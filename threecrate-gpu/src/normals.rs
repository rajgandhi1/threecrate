//! GPU-accelerated normal estimation

use threecrate_core::{PointCloud, Result, Point3f, NormalPoint3f};
use crate::GpuContext;
// use wgpu::util::DeviceExt; // Used in device.rs

const NORMALS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output_normals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> neighbors: array<array<u32, 64>>;
@group(0) @binding(3) var<uniform> params: NormalParams;

struct NormalParams {
    num_points: u32,
    k_neighbors: u32,
    consistent_orientation: u32,
    _pad: u32,
    viewpoint: vec4<f32>,
}

fn mat3_mul_vec3(m00: f32, m01: f32, m02: f32,
                 m10: f32, m11: f32, m12: f32,
                 m20: f32, m21: f32, m22: f32,
                 v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        m00 * v.x + m01 * v.y + m02 * v.z,
        m10 * v.x + m11 * v.y + m12 * v.z,
        m20 * v.x + m21 * v.y + m22 * v.z
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_points) {
        return;
    }
    
    let center_point = input_points[index].xyz;
    
    // Compute covariance matrix via neighbor centroid
    var centroid = vec3<f32>(0.0);
    var count = 0u;
    for (var i = 0u; i < params.k_neighbors; i++) {
        let neighbor_idx = neighbors[index][i];
        if (neighbor_idx < params.num_points) {
            centroid += input_points[neighbor_idx].xyz;
            count++;
        }
    }
    
    if (count < 3u) {
        output_normals[index] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return;
    }
    
    centroid /= f32(count);
    
    // Covariance components
    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0;
    var c11 = 0.0; var c12 = 0.0; var c22 = 0.0;
    for (var i = 0u; i < params.k_neighbors; i++) {
        let neighbor_idx = neighbors[index][i];
        if (neighbor_idx < params.num_points) {
            let d = input_points[neighbor_idx].xyz - centroid;
            c00 += d.x * d.x;
            c01 += d.x * d.y;
            c02 += d.x * d.z;
            c11 += d.y * d.y;
            c12 += d.y * d.z;
            c22 += d.z * d.z;
        }
    }
    let inv_count = 1.0 / f32(count);
    c00 *= inv_count; c01 *= inv_count; c02 *= inv_count;
    c11 *= inv_count; c12 *= inv_count; c22 *= inv_count;
    
    // Power iteration on shifted matrix D = trace(C) * I - C to get eigenvector of smallest eigenvalue of C
    let trace_c = c00 + c11 + c22;
    
    // Initial vector using cross of two neighbor directions for stability
    let n0 = neighbors[index][0u];
    let n1 = neighbors[index][1u];
    var v = vec3<f32>(0.0, 0.0, 1.0);
    if (n0 < params.num_points && n1 < params.num_points) {
        let d1 = normalize(input_points[n0].xyz - center_point);
        let d2 = normalize(input_points[n1].xyz - center_point);
        let cp = cross(d1, d2);
        if (length(cp) > 1e-6) {
            v = normalize(cp);
        }
    }
    
    // Perform fixed number of iterations
    for (var it = 0u; it < 8u; it++) {
        // Multiply v by D = trace*I - C
        let Cv = mat3_mul_vec3(c00, c01, c02, c01, c11, c12, c02, c12, c22, v);
        let Dv = vec3<f32>(trace_c * v.x, trace_c * v.y, trace_c * v.z) - Cv;
        let lenDv = length(Dv);
        if (lenDv > 1e-8) {
            v = Dv / lenDv;
        }
    }
    var normal = v;
    
    // Orientation consistency
    if (params.consistent_orientation == 1u) {
        let to_view = normalize(params.viewpoint.xyz - center_point);
        if (dot(normal, to_view) < 0.0) {
            normal = -normal;
        }
    }
    
    output_normals[index] = vec4<f32>(normal, 0.0);
}
"#;

impl GpuContext {
    /// Compute normals for a point cloud using GPU acceleration with options
    pub async fn compute_normals_with_options(
        &self,
        points: &[Point3f],
        k_neighbors: usize,
        consistent_orientation: bool,
        viewpoint: Option<[f32; 3]>,
    ) -> Result<Vec<nalgebra::Vector3<f32>>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        // Convert points to GPU format (std430 alignment prefers vec4)
        let point_data: Vec<[f32; 4]> = points
            .iter()
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();

        // Create buffers
        let input_buffer = self.create_buffer_init(
            "Input Points",
            &point_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let output_buffer = self.create_buffer(
            "Output Normals",
            (point_data.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // For now, use a simple neighbor computation (could be replaced with KD-tree)
        let k_neighbors = k_neighbors.max(3).min(64);
        let neighbors = self.compute_neighbors_simple_points3(&points.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<[f32;3]>>(), k_neighbors);
        let neighbors_buffer = self.create_buffer_init(
            "Neighbors",
            &neighbors,
            wgpu::BufferUsages::STORAGE,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct NormalParams {
            num_points: u32,
            k_neighbors: u32,
            consistent_orientation: u32,
            _pad: u32,
            viewpoint: [f32; 4],
        }

        // Determine default viewpoint (mimic CPU implementation): above bbox center along +Z by extent
        let vp = if let Some(vp) = viewpoint {
            [vp[0], vp[1], vp[2], 0.0]
        } else {
            let mut min_x = point_data[0][0];
            let mut min_y = point_data[0][1];
            let mut min_z = point_data[0][2];
            let mut max_x = point_data[0][0];
            let mut max_y = point_data[0][1];
            let mut max_z = point_data[0][2];
            for p in &point_data {
                min_x = min_x.min(p[0]);
                min_y = min_y.min(p[1]);
                min_z = min_z.min(p[2]);
                max_x = max_x.max(p[0]);
                max_y = max_y.max(p[1]);
                max_z = max_z.max(p[2]);
            }
            let cx = (min_x + max_x) * 0.5;
            let cy = (min_y + max_y) * 0.5;
            let cz = (min_z + max_z) * 0.5;
            let dx = max_x - min_x;
            let dy = max_y - min_y;
            let dz = max_z - min_z;
            let extent = (dx * dx + dy * dy + dz * dz).sqrt();
            [cx, cy, cz + extent, 0.0]
        };

        let params = NormalParams {
            num_points: points.len() as u32,
            k_neighbors: k_neighbors as u32,
            consistent_orientation: if consistent_orientation { 1 } else { 0 },
            _pad: 0,
            viewpoint: vp,
        };

        let params_buffer = self.create_buffer_init(
            "Params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create shader
        let shader = self.create_shader_module("Normals Compute", NORMALS_SHADER);

        // Create bind group layout
        let bind_group_layout = self.create_bind_group_layout(
            "Normal Computation",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Normal Computation Pipeline"),
            layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Normal Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            })),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = self.create_bind_group(
            "Normal Computation",
            &bind_group_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: neighbors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Normal Computation"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Normal Computation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (points.len() + 63) / 64; // 64 is workgroup size
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Read back results
        let staging_buffer = self.create_buffer(
            "Staging Buffer",
            (point_data.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (point_data.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let normals4: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
            
            let result = normals4
                .into_iter()
                .map(|n| nalgebra::Vector3::new(n[0], n[1], n[2]))
                .collect();
            
            drop(data);
            staging_buffer.unmap();
            
            Ok(result)
        } else {
            Err(threecrate_core::Error::Gpu("Failed to read GPU results".to_string()))
        }
    }

    /// Compute normals for a point cloud using GPU acceleration with default options
    pub async fn compute_normals(&self, points: &[Point3f], k_neighbors: usize) -> Result<Vec<nalgebra::Vector3<f32>>> {
        self.compute_normals_with_options(points, k_neighbors, true, None).await
    }

    /// Simple neighbor computation (brute force - could be replaced with KD-tree)
    pub fn compute_neighbors_simple(&self, points: &[[f32; 3]], k: usize) -> Vec<[u32; 64]> {
        let mut neighbors = vec![[0u32; 64]; points.len()];
        let k = k.min(64).min(points.len()); // Limit to 64 neighbors and available points
        
        for (i, point) in points.iter().enumerate() {
            let mut distances: Vec<(f32, usize)> = points
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, other)| {
                    let dx = point[0] - other[0];
                    let dy = point[1] - other[1];
                    let dz = point[2] - other[2];
                    (dx * dx + dy * dy + dz * dz, j)
                })
                .collect();
            
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            for (idx, &(_, neighbor_idx)) in distances.iter().take(k).enumerate() {
                neighbors[i][idx] = neighbor_idx as u32;
            }
            
            // Fill remaining slots with the same neighbor to avoid issues
            for idx in k..64 {
                neighbors[i][idx] = if k > 0 { neighbors[i][k - 1] } else { i as u32 };
            }
        }
        
        neighbors
    }

    /// Helper to compute neighbors from Vec<[f32;3]> built from owned data
    pub fn compute_neighbors_simple_points3(&self, points: &[[f32; 3]], k: usize) -> Vec<[u32; 64]> {
        self.compute_neighbors_simple(points, k)
    }
}

/// GPU-accelerated normal estimation for point clouds
pub async fn gpu_estimate_normals(
    gpu_context: &GpuContext,
    cloud: &mut PointCloud<Point3f>,
    k: usize,
) -> Result<PointCloud<NormalPoint3f>> {
    let normals = gpu_context.compute_normals(&cloud.points, k).await?;
    
    let normal_points: Vec<NormalPoint3f> = cloud
        .points
        .iter()
        .zip(normals.iter())
        .map(|(point, normal)| NormalPoint3f {
            position: *point,
            normal: *normal,
        })
        .collect();
    
    Ok(PointCloud::from_points(normal_points))
} 

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{Point3f, PointCloud};

    /// Try to create a GPU context, return None if not available
    async fn try_create_gpu_context() -> Option<crate::GpuContext> {
        match crate::GpuContext::new().await {
            Ok(gpu) => Some(gpu),
            Err(_) => {
                println!("⚠️  GPU not available, skipping GPU-dependent test");
                None
            }
        }
    }

    #[tokio::test]
    async fn test_gpu_normals_plane() {
        let Some(gpu) = try_create_gpu_context().await else {
            return;
        };
        
        let mut cloud = PointCloud::new();
        // Create XY plane grid
        for i in 0..15 { for j in 0..15 {
            cloud.push(Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
        }}
        let result = gpu_estimate_normals(&gpu, &mut cloud, 8).await.unwrap();
        assert_eq!(result.len(), 225);
        let mut z_count = 0;
        for p in result.iter() {
            if p.normal.z.abs() > 0.8 { z_count += 1; }
        }
        let pct = (z_count as f32 / result.len() as f32) * 100.0;
        assert!(pct > 80.0, "Only {:.1}% normals in Z direction", pct);
    }

    #[tokio::test]
    async fn test_gpu_normals_compare_cpu_plane() {
        use threecrate_algorithms::estimate_normals as cpu_estimate_normals;
        let Some(gpu) = try_create_gpu_context().await else {
            return;
        };
        
        let mut cloud = PointCloud::new();
        for i in 0..10 { for j in 0..10 {
            cloud.push(Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
        }}
        let gpu_cloud = gpu_estimate_normals(&gpu, &mut cloud.clone(), 8).await.unwrap();
        let cpu_cloud = cpu_estimate_normals(&cloud, 8).unwrap();
        // Compare orientation alignment percentage
        let mut agree = 0usize;
        for (g, c) in gpu_cloud.iter().zip(cpu_cloud.iter()) {
            let dot = g.normal.dot(&c.normal);
            if dot.abs() > 0.7 { agree += 1; }
        }
        let pct = (agree as f32 / gpu_cloud.len() as f32) * 100.0;
        assert!(pct > 70.0, "GPU-CPU normals agree only {:.1}%", pct);
    }
}