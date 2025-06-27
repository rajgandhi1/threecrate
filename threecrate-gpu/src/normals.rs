//! GPU-accelerated normal estimation

use threecrate_core::{PointCloud, Result, Point3f, NormalPoint3f};
use crate::GpuContext;
// use wgpu::util::DeviceExt; // Used in device.rs

const NORMALS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> output_normals: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> neighbors: array<array<u32, MAX_NEIGHBORS>>;
@group(0) @binding(3) var<uniform> params: NormalParams;

struct NormalParams {
    num_points: u32,
    k_neighbors: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_points) {
        return;
    }
    
    let center_point = input_points[index];
    
    // Compute covariance matrix
    var centroid = vec3<f32>(0.0);
    var count = 0u;
    
    // Calculate centroid of neighbors
    for (var i = 0u; i < params.k_neighbors; i++) {
        let neighbor_idx = neighbors[index][i];
        if (neighbor_idx < params.num_points) {
            centroid += input_points[neighbor_idx];
            count++;
        }
    }
    
    if (count == 0u) {
        output_normals[index] = vec3<f32>(0.0, 0.0, 1.0);
        return;
    }
    
    centroid /= f32(count);
    
    // Compute covariance matrix
    var cov00 = 0.0; var cov01 = 0.0; var cov02 = 0.0;
    var cov11 = 0.0; var cov12 = 0.0; var cov22 = 0.0;
    
    for (var i = 0u; i < params.k_neighbors; i++) {
        let neighbor_idx = neighbors[index][i];
        if (neighbor_idx < params.num_points) {
            let diff = input_points[neighbor_idx] - centroid;
            cov00 += diff.x * diff.x;
            cov01 += diff.x * diff.y;
            cov02 += diff.x * diff.z;
            cov11 += diff.y * diff.y;
            cov12 += diff.y * diff.z;
            cov22 += diff.z * diff.z;
        }
    }
    
    let scale = 1.0 / f32(count);
    cov00 *= scale; cov01 *= scale; cov02 *= scale;
    cov11 *= scale; cov12 *= scale; cov22 *= scale;
    
    // Simplified eigenvalue computation for smallest eigenvalue's eigenvector
    // This is a simplified approach - in practice, you'd want a more robust method
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    
    // Use cross product method for robustness
    let diff1 = input_points[neighbors[index][0]] - center_point;
    let diff2 = input_points[neighbors[index][min(1u, params.k_neighbors - 1u)]] - center_point;
    let cross_prod = cross(diff1, diff2);
    
    if (length(cross_prod) > 1e-6) {
        normal = normalize(cross_prod);
    }
    
    output_normals[index] = normal;
}
"#;

impl GpuContext {
    /// Compute normals for a point cloud using GPU acceleration
    pub async fn compute_normals(&self, points: &[Point3f], k_neighbors: usize) -> Result<Vec<nalgebra::Vector3<f32>>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        // Convert points to GPU format
        let point_data: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();

        // Create buffers
        let input_buffer = self.create_buffer_init(
            "Input Points",
            &point_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let output_buffer = self.create_buffer(
            "Output Normals",
            (point_data.len() * std::mem::size_of::<[f32; 3]>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // For now, use a simple neighbor computation (could be replaced with KD-tree)
        let neighbors = self.compute_neighbors_simple(&point_data, k_neighbors);
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
        }

        let params = NormalParams {
            num_points: points.len() as u32,
            k_neighbors: k_neighbors as u32,
        };

        let params_buffer = self.create_buffer_init(
            "Params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create shader with MAX_NEIGHBORS constant
        let shader_source = NORMALS_SHADER.replace("MAX_NEIGHBORS", &k_neighbors.to_string());
        let shader = self.create_shader_module("Normals Compute", &shader_source);

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
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
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
            (point_data.len() * std::mem::size_of::<[f32; 3]>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (point_data.len() * std::mem::size_of::<[f32; 3]>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let normals: Vec<[f32; 3]> = bytemuck::cast_slice(&data).to_vec();
            
            let result = normals
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