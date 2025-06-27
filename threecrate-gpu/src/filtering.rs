//! GPU-accelerated filtering

use threecrate_core::{PointCloud, Result, Point3f};
use crate::GpuContext;

const STATISTICAL_OUTLIER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> neighbors: array<array<u32, MAX_NEIGHBORS>>;
@group(0) @binding(2) var<storage, read_write> is_outlier: array<u32>;
@group(0) @binding(3) var<uniform> params: FilterParams;

struct FilterParams {
    num_points: u32,
    k_neighbors: u32,
    std_dev_threshold: f32,
    mean_distance: f32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_points) {
        return;
    }
    
    let center_point = input_points[index];
    
    // Compute mean distance to k-nearest neighbors
    var total_distance = 0.0;
    var count = 0u;
    
    for (var i = 0u; i < params.k_neighbors; i++) {
        let neighbor_idx = neighbors[index][i];
        if (neighbor_idx < params.num_points) {
            let neighbor_point = input_points[neighbor_idx];
            let distance = length(center_point - neighbor_point);
            total_distance += distance;
            count++;
        }
    }
    
    if (count == 0u) {
        is_outlier[index] = 0u;
        return;
    }
    
    let mean_distance = total_distance / f32(count);
    
    // Mark as outlier if distance is beyond threshold
    let deviation = abs(mean_distance - params.mean_distance);
    is_outlier[index] = select(0u, 1u, deviation > params.std_dev_threshold);
}
"#;

impl GpuContext {
    /// Remove statistical outliers from point cloud using GPU acceleration
    pub async fn remove_statistical_outliers(
        &self,
        points: &[Point3f],
        k_neighbors: usize,
        std_dev_multiplier: f32,
    ) -> Result<Vec<Point3f>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        // Convert points to GPU format
        let point_data: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();

        // Compute neighbors (reuse from normals computation)
        let neighbors = self.compute_neighbors_simple(&point_data, k_neighbors);
        
        // Compute global statistics on CPU first (could be moved to GPU)
        let global_mean = self.compute_global_mean_distance(&point_data, &neighbors, k_neighbors);

        // Create buffers
        let input_buffer = self.create_buffer_init(
            "Input Points",
            &point_data,
            wgpu::BufferUsages::STORAGE,
        );

        let neighbors_buffer = self.create_buffer_init(
            "Neighbors",
            &neighbors,
            wgpu::BufferUsages::STORAGE,
        );

        let outlier_buffer = self.create_buffer(
            "Outlier Flags",
            (point_data.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FilterParams {
            num_points: u32,
            k_neighbors: u32,
            std_dev_threshold: f32,
            mean_distance: f32,
        }

        let params = FilterParams {
            num_points: points.len() as u32,
            k_neighbors: k_neighbors as u32,
            std_dev_threshold: global_mean * std_dev_multiplier,
            mean_distance: global_mean,
        };

        let params_buffer = self.create_buffer_init(
            "Filter Params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create shader with MAX_NEIGHBORS constant
        let shader_source = STATISTICAL_OUTLIER_SHADER.replace("MAX_NEIGHBORS", &k_neighbors.to_string());
        let shader = self.create_shader_module("Statistical Outlier Filter", &shader_source);

        // Create bind group layout
        let bind_group_layout = self.create_bind_group_layout(
            "Outlier Filter",
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            label: Some("Outlier Filter Pipeline"),
            layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Outlier Filter Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
        });

        // Create bind group
        let bind_group = self.create_bind_group(
            "Outlier Filter",
            &bind_group_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: neighbors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: outlier_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Outlier Filter"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Outlier Filter Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (points.len() + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Read back results
        let staging_buffer = self.create_buffer(
            "Outlier Staging",
            (point_data.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        encoder.copy_buffer_to_buffer(
            &outlier_buffer,
            0,
            &staging_buffer,
            0,
            (point_data.len() * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let outlier_flags: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            
            let filtered_points: Vec<Point3f> = points
                .iter()
                .zip(outlier_flags.iter())
                .filter(|&(_, &is_outlier)| is_outlier == 0)
                .map(|(point, _)| *point)
                .collect();
            
            drop(data);
            staging_buffer.unmap();
            
            Ok(filtered_points)
        } else {
            Err(threecrate_core::Error::Gpu("Failed to read GPU filtering results".to_string()))
        }
    }

    /// Compute global mean distance for statistical filtering
    fn compute_global_mean_distance(&self, points: &[[f32; 3]], neighbors: &[[u32; 64]], k: usize) -> f32 {
        let k = k.min(64).min(points.len());
        let mut total_distance = 0.0;
        let mut count = 0;

        for (i, point) in points.iter().enumerate() {
            for j in 0..k {
                let neighbor_idx = neighbors[i][j] as usize;
                if neighbor_idx < points.len() {
                    let neighbor_point = &points[neighbor_idx];
                    let dx = point[0] - neighbor_point[0];
                    let dy = point[1] - neighbor_point[1];
                    let dz = point[2] - neighbor_point[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    total_distance += distance;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_distance / count as f32
        } else {
            0.0
        }
    }
}

/// GPU-accelerated statistical outlier removal
pub async fn gpu_remove_statistical_outliers(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    k_neighbors: usize,
    std_dev_multiplier: f32,
) -> Result<PointCloud<Point3f>> {
    let filtered_points = gpu_context.remove_statistical_outliers(&cloud.points, k_neighbors, std_dev_multiplier).await?;
    Ok(PointCloud::from_points(filtered_points))
}

/// GPU-accelerated voxel grid filtering
pub fn gpu_voxel_grid_filter(_cloud: &PointCloud<Point3f>, _voxel_size: f32) -> Result<PointCloud<Point3f>> {
    // TODO: Implement GPU voxel grid filtering
    todo!("GPU voxel grid filtering not yet implemented")
} 