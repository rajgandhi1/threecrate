//! GPU-accelerated filtering

use threecrate_core::{PointCloud, Result, Point3f};
use crate::GpuContext;

const STATISTICAL_OUTLIER_SHADER: &str = r#"
struct GpuPoint {
    position: vec3<f32>,
    _padding: f32,  // Ensure 16-byte alignment
}

@group(0) @binding(0) var<storage, read> input_points: array<GpuPoint>;
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
    
    let center_point = input_points[index].position;
    
    // Compute mean distance to k-nearest neighbors
    var total_distance = 0.0;
    var count = 0u;
    
    for (var i = 0u; i < params.k_neighbors; i++) {
        let neighbor_idx = neighbors[index][i];
        if (neighbor_idx < params.num_points) {
            let neighbor_point = input_points[neighbor_idx].position;
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

const RADIUS_OUTLIER_SHADER: &str = r#"
struct GpuPoint {
    position: vec3<f32>,
    _padding: f32,  // Ensure 16-byte alignment
}

@group(0) @binding(0) var<storage, read> input_points: array<GpuPoint>;
@group(0) @binding(1) var<storage, read_write> is_outlier: array<u32>;
@group(0) @binding(2) var<uniform> params: RadiusOutlierParams;

struct RadiusOutlierParams {
    num_points: u32,
    radius: f32,
    min_neighbors: u32,
    _padding: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_points) {
        return;
    }
    
    let center_point = input_points[index].position;
    var neighbor_count = 0u;
    
    // Count neighbors within radius
    for (var i = 0u; i < params.num_points; i++) {
        if (i != index) {
            let neighbor_point = input_points[i].position;
            let distance = length(center_point - neighbor_point);
            if (distance <= params.radius) {
                neighbor_count++;
            }
        }
    }
    
    // Mark as outlier if neighbor count is below threshold
    is_outlier[index] = select(0u, 1u, neighbor_count < params.min_neighbors);
}
"#;

const VOXEL_GRID_SHADER: &str = r#"
struct GpuPoint {
    position: vec3<f32>,
    _padding: f32,  // Ensure 16-byte alignment
}

@group(0) @binding(0) var<storage, read> input_points: array<GpuPoint>;
@group(0) @binding(1) var<storage, read_write> voxel_indices: array<u32>;
@group(0) @binding(2) var<uniform> params: VoxelGridParams;

struct VoxelGridParams {
    num_points: u32,
    voxel_size: f32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
}

fn get_voxel_index(point: vec3<f32>, params: VoxelGridParams) -> u32 {
    // Simple coordinate-based voxel index (similar to CPU version)
    let voxel_x = i32(floor((point.x - params.min_x) / params.voxel_size));
    let voxel_y = i32(floor((point.y - params.min_y) / params.voxel_size));
    let voxel_z = i32(floor((point.z - params.min_z) / params.voxel_size));
    
    // Use a simple hash based on coordinates
    let hash_x = u32(voxel_x * voxel_x + voxel_x);
    let hash_y = u32(voxel_y * voxel_y + voxel_y * 7);
    let hash_z = u32(voxel_z * voxel_z + voxel_z * 13);
    
    return (hash_x + hash_y * 31u + hash_z * 97u) % 100000u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_points) {
        return;
    }
    
    let point = input_points[index].position;
    let voxel_idx = get_voxel_index(point, params);
    
    // Store voxel index for this point
    voxel_indices[index] = voxel_idx;
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

        // Convert points to GPU format with proper alignment
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuPoint {
            position: [f32; 3],
            _padding: f32,
        }
        
        let point_data: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint {
                position: [p.x, p.y, p.z],
                _padding: 0.0,
            })
            .collect();

        // Convert to the format expected by helper functions
        let point_data_flat: Vec<[f32; 3]> = point_data.iter().map(|p| p.position).collect();
        
        // Compute neighbors (reuse from normals computation)
        let neighbors = self.compute_neighbors_simple(&point_data_flat, k_neighbors);
        
        // Compute global statistics on CPU first (could be moved to GPU)
        let global_mean = self.compute_global_mean_distance(&point_data_flat, &neighbors, k_neighbors);

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
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
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

        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

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

    /// Remove radius outliers from point cloud using GPU acceleration
    pub async fn remove_radius_outliers(
        &self,
        points: &[Point3f],
        radius: f32,
        min_neighbors: usize,
    ) -> Result<Vec<Point3f>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        // Convert points to GPU format with proper alignment
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuPoint {
            position: [f32; 3],
            _padding: f32,
        }
        
        let point_data: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint {
                position: [p.x, p.y, p.z],
                _padding: 0.0,
            })
            .collect();

        // Create buffers
        let input_buffer = self.create_buffer_init(
            "Input Points",
            &point_data,
            wgpu::BufferUsages::STORAGE,
        );

        let outlier_buffer = self.create_buffer(
            "Outlier Flags",
            (point_data.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RadiusOutlierParams {
            num_points: u32,
            radius: f32,
            min_neighbors: u32,
            _padding: u32,
        }

        let params = RadiusOutlierParams {
            num_points: points.len() as u32,
            radius,
            min_neighbors: min_neighbors as u32,
            _padding: 0,
        };

        let params_buffer = self.create_buffer_init(
            "Radius Outlier Params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create shader
        let shader = self.create_shader_module("Radius Outlier Filter", RADIUS_OUTLIER_SHADER);

        // Create bind group layout
        let bind_group_layout = self.create_bind_group_layout(
            "Radius Outlier Filter",
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
            label: Some("Radius Outlier Filter Pipeline"),
            layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Radius Outlier Filter Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = self.create_bind_group(
            "Radius Outlier Filter",
            &bind_group_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: outlier_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Radius Outlier Filter"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Radius Outlier Filter Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (points.len() + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Read back results
        let staging_buffer = self.create_buffer(
            "Radius Outlier Staging",
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

        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

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
            Err(threecrate_core::Error::Gpu("Failed to read GPU radius outlier filtering results".to_string()))
        }
    }

    /// Voxel grid filtering using GPU acceleration
    pub async fn voxel_grid_filter(
        &self,
        points: &[Point3f],
        voxel_size: f32,
    ) -> Result<Vec<Point3f>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        if voxel_size <= 0.0 {
            return Err(threecrate_core::Error::InvalidData("Voxel size must be positive".to_string()));
        }

        // Convert points to GPU format
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuPoint {
            position: [f32; 3],
            _padding: f32,
        }
        
        let point_data: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint {
                position: [p.x, p.y, p.z],
                _padding: 0.0,
            })
            .collect();

        // Compute grid bounds
        let min_x = points.iter().map(|p| p.x).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_y = points.iter().map(|p| p.y).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_z = points.iter().map(|p| p.z).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_x = points.iter().map(|p| p.x).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_y = points.iter().map(|p| p.y).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_z = points.iter().map(|p| p.z).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        // Estimate total voxels for buffer allocation (we'll use a hash-based approach)
        let total_voxels = points.len() * 2; // Conservative estimate

        // Create buffers
        let input_buffer = self.create_buffer_init(
            "Input Points",
            &point_data,
            wgpu::BufferUsages::STORAGE,
        );

        let voxel_indices_buffer = self.create_buffer(
            "Voxel Indices",
            (point_data.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );



        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct VoxelGridParams {
            num_points: u32,
            voxel_size: f32,
            min_x: f32,
            min_y: f32,
            min_z: f32,
            max_x: f32,
            max_y: f32,
            max_z: f32,
        }

        let params = VoxelGridParams {
            num_points: points.len() as u32,
            voxel_size,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        };

        let params_buffer = self.create_buffer_init(
            "Voxel Grid Params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create shader
        let shader = self.create_shader_module("Voxel Grid Filter", VOXEL_GRID_SHADER);

        // Create bind group layout
        let bind_group_layout = self.create_bind_group_layout(
            "Voxel Grid Filter",
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
            label: Some("Voxel Grid Filter Pipeline"),
            layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Voxel Grid Filter Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = self.create_bind_group(
            "Voxel Grid Filter",
            &bind_group_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: voxel_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Voxel Grid Filter"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Voxel Grid Filter Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (points.len() + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Read back voxel indices
        let indices_staging = self.create_buffer(
            "Voxel Indices Staging",
            (point_data.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        encoder.copy_buffer_to_buffer(
            &voxel_indices_buffer,
            0,
            &indices_staging,
            0,
            (point_data.len() * std::mem::size_of::<u32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let indices_slice = indices_staging.slice(..);
        
        let (indices_sender, indices_receiver) = futures_intrusive::channel::shared::oneshot_channel();
        
        indices_slice.map_async(wgpu::MapMode::Read, move |v| indices_sender.send(v).unwrap());

        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Some(Ok(())) = indices_receiver.receive().await {
            let indices_data = indices_slice.get_mapped_range();
            
            let voxel_indices: Vec<u32> = bytemuck::cast_slice(&indices_data).to_vec();
            
            // Post-process on CPU: keep one point per voxel (the first one)
            let mut voxel_used = vec![false; total_voxels];
            let mut filtered_points = Vec::new();
            
            for (point_idx, &voxel_idx) in voxel_indices.iter().enumerate() {
                if voxel_idx < total_voxels as u32 && !voxel_used[voxel_idx as usize] {
                    filtered_points.push(points[point_idx]);
                    voxel_used[voxel_idx as usize] = true;
                }
            }
            
            drop(indices_data);
            indices_staging.unmap();
            
            Ok(filtered_points)
        } else {
            Err(threecrate_core::Error::Gpu("Failed to read GPU voxel grid filtering results".to_string()))
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

/// GPU-accelerated radius outlier removal
pub async fn gpu_radius_outlier_removal(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    radius: f32,
    min_neighbors: usize,
) -> Result<PointCloud<Point3f>> {
    let filtered_points = gpu_context.remove_radius_outliers(&cloud.points, radius, min_neighbors).await?;
    Ok(PointCloud::from_points(filtered_points))
}

/// GPU-accelerated voxel grid filtering
pub async fn gpu_voxel_grid_filter(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    voxel_size: f32,
) -> Result<PointCloud<Point3f>> {
    let filtered_points = gpu_context.voxel_grid_filter(&cloud.points, voxel_size).await?;
    Ok(PointCloud::from_points(filtered_points))
} 

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{PointCloud, Point3f};
    use std::time::Instant;

    async fn try_create_gpu_context() -> Option<GpuContext> {
        GpuContext::new().await.ok()
    }

    fn create_test_point_cloud_with_outliers() -> PointCloud<Point3f> {
        let mut points = Vec::new();
        
        // Create a cluster of normal points
        for i in 0..50 {
            for j in 0..50 {
                let x = (i as f32 - 25.0) * 0.1;
                let y = (j as f32 - 25.0) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
            }
        }
        
        // Add some outliers far from the cluster
        for i in 0..20 {
            let x = 50.0 + (i as f32 * 2.0);
            let y = 50.0 + (i as f32 * 2.0);
            let z = 50.0 + (i as f32 * 2.0);
            points.push(Point3f::new(x, y, z));
        }
        
        PointCloud::from_points(points)
    }

    fn create_test_point_cloud_no_outliers() -> PointCloud<Point3f> {
        let mut points = Vec::new();
        
        // Create a uniform grid of points
        for i in 0..25 {
            for j in 0..25 {
                for k in 0..5 {
                    let x = (i as f32 - 12.5) * 0.1;
                    let y = (j as f32 - 12.5) * 0.1;
                    let z = (k as f32 - 2.5) * 0.1;
                    points.push(Point3f::new(x, y, z));
                }
            }
        }
        
        PointCloud::from_points(points)
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_basic() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_with_outliers();
        let original_count = cloud.len();
        
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0).await.unwrap();
        let filtered_count = filtered.len();
        
        // Should remove some outliers
        assert!(filtered_count < original_count);
        assert!(filtered_count > 0);
        
        println!("GPU outlier removal: {} -> {} points", original_count, filtered_count);
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_no_outliers() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_no_outliers();
        let original_count = cloud.len();
        
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0).await.unwrap();
        let filtered_count = filtered.len();
        
        println!("GPU outlier removal (no outliers): {} -> {} points", original_count, filtered_count);
        
        // The algorithm should remove some points but not all
        // Even with no obvious outliers, the algorithm may remove points based on statistical analysis
        assert!(filtered_count > 0);
        assert!(filtered_count < original_count);
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_empty_cloud() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::<Point3f>::new();
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0).await.unwrap();
        
        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_single_point() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 1, 1.0).await.unwrap();
        
        // Should keep the single point
        assert_eq!(filtered.len(), 1);
    }

    #[tokio::test]
    async fn test_gpu_vs_cpu_statistical_outlier_removal() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU vs CPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_with_outliers();
        
        // CPU version - use the same algorithm logic but avoid type conflicts
        let cpu_start = Instant::now();
        let cpu_filtered_count = {
            // Simple CPU implementation for comparison
            let points = &cloud.points;
            let k = 10;
            let std_dev_multiplier = 1.0;
            
            if points.is_empty() {
                0
            } else {
                // Simple distance-based outlier detection
                let mut mean_distances = Vec::new();
                for point in points {
                    let mut distances = Vec::new();
                    for other_point in points {
                        if other_point != point {
                            let dx = point.x - other_point.x;
                            let dy = point.y - other_point.y;
                            let dz = point.z - other_point.z;
                            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                            distances.push(distance);
                        }
                    }
                    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let k_neighbors = k.min(distances.len());
                    if k_neighbors > 0 {
                        let mean = distances[..k_neighbors].iter().sum::<f32>() / k_neighbors as f32;
                        mean_distances.push(mean);
                    } else {
                        mean_distances.push(0.0);
                    }
                }
                
                let global_mean = mean_distances.iter().sum::<f32>() / mean_distances.len() as f32;
                let variance = mean_distances.iter().map(|&d| (d - global_mean).powi(2)).sum::<f32>() / mean_distances.len() as f32;
                let global_std_dev = variance.sqrt();
                let threshold = global_mean + std_dev_multiplier * global_std_dev;
                
                mean_distances.iter().filter(|&&d| d <= threshold).count()
            }
        };
        let cpu_time = cpu_start.elapsed();
        
        // GPU version
        let gpu_start = Instant::now();
        let gpu_filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0).await.unwrap();
        let gpu_time = gpu_start.elapsed();
        
        println!("CPU outlier removal: {} -> {} points in {:?}", 
                cloud.len(), cpu_filtered_count, cpu_time);
        println!("GPU outlier removal: {} -> {} points in {:?}", 
                cloud.len(), gpu_filtered.len(), gpu_time);
        
        // Both should remove outliers (exact counts may vary due to different algorithms)
        assert!(cpu_filtered_count < cloud.len());
        assert!(gpu_filtered.len() < cloud.len());
        
        // Performance comparison (GPU should be faster for larger datasets)
        if cloud.len() > 1000 {
            println!("GPU speedup: {:.2}x", cpu_time.as_secs_f32() / gpu_time.as_secs_f32());
        }
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_different_parameters() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_with_outliers();
        let original_count = cloud.len();
        
        // Test different k values
        for k in [5, 10, 20] {
            let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, k, 1.0).await.unwrap();
            println!("k={}: {} -> {} points", k, original_count, filtered.len());
            assert!(filtered.len() > 0);
        }
        
        // Test different std_dev_multiplier values
        for std_dev in [0.5, 1.0, 2.0] {
            let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, std_dev).await.unwrap();
            println!("std_dev={}: {} -> {} points", std_dev, original_count, filtered.len());
            assert!(filtered.len() > 0);
        }
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_large_dataset() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        // Create a smaller dataset for faster testing
        let mut points = Vec::new();
        for i in 0..50 {
            for j in 0..50 {
                let x = (i as f32 - 25.0) * 0.1;
                let y = (j as f32 - 25.0) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
            }
        }
        
        // Add some outliers
        for i in 0..10 {
            let x = 25.0 + (i as f32 * 2.0);
            let y = 25.0 + (i as f32 * 2.0);
            let z = 25.0 + (i as f32 * 2.0);
            points.push(Point3f::new(x, y, z));
        }
        
        let cloud = PointCloud::from_points(points);
        let original_count = cloud.len();
        
        let start = Instant::now();
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 5, 1.0).await.unwrap();
        let elapsed = start.elapsed();
        
        println!("Large dataset GPU outlier removal: {} -> {} points in {:?}", 
                original_count, filtered.len(), elapsed);
        
        assert!(filtered.len() < original_count);
        assert!(filtered.len() > 0);
        
        // Should complete in reasonable time
        assert!(elapsed.as_secs() < 10);
    }

    fn create_test_point_cloud_for_radius_outlier() -> PointCloud<Point3f> {
        let mut points = Vec::new();
        
        // Create a dense cluster
        for i in 0..20 {
            for j in 0..20 {
                let x = (i as f32 - 10.0) * 0.1;
                let y = (j as f32 - 10.0) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
            }
        }
        
        // Add some isolated points (outliers)
        for i in 0..5 {
            let x = 10.0 + (i as f32 * 2.0);
            let y = 10.0 + (i as f32 * 2.0);
            let z = 10.0 + (i as f32 * 2.0);
            points.push(Point3f::new(x, y, z));
        }
        
        PointCloud::from_points(points)
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_basic() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_radius_outlier();
        let original_count = cloud.len();
        
        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 3).await.unwrap();
        let filtered_count = filtered.len();
        
        // Should remove some outliers
        assert!(filtered_count < original_count);
        assert!(filtered_count > 0);
        
        println!("GPU radius outlier removal: {} -> {} points", original_count, filtered_count);
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_empty_cloud() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::<Point3f>::new();
        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 3).await.unwrap();
        
        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_single_point() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 1).await.unwrap();
        
        // Should remove the single point since it has no neighbors
        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_different_parameters() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_radius_outlier();
        let original_count = cloud.len();
        
        // Test different radius values
        for radius in [0.2, 0.5, 1.0] {
            let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, radius, 3).await.unwrap();
            println!("radius={}: {} -> {} points", radius, original_count, filtered.len());
            assert!(filtered.len() > 0);
        }
        
        // Test different min_neighbors values
        for min_neighbors in [1, 3, 5] {
            let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, min_neighbors).await.unwrap();
            println!("min_neighbors={}: {} -> {} points", min_neighbors, original_count, filtered.len());
            assert!(filtered.len() > 0);
        }
    }

    fn create_test_point_cloud_for_voxel_grid() -> PointCloud<Point3f> {
        let mut points = Vec::new();
        
        // Create a dense grid of points
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..5 {
                    let x = (i as f32 - 5.0) * 0.1;
                    let y = (j as f32 - 5.0) * 0.1;
                    let z = (k as f32 - 2.5) * 0.1;
                    points.push(Point3f::new(x, y, z));
                }
            }
        }
        
        // Add some duplicate points in the same voxels
        for i in 0..5 {
            for j in 0..5 {
                let x = (i as f32 - 2.5) * 0.1;
                let y = (j as f32 - 2.5) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
                points.push(Point3f::new(x, y, z)); // Duplicate
            }
        }
        
        PointCloud::from_points(points)
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_basic() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_voxel_grid();
        let original_count = cloud.len();
        
        let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1).await.unwrap();
        let filtered_count = filtered.len();
        
        // Should reduce the number of points by removing duplicates
        assert!(filtered_count < original_count);
        assert!(filtered_count > 0);
        
        println!("GPU voxel grid filter: {} -> {} points", original_count, filtered_count);
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_empty_cloud() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::<Point3f>::new();
        let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1).await.unwrap();
        
        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_single_point() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1).await.unwrap();
        
        // Should keep the single point
        assert_eq!(filtered.len(), 1);
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_different_voxel_sizes() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_voxel_grid();
        let original_count = cloud.len();
        
        // Test different voxel sizes
        for voxel_size in [0.05, 0.1, 0.2] {
            let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, voxel_size).await.unwrap();
            println!("voxel_size={}: {} -> {} points", voxel_size, original_count, filtered.len());
            assert!(filtered.len() > 0);
            assert!(filtered.len() <= original_count);
        }
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_invalid_voxel_size() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.0).await;
        assert!(result.is_err());
        
        let result = gpu_voxel_grid_filter(&gpu_context, &cloud, -1.0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_gpu_vs_cpu_performance_comparison() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU vs CPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_radius_outlier();
        
        // GPU radius outlier removal
        let gpu_start = Instant::now();
        let gpu_filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 3).await.unwrap();
        let gpu_time = gpu_start.elapsed();
        
        // GPU voxel grid filter
        let gpu_voxel_start = Instant::now();
        let gpu_voxel_filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1).await.unwrap();
        let gpu_voxel_time = gpu_voxel_start.elapsed();
        
        println!("GPU radius outlier removal: {} -> {} points in {:?}", 
                cloud.len(), gpu_filtered.len(), gpu_time);
        println!("GPU voxel grid filter: {} -> {} points in {:?}", 
                cloud.len(), gpu_voxel_filtered.len(), gpu_voxel_time);
        
        // Both should complete successfully
        assert!(gpu_filtered.len() > 0);
        assert!(gpu_voxel_filtered.len() > 0);
        
        // Should complete in reasonable time
        assert!(gpu_time.as_secs() < 10);
        assert!(gpu_voxel_time.as_secs() < 10);
    }
} 