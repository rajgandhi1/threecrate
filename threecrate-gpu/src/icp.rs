//! GPU-accelerated ICP

use threecrate_core::{PointCloud, Result, Point3f};
use crate::GpuContext;
use nalgebra::Isometry3;

const ICP_NEAREST_NEIGHBOR_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> source_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> target_points: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> correspondences: array<u32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;
@group(0) @binding(4) var<uniform> params: ICPParams;

struct ICPParams {
    num_source: u32,
    num_target: u32,
    max_distance: f32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_source) {
        return;
    }
    
    let source_point = source_points[index];
    var min_distance = params.max_distance;
    var best_match = 0u;
    
    // Find nearest neighbor in target
    for (var i = 0u; i < params.num_target; i++) {
        let target_point = target_points[i];
        let diff = source_point - target_point;
        let distance = length(diff);
        
        if (distance < min_distance) {
            min_distance = distance;
            best_match = i;
        }
    }
    
    correspondences[index] = best_match;
    distances[index] = min_distance;
}
"#;

impl GpuContext {
    /// GPU-accelerated ICP alignment
    pub async fn icp_align(
        &self,
        source: &PointCloud<Point3f>,
        target: &PointCloud<Point3f>,
        max_iterations: usize,
        convergence_threshold: f32,
        max_correspondence_distance: f32,
    ) -> Result<Isometry3<f32>> {
        if source.is_empty() || target.is_empty() {
            return Err(threecrate_core::Error::InvalidData("Empty point clouds".to_string()));
        }

        let mut current_transform = Isometry3::identity();
        let mut transformed_source = source.clone();
        
        for _iteration in 0..max_iterations {
            // Find correspondences using GPU
            let correspondences = self.find_correspondences(
                &transformed_source.points,
                &target.points,
                max_correspondence_distance,
            ).await?;
            
            // Compute transformation on CPU (could be moved to GPU for larger datasets)
            let transform_delta = self.compute_transformation(&transformed_source.points, &target.points, &correspondences)?;
            
            // Update current transform
            current_transform = transform_delta * current_transform;
            
            // Transform source points
            transformed_source = source.clone();
            for point in &mut transformed_source.points {
                *point = current_transform.transform_point(point);
            }
            
            // Check convergence
            let translation_norm = transform_delta.translation.vector.norm();
            let rotation_angle = transform_delta.rotation.angle();
            
            if translation_norm < convergence_threshold && rotation_angle < convergence_threshold {
                break;
            }
        }
        
        Ok(current_transform)
    }
    
    /// Find nearest neighbor correspondences using GPU
    async fn find_correspondences(
        &self,
        source_points: &[Point3f],
        target_points: &[Point3f],
        max_distance: f32,
    ) -> Result<Vec<(usize, usize, f32)>> {
        // Convert points to GPU format
        let source_data: Vec<[f32; 3]> = source_points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
            
        let target_data: Vec<[f32; 3]> = target_points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();

        // Create buffers
        let source_buffer = self.create_buffer_init(
            "Source Points",
            &source_data,
            wgpu::BufferUsages::STORAGE,
        );

        let target_buffer = self.create_buffer_init(
            "Target Points", 
            &target_data,
            wgpu::BufferUsages::STORAGE,
        );

        let correspondences_buffer = self.create_buffer(
            "Correspondences",
            (source_data.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let distances_buffer = self.create_buffer(
            "Distances",
            (source_data.len() * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ICPParams {
            num_source: u32,
            num_target: u32,
            max_distance: f32,
        }

        let params = ICPParams {
            num_source: source_data.len() as u32,
            num_target: target_data.len() as u32,
            max_distance,
        };

        let params_buffer = self.create_buffer_init(
            "ICP Params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create shader and pipeline
        let shader = self.create_shader_module("ICP Nearest Neighbor", ICP_NEAREST_NEIGHBOR_SHADER);
        
        let bind_group_layout = self.create_bind_group_layout(
            "ICP Correspondence",
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ICP Correspondence"),
            layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ICP Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
        });

        let bind_group = self.create_bind_group(
            "ICP Correspondence",
            &bind_group_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: correspondences_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: distances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ICP Correspondence"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ICP Correspondence Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (source_data.len() + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Read back results
        let correspondences_staging = self.create_buffer(
            "Correspondences Staging",
            (source_data.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        let distances_staging = self.create_buffer(
            "Distances Staging",
            (source_data.len() * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        encoder.copy_buffer_to_buffer(
            &correspondences_buffer,
            0,
            &correspondences_staging,
            0,
            (source_data.len() * std::mem::size_of::<u32>()) as u64,
        );

        encoder.copy_buffer_to_buffer(
            &distances_buffer,
            0,
            &distances_staging,
            0,
            (source_data.len() * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read correspondences
        let corr_slice = correspondences_staging.slice(..);
        let dist_slice = distances_staging.slice(..);
        
        let (corr_sender, corr_receiver) = futures_intrusive::channel::shared::oneshot_channel();
        let (dist_sender, dist_receiver) = futures_intrusive::channel::shared::oneshot_channel();
        
        corr_slice.map_async(wgpu::MapMode::Read, move |v| corr_sender.send(v).unwrap());
        dist_slice.map_async(wgpu::MapMode::Read, move |v| dist_sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        if let (Some(Ok(())), Some(Ok(()))) = (corr_receiver.receive().await, dist_receiver.receive().await) {
            let corr_data = corr_slice.get_mapped_range();
            let dist_data = dist_slice.get_mapped_range();
            
            let correspondences: Vec<u32> = bytemuck::cast_slice(&corr_data).to_vec();
            let distances: Vec<f32> = bytemuck::cast_slice(&dist_data).to_vec();
            
            let result: Vec<(usize, usize, f32)> = correspondences
                .into_iter()
                .zip(distances.into_iter())
                .enumerate()
                .filter(|(_, (_, distance))| *distance < max_distance)
                .map(|(i, (target_idx, distance))| (i, target_idx as usize, distance))
                .collect();
            
            drop(corr_data);
            drop(dist_data);
            correspondences_staging.unmap();
            distances_staging.unmap();
            
            Ok(result)
        } else {
            Err(threecrate_core::Error::Gpu("Failed to read GPU correspondence results".to_string()))
        }
    }
    
    /// Compute transformation from correspondences using SVD (CPU implementation)
    fn compute_transformation(
        &self,
        source_points: &[Point3f],
        target_points: &[Point3f],
        correspondences: &[(usize, usize, f32)],
    ) -> Result<Isometry3<f32>> {
        if correspondences.is_empty() {
            return Ok(Isometry3::identity());
        }
        
        // Compute centroids
        let mut source_centroid = nalgebra::Vector3::zeros();
        let mut target_centroid = nalgebra::Vector3::zeros();
        
        for &(src_idx, tgt_idx, _) in correspondences {
            source_centroid += source_points[src_idx].coords;
            target_centroid += target_points[tgt_idx].coords;
        }
        
        let n = correspondences.len() as f32;
        source_centroid /= n;
        target_centroid /= n;
        
        // Compute cross-covariance matrix
        let mut h = nalgebra::Matrix3::zeros();
        
        for &(src_idx, tgt_idx, _) in correspondences {
            let source_centered = source_points[src_idx].coords - source_centroid;
            let target_centered = target_points[tgt_idx].coords - target_centroid;
            h += source_centered * target_centered.transpose();
        }
        
        // Compute SVD
        let svd = h.svd(true, true);
        let u = svd.u.ok_or_else(|| threecrate_core::Error::Algorithm("SVD failed".to_string()))?;
        let v_t = svd.v_t.ok_or_else(|| threecrate_core::Error::Algorithm("SVD failed".to_string()))?;
        
        let mut rotation = v_t.transpose() * u.transpose();
        
        // Ensure proper rotation (det = 1)
        if rotation.determinant() < 0.0 {
            let mut v_corrected = v_t.transpose();
            v_corrected.column_mut(2).scale_mut(-1.0);
            rotation = v_corrected * u.transpose();
        }
        
        let translation = target_centroid - rotation * source_centroid;
        
        Ok(Isometry3::from_parts(
            nalgebra::Translation3::from(translation),
            nalgebra::UnitQuaternion::from_matrix(&rotation),
        ))
    }
}

/// GPU-accelerated ICP registration
pub async fn gpu_icp(
    gpu_context: &GpuContext,
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    max_iterations: usize,
    convergence_threshold: f32,
    max_correspondence_distance: f32,
) -> Result<Isometry3<f32>> {
    gpu_context.icp_align(source, target, max_iterations, convergence_threshold, max_correspondence_distance).await
} 