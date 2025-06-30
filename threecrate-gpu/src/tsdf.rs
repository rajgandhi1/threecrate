use crate::device::GpuContext;
use threecrate_core::{PointCloud, Point3f, Error, Result};
use nalgebra::{Matrix4, Point3};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// TSDF voxel data for GPU processing
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TsdfVoxel {
    pub tsdf_value: f32,
    pub weight: f32,
    pub color_r: u8,
    pub color_g: u8, 
    pub color_b: u8,
    pub _padding: u8,
}

/// TSDF volume parameters
#[derive(Debug, Clone)]
pub struct TsdfVolume {
    pub voxel_size: f32,
    pub truncation_distance: f32,
    pub resolution: [u32; 3], // [width, height, depth]
    pub origin: Point3<f32>,
}

/// Camera intrinsic parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    pub depth_scale: f32,
    pub _padding: f32,
}

/// TSDF integration parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TsdfParams {
    pub voxel_size: f32,
    pub truncation_distance: f32,
    pub max_weight: f32,
    pub _padding: f32,
    pub resolution: [u32; 3],
    pub _padding2: u32,
    pub origin: [f32; 3],
    pub _padding3: f32,
}

impl GpuContext {
    /// Integrate depth image into TSDF volume
    pub async fn tsdf_integrate(
        &self,
        volume: &mut TsdfVolume,
        depth_image: &[f32],
        color_image: Option<&[u8]>, // RGB color data
        camera_pose: &Matrix4<f32>,
        intrinsics: &CameraIntrinsics,
    ) -> Result<Vec<TsdfVoxel>> {
        let total_voxels = (volume.resolution[0] * volume.resolution[1] * volume.resolution[2]) as usize;
        
        // Create buffers
        let depth_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Depth Buffer"),
            contents: bytemuck::cast_slice(depth_image),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let color_buffer = if let Some(color_data) = color_image {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TSDF Color Buffer"),
                contents: color_data,
                usage: wgpu::BufferUsages::STORAGE,
            }))
        } else {
            None
        };

        // Initialize TSDF volume if needed
        let initial_voxels = vec![TsdfVoxel {
            tsdf_value: 1.0,
            weight: 0.0,
            color_r: 0,
            color_g: 0,
            color_b: 0,
            _padding: 0,
        }; total_voxels];

        let voxel_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Voxel Buffer"),
            contents: bytemuck::cast_slice(&initial_voxels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Camera transform (world to camera)
        let camera_transform: [[f32; 4]; 4] = camera_pose.try_inverse()
            .unwrap_or(*camera_pose)
            .into();

        let transform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Transform Buffer"),
            contents: bytemuck::cast_slice(&[camera_transform]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let intrinsics_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Intrinsics Buffer"),
            contents: bytemuck::bytes_of(intrinsics),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params = TsdfParams {
            voxel_size: volume.voxel_size,
            truncation_distance: volume.truncation_distance,
            max_weight: 100.0,
            _padding: 0.0,
            resolution: volume.resolution,
            _padding2: 0,
            origin: [volume.origin.x, volume.origin.y, volume.origin.z],
            _padding3: 0.0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create compute pipeline
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TSDF Integration Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tsdf_integration.wgsl").into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TSDF Integration Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Create bind group
        let mut bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: voxel_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: depth_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: transform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: intrinsics_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ];

        if let Some(ref color_buf) = color_buffer {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: color_buf.as_entire_binding(),
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TSDF Integration Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bind_group_entries,
        });

        // Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TSDF Integration Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TSDF Integration Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with 8x8x8 workgroups
            let workgroup_size = 8;
            let dispatch_x = (volume.resolution[0] + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (volume.resolution[1] + workgroup_size - 1) / workgroup_size;
            let dispatch_z = (volume.resolution[2] + workgroup_size - 1) / workgroup_size;
            
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSDF Staging Buffer"),
            size: (total_voxels * std::mem::size_of::<TsdfVoxel>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &voxel_buffer,
            0,
            &staging_buffer,
            0,
            staging_buffer.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::unbounded();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv_async().await.map_err(|_| Error::Gpu("Failed to receive mapping result".into()))?
            .map_err(|e| Error::Gpu(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<TsdfVoxel> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Extract point cloud from TSDF volume using marching cubes
    pub async fn tsdf_extract_surface(
        &self,
        volume: &TsdfVolume,
        voxels: &[TsdfVoxel],
        iso_value: f32,
    ) -> Result<PointCloud<Point3f>> {
        // Create buffers
        let voxel_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Voxel Buffer"),
            contents: bytemuck::cast_slice(voxels),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Estimate maximum number of vertices (conservative)
        let max_vertices = (volume.resolution[0] * volume.resolution[1] * volume.resolution[2] * 3) as usize;
        let vertices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSDF Vertices Buffer"),
            size: (max_vertices * std::mem::size_of::<Point3f>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Counter for actual vertices written
        let counter_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Counter Buffer"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params = TsdfParams {
            voxel_size: volume.voxel_size,
            truncation_distance: volume.truncation_distance,
            max_weight: 100.0,
            _padding: 0.0,
            resolution: volume.resolution,
            _padding2: 0,
            origin: [volume.origin.x, volume.origin.y, volume.origin.z],
            _padding3: 0.0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let iso_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Iso Value Buffer"),
            contents: bytemuck::bytes_of(&iso_value),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create compute pipeline for surface extraction
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TSDF Surface Extraction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tsdf_surface_extraction.wgsl").into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TSDF Surface Extraction Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TSDF Surface Extraction Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vertices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: iso_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch surface extraction
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TSDF Surface Extraction Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TSDF Surface Extraction Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_size = 8;
            let dispatch_x = (volume.resolution[0] + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (volume.resolution[1] + workgroup_size - 1) / workgroup_size;
            let dispatch_z = (volume.resolution[2] + workgroup_size - 1) / workgroup_size;
            
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        // Read back vertex count
        let counter_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counter Staging Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&counter_buffer, 0, &counter_staging, 0, 4);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Get vertex count
        let buffer_slice = counter_staging.slice(..);
        let (sender, receiver) = flume::unbounded();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv_async().await.map_err(|_| Error::Gpu("Failed to receive mapping result".into()))?
            .map_err(|e| Error::Gpu(format!("Buffer mapping failed: {:?}", e)))?;

        let count_data = buffer_slice.get_mapped_range();
        let vertex_count = bytemuck::cast_slice::<u8, u32>(&count_data)[0] as usize;
        drop(count_data);
        counter_staging.unmap();

        if vertex_count == 0 {
            return Ok(PointCloud::new());
        }

        // Read back vertices
        let vertices_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertices Staging Buffer"),
            size: (vertex_count * std::mem::size_of::<Point3f>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TSDF Copy Vertices Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &vertices_buffer,
            0,
            &vertices_staging,
            0,
            vertices_staging.size(),
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = vertices_staging.slice(..);
        let (sender, receiver) = flume::unbounded();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv_async().await.map_err(|_| Error::Gpu("Failed to receive mapping result".into()))?
            .map_err(|e| Error::Gpu(format!("Buffer mapping failed: {:?}", e)))?;

        let vertices_data = buffer_slice.get_mapped_range();
        let vertex_data: &[[f32; 3]] = bytemuck::cast_slice(&vertices_data);
        let vertices: Vec<Point3f> = vertex_data.iter()
            .map(|&[x, y, z]| Point3f::new(x, y, z))
            .collect();
        
        drop(vertices_data);
        vertices_staging.unmap();

        let mut point_cloud = PointCloud::new();
        point_cloud.points = vertices;

        Ok(point_cloud)
    }
}

/// Create a new TSDF volume with specified parameters
pub fn create_tsdf_volume(
    voxel_size: f32,
    truncation_distance: f32,
    resolution: [u32; 3],
    origin: Point3<f32>,
) -> TsdfVolume {
    TsdfVolume {
        voxel_size,
        truncation_distance,
        resolution,
        origin,
    }
}

/// GPU-accelerated TSDF integration from depth image
pub async fn gpu_tsdf_integrate(
    gpu_context: &GpuContext,
    volume: &mut TsdfVolume,
    depth_image: &[f32],
    color_image: Option<&[u8]>,
    camera_pose: &Matrix4<f32>,
    intrinsics: &CameraIntrinsics,
) -> Result<Vec<TsdfVoxel>> {
    gpu_context.tsdf_integrate(volume, depth_image, color_image, camera_pose, intrinsics).await
}

/// GPU-accelerated surface extraction from TSDF volume
pub async fn gpu_tsdf_extract_surface(
    gpu_context: &GpuContext,
    volume: &TsdfVolume,
    voxels: &[TsdfVoxel],
    iso_value: f32,
) -> Result<PointCloud<Point3f>> {
    gpu_context.tsdf_extract_surface(volume, voxels, iso_value).await
} 