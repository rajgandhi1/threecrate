use crate::device::GpuContext;
use threecrate_core::{PointCloud, ColoredPoint3f, Error, Result};
use nalgebra::{Matrix4, Point3};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// TSDF voxel data for GPU processing
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(align(16))]  // Ensure 16-byte alignment for GPU
pub struct TsdfVoxel {
    pub tsdf_value: f32,
    pub weight: f32,
    pub color_r: u32,
    pub color_g: u32,
    pub color_b: u32,
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
}

/// TSDF volume parameters
#[derive(Debug, Clone)]
pub struct TsdfVolume {
    pub voxel_size: f32,
    pub truncation_distance: f32,
    pub resolution: [u32; 3], // [width, height, depth]
    pub origin: Point3<f32>,
}

/// Represents a TSDF volume stored on the GPU.
pub struct TsdfVolumeGpu {
    pub volume: TsdfVolume,
    pub voxel_buffer: wgpu::Buffer,
}

/// Camera intrinsic parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(align(16))]  // Ensure 16-byte alignment for GPU
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
#[repr(align(16))]
pub struct TsdfParams {
    pub voxel_size: f32,
    pub truncation_distance: f32,
    pub max_weight: f32,
    pub iso_value: f32,
    pub resolution: [u32; 3],
    pub _padding2: u32,
    pub origin: [f32; 3],
    pub _padding3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(align(16))]  // Ensure 16-byte alignment for GPU
pub struct GpuPoint3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u32,
    pub g: u32,
    pub b: u32,
    pub _padding1: u32,
    pub _padding2: u32,
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
            // Convert RGB u8 data to packed u32 RGB values
            let mut packed_colors = Vec::with_capacity(color_data.len() / 3);
            for chunk in color_data.chunks_exact(3) {
                let r = chunk[0] as u32;
                let g = chunk[1] as u32;
                let b = chunk[2] as u32;
                let packed = (r << 16) | (g << 8) | b;
                packed_colors.push(packed);
            }
            
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TSDF Color Buffer"),
                contents: bytemuck::cast_slice(&packed_colors),
                usage: wgpu::BufferUsages::STORAGE,
            })
        } else {
            // Create empty buffer if no color data
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TSDF Empty Color Buffer"),
                contents: bytemuck::cast_slice(&[0u32; 4]), // Small dummy buffer
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // Initialize TSDF volume if needed
        let initial_voxels = vec![TsdfVoxel {
            tsdf_value: 1.0,
            weight: 0.0,
            color_r: 0,
            color_g: 0,
            color_b: 0,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        }; total_voxels];

        let voxel_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Voxel Buffer"),
            contents: bytemuck::cast_slice(&initial_voxels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Convert camera transform to world-to-camera matrix (inverse of camera pose)
        let world_to_camera = camera_pose.try_inverse()
            .ok_or_else(|| Error::Gpu("Failed to invert camera pose matrix".into()))?;
        
        let mut camera_transform = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                camera_transform[i][j] = world_to_camera[(i, j)];
            }
        }

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
            iso_value: 0.0,
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
        let bind_group_entries = vec![
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
            wgpu::BindGroupEntry {
                binding: 5,
                resource: color_buffer.as_entire_binding(),
            },
        ];

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
            
            // Dispatch with 4x4x4 workgroups
            let workgroup_size = 4;
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
    ) -> Result<PointCloud<ColoredPoint3f>> {
        let total_voxels = (volume.resolution[0] * volume.resolution[1] * volume.resolution[2]) as usize;
        let max_points = std::cmp::min(total_voxels, 1_000_000); // Limit to reasonable size
        
        // Create voxel buffer
        let voxel_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Voxel Buffer"),
            contents: bytemuck::cast_slice(voxels),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create output buffers
        let points_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Surface Points Buffer"),
            size: (max_points * std::mem::size_of::<GpuPoint3f>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let point_count_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Count Buffer"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params = TsdfParams {
            voxel_size: volume.voxel_size,
            truncation_distance: volume.truncation_distance,
            max_weight: 100.0,
            iso_value,
            resolution: volume.resolution,
            _padding2: 0,
            origin: [volume.origin.x, volume.origin.y, volume.origin.z],
            _padding3: 0.0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Surface Extraction Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create compute pipeline
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Surface Extraction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/surface_extraction.wgsl").into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Surface Extraction Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Surface Extraction Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: point_count_buffer.as_entire_binding(),
                },
            ],
        });

        // Create staging buffer for reading back results
        let point_count_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Point Count Staging Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Surface Extraction Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Surface Extraction Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                (volume.resolution[0] + 3) / 4,
                (volume.resolution[1] + 3) / 4,
                (volume.resolution[2] + 3) / 4,
            );
        }

        // Copy point count to staging buffer
        encoder.copy_buffer_to_buffer(
            &point_count_buffer,
            0,
            &point_count_staging_buffer,
            0,
            std::mem::size_of::<u32>() as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read point count
        let point_count_slice = point_count_staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        point_count_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap()?;

        let mapped_range = point_count_slice.get_mapped_range();
        let point_count = bytemuck::cast_slice::<u8, u32>(mapped_range.as_ref())[0] as usize;
        drop(mapped_range);
        point_count_staging_buffer.unmap();

        if point_count == 0 {
            return Ok(PointCloud { points: Vec::new() });
        }

        // Create staging buffer for points
        let points_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Points Staging Buffer"),
            size: (point_count * std::mem::size_of::<GpuPoint3f>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy points to staging buffer
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Points Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &points_buffer,
            0,
            &points_staging_buffer,
            0,
            (point_count * std::mem::size_of::<GpuPoint3f>()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read points
        let points_slice = points_staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        points_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap()?;

        let mapped_range = points_slice.get_mapped_range();
        let gpu_points = bytemuck::cast_slice::<u8, GpuPoint3f>(mapped_range.as_ref());
        let mut points = Vec::with_capacity(point_count);

        for gpu_point in gpu_points.iter().take(point_count) {
            points.push(ColoredPoint3f {
                position: Point3::new(gpu_point.x, gpu_point.y, gpu_point.z),
                color: [gpu_point.r as u8, gpu_point.g as u8, gpu_point.b as u8],
            });
        }

        drop(mapped_range);  // Explicitly drop the mapped range before unmapping
        points_staging_buffer.unmap();

        Ok(PointCloud { points })
    }
}

impl TsdfVolumeGpu {
    /// Creates a new TSDF volume on the GPU.
    pub fn new(gpu: &GpuContext, volume_params: TsdfVolume) -> Self {
        let total_voxels = (volume_params.resolution[0] * volume_params.resolution[1] * volume_params.resolution[2]) as usize;
        
        // Initialize voxels with default values
        let initial_voxels = vec![TsdfVoxel {
            tsdf_value: 1.0,
            weight: 0.0,
            color_r: 0,
            color_g: 0,
            color_b: 0,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        }; total_voxels];

        let voxel_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TSDF Voxel Buffer"),
            contents: bytemuck::cast_slice(&initial_voxels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            volume: volume_params,
            voxel_buffer,
        }
    }

    /// Integrates a depth image into the TSDF volume.
    pub async fn integrate(
        &self,
        gpu: &GpuContext,
        depth_image: &[f32],
        color_image: Option<&[u8]>, // RGB color data
        camera_pose: &Matrix4<f32>,
        intrinsics: &CameraIntrinsics,
    ) -> Result<()> {
        // Create buffers for depth, color, transform, and parameters
        let depth_buffer = gpu.create_buffer_init("TSDF Depth Buffer", depth_image, wgpu::BufferUsages::STORAGE);

        let color_buffer = if let Some(data) = color_image {
            gpu.create_buffer_init("TSDF Color Buffer", data, wgpu::BufferUsages::STORAGE)
        } else {
            // Create a dummy buffer if no color image is provided
            gpu.create_buffer_init("TSDF Dummy Color Buffer", &[0u32; 4], wgpu::BufferUsages::STORAGE)
        };

        // Convert camera transform to world-to-camera matrix (inverse of camera pose)
        let world_to_camera = camera_pose.try_inverse()
            .ok_or_else(|| Error::Gpu("Failed to invert camera pose matrix".into()))?;
        
        let mut camera_transform = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                camera_transform[i][j] = world_to_camera[(i, j)];
            }
        }

        let transform_buffer = gpu.create_buffer_init(
            "TSDF Transform Buffer",
            &[camera_transform],
            wgpu::BufferUsages::UNIFORM,
        );

        let intrinsics_buffer = gpu.create_buffer_init(
            "TSDF Intrinsics Buffer",
            &[*intrinsics],
            wgpu::BufferUsages::UNIFORM,
        );

        let params = TsdfParams {
            voxel_size: self.volume.voxel_size,
            truncation_distance: self.volume.truncation_distance,
            max_weight: 100.0,
            iso_value: 0.0,
            resolution: self.volume.resolution,
            _padding2: 0,
            origin: [self.volume.origin.x, self.volume.origin.y, self.volume.origin.z],
            _padding3: 0.0,
        };
        let params_buffer = gpu.create_buffer_init("TSDF Params Buffer", &[params], wgpu::BufferUsages::UNIFORM);

        // Create compute pipeline
        let shader = gpu.create_shader_module("TSDF Integration Shader", include_str!("shaders/tsdf_integration.wgsl"));
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TSDF Integration Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Create bind group
        let bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.voxel_buffer.as_entire_binding(),
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
            wgpu::BindGroupEntry {
                binding: 5,
                resource: color_buffer.as_entire_binding(),
            },
        ];

        let bind_group = gpu.create_bind_group("TSDF Integration Bind Group", &pipeline.get_bind_group_layout(0), &bind_group_entries);

        // Dispatch compute shader
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TSDF Integration Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TSDF Integration Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with 4x4x4 workgroups
            let workgroup_size = 4;
            let dispatch_x = (self.volume.resolution[0] + workgroup_size - 1) / workgroup_size;
            let dispatch_y = (self.volume.resolution[1] + workgroup_size - 1) / workgroup_size;
            let dispatch_z = (self.volume.resolution[2] + workgroup_size - 1) / workgroup_size;
            
            println!("Dispatching compute shader with {} x {} x {} workgroups", dispatch_x, dispatch_y, dispatch_z);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Downloads the TSDF voxel data from the GPU.
    pub async fn download_voxels(&self, gpu: &GpuContext) -> Result<Vec<TsdfVoxel>> {
        let total_voxels = (self.volume.resolution[0] * self.volume.resolution[1] * self.volume.resolution[2]) as usize;
        let buffer_size = (total_voxels * std::mem::size_of::<TsdfVoxel>()) as u64;

        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TSDF Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TSDF Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.voxel_buffer,
            0,
            &staging_buffer,
            0,
            buffer_size,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::unbounded();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        gpu.device.poll(wgpu::Maintain::Wait);
        receiver.recv_async().await.map_err(|_| Error::Gpu("Failed to receive mapping result".into()))??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<TsdfVoxel> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Extract point cloud from TSDF volume using marching cubes
    pub async fn extract_surface(&self, gpu: &GpuContext, iso_value: f32) -> Result<PointCloud<ColoredPoint3f>> {
        let voxels = self.download_voxels(gpu).await?;
        gpu.tsdf_extract_surface(&self.volume, &voxels, iso_value).await
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
) -> Result<PointCloud<ColoredPoint3f>> {
    gpu_context.tsdf_extract_surface(volume, voxels, iso_value).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuContext;
    use nalgebra::{Matrix4, Point3};
    use approx::assert_relative_eq;

    /// Try to create a GPU context, return None if not available
    async fn try_create_gpu_context() -> Option<GpuContext> {
        match GpuContext::new().await {
            Ok(gpu) => Some(gpu),
            Err(_) => {
                println!("⚠️  GPU not available, skipping GPU-dependent test");
                None
            }
        }
    }

    /// Create simple depth image for basic testing
    fn create_simple_depth_image(width: u32, height: u32, depth: f32) -> Vec<f32> {
        vec![depth; (width * height) as usize]
    }

    /// Create a basic camera setup for testing
    fn create_test_camera() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
            width: 640,
            height: 480,
            depth_scale: 1.0,
            _padding: 0.0,
        }
    }

    /// Create identity camera pose
    fn create_identity_pose() -> Matrix4<f32> {
        Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[test]
    fn test_tsdf_basic_integration() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            // Create a simple TSDF volume
            let voxel_size = 0.02; // 2cm voxels for faster processing
            let truncation_distance = 0.1; 
            let resolution = [32, 32, 32]; // Smaller resolution for speed
            let origin = Point3::new(-0.32, -0.32, 0.0);

            let volume_params = create_tsdf_volume(
                voxel_size,
                truncation_distance,
                resolution,
                origin,
            );
            let tsdf_volume_gpu = TsdfVolumeGpu::new(&gpu, volume_params);

            // Create simple depth image with constant depth
            let intrinsics = create_test_camera();
            let depth_image = create_simple_depth_image(intrinsics.width, intrinsics.height, 0.5);
            let camera_pose = create_identity_pose();

            // Test integration
            let result = tsdf_volume_gpu.integrate(&gpu, &depth_image, None, &camera_pose, &intrinsics).await;
            assert!(result.is_ok(), "TSDF integration should succeed");

            // Test voxel download
            let voxels = tsdf_volume_gpu.download_voxels(&gpu).await.unwrap();
            assert_eq!(voxels.len(), (32 * 32 * 32) as usize, "Should have correct number of voxels");

            // Check that some voxels have been updated
            let updated_voxels = voxels.iter().filter(|v| v.weight > 0.0).count();
            assert!(updated_voxels > 0, "Some voxels should have been updated");

            println!("✓ Basic integration test passed: {} voxels updated", updated_voxels);
        });
    }

    #[test]
    fn test_tsdf_surface_extraction() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            // Create TSDF volume 
            let voxel_size = 0.02;
            let truncation_distance = 0.1;
            let resolution = [32, 32, 32];
            let origin = Point3::new(-0.32, -0.32, 0.0);

            let volume_params = create_tsdf_volume(
                voxel_size,
                truncation_distance,
                resolution,
                origin,
            );
            let tsdf_volume_gpu = TsdfVolumeGpu::new(&gpu, volume_params);

            // Integrate a simple depth image
            let intrinsics = create_test_camera();
            let depth_image = create_simple_depth_image(intrinsics.width, intrinsics.height, 0.3);
            let camera_pose = create_identity_pose();

            tsdf_volume_gpu.integrate(&gpu, &depth_image, None, &camera_pose, &intrinsics)
                .await
                .unwrap();

            // Extract surface
            let point_cloud = tsdf_volume_gpu.extract_surface(&gpu, 0.0).await.unwrap();
            
            // Should extract some points
            assert!(!point_cloud.points.is_empty(), "Should extract surface points");
            
            // Points should be in reasonable Z range around the depth value
            let avg_z = point_cloud.points.iter()
                .map(|p| p.position.z)
                .sum::<f32>() / point_cloud.points.len() as f32;
            
            assert!(avg_z > 0.2 && avg_z < 0.4, "Average Z should be near depth value of 0.3");
            
            println!("✓ Surface extraction test passed: {} points extracted, avg Z: {:.3}", 
                     point_cloud.points.len(), avg_z);
        });
    }

    #[test]
    fn test_tsdf_multiple_integrations() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            // Create TSDF volume
            let voxel_size = 0.02;
            let truncation_distance = 0.1;
            let resolution = [32, 32, 32];
            let origin = Point3::new(-0.32, -0.32, 0.0);

            let volume_params = create_tsdf_volume(
                voxel_size,
                truncation_distance,
                resolution,
                origin,
            );
            let tsdf_volume_gpu = TsdfVolumeGpu::new(&gpu, volume_params);

            let intrinsics = create_test_camera();
            let camera_pose = create_identity_pose();

            // Integrate multiple depth images
            let depths = [0.25, 0.3, 0.35];
            for &depth in &depths {
                let depth_image = create_simple_depth_image(intrinsics.width, intrinsics.height, depth);
                tsdf_volume_gpu.integrate(&gpu, &depth_image, None, &camera_pose, &intrinsics)
                    .await
                    .unwrap();
            }

            // Check voxel weights have increased
            let voxels = tsdf_volume_gpu.download_voxels(&gpu).await.unwrap();
            let max_weight = voxels.iter().map(|v| v.weight).fold(0.0, f32::max);
            assert!(max_weight > 1.0, "Multiple integrations should increase voxel weights");

            // Extract surface
            let point_cloud = tsdf_volume_gpu.extract_surface(&gpu, 0.0).await.unwrap();
            assert!(!point_cloud.points.is_empty(), "Should extract surface after multiple integrations");

            println!("✓ Multiple integration test passed: max weight {:.1}, {} points extracted", 
                     max_weight, point_cloud.points.len());
        });
    }

    #[test]
    fn test_tsdf_coordinate_system() {
        pollster::block_on(async {
            let Some(_gpu) = try_create_gpu_context().await else {
                return;
            };

            // Test basic coordinate system consistency
            let voxel_size = 0.02;
            let resolution = [32, 32, 32];
            let origin = Point3::new(-0.32, -0.32, 0.0);

            // Check volume bounds
            let max_coord = Point3::new(
                origin.x + (resolution[0] as f32) * voxel_size,
                origin.y + (resolution[1] as f32) * voxel_size,
                origin.z + (resolution[2] as f32) * voxel_size,
            );

            assert_relative_eq!(max_coord.x, 0.32, epsilon = 0.01);
            assert_relative_eq!(max_coord.y, 0.32, epsilon = 0.01);
            assert_relative_eq!(max_coord.z, 0.64, epsilon = 0.01);

            // Test camera transforms
            let camera_pose = create_identity_pose();
            let world_to_camera = camera_pose.try_inverse().unwrap();
            
            let test_point = Point3::new(0.1, 0.2, 0.3);
            let camera_point = world_to_camera.transform_point(&test_point);
            
            // For identity transform, should be the same
            assert_relative_eq!(test_point.x, camera_point.x, epsilon = 0.001);
            assert_relative_eq!(test_point.y, camera_point.y, epsilon = 0.001);
            assert_relative_eq!(test_point.z, camera_point.z, epsilon = 0.001);

            println!("✓ Coordinate system test passed");
        });
    }

    #[test]
    fn test_tsdf_color_integration() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            // Create TSDF volume
            let voxel_size = 0.02;
            let truncation_distance = 0.1;
            let resolution = [32, 32, 32];
            let origin = Point3::new(-0.32, -0.32, 0.0);

            let volume_params = create_tsdf_volume(
                voxel_size,
                truncation_distance,
                resolution,
                origin,
            );
            let tsdf_volume_gpu = TsdfVolumeGpu::new(&gpu, volume_params);

            // Create depth and color images
            let intrinsics = create_test_camera();
            let depth_image = create_simple_depth_image(intrinsics.width, intrinsics.height, 0.3);
            
            // Simple red color image
            let pixel_count = (intrinsics.width * intrinsics.height) as usize;
            let mut color_image = Vec::with_capacity(pixel_count * 3);
            for _ in 0..pixel_count {
                color_image.extend_from_slice(&[255u8, 0u8, 0u8]); // RGB: red
            }
            
            let camera_pose = create_identity_pose();

            // Integrate with color
            tsdf_volume_gpu.integrate(&gpu, &depth_image, Some(&color_image), &camera_pose, &intrinsics)
                .await
                .unwrap();

            // Extract surface
            let point_cloud = tsdf_volume_gpu.extract_surface(&gpu, 0.0).await.unwrap();
            
            assert!(!point_cloud.points.is_empty(), "Should extract colored surface points");
            
            // Check that some points have red color
            let red_points = point_cloud.points.iter()
                .filter(|p| p.color[0] > 200)
                .count();
            
            assert!(red_points > 0, "Some points should have red color");

            println!("✓ Color integration test passed: {} red points", red_points);
        });
    }
}  