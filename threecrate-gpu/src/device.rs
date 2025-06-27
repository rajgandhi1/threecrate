//! GPU device management

use threecrate_core::Result;
use wgpu::util::DeviceExt;

/// GPU context for managing compute and rendering operations
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl GpuContext {
    /// Create a new GPU context
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| threecrate_core::Error::Gpu("Failed to find suitable adapter".to_string()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ThreeCrate GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| threecrate_core::Error::Gpu(format!("Failed to create device: {}", e)))?;

        Ok(Self {
            device,
            queue,
            adapter,
        })
    }

    /// Create a buffer from data
    pub fn create_buffer_init<T: bytemuck::Pod>(&self, label: &str, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }

    /// Create an empty buffer
    pub fn create_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a compute pipeline
    pub fn create_compute_pipeline(&self, label: &str, shader: &wgpu::ShaderModule, entry_point: &str) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: shader,
            entry_point,
        })
    }

    /// Create a shader module from WGSL source
    pub fn create_shader_module(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    /// Create a bind group layout
    pub fn create_bind_group_layout(&self, label: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
    }

    /// Create a bind group
    pub fn create_bind_group(&self, label: &str, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        })
    }
} 