//! GPU device management

use std::sync::OnceLock;
use threecrate_core::{Error, Result};
use wgpu::util::DeviceExt;

/// Compute pipelines that are built once per context and reused across calls.
/// Building a pipeline compiles its shader, which costs far more than running
/// it on small inputs.
#[derive(Default)]
pub(crate) struct PipelineCache {
    pub(crate) knn: OnceLock<wgpu::ComputePipeline>,
    pub(crate) normals: OnceLock<wgpu::ComputePipeline>,
    pub(crate) icp_match: OnceLock<wgpu::ComputePipeline>,
    pub(crate) icp_match_plane: OnceLock<wgpu::ComputePipeline>,
    pub(crate) radius_outlier: OnceLock<wgpu::ComputePipeline>,
}

/// GPU context for managing compute and rendering operations
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    pub(crate) pipelines: PipelineCache,
}

impl GpuContext {
    /// Create a new GPU context
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                threecrate_core::Error::Gpu(format!("Failed to find suitable adapter: {:?}", e))
            })?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ThreeCrate GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| threecrate_core::Error::Gpu(format!("Failed to create device: {}", e)))?;

        Ok(Self::from_parts(instance, adapter, device, queue))
    }

    /// Wrap a device the caller already created (for example to share it with
    /// a renderer). Pipelines are compiled on first use, as with [`Self::new`].
    pub fn from_parts(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        Self {
            instance,
            device,
            queue,
            adapter,
            pipelines: PipelineCache::default(),
        }
    }

    /// Return the pipeline in `slot`, compiling `source` on first use. The bind
    /// group layout is derived from the shader (`layout: None`).
    pub(crate) fn cached_pipeline<'a>(
        &self,
        slot: &'a OnceLock<wgpu::ComputePipeline>,
        label: &str,
        source: impl FnOnce() -> String,
        entry_point: &str,
    ) -> &'a wgpu::ComputePipeline {
        slot.get_or_init(|| {
            let shader = self.create_shader_module(label, &source());
            self.create_compute_pipeline(label, &shader, entry_point)
        })
    }

    /// Copy `buffer` (which needs `COPY_SRC`) to the CPU and return its contents
    /// as `T`s. Blocks until the GPU has finished all submitted work. For
    /// repeated reads of the same size, keep a staging buffer and use
    /// [`Self::read_staging`] instead.
    pub(crate) fn read_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer) -> Result<Vec<T>> {
        let staging = self.create_staging_buffer("Readback", buffer.size());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit(std::iter::once(encoder.finish()));
        self.read_staging(&staging)
    }

    /// A `MAP_READ` buffer that GPU results can be copied into.
    pub(crate) fn create_staging_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.create_buffer(
            label,
            size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )
    }

    /// Read a staging buffer whose copy has already been submitted, and unmap
    /// it again so it can be reused. Blocks until the GPU has finished.
    pub(crate) fn read_staging<T: bytemuck::Pod>(&self, staging: &wgpu::Buffer) -> Result<Vec<T>> {
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| Error::Gpu(format!("GPU poll failed: {e}")))?;
        receiver
            .recv()
            .map_err(|_| Error::Gpu("GPU readback was dropped".to_string()))?
            .map_err(|e| Error::Gpu(format!("Failed to map GPU buffer: {e}")))?;

        // `pod_collect_to_vec` copies into a correctly aligned Vec; the mapped
        // range itself is only guaranteed 8-byte alignment.
        let data = bytemuck::pod_collect_to_vec(&slice.get_mapped_range());
        staging.unmap();
        Ok(data)
    }

    /// Create a buffer from data
    pub fn create_buffer_init<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    pub fn create_compute_pipeline(
        &self,
        label: &str,
        shader: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }

    /// Create a shader module from WGSL source
    pub fn create_shader_module(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    /// Create a bind group layout
    pub fn create_bind_group_layout(
        &self,
        label: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::BindGroupLayout {
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
    }

    /// Create a bind group
    pub fn create_bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        })
    }
}
