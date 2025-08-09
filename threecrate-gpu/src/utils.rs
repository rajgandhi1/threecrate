//! GPU utilities

use threecrate_core::Result;

/// Create a compute shader from WGSL source
pub fn create_compute_shader(_device: &wgpu::Device, _source: &str) -> Result<wgpu::ShaderModule> {
    // Note: Shader module creation is infallible; validation happens at pipeline creation time
    let module = _device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ThreeCrate Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(_source.into()),
    });
    Ok(module)
} 