# ThreeCrate GPU

[![Crates.io](https://img.shields.io/crates/v/threecrate-gpu.svg)](https://crates.io/crates/threecrate-gpu)
[![Documentation](https://docs.rs/threecrate-gpu/badge.svg)](https://docs.rs/threecrate-gpu)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/3DCrate#license)

GPU-accelerated algorithms for 3D point cloud processing using WGPU.

## Features

- **GPU Computing**: Hardware-accelerated point cloud processing
- **Real-time Rendering**: Point cloud and mesh visualization
- **Parallel Algorithms**: Massively parallel GPU implementations
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Modern Graphics**: Uses WGPU for cross-platform GPU access

## Algorithms

- **Point Cloud Rendering**: Real-time visualization with splatting
- **Normal Estimation**: GPU-accelerated surface normal computation
- **Filtering**: Parallel outlier removal and downsampling
- **ICP Registration**: GPU-accelerated point cloud alignment
- **TSDF Integration**: Truncated Signed Distance Function processing

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-gpu = "0.1.0"
threecrate-core = { version = "0.1.0", features = ["gpu"] }
```

## Example

```rust
use threecrate_gpu::{GpuContext, PointCloudRenderer, RenderConfig};
use threecrate_core::{PointCloud, Point3f};

// Initialize GPU context
let gpu_context = GpuContext::new().await?;

// Create point cloud renderer
let config = RenderConfig::default();
let renderer = PointCloudRenderer::new(&window, config).await?;

// Render point cloud
let cloud = PointCloud::from_points(vec![/* points */]);
let vertices = point_cloud_to_vertices(&cloud, [1.0, 1.0, 1.0], 4.0);
renderer.render(&vertices)?;
```

## GPU Requirements

- **Vulkan**: Preferred backend for best performance
- **Metal**: macOS support via Metal API
- **DirectX 12**: Windows support via DX12
- **OpenGL**: Fallback support for older systems

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 