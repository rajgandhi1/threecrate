# ThreeCrate GPU

[![Crates.io](https://img.shields.io/crates/v/threecrate-gpu.svg)](https://crates.io/crates/threecrate-gpu)
[![Documentation](https://docs.rs/threecrate-gpu/badge.svg)](https://docs.rs/threecrate-gpu)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

GPU-accelerated algorithms for 3D point cloud processing using wgpu.

## Features

- **Filtering**: GPU statistical outlier removal, radius filtering, voxel grid downsampling
- **Normal Estimation**: Parallel GPU-based surface normal computation
- **Nearest Neighbor Search**: GPU K-nearest and radius neighbor search
- **Segmentation**: GPU-scored RANSAC plane segmentation and GPU-accelerated Euclidean clustering
- **ICP Registration**: GPU-accelerated Iterative Closest Point
- **TSDF**: Truncated Signed Distance Function volume integration and surface extraction
- **Rendering**: Real-time point cloud and mesh rendering with PBR material support
- **Cross-platform**: Vulkan, Metal, and DirectX 12 via wgpu

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-gpu = "0.6.0"
threecrate-core = { version = "0.6.0", features = ["gpu"] }
```

## Example

```rust
use threecrate_gpu::{GpuContext, PointCloudRenderer, RenderConfig, point_cloud_to_vertices};
use threecrate_core::{PointCloud, Point3f};

// Initialize GPU context
let gpu_context = GpuContext::new().await?;

// GPU-accelerated filtering
let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.05).await?;

// GPU RANSAC plane segmentation
let plane_config = GpuPlaneSegmentationConfig {
    max_iterations: 1000,
    distance_threshold: 0.01,
    min_inliers: 100,
};
let plane = gpu_segment_plane(&gpu_context, &cloud, plane_config).await?;

// GPU-accelerated Euclidean clustering
let cluster_config = GpuClusterConfig::new(0.02, 100, 25000);
let clusters = gpu_extract_clusters(&gpu_context, &cloud, cluster_config).await?;

// GPU ICP registration
let result = gpu_icp(&gpu_context, &source, &target).await?;

// TSDF volume integration
let mut volume = create_tsdf_volume(resolution, voxel_size);
gpu_tsdf_integrate(&gpu_context, &mut volume, &depth_image, &intrinsics, &pose).await?;
let mesh = gpu_tsdf_extract_surface(&gpu_context, &volume).await?;

// Real-time rendering
let config = RenderConfig::default();
let renderer = PointCloudRenderer::new(&window, config).await?;
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
