# ThreeCrate Visualization

[![Crates.io](https://img.shields.io/crates/v/threecrate-visualization.svg)](https://crates.io/crates/threecrate-visualization)
[![Documentation](https://docs.rs/threecrate-visualization/badge.svg)](https://docs.rs/threecrate-visualization)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/3DCrate#license)

Real-time 3D visualization and interactive viewing for point clouds and meshes.

## Features

- **Interactive Viewer**: Real-time 3D visualization with camera controls
- **Point Cloud Rendering**: GPU-accelerated point cloud display
- **Mesh Rendering**: Triangle mesh visualization with lighting
- **Camera Controls**: Orbit, pan, zoom, and reset functionality
- **Cross-platform**: Works on Windows, macOS, and Linux via winit and wgpu

## Camera Controls

- **Mouse Drag**: Orbit around the scene
- **Mouse Scroll**: Zoom in/out
- **Keyboard Shortcuts**:
  - `O`: Switch to orbit mode
  - `P`: Switch to pan mode
  - `Z`: Switch to zoom mode
  - `R`: Reset camera position

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-visualization = "0.1.0"
threecrate-core = "0.1.0"
```

## Example

```rust
use threecrate_visualization::InteractiveViewer;
use threecrate_core::{PointCloud, Point3f};

// Create point cloud
let points = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.0, 1.0, 0.0),
];
let cloud = PointCloud::from_points(points);

// Create and run viewer
let mut viewer = InteractiveViewer::new()?;
viewer.set_point_cloud(&cloud);
viewer.run()?;
```

## Architecture

- **Camera System**: Flexible camera with spherical coordinates
- **GPU Rendering**: Hardware-accelerated rendering via wgpu
- **Event Handling**: Responsive input handling with winit
- **Cross-platform**: Native windowing and input across platforms

## Requirements

- **GPU**: Modern graphics card with Vulkan, Metal, or DirectX 12 support
- **RAM**: Minimum 2GB for large point clouds
- **OS**: Windows 10+, macOS 10.15+, or Linux with graphics drivers

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 