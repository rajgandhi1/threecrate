# ThreeCrate Visualization

[![Crates.io](https://img.shields.io/crates/v/threecrate-visualization.svg)](https://crates.io/crates/threecrate-visualization)
[![Documentation](https://docs.rs/threecrate-visualization/badge.svg)](https://docs.rs/threecrate-visualization)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

Real-time 3D visualization and interactive viewing for point clouds and meshes.

## Features

- `show_point_cloud()` and `show_mesh()` convenience functions for quick display
- `InteractiveViewer` for full control over the viewing session
- GPU-accelerated rendering via wgpu
- Triangle mesh rendering with lighting
- Spherical coordinate camera with orbit, pan, and zoom modes
- Cross-platform (Windows, macOS, Linux) via winit and wgpu

## Camera Controls

- **Mouse Drag**: Orbit around the scene
- **Mouse Scroll**: Zoom in/out
- **O**: Switch to orbit mode
- **P**: Switch to pan mode
- **Z**: Switch to zoom mode
- **R**: Reset camera position

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-visualization = "0.6.0"
threecrate-core = "0.6.0"
```

## Example

```rust
use threecrate_visualization::{show_point_cloud, show_mesh, InteractiveViewer};
use threecrate_core::{PointCloud, Point3f};

// Quick display
show_point_cloud(&cloud)?;
show_mesh(&mesh)?;

// Full control with InteractiveViewer
let mut viewer = InteractiveViewer::new()?;
viewer.set_point_cloud(&cloud);
viewer.run()?;
```

## Requirements

- A GPU with Vulkan, Metal, or DirectX 12 support
- Windows 10+, macOS 10.15+, or Linux with appropriate graphics drivers

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
