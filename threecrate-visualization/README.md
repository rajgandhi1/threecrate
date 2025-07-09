# ThreeCrate Interactive Viewer

A comprehensive interactive 3D viewer for point clouds and meshes with real-time rendering and algorithm controls.

## Features

- **Real-time 3D Rendering**: GPU-accelerated rendering of point clouds and meshes
- **Interactive UI Controls**: Comprehensive UI panels for all settings and parameters
- **Camera Navigation**: Smooth orbit, pan, and zoom controls
- **Algorithm Integration**: Interactive controls for ICP, RANSAC, and other algorithms
- **File I/O**: Load and save point clouds and meshes in various formats
- **Screenshot Capture**: Save high-quality screenshots of your visualizations
- **CPU/GPU Pipeline Switching**: Choose between CPU and GPU processing pipelines

## Dependencies

The interactive viewer depends on:

- **winit** - Window management and input handling
- **wgpu** - Modern GPU graphics API
- **egui** - Immediate mode GUI library
- **nalgebra** - Linear algebra operations
- **image** - Image processing for screenshots
- **rfd** - Native file dialogs

## Quick Start

### Basic Usage

```rust
use threecrate_visualization::{InteractiveViewer, show_point_cloud};
use threecrate_core::{PointCloud, Point3f};

// Create a point cloud
let mut cloud = PointCloud::new();
cloud.push(Point3f::new(0.0, 0.0, 0.0));
cloud.push(Point3f::new(1.0, 0.0, 0.0));
cloud.push(Point3f::new(0.0, 1.0, 0.0));

// Show it in the viewer
show_point_cloud(&cloud)?;
```

### Advanced Usage

```rust
use threecrate_visualization::InteractiveViewer;
use threecrate_core::{PointCloud, Point3f, ColoredPoint3f};

// Create viewer
let mut viewer = InteractiveViewer::new()?;

// Set data
viewer.set_point_cloud(&cloud);
// Or: viewer.set_colored_point_cloud(&colored_cloud);
// Or: viewer.set_mesh(&mesh);

// Run the viewer
viewer.run()?;
```

## Controls

### Mouse Controls
- **Left Click + Drag**: Orbit camera around target
- **Mouse Wheel**: Zoom in/out
- **Middle Click + Drag**: Pan camera (when in pan mode)

### Keyboard Controls
- **O**: Switch to orbit mode
- **P**: Switch to pan mode
- **Z**: Switch to zoom mode
- **R**: Reset camera to fit all data
- **S**: Take screenshot
- **F1**: Toggle render settings panel
- **F2**: Toggle algorithms panel
- **F3**: Toggle camera info panel
- **F4**: Toggle statistics panel

## UI Panels

### Render Settings (F1)
- **Point Size**: Adjust point rendering size
- **Alpha Threshold**: Control point transparency
- **Enable Splatting**: Toggle point splatting rendering
- **Lighting Controls**: Ambient, diffuse, and specular lighting
- **Background Color**: Change background color

### Algorithms (F2)
- **ICP Registration**: Interactive Iterative Closest Point
  - Max iterations
  - Convergence threshold
  - Maximum correspondence distance
- **RANSAC Plane Segmentation**: Random Sample Consensus
  - Max iterations
  - Distance threshold
  - View results and statistics

### Camera Info (F3)
- View current camera position and orientation
- Camera parameters (FOV, aspect ratio)
- Current camera mode
- Reset camera button

### Statistics (F4)
- Real-time performance metrics
- Frame rate and frame time
- Data statistics (point count, face count)

## Example Applications

### Basic Point Cloud Viewer
```bash
cargo run --bin interactive_viewer_example
```

Choose option 1 for a basic point cloud example.

### Colored Point Cloud
```bash
cargo run --bin interactive_viewer_example
```

Choose option 2 for a colored point cloud example.

### Mesh Rendering
```bash
cargo run --bin interactive_viewer_example
```

Choose option 3 for a mesh rendering example.

### Algorithm Demo
```bash
cargo run --bin interactive_viewer_example
```

Choose option 4 for an algorithm demonstration with RANSAC plane segmentation.

### Custom Data
```bash
cargo run --bin interactive_viewer_example
```

Choose option 5 to start with an empty viewer and load your own data files.

## Supported File Formats

### Point Clouds
- **PLY** (Stanford Polygon Library)
- **LAS/LAZ** (via pasture integration - coming soon)
- **PCD** (Point Cloud Data - coming soon)

### Meshes
- **OBJ** (Wavefront OBJ)
- **PLY** (Stanford Polygon Library)

## Pipeline Types

### GPU Pipeline (Default)
- Uses wgpu for GPU-accelerated rendering
- Supports advanced features like splatting and lighting
- Better performance for large datasets

### CPU Pipeline
- Fallback CPU-based rendering
- More compatible but slower
- Useful for debugging or systems without GPU support

## Performance Tips

1. **Large Point Clouds**: Use GPU pipeline for better performance
2. **Rendering Quality**: Adjust point size and enable splatting for better visual quality
3. **Smooth Navigation**: Use orbit mode for smooth camera movement
4. **Memory Usage**: Monitor statistics panel for memory usage

## Integration with Algorithms

The viewer provides interactive controls for various algorithms:

### ICP (Iterative Closest Point)
```rust
// Load two point clouds
let source = read_point_cloud("source.ply")?;
let target = read_point_cloud("target.ply")?;

// Set up viewer with source cloud
let mut viewer = InteractiveViewer::new()?;
viewer.set_point_cloud(&source);

// Use UI controls to set target cloud and run ICP
viewer.run()?;
```

### RANSAC Plane Segmentation
```rust
// Load point cloud with plane structure
let cloud = read_point_cloud("plane_data.ply")?;

// Set up viewer
let mut viewer = InteractiveViewer::new()?;
viewer.set_point_cloud(&cloud);

// Use Algorithms panel (F2) to run RANSAC
viewer.run()?;
```

## Troubleshooting

### Common Issues

1. **Black Screen**: Check that your GPU supports the required features
2. **Slow Performance**: Try switching to GPU pipeline or reducing point cloud size
3. **File Loading Errors**: Ensure file format is supported and path is correct
4. **UI Not Responding**: Check that egui is properly initialized

### Performance Issues

1. **Large Point Clouds**: Consider downsampling or using level-of-detail
2. **Memory Usage**: Monitor memory usage in statistics panel
3. **GPU Compatibility**: Some older GPUs may not support all features

## Building

To build the interactive viewer:

```bash
# Build visualization crate
cargo build -p threecrate-visualization

# Build examples
cargo build -p threecrate-examples

# Run interactive viewer example
cargo run --bin interactive_viewer_example
```

## Dependencies Overview

- **winit**: Window management
- **wgpu**: Graphics API
- **egui**: Immediate mode GUI
- **nalgebra**: Linear algebra
- **image**: Image processing
- **rfd**: File dialogs
- **instant**: Cross-platform timing

## License

This project is dual-licensed under the MIT and Apache 2.0 licenses. 