# threecrate

A comprehensive 3D point cloud processing library for Rust.

[![Crates.io](https://img.shields.io/crates/v/threecrate.svg)](https://crates.io/crates/threecrate)
[![Documentation](https://docs.rs/threecrate/badge.svg)](https://docs.rs/threecrate)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate)

## Overview

threecrate is a modular, high-performance library for 3D point cloud and mesh processing. This umbrella crate provides convenient access to all threecrate functionality in one place.

## Features

- **Core**: Basic 3D data structures (Point, PointCloud, Mesh, Transform)
- **Algorithms**: Point cloud processing (filtering, registration, segmentation, normals)
- **GPU**: GPU-accelerated processing using wgpu
- **I/O**: File format support (PLY, OBJ, LAS, Pasture formats)
- **Simplification**: Mesh and point cloud simplification algorithms
- **Reconstruction**: Surface reconstruction from point clouds
- **Visualization**: Interactive 3D visualization tools

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate = "0.5.0"
```

Basic usage:

```rust
use threecrate::prelude::*;

// Create a point cloud
let points = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.0, 1.0, 0.0),
];
let cloud = PointCloud::from_points(points);

// Apply algorithms (many still in development)
println!("Point cloud with {} points", cloud.len());
```

## Installation Options

### Option 1: Umbrella Crate (Recommended for most users)

```toml
[dependencies]
threecrate = "0.5.0"
```

### Option 2: Individual Crates (For minimal dependencies)

```toml
[dependencies]
threecrate-core = "0.5.0"         # Core data structures
threecrate-algorithms = "0.5.0"   # Processing algorithms
threecrate-gpu = "0.5.0"          # GPU acceleration
threecrate-io = "0.5.0"           # File I/O
threecrate-simplification = "0.5.0"  # Simplification
threecrate-reconstruction = "0.5.0"  # Surface reconstruction
threecrate-visualization = "0.5.0"  # Visualization
```

## Feature Flags

The umbrella crate supports granular feature control:

```toml
[dependencies]
threecrate = { version = "0.5.0", features = ["all"] }
```

Available features:
- `default`: core, algorithms, io, simplification
- `core`: Core data structures (always enabled)
- `algorithms`: Point cloud processing algorithms
- `gpu`: GPU-accelerated processing
- `io`: File format support
- `simplification`: Mesh and point cloud simplification
- `reconstruction`: Surface reconstruction
- `visualization`: Interactive visualization
- `all`: All features

## Individual Crates

### [`threecrate-core`](https://crates.io/crates/threecrate-core)
Core data structures and traits for 3D processing.

**Key types:**
- `Point3f`: 3D point with floating-point coordinates
- `PointCloud`: Collection of 3D points with spatial operations
- `TriangleMesh`: 3D mesh with vertices, faces, and normals
- `Transform3D`: 3D transformations (rotation, translation, scaling)

### [`threecrate-algorithms`](https://crates.io/crates/threecrate-algorithms)
Algorithms for point cloud and mesh processing.

**Features:**
- **Filtering**: Statistical, radius, voxel grid filters
- **Registration**: ICP (Iterative Closest Point) algorithm
- **Segmentation**: RANSAC plane detection
- **Normals**: Normal estimation and orientation
- **Nearest Neighbor**: Efficient spatial queries

### [`threecrate-gpu`](https://crates.io/crates/threecrate-gpu)
GPU-accelerated processing using wgpu.

**Features:**
- GPU-accelerated filtering and processing
- Parallel normal computation
- GPU-based ICP registration
- Efficient mesh rendering

### [`threecrate-io`](https://crates.io/crates/threecrate-io)
File format support for point clouds and meshes.

**Supported formats:**
- PLY (Stanford Polygon format)
- OBJ (Wavefront OBJ)
- LAS/LAZ (LiDAR formats)
- Pasture formats

### [`threecrate-simplification`](https://crates.io/crates/threecrate-simplification)
Mesh and point cloud simplification algorithms.

**Features:**
- Quadric error metrics
- Edge collapse simplification
- Clustering-based simplification

### [`threecrate-reconstruction`](https://crates.io/crates/threecrate-reconstruction)
Advanced surface reconstruction from point clouds with intelligent algorithm selection.

**Features:**
- Poisson surface reconstruction with automatic parameter tuning
- Enhanced Ball Pivoting with multi-scale and adaptive radius selection
- Complete Delaunay triangulation with multiple projection methods
- Marching Cubes for volumetric reconstruction from implicit surfaces
- Moving Least Squares (MLS) surface fitting with multiple weight functions
- Alpha shapes for non-convex surface extraction
- Unified reconstruction pipeline with automatic algorithm selection
- Parallel processing using rayon for optimal performance
- Comprehensive quality metrics and validation
- Data analysis engine for point cloud characteristics

### [`threecrate-visualization`](https://crates.io/crates/threecrate-visualization)
Interactive 3D visualization tools.

**Features:**
- Real-time point cloud visualization
- Interactive mesh rendering
- Camera controls (orbit, pan, zoom)
- Cross-platform support

## Examples

### Point Cloud Processing

```rust
use threecrate::prelude::*;

// Create a point cloud
let points = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.0, 1.0, 0.0),
];
let cloud = PointCloud::from_points(points);

// Estimate normals
let normals_cloud = estimate_normals(&cloud, 10)?;

// Perform surface reconstruction
let mesh = auto_reconstruct(&cloud)?;
println!("Reconstructed mesh with {} triangles", mesh.face_count());

// Or use specific algorithm
let poisson_mesh = poisson_reconstruction_default(&normals_cloud)?;

// Save processed cloud
// cloud.save("output.ply")?; // I/O functionality
```

### Mesh Processing

```rust
use threecrate::prelude::*;

// Create a mesh
let vertices = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.0, 1.0, 0.0),
];
let faces = vec![[0, 1, 2]];
let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);

// Simplify mesh (algorithms in development)
println!("Mesh with {} vertices", mesh.vertices.len());
```

### GPU Acceleration

```rust
use threecrate::prelude::*;

// Initialize GPU context
let gpu_context = GpuContext::new().await?;

// GPU-accelerated processing (in development)
println!("GPU context initialized");
```

## Performance

threecrate is designed for high performance:

- **Parallel processing**: Uses `rayon` for CPU parallelism
- **GPU acceleration**: Optional wgpu-based GPU processing
- **Efficient data structures**: Optimized for cache locality
- **Spatial indexing**: KD-tree and other spatial data structures

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Changelog

### v0.5.0

- Major reconstruction module overhaul with 5 advanced algorithms
- Fixed Poisson reconstruction with proper API integration
- Enhanced Ball Pivoting with multi-scale capabilities and adaptive features
- Complete Delaunay triangulation implementation with 5 projection methods
- New Marching Cubes algorithm for volumetric surface reconstruction
- New Moving Least Squares (MLS) surface fitting with 4 weight functions
- Unified reconstruction pipeline with intelligent algorithm auto-selection
- Comprehensive parallel processing integration using rayon
- Advanced data analysis engine for point cloud characteristics
- Quality metrics and validation for all reconstruction algorithms
- 65 comprehensive tests ensuring reliability and correctness

### v0.1.0

- Initial release with core functionality
- Point cloud and mesh processing algorithms
- GPU acceleration support
- File I/O for common formats
- Interactive visualization tools
