# threecrate

A comprehensive 3D point cloud and mesh processing library for Rust.

[![Crates.io](https://img.shields.io/crates/v/threecrate.svg)](https://crates.io/crates/threecrate)
[![Documentation](https://docs.rs/threecrate/badge.svg)](https://docs.rs/threecrate)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate)

## Overview

threecrate is a modular, high-performance library for 3D point cloud and mesh processing. This umbrella crate provides convenient access to all threecrate functionality in one place.

## Features

- **Core**: 3D data structures (Point, PointCloud, TriangleMesh, Transform)
- **Algorithms**: Filtering, registration (ICP, NDT, global), segmentation, normal estimation, feature descriptors (FPFH, SHOT), mesh boolean operations, mesh smoothing, colorization, SIMD-accelerated distances, and streaming processing
- **GPU**: GPU-accelerated filtering, normal estimation, ICP, nearest neighbor search, TSDF volume integration, and real-time rendering via wgpu
- **I/O**: PLY, OBJ, PCD, LAS/LAZ, XYZ/CSV, E57 with streaming and memory-mapped I/O support
- **Reconstruction**: Poisson, Ball Pivoting, Alpha Shapes, Delaunay, Marching Cubes, MLS, and automatic algorithm selection
- **Simplification**: Quadric error metrics, edge collapse, clustering-based, and progressive mesh simplification
- **Visualization**: Interactive 3D viewer with orbit, pan, and zoom controls

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate = "0.6.0"
```

Basic usage:

```rust
use threecrate::prelude::*;

let cloud = read_point_cloud("scan.ply")?;
let cloud = voxel_grid_filter(&cloud, 0.05)?;
let normals_cloud = estimate_normals(&cloud, 10)?;
let mesh = auto_reconstruct(&normals_cloud)?;
write_mesh("output.obj", &mesh)?;
```

## Installation Options

### Option 1: Umbrella Crate (Recommended)

```toml
[dependencies]
threecrate = "0.6.0"
```

### Option 2: Individual Crates (Minimal dependencies)

```toml
[dependencies]
threecrate-core = "0.6.0"
threecrate-algorithms = "0.6.0"
threecrate-gpu = "0.6.0"
threecrate-io = "0.6.0"
threecrate-simplification = "0.6.0"
threecrate-reconstruction = "0.6.0"
threecrate-visualization = "0.6.0"
```

## Feature Flags

```toml
[dependencies]
threecrate = { version = "0.6.0", features = ["all"] }
```

Available features:
- `default`: core, algorithms, io, simplification
- `core`: Core data structures (always enabled)
- `algorithms`: Point cloud processing algorithms
- `gpu`: GPU-accelerated processing
- `io`: File format support
- `simplification`: Mesh simplification
- `reconstruction`: Surface reconstruction
- `visualization`: Interactive visualization
- `all`: All features

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
