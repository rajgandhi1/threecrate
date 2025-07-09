# ThreeCrate Reconstruction

[![Crates.io](https://img.shields.io/crates/v/threecrate-reconstruction.svg)](https://crates.io/crates/threecrate-reconstruction)
[![Documentation](https://docs.rs/threecrate-reconstruction/badge.svg)](https://docs.rs/threecrate-reconstruction)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/3DCrate#license)

Surface reconstruction algorithms for generating meshes from point clouds.

## Features

- **Poisson Reconstruction**: High-quality surface reconstruction using Poisson solving
- **Ball Pivoting**: Fast surface reconstruction for uniformly sampled point clouds
- **Alpha Shapes**: Geometric reconstruction using alpha complex
- **Delaunay Triangulation**: 2D/3D triangulation for mesh generation
- **Parallel Processing**: Multi-threaded algorithms for large datasets

## Algorithms

### Poisson Surface Reconstruction
- High-quality mesh generation from oriented point clouds
- Handles noise and irregular sampling well
- Produces watertight meshes
- Configurable octree depth and sample density

### Ball Pivoting Algorithm (BPA)
- Fast reconstruction for uniformly sampled point clouds
- Good for dense, noise-free data
- Preserves sharp features and boundaries
- Configurable ball radius and clustering

### Alpha Shapes
- Geometric approach using alpha complex
- Good for shape analysis and boundary detection
- Multiple levels of detail with different alpha values
- Handles complex topologies

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-reconstruction = "0.1.0"
threecrate-core = "0.1.0"
```

## Example

```rust
use threecrate_reconstruction::{poisson_reconstruction, ball_pivoting_reconstruction};
use threecrate_core::{PointCloud, Point3f};

// Load point cloud with normals
let cloud = PointCloud::from_points(vec![/* points */]);

// Poisson reconstruction
let mesh = poisson_reconstruction(&cloud, 6, 1.0, 0.1)?;
println!("Generated mesh with {} faces", mesh.faces.len());

// Ball pivoting reconstruction
let radius = 0.1;
let mesh = ball_pivoting_reconstruction(&cloud, radius)?;
println!("Generated mesh with {} faces", mesh.faces.len());
```

## Algorithm Details

### Poisson Parameters
- **Octree Depth**: Controls mesh resolution (6-10 recommended)
- **Sample Density**: Point density factor (0.5-2.0)
- **Confidence Threshold**: Quality filtering (0.0-1.0)

### Ball Pivoting Parameters
- **Ball Radius**: Reconstruction radius (depends on point density)
- **Clustering**: Remove duplicate vertices and faces
- **Normal Consistency**: Ensure consistent face orientations

## Requirements

- Point clouds with estimated normals for best results
- Sufficient point density for reconstruction
- Reasonable memory for large point clouds

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 