# ThreeCrate Algorithms

[![Crates.io](https://img.shields.io/crates/v/threecrate-algorithms.svg)](https://crates.io/crates/threecrate-algorithms)
[![Documentation](https://docs.rs/threecrate-algorithms/badge.svg)](https://docs.rs/threecrate-algorithms)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

High-performance algorithms for 3D point cloud and mesh processing.

## Features

- **Point Cloud Processing**: Filtering, downsampling, and outlier removal
- **Registration**: ICP (Iterative Closest Point) algorithm for point cloud alignment
- **Segmentation**: RANSAC plane segmentation and clustering algorithms
- **Spatial Queries**: K-nearest neighbor search and spatial indexing
- **Normal Estimation**: Surface normal computation for point clouds
- **Parallel Processing**: Multi-threaded algorithms using Rayon

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-algorithms = "0.1.0"
threecrate-core = "0.1.0"
```

## Example

```rust
use threecrate_core::{PointCloud, Point3f};
use threecrate_algorithms::{icp_registration, estimate_normals, plane_segmentation_ransac};

// Load or create point clouds
let source = PointCloud::from_points(vec![/* points */]);
let target = PointCloud::from_points(vec![/* points */]);

// ICP registration
let result = icp_registration(&source, &target, 50, 0.001, 1.0)?;
println!("Registration converged: {}", result.converged);

// Estimate normals
let normals = estimate_normals(&source, 10)?;

// RANSAC plane segmentation
let plane_result = plane_segmentation_ransac(&source, 1000, 0.01)?;
println!("Found {} inliers", plane_result.inliers.len());
```

## Algorithms

- **ICP Registration**: Point cloud alignment using iterative closest point
- **RANSAC Segmentation**: Robust plane fitting and outlier detection
- **Normal Estimation**: Surface normal computation using local neighborhoods
- **Filtering**: Statistical outlier removal and radius filtering
- **Spatial Indexing**: KD-tree and R-tree based spatial queries

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 