# ThreeCrate Algorithms

[![Crates.io](https://img.shields.io/crates/v/threecrate-algorithms.svg)](https://crates.io/crates/threecrate-algorithms)
[![Documentation](https://docs.rs/threecrate-algorithms/badge.svg)](https://docs.rs/threecrate-algorithms)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

High-performance algorithms for 3D point cloud and mesh processing.

## Features

- **Filtering**: Voxel grid, radius outlier removal, statistical outlier removal
- **Registration**: ICP (point-to-point and point-to-plane), NDT, global registration (FPFH + RANSAC)
- **Segmentation**: RANSAC plane detection, Euclidean cluster extraction (sequential and parallel)
- **Normal Estimation**: K-nearest and radius-based normal computation with configurable orientation
- **Feature Descriptors**: FPFH (Fast Point Feature Histograms) and SHOT (Signature of Histograms of Orientations)
- **Nearest Neighbor Search**: KD-tree and brute-force, with SIMD-accelerated distance computations
- **Mesh Boolean Operations**: Union, intersection, and difference
- **Mesh Smoothing**: Laplacian, Taubin, and HC smoothing
- **Colorization**: Project colors from registered RGB images onto point clouds
- **Streaming Processing**: Memory-efficient pipeline for large point clouds
- **Parallel Processing**: Multi-threaded algorithms via rayon

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-algorithms = "0.6.0"
threecrate-core = "0.6.0"
```

## Example

```rust
use threecrate_core::{PointCloud, Point3f};
use threecrate_algorithms::{
    icp, estimate_normals, segment_plane_ransac,
    voxel_grid_filter, statistical_outlier_removal,
    extract_euclidean_clusters, global_registration,
};

// Filter the point cloud
let cloud = voxel_grid_filter(&cloud, 0.05)?;
let cloud = statistical_outlier_removal(&cloud, 10, 2.0)?;

// Estimate normals
let normals = estimate_normals(&cloud, 10)?;

// ICP registration
let result = icp(&source, &target, Default::default())?;
println!("Registration converged: {}", result.converged);

// RANSAC plane segmentation
let plane_result = segment_plane_ransac(&cloud, 1000, 0.01)?;
println!("Found {} inliers", plane_result.inliers.len());

// Euclidean cluster extraction
let clusters = extract_euclidean_clusters(&cloud, 0.1, 10, 10000)?;
println!("Found {} clusters", clusters.len());

// Global registration using FPFH features
let result = global_registration(&source, &target, Default::default())?;
```

## Algorithms

- **Filtering**: `voxel_grid_filter`, `radius_outlier_removal`, `statistical_outlier_removal`
- **Registration**: `icp`, `icp_point_to_point`, `icp_point_to_plane`, `ndt_registration`, `global_registration`
- **Segmentation**: `segment_plane_ransac`, `extract_euclidean_clusters`, `extract_euclidean_clusters_parallel`
- **Normal Estimation**: `estimate_normals`, `estimate_normals_radius`, `estimate_normals_with_config`
- **Feature Descriptors**: `extract_fpfh_features`, `extract_shot_features`
- **Nearest Neighbor**: `KdTree`, `BruteForceSearch`
- **Mesh Boolean**: `mesh_union`, `mesh_intersection`, `mesh_difference`
- **Mesh Smoothing**: `smooth_laplacian`, `smooth_taubin`, `smooth_hc`
- **Colorization**: `colorize_point_cloud`, `colorize_from_images`

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
