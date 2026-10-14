# ThreeCrate Reconstruction

[![Crates.io](https://img.shields.io/crates/v/threecrate-reconstruction.svg)](https://crates.io/crates/threecrate-reconstruction)
[![Documentation](https://docs.rs/threecrate-reconstruction/badge.svg)](https://docs.rs/threecrate-reconstruction)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

Surface reconstruction algorithms for generating meshes from point clouds.

## Algorithms

- **Poisson Surface Reconstruction**: Watertight meshes from oriented point clouds
- **Ball Pivoting Algorithm (BPA)**: Fast reconstruction for uniformly sampled clouds; multi-scale and adaptive radius
- **Alpha Shapes**: Non-convex surface extraction via alpha complex; multiple levels of detail
- **Delaunay Triangulation**: Complete 3D triangulation with multiple projection methods
- **Marching Cubes**: Volumetric/implicit surface to mesh conversion
- **Moving Least Squares (MLS)**: Smooth surface fitting with multiple weight functions
- **Unified Pipeline**: `auto_reconstruct()` — automatic algorithm selection based on point cloud characteristics, with quality metrics and validation

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-reconstruction = "0.6.0"
threecrate-core = "0.6.0"
```

## Example

```rust
use threecrate_reconstruction::{
    auto_reconstruct,
    poisson_reconstruction, poisson_reconstruction_default,
    ball_pivoting_reconstruction, BallPivotingConfig,
};
use threecrate_core::PointCloud;

// Automatic algorithm selection
let mesh = auto_reconstruct(&cloud)?;
println!("Generated mesh: {} faces", mesh.faces.len());

// Poisson reconstruction
let mesh = poisson_reconstruction_default(&normals_cloud)?;

// Ball pivoting with custom config
let config = BallPivotingConfig { radius: 0.1, ..Default::default() };
let mesh = ball_pivoting_reconstruction(&cloud, config)?;
```

## Algorithm Details

### Poisson Parameters
- **depth**: the grid has 2^depth cells per side (3 to 8, default 7). Each step up gives finer detail and takes about 8x longer. Depth 7 runs in about 1 second.
- **smoothing**: how far each normal is spread, in point spacings (default 1.0). Raise it to fill bigger holes.
- **trim_distance**: drop triangles farther than this from any input point. Poisson always closes the surface, so use this on open scans.
- **iso_offset**: move the surface in (positive) or out (negative).

### Ball Pivoting Parameters
- **Ball Radius**: Reconstruction radius (depends on point density)
- **Multi-scale**: Adaptive radius selection for varying densities

## Requirements

- Point clouds with estimated normals produce the best results with Poisson and MLS
- Sufficient point density relative to the ball radius for BPA

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
