# ThreeCrate Simplification

[![Crates.io](https://img.shields.io/crates/v/threecrate-simplification.svg)](https://crates.io/crates/threecrate-simplification)
[![Documentation](https://docs.rs/threecrate-simplification/badge.svg)](https://docs.rs/threecrate-simplification)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

Mesh simplification and decimation algorithms for reducing triangle count while preserving quality.

## Algorithms

- **Quadric Error Metrics**: Quality-aware edge collapse; preserves topology, texture coordinates, and vertex attributes
- **Edge Collapse**: Fast triangle reduction; maintains manifold properties with boundary preservation
- **Clustering**: Groups similar vertices/faces; uniform simplification for large meshes
- **Progressive Mesh**: Multi-resolution representation with incremental simplification and curvature-adaptive quality feedback

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-simplification = "0.6.0"
threecrate-core = "0.6.0"
```

## Example

```rust
use threecrate_simplification::{quadric_error_simplification, edge_collapse_simplification};
use threecrate_core::TriangleMesh;

// Simplify using quadric error metrics (50% reduction)
let simplified = quadric_error_simplification(&mesh, 0.5, 0.01)?;
println!("Simplified: {} faces", simplified.faces.len());

// Edge collapse to a specific triangle count
let simplified = edge_collapse_simplification(&mesh, 1000)?;
println!("Target 1000 faces: {} faces", simplified.faces.len());
```

## Parameters

### Quadric Error Metrics
- **Reduction Ratio**: Target percentage of original triangles (0.0-1.0)
- **Error Threshold**: Maximum allowed quadric error per collapse
- **Boundary Weight**: Higher values preserve boundary edges more strictly

### Edge Collapse
- **Target Count**: Desired number of output triangles
- **Max Error**: Maximum allowed collapse error
- **Preserve Boundaries**: Keep mesh boundary edges intact

## Quality Metrics

- Hausdorff distance from original mesh
- Volume preservation
- Normal deviation tracking
- Texture distortion minimization

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
