# ThreeCrate Simplification

[![Crates.io](https://img.shields.io/crates/v/threecrate-simplification.svg)](https://crates.io/crates/threecrate-simplification)
[![Documentation](https://docs.rs/threecrate-simplification/badge.svg)](https://docs.rs/threecrate-simplification)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/3DCrate#license)

Mesh simplification and decimation algorithms for reducing triangle count while preserving quality.

## Features

- **Quadric Error Metrics**: Quality-aware edge collapse simplification
- **Progressive Mesh**: Multi-resolution mesh representations
- **Edge Collapse**: Efficient triangle reduction algorithms
- **Clustering**: Point cloud and mesh clustering for LOD
- **Parallel Processing**: Multi-threaded simplification for large meshes

## Algorithms

### Quadric Error Metrics
- High-quality mesh decimation using quadric error matrices
- Preserves mesh topology and important features
- Configurable error thresholds and target triangle counts
- Handles texture coordinates and vertex attributes

### Edge Collapse
- Fast triangle reduction through edge collapse operations
- Maintains mesh manifold properties
- Configurable collapse criteria and error bounds
- Supports boundary preservation

### Clustering-based Simplification
- Groups similar vertices and faces for reduction
- Good for uniform simplification across the mesh
- Handles large meshes efficiently
- Configurable cluster size and merge criteria

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-simplification = "0.1.0"
threecrate-core = "0.1.0"
```

## Example

```rust
use threecrate_simplification::{quadric_error_simplification, edge_collapse_simplification};
use threecrate_core::TriangleMesh;

// Load a high-resolution mesh
let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
println!("Original mesh: {} faces", mesh.faces.len());

// Simplify using quadric error metrics
let simplified = quadric_error_simplification(&mesh, 0.5, 0.01)?;
println!("Simplified mesh: {} faces", simplified.faces.len());

// Alternative: Edge collapse simplification
let simplified = edge_collapse_simplification(&mesh, 1000)?;
println!("Target 1000 faces: {} faces", simplified.faces.len());
```

## Algorithm Parameters

### Quadric Error Metrics
- **Reduction Ratio**: Target percentage of original triangles (0.0-1.0)
- **Error Threshold**: Maximum allowed quadric error per collapse
- **Boundary Weight**: Preserve boundary edges (higher = more preservation)
- **Normal Weight**: Preserve surface normals (higher = smoother results)

### Edge Collapse
- **Target Count**: Desired number of triangles in output
- **Max Error**: Maximum collapse error allowed
- **Preserve Boundaries**: Keep mesh boundaries intact
- **Aspect Ratio**: Prevent degenerate triangles

## Quality Metrics

- **Hausdorff Distance**: Measure geometric deviation from original
- **Volume Preservation**: Maintain original mesh volume
- **Normal Deviation**: Preserve surface orientation
- **Texture Distortion**: Minimize UV coordinate stretching

## Performance

- **Parallel Processing**: Multi-threaded for large meshes
- **Memory Efficient**: Streaming algorithms for huge datasets
- **Incremental**: Progressive simplification with quality feedback
- **Adaptive**: Locally adaptive simplification based on curvature

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 