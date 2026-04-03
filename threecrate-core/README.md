# ThreeCrate Core

[![Crates.io](https://img.shields.io/crates/v/threecrate-core.svg)](https://crates.io/crates/threecrate-core)
[![Documentation](https://docs.rs/threecrate-core/badge.svg)](https://docs.rs/threecrate-core)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

Core data structures and traits for the threecrate library ecosystem.

## Features

- **Point Types**: `Point3f`, `ColoredPoint3f`, `NormalPoint3f`, `ColoredNormalPoint3f`
- **Point Cloud**: Generic `PointCloud<T>` with spatial operations
- **Mesh**: `TriangleMesh` with vertices, faces, normals, and optional UV coordinates
- **Transformations**: `Transform3D` for rotation, translation, and scaling
- **Traits**: `Drawable`, `Transformable`, `NearestNeighborSearch`
- **Error Handling**: Comprehensive error types for 3D processing operations
- **Bevy Integration**: Optional Bevy game engine interop via the `bevy` feature flag

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-core = "0.6.0"
```

For Bevy integration:

```toml
[dependencies]
threecrate-core = { version = "0.6.0", features = ["bevy"] }
```

## Example

```rust
use threecrate_core::{PointCloud, Point3f, TriangleMesh};

// Create a point cloud
let points = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.0, 1.0, 0.0),
];
let cloud = PointCloud::from_points(points);

// Create a triangle mesh
let vertices = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.0, 1.0, 0.0),
];
let faces = vec![[0, 1, 2]];
let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
```

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
