# ThreeCrate I/O

[![Crates.io](https://img.shields.io/crates/v/threecrate-io.svg)](https://crates.io/crates/threecrate-io)
[![Documentation](https://docs.rs/threecrate-io/badge.svg)](https://docs.rs/threecrate-io)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

File I/O operations for point clouds and meshes in the threecrate ecosystem.

## Features

- **Point Cloud Formats**: PLY, LAS, LAZ file support
- **Mesh Formats**: OBJ file support with normals and textures
- **Streaming I/O**: Memory-efficient reading and writing
- **Error Handling**: Comprehensive error reporting for invalid files
- **Cross-platform**: Works on Windows, macOS, and Linux

## Supported Formats

### Point Clouds
- **PLY**: Polygon File Format (ASCII and binary)
- **LAS/LAZ**: LiDAR data formats via Pasture
- **CSV**: Comma-separated values with configurable columns

### Meshes
- **OBJ**: Wavefront OBJ format with materials
- **PLY**: Triangle meshes in PLY format

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-io = "0.1.0"
threecrate-core = "0.1.0"
```

## Example

```rust
use threecrate_io::{load_ply_point_cloud, save_ply_point_cloud, load_obj_mesh};
use threecrate_core::{PointCloud, Point3f};

// Load point cloud from PLY file
let cloud = load_ply_point_cloud("input.ply")?;
println!("Loaded {} points", cloud.len());

// Save point cloud to PLY file
save_ply_point_cloud(&cloud, "output.ply")?;

// Load mesh from OBJ file
let mesh = load_obj_mesh("model.obj")?;
println!("Loaded mesh with {} vertices", mesh.vertices.len());
```

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 