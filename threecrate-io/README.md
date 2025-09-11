# ThreeCrate I/O

[![Crates.io](https://img.shields.io/crates/v/threecrate-io.svg)](https://crates.io/crates/threecrate-io)
[![Documentation](https://docs.rs/threecrate-io/badge.svg)](https://docs.rs/threecrate-io)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

File I/O operations for point clouds and meshes in the threecrate ecosystem.

## Features

- **Point Cloud Formats**: PLY, PCD, LAS, LAZ, XYZ/CSV file support
- **Mesh Formats**: OBJ, PLY file support with normals and textures
- **Memory-Mapped I/O**: High-performance reading of large binary files (optional)
- **Streaming I/O**: Memory-efficient reading and writing
- **Error Handling**: Comprehensive error reporting for invalid files
- **Cross-platform**: Works on Windows, macOS, and Linux
- **OS-Gated Features**: Platform-specific optimizations with automatic fallback

## Supported Formats

### Point Clouds
- **PLY**: Polygon File Format (ASCII and binary)
- **PCD**: Point Cloud Data format (ASCII and binary)
- **LAS/LAZ**: LiDAR data formats via Pasture
- **XYZ/CSV**: Comma-separated values with configurable columns

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

### Optional Features

- `las_laz`: Enable LAS/LAZ format support via Pasture
- `io-mmap`: Enable memory-mapped I/O for improved performance on large binary files

```toml
[dependencies]
threecrate-io = { version = "0.1.0", features = ["io-mmap"] }
```

## Examples

### Basic Usage

```rust
use threecrate_io::{read_point_cloud, write_point_cloud, read_mesh};
use threecrate_core::{PointCloud, Point3f};

// Auto-detect format and load point cloud
let cloud = read_point_cloud("input.ply")?;
println!("Loaded {} points", cloud.len());

// Save point cloud (format determined by extension)
write_point_cloud(&cloud, "output.pcd")?;

// Load mesh from OBJ file
let mesh = read_mesh("model.obj")?;
println!("Loaded mesh with {} vertices", mesh.vertices.len());
```

### Memory-Mapped I/O (io-mmap feature)

For large binary files, enable memory-mapped I/O for better performance:

```rust
use threecrate_io::{RobustPlyReader, RobustPcdReader};

// Memory mapping is automatically used for large binary files
// when the io-mmap feature is enabled
let ply_data = RobustPlyReader::read_ply_file("large_cloud.ply")?;
let (header, points) = RobustPcdReader::read_pcd_file("large_cloud.pcd")?;

// Check if memory mapping would be used
#[cfg(feature = "io-mmap")]
{
    let would_use_mmap = threecrate_io::mmap::should_use_mmap("large_cloud.ply");
    println!("Would use memory mapping: {}", would_use_mmap);
}
```

### Benchmarking

Compare performance between memory-mapped and standard I/O:

```bash
# Run benchmarks (requires io-mmap feature)
cargo bench --features io-mmap mmap_benchmarks

# Run the memory mapping example
cargo run --example mmap_example --features io-mmap
```

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 