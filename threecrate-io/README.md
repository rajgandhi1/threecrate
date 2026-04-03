# ThreeCrate I/O

[![Crates.io](https://img.shields.io/crates/v/threecrate-io.svg)](https://crates.io/crates/threecrate-io)
[![Documentation](https://docs.rs/threecrate-io/badge.svg)](https://docs.rs/threecrate-io)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

File I/O operations for point clouds and meshes in the threecrate ecosystem.

## Supported Formats

### Point Clouds
- **PLY**: Polygon File Format (ASCII and binary)
- **PCD**: Point Cloud Data format (ASCII and binary)
- **LAS/LAZ**: LiDAR data formats via Pasture (feature-gated)
- **XYZ/CSV**: Comma-separated values with configurable columns
- **E57**: 3D imaging format (feature-gated)

### Meshes
- **OBJ**: Wavefront OBJ with materials and groups
- **PLY**: Triangle meshes in PLY format

## Features

- Auto-detect format from file extension
- Streaming readers for memory-efficient processing of large files
- Memory-mapped I/O for large binary files (`io-mmap` feature)
- Attribute-preserving read/write with metadata support
- Robust readers with error recovery

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate-io = "0.6.0"
threecrate-core = "0.6.0"
```

### Optional Features

- `las_laz`: Enable LAS/LAZ format support via Pasture
- `e57`: Enable E57 format support
- `io-mmap`: Enable memory-mapped I/O for improved performance on large binary files

```toml
[dependencies]
threecrate-io = { version = "0.6.0", features = ["io-mmap", "e57"] }
```

## Examples

### Basic Usage

```rust
use threecrate_io::{read_point_cloud, write_point_cloud, read_mesh, write_mesh};

// Auto-detect format and load point cloud
let cloud = read_point_cloud("input.ply")?;
println!("Loaded {} points", cloud.len());

// Save point cloud (format determined by extension)
write_point_cloud(&cloud, "output.pcd")?;

// Load and save mesh
let mesh = read_mesh("model.obj")?;
write_mesh("output.ply", &mesh)?;
```

### Streaming I/O

```rust
use threecrate_io::read_point_cloud_iter;

// Process large files without loading everything into memory
for chunk in read_point_cloud_iter("large_scan.ply")? {
    let points = chunk?;
    // process chunk
}
```

### Memory-Mapped I/O

For large binary files, enable the `io-mmap` feature for better performance:

```bash
cargo run --example mmap_example --features io-mmap
cargo bench --features io-mmap mmap_benchmarks
```

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
