# ThreeCrate I/O

This crate provides comprehensive I/O operations for point clouds and meshes in the ThreeCrate 3D processing library.

## Supported Formats

### Point Clouds
- **PLY** - Polygon File Format (✅ Implemented)
  - Read and write point clouds with positions
  - Automatic type conversion
  - Support for various PLY property types

### Meshes  
- **PLY** - Polygon File Format (✅ Implemented)
  - Read and write triangle meshes
  - Support for vertex normals
  - Automatic face triangulation for complex polygons
  
- **OBJ** - Wavefront OBJ (✅ Implemented)
  - Read and write triangle meshes
  - Support for vertex normals
  - Automatic triangulation of quads and n-gons
  - Helper functions for vertex-only operations

### Point Cloud Formats (Future)
- **LAS/LAZ** - LIDAR formats (🚧 Planned via pasture)
- **PCD** - Point Cloud Data format (🚧 Planned via pasture)

## Dependencies

The crate uses the following key dependencies for I/O operations:

- `ply-rs` - For PLY format support
- `obj` - For OBJ format support  
- `pasture-core` & `pasture-io` - For advanced point cloud formats (planned)

## Usage

### Reading Files

```rust
use threecrate_io::*;

// Auto-detect format and read
let point_cloud = read_point_cloud("data.ply")?;
let mesh = read_mesh("model.obj")?;

// Use specific readers
let cloud = ply::PlyReader::read_point_cloud("data.ply")?;
let mesh = obj::ObjReader::read_mesh("model.obj")?;
```

### Writing Files

```rust
use threecrate_io::*;

// Write using specific writers
ply::PlyWriter::write_point_cloud(&cloud, "output.ply")?;
obj::ObjWriter::write_mesh(&mesh, "output.obj")?;
```

### Creating Data

```rust
use threecrate_core::*;

// Create a point cloud
let mut cloud = PointCloud::new();
cloud.push(Point3f::new(0.0, 0.0, 0.0));
cloud.push(Point3f::new(1.0, 0.0, 0.0));

// Create a triangle mesh
let vertices = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.5, 1.0, 0.0),
];
let faces = vec![[0, 1, 2]];
let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
```

## Features

### PLY Support
- ✅ Read/write point clouds
- ✅ Read/write triangle meshes
- ✅ Support for vertex normals
- ✅ Automatic type conversion
- ✅ Binary and ASCII format support

### OBJ Support  
- ✅ Read/write triangle meshes
- ✅ Support for vertex normals
- ✅ Automatic triangulation of complex polygons
- ✅ Helper functions for vertices-only I/O

### Advanced Features
- ✅ Auto-detection of file formats by extension
- ✅ Comprehensive error handling
- ✅ Memory-efficient processing
- ✅ Type-safe API design

## Error Handling

The crate provides comprehensive error handling through the `threecrate_core::Error` type:

```rust
match read_point_cloud("data.ply") {
    Ok(cloud) => println!("Loaded {} points", cloud.len()),
    Err(Error::UnsupportedFormat(msg)) => println!("Format not supported: {}", msg),
    Err(Error::InvalidData(msg)) => println!("Invalid data: {}", msg),
    Err(Error::Io(err)) => println!("I/O error: {}", err),
    Err(err) => println!("Other error: {}", err),
}
```

## Examples

See `examples/basic_usage.rs` for a comprehensive demonstration of all I/O functionality.

Run the example with:
```bash
cargo run --bin basic_usage
```

## Future Plans

- Complete pasture integration for LAS/LAZ/PCD support
- Add support for colored point clouds
- Implement streaming I/O for large files
- Add compression support for various formats
- Support for additional mesh formats (STL, 3MF, etc.) 