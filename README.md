# 3DCrate

A comprehensive 3D point cloud processing library for Rust, providing high-performance algorithms for point cloud manipulation, analysis, and visualization.

## Overview

3DCrate is a modular library designed for 3D point cloud processing with a focus on performance, safety, and ease of use. The library is organized into several specialized crates:
![image](https://github.com/user-attachments/assets/ecf23d60-0bb3-431d-9894-bf3356493a28)


**Note**: This project has core algorithms implemented including normals estimation, ICP registration, RANSAC segmentation, and surface reconstruction. See the implementation status below.

## Architecture

### Core Modules

- **`threecrate-core`** - Core data structures and traits for 3D geometry
- **`threecrate-algorithms`** - Point cloud processing algorithms (filtering, normals, registration)
- **`threecrate-gpu`** - GPU-accelerated computing using WGPU
- **`threecrate-io`** - File I/O support for various 3D formats (PLY, OBJ)
- **`threecrate-reconstruction`** - Surface reconstruction algorithms
- **`threecrate-simplification`** - Mesh and point cloud simplification
- **`threecrate-visualization`** - Real-time 3D visualization

## Implementation Status

###  **Implemented**
- **Core Data Structures**: `PointCloud<T>`, `TriangleMesh`, `Point3f`, `Transform3D`
- **Basic Point Cloud Operations**: Creation, iteration, indexing, transformation
- **Traits**: `Drawable`, `Transformable`, `NearestNeighborSearch`
- **Type System**: Generic point cloud containers with specialized types
- **Build System**: Complete workspace with all crates compiling successfully
- **Basic Examples**: Working example demonstrating core functionality

###  **Under Implementation**
- **File I/O**: PLY and OBJ format support (skeleton structure in place)
- **Visualization**: Camera, renderer, and shader infrastructure (basic structure ready)
- **GPU Infrastructure**: Device management and compute pipeline setup

###  **Recently Implemented**

#### Point Cloud Processing
- ✅ **Normal Estimation**: k-NN based surface normal computation with PCA
- ✅ **Registration**: ICP (Iterative Closest Point) alignment algorithm
- ✅ **Segmentation**: RANSAC plane detection and clustering
- **Filtering**: Statistical outlier removal, voxel grid downsampling (planned)
- **Feature Detection**: Keypoint extraction and descriptors (planned)

#### Surface Reconstruction
- ✅ **Ball Pivoting**: Simplified ball pivoting algorithm for surface reconstruction
- ✅ **Alpha Shapes**: CGAL-style alpha complex with multiple computation modes
- ✅ **Poisson Reconstruction**: Integration ready (external crate wrapper)
- **Delaunay Triangulation**: 2D/3D triangulation algorithms (planned)

#### Mesh Processing
- ✅ **Quadric Error Decimation**: Garland-Heckbert mesh simplification algorithm

#### GPU Acceleration
- **Parallel Processing**: GPU-accelerated algorithms using WGPU
- **Memory Management**: Efficient GPU buffer management
- **Cross-Platform**: Vulkan, Metal, DirectX 12, and WebGL support

#### Visualization
- **Real-time Rendering**: Interactive 3D point cloud and mesh visualization
- **Multiple Formats**: Support for various rendering primitives
- **Camera Controls**: Orbit, pan, zoom camera interactions

## Quick Start

Add 3DCrate to your `Cargo.toml`:

```toml
[dependencies]
threecrate-core = "0.1.0"
threecrate-algorithms = "0.1.0"        # Normals, ICP, RANSAC
threecrate-reconstruction = "0.1.0"     # Surface reconstruction
threecrate-simplification = "0.1.0"     # Mesh simplification
threecrate-io = "0.1.0"
```

### Basic Usage

```rust
use threecrate_core::{PointCloud, Point3f, Transform3D};
use threecrate_algorithms::{estimate_normals, segment_plane, icp};
use threecrate_reconstruction::ball_pivoting_reconstruction;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a point cloud
    let points = vec![
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(1.0, 0.0, 0.0),
        Point3f::new(0.0, 1.0, 0.0),
        Point3f::new(0.0, 0.0, 1.0),
    ];
    
    let mut cloud = PointCloud::from_points(points);
    println!("Created point cloud with {} points", cloud.len());
    
    // Estimate surface normals
    let normals_cloud = estimate_normals(&cloud, 3)?;
    println!("Estimated normals for {} points", normals_cloud.len());
    
    // Detect planes with RANSAC
    let plane_result = segment_plane(&cloud, 0.01, 1000)?;
    println!("Found plane with {} inliers", plane_result.inliers.len());
    
    // Surface reconstruction
    let mesh = ball_pivoting_reconstruction(&cloud, 0.1)?;
    println!("Reconstructed mesh with {} triangles", mesh.face_count());
    
    Ok(())
}
```


## Examples

Currently available examples:

```bash
# Run the basic usage example (works)
cargo run --bin basic_usage
```


## Building

```bash
# Build all crates
cargo build --workspace

# Run tests (currently no tests implemented)
cargo test --workspace

# Check compilation
cargo check --workspace
```

## Development Roadmap

### Phase 1: Core Implementation (Current)
- [x] Core data structures and traits
- [x] Basic point cloud operations
- [x] Workspace setup and compilation
- [ ] File I/O implementations
- [ ] Basic algorithms (filtering, normals)

### Phase 2: Algorithm Implementation
- ✅ Point cloud filtering algorithms (partial)
- ✅ Normal estimation (k-NN with PCA)
- ✅ Registration (ICP)
- ✅ Segmentation algorithms (RANSAC plane detection)
- ✅ Surface reconstruction (Ball pivoting, Alpha shapes)
- ✅ Mesh simplification (Quadric error decimation)

### Phase 3: GPU Acceleration
- [ ] WGPU compute pipeline setup
- [ ] GPU-accelerated filtering
- [ ] GPU-accelerated normal estimation
- [ ] GPU-accelerated ICP

### Phase 4: Advanced Features
- ✅ Surface reconstruction (Ball pivoting, Alpha shapes, Poisson wrapper)
- ✅ Mesh simplification (Quadric error decimation)
- [ ] Real-time visualization
- [ ] Advanced file format support

## Performance Goals

3DCrate aims for high performance through:

- **Zero-cost abstractions**: Minimal runtime overhead
- **SIMD optimization**: Vectorized operations where possible
- **Parallel processing**: Multi-threaded algorithms using Rayon
- **GPU acceleration**: WGPU-based compute shaders
- **Memory efficiency**: Cache-friendly data structures

**Note**: Performance benchmarks will be available once core algorithms are implemented.

## Planned Supported Formats

### Input/Output
- **PLY** - Stanford Polygon Format (skeleton implemented)
- **OBJ** - Wavefront OBJ (skeleton implemented)
- **PCD** - Point Cloud Data (planned)
- **LAS/LAZ** - LIDAR formats (planned)

## Requirements

- **Rust**: 1.70+
- **GPU**: DirectX 11+, Vulkan 1.1+, or Metal 2.0+ (for future GPU features)

## Contributing

We welcome contributions! This project is in early development and there are many opportunities to implement core algorithms.

### Development Setup

```bash
git clone https://github.com/yourusername/3DCrate.git
cd 3DCrate
cargo build --workspace
cargo check --workspace
```

### Contributing Guidelines

- Core algorithms need implementation (currently using `todo!()` placeholders)
- Focus on correctness first, then optimization
- Add tests for new functionality
- Update documentation and examples

## License

Licensed under
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

## Acknowledgments

- Built with [nalgebra](https://nalgebra.org/) for linear algebra
- GPU computing planned with [WGPU](https://wgpu.rs/)
- Visualization planned using [winit](https://github.com/rust-windowing/winit)

---

**Current Status**: Early development phase. The library compiles and basic functionality works, but most algorithms are skeleton implementations awaiting development. See the roadmap above for planned features and implementation timeline.
