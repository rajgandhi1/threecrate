# threecrate

A comprehensive 3D point cloud and mesh processing library for Rust.

![logo_3crate Small](https://github.com/user-attachments/assets/8bd23278-0638-4fb6-a187-5a8d20beebd1)

[![Crates.io](https://img.shields.io/crates/v/threecrate.svg)](https://crates.io/crates/threecrate)
[![PyPI](https://img.shields.io/pypi/v/threecrate.svg)](https://pypi.org/project/threecrate/)
[![Documentation](https://docs.rs/threecrate/badge.svg)](https://docs.rs/threecrate)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate)
[![CI](https://github.com/rajgandhi1/threecrate/actions/workflows/rust.yml/badge.svg)](https://github.com/rajgandhi1/threecrate/actions)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Overview

threecrate is a modular, high-performance library for 3D point cloud and mesh processing in Rust. It is organized as a workspace of specialized crates, allowing you to include only the functionality you need.

## Features

- **Core**: 3D data structures (Point, PointCloud, TriangleMesh, Transform)
- **Algorithms**: Filtering, registration (ICP, NDT, global), segmentation, normal estimation, feature descriptors (FPFH, SHOT), mesh boolean operations, mesh smoothing, point cloud colorization, SIMD-accelerated distance computations, and streaming processing
- **GPU**: GPU-accelerated filtering, normal estimation, ICP, nearest neighbor search, TSDF volume integration, and real-time rendering via wgpu
- **I/O**: PLY, OBJ, PCD, LAS/LAZ, XYZ/CSV, E57 with streaming and memory-mapped I/O support
- **Reconstruction**: Poisson, Ball Pivoting, Alpha Shapes, Delaunay, Marching Cubes, Moving Least Squares, and automatic algorithm selection
- **Simplification**: Quadric error metrics, edge collapse, clustering-based, and progressive mesh simplification
- **Visualization**: Interactive 3D viewer with orbit, pan, and zoom controls

## Viewer

![ThreeCrate Mesh Viewer](assets/mesh_viewer.png)

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
threecrate = "0.6.0"
```

Basic usage:

```rust
use threecrate::prelude::*;

// Create a point cloud
let points = vec![
    Point3f::new(0.0, 0.0, 0.0),
    Point3f::new(1.0, 0.0, 0.0),
    Point3f::new(0.0, 1.0, 0.0),
];
let cloud = PointCloud::from_points(points);
println!("Point cloud with {} points", cloud.len());
```

## Installation Options

### Option 1: Umbrella Crate (Recommended)

```toml
[dependencies]
threecrate = "0.6.0"
```

### Option 2: Individual Crates (Minimal dependencies)

```toml
[dependencies]
threecrate-core = "0.6.0"
threecrate-algorithms = "0.6.0"
threecrate-gpu = "0.6.0"
threecrate-io = "0.6.0"
threecrate-simplification = "0.6.0"
threecrate-reconstruction = "0.6.0"
threecrate-visualization = "0.6.0"
```

## Feature Flags

```toml
[dependencies]
threecrate = { version = "0.6.0", features = ["all"] }
```

Available features:
- `default`: core, algorithms, io, simplification
- `core`: Core data structures (always enabled)
- `algorithms`: Point cloud processing algorithms
- `gpu`: GPU-accelerated processing
- `io`: File format support
- `simplification`: Mesh and point cloud simplification
- `reconstruction`: Surface reconstruction
- `visualization`: Interactive visualization
- `all`: All features

## Individual Crates

### [`threecrate-core`](https://crates.io/crates/threecrate-core)

Core data structures and traits.

- `Point3f`, `ColoredPoint3f`, `NormalPoint3f` — point types
- `PointCloud<T>` — generic point cloud with spatial operations
- `TriangleMesh` — mesh with vertices, faces, normals, and optional UVs
- `Transform3D` — rotation, translation, and scaling
- Optional Bevy game engine integration

### [`threecrate-algorithms`](https://crates.io/crates/threecrate-algorithms)

Algorithms for point cloud and mesh processing.

- **Filtering**: Voxel grid, radius outlier removal, statistical outlier removal
- **Registration**: ICP (point-to-point and point-to-plane), NDT, global registration (FPFH + RANSAC)
- **Segmentation**: RANSAC plane detection, Euclidean cluster extraction (CPU and parallel)
- **Normal Estimation**: K-nearest and radius-based, with configurable orientation
- **Feature Descriptors**: FPFH and SHOT
- **Nearest Neighbor Search**: KD-tree and brute-force, with SIMD-accelerated distance computations
- **Mesh Boolean Operations**: Union, intersection, and difference
- **Mesh Smoothing**: Laplacian, Taubin, and HC smoothing
- **Colorization**: Project colors from registered RGB camera images onto point clouds
- **Streaming Processing**: Memory-efficient pipeline for large point clouds

### [`threecrate-gpu`](https://crates.io/crates/threecrate-gpu)

GPU-accelerated computing via wgpu (Vulkan, Metal, DirectX 12).

- GPU filtering: statistical outlier removal, radius filtering, voxel grid
- GPU normal estimation
- GPU nearest neighbor search (K-nearest and radius)
- GPU ICP registration
- TSDF volume integration and surface extraction
- Real-time point cloud and mesh rendering with PBR material support

### [`threecrate-io`](https://crates.io/crates/threecrate-io)

File I/O for point clouds and meshes.

**Supported formats:**
- Point clouds: PLY (ASCII/binary), PCD (ASCII/binary), LAS/LAZ (LiDAR), XYZ/CSV, E57
- Meshes: OBJ, PLY

**Advanced features:**
- Auto-detect format from file extension
- Streaming readers for memory-efficient processing of large files
- Memory-mapped I/O for large binary files
- Attribute-preserving read/write with metadata support

### [`threecrate-reconstruction`](https://crates.io/crates/threecrate-reconstruction)

Surface reconstruction from point clouds.

- Poisson surface reconstruction (watertight meshes)
- Ball Pivoting Algorithm (BPA) with multi-scale and adaptive radius
- Alpha shapes for non-convex surfaces
- Delaunay triangulation
- Marching Cubes for volumetric/implicit surfaces
- Moving Least Squares (MLS) surface fitting
- `auto_reconstruct()` — unified pipeline with automatic algorithm selection and quality metrics
- Parallel processing via rayon

### [`threecrate-simplification`](https://crates.io/crates/threecrate-simplification)

Mesh simplification and decimation.

- Quadric error metrics (topology-preserving edge collapse)
- Edge collapse simplification with boundary preservation
- Clustering-based simplification
- Progressive mesh representation
- Quality metrics: Hausdorff distance, volume preservation, normal deviation

### [`threecrate-visualization`](https://crates.io/crates/threecrate-visualization)

Interactive real-time 3D visualization.

- `show_point_cloud()` and `show_mesh()` for quick display
- Orbit, pan, and zoom camera controls (mouse drag, scroll, keyboard shortcuts)
- GPU-accelerated rendering via wgpu
- Cross-platform (Windows, macOS, Linux)

## Examples

### Point Cloud Processing

```rust
use threecrate::prelude::*;

// Load a point cloud
let cloud = read_point_cloud("scan.ply")?;

// Filter and estimate normals
let cloud = voxel_grid_filter(&cloud, 0.05)?;
let cloud = statistical_outlier_removal(&cloud, 10, 2.0)?;
let normals_cloud = estimate_normals(&cloud, 10)?;

// Register two scans with ICP
let result = icp(&source, &target, Default::default())?;

// Surface reconstruction
let mesh = auto_reconstruct(&normals_cloud)?;
println!("Reconstructed mesh: {} triangles", mesh.face_count());

// Save result
write_mesh("output.obj", &mesh)?;
```

### GPU Acceleration

```rust
use threecrate::prelude::*;

let gpu = GpuContext::new().await?;

// GPU-accelerated filtering
let filtered = gpu_voxel_grid_filter(&gpu, &cloud, 0.05).await?;

// GPU ICP registration
let result = gpu_icp(&gpu, &source, &target).await?;
```

### Interactive Visualization

```rust
use threecrate::prelude::*;

show_point_cloud(&cloud)?;
show_mesh(&mesh)?;
```

## Python Bindings

threecrate is available as a pip-installable Python package built with PyO3 and maturin:

```bash
pip install threecrate
```

```python
import numpy as np
import threecrate as tc

# Load and filter
cloud = tc.read_point_cloud("scan.ply")
cloud = tc.voxel_downsample(cloud, voxel_size=0.02)
cloud = tc.remove_statistical_outliers(cloud)

# Register two scans
result = tc.icp(source, target, max_iterations=100)
print(result.converged, result.transformation())

# Reconstruct surface
normal_cloud = tc.estimate_normals(cloud)
mesh = tc.poisson_reconstruct(normal_cloud)
tc.write_mesh(mesh, "output.ply")
```

NumPy interop is first-class — `PointCloud.from_numpy()` and `.to_numpy()` convert to/from `(N, 3)` float32 arrays with zero unnecessary copies.

See [`threecrate-python/README.md`](threecrate-python/README.md) for the full Python API reference and build instructions.

## Performance

- **Parallel CPU processing**: rayon-based parallelism across algorithms
- **GPU acceleration**: optional wgpu-based GPU compute for filtering, registration, and rendering
- **SIMD distance computations**: optimized nearest neighbor search
- **Streaming I/O**: process large point clouds without loading everything into memory
- **Spatial indexing**: KD-tree and R*-tree for efficient neighbor queries

## Comparison

| Feature | threecrate | Open3D (Python) | PCL (C++) |
|---|---|---|---|
| Language | Rust + Python | Python (C++ core) | C++ |
| pip install | ✅ | ✅ | ❌ |
| Memory safety | ✅ (Rust) | ❌ | ❌ |
| GPU (wgpu) | ✅ | Partial | Partial |
| Parallel CPU | ✅ rayon | ✅ | ✅ |
| ICP registration | ✅ | ✅ | ✅ |
| Global registration | ✅ FPFH+RANSAC | ✅ | ✅ |
| Surface reconstruction | ✅ 6 algorithms | ✅ | ✅ |
| Mesh smoothing | ✅ Laplacian/Taubin/HC | ✅ | ✅ |
| Mesh boolean ops | ✅ | ✅ | ❌ |
| Streaming I/O | ✅ | ❌ | ❌ |
| LAS/LAZ support | ✅ | ✅ | ✅ |
| E57 support | ✅ | ❌ | ❌ |
| Async/WebAssembly | Roadmap | ❌ | ❌ |

## Contributing

Contributions are very welcome — from bug fixes and new algorithms to Python bindings and documentation.

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines
- Browse [open issues](https://github.com/rajgandhi1/threecrate/issues) — issues labelled **`good first issue`** are a great starting point
- Ask questions or propose ideas in [GitHub Discussions](https://github.com/rajgandhi1/threecrate/discussions)

The biggest area for contribution right now is **expanding the Python API** — most Rust algorithms are implemented but not yet exposed to Python. See the [gap table in CONTRIBUTING.md](CONTRIBUTING.md#current-gaps-in-the-python-bindings) for specifics.

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) for details.
