# Crate Reference

threecrate is a workspace of focused crates. Use the umbrella crate `threecrate` to pull them all in, or depend on individual crates to keep your build lean.

## threecrate-core

[![Crates.io](https://img.shields.io/crates/v/threecrate-core.svg)](https://crates.io/crates/threecrate-core)

Core data structures and traits.

- `Point3f`, `ColoredPoint3f`, `NormalPoint3f` — point types
- `PointCloud<T>` — generic point cloud with spatial indexing
- `TriangleMesh` — mesh with vertices, faces, normals, and optional UVs
- `Transform3D` — rotation, translation, and scaling
- Optional Bevy game engine integration

## threecrate-algorithms

[![Crates.io](https://img.shields.io/crates/v/threecrate-algorithms.svg)](https://crates.io/crates/threecrate-algorithms)

Point cloud and mesh processing algorithms.

- **Filtering**: voxel grid, radius outlier removal, statistical outlier removal
- **Registration**: ICP (point-to-point and point-to-plane), NDT, global registration (FPFH + RANSAC)
- **Segmentation**: RANSAC plane detection, Euclidean cluster extraction (CPU and parallel)
- **Normal estimation**: K-nearest and radius-based
- **Feature descriptors**: FPFH, SHOT
- **Nearest neighbor search**: KD-tree and brute-force; AVX2/SSE2 SIMD on x86/x86_64
- **Mesh boolean ops**: union, intersection, difference (BSP-based)
- **Mesh smoothing**: Laplacian, Taubin, HC
- **Colorization**: project RGB camera images onto point clouds
- **Streaming processing**: memory-efficient pipeline for large point clouds

## threecrate-gpu

[![Crates.io](https://img.shields.io/crates/v/threecrate-gpu.svg)](https://crates.io/crates/threecrate-gpu)

GPU-accelerated computing via wgpu (Vulkan, Metal, DirectX 12).

- GPU filtering: statistical outlier removal, radius filtering, voxel grid
- GPU normal estimation
- GPU nearest neighbor search (K-nearest and radius)
- GPU ICP registration
- TSDF volume integration and surface extraction
- Real-time point cloud and mesh rendering with PBR material support

## threecrate-io

[![Crates.io](https://img.shields.io/crates/v/threecrate-io.svg)](https://crates.io/crates/threecrate-io)

File I/O for point clouds and meshes.

**Always available:**
- Point clouds: PLY (ASCII/binary), PCD (ASCII/binary), XYZ/CSV
- Meshes: OBJ, PLY
- Auto-detect format from file extension
- Streaming readers for large files
- Memory-mapped I/O (`io-mmap` feature)

**Opt-in:**
- LAS/LAZ — enable `las_laz` feature
- E57 — enable `e57` feature

## threecrate-reconstruction

[![Crates.io](https://img.shields.io/crates/v/threecrate-reconstruction.svg)](https://crates.io/crates/threecrate-reconstruction)

Surface reconstruction from point clouds.

- Poisson (watertight meshes)
- Ball Pivoting Algorithm (BPA) with multi-scale and adaptive radius
- Alpha shapes
- Delaunay triangulation
- Marching Cubes
- Moving Least Squares (MLS) surface fitting
- `auto_reconstruct()` — automatic algorithm selection with quality metrics
- Parallel processing via rayon

## threecrate-simplification

[![Crates.io](https://img.shields.io/crates/v/threecrate-simplification.svg)](https://crates.io/crates/threecrate-simplification)

Mesh simplification and decimation.

- Quadric error metrics / Garland–Heckbert edge collapse
- Edge collapse with boundary preservation
- Clustering-based simplification
- Progressive mesh representation

## threecrate-visualization

[![Crates.io](https://img.shields.io/crates/v/threecrate-visualization.svg)](https://crates.io/crates/threecrate-visualization)

Interactive real-time 3D visualization.

- `show_point_cloud()` and `show_mesh()` for one-line display
- Orbit, pan, zoom (mouse + keyboard)
- GPU-accelerated rendering via wgpu
- Cross-platform: Windows, macOS, Linux
