# threecrate (Python)

Python bindings for [threecrate](https://github.com/rajgandhi1/threecrate) — a high-performance 3D point cloud and mesh processing library written in Rust.

[![PyPI](https://img.shields.io/pypi/v/threecrate.svg)](https://pypi.org/project/threecrate/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate)

## Installation

Pre-built wheels (no Rust required):

```bash
pip install threecrate
```

Build from source (requires Rust and maturin):

```bash
pip install maturin
cd threecrate-python
maturin develop --release
```

## Quick Start

```python
import numpy as np
import threecrate as tc

# Load a point cloud
cloud = tc.read_point_cloud("scan.ply")
print(cloud)  # PointCloud(120000 points)

# Or create from a numpy array (N, 3) float32
pts = np.random.rand(1000, 3).astype(np.float32)
cloud = tc.PointCloud.from_numpy(pts)

# Get points back as numpy
arr = cloud.to_numpy()  # shape (N, 3), dtype float32
```

## API Reference

### Types

| Class | Description |
|---|---|
| `PointCloud` | XYZ point cloud. Construct with `from_numpy()` or `read_point_cloud()`. |
| `NormalPointCloud` | Point cloud with per-point surface normals. Returned by `estimate_normals()`. |
| `TriangleMesh` | Triangle mesh with vertices and faces. |
| `IcpResult` | Registration result: `transformation`, `mse`, `iterations`, `converged`. |

### Filtering

```python
# Voxel grid downsampling
cloud = tc.voxel_downsample(cloud, voxel_size=0.05)

# Statistical outlier removal (default: k=20, std_ratio=2.0)
cloud = tc.remove_statistical_outliers(cloud, k_neighbors=20, std_ratio=2.0)

# Radius outlier removal
cloud = tc.remove_radius_outliers(cloud, radius=0.1, min_neighbors=5)
```

### Normal Estimation

```python
# Estimate normals using K nearest neighbours (default k=10)
normal_cloud = tc.estimate_normals(cloud, k_neighbors=10)

# Access positions and normals as numpy arrays
positions = normal_cloud.positions()  # (N, 3) float32
normals   = normal_cloud.normals()    # (N, 3) float32
```

### Registration

```python
# Point-to-point ICP
result = tc.icp(source, target, max_iterations=50)

print(result.converged)           # True / False
print(result.mse)                 # float
print(result.iterations)          # int
T = result.transformation()       # (4, 4) float32 numpy array
```

### Surface Reconstruction

```python
# Automatic algorithm selection
mesh = tc.reconstruct(cloud)

# Poisson reconstruction (higher quality, requires normals)
normal_cloud = tc.estimate_normals(cloud)
mesh = tc.poisson_reconstruct(normal_cloud)

print(mesh.vertex_count)
print(mesh.face_count)
verts = mesh.vertices()  # (N, 3) float32
faces = mesh.faces()     # (M, 3) uint32
```

### I/O

```python
# Point clouds — PLY, PCD, XYZ, CSV, LAS, LAZ, E57
cloud = tc.read_point_cloud("scan.ply")
tc.write_point_cloud(cloud, "output.pcd")

# Meshes — PLY, OBJ
mesh = tc.read_mesh("model.obj")
tc.write_mesh(mesh, "output.ply")
```

## Full Example

```python
import numpy as np
import threecrate as tc

# Load and preprocess
cloud = tc.read_point_cloud("scene.ply")
cloud = tc.voxel_downsample(cloud, voxel_size=0.02)
cloud = tc.remove_statistical_outliers(cloud, k_neighbors=20, std_ratio=2.0)

# Register two scans
source = tc.read_point_cloud("scan_a.ply")
target = tc.read_point_cloud("scan_b.ply")
result = tc.icp(source, target, max_iterations=100)
if result.converged:
    print(f"Aligned with MSE {result.mse:.4f}")
    print(result.transformation())

# Reconstruct surface
normal_cloud = tc.estimate_normals(cloud, k_neighbors=15)
mesh = tc.poisson_reconstruct(normal_cloud)
tc.write_mesh(mesh, "reconstruction.ply")
print(f"Mesh: {mesh.vertex_count} vertices, {mesh.face_count} faces")
```

## Building from Source

Requirements: Rust 1.70+, Python 3.8+, maturin 1.x

```bash
# Install maturin
pip install maturin

# Development build (editable install)
cd threecrate-python
maturin develop --release

# Build a distributable wheel
maturin build --release --out dist/
pip install dist/threecrate-*.whl
```

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](../LICENSE-MIT) for details.
