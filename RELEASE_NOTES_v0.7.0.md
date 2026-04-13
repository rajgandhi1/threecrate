# v0.7.0 Release Notes

## Python Bindings

threecrate is now available as a Python package via PyPI.

```
pip install threecrate
```

Wraps the core Rust library using PyO3 and exposes the following API:

- `PointCloud` — create from / convert to numpy arrays (`from_numpy`, `to_numpy`)
- `NormalPointCloud` — point cloud with estimated surface normals
- `TriangleMesh` — triangle mesh with vertex and face access
- `voxel_downsample(cloud, voxel_size)` — voxel grid downsampling
- `remove_statistical_outliers(cloud, k, std_ratio)` — statistical outlier removal
- `remove_radius_outliers(cloud, radius, min_neighbors)` — radius outlier removal
- `estimate_normals(cloud, k_neighbors)` — KNN normal estimation
- `icp(source, target, max_iterations)` — point-to-point ICP registration
- `reconstruct(cloud)` — automatic surface reconstruction
- `poisson_reconstruct(cloud)` — Poisson surface reconstruction
- `read_point_cloud(path)` / `write_point_cloud(cloud, path)` — PLY, PCD, XYZ, CSV, LAS, E57
- `read_mesh(path)` / `write_mesh(mesh, path)` — PLY, OBJ

Wheels are provided for Linux, macOS, and Windows (Python 3.11+).

## CI/CD

- Added GitHub Actions workflow for automated PyPI publishing via OIDC trusted publishers
- Added GitHub Actions workflow for automated crates.io publishing via OIDC trusted publishers
- Both registries publish automatically on GitHub release

## Crates

All crates bumped to `0.7.0`. New crate: `threecrate-python` (not published to crates.io; distributed via PyPI only).
