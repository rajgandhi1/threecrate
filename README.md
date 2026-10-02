# threecrate

A high-performance 3D point cloud and mesh processing library for Rust, with Python bindings.

![logo_3crate Small](https://github.com/user-attachments/assets/8bd23278-0638-4fb6-a187-5a8d20beebd1)

[![Crates.io](https://img.shields.io/crates/v/threecrate.svg)](https://crates.io/crates/threecrate)
[![PyPI](https://img.shields.io/pypi/v/threecrate.svg)](https://pypi.org/project/threecrate/)
[![Documentation](https://docs.rs/threecrate/badge.svg)](https://docs.rs/threecrate)
[![CI](https://github.com/rajgandhi1/threecrate/actions/workflows/rust.yml/badge.svg)](https://github.com/rajgandhi1/threecrate/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

## What's inside

| Crate | What it does |
|---|---|
| `threecrate-core` | Point, PointCloud, TriangleMesh, Transform3D |
| `threecrate-algorithms` | Filtering, ICP, NDT, global registration, segmentation, normals, FPFH/SHOT, mesh boolean, smoothing |
| `threecrate-gpu` | GPU filtering, segmentation, ICP, normals, nearest-neighbor, TSDF, real-time rendering (wgpu) |
| `threecrate-io` | PLY, OBJ, PCD, XYZ/CSV, LAS/LAZ\*, E57\* — streaming and memory-mapped |
| `threecrate-reconstruction` | Poisson, BPA, alpha shapes, Delaunay, Marching Cubes, MLS, auto-select |
| `threecrate-simplification` | Quadric error, edge collapse, clustering, progressive mesh |
| `threecrate-visualization` | Interactive viewer — orbit/pan/zoom, GPU-accelerated |

\* opt-in feature flags

## Viewer

![ThreeCrate Mesh Viewer](assets/mesh_viewer.png)

## Quick start

**Rust**

```toml
[dependencies]
threecrate = "0.8.0"
```

```rust
use threecrate::prelude::*;

let cloud = read_point_cloud("scan.ply")?;
let cloud = voxel_grid_filter(&cloud, 0.05)?;
let normals = estimate_normals(&cloud, 10)?;
let mesh = auto_reconstruct(&normals)?;
write_mesh("output.obj", &mesh)?;
```

**Python**

```bash
pip install threecrate
```

```python
import threecrate as tc

cloud = tc.read_point_cloud("scan.ply")
cloud = tc.voxel_downsample(cloud, voxel_size=0.05)
normal_cloud = tc.estimate_normals(cloud)
mesh = tc.poisson_reconstruct(normal_cloud)
tc.write_mesh(mesh, "output.ply")
```

## Comparison

| Feature | threecrate | Open3D | PCL |
|---|---|---|---|
| Language | Rust + Python | Python (C++ core) | C++ |
| `pip install` | ✅ | ✅ | ❌ |
| Memory safety | ✅ Rust | ❌ | ❌ |
| GPU compute | ✅ wgpu | ✅ CUDA | Partial |
| Global registration | ✅ FPFH+RANSAC | ✅ | ✅ |
| Surface reconstruction | ✅ 6 algorithms | ✅ | ✅ |
| Streaming I/O | ✅ PLY/OBJ/XYZ | ❌ | ❌ |
| E57 support | ✅ opt-in | ❌ | ❌ |
| WebAssembly | Roadmap | ❌ | ❌ |

## Benchmarks

We benchmarked ThreeCrate against **Open3D 0.19** on the same machine, using
full-resolution frames from three real datasets: TUM RGB-D, KITTI, and
nuScenes-mini. Everything runs on CPU. A ratio above 1 means ThreeCrate is
faster than Open3D.

| Workload | How ThreeCrate compares |
|---|---:|
| Reading files (raw float parsing) | **1.5x to 2.0x faster** |
| Voxel downsampling (CPU) | **1.5x to 1.7x faster** |
| Voxel downsampling (GPU, wgpu) | **1.4x to 2.7x faster** *(vs our own CPU path, not Open3D)* |
| Normal estimation | **1.5x to 2.1x faster** |
| ICP | **1.8x to 3.4x faster**, with the same accuracy |

In short: ThreeCrate is faster than Open3D at every task we measure, and its ICP
results are just as accurate.

About the GPU: the compute backend is [wgpu](https://wgpu.rs/), so it runs on any
GPU (NVIDIA/AMD/Intel/Apple) with no CUDA lock-in. On an RTX 3050 Ti laptop GPU,
ICP is 2.6x to 3.4x faster and normal estimation 1.2x to 2.4x faster than our
16-core CPU path, with the same results. Details in
[docs/benchmarks.md](docs/benchmarks.md).

I also ran ThreeCrate, Open3D and PCL together in one Linux Docker container.
There, ThreeCrate's ICP is 6x to 12x faster than PCL's with the same accuracy,
normals are about even, and PCL's voxel filter is fastest. ThreeCrate and Open3D
are close overall in that container, so the Windows lead above is not universal.

Want the full picture? [docs/benchmarks.md](docs/benchmarks.md) has every number
(full-resolution and capped), how we measured, the caveats we ran into, and the
exact command to reproduce it yourself.

## Docs

- [Installation & feature flags](docs/installation.md)
- [Crate reference](docs/crates.md)
- [Examples](docs/examples.md)
- [Python API reference](threecrate-python/README.md)
- [API docs on docs.rs](https://docs.rs/threecrate)

## Contributing

Contributions are welcome — algorithms, Python bindings, new formats, docs.

- [ROADMAP.md](ROADMAP.md) — where we're ahead, where we trail, and what's next
- [CONTRIBUTING.md](CONTRIBUTING.md) — setup and guidelines
- [Open issues](https://github.com/rajgandhi1/threecrate/issues) — look for `good first issue`
- [GitHub Discussions](https://github.com/rajgandhi1/threecrate/discussions) — questions and ideas

## License

licensed under MIT
