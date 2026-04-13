# Contributing to threecrate

Thank you for your interest in contributing! threecrate is a modular, high-performance 3D point cloud and mesh processing library written in Rust, with Python bindings via PyO3.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [What to Work On](#what-to-work-on)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

---

## Getting Started

1. **Fork and clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/threecrate.git
   cd threecrate
   git remote add upstream https://github.com/rajgandhi1/threecrate.git
   ```

2. **Build the workspace**
   ```bash
   cargo build --workspace
   cargo test --workspace
   ```

3. **Run an example**
   ```bash
   cargo run --example basic_usage
   cargo run --example ransac_plane_example
   ```

---

## Project Structure

```
threecrate/
├── threecrate-core/           # Point, PointCloud, TriangleMesh, Transform3D
├── threecrate-algorithms/     # Filtering, ICP, segmentation, normals, features, smoothing
├── threecrate-gpu/            # GPU-accelerated compute via wgpu
├── threecrate-io/             # PLY, OBJ, PCD, LAS/LAZ, XYZ/CSV, E57
├── threecrate-reconstruction/ # Poisson, BPA, alpha shapes, Marching Cubes, MLS, Delaunay
├── threecrate-simplification/ # Quadric error, edge collapse, clustering, progressive mesh
├── threecrate-visualization/  # Interactive 3D viewer (wgpu)
├── threecrate-python/         # Python bindings (PyO3 + maturin)
├── threecrate-umbrella/       # `threecrate` re-export crate published to crates.io
└── examples/                  # Runnable examples for every major feature
```

---

## Development Setup

### Rust (all crates)

Requirements: **Rust 1.75+**

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

### Python bindings

Requirements: **Python 3.8+**, **maturin 1.x**

```bash
pip install maturin
cd threecrate-python
maturin develop --release   # installs an editable wheel into the current venv
python -c "import threecrate as tc; print(tc.PointCloud())"
```

To run the full build as CI does:
```bash
maturin build --release --out dist/
pip install dist/threecrate-*.whl --force-reinstall
```

### Benchmarks

```bash
cargo bench -p threecrate-bench
```

---

## What to Work On

Check the [issues list](https://github.com/rajgandhi1/threecrate/issues) for open tasks. Issues labelled **`good first issue`** are intentionally scoped to be approachable.

### Current gaps in the Python bindings (`threecrate-python`)

The Rust library is feature-complete in most areas, but the Python API still has significant gaps. These are great entry points for contributors:

| Feature | Rust crate | Python status |
|---|---|---|
| Point-to-plane ICP | `threecrate-algorithms` | Not exposed |
| Global registration (FPFH+RANSAC) | `threecrate-algorithms` | Not exposed |
| Radius-based normal estimation | `threecrate-algorithms` | Not exposed |
| Mesh boolean ops (union/intersect/diff) | `threecrate-algorithms` | Not exposed |
| Ball Pivoting reconstruction | `threecrate-reconstruction` | Not exposed |
| Alpha shapes reconstruction | `threecrate-reconstruction` | Not exposed |
| Delaunay triangulation | `threecrate-reconstruction` | Not exposed |
| Moving Least Squares | `threecrate-reconstruction` | Not exposed |
| Edge collapse simplification | `threecrate-simplification` | Not exposed |
| Clustering simplification | `threecrate-simplification` | Not exposed |

The pattern for adding a new binding is in `threecrate-python/src/lib.rs`. Look at how `segment_plane` or `simplify_mesh` are implemented — it's usually 15–30 lines.

### Other areas

- **New file formats**: STL, glTF/GLB support in `threecrate-io`
- **Examples**: new `examples/` entries demonstrating existing features
- **Documentation**: `///` doc comments on any public item that lacks them
- **Benchmarks**: add criterion benchmarks in `threecrate-bench` for algorithms that don't have one yet

---

## Code Style

| Tool | Command | Required |
|---|---|---|
| Formatting | `cargo fmt` | Yes |
| Linting | `cargo clippy --workspace -- -D warnings` | Yes |
| Doc tests | `cargo test --doc` | For public API |

**Error handling**: use `threecrate_core::Result<T>` (aliased to `Result<T, threecrate_core::Error>`) for all fallible public functions. Do not use `unwrap()` or `expect()` in library code.

**Doc comment format**:
```rust
/// One-line summary of what this does.
///
/// Longer explanation if needed. Include the algorithm name and a reference
/// if it implements a known technique.
///
/// # Arguments
/// * `cloud` - The input point cloud (must be non-empty)
/// * `k` - Number of nearest neighbours
///
/// # Returns
/// A new `PointCloud<NormalPoint3f>` with estimated surface normals.
///
/// # Errors
/// Returns `Err` if `cloud` is empty or `k` is zero.
pub fn estimate_normals(cloud: &PointCloud<Point3f>, k: usize) -> Result<PointCloud<NormalPoint3f>> {
```

---

## Testing

**All new algorithms must include tests.**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{PointCloud, Point3f};

    #[test]
    fn test_algorithm_basic() {
        let cloud = PointCloud::from_points(vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
        ]);
        let result = your_algorithm(&cloud, /* params */);
        assert!(result.is_ok());
    }

    #[test]
    fn test_algorithm_empty_input_errors() {
        let empty = PointCloud::new();
        assert!(your_algorithm(&empty, /* params */).is_err());
    }
}
```

Run tests:
```bash
cargo test --workspace                  # all crates
cargo test -p threecrate-algorithms     # one crate
cargo test algorithm_name               # one test by name
```

---

## Pull Request Process

1. **Branch from main**
   ```bash
   git checkout main && git pull upstream main
   git checkout -b your-feature-branch
   ```

2. **Before pushing**, make sure these all pass:
   ```bash
   cargo fmt --check
   cargo clippy --workspace -- -D warnings
   cargo test --workspace
   ```

3. **PR title format**: `Add <feature>` / `Fix <bug>` / `Improve <thing>` — one line, no period.

4. **Keep PRs focused** — one feature or fix per PR. Avoid mixing refactors with new functionality.

5. A maintainer will review and merge. For larger changes, open an issue first to align on approach before writing code.

---

Questions? Open a [GitHub Discussion](https://github.com/rajgandhi1/threecrate/discussions) or file an issue.
