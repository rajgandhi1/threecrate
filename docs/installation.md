# Installation

## Rust

### Umbrella crate (recommended)

Includes core, algorithms, I/O, and simplification by default:

```toml
[dependencies]
threecrate = "0.7.1"
```

### Individual crates (minimal dependencies)

Pick only what you need:

```toml
[dependencies]
threecrate-core = "0.7.1"
threecrate-algorithms = "0.7.1"
threecrate-io = "0.7.1"
threecrate-reconstruction = "0.7.1"
threecrate-simplification = "0.7.1"
threecrate-gpu = "0.7.1"
threecrate-visualization = "0.7.1"
```

### Feature flags

```toml
[dependencies]
threecrate = { version = "0.7.1", features = ["all"] }
```

| Feature | Description | Default |
|---|---|---|
| `core` | Core data structures | ✅ always |
| `algorithms` | Point cloud processing algorithms | ✅ |
| `io` | File format support (PLY, OBJ, PCD, XYZ) | ✅ |
| `simplification` | Mesh simplification | ✅ |
| `reconstruction` | Surface reconstruction | ❌ opt-in |
| `gpu` | GPU-accelerated compute and rendering | ❌ opt-in |
| `visualization` | Interactive 3D viewer | ❌ opt-in |
| `all` | Everything above | ❌ opt-in |

#### I/O opt-in features

These formats require extra dependencies and must be enabled explicitly:

```toml
threecrate-io = { version = "0.7.1", features = ["las_laz", "e57", "io-mmap"] }
```

| Feature | Formats | Notes |
|---|---|---|
| `las_laz` | LAS, LAZ (LiDAR) | via pasture |
| `e57` | E57 | via e57 crate |
| `io-mmap` | Memory-mapped binary reads | large file optimization |

## Python

Pre-built wheels — no Rust required:

```bash
pip install threecrate
```

Supported: Python 3.8+, Linux / macOS / Windows.

### Build from source

Requires Rust 1.75+ and maturin 1.x:

```bash
pip install maturin
cd threecrate-python
maturin develop --release
```
