# 3DCrate

**[Examples](#running-examples)** | **[Documentation](https://docs.rs/threecrate-core)** | **[Contribution Guide](CONTRIBUTING.md)**

## About

3DCrate is a **comprehensive 3D point cloud processing library** for Rust, providing **high-performance algorithms** for point cloud manipulation, analysis, and visualization. Built with a focus on **GPU acceleration** and **modular architecture**.

As a 3D processing library, the project's main goal is to provide **a solid foundation for developers to build on top of**, whether for robotics applications, 3D scanning pipelines, computer vision systems, or specialized point cloud processing tools.

In doing so, ThreeCrate follows these principles:

* **GPU-first design** - Leverage modern graphics hardware for maximum performance using WGPU compute shaders
* **Modular architecture** - Pick and choose only the components you need for your specific use case  
* **Safety and reliability** - Rust's memory safety guarantees combined with comprehensive error handling
* **Cross-platform compatibility** - Support for Windows, macOS, Linux, and WebAssembly through WGPU
* **Real-time performance** - Designed for interactive applications requiring low-latency processing

ThreeCrate is actively developed and suitable for production use in specialized applications, though some advanced features are still being implemented.

## Status

3DCrate is usable for a wide range of point cloud processing tasks including normals estimation, ICP registration, RANSAC segmentation, and surface reconstruction. GPU acceleration is available for core algorithms with ongoing expansion to additional operations.

Current capabilities support:
- Point clouds up to millions of points with GPU acceleration
- Real-time normal estimation and surface reconstruction  
- Multi-format I/O (PLY, OBJ with more planned)
- Cross-platform deployment including WebAssembly

## Overview

3DCrate features a modular architecture, allowing you to pick and choose which parts of it you want to use. It is made up of the following libraries:

* **threecrate-core**: Core data structures (`PointCloud`, `TriangleMesh`, `Point3f`) and fundamental traits
* **threecrate-algorithms**: CPU-based point cloud processing algorithms (normals, ICP, RANSAC, filtering)
* **threecrate-gpu**: GPU-accelerated computing using WGPU with compute shaders for high-performance operations
* **threecrate-io**: File I/O support for 3D formats (PLY, OBJ, with PCD and LAS/LAZ planned)
* **threecrate-reconstruction**: Surface reconstruction algorithms (Ball Pivoting, Alpha Shapes, Poisson)
* **threecrate-simplification**: Mesh and point cloud simplification algorithms (Quadric Error Decimation)
* **threecrate-visualization**: Real-time 3D visualization and rendering capabilities

## Usage

3DCrate is a set of Rust libraries (see list above). Add the components you need to your `Cargo.toml`:

```toml
[dependencies]
threecrate-core = "0.1.0"
threecrate-algorithms = "0.1.0"        # CPU algorithms
threecrate-gpu = "0.1.0"               # GPU acceleration  
threecrate-reconstruction = "0.1.0"     # Surface reconstruction
threecrate-io = "0.1.0"                # File I/O
```

### Basic Example

```rust
use threecrate_core::{PointCloud, Point3f};
use threecrate_algorithms::{estimate_normals, segment_plane};
use threecrate_reconstruction::ball_pivoting_reconstruction;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a point cloud
    let points = vec![
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(1.0, 0.0, 0.0),
        Point3f::new(0.0, 1.0, 0.0),
        Point3f::new(0.0, 0.0, 1.0),
    ];
    
    let cloud = PointCloud::from_points(points);
    
    // Estimate surface normals
    let normals_cloud = estimate_normals(&cloud, 3)?;
    
    // Detect planes with RANSAC
    let plane_result = segment_plane(&cloud, 0.01, 1000)?;
    
    // Surface reconstruction
    let mesh = ball_pivoting_reconstruction(&cloud, 0.1)?;
    
    Ok(())
}
```

### GPU-Accelerated Example

```rust
use threecrate_gpu::{GpuContext, gpu_estimate_normals, gpu_icp};
use threecrate_core::PointCloud;

#[tokio::main] 
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let gpu_context = GpuContext::new().await?;
    
    // GPU-accelerated normal estimation
    let mut cloud = PointCloud::from_file("model.ply")?;
    let normals_cloud = gpu_estimate_normals(&gpu_context, &mut cloud, 10).await?;
    
    // GPU-accelerated ICP registration
    let transform = gpu_icp(&gpu_context, &source_cloud, &target_cloud, 50, 0.001, 0.1).await?;
    
    Ok(())
}
```

### Running Examples

* To run basic examples: `cargo run --bin basic_usage`
* To run GPU examples: `cargo run --bin comprehensive_gpu_example`  
* To see all available examples: `ls examples/`

## Building

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Check compilation  
cargo check --workspace
```

**Requirements:**
- Rust 1.70+
- GPU with Vulkan 1.1+, DirectX 11+, or Metal 2.0+ (for GPU features)

## Community

If you are interested in 3DCrate, please consider joining the community. We'd love to have you!

* **Issues**: Report bugs and request features on GitHub
* **Discussions**: Share your projects and ask questions
* **Examples**: Check out community projects using 3DCrate

## Get Involved

If you are interested in helping out, just fork the GitHub repository and submit a pull request:

* **Main repository**: Submit PRs for new features, bug fixes, or improvements
* **Good first issues**: Look for issues labeled "good first issue" for contribution ideas

If you don't know what to work on, check out our roadmap in the issues or ask in discussions. Areas where we especially welcome contributions:

* GPU shader optimization
* New file format support  
* Algorithm implementations
* Performance benchmarks
* Documentation and examples

## License

This project is open source, licensed under the terms of the MIT License. This means you can use it in both open source and commercial projects.

See `LICENSE` for full details.