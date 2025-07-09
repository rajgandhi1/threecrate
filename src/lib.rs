//! # 3DCrate
//!
//! A comprehensive 3D point cloud processing library for Rust.
//!
//! This is the umbrella crate that provides convenient access to all 3DCrate functionality.
//! You can use this crate to get everything in one place, or use individual crates for
//! more granular control over dependencies.
//!
//! ## Features
//!
//! - **Core**: Basic 3D data structures (Point, PointCloud, Mesh, etc.)
//! - **Algorithms**: Point cloud processing algorithms (filtering, registration, etc.)
//! - **GPU**: GPU-accelerated processing using wgpu
//! - **I/O**: File format support (PLY, OBJ, LAS, etc.)
//! - **Simplification**: Mesh and point cloud simplification algorithms
//! - **Reconstruction**: Surface reconstruction from point clouds
//! - **Visualization**: Interactive 3D visualization tools
//!
//! ## Quick Start
//!
//! ```rust
//! use threecrate::prelude::*;
//!
//! // Create a point cloud
//! let points = vec![
//!     Point3D::new(0.0, 0.0, 0.0),
//!     Point3D::new(1.0, 0.0, 0.0),
//!     Point3D::new(0.0, 1.0, 0.0),
//! ];
//! let cloud = PointCloud::from_points(points);
//!
//! // Apply algorithms
//! let filtered = cloud.statistical_filter(50, 1.0);
//! ```
//!
//! ## Feature Flags
//!
//! - `default`: Enables core, algorithms, io, and simplification
//! - `core`: Core data structures (always enabled)
//! - `algorithms`: Point cloud processing algorithms
//! - `gpu`: GPU-accelerated processing
//! - `io`: File format support
//! - `simplification`: Mesh and point cloud simplification
//! - `reconstruction`: Surface reconstruction (coming soon)
//! - `visualization`: Interactive visualization (coming soon)
//! - `all`: Enables all features

// Re-export core functionality
pub use threecrate_core::*;

// Re-export sub-crates
#[cfg(feature = "algorithms")]
pub use threecrate_algorithms as algorithms;

#[cfg(feature = "gpu")]
pub use threecrate_gpu as gpu;

#[cfg(feature = "io")]
pub use threecrate_io as io;

#[cfg(feature = "simplification")]
pub use threecrate_simplification as simplification;

// TODO: Enable once published
// #[cfg(feature = "reconstruction")]
// pub use threecrate_reconstruction as reconstruction;

// #[cfg(feature = "visualization")]
// pub use threecrate_visualization as visualization;

/// Convenient imports for common use cases
pub mod prelude {
    pub use threecrate_core::*;
    
    #[cfg(feature = "algorithms")]
    pub use threecrate_algorithms::*;
    
    #[cfg(feature = "io")]
    pub use threecrate_io::*;
    
    #[cfg(feature = "simplification")]
    pub use threecrate_simplification::*;
    
    // TODO: Enable once published
    // #[cfg(feature = "reconstruction")]
    // pub use threecrate_reconstruction::*;
    
    // #[cfg(feature = "visualization")]
    // pub use threecrate_visualization::*;
} 