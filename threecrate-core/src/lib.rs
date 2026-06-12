//! Core data structures and traits for threecrate
//!
//! This crate provides fundamental types for 3D point cloud and mesh processing,
//! including points, point clouds, meshes, and essential traits.

pub mod error;
pub mod mesh;
pub mod organized_point_cloud;
pub mod point;
pub mod point_cloud;
pub mod traits;
pub mod transform;

#[cfg(feature = "bevy_interop")]
pub mod bevy_interop;

pub use error::*;
pub use mesh::*;
pub use organized_point_cloud::*;
pub use point::*;
pub use point_cloud::*;
pub use traits::*;
pub use transform::*;

/// Re-export commonly used types from nalgebra
pub use nalgebra::{Isometry3, Matrix3, Matrix4, Point3, Transform3, Vector3};

/// Common result type for threecrate operations
pub type Result<T> = std::result::Result<T, Error>;

// Type aliases for easier imports
pub type Point = Point3f;
pub type Mesh = TriangleMesh;
