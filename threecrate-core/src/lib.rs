//! Core data structures and traits for threecrate
//! 
//! This crate provides fundamental types for 3D point cloud and mesh processing,
//! including points, point clouds, meshes, and essential traits.

pub mod point;
pub mod point_cloud;
pub mod mesh;
pub mod traits;
pub mod transform;
pub mod error;

#[cfg(feature = "bevy_interop")]
pub mod bevy_interop;

pub use point::*;
pub use point_cloud::*;
pub use mesh::*;
pub use traits::*;
pub use transform::*;
pub use error::*;

/// Re-export commonly used types from nalgebra
pub use nalgebra::{Point3, Vector3, Matrix3, Matrix4, Isometry3, Transform3};

/// Common result type for threecrate operations
pub type Result<T> = std::result::Result<T, Error>;

// Type aliases for easier imports
pub type Point = Point3f;
pub type Mesh = TriangleMesh; 