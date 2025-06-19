//! # ThreeCrate Algorithms
//!
//! A collection of algorithms for 3D point cloud and mesh processing.
//!
//! This crate provides various algorithms for processing 3D point clouds and meshes,
//! including filtering, normal estimation, registration, segmentation, and feature detection.

pub mod filtering;
pub mod normals;
pub mod nearest_neighbor;
pub mod registration;
pub mod segmentation;
pub mod features;

// Re-export commonly used items
pub use filtering::*;
pub use normals::*;
pub use nearest_neighbor::*;
pub use registration::*;
pub use segmentation::*;
pub use features::*; 