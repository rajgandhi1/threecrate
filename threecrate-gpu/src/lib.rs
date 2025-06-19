//! # ThreeCrate GPU
//!
//! GPU-accelerated computing for 3D point cloud processing using WGPU.
//!
//! This crate provides GPU-accelerated implementations of common 3D point cloud
//! processing algorithms, leveraging the power of modern graphics hardware.

pub mod device;
pub mod filtering;
pub mod normals;
pub mod nearest_neighbor;
pub mod icp;
pub mod utils;

// Re-export commonly used items
pub use device::*;
pub use filtering::*;
pub use normals::*;
pub use nearest_neighbor::*;
pub use icp::*;
pub use utils::*; 