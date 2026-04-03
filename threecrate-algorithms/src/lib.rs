//! Algorithms for 3D point cloud and mesh processing

pub mod nearest_neighbor;
pub mod point_cloud_ops;
pub mod filtering;
pub mod normals;
pub mod registration;
pub mod ndt_registration;
pub mod global_registration;
pub mod segmentation;
pub mod features;
pub mod mesh_boolean;
pub mod mesh_smoothing;
pub mod colorization;
pub mod simd_distance;
pub mod streaming;

// Re-export commonly used items
pub use filtering::*;
pub use normals::*;
pub use nearest_neighbor::*;
pub use registration::*;
pub use ndt_registration::*;
pub use global_registration::*;
pub use segmentation::*;
pub use features::*;
pub use mesh_boolean::*;
pub use mesh_smoothing::*;
pub use colorization::*;
pub use simd_distance::*;