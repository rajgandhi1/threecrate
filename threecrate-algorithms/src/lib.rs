//! Algorithms for 3D point cloud and mesh processing

pub mod colorization;
pub mod features;
pub mod filtering;
pub mod gicp;
pub mod global_registration;
pub mod ground_segmentation;
pub mod kiss_icp;
pub mod mesh_boolean;
pub mod mesh_smoothing;
pub mod ndt_registration;
pub mod nearest_neighbor;
pub mod normals;
pub mod point_cloud_ops;
pub mod registration;
pub mod segmentation;
pub mod simd_distance;
pub mod streaming;

// Re-export commonly used items
pub use colorization::*;
pub use features::*;
pub use filtering::*;
pub use gicp::{gicp, GicpConfig};
pub use global_registration::*;
pub use ground_segmentation::{
    patchwork_plus_plus, segment_ground, GroundSegmentationResult, PatchworkConfig,
};
pub use kiss_icp::{kiss_icp, KissIcpConfig};
pub use mesh_boolean::*;
pub use mesh_smoothing::*;
pub use ndt_registration::*;
pub use nearest_neighbor::*;
pub use normals::*;
pub use registration::*;
pub use segmentation::*;
pub use simd_distance::*;
