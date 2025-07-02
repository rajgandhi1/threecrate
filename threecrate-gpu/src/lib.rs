//! # ThreeCrate GPU
//!
//! GPU-accelerated computing for 3D point cloud processing using WGPU.
//!
//! This crate provides GPU-accelerated implementations of common 3D point cloud
//! processing algorithms, leveraging the power of modern graphics hardware.
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use threecrate_gpu::GpuContext;
//! use threecrate_core::{PointCloud, Point3f};
//!
//! async fn example() -> threecrate_core::Result<()> {
//!     let gpu_context = GpuContext::new().await?;
//!     
//!     let mut point_cloud = PointCloud::<Point3f>::new();
//!     // ... populate point cloud
//!     
//!     let normals = gpu_context.compute_normals(&point_cloud.points, 10).await?;
//!     Ok(())
//! }
//! ```

pub mod device;
pub mod filtering;
pub mod normals;
pub mod nearest_neighbor;
pub mod icp;
pub mod tsdf;
pub mod renderer;
pub mod utils;

// Re-export commonly used items
pub use device::GpuContext;
pub use filtering::gpu_remove_statistical_outliers;
pub use normals::gpu_estimate_normals;
pub use nearest_neighbor::*;
pub use icp::gpu_icp;
pub use tsdf::{gpu_tsdf_integrate, gpu_tsdf_extract_surface, create_tsdf_volume, TsdfVolume, TsdfVoxel, CameraIntrinsics, TsdfVolumeGpu};
pub use renderer::{PointCloudRenderer, PointVertex, RenderConfig, point_cloud_to_vertices, point_cloud_to_vertices_colored};
pub use utils::*; 