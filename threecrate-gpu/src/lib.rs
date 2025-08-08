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
//! use threecrate_gpu::{GpuContext, gpu_remove_statistical_outliers, gpu_radius_outlier_removal, gpu_voxel_grid_filter};
//! use threecrate_core::{PointCloud, Point3f};
//!
//! async fn example() -> threecrate_core::Result<()> {
//!     let gpu_context = GpuContext::new().await?;
//!     
//!     let mut point_cloud = PointCloud::<Point3f>::new();
//!     // ... populate point cloud
//!     
//!     // Compute normals using GPU acceleration
//!     let normals = gpu_context.compute_normals(&point_cloud.points, 10).await?;
//!     
//!     // Filter outliers using GPU acceleration
//!     let filtered = gpu_remove_statistical_outliers(&gpu_context, &point_cloud, 10, 1.0).await?;
//!     
//!     // Remove isolated points using radius-based filtering
//!     let filtered = gpu_radius_outlier_removal(&gpu_context, &point_cloud, 0.1, 5).await?;
//!     
//!     // Downsample using voxel grid filtering
//!     let downsampled = gpu_voxel_grid_filter(&gpu_context, &point_cloud, 0.05).await?;
//!     
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
pub mod mesh;
pub mod utils;

// Re-export commonly used items
pub use device::GpuContext;
pub use filtering::{gpu_remove_statistical_outliers, gpu_radius_outlier_removal, gpu_voxel_grid_filter};
pub use normals::gpu_estimate_normals;
pub use nearest_neighbor::{gpu_find_k_nearest, gpu_find_k_nearest_batch, gpu_find_radius_neighbors};
pub use icp::gpu_icp;
pub use tsdf::{gpu_tsdf_integrate, gpu_tsdf_extract_surface, create_tsdf_volume, TsdfVolume, TsdfVoxel, CameraIntrinsics, TsdfVolumeGpu};
pub use renderer::{
    PointCloudRenderer, PointVertex, RenderConfig, RenderParams, CameraUniform,
    point_cloud_to_vertices, point_cloud_to_vertices_colored, colored_point_cloud_to_vertices
};
pub use mesh::{
    MeshRenderer, MeshVertex, MeshCameraUniform, PbrMaterial, FlatMaterial, 
    MeshLightingParams, MeshRenderConfig, GpuMesh, ShadingMode, mesh_to_gpu_mesh
};
pub use utils::*; 