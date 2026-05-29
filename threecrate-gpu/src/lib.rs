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
pub mod icp;
pub mod mesh;
pub mod nearest_neighbor;
pub mod normals;
pub mod renderer;
pub mod segmentation;
pub mod tsdf;
pub mod utils;

// Re-export commonly used items
pub use device::GpuContext;
pub use filtering::{
    gpu_radius_outlier_removal, gpu_remove_statistical_outliers, gpu_voxel_grid_filter,
};
pub use icp::gpu_icp;
pub use mesh::{
    mesh_to_gpu_mesh, FlatMaterial, GpuMesh, LodMesh, MeshCameraUniform, MeshLightingParams,
    MeshRenderConfig, MeshRenderer, MeshVertex, PbrMaterial, ShadingMode,
};
pub use nearest_neighbor::{
    gpu_find_k_nearest, gpu_find_k_nearest_batch, gpu_find_radius_neighbors,
};
pub use normals::gpu_estimate_normals;
pub use renderer::{
    colored_point_cloud_to_vertices, point_cloud_to_vertices, point_cloud_to_vertices_colored,
    CameraUniform, PointCloudRenderer, PointVertex, RenderConfig, RenderParams,
};
pub use segmentation::{
    gpu_extract_clusters, gpu_extract_euclidean_clusters, gpu_segment_plane,
    gpu_segment_plane_ransac, GpuClusterConfig, GpuClusterExtractionResult,
    GpuEuclideanClusterConfig, GpuPlaneModel, GpuPlaneSegmentationConfig,
    GpuPlaneSegmentationResult,
};
pub use tsdf::{
    create_tsdf_volume, gpu_tsdf_extract_surface, gpu_tsdf_integrate, CameraIntrinsics, TsdfVolume,
    TsdfVolumeGpu, TsdfVoxel,
};
pub use utils::*;
