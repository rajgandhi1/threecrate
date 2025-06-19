//! Visualization and rendering for 3D data
//! 
//! This crate provides real-time visualization capabilities for point clouds
//! and meshes using wgpu and winit:
//! - Interactive 3D viewer
//! - Point cloud rendering
//! - Mesh rendering with lighting
//! - Camera controls

pub mod viewer;
pub mod renderer;
pub mod camera;
pub mod shaders;

pub use viewer::*;
pub use renderer::*;
pub use camera::*;

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f};

/// Show a point cloud in an interactive viewer
pub fn show_point_cloud(_cloud: &PointCloud<Point3f>) -> Result<()> {
    let _viewer = Viewer::new()?;
    // viewer.show_point_cloud(cloud)
    todo!("Point cloud visualization not yet implemented")
}

/// Show a mesh in an interactive viewer
pub fn show_mesh(_mesh: &TriangleMesh) -> Result<()> {
    let _viewer = Viewer::new()?;
    // viewer.show_mesh(mesh)
    todo!("Mesh visualization not yet implemented")
} 