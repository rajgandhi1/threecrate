//! Visualization and rendering for 3D data
//! 
//! This crate provides real-time visualization capabilities for point clouds
//! and meshes using wgpu and winit:
//! - Interactive 3D viewer with UI controls
//! - Point cloud rendering
//! - Mesh rendering with lighting
//! - Camera controls
//! - Algorithm parameter controls

pub mod camera;
pub mod shaders;
pub mod interactive_viewer;

pub use camera::*;
pub use interactive_viewer::*;

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f};

/// Show a point cloud in an interactive viewer
pub fn show_point_cloud(cloud: &PointCloud<Point3f>) -> Result<()> {
    let mut viewer = InteractiveViewer::new()?;
    viewer.set_point_cloud(cloud);
    viewer.run()
}

/// Show a mesh in an interactive viewer
pub fn show_mesh(mesh: &TriangleMesh) -> Result<()> {
    let mut viewer = InteractiveViewer::new()?;
    viewer.set_mesh(mesh);
    viewer.run()
} 