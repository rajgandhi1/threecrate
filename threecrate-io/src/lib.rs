//! I/O operations for point clouds and meshes
//! 
//! This crate provides functionality to read and write various 3D file formats
//! including PLY, OBJ, and other common point cloud and mesh formats.

pub mod ply;
pub mod obj;
pub mod error;

pub use error::*;

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f};

/// Trait for reading point clouds from files
pub trait PointCloudReader {
    fn read_point_cloud<P: AsRef<std::path::Path>>(path: P) -> Result<PointCloud<Point3f>>;
}

/// Trait for writing point clouds to files
pub trait PointCloudWriter {
    fn write_point_cloud<P: AsRef<std::path::Path>>(cloud: &PointCloud<Point3f>, path: P) -> Result<()>;
}

/// Trait for reading meshes from files
pub trait MeshReader {
    fn read_mesh<P: AsRef<std::path::Path>>(path: P) -> Result<TriangleMesh>;
}

/// Trait for writing meshes to files
pub trait MeshWriter {
    fn write_mesh<P: AsRef<std::path::Path>>(mesh: &TriangleMesh, path: P) -> Result<()>;
} 