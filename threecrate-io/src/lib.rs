//! I/O operations for point clouds and meshes
//! 
//! This crate provides functionality to read and write various 3D file formats
//! including PLY, OBJ, and other common point cloud and mesh formats.

pub mod ply;
pub mod obj;
#[cfg(feature = "las_laz")]
pub mod pasture;
pub mod pcd;
pub mod xyz_csv;
pub mod error;
pub mod registry;
pub mod mesh_attributes;
pub mod serialization;

#[cfg(test)]
pub mod tests;

pub use error::*;
pub use ply::{RobustPlyReader, RobustPlyWriter, PlyWriteOptions, PlyFormat, PlyValue};
pub use obj::{RobustObjReader, RobustObjWriter, ObjData, ObjWriteOptions, Material, FaceVertex, Face, Group};
pub use pcd::{RobustPcdReader, RobustPcdWriter, PcdWriteOptions, PcdDataFormat, PcdFieldType, PcdHeader, PcdValue};
pub use xyz_csv::{XyzCsvReader, XyzCsvWriter, XyzCsvStreamingReader, XyzCsvWriteOptions, XyzCsvSchema, XyzCsvPoint, Delimiter, ColumnType};
pub use registry::{IoRegistry, FormatHandler};
pub use mesh_attributes::{ExtendedTriangleMesh, MeshAttributeOptions, MeshMetadata, Tangent, UV};
pub use serialization::{SerializationOptions, AttributePreservingReader, AttributePreservingWriter};

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f};
use std::path::Path;

// Legacy traits for backward compatibility
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

// Global IO registry instance
lazy_static::lazy_static! {
    static ref IO_REGISTRY: IoRegistry = {
        let mut registry = IoRegistry::new();
        
        // Register PLY format handlers
        registry.register_point_cloud_handler("ply", Box::new(ply::PlyReader));
        registry.register_mesh_handler("ply", Box::new(ply::PlyReader));
        registry.register_point_cloud_writer("ply", Box::new(ply::PlyWriter));
        registry.register_mesh_writer("ply", Box::new(ply::PlyWriter));
        
        // Register OBJ format handlers
        registry.register_mesh_handler("obj", Box::new(obj::ObjReader));
        registry.register_mesh_writer("obj", Box::new(obj::ObjWriter));
        
        // Register pasture format handlers (when feature is enabled)
        #[cfg(feature = "las_laz")]
        {
            registry.register_point_cloud_handler("las", Box::new(pasture::PastureReader));
            registry.register_point_cloud_handler("laz", Box::new(pasture::PastureReader));
            registry.register_point_cloud_writer("las", Box::new(pasture::PastureWriter));
            registry.register_point_cloud_writer("laz", Box::new(pasture::PastureWriter));
        }
        registry.register_point_cloud_handler("pcd", Box::new(pcd::PcdReader));
        registry.register_point_cloud_writer("pcd", Box::new(pcd::PcdWriter));
        
        // Register XYZ/CSV format handlers
        registry.register_point_cloud_handler("xyz", Box::new(xyz_csv::XyzCsvReader));
        registry.register_point_cloud_handler("csv", Box::new(xyz_csv::XyzCsvReader));
        registry.register_point_cloud_handler("txt", Box::new(xyz_csv::XyzCsvReader));
        registry.register_point_cloud_writer("xyz", Box::new(xyz_csv::XyzCsvWriter));
        registry.register_point_cloud_writer("csv", Box::new(xyz_csv::XyzCsvWriter));
        registry.register_point_cloud_writer("txt", Box::new(xyz_csv::XyzCsvWriter));
        
        registry
    };
}

/// Auto-detect format and read point cloud using the unified registry
pub fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
    let path = path.as_ref();
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| threecrate_core::Error::UnsupportedFormat(
            "No file extension found".to_string()
        ))?;
    
    IO_REGISTRY.read_point_cloud(path, extension)
}

/// Auto-detect format and read mesh using the unified registry
pub fn read_mesh<P: AsRef<Path>>(path: P) -> Result<TriangleMesh> {
    let path = path.as_ref();
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| threecrate_core::Error::UnsupportedFormat(
            "No file extension found".to_string()
        ))?;
    
    IO_REGISTRY.read_mesh(path, extension)
}

/// Write point cloud with format auto-detection using the unified registry
pub fn write_point_cloud<P: AsRef<Path>>(cloud: &PointCloud<Point3f>, path: P) -> Result<()> {
    let path = path.as_ref();
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| threecrate_core::Error::UnsupportedFormat(
            "No file extension found".to_string()
        ))?;
    
    IO_REGISTRY.write_point_cloud(cloud, path, extension)
}

/// Write mesh with format auto-detection using the unified registry
pub fn write_mesh<P: AsRef<Path>>(mesh: &TriangleMesh, path: P) -> Result<()> {
    let path = path.as_ref();
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| threecrate_core::Error::UnsupportedFormat(
            "No file extension found".to_string()
        ))?;
    
    IO_REGISTRY.write_mesh(mesh, path, extension)
}

/// Get the global IO registry for advanced usage
pub fn get_io_registry() -> &'static IoRegistry {
    &IO_REGISTRY
}

/// Streaming point cloud reader for large files
/// 
/// This function returns an iterator that reads points one by one without loading
/// the entire file into memory. Useful for processing very large point cloud files.
/// 
/// # Arguments
/// * `path` - Path to the point cloud file
/// * `chunk_size` - Optional chunk size for internal buffering (default: 1000)
/// 
/// # Returns
/// An iterator over `Result<Point3f>` where each item is either a point or an error
/// 
/// # Example
/// ```rust
/// use threecrate_io::read_point_cloud_iter;
/// 
/// // Note: This will fail if the file doesn't exist, but demonstrates the API
/// match read_point_cloud_iter("large_cloud.ply", Some(5000)) {
///     Ok(iter) => {
///         for result in iter {
///             match result {
///                 Ok(point) => println!("Point: {:?}", point),
///                 Err(e) => eprintln!("Error: {}", e),
///             }
///         }
///     }
///     Err(e) => eprintln!("Failed to open file: {}", e),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn read_point_cloud_iter<P: AsRef<Path>>(
    path: P, 
    chunk_size: Option<usize>
) -> Result<Box<dyn Iterator<Item = Result<Point3f>> + Send + Sync>> {
    let path = path.as_ref();
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| threecrate_core::Error::UnsupportedFormat(
            "No file extension found".to_string()
        ))?;
    
    match extension {
        "ply" => {
            let iter = ply::PlyStreamingReader::new(path, chunk_size.unwrap_or(1000))?;
            Ok(Box::new(iter))
        }
        "obj" => {
            let iter = obj::ObjStreamingReader::new(path, chunk_size.unwrap_or(1000))?;
            Ok(Box::new(iter))
        }
        "xyz" | "csv" | "txt" => {
            let iter = xyz_csv::XyzCsvStreamingReader::new(path, chunk_size.unwrap_or(1000))?;
            Ok(Box::new(iter))
        }
        _ => Err(threecrate_core::Error::UnsupportedFormat(
            format!("Streaming not supported for format: {}", extension)
        ))
    }
}

/// Streaming mesh reader for large files
/// 
/// This function returns an iterator that reads mesh faces one by one without loading
/// the entire file into memory. Useful for processing very large mesh files.
/// 
/// # Arguments
/// * `path` - Path to the mesh file
/// * `chunk_size` - Optional chunk size for internal buffering (default: 1000)
/// 
/// # Returns
/// An iterator over `Result<[usize; 3]>` where each item is either a face or an error
/// 
/// # Example
/// ```rust
/// use threecrate_io::read_mesh_iter;
/// 
/// // Note: This will fail if the file doesn't exist, but demonstrates the API
/// match read_mesh_iter("large_mesh.obj", Some(5000)) {
///     Ok(iter) => {
///         for result in iter {
///             match result {
///                 Ok(face) => println!("Face: {:?}", face),
///                 Err(e) => eprintln!("Error: {}", e),
///             }
///         }
///     }
///     Err(e) => eprintln!("Failed to open file: {}", e),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn read_mesh_iter<P: AsRef<Path>>(
    path: P, 
    chunk_size: Option<usize>
) -> Result<Box<dyn Iterator<Item = Result<[usize; 3]>> + Send + Sync>> {
    let path = path.as_ref();
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| threecrate_core::Error::UnsupportedFormat(
            "No file extension found".to_string()
        ))?;
    
    match extension {
        "ply" => {
            let iter = ply::PlyMeshStreamingReader::new(path, chunk_size.unwrap_or(1000))?;
            Ok(Box::new(iter))
        }
        "obj" => {
            let iter = obj::ObjMeshStreamingReader::new(path, chunk_size.unwrap_or(1000))?;
            Ok(Box::new(iter))
        }
        _ => Err(threecrate_core::Error::UnsupportedFormat(
            format!("Streaming not supported for format: {}", extension)
        ))
    }
}

// Legacy tests moved to tests/ module

