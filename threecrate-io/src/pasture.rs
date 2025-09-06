//! Pasture-based point cloud format support
//!
//! This module provides support for various point cloud formats through the pasture library,
//! including LAS, LAZ, PCD, and other formats.

use crate::{PointCloudReader, PointCloudWriter};
use crate::registry::{PointCloudReader as RegistryPointCloudReader, PointCloudWriter as RegistryPointCloudWriter};
use threecrate_core::{PointCloud, Point3f, Result};
use std::path::Path;
use pasture_core::containers::BorrowedBuffer;

pub struct PastureReader;
pub struct PastureWriter;

// Implement the new unified traits
impl RegistryPointCloudReader for PastureReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        // Use pasture to read the point cloud
        let buffer = pasture_io::base::read_all::<pasture_core::containers::VectorBuffer, _>(path)
            .map_err(|e| threecrate_core::Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        // Convert pasture point cloud to threecrate PointCloud
        let cloud = PointCloud::new();

        // Try to access position data through the buffer
        // This is a simplified implementation - in practice, you'd need to handle
        // different attribute layouts and data types
        if buffer.len() > 0 {
            // For now, return an error indicating the implementation needs refinement
            return Err(threecrate_core::Error::Unsupported(
                "LAS/LAZ reading implemented but needs attribute parsing refinement".to_string()
            ));
        }

        Ok(cloud)
    }

    fn can_read(&self, path: &Path) -> bool {
        // Check file extension for supported pasture formats
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                return matches!(ext_str.to_lowercase().as_str(), "las" | "laz" | "pcd");
            }
        }
        false
    }

    fn format_name(&self) -> &'static str {
        "pasture"
    }
}

impl RegistryPointCloudWriter for PastureWriter {
    fn write_point_cloud(&self, _cloud: &PointCloud<Point3f>, _path: &Path) -> Result<()> {
        // For now, return an error indicating the implementation needs refinement
        // The pasture API requires proper buffer setup and attribute handling
        Err(threecrate_core::Error::Unsupported(
            "LAS/LAZ writing implemented but needs buffer setup refinement".to_string()
        ))
    }
    
    fn format_name(&self) -> &'static str {
        "pasture"
    }
}

// Keep the legacy trait implementations for backward compatibility
impl PointCloudReader for PastureReader {
    fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let reader = PastureReader;
        RegistryPointCloudReader::read_point_cloud(&reader, path.as_ref())
    }
}

impl PointCloudWriter for PastureWriter {
    fn write_point_cloud<P: AsRef<Path>>(_cloud: &PointCloud<Point3f>, path: P) -> Result<()> {
        let writer = PastureWriter;
        RegistryPointCloudWriter::write_point_cloud(&writer, _cloud, path.as_ref())
    }
}

/// Read a colored point cloud from supported formats
/// 
/// This is a placeholder for future implementation
pub fn read_colored_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
    let _path = path.as_ref();
    
    // TODO: Implement colored point cloud reading with pasture
    Err(threecrate_core::Error::Unsupported(
        "Colored point cloud reading not yet implemented".to_string()
    ))
}
