//! E57 point cloud format support (ASTM E2807)
//!
//! Provides read/write support for the E57 format, the ISO standard for 3D imaging
//! data used by terrestrial laser scanners (Faro, Leica, Trimble).
//!
//! # Feature flag
//! Enable with `features = ["e57"]` in your `Cargo.toml`.

use crate::registry::{
    MeshReader as RegistryMeshReader, MeshWriter as RegistryMeshWriter,
    PointCloudReader as RegistryPointCloudReader, PointCloudWriter as RegistryPointCloudWriter,
};
use std::path::Path;
use threecrate_core::{Error, Point3f, PointCloud, Result, TriangleMesh};

/// Default GUID embedded in the E57 file header when none is supplied.
const DEFAULT_FILE_GUID: &str = "{3F2504E0-4F89-11D3-9A0C-0305E82C3301}";
/// Default GUID used for the point cloud scan record when none is supplied.
const DEFAULT_CLOUD_GUID: &str = "{3F2504E0-4F89-11D3-9A0C-0305E82C3302}";

/// Write options for the E57 format.
#[derive(Debug, Clone)]
pub struct E57WriteOptions {
    /// GUID embedded in the E57 file header.
    pub file_guid: String,
    /// GUID for the individual point cloud scan record.
    pub cloud_guid: String,
}

impl Default for E57WriteOptions {
    fn default() -> Self {
        Self {
            file_guid: DEFAULT_FILE_GUID.to_string(),
            cloud_guid: DEFAULT_CLOUD_GUID.to_string(),
        }
    }
}

impl E57WriteOptions {
    pub fn with_file_guid(mut self, guid: impl Into<String>) -> Self {
        self.file_guid = guid.into();
        self
    }

    pub fn with_cloud_guid(mut self, guid: impl Into<String>) -> Self {
        self.cloud_guid = guid.into();
        self
    }
}

// ── Reader ────────────────────────────────────────────────────────────────────

/// E57 file reader (ASTM E2807).
///
/// All scan data in the file is merged into a single `PointCloud<Point3f>`.
/// Only Cartesian-valid points are included; spherical-only scans are skipped.
pub struct RobustE57Reader;

impl RobustE57Reader {
    /// Read all point clouds from an E57 file and merge them into one cloud.
    pub fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let path = path.as_ref();
        let mut reader = e57::E57Reader::from_file(path)
            .map_err(|e| Error::InvalidData(format!("Failed to open E57 file: {e}")))?;

        let point_clouds = reader.pointclouds();
        let mut cloud = PointCloud::new();

        for pc in &point_clouds {
            let simple_reader = reader
                .pointcloud_simple(pc)
                .map_err(|e| Error::InvalidData(format!("Failed to access E57 scan: {e}")))?;

            for result in simple_reader {
                let point = result
                    .map_err(|e| Error::InvalidData(format!("Failed to read E57 point: {e}")))?;

                if let e57::CartesianCoordinate::Valid { x, y, z } = point.cartesian {
                    cloud.push(Point3f::new(x as f32, y as f32, z as f32));
                }
            }
        }

        Ok(cloud)
    }
}

// ── Writer ────────────────────────────────────────────────────────────────────

/// E57 file writer (ASTM E2807).
pub struct RobustE57Writer;

impl RobustE57Writer {
    /// Write a `PointCloud<Point3f>` to an E57 file.
    pub fn write_point_cloud<P: AsRef<Path>>(
        cloud: &PointCloud<Point3f>,
        path: P,
        options: &E57WriteOptions,
    ) -> Result<()> {
        let path = path.as_ref();

        let mut writer = e57::E57Writer::from_file(path, &options.file_guid)
            .map_err(|e| Error::InvalidData(format!("Failed to create E57 file: {e}")))?;

        let prototype = vec![
            e57::Record::CARTESIAN_X_F64,
            e57::Record::CARTESIAN_Y_F64,
            e57::Record::CARTESIAN_Z_F64,
        ];

        let mut pc_writer = writer
            .add_pointcloud(&options.cloud_guid, prototype)
            .map_err(|e| Error::InvalidData(format!("Failed to add E57 point cloud: {e}")))?;

        for point in cloud.iter() {
            pc_writer
                .add_point(vec![
                    e57::RecordValue::Double(point.x as f64),
                    e57::RecordValue::Double(point.y as f64),
                    e57::RecordValue::Double(point.z as f64),
                ])
                .map_err(|e| Error::InvalidData(format!("Failed to write E57 point: {e}")))?;
        }

        pc_writer
            .finalize()
            .map_err(|e| Error::InvalidData(format!("Failed to finalize E57 scan: {e}")))?;

        writer
            .finalize()
            .map_err(|e| Error::InvalidData(format!("Failed to finalize E57 file: {e}")))?;

        Ok(())
    }
}

// ── Registry adapters ─────────────────────────────────────────────────────────

pub(crate) struct E57Reader;
pub(crate) struct E57Writer;

impl RegistryPointCloudReader for E57Reader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        RobustE57Reader::read_point_cloud(path)
    }

    fn can_read(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("e57"))
            .unwrap_or(false)
    }

    fn format_name(&self) -> &'static str {
        "e57"
    }
}

impl RegistryPointCloudWriter for E57Writer {
    fn write_point_cloud(&self, cloud: &PointCloud<Point3f>, path: &Path) -> Result<()> {
        RobustE57Writer::write_point_cloud(cloud, path, &E57WriteOptions::default())
    }

    fn format_name(&self) -> &'static str {
        "e57"
    }
}

/// E57 does not store triangle meshes. Reading returns a vertex-only mesh.
impl RegistryMeshReader for E57Reader {
    fn read_mesh(&self, path: &Path) -> Result<TriangleMesh> {
        let cloud = RobustE57Reader::read_point_cloud(path)?;
        let vertices: Vec<Point3f> = cloud.iter().cloned().collect();
        Ok(TriangleMesh::from_vertices_and_faces(vertices, vec![]))
    }

    fn can_read(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("e57"))
            .unwrap_or(false)
    }

    fn format_name(&self) -> &'static str {
        "e57"
    }
}

/// E57 does not store triangle meshes. Only mesh vertices are written.
impl RegistryMeshWriter for E57Writer {
    fn write_mesh(&self, mesh: &TriangleMesh, path: &Path) -> Result<()> {
        let mut cloud = PointCloud::new();
        for vertex in mesh.vertices.iter() {
            cloud.push(*vertex);
        }
        RobustE57Writer::write_point_cloud(&cloud, path, &E57WriteOptions::default())
    }

    fn format_name(&self) -> &'static str {
        "e57"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_cloud() -> PointCloud<Point3f> {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(1.0, 2.0, 3.0));
        cloud.push(Point3f::new(4.0, 5.0, 6.0));
        cloud.push(Point3f::new(7.0, 8.0, 9.0));
        cloud
    }

    #[test]
    fn test_write_read_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("threecrate_e57_test.e57");

        let original = make_test_cloud();
        RobustE57Writer::write_point_cloud(&original, &path, &E57WriteOptions::default())
            .expect("write should succeed");

        let loaded = RobustE57Reader::read_point_cloud(&path).expect("read should succeed");

        assert_eq!(loaded.len(), original.len());
        for (a, b) in original.iter().zip(loaded.iter()) {
            assert!((a.x - b.x).abs() < 1e-5);
            assert!((a.y - b.y).abs() < 1e-5);
            assert!((a.z - b.z).abs() < 1e-5);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_options_builder() {
        let opts = E57WriteOptions::default()
            .with_file_guid("{AABBCCDD-0000-0000-0000-000000000001}")
            .with_cloud_guid("{AABBCCDD-0000-0000-0000-000000000002}");
        assert_eq!(opts.file_guid, "{AABBCCDD-0000-0000-0000-000000000001}");
        assert_eq!(opts.cloud_guid, "{AABBCCDD-0000-0000-0000-000000000002}");
    }

    #[test]
    fn test_can_read_detects_extension() {
        let reader = E57Reader;
        assert!(RegistryPointCloudReader::can_read(&reader, Path::new("scan.e57")));
        assert!(RegistryPointCloudReader::can_read(&reader, Path::new("scan.E57")));
        assert!(!RegistryPointCloudReader::can_read(&reader, Path::new("scan.ply")));
        assert!(!RegistryPointCloudReader::can_read(&reader, Path::new("scan")));
    }
}
