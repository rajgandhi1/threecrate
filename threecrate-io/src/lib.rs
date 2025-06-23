//! I/O operations for point clouds and meshes
//! 
//! This crate provides functionality to read and write various 3D file formats
//! including PLY, OBJ, and other common point cloud and mesh formats.

pub mod ply;
pub mod obj;
pub mod pasture;
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

/// Auto-detect format and read point cloud
pub fn read_point_cloud<P: AsRef<std::path::Path>>(path: P) -> Result<PointCloud<Point3f>> {
    let path = path.as_ref();
    match path.extension().and_then(|s| s.to_str()) {
        Some("ply") => ply::PlyReader::read_point_cloud(path),
        Some("las") | Some("laz") | Some("pcd") => {
            // For now, return an error since pasture is not fully implemented
            Err(threecrate_core::Error::Unsupported(
                format!("Point cloud format {:?} not yet supported (pasture integration incomplete)", 
                        path.extension())
            ))
        }
        _ => Err(threecrate_core::Error::UnsupportedFormat(
            format!("Unsupported point cloud format: {:?}", path.extension())
        )),
    }
}

/// Auto-detect format and read mesh
pub fn read_mesh<P: AsRef<std::path::Path>>(path: P) -> Result<TriangleMesh> {
    let path = path.as_ref();
    match path.extension().and_then(|s| s.to_str()) {
        Some("obj") => obj::ObjReader::read_mesh(path),
        Some("ply") => ply::PlyReader::read_mesh(path),
        _ => Err(threecrate_core::Error::UnsupportedFormat(
            format!("Unsupported mesh format: {:?}", path.extension())
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{Point3f, Vector3f};
    use std::fs;

    #[test]
    fn test_ply_point_cloud_roundtrip() {
        let temp_file = "test_cloud.ply";
        
        // Create test point cloud
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        
        // Write and read back
        ply::PlyWriter::write_point_cloud(&cloud, temp_file).unwrap();
        let loaded_cloud = ply::PlyReader::read_point_cloud(temp_file).unwrap();
        
        // Verify
        assert_eq!(cloud.len(), loaded_cloud.len());
        for (original, loaded) in cloud.iter().zip(loaded_cloud.iter()) {
            assert!((original.x - loaded.x).abs() < 1e-6);
            assert!((original.y - loaded.y).abs() < 1e-6);
            assert!((original.z - loaded.z).abs() < 1e-6);
        }
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_obj_mesh_roundtrip() {
        let temp_file = "test_mesh.obj";
        
        // Create test mesh
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        
        // Write and read back
        obj::ObjWriter::write_mesh(&mesh, temp_file).unwrap();
        let loaded_mesh = obj::ObjReader::read_mesh(temp_file).unwrap();
        
        // Verify
        assert_eq!(mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(mesh.face_count(), loaded_mesh.face_count());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_ply_mesh_with_normals() {
        let temp_file = "test_mesh_normals.ply";
        
        // Create test mesh with normals
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let normals = vec![
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 0.0, 1.0),
        ];
        
        let mut mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        mesh.set_normals(normals);
        
        // Write and read back
        ply::PlyWriter::write_mesh(&mesh, temp_file).unwrap();
        let loaded_mesh = ply::PlyReader::read_mesh(temp_file).unwrap();
        
        // Verify
        assert_eq!(mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(mesh.face_count(), loaded_mesh.face_count());
        assert!(loaded_mesh.normals.is_some());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_auto_detect_functions() {
        // Test PLY auto-detection
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        ply::PlyWriter::write_point_cloud(&cloud, "test_auto.ply").unwrap();
        
        let loaded = read_point_cloud("test_auto.ply").unwrap();
        assert_eq!(cloud.len(), loaded.len());
        
        // Test OBJ auto-detection
        let mesh = TriangleMesh::from_vertices_and_faces(
            vec![Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 0.0, 0.0), Point3f::new(0.5, 1.0, 0.0)],
            vec![[0, 1, 2]]
        );
        obj::ObjWriter::write_mesh(&mesh, "test_auto.obj").unwrap();
        
        let loaded = read_mesh("test_auto.obj").unwrap();
        assert_eq!(mesh.vertex_count(), loaded.vertex_count());
        
        // Cleanup
        let _ = fs::remove_file("test_auto.ply");
        let _ = fs::remove_file("test_auto.obj");
    }

    #[test]
    fn test_unsupported_format() {
        // Test unsupported point cloud format
        let result = read_point_cloud("test.xyz");
        assert!(result.is_err());
        
        // Test unsupported mesh format
        let result = read_mesh("test.stl");
        assert!(result.is_err());
    }
} 