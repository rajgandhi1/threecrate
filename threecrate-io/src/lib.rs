//! I/O operations for point clouds and meshes
//! 
//! This crate provides functionality to read and write various 3D file formats
//! including PLY, OBJ, and other common point cloud and mesh formats.

pub mod ply;
pub mod obj;
pub mod pasture;
pub mod error;

pub use error::*;
pub use ply::{RobustPlyReader, RobustPlyWriter, PlyWriteOptions, PlyFormat, PlyValue};

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
    use std::io::Write;

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
    fn test_robust_ply_ascii_parsing() {
        let temp_file = "test_robust_ascii.ply";
        
        // Create a test ASCII PLY file manually
        let ply_content = r#"ply
format ascii 1.0
comment This is a test file
obj_info Created by threecrate test
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 0.0 0.0 0.0 1.0 255 0 0
1.0 0.0 0.0 0.0 0.0 1.0 0 255 0
1.0 1.0 0.0 0.0 0.0 1.0 0 0 255
0.0 1.0 0.0 0.0 0.0 1.0 255 255 255
3 0 1 2
3 0 2 3
"#;
        
        std::fs::write(temp_file, ply_content).unwrap();
        
        // Test robust reader
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        
        // Verify header
        assert_eq!(ply_data.header.format, ply::PlyFormat::Ascii);
        assert_eq!(ply_data.header.version, "1.0");
        assert_eq!(ply_data.header.comments.len(), 1);
        assert_eq!(ply_data.header.comments[0], "This is a test file");
        assert_eq!(ply_data.header.obj_info.len(), 1);
        assert_eq!(ply_data.header.obj_info[0], "Created by threecrate test");
        
        // Verify elements
        assert_eq!(ply_data.header.elements.len(), 2);
        assert_eq!(ply_data.header.elements[0].name, "vertex");
        assert_eq!(ply_data.header.elements[0].count, 4);
        assert_eq!(ply_data.header.elements[1].name, "face");
        assert_eq!(ply_data.header.elements[1].count, 2);
        
        // Verify vertex data
        let vertices = ply_data.elements.get("vertex").unwrap();
        assert_eq!(vertices.len(), 4);
        
        let first_vertex = &vertices[0];
        assert_eq!(first_vertex.get("x").unwrap().as_f32().unwrap(), 0.0);
        assert_eq!(first_vertex.get("y").unwrap().as_f32().unwrap(), 0.0);
        assert_eq!(first_vertex.get("z").unwrap().as_f32().unwrap(), 0.0);
        assert_eq!(first_vertex.get("nx").unwrap().as_f32().unwrap(), 0.0);
        assert_eq!(first_vertex.get("red").unwrap().as_f32().unwrap(), 255.0);
        
        // Verify face data
        let faces = ply_data.elements.get("face").unwrap();
        assert_eq!(faces.len(), 2);
        
        let first_face = &faces[0];
        let indices = first_face.get("vertex_indices").unwrap().as_usize_list().unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
        
        // Test mesh reading through the standard interface
        let mesh = ply::PlyReader::read_mesh(temp_file).unwrap();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.face_count(), 2);
        assert!(mesh.normals.is_some());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_binary_little_endian() {
        let temp_file = "test_binary_le.ply";
        
        // Create a simple binary PLY file
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        let mut file = std::fs::File::create(temp_file).unwrap();
        file.write_all(header.as_bytes()).unwrap();
        
        // Write binary vertex data (little endian)
        use byteorder::{LittleEndian, WriteBytesExt};
        file.write_f32::<LittleEndian>(1.0).unwrap(); // x
        file.write_f32::<LittleEndian>(2.0).unwrap(); // y
        file.write_f32::<LittleEndian>(3.0).unwrap(); // z
        file.write_f32::<LittleEndian>(4.0).unwrap(); // x
        file.write_f32::<LittleEndian>(5.0).unwrap(); // y
        file.write_f32::<LittleEndian>(6.0).unwrap(); // z
        file.flush().unwrap();
        drop(file);
        
        // Test reading
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        assert_eq!(ply_data.header.format, ply::PlyFormat::BinaryLittleEndian);
        
        let vertices = ply_data.elements.get("vertex").unwrap();
        assert_eq!(vertices.len(), 2);
        
        let first_vertex = &vertices[0];
        assert_eq!(first_vertex.get("x").unwrap().as_f32().unwrap(), 1.0);
        assert_eq!(first_vertex.get("y").unwrap().as_f32().unwrap(), 2.0);
        assert_eq!(first_vertex.get("z").unwrap().as_f32().unwrap(), 3.0);
        
        let second_vertex = &vertices[1];
        assert_eq!(second_vertex.get("x").unwrap().as_f32().unwrap(), 4.0);
        assert_eq!(second_vertex.get("y").unwrap().as_f32().unwrap(), 5.0);
        assert_eq!(second_vertex.get("z").unwrap().as_f32().unwrap(), 6.0);
        
        // Test point cloud reading
        let cloud = ply::PlyReader::read_point_cloud(temp_file).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud[0], Point3f::new(1.0, 2.0, 3.0));
        assert_eq!(cloud[1], Point3f::new(4.0, 5.0, 6.0));
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_binary_big_endian() {
        let temp_file = "test_binary_be.ply";
        
        // Create a simple binary PLY file
        let header = "ply\nformat binary_big_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        let mut file = std::fs::File::create(temp_file).unwrap();
        file.write_all(header.as_bytes()).unwrap();
        
        // Write binary vertex data (big endian)
        use byteorder::{BigEndian, WriteBytesExt};
        file.write_f32::<BigEndian>(10.0).unwrap(); // x
        file.write_f32::<BigEndian>(20.0).unwrap(); // y
        file.write_f32::<BigEndian>(30.0).unwrap(); // z
        file.flush().unwrap();
        drop(file);
        
        // Test reading
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        assert_eq!(ply_data.header.format, ply::PlyFormat::BinaryBigEndian);
        
        let vertices = ply_data.elements.get("vertex").unwrap();
        assert_eq!(vertices.len(), 1);
        
        let vertex = &vertices[0];
        assert_eq!(vertex.get("x").unwrap().as_f32().unwrap(), 10.0);
        assert_eq!(vertex.get("y").unwrap().as_f32().unwrap(), 20.0);
        assert_eq!(vertex.get("z").unwrap().as_f32().unwrap(), 30.0);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_error_handling() {
        let temp_file = "test_invalid.ply";
        
        // Test invalid magic number
        std::fs::write(temp_file, "not_ply\n").unwrap();
        let result = ply::RobustPlyReader::read_ply_file(temp_file);
        assert!(result.is_err());
        
        // Test missing format
        std::fs::write(temp_file, "ply\nelement vertex 1\nend_header\n").unwrap();
        let result = ply::RobustPlyReader::read_ply_file(temp_file);
        assert!(result.is_err());
        
        // Test invalid format
        std::fs::write(temp_file, "ply\nformat unknown_format 1.0\nend_header\n").unwrap();
        let result = ply::RobustPlyReader::read_ply_file(temp_file);
        assert!(result.is_err());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_list_properties() {
        let temp_file = "test_list_props.ply";
        
        // Create PLY with list properties
        let ply_content = r#"ply
format ascii 1.0
element face 1
property list uchar int vertex_indices
end_header
4 10 20 30 40
"#;
        
        std::fs::write(temp_file, ply_content).unwrap();
        
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        let faces = ply_data.elements.get("face").unwrap();
        let face = &faces[0];
        let indices = face.get("vertex_indices").unwrap().as_usize_list().unwrap();
        assert_eq!(indices, vec![10, 20, 30, 40]);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_writer_ascii() {
        let temp_file = "test_writer_ascii.ply";
        
        // Create test point cloud
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(1.0, 2.0, 3.0));
        cloud.push(Point3f::new(4.0, 5.0, 6.0));
        cloud.push(Point3f::new(7.0, 8.0, 9.0));
        
        // Write with ASCII format and custom options
        let options = ply::PlyWriteOptions::ascii()
            .with_comment("Test point cloud")
            .with_obj_info("Created by threecrate test");
        
        ply::RobustPlyWriter::write_point_cloud(&cloud, temp_file, &options).unwrap();
        
        // Read back and verify
        let loaded_cloud = ply::PlyReader::read_point_cloud(temp_file).unwrap();
        assert_eq!(cloud.len(), loaded_cloud.len());
        
        for (original, loaded) in cloud.iter().zip(loaded_cloud.iter()) {
            assert!((original.x - loaded.x).abs() < 1e-6);
            assert!((original.y - loaded.y).abs() < 1e-6);
            assert!((original.z - loaded.z).abs() < 1e-6);
        }
        
        // Verify metadata
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        assert_eq!(ply_data.header.format, ply::PlyFormat::Ascii);
        assert_eq!(ply_data.header.comments, vec!["Test point cloud"]);
        assert_eq!(ply_data.header.obj_info, vec!["Created by threecrate test"]);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_writer_binary_little_endian() {
        let temp_file = "test_writer_binary_le.ply";
        
        // Create test mesh
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
        
        // Write with binary little endian format
        let options = ply::PlyWriteOptions::binary_little_endian()
            .with_normals(true);
        
        ply::RobustPlyWriter::write_mesh(&mesh, temp_file, &options).unwrap();
        
        // Read back and verify
        let loaded_mesh = ply::PlyReader::read_mesh(temp_file).unwrap();
        assert_eq!(mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(mesh.face_count(), loaded_mesh.face_count());
        assert!(loaded_mesh.normals.is_some());
        
        // Verify format
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        assert_eq!(ply_data.header.format, ply::PlyFormat::BinaryLittleEndian);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_writer_binary_big_endian() {
        let temp_file = "test_writer_binary_be.ply";
        
        // Create test point cloud
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(10.0, 20.0, 30.0));
        cloud.push(Point3f::new(40.0, 50.0, 60.0));
        
        // Write with binary big endian format
        let options = ply::PlyWriteOptions::binary_big_endian()
            .with_comment("Binary big endian test");
        
        ply::RobustPlyWriter::write_point_cloud(&cloud, temp_file, &options).unwrap();
        
        // Read back and verify
        let loaded_cloud = ply::PlyReader::read_point_cloud(temp_file).unwrap();
        assert_eq!(cloud.len(), loaded_cloud.len());
        
        // Verify format
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        assert_eq!(ply_data.header.format, ply::PlyFormat::BinaryBigEndian);
        assert_eq!(ply_data.header.comments, vec!["Binary big endian test"]);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_writer_custom_properties() {
        let temp_file = "test_writer_custom_props.ply";
        
        // Create test point cloud
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(1.0, 2.0, 3.0));
        cloud.push(Point3f::new(4.0, 5.0, 6.0));
        
        // Create custom properties
        let colors = vec![
            ply::PlyValue::UChar(255),
            ply::PlyValue::UChar(128),
        ];
        let intensities = vec![
            ply::PlyValue::Float(0.8),
            ply::PlyValue::Float(0.5),
        ];
        
        // Write with custom properties
        let options = ply::PlyWriteOptions::ascii()
            .with_custom_vertex_property("red", colors)
            .with_custom_vertex_property("intensity", intensities)
            .with_vertex_property_order(vec!["x".to_string(), "y".to_string(), "z".to_string(), "red".to_string(), "intensity".to_string()]);
        
        ply::RobustPlyWriter::write_point_cloud(&cloud, temp_file, &options).unwrap();
        
        // Read back and verify structure
        let ply_data = ply::RobustPlyReader::read_ply_file(temp_file).unwrap();
        let vertex_element = &ply_data.header.elements[0];
        assert_eq!(vertex_element.properties.len(), 5); // x, y, z, red, intensity
        
        // Verify custom properties exist
        let vertices = ply_data.elements.get("vertex").unwrap();
        let first_vertex = &vertices[0];
        assert!(first_vertex.contains_key("red"));
        assert!(first_vertex.contains_key("intensity"));
        
        // Verify values
        match first_vertex.get("red").unwrap() {
            ply::PlyValue::UChar(255) => {},
            _ => panic!("Expected UChar(255) for red property"),
        }
        match first_vertex.get("intensity").unwrap() {
            ply::PlyValue::Float(val) if (*val - 0.8).abs() < 1e-6 => {},
            _ => panic!("Expected Float(0.8) for intensity property"),
        }
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_ply_writer_round_trip_all_formats() {
        let formats = vec![
            ply::PlyFormat::Ascii,
            ply::PlyFormat::BinaryLittleEndian,
            ply::PlyFormat::BinaryBigEndian,
        ];
        
        for (i, format) in formats.iter().enumerate() {
            let temp_file = format!("test_roundtrip_{}.ply", i);
            
            // Create test mesh with normals
            let vertices = vec![
                Point3f::new(-1.0, -1.0, 0.0),
                Point3f::new(1.0, -1.0, 0.0),
                Point3f::new(0.0, 1.0, 0.0),
            ];
            let faces = vec![[0, 1, 2]];
            let normals = vec![
                Vector3f::new(0.0, 0.0, 1.0),
                Vector3f::new(0.0, 0.0, 1.0),
                Vector3f::new(0.0, 0.0, 1.0),
            ];
            
            let mut original_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
            original_mesh.set_normals(normals);
            
            // Write mesh
            let options = ply::PlyWriteOptions {
                format: *format,
                include_normals: true,
                comments: vec!["Round-trip test".to_string()],
                ..Default::default()
            };
            
            ply::RobustPlyWriter::write_mesh(&original_mesh, &temp_file, &options).unwrap();
            
            // Read back and verify
            let loaded_mesh = ply::PlyReader::read_mesh(&temp_file).unwrap();
            
            // Verify structure
            assert_eq!(original_mesh.vertex_count(), loaded_mesh.vertex_count());
            assert_eq!(original_mesh.face_count(), loaded_mesh.face_count());
            assert!(loaded_mesh.normals.is_some());
            
            // Verify vertices
            for (orig, loaded) in original_mesh.vertices.iter().zip(loaded_mesh.vertices.iter()) {
                assert!((orig.x - loaded.x).abs() < 1e-5, "Vertex X mismatch for format {:?}", format);
                assert!((orig.y - loaded.y).abs() < 1e-5, "Vertex Y mismatch for format {:?}", format);
                assert!((orig.z - loaded.z).abs() < 1e-5, "Vertex Z mismatch for format {:?}", format);
            }
            
            // Verify faces
            for (orig, loaded) in original_mesh.faces.iter().zip(loaded_mesh.faces.iter()) {
                assert_eq!(orig, loaded, "Face mismatch for format {:?}", format);
            }
            
            // Verify normals
            if let (Some(orig_normals), Some(loaded_normals)) = (&original_mesh.normals, &loaded_mesh.normals) {
                for (orig, loaded) in orig_normals.iter().zip(loaded_normals.iter()) {
                    assert!((orig.x - loaded.x).abs() < 1e-5, "Normal X mismatch for format {:?}", format);
                    assert!((orig.y - loaded.y).abs() < 1e-5, "Normal Y mismatch for format {:?}", format);
                    assert!((orig.z - loaded.z).abs() < 1e-5, "Normal Z mismatch for format {:?}", format);
                }
            }
            
            // Verify format was preserved
            let ply_data = ply::RobustPlyReader::read_ply_file(&temp_file).unwrap();
            assert_eq!(ply_data.header.format, *format);
            
            // Cleanup
            let _ = fs::remove_file(temp_file);
        }
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