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
pub use obj::{RobustObjReader, RobustObjWriter, ObjData, ObjWriteOptions, Material, FaceVertex, Face, Group};

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
    use std::collections::HashMap;

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
    fn test_robust_obj_reader_basic() {
        let temp_file = "test_basic.obj";
        
        // Create a basic OBJ file
        let obj_content = r#"# Basic OBJ test
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f 1 2 3
"#;
        
        std::fs::write(temp_file, obj_content).unwrap();
        
        // Test reading
        let obj_data = obj::RobustObjReader::read_obj_file(temp_file).unwrap();
        assert_eq!(obj_data.vertices.len(), 3);
        assert_eq!(obj_data.groups.len(), 1);
        assert_eq!(obj_data.groups[0].faces.len(), 1);
        
        // Test mesh conversion
        let mesh = obj::RobustObjReader::obj_data_to_mesh(&obj_data).unwrap();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_obj_reader_with_normals_and_textures() {
        let temp_file = "test_with_normals.obj";
        
        // Create OBJ file with normals and texture coordinates
        let obj_content = r#"# OBJ with normals and textures
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/2 3/3/3
"#;
        
        std::fs::write(temp_file, obj_content).unwrap();
        
        // Test reading
        let obj_data = obj::RobustObjReader::read_obj_file(temp_file).unwrap();
        assert_eq!(obj_data.vertices.len(), 3);
        assert_eq!(obj_data.texture_coords.len(), 3);
        assert_eq!(obj_data.normals.len(), 3);
        assert_eq!(obj_data.groups[0].faces.len(), 1);
        
        // Verify face vertex data
        let face = &obj_data.groups[0].faces[0];
        assert_eq!(face.vertices.len(), 3);
        assert_eq!(face.vertices[0].vertex, 0);
        assert_eq!(face.vertices[0].texture, Some(0));
        assert_eq!(face.vertices[0].normal, Some(0));
        
        // Test mesh conversion with normals
        let mesh = obj::RobustObjReader::obj_data_to_mesh(&obj_data).unwrap();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
        assert!(mesh.normals.is_some());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_obj_reader_with_groups() {
        let temp_file = "test_with_groups.obj";
        
        // Create OBJ file with multiple groups
        let obj_content = r#"# OBJ with groups
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
v 0.0 0.0 1.0

g group1
f 1 2 3

g group2
f 1 3 4
"#;
        
        std::fs::write(temp_file, obj_content).unwrap();
        
        // Test reading
        let obj_data = obj::RobustObjReader::read_obj_file(temp_file).unwrap();
        assert_eq!(obj_data.vertices.len(), 4);
        assert_eq!(obj_data.groups.len(), 2);
        assert_eq!(obj_data.groups[0].name, "group1");
        assert_eq!(obj_data.groups[1].name, "group2");
        assert_eq!(obj_data.groups[0].faces.len(), 1);
        assert_eq!(obj_data.groups[1].faces.len(), 1);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_obj_reader_with_materials() {
        let obj_file = "test_with_materials.obj";
        let mtl_file = "test_materials.mtl";
        
        // Create MTL file
        let mtl_content = r#"# Test materials
newmtl red_material
Ka 0.2 0.0 0.0
Kd 1.0 0.0 0.0
Ks 0.5 0.5 0.5
Ns 32.0
d 1.0
illum 2
map_Kd red_texture.jpg

newmtl blue_material
Ka 0.0 0.0 0.2
Kd 0.0 0.0 1.0
Ks 0.5 0.5 0.5
Ns 16.0
d 0.8
"#;
        
        std::fs::write(mtl_file, mtl_content).unwrap();
        
        // Create OBJ file with material references
        let obj_content = r#"# OBJ with materials
mtllib test_materials.mtl
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
v 0.0 0.0 1.0

usemtl red_material
f 1 2 3

usemtl blue_material
f 1 3 4
"#;
        
        std::fs::write(obj_file, obj_content).unwrap();
        
        // Test reading
        let obj_data = obj::RobustObjReader::read_obj_file(obj_file).unwrap();
        assert_eq!(obj_data.vertices.len(), 4);
        assert_eq!(obj_data.mtl_files.len(), 1);
        assert_eq!(obj_data.materials.len(), 2);
        
        // Verify materials
        let red_material = obj_data.materials.get("red_material").unwrap();
        assert_eq!(red_material.diffuse, Some([1.0, 0.0, 0.0]));
        assert_eq!(red_material.shininess, Some(32.0));
        assert_eq!(red_material.diffuse_map, Some("red_texture.jpg".to_string()));
        
        let blue_material = obj_data.materials.get("blue_material").unwrap();
        assert_eq!(blue_material.diffuse, Some([0.0, 0.0, 1.0]));
        assert_eq!(blue_material.transparency, Some(0.8));
        
        // Verify face materials
        assert_eq!(obj_data.groups[0].faces[0].material, Some("red_material".to_string()));
        assert_eq!(obj_data.groups[0].faces[1].material, Some("blue_material".to_string()));
        
        // Cleanup
        let _ = fs::remove_file(obj_file);
        let _ = fs::remove_file(mtl_file);
    }

    #[test]
    fn test_robust_obj_reader_polygon_triangulation() {
        let temp_file = "test_polygons.obj";
        
        // Create OBJ file with quads and n-gons
        let obj_content = r#"# OBJ with polygons
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 2.0 0.0 0.0
v 2.0 1.0 0.0

# Quad (should become 2 triangles)
f 1 2 3 4

# Pentagon (should become 3 triangles)
f 1 2 5 6 4
"#;
        
        std::fs::write(temp_file, obj_content).unwrap();
        
        // Test reading
        let obj_data = obj::RobustObjReader::read_obj_file(temp_file).unwrap();
        assert_eq!(obj_data.vertices.len(), 6);
        
        // Quad should become 2 triangles, pentagon should become 3 triangles
        assert_eq!(obj_data.groups[0].faces.len(), 5);
        
        // All faces should be triangles
        for face in &obj_data.groups[0].faces {
            assert_eq!(face.vertices.len(), 3);
        }
        
        // Test mesh conversion
        let mesh = obj::RobustObjReader::obj_data_to_mesh(&obj_data).unwrap();
        assert_eq!(mesh.vertex_count(), 6);
        assert_eq!(mesh.face_count(), 5);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_obj_reader_error_handling() {
        // Test invalid vertex
        let invalid_vertex = "test_invalid_vertex.obj";
        std::fs::write(invalid_vertex, "v 1.0 invalid 3.0\n").unwrap();
        let result = obj::RobustObjReader::read_obj_file(invalid_vertex);
        assert!(result.is_err());
        let _ = fs::remove_file(invalid_vertex);
        
        // Test invalid face
        let invalid_face = "test_invalid_face.obj";
        std::fs::write(invalid_face, "v 0.0 0.0 0.0\nf 1 invalid 3\n").unwrap();
        let result = obj::RobustObjReader::read_obj_file(invalid_face);
        assert!(result.is_err());
        let _ = fs::remove_file(invalid_face);
        
        // Test out of range vertex index
        let out_of_range = "test_out_of_range.obj";
        std::fs::write(out_of_range, "v 0.0 0.0 0.0\nf 1 2 3\n").unwrap();
        let obj_data = obj::RobustObjReader::read_obj_file(out_of_range).unwrap();
        let result = obj::RobustObjReader::obj_data_to_mesh(&obj_data);
        assert!(result.is_err());
        let _ = fs::remove_file(out_of_range);
    }

    #[test]
    fn test_mtl_reader_standalone() {
        let mtl_file = "test_standalone.mtl";
        
        // Create comprehensive MTL file
        let mtl_content = r#"# Comprehensive MTL test
newmtl material1
Ka 0.1 0.2 0.3
Kd 0.4 0.5 0.6
Ks 0.7 0.8 0.9
Ns 96.0
d 0.9
Tr 0.1
illum 2
map_Kd diffuse.png
map_Bump normal.png
map_Ks specular.png

newmtl material2
Kd 1.0 0.0 0.0
Ns 32
d 1.0
illum 1
"#;
        
        std::fs::write(mtl_file, mtl_content).unwrap();
        
        // Test reading
        let materials = obj::RobustObjReader::read_mtl_file(mtl_file).unwrap();
        assert_eq!(materials.len(), 2);
        
        // Test material1
        let mat1 = materials.get("material1").unwrap();
        assert_eq!(mat1.ambient, Some([0.1, 0.2, 0.3]));
        assert_eq!(mat1.diffuse, Some([0.4, 0.5, 0.6]));
        assert_eq!(mat1.specular, Some([0.7, 0.8, 0.9]));
        assert_eq!(mat1.shininess, Some(96.0));
        assert_eq!(mat1.transparency, Some(0.9));
        assert_eq!(mat1.illumination, Some(2));
        assert_eq!(mat1.diffuse_map, Some("diffuse.png".to_string()));
        assert_eq!(mat1.normal_map, Some("normal.png".to_string()));
        assert_eq!(mat1.specular_map, Some("specular.png".to_string()));
        
        // Test material2
        let mat2 = materials.get("material2").unwrap();
        assert_eq!(mat2.diffuse, Some([1.0, 0.0, 0.0]));
        assert_eq!(mat2.shininess, Some(32.0));
        assert_eq!(mat2.transparency, Some(1.0));
        assert_eq!(mat2.illumination, Some(1));
        
        // Cleanup
        let _ = fs::remove_file(mtl_file);
    }

    #[test]
    fn test_robust_obj_writer_basic() {
        let temp_file = "test_writer_basic.obj";
        
        // Create test mesh
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        
        // Write with basic options
        let options = obj::ObjWriteOptions::new()
            .with_comment("Test mesh")
            .with_normals(false);
        
        obj::RobustObjWriter::write_mesh(&mesh, temp_file, &options).unwrap();
        
        // Read back and verify
        let loaded_mesh = obj::ObjReader::read_mesh(temp_file).unwrap();
        assert_eq!(mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(mesh.face_count(), loaded_mesh.face_count());
        
        // Verify file contents
        let content = std::fs::read_to_string(temp_file).unwrap();
        assert!(content.contains("# Test mesh"));
        assert!(content.contains("v 0 0 0"));
        assert!(content.contains("v 1 0 0"));
        assert!(content.contains("v 0.5 1 0"));
        assert!(content.contains("f 1 2 3"));
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_obj_writer_with_normals() {
        let temp_file = "test_writer_normals.obj";
        
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
        
        // Write with normals enabled
        let options = obj::ObjWriteOptions::new()
            .with_normals(true)
            .with_group_name("test_group");
        
        obj::RobustObjWriter::write_mesh(&mesh, temp_file, &options).unwrap();
        
        // Read back and verify
        let loaded_mesh = obj::ObjReader::read_mesh(temp_file).unwrap();
        assert_eq!(mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(mesh.face_count(), loaded_mesh.face_count());
        assert!(loaded_mesh.normals.is_some());
        
        // Verify file contents
        let content = std::fs::read_to_string(temp_file).unwrap();
        assert!(content.contains("vn 0 0 1"));
        assert!(content.contains("g test_group"));
        assert!(content.contains("f 1//1 2//2 3//3"));
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_robust_obj_writer_with_materials() {
        let obj_file = "test_writer_materials.obj";
        let mtl_file = "test_writer_materials.mtl";
        
        // Create test mesh
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        
        // Write with materials
        let options = obj::ObjWriteOptions::new()
            .with_materials(true)
            .with_material_name("test_material")
            .with_object_name("test_object");
        
        obj::RobustObjWriter::write_mesh(&mesh, obj_file, &options).unwrap();
        
        // Verify OBJ file
        let obj_content = std::fs::read_to_string(obj_file).unwrap();
        assert!(obj_content.contains("mtllib test_writer_materials.mtl"));
        assert!(obj_content.contains("o test_object"));
        assert!(obj_content.contains("usemtl test_material"));
        
        // Verify MTL file was created
        assert!(std::path::Path::new(mtl_file).exists());
        let mtl_content = std::fs::read_to_string(mtl_file).unwrap();
        assert!(mtl_content.contains("newmtl test_material"));
        
        // Test round-trip
        let loaded_obj_data = obj::RobustObjReader::read_obj_file(obj_file).unwrap();
        assert_eq!(loaded_obj_data.materials.len(), 1);
        assert!(loaded_obj_data.materials.contains_key("test_material"));
        
        // Cleanup
        let _ = fs::remove_file(obj_file);
        let _ = fs::remove_file(mtl_file);
    }

    #[test]
    fn test_robust_obj_writer_obj_data_round_trip() {
        let obj_file = "test_round_trip.obj";
        let mtl_file = "test_round_trip.mtl";
        
        // Create complex ObjData
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0),
        ];
        
        let texture_coords = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 1.0],
        ];
        
        let normals = vec![
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 1.0, 0.0),
        ];
        
        // Create materials
        let mut materials = HashMap::new();
        let mut red_material = obj::Material::new("red".to_string());
        red_material.diffuse = Some([1.0, 0.0, 0.0]);
        red_material.shininess = Some(32.0);
        materials.insert("red".to_string(), red_material);
        
        let mut blue_material = obj::Material::new("blue".to_string());
        blue_material.diffuse = Some([0.0, 0.0, 1.0]);
        blue_material.transparency = Some(0.8);
        materials.insert("blue".to_string(), blue_material);
        
        // Create faces with different materials
        let face1 = obj::Face {
            vertices: vec![
                obj::FaceVertex { vertex: 0, texture: Some(0), normal: Some(0) },
                obj::FaceVertex { vertex: 1, texture: Some(1), normal: Some(1) },
                obj::FaceVertex { vertex: 2, texture: Some(2), normal: Some(2) },
            ],
            material: Some("red".to_string()),
        };
        
        let face2 = obj::Face {
            vertices: vec![
                obj::FaceVertex { vertex: 0, texture: Some(0), normal: Some(0) },
                obj::FaceVertex { vertex: 2, texture: Some(2), normal: Some(2) },
                obj::FaceVertex { vertex: 3, texture: Some(3), normal: Some(3) },
            ],
            material: Some("blue".to_string()),
        };
        
        let group = obj::Group {
            name: "test_group".to_string(),
            faces: vec![face1, face2],
        };
        
        let original_obj_data = obj::ObjData {
            vertices,
            texture_coords,
            normals,
            groups: vec![group],
            materials,
            mtl_files: vec!["test_round_trip.mtl".to_string()],
        };
        
        // Write OBJ data
        let options = obj::ObjWriteOptions::new()
            .with_normals(true)
            .with_texcoords(true)
            .with_materials(true);
        
        obj::RobustObjWriter::write_obj_file(&original_obj_data, obj_file, &options).unwrap();
        
        // Read back
        let loaded_obj_data = obj::RobustObjReader::read_obj_file(obj_file).unwrap();
        
        // Verify structure
        assert_eq!(original_obj_data.vertices.len(), loaded_obj_data.vertices.len());
        assert_eq!(original_obj_data.texture_coords.len(), loaded_obj_data.texture_coords.len());
        assert_eq!(original_obj_data.normals.len(), loaded_obj_data.normals.len());
        assert_eq!(original_obj_data.groups.len(), loaded_obj_data.groups.len());
        assert_eq!(original_obj_data.materials.len(), loaded_obj_data.materials.len());
        
        // Verify vertices
        for (orig, loaded) in original_obj_data.vertices.iter().zip(loaded_obj_data.vertices.iter()) {
            assert!((orig.x - loaded.x).abs() < 1e-6);
            assert!((orig.y - loaded.y).abs() < 1e-6);
            assert!((orig.z - loaded.z).abs() < 1e-6);
        }
        
        // Verify materials
        for (name, orig_material) in &original_obj_data.materials {
            let loaded_material = loaded_obj_data.materials.get(name).unwrap();
            assert_eq!(orig_material.diffuse, loaded_material.diffuse);
            assert_eq!(orig_material.shininess, loaded_material.shininess);
            assert_eq!(orig_material.transparency, loaded_material.transparency);
        }
        
        // Verify groups and faces
        let orig_group = &original_obj_data.groups[0];
        let loaded_group = &loaded_obj_data.groups[0];
        assert_eq!(orig_group.name, loaded_group.name);
        assert_eq!(orig_group.faces.len(), loaded_group.faces.len());
        
        // Cleanup
        let _ = fs::remove_file(obj_file);
        let _ = fs::remove_file(mtl_file);
    }

    #[test]
    fn test_mtl_writer_standalone() {
        let mtl_file = "test_mtl_writer.mtl";
        
        // Create materials
        let mut materials = HashMap::new();
        
        let mut plastic = obj::Material::new("plastic".to_string());
        plastic.ambient = Some([0.1, 0.1, 0.1]);
        plastic.diffuse = Some([0.8, 0.2, 0.2]);
        plastic.specular = Some([0.9, 0.9, 0.9]);
        plastic.shininess = Some(32.0);
        plastic.transparency = Some(1.0);
        plastic.illumination = Some(2);
        plastic.diffuse_map = Some("texture.jpg".to_string());
        materials.insert("plastic".to_string(), plastic);
        
        let mut glass = obj::Material::new("glass".to_string());
        glass.diffuse = Some([0.0, 0.8, 0.0]);
        glass.transparency = Some(0.5);
        glass.illumination = Some(4);
        materials.insert("glass".to_string(), glass);
        
        // Write MTL file
        obj::RobustObjWriter::write_mtl_file(&materials, mtl_file).unwrap();
        
        // Read back and verify
        let loaded_materials = obj::RobustObjReader::read_mtl_file(mtl_file).unwrap();
        assert_eq!(materials.len(), loaded_materials.len());
        
        // Verify plastic material
        let loaded_plastic = loaded_materials.get("plastic").unwrap();
        assert_eq!(loaded_plastic.ambient, Some([0.1, 0.1, 0.1]));
        assert_eq!(loaded_plastic.diffuse, Some([0.8, 0.2, 0.2]));
        assert_eq!(loaded_plastic.specular, Some([0.9, 0.9, 0.9]));
        assert_eq!(loaded_plastic.shininess, Some(32.0));
        assert_eq!(loaded_plastic.transparency, Some(1.0));
        assert_eq!(loaded_plastic.illumination, Some(2));
        assert_eq!(loaded_plastic.diffuse_map, Some("texture.jpg".to_string()));
        
        // Verify glass material
        let loaded_glass = loaded_materials.get("glass").unwrap();
        assert_eq!(loaded_glass.diffuse, Some([0.0, 0.8, 0.0]));
        assert_eq!(loaded_glass.transparency, Some(0.5));
        assert_eq!(loaded_glass.illumination, Some(4));
        
        // Cleanup
        let _ = fs::remove_file(mtl_file);
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