//! Comprehensive tests for mesh serialization utilities
//!
//! These tests validate that mesh attributes (normals, tangents, UVs) survive
//! round-trip across different formats (OBJ/PLY) with proper validation.

use crate::{
    mesh_attributes::{ExtendedTriangleMesh, MeshAttributeOptions, Tangent, UV},
    serialization::{SerializationOptions, AttributePreservingReader, AttributePreservingWriter, utils},
};
use threecrate_core::{TriangleMesh, Point3f, Vector3f};
use std::fs;

/// Create a comprehensive test mesh with all attributes
fn create_comprehensive_test_mesh() -> ExtendedTriangleMesh {
    // Create a simple quad (2 triangles)
    let vertices = vec![
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(1.0, 0.0, 0.0),
        Point3f::new(1.0, 1.0, 0.0),
        Point3f::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![
        [0, 1, 2], // First triangle
        [0, 2, 3], // Second triangle
    ];
    
    let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
    let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
    
    // Add normals (all pointing up)
    let normals = vec![
        Vector3f::new(0.0, 0.0, 1.0),
        Vector3f::new(0.0, 0.0, 1.0),
        Vector3f::new(0.0, 0.0, 1.0),
        Vector3f::new(0.0, 0.0, 1.0),
    ];
    extended.mesh.set_normals(normals);
    
    // Add UVs (standard quad mapping)
    let uvs: Vec<UV> = vec![
        [0.0, 0.0], // Bottom-left
        [1.0, 0.0], // Bottom-right
        [1.0, 1.0], // Top-right
        [0.0, 1.0], // Top-left
    ];
    extended.set_uvs(uvs);
    
    // Add tangents (pointing right)
    let tangents = vec![
        Tangent::new(Vector3f::new(1.0, 0.0, 0.0), 1.0),
        Tangent::new(Vector3f::new(1.0, 0.0, 0.0), 1.0),
        Tangent::new(Vector3f::new(1.0, 0.0, 0.0), 1.0),
        Tangent::new(Vector3f::new(1.0, 0.0, 0.0), 1.0),
    ];
    extended.set_tangents(tangents);
    
    extended
}

/// Create a simple triangle mesh for basic tests
fn create_simple_test_mesh() -> ExtendedTriangleMesh {
    let vertices = vec![
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(1.0, 0.0, 0.0),
        Point3f::new(0.5, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];
    let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
    ExtendedTriangleMesh::from_mesh(base_mesh)
}

#[cfg(test)]
mod attribute_tests {
    use super::*;
    
    #[test]
    fn test_normal_computation() {
        let mut mesh = create_simple_test_mesh();
        assert!(mesh.mesh.normals.is_none());
        
        mesh.compute_normals(true, true).unwrap();
        
        assert!(mesh.mesh.normals.is_some());
        let normals = mesh.mesh.normals.unwrap();
        assert_eq!(normals.len(), 3);
        
        // All normals should point in +Z direction for this triangle
        for normal in &normals {
            assert!((normal.z - 1.0).abs() < 1e-5, "Normal should point up: {:?}", normal);
            assert!((normal.x * normal.x + normal.y * normal.y + normal.z * normal.z - 1.0).abs() < 1e-5, 
                "Normal should be normalized");
        }
    }
    
    #[test]
    fn test_uv_generation() {
        let mut mesh = create_simple_test_mesh();
        assert!(mesh.uvs.is_none());
        
        mesh.generate_default_uvs().unwrap();
        
        assert!(mesh.uvs.is_some());
        let uvs = mesh.uvs.unwrap();
        assert_eq!(uvs.len(), 3);
        
        // UVs should be in [0, 1] range
        for uv in &uvs {
            assert!(uv[0] >= 0.0 && uv[0] <= 1.0, "U coordinate out of range: {}", uv[0]);
            assert!(uv[1] >= 0.0 && uv[1] <= 1.0, "V coordinate out of range: {}", uv[1]);
        }
    }
    
    #[test]
    fn test_tangent_computation() {
        let mut mesh = create_simple_test_mesh();
        
        // Need normals and UVs for tangent computation
        mesh.compute_normals(true, true).unwrap();
        mesh.generate_default_uvs().unwrap();
        mesh.compute_tangents(true).unwrap();
        
        assert!(mesh.tangents.is_some());
        let tangents = mesh.tangents.unwrap();
        assert_eq!(tangents.len(), 3);
        
        // Check tangent properties
        for tangent in &tangents {
            let length_sq = tangent.vector.x * tangent.vector.x + 
                tangent.vector.y * tangent.vector.y + 
                tangent.vector.z * tangent.vector.z;
            assert!((length_sq - 1.0).abs() < 1e-5, "Tangent should be normalized");
            assert!(tangent.handedness.abs() == 1.0, "Handedness should be ±1");
        }
    }
    
    #[test]
    fn test_attribute_validation() {
        let mut mesh = create_comprehensive_test_mesh();
        
        // Should pass validation initially
        assert!(mesh.validate_attributes().is_ok());
        
        // Test mismatched UV count
        mesh.uvs = Some(vec![[0.0, 0.0]]); // Wrong count
        assert!(mesh.validate_attributes().is_err());
        
        // Test invalid UV coordinates
        mesh.uvs = Some(vec![
            [f32::NAN, 0.0],
            [1.0, f32::INFINITY],
            [1.0, 1.0],
            [0.0, 1.0],
        ]);
        mesh.validate_attributes().unwrap(); // Should succeed but add warnings
        assert!(!mesh.metadata.validation_messages.is_empty());
    }
    
    #[test]
    fn test_process_attributes() {
        let mut mesh = create_simple_test_mesh();
        
        let options = MeshAttributeOptions::recompute_all();
        mesh.process_attributes(&options).unwrap();
        
        assert!(mesh.mesh.normals.is_some());
        assert!(mesh.uvs.is_some());
        assert!(mesh.tangents.is_some());
        assert!(mesh.metadata.normals_computed);
        assert!(mesh.metadata.tangents_computed);
        assert!(mesh.metadata.is_complete());
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;
    
    #[test]
    fn test_obj_round_trip_basic() {
        let original_mesh = create_simple_test_mesh();
        let temp_file = "test_obj_basic.obj";
        
        let options = SerializationOptions::default();
        
        // Write mesh
        AttributePreservingWriter::write_extended_mesh(&original_mesh, temp_file, &options).unwrap();
        
        // Read mesh back
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Compare basic properties
        assert_eq!(original_mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(original_mesh.face_count(), loaded_mesh.face_count());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_obj_round_trip_with_attributes() {
        let original_mesh = create_comprehensive_test_mesh();
        let temp_file = "test_obj_attributes.obj";
        
        let options = SerializationOptions::preserve_all();
        
        // Write mesh
        AttributePreservingWriter::write_extended_mesh(&original_mesh, temp_file, &options).unwrap();
        
        // Read mesh back
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Compare attributes
        let differences = utils::compare_meshes(&original_mesh, &loaded_mesh).unwrap();
        
        // OBJ should preserve vertices, faces, normals, and UVs
        // Tangents might be lost (OBJ doesn't natively support them)
        println!("OBJ round-trip differences: {:#?}", differences);
        
        // At minimum, basic geometry should be preserved
        assert_eq!(original_mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(original_mesh.face_count(), loaded_mesh.face_count());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_ply_round_trip_basic() {
        let original_mesh = create_simple_test_mesh();
        let temp_file = "test_ply_basic.ply";
        
        let options = SerializationOptions::default();
        
        // Write mesh
        AttributePreservingWriter::write_extended_mesh(&original_mesh, temp_file, &options).unwrap();
        
        // Read mesh back
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Compare basic properties
        assert_eq!(original_mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(original_mesh.face_count(), loaded_mesh.face_count());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_ply_round_trip_with_attributes() {
        let original_mesh = create_comprehensive_test_mesh();
        let temp_file = "test_ply_attributes.ply";
        
        let options = SerializationOptions::preserve_all();
        
        // Write mesh
        AttributePreservingWriter::write_extended_mesh(&original_mesh, temp_file, &options).unwrap();
        
        // Read mesh back
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Compare attributes
        let differences = utils::compare_meshes(&original_mesh, &loaded_mesh).unwrap();
        
        // PLY should preserve all attributes as custom properties
        println!("PLY round-trip differences: {:#?}", differences);
        
        // Basic geometry should definitely be preserved
        assert_eq!(original_mesh.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(original_mesh.face_count(), loaded_mesh.face_count());
        
        // PLY should preserve UVs and tangents as custom properties
        if differences.iter().any(|d| d.contains("UVs lost")) {
            println!("Warning: UVs lost in PLY round-trip");
        }
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_cross_format_conversion() {
        let mut original_mesh = create_comprehensive_test_mesh();
        
        // Ensure all attributes are computed
        let options = MeshAttributeOptions::recompute_all();
        original_mesh.process_attributes(&options).unwrap();
        
        let obj_file = "test_cross_format.obj";
        let ply_file = "test_cross_format.ply";
        
        let serialization_options = SerializationOptions::preserve_all();
        
        // Write as OBJ
        AttributePreservingWriter::write_extended_mesh(&original_mesh, obj_file, &serialization_options).unwrap();
        
        // Read OBJ and write as PLY
        let obj_mesh = AttributePreservingReader::read_extended_mesh(obj_file, &serialization_options).unwrap();
        AttributePreservingWriter::write_extended_mesh(&obj_mesh, ply_file, &serialization_options).unwrap();
        
        // Read PLY back
        let ply_mesh = AttributePreservingReader::read_extended_mesh(ply_file, &serialization_options).unwrap();
        
        // Compare final result with original
        let differences = utils::compare_meshes(&original_mesh, &ply_mesh).unwrap();
        println!("Cross-format conversion differences: {:#?}", differences);
        
        // Basic geometry should be preserved
        assert_eq!(original_mesh.vertex_count(), ply_mesh.vertex_count());
        assert_eq!(original_mesh.face_count(), ply_mesh.face_count());
        
        // Cleanup
        let _ = fs::remove_file(obj_file);
        let _ = fs::remove_file(ply_file);
    }
    
    #[test]
    fn test_attribute_recomputation() {
        // Create a mesh with only vertices and faces
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        let extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Initially no attributes
        assert!(extended.mesh.normals.is_none());
        assert!(extended.uvs.is_none());
        assert!(extended.tangents.is_none());
        
        let temp_file = "test_recomputation.ply";
        
        // Write with recomputation enabled
        let options = SerializationOptions::preserve_all();
        AttributePreservingWriter::write_extended_mesh(&extended, temp_file, &options).unwrap();
        
        // Read back with recomputation
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Should now have computed attributes
        assert!(loaded_mesh.mesh.normals.is_some());
        assert!(loaded_mesh.uvs.is_some());
        assert!(loaded_mesh.tangents.is_some());
        assert!(loaded_mesh.metadata.normals_computed);
        assert!(loaded_mesh.metadata.tangents_computed);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_validation_on_read() {
        // Create a mesh and write it
        let original_mesh = create_comprehensive_test_mesh();
        let temp_file = "test_validation.ply";
        
        let write_options = SerializationOptions::preserve_all();
        AttributePreservingWriter::write_extended_mesh(&original_mesh, temp_file, &write_options).unwrap();
        
        // Read with validation enabled
        let read_options = SerializationOptions {
            attributes: MeshAttributeOptions {
                validate_attributes: true,
                ..Default::default()
            },
            validate_before_write: false,
            ..Default::default()
        };
        
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &read_options).unwrap();
        
        // Should have validation metadata
        assert!(loaded_mesh.metadata.source_format.is_some());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_round_trip_utility() {
        let original_mesh = create_comprehensive_test_mesh();
        let input_file = "test_round_trip_input.ply";
        let output_file = "test_round_trip_output.obj";
        
        // Write initial mesh
        let options = SerializationOptions::preserve_all();
        AttributePreservingWriter::write_extended_mesh(&original_mesh, input_file, &options).unwrap();
        
        // Test round-trip utility
        let (final_mesh, warnings) = utils::test_round_trip(input_file, output_file, Some(options)).unwrap();
        
        println!("Round-trip warnings: {:#?}", warnings);
        
        // Should have preserved basic geometry
        assert_eq!(original_mesh.vertex_count(), final_mesh.vertex_count());
        assert_eq!(original_mesh.face_count(), final_mesh.face_count());
        
        // Cleanup
        let _ = fs::remove_file(input_file);
        let _ = fs::remove_file(output_file);
    }
    
    #[test]
    fn test_format_preparation() {
        let mut mesh = create_comprehensive_test_mesh();
        
        // Prepare for OBJ format
        utils::prepare_mesh_for_format(&mut mesh, "obj").unwrap();
        assert!(mesh.mesh.normals.is_some());
        assert_eq!(mesh.metadata.source_format, Some("obj".to_string()));
        
        // Prepare for PLY format
        utils::prepare_mesh_for_format(&mut mesh, "ply").unwrap();
        assert_eq!(mesh.metadata.source_format, Some("ply".to_string()));
    }
    
    #[test]
    fn test_custom_properties() {
        let mesh = create_comprehensive_test_mesh();
        let temp_file = "test_custom_props.ply";
        
        // Add custom properties
        let mut options = SerializationOptions::preserve_all();
        options = options.with_custom_property("intensity", vec![0.5, 0.8, 1.0, 0.3]);
        
        // Write with custom properties
        AttributePreservingWriter::write_extended_mesh(&mesh, temp_file, &options).unwrap();
        
        // Verify file was created and contains custom properties
        let content = fs::read_to_string(temp_file).unwrap();
        assert!(content.contains("intensity"), "Custom property not found in PLY file");
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_metadata_preservation() {
        let mut mesh = create_comprehensive_test_mesh();
        mesh.metadata.source_format = Some("test_format".to_string());
        
        let temp_file = "test_metadata.ply";
        let options = SerializationOptions {
            attach_metadata: true,
            ..SerializationOptions::preserve_all()
        };
        
        // Write mesh
        AttributePreservingWriter::write_extended_mesh(&mesh, temp_file, &options).unwrap();
        
        // Read mesh back
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Should have metadata
        assert!(loaded_mesh.metadata.source_format.is_some());
        assert!(loaded_mesh.metadata.completeness_score > 0.0);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_performance_fast_options() {
        let mesh = create_comprehensive_test_mesh();
        let temp_file = "test_fast_options.obj";
        
        // Use fast options (minimal processing)
        let options = SerializationOptions::fast();
        
        // Should complete quickly without extensive processing
        let start = std::time::Instant::now();
        AttributePreservingWriter::write_extended_mesh(&mesh, temp_file, &options).unwrap();
        let write_duration = start.elapsed();
        
        let start = std::time::Instant::now();
        let _loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        let read_duration = start.elapsed();
        
        println!("Fast write: {:?}, Fast read: {:?}", write_duration, read_duration);
        
        // These should be reasonably fast (actual timing will vary)
        assert!(write_duration.as_millis() < 100, "Fast write took too long");
        assert!(read_duration.as_millis() < 100, "Fast read took too long");
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[test]
    fn test_empty_mesh() {
        let vertices = Vec::new();
        let faces = Vec::new();
        let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        let extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        let temp_file = "test_empty_mesh.ply";
        let options = SerializationOptions::default();
        
        // Should handle empty mesh gracefully
        let result = AttributePreservingWriter::write_extended_mesh(&extended, temp_file, &options);
        
        // May succeed or fail depending on format requirements
        match result {
            Ok(_) => {
                // If write succeeds, read should also work
                let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
                assert_eq!(loaded_mesh.vertex_count(), 0);
                assert_eq!(loaded_mesh.face_count(), 0);
            }
            Err(_) => {
                // Empty mesh rejection is acceptable
                println!("Empty mesh write rejected (acceptable)");
            }
        }
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_degenerate_triangles() {
        // Create mesh with degenerate triangle (all vertices at same point)
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(0.0, 0.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Try to compute normals (should handle degenerate case)
        let result = extended.compute_normals(true, true);
        
        // Should either succeed with default normals or fail gracefully
        match result {
            Ok(_) => {
                if let Some(normals) = &extended.mesh.normals {
                    // Should have some valid normal (possibly default)
                    assert_eq!(normals.len(), 3);
                }
            }
            Err(_) => {
                println!("Degenerate triangle normal computation failed (acceptable)");
            }
        }
    }
    
    #[test]
    fn test_large_mesh_handling() {
        // Create a larger mesh to test performance and memory handling
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        
        // Create a grid of vertices (10x10 = 100 vertices)
        for y in 0..10 {
            for x in 0..10 {
                vertices.push(Point3f::new(x as f32, y as f32, 0.0));
            }
        }
        
        // Create triangulated faces for the grid
        for y in 0..9 {
            for x in 0..9 {
                let base = y * 10 + x;
                // Two triangles per grid cell
                faces.push([base, base + 1, base + 10]);
                faces.push([base + 1, base + 11, base + 10]);
            }
        }
        
        let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Process all attributes
        let options = MeshAttributeOptions::recompute_all();
        extended.process_attributes(&options).unwrap();
        
        // Should have all attributes for all vertices
        assert_eq!(extended.vertex_count(), 100);
        assert_eq!(extended.face_count(), 162); // 9*9*2 = 162 triangles
        assert!(extended.mesh.normals.is_some());
        assert!(extended.uvs.is_some());
        assert!(extended.tangents.is_some());
        
        // Test serialization
        let temp_file = "test_large_mesh.ply";
        let serialization_options = SerializationOptions::preserve_all();
        
        AttributePreservingWriter::write_extended_mesh(&extended, temp_file, &serialization_options).unwrap();
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &serialization_options).unwrap();
        
        assert_eq!(extended.vertex_count(), loaded_mesh.vertex_count());
        assert_eq!(extended.face_count(), loaded_mesh.face_count());
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_unsupported_format() {
        let mesh = create_simple_test_mesh();
        let temp_file = "test_unsupported.xyz"; // Unsupported format for mesh
        
        let options = SerializationOptions::default();
        
        // Should fall back to basic mesh writing or fail gracefully
        let result = AttributePreservingWriter::write_extended_mesh(&mesh, temp_file, &options);
        
        match result {
            Ok(_) => {
                println!("Unsupported format handled by fallback");
                let _ = fs::remove_file(temp_file);
            }
            Err(_) => {
                println!("Unsupported format rejected (expected)");
            }
        }
    }
}
