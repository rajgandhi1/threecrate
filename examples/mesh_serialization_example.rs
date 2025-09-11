//! Example demonstrating mesh serialization utilities with attribute preservation
//!
//! This example shows how to use the new mesh serialization utilities to ensure
//! that mesh attributes (normals, tangents, UVs) survive round-trip across formats.

use threecrate_core::{TriangleMesh, Point3f};
use threecrate_io::{
    ExtendedTriangleMesh, MeshAttributeOptions, SerializationOptions,
    AttributePreservingReader, AttributePreservingWriter,
};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mesh Serialization Utilities Example");
    println!("=====================================");
    
    // Create a simple pyramid mesh
    let vertices = vec![
        Point3f::new(0.0, 0.0, 0.0),   // Base center
        Point3f::new(1.0, 0.0, 0.0),   // Base corner 1
        Point3f::new(0.0, 1.0, 0.0),   // Base corner 2
        Point3f::new(-1.0, 0.0, 0.0),  // Base corner 3
        Point3f::new(0.0, -1.0, 0.0),  // Base corner 4
        Point3f::new(0.0, 0.0, 1.0),   // Apex
    ];
    
    let faces = vec![
        // Base (two triangles)
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        // Sides
        [1, 5, 2],
        [2, 5, 3],
        [3, 5, 4],
        [4, 5, 1],
    ];
    
    println!("Created pyramid with {} vertices and {} faces", vertices.len(), faces.len());
    
    // Create base mesh
    let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
    let mut extended_mesh = ExtendedTriangleMesh::from_mesh(base_mesh);
    
    println!("Initial mesh completeness: {:.1}%", extended_mesh.metadata.completeness_score * 100.0);
    
    // Process attributes with different options
    println!("\n1. Computing mesh attributes...");
    let attribute_options = MeshAttributeOptions::recompute_all();
    extended_mesh.process_attributes(&attribute_options)?;
    
    println!("   ✓ Normals computed: {}", extended_mesh.mesh.normals.is_some());
    println!("   ✓ UVs generated: {}", extended_mesh.uvs.is_some());
    println!("   ✓ Tangents computed: {}", extended_mesh.tangents.is_some());
    println!("   Completeness after processing: {:.1}%", extended_mesh.metadata.completeness_score * 100.0);
    
    // Test OBJ round-trip
    println!("\n2. Testing OBJ round-trip...");
    let obj_file = "example_pyramid.obj";
    
    let serialization_options = SerializationOptions::preserve_all();
    
    // Write to OBJ
    AttributePreservingWriter::write_extended_mesh(&extended_mesh, obj_file, &serialization_options)?;
    println!("   ✓ Written to {}", obj_file);
    
    // Read back from OBJ
    let obj_mesh = AttributePreservingReader::read_extended_mesh(obj_file, &serialization_options)?;
    println!("   ✓ Read back from {}", obj_file);
    println!("   OBJ mesh completeness: {:.1}%", obj_mesh.metadata.completeness_score * 100.0);
    
    // Compare meshes
    let obj_differences = threecrate_io::serialization::utils::compare_meshes(&extended_mesh, &obj_mesh)?;
    if obj_differences.is_empty() {
        println!("   ✓ Perfect OBJ round-trip!");
    } else {
        println!("   ⚠ OBJ round-trip differences: {}", obj_differences.len());
        for diff in &obj_differences {
            println!("     - {}", diff);
        }
    }
    
    // Test PLY round-trip
    println!("\n3. Testing PLY round-trip...");
    let ply_file = "example_pyramid.ply";
    
    // Write to PLY
    AttributePreservingWriter::write_extended_mesh(&extended_mesh, ply_file, &serialization_options)?;
    println!("   ✓ Written to {}", ply_file);
    
    // Read back from PLY
    let ply_mesh = AttributePreservingReader::read_extended_mesh(ply_file, &serialization_options)?;
    println!("   ✓ Read back from {}", ply_file);
    println!("   PLY mesh completeness: {:.1}%", ply_mesh.metadata.completeness_score * 100.0);
    
    // Compare meshes
    let ply_differences = threecrate_io::serialization::utils::compare_meshes(&extended_mesh, &ply_mesh)?;
    if ply_differences.is_empty() {
        println!("   ✓ Perfect PLY round-trip!");
    } else {
        println!("   ⚠ PLY round-trip differences: {}", ply_differences.len());
        for diff in &ply_differences {
            println!("     - {}", diff);
        }
    }
    
    // Test cross-format conversion
    println!("\n4. Testing cross-format conversion (OBJ → PLY)...");
    let converted_file = "example_pyramid_converted.ply";
    
    // Convert OBJ mesh to PLY
    AttributePreservingWriter::write_extended_mesh(&obj_mesh, converted_file, &serialization_options)?;
    let converted_mesh = AttributePreservingReader::read_extended_mesh(converted_file, &serialization_options)?;
    
    let conversion_differences = threecrate_io::serialization::utils::compare_meshes(&extended_mesh, &converted_mesh)?;
    if conversion_differences.is_empty() {
        println!("   ✓ Perfect cross-format conversion!");
    } else {
        println!("   ⚠ Cross-format differences: {}", conversion_differences.len());
        for diff in &conversion_differences {
            println!("     - {}", diff);
        }
    }
    
    // Demonstrate validation
    println!("\n5. Validation results:");
    let validation_warnings = threecrate_io::mesh_attributes::utils::validate_round_trip(&extended_mesh)?;
    if validation_warnings.is_empty() {
        println!("   ✓ No validation warnings");
    } else {
        println!("   ⚠ Validation warnings: {}", validation_warnings.len());
        for warning in &validation_warnings {
            println!("     - {}", warning);
        }
    }
    
    // Show metadata
    println!("\n6. Mesh metadata:");
    println!("   Source format: {:?}", extended_mesh.metadata.source_format);
    println!("   Normals computed: {}", extended_mesh.metadata.normals_computed);
    println!("   Tangents computed: {}", extended_mesh.metadata.tangents_computed);
    println!("   UVs loaded: {}", extended_mesh.metadata.uvs_loaded);
    println!("   Completeness score: {:.1}%", extended_mesh.metadata.completeness_score * 100.0);
    println!("   Missing attributes: {:?}", extended_mesh.metadata.missing_attributes());
    
    // Demonstrate fast serialization
    println!("\n7. Testing fast serialization options...");
    let fast_file = "example_pyramid_fast.obj";
    let fast_options = SerializationOptions::fast();
    
    let start_time = std::time::Instant::now();
    AttributePreservingWriter::write_extended_mesh(&extended_mesh, fast_file, &fast_options)?;
    let write_duration = start_time.elapsed();
    
    let start_time = std::time::Instant::now();
    let _fast_mesh = AttributePreservingReader::read_extended_mesh(fast_file, &fast_options)?;
    let read_duration = start_time.elapsed();
    
    println!("   ✓ Fast write: {:?}", write_duration);
    println!("   ✓ Fast read: {:?}", read_duration);
    
    // Cleanup
    println!("\n8. Cleaning up temporary files...");
    let files_to_remove = [obj_file, ply_file, converted_file, fast_file];
    for file in &files_to_remove {
        if let Err(e) = fs::remove_file(file) {
            println!("   ⚠ Could not remove {}: {}", file, e);
        } else {
            println!("   ✓ Removed {}", file);
        }
    }
    
    println!("\n🎉 Mesh serialization utilities example completed successfully!");
    println!("\nKey features demonstrated:");
    println!("  • Automatic attribute computation (normals, UVs, tangents)");
    println!("  • Round-trip preservation across OBJ and PLY formats");
    println!("  • Cross-format conversion with attribute preservation");
    println!("  • Comprehensive validation and metadata tracking");
    println!("  • Configurable serialization options for performance vs quality");
    
    Ok(())
}
