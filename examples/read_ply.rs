//! Example demonstrating robust PLY file reading capabilities
//! 
//! This example shows how to use the enhanced PLY reader to read various PLY formats
//! including ASCII and binary (little/big endian) with comprehensive metadata support.

use std::env;
use std::process;
use threecrate_io::ply::{RobustPlyReader, PlyFormat};
use threecrate_io::{read_point_cloud, read_mesh};

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <ply_file>", args[0]);
        eprintln!("Example: {} bunny.ply", args[0]);
        process::exit(1);
    }
    
    let ply_file = &args[1];
    
    println!("Reading PLY file: {}", ply_file);
    println!("{}", "=".repeat(50));
    
    // Read complete PLY data with metadata
    match RobustPlyReader::read_ply_file(ply_file) {
        Ok(ply_data) => {
            // Display header information
            println!("📋 PLY Header Information:");
            println!("  Format: {:?}", ply_data.header.format);
            println!("  Version: {}", ply_data.header.version);
            
            if !ply_data.header.comments.is_empty() {
                println!("  Comments:");
                for comment in &ply_data.header.comments {
                    println!("    💬 {}", comment);
                }
            }
            
            if !ply_data.header.obj_info.is_empty() {
                println!("  Object Info:");
                for info in &ply_data.header.obj_info {
                    println!("    ℹ️  {}", info);
                }
            }
            
            println!();
            
            // Display element information
            println!("📊 Element Information:");
            for element in &ply_data.header.elements {
                println!("  {} (count: {})", element.name, element.count);
                for property in &element.properties {
                    println!("    - {} ({:?})", property.name, property.property_type);
                }
            }
            
            println!();
            
            // Try to read as point cloud
            match read_point_cloud(ply_file) {
                Ok(point_cloud) => {
                    println!("☁️  Point Cloud Data:");
                    println!("  Total points: {}", point_cloud.len());
                    
                    if !point_cloud.is_empty() {
                        println!("  First few points:");
                        for (i, point) in point_cloud.iter().take(5).enumerate() {
                            println!("    Point {}: ({:.3}, {:.3}, {:.3})", i, point.x, point.y, point.z);
                        }
                        
                        if point_cloud.len() > 5 {
                            println!("    ... and {} more points", point_cloud.len() - 5);
                        }
                        
                        // Calculate bounding box
                        let mut min_x = f32::INFINITY;
                        let mut max_x = f32::NEG_INFINITY;
                        let mut min_y = f32::INFINITY;
                        let mut max_y = f32::NEG_INFINITY;
                        let mut min_z = f32::INFINITY;
                        let mut max_z = f32::NEG_INFINITY;
                        
                        for point in &point_cloud {
                            min_x = min_x.min(point.x);
                            max_x = max_x.max(point.x);
                            min_y = min_y.min(point.y);
                            max_y = max_y.max(point.y);
                            min_z = min_z.min(point.z);
                            max_z = max_z.max(point.z);
                        }
                        
                        println!("  Bounding Box:");
                        println!("    X: [{:.3}, {:.3}] (size: {:.3})", min_x, max_x, max_x - min_x);
                        println!("    Y: [{:.3}, {:.3}] (size: {:.3})", min_y, max_y, max_y - min_y);
                        println!("    Z: [{:.3}, {:.3}] (size: {:.3})", min_z, max_z, max_z - min_z);
                    }
                }
                Err(e) => println!("  ⚠️  Could not read as point cloud: {}", e),
            }
            
            println!();
            
            // Try to read as mesh
            match read_mesh(ply_file) {
                Ok(mesh) => {
                    println!("🔺 Mesh Data:");
                    println!("  Vertices: {}", mesh.vertex_count());
                    println!("  Faces: {}", mesh.face_count());
                    println!("  Has normals: {}", mesh.normals.is_some());
                    
                    if mesh.vertex_count() > 0 {
                        println!("  First few vertices:");
                        for (i, vertex) in mesh.vertices.iter().take(3).enumerate() {
                            println!("    Vertex {}: ({:.3}, {:.3}, {:.3})", i, vertex.x, vertex.y, vertex.z);
                        }
                    }
                    
                    if mesh.face_count() > 0 {
                        println!("  First few faces:");
                        for (i, face) in mesh.faces.iter().take(3).enumerate() {
                            println!("    Face {}: [{}, {}, {}]", i, face[0], face[1], face[2]);
                        }
                    }
                    
                    if let Some(normals) = &mesh.normals {
                        println!("  First few normals:");
                        for (i, normal) in normals.iter().take(3).enumerate() {
                            println!("    Normal {}: ({:.3}, {:.3}, {:.3})", i, normal.x, normal.y, normal.z);
                        }
                    }
                }
                Err(e) => println!("  ⚠️  Could not read as mesh: {}", e),
            }
            
            println!();
            
            // Display raw element data for first element (if any)
            if let Some((element_name, element_data)) = ply_data.elements.iter().next() {
                println!("🔍 Raw Element Data ({}): ", element_name);
                if let Some(first_instance) = element_data.first() {
                    println!("  First instance properties:");
                    for (prop_name, prop_value) in first_instance {
                        match prop_value {
                            threecrate_io::ply::PlyValue::Float(v) => println!("    {}: {:.6} (float)", prop_name, v),
                            threecrate_io::ply::PlyValue::Double(v) => println!("    {}: {:.6} (double)", prop_name, v),
                            threecrate_io::ply::PlyValue::Int(v) => println!("    {}: {} (int)", prop_name, v),
                            threecrate_io::ply::PlyValue::UInt(v) => println!("    {}: {} (uint)", prop_name, v),
                            threecrate_io::ply::PlyValue::UChar(v) => println!("    {}: {} (uchar)", prop_name, v),
                            threecrate_io::ply::PlyValue::Char(v) => println!("    {}: {} (char)", prop_name, v),
                            threecrate_io::ply::PlyValue::Short(v) => println!("    {}: {} (short)", prop_name, v),
                            threecrate_io::ply::PlyValue::UShort(v) => println!("    {}: {} (ushort)", prop_name, v),
                            threecrate_io::ply::PlyValue::List(values) => {
                                println!("    {}: [list with {} items]", prop_name, values.len());
                                if values.len() <= 10 {
                                    print!("      Values: ");
                                    for (i, val) in values.iter().enumerate() {
                                        if i > 0 { print!(", "); }
                                        match val {
                                            threecrate_io::ply::PlyValue::Int(v) => print!("{}", v),
                                            threecrate_io::ply::PlyValue::UInt(v) => print!("{}", v),
                                            threecrate_io::ply::PlyValue::Float(v) => print!("{:.3}", v),
                                            _ => print!("{:?}", val),
                                        }
                                    }
                                    println!();
                                }
                            }
                        }
                    }
                }
            }
            
            println!();
            println!("✅ Successfully read PLY file!");
            
            // Performance info
            match ply_data.header.format {
                PlyFormat::Ascii => println!("📝 Format: ASCII (human-readable, slower to parse)"),
                PlyFormat::BinaryLittleEndian => println!("💾 Format: Binary Little Endian (compact, fast)"),
                PlyFormat::BinaryBigEndian => println!("💾 Format: Binary Big Endian (compact, fast)"),
            }
        }
        Err(e) => {
            eprintln!("❌ Error reading PLY file: {}", e);
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_with_sample_file() {
        // Create a sample PLY file for testing
        let sample_content = r#"ply
format ascii 1.0
comment Created by threecrate example
obj_info Test mesh with colors and normals
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0 0.0 0.0 1.0 255 0 0
1.0 0.0 0.0 0.0 0.0 1.0 0 255 0
0.5 1.0 0.0 0.0 0.0 1.0 0 0 255
3 0 1 2
"#;
        
        let temp_file = "sample_test.ply";
        std::fs::write(temp_file, sample_content).unwrap();
        
        // Test that the example can read the file without panicking
        let ply_data = RobustPlyReader::read_ply_file(temp_file).unwrap();
        assert_eq!(ply_data.header.format, PlyFormat::Ascii);
        assert_eq!(ply_data.header.comments.len(), 1);
        assert_eq!(ply_data.header.obj_info.len(), 1);
        
        // Cleanup
        let _ = std::fs::remove_file(temp_file);
    }
}
