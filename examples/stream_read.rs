//! Streaming file reading example
//! 
//! This example demonstrates how to use the streaming API to read large point cloud
//! and mesh files without loading them entirely into memory. This is particularly
//! useful for processing very large files that might not fit in available RAM.

use threecrate_io::{read_point_cloud_iter, read_mesh_iter};
use threecrate_core::Point3f;
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: {} <file_path> [chunk_size]", args[0]);
        eprintln!("  file_path: Path to PLY or OBJ file");
        eprintln!("  chunk_size: Optional chunk size for buffering (default: 1000)");
        return Ok(());
    }
    
    let file_path = &args[1];
    let chunk_size = args.get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);
    
    println!("Streaming file: {}", file_path);
    println!("Chunk size: {}", chunk_size);
    println!();
    
    // Determine file type and process accordingly
    if file_path.ends_with(".ply") || file_path.ends_with(".obj") {
        process_point_cloud(file_path, chunk_size)?;
        process_mesh(file_path, chunk_size)?;
    } else {
        eprintln!("Unsupported file format. Please use .ply or .obj files.");
        return Ok(());
    }
    
    Ok(())
}

fn process_point_cloud(file_path: &str, chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Processing Point Cloud ===");
    
    let start_time = Instant::now();
    let mut point_count = 0;
    let mut min_bounds = Point3f::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_bounds = Point3f::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    
    // Create streaming iterator
    let mut iter = read_point_cloud_iter(file_path, Some(chunk_size))?;
    
    // Process points one by one
    while let Some(result) = iter.next() {
        match result {
            Ok(point) => {
                point_count += 1;
                
                // Update bounding box
                min_bounds.x = min_bounds.x.min(point.x);
                min_bounds.y = min_bounds.y.min(point.y);
                min_bounds.z = min_bounds.z.min(point.z);
                
                max_bounds.x = max_bounds.x.max(point.x);
                max_bounds.y = max_bounds.y.max(point.y);
                max_bounds.z = max_bounds.z.max(point.z);
                
                // Print progress every 10000 points
                if point_count % 10000 == 0 {
                    println!("Processed {} points...", point_count);
                }
            }
            Err(e) => {
                eprintln!("Error reading point: {}", e);
                return Err(e.into());
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    
    println!("Point cloud processing complete!");
    println!("Total points: {}", point_count);
    println!("Bounding box:");
    println!("  Min: ({:.3}, {:.3}, {:.3})", min_bounds.x, min_bounds.y, min_bounds.z);
    println!("  Max: ({:.3}, {:.3}, {:.3})", max_bounds.x, max_bounds.y, max_bounds.z);
    println!("  Size: ({:.3}, {:.3}, {:.3})", 
        max_bounds.x - min_bounds.x,
        max_bounds.y - min_bounds.y,
        max_bounds.z - min_bounds.z
    );
    println!("Processing time: {:.2?}", elapsed);
    println!("Points per second: {:.0}", point_count as f64 / elapsed.as_secs_f64());
    println!();
    
    Ok(())
}

fn process_mesh(file_path: &str, chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Processing Mesh Faces ===");
    
    let start_time = Instant::now();
    let mut face_count = 0;
    let mut total_vertices = 0;
    
    // Create streaming iterator for faces
    let mut iter = read_mesh_iter(file_path, Some(chunk_size))?;
    
    // Process faces one by one
        while let Some(result) = iter.next() {
            match result {
                Ok(_face) => {
                    face_count += 1;
                    total_vertices += 3; // Each face has 3 vertices
                    
                    // Print progress every 10000 faces
                    if face_count % 10000 == 0 {
                        println!("Processed {} faces...", face_count);
                    }
                }
            Err(e) => {
                eprintln!("Error reading face: {}", e);
                return Err(e.into());
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    
    println!("Mesh processing complete!");
    println!("Total faces: {}", face_count);
    println!("Total vertices: {}", total_vertices);
    println!("Processing time: {:.2?}", elapsed);
    println!("Faces per second: {:.0}", face_count as f64 / elapsed.as_secs_f64());
    println!();
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    
    #[test]
    fn test_streaming_ply_point_cloud() {
        // Create a test PLY file
        let ply_content = r#"ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 1.0 1.0
2.0 2.0 2.0
"#;
        
        let temp_file = "test_streaming.ply";
        let mut file = File::create(temp_file).unwrap();
        file.write_all(ply_content.as_bytes()).unwrap();
        drop(file);
        
        // Test streaming reader
        let mut iter = read_point_cloud_iter(temp_file, Some(2)).unwrap();
        let mut points = Vec::new();
        
        while let Some(result) = iter.next() {
            points.push(result.unwrap());
        }
        
        assert_eq!(points.len(), 3);
        assert_eq!(points[0], Point3f::new(0.0, 0.0, 0.0));
        assert_eq!(points[1], Point3f::new(1.0, 1.0, 1.0));
        assert_eq!(points[2], Point3f::new(2.0, 2.0, 2.0));
        
        // Cleanup
        std::fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_streaming_obj_point_cloud() {
        // Create a test OBJ file
        let obj_content = r#"# Test OBJ file
v 0.0 0.0 0.0
v 1.0 1.0 1.0
v 2.0 2.0 2.0
f 1 2 3
"#;
        
        let temp_file = "test_streaming.obj";
        let mut file = File::create(temp_file).unwrap();
        file.write_all(obj_content.as_bytes()).unwrap();
        drop(file);
        
        // Test streaming reader
        let mut iter = read_point_cloud_iter(temp_file, Some(2)).unwrap();
        let mut points = Vec::new();
        
        while let Some(result) = iter.next() {
            points.push(result.unwrap());
        }
        
        assert_eq!(points.len(), 3);
        assert_eq!(points[0], Point3f::new(0.0, 0.0, 0.0));
        assert_eq!(points[1], Point3f::new(1.0, 1.0, 1.0));
        assert_eq!(points[2], Point3f::new(2.0, 2.0, 2.0));
        
        // Cleanup
        std::fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_streaming_obj_mesh() {
        // Create a test OBJ file with faces
        let obj_content = r#"# Test OBJ file
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 1.0 1.0 0.0
f 1 2 3
f 2 3 4
"#;
        
        let temp_file = "test_streaming_mesh.obj";
        let mut file = File::create(temp_file).unwrap();
        file.write_all(obj_content.as_bytes()).unwrap();
        drop(file);
        
        // Test streaming mesh reader
        let mut iter = read_mesh_iter(temp_file, Some(2)).unwrap();
        let mut faces = Vec::new();
        
        while let Some(result) = iter.next() {
            faces.push(result.unwrap());
        }
        
        assert_eq!(faces.len(), 2);
        assert_eq!(faces[0], [0, 1, 2]);
        assert_eq!(faces[1], [1, 2, 3]);
        
        // Cleanup
        std::fs::remove_file(temp_file).unwrap();
    }
}
