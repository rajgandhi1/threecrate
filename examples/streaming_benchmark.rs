//! Streaming vs Full-Load Benchmark
//! 
//! This benchmark compares the performance and memory usage of streaming
//! vs full-load approaches for reading large point cloud and mesh files.

use threecrate_io::{read_point_cloud, read_point_cloud_iter, read_mesh, read_mesh_iter};
use threecrate_core::Point3f;
use std::env;
use std::time::Instant;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <file_path> [chunk_size]", args[0]);
        eprintln!("  file_path: Path to PLY or OBJ file");
        eprintln!("  chunk_size: Optional chunk size for streaming (default: 1000)");
        return Ok(());
    }
    
    let file_path = &args[1];
    let chunk_size = args.get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);
    
    println!("Benchmarking: {}", file_path);
    println!("Chunk size: {}", chunk_size);
    println!();
    
    // Run benchmarks
    if file_path.ends_with(".ply") || file_path.ends_with(".obj") {
        benchmark_point_cloud(file_path, chunk_size)?;
        benchmark_mesh(file_path, chunk_size)?;
    } else {
        eprintln!("Unsupported file format. Please use .ply or .obj files.");
        return Ok(());
    }
    
    Ok(())
}

fn benchmark_point_cloud(file_path: &str, chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Point Cloud Benchmark ===");
    
    // Test full-load approach
    println!("Testing full-load approach...");
    let start_time = Instant::now();
    let cloud = read_point_cloud(file_path)?;
    let full_load_time = start_time.elapsed();
    let point_count = cloud.len();
    let memory_usage = std::mem::size_of_val(&cloud) + 
        (cloud.len() * std::mem::size_of::<Point3f>());
    
    println!("Full-load results:");
    println!("  Points: {}", point_count);
    println!("  Time: {:.2?}", full_load_time);
    println!("  Memory: {:.2} MB", memory_usage as f64 / 1_000_000.0);
    println!("  Points/sec: {:.0}", point_count as f64 / full_load_time.as_secs_f64());
    println!();
    
    // Test streaming approach
    println!("Testing streaming approach...");
    let start_time = Instant::now();
    let mut stream_point_count = 0;
    let mut iter = read_point_cloud_iter(file_path, Some(chunk_size))?;
    
    while let Some(result) = iter.next() {
        match result {
            Ok(_point) => {
                stream_point_count += 1;
            }
            Err(e) => {
                eprintln!("Error reading point: {}", e);
                return Err(e.into());
            }
        }
    }
    
    let streaming_time = start_time.elapsed();
    let streaming_memory = std::mem::size_of::<Vec<u8>>() + (chunk_size * 12); // Estimate
    
    println!("Streaming results:");
    println!("  Points: {}", stream_point_count);
    println!("  Time: {:.2?}", streaming_time);
    println!("  Memory: {:.2} KB", streaming_memory as f64 / 1_000.0);
    println!("  Points/sec: {:.0}", stream_point_count as f64 / streaming_time.as_secs_f64());
    println!();
    
    // Compare results
    println!("Comparison:");
    println!("  Time ratio (streaming/full): {:.2}x", 
        streaming_time.as_secs_f64() / full_load_time.as_secs_f64());
    println!("  Memory ratio (streaming/full): {:.2}x", 
        streaming_memory as f64 / memory_usage as f64);
    println!("  Speed ratio (streaming/full): {:.2}x", 
        full_load_time.as_secs_f64() / streaming_time.as_secs_f64());
    println!();
    
    Ok(())
}

fn benchmark_mesh(file_path: &str, chunk_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mesh Benchmark ===");
    
    // Test full-load approach
    println!("Testing full-load approach...");
    let start_time = Instant::now();
    let mesh = read_mesh(file_path)?;
    let full_load_time = start_time.elapsed();
    let face_count = mesh.face_count();
    let vertex_count = mesh.vertex_count();
    let memory_usage = std::mem::size_of_val(&mesh) + 
        (mesh.vertices.len() * std::mem::size_of::<Point3f>()) +
        (mesh.faces.len() * std::mem::size_of::<[usize; 3]>());
    
    println!("Full-load results:");
    println!("  Vertices: {}", vertex_count);
    println!("  Faces: {}", face_count);
    println!("  Time: {:.2?}", full_load_time);
    println!("  Memory: {:.2} MB", memory_usage as f64 / 1_000_000.0);
    println!("  Faces/sec: {:.0}", face_count as f64 / full_load_time.as_secs_f64());
    println!();
    
    // Test streaming approach
    println!("Testing streaming approach...");
    let start_time = Instant::now();
    let mut stream_face_count = 0;
    let mut iter = read_mesh_iter(file_path, Some(chunk_size))?;
    
    while let Some(result) = iter.next() {
        match result {
            Ok(_face) => {
                stream_face_count += 1;
            }
            Err(e) => {
                eprintln!("Error reading face: {}", e);
                return Err(e.into());
            }
        }
    }
    
    let streaming_time = start_time.elapsed();
    let streaming_memory = std::mem::size_of::<Vec<u8>>() + (chunk_size * 16); // Estimate
    
    println!("Streaming results:");
    println!("  Faces: {}", stream_face_count);
    println!("  Time: {:.2?}", streaming_time);
    println!("  Memory: {:.2} KB", streaming_memory as f64 / 1_000.0);
    println!("  Faces/sec: {:.0}", stream_face_count as f64 / streaming_time.as_secs_f64());
    println!();
    
    // Compare results
    println!("Comparison:");
    println!("  Time ratio (streaming/full): {:.2}x", 
        streaming_time.as_secs_f64() / full_load_time.as_secs_f64());
    println!("  Memory ratio (streaming/full): {:.2}x", 
        streaming_memory as f64 / memory_usage as f64);
    println!("  Speed ratio (streaming/full): {:.2}x", 
        full_load_time.as_secs_f64() / streaming_time.as_secs_f64());
    println!();
    
    Ok(())
}

#[allow(dead_code)]
fn create_test_files() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large test PLY file
    let mut file = File::create("large_test.ply")?;
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex 10000")?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "end_header")?;
    
    for i in 0..10000 {
        let x = (i as f32) * 0.1;
        let y = (i as f32) * 0.2;
        let z = (i as f32) * 0.3;
        writeln!(file, "{} {} {}", x, y, z)?;
    }
    
    // Create a large test OBJ file
    let mut file = File::create("large_test.obj")?;
    writeln!(file, "# Large test OBJ file")?;
    
    for i in 0..10000 {
        let x = (i as f32) * 0.1;
        let y = (i as f32) * 0.2;
        let z = (i as f32) * 0.3;
        writeln!(file, "v {} {} {}", x, y, z)?;
    }
    
    // Add some faces
    for i in 0..5000 {
        let v1 = i * 2 + 1;
        let v2 = i * 2 + 2;
        let v3 = i * 2 + 3;
        writeln!(file, "f {} {} {}", v1, v2, v3)?;
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_creation() {
        create_test_files().unwrap();
        
        // Test that files were created
        assert!(std::path::Path::new("large_test.ply").exists());
        assert!(std::path::Path::new("large_test.obj").exists());
        
        // Cleanup
        std::fs::remove_file("large_test.ply").unwrap();
        std::fs::remove_file("large_test.obj").unwrap();
    }
}
