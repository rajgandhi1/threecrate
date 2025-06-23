//! Basic usage example for 3DCrate
//! 
//! This example demonstrates fundamental operations:
//! - Creating point clouds
//! - Loading and saving data
//! - Basic algorithms
//! - Visualization

use threecrate_core::{PointCloud, TriangleMesh, Point3f, Vector3f};
use threecrate_io::{
    ply::{PlyReader, PlyWriter},
    obj::{ObjReader, ObjWriter},
    pasture::PastureWriter,
    PointCloudReader, PointCloudWriter, MeshReader, MeshWriter,
};
// use threecrate_algorithms::estimate_normals;
// use threecrate_visualization::show_point_cloud;
use anyhow::Result;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ThreeCrate Basic I/O Usage Example");
    println!("==================================");

    // Create a sample point cloud
    let mut point_cloud = PointCloud::new();
    point_cloud.push(Point3f::new(0.0, 0.0, 0.0));
    point_cloud.push(Point3f::new(1.0, 0.0, 0.0));
    point_cloud.push(Point3f::new(0.0, 1.0, 0.0));
    point_cloud.push(Point3f::new(0.0, 0.0, 1.0));
    point_cloud.push(Point3f::new(1.0, 1.0, 1.0));

    println!("Created point cloud with {} points", point_cloud.len());

    // Example 1: PLY Point Cloud I/O
    println!("\n1. PLY Point Cloud I/O:");
    let ply_path = "example_cloud.ply";
    
    // Write PLY point cloud
    match PlyWriter::write_point_cloud(&point_cloud, ply_path) {
        Ok(_) => println!("✓ Successfully wrote PLY point cloud to {}", ply_path),
        Err(e) => println!("✗ Failed to write PLY point cloud: {}", e),
    }

    // Read PLY point cloud
    match PlyReader::read_point_cloud(ply_path) {
        Ok(loaded_cloud) => {
            println!("✓ Successfully read PLY point cloud with {} points", loaded_cloud.len());
        }
        Err(e) => println!("✗ Failed to read PLY point cloud: {}", e),
    }

    // Example 2: Create and save a triangle mesh
    println!("\n2. Triangle Mesh I/O:");
    let vertices = vec![
        Point3f::new(0.0, 0.0, 0.0),  // vertex 0
        Point3f::new(1.0, 0.0, 0.0),  // vertex 1
        Point3f::new(0.5, 1.0, 0.0),  // vertex 2
        Point3f::new(0.5, 0.5, 1.0),  // vertex 3
    ];
    
    let faces = vec![
        [0, 1, 2],  // bottom triangle
        [0, 1, 3],  // front triangle
        [1, 2, 3],  // right triangle
        [0, 2, 3],  // left triangle
    ];
    
    let mut mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
    
    // Add normals
    let normals = vec![
        Vector3f::new(0.0, 0.0, -1.0),
        Vector3f::new(0.0, -1.0, 0.0),
        Vector3f::new(1.0, 0.0, 0.0),
        Vector3f::new(0.0, 1.0, 0.0),
    ];
    mesh.set_normals(normals);
    
    println!("Created triangle mesh with {} vertices and {} faces", 
             mesh.vertex_count(), mesh.face_count());

    // Example 3: OBJ Mesh I/O
    println!("\n3. OBJ Mesh I/O:");
    let obj_path = "example_mesh.obj";
    
    // Write OBJ mesh
    match ObjWriter::write_mesh(&mesh, obj_path) {
        Ok(_) => println!("✓ Successfully wrote OBJ mesh to {}", obj_path),
        Err(e) => println!("✗ Failed to write OBJ mesh: {}", e),
    }

    // Read OBJ mesh
    match ObjReader::read_mesh(obj_path) {
        Ok(loaded_mesh) => {
            println!("✓ Successfully read OBJ mesh with {} vertices and {} faces", 
                     loaded_mesh.vertex_count(), loaded_mesh.face_count());
        }
        Err(e) => println!("✗ Failed to read OBJ mesh: {}", e),
    }

    // Example 4: PLY Mesh I/O
    println!("\n4. PLY Mesh I/O:");
    let ply_mesh_path = "example_mesh.ply";
    
    // Write PLY mesh
    match PlyWriter::write_mesh(&mesh, ply_mesh_path) {
        Ok(_) => println!("✓ Successfully wrote PLY mesh to {}", ply_mesh_path),
        Err(e) => println!("✗ Failed to write PLY mesh: {}", e),
    }

    // Read PLY mesh
    match PlyReader::read_mesh(ply_mesh_path) {
        Ok(loaded_mesh) => {
            println!("✓ Successfully read PLY mesh with {} vertices and {} faces", 
                     loaded_mesh.vertex_count(), loaded_mesh.face_count());
        }
        Err(e) => println!("✗ Failed to read PLY mesh: {}", e),
    }

    // Example 5: Pasture Point Cloud Formats (LAS/LAZ/PCD)
    println!("\n5. Pasture Point Cloud Formats:");
    
    // Note: Pasture integration is not yet complete
    println!("⚠ Pasture integration is currently incomplete and will be implemented in future versions");
    
    // Try to write as PCD format (this will show the current state)
    let pcd_path = "example_cloud.pcd";
    match PastureWriter::write_point_cloud(&point_cloud, pcd_path) {
        Ok(_) => println!("✓ Successfully wrote PCD point cloud to {}", pcd_path),
        Err(e) => println!("ℹ PCD format: {}", e),
    }

    // Example 6: Auto-detection based on file extension
    println!("\n6. Auto-detection I/O:");
    
    // Using the convenience functions that auto-detect format
    match threecrate_io::read_point_cloud("example_cloud.ply") {
        Ok(cloud) => println!("✓ Auto-detected and read PLY point cloud with {} points", cloud.len()),
        Err(e) => println!("✗ Failed to auto-read point cloud: {}", e),
    }
    
    match threecrate_io::read_mesh("example_mesh.obj") {
        Ok(mesh) => println!("✓ Auto-detected and read OBJ mesh with {} vertices", mesh.vertex_count()),
        Err(e) => println!("✗ Failed to auto-read mesh: {}", e),
    }

    // Clean up example files
    let _ = std::fs::remove_file("example_cloud.ply");
    let _ = std::fs::remove_file("example_mesh.obj");
    let _ = std::fs::remove_file("example_mesh.ply");

    println!("\n✓ All examples completed!");
    
    Ok(())
} 