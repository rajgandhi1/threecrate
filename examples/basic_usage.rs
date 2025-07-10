//! Basic usage example for threecrate
//! 
//! This example demonstrates fundamental operations:
//! - Creating point clouds
//! - Loading and saving data
//! - Basic algorithms
//! - Visualization

use threecrate_core::{PointCloud, Point3f, TriangleMesh};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("threecrate Umbrella Crate Example");
    println!("==============================");

    // Create a simple point cloud
    let points = vec![
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(1.0, 0.0, 0.0),
        Point3f::new(0.0, 1.0, 0.0),
        Point3f::new(0.0, 0.0, 1.0),
        Point3f::new(1.0, 1.0, 0.0),
        Point3f::new(1.0, 0.0, 1.0),
        Point3f::new(0.0, 1.0, 1.0),
        Point3f::new(1.0, 1.0, 1.0),
    ];

    let cloud = PointCloud::from_points(points);
    println!("Created point cloud with {} points", cloud.len());

    // Demonstrate core functionality
    println!("\nCore functionality:");
    println!("- Point cloud has {} points", cloud.len());
    println!("- Point cloud is empty: {}", cloud.is_empty());

    // Demonstrate algorithm functionality
    println!("\nAlgorithms:");
    println!("- Algorithms module is available");
    println!("- Many algorithms are still being implemented (marked with todo!())");

    // Demonstrate I/O functionality
    println!("\nI/O:");
    println!("- I/O module is available");
    println!("- File format support: PLY, OBJ, LAS/LAZ");

    // Create a simple mesh
    let vertices = vec![
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(1.0, 0.0, 0.0),
        Point3f::new(0.0, 1.0, 0.0),
    ];
    
    let faces = vec![
        [0, 1, 2],
    ];
    
    let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
    println!("\nCreated mesh with {} vertices and {} faces", mesh.vertices.len(), mesh.faces.len());

    // Demonstrate simplification functionality
    println!("\nSimplification:");
    println!("- Simplification module is available");
    println!("- Many simplification algorithms are still being implemented");

    println!("\nExample completed successfully!");
    Ok(())
} 