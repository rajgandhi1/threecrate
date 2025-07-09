//! Basic usage example for 3DCrate
//! 
//! This example demonstrates fundamental operations:
//! - Creating point clouds
//! - Loading and saving data
//! - Basic algorithms
//! - Visualization

use threecrate::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("3DCrate Umbrella Crate Example");
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
    println!("- Point cloud bounds: {:?}", cloud.bounding_box());
    println!("- Point cloud center: {:?}", cloud.center());

    // Demonstrate algorithm functionality (if enabled)
    #[cfg(feature = "algorithms")]
    {
        println!("\nAlgorithms:");
        println!("- Algorithms module is available");
        println!("- Many algorithms are still being implemented (marked with todo!())");
    }

    // Demonstrate I/O functionality (if enabled)
    #[cfg(feature = "io")]
    {
        use threecrate_io::ply::PlyWriter;
        
        println!("\nI/O:");
        
        // Save to PLY format
        match PlyWriter::write_point_cloud(&cloud, "output.ply") {
            Ok(_) => println!("- Saved point cloud to output.ply"),
            Err(e) => println!("- Failed to save: {}", e),
        }
    }

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

    // Demonstrate simplification functionality (if enabled)
    #[cfg(feature = "simplification")]
    {
        println!("\nSimplification:");
        println!("- Simplification module is available");
        println!("- Many simplification algorithms are still being implemented");
    }

    println!("\nExample completed successfully!");
    Ok(())
} 