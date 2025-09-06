//! Example demonstrating XYZ/CSV point cloud reading and writing
//! 
//! This example shows how to use the XYZ/CSV reader and writer with various
//! file formats and schemas.

use threecrate_io::{XyzCsvReader, XyzCsvWriter, XyzCsvWriteOptions, XyzCsvPoint};
use threecrate_core::{PointCloud, Point3f, Vector3f};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("XYZ/CSV Point Cloud Example");
    println!("===========================");
    
    // Example 1: Basic XYZ format (space-separated, no header)
    println!("\n1. Basic XYZ format:");
    let xyz_content = "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n";
    fs::write("example.xyz", xyz_content)?;
    
    let cloud = XyzCsvReader::read_point_cloud("example.xyz")?;
    println!("   Read {} points from XYZ file", cloud.len());
    for (i, point) in cloud.iter().enumerate() {
        println!("   Point {}: ({}, {}, {})", i, point.x, point.y, point.z);
    }
    
    // Example 2: CSV format with header
    println!("\n2. CSV format with header:");
    let csv_content = "x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n";
    fs::write("example.csv", csv_content)?;
    
    let cloud = XyzCsvReader::read_point_cloud("example.csv")?;
    println!("   Read {} points from CSV file", cloud.len());
    
    // Example 3: CSV with colors and normals
    println!("\n3. CSV with colors and normals:");
    let detailed_content = "x,y,z,intensity,r,g,b,nx,ny,nz\n1.0,2.0,3.0,0.8,255,0,0,0.0,0.0,1.0\n4.0,5.0,6.0,0.6,0,255,0,0.0,1.0,0.0\n";
    fs::write("example_detailed.csv", detailed_content)?;
    
    let points = XyzCsvReader::read_detailed_points("example_detailed.csv")?;
    println!("   Read {} detailed points", points.len());
    for (i, point) in points.iter().enumerate() {
        println!("   Point {}: pos=({}, {}, {}), intensity={:?}, color={:?}, normal={:?}", 
                i, point.position.x, point.position.y, point.position.z,
                point.intensity, point.color, point.normal);
    }
    
    // Example 4: Writing point clouds
    println!("\n4. Writing point clouds:");
    
    // Create a simple point cloud
    let mut cloud = PointCloud::new();
    cloud.push(Point3f::new(0.0, 0.0, 0.0));
    cloud.push(Point3f::new(1.0, 1.0, 1.0));
    cloud.push(Point3f::new(2.0, 2.0, 2.0));
    
    // Write as XYZ
    let xyz_options = XyzCsvWriteOptions::xyz();
    XyzCsvWriter::write_point_cloud(&cloud, "output.xyz", &xyz_options)?;
    println!("   Written XYZ file with {} points", cloud.len());
    
    // Write as CSV with header
    let csv_options = XyzCsvWriteOptions::csv_with_header();
    XyzCsvWriter::write_point_cloud(&cloud, "output.csv", &csv_options)?;
    println!("   Written CSV file with header");
    
    // Write detailed points with colors
    let detailed_points = vec![
        XyzCsvPoint::with_color(Point3f::new(0.0, 0.0, 0.0), [255, 0, 0]),
        XyzCsvPoint::with_intensity(Point3f::new(1.0, 1.0, 1.0), 0.8),
        XyzCsvPoint::with_normal(Point3f::new(2.0, 2.0, 2.0), Vector3f::new(0.0, 0.0, 1.0)),
    ];
    
    let complete_options = XyzCsvWriteOptions::csv_complete();
    XyzCsvWriter::write_detailed_points(&detailed_points, "output_detailed.csv", &complete_options)?;
    println!("   Written detailed CSV file with all attributes");
    
    // Example 5: Auto-detection
    println!("\n5. Auto-detection capabilities:");
    
    // Test different delimiters
    let tab_content = "x\ty\tz\n1.0\t2.0\t3.0\n";
    fs::write("example_tab.txt", tab_content)?;
    
    let cloud = XyzCsvReader::read_point_cloud("example_tab.txt")?;
    println!("   Auto-detected tab delimiter and read {} points", cloud.len());
    
    // Example 6: Using the unified I/O interface
    println!("\n6. Using unified I/O interface:");
    
    // This works with the registry system
    let cloud = threecrate_io::read_point_cloud("example.xyz")?;
    println!("   Read {} points using unified interface", cloud.len());
    
    threecrate_io::write_point_cloud(&cloud, "unified_output.xyz")?;
    println!("   Written using unified interface");
    
    // Cleanup
    let files = ["example.xyz", "example.csv", "example_detailed.csv", 
                 "output.xyz", "output.csv", "output_detailed.csv", 
                 "example_tab.txt", "unified_output.xyz"];
    for file in &files {
        let _ = fs::remove_file(file);
    }
    
    println!("\nExample completed successfully!");
    Ok(())
}
