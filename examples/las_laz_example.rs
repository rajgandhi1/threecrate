//! Example demonstrating LAS/LAZ file reading and writing using threecrate-io
//!
//! This example requires the `las_laz` feature to be enabled:
//! `cargo run --example las_laz_example --features las_laz`

#[cfg(feature = "las_laz")]
use threecrate_core::{PointCloud, Point3f};
#[cfg(feature = "las_laz")]
use threecrate_io::{read_point_cloud, write_point_cloud};

#[cfg(not(feature = "las_laz"))]
fn main() {
    eprintln!("This example requires the 'las_laz' feature to be enabled.");
    eprintln!("Run with: cargo run --example las_laz_example --features las_laz");
    std::process::exit(1);
}

#[cfg(feature = "las_laz")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LAS/LAZ Example");
    println!("===============");

    // Create a sample point cloud
    let mut cloud = PointCloud::new();
    cloud.push(Point3f::new(0.0, 0.0, 0.0));
    cloud.push(Point3f::new(1.0, 0.0, 0.0));
    cloud.push(Point3f::new(1.0, 1.0, 0.0));
    cloud.push(Point3f::new(0.0, 1.0, 0.0));
    cloud.push(Point3f::new(0.5, 0.5, 1.0));

    println!("Created point cloud with {} points", cloud.len());

    // Write to LAS file
    let las_file = "example_output.las";
    println!("Writing to {}...", las_file);
    write_point_cloud(&cloud, las_file)?;

    // Read back from LAS file
    println!("Reading from {}...", las_file);
    let loaded_cloud = read_point_cloud(las_file)?;

    println!("Loaded point cloud with {} points", loaded_cloud.len());

    // Verify the data
    if cloud.len() == loaded_cloud.len() {
        println!("✓ Point count matches");
    } else {
        println!("✗ Point count mismatch: expected {}, got {}", cloud.len(), loaded_cloud.len());
    }

    // Compare first few points
    for i in 0..std::cmp::min(3, cloud.len()) {
        let orig = cloud[i];
        let loaded = loaded_cloud[i];
        println!("Point {}: ({:.3}, {:.3}, {:.3}) -> ({:.3}, {:.3}, {:.3})",
                i, orig.x, orig.y, orig.z, loaded.x, loaded.y, loaded.z);
    }

    // Write to LAZ file as well
    let laz_file = "example_output.laz";
    println!("Writing to {}...", laz_file);
    write_point_cloud(&cloud, laz_file)?;

    // Clean up
    std::fs::remove_file(las_file)?;
    std::fs::remove_file(laz_file)?;

    println!("✓ Example completed successfully!");
    Ok(())
}
