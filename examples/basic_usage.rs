//! Basic usage example for threecrate
//! 
//! This example demonstrates fundamental operations:
//! - Creating point clouds
//! - Loading and saving data
//! - Basic algorithms
//! - Visualization

use threecrate_core::{PointCloud, Point3f, Drawable};
use threecrate_algorithms::statistical_outlier_removal;
use rand::prelude::*;

fn main() -> threecrate_core::Result<()> {
    println!("ThreeCrate Basic Usage Example");
    println!("==============================");

    // Create a point cloud with some outliers
    let mut points = Vec::new();
    let mut rng = rand::thread_rng();

    // Generate main cluster of points
    for _ in 0..100 {
        points.push(Point3f::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ));
    }

    // Add some outliers
    for _ in 0..10 {
        points.push(Point3f::new(
            rng.gen_range(-10.0..-5.0),
            rng.gen_range(-10.0..-5.0),
            rng.gen_range(-10.0..-5.0),
        ));
    }

    for _ in 0..10 {
        points.push(Point3f::new(
            rng.gen_range(5.0..10.0),
            rng.gen_range(5.0..10.0),
            rng.gen_range(5.0..10.0),
        ));
    }

    let cloud = PointCloud::from_points(points);
    println!("Original point cloud: {} points", cloud.len());

    // Apply statistical outlier removal
    println!("\nApplying statistical outlier removal...");
    let filtered_cloud = statistical_outlier_removal(&cloud, 10, 1.0)?;
    println!("Filtered point cloud: {} points", filtered_cloud.len());
    println!("Removed {} outliers", cloud.len() - filtered_cloud.len());

    // Show some statistics
    let (min, max) = cloud.bounding_box();
    println!("\nOriginal cloud bounding box:");
    println!("  Min: ({:.2}, {:.2}, {:.2})", min.x, min.y, min.z);
    println!("  Max: ({:.2}, {:.2}, {:.2})", max.x, max.y, max.z);

    let (min_filtered, max_filtered) = filtered_cloud.bounding_box();
    println!("\nFiltered cloud bounding box:");
    println!("  Min: ({:.2}, {:.2}, {:.2})", min_filtered.x, min_filtered.y, min_filtered.z);
    println!("  Max: ({:.2}, {:.2}, {:.2})", max_filtered.x, max_filtered.y, max_filtered.z);

    Ok(())
} 