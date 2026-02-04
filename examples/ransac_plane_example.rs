//! RANSAC Plane Segmentation Example
//! 
//! This example demonstrates the RANSAC plane segmentation algorithm
//! on noisy planar point clouds, as requested in GitHub issue #4.

use threecrate_core::{PointCloud, Point3f};
use threecrate_algorithms::{segment_plane_ransac, plane_segmentation_ransac};
use rand::prelude::*;
use rand::thread_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RANSAC Plane Segmentation Example ===\n");
    
    // Example 1: Simple planar point cloud
    println!("1. Simple planar point cloud:");
    let simple_cloud = create_simple_planar_cloud();
    let (coefficients, inliers) = segment_plane_ransac(&simple_cloud, 1000, 0.01)?;
    
    println!("   Input points: {}", simple_cloud.len());
    println!("   Found inliers: {}", inliers.len());
    println!("   Plane coefficients: {:?}", coefficients);
    println!("   Inlier percentage: {:.1}%", (inliers.len() as f32 / simple_cloud.len() as f32) * 100.0);
    
    // Example 2: Noisy planar point cloud
    println!("\n2. Noisy planar point cloud:");
    let noisy_cloud = create_noisy_planar_cloud();
    let (coefficients, inliers) = segment_plane_ransac(&noisy_cloud, 1000, 0.05)?;
    
    println!("   Input points: {}", noisy_cloud.len());
    println!("   Found inliers: {}", inliers.len());
    println!("   Plane coefficients: {:?}", coefficients);
    println!("   Inlier percentage: {:.1}%", (inliers.len() as f32 / noisy_cloud.len() as f32) * 100.0);
    
    // Example 3: Tilted plane with outliers
    println!("\n3. Tilted plane with outliers:");
    let tilted_cloud = create_tilted_plane_with_outliers();
    let (coefficients, inliers) = segment_plane_ransac(&tilted_cloud, 2000, 0.1)?;
    
    println!("   Input points: {}", tilted_cloud.len());
    println!("   Found inliers: {}", inliers.len());
    println!("   Plane coefficients: {:?}", coefficients);
    println!("   Inlier percentage: {:.1}%", (inliers.len() as f32 / tilted_cloud.len() as f32) * 100.0);
    
    // Example 4: Multiple planes (demonstrating single plane detection)
    println!("\n4. Multiple planes (detecting largest plane):");
    let multi_plane_cloud = create_multiple_planes();
    let (coefficients, inliers) = segment_plane_ransac(&multi_plane_cloud, 3000, 0.05)?;
    
    println!("   Input points: {}", multi_plane_cloud.len());
    println!("   Found inliers: {}", inliers.len());
    println!("   Plane coefficients: {:?}", coefficients);
    println!("   Inlier percentage: {:.1}%", (inliers.len() as f32 / multi_plane_cloud.len() as f32) * 100.0);
    
    // Example 5: Using the alias function
    println!("\n5. Using plane_segmentation_ransac alias:");
    let (coefficients_alias, inliers_alias) = plane_segmentation_ransac(&simple_cloud, 1000, 0.01)?;
    
    println!("   Results identical: {}", coefficients == coefficients_alias && inliers == inliers_alias);
    println!("   Note: Results may differ due to RANSAC's random nature");
    
    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Create a simple planar point cloud on the XY plane
fn create_simple_planar_cloud() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    
    // Create a 10x10 grid on the XY plane (z=0)
    for i in 0..10 {
        for j in 0..10 {
            cloud.push(Point3f::new(i as f32, j as f32, 0.0));
        }
    }
    
    // Add a few outliers
    cloud.push(Point3f::new(5.0, 5.0, 10.0));
    cloud.push(Point3f::new(5.0, 5.0, -10.0));
    cloud.push(Point3f::new(15.0, 15.0, 5.0));
    
    cloud
}

/// Create a noisy planar point cloud
fn create_noisy_planar_cloud() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    let mut rng = thread_rng();
    
    // Create a 20x20 grid on the XY plane with noise
    for i in 0..20 {
        for j in 0..20 {
            let x = i as f32;
            let y = j as f32;
            let z = rng.gen_range(-0.03..0.03); // Add noise to z coordinate
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    // Add outliers
    for _ in 0..30 {
        let x = rng.gen_range(0.0..20.0);
        let y = rng.gen_range(0.0..20.0);
        let z = rng.gen_range(2.0..8.0); // Outliers above the plane
        cloud.push(Point3f::new(x, y, z));
    }
    
    cloud
}

/// Create a tilted plane with outliers
fn create_tilted_plane_with_outliers() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    let mut rng = thread_rng();
    
    // Create a tilted plane: x + y + z = 0
    for i in 0..15 {
        for j in 0..15 {
            let x = i as f32;
            let y = j as f32;
            let z = -(x + y); // Points on the plane x + y + z = 0
            
            // Add some noise
            let noise_x = rng.gen_range(-0.02..0.02);
            let noise_y = rng.gen_range(-0.02..0.02);
            let noise_z = rng.gen_range(-0.02..0.02);
            
            cloud.push(Point3f::new(x + noise_x, y + noise_y, z + noise_z));
        }
    }
    
    // Add outliers
    for _ in 0..50 {
        let x = rng.gen_range(0.0..15.0);
        let y = rng.gen_range(0.0..15.0);
        let z = rng.gen_range(5.0..15.0); // Outliers above the plane
        cloud.push(Point3f::new(x, y, z));
    }
    
    cloud
}

/// Create multiple planes (for demonstrating single plane detection)
fn create_multiple_planes() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    let mut rng = thread_rng();
    
    // First plane: z = 0 (largest)
    for i in 0..25 {
        for j in 0..25 {
            let x = i as f32;
            let y = j as f32;
            let z = rng.gen_range(-0.02..0.02); // Small noise
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    // Second plane: z = 5 (smaller)
    for i in 0..10 {
        for j in 0..10 {
            let x = i as f32;
            let y = j as f32;
            let z = 5.0 + rng.gen_range(-0.02..0.02); // Small noise
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    // Third plane: x = 0 (smallest)
    for i in 0..5 {
        for j in 0..5 {
            let x = rng.gen_range(-0.02..0.02); // Small noise
            let y = i as f32;
            let z = j as f32;
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    // Add some random outliers
    for _ in 0..20 {
        let x = rng.gen_range(-5.0..30.0);
        let y = rng.gen_range(-5.0..30.0);
        let z = rng.gen_range(-5.0..10.0);
        cloud.push(Point3f::new(x, y, z));
    }
    
    cloud
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_planar_cloud() {
        let cloud = create_simple_planar_cloud();
        assert_eq!(cloud.len(), 103); // 10x10 + 3 outliers
        
        let result = segment_plane_ransac(&cloud, 100, 0.01);
        assert!(result.is_ok());
        
        let (_coefficients, inliers) = result.unwrap();
        assert!(inliers.len() >= 95); // Should find most planar points
    }

    #[test]
    fn test_noisy_planar_cloud() {
        let cloud = create_noisy_planar_cloud();
        assert_eq!(cloud.len(), 430); // 20x20 + 30 outliers
        
        let result = segment_plane_ransac(&cloud, 100, 0.05);
        assert!(result.is_ok());
        
        let (_coefficients, inliers) = result.unwrap();
        assert!(inliers.len() >= 350); // Should find most planar points
    }

    #[test]
    fn test_tilted_plane() {
        let cloud = create_tilted_plane_with_outliers();
        assert_eq!(cloud.len(), 275); // 15x15 + 50 outliers
        
        let result = segment_plane_ransac(&cloud, 100, 0.1);
        assert!(result.is_ok());
        
        let (_coefficients, inliers) = result.unwrap();
        assert!(inliers.len() >= 200); // Should find most planar points
    }
} 