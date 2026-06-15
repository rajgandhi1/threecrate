//! Euclidean Cluster Extraction Example
//!
//! Demonstrates region-growing Euclidean cluster extraction on point clouds
//! with multiple spatially-separated object blobs, as implemented for issue #95.

use rand::prelude::*;
use rand::rng;
use threecrate_algorithms::{
    extract_euclidean_clusters, extract_euclidean_clusters_parallel, EuclideanClusterConfig,
};
use threecrate_core::{Point3f, PointCloud};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Euclidean Cluster Extraction Example ===\n");

    // Example 1: Three well-separated blobs
    println!("1. Three well-separated blobs:");
    let cloud1 = create_three_blobs();
    let config1 = EuclideanClusterConfig::new(0.5, 50, 10_000);
    let result1 = extract_euclidean_clusters(&cloud1, &config1)?;

    println!("   Input points : {}", cloud1.len());
    println!("   Clusters found: {}", result1.num_clusters());
    for (i, cluster) in result1.clusters.iter().enumerate() {
        println!("   Cluster {}: {} points", i + 1, cluster.len());
    }

    // Example 2: Parallel variant on a larger cloud
    println!("\n2. Parallel extraction on a denser scene:");
    let cloud2 = create_dense_scene();
    let config2 = EuclideanClusterConfig::new(0.4, 100, 50_000);
    let result2 = extract_euclidean_clusters_parallel(&cloud2, &config2)?;

    println!("   Input points : {}", cloud2.len());
    println!("   Clusters found: {}", result2.num_clusters());
    for (i, cluster) in result2.clusters.iter().enumerate() {
        println!("   Cluster {}: {} points", i + 1, cluster.len());
    }

    // Example 3: Size filtering – small noise blobs are suppressed
    println!("\n3. Size filtering (small blobs removed):");
    let cloud3 = create_cloud_with_noise_blobs();
    let config3 = EuclideanClusterConfig::new(0.5, 200, 10_000);
    let result3 = extract_euclidean_clusters(&cloud3, &config3)?;

    println!("   Input points  : {}", cloud3.len());
    println!(
        "   Clusters found: {} (noise blobs filtered out)",
        result3.num_clusters()
    );

    // Example 4: Extract sub-clouds per cluster
    println!("\n4. Sub-cloud extraction per cluster:");
    let cloud4 = create_three_blobs();
    let config4 = EuclideanClusterConfig::new(0.5, 50, 10_000);
    let result4 = extract_euclidean_clusters(&cloud4, &config4)?;

    for i in 0..result4.num_clusters() {
        let sub = result4.get_cluster_cloud(&cloud4, i).unwrap();
        println!("   Cluster {} sub-cloud: {} points", i + 1, sub.len());
    }

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Three sphere-shaped blobs at well-separated positions
fn create_three_blobs() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    let mut rng = rng();

    let blobs = [
        (Point3f::new(0.0, 0.0, 0.0), 0.4_f32, 500_usize),
        (Point3f::new(5.0, 0.0, 0.0), 0.4, 400),
        (Point3f::new(0.0, 5.0, 0.0), 0.4, 300),
    ];

    for (center, radius, count) in &blobs {
        let mut added = 0;
        while added < *count {
            let x: f32 = rng.random_range(-radius..=*radius);
            let y: f32 = rng.random_range(-radius..=*radius);
            let z: f32 = rng.random_range(-radius..=*radius);
            if x * x + y * y + z * z <= radius * radius {
                cloud.push(Point3f::new(center.x + x, center.y + y, center.z + z));
                added += 1;
            }
        }
    }

    cloud
}

/// A denser scene with four blobs
fn create_dense_scene() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    let mut rng = rng();

    let blobs = [
        (Point3f::new(0.0, 0.0, 0.0), 0.5_f32, 1000_usize),
        (Point3f::new(6.0, 0.0, 0.0), 0.5, 800),
        (Point3f::new(0.0, 6.0, 0.0), 0.5, 600),
        (Point3f::new(6.0, 6.0, 0.0), 0.5, 400),
    ];

    for (center, radius, count) in &blobs {
        let mut added = 0;
        while added < *count {
            let x: f32 = rng.random_range(-radius..=*radius);
            let y: f32 = rng.random_range(-radius..=*radius);
            let z: f32 = rng.random_range(-radius..=*radius);
            if x * x + y * y + z * z <= radius * radius {
                cloud.push(Point3f::new(center.x + x, center.y + y, center.z + z));
                added += 1;
            }
        }
    }

    cloud
}

/// Two large blobs plus several tiny noise blobs that should be filtered out
fn create_cloud_with_noise_blobs() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    let mut rng = rng();

    // Large blobs
    for (center, radius, count) in [
        (Point3f::new(0.0, 0.0, 0.0), 0.4_f32, 500_usize),
        (Point3f::new(8.0, 0.0, 0.0), 0.4, 400),
    ] {
        let mut added = 0;
        while added < count {
            let x: f32 = rng.random_range(-radius..=radius);
            let y: f32 = rng.random_range(-radius..=radius);
            let z: f32 = rng.random_range(-radius..=radius);
            if x * x + y * y + z * z <= radius * radius {
                cloud.push(Point3f::new(center.x + x, center.y + y, center.z + z));
                added += 1;
            }
        }
    }

    // Tiny noise blobs (10 points each) – below min_cluster_size of 200
    for cx in [3.0_f32, 5.0, 7.0] {
        for _ in 0..10 {
            let x: f32 = rng.random_range(-0.1..0.1);
            let y: f32 = rng.random_range(-0.1..0.1);
            let z: f32 = rng.random_range(-0.1..0.1);
            cloud.push(Point3f::new(cx + x, 3.0 + y, z));
        }
    }

    cloud
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_three_blobs_example() {
        let cloud = create_three_blobs();
        let config = EuclideanClusterConfig::new(0.5, 50, 10_000);
        let result = extract_euclidean_clusters(&cloud, &config).unwrap();
        assert_eq!(result.num_clusters(), 3);
    }

    #[test]
    fn test_dense_scene_example() {
        let cloud = create_dense_scene();
        let config = EuclideanClusterConfig::new(0.4, 100, 50_000);
        let result = extract_euclidean_clusters_parallel(&cloud, &config).unwrap();
        assert_eq!(result.num_clusters(), 4);
    }

    #[test]
    fn test_noise_filtered_example() {
        let cloud = create_cloud_with_noise_blobs();
        let config = EuclideanClusterConfig::new(0.5, 200, 10_000);
        let result = extract_euclidean_clusters(&cloud, &config).unwrap();
        assert_eq!(
            result.num_clusters(),
            2,
            "Only the two large blobs should survive"
        );
    }
}
