//! Example demonstrating k-nearest neighbors functionality
//! 
//! This example shows how to use the k-nearest neighbors search functionality
//! with both KD-tree and brute force implementations.

use threecrate_core::{PointCloud, Point3f, NearestNeighborSearch};
use threecrate_algorithms::nearest_neighbor::{KdTree, BruteForceSearch};
use threecrate_algorithms::point_cloud_ops::PointCloudNeighbors;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== K-Nearest Neighbors Example ===\n");

    // Create a point cloud with some sample points
    let mut cloud = PointCloud::new();
    
    // Add points in a grid pattern
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..3 {
                cloud.push(Point3f::new(x as f32, y as f32, z as f32));
            }
        }
    }
    
    println!("Created point cloud with {} points", cloud.len());
    
    // Add some random points for more interesting results
    let mut rng = rand::thread_rng();
    for _ in 0..20 {
        cloud.push(Point3f::new(
            rng.gen_range(-2.0..7.0),
            rng.gen_range(-2.0..7.0),
            rng.gen_range(-1.0..4.0),
        ));
    }
    
    println!("Added 20 random points. Total: {} points\n", cloud.len());

    // Example 1: Find k-nearest neighbors for each point in the cloud
    println!("1. Finding k-nearest neighbors for each point:");
    let k = 3;
    let neighbors = cloud.k_nearest_neighbors(k);
    
    println!("   Found {} nearest neighbors for each point:", k);
    for (i, point_neighbors) in neighbors.iter().take(5).enumerate() {
        println!("   Point {}: {:?}", i, point_neighbors);
    }
    if neighbors.len() > 5 {
        println!("   ... and {} more points", neighbors.len() - 5);
    }
    println!();

    // Example 2: Find k-nearest neighbors for a specific query point
    println!("2. Finding k-nearest neighbors for a specific query point:");
    let query_point = Point3f::new(2.5, 2.5, 1.5);
    let k = 5;
    
    let nearest = cloud.find_k_nearest(&query_point, k);
    println!("   Query point: {:?}", query_point);
    println!("   {} nearest neighbors:", k);
    for (i, (idx, distance)) in nearest.iter().enumerate() {
        println!("   {}. Point {} at distance {:.3}", i + 1, idx, distance);
    }
    println!();

    // Example 3: Find neighbors within a radius
    println!("3. Finding neighbors within a radius:");
    let radius = 2.0;
    let radius_neighbors = cloud.find_radius_neighbors(&query_point, radius);
    
    println!("   Query point: {:?}", query_point);
    println!("   Neighbors within radius {}: {}", radius, radius_neighbors.len());
    for (i, (idx, distance)) in radius_neighbors.iter().take(10).enumerate() {
        println!("   {}. Point {} at distance {:.3}", i + 1, idx, distance);
    }
    if radius_neighbors.len() > 10 {
        println!("   ... and {} more neighbors", radius_neighbors.len() - 10);
    }
    println!();

    // Example 4: Compare KD-tree vs Brute Force
    println!("4. Comparing KD-tree vs Brute Force:");
    
    let kdtree = KdTree::new(&cloud.points)?;
    let brute_force = BruteForceSearch::new(&cloud.points);
    
    let test_query = Point3f::new(1.0, 1.0, 1.0);
    let test_k = 4;
    
    let kdtree_result = kdtree.find_k_nearest(&test_query, test_k);
    let brute_result = brute_force.find_k_nearest(&test_query, test_k);
    
    println!("   Query point: {:?}", test_query);
    println!("   KD-tree result: {:?}", kdtree_result);
    println!("   Brute force result: {:?}", brute_result);
    
    // Verify results are consistent
    let results_match = kdtree_result.len() == brute_result.len() &&
        kdtree_result.iter().zip(brute_result.iter()).all(|(a, b)| {
            a.0 == b.0 && (a.1 - b.1).abs() < 1e-6
        });
    
    println!("   Results match: {}", results_match);
    println!();

    // Example 5: Performance comparison
    println!("5. Performance comparison:");
    
    let iterations = 100;
    let query_points: Vec<Point3f> = (0..iterations)
        .map(|_| Point3f::new(
            rng.gen_range(-1.0..6.0),
            rng.gen_range(-1.0..6.0),
            rng.gen_range(0.0..3.0),
        ))
        .collect();
    
    // Time KD-tree searches
    let start = std::time::Instant::now();
    for query in &query_points {
        let _ = kdtree.find_k_nearest(query, 5);
    }
    let kdtree_time = start.elapsed();
    
    // Time brute force searches
    let start = std::time::Instant::now();
    for query in &query_points {
        let _ = brute_force.find_k_nearest(query, 5);
    }
    let brute_time = start.elapsed();
    
    println!("   {} queries with k=5:", iterations);
    println!("   KD-tree time: {:?}", kdtree_time);
    println!("   Brute force time: {:?}", brute_time);
    println!("   Speedup: {:.2}x", brute_time.as_nanos() as f64 / kdtree_time.as_nanos() as f64);
    println!();

    // Example 6: Edge cases
    println!("6. Testing edge cases:");
    
    // Empty point cloud
    let empty_cloud = PointCloud::new();
    let empty_result = empty_cloud.find_k_nearest(&Point3f::new(0.0, 0.0, 0.0), 5);
    println!("   Empty cloud result: {} neighbors", empty_result.len());
    
    // k = 0
    let zero_k_result = cloud.find_k_nearest(&query_point, 0);
    println!("   k = 0 result: {} neighbors", zero_k_result.len());
    
    // k larger than number of points
    let large_k_result = cloud.find_k_nearest(&query_point, cloud.len() + 10);
    println!("   k > points result: {} neighbors", large_k_result.len());
    
    // Radius = 0
    let zero_radius_result = cloud.find_radius_neighbors(&query_point, 0.0);
    println!("   radius = 0 result: {} neighbors", zero_radius_result.len());
    
    // Negative radius
    let negative_radius_result = cloud.find_radius_neighbors(&query_point, -1.0);
    println!("   negative radius result: {} neighbors", negative_radius_result.len());

    println!("\n=== Example completed successfully! ===");
    Ok(())
} 