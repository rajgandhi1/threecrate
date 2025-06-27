//! GPU-accelerated 3D point cloud processing example
//!
//! This example demonstrates how to use the GPU-accelerated algorithms
//! provided by threecrate-gpu for high-performance point cloud processing.

use threecrate_core::{PointCloud, Point3f};
use threecrate_gpu::{GpuContext, gpu_estimate_normals, gpu_icp, gpu_remove_statistical_outliers};
use nalgebra::Point3;
use std::time::Instant;

#[tokio::main]
async fn main() -> threecrate_core::Result<()> {
    println!("🚀 GPU-Accelerated 3D Point Cloud Processing Example");
    println!("===================================================");

    // Initialize GPU context
    println!("\n📱 Initializing GPU context...");
    let gpu_context = GpuContext::new().await?;
    println!("✅ GPU context initialized successfully!");

    // Create sample point clouds
    let source_cloud = create_sample_point_cloud(1000, 0.0);
    let mut target_cloud = create_sample_point_cloud(1000, 0.1); // Slightly offset
    
    // Add some noise/outliers to demonstrate filtering
    add_outliers(&mut target_cloud, 50);
    
    println!("\n📊 Created sample point clouds:");
    println!("   - Source: {} points", source_cloud.len());
    println!("   - Target: {} points (with outliers)", target_cloud.len());

    // GPU Normal Estimation
    println!("\n🧭 Computing normals using GPU...");
    let start = Instant::now();
    let normal_cloud = gpu_estimate_normals(&gpu_context, &mut source_cloud.clone(), 10).await?;
    let gpu_normals_time = start.elapsed();
    println!("✅ GPU normals computed in {:?}", gpu_normals_time);
    println!("   - Normal cloud size: {}", normal_cloud.len());

    // GPU Statistical Outlier Removal
    println!("\n🧹 Removing outliers using GPU...");
    let start = Instant::now();
    let filtered_cloud = gpu_remove_statistical_outliers(&gpu_context, &target_cloud, 10, 2.0).await?;
    let gpu_filter_time = start.elapsed();
    println!("✅ GPU filtering completed in {:?}", gpu_filter_time);
    println!("   - Original: {} points", target_cloud.len());
    println!("   - Filtered: {} points", filtered_cloud.len());
    println!("   - Removed: {} outliers", target_cloud.len() - filtered_cloud.len());

    // GPU ICP Registration
    println!("\n🎯 Performing ICP registration using GPU...");
    let start = Instant::now();
    let transformation = gpu_icp(
        &gpu_context,
        &source_cloud,
        &filtered_cloud,
        50,     // max iterations
        1e-6,   // convergence threshold
        1.0,    // max correspondence distance
    ).await?;
    let gpu_icp_time = start.elapsed();
    println!("✅ GPU ICP completed in {:?}", gpu_icp_time);
    println!("   - Translation: {:?}", transformation.translation.vector);
    println!("   - Rotation angle: {:.6} radians", transformation.rotation.angle());

    // Direct GPU Context Usage
    println!("\n⚡ Direct GPU context usage:");
    let start = Instant::now();
    let normals_vec = gpu_context.compute_normals(&source_cloud.points, 10).await?;
    let direct_time = start.elapsed();
    println!("✅ Direct normal computation: {:?}", direct_time);
    println!("   - Computed {} normal vectors", normals_vec.len());

    println!("\n🎉 GPU processing example completed successfully!");
    println!("💡 All operations used GPU acceleration for maximum performance");

    Ok(())
}

/// Create a sample point cloud in the shape of a sphere
fn create_sample_point_cloud(num_points: usize, offset: f32) -> PointCloud<Point3f> {
    let mut points = Vec::with_capacity(num_points);
    
    for i in 0..num_points {
        // Generate points on a sphere with some noise
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / (num_points as f32);
        let phi = std::f32::consts::PI * ((i * 7) % num_points) as f32 / (num_points as f32);
        
        let radius = 1.0 + 0.1 * ((i as f32) * 0.1).sin(); // Add some variation
        
        let x = radius * phi.sin() * theta.cos() + offset;
        let y = radius * phi.sin() * theta.sin() + offset;
        let z = radius * phi.cos() + offset;
        
        points.push(Point3::new(x, y, z));
    }
    
    PointCloud::from_points(points)
}

/// Add random outliers to a point cloud for testing filtering
fn add_outliers(cloud: &mut PointCloud<Point3f>, num_outliers: usize) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for _ in 0..num_outliers {
        // Add points far from the main cluster
        let x = rng.gen_range(-5.0..5.0);
        let y = rng.gen_range(-5.0..5.0);
        let z = rng.gen_range(-5.0..5.0);
        
        cloud.push(Point3::new(x, y, z));
    }
} 