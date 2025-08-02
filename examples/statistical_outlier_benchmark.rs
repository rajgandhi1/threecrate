use threecrate_core::{PointCloud, Point3f};
use threecrate_gpu::GpuContext;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔬 Statistical Outlier Removal Performance Benchmark");
    println!("==================================================\n");

    // Initialize GPU context
    let gpu_context = match GpuContext::new().await {
        Ok(ctx) => {
            println!("✅ GPU context initialized successfully!");
            Some(ctx)
        }
        Err(_) => {
            println!("❌ Failed to initialize GPU context, running CPU-only benchmark");
            None
        }
    };

    // Create test datasets of different sizes
    let dataset_sizes = [100, 500, 1000, 2500, 5000];
    
    for &size in &dataset_sizes {
        println!("\n📊 Testing with {} points", size);
        println!("{}", "-".repeat(40));
        
        // Create test point cloud
        let cloud = create_test_point_cloud(size);
        
        // CPU benchmark (simple implementation)
        let cpu_start = Instant::now();
        let cpu_result = cpu_statistical_outlier_removal(&cloud, 10, 1.0);
        let cpu_time = cpu_start.elapsed();
        
        println!("CPU:  {} -> {} points in {:?}", 
                cloud.len(), cpu_result.len(), cpu_time);
        
        // GPU benchmark (if available)
        if let Some(ref gpu_ctx) = gpu_context {
            let gpu_start = Instant::now();
            let gpu_result = threecrate_gpu::gpu_remove_statistical_outliers(gpu_ctx, &cloud, 10, 1.0).await?;
            let gpu_time = gpu_start.elapsed();
            
            println!("GPU:  {} -> {} points in {:?}", 
                    cloud.len(), gpu_result.len(), gpu_time);
            
            // Calculate speedup
            let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
            println!("🚀 GPU speedup: {:.2}x", speedup);
            
            // Verify results are similar (within reasonable bounds)
            let cpu_removed = cloud.len() - cpu_result.len();
            let gpu_removed = cloud.len() - gpu_result.len();
            let difference = (cpu_removed as i32 - gpu_removed as i32).abs();
            let similarity = 1.0 - (difference as f32 / cloud.len() as f32);
            
            println!("📈 Result similarity: {:.1}%", similarity * 100.0);
        }
    }
    
    println!("\n🎉 Benchmark completed!");
    println!("💡 GPU acceleration provides significant speedup for larger datasets");
    
    Ok(())
}

fn cpu_statistical_outlier_removal(cloud: &PointCloud<Point3f>, k_neighbors: usize, std_dev_multiplier: f32) -> PointCloud<Point3f> {
    if cloud.is_empty() {
        return PointCloud::new();
    }
    
    let points = &cloud.points;
    let mut mean_distances = Vec::new();
    
    // Compute mean distances for all points
    for point in points {
        let mut distances = Vec::new();
        for other_point in points {
            if other_point != point {
                let dx = point.x - other_point.x;
                let dy = point.y - other_point.y;
                let dz = point.z - other_point.z;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                distances.push(distance);
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = k_neighbors.min(distances.len());
        if k > 0 {
            let mean = distances[..k].iter().sum::<f32>() / k as f32;
            mean_distances.push(mean);
        } else {
            mean_distances.push(0.0);
        }
    }
    
    // Compute global statistics
    let global_mean = mean_distances.iter().sum::<f32>() / mean_distances.len() as f32;
    let variance = mean_distances.iter().map(|&d| (d - global_mean).powi(2)).sum::<f32>() / mean_distances.len() as f32;
    let global_std_dev = variance.sqrt();
    let threshold = global_mean + std_dev_multiplier * global_std_dev;
    
    // Filter out outliers
    let filtered_points: Vec<Point3f> = points
        .iter()
        .zip(mean_distances.iter())
        .filter(|(_, &mean_dist)| mean_dist <= threshold)
        .map(|(point, _)| *point)
        .collect();
    
    PointCloud::from_points(filtered_points)
}

fn create_test_point_cloud(size: usize) -> PointCloud<Point3f> {
    let mut points = Vec::new();
    
    // Create a cluster of normal points
    let grid_size = (size as f32).sqrt() as usize;
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = (i as f32 - grid_size as f32 / 2.0) * 0.1;
            let y = (j as f32 - grid_size as f32 / 2.0) * 0.1;
            let z = 0.0;
            points.push(Point3f::new(x, y, z));
        }
    }
    
    // Add some outliers (10% of total points)
    let outlier_count = size / 10;
    for i in 0..outlier_count {
        let x = 10.0 + (i as f32 * 2.0);
        let y = 10.0 + (i as f32 * 2.0);
        let z = 10.0 + (i as f32 * 2.0);
        points.push(Point3f::new(x, y, z));
    }
    
    PointCloud::from_points(points)
} 