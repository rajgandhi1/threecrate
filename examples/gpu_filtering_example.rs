use threecrate_core::{PointCloud, Point3f};
use threecrate_gpu::GpuContext;
use threecrate_algorithms::{statistical_outlier_removal, radius_outlier_removal, voxel_grid_filter};
use threecrate_gpu::filtering::{
    gpu_remove_statistical_outliers,
    gpu_radius_outlier_removal,
    gpu_voxel_grid_filter,
};
use std::time::Instant;

fn create_test_point_cloud_with_outliers() -> PointCloud<Point3f> {
    let mut points = Vec::new();
    
    // Create a dense cluster
    for i in 0..30 {
        for j in 0..30 {
            let x = (i as f32 - 15.0) * 0.1;
            let y = (j as f32 - 15.0) * 0.1;
            let z = 0.0;
            points.push(Point3f::new(x, y, z));
        }
    }
    
    // Add some outliers far from the cluster
    for i in 0..10 {
        let x = 50.0 + (i as f32 * 2.0);
        let y = 50.0 + (i as f32 * 2.0);
        let z = 50.0 + (i as f32 * 2.0);
        points.push(Point3f::new(x, y, z));
    }
    
    PointCloud::from_points(points)
}

fn create_test_point_cloud_for_voxel_grid() -> PointCloud<Point3f> {
    let mut points = Vec::new();
    
    // Create a dense grid of points
    for i in 0..20 {
        for j in 0..20 {
            for k in 0..10 {
                let x = (i as f32 - 10.0) * 0.1;
                let y = (j as f32 - 10.0) * 0.1;
                let z = (k as f32 - 5.0) * 0.1;
                points.push(Point3f::new(x, y, z));
            }
        }
    }
    
    // Add some duplicate points in the same voxels
    for i in 0..10 {
        for j in 0..10 {
            let x = (i as f32 - 5.0) * 0.1;
            let y = (j as f32 - 5.0) * 0.1;
            let z = 0.0;
            points.push(Point3f::new(x, y, z));
            points.push(Point3f::new(x, y, z)); // Duplicate
        }
    }
    
    PointCloud::from_points(points)
}

#[tokio::main]
async fn main() -> threecrate_core::Result<()> {
    println!("=== GPU-Accelerated Filtering Example ===\n");

    // Create GPU context
    let gpu_context = match GpuContext::new().await {
        Ok(ctx) => {
            println!("✓ GPU context created successfully");
            ctx
        }
        Err(e) => {
            println!("✗ Failed to create GPU context: {}", e);
            println!("Falling back to CPU-only mode");
            return run_cpu_only_example().await;
        }
    };

    // Test 1: Statistical Outlier Removal
    println!("\n--- Statistical Outlier Removal ---");
    let cloud = create_test_point_cloud_with_outliers();
    println!("Original point cloud: {} points", cloud.len());

    // CPU version
    let cpu_start = Instant::now();
    let cpu_filtered = statistical_outlier_removal(&cloud, 10, 1.0)?;
    let cpu_time = cpu_start.elapsed();

    // GPU version
    let gpu_start = Instant::now();
    let gpu_filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0).await?;
    let gpu_time = gpu_start.elapsed();

    println!("CPU filtered: {} points in {:?}", cpu_filtered.len(), cpu_time);
    println!("GPU filtered: {} points in {:?}", gpu_filtered.len(), gpu_time);
    
    if gpu_time.as_secs_f32() > 0.0 {
        let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
        println!("GPU speedup: {:.2}x", speedup);
    }

    // Test 2: Radius Outlier Removal
    println!("\n--- Radius Outlier Removal ---");
    let cloud = create_test_point_cloud_with_outliers();
    println!("Original point cloud: {} points", cloud.len());

    // CPU version
    let cpu_start = Instant::now();
    let cpu_filtered = radius_outlier_removal(&cloud, 0.5, 3)?;
    let cpu_time = cpu_start.elapsed();

    // GPU version
    let gpu_start = Instant::now();
    let gpu_filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 3).await?;
    let gpu_time = gpu_start.elapsed();

    println!("CPU filtered: {} points in {:?}", cpu_filtered.len(), cpu_time);
    println!("GPU filtered: {} points in {:?}", gpu_filtered.len(), gpu_time);
    
    if gpu_time.as_secs_f32() > 0.0 {
        let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
        println!("GPU speedup: {:.2}x", speedup);
    }

    // Test 3: Voxel Grid Filtering
    println!("\n--- Voxel Grid Filtering ---");
    let cloud = create_test_point_cloud_for_voxel_grid();
    println!("Original point cloud: {} points", cloud.len());

    // CPU version
    let cpu_start = Instant::now();
    let cpu_filtered = voxel_grid_filter(&cloud, 0.1)?;
    let cpu_time = cpu_start.elapsed();

    // GPU version
    let gpu_start = Instant::now();
    let gpu_filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1).await?;
    let gpu_time = gpu_start.elapsed();

    println!("CPU filtered: {} points in {:?}", cpu_filtered.len(), cpu_time);
    println!("GPU filtered: {} points in {:?}", gpu_filtered.len(), gpu_time);
    
    if gpu_time.as_secs_f32() > 0.0 {
        let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
        println!("GPU speedup: {:.2}x", speedup);
    }

    // Test 4: Parameter Comparison
    println!("\n--- Parameter Comparison ---");
    let cloud = create_test_point_cloud_with_outliers();
    
    // Test different radius values for radius outlier removal
    println!("Radius outlier removal with different parameters:");
    for radius in [0.2, 0.5, 1.0] {
        let gpu_start = Instant::now();
        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, radius, 3).await?;
        let gpu_time = gpu_start.elapsed();
        println!("  radius={}: {} -> {} points in {:?}", radius, cloud.len(), filtered.len(), gpu_time);
    }

    // Test different voxel sizes for voxel grid filtering
    println!("Voxel grid filtering with different parameters:");
    let cloud = create_test_point_cloud_for_voxel_grid();
    for voxel_size in [0.05, 0.1, 0.2] {
        let gpu_start = Instant::now();
        let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, voxel_size).await?;
        let gpu_time = gpu_start.elapsed();
        println!("  voxel_size={}: {} -> {} points in {:?}", voxel_size, cloud.len(), filtered.len(), gpu_time);
    }

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

async fn run_cpu_only_example() -> threecrate_core::Result<()> {
    println!("\n--- CPU-Only Filtering Example ---");
    
    let cloud = create_test_point_cloud_with_outliers();
    println!("Original point cloud: {} points", cloud.len());

    // Statistical outlier removal
    let start = Instant::now();
    let filtered = statistical_outlier_removal(&cloud, 10, 1.0)?;
    let time = start.elapsed();
    println!("Statistical outlier removal: {} -> {} points in {:?}", 
             cloud.len(), filtered.len(), time);

    // Radius outlier removal
    let start = Instant::now();
    let filtered = radius_outlier_removal(&cloud, 0.5, 3)?;
    let time = start.elapsed();
    println!("Radius outlier removal: {} -> {} points in {:?}", 
             cloud.len(), filtered.len(), time);

    // Voxel grid filtering
    let cloud = create_test_point_cloud_for_voxel_grid();
    let start = Instant::now();
    let filtered = voxel_grid_filter(&cloud, 0.1)?;
    let time = start.elapsed();
    println!("Voxel grid filtering: {} -> {} points in {:?}", 
             cloud.len(), filtered.len(), time);

    println!("\n=== CPU-only example completed! ===");
    Ok(())
} 