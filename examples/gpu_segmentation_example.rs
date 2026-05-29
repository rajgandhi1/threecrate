//! GPU-accelerated RANSAC plane segmentation and Euclidean clustering.

use threecrate_core::{Point3f, PointCloud};
use threecrate_gpu::{
    gpu_extract_clusters, gpu_segment_plane, GpuClusterConfig, GpuContext,
    GpuPlaneSegmentationConfig,
};

#[tokio::main]
async fn main() -> threecrate_core::Result<()> {
    println!("GPU Segmentation Example");
    println!("========================");

    let gpu = GpuContext::new().await?;

    let plane_cloud = create_plane_with_outliers();
    let plane_config = GpuPlaneSegmentationConfig {
        max_iterations: 512,
        distance_threshold: 0.02,
        min_inliers: 100,
    };
    let plane = gpu_segment_plane(&gpu, &plane_cloud, plane_config).await?;
    println!(
        "Plane inliers: {} / {}",
        plane.inliers.len(),
        plane_cloud.len()
    );
    println!("Plane coefficients: {:?}", plane.plane.coefficients);

    let cluster_cloud = create_clustered_cloud();
    let config = GpuClusterConfig::with_max_neighbors(0.12, 10, 1_000, 32);
    let clusters = gpu_extract_clusters(&gpu, &cluster_cloud, config).await?;
    println!("Clusters found: {}", clusters.len());
    for (idx, cluster) in clusters.iter().enumerate() {
        println!("  Cluster {}: {} points", idx + 1, cluster.len());
    }

    Ok(())
}

fn create_plane_with_outliers() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();

    for x in 0..24 {
        for y in 0..24 {
            let z = ((x * 17 + y * 31) % 7) as f32 * 0.001;
            cloud.push(Point3f::new(x as f32 * 0.05, y as f32 * 0.05, z));
        }
    }

    for i in 0..40 {
        cloud.push(Point3f::new(
            (i % 10) as f32 * 0.12,
            (i / 10) as f32 * 0.12,
            1.5 + (i % 5) as f32 * 0.2,
        ));
    }

    cloud
}

fn create_clustered_cloud() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    add_grid_cluster(&mut cloud, Point3f::new(0.0, 0.0, 0.0), 40);
    add_grid_cluster(&mut cloud, Point3f::new(3.0, 0.0, 0.0), 30);
    add_grid_cluster(&mut cloud, Point3f::new(0.0, 3.0, 0.0), 20);
    cloud
}

fn add_grid_cluster(cloud: &mut PointCloud<Point3f>, center: Point3f, count: usize) {
    for i in 0..count {
        let x = center.x + (i % 5) as f32 * 0.04;
        let y = center.y + ((i / 5) % 5) as f32 * 0.04;
        let z = center.z + (i / 25) as f32 * 0.04;
        cloud.push(Point3f::new(x, y, z));
    }
}
