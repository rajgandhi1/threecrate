//! GPU-accelerated filtering

use crate::device::workgroup_grid;
use crate::spatial::{GpuKdTree, KD_TREE_WGSL};
use crate::GpuContext;
use threecrate_core::{Point3f, PointCloud, Result};

/// Radius outlier kernel: one thread per kd-tree node. Counts the other points
/// within `radius` with the shared tree walk, stopping as soon as it has
/// `min_neighbors`, since the point is then known to stay.
fn radius_outlier_shader() -> String {
    format!(
        "{}{}",
        KD_TREE_WGSL,
        r#"
struct RadiusOutlierParams {
    num_nodes: u32,
    radius_sq: f32,
    min_neighbors: u32,
    _padding: u32,
}

@group(0) @binding(1) var<storage, read_write> is_outlier: array<u32>;
@group(0) @binding(2) var<uniform> params: RadiusOutlierParams;

@compute @workgroup_size(64)
fn radius_outlier(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let node_id = gid.x + gid.y * nwg.x * 64u;
    if (node_id >= params.num_nodes) {
        return;
    }
    let count = count_within(
        nodes[node_id].pos,
        params.radius_sq,
        node_id,
        params.min_neighbors,
    );
    is_outlier[nodes[node_id].index] = select(0u, 1u, count < params.min_neighbors);
}
"#
    )
}

impl GpuContext {
    /// Remove statistical outliers. Runs on the CPU (`threecrate_algorithms`):
    /// the neighbour search dominates and already ran there, and the old GPU
    /// kernel read the wrong neighbours for any k other than 64 (#216).
    pub async fn remove_statistical_outliers(
        &self,
        points: &[Point3f],
        k_neighbors: usize,
        std_dev_multiplier: f32,
    ) -> Result<Vec<Point3f>> {
        let cloud = PointCloud::from_points(points.to_vec());
        let kept = threecrate_algorithms::statistical_outlier_removal(
            &cloud,
            k_neighbors,
            std_dev_multiplier,
        )?;
        Ok(kept.points)
    }

    /// Remove radius outliers from point cloud using GPU acceleration.
    ///
    /// A point is kept when at least `min_neighbors` other points lie within
    /// `radius` of it. Points with a NaN or infinite coordinate have no
    /// neighbors, so they are removed whenever `min_neighbors > 0`.
    pub async fn remove_radius_outliers(
        &self,
        points: &[Point3f],
        radius: f32,
        min_neighbors: usize,
    ) -> Result<Vec<Point3f>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let tree = GpuKdTree::new(self, points)?;
        // Every point starts flagged as an outlier unless no neighbors are
        // required; the kernel then decides for each finite point.
        let initial_flag = (min_neighbors > 0) as u32;
        let outlier_buffer = self.create_buffer_init(
            "Outlier Flags",
            &vec![initial_flag; points.len()],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        if !tree.is_empty() {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct RadiusOutlierParams {
                num_nodes: u32,
                radius_sq: f32,
                min_neighbors: u32,
                _padding: u32,
            }
            let params = RadiusOutlierParams {
                num_nodes: tree.node_to_index.len() as u32,
                // A negative radius matches nothing (as the old brute-force
                // kernel did); squaring it would make it a valid range.
                radius_sq: if radius < 0.0 { -1.0 } else { radius * radius },
                min_neighbors: min_neighbors.min(u32::MAX as usize) as u32,
                _padding: 0,
            };
            let params_buffer = self.create_buffer_init(
                "Radius Outlier Params",
                &[params],
                wgpu::BufferUsages::UNIFORM,
            );

            let pipeline = self.cached_pipeline(
                &self.pipelines.radius_outlier,
                "Radius Outlier Filter",
                radius_outlier_shader,
                "radius_outlier",
            );
            let bind_group = self.create_bind_group(
                "Radius Outlier Filter",
                &pipeline.get_bind_group_layout(0),
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tree.nodes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: outlier_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Radius Outlier Filter"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Radius Outlier Filter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let (x, y) = workgroup_grid(tree.node_to_index.len().div_ceil(64));
                pass.dispatch_workgroups(x, y, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        let outlier_flags: Vec<u32> = self.read_buffer(&outlier_buffer)?;
        Ok(points
            .iter()
            .zip(&outlier_flags)
            .filter(|&(_, &is_outlier)| is_outlier == 0)
            .map(|(point, _)| *point)
            .collect())
    }

    /// Voxel grid filtering. Runs on the CPU (`threecrate_algorithms`), which
    /// is exact and already fast; the old GPU version hashed voxels into a
    /// fixed table and dropped most points (#216).
    pub async fn voxel_grid_filter(
        &self,
        points: &[Point3f],
        voxel_size: f32,
    ) -> Result<Vec<Point3f>> {
        let cloud = PointCloud::from_points(points.to_vec());
        Ok(threecrate_algorithms::voxel_grid_filter(&cloud, voxel_size)?.points)
    }
}

/// GPU-accelerated statistical outlier removal
pub async fn gpu_remove_statistical_outliers(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    k_neighbors: usize,
    std_dev_multiplier: f32,
) -> Result<PointCloud<Point3f>> {
    let filtered_points = gpu_context
        .remove_statistical_outliers(&cloud.points, k_neighbors, std_dev_multiplier)
        .await?;
    Ok(PointCloud::from_points(filtered_points))
}

/// GPU-accelerated radius outlier removal
pub async fn gpu_radius_outlier_removal(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    radius: f32,
    min_neighbors: usize,
) -> Result<PointCloud<Point3f>> {
    let filtered_points = gpu_context
        .remove_radius_outliers(&cloud.points, radius, min_neighbors)
        .await?;
    Ok(PointCloud::from_points(filtered_points))
}

/// GPU-accelerated voxel grid filtering
pub async fn gpu_voxel_grid_filter(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    voxel_size: f32,
) -> Result<PointCloud<Point3f>> {
    let filtered_points = gpu_context
        .voxel_grid_filter(&cloud.points, voxel_size)
        .await?;
    Ok(PointCloud::from_points(filtered_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use threecrate_core::{Point3f, PointCloud};

    async fn try_create_gpu_context() -> Option<GpuContext> {
        GpuContext::new().await.ok()
    }

    fn create_test_point_cloud_with_outliers() -> PointCloud<Point3f> {
        let mut points = Vec::new();

        // Create a cluster of normal points
        for i in 0..50 {
            for j in 0..50 {
                let x = (i as f32 - 25.0) * 0.1;
                let y = (j as f32 - 25.0) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
            }
        }

        // Add some outliers far from the cluster
        for i in 0..20 {
            let x = 50.0 + (i as f32 * 2.0);
            let y = 50.0 + (i as f32 * 2.0);
            let z = 50.0 + (i as f32 * 2.0);
            points.push(Point3f::new(x, y, z));
        }

        PointCloud::from_points(points)
    }

    fn create_test_point_cloud_no_outliers() -> PointCloud<Point3f> {
        let mut points = Vec::new();

        // Create a uniform grid of points
        for i in 0..25 {
            for j in 0..25 {
                for k in 0..5 {
                    let x = (i as f32 - 12.5) * 0.1;
                    let y = (j as f32 - 12.5) * 0.1;
                    let z = (k as f32 - 2.5) * 0.1;
                    points.push(Point3f::new(x, y, z));
                }
            }
        }

        PointCloud::from_points(points)
    }

    /// The two cases from #216: the old kernels kept 20 of 400 grid points
    /// (k = 8) and 7 of 100 points in a line. Results must match the CPU.
    #[tokio::test]
    async fn test_gpu_filters_match_cpu_on_216_cases() {
        let Some(gpu_context) = try_create_gpu_context().await else {
            println!("Skipping GPU test - no GPU available");
            return;
        };
        let mut grid: Vec<Point3f> = (0..400)
            .map(|i| Point3f::new((i % 20) as f32, (i / 20) as f32, 0.0))
            .collect();
        grid.push(Point3f::new(100.0, 100.0, 100.0));
        let grid = PointCloud::from_points(grid);
        for k in [8, 10, 64] {
            let gpu = gpu_remove_statistical_outliers(&gpu_context, &grid, k, 1.0)
                .await
                .unwrap();
            let cpu = threecrate_algorithms::statistical_outlier_removal(&grid, k, 1.0).unwrap();
            assert_eq!(gpu.len(), cpu.len(), "k = {k}");
            assert_eq!(gpu.len(), 400, "k = {k}: only the far point should go");
        }

        let line =
            PointCloud::from_points((0..100).map(|x| Point3f::new(x as f32, 0.0, 0.0)).collect());
        let kept = gpu_voxel_grid_filter(&gpu_context, &line, 0.5)
            .await
            .unwrap();
        assert_eq!(kept.len(), 100);
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_basic() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_with_outliers();
        let original_count = cloud.len();

        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0)
            .await
            .unwrap();
        let filtered_count = filtered.len();

        // Should remove some outliers
        assert!(filtered_count < original_count);
        assert!(filtered_count > 0);

        println!(
            "GPU outlier removal: {} -> {} points",
            original_count, filtered_count
        );
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_no_outliers() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_no_outliers();
        let original_count = cloud.len();

        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0)
            .await
            .unwrap();
        let filtered_count = filtered.len();

        println!(
            "GPU outlier removal (no outliers): {} -> {} points",
            original_count, filtered_count
        );

        // The algorithm should remove some points but not all
        // Even with no obvious outliers, the algorithm may remove points based on statistical analysis
        assert!(filtered_count > 0);
        assert!(filtered_count < original_count);
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_empty_cloud() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::<Point3f>::new();
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0)
            .await
            .unwrap();

        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_single_point() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 1, 1.0)
            .await
            .unwrap();

        // Should keep the single point
        assert_eq!(filtered.len(), 1);
    }

    #[tokio::test]
    async fn test_gpu_vs_cpu_statistical_outlier_removal() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU vs CPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_with_outliers();

        // CPU version - use the same algorithm logic but avoid type conflicts
        let cpu_start = Instant::now();
        let cpu_filtered_count = {
            // Simple CPU implementation for comparison
            let points = &cloud.points;
            let k = 10;
            let std_dev_multiplier = 1.0;

            if points.is_empty() {
                0
            } else {
                // Simple distance-based outlier detection
                let mut mean_distances = Vec::new();
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
                    let k_neighbors = k.min(distances.len());
                    if k_neighbors > 0 {
                        let mean =
                            distances[..k_neighbors].iter().sum::<f32>() / k_neighbors as f32;
                        mean_distances.push(mean);
                    } else {
                        mean_distances.push(0.0);
                    }
                }

                let global_mean = mean_distances.iter().sum::<f32>() / mean_distances.len() as f32;
                let variance = mean_distances
                    .iter()
                    .map(|&d| (d - global_mean).powi(2))
                    .sum::<f32>()
                    / mean_distances.len() as f32;
                let global_std_dev = variance.sqrt();
                let threshold = global_mean + std_dev_multiplier * global_std_dev;

                mean_distances.iter().filter(|&&d| d <= threshold).count()
            }
        };
        let cpu_time = cpu_start.elapsed();

        // GPU version
        let gpu_start = Instant::now();
        let gpu_filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, 1.0)
            .await
            .unwrap();
        let gpu_time = gpu_start.elapsed();

        println!(
            "CPU outlier removal: {} -> {} points in {:?}",
            cloud.len(),
            cpu_filtered_count,
            cpu_time
        );
        println!(
            "GPU outlier removal: {} -> {} points in {:?}",
            cloud.len(),
            gpu_filtered.len(),
            gpu_time
        );

        // Both should remove outliers (exact counts may vary due to different algorithms)
        assert!(cpu_filtered_count < cloud.len());
        assert!(gpu_filtered.len() < cloud.len());

        // Performance comparison (GPU should be faster for larger datasets)
        if cloud.len() > 1000 {
            println!(
                "GPU speedup: {:.2}x",
                cpu_time.as_secs_f32() / gpu_time.as_secs_f32()
            );
        }
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_different_parameters() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_with_outliers();
        let original_count = cloud.len();

        // Test different k values
        for k in [5, 10, 20] {
            let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, k, 1.0)
                .await
                .unwrap();
            println!("k={}: {} -> {} points", k, original_count, filtered.len());
            assert!(filtered.len() > 0);
        }

        // Test different std_dev_multiplier values
        for std_dev in [0.5, 1.0, 2.0] {
            let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 10, std_dev)
                .await
                .unwrap();
            println!(
                "std_dev={}: {} -> {} points",
                std_dev,
                original_count,
                filtered.len()
            );
            assert!(filtered.len() > 0);
        }
    }

    #[tokio::test]
    async fn test_gpu_statistical_outlier_removal_large_dataset() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        // Create a smaller dataset for faster testing
        let mut points = Vec::new();
        for i in 0..50 {
            for j in 0..50 {
                let x = (i as f32 - 25.0) * 0.1;
                let y = (j as f32 - 25.0) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
            }
        }

        // Add some outliers
        for i in 0..10 {
            let x = 25.0 + (i as f32 * 2.0);
            let y = 25.0 + (i as f32 * 2.0);
            let z = 25.0 + (i as f32 * 2.0);
            points.push(Point3f::new(x, y, z));
        }

        let cloud = PointCloud::from_points(points);
        let original_count = cloud.len();

        let start = Instant::now();
        let filtered = gpu_remove_statistical_outliers(&gpu_context, &cloud, 5, 1.0)
            .await
            .unwrap();
        let elapsed = start.elapsed();

        println!(
            "Large dataset GPU outlier removal: {} -> {} points in {:?}",
            original_count,
            filtered.len(),
            elapsed
        );

        assert!(filtered.len() < original_count);
        assert!(filtered.len() > 0);

        // Should complete in reasonable time
        assert!(elapsed.as_secs() < 10);
    }

    fn create_test_point_cloud_for_radius_outlier() -> PointCloud<Point3f> {
        let mut points = Vec::new();

        // Create a dense cluster
        for i in 0..20 {
            for j in 0..20 {
                let x = (i as f32 - 10.0) * 0.1;
                let y = (j as f32 - 10.0) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
            }
        }

        // Add some isolated points (outliers)
        for i in 0..5 {
            let x = 10.0 + (i as f32 * 2.0);
            let y = 10.0 + (i as f32 * 2.0);
            let z = 10.0 + (i as f32 * 2.0);
            points.push(Point3f::new(x, y, z));
        }

        PointCloud::from_points(points)
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_matches_cpu() {
        let Some(gpu_context) = try_create_gpu_context().await else {
            return;
        };

        // Uneven density: a dense patch, a sparse patch, and scattered points
        let mut state = 7u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut cloud = PointCloud::new();
        for _ in 0..15_000 {
            cloud.push(Point3f::new(next(), next(), 0.1 * next()));
        }
        for _ in 0..3_000 {
            cloud.push(Point3f::new(2.0 + 3.0 * next(), 3.0 * next(), next()));
        }
        cloud.push(Point3f::new(f32::NAN, 0.0, 0.0));

        let gpu = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.05, 4)
            .await
            .unwrap();
        let finite = PointCloud::from_points(
            cloud
                .points
                .iter()
                .copied()
                .filter(|p| p.coords.iter().all(|c| c.is_finite()))
                .collect(),
        );
        let cpu = threecrate_algorithms::radius_outlier_removal(&finite, 0.05, 4).unwrap();

        assert_eq!(gpu.len(), cpu.len());
        assert!(gpu.points.iter().zip(&cpu.points).all(|(g, c)| g == c));
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_negative_radius() {
        let Some(gpu_context) = try_create_gpu_context().await else {
            return;
        };
        let cloud = PointCloud::from_points(
            (0..50)
                .map(|i| Point3f::new(i as f32 * 0.01, 0.0, 0.0))
                .collect(),
        );
        // A negative radius matches nothing, so every point is an outlier
        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, -0.2, 3)
            .await
            .unwrap();
        assert!(filtered.is_empty());
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_basic() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_radius_outlier();
        let original_count = cloud.len();

        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 3)
            .await
            .unwrap();
        let filtered_count = filtered.len();

        // Should remove some outliers
        assert!(filtered_count < original_count);
        assert!(filtered_count > 0);

        println!(
            "GPU radius outlier removal: {} -> {} points",
            original_count, filtered_count
        );
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_empty_cloud() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::<Point3f>::new();
        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 3)
            .await
            .unwrap();

        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_single_point() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 1)
            .await
            .unwrap();

        // Should remove the single point since it has no neighbors
        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_radius_outlier_removal_different_parameters() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_radius_outlier();
        let original_count = cloud.len();

        // Test different radius values
        for radius in [0.2, 0.5, 1.0] {
            let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, radius, 3)
                .await
                .unwrap();
            println!(
                "radius={}: {} -> {} points",
                radius,
                original_count,
                filtered.len()
            );
            assert!(filtered.len() > 0);
        }

        // Test different min_neighbors values
        for min_neighbors in [1, 3, 5] {
            let filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, min_neighbors)
                .await
                .unwrap();
            println!(
                "min_neighbors={}: {} -> {} points",
                min_neighbors,
                original_count,
                filtered.len()
            );
            assert!(filtered.len() > 0);
        }
    }

    fn create_test_point_cloud_for_voxel_grid() -> PointCloud<Point3f> {
        let mut points = Vec::new();

        // Create a dense grid of points
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..5 {
                    let x = (i as f32 - 5.0) * 0.1;
                    let y = (j as f32 - 5.0) * 0.1;
                    let z = (k as f32 - 2.5) * 0.1;
                    points.push(Point3f::new(x, y, z));
                }
            }
        }

        // Add some duplicate points in the same voxels
        for i in 0..5 {
            for j in 0..5 {
                let x = (i as f32 - 2.5) * 0.1;
                let y = (j as f32 - 2.5) * 0.1;
                let z = 0.0;
                points.push(Point3f::new(x, y, z));
                points.push(Point3f::new(x, y, z)); // Duplicate
            }
        }

        PointCloud::from_points(points)
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_basic() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_voxel_grid();
        let original_count = cloud.len();

        let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1)
            .await
            .unwrap();
        let filtered_count = filtered.len();

        // Should reduce the number of points by removing duplicates
        assert!(filtered_count < original_count);
        assert!(filtered_count > 0);

        println!(
            "GPU voxel grid filter: {} -> {} points",
            original_count, filtered_count
        );
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_empty_cloud() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::<Point3f>::new();
        let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1)
            .await
            .unwrap();

        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_single_point() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1)
            .await
            .unwrap();

        // Should keep the single point
        assert_eq!(filtered.len(), 1);
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_different_voxel_sizes() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_voxel_grid();
        let original_count = cloud.len();

        // Test different voxel sizes
        for voxel_size in [0.05, 0.1, 0.2] {
            let filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, voxel_size)
                .await
                .unwrap();
            println!(
                "voxel_size={}: {} -> {} points",
                voxel_size,
                original_count,
                filtered.len()
            );
            assert!(filtered.len() > 0);
            assert!(filtered.len() <= original_count);
        }
    }

    #[tokio::test]
    async fn test_gpu_voxel_grid_filter_invalid_voxel_size() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU test - no GPU available");
                return;
            }
        };

        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.0).await;
        assert!(result.is_err());

        let result = gpu_voxel_grid_filter(&gpu_context, &cloud, -1.0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_gpu_vs_cpu_performance_comparison() {
        let gpu_context = match try_create_gpu_context().await {
            Some(ctx) => ctx,
            None => {
                println!("Skipping GPU vs CPU test - no GPU available");
                return;
            }
        };

        let cloud = create_test_point_cloud_for_radius_outlier();

        // GPU radius outlier removal
        let gpu_start = Instant::now();
        let gpu_filtered = gpu_radius_outlier_removal(&gpu_context, &cloud, 0.5, 3)
            .await
            .unwrap();
        let gpu_time = gpu_start.elapsed();

        // GPU voxel grid filter
        let gpu_voxel_start = Instant::now();
        let gpu_voxel_filtered = gpu_voxel_grid_filter(&gpu_context, &cloud, 0.1)
            .await
            .unwrap();
        let gpu_voxel_time = gpu_voxel_start.elapsed();

        println!(
            "GPU radius outlier removal: {} -> {} points in {:?}",
            cloud.len(),
            gpu_filtered.len(),
            gpu_time
        );
        println!(
            "GPU voxel grid filter: {} -> {} points in {:?}",
            cloud.len(),
            gpu_voxel_filtered.len(),
            gpu_voxel_time
        );

        // Both should complete successfully
        assert!(gpu_filtered.len() > 0);
        assert!(gpu_voxel_filtered.len() > 0);

        // Should complete in reasonable time
        assert!(gpu_time.as_secs() < 10);
        assert!(gpu_voxel_time.as_secs() < 10);
    }
}
