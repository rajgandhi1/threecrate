//! GPU-accelerated nearest neighbor search

use crate::spatial::{vec4s, GpuKdTree, KD_TREE_WGSL, MAX_K};
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use threecrate_core::{Point3f, Result};

/// Parameters for nearest neighbor search
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NearestNeighborParams {
    pub num_points: u32,
    pub k_neighbors: u32,
    pub max_distance: f32,
    pub _padding: u32,
}

/// GPU representation of a point for nearest neighbor search
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(align(16))]
pub struct GpuPoint {
    pub position: [f32; 3],
    pub _padding: f32,
}

/// Result of nearest neighbor search
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NeighborResult {
    pub index: u32,
    pub distance: f32,
    pub _padding: [u32; 2],
}

/// k-NN kernel: one invocation per query point, walking the GPU kd-tree.
fn knn_shader() -> String {
    format!(
        "{}{}",
        KD_TREE_WGSL,
        r#"
struct KnnParams {
    num_queries: u32,
    k: u32,
    limit_sq: f32,
    _pad: u32,
}

@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
// k slots per query: (original point index, distance bits). Unused slots have
// index NO_NODE.
@group(0) @binding(2) var<storage, read_write> results: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: KnnParams;

@compute @workgroup_size(64)
fn knn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= params.num_queries) {
        return;
    }
    find_k_nearest(queries[q].xyz, params.k, params.limit_sq);
    let base = q * params.k;
    for (var i = 0u; i < params.k; i++) {
        if (i < knn_count) {
            results[base + i] = vec2<u32>(nodes[knn_node[i]].index, bitcast<u32>(sqrt(knn_dist[i])));
        } else {
            results[base + i] = vec2<u32>(NO_NODE, 0u);
        }
    }
}
"#
    )
}

impl GpuContext {
    /// GPU-accelerated k-nearest neighbor search.
    ///
    /// Builds a kd-tree over `points` once, uploads it, and answers every query
    /// on the GPU by walking the tree. Returns, for each query, up to `k`
    /// `(point index, distance)` pairs closer than `max_distance`, closest
    /// first. `k` is clamped to 1..=32.
    pub async fn find_k_nearest_neighbors(
        &self,
        points: &[Point3f],
        query_points: &[Point3f],
        k: usize,
        max_distance: f32,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        if points.is_empty() || query_points.is_empty() {
            return Ok(vec![Vec::new(); query_points.len()]);
        }
        let k = k.clamp(1, MAX_K);

        let tree = GpuKdTree::new(self, points)?;
        if tree.is_empty() {
            return Ok(vec![Vec::new(); query_points.len()]);
        }

        let queries = self.create_buffer_init(
            "kNN Queries",
            &vec4s(&tree.frame.points_to_local(query_points)),
            wgpu::BufferUsages::STORAGE,
        );
        let results = self.create_buffer(
            "kNN Results",
            (query_points.len() * k * std::mem::size_of::<[u32; 2]>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params = KnnParams {
            num_queries: query_points.len() as u32,
            k: k as u32,
            // Negative means nothing matches, as with the old brute-force
            // search (squaring it would turn it into a valid range).
            limit_sq: if max_distance < 0.0 {
                0.0
            } else {
                max_distance * max_distance
            },
            _pad: 0,
        };
        let params_buffer =
            self.create_buffer_init("kNN Params", &[params], wgpu::BufferUsages::UNIFORM);

        let pipeline = self.cached_pipeline(&self.pipelines.knn, "kNN", knn_shader, "knn");
        let bind_group = self.create_bind_group(
            "kNN",
            &pipeline.get_bind_group_layout(0),
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tree.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: queries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kNN") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kNN"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(query_points.len().div_ceil(64) as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        let raw: Vec<[u32; 2]> = self.read_buffer(&results)?;
        Ok(raw
            .chunks_exact(k)
            .map(|slots| {
                slots
                    .iter()
                    .take_while(|slot| slot[0] != u32::MAX)
                    .map(|slot| (slot[0] as usize, f32::from_bits(slot[1])))
                    .collect()
            })
            .collect())
    }
}

/// Uniform block of the k-NN kernel (`KnnParams` in WGSL).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct KnnParams {
    num_queries: u32,
    k: u32,
    limit_sq: f32,
    _pad: u32,
}

/// GPU-accelerated nearest neighbor search for single query point
pub async fn gpu_find_k_nearest(
    gpu_context: &GpuContext,
    points: &[Point3f],
    query: &Point3f,
    k: usize,
) -> Result<Vec<(usize, f32)>> {
    let results = gpu_context
        .find_k_nearest_neighbors(points, &[*query], k, f32::INFINITY)
        .await?;
    Ok(results.into_iter().next().unwrap_or_default())
}

/// GPU-accelerated nearest neighbor search for multiple query points
pub async fn gpu_find_k_nearest_batch(
    gpu_context: &GpuContext,
    points: &[Point3f],
    query_points: &[Point3f],
    k: usize,
) -> Result<Vec<Vec<(usize, f32)>>> {
    gpu_context
        .find_k_nearest_neighbors(points, query_points, k, f32::INFINITY)
        .await
}

/// GPU-accelerated radius-based nearest neighbor search
pub async fn gpu_find_radius_neighbors(
    gpu_context: &GpuContext,
    points: &[Point3f],
    query: &Point3f,
    radius: f32,
) -> Result<Vec<(usize, f32)>> {
    let results = gpu_context
        .find_k_nearest_neighbors(points, &[*query], 32, radius)
        .await?;
    Ok(results.into_iter().next().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuContext;
    use approx::assert_relative_eq;
    use threecrate_core::Point3f;

    /// Try to create a GPU context, return None if not available
    async fn try_create_gpu_context() -> Option<GpuContext> {
        match GpuContext::new().await {
            Ok(gpu) => Some(gpu),
            Err(_) => {
                println!("⚠️  GPU not available, skipping GPU-dependent test");
                None
            }
        }
    }

    /// Create a simple test point cloud
    fn create_test_points() -> Vec<Point3f> {
        vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0),
            Point3f::new(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn test_gpu_nearest_neighbor_single() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let query = Point3f::new(0.1, 0.1, 0.1);

            let neighbors = gpu_find_k_nearest(&gpu, &points, &query, 3).await.unwrap();

            assert_eq!(neighbors.len(), 3);
            assert_eq!(neighbors[0].0, 0); // Closest should be origin
            assert!(neighbors[0].1 < 0.2); // Distance should be small

            println!("✓ GPU single nearest neighbor test passed");
        });
    }

    #[test]
    fn test_gpu_nearest_neighbor_batch() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let queries = vec![Point3f::new(0.1, 0.1, 0.1), Point3f::new(0.9, 0.1, 0.1)];

            let results = gpu_find_k_nearest_batch(&gpu, &points, &queries, 2)
                .await
                .unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].len(), 2);
            assert_eq!(results[1].len(), 2);

            // First query should find origin as closest
            assert_eq!(results[0][0].0, 0);

            // Second query should find (1,0,0) as closest
            assert_eq!(results[1][0].0, 1);

            println!("✓ GPU batch nearest neighbor test passed");
        });
    }

    #[test]
    fn test_gpu_radius_neighbors() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let query = Point3f::new(0.0, 0.0, 0.0);
            let radius = 1.5;

            let neighbors = gpu_find_radius_neighbors(&gpu, &points, &query, radius)
                .await
                .unwrap();

            // Should find points within radius
            assert!(!neighbors.is_empty());

            // All distances should be within radius
            for (_, distance) in &neighbors {
                assert!(*distance <= radius);
            }

            println!(
                "✓ GPU radius neighbors test passed: {} neighbors found",
                neighbors.len()
            );
        });
    }

    #[test]
    fn test_gpu_knn_matches_cpu_kd_tree() {
        use threecrate_algorithms::KdTree;
        use threecrate_core::NearestNeighborSearch;

        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            // Deterministic pseudo-random cloud, offset from the origin
            let mut state = 12345u32;
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32 / (1u32 << 24) as f32
            };
            let points: Vec<Point3f> = (0..20_000)
                .map(|_| Point3f::new(100.0 + 20.0 * next(), -50.0 + 20.0 * next(), 5.0 * next()))
                .collect();
            let queries = &points[..2_000];
            let k = 10;

            let gpu_results = gpu_find_k_nearest_batch(&gpu, &points, queries, k)
                .await
                .unwrap();
            let tree = KdTree::new(&points).unwrap();

            for (query, gpu_neighbors) in queries.iter().zip(&gpu_results) {
                let cpu_neighbors = tree.find_k_nearest(query, k);
                assert_eq!(gpu_neighbors.len(), k);
                for (g, c) in gpu_neighbors.iter().zip(&cpu_neighbors) {
                    assert!((g.1 - c.1).abs() < 1e-3, "gpu {:?} vs cpu {:?}", g, c);
                }
            }
        });
    }

    #[test]
    fn test_gpu_negative_max_distance_matches_nothing() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };
            let points = create_test_points();
            let results = gpu
                .find_k_nearest_neighbors(&points, &[Point3f::origin()], 3, -1.0)
                .await
                .unwrap();
            assert!(results[0].is_empty());
        });
    }

    #[test]
    fn test_gpu_nearest_neighbor_accuracy() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let query = Point3f::new(0.5, 0.5, 0.5);

            let neighbors = gpu_find_k_nearest(&gpu, &points, &query, 1).await.unwrap();

            assert_eq!(neighbors.len(), 1);

            // Manually verify the nearest neighbor
            let min_dist = points
                .iter()
                .map(|p| (query - *p).magnitude())
                .fold(f32::INFINITY, f32::min);

            // Every point here is equally far from the query, so any of them is
            // a correct answer; check the returned one is at the minimum.
            let (index, distance) = neighbors[0];
            assert_relative_eq!(
                (query - points[index]).magnitude(),
                min_dist,
                epsilon = 0.001
            );
            assert_relative_eq!(distance, min_dist, epsilon = 0.001);

            println!("✓ GPU nearest neighbor accuracy test passed");
        });
    }
}
