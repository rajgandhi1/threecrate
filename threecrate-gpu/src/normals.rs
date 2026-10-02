//! GPU-accelerated normal estimation

use crate::spatial::{vec4, GpuKdTree, KD_TREE_WGSL, MAX_K};
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use threecrate_algorithms::KdTree;
use threecrate_core::{NormalPoint3f, Point3f, PointCloud, Result};

/// Normals kernel: per point, find its neighbors in the GPU kd-tree, build the
/// neighborhood covariance, and take the eigenvector of its smallest
/// eigenvalue. Matches the CPU `estimate_normals`: the k nearest neighbors
/// (not counting the point itself) plus the point.
fn normals_shader() -> String {
    format!(
        "{}{}",
        KD_TREE_WGSL,
        r#"
struct NormalParams {
    num_points: u32,
    k: u32,
    consistent_orientation: u32,
    _pad: u32,
    viewpoint: vec4<f32>,
}

@group(0) @binding(2) var<storage, read_write> output_normals: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: NormalParams;

// Eigenvector for the smallest eigenvalue of the symmetric matrix
// [[a00 a01 a02] [a01 a11 a12] [a02 a12 a22]], via the closed-form
// (trigonometric) eigenvalues and a cross product of two rows of A - lambda*I.
fn smallest_eigenvector(c00: f32, c01: f32, c02: f32, c11: f32, c12: f32, c22: f32) -> vec3<f32> {
    // Scale to unit size so f32 keeps its precision for tiny or huge clouds.
    let scale = max(max(max(abs(c00), abs(c01)), max(abs(c02), abs(c11))), max(abs(c12), abs(c22)));
    if (scale <= 0.0) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let a00 = c00 / scale;
    let a01 = c01 / scale;
    let a02 = c02 / scale;
    let a11 = c11 / scale;
    let a12 = c12 / scale;
    let a22 = c22 / scale;

    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    let q = (a00 + a11 + a22) / 3.0;
    let b00 = a00 - q;
    let b11 = a11 - q;
    let b22 = a22 - q;
    let p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
    if (p2 < 1e-12) {
        // All eigenvalues equal: no preferred direction.
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let p = sqrt(p2 / 6.0);
    let det_b = b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02) + a02 * (a01 * a12 - b11 * a02);
    let r = clamp(det_b / (2.0 * p * p * p), -1.0, 1.0);
    let phi = acos(r) / 3.0;
    // Eigenvalues are q + 2p cos(phi + 2 pi j / 3); j = 1 gives the smallest.
    let lambda = q + 2.0 * p * cos(phi + 2.0943951);

    let r0 = vec3<f32>(a00 - lambda, a01, a02);
    let r1 = vec3<f32>(a01, a11 - lambda, a12);
    let r2 = vec3<f32>(a02, a12, a22 - lambda);
    let x01 = cross(r0, r1);
    let x02 = cross(r0, r2);
    let x12 = cross(r1, r2);
    let d01 = dot(x01, x01);
    let d02 = dot(x02, x02);
    let d12 = dot(x12, x12);
    var best = x01;
    var best_len = d01;
    if (d02 > best_len) {
        best = x02;
        best_len = d02;
    }
    if (d12 > best_len) {
        best = x12;
        best_len = d12;
    }
    if (best_len < 1e-20) {
        // Two smallest eigenvalues equal (points on a line): any perpendicular works.
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return best / sqrt(best_len);
}

@compute @workgroup_size(64)
fn normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    // One thread per tree node rather than per input point: nodes are stored
    // in tree order, so neighboring threads search neighboring parts of the
    // cloud, which keeps them in step and their memory reads cached.
    let node_id = gid.x;
    if (node_id >= arrayLength(&nodes) || node_id >= params.num_points) {
        return;
    }
    let index = nodes[node_id].index;
    let center = nodes[node_id].pos;

    // k + 1 so the point itself (distance 0) can be dropped and still leave k.
    find_k_nearest(center, params.k + 1u, 3.4e38);

    // Neighborhood = up to k neighbors other than the point, plus the point.
    var sum = center;
    var count = 1u;
    for (var i = 0u; i < knn_count; i++) {
        let node = nodes[knn_node[i]];
        if (node.index != index && count <= params.k) {
            sum += node.pos;
            count += 1u;
        }
    }
    if (count < 4u) {
        output_normals[index] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
        return;
    }
    let centroid = sum / f32(count);

    let dc = center - centroid;
    var c00 = dc.x * dc.x;
    var c01 = dc.x * dc.y;
    var c02 = dc.x * dc.z;
    var c11 = dc.y * dc.y;
    var c12 = dc.y * dc.z;
    var c22 = dc.z * dc.z;
    var used = 1u;
    for (var i = 0u; i < knn_count; i++) {
        let node = nodes[knn_node[i]];
        if (node.index != index && used <= params.k) {
            let d = node.pos - centroid;
            c00 += d.x * d.x;
            c01 += d.x * d.y;
            c02 += d.x * d.z;
            c11 += d.y * d.y;
            c12 += d.y * d.z;
            c22 += d.z * d.z;
            used += 1u;
        }
    }

    var normal = smallest_eigenvector(c00, c01, c02, c11, c12, c22);
    if (params.consistent_orientation == 1u) {
        let to_view = params.viewpoint.xyz - center;
        if (dot(normal, to_view) < 0.0) {
            normal = -normal;
        }
    }
    output_normals[index] = vec4<f32>(normal, 0.0);
}
"#
    )
}

/// Uniform block of the normals kernel (`NormalParams` in WGSL).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct NormalParams {
    num_points: u32,
    k: u32,
    consistent_orientation: u32,
    _pad: u32,
    viewpoint: [f32; 4],
}

impl GpuContext {
    /// Compute normals for a point cloud using GPU acceleration with options.
    ///
    /// Uses the `k` nearest neighbors of each point (plus the point), like the
    /// CPU `estimate_normals`. `k` is clamped to 3..=31. With
    /// `consistent_orientation`, normals point towards `viewpoint`, or by
    /// default towards a point above the cloud's bounding-box centre.
    pub async fn compute_normals_with_options(
        &self,
        points: &[Point3f],
        k_neighbors: usize,
        consistent_orientation: bool,
        viewpoint: Option<[f32; 3]>,
    ) -> Result<Vec<nalgebra::Vector3<f32>>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let k = k_neighbors.clamp(3, MAX_K - 1);

        let tree = GpuKdTree::new(self, points)?;
        if tree.is_empty() {
            return Ok(vec![nalgebra::Vector3::z(); points.len()]);
        }
        let frame = tree.frame;

        // Default viewpoint (same as the CPU path): above the bounding-box
        // centre by the bounding-box diagonal.
        let viewpoint = viewpoint.unwrap_or_else(|| {
            let (min, max) =
                points
                    .iter()
                    .fold(([f32::MAX; 3], [f32::MIN; 3]), |(mut lo, mut hi), p| {
                        for i in 0..3 {
                            lo[i] = lo[i].min(p[i]);
                            hi[i] = hi[i].max(p[i]);
                        }
                        (lo, hi)
                    });
            let extent =
                ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2))
                    .sqrt();
            [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5 + extent,
            ]
        });

        // Points with a NaN or infinite coordinate have no tree node, so the
        // kernel never writes them; they keep this default unit normal.
        let output = self.create_buffer_init(
            "Normals Output",
            &vec![[0.0f32, 0.0, 1.0, 0.0]; points.len()],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params = NormalParams {
            // Threads run per tree node; non-finite points have no node.
            num_points: tree.node_to_index.len() as u32,
            k: k as u32,
            consistent_orientation: consistent_orientation as u32,
            _pad: 0,
            viewpoint: vec4(&frame.point_to_local(&Point3f::from(viewpoint))),
        };
        let params_buffer =
            self.create_buffer_init("Normals Params", &[params], wgpu::BufferUsages::UNIFORM);

        let pipeline = self.cached_pipeline(
            &self.pipelines.normals,
            "Normals",
            normals_shader,
            "normals",
        );
        let bind_group = self.create_bind_group(
            "Normals",
            &pipeline.get_bind_group_layout(0),
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tree.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Normals"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Normals"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(tree.node_to_index.len().div_ceil(64) as u32, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        let normals: Vec<[f32; 4]> = self.read_buffer(&output)?;
        Ok(normals
            .into_iter()
            .map(|n| nalgebra::Vector3::new(n[0], n[1], n[2]))
            .collect())
    }

    /// Compute normals for a point cloud using GPU acceleration with default options
    pub async fn compute_normals(
        &self,
        points: &[Point3f],
        k_neighbors: usize,
    ) -> Result<Vec<nalgebra::Vector3<f32>>> {
        self.compute_normals_with_options(points, k_neighbors, true, None)
            .await
    }

    /// The `k` nearest neighbors of every point (not counting the point
    /// itself), padded to 64 slots by repeating the last neighbor. Used by the
    /// GPU statistical outlier filter. Runs on the CPU with a kd-tree.
    pub fn compute_neighbors_simple(&self, points: &[[f32; 3]], k: usize) -> Vec<[u32; 64]> {
        let k = k.min(64).min(points.len().saturating_sub(1));
        if points.is_empty() {
            return Vec::new();
        }
        let cloud: Vec<Point3f> = points
            .iter()
            .map(|p| Point3f::new(p[0], p[1], p[2]))
            .collect();
        let Ok(tree) = KdTree::new(&cloud) else {
            return vec![[0; 64]; points.len()];
        };

        cloud
            .par_iter()
            .enumerate()
            .map_init(Vec::new, |knn, (i, point)| {
                tree.find_k_nearest_into(point, k + 1, knn);
                let mut neighbors = [i as u32; 64];
                let mut count = 0;
                for &(index, _) in knn.iter() {
                    if index != i && count < k {
                        neighbors[count] = index as u32;
                        count += 1;
                    }
                }
                if count > 0 {
                    let last = neighbors[count - 1];
                    neighbors[count..].fill(last);
                }
                neighbors
            })
            .collect()
    }

    /// Helper to compute neighbors from Vec<[f32;3]> built from owned data
    pub fn compute_neighbors_simple_points3(
        &self,
        points: &[[f32; 3]],
        k: usize,
    ) -> Vec<[u32; 64]> {
        self.compute_neighbors_simple(points, k)
    }
}

/// GPU-accelerated normal estimation for point clouds
pub async fn gpu_estimate_normals(
    gpu_context: &GpuContext,
    cloud: &mut PointCloud<Point3f>,
    k: usize,
) -> Result<PointCloud<NormalPoint3f>> {
    let normals = gpu_context.compute_normals(&cloud.points, k).await?;

    let normal_points: Vec<NormalPoint3f> = cloud
        .points
        .iter()
        .zip(normals.iter())
        .map(|(point, normal)| NormalPoint3f {
            position: *point,
            normal: *normal,
        })
        .collect();

    Ok(PointCloud::from_points(normal_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{Point3f, PointCloud};

    /// Try to create a GPU context, return None if not available
    async fn try_create_gpu_context() -> Option<crate::GpuContext> {
        match crate::GpuContext::new().await {
            Ok(gpu) => Some(gpu),
            Err(_) => {
                println!("⚠️  GPU not available, skipping GPU-dependent test");
                None
            }
        }
    }

    #[tokio::test]
    async fn test_gpu_normals_plane() {
        let Some(gpu) = try_create_gpu_context().await else {
            return;
        };

        let mut cloud = PointCloud::new();
        // Create XY plane grid
        for i in 0..15 {
            for j in 0..15 {
                cloud.push(Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        let result = gpu_estimate_normals(&gpu, &mut cloud, 8).await.unwrap();
        assert_eq!(result.len(), 225);
        let mut z_count = 0;
        for p in result.iter() {
            if p.normal.z.abs() > 0.8 {
                z_count += 1;
            }
        }
        let pct = (z_count as f32 / result.len() as f32) * 100.0;
        assert!(pct > 80.0, "Only {:.1}% normals in Z direction", pct);
    }

    #[tokio::test]
    async fn test_gpu_normals_compare_cpu_plane() {
        use threecrate_algorithms::estimate_normals as cpu_estimate_normals;
        let Some(gpu) = try_create_gpu_context().await else {
            return;
        };

        let mut cloud = PointCloud::new();
        for i in 0..10 {
            for j in 0..10 {
                cloud.push(Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        let gpu_cloud = gpu_estimate_normals(&gpu, &mut cloud.clone(), 8)
            .await
            .unwrap();
        let cpu_cloud = cpu_estimate_normals(&cloud, 8).unwrap();
        // Compare orientation alignment percentage
        let mut agree = 0usize;
        for (g, c) in gpu_cloud.iter().zip(cpu_cloud.iter()) {
            let dot = g.normal.dot(&c.normal);
            if dot.abs() > 0.7 {
                agree += 1;
            }
        }
        let pct = (agree as f32 / gpu_cloud.len() as f32) * 100.0;
        assert!(pct > 70.0, "GPU-CPU normals agree only {:.1}%", pct);
    }

    #[tokio::test]
    async fn test_gpu_normals_non_finite_point_gets_unit_normal() {
        let Some(gpu) = try_create_gpu_context().await else {
            return;
        };
        let mut cloud = PointCloud::new();
        for i in 0..10 {
            for j in 0..10 {
                cloud.push(Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        cloud.push(Point3f::new(f32::NAN, 0.0, 0.0));
        let normals = gpu_estimate_normals(&gpu, &mut cloud, 8).await.unwrap();
        assert_eq!(
            normals.points[100].normal,
            nalgebra::Vector3::new(0.0, 0.0, 1.0)
        );
        assert!(normals.points[..100]
            .iter()
            .all(|p| (p.normal.norm() - 1.0).abs() < 1e-4));
    }

    #[tokio::test]
    async fn test_gpu_normals_match_cpu_on_curved_surface() {
        use threecrate_algorithms::estimate_normals as cpu_estimate_normals;
        let Some(gpu) = try_create_gpu_context().await else {
            return;
        };

        // Curved, slightly irregular surface far from the origin
        let mut cloud = PointCloud::new();
        for i in 0..120 {
            for j in 0..120 {
                let x = i as f32 * 0.05 + 0.013 * ((i * 7 + j * 3) % 5) as f32;
                let y = j as f32 * 0.05 + 0.011 * ((i * 3 + j * 5) % 7) as f32;
                let z = 0.5 * (x * 1.3).sin() * (y * 0.9).cos();
                cloud.push(Point3f::new(x + 1000.0, y - 500.0, z + 20.0));
            }
        }

        let gpu_cloud = gpu_estimate_normals(&gpu, &mut cloud.clone(), 10)
            .await
            .unwrap();
        let cpu_cloud = cpu_estimate_normals(&cloud, 10).unwrap();

        // Same direction and same orientation (both point to the viewpoint)
        let agree = gpu_cloud
            .iter()
            .zip(cpu_cloud.iter())
            .filter(|(g, c)| g.normal.dot(&c.normal) > 0.999)
            .count();
        let pct = 100.0 * agree as f32 / gpu_cloud.len() as f32;
        assert!(
            pct > 99.0,
            "GPU and CPU normals agree on only {pct:.2}% of points"
        );
    }
}
