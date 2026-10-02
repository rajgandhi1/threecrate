//! GPU-accelerated ICP
//!
//! The target cloud is uploaded once as a kd-tree. Each iteration runs one
//! kernel that moves every source point by the current estimate, finds its
//! nearest target point in the tree (starting from last iteration's match), and
//! sums the values the update needs within each workgroup. Only those small
//! per-workgroup sums come back to the CPU.

use crate::spatial::{morton_order, vec4s, GpuKdTree, KD_TREE_WGSL};
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use nalgebra::{Isometry3, Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};
use threecrate_algorithms::{icp_converged, rigid_transform_from_sums, CloudScale};
use threecrate_core::{Error, Point3f, PointCloud, Result, Vector3f};

/// Sums per workgroup for point-to-point: count, source sum (3), target sum
/// (3), source * target^T (9), squared distance, distance.
const STATS_POINT: usize = 18;
/// Sums per workgroup for point-to-plane: count, squared residual, the upper
/// triangle of A^T A (21), and A^T b (6).
const STATS_PLANE: usize = 29;
/// Workgroup size of the matching kernels (`WG` in WGSL).
const WORKGROUP: usize = 128;

fn icp_match_shader() -> String {
    format!(
        "{}{}",
        KD_TREE_WGSL,
        r#"
struct IcpParams {
    // Rows of the current transform; w holds the translation.
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    num_source: u32,
    max_dist_sq: f32,
    use_seed: u32,
    _pad: u32,
}

const WG: u32 = 128u;

@group(0) @binding(1) var<storage, read> source: array<vec4<f32>>;
// Matched node per source point (NO_NODE if none); also the seed for the next
// iteration's search.
@group(0) @binding(2) var<storage, read_write> matches: array<u32>;
// Sums per workgroup.
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(0) @binding(4) var<uniform> params: IcpParams;
// Target normals by original target index (point-to-plane only).
@group(0) @binding(5) var<storage, read> target_normals: array<vec4<f32>>;

var<workgroup> scratch_point: array<f32, 2304>; // WG * 18
var<workgroup> scratch_plane: array<f32, 3712>; // WG * 29

struct Match {
    found: Nearest,
    moved: vec3<f32>,
}

// Move source point `i` by the current transform and find its nearest target.
fn match_point(i: u32) -> Match {
    let p = source[i].xyz;
    let q = vec3<f32>(
        dot(params.row0.xyz, p) + params.row0.w,
        dot(params.row1.xyz, p) + params.row1.w,
        dot(params.row2.xyz, p) + params.row2.w,
    );
    var seed = NO_NODE;
    if (params.use_seed == 1u) {
        seed = matches[i];
    }
    let found = find_nearest(q, params.max_dist_sq, seed);
    matches[i] = found.node;
    return Match(found, q);
}

@compute @workgroup_size(128)
fn icp_match(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    const STATS: u32 = 18u;
    var s: array<f32, 18>;
    for (var j = 0u; j < STATS; j++) {
        s[j] = 0.0;
    }

    if (gid.x < params.num_source) {
        let m = match_point(gid.x);
        if (m.found.node != NO_NODE) {
            let q = m.moved;
            let t = nodes[m.found.node].pos;
            s[0] = 1.0;
            s[1] = q.x; s[2] = q.y; s[3] = q.z;
            s[4] = t.x; s[5] = t.y; s[6] = t.z;
            s[7] = q.x * t.x; s[8] = q.x * t.y; s[9] = q.x * t.z;
            s[10] = q.y * t.x; s[11] = q.y * t.y; s[12] = q.y * t.z;
            s[13] = q.z * t.x; s[14] = q.z * t.y; s[15] = q.z * t.z;
            s[16] = m.found.dist_sq;
            s[17] = sqrt(m.found.dist_sq);
        }
    }

    // Tree reduction of the per-point values within the workgroup.
    let base = lid.x * STATS;
    for (var j = 0u; j < STATS; j++) {
        scratch_point[base + j] = s[j];
    }
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            let other = (lid.x + stride) * STATS;
            for (var j = 0u; j < STATS; j++) {
                scratch_point[base + j] += scratch_point[other + j];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        for (var j = 0u; j < STATS; j++) {
            partials[wid.x * STATS + j] = scratch_point[j];
        }
    }
}

// Point-to-plane: each match adds row a = (q x n, n) and residual
// b = n . (t - q) to the linearised 6x6 system (Chen & Medioni 1992).
@compute @workgroup_size(128)
fn icp_match_plane(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    const STATS: u32 = 29u;
    var s: array<f32, 29>;
    for (var j = 0u; j < STATS; j++) {
        s[j] = 0.0;
    }

    if (gid.x < params.num_source) {
        let m = match_point(gid.x);
        if (m.found.node != NO_NODE) {
            let node = nodes[m.found.node];
            let n = target_normals[node.index].xyz;
            let q = m.moved;
            let c = cross(q, n);
            var a = array<f32, 6>(c.x, c.y, c.z, n.x, n.y, n.z);
            let b = dot(n, node.pos - q);
            s[0] = 1.0;
            s[1] = b * b;
            var k = 2u;
            for (var r = 0u; r < 6u; r++) {
                for (var col = r; col < 6u; col++) {
                    s[k] = a[r] * a[col];
                    k += 1u;
                }
            }
            for (var r = 0u; r < 6u; r++) {
                s[23u + r] = a[r] * b;
            }
        }
    }

    let base = lid.x * STATS;
    for (var j = 0u; j < STATS; j++) {
        scratch_plane[base + j] = s[j];
    }
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            let other = (lid.x + stride) * STATS;
            for (var j = 0u; j < STATS; j++) {
                scratch_plane[base + j] += scratch_plane[other + j];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        for (var j = 0u; j < STATS; j++) {
            partials[wid.x * STATS + j] = scratch_plane[j];
        }
    }
}
"#
    )
}

/// Uniform block of the matching kernels (`IcpParams` in WGSL).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IcpParams {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    num_source: u32,
    max_dist_sq: f32,
    use_seed: u32,
    _pad: u32,
}

/// GPU state for one ICP run: target tree, source points, and the buffers the
/// matching kernel reads and writes. Everything stays on the GPU between
/// iterations except the small per-workgroup sums, which are copied into one
/// staging buffer that is reused every iteration.
struct IcpSession<'a> {
    gpu: &'a GpuContext,
    tree: GpuKdTree,
    /// Size of the source cloud (centroid and spread) in the tree's frame.
    source_scale: CloudScale,
    num_source: usize,
    pipeline: &'a wgpu::ComputePipeline,
    partials: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    stats_len: usize,
    max_dist_sq: f32,
    iteration: usize,
}

impl<'a> IcpSession<'a> {
    /// `target_normals` selects point-to-plane (`Some`) or point-to-point.
    fn new(
        gpu: &'a GpuContext,
        source: &[Point3f],
        target: &[Point3f],
        target_normals: Option<&[Vector3f]>,
        max_correspondence_distance: f32,
    ) -> Result<Self> {
        let tree = GpuKdTree::new(gpu, target)?;
        if tree.is_empty() {
            return Err(Error::InvalidData(
                "Target point cloud has no finite points".to_string(),
            ));
        }
        let source_local = tree.frame.points_to_local(source);
        let source_scale = CloudScale::of(&source_local);

        // Upload in Morton order so neighboring GPU threads search neighboring
        // parts of the target tree. The sums do not depend on point order.
        let uploaded: Vec<Point3f> = morton_order(&source_local)
            .into_iter()
            .map(|i| source_local[i as usize])
            .collect();
        let source_buffer =
            gpu.create_buffer_init("ICP Source", &vec4s(&uploaded), wgpu::BufferUsages::STORAGE);
        let matches = gpu.create_buffer_init(
            "ICP Matches",
            &vec![u32::MAX; source.len()],
            wgpu::BufferUsages::STORAGE,
        );

        let stats_len = if target_normals.is_some() {
            STATS_PLANE
        } else {
            STATS_POINT
        };
        let partials_size =
            (source.len().div_ceil(WORKGROUP) * stats_len * std::mem::size_of::<f32>()) as u64;
        let partials = gpu.create_buffer(
            "ICP Partials",
            partials_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = gpu.create_staging_buffer("ICP Partials Staging", partials_size);
        let params = gpu.create_buffer(
            "ICP Params",
            std::mem::size_of::<IcpParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tree.nodes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: source_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matches.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: partials.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ];
        let normals_buffer;
        let pipeline = if let Some(normals) = target_normals {
            let data: Vec<[f32; 4]> = normals.iter().map(|n| [n.x, n.y, n.z, 0.0]).collect();
            normals_buffer =
                gpu.create_buffer_init("ICP Target Normals", &data, wgpu::BufferUsages::STORAGE);
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: normals_buffer.as_entire_binding(),
            });
            gpu.cached_pipeline(
                &gpu.pipelines.icp_match_plane,
                "ICP Match Plane",
                icp_match_shader,
                "icp_match_plane",
            )
        } else {
            gpu.cached_pipeline(
                &gpu.pipelines.icp_match,
                "ICP Match",
                icp_match_shader,
                "icp_match",
            )
        };
        let bind_group =
            gpu.create_bind_group("ICP Match", &pipeline.get_bind_group_layout(0), &entries);

        // Same cutoff semantics as the CPU ICP: negative rejects everything.
        let max_dist_sq = if max_correspondence_distance < 0.0 {
            0.0
        } else {
            max_correspondence_distance * max_correspondence_distance
        };

        Ok(Self {
            gpu,
            tree,
            source_scale,
            num_source: source.len(),
            pipeline,
            partials,
            staging,
            params,
            bind_group,
            stats_len,
            max_dist_sq,
            iteration: 0,
        })
    }

    /// Match every source point (moved by `transform`, in the tree's frame) to
    /// its nearest target point on the GPU, and return the summed statistics.
    fn match_points(&mut self, transform: &Isometry3<f32>) -> Result<Vec<f64>> {
        let m = transform.to_homogeneous();
        let row = |r: usize| [m[(r, 0)], m[(r, 1)], m[(r, 2)], m[(r, 3)]];
        let params = IcpParams {
            row0: row(0),
            row1: row(1),
            row2: row(2),
            num_source: self.num_source as u32,
            max_dist_sq: self.max_dist_sq,
            use_seed: (self.iteration > 0) as u32,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ICP Match"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ICP Match"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.num_source.div_ceil(WORKGROUP) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.partials, 0, &self.staging, 0, self.partials.size());
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        self.iteration += 1;

        let partials: Vec<f32> = self.gpu.read_staging(&self.staging)?;
        let mut totals = vec![0.0f64; self.stats_len];
        for group in partials.chunks_exact(self.stats_len) {
            for (total, value) in totals.iter_mut().zip(group) {
                *total += *value as f64;
            }
        }
        Ok(totals)
    }

    /// The source centroid moved by `transform`, for the convergence check.
    fn moved_centroid(&self, transform: &Isometry3<f32>) -> Point3f {
        transform * self.source_scale.centroid
    }
}

/// Batch ICP operation for multiple point cloud pairs
#[derive(Debug, Clone)]
pub struct BatchICPJob {
    pub source: PointCloud<Point3f>,
    pub target: PointCloud<Point3f>,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub max_correspondence_distance: f32,
}

/// Result of a batch ICP operation
#[derive(Debug, Clone)]
pub struct BatchICPResult {
    pub transformation: Isometry3<f32>,
    pub final_error: f32,
    pub iterations: usize,
}

impl GpuContext {
    /// Execute multiple ICP operations in parallel batches
    pub async fn batch_icp_align(&self, jobs: &[BatchICPJob]) -> Result<Vec<BatchICPResult>> {
        let mut results = Vec::with_capacity(jobs.len());
        for job in jobs {
            results.push(
                self.optimized_icp_align(
                    &job.source,
                    &job.target,
                    job.max_iterations,
                    job.convergence_threshold,
                    job.max_correspondence_distance,
                )
                .await?,
            );
        }
        Ok(results)
    }

    /// Point-to-point ICP with GPU matching.
    ///
    /// Uses the same stopping rule as the CPU ICP: stop when the RMSE improves
    /// by less than `convergence_threshold` as a fraction, or an update barely
    /// moves the points relative to the cloud's size. `final_error` is the
    /// mean correspondence distance of the last iteration.
    async fn optimized_icp_align(
        &self,
        source: &PointCloud<Point3f>,
        target: &PointCloud<Point3f>,
        max_iterations: usize,
        convergence_threshold: f32,
        max_correspondence_distance: f32,
    ) -> Result<BatchICPResult> {
        if source.is_empty() || target.is_empty() {
            return Err(Error::InvalidData("Empty point clouds".to_string()));
        }

        let mut session = IcpSession::new(
            self,
            &source.points,
            &target.points,
            None,
            max_correspondence_distance,
        )?;
        let mut current = Isometry3::identity();
        let mut previous_mse = f32::INFINITY;
        let mut final_error = f32::INFINITY;
        let mut iterations_used = 0;

        for iteration in 0..max_iterations {
            iterations_used = iteration + 1;
            let s = session.match_points(&current)?;
            let count = s[0];
            if count < 3.0 {
                break;
            }
            final_error = (s[17] / count) as f32;
            let mse = (s[16] / count) as f32;

            let delta = rigid_transform_from_sums(
                count,
                &Vector3::new(s[1], s[2], s[3]),
                &Vector3::new(s[4], s[5], s[6]),
                &Matrix3::new(s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]),
            )?;
            let moved_centroid = session.moved_centroid(&current);
            current = delta * current;

            if icp_converged(
                previous_mse,
                mse,
                &delta,
                &moved_centroid,
                session.source_scale.radius,
                convergence_threshold,
            ) {
                break;
            }
            previous_mse = mse;
        }

        Ok(BatchICPResult {
            transformation: session.tree.frame.transform_to_world(&current),
            final_error,
            iterations: iterations_used,
        })
    }

    /// GPU-accelerated ICP alignment (original single implementation)
    pub async fn icp_align(
        &self,
        source: &PointCloud<Point3f>,
        target: &PointCloud<Point3f>,
        max_iterations: usize,
        convergence_threshold: f32,
        max_correspondence_distance: f32,
    ) -> Result<Isometry3<f32>> {
        let result = self
            .optimized_icp_align(
                source,
                target,
                max_iterations,
                convergence_threshold,
                max_correspondence_distance,
            )
            .await?;
        Ok(result.transformation)
    }
}

/// Result of GPU point-to-plane ICP
#[derive(Debug, Clone)]
pub struct GpuPointToPlaneICPResult {
    pub transformation: Isometry3<f32>,
    pub final_error: f32,
    pub iterations: usize,
    pub converged: bool,
}

impl GpuContext {
    /// GPU-accelerated point-to-plane ICP.
    ///
    /// The GPU matches points and sums the linearised 6×6 system
    /// (Chen & Medioni 1992); the CPU only solves it. Stops with the same rule
    /// as the CPU ICP (see [`icp_converged`]).
    pub async fn icp_point_to_plane_align(
        &self,
        source: &PointCloud<Point3f>,
        target: &PointCloud<Point3f>,
        target_normals: &[Vector3f],
        max_iterations: usize,
        convergence_threshold: f32,
        max_correspondence_distance: f32,
    ) -> Result<GpuPointToPlaneICPResult> {
        if source.is_empty() || target.is_empty() {
            return Err(Error::InvalidData("Empty point clouds".to_string()));
        }
        if target_normals.len() != target.points.len() {
            return Err(Error::InvalidData(
                "target_normals length must equal target point count".to_string(),
            ));
        }

        let mut session = IcpSession::new(
            self,
            &source.points,
            &target.points,
            Some(target_normals),
            max_correspondence_distance,
        )?;
        let mut current = Isometry3::identity();
        let mut previous_mse = f32::INFINITY;
        let mut final_error = f32::INFINITY;
        let mut iterations_used = 0;
        let mut converged = false;

        for iteration in 0..max_iterations {
            iterations_used = iteration + 1;
            let s = session.match_points(&current)?;
            let count = s[0];
            if count < 6.0 {
                break;
            }
            let mse = (s[1] / count) as f32;
            final_error = mse;

            let delta = solve_point_to_plane(&s[2..23], &s[23..29])?;
            let moved_centroid = session.moved_centroid(&current);
            current = delta * current;

            if icp_converged(
                previous_mse,
                mse,
                &delta,
                &moved_centroid,
                session.source_scale.radius,
                convergence_threshold,
            ) {
                converged = true;
                break;
            }
            previous_mse = mse;
        }

        Ok(GpuPointToPlaneICPResult {
            transformation: session.tree.frame.transform_to_world(&current),
            final_error,
            iterations: iterations_used,
            converged,
        })
    }
}

/// Solve the linearised point-to-plane system `AᵀA x = Aᵀb` given the upper
/// triangle of `AᵀA` (row-major, 21 values) and `Aᵀb`, and turn
/// `x = (rx, ry, rz, tx, ty, tz)` into a rigid update.
fn solve_point_to_plane(ata_upper: &[f64], atb: &[f64]) -> Result<Isometry3<f32>> {
    let mut ata = Matrix6::<f64>::zeros();
    let mut k = 0;
    for r in 0..6 {
        for c in r..6 {
            ata[(r, c)] = ata_upper[k];
            ata[(c, r)] = ata_upper[k];
            k += 1;
        }
    }
    let atb = Vector6::from_column_slice(atb);

    let x = if let Some(chol) = ata.cholesky() {
        chol.solve(&atb)
    } else {
        ata.lu().solve(&atb).ok_or_else(|| {
            Error::Algorithm("Point-to-plane GPU system is ill-conditioned".to_string())
        })?
    };
    let x = x.cast::<f32>();

    let rot_x = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), x[0]);
    let rot_y = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), x[1]);
    let rot_z = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), x[2]);

    Ok(Isometry3::from_parts(
        Translation3::new(x[3], x[4], x[5]),
        rot_z * rot_y * rot_x,
    ))
}

/// GPU-accelerated ICP registration
pub async fn gpu_icp(
    gpu_context: &GpuContext,
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    max_iterations: usize,
    convergence_threshold: f32,
    max_correspondence_distance: f32,
) -> Result<Isometry3<f32>> {
    gpu_context
        .icp_align(
            source,
            target,
            max_iterations,
            convergence_threshold,
            max_correspondence_distance,
        )
        .await
}

/// Execute batch ICP operations on multiple point cloud pairs
pub async fn gpu_batch_icp(
    gpu_context: &GpuContext,
    jobs: &[BatchICPJob],
) -> Result<Vec<BatchICPResult>> {
    gpu_context.batch_icp_align(jobs).await
}

/// GPU-accelerated point-to-plane ICP registration.
///
/// The GPU finds correspondences and sums the linearised 6×6 system; the CPU
/// only solves it (Chen & Medioni 1992).
///
/// # Arguments
/// * `gpu_context`                  - Initialized GPU context
/// * `source`                       - Source point cloud
/// * `target`                       - Target point cloud
/// * `target_normals`               - Surface normals at each target point
/// * `max_iterations`               - Maximum number of iterations
/// * `convergence_threshold`        - Stop when the RMSE improves by less than this
///   fraction, or an update barely moves the points (same rule as the CPU ICP)
/// * `max_correspondence_distance`  - Maximum distance for valid correspondences
pub async fn gpu_icp_point_to_plane(
    gpu_context: &GpuContext,
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    target_normals: &[Vector3f],
    max_iterations: usize,
    convergence_threshold: f32,
    max_correspondence_distance: f32,
) -> Result<GpuPointToPlaneICPResult> {
    gpu_context
        .icp_point_to_plane_align(
            source,
            target,
            target_normals,
            max_iterations,
            convergence_threshold,
            max_correspondence_distance,
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn try_create_gpu_context() -> Option<GpuContext> {
        GpuContext::new().await.ok()
    }

    /// Curved grid far from the origin, scaled by `scale`, and a copy moved by
    /// a known transform. The move is under half the grid spacing at the
    /// edges; larger moves can trap point-to-point ICP on a regular grid (on
    /// the CPU as well).
    fn test_pair(scale: f32) -> (PointCloud<Point3f>, PointCloud<Point3f>, Isometry3<f32>) {
        let mut source = PointCloud::new();
        for i in 0..60 {
            for j in 0..60 {
                let (x, y) = (i as f32 * 0.1, j as f32 * 0.1);
                let z = 0.4 * (x * 1.1).sin() * (y * 0.8).cos();
                source.push(Point3f::new(x + 500.0, y - 200.0, z + 10.0) * scale);
            }
        }
        let centre = Translation3::new(503.0 * scale, -197.0 * scale, 10.0 * scale);
        let truth = centre
            * Isometry3::from_parts(
                Translation3::new(0.02 * scale, -0.01 * scale, 0.01 * scale),
                UnitQuaternion::from_euler_angles(0.003, -0.002, 0.01),
            )
            * centre.inverse();
        let target = PointCloud::from_points(source.points.iter().map(|p| truth * p).collect());
        (source, target, truth)
    }

    fn max_point_error(
        estimate: &Isometry3<f32>,
        truth: &Isometry3<f32>,
        cloud: &[Point3f],
    ) -> f32 {
        cloud
            .iter()
            .map(|p| (estimate * p - truth * p).norm())
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_gpu_icp_recovers_known_transform() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };
            let (source, target, truth) = test_pair(1.0);
            let estimate = gpu_icp(&gpu, &source, &target, 50, 1e-7, 1.0)
                .await
                .unwrap();
            let error = max_point_error(&estimate, &truth, &source.points);
            assert!(error < 1e-3, "GPU ICP points are {error} m off");

            // Same answer as the CPU implementation
            let cpu = threecrate_algorithms::icp_point_to_point(
                &source,
                &target,
                Isometry3::identity(),
                50,
                1e-7,
                Some(1.0),
            )
            .unwrap();
            let gap = max_point_error(&estimate, &cpu.transformation, &source.points);
            assert!(gap < 1e-3, "GPU and CPU ICP differ by {gap} m");
        });
    }

    #[test]
    fn test_gpu_icp_convergence_is_scale_invariant() {
        // The same problem at full and 1000x scale must stop at the same point
        // and be solved just as well. An absolute translation threshold made
        // the stopping point depend on the scene size: rounding noise in a large
        // scene stayed above it, so ICP never stopped.
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };
            let mut runs = Vec::new();
            for scale in [1.0f32, 1000.0] {
                let (source, target, truth) = test_pair(scale);
                let result = gpu
                    .optimized_icp_align(&source, &target, 50, 1e-5, scale)
                    .await
                    .unwrap();
                let error = max_point_error(&result.transformation, &truth, &source.points) / scale;
                assert!(error < 1e-3, "scale {scale}: {error} (relative) off");
                runs.push(result.iterations);
            }
            assert!(
                runs[0].abs_diff(runs[1]) <= 1,
                "iterations differ with scale: {runs:?}"
            );
        });
    }

    #[test]
    fn test_gpu_point_to_plane_recovers_known_transform() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };
            let (source, target, truth) = test_pair(1.0);
            let normals: Vec<Vector3f> = threecrate_algorithms::estimate_normals(&target, 10)
                .unwrap()
                .points
                .iter()
                .map(|p| p.normal)
                .collect();
            let result = gpu_icp_point_to_plane(&gpu, &source, &target, &normals, 50, 1e-7, 1.0)
                .await
                .unwrap();
            let error = max_point_error(&result.transformation, &truth, &source.points);
            assert!(
                error < 1e-3,
                "GPU point-to-plane ICP points are {error} m off"
            );
            assert!(result.converged);
        });
    }
}
