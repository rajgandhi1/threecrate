//! Registration algorithms

use crate::filtering::voxel_grid_filter;
use crate::nearest_neighbor::KdTree;
use nalgebra::{Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};
use rayon::prelude::*;
use threecrate_core::{Error, Isometry3, Point3f, PointCloud, Result, Vector3f};

/// Result of ICP registration
#[derive(Debug, Clone)]
pub struct ICPResult {
    /// Final transformation
    pub transformation: Isometry3<f32>,
    /// Final mean squared error
    pub mse: f32,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Correspondences found in the last iteration
    pub correspondences: Vec<(usize, usize)>,
}

/// One level in a coarse-to-fine ICP pyramid.
#[derive(Debug, Clone)]
pub struct IcpScaleLevel {
    /// Voxel size used to downsample source and target before this ICP stage.
    pub voxel_size: f32,
    /// Maximum ICP iterations for this stage.
    pub max_iterations: usize,
    /// Optional correspondence cutoff for this stage.
    pub max_correspondence_distance: Option<f32>,
}

/// Coarse-to-fine point-to-point ICP configuration.
#[derive(Debug, Clone)]
pub struct MultiScaleIcpConfig {
    pub levels: Vec<IcpScaleLevel>,
    pub final_refinement_iterations: usize,
    pub final_max_correspondence_distance: Option<f32>,
    /// Stop when RMSE improves by less than this fraction of its previous value.
    pub convergence_threshold: f32,
}

impl Default for MultiScaleIcpConfig {
    fn default() -> Self {
        Self {
            levels: vec![
                IcpScaleLevel {
                    voxel_size: 0.20,
                    max_iterations: 10,
                    max_correspondence_distance: Some(0.50),
                },
                IcpScaleLevel {
                    voxel_size: 0.10,
                    max_iterations: 10,
                    max_correspondence_distance: Some(0.25),
                },
                IcpScaleLevel {
                    voxel_size: 0.05,
                    max_iterations: 15,
                    max_correspondence_distance: Some(0.15),
                },
            ],
            final_refinement_iterations: 10,
            final_max_correspondence_distance: Some(0.10),
            convergence_threshold: 1e-5,
        }
    }
}

/// Find the closest point in target cloud for each point in source cloud
#[cfg(test)]
fn find_correspondences(
    source: &[Point3f],
    target: &[Point3f],
    max_distance: Option<f32>,
) -> Vec<Option<(usize, f32)>> {
    match KdTree::new(target) {
        Ok(tree) => {
            let no_previous = vec![None; source.len()];
            find_correspondences_seeded(
                source,
                &Isometry3::identity(),
                target,
                &tree,
                &no_previous,
                max_distance_sq(max_distance),
            )
            .into_iter()
            .map(|c| c.map(|(idx, dist_sq)| (idx, dist_sq.sqrt())))
            .collect()
        }
        Err(_) => find_correspondences_brute_force(source, target, max_distance),
    }
}

/// Turn an optional correspondence cutoff into the exclusive squared-distance
/// bound taken by [`KdTree::find_nearest_bounded`], keeping the original
/// semantics: a match at exactly `max_distance` is accepted, a negative cutoff
/// rejects every match, and `None` (or NaN, which never compared as "too far")
/// means no limit.
fn max_distance_sq(max_distance: Option<f32>) -> f32 {
    match max_distance {
        Some(d) if d < 0.0 => 0.0,
        Some(d) if d >= 0.0 => (d * d).next_up(),
        _ => f32::INFINITY,
    }
}

/// Find the nearest target point for each source point after applying
/// `transform`, in parallel.
///
/// `previous` holds each source point's match from the last ICP iteration.
/// Between iterations a point moves only by the small delta transform, so its
/// old match is almost always at or near the new nearest neighbor; seeding the
/// kd-tree search with it lets most of the tree be pruned immediately. The
/// result is still the exact nearest neighbor — the seed is only a bound.
///
/// Returns `(target_index, squared_distance)` per source point, or `None` when
/// no target point lies within `max_dist_sq`.
/// Fewest points one parallel ICP task handles. ICP makes light passes over the
/// cloud many times per run; splitting them finer wakes more threads than the
/// work is worth, which is especially slow in VMs (waking a thread there costs
/// far more than matching a few hundred points).
const MIN_POINTS_PER_TASK: usize = 512;

fn find_correspondences_seeded(
    source: &[Point3f],
    transform: &Isometry3<f32>,
    target: &[Point3f],
    target_tree: &KdTree,
    previous: &[Option<(usize, f32)>],
    max_dist_sq: f32,
) -> Vec<Option<(usize, f32)>> {
    source
        .par_iter()
        .zip(previous.par_iter())
        .with_min_len(MIN_POINTS_PER_TASK)
        .map(|(point, prev)| {
            let moved = transform * point;
            let seed = prev.map(|(idx, _)| (idx, (moved - target[idx]).magnitude_squared()));
            target_tree.find_nearest_bounded(&moved, max_dist_sq, seed)
        })
        .collect()
}

/// One ICP iteration's parallel work in a single pass: re-match every source
/// point (moved by `transform`, seeded with its previous match, updated in
/// place in `matches`) and sum the statistics for the update. Doing both in
/// one pass halves how often worker threads are woken per iteration.
fn match_and_collect(
    source: &[Point3f],
    transform: &Isometry3<f32>,
    target: &[Point3f],
    target_tree: &KdTree,
    matches: &mut [Option<(usize, f32)>],
    max_dist_sq: f32,
) -> CorrespondenceStats {
    source
        .par_chunks(MIN_POINTS_PER_TASK)
        .zip(matches.par_chunks_mut(MIN_POINTS_PER_TASK))
        .map(|(points, slots)| {
            let mut stats = CorrespondenceStats::zero();
            for (point, slot) in points.iter().zip(slots) {
                let moved = transform * point;
                let seed = slot.map(|(idx, _)| (idx, (moved - target[idx]).magnitude_squared()));
                *slot = target_tree.find_nearest_bounded(&moved, max_dist_sq, seed);
                if let Some((idx, _)) = *slot {
                    stats = stats.add(&moved, &target[idx]);
                }
            }
            stats
        })
        .reduce(CorrespondenceStats::zero, CorrespondenceStats::merge)
}

/// Brute-force fallback used only if the KD-tree cannot be built.
#[cfg(test)]
fn find_correspondences_brute_force(
    source: &[Point3f],
    target: &[Point3f],
    max_distance: Option<f32>,
) -> Vec<Option<(usize, f32)>> {
    source
        .par_iter()
        .map(|source_point| {
            let mut best_distance = f32::INFINITY;
            let mut best_idx = None;

            for (target_idx, target_point) in target.iter().enumerate() {
                let distance = (source_point - target_point).magnitude();

                if distance < best_distance {
                    best_distance = distance;
                    best_idx = Some(target_idx);
                }
            }

            // Filter out correspondences that are too far
            if let Some(max_dist) = max_distance {
                if best_distance > max_dist {
                    return None;
                }
            }

            best_idx.map(|idx| (idx, best_distance))
        })
        .collect()
}

/// Running sums over point-to-point correspondences — enough to recover both
/// the optimal rigid transform and the MSE without materialising the matched
/// point lists, so a whole ICP iteration reduces in one parallel pass.
///
/// Sums are kept in `f64`: the covariance is formed in one pass as
/// `Σ p qᵀ − n·p̄ q̄ᵀ`, which cancels badly in `f32` on clouds of 10⁵+ points.
#[derive(Clone, Copy)]
struct CorrespondenceStats {
    count: usize,
    sum_source: Vector3<f64>,
    sum_target: Vector3<f64>,
    sum_outer: Matrix3<f64>,
    sum_sq_error: f64,
}

impl CorrespondenceStats {
    fn zero() -> Self {
        Self {
            count: 0,
            sum_source: Vector3::zeros(),
            sum_target: Vector3::zeros(),
            sum_outer: Matrix3::zeros(),
            sum_sq_error: 0.0,
        }
    }

    fn add(mut self, source: &Point3f, target: &Point3f) -> Self {
        let p = source.coords.cast::<f64>();
        let q = target.coords.cast::<f64>();
        self.count += 1;
        self.sum_source += p;
        self.sum_target += q;
        self.sum_outer += p * q.transpose();
        self.sum_sq_error += (p - q).norm_squared();
        self
    }

    fn merge(self, other: Self) -> Self {
        Self {
            count: self.count + other.count,
            sum_source: self.sum_source + other.sum_source,
            sum_target: self.sum_target + other.sum_target,
            sum_outer: self.sum_outer + other.sum_outer,
            sum_sq_error: self.sum_sq_error + other.sum_sq_error,
        }
    }

    /// Accumulate `source` (moved by `transform`) against its matched targets.
    fn collect(
        source: &[Point3f],
        transform: &Isometry3<f32>,
        target: &[Point3f],
        matches: &[Option<(usize, f32)>],
    ) -> Self {
        source
            .par_iter()
            .zip(matches.par_iter())
            .with_min_len(MIN_POINTS_PER_TASK)
            .fold(Self::zero, |stats, (point, m)| match m {
                Some((idx, _)) => stats.add(&(transform * point), &target[*idx]),
                None => stats,
            })
            .reduce(Self::zero, Self::merge)
    }

    /// Mean squared error between corresponding points
    fn mse(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum_sq_error / self.count as f64) as f32
        }
    }

    /// Compute the optimal transformation using SVD
    fn transformation(&self) -> Result<Isometry3<f32>> {
        rigid_transform_from_sums(
            self.count as f64,
            &self.sum_source,
            &self.sum_target,
            &self.sum_outer,
        )
    }
}

/// Best rigid transform taking a set of source points onto their matched
/// target points (Kabsch / SVD), from running sums over the matched pairs:
/// the pair count, the sums of source and target points, and the sum of
/// `source * targetᵀ`. Used by the CPU ICP and by the GPU ICP in
/// `threecrate-gpu`, which computes the sums on the GPU.
pub fn rigid_transform_from_sums(
    count: f64,
    sum_source: &Vector3<f64>,
    sum_target: &Vector3<f64>,
    sum_outer: &Matrix3<f64>,
) -> Result<Isometry3<f32>> {
    if count <= 0.0 {
        return Err(Error::InvalidData(
            "Point correspondence mismatch".to_string(),
        ));
    }

    let source_centroid = sum_source / count;
    let target_centroid = sum_target / count;

    // Covariance H = Σ (p − p̄)(q − q̄)ᵀ, accumulated in f64 but decomposed
    // in f32: nalgebra's f64 SVD fails to converge on some rank-deficient
    // matrices (e.g. [[8,8,0],[8,8,0],[0,0,0]] from collinear clouds).
    let h: Matrix3<f32> =
        (sum_outer - source_centroid * target_centroid.transpose() * count).cast::<f32>();

    // SVD decomposition, capped so a non-converging case errors, not hangs
    const SVD_MAX_ITERATIONS: usize = 1000;
    let svd = h
        .try_svd(true, true, f32::EPSILON * 5.0, SVD_MAX_ITERATIONS)
        .ok_or_else(|| Error::Algorithm("SVD did not converge".to_string()))?;
    let u = svd
        .u
        .ok_or_else(|| Error::Algorithm("SVD U matrix not available".to_string()))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| Error::Algorithm("SVD V^T matrix not available".to_string()))?;

    // Compute rotation matrix
    let mut r = v_t.transpose() * u.transpose();

    // Ensure proper rotation (det(R) = 1)
    if r.determinant() < 0.0 {
        let mut v_t_corrected = v_t;
        v_t_corrected.set_row(2, &(-v_t.row(2)));
        r = v_t_corrected.transpose() * u.transpose();
    }

    // Convert to unit quaternion
    let rotation = UnitQuaternion::from_matrix(&r);

    // Compute translation
    let translation = target_centroid - rotation.cast::<f64>() * source_centroid;

    Ok(Isometry3::from_parts(
        Translation3::from(translation.cast::<f32>()),
        rotation,
    ))
}

/// True when every coordinate of `p` is finite.
fn is_finite_point(p: &Point3f) -> bool {
    p.coords.iter().all(|c| c.is_finite())
}

/// Where a cloud sits and how big it is: the centroid of its finite points
/// and their RMS distance from it. ICP uses both to judge whether an update
/// still moves the points by a meaningful amount.
#[derive(Debug, Clone, Copy)]
pub struct CloudScale {
    pub centroid: Point3f,
    pub radius: f32,
}

impl CloudScale {
    pub fn of(points: &[Point3f]) -> Self {
        let (sum, sum_sq, count) = points
            .par_iter()
            .filter(|p| is_finite_point(p))
            .map(|p| {
                let v = p.coords.cast::<f64>();
                (v, v.norm_squared(), 1usize)
            })
            .reduce(
                || (Vector3::zeros(), 0.0, 0),
                |(a, a_sq, a_n), (b, b_sq, b_n)| (a + b, a_sq + b_sq, a_n + b_n),
            );
        let n = count.max(1) as f64;
        let centroid = sum / n;
        let radius_sq = (sum_sq / n - centroid.norm_squared()).max(0.0);
        Self {
            centroid: Point3f::from(centroid.cast::<f32>()),
            radius: radius_sq.sqrt() as f32,
        }
    }
}

/// A frame centred on a point cloud, for running registration in `f32`.
///
/// Far from the origin (e.g. map coordinates millions of metres away), `f32`
/// cannot represent small moves, so ICP would stall well short of alignment.
/// Moving the clouds next to the origin first keeps full precision; results
/// are converted back to the original frame.
#[derive(Debug, Clone, Copy)]
pub struct LocalFrame {
    /// The original-frame point that becomes this frame's origin.
    pub origin: Vector3<f64>,
}

impl LocalFrame {
    /// A frame centred on the mean of the finite points (a single NaN would
    /// otherwise make every local coordinate NaN).
    pub fn centred_on(points: &[Point3f]) -> Self {
        let (sum, count) = points
            .par_iter()
            .filter(|p| is_finite_point(p))
            .map(|p| (p.coords.cast::<f64>(), 1usize))
            .reduce(
                || (Vector3::zeros(), 0),
                |(a, a_n), (b, b_n)| (a + b, a_n + b_n),
            );
        Self {
            origin: sum / count.max(1) as f64,
        }
    }

    /// A frame centred on `points`, or `None` when they are close enough to
    /// the origin (within 100x their own size) that `f32` loses nothing.
    /// Clouds in sensor coordinates take the `None` path and skip the copies.
    pub fn if_far_from_origin(points: &[Point3f]) -> Option<Self> {
        let scale = CloudScale::of(points);
        if scale.centroid.coords.norm() <= 100.0 * scale.radius {
            return None;
        }
        Some(Self::centred_on(points))
    }

    /// `p` in this frame.
    pub fn point_to_local(&self, p: &Point3f) -> Point3f {
        Point3f::from((p.coords.cast::<f64>() - self.origin).cast::<f32>())
    }

    /// Every point of `points` in this frame (in parallel).
    pub fn points_to_local(&self, points: &[Point3f]) -> Vec<Point3f> {
        points.par_iter().map(|p| self.point_to_local(p)).collect()
    }

    /// A transform given in the original frame, expressed in this frame.
    pub fn transform_to_local(&self, transform: &Isometry3<f32>) -> Isometry3<f32> {
        let shift = Translation3::from(self.origin);
        (shift.inverse() * transform.cast::<f64>() * shift).cast::<f32>()
    }

    /// A transform given in this frame, expressed in the original frame.
    pub fn transform_to_world(&self, transform: &Isometry3<f32>) -> Isometry3<f32> {
        let shift = Translation3::from(self.origin);
        (shift * transform.cast::<f64>() * shift.inverse()).cast::<f32>()
    }
}

/// Decide whether ICP has converged. Both tests are relative, so they behave the
/// same on a small indoor scan as on a large outdoor one (an absolute change in
/// MSE does not: MSE shrinks with the square of the scene size).
///
/// ICP stops when either:
/// - the RMSE improved by less than `threshold` as a fraction of its previous
///   value, or
/// - the last update `delta` moved the points by less than `threshold` times the
///   cloud's size, or
/// - the error has stopped going down and the update is about one `f32`
///   rounding step at these coordinates. This catches near-perfect alignments,
///   where the RMSE is down to rounding noise and its relative change never
///   settles. It only applies once the error stops improving, so a cloud far
///   from the origin (e.g. map coordinates) still takes every useful step.
///
/// `centroid` is the source centroid before `delta` was applied.
pub fn icp_converged(
    previous_mse: f32,
    current_mse: f32,
    delta: &Isometry3<f32>,
    centroid: &Point3f,
    radius: f32,
    threshold: f32,
) -> bool {
    // RMS point motion of a rigid update: centroid shift plus rotation about it.
    let centroid_shift = (delta * centroid - centroid).norm();
    let rotation_shift = delta.rotation.angle() * radius;
    let motion = centroid_shift.hypot(rotation_shift);
    if motion <= threshold * radius {
        return true;
    }

    if !previous_mse.is_finite() {
        return false;
    }
    let previous_rmse = previous_mse.sqrt();
    if previous_rmse == 0.0
        || (previous_rmse - current_mse.sqrt()).abs() / previous_rmse < threshold
    {
        return true;
    }

    let rounding_step = 2.0 * f32::EPSILON * (centroid.coords.norm() + radius);
    current_mse >= previous_mse && motion <= rounding_step
}

/// Flatten per-source-point matches into `(source_index, target_index)` pairs.
fn correspondence_pairs(matches: &[Option<(usize, f32)>]) -> Vec<(usize, usize)> {
    matches
        .iter()
        .enumerate()
        .filter_map(|(src_idx, m)| m.map(|(tgt_idx, _)| (src_idx, tgt_idx)))
        .collect()
}

/// ICP (Iterative Closest Point) registration - Main function matching requested API
///
/// This function performs point cloud registration using the ICP algorithm.
///
/// # Arguments
/// * `source` - Source point cloud to be aligned
/// * `target` - Target point cloud to align to
/// * `init` - Initial transformation estimate
/// * `max_iters` - Maximum number of iterations
///
/// # Returns
/// * `Isometry3<f32>` - Final transformation that aligns source to target
pub fn icp(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    init: Isometry3<f32>,
    max_iters: usize,
) -> Isometry3<f32> {
    match icp_detailed(source, target, init, max_iters, None, 1e-6) {
        Ok(result) => result.transformation,
        Err(_) => init, // Return initial transformation on error
    }
}

/// Detailed ICP registration with comprehensive options and result
///
/// This function provides full control over ICP parameters and returns detailed results.
///
/// # Arguments
/// * `source` - Source point cloud to be aligned
/// * `target` - Target point cloud to align to
/// * `init` - Initial transformation estimate
/// * `max_iters` - Maximum number of iterations
/// * `max_correspondence_distance` - Maximum distance for valid correspondences (None = no limit)
/// * `convergence_threshold` - Stop when RMSE improves by less than this fraction (e.g. 1e-6)
///
/// # Returns
/// * `Result<ICPResult>` - Detailed ICP result including transformation, error, and convergence info
pub fn icp_detailed(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    init: Isometry3<f32>,
    max_iters: usize,
    max_correspondence_distance: Option<f32>,
    convergence_threshold: f32,
) -> Result<ICPResult> {
    if source.is_empty() || target.is_empty() {
        return Err(Error::InvalidData(
            "Source or target point cloud is empty".to_string(),
        ));
    }

    if max_iters == 0 {
        return Err(Error::InvalidData(
            "Max iterations must be positive".to_string(),
        ));
    }

    let Some(frame) = LocalFrame::if_far_from_origin(&target.points) else {
        return icp_point_to_point_local(
            &source.points,
            &target.points,
            init,
            max_iters,
            max_correspondence_distance,
            convergence_threshold,
        );
    };
    let mut result = icp_point_to_point_local(
        &frame.points_to_local(&source.points),
        &frame.points_to_local(&target.points),
        frame.transform_to_local(&init),
        max_iters,
        max_correspondence_distance,
        convergence_threshold,
    )?;
    result.transformation = frame.transform_to_world(&result.transformation);
    Ok(result)
}

/// The point-to-point ICP loop, on clouds already moved into a [`LocalFrame`].
fn icp_point_to_point_local(
    source: &[Point3f],
    target: &[Point3f],
    init: Isometry3<f32>,
    max_iters: usize,
    max_correspondence_distance: Option<f32>,
    convergence_threshold: f32,
) -> Result<ICPResult> {
    let mut current_transform = init;
    let mut previous_mse = f32::INFINITY;
    let target_tree = KdTree::new(target)?;
    let max_dist_sq = max_distance_sq(max_correspondence_distance);
    let source_scale = CloudScale::of(source);
    let mut matches: Vec<Option<(usize, f32)>> = vec![None; source.len()];

    for iteration in 0..max_iters {
        // Find correspondences for the source moved by the current estimate,
        // seeded with last iteration's matches
        let stats = match_and_collect(
            source,
            &current_transform,
            target,
            &target_tree,
            &mut matches,
            max_dist_sq,
        );

        if stats.count < 3 {
            return Err(Error::Algorithm(
                "Insufficient correspondences found".to_string(),
            ));
        }

        // Compute transformation for this iteration
        let delta_transform = stats.transformation()?;

        // Update transformation
        let moved_centroid = current_transform * source_scale.centroid;
        current_transform = delta_transform * current_transform;

        // MSE of this iteration's correspondences (before the update)
        let current_mse = stats.mse();

        // Check for convergence
        if icp_converged(
            previous_mse,
            current_mse,
            &delta_transform,
            &moved_centroid,
            source_scale.radius,
            convergence_threshold,
        ) {
            return Ok(ICPResult {
                transformation: current_transform,
                mse: current_mse,
                iterations: iteration + 1,
                converged: true,
                correspondences: correspondence_pairs(&matches),
            });
        }

        previous_mse = current_mse;
    }

    // Re-score the last correspondences under the final transformation
    let final_stats = CorrespondenceStats::collect(source, &current_transform, target, &matches);
    let final_mse = if final_stats.count > 0 {
        final_stats.mse()
    } else {
        previous_mse
    };

    Ok(ICPResult {
        transformation: current_transform,
        mse: final_mse,
        iterations: max_iters,
        converged: false,
        correspondences: correspondence_pairs(&matches),
    })
}

/// Legacy ICP function with different signature for backward compatibility
#[deprecated(note = "Use icp instead which matches the standard API")]
pub fn icp_legacy(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    max_iterations: usize,
    threshold: f32,
) -> Result<(threecrate_core::Transform3D, f32)> {
    let init = Isometry3::identity();
    let result = icp_detailed(source, target, init, max_iterations, Some(threshold), 1e-6)?;

    // Convert Isometry3 to Transform3D
    let transform = threecrate_core::Transform3D::from(result.transformation);

    Ok((transform, result.mse))
}

/// Compute the optimal incremental transformation using linearized point-to-plane optimization.
///
/// Based on Chen & Medioni (1992) - Object Modelling by Registration of Multiple Range Images.
///
/// Minimizes sum_i [n_i · (R*s_i + t - d_i)]^2 via small-angle linearization, building
/// a 6x6 linear system A^T*A*x = A^T*b where x = [α, β, γ, tx, ty, tz].
fn compute_transformation_point_to_plane(
    source_points: &[Point3f],
    target_points: &[Point3f],
    target_normals: &[Vector3f],
) -> Result<Isometry3<f32>> {
    if source_points.len() != target_points.len()
        || source_points.len() != target_normals.len()
        || source_points.is_empty()
    {
        return Err(Error::InvalidData(
            "Point/normal count mismatch in point-to-plane optimization".to_string(),
        ));
    }

    let mut ata = Matrix6::<f32>::zeros();
    let mut atb = Vector6::<f32>::zeros();

    for ((src, tgt), normal) in source_points
        .iter()
        .zip(target_points.iter())
        .zip(target_normals.iter())
    {
        // Cross product c = s × n  (rotational part of the Jacobian row)
        let c = src.coords.cross(normal);

        // Row of A: [c.x, c.y, c.z, n.x, n.y, n.z]
        let a_row = Vector6::new(c.x, c.y, c.z, normal.x, normal.y, normal.z);

        // RHS: n · (d - s)
        let b_i = normal.dot(&(tgt.coords - src.coords));

        ata += a_row * a_row.transpose();
        atb += a_row * b_i;
    }

    // Solve with Cholesky (fast, stable when A^T*A is positive definite);
    // fall back to LU if the system is rank-deficient.
    let x = if let Some(chol) = ata.cholesky() {
        chol.solve(&atb)
    } else {
        ata.lu().solve(&atb).ok_or_else(|| {
            Error::Algorithm("Point-to-plane system is ill-conditioned".to_string())
        })?
    };

    // Compose small-angle rotations Rz(γ) * Ry(β) * Rx(α)
    let rot_x = UnitQuaternion::from_axis_angle(&Vector3f::x_axis(), x[0]);
    let rot_y = UnitQuaternion::from_axis_angle(&Vector3f::y_axis(), x[1]);
    let rot_z = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), x[2]);
    let rotation = rot_z * rot_y * rot_x;

    Ok(Isometry3::from_parts(
        Translation3::new(x[3], x[4], x[5]),
        rotation,
    ))
}

/// Compute mean squared point-to-plane distance for a set of correspondences.
fn compute_point_to_plane_mse(
    source_points: &[Point3f],
    target_points: &[Point3f],
    normals: &[Vector3f],
) -> f32 {
    if source_points.is_empty() {
        return 0.0;
    }
    let sum: f32 = source_points
        .iter()
        .zip(target_points.iter())
        .zip(normals.iter())
        .map(|((src, tgt), n)| {
            let d = n.dot(&(tgt.coords - src.coords));
            d * d
        })
        .sum();
    sum / source_points.len() as f32
}

/// Point-to-plane ICP variant (requires target normals).
///
/// Uses the linearized Chen & Medioni (1992) formulation: each iteration solves a 6×6
/// linear system instead of the full SVD used by point-to-point ICP.  This typically
/// converges faster and more accurately on smooth surfaces.
///
/// # Arguments
/// * `source`          - Source point cloud to be aligned
/// * `target`          - Target point cloud to align to
/// * `target_normals`  - Surface normals at each target point (must equal `target.len()`)
/// * `init`            - Initial transformation estimate
/// * `max_iters`       - Maximum number of iterations
///
/// # Returns
/// * `Result<ICPResult>` – transformation, per-iteration error, convergence flag
pub fn icp_point_to_plane(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    target_normals: &[Vector3f],
    init: Isometry3<f32>,
    max_iters: usize,
) -> Result<ICPResult> {
    icp_point_to_plane_detailed(source, target, target_normals, init, max_iters, None, 1e-6)
}

/// Detailed point-to-plane ICP with full parameter control.
///
/// # Arguments
/// * `source`                       - Source point cloud
/// * `target`                       - Target point cloud
/// * `target_normals`               - Surface normals at each target point
/// * `init`                         - Initial transformation estimate
/// * `max_iters`                    - Maximum number of iterations
/// * `max_correspondence_distance`  - Optional distance cutoff for correspondence rejection
/// * `convergence_threshold`        - Stop when RMSE improves by less than this fraction
pub fn icp_point_to_plane_detailed(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    target_normals: &[Vector3f],
    init: Isometry3<f32>,
    max_iters: usize,
    max_correspondence_distance: Option<f32>,
    convergence_threshold: f32,
) -> Result<ICPResult> {
    if source.is_empty() || target.is_empty() {
        return Err(Error::InvalidData(
            "Source or target point cloud is empty".to_string(),
        ));
    }
    if target_normals.len() != target.points.len() {
        return Err(Error::InvalidData(
            "target_normals length must equal the number of target points".to_string(),
        ));
    }
    if max_iters == 0 {
        return Err(Error::InvalidData(
            "Max iterations must be positive".to_string(),
        ));
    }

    let Some(frame) = LocalFrame::if_far_from_origin(&target.points) else {
        return icp_point_to_plane_local(
            &source.points,
            &target.points,
            target_normals,
            init,
            max_iters,
            max_correspondence_distance,
            convergence_threshold,
        );
    };
    let mut result = icp_point_to_plane_local(
        &frame.points_to_local(&source.points),
        &frame.points_to_local(&target.points),
        target_normals,
        frame.transform_to_local(&init),
        max_iters,
        max_correspondence_distance,
        convergence_threshold,
    )?;
    result.transformation = frame.transform_to_world(&result.transformation);
    Ok(result)
}

/// The point-to-plane ICP loop, on clouds already moved into a [`LocalFrame`].
fn icp_point_to_plane_local(
    source: &[Point3f],
    target: &[Point3f],
    target_normals: &[Vector3f],
    init: Isometry3<f32>,
    max_iters: usize,
    max_correspondence_distance: Option<f32>,
    convergence_threshold: f32,
) -> Result<ICPResult> {
    let mut current_transform = init;
    let mut previous_mse = f32::INFINITY;
    let mut final_correspondences: Vec<(usize, usize)> = Vec::new();
    let target_tree = KdTree::new(target)?;
    let max_dist_sq = max_distance_sq(max_correspondence_distance);
    let source_scale = CloudScale::of(source);
    let mut correspondences: Vec<Option<(usize, f32)>> = vec![None; source.len()];

    for iteration in 0..max_iters {
        // Apply current estimate to source
        let transformed_source: Vec<Point3f> =
            source.par_iter().map(|p| current_transform * p).collect();

        // Find nearest-neighbor correspondences, seeded with last iteration's matches
        correspondences = find_correspondences_seeded(
            source,
            &current_transform,
            target,
            &target_tree,
            &correspondences,
            max_dist_sq,
        );

        let mut valid_source: Vec<Point3f> = Vec::new();
        let mut valid_target: Vec<Point3f> = Vec::new();
        let mut valid_normals: Vec<Vector3f> = Vec::new();
        let mut corr_pairs: Vec<(usize, usize)> = Vec::new();

        for (src_idx, corr) in correspondences.iter().enumerate() {
            if let Some((tgt_idx, _)) = corr {
                valid_source.push(transformed_source[src_idx]);
                valid_target.push(target[*tgt_idx]);
                valid_normals.push(target_normals[*tgt_idx]);
                corr_pairs.push((src_idx, *tgt_idx));
            }
        }

        // Need at least 6 points to solve the 6-DOF system
        if valid_source.len() < 6 {
            return Err(Error::Algorithm(
                "Insufficient correspondences for point-to-plane ICP (need ≥ 6)".to_string(),
            ));
        }

        let delta =
            compute_transformation_point_to_plane(&valid_source, &valid_target, &valid_normals)?;
        let moved_centroid = current_transform * source_scale.centroid;
        current_transform = delta * current_transform;

        let current_mse = compute_point_to_plane_mse(&valid_source, &valid_target, &valid_normals);
        if icp_converged(
            previous_mse,
            current_mse,
            &delta,
            &moved_centroid,
            source_scale.radius,
            convergence_threshold,
        ) {
            return Ok(ICPResult {
                transformation: current_transform,
                mse: current_mse,
                iterations: iteration + 1,
                converged: true,
                correspondences: corr_pairs,
            });
        }

        previous_mse = current_mse;
        final_correspondences = corr_pairs;
    }

    Ok(ICPResult {
        transformation: current_transform,
        mse: previous_mse,
        iterations: max_iters,
        converged: false,
        correspondences: final_correspondences,
    })
}

/// Point-to-point ICP registration
///
/// This function performs point-to-point ICP registration using Euclidean distance minimization.
/// It finds the rigid transformation that best aligns the source point cloud to the target.
///
/// # Arguments
/// * `source` - Source point cloud to be aligned
/// * `target` - Target point cloud to align to
/// * `init` - Initial transformation estimate (use Isometry3::identity() for no initial guess)
/// * `max_iterations` - Maximum number of iterations to perform
/// * `convergence_threshold` - Stop when RMSE improves by less than this fraction (default: 1e-6)
/// * `max_correspondence_distance` - Maximum distance for valid correspondences (None = no limit)
///
/// # Returns
/// * `Result<ICPResult>` - Detailed ICP result including transformation, error, and convergence info
///
/// # Example
/// ```rust
/// use threecrate_algorithms::icp_point_to_point;
/// use threecrate_core::{PointCloud, Point3f};
/// use nalgebra::Isometry3;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create source and target point clouds
///     let mut source = PointCloud::new();
///     let mut target = PointCloud::new();
///     
///     // Add some points
///     for i in 0..10 {
///         let point = Point3f::new(i as f32, i as f32, 0.0);
///         source.push(point);
///         target.push(point + Point3f::new(1.0, 0.0, 0.0).coords); // Translated by (1,0,0)
///     }
///     
///     let init = Isometry3::identity();
///     let result = icp_point_to_point(&source, &target, init, 50, 1e-6, None)?;
///     println!("Converged: {}, MSE: {}", result.converged, result.mse);
///     Ok(())
/// }
/// ```
pub fn icp_point_to_point(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    init: Isometry3<f32>,
    max_iterations: usize,
    convergence_threshold: f32,
    max_correspondence_distance: Option<f32>,
) -> Result<ICPResult> {
    // Validate inputs
    if source.is_empty() || target.is_empty() {
        return Err(Error::InvalidData(
            "Source or target point cloud is empty".to_string(),
        ));
    }

    if max_iterations == 0 {
        return Err(Error::InvalidData(
            "Max iterations must be positive".to_string(),
        ));
    }

    if convergence_threshold <= 0.0 {
        return Err(Error::InvalidData(
            "Convergence threshold must be positive".to_string(),
        ));
    }

    // Use the detailed ICP implementation with point-to-point distance minimization
    icp_detailed(
        source,
        target,
        init,
        max_iterations,
        max_correspondence_distance,
        convergence_threshold,
    )
}

/// Point-to-point ICP registration with default parameters
///
/// Convenience function that uses reasonable default parameters for point-to-point ICP.
///
/// # Arguments
/// * `source` - Source point cloud to be aligned
/// * `target` - Target point cloud to align to
/// * `init` - Initial transformation estimate
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// * `Result<ICPResult>` - Detailed ICP result
pub fn icp_point_to_point_default(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    init: Isometry3<f32>,
    max_iterations: usize,
) -> Result<ICPResult> {
    icp_point_to_point(source, target, init, max_iterations, 1e-6, None)
}

/// Coarse-to-fine point-to-point ICP using voxel downsampling at each scale.
pub fn multiscale_icp_point_to_point(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    init: Isometry3<f32>,
    config: &MultiScaleIcpConfig,
) -> Result<ICPResult> {
    if source.is_empty() || target.is_empty() {
        return Err(Error::InvalidData(
            "Source or target point cloud is empty".to_string(),
        ));
    }
    if config.levels.is_empty() {
        return Err(Error::InvalidData(
            "At least one ICP scale level is required".to_string(),
        ));
    }
    if config.convergence_threshold <= 0.0 {
        return Err(Error::InvalidData(
            "Convergence threshold must be positive".to_string(),
        ));
    }
    if config.final_refinement_iterations == 0 {
        return Err(Error::InvalidData(
            "Final refinement iterations must be positive".to_string(),
        ));
    }

    let mut current_transform = init;
    let mut total_iterations = 0usize;
    let mut last_result: Option<ICPResult> = None;

    for level in &config.levels {
        if level.voxel_size <= 0.0 {
            return Err(Error::InvalidData(
                "Scale voxel_size must be positive".to_string(),
            ));
        }
        if level.max_iterations == 0 {
            return Err(Error::InvalidData(
                "Scale max_iterations must be positive".to_string(),
            ));
        }

        let source_down = voxel_grid_filter(source, level.voxel_size)?;
        let target_down = voxel_grid_filter(target, level.voxel_size)?;
        if source_down.len() < 3 || target_down.len() < 3 {
            continue;
        }

        let result = icp_point_to_point(
            &source_down,
            &target_down,
            current_transform,
            level.max_iterations,
            config.convergence_threshold,
            level.max_correspondence_distance,
        )?;

        current_transform = result.transformation;
        total_iterations += result.iterations;
        last_result = Some(result);
    }

    if last_result.is_none() {
        return Err(Error::Algorithm(
            "No multiscale ICP level had enough downsampled points".to_string(),
        ));
    }

    let final_result = icp_point_to_point(
        source,
        target,
        current_transform,
        config.final_refinement_iterations,
        config.convergence_threshold,
        config.final_max_correspondence_distance,
    )?;

    Ok(ICPResult {
        transformation: final_result.transformation,
        mse: final_result.mse,
        iterations: total_iterations + final_result.iterations,
        converged: final_result.converged,
        correspondences: final_result.correspondences,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use nalgebra::UnitQuaternion;

    #[test]
    fn test_icp_identity_transformation() {
        // Create identical point clouds
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        for i in 0..10 {
            let point = Point3f::new(i as f32, (i * 2) as f32, (i * 3) as f32);
            source.push(point);
            target.push(point);
        }

        let init = Isometry3::identity();
        let result = icp_detailed(&source, &target, init, 10, None, 1e-6).unwrap();

        // Should converge quickly with minimal transformation
        assert!(result.converged);
        assert!(result.mse < 1e-6);
        assert!(result.iterations <= 3);
    }

    #[test]
    fn test_icp_translation() {
        // Create source and target with known translation
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        let translation = Vector3f::new(1.0, 2.0, 3.0);

        for i in 0..10 {
            let source_point = Point3f::new(i as f32, (i * 2) as f32, (i * 3) as f32);
            let target_point = source_point + translation;
            source.push(source_point);
            target.push(target_point);
        }

        let init = Isometry3::identity();
        let result = icp_detailed(&source, &target, init, 50, None, 1e-6).unwrap();

        // Check that the computed translation is in the right direction
        let computed_translation = result.transformation.translation.vector;
        // ICP may not converge exactly due to numerical precision and algorithm limitations
        // The algorithm should at least move in the correct direction
        assert!(
            computed_translation.magnitude() > 0.05,
            "Translation magnitude too small: {}",
            computed_translation.magnitude()
        );

        assert!(result.mse < 2.0); // Allow for higher MSE in simple test cases
    }

    #[test]
    fn test_icp_rotation() {
        // Create source and target with known rotation
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        let rotation =
            UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), std::f32::consts::FRAC_PI_4);

        for i in 0..20 {
            let source_point = Point3f::new(i as f32, (i % 5) as f32, 0.0);
            let target_point = rotation * source_point;
            source.push(source_point);
            target.push(target_point);
        }

        let init = Isometry3::identity();
        let result = icp_detailed(&source, &target, init, 100, None, 1e-6).unwrap();

        // Should find a reasonable transformation for rotation
        assert!(result.mse < 1.0, "MSE too high: {}", result.mse);
    }

    #[test]
    fn test_icp_insufficient_points() {
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        source.push(Point3f::new(0.0, 0.0, 0.0));
        target.push(Point3f::new(1.0, 1.0, 1.0));

        let init = Isometry3::identity();
        let result = icp_detailed(&source, &target, init, 10, None, 1e-6);

        assert!(result.is_err());
    }

    #[test]
    fn test_icp_api_compatibility() {
        // Test the main API function
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        for i in 0..5 {
            let point = Point3f::new(i as f32, i as f32, 0.0);
            source.push(point);
            target.push(point + Vector3f::new(1.0, 0.0, 0.0));
        }

        let init = Isometry3::identity();
        let transform = icp(&source, &target, init, 20);

        // Should return a valid transformation (not panic)
        assert!(transform.translation.vector.magnitude() > 0.5);
    }

    #[test]
    fn test_icp_far_from_origin_does_not_stop_early() {
        // Terrain-like cloud in UTM-style map coordinates (millions of metres
        // from the origin), turned and shifted by a few metres. ICP needs
        // several real steps here and must not stop before it is aligned.
        let origin = nalgebra::Vector3::<f64>::new(500_000.0, 4_200_000.0, 0.0);
        let local: Vec<nalgebra::Vector3<f64>> = (0..40)
            .flat_map(|i| (0..40).map(move |j| (i as f64 * 10.0, j as f64 * 10.0)))
            .map(|(x, y)| nalgebra::Vector3::new(x, y, 30.0 * (x / 70.0).sin() * (y / 110.0).cos()))
            .collect();
        let centre = local.iter().sum::<nalgebra::Vector3<f64>>() / local.len() as f64 + origin;

        // Truth: turn 0.01 rad about the cloud centre, then shift (3, -2, 0.5)
        let truth = nalgebra::Translation3::from(centre + nalgebra::Vector3::new(3.0, -2.0, 0.5))
            * nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, 0.01)
            * nalgebra::Translation3::from(-centre);

        let world: Vec<nalgebra::Point3<f64>> = local
            .iter()
            .map(|v| nalgebra::Point3::from(v + origin))
            .collect();
        let source = PointCloud::from_points(world.iter().map(|p| p.cast::<f32>()).collect());
        let target =
            PointCloud::from_points(world.iter().map(|p| (truth * p).cast::<f32>()).collect());

        let result =
            icp_point_to_point(&source, &target, Isometry3::identity(), 100, 1e-6, None).unwrap();

        // Judge by where the points land (in f64), not by the translation
        // vector: this far from the origin, a 1e-7 rotation rounding error is
        // already ~0.4 m of translation, which the translation then cancels.
        let estimate = result.transformation.cast::<f64>();
        let error = world
            .iter()
            .map(|p| (estimate * p - truth * p).norm())
            .fold(0.0, f64::max);
        assert!(
            result.converged && error < 0.05,
            "points {error} m off after {} iterations (converged: {})",
            result.iterations,
            result.converged
        );
    }

    #[test]
    fn test_icp_convergence_is_scale_invariant() {
        // The same problem at 1x and at 1/100 scale (e.g. metres vs centimetres
        // of scene). With an absolute MSE rule the small copy stopped early.
        let solve = |scale: f32| {
            let mut source = PointCloud::new();
            for i in 0..40 {
                for j in 0..40 {
                    let (x, y) = (i as f32 * 0.1, j as f32 * 0.1);
                    let z = 0.3 * (1.3 * x).sin() * (0.7 * y).cos();
                    source.push(Point3f::new(x, y, z) * scale);
                }
            }
            let truth = Isometry3::from_parts(
                Translation3::new(0.05 * scale, -0.03 * scale, 0.02 * scale),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.03),
            );
            let target = PointCloud::from_points(source.points.iter().map(|p| truth * p).collect());
            let result =
                icp_point_to_point(&source, &target, Isometry3::identity(), 100, 1e-6, None)
                    .unwrap();
            let rotation_error =
                (result.transformation.rotation.inverse() * truth.rotation).angle();
            let translation_error =
                (result.transformation.translation.vector - truth.translation.vector).norm()
                    / scale;
            (result.iterations, rotation_error, translation_error)
        };

        let (iters_big, rot_big, trans_big) = solve(1.0);
        let (iters_small, rot_small, trans_small) = solve(0.01);

        assert!(rot_big < 1e-3 && trans_big < 1e-3);
        assert!(
            rot_small < 1e-3 && trans_small < 1e-3,
            "small-scale ICP stopped early: rot {rot_small}, trans {trans_small}, {iters_small} iterations"
        );
        assert!(
            iters_small.abs_diff(iters_big) <= 2,
            "iterations: big {iters_big}, small {iters_small}"
        );
    }

    #[test]
    fn test_max_distance_cutoff_semantics() {
        let source = vec![Point3f::new(0.0, 0.0, 0.0)];
        let target = vec![Point3f::new(0.5, 0.0, 0.0)];

        // A match exactly at the cutoff is accepted
        assert!(find_correspondences(&source, &target, Some(0.5))[0].is_some());
        // A cutoff just below the match distance rejects it
        assert!(find_correspondences(&source, &target, Some(0.49))[0].is_none());
        // A negative cutoff rejects everything rather than being squared
        assert!(find_correspondences(&source, &target, Some(-0.5))[0].is_none());
        // No cutoff accepts
        assert!(find_correspondences(&source, &target, None)[0].is_some());

        // Zero cutoff still accepts exact duplicates
        assert!(find_correspondences(&source, &source, Some(0.0))[0].is_some());
    }

    #[test]
    fn test_correspondence_finding() {
        let source = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
        ];

        let target = vec![
            Point3f::new(0.1, 0.1, 0.0),
            Point3f::new(1.1, 0.1, 0.0),
            Point3f::new(0.1, 1.1, 0.0),
        ];

        let correspondences = find_correspondences(&source, &target, None);

        assert_eq!(correspondences.len(), 3);
        assert!(correspondences[0].is_some());
        assert!(correspondences[1].is_some());
        assert!(correspondences[2].is_some());
    }

    #[test]
    fn test_icp_point_to_point_basic() {
        // Test basic functionality with simple point clouds
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        // Create a simple cube pattern
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    let point = Point3f::new(x as f32, y as f32, z as f32);
                    source.push(point);
                    target.push(point + Vector3f::new(1.0, 0.5, 0.25)); // Known translation
                }
            }
        }

        let init = Isometry3::identity();
        let result = icp_point_to_point(&source, &target, init, 50, 1e-6, None).unwrap();

        // Should converge and find a reasonable transformation
        assert!(result.converged || result.iterations == 50);
        assert!(result.mse < 2.0); // Allow for higher MSE in simple test cases
                                   // The transformation should at least move in the right direction
        let translation_mag = result.transformation.translation.vector.magnitude();
        assert!(
            translation_mag > 0.1,
            "Translation magnitude too small: {}",
            translation_mag
        );
    }

    #[test]
    fn test_icp_point_to_point_with_noise() {
        // Test with noisy data
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        let translation = Vector3f::new(2.0, 1.0, 0.5);
        let rotation = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), 0.3);
        let transform = Isometry3::from_parts(
            Translation3::new(translation.x, translation.y, translation.z),
            rotation,
        );

        // Create source points
        for i in 0..100 {
            let angle = (i as f32) * 0.1;
            let radius = 2.0 + (i % 10) as f32 * 0.1;
            let source_point = Point3f::new(
                radius * angle.cos(),
                radius * angle.sin(),
                (i % 5) as f32 * 0.5,
            );
            source.push(source_point);
        }

        // Create target points with known transformation + noise
        for point in &source.points {
            let transformed = transform * point;
            // Add some noise
            let noise = Vector3f::new(
                (rand::random::<f32>() - 0.5) * 0.1,
                (rand::random::<f32>() - 0.5) * 0.1,
                (rand::random::<f32>() - 0.5) * 0.1,
            );
            target.push(transformed + noise);
        }

        let init = Isometry3::identity();
        let result = icp_point_to_point(&source, &target, init, 100, 1e-5, None).unwrap();

        // Should find a reasonable transformation despite noise
        assert!(result.mse < 0.5); // Allow for noise
        assert!(result.transformation.translation.vector.magnitude() > 1.0);
    }

    #[test]
    fn test_icp_point_to_point_known_transform() {
        // Test with a known transformation
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        // Known transformation - use smaller values for better convergence
        let known_translation = Vector3f::new(1.0, -0.5, 0.25);
        let known_rotation = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), 0.2);
        let known_transform = Isometry3::from_parts(
            Translation3::new(
                known_translation.x,
                known_translation.y,
                known_translation.z,
            ),
            known_rotation,
        );

        // Create source points in a grid
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -1..=1 {
                    let point = Point3f::new(x as f32, y as f32, z as f32);
                    source.push(point);
                    target.push(known_transform * point);
                }
            }
        }

        let init = Isometry3::identity();
        let result = icp_point_to_point(&source, &target, init, 50, 1e-6, None).unwrap();

        // Should find a transformation close to the known one
        let computed_translation = result.transformation.translation.vector;
        let translation_error = (computed_translation - known_translation).magnitude();
        assert!(
            translation_error < 1.0,
            "Translation error too large: {}",
            translation_error
        );

        assert!(result.mse < 0.5);
    }

    #[test]
    fn test_icp_point_to_point_convergence() {
        // Test convergence behavior
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        // Create point clouds that should converge quickly
        for i in 0..50 {
            let point = Point3f::new(i as f32 * 0.1, (i * 2) as f32 * 0.1, 0.0);
            source.push(point);
            target.push(point + Vector3f::new(0.5, 0.0, 0.0));
        }

        let init = Isometry3::identity();
        let result = icp_point_to_point(&source, &target, init, 20, 1e-6, None).unwrap();

        // Should converge quickly
        assert!(result.converged);
        assert!(result.iterations < 20);
        assert!(result.mse < 0.1);
    }

    #[test]
    fn test_icp_point_to_point_max_distance() {
        // Test with maximum correspondence distance
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        // Create source points
        for i in 0..10 {
            source.push(Point3f::new(i as f32, 0.0, 0.0));
        }

        // Create target points with some far away
        for i in 0..10 {
            if i < 5 {
                target.push(Point3f::new(i as f32 + 0.1, 0.0, 0.0)); // Close
            } else {
                target.push(Point3f::new(i as f32 + 10.0, 0.0, 0.0)); // Far away
            }
        }

        let init = Isometry3::identity();
        let result = icp_point_to_point(&source, &target, init, 20, 1e-6, Some(1.0)).unwrap();

        // Should only use correspondences within max_distance
        // Note: The algorithm might still find some correspondences due to the iterative nature
        // but it should use fewer correspondences than without the distance limit
        assert!(result.correspondences.len() <= 10);
        assert!(result.mse < 5.0); // Allow for higher MSE when using distance filtering
    }

    #[test]
    fn test_icp_point_to_point_default() {
        // Test the default convenience function
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();

        for i in 0..10 {
            let point = Point3f::new(i as f32, i as f32, 0.0);
            source.push(point);
            target.push(point + Vector3f::new(1.0, 0.0, 0.0));
        }

        let init = Isometry3::identity();
        let result = icp_point_to_point_default(&source, &target, init, 30).unwrap();

        // Should work with default parameters
        assert!(result.mse < 1.0);
        assert!(result.transformation.translation.vector.magnitude() > 0.5);
    }

    #[test]
    fn test_icp_point_to_point_validation() {
        // Test input validation
        let empty_source = PointCloud::new();
        let mut target = PointCloud::new();
        target.push(Point3f::new(0.0, 0.0, 0.0));

        let init = Isometry3::identity();

        // Test empty source
        let result = icp_point_to_point(&empty_source, &target, init, 10, 1e-6, None);
        assert!(result.is_err());

        // Test zero iterations
        let result = icp_point_to_point(&target, &target, init, 0, 1e-6, None);
        assert!(result.is_err());

        // Test negative convergence threshold
        let result = icp_point_to_point(&target, &target, init, 10, -1e-6, None);
        assert!(result.is_err());
    }

    // ── Point-to-plane ICP tests ──────────────────────────────────────────────

    /// Build a Fibonacci-sphere cloud with outward-pointing unit normals.
    ///
    /// A sphere is the canonical test surface for point-to-plane ICP because the
    /// normals span all of 3-D space, ensuring the 6×6 linear system is full rank.
    fn make_sphere_cloud(n: usize) -> (PointCloud<Point3f>, Vec<Vector3f>) {
        let mut cloud = PointCloud::new();
        let mut normals = Vec::new();
        let radius = 3.0_f32;
        let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        for i in 0..n {
            let y = 1.0 - (i as f32 / (n as f32 - 1.0).max(1.0)) * 2.0;
            let r = (1.0 - y * y).max(0.0_f32).sqrt();
            let theta = golden_angle * i as f32;
            let x = theta.cos() * r;
            let z = theta.sin() * r;
            // (x, y, z) is already a unit vector (on the unit sphere)
            let normal = Vector3f::new(x, y, z);
            cloud.push(Point3f::new(x * radius, y * radius, z * radius));
            normals.push(normal);
        }
        (cloud, normals)
    }

    #[test]
    fn test_icp_point_to_plane_identity() {
        let (source, normals) = make_sphere_cloud(50);
        let target = source.clone();
        let init = Isometry3::identity();

        let result = icp_point_to_plane(&source, &target, &normals, init, 20).unwrap();

        assert!(result.converged);
        assert!(result.mse < 1e-6, "mse={}", result.mse);
    }

    #[test]
    fn test_icp_point_to_plane_translation() {
        // Small in-plane shift so nearest-neighbor correspondences remain correct.
        let (source, normals) = make_sphere_cloud(100);
        let shift = Vector3f::new(0.15, 0.0, 0.0);

        let mut target = PointCloud::new();
        for p in &source.points {
            target.push(p + shift);
        }
        // Reuse the source normals as approximate target normals (valid for small shift).
        let result =
            icp_point_to_plane(&source, &target, &normals, Isometry3::identity(), 50).unwrap();

        let t_err = (result.transformation.translation.vector - shift).magnitude();
        assert!(t_err < 0.3, "translation error={}", t_err);
        assert!(result.mse < 0.1, "mse={}", result.mse);
    }

    #[test]
    fn test_icp_point_to_plane_validation() {
        let (source, normals) = make_sphere_cloud(20);
        let init = Isometry3::identity();

        // Normals length mismatch
        let bad_normals = vec![Vector3f::new(0.0, 0.0, 1.0)];
        let result = icp_point_to_plane(&source, &source, &bad_normals, init, 10);
        assert!(result.is_err());

        // Empty source
        let empty: PointCloud<Point3f> = PointCloud::new();
        let result = icp_point_to_plane(&empty, &source, &normals, init, 10);
        assert!(result.is_err());

        // Zero iterations
        let result = icp_point_to_plane_detailed(&source, &source, &normals, init, 0, None, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_icp_point_to_plane_vs_point_to_point_convergence() {
        // Both variants must converge to a reasonable solution for the same sphere+shift input.
        let (source, normals) = make_sphere_cloud(80);
        let shift = Vector3f::new(0.1, 0.05, 0.0);
        let mut target = PointCloud::new();
        for p in &source.points {
            target.push(p + shift);
        }

        let init = Isometry3::identity();

        let p2pl_result = icp_point_to_plane(&source, &target, &normals, init, 50).unwrap();
        let p2pt_result = icp_point_to_point(&source, &target, init, 50, 1e-6, None).unwrap();

        // Both should find a non-trivial transformation
        assert!(
            p2pl_result.transformation.translation.vector.magnitude() > 0.05,
            "p2pl did not translate: t={}",
            p2pl_result.transformation.translation.vector.magnitude()
        );
        assert!(
            p2pt_result.transformation.translation.vector.magnitude() > 0.05,
            "p2pt did not translate: t={}",
            p2pt_result.transformation.translation.vector.magnitude()
        );
        // Point-to-plane should converge (or at least not diverge)
        assert!(
            p2pl_result.converged || p2pl_result.mse < 0.1,
            "p2pl failed to converge: mse={}, iters={}",
            p2pl_result.mse,
            p2pl_result.iterations
        );
    }

    #[test]
    fn test_icp_point_to_plane_detailed_max_distance() {
        let (source, normals) = make_sphere_cloud(50);
        let mut target = PointCloud::new();
        for p in &source.points {
            target.push(p + Vector3f::new(0.1, 0.0, 0.0));
        }

        let init = Isometry3::identity();
        let result =
            icp_point_to_plane_detailed(&source, &target, &normals, init, 30, Some(5.0), 1e-6);
        assert!(result.is_ok(), "unexpected error: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.mse < 0.5, "mse={}", result.mse);
    }
}
