//! KISS-ICP: Keep It Simple and Straightforward ICP.
//!
//! Bai et al., *KISS-ICP: In Defense of Point-to-Point ICP -- Simple, Accurate,
//! and Robust Registration If Done the Right Way*, IROS 2023.
//! <https://arxiv.org/abs/2209.15397>
//!
//! Key ideas (from the paper):
//! 1. **Range filtering** — discard points outside `[min_range, max_range]`.
//! 2. **Voxel downsampling** of the source scan with `voxel_size`.
//! 3. **Adaptive correspondence threshold** σ — derived from recent motion magnitude
//!    so the user never needs to tune a distance parameter.
//! 4. **Standard point-to-point ICP** with the adaptive threshold.
//!    No covariance matrices, no normals, no complex data structures.

use nalgebra::{Matrix3, Translation3, UnitQuaternion};
use threecrate_core::{Error, Isometry3, NearestNeighborSearch, Point3f, PointCloud, Result};

use crate::filtering::voxel_grid_filter;
use crate::nearest_neighbor::KdTree;
use crate::registration::ICPResult;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parameters for [`kiss_icp`].
#[derive(Debug, Clone)]
pub struct KissIcpConfig {
    /// Voxel size (metres) used to downsample the source scan.
    /// Also anchors the adaptive correspondence-distance threshold.
    pub voxel_size: f32,
    /// Discard source points farther than this from the sensor origin.
    pub max_range: f32,
    /// Discard source points closer than this from the sensor origin (removes ego-vehicle noise).
    pub min_range: f32,
    /// Maximum ICP iterations per scan.
    pub max_iterations: usize,
}

impl Default for KissIcpConfig {
    fn default() -> Self {
        Self {
            voxel_size: 1.0,
            max_range: 100.0,
            min_range: 0.5,
            max_iterations: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Remove points whose Euclidean distance from the origin is outside [min_r, max_r].
fn range_filter(cloud: &PointCloud<Point3f>, min_r: f32, max_r: f32) -> PointCloud<Point3f> {
    let min_sq = min_r * min_r;
    let max_sq = max_r * max_r;
    PointCloud::from_points(
        cloud
            .points
            .iter()
            .filter(|p| {
                let r2 = p.coords.magnitude_squared();
                r2 >= min_sq && r2 <= max_sq
            })
            .copied()
            .collect(),
    )
}

/// Compute the adaptive correspondence threshold σ.
///
/// In the online KISS-ICP pipeline σ is updated per-scan from the deviation
/// between the predicted and actual motion.  For this stateless (single-pair)
/// API we approximate it from the magnitude of `init`:
///
/// `σ = clamp(3 · ‖init‖, 3·voxel_size, 10·voxel_size)`
///
/// When `init = I` (no prior), σ defaults to `3·voxel_size`, which matches the
/// paper's recommendation for the first frame.
fn adaptive_threshold(init: &Isometry3<f32>, voxel_size: f32) -> f32 {
    let trans = init.translation.vector.magnitude();
    // Convert rotation to an equivalent linear displacement at the scale of one
    // voxel.  Using the quaternion's imaginary-part magnitude (= sin(θ/2)) keeps
    // the result dimensionally consistent with `trans` (both in metres) and
    // stays bounded in [0, voxel_size] for all rotation angles.
    //   • θ → 0  : rot_disp ≈ (θ/2) · voxel_size  (linear, small-angle)
    //   • θ = π  : rot_disp = voxel_size            (largest rotation)
    let rot_disp = 2.0 * init.rotation.imag().magnitude() * voxel_size;
    let motion = trans + rot_disp;
    // Lower bound: always allow 3 voxels; upper bound: 10 voxels to stay sane.
    (3.0 * motion).max(3.0 * voxel_size).min(10.0 * voxel_size)
}

/// SVD-based rigid transform: Procrustes / Kabsch algorithm.
fn svd_transform(src: &[Point3f], tgt: &[Point3f]) -> Result<Isometry3<f32>> {
    debug_assert_eq!(src.len(), tgt.len());
    if src.len() < 3 {
        return Err(Error::Algorithm(
            "KISS-ICP SVD: need at least 3 point pairs for a rigid transform".into(),
        ));
    }
    let n = src.len() as f32;

    let src_mean = src
        .iter()
        .fold(nalgebra::Vector3::zeros(), |a, p| a + p.coords)
        / n;
    let tgt_mean = tgt
        .iter()
        .fold(nalgebra::Vector3::zeros(), |a, p| a + p.coords)
        / n;

    let mut h = Matrix3::<f32>::zeros();
    for (s, t) in src.iter().zip(tgt.iter()) {
        h += (s.coords - src_mean) * (t.coords - tgt_mean).transpose();
    }

    // Degenerate correspondence set (all points collinear / identical after centering).
    if h.norm() < 1e-10 {
        return Err(Error::Algorithm(
            "KISS-ICP SVD: cross-covariance matrix H is near-zero — \
             all correspondence points are identical or collinear"
                .into(),
        ));
    }

    let svd = h.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| Error::Algorithm("KISS-ICP SVD failed (U)".into()))?;
    let vt = svd
        .v_t
        .ok_or_else(|| Error::Algorithm("KISS-ICP SVD failed (V^T)".into()))?;

    let mut r = vt.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut vt_fix = vt;
        vt_fix.set_row(2, &(-vt.row(2)));
        r = vt_fix.transpose() * u.transpose();
    }

    let rotation = UnitQuaternion::from_matrix(&r);
    let translation = tgt_mean - rotation * src_mean;

    Ok(Isometry3::from_parts(
        Translation3::new(translation.x, translation.y, translation.z),
        rotation,
    ))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Align `source` to `target` using KISS-ICP.
///
/// # Algorithm
///
/// 1. Range-filter `source` to keep points with `min_range ≤ ‖p‖ ≤ max_range`.
/// 2. Voxel-downsample the filtered source with `voxel_size`.
/// 3. Compute an adaptive correspondence threshold σ from the magnitude of `init`.
/// 4. Iteratively:
///    a. Transform source with the current estimate.
///    b. Find nearest-neighbour correspondences within σ.
///    c. Compute the optimal rigid transform via SVD (Kabsch).
///    d. Update the current estimate.
///    e. Declare convergence when |ΔMSE| < 1 × 10⁻⁶.
///
/// # Notes
///
/// * In a full SLAM system, `init` is typically supplied by a constant-velocity
///   or IMU motion model; σ then adapts across scans.  For a single-pair call
///   pass `Isometry3::identity()` and σ defaults to `3 × voxel_size`.
/// * `source` and `target` do **not** need to be pre-filtered or downsampled —
///   the function handles that internally.
///
/// # References
///
/// Bai et al. (2023). *KISS-ICP: In Defense of Point-to-Point ICP.*
/// IEEE Robotics and Automation Letters / IROS 2023.
pub fn kiss_icp(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    init: Isometry3<f32>,
    config: KissIcpConfig,
) -> Result<ICPResult> {
    if source.is_empty() || target.is_empty() {
        return Err(Error::InvalidData(
            "KISS-ICP: source or target point cloud is empty".into(),
        ));
    }
    if config.max_iterations == 0 {
        return Err(Error::InvalidData(
            "KISS-ICP: max_iterations must be > 0".into(),
        ));
    }
    if config.voxel_size <= 0.0 {
        return Err(Error::InvalidData(
            "KISS-ICP: voxel_size must be > 0".into(),
        ));
    }

    // --- Step 1: Preprocess source scan ---
    let source_ranged = range_filter(source, config.min_range, config.max_range);
    if source_ranged.is_empty() {
        return Err(Error::InvalidData(
            "KISS-ICP: no source points remain after range filtering — \
             check min_range / max_range relative to your data"
                .into(),
        ));
    }
    let source_down = voxel_grid_filter(&source_ranged, config.voxel_size)?;
    if source_down.is_empty() {
        return Err(Error::InvalidData(
            "KISS-ICP: no source points remain after voxel downsampling".into(),
        ));
    }

    // --- Step 2: Adaptive threshold ---
    let sigma = adaptive_threshold(&init, config.voxel_size);

    // --- Step 3: KD-tree for the target ---
    let target_tree = KdTree::new(&target.points)?;

    let mut current_transform = init;
    let mut prev_mse = f32::INFINITY;
    let mut final_corr: Vec<(usize, usize)> = Vec::new();

    for iteration in 0..config.max_iterations {
        // Apply current estimate to the (downsampled) source.
        let transformed: Vec<Point3f> = source_down
            .points
            .iter()
            .map(|p| current_transform * p)
            .collect();

        // Collect valid correspondences within σ.
        let mut src_matched: Vec<Point3f> = Vec::new();
        let mut tgt_matched: Vec<Point3f> = Vec::new();
        let mut corr_pairs: Vec<(usize, usize)> = Vec::new();

        for (src_idx, ts) in transformed.iter().enumerate() {
            let neighbours = target_tree.find_k_nearest(ts, 1);
            if neighbours.is_empty() {
                continue;
            }
            let (tgt_idx, dist) = neighbours[0];
            if dist > sigma {
                continue;
            }
            src_matched.push(*ts);
            tgt_matched.push(target.points[tgt_idx]);
            corr_pairs.push((src_idx, tgt_idx));
        }

        if src_matched.len() < 3 {
            return Err(Error::Algorithm(
                "KISS-ICP: too few correspondences within the adaptive threshold — \
                 try increasing voxel_size or check point cloud overlap"
                    .into(),
            ));
        }

        // Compute optimal rigid transform via SVD.
        let delta = svd_transform(&src_matched, &tgt_matched)?;
        current_transform = delta * current_transform;

        // MSE over correspondences (using points *after* applying delta).
        let mse: f32 = src_matched
            .iter()
            .zip(tgt_matched.iter())
            .map(|(s, t)| (delta * s - t).magnitude_squared())
            .sum::<f32>()
            / src_matched.len() as f32;

        if (prev_mse - mse).abs() < 1e-6 {
            return Ok(ICPResult {
                transformation: current_transform,
                mse,
                iterations: iteration + 1,
                converged: true,
                correspondences: corr_pairs,
            });
        }

        prev_mse = mse;
        final_corr = corr_pairs;
    }

    Ok(ICPResult {
        transformation: current_transform,
        mse: prev_mse,
        iterations: config.max_iterations,
        converged: false,
        correspondences: final_corr,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use threecrate_core::Vector3f;

    /// Fibonacci sphere with irregular point spacing — good for rotation tests
    /// because there are no repeated angular distances that create false NN matches.
    /// All points at `radius` from the origin, so they pass KISS-ICP's range filter.
    fn make_sphere(n: usize, radius: f32) -> PointCloud<Point3f> {
        let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        PointCloud::from_points(
            (0..n)
                .map(|i| {
                    let y = 1.0 - (i as f32 / (n as f32 - 1.0).max(1.0)) * 2.0;
                    let r = (1.0 - y * y).max(0.0_f32).sqrt();
                    let theta = golden * i as f32;
                    Point3f::new(
                        theta.cos() * r * radius,
                        y * radius,
                        theta.sin() * r * radius,
                    )
                })
                .collect(),
        )
    }

    /// Dense flat grid — good for pure translation tests.
    fn make_grid(n: usize, spacing: f32, z_offset: f32) -> PointCloud<Point3f> {
        let side = (n as f32).sqrt().ceil() as usize;
        PointCloud::from_points(
            (0..side)
                .flat_map(|i| {
                    (0..side).map(move |j| {
                        Point3f::new(i as f32 * spacing, j as f32 * spacing, z_offset)
                    })
                })
                .take(n)
                .collect(),
        )
    }

    /// All points at range 5 m (outside default [0.5, 100] so they pass the filter).
    fn make_ring(n: usize, range: f32) -> PointCloud<Point3f> {
        PointCloud::from_points(
            (0..n)
                .map(|i| {
                    let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
                    Point3f::new(angle.cos() * range, angle.sin() * range, 0.0)
                })
                .collect(),
        )
    }

    #[test]
    fn kiss_icp_identity_converges() {
        // source == target, init == I → should converge to identity in few iters.
        let cloud = make_ring(200, 5.0);
        let config = KissIcpConfig {
            voxel_size: 0.2,
            max_range: 50.0,
            min_range: 0.1,
            max_iterations: 30,
        };
        let result = kiss_icp(&cloud, &cloud, Isometry3::identity(), config).unwrap();
        assert!(result.converged || result.mse < 1e-4, "mse={}", result.mse);
    }

    #[test]
    fn kiss_icp_recovers_small_translation() {
        // Shift must be < half the grid spacing (0.5/2 = 0.25) so each source point's
        // nearest target neighbour is unambiguously the "correct" one.
        let source = make_grid(100, 0.5, 5.0); // all points at z=5 → within [0.5, 100]
        let shift = Vector3f::new(0.1, 0.0, 0.0);
        let target = PointCloud::from_points(source.points.iter().map(|p| p + shift).collect());

        let config = KissIcpConfig {
            voxel_size: 0.2,
            max_range: 50.0,
            min_range: 0.1,
            max_iterations: 50,
        };
        let result = kiss_icp(&source, &target, Isometry3::identity(), config).unwrap();

        let t_err = (result.transformation.translation.vector - shift).magnitude();
        assert!(t_err < 0.05, "translation error={}", t_err);
        assert!(result.mse < 0.1, "mse={}", result.mse);
    }

    #[test]
    fn kiss_icp_empty_source_errors() {
        let empty: PointCloud<Point3f> = PointCloud::new();
        let cloud = make_ring(30, 5.0);
        assert!(kiss_icp(
            &empty,
            &cloud,
            Isometry3::identity(),
            KissIcpConfig::default()
        )
        .is_err());
    }

    #[test]
    fn kiss_icp_all_points_outside_range_errors() {
        // Points at range 0.05 m (< min_range 0.5) → filtered to empty.
        let cloud = make_ring(50, 0.05);
        let config = KissIcpConfig {
            min_range: 0.5,
            ..Default::default()
        };
        assert!(kiss_icp(&cloud, &cloud, Isometry3::identity(), config).is_err());
    }

    #[test]
    fn kiss_icp_zero_voxel_size_errors() {
        let cloud = make_ring(30, 5.0);
        let config = KissIcpConfig {
            voxel_size: 0.0,
            ..Default::default()
        };
        assert!(kiss_icp(&cloud, &cloud, Isometry3::identity(), config).is_err());
    }

    #[test]
    fn kiss_icp_result_fields_populated() {
        let cloud = make_ring(60, 5.0);
        let config = KissIcpConfig {
            voxel_size: 0.3,
            max_range: 50.0,
            min_range: 0.1,
            max_iterations: 10,
        };
        let result = kiss_icp(&cloud, &cloud, Isometry3::identity(), config).unwrap();
        assert!(result.iterations > 0);
        assert!(!result.correspondences.is_empty());
    }

    #[test]
    fn adaptive_threshold_no_prior() {
        // init = I → motion magnitude = 0 → σ = 3 * voxel_size
        let sigma = adaptive_threshold(&Isometry3::identity(), 1.0);
        assert_relative_eq!(sigma, 3.0, epsilon = 1e-5);
    }

    #[test]
    fn range_filter_removes_out_of_range() {
        let pts = vec![
            Point3f::new(0.1, 0.0, 0.0),   // too close (range=0.1)
            Point3f::new(5.0, 0.0, 0.0),   // ok
            Point3f::new(200.0, 0.0, 0.0), // too far
        ];
        let cloud = PointCloud::from_points(pts);
        let filtered = range_filter(&cloud, 0.5, 100.0);
        assert_eq!(filtered.len(), 1);
        assert_relative_eq!(filtered.points[0].x, 5.0, epsilon = 1e-5);
    }

    // -------------------------------------------------------------------------
    // Adaptive threshold — fixed unit-consistency
    // -------------------------------------------------------------------------

    #[test]
    fn adaptive_threshold_large_rotation_stays_bounded() {
        // 90° rotation: sin(45°) = √2/2 ≈ 0.707
        // rot_disp = 2 · 0.707 · 1.0 ≈ 1.414 m
        // motion = 0 + 1.414 = 1.414
        // σ = clamp(3·1.414, 3, 10) = clamp(4.24, 3, 10) = 4.24
        let rot90 = Isometry3::from_parts(
            nalgebra::Translation3::identity(),
            nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::z_axis(),
                std::f32::consts::FRAC_PI_2,
            ),
        );
        let sigma = adaptive_threshold(&rot90, 1.0);
        assert!(sigma >= 3.0, "sigma below lower bound: {}", sigma);
        assert!(sigma <= 10.0, "sigma above upper bound: {}", sigma);
        // With the old (broken) code: σ = clamp(3·(0 + π/2), 3, 10) = 4.71
        // The new code should give a smaller, dimensionally-consistent value ≈ 4.24.
        assert!(
            sigma < 4.5,
            "sigma suspiciously large — possible unit mixing: {}",
            sigma
        );
    }

    #[test]
    fn adaptive_threshold_180_deg_saturates_at_voxel_scale() {
        // 180° rotation: sin(90°) = 1 → rot_disp = 2 · 1 · 1 = 2 m
        // σ = clamp(6, 3, 10) = 6
        let rot180 = Isometry3::from_parts(
            nalgebra::Translation3::identity(),
            nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::z_axis(),
                std::f32::consts::PI,
            ),
        );
        let sigma = adaptive_threshold(&rot180, 1.0);
        assert!(sigma <= 10.0, "sigma exceeds 10·voxel_size: {}", sigma);
        assert!(sigma >= 3.0);
    }

    // -------------------------------------------------------------------------
    // Rotation recovery
    //
    // KISS-ICP is a LOCAL method.  A Fibonacci sphere (irregular point spacing,
    // ~11° average angular gap for 300 points at radius 5 m) avoids the
    // false-nearest-neighbour problem that regular grids/rings have.
    //
    // Rule of thumb: use rotation < half the average angular spacing for
    // identity-initialised tests.  For larger rotations, supply a near-correct
    // initial guess (the real SLAM use-case where IMU/odometry provides a prior).
    // -------------------------------------------------------------------------

    #[test]
    fn kiss_icp_recovers_tiny_rotation_from_identity() {
        // Fibonacci sphere: 300 points at radius 5 → avg angular spacing ≈ 11°,
        // half ≈ 5.5°.  3° rotation is well within the convergence basin.
        // σ = 3·voxel_size = 1.5 m; chord at 5 m for 3° = 0.26 m < 1.5 m ✓
        let source = make_sphere(300, 5.0);
        let angle = 3_f32.to_radians();
        let rot = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), angle);
        let target = PointCloud::from_points(source.points.iter().map(|p| rot * p).collect());

        let config = KissIcpConfig {
            voxel_size: 0.5,
            max_range: 50.0,
            min_range: 0.1,
            max_iterations: 60,
        };
        let result = kiss_icp(&source, &target, Isometry3::identity(), config).unwrap();

        let rot_err = result.transformation.rotation.angle_to(&rot);
        assert!(
            rot_err < 1_f32.to_radians(),
            "rotation error = {:.2}°",
            rot_err.to_degrees()
        );
        let t_err = result.transformation.translation.vector.magnitude();
        assert!(t_err < 0.1, "spurious translation={}", t_err);
    }

    #[test]
    fn kiss_icp_refines_rotation_from_near_correct_init() {
        // SLAM use-case: IMU/odometry gives a 6° initial guess for a true 8°
        // rotation.  Residual 2° < 5.5° half-spacing → correct correspondences.
        let source = make_sphere(300, 5.0);
        let true_angle = 8_f32.to_radians();
        let init_angle = 6_f32.to_radians();
        let true_rot =
            nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), true_angle);
        let target = PointCloud::from_points(source.points.iter().map(|p| true_rot * p).collect());
        let init = Isometry3::from_parts(
            nalgebra::Translation3::identity(),
            nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), init_angle),
        );

        let config = KissIcpConfig {
            voxel_size: 0.5,
            max_range: 50.0,
            min_range: 0.1,
            max_iterations: 60,
        };
        let result = kiss_icp(&source, &target, init, config).unwrap();

        let rot_err = result.transformation.rotation.angle_to(&true_rot);
        assert!(
            rot_err < 1_f32.to_radians(),
            "rotation error = {:.2}°",
            rot_err.to_degrees()
        );
    }

    // -------------------------------------------------------------------------
    // Noise robustness
    // -------------------------------------------------------------------------

    #[test]
    fn kiss_icp_robust_to_gaussian_noise() {
        // Source with ±0.05 m deterministic noise; clean target.
        let base = make_ring(200, 5.0);
        let noise_half = 0.05_f32;
        let source = PointCloud::from_points(
            base.points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let t = (i as f32 * 1.6180339887_f32).sin() * noise_half;
                    let u = (i as f32 * 2.7182818284_f32).cos() * noise_half;
                    let v = (i as f32 * 3.1415926535_f32).sin() * noise_half;
                    Point3f::new(p.x + t, p.y + u, p.z + v)
                })
                .collect(),
        );

        let config = KissIcpConfig {
            voxel_size: 0.2,
            max_range: 50.0,
            min_range: 0.1,
            max_iterations: 40,
        };
        let result = kiss_icp(&source, &base, Isometry3::identity(), config).unwrap();

        // Voxel downsampling averages noise; expect small residual.
        assert!(result.mse < 0.05, "mse={}", result.mse);
        let t_err = result.transformation.translation.vector.magnitude();
        assert!(t_err < 0.1, "spurious drift={}", t_err);
    }

    // -------------------------------------------------------------------------
    // Degenerate inputs
    // -------------------------------------------------------------------------

    #[test]
    fn kiss_icp_collinear_source_recovers_translation() {
        // All source points collinear along the X axis.  The SVD H matrix is
        // rank-1 (only the X component is determined), so rotation is degenerate
        // but the translation along X should still be recovered correctly.
        let source = PointCloud::from_points(
            (0..30)
                .map(|i| Point3f::new(i as f32 * 0.3 + 1.0, 0.0, 5.0))
                .collect(),
        );
        let shift_x = 0.05_f32;
        let target = PointCloud::from_points(
            source
                .points
                .iter()
                .map(|p| Point3f::new(p.x + shift_x, p.y, p.z))
                .collect(),
        );
        let config = KissIcpConfig {
            voxel_size: 0.1,
            max_range: 50.0,
            min_range: 0.1,
            max_iterations: 30,
        };
        // Should succeed (rank-1 SVD is valid); translation along X should be close.
        let result = kiss_icp(&source, &target, Isometry3::identity(), config).unwrap();
        let tx = result.transformation.translation.vector.x;
        assert!(
            (tx - shift_x).abs() < 0.02,
            "x translation error={}",
            (tx - shift_x).abs()
        );
    }

    #[test]
    fn kiss_icp_identical_points_reduced_to_one_errors() {
        // 20 identical points → voxel downsampling leaves 1 → <3 correspondences.
        let source = PointCloud::from_points(
            std::iter::repeat(Point3f::new(5.0, 0.0, 0.0))
                .take(20)
                .collect(),
        );
        let config = KissIcpConfig {
            min_range: 0.1,
            max_range: 50.0,
            voxel_size: 1.0,
            max_iterations: 5,
        };
        assert!(
            kiss_icp(&source, &source, Isometry3::identity(), config).is_err(),
            "expected error for single-point cloud after downsampling"
        );
    }
}
