//! Generalized ICP (GICP) registration algorithm.
//!
//! GICP (Segal et al., 2009) extends standard ICP by modelling each point as a local
//! Gaussian distribution whose covariance is estimated from its k nearest neighbours.
//! Each correspondence is weighted by the combined source + target covariance, giving
//! much stronger robustness to noise than point-to-point or point-to-plane ICP.
//!
//! Used in: LOAM, LIO-SAM, and most modern LiDAR odometry pipelines.

use nalgebra::{Matrix3, Matrix6, Translation3, UnitQuaternion, Vector6};
use rayon::prelude::*;
use threecrate_core::{
    Error, Isometry3, NearestNeighborSearch, Point3f, PointCloud, Result, Vector3f,
};

use crate::nearest_neighbor::KdTree;
use crate::registration::ICPResult;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parameters for [`gicp`].
#[derive(Debug, Clone)]
pub struct GicpConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: usize,
    /// Maximum Euclidean distance for accepting a source–target correspondence.
    pub max_correspondence_distance: f32,
    /// Convergence threshold: stop when |ΔMSE| < this value.
    pub convergence_threshold: f32,
    /// Number of nearest neighbours used to estimate per-point covariance matrices.
    pub k_correspondences: usize,
}

impl Default for GicpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            convergence_threshold: 1e-6,
            k_correspondences: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Skew-symmetric ("cross-product") matrix of a 3-D vector.
#[inline]
fn skew_sym(v: &nalgebra::Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Estimate per-point 3×3 covariance matrices from k nearest neighbours.
///
/// Returns a `Vec` of length `points.len()`, aligned index-for-index.
fn compute_covariances(points: &[Point3f], k: usize) -> Result<Vec<Matrix3<f32>>> {
    let k = k.max(4); // need ≥ 4 neighbours for a non-degenerate 3-D covariance
    let tree = KdTree::new(points)?;

    let covs: Vec<Matrix3<f32>> = points
        .par_iter()
        .map(|p| {
            let neighbours = tree.find_k_nearest(p, k);
            let n = neighbours.len();
            if n < 3 {
                // Sparse neighbourhood: isotropic fallback.
                return Matrix3::identity() * 1e-3_f32;
            }

            let nf = n as f32;
            let mean = neighbours
                .iter()
                .map(|(idx, _)| points[*idx].coords)
                .fold(nalgebra::Vector3::zeros(), |acc, v| acc + v)
                / nf;

            let mut cov = Matrix3::zeros();
            for (idx, _) in &neighbours {
                let d = points[*idx].coords - mean;
                cov += d * d.transpose();
            }
            let mut cov = cov / (nf - 1.0).max(1.0);
            // Regularise: add a small isotropic term so that locally planar or
            // collinear neighbourhoods still have an invertible covariance matrix.
            cov += Matrix3::identity() * 1e-4_f32;
            cov
        })
        .collect();

    Ok(covs)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Align `source` to `target` using Generalized ICP.
///
/// # Algorithm
///
/// 1. Estimate per-point covariances for both clouds using k-NN.
/// 2. Iteratively:
///    a. Transform source with the current estimate.
///    b. Find nearest-neighbour correspondences within `max_correspondence_distance`.
///    c. Build the weighted 6×6 Gauss-Newton system using combined covariances
///       `M_i = C_i^T + R C_i^S R^T`.
///    d. Solve for the incremental 6-DOF update and compose with the current transform.
///    e. Declare convergence when |ΔMSE| < `convergence_threshold`.
///
/// # References
/// Segal, A., Haehnel, D., & Thrun, S. (2009). *Generalized-ICP.*
/// Robotics: Science and Systems.
pub fn gicp(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    init: Isometry3<f32>,
    config: GicpConfig,
) -> Result<ICPResult> {
    if source.is_empty() || target.is_empty() {
        return Err(Error::InvalidData(
            "GICP: source or target point cloud is empty".into(),
        ));
    }
    if config.max_iterations == 0 {
        return Err(Error::InvalidData(
            "GICP: max_iterations must be > 0".into(),
        ));
    }
    // Covariance estimation needs at least k neighbours per point; fewer points
    // than k_correspondences produces a degenerate (global) covariance.
    let min_k = config.k_correspondences.max(4);
    if source.len() < min_k || target.len() < min_k {
        return Err(Error::InvalidData(format!(
            "GICP: clouds must have at least {} points for reliable covariance estimation \
             (k_correspondences={}); got source={}, target={}",
            min_k,
            config.k_correspondences,
            source.len(),
            target.len()
        )));
    }
    // Reject degenerate clouds where all points lie on a plane or line — GICP
    // requires 3-D structure to build meaningful per-point covariance matrices.
    for (label, cloud) in [("source", source), ("target", target)] {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &cloud.points {
            for ax in 0..3 {
                min[ax] = min[ax].min(p.coords[ax]);
                max[ax] = max[ax].max(p.coords[ax]);
            }
        }
        let min_extent = (0..3)
            .map(|ax| max[ax] - min[ax])
            .fold(f32::INFINITY, f32::min);
        if min_extent < 1e-4 {
            return Err(Error::InvalidData(format!(
                "GICP: {label} point cloud appears to be coplanar or collinear \
                 (smallest bounding-box dimension = {min_extent:.2e}); \
                 GICP requires 3-D structure"
            )));
        }
    }

    // Step 1: per-point covariances.
    let source_covs = compute_covariances(&source.points, config.k_correspondences)?;
    let target_covs = compute_covariances(&target.points, config.k_correspondences)?;

    // Step 2: KD-tree over target.
    let target_tree = KdTree::new(&target.points)?;

    let mut current_transform = init;
    let mut prev_mse = f32::INFINITY;
    let mut final_corr: Vec<(usize, usize)> = Vec::new();

    for iteration in 0..config.max_iterations {
        // Transform source points with the current estimate.
        let transformed_source: Vec<Point3f> = source
            .points
            .iter()
            .map(|p| current_transform * p)
            .collect();

        // Rotation matrix for covariance transformation.
        let r_mat = *current_transform.rotation.to_rotation_matrix().matrix();

        // Accumulate the 6×6 Gauss-Newton Hessian H and gradient g.
        // Pose parameterisation: x = [δω_x, δω_y, δω_z, δt_x, δt_y, δt_z]
        // (rotation first, matching the existing point-to-plane convention).
        let mut h = Matrix6::<f32>::zeros();
        let mut gvec = Vector6::<f32>::zeros();
        let mut n_corr = 0usize;
        let mut mse_sum = 0.0f32;
        let mut corr_pairs: Vec<(usize, usize)> = Vec::new();

        for (src_idx, (ts, c_s)) in transformed_source
            .iter()
            .zip(source_covs.iter())
            .enumerate()
        {
            let neighbours = target_tree.find_k_nearest(ts, 1);
            if neighbours.is_empty() {
                continue;
            }
            let (tgt_idx, dist) = neighbours[0];
            if dist > config.max_correspondence_distance {
                continue;
            }

            let c_t = &target_covs[tgt_idx];

            // Combined covariance M = C_t + R C_s R^T.
            let m = c_t + r_mat * c_s * r_mat.transpose();
            let m_inv = match m.try_inverse() {
                Some(inv) => inv,
                None => continue, // numerically degenerate — skip
            };

            // Residual r = t_i – T s_i.
            let residual = target.points[tgt_idx].coords - ts.coords;

            // 3×6 Jacobian J = [A | I] where A = −skew(T s_i).
            // Contribution to H and g:
            //   H_rr (0..3, 0..3) = A^T M⁻¹ A
            //   H_rt (0..3, 3..6) = A^T M⁻¹
            //   H_tr (3..6, 0..3) = M⁻¹ A = H_rt^T
            //   H_tt (3..6, 3..6) = M⁻¹
            //   g_r  (0..3)       = A^T M⁻¹ r
            //   g_t  (3..6)       = M⁻¹ r

            let a = -skew_sym(&ts.coords); // rotation block
            let h_rr = a.transpose() * (m_inv * a);
            let h_rt = a.transpose() * m_inv;
            let wr = m_inv * residual;
            let g_r = a.transpose() * wr;

            for i in 0..3 {
                for j in 0..3 {
                    h[(i, j)] += h_rr[(i, j)];
                    h[(i, j + 3)] += h_rt[(i, j)];
                    h[(i + 3, j)] += h_rt[(j, i)]; // H_tr = H_rt^T
                    h[(i + 3, j + 3)] += m_inv[(i, j)];
                }
                gvec[i] += g_r[i];
                gvec[i + 3] += wr[i];
            }

            n_corr += 1;
            mse_sum += dist * dist;
            corr_pairs.push((src_idx, tgt_idx));
        }

        if n_corr < 6 {
            return Err(Error::Algorithm(
                "GICP: insufficient correspondences (need ≥ 6)".into(),
            ));
        }

        let mse = mse_sum / n_corr as f32;

        // Solve H δ = g (Cholesky preferred; LU fallback).
        let delta = if let Some(chol) = h.cholesky() {
            chol.solve(&gvec)
        } else {
            h.lu().solve(&gvec).ok_or_else(|| {
                Error::Algorithm("GICP: Gauss-Newton system is ill-conditioned".into())
            })?
        };

        // Compose incremental transform δT from x = [δω_x, δω_y, δω_z, δt_x, δt_y, δt_z].
        let delta_rot = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), delta[2])
            * UnitQuaternion::from_axis_angle(&Vector3f::y_axis(), delta[1])
            * UnitQuaternion::from_axis_angle(&Vector3f::x_axis(), delta[0]);
        let delta_iso =
            Isometry3::from_parts(Translation3::new(delta[3], delta[4], delta[5]), delta_rot);
        current_transform = delta_iso * current_transform;

        // Convergence check.
        if (prev_mse - mse).abs() < config.convergence_threshold {
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
    #[allow(unused_imports)]
    use nalgebra::Translation3;

    /// Fibonacci-sphere point cloud with outward unit normals (not used by GICP
    /// directly, but a good well-conditioned test surface).
    fn make_sphere(n: usize, radius: f32) -> PointCloud<Point3f> {
        let mut cloud = PointCloud::new();
        let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        for i in 0..n {
            let y = 1.0 - (i as f32 / (n as f32 - 1.0).max(1.0)) * 2.0;
            let r = (1.0 - y * y).max(0.0_f32).sqrt();
            let theta = golden * i as f32;
            cloud.push(Point3f::new(
                theta.cos() * r * radius,
                y * radius,
                theta.sin() * r * radius,
            ));
        }
        cloud
    }

    #[test]
    fn gicp_identity_converges() {
        let cloud = make_sphere(100, 3.0);
        let config = GicpConfig {
            max_iterations: 30,
            ..Default::default()
        };
        let result = gicp(&cloud, &cloud, Isometry3::identity(), config).unwrap();
        assert!(result.converged, "should converge for identical clouds");
        assert!(result.mse < 1e-4, "mse={}", result.mse);
    }

    #[test]
    fn gicp_recovers_small_translation() {
        let source = make_sphere(150, 3.0);
        let shift = Vector3f::new(0.1, 0.0, 0.0);
        let target = PointCloud::from_points(source.points.iter().map(|p| p + shift).collect());

        let config = GicpConfig {
            max_iterations: 60,
            max_correspondence_distance: 2.0,
            ..Default::default()
        };
        let result = gicp(&source, &target, Isometry3::identity(), config).unwrap();

        let t_err = (result.transformation.translation.vector - shift).magnitude();
        assert!(t_err < 0.05, "translation error={}", t_err);
        assert!(result.mse < 0.1, "mse={}", result.mse);
    }

    #[test]
    fn gicp_empty_source_errors() {
        let empty: PointCloud<Point3f> = PointCloud::new();
        let cloud = make_sphere(30, 1.0);
        assert!(gicp(&empty, &cloud, Isometry3::identity(), GicpConfig::default()).is_err());
    }

    #[test]
    fn gicp_zero_iterations_errors() {
        let cloud = make_sphere(30, 1.0);
        let config = GicpConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(gicp(&cloud, &cloud, Isometry3::identity(), config).is_err());
    }

    #[test]
    fn gicp_result_fields_populated() {
        let cloud = make_sphere(60, 2.0);
        let config = GicpConfig {
            max_iterations: 10,
            ..Default::default()
        };
        let result = gicp(&cloud, &cloud, Isometry3::identity(), config).unwrap();
        assert!(result.iterations > 0);
        assert!(!result.correspondences.is_empty());
    }

    // -------------------------------------------------------------------------
    // Rotation recovery
    //
    // ICP is a LOCAL optimizer: it refines an initial guess rather than solving
    // the global alignment problem.  Tests use either:
    //   (a) identity init + tiny rotation (< half the average point spacing), or
    //   (b) a near-correct initial guess for larger rotations, matching the real
    //       SLAM use-case where IMU/odometry supplies a rough estimate.
    // -------------------------------------------------------------------------

    #[test]
    fn gicp_recovers_tiny_rotation_from_identity() {
        // 300-point sphere → average angular spacing ≈ 11°.
        // 2° rotation chord ≈ 0.21 m — well within the half-spacing convergence
        // basin so every nearest-neighbour correspondence is the correct one.
        let source = make_sphere(300, 3.0);
        let angle = 2_f32.to_radians();
        let rot = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), angle);
        let target = PointCloud::from_points(source.points.iter().map(|p| rot * p).collect());

        let config = GicpConfig {
            max_iterations: 60,
            max_correspondence_distance: 0.8,
            ..Default::default()
        };
        let result = gicp(&source, &target, Isometry3::identity(), config).unwrap();

        let rot_err = result.transformation.rotation.angle_to(&rot);
        assert!(
            rot_err < 0.5_f32.to_radians(),
            "rotation error = {:.2}°",
            rot_err.to_degrees()
        );
        assert!(result.mse < 0.01, "mse={}", result.mse);
    }

    #[test]
    fn gicp_refines_rotation_from_near_correct_init() {
        // Simulate the SLAM use-case: odometry gives an 80%-correct initial guess
        // (6° out of 8°).  GICP refines the 2° residual to < 0.5°.
        let source = make_sphere(200, 3.0);
        let true_angle = 8_f32.to_radians();
        let init_angle = 6_f32.to_radians(); // ← provided by odometry / IMU
        let true_rot = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), true_angle);
        let target = PointCloud::from_points(source.points.iter().map(|p| true_rot * p).collect());
        let init = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), init_angle),
        );

        let config = GicpConfig {
            max_iterations: 60,
            max_correspondence_distance: 0.8,
            ..Default::default()
        };
        let result = gicp(&source, &target, init, config).unwrap();

        let rot_err = result.transformation.rotation.angle_to(&true_rot);
        assert!(
            rot_err < 0.5_f32.to_radians(),
            "rotation error = {:.2}°",
            rot_err.to_degrees()
        );
    }

    #[test]
    fn gicp_refines_combined_rotation_and_translation() {
        // Near-correct init for combined 6° rotation + 0.3 m translation.
        let source = make_sphere(200, 3.0);
        let true_angle = 6_f32.to_radians();
        let true_shift = Vector3f::new(0.3, 0.0, 0.0);
        let true_rot = UnitQuaternion::from_axis_angle(&Vector3f::y_axis(), true_angle);
        let true_iso = Isometry3::from_parts(
            Translation3::new(true_shift.x, true_shift.y, true_shift.z),
            true_rot,
        );
        let target = PointCloud::from_points(source.points.iter().map(|p| true_iso * p).collect());

        // Initial guess: 80% of the rotation, 80% of the translation.
        let init = Isometry3::from_parts(
            Translation3::new(0.24, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3f::y_axis(), 4.8_f32.to_radians()),
        );

        let config = GicpConfig {
            max_iterations: 80,
            max_correspondence_distance: 0.8,
            ..Default::default()
        };
        let result = gicp(&source, &target, init, config).unwrap();

        let t_err = (result.transformation.translation.vector - true_shift).magnitude();
        let rot_err = result.transformation.rotation.angle_to(&true_rot);
        assert!(t_err < 0.05, "translation error={}", t_err);
        assert!(
            rot_err < 0.5_f32.to_radians(),
            "rotation error = {:.2}°",
            rot_err.to_degrees()
        );
    }

    // -------------------------------------------------------------------------
    // Noise robustness
    // -------------------------------------------------------------------------

    #[test]
    fn gicp_robust_to_gaussian_noise() {
        // Target = source + uniform ±0.05 m noise per axis.
        let source = make_sphere(200, 3.0);
        let noise_half = 0.05_f32;
        // Deterministic noise via a simple LCG so tests are reproducible.
        let target = PointCloud::from_points(
            source
                .points
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

        let config = GicpConfig {
            max_iterations: 50,
            max_correspondence_distance: 1.0,
            ..Default::default()
        };
        let result = gicp(&source, &target, Isometry3::identity(), config).unwrap();

        // MSE should be in the noise-variance ballpark (3 × 0.05² ≈ 0.0075),
        // well below the correspondence-distance threshold.
        assert!(result.mse < 0.05, "mse={}", result.mse);
        let t_err = result.transformation.translation.vector.magnitude();
        assert!(t_err < 0.1, "spurious translation drift={}", t_err);
    }

    #[test]
    fn gicp_robust_to_outlier_points() {
        // Target = source union 20 far outliers. Tight correspondence distance
        // should exclude the outliers so the result stays close to identity.
        let source = make_sphere(200, 3.0);
        let mut target_pts = source.points.clone();
        for i in 0..20 {
            let t = i as f32;
            target_pts.push(Point3f::new(t * 7.3 - 50.0, t * 3.1 - 30.0, t * 5.7 - 40.0));
        }
        let target = PointCloud::from_points(target_pts);

        let config = GicpConfig {
            max_iterations: 40,
            max_correspondence_distance: 0.5,
            ..Default::default()
        };
        let result = gicp(&source, &target, Isometry3::identity(), config).unwrap();

        let t_err = result.transformation.translation.vector.magnitude();
        assert!(t_err < 0.05, "spurious drift from outliers={}", t_err);
        assert!(result.mse < 0.01);
    }

    // -------------------------------------------------------------------------
    // Degenerate inputs
    // -------------------------------------------------------------------------

    #[test]
    fn gicp_too_few_points_errors() {
        // Fewer points than k_correspondences (default 20).
        let cloud = make_sphere(10, 1.0);
        assert!(
            gicp(&cloud, &cloud, Isometry3::identity(), GicpConfig::default()).is_err(),
            "expected error for cloud with too few points"
        );
    }

    #[test]
    fn gicp_coplanar_points_errors() {
        // All points on Z = 0 plane — bounding box extent in Z is 0.
        let pts: Vec<Point3f> = (0..50)
            .flat_map(|i| (0..50).map(move |j| Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0)))
            .collect();
        let cloud = PointCloud::from_points(pts);
        assert!(
            gicp(&cloud, &cloud, Isometry3::identity(), GicpConfig::default()).is_err(),
            "expected error for coplanar point cloud"
        );
    }
}
