//! Global point cloud registration via FPFH feature matching and RANSAC.
//!
//! Implements a coarse-to-fine pipeline:
//! 1. Estimate surface normals (if not pre-computed)
//! 2. Extract FPFH descriptors from source and target
//! 3. Match descriptors to build putative correspondences
//! 4. RANSAC to find the best rigid transformation consistent with those correspondences
//! 5. Optional ICP refinement to polish the coarse alignment
//!
//! Expected use: call `global_registration()` to get a good initial pose, then hand off to ICP
//! or NDT for sub-millimetre refinement.

use threecrate_core::{PointCloud, Result, Point3f, Error, Isometry3};
use nalgebra::{Matrix3, Vector3, Rotation3, UnitQuaternion, Translation3};
use rand::Rng;
use rayon::prelude::*;
use crate::features::{extract_fpfh_features_with_normals, FpfhConfig, FPFH_DIM};
use crate::normals::estimate_normals;
use crate::registration::{icp_point_to_point, ICPResult};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the global registration pipeline.
#[derive(Debug, Clone)]
pub struct GlobalRegistrationConfig {
    /// Maximum RANSAC iterations.  Higher = more robust, slower.
    pub ransac_iterations: usize,
    /// Maximum Euclidean distance (in model units) for a correspondence to count as an inlier.
    pub distance_threshold: f32,
    /// Early-exit RANSAC when the fraction of inlier correspondences exceeds this value.
    pub inlier_ratio: f32,
    /// Radius used for FPFH feature extraction.
    pub fpfh_radius: f32,
    /// Minimum neighbours required by radius search; falls back to k-NN when fewer found.
    pub fpfh_k_neighbors: usize,
    /// Number of nearest neighbours used for surface normal estimation.
    pub normal_k_neighbors: usize,
    /// Run ICP after RANSAC to refine the coarse alignment.
    pub refine_with_icp: bool,
    /// Maximum ICP iterations (only used when `refine_with_icp` is true).
    pub icp_max_iterations: usize,
    /// Maximum point-to-point correspondence distance for ICP (None = unlimited).
    pub icp_distance_threshold: Option<f32>,
}

impl Default for GlobalRegistrationConfig {
    fn default() -> Self {
        Self {
            ransac_iterations: 50_000,
            distance_threshold: 0.05,
            inlier_ratio: 0.25,
            fpfh_radius: 0.25,
            fpfh_k_neighbors: 10,
            normal_k_neighbors: 10,
            refine_with_icp: true,
            icp_max_iterations: 50,
            icp_distance_threshold: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of the global registration pipeline.
#[derive(Debug, Clone)]
pub struct GlobalRegistrationResult {
    /// Best-found rigid transformation that maps `source` onto `target`.
    pub transformation: Isometry3<f32>,
    /// Number of correspondences classified as inliers under `transformation`.
    pub inlier_count: usize,
    /// Fraction of total correspondences that are inliers (0.0 – 1.0).
    pub inlier_ratio: f32,
    /// ICP refinement result, present when `config.refine_with_icp` is `true`.
    pub icp_result: Option<ICPResult>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Squared L2 distance between two FPFH descriptors.
#[inline]
fn fpfh_dist_sq(a: &[f32; FPFH_DIM], b: &[f32; FPFH_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Match source descriptors to target descriptors (nearest-neighbour in descriptor space).
///
/// Returns a list of `(source_idx, target_idx)` pairs.
fn find_feature_correspondences(
    src_descs: &[[f32; FPFH_DIM]],
    tgt_descs: &[[f32; FPFH_DIM]],
) -> Vec<(usize, usize)> {
    src_descs
        .par_iter()
        .enumerate()
        .filter_map(|(i, sd)| {
            let best = tgt_descs
                .iter()
                .enumerate()
                .min_by(|(_, ta), (_, tb)| {
                    fpfh_dist_sq(sd, ta)
                        .partial_cmp(&fpfh_dist_sq(sd, tb))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?;
            Some((i, best.0))
        })
        .collect()
}

/// Estimate a rigid transformation from ≥ 3 point pairs using SVD (same method as ICP).
fn estimate_transform_svd(src_pts: &[Point3f], tgt_pts: &[Point3f]) -> Option<Isometry3<f32>> {
    let n = src_pts.len();
    if n < 3 {
        return None;
    }
    let scale = 1.0 / n as f32;

    let src_centroid = src_pts.iter().fold(Vector3::zeros(), |a, p| a + p.coords) * scale;
    let tgt_centroid = tgt_pts.iter().fold(Vector3::zeros(), |a, p| a + p.coords) * scale;

    let mut h = Matrix3::<f32>::zeros();
    for (s, t) in src_pts.iter().zip(tgt_pts.iter()) {
        let ds = s.coords - src_centroid;
        let dt = t.coords - tgt_centroid;
        h += ds * dt.transpose();
    }

    let svd = h.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let mut r = vt.transpose() * u.transpose();

    // Fix reflection
    if r.determinant() < 0.0 {
        let mut vt_fix = vt;
        vt_fix.row_mut(2).neg_mut();
        r = vt_fix.transpose() * u.transpose();
    }

    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    let t = tgt_centroid - rotation * src_centroid;
    Some(Isometry3::from_parts(Translation3::from(t), rotation))
}

/// Count correspondences that are inliers under `transform`.
fn count_inliers(
    corrs: &[(usize, usize)],
    src_pts: &[Point3f],
    tgt_pts: &[Point3f],
    transform: &Isometry3<f32>,
    threshold: f32,
) -> usize {
    let thr_sq = threshold * threshold;
    corrs
        .iter()
        .filter(|&&(si, ti)| {
            let tp = transform * src_pts[si];
            (tp - tgt_pts[ti]).magnitude_squared() <= thr_sq
        })
        .count()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the full global registration pipeline on raw (un-normalised) point clouds.
///
/// This function estimates normals internally. For more control (e.g. when normals are
/// already computed or cloud density is irregular) call `global_registration_with_normals`.
///
/// # Workflow
/// 1. Normal estimation → FPFH extraction → feature matching → RANSAC → optional ICP
///
/// # Arguments
/// * `source` – Point cloud to align (the "model")
/// * `target` – Reference point cloud (the "scene")
/// * `config` – Pipeline parameters
///
/// # Returns
/// [`GlobalRegistrationResult`] containing the best transformation found, inlier statistics,
/// and (optionally) the ICP refinement result.
pub fn global_registration(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    config: &GlobalRegistrationConfig,
) -> Result<GlobalRegistrationResult> {
    if source.is_empty() {
        return Err(Error::Algorithm("Source point cloud is empty".into()));
    }
    if target.is_empty() {
        return Err(Error::Algorithm("Target point cloud is empty".into()));
    }

    let src_normals = estimate_normals(source, config.normal_k_neighbors)?;
    let tgt_normals = estimate_normals(target, config.normal_k_neighbors)?;

    global_registration_with_normals(&src_normals, &tgt_normals, source, target, config)
}

/// Global registration when surface normals are already available.
///
/// Skips normal estimation; otherwise identical to [`global_registration`].
///
/// # Arguments
/// * `source_n` – Source cloud with normals (for FPFH)
/// * `target_n` – Target cloud with normals (for FPFH)
/// * `source`   – Raw source positions (for ICP refinement)
/// * `target`   – Raw target positions (for ICP refinement)
/// * `config`   – Pipeline parameters
pub fn global_registration_with_normals(
    source_n: &PointCloud<threecrate_core::NormalPoint3f>,
    target_n: &PointCloud<threecrate_core::NormalPoint3f>,
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    config: &GlobalRegistrationConfig,
) -> Result<GlobalRegistrationResult> {
    if source_n.is_empty() || target_n.is_empty() {
        return Err(Error::Algorithm("Source or target cloud is empty".into()));
    }

    // --- FPFH extraction ---
    let fpfh_cfg = FpfhConfig {
        search_radius: config.fpfh_radius,
        k_neighbors: config.fpfh_k_neighbors,
    };
    let src_descs = extract_fpfh_features_with_normals(source_n, &fpfh_cfg)?;
    let tgt_descs = extract_fpfh_features_with_normals(target_n, &fpfh_cfg)?;

    // --- Feature correspondences ---
    let corrs = find_feature_correspondences(&src_descs, &tgt_descs);

    if corrs.len() < 3 {
        return Err(Error::Algorithm(
            "Too few feature correspondences for RANSAC (need ≥ 3)".into(),
        ));
    }

    let src_pts = &source_n.points.iter().map(|p| p.position).collect::<Vec<_>>();
    let tgt_pts = &target_n.points.iter().map(|p| p.position).collect::<Vec<_>>();

    // --- RANSAC ---
    let mut best_transform = Isometry3::identity();
    let mut best_inliers = 0usize;
    let early_exit_count =
        ((config.inlier_ratio * corrs.len() as f32).ceil() as usize).max(3);
    let mut rng = rand::rng();
    let n_corrs = corrs.len();

    for _ in 0..config.ransac_iterations {
        // Pick 3 unique random correspondence indices
        let i0 = rng.random_range(0..n_corrs);
        let mut i1 = rng.random_range(0..n_corrs - 1);
        if i1 >= i0 { i1 += 1; }
        let mut i2 = rng.random_range(0..n_corrs - 2);
        if i2 >= i0.min(i1) { i2 += 1; }
        if i2 >= i0.max(i1) { i2 += 1; }
        let sample = [i0, i1, i2];

        let s_pts: Vec<Point3f> = sample.iter().map(|&i| src_pts[corrs[i].0]).collect();
        let t_pts: Vec<Point3f> = sample.iter().map(|&i| tgt_pts[corrs[i].1]).collect();

        let transform = match estimate_transform_svd(&s_pts, &t_pts) {
            Some(t) => t,
            None => continue,
        };

        let inliers = count_inliers(&corrs, src_pts, tgt_pts, &transform, config.distance_threshold);

        if inliers > best_inliers {
            best_inliers = inliers;
            best_transform = transform;

            if inliers >= early_exit_count {
                break;
            }
        }
    }

    let total_corrs = corrs.len();
    let inlier_ratio = if total_corrs > 0 {
        best_inliers as f32 / total_corrs as f32
    } else {
        0.0
    };

    // --- ICP refinement ---
    let icp_result = if config.refine_with_icp {
        Some(icp_point_to_point(
            source,
            target,
            best_transform,
            config.icp_max_iterations,
            1e-6,
            config.icp_distance_threshold,
        )?)
    } else {
        None
    };

    let final_transform = icp_result
        .as_ref()
        .map(|r| r.transformation)
        .unwrap_or(best_transform);

    Ok(GlobalRegistrationResult {
        transformation: final_transform,
        inlier_count: best_inliers,
        inlier_ratio,
        icp_result,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{PointCloud, Point3f};
    use nalgebra::{Isometry3, Translation3, UnitQuaternion};

    fn grid_cloud(nx: usize, ny: usize, nz: usize, scale: f32) -> PointCloud<Point3f> {
        let mut pts = Vec::new();
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    pts.push(Point3f::new(ix as f32 * scale, iy as f32 * scale, iz as f32 * scale));
                }
            }
        }
        PointCloud { points: pts }
    }

    fn apply(cloud: &PointCloud<Point3f>, iso: &Isometry3<f32>) -> PointCloud<Point3f> {
        PointCloud { points: cloud.points.iter().map(|p| iso * p).collect() }
    }

    #[test]
    fn test_global_reg_empty_source() {
        let empty: PointCloud<Point3f> = PointCloud { points: vec![] };
        let target = grid_cloud(4, 4, 4, 1.0);
        let cfg = GlobalRegistrationConfig::default();
        assert!(global_registration(&empty, &target, &cfg).is_err());
    }

    #[test]
    fn test_global_reg_empty_target() {
        let source = grid_cloud(4, 4, 4, 1.0);
        let empty: PointCloud<Point3f> = PointCloud { points: vec![] };
        let cfg = GlobalRegistrationConfig::default();
        assert!(global_registration(&source, &empty, &cfg).is_err());
    }

    #[test]
    fn test_global_reg_identity() {
        let cloud = grid_cloud(4, 4, 4, 1.0);
        let cfg = GlobalRegistrationConfig {
            ransac_iterations: 200,
            distance_threshold: 0.5,
            fpfh_radius: 3.0,
            refine_with_icp: false,
            ..Default::default()
        };
        let result = global_registration(&cloud, &cloud, &cfg).unwrap();
        // With identical clouds, should find many inliers
        assert!(result.inlier_count > 0);
        assert!(result.inlier_ratio > 0.0);
    }

    #[test]
    fn test_global_reg_returns_valid_isometry() {
        let cloud = grid_cloud(4, 4, 4, 1.0);
        let cfg = GlobalRegistrationConfig {
            ransac_iterations: 100,
            distance_threshold: 0.5,
            fpfh_radius: 3.0,
            refine_with_icp: false,
            ..Default::default()
        };
        let result = global_registration(&cloud, &cloud, &cfg).unwrap();
        // Rotation must have unit quaternion norm
        let qnorm = result.transformation.rotation.norm();
        assert!((qnorm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_global_reg_with_icp() {
        let target = grid_cloud(4, 4, 4, 1.0);
        let t = Isometry3::from_parts(
            Translation3::new(0.3, 0.2, 0.1),
            UnitQuaternion::identity(),
        );
        let source = apply(&target, &t);
        let cfg = GlobalRegistrationConfig {
            ransac_iterations: 500,
            distance_threshold: 1.0,
            fpfh_radius: 3.0,
            refine_with_icp: true,
            icp_max_iterations: 30,
            icp_distance_threshold: Some(2.0),
            ..Default::default()
        };
        let result = global_registration(&source, &target, &cfg).unwrap();
        assert!(result.icp_result.is_some());
    }

    #[test]
    fn test_estimate_transform_svd_three_points() {
        let src = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
        ];
        let shift = Vector3::new(1.0f32, 2.0, 3.0);
        let tgt: Vec<Point3f> = src.iter().map(|p| Point3f::from(p.coords + shift)).collect();
        let iso = estimate_transform_svd(&src, &tgt).unwrap();
        let t = iso.translation.vector;
        assert!((t.x - 1.0).abs() < 1e-4);
        assert!((t.y - 2.0).abs() < 1e-4);
        assert!((t.z - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_config_defaults() {
        let cfg = GlobalRegistrationConfig::default();
        assert!(cfg.ransac_iterations > 0);
        assert!(cfg.distance_threshold > 0.0);
        assert!(cfg.inlier_ratio > 0.0 && cfg.inlier_ratio < 1.0);
        assert!(cfg.fpfh_radius > 0.0);
    }
}
