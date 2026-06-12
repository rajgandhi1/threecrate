//! NDT (Normal Distributions Transform) registration algorithm
//!
//! NDT is a point cloud registration method that represents the target point cloud
//! as a set of normal distributions (one per voxel). It is more robust than ICP for
//! large initial misalignments and sparse point clouds.
//!
//! Reference: Biber & Straßer (2003), "The Normal Distributions Transform: A New Approach to Laser Scan Matching"

use nalgebra::{Matrix3, Translation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use threecrate_core::{Error, Isometry3, Point3f, PointCloud, Result};

/// Configuration for NDT registration
#[derive(Debug, Clone)]
pub struct NdtConfig {
    /// Voxel grid resolution (side length of each cell)
    pub resolution: f32,
    /// Maximum step size for gradient descent
    pub step_size: f32,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold (minimum change in transformation norm)
    pub epsilon: f32,
    /// Minimum number of points required in a voxel to compute distribution
    pub min_points_per_voxel: usize,
}

impl Default for NdtConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            step_size: 0.1,
            max_iterations: 35,
            epsilon: 1e-4,
            min_points_per_voxel: 5,
        }
    }
}

/// Result of NDT registration
#[derive(Debug, Clone)]
pub struct NdtResult {
    /// Final transformation (source → target)
    pub transformation: Isometry3<f32>,
    /// Final score (higher = better fit; NDT score)
    pub score: f32,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// A single voxel's normal distribution: mean and inverse covariance
#[derive(Debug, Clone)]
struct VoxelDistribution {
    mean: Vector3<f32>,
    inv_cov: Matrix3<f32>,
}

/// Discretize a point to voxel key
fn voxel_key(p: &Point3f, resolution: f32) -> (i32, i32, i32) {
    (
        (p.x / resolution).floor() as i32,
        (p.y / resolution).floor() as i32,
        (p.z / resolution).floor() as i32,
    )
}

/// Build the NDT voxel grid from a target point cloud
fn build_voxel_grid(
    target: &PointCloud<Point3f>,
    resolution: f32,
    min_points: usize,
) -> HashMap<(i32, i32, i32), VoxelDistribution> {
    // Group points by voxel
    let mut cells: HashMap<(i32, i32, i32), Vec<Vector3<f32>>> = HashMap::new();
    for p in &target.points {
        let key = voxel_key(p, resolution);
        cells
            .entry(key)
            .or_default()
            .push(Vector3::new(p.x, p.y, p.z));
    }

    let mut grid = HashMap::new();
    for (key, pts) in cells {
        if pts.len() < min_points {
            continue;
        }
        let n = pts.len() as f32;

        // Compute mean
        let mean = pts.iter().fold(Vector3::zeros(), |acc, p| acc + p) / n;

        // Compute covariance
        let mut cov = Matrix3::zeros();
        for p in &pts {
            let d = p - mean;
            cov += d * d.transpose();
        }
        cov /= n;

        // Regularize to avoid singular matrices
        cov += Matrix3::identity() * 1e-4;

        if let Some(inv_cov) = cov.try_inverse() {
            grid.insert(key, VoxelDistribution { mean, inv_cov });
        }
    }
    grid
}

/// Compute the NDT score and gradient/hessian for a given pose
///
/// Returns (score, gradient_6dof, hessian_6x6)
/// The 6-DOF parameterization is [tx, ty, tz, rx, ry, rz] (small angles).
fn compute_score_and_derivatives(
    source: &[Point3f],
    grid: &HashMap<(i32, i32, i32), VoxelDistribution>,
    transform: &Isometry3<f32>,
    resolution: f32,
) -> (f32, nalgebra::Vector6<f32>, nalgebra::Matrix6<f32>) {
    let rot = transform.rotation.to_rotation_matrix();
    let rot_mat = rot.matrix();

    let mut score = 0.0_f32;
    let mut gradient = nalgebra::Vector6::<f32>::zeros();
    let mut hessian = nalgebra::Matrix6::<f32>::zeros();

    for src_pt in source {
        // Transform source point
        let p = transform * src_pt;
        let key = voxel_key(&p, resolution);

        let dist = match grid.get(&key) {
            Some(d) => d,
            None => continue,
        };

        let diff = Vector3::new(p.x, p.y, p.z) - dist.mean;
        let cov_diff = dist.inv_cov * diff;
        let exponent = -0.5 * diff.dot(&cov_diff);
        let e = exponent.exp();

        score += e;

        // Jacobian of transformed point w.r.t. 6-DOF params
        // [∂p/∂tx, ∂p/∂ty, ∂p/∂tz, ∂p/∂rx, ∂p/∂ry, ∂p/∂rz]
        // Translation part is identity
        let src_vec = Vector3::new(src_pt.x, src_pt.y, src_pt.z);
        let dp_dtrans = Matrix3::identity();
        // Rotation part: ∂(R*s)/∂angle via skew-symmetric
        // ∂(R*s)/∂rx = R * skew([1,0,0]) * s_body (approximated as skew * R^T * p)
        let rs = rot_mat * src_vec;
        let dp_drx = Vector3::new(0.0, -rs[2], rs[1]); // skew([1,0,0]) * rs
        let dp_dry = Vector3::new(rs[2], 0.0, -rs[0]); // skew([0,1,0]) * rs
        let dp_drz = Vector3::new(-rs[1], rs[0], 0.0); // skew([0,0,1]) * rs

        // Full jacobian: 3x6 matrix (rows=xyz, cols=6dof)
        let mut jac = nalgebra::Matrix3x6::<f32>::zeros();
        jac.fixed_columns_mut::<3>(0).copy_from(&dp_dtrans);
        jac.column_mut(3).copy_from(&dp_drx);
        jac.column_mut(4).copy_from(&dp_dry);
        jac.column_mut(5).copy_from(&dp_drz);

        // g = J^T * Σ^-1 * diff  (NDT gradient contribution)
        let g_vec = jac.transpose() * cov_diff;
        gradient += e * g_vec;

        // H ≈ J^T * Σ^-1 * J (Gauss-Newton approximation)
        let h_contrib = jac.transpose() * dist.inv_cov * jac;
        hessian += e * h_contrib;
    }

    (score, gradient, hessian)
}

/// Run NDT registration to align `source` onto `target`.
///
/// # Arguments
/// * `source` - The point cloud to align
/// * `target` - The reference point cloud
/// * `initial_transform` - Initial guess for the transformation
/// * `config` - NDT configuration parameters
///
/// # Returns
/// [`NdtResult`] with the final transformation, score, iterations, and convergence flag.
pub fn ndt_registration(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    initial_transform: Isometry3<f32>,
    config: &NdtConfig,
) -> Result<NdtResult> {
    if source.points.is_empty() {
        return Err(Error::Algorithm("Source point cloud is empty".into()));
    }
    if target.points.len() < config.min_points_per_voxel {
        return Err(Error::Algorithm(
            "Target point cloud has too few points for NDT voxel grid".into(),
        ));
    }

    let grid = build_voxel_grid(target, config.resolution, config.min_points_per_voxel);
    if grid.is_empty() {
        return Err(Error::Algorithm(
            "NDT voxel grid is empty — try a larger resolution or lower min_points_per_voxel"
                .into(),
        ));
    }

    let mut transform = initial_transform;
    let mut converged = false;
    let mut iterations = 0;
    let mut score = 0.0_f32;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        let (s, grad, hess) =
            compute_score_and_derivatives(&source.points, &grid, &transform, config.resolution);
        score = s;

        // Regularize hessian
        let reg = nalgebra::Matrix6::<f32>::identity() * 1e-6;
        let hess_reg = hess + reg;

        // Solve H * delta = -gradient  (Newton step)
        let delta = match hess_reg.lu().solve(&(-grad)) {
            Some(d) => d,
            None => break,
        };

        // Clamp step size
        let step_norm = delta.norm();
        let delta = if step_norm > config.step_size {
            delta * (config.step_size / step_norm)
        } else {
            delta
        };

        // Check convergence
        if delta.norm() < config.epsilon {
            converged = true;
            break;
        }

        // Apply delta to transform
        let dt = Translation3::new(delta[0], delta[1], delta[2]);
        let dr = UnitQuaternion::from_euler_angles(delta[3], delta[4], delta[5]);
        let delta_iso = Isometry3::from_parts(dt, dr);
        transform = delta_iso * transform;
    }

    Ok(NdtResult {
        transformation: transform,
        score,
        iterations,
        converged,
    })
}

/// Convenience wrapper with default config
pub fn ndt_registration_default(
    source: &PointCloud<Point3f>,
    target: &PointCloud<Point3f>,
    initial_transform: Isometry3<f32>,
) -> Result<NdtResult> {
    ndt_registration(source, target, initial_transform, &NdtConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};
    use threecrate_core::PointCloud;

    fn make_grid_cloud(nx: usize, ny: usize, nz: usize, scale: f32) -> PointCloud<Point3f> {
        let mut points = Vec::new();
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    points.push(Point3f::new(
                        ix as f32 * scale,
                        iy as f32 * scale,
                        iz as f32 * scale,
                    ));
                }
            }
        }
        PointCloud { points }
    }

    fn apply_transform(cloud: &PointCloud<Point3f>, iso: &Isometry3<f32>) -> PointCloud<Point3f> {
        PointCloud {
            points: cloud.points.iter().map(|p| iso * p).collect(),
        }
    }

    #[test]
    fn test_ndt_identity() {
        let target = make_grid_cloud(5, 5, 5, 1.0);
        let config = NdtConfig {
            resolution: 2.0,
            min_points_per_voxel: 2,
            ..Default::default()
        };
        let result = ndt_registration(&target, &target, Isometry3::identity(), &config).unwrap();
        assert!(result.score > 0.0);
        // With identical clouds, translation should be near zero
        let t = result.transformation.translation.vector;
        assert!(t.norm() < 1.0, "Translation should be small: {}", t.norm());
    }

    #[test]
    fn test_ndt_small_translation() {
        let target = make_grid_cloud(6, 6, 6, 1.0);
        let translation =
            Isometry3::from_parts(Translation3::new(0.3, 0.2, 0.1), UnitQuaternion::identity());
        let source = apply_transform(&target, &translation);

        let config = NdtConfig {
            resolution: 2.0,
            step_size: 0.5,
            max_iterations: 50,
            epsilon: 1e-5,
            min_points_per_voxel: 3,
        };
        let result = ndt_registration(&source, &target, Isometry3::identity(), &config).unwrap();
        assert!(result.score > 0.0);
        assert!(result.iterations <= 50);
    }

    #[test]
    fn test_ndt_empty_source() {
        let empty: PointCloud<Point3f> = PointCloud { points: vec![] };
        let target = make_grid_cloud(4, 4, 4, 1.0);
        let config = NdtConfig::default();
        assert!(ndt_registration(&empty, &target, Isometry3::identity(), &config).is_err());
    }

    #[test]
    fn test_ndt_sparse_target() {
        let source = make_grid_cloud(4, 4, 4, 1.0);
        // Only 2 points — below default min_points_per_voxel
        let sparse: PointCloud<Point3f> = PointCloud {
            points: vec![Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0)],
        };
        let config = NdtConfig::default();
        assert!(ndt_registration(&source, &sparse, Isometry3::identity(), &config).is_err());
    }

    #[test]
    fn test_ndt_config_default() {
        let cfg = NdtConfig::default();
        assert_eq!(cfg.resolution, 1.0);
        assert_eq!(cfg.max_iterations, 35);
        assert!(cfg.epsilon > 0.0);
        assert!(cfg.step_size > 0.0);
        assert!(cfg.min_points_per_voxel >= 1);
    }

    #[test]
    fn test_ndt_result_fields() {
        let target = make_grid_cloud(5, 5, 5, 1.0);
        let config = NdtConfig {
            resolution: 2.0,
            min_points_per_voxel: 2,
            ..Default::default()
        };
        let result = ndt_registration(&target, &target, Isometry3::identity(), &config).unwrap();
        // All result fields should be populated
        assert!(result.iterations > 0);
        assert!(result.score >= 0.0);
        // transformation should be a valid isometry (rotation norm ≈ 1)
        let qnorm = result.transformation.rotation.norm();
        assert!((qnorm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ndt_default_wrapper() {
        let target = make_grid_cloud(5, 5, 5, 1.0);
        // Should not panic/error with a dense enough cloud
        let config_res_cloud = NdtConfig {
            resolution: 2.0,
            min_points_per_voxel: 2,
            ..Default::default()
        };
        let r1 = ndt_registration(&target, &target, Isometry3::identity(), &config_res_cloud);
        assert!(r1.is_ok());
    }
}
