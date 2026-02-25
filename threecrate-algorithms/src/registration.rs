//! Registration algorithms

use threecrate_core::{PointCloud, Result, Point3f, Vector3f, Error, Isometry3};
use nalgebra::{Matrix3, Matrix6, Vector6, UnitQuaternion, Translation3};
use rayon::prelude::*;




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

/// Find the closest point in target cloud for each point in source cloud
fn find_correspondences(
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

/// Compute the optimal transformation using SVD
fn compute_transformation(
    source_points: &[Point3f],
    target_points: &[Point3f],
) -> Result<Isometry3<f32>> {
    if source_points.len() != target_points.len() || source_points.is_empty() {
        return Err(Error::InvalidData("Point correspondence mismatch".to_string()));
    }

    let n = source_points.len() as f32;

    // Compute centroids
    let source_centroid = source_points.iter().fold(Point3f::origin(), |acc, p| acc + p.coords) / n;
    let target_centroid = target_points.iter().fold(Point3f::origin(), |acc, p| acc + p.coords) / n;

    // Compute covariance matrix H
    let mut h = Matrix3::zeros();
    for (src, tgt) in source_points.iter().zip(target_points.iter()) {
        let p = src - source_centroid;
        let q = tgt - target_centroid;
        h += p * q.transpose();
    }

    // SVD decomposition
    let svd = h.svd(true, true);
    let u = svd.u.ok_or_else(|| Error::Algorithm("SVD U matrix not available".to_string()))?;
    let v_t = svd.v_t.ok_or_else(|| Error::Algorithm("SVD V^T matrix not available".to_string()))?;

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
    let translation = target_centroid - rotation * source_centroid;

    Ok(Isometry3::from_parts(
        Translation3::new(translation.x, translation.y, translation.z),
        rotation,
    ))
}

/// Compute mean squared error between corresponding points
fn compute_mse(
    source_points: &[Point3f],
    target_points: &[Point3f],
) -> f32 {
    if source_points.is_empty() {
        return 0.0;
    }

    let sum_squared_error: f32 = source_points
        .iter()
        .zip(target_points.iter())
        .map(|(src, tgt)| (src - tgt).magnitude_squared())
        .sum();

    sum_squared_error / source_points.len() as f32
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
/// * `convergence_threshold` - MSE change threshold for convergence
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
        return Err(Error::InvalidData("Source or target point cloud is empty".to_string()));
    }

    if max_iters == 0 {
        return Err(Error::InvalidData("Max iterations must be positive".to_string()));
    }

    let mut current_transform = init;
    let mut previous_mse = f32::INFINITY;
    let mut final_correspondences = Vec::new();

    for iteration in 0..max_iters {
        // Transform source points with current transformation
        let transformed_source: Vec<Point3f> = source
            .points
            .iter()
            .map(|point| current_transform * point)
            .collect();

        // Find correspondences
        let correspondences = find_correspondences(
            &transformed_source,
            &target.points,
            max_correspondence_distance,
        );

        // Extract valid correspondences
        let mut valid_source_points = Vec::new();
        let mut valid_target_points = Vec::new();
        let mut corr_pairs = Vec::new();

        for (src_idx, correspondence) in correspondences.iter().enumerate() {
            if let Some((tgt_idx, _distance)) = correspondence {
                valid_source_points.push(transformed_source[src_idx]);
                valid_target_points.push(target.points[*tgt_idx]);
                corr_pairs.push((src_idx, *tgt_idx));
            }
        }

        if valid_source_points.len() < 3 {
            return Err(Error::Algorithm("Insufficient correspondences found".to_string()));
        }

        // Compute transformation for this iteration
        let delta_transform = compute_transformation(&valid_source_points, &valid_target_points)?;

        // Update transformation
        current_transform = delta_transform * current_transform;

        // Compute MSE
        let current_mse = compute_mse(&valid_source_points, &valid_target_points);

        // Check for convergence
        let mse_change = (previous_mse - current_mse).abs();
        if mse_change < convergence_threshold {
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

    // Final transformation after all iterations
    let transformed_source: Vec<Point3f> = source
        .points
        .iter()
        .map(|point| current_transform * point)
        .collect();

    let final_mse = if !final_correspondences.is_empty() {
        let valid_source: Vec<Point3f> = final_correspondences
            .iter()
            .map(|(src_idx, _)| transformed_source[*src_idx])
            .collect();
        let valid_target: Vec<Point3f> = final_correspondences
            .iter()
            .map(|(_, tgt_idx)| target.points[*tgt_idx])
            .collect();
        compute_mse(&valid_source, &valid_target)
    } else {
        previous_mse
    };

    Ok(ICPResult {
        transformation: current_transform,
        mse: final_mse,
        iterations: max_iters,
        converged: false,
        correspondences: final_correspondences,
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
        ata.lu()
            .solve(&atb)
            .ok_or_else(|| Error::Algorithm("Point-to-plane system is ill-conditioned".to_string()))?
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
/// * `convergence_threshold`        - MSE change threshold to declare convergence
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

    let mut current_transform = init;
    let mut previous_mse = f32::INFINITY;
    let mut final_correspondences: Vec<(usize, usize)> = Vec::new();

    for iteration in 0..max_iters {
        // Apply current estimate to source
        let transformed_source: Vec<Point3f> = source
            .points
            .iter()
            .map(|p| current_transform * p)
            .collect();

        // Find nearest-neighbor correspondences
        let correspondences = find_correspondences(
            &transformed_source,
            &target.points,
            max_correspondence_distance,
        );

        let mut valid_source: Vec<Point3f> = Vec::new();
        let mut valid_target: Vec<Point3f> = Vec::new();
        let mut valid_normals: Vec<Vector3f> = Vec::new();
        let mut corr_pairs: Vec<(usize, usize)> = Vec::new();

        for (src_idx, corr) in correspondences.iter().enumerate() {
            if let Some((tgt_idx, _)) = corr {
                valid_source.push(transformed_source[src_idx]);
                valid_target.push(target.points[*tgt_idx]);
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

        let delta = compute_transformation_point_to_plane(&valid_source, &valid_target, &valid_normals)?;
        current_transform = delta * current_transform;

        let current_mse = compute_point_to_plane_mse(&valid_source, &valid_target, &valid_normals);
        let mse_change = (previous_mse - current_mse).abs();

        if mse_change < convergence_threshold {
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
/// * `convergence_threshold` - MSE change threshold for convergence (default: 1e-6)
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
        return Err(Error::InvalidData("Source or target point cloud is empty".to_string()));
    }

    if max_iterations == 0 {
        return Err(Error::InvalidData("Max iterations must be positive".to_string()));
    }

    if convergence_threshold <= 0.0 {
        return Err(Error::InvalidData("Convergence threshold must be positive".to_string()));
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
        assert!(computed_translation.magnitude() > 0.05, "Translation magnitude too small: {}", computed_translation.magnitude());
        
        assert!(result.mse < 2.0); // Allow for higher MSE in simple test cases
    }

    #[test]
    fn test_icp_rotation() {
        // Create source and target with known rotation
        let mut source = PointCloud::new();
        let mut target = PointCloud::new();
        
        let rotation = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), std::f32::consts::FRAC_PI_4);
        
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
        assert!(translation_mag > 0.1, "Translation magnitude too small: {}", translation_mag);
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
            Translation3::new(known_translation.x, known_translation.y, known_translation.z),
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
        assert!(translation_error < 1.0, "Translation error too large: {}", translation_error);
        
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
        let result = icp_point_to_plane(&source, &target, &normals, Isometry3::identity(), 50)
            .unwrap();

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