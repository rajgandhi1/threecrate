//! Segmentation algorithms

use threecrate_core::{PointCloud, Result, Point3f, Vector3f, Error};
use nalgebra::{Vector4};
use rayon::prelude::*;
use rand::prelude::*;
use std::collections::HashSet;

/// A 3D plane model defined by the equation ax + by + cz + d = 0
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneModel {
    /// Plane coefficients [a, b, c, d] where ax + by + cz + d = 0
    pub coefficients: Vector4<f32>,
}

impl PlaneModel {
    /// Create a new plane model from coefficients
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            coefficients: Vector4::new(a, b, c, d),
        }
    }

    /// Create a plane model from three points
    pub fn from_points(p1: &Point3f, p2: &Point3f, p3: &Point3f) -> Option<Self> {
        // Calculate two vectors in the plane
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        
        // Calculate normal vector using cross product
        let normal = v1.cross(&v2);
        
        // Check if points are collinear
        if normal.magnitude() < 1e-8 {
            return None;
        }
        
        let normal = normal.normalize();
        
        // Calculate d coefficient using point p1
        let d = -normal.dot(&p1.coords);
        
        Some(PlaneModel::new(normal.x, normal.y, normal.z, d))
    }

    /// Get the normal vector of the plane
    pub fn normal(&self) -> Vector3f {
        Vector3f::new(
            self.coefficients.x,
            self.coefficients.y,
            self.coefficients.z,
        )
    }

    /// Calculate the distance from a point to the plane
    pub fn distance_to_point(&self, point: &Point3f) -> f32 {
        let normal = self.normal();
        let normal_magnitude = normal.magnitude();
        
        if normal_magnitude < 1e-8 {
            return f32::INFINITY;
        }
        
        (self.coefficients.x * point.x + 
         self.coefficients.y * point.y + 
         self.coefficients.z * point.z + 
         self.coefficients.w).abs() / normal_magnitude
    }

    /// Count inliers within a distance threshold
    pub fn count_inliers(&self, points: &[Point3f], threshold: f32) -> usize {
        points.iter()
            .filter(|point| self.distance_to_point(point) <= threshold)
            .count()
    }

    /// Get indices of inlier points within a distance threshold
    pub fn get_inliers(&self, points: &[Point3f], threshold: f32) -> Vec<usize> {
        points.iter()
            .enumerate()
            .filter(|(_, point)| self.distance_to_point(point) <= threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

/// RANSAC plane segmentation result
#[derive(Debug, Clone)]
pub struct PlaneSegmentationResult {
    /// The best plane model found
    pub model: PlaneModel,
    /// Indices of inlier points
    pub inliers: Vec<usize>,
    /// Number of RANSAC iterations performed
    pub iterations: usize,
}

/// Plane segmentation using RANSAC algorithm
/// 
/// This function finds the best plane that fits the most points in the cloud
/// using the RANSAC (Random Sample Consensus) algorithm.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `threshold` - Maximum distance for a point to be considered an inlier
/// * `max_iters` - Maximum number of RANSAC iterations
/// 
/// # Returns
/// * `Result<PlaneSegmentationResult>` - The best plane model and inlier indices
pub fn segment_plane(
    cloud: &PointCloud<Point3f>, 
    threshold: f32, 
    max_iters: usize
) -> Result<PlaneSegmentationResult> {
    if cloud.len() < 3 {
        return Err(Error::InvalidData("Need at least 3 points for plane segmentation".to_string()));
    }

    if threshold <= 0.0 {
        return Err(Error::InvalidData("Threshold must be positive".to_string()));
    }

    if max_iters == 0 {
        return Err(Error::InvalidData("Max iterations must be positive".to_string()));
    }

    let points = &cloud.points;
    let mut rng = thread_rng();
    let mut best_model: Option<PlaneModel> = None;
    let mut best_inliers = Vec::new();
    let mut best_score = 0;

    for _iteration in 0..max_iters {
        // Randomly sample 3 points
        let mut indices = HashSet::new();
        while indices.len() < 3 {
            indices.insert(rng.gen_range(0..points.len()));
        }
        let indices: Vec<usize> = indices.into_iter().collect();

        let p1 = &points[indices[0]];
        let p2 = &points[indices[1]];
        let p3 = &points[indices[2]];

        // Try to create a plane model from these points
        if let Some(model) = PlaneModel::from_points(p1, p2, p3) {
            // Count inliers
            let inlier_count = model.count_inliers(points, threshold);
            
            // Update best model if this one is better
            if inlier_count > best_score {
                best_score = inlier_count;
                best_inliers = model.get_inliers(points, threshold);
                best_model = Some(model);
            }
        }
    }

    match best_model {
        Some(model) => Ok(PlaneSegmentationResult {
            model,
            inliers: best_inliers,
            iterations: max_iters,
        }),
        None => Err(Error::Algorithm("Failed to find valid plane model".to_string())),
    }
}

/// Parallel RANSAC plane segmentation for better performance on large point clouds
/// 
/// This version uses parallel processing to speed up the RANSAC algorithm
/// by running multiple iterations in parallel.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `threshold` - Maximum distance for a point to be considered an inlier
/// * `max_iters` - Maximum number of RANSAC iterations
/// 
/// # Returns
/// * `Result<PlaneSegmentationResult>` - The best plane model and inlier indices
pub fn segment_plane_parallel(
    cloud: &PointCloud<Point3f>, 
    threshold: f32, 
    max_iters: usize
) -> Result<PlaneSegmentationResult> {
    if cloud.len() < 3 {
        return Err(Error::InvalidData("Need at least 3 points for plane segmentation".to_string()));
    }

    if threshold <= 0.0 {
        return Err(Error::InvalidData("Threshold must be positive".to_string()));
    }

    if max_iters == 0 {
        return Err(Error::InvalidData("Max iterations must be positive".to_string()));
    }

    let points = &cloud.points;
    
    // Run RANSAC iterations in parallel
    let results: Vec<_> = (0..max_iters)
        .into_par_iter()
        .filter_map(|_| {
            let mut rng = thread_rng();
            
            // Randomly sample 3 points
            let mut indices = HashSet::new();
            while indices.len() < 3 {
                indices.insert(rng.gen_range(0..points.len()));
            }
            let indices: Vec<usize> = indices.into_iter().collect();

            let p1 = &points[indices[0]];
            let p2 = &points[indices[1]];
            let p3 = &points[indices[2]];

            // Try to create a plane model from these points
            PlaneModel::from_points(p1, p2, p3).map(|model| {
                let inliers = model.get_inliers(points, threshold);
                let score = inliers.len();
                (model, inliers, score)
            })
        })
        .collect();

    // Find the best result
    let best = results.into_iter()
        .max_by_key(|(_, _, score)| *score);

    match best {
        Some((model, inliers, _)) => Ok(PlaneSegmentationResult {
            model,
            inliers,
            iterations: max_iters,
        }),
        None => Err(Error::Algorithm("Failed to find valid plane model".to_string())),
    }
}

/// Legacy function for backward compatibility
#[deprecated(note = "Use segment_plane instead which returns a complete result")]
pub fn segment_plane_legacy(cloud: &PointCloud<Point3f>, threshold: f32) -> Result<Vec<usize>> {
    let result = segment_plane(cloud, threshold, 1000)?;
    Ok(result.inliers)
}

/// RANSAC plane segmentation with simplified interface
/// 
/// This function provides a simplified interface for RANSAC plane segmentation
/// that returns plane coefficients and inlier indices directly.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `max_iters` - Maximum number of RANSAC iterations
/// * `threshold` - Maximum distance for a point to be considered an inlier
/// 
/// # Returns
/// * `Result<(Vector4<f32>, Vec<usize>)>` - Plane coefficients and inlier indices
/// 
/// # Example
/// ```rust
/// use threecrate_algorithms::segment_plane_ransac;
/// use threecrate_core::{PointCloud, Point3f};
/// use nalgebra::Vector4;
/// 
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let cloud = PointCloud::from_points(vec![
///         Point3f::new(0.0, 0.0, 0.0),
///         Point3f::new(1.0, 0.0, 0.0),
///         Point3f::new(0.0, 1.0, 0.0),
///     ]);
/// 
///     let (coefficients, inliers) = segment_plane_ransac(&cloud, 1000, 0.01)?;
///     println!("Plane coefficients: {:?}", coefficients);
///     println!("Found {} inliers", inliers.len());
///     Ok(())
/// }
/// ```
pub fn segment_plane_ransac(
    cloud: &PointCloud<Point3f>,
    max_iters: usize,
    threshold: f32,
) -> Result<(Vector4<f32>, Vec<usize>)> {
    let result = segment_plane(cloud, threshold, max_iters)?;
    Ok((result.model.coefficients, result.inliers))
}

/// RANSAC plane segmentation (alias for segment_plane_ransac)
/// 
/// This function is an alias for `segment_plane_ransac` to maintain compatibility
/// with the README documentation.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `max_iters` - Maximum number of RANSAC iterations
/// * `threshold` - Maximum distance for a point to be considered an inlier
/// 
/// # Returns
/// * `Result<(Vector4<f32>, Vec<usize>)>` - Plane coefficients and inlier indices
pub fn plane_segmentation_ransac(
    cloud: &PointCloud<Point3f>,
    max_iters: usize,
    threshold: f32,
) -> Result<(Vector4<f32>, Vec<usize>)> {
    segment_plane_ransac(cloud, max_iters, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_plane_model_from_points() {
        // Create a plane in XY plane (z=0)
        let p1 = Point3f::new(0.0, 0.0, 0.0);
        let p2 = Point3f::new(1.0, 0.0, 0.0);
        let p3 = Point3f::new(0.0, 1.0, 0.0);

        let model = PlaneModel::from_points(&p1, &p2, &p3).unwrap();
        
        // Normal should be close to (0, 0, 1) or (0, 0, -1)
        let normal = model.normal();
        assert!(normal.z.abs() > 0.9, "Normal should be primarily in Z direction: {:?}", normal);
        
        // Distance to points on the plane should be ~0
        assert!(model.distance_to_point(&p1) < 1e-6);
        assert!(model.distance_to_point(&p2) < 1e-6);
        assert!(model.distance_to_point(&p3) < 1e-6);
    }

    #[test]
    fn test_plane_model_collinear_points() {
        // Create collinear points
        let p1 = Point3f::new(0.0, 0.0, 0.0);
        let p2 = Point3f::new(1.0, 0.0, 0.0);
        let p3 = Point3f::new(2.0, 0.0, 0.0);

        let model = PlaneModel::from_points(&p1, &p2, &p3);
        assert!(model.is_none(), "Should return None for collinear points");
    }

    #[test]
    fn test_plane_distance_calculation() {
        // Create a plane at z=1
        let model = PlaneModel::new(0.0, 0.0, 1.0, -1.0);
        
        let point_on_plane = Point3f::new(0.0, 0.0, 1.0);
        let point_above_plane = Point3f::new(0.0, 0.0, 2.0);
        let point_below_plane = Point3f::new(0.0, 0.0, 0.0);
        
        assert_relative_eq!(model.distance_to_point(&point_on_plane), 0.0, epsilon = 1e-6);
        assert_relative_eq!(model.distance_to_point(&point_above_plane), 1.0, epsilon = 1e-6);
        assert_relative_eq!(model.distance_to_point(&point_below_plane), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_segment_plane_simple() {
        // Create a point cloud with most points on a plane
        let mut cloud = PointCloud::new();
        
        // Add points on XY plane (z=0)
        for i in 0..10 {
            for j in 0..10 {
                cloud.push(Point3f::new(i as f32, j as f32, 0.0));
            }
        }
        
        // Add a few outliers
        cloud.push(Point3f::new(5.0, 5.0, 10.0));
        cloud.push(Point3f::new(5.0, 5.0, -10.0));
        
        let result = segment_plane(&cloud, 0.1, 100).unwrap();
        
        // Should find most of the points as inliers
        assert!(result.inliers.len() >= 95, "Should find most points as inliers");
        
        // Normal should be close to (0, 0, 1) or (0, 0, -1)
        let normal = result.model.normal();
        assert!(normal.z.abs() > 0.9, "Normal should be primarily in Z direction");
    }

    #[test]
    fn test_segment_plane_insufficient_points() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        
        let result = segment_plane(&cloud, 0.1, 100);
        assert!(result.is_err(), "Should fail with insufficient points");
    }

    #[test]
    fn test_segment_plane_invalid_threshold() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        
        let result = segment_plane(&cloud, -0.1, 100);
        assert!(result.is_err(), "Should fail with negative threshold");
    }

    #[test]
    fn test_segment_plane_parallel() {
        // Create a point cloud with most points on a plane
        let mut cloud = PointCloud::new();
        
        // Add points on XY plane (z=0)
        for i in 0..10 {
            for j in 0..10 {
                cloud.push(Point3f::new(i as f32, j as f32, 0.0));
            }
        }
        
        let result = segment_plane_parallel(&cloud, 0.1, 100).unwrap();
        
        // Should find most of the points as inliers
        assert!(result.inliers.len() >= 95, "Should find most points as inliers");
    }

    #[test]
    fn test_segment_plane_ransac_simple() {
        // Create a point cloud with most points on a plane
        let mut cloud = PointCloud::new();
        
        // Add points on XY plane (z=0)
        for i in 0..10 {
            for j in 0..10 {
                cloud.push(Point3f::new(i as f32, j as f32, 0.0));
            }
        }
        
        // Add a few outliers
        cloud.push(Point3f::new(5.0, 5.0, 10.0));
        cloud.push(Point3f::new(5.0, 5.0, -10.0));
        
        let (coefficients, inliers) = segment_plane_ransac(&cloud, 100, 0.1).unwrap();
        
        // Should find most of the points as inliers
        assert!(inliers.len() >= 95, "Should find most points as inliers");
        
        // Normal should be close to (0, 0, 1) or (0, 0, -1)
        let normal = Vector3f::new(coefficients.x, coefficients.y, coefficients.z);
        assert!(normal.z.abs() > 0.9, "Normal should be primarily in Z direction: {:?}", normal);
    }

    #[test]
    fn test_segment_plane_ransac_noisy() {
        // Create a point cloud with noisy planar points
        let mut cloud = PointCloud::new();
        let mut rng = thread_rng();
        
        // Add points on XY plane (z=0) with noise
        for i in 0..20 {
            for j in 0..20 {
                let x = i as f32;
                let y = j as f32;
                let z = rng.gen_range(-0.05..0.05); // Add noise to z coordinate
                cloud.push(Point3f::new(x, y, z));
            }
        }
        
        // Add some outliers
        for _ in 0..20 {
            let x = rng.gen_range(0.0..20.0);
            let y = rng.gen_range(0.0..20.0);
            let z = rng.gen_range(1.0..5.0); // Outliers above the plane
            cloud.push(Point3f::new(x, y, z));
        }
        
        let (coefficients, inliers) = segment_plane_ransac(&cloud, 1000, 0.1).unwrap();
        
        // Should find most of the planar points as inliers
        assert!(inliers.len() >= 350, "Should find most planar points as inliers");
        
        // Normal should be close to (0, 0, 1) or (0, 0, -1)
        let normal = Vector3f::new(coefficients.x, coefficients.y, coefficients.z);
        assert!(normal.z.abs() > 0.8, "Normal should be primarily in Z direction: {:?}", normal);
        
        // Test that outliers are not included in inliers
        let outlier_indices: Vec<usize> = (400..420).collect();
        let outlier_inliers: Vec<usize> = inliers.iter()
            .filter(|&&idx| outlier_indices.contains(&idx))
            .cloned()
            .collect();
        assert!(outlier_inliers.len() <= 2, "Should not include many outliers in inliers");
    }

    #[test]
    fn test_segment_plane_ransac_tilted_plane() {
        // Create a tilted plane (not aligned with coordinate axes)
        let mut cloud = PointCloud::new();
        let mut rng = thread_rng();
        
        // Create a tilted plane: x + y + z = 0
        for i in 0..15 {
            for j in 0..15 {
                let x = i as f32;
                let y = j as f32;
                let z = -(x + y); // Points on the plane x + y + z = 0
                
                // Add some noise
                let noise_x = rng.gen_range(-0.02..0.02);
                let noise_y = rng.gen_range(-0.02..0.02);
                let noise_z = rng.gen_range(-0.02..0.02);
                
                cloud.push(Point3f::new(x + noise_x, y + noise_y, z + noise_z));
            }
        }
        
        // Add outliers
        for _ in 0..30 {
            let x = rng.gen_range(0.0..15.0);
            let y = rng.gen_range(0.0..15.0);
            let z = rng.gen_range(5.0..10.0); // Outliers above the plane
            cloud.push(Point3f::new(x, y, z));
        }
        
        let (coefficients, inliers) = segment_plane_ransac(&cloud, 1000, 0.1).unwrap();
        
        // Should find most of the planar points as inliers
        assert!(inliers.len() >= 200, "Should find most planar points as inliers");
        
        // Normal should be close to (1, 1, 1) normalized
        let normal = Vector3f::new(coefficients.x, coefficients.y, coefficients.z);
        let expected_normal = Vector3f::new(1.0, 1.0, 1.0).normalize();
        let dot_product = normal.dot(&expected_normal).abs();
        assert!(dot_product > 0.8, "Normal should be close to expected direction: {:?}", normal);
    }

    #[test]
    fn test_plane_segmentation_ransac_alias() {
        // Test that plane_segmentation_ransac is an alias for segment_plane_ransac
        let mut cloud = PointCloud::new();
        
        // Add points on XY plane (z=0)
        for i in 0..5 {
            for j in 0..5 {
                cloud.push(Point3f::new(i as f32, j as f32, 0.0));
            }
        }
        
        let result1 = segment_plane_ransac(&cloud, 100, 0.1).unwrap();
        let result2 = plane_segmentation_ransac(&cloud, 100, 0.1).unwrap();
        
        // Both should return valid results (RANSAC is stochastic, so exact values may differ)
        assert!(result1.1.len() >= 20, "Should find most points as inliers");
        assert!(result2.1.len() >= 20, "Should find most points as inliers");
        
        // Both should have similar inlier counts (within reasonable bounds)
        let diff = (result1.1.len() as i32 - result2.1.len() as i32).abs();
        assert!(diff <= 5, "Inlier counts should be similar: {} vs {}", result1.1.len(), result2.1.len());
    }

    #[test]
    fn test_segment_plane_ransac_insufficient_points() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        
        let result = segment_plane_ransac(&cloud, 100, 0.1);
        assert!(result.is_err(), "Should fail with insufficient points");
    }

    #[test]
    fn test_segment_plane_ransac_invalid_threshold() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        
        let result = segment_plane_ransac(&cloud, 100, -0.1);
        assert!(result.is_err(), "Should fail with negative threshold");
    }

    #[test]
    fn test_segment_plane_ransac_zero_iterations() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        
        let result = segment_plane_ransac(&cloud, 0, 0.1);
        assert!(result.is_err(), "Should fail with zero iterations");
    }
}