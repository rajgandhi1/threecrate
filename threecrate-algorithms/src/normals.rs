//! Normal estimation algorithms

use threecrate_core::{PointCloud, Result, Point3f, Vector3f, NormalPoint3f, Error};
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Configuration for normal estimation
#[derive(Debug, Clone)]
pub struct NormalEstimationConfig {
    /// Number of nearest neighbors to use (k-NN)
    pub k_neighbors: usize,
    /// Optional radius for radius-based neighbor search
    pub radius: Option<f32>,
    /// Whether to enforce orientation consistency
    pub consistent_orientation: bool,
    /// Viewpoint for orientation consistency (if None, uses positive Z direction)
    pub viewpoint: Option<Point3f>,
}

impl Default for NormalEstimationConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 10,
            radius: None,
            consistent_orientation: true,
            viewpoint: None,
        }
    }
}

/// A simple distance-based neighbor for priority queue
#[derive(Debug, Clone)]
struct Neighbor {
    index: usize,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// Find k-nearest neighbors using brute force search
fn find_k_nearest_neighbors(points: &[Point3f], query_idx: usize, k: usize) -> Vec<usize> {
    let query = &points[query_idx];
    let mut heap = BinaryHeap::with_capacity(k + 1);
    
    for (i, point) in points.iter().enumerate() {
        if i == query_idx {
            continue; // Skip the query point itself
        }
        
        let distance = (point - query).magnitude_squared();
        let neighbor = Neighbor { index: i, distance };
        
        if heap.len() < k {
            heap.push(neighbor);
        } else if let Some(farthest) = heap.peek() {
            if neighbor.distance < farthest.distance {
                heap.pop();
                heap.push(neighbor);
            }
        }
    }
    
    heap.into_iter().map(|n| n.index).collect()
}

/// Find neighbors within a radius
fn find_radius_neighbors(points: &[Point3f], query_idx: usize, radius: f32) -> Vec<usize> {
    let query = &points[query_idx];
    let radius_squared = radius * radius;
    
    points.iter()
        .enumerate()
        .filter(|(i, point)| {
            *i != query_idx && (**point - query).magnitude_squared() <= radius_squared
        })
        .map(|(i, _)| i)
        .collect()
}

/// Find neighbors using either k-NN or radius-based search
fn find_neighbors(points: &[Point3f], query_idx: usize, config: &NormalEstimationConfig) -> Vec<usize> {
    if let Some(radius) = config.radius {
        // Use radius-based search
        find_radius_neighbors(points, query_idx, radius)
    } else {
        // Use k-NN search
        find_k_nearest_neighbors(points, query_idx, config.k_neighbors)
    }
}

/// Compute normal using PCA on the neighborhood points
fn compute_normal_pca(points: &[Point3f], indices: &[usize]) -> Vector3f {
    if indices.len() < 3 {
        // Default normal if not enough points
        return Vector3f::new(0.0, 0.0, 1.0);
    }
    
    // Compute centroid
    let mut centroid = Point3f::origin();
    for &idx in indices {
        centroid += points[idx].coords;
    }
    centroid /= indices.len() as f32;
    
    // Build covariance matrix
    let mut covariance = Matrix3::zeros();
    for &idx in indices {
        let diff = points[idx] - centroid;
        covariance += diff * diff.transpose();
    }
    covariance /= indices.len() as f32;
    
    // Find eigenvector corresponding to smallest eigenvalue
    // This is the normal direction
    let eigen = covariance.symmetric_eigen();
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;
    
    // Find index of smallest eigenvalue
    let mut min_idx = 0;
    for i in 1..3 {
        if eigenvalues[i] < eigenvalues[min_idx] {
            min_idx = i;
        }
    }
    

    
    // Return the corresponding eigenvector as normal
    let mut normal: Vector3f = eigenvectors.column(min_idx).into();
    
    // Ensure the normal is normalized
    let magnitude = normal.magnitude();
    if magnitude > 1e-6 {
        normal /= magnitude;
    } else {
        normal = Vector3f::new(0.0, 0.0, 1.0);
    }
    
    normal
}

/// Orient normal towards viewpoint for consistency
fn orient_normal_towards_viewpoint(normal: Vector3f, point: Point3f, viewpoint: Point3f) -> Vector3f {
    let to_viewpoint = (viewpoint - point).normalize();
    let dot_product = normal.dot(&to_viewpoint);
    
    // If the angle between normal and viewpoint direction is > 90 degrees, flip the normal
    if dot_product < 0.0 {
        -normal
    } else {
        normal
    }
}

/// Estimate normals for a point cloud using k-nearest neighbors
/// 
/// This function computes surface normals for each point in the cloud by:
/// 1. Finding k nearest neighbors for each point (or neighbors within radius)
/// 2. Computing the normal using Principal Component Analysis (PCA)
/// 3. The normal is the eigenvector corresponding to the smallest eigenvalue
/// 4. Optionally enforcing orientation consistency towards a viewpoint
/// 
/// # Arguments
/// * `cloud` - Reference to the point cloud
/// * `k` - Number of nearest neighbors to use (typically 10-30)
/// 
/// # Returns
/// * `Result<PointCloud<NormalPoint3f>>` - A new point cloud with normals
pub fn estimate_normals(cloud: &PointCloud<Point3f>, k: usize) -> Result<PointCloud<NormalPoint3f>> {
    let config = NormalEstimationConfig {
        k_neighbors: k,
        ..Default::default()
    };
    estimate_normals_with_config(cloud, &config)
}

/// Estimate normals with advanced configuration
/// 
/// # Arguments
/// * `cloud` - Reference to the point cloud
/// * `config` - Configuration for normal estimation
/// 
/// # Returns
/// * `Result<PointCloud<NormalPoint3f>>` - A new point cloud with normals
pub fn estimate_normals_with_config(
    cloud: &PointCloud<Point3f>, 
    config: &NormalEstimationConfig
) -> Result<PointCloud<NormalPoint3f>> {
    if cloud.is_empty() {
        return Ok(PointCloud::new());
    }
    
    if config.k_neighbors < 3 {
        return Err(Error::InvalidData("k_neighbors must be at least 3".to_string()));
    }
    
    let points = &cloud.points;
    
    // Determine viewpoint for orientation consistency
    let viewpoint = config.viewpoint.unwrap_or_else(|| {
        // Default viewpoint: compute a good viewpoint based on the point cloud bounds
        let mut min_x = points[0].x;
        let mut min_y = points[0].y;
        let mut min_z = points[0].z;
        let mut max_x = points[0].x;
        let mut max_y = points[0].y;
        let mut max_z = points[0].z;
        
        for point in points {
            min_x = min_x.min(point.x);
            min_y = min_y.min(point.y);
            min_z = min_z.min(point.z);
            max_x = max_x.max(point.x);
            max_y = max_y.max(point.y);
            max_z = max_z.max(point.z);
        }
        
        let center = Point3f::new(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0,
        );
        let extent = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2) + (max_z - min_z).powi(2)).sqrt();
        
        // Viewpoint is above the center of the point cloud
        center + Vector3f::new(0.0, 0.0, extent)
    });
    
    // Compute normals in parallel
    let normals: Vec<NormalPoint3f> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            let neighbors = find_neighbors(points, i, config);
            
            // Use only the neighbors for PCA, not the query point itself
            let mut neighborhood = neighbors;
            
            // If radius-based search didn't find enough neighbors, fall back to k-NN
            if config.radius.is_some() && neighborhood.len() < config.k_neighbors {
                neighborhood = find_k_nearest_neighbors(points, i, config.k_neighbors);
            }
            
            // Ensure we have enough neighbors for PCA
            if neighborhood.len() < 3 {
                // If we still don't have enough neighbors, use a larger k
                neighborhood = find_k_nearest_neighbors(points, i, config.k_neighbors.max(5));
            }
            
            let mut normal = compute_normal_pca(points, &neighborhood);
            
            // Apply orientation consistency if requested
            if config.consistent_orientation {
                normal = orient_normal_towards_viewpoint(normal, points[i], viewpoint);
            }
            
            NormalPoint3f {
                position: points[i],
                normal,
            }
        })
        .collect();
    
    Ok(PointCloud::from_points(normals))
}

/// Estimate normals using radius-based neighbor search
/// 
/// # Arguments
/// * `cloud` - Reference to the point cloud
/// * `radius` - Search radius for neighbors
/// * `consistent_orientation` - Whether to enforce orientation consistency
/// 
/// # Returns
/// * `Result<PointCloud<NormalPoint3f>>` - A new point cloud with normals
pub fn estimate_normals_radius(
    cloud: &PointCloud<Point3f>, 
    radius: f32, 
    consistent_orientation: bool
) -> Result<PointCloud<NormalPoint3f>> {
    let config = NormalEstimationConfig {
        k_neighbors: 10, // Fallback value
        radius: Some(radius),
        consistent_orientation,
        viewpoint: None,
    };
    estimate_normals_with_config(cloud, &config)
}

/// Estimate normals and modify the input cloud in-place (legacy API)
/// This function is deprecated in favor of the version that returns a new cloud
#[deprecated(note = "Use estimate_normals instead which returns a new point cloud")]
pub fn estimate_normals_inplace(_cloud: &mut PointCloud<Point3f>, k: usize) -> Result<()> {
    // This would require converting the point cloud type, which isn't straightforward
    // with the current type system. The new API is cleaner.
    let _ = k;
    Err(Error::Unsupported("Use estimate_normals instead".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_normals_simple() {
        // Create a simple planar point cloud (XY plane)
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        cloud.push(Point3f::new(1.0, 1.0, 0.0));
        cloud.push(Point3f::new(0.5, 0.5, 0.0));
        
        let result = estimate_normals(&cloud, 3).unwrap();
        
        assert_eq!(result.len(), 5);
        
        // For a planar surface in XY plane, normals should point along Z axis
        for point in result.iter() {
            let normal = point.normal;
            // Normal should be close to (0, 0, 1) or (0, 0, -1)
            assert!(normal.z.abs() > 0.8, "Normal should be primarily in Z direction: {:?}", normal);
        }
    }
    
    #[test]
    fn test_estimate_normals_empty() {
        let cloud = PointCloud::<Point3f>::new();
        let result = estimate_normals(&cloud, 5).unwrap();
        assert!(result.is_empty());
    }
    
    #[test]
    fn test_estimate_normals_insufficient_k() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        
        let result = estimate_normals(&cloud, 2);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_estimate_normals_radius() {
        // Create a simple planar point cloud for testing radius-based search
        let mut cloud = PointCloud::new();
        for i in 0..20 {
            for j in 0..20 {
                let x = (i as f32) * 0.1;
                let y = (j as f32) * 0.1;
                let z = 0.0;
                cloud.push(Point3f::new(x, y, z));
            }
        }
        
        let result = estimate_normals_radius(&cloud, 0.2, true).unwrap();
        assert_eq!(result.len(), 400);
        
        // Check that normals are computed and have reasonable values
        let mut z_direction_count = 0;
        for point in result.iter() {
            let normal_magnitude = point.normal.magnitude();
            // Normals should be unit vectors
            assert!((normal_magnitude - 1.0).abs() < 0.1, "Normal should be unit vector: magnitude={}", normal_magnitude);
            
            // For a planar surface, normals should be primarily in Z direction
            if point.normal.z.abs() > 0.8 {
                z_direction_count += 1;
            }
        }
        
        // At least 80% of normals should be in Z direction for a planar surface
        let percentage = (z_direction_count as f32 / result.len() as f32) * 100.0;
        assert!(percentage > 80.0, "Only {:.1}% of normals are in Z direction", percentage);
    }
    
    #[test]
    fn test_estimate_normals_cylinder() {
        // Create a true cylindrical point cloud (points on a cylinder surface)
        let mut cloud = PointCloud::new();
        for i in 0..10 {
            for j in 0..10 {
                let angle = (i as f32) * 0.6;
                let height = (j as f32) * 0.2 - 1.0;
                let x = angle.cos();
                let y = angle.sin();
                let z = height;
                cloud.push(Point3f::new(x, y, z));
            }
        }
        
        let config = NormalEstimationConfig {
            k_neighbors: 8, // Increase k for better results
            radius: None,
            consistent_orientation: true,
            viewpoint: Some(Point3f::new(0.0, 0.0, 2.0)), // View from above
        };
        
        let result = estimate_normals_with_config(&cloud, &config).unwrap();
        assert_eq!(result.len(), 100);
        
        // Check that normals are computed and have reasonable values
        let mut perpendicular_count = 0;
        let mut outward_count = 0;
        for point in result.iter() {
            let normal_magnitude = point.normal.magnitude();
            // Normals should be unit vectors
            assert!((normal_magnitude - 1.0).abs() < 0.1, "Normal should be unit vector: magnitude={}", normal_magnitude);
            
            // For a cylinder, normals should be roughly perpendicular to the cylinder axis (Z-axis)
            let dot_with_z = point.normal.z.abs();
            if dot_with_z < 0.8 {
                perpendicular_count += 1;
            }
            
            // Check if normal points outward from center
            let to_center = Vector3f::new(-point.position.x, -point.position.y, 0.0).normalize();
            let dot_outward = point.normal.dot(&to_center);
            if dot_outward > 0.5 {
                outward_count += 1;
            }
        }
        
        // At least 60% of normals should be perpendicular to Z-axis for a cylinder
        let percentage_perpendicular = (perpendicular_count as f32 / result.len() as f32) * 100.0;
        let percentage_outward = (outward_count as f32 / result.len() as f32) * 100.0;
        
        println!("Cylinder test: {:.1}% perpendicular to Z, {:.1}% pointing outward", 
                percentage_perpendicular, percentage_outward);
        
        // For a cylinder, normals should be perpendicular to Z-axis
        assert!(percentage_perpendicular > 60.0, "Only {:.1}% of normals are perpendicular to Z-axis", percentage_perpendicular);
    }
    
    #[test]
    fn test_estimate_normals_orientation_consistency() {
        // Create a simple planar point cloud
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        cloud.push(Point3f::new(1.0, 1.0, 0.0));
        
        // Test with orientation consistency enabled
        let config_consistent = NormalEstimationConfig {
            k_neighbors: 3,
            radius: None,
            consistent_orientation: true,
            viewpoint: Some(Point3f::new(0.0, 0.0, 1.0)), // View from positive Z
        };
        
        let result_consistent = estimate_normals_with_config(&cloud, &config_consistent).unwrap();
        
        // Test with orientation consistency disabled
        let config_inconsistent = NormalEstimationConfig {
            k_neighbors: 3,
            radius: None,
            consistent_orientation: false,
            viewpoint: None,
        };
        
        let _result_inconsistent = estimate_normals_with_config(&cloud, &config_inconsistent).unwrap();
        
        // With consistent orientation, all normals should point in the same direction (positive Z)
        let first_normal_consistent = result_consistent.points[0].normal.z;
        for point in result_consistent.iter() {
            assert!((point.normal.z * first_normal_consistent) > 0.0, 
                   "Normals should have consistent orientation");
        }
        
        // Without consistent orientation, normals might point in different directions
        // (This test is less strict since the algorithm might still produce consistent results)
        println!("Consistent orientation test completed");
    }
    
    #[test]
    fn test_find_neighbors() {
        let points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(2.0, 0.0, 0.0),
        ];
        
        // Test k-NN
        let config_knn = NormalEstimationConfig {
            k_neighbors: 2,
            radius: None,
            consistent_orientation: false,
            viewpoint: None,
        };
        
        let neighbors_knn = find_neighbors(&points, 0, &config_knn);
        assert_eq!(neighbors_knn.len(), 2);
        
        // Test radius-based
        let config_radius = NormalEstimationConfig {
            k_neighbors: 10,
            radius: Some(1.5),
            consistent_orientation: false,
            viewpoint: None,
        };
        
        let neighbors_radius = find_neighbors(&points, 0, &config_radius);
        assert_eq!(neighbors_radius.len(), 2); // Points at (1,0,0) and (0,1,0) are within radius 1.5
    }
} 