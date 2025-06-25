//! Normal estimation algorithms

use threecrate_core::{PointCloud, Result, Point3f, Vector3f, NormalPoint3f, Error};
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

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
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
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
    let normal = eigenvectors.column(min_idx).into();
    
    // Ensure consistent orientation (optional: could orient towards viewpoint)
    normal
}

/// Estimate normals for a point cloud using k-nearest neighbors
/// 
/// This function computes surface normals for each point in the cloud by:
/// 1. Finding k nearest neighbors for each point
/// 2. Computing the normal using Principal Component Analysis (PCA)
/// 3. The normal is the eigenvector corresponding to the smallest eigenvalue
/// 
/// # Arguments
/// * `cloud` - Mutable reference to the point cloud
/// * `k` - Number of nearest neighbors to use (typically 10-30)
/// 
/// # Returns
/// * `Result<PointCloud<NormalPoint3f>>` - A new point cloud with normals
pub fn estimate_normals(cloud: &PointCloud<Point3f>, k: usize) -> Result<PointCloud<NormalPoint3f>> {
    if cloud.is_empty() {
        return Ok(PointCloud::new());
    }
    
    if k < 3 {
        return Err(Error::InvalidData("k must be at least 3".to_string()));
    }
    
    let points = &cloud.points;
    
    // Compute normals in parallel
    let normals: Vec<NormalPoint3f> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            let neighbors = find_k_nearest_neighbors(points, i, k);
            let mut neighborhood = vec![i]; // Include the point itself
            neighborhood.extend(neighbors);
            
            let normal = compute_normal_pca(points, &neighborhood);
            
            NormalPoint3f {
                position: points[i],
                normal,
            }
        })
        .collect();
    
    Ok(PointCloud::from_points(normals))
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
} 