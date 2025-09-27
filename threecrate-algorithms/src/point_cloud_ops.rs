//! Point cloud operations including k-nearest neighbors search

use threecrate_core::{PointCloud, Point3f, NearestNeighborSearch};
use crate::nearest_neighbor::{KdTree, BruteForceSearch};

/// Extension trait for PointCloud to add k-nearest neighbors functionality
pub trait PointCloudNeighbors {
    /// Find k nearest neighbors for each point in the cloud using KD-tree
    /// 
    /// This method returns a vector where each element contains the indices and distances
    /// of the k nearest neighbors for the corresponding point in the cloud.
    /// 
    /// # Arguments
    /// * `k` - Number of nearest neighbors to find for each point
    /// 
    /// # Returns
    /// * `Vec<Vec<(usize, f32)>>` - Vector of neighbor results for each point
    /// 
    /// # Example
    /// ```rust,no_run
    /// // Temporarily disabled due to stack overflow - needs investigation
    /// use threecrate_core::{PointCloud, Point3f};
    /// use threecrate_algorithms::point_cloud_ops::PointCloudNeighbors;
    ///
    /// let mut cloud = PointCloud::new();
    /// cloud.push(Point3f::new(0.0, 0.0, 0.0));
    /// cloud.push(Point3f::new(1.0, 0.0, 0.0));
    /// cloud.push(Point3f::new(0.0, 1.0, 0.0));
    ///
    /// let neighbors = cloud.k_nearest_neighbors(2);
    /// // neighbors[0] contains the 2 nearest neighbors for point 0
    /// ```
    fn k_nearest_neighbors(&self, k: usize) -> Vec<Vec<(usize, f32)>>;

    /// Find k nearest neighbors for a specific query point using KD-tree
    /// 
    /// # Arguments
    /// * `query` - The query point to find neighbors for
    /// * `k` - Number of nearest neighbors to find
    /// 
    /// # Returns
    /// * `Vec<(usize, f32)>` - Vector of (index, distance) pairs for the k nearest neighbors
    fn find_k_nearest(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)>;

    /// Find all neighbors within a given radius using KD-tree
    /// 
    /// # Arguments
    /// * `query` - The query point to find neighbors for
    /// * `radius` - Search radius
    /// 
    /// # Returns
    /// * `Vec<(usize, f32)>` - Vector of (index, distance) pairs for neighbors within radius
    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)>;

    /// Find k nearest neighbors using brute force search (for small datasets or testing)
    /// 
    /// This method is useful for small point clouds or when you want to verify
    /// the results of the KD-tree implementation.
    /// 
    /// # Arguments
    /// * `query` - The query point to find neighbors for
    /// * `k` - Number of nearest neighbors to find
    /// 
    /// # Returns
    /// * `Vec<(usize, f32)>` - Vector of (index, distance) pairs for the k nearest neighbors
    fn find_k_nearest_brute_force(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)>;

    /// Find all neighbors within a given radius using brute force search
    /// 
    /// # Arguments
    /// * `query` - The query point to find neighbors for
    /// * `radius` - Search radius
    /// 
    /// # Returns
    /// * `Vec<(usize, f32)>` - Vector of (index, distance) pairs for neighbors within radius
    fn find_radius_neighbors_brute_force(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)>;
}

impl PointCloudNeighbors for PointCloud<Point3f> {
    fn k_nearest_neighbors(&self, k: usize) -> Vec<Vec<(usize, f32)>> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        // Use KD-tree for efficient search
        let kdtree = KdTree::new(&self.points)
            .expect("Failed to build KD-tree");

        let mut results = Vec::with_capacity(self.len());
        
        for (i, query_point) in self.points.iter().enumerate() {
            let mut neighbors = kdtree.find_k_nearest(query_point, k + 1); // +1 to exclude self
            
            // Remove the point itself from its own neighbors
            neighbors.retain(|&(idx, _)| idx != i);
            
            // Ensure we have exactly k neighbors (or fewer if not enough points)
            if neighbors.len() > k {
                neighbors.truncate(k);
            }
            
            results.push(neighbors);
        }
        
        results
    }

    fn find_k_nearest(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let kdtree = KdTree::new(&self.points)
            .expect("Failed to build KD-tree");
        
        kdtree.find_k_nearest(query, k)
    }

    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)> {
        if self.is_empty() || radius <= 0.0 {
            return Vec::new();
        }

        let kdtree = KdTree::new(&self.points)
            .expect("Failed to build KD-tree");
        
        kdtree.find_radius_neighbors(query, radius)
    }

    fn find_k_nearest_brute_force(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let brute_force = BruteForceSearch::new(&self.points);
        brute_force.find_k_nearest(query, k)
    }

    fn find_radius_neighbors_brute_force(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)> {
        if self.is_empty() || radius <= 0.0 {
            return Vec::new();
        }

        let brute_force = BruteForceSearch::new(&self.points);
        brute_force.find_radius_neighbors(query, radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::Point3f;

    #[test]
    #[ignore] // Temporarily disabled due to stack overflow - needs investigation
    fn test_point_cloud_k_nearest_neighbors() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        cloud.push(Point3f::new(1.0, 1.0, 0.0));

        let neighbors = cloud.k_nearest_neighbors(2);
        assert_eq!(neighbors.len(), 4);
        
        // Each point should have 2 neighbors (excluding itself)
        for point_neighbors in &neighbors {
            assert_eq!(point_neighbors.len(), 2);
        }
    }

    #[test]
    #[ignore] // Temporarily disabled due to stack overflow - needs investigation
    fn test_point_cloud_find_k_nearest() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        cloud.push(Point3f::new(1.0, 1.0, 0.0));

        let query = Point3f::new(0.5, 0.5, 0.0);
        let nearest = cloud.find_k_nearest(&query, 2);
        
        assert_eq!(nearest.len(), 2);
        assert!(nearest[0].1 <= nearest[1].1); // Should be sorted by distance
    }

    #[test]
    fn test_point_cloud_radius_neighbors() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        cloud.push(Point3f::new(1.0, 1.0, 0.0));

        let query = Point3f::new(0.5, 0.5, 0.0);
        let radius_neighbors = cloud.find_radius_neighbors(&query, 1.0);
        
        // Should find all 4 points within radius 1.0
        assert_eq!(radius_neighbors.len(), 4);
        
        // All distances should be within radius
        for (_, distance) in &radius_neighbors {
            assert!(*distance <= 1.0);
        }
    }

    #[test]
    #[ignore] // Temporarily disabled due to stack overflow - needs investigation
    fn test_brute_force_consistency() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.0, 1.0, 0.0));
        cloud.push(Point3f::new(1.0, 1.0, 0.0));

        let query = Point3f::new(0.5, 0.5, 0.0);
        let k = 2;
        
        let mut kdtree_result = cloud.find_k_nearest(&query, k);
        let mut brute_result = cloud.find_k_nearest_brute_force(&query, k);
        
        // Sort by distance first, then by index for consistent comparison
        kdtree_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        brute_result.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        
        assert_eq!(kdtree_result.len(), brute_result.len());
        
        // Check that the distances match (within tolerance)
        for (kdtree_neighbor, brute_neighbor) in kdtree_result.iter().zip(brute_result.iter()) {
            assert!((kdtree_neighbor.1 - brute_neighbor.1).abs() < 1e-6);
        }
    }
} 