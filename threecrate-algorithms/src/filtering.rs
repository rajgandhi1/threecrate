//! Filtering algorithms

use threecrate_core::{PointCloud, Result, Point3f, NearestNeighborSearch};
use crate::nearest_neighbor::BruteForceSearch;
use rayon::prelude::*;

/// Voxel grid filtering
/// 
/// This algorithm reduces the density of a point cloud by grouping points into voxels
/// and keeping only one representative point per voxel. This is useful for downsampling
/// large point clouds while preserving their overall structure.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `voxel_size` - Size of each voxel cube
/// 
/// # Returns
/// * `Result<PointCloud<Point3f>>` - Downsampled point cloud
/// 
/// # Example
/// ```rust
/// use threecrate_core::{PointCloud, Point3f};
/// use threecrate_algorithms::voxel_grid_filter;
/// 
/// fn main() -> threecrate_core::Result<()> {
///     let cloud = PointCloud::from_points(vec![
///         Point3f::new(0.0, 0.0, 0.0),
///         Point3f::new(0.1, 0.0, 0.0),
///         Point3f::new(0.0, 0.1, 0.0),
///         Point3f::new(0.0, 0.0, 0.1),
///     ]);
/// 
///     let filtered = voxel_grid_filter(&cloud, 0.2)?;
///     println!("Filtered cloud has {} points", filtered.len());
///     Ok(())
/// }
/// ```
pub fn voxel_grid_filter(cloud: &PointCloud<Point3f>, voxel_size: f32) -> Result<PointCloud<Point3f>> {
    if cloud.is_empty() {
        return Ok(PointCloud::new());
    }
    
    if voxel_size <= 0.0 {
        return Err(threecrate_core::Error::InvalidData(
            "voxel_size must be positive".to_string()
        ));
    }
    
    // Compute bounding box
    let min_x = cloud.points.iter().map(|p| p.x).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_y = cloud.points.iter().map(|p| p.y).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_z = cloud.points.iter().map(|p| p.z).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_x = cloud.points.iter().map(|p| p.x).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_y = cloud.points.iter().map(|p| p.y).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_z = cloud.points.iter().map(|p| p.z).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    // Compute grid dimensions (for reference, not used in hash-based approach)
    let _grid_dim_x = ((max_x - min_x) / voxel_size).ceil() as i32 + 1;
    let _grid_dim_y = ((max_y - min_y) / voxel_size).ceil() as i32 + 1;
    let _grid_dim_z = ((max_z - min_z) / voxel_size).ceil() as i32 + 1;
    
    // Function to get voxel coordinates for a point
    let get_voxel_coords = |point: &Point3f| -> (i32, i32, i32) {
        let x = ((point.x - min_x) / voxel_size).floor() as i32;
        let y = ((point.y - min_y) / voxel_size).floor() as i32;
        let z = ((point.z - min_z) / voxel_size).floor() as i32;
        (x, y, z)
    };
    
    // Group points by voxel
    let mut voxel_map = std::collections::HashMap::new();
    
    for (idx, point) in cloud.points.iter().enumerate() {
        let voxel_coords = get_voxel_coords(point);
        voxel_map.entry(voxel_coords).or_insert_with(Vec::new).push(idx);
    }
    
    // Keep one point per voxel (the first one)
    let filtered_points: Vec<Point3f> = voxel_map
        .values()
        .map(|indices| cloud.points[indices[0]])
        .collect();
    
    Ok(PointCloud::from_points(filtered_points))
}

/// Radius outlier removal filter
/// 
/// This algorithm removes points that have fewer than `min_neighbors` neighbors
/// within a specified radius. This is useful for removing isolated noise points
/// that are far from the main point cloud.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `radius` - Search radius for neighbor counting
/// * `min_neighbors` - Minimum number of neighbors required to keep a point
/// 
/// # Returns
/// * `Result<PointCloud<Point3f>>` - Filtered point cloud with outliers removed
/// 
/// # Example
/// ```rust
/// use threecrate_core::{PointCloud, Point3f};
/// use threecrate_algorithms::radius_outlier_removal;
/// 
/// fn main() -> threecrate_core::Result<()> {
///     let cloud = PointCloud::from_points(vec![
///         Point3f::new(0.0, 0.0, 0.0),
///         Point3f::new(0.1, 0.0, 0.0),
///         Point3f::new(0.0, 0.1, 0.0),
///         Point3f::new(10.0, 10.0, 10.0), // outlier
///     ]);
/// 
///     let filtered = radius_outlier_removal(&cloud, 0.5, 2)?;
///     println!("Filtered cloud has {} points", filtered.len());
///     Ok(())
/// }
/// ```
pub fn radius_outlier_removal(
    cloud: &PointCloud<Point3f>,
    radius: f32,
    min_neighbors: usize,
) -> Result<PointCloud<Point3f>> {
    if cloud.is_empty() {
        return Ok(PointCloud::new());
    }
    
    if radius <= 0.0 {
        return Err(threecrate_core::Error::InvalidData(
            "radius must be positive".to_string()
        ));
    }
    
    if min_neighbors == 0 {
        return Err(threecrate_core::Error::InvalidData(
            "min_neighbors must be greater than 0".to_string()
        ));
    }
    
    // Create nearest neighbor search structure
    let nn_search = BruteForceSearch::new(&cloud.points);
    
    // Count neighbors within radius for each point
    let neighbor_counts: Vec<usize> = cloud.points
        .par_iter()
        .map(|point| {
            let neighbors = nn_search.find_radius_neighbors(point, radius);
            // Subtract 1 to exclude the point itself
            neighbors.len().saturating_sub(1)
        })
        .collect();
    
    // Filter out points with insufficient neighbors
    let filtered_points: Vec<Point3f> = cloud.points
        .iter()
        .zip(neighbor_counts.iter())
        .filter(|(_, &count)| count >= min_neighbors)
        .map(|(point, _)| *point)
        .collect();
    
    Ok(PointCloud::from_points(filtered_points))
}

/// Statistical outlier removal filter
/// 
/// This algorithm removes points that are statistical outliers based on the distance
/// to their k-nearest neighbors. For each point, it computes the mean distance to
/// its k nearest neighbors. Points with mean distances that deviate more than
/// `std_dev_multiplier` standard deviations from the global mean are considered
/// outliers and removed.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `k_neighbors` - Number of nearest neighbors to consider for each point
/// * `std_dev_multiplier` - Standard deviation multiplier for outlier detection
/// 
/// # Returns
/// * `Result<PointCloud<Point3f>>` - Filtered point cloud with outliers removed
/// 
/// # Example
/// ```rust
/// use threecrate_core::{PointCloud, Point3f};
/// use threecrate_algorithms::statistical_outlier_removal;
/// 
/// fn main() -> threecrate_core::Result<()> {
///     let cloud = PointCloud::from_points(vec![
///         Point3f::new(0.0, 0.0, 0.0),
///         Point3f::new(1.0, 0.0, 0.0),
///         Point3f::new(0.0, 1.0, 0.0),
///         Point3f::new(10.0, 10.0, 10.0), // outlier
///     ]);
/// 
///     let filtered = statistical_outlier_removal(&cloud, 3, 1.0)?;
///     println!("Filtered cloud has {} points", filtered.len());
///     Ok(())
/// }
/// ```
pub fn statistical_outlier_removal(
    cloud: &PointCloud<Point3f>,
    k_neighbors: usize,
    std_dev_multiplier: f32,
) -> Result<PointCloud<Point3f>> {
    if cloud.is_empty() {
        return Ok(PointCloud::new());
    }
    
    if k_neighbors == 0 {
        return Err(threecrate_core::Error::InvalidData(
            "k_neighbors must be greater than 0".to_string()
        ));
    }
    
    if std_dev_multiplier <= 0.0 {
        return Err(threecrate_core::Error::InvalidData(
            "std_dev_multiplier must be positive".to_string()
        ));
    }
    
    // Create nearest neighbor search structure
    let nn_search = BruteForceSearch::new(&cloud.points);
    
    // Compute mean distances for all points
    let mean_distances: Vec<f32> = cloud.points
        .par_iter()
        .map(|point| {
            let neighbors = nn_search.find_k_nearest(point, k_neighbors + 1); // +1 to exclude self
            if neighbors.is_empty() {
                return 0.0;
            }
            
            // Calculate mean distance to neighbors (skip first neighbor if it's the point itself)
            let distances: Vec<f32> = neighbors
                .iter()
                .filter(|(idx, _)| cloud.points[*idx] != *point) // Skip self
                .map(|(_, distance)| *distance)
                .collect();
            
            if distances.is_empty() {
                return 0.0;
            }
            
            distances.iter().sum::<f32>() / distances.len() as f32
        })
        .collect();
    
    // Compute global statistics
    let global_mean = mean_distances.iter().sum::<f32>() / mean_distances.len() as f32;
    
    let variance = mean_distances
        .iter()
        .map(|&d| (d - global_mean).powi(2))
        .sum::<f32>() / mean_distances.len() as f32;
    
    let global_std_dev = variance.sqrt();
    let threshold = global_mean + std_dev_multiplier * global_std_dev;
    
    // Filter out outliers
    let filtered_points: Vec<Point3f> = cloud.points
        .iter()
        .zip(mean_distances.iter())
        .filter(|(_, &mean_dist)| mean_dist <= threshold)
        .map(|(point, _)| *point)
        .collect();
    
    Ok(PointCloud::from_points(filtered_points))
}

/// Statistical outlier removal with custom threshold
/// 
/// This variant allows you to specify a custom threshold instead of using
/// the automatic standard deviation calculation.
/// 
/// # Arguments
/// * `cloud` - Input point cloud
/// * `k_neighbors` - Number of nearest neighbors to consider for each point
/// * `threshold` - Custom threshold for outlier detection
/// 
/// # Returns
/// * `Result<PointCloud<Point3f>>` - Filtered point cloud with outliers removed
pub fn statistical_outlier_removal_with_threshold(
    cloud: &PointCloud<Point3f>,
    k_neighbors: usize,
    threshold: f32,
) -> Result<PointCloud<Point3f>> {
    if cloud.is_empty() {
        return Ok(PointCloud::new());
    }
    
    if k_neighbors == 0 {
        return Err(threecrate_core::Error::InvalidData(
            "k_neighbors must be greater than 0".to_string()
        ));
    }
    
    if threshold <= 0.0 {
        return Err(threecrate_core::Error::InvalidData(
            "threshold must be positive".to_string()
        ));
    }
    
    // Create nearest neighbor search structure
    let nn_search = BruteForceSearch::new(&cloud.points);
    
    // Compute mean distances for all points
    let mean_distances: Vec<f32> = cloud.points
        .par_iter()
        .map(|point| {
            let neighbors = nn_search.find_k_nearest(point, k_neighbors + 1); // +1 to exclude self
            if neighbors.is_empty() {
                return 0.0;
            }
            
            // Calculate mean distance to neighbors (skip first neighbor if it's the point itself)
            let distances: Vec<f32> = neighbors
                .iter()
                .filter(|(idx, _)| cloud.points[*idx] != *point) // Skip self
                .map(|(_, distance)| *distance)
                .collect();
            
            if distances.is_empty() {
                return 0.0;
            }
            
            distances.iter().sum::<f32>() / distances.len() as f32
        })
        .collect();
    
    // Filter out outliers using custom threshold
    let filtered_points: Vec<Point3f> = cloud.points
        .iter()
        .zip(mean_distances.iter())
        .filter(|(_, &mean_dist)| mean_dist <= threshold)
        .map(|(point, _)| *point)
        .collect();
    
    Ok(PointCloud::from_points(filtered_points))
} 

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::Point3f;

    #[test]
    fn test_statistical_outlier_removal_empty_cloud() {
        let cloud = PointCloud::<Point3f>::new();
        let result = statistical_outlier_removal(&cloud, 5, 1.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_statistical_outlier_removal_single_point() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = statistical_outlier_removal(&cloud, 1, 1.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_statistical_outlier_removal_with_outliers() {
        // Create a point cloud with some outliers
        let mut points = Vec::new();
        
        // Main cluster
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    points.push(Point3f::new(
                        i as f32 * 0.1,
                        j as f32 * 0.1,
                        k as f32 * 0.1,
                    ));
                }
            }
        }
        
        // Add some outliers
        points.push(Point3f::new(10.0, 10.0, 10.0));
        points.push(Point3f::new(-10.0, -10.0, -10.0));
        points.push(Point3f::new(5.0, 5.0, 5.0));
        
        let cloud = PointCloud::from_points(points);
        let original_count = cloud.len();
        
        let result = statistical_outlier_removal(&cloud, 5, 1.0);
        assert!(result.is_ok());
        
        let filtered = result.unwrap();
        assert!(filtered.len() < original_count);
        assert!(filtered.len() > 0);
        
        // Check that outliers were removed
        let has_outlier_1 = filtered.points.iter().any(|p| 
            (p.x - 10.0).abs() < 0.1 && (p.y - 10.0).abs() < 0.1 && (p.z - 10.0).abs() < 0.1
        );
        let has_outlier_2 = filtered.points.iter().any(|p| 
            (p.x + 10.0).abs() < 0.1 && (p.y + 10.0).abs() < 0.1 && (p.z + 10.0).abs() < 0.1
        );
        
        assert!(!has_outlier_1);
        assert!(!has_outlier_2);
    }

    #[test]
    fn test_statistical_outlier_removal_no_outliers() {
        // Create a uniform point cloud without outliers
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    points.push(Point3f::new(
                        i as f32 * 0.1,
                        j as f32 * 0.1,
                        k as f32 * 0.1,
                    ));
                }
            }
        }
        
        let cloud = PointCloud::from_points(points);
        let original_count = cloud.len();
        
        let result = statistical_outlier_removal(&cloud, 5, 1.0);
        assert!(result.is_ok());
        
        let filtered = result.unwrap();
        // Should keep most points since there are no real outliers
        assert!(filtered.len() > original_count * 8 / 10);
    }

    #[test]
    fn test_statistical_outlier_removal_invalid_k() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = statistical_outlier_removal(&cloud, 0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_statistical_outlier_removal_invalid_std_dev() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = statistical_outlier_removal(&cloud, 5, 0.0);
        assert!(result.is_err());
        
        let result = statistical_outlier_removal(&cloud, 5, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_statistical_outlier_removal_with_threshold() {
        // Create a point cloud with known outliers
        let points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(0.1, 0.0, 0.0),
            Point3f::new(0.0, 0.1, 0.0),
            Point3f::new(0.0, 0.0, 0.1),
            Point3f::new(10.0, 10.0, 10.0), // outlier
        ];
        
        let cloud = PointCloud::from_points(points);
        
        // Use a very low threshold to remove outliers
        let result = statistical_outlier_removal_with_threshold(&cloud, 3, 0.5);
        assert!(result.is_ok());
        
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 4); // Should remove the outlier
        
        // Check that the outlier was removed
        let has_outlier = filtered.points.iter().any(|p| 
            (p.x - 10.0).abs() < 0.1 && (p.y - 10.0).abs() < 0.1 && (p.z - 10.0).abs() < 0.1
        );
        assert!(!has_outlier);
    }

    #[test]
    fn test_statistical_outlier_removal_with_threshold_invalid() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = statistical_outlier_removal_with_threshold(&cloud, 0, 1.0);
        assert!(result.is_err());
        
        let result = statistical_outlier_removal_with_threshold(&cloud, 5, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_voxel_grid_filter_empty_cloud() {
        let cloud = PointCloud::<Point3f>::new();
        let result = voxel_grid_filter(&cloud, 0.1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_voxel_grid_filter_single_point() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = voxel_grid_filter(&cloud, 0.1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_voxel_grid_filter_with_duplicates() {
        let cloud = PointCloud::from_points(vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(0.0, 0.0, 0.0), // duplicate
            Point3f::new(0.1, 0.0, 0.0),
            Point3f::new(0.1, 0.0, 0.0), // duplicate
            Point3f::new(0.0, 0.1, 0.0),
        ]);
        
        let result = voxel_grid_filter(&cloud, 0.05);
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), 3); // Should remove duplicates
    }

    #[test]
    fn test_voxel_grid_filter_invalid_voxel_size() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = voxel_grid_filter(&cloud, 0.0);
        assert!(result.is_err());
        
        let result = voxel_grid_filter(&cloud, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_radius_outlier_removal_empty_cloud() {
        let cloud = PointCloud::<Point3f>::new();
        let result = radius_outlier_removal(&cloud, 0.5, 3);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_radius_outlier_removal_single_point() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = radius_outlier_removal(&cloud, 0.5, 1);
        assert!(result.is_ok());
        // Should remove the single point since it has no neighbors
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_radius_outlier_removal_with_outliers() {
        // Create a point cloud with some outliers
        let mut points = Vec::new();
        
        // Main cluster
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3f::new(
                    i as f32 * 0.1,
                    j as f32 * 0.1,
                    0.0,
                ));
            }
        }
        
        // Add some outliers
        points.push(Point3f::new(10.0, 10.0, 10.0));
        points.push(Point3f::new(-10.0, -10.0, -10.0));
        
        let cloud = PointCloud::from_points(points);
        let original_count = cloud.len();
        
        let result = radius_outlier_removal(&cloud, 0.5, 2);
        assert!(result.is_ok());
        
        let filtered = result.unwrap();
        assert!(filtered.len() < original_count);
        assert!(filtered.len() > 0);
        
        // Check that outliers were removed
        let has_outlier_1 = filtered.points.iter().any(|p| 
            (p.x - 10.0).abs() < 0.1 && (p.y - 10.0).abs() < 0.1 && (p.z - 10.0).abs() < 0.1
        );
        let has_outlier_2 = filtered.points.iter().any(|p| 
            (p.x + 10.0).abs() < 0.1 && (p.y + 10.0).abs() < 0.1 && (p.z + 10.0).abs() < 0.1
        );
        
        assert!(!has_outlier_1);
        assert!(!has_outlier_2);
    }

    #[test]
    fn test_radius_outlier_removal_invalid_parameters() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        
        let result = radius_outlier_removal(&cloud, 0.0, 3);
        assert!(result.is_err());
        
        let result = radius_outlier_removal(&cloud, -1.0, 3);
        assert!(result.is_err());
        
        let result = radius_outlier_removal(&cloud, 0.5, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_radius_outlier_removal_different_parameters() {
        let mut points = Vec::new();
        
        // Create a cluster
        for i in 0..3 {
            for j in 0..3 {
                points.push(Point3f::new(
                    i as f32 * 0.1,
                    j as f32 * 0.1,
                    0.0,
                ));
            }
        }
        
        // Add some outliers
        points.push(Point3f::new(1.0, 1.0, 1.0));
        
        let cloud = PointCloud::from_points(points);
        
        // Test different radius values
        let result1 = radius_outlier_removal(&cloud, 0.2, 2).unwrap();
        let result2 = radius_outlier_removal(&cloud, 0.5, 2).unwrap();
        
        // Larger radius should keep more points
        assert!(result2.len() >= result1.len());
    }
} 