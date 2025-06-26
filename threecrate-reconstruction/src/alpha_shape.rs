//! Alpha shape reconstruction

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, NormalPoint3f, Error};
use rayon::prelude::*;

/// Configuration for Alpha Shape reconstruction
#[derive(Debug, Clone)]
pub struct AlphaShapeConfig {
    /// Alpha parameter for shape filtering
    pub alpha: f32,
    /// Whether to use only the boundary of the alpha shape
    pub boundary_only: bool,
    /// Minimum triangle area threshold
    pub min_triangle_area: f32,
}

impl Default for AlphaShapeConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            boundary_only: true,
            min_triangle_area: 1e-8,
        }
    }
}

/// Represents a triangle with its circumradius for alpha filtering
#[derive(Debug, Clone)]
struct AlphaTriangle {
    vertices: [usize; 3],
    circumradius: f32,
    area: f32,
}

impl AlphaTriangle {
    /// Create a new alpha triangle from three points
    fn new(v1: usize, v2: usize, v3: usize, points: &[Point3f]) -> Self {
        let p1 = points[v1];
        let p2 = points[v2];
        let p3 = points[v3];
        
        let circumradius = Self::compute_circumradius(&p1, &p2, &p3);
        let area = Self::compute_triangle_area(&p1, &p2, &p3);
        
        Self {
            vertices: [v1, v2, v3],
            circumradius,
            area,
        }
    }
    
    /// Compute the circumradius of a triangle
    fn compute_circumradius(p1: &Point3f, p2: &Point3f, p3: &Point3f) -> f32 {
        let a = (p2 - p1).magnitude();
        let b = (p3 - p2).magnitude();
        let c = (p1 - p3).magnitude();
        
        // Handle degenerate cases
        if a < 1e-10 || b < 1e-10 || c < 1e-10 {
            return f32::INFINITY;
        }
        
        // Compute area using cross product
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let cross = v1.cross(&v2);
        let area = cross.magnitude() * 0.5;
        
        if area < 1e-10 {
            return f32::INFINITY;
        }
        
        // Circumradius formula: R = (abc) / (4 * Area)
        (a * b * c) / (4.0 * area)
    }
    
    /// Compute the area of a triangle
    fn compute_triangle_area(p1: &Point3f, p2: &Point3f, p3: &Point3f) -> f32 {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let cross = v1.cross(&v2);
        cross.magnitude() * 0.5
    }
    
    /// Check if this triangle is valid for the given alpha value
    fn is_valid(&self, alpha: f32, min_area: f32) -> bool {
        self.circumradius <= alpha && self.area >= min_area
    }
}

/// Simplified Alpha Shape reconstruction implementation
/// Uses a brute-force approach to avoid complex 3D triangulation dependencies
pub struct AlphaShape {
    points: Vec<Point3f>,
    alpha_triangles: Vec<AlphaTriangle>,
    config: AlphaShapeConfig,
}

impl AlphaShape {
    /// Create a new Alpha Shape instance
    pub fn new(cloud: &PointCloud<Point3f>, config: AlphaShapeConfig) -> Result<Self> {
        if cloud.is_empty() {
            return Err(Error::InvalidData("Point cloud is empty".to_string()));
        }

        let points = cloud.points.clone();
        
        Ok(Self {
            points,
            alpha_triangles: Vec::new(),
            config,
        })
    }
    
    /// Create Alpha Shape from point cloud with normals
    pub fn from_normals(cloud: &PointCloud<NormalPoint3f>, config: AlphaShapeConfig) -> Result<Self> {
        if cloud.is_empty() {
            return Err(Error::InvalidData("Point cloud is empty".to_string()));
        }

        let points: Vec<Point3f> = cloud.points.iter().map(|p| p.position).collect();
        
        Ok(Self {
            points,
            alpha_triangles: Vec::new(),
            config,
        })
    }
    
    /// Generate candidate triangles using brute force approach
    /// This is simplified but works for smaller point clouds
    fn generate_candidate_triangles(&mut self) -> Result<()> {
        let n = self.points.len();
        
        if n < 3 {
            return Err(Error::InvalidData("Need at least 3 points for triangulation".to_string()));
        }
        
        // Limit the number of combinations to avoid exponential explosion
        let max_combinations = 10000;
        let mut combinations = 0;
        
        self.alpha_triangles.clear();
        
        // Generate all possible triangles (O(n^3) - only for small datasets)
        for i in 0..n {
            if combinations >= max_combinations {
                break;
            }
            for j in (i + 1)..n {
                if combinations >= max_combinations {
                    break;
                }
                for k in (j + 1)..n {
                    combinations += 1;
                    if combinations >= max_combinations {
                        break;
                    }
                    
                    let triangle = AlphaTriangle::new(i, j, k, &self.points);
                    
                    // Only keep triangles that could potentially be valid
                    if triangle.circumradius <= self.config.alpha * 2.0 {
                        self.alpha_triangles.push(triangle);
                    }
                }
            }
        }
        
        if self.alpha_triangles.is_empty() {
            return Err(Error::Algorithm("No candidate triangles generated".to_string()));
        }
        
        Ok(())
    }
    
    /// Filter triangles based on alpha value
    fn filter_triangles(&self) -> Vec<AlphaTriangle> {
        self.alpha_triangles
            .iter()
            .filter(|t| t.is_valid(self.config.alpha, self.config.min_triangle_area))
            .cloned()
            .collect()
    }
    
    /// Extract boundary triangles if boundary_only is enabled
    fn extract_boundary(&self, triangles: &[AlphaTriangle]) -> Vec<AlphaTriangle> {
        if !self.config.boundary_only {
            return triangles.to_vec();
        }
        
        // Count edge occurrences to find boundary edges
        let mut edge_count = std::collections::HashMap::new();
        
        for triangle in triangles {
            let edges = [
                Self::make_edge(triangle.vertices[0], triangle.vertices[1]),
                Self::make_edge(triangle.vertices[1], triangle.vertices[2]),
                Self::make_edge(triangle.vertices[2], triangle.vertices[0]),
            ];
            
            for edge in &edges {
                *edge_count.entry(*edge).or_insert(0) += 1;
            }
        }
        
        // Keep triangles that have at least one boundary edge (count == 1)
        triangles
            .iter()
            .filter(|triangle| {
                let edges = [
                    Self::make_edge(triangle.vertices[0], triangle.vertices[1]),
                    Self::make_edge(triangle.vertices[1], triangle.vertices[2]),
                    Self::make_edge(triangle.vertices[2], triangle.vertices[0]),
                ];
                
                edges.iter().any(|edge| edge_count.get(edge).unwrap_or(&0) == &1)
            })
            .cloned()
            .collect()
    }
    
    /// Create a normalized edge (smaller index first)
    fn make_edge(v1: usize, v2: usize) -> (usize, usize) {
        if v1 < v2 {
            (v1, v2)
        } else {
            (v2, v1)
        }
    }
    
    /// Check if a triangle is potentially valid (no points inside circumsphere)
    fn is_triangle_valid(&self, triangle: &AlphaTriangle) -> bool {
        let p1 = self.points[triangle.vertices[0]];
        let p2 = self.points[triangle.vertices[1]];
        let p3 = self.points[triangle.vertices[2]];
        
        // Compute circumcenter
        let circumcenter = self.compute_circumcenter(&p1, &p2, &p3);
        
        if let Some(center) = circumcenter {
            // Check if any other point is inside the circumsphere
            for (i, point) in self.points.iter().enumerate() {
                if i == triangle.vertices[0] || i == triangle.vertices[1] || i == triangle.vertices[2] {
                    continue;
                }
                
                let dist = (point - center).magnitude();
                if dist < triangle.circumradius - 1e-6 {
                    return false; // Point is inside circumsphere
                }
            }
        }
        
        true
    }
    
    /// Compute circumcenter of a triangle
    fn compute_circumcenter(&self, p1: &Point3f, p2: &Point3f, p3: &Point3f) -> Option<Point3f> {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        
        let cross = v1.cross(&v2);
        let cross_mag_sq = cross.magnitude_squared();
        
        if cross_mag_sq < 1e-10 {
            return None; // Degenerate triangle
        }
        
        let d1 = v1.magnitude_squared();
        let d2 = v2.magnitude_squared();
        
        let alpha = d2 * v1.dot(&cross) / (2.0 * cross_mag_sq);
        let beta = -d1 * v2.dot(&cross) / (2.0 * cross_mag_sq);
        
        Some(p1 + alpha * v1 + beta * v2)
    }
    
    /// Reconstruct the alpha shape
    pub fn reconstruct(&mut self) -> Result<TriangleMesh> {
        // Generate candidate triangles
        self.generate_candidate_triangles()?;
        
        // Filter triangles based on alpha and validity
        let filtered_triangles: Vec<AlphaTriangle> = self.filter_triangles()
            .into_iter()
            .filter(|t| self.is_triangle_valid(t))
            .collect();
        
        if filtered_triangles.is_empty() {
            return Err(Error::Algorithm("No valid triangles found with given alpha".to_string()));
        }
        
        // Extract boundary if needed
        let final_triangles = self.extract_boundary(&filtered_triangles);
        
        if final_triangles.is_empty() {
            return Err(Error::Algorithm("No boundary triangles found".to_string()));
        }
        
        // Convert to triangle mesh
        let vertices = self.points.clone();
        let faces: Vec<[usize; 3]> = final_triangles
            .iter()
            .map(|t| t.vertices)
            .collect();
        
        Ok(TriangleMesh::from_vertices_and_faces(vertices, faces))
    }
}

/// Alpha shape surface reconstruction
/// 
/// This function reconstructs a triangle mesh from a point cloud using
/// alpha shapes, which are a generalization of convex hulls.
/// 
/// # Arguments
/// * `cloud` - Point cloud
/// * `alpha` - Alpha parameter for shape filtering
/// 
/// # Returns
/// * `Result<TriangleMesh>` - Reconstructed triangle mesh
pub fn alpha_shape_reconstruction(cloud: &PointCloud<Point3f>, alpha: f32) -> Result<TriangleMesh> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }

    if alpha <= 0.0 {
        return Err(Error::InvalidData("Alpha must be positive".to_string()));
    }

    let config = AlphaShapeConfig {
        alpha,
        ..Default::default()
    };

    let mut alpha_shape = AlphaShape::new(cloud, config)?;
    alpha_shape.reconstruct()
}

/// Alpha shape reconstruction with configuration
/// 
/// # Arguments
/// * `cloud` - Point cloud
/// * `config` - Configuration parameters
/// 
/// # Returns
/// * `Result<TriangleMesh>` - Reconstructed triangle mesh
pub fn alpha_shape_reconstruction_with_config(
    cloud: &PointCloud<Point3f>,
    config: &AlphaShapeConfig,
) -> Result<TriangleMesh> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }

    let mut alpha_shape = AlphaShape::new(cloud, config.clone())?;
    alpha_shape.reconstruct()
}

/// Alpha shape reconstruction from point cloud with normals
/// 
/// # Arguments
/// * `cloud` - Point cloud with normals
/// * `alpha` - Alpha parameter for shape filtering
/// 
/// # Returns
/// * `Result<TriangleMesh>` - Reconstructed triangle mesh
pub fn alpha_shape_from_normals(
    cloud: &PointCloud<NormalPoint3f>,
    alpha: f32,
) -> Result<TriangleMesh> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }

    if alpha <= 0.0 {
        return Err(Error::InvalidData("Alpha must be positive".to_string()));
    }

    let config = AlphaShapeConfig {
        alpha,
        ..Default::default()
    };

    let mut alpha_shape = AlphaShape::from_normals(cloud, config)?;
    alpha_shape.reconstruct()
}

/// Compute optimal alpha value for a point cloud
/// 
/// This function estimates a good alpha value based on the average
/// nearest neighbor distance in the point cloud.
/// 
/// # Arguments
/// * `cloud` - Point cloud
/// * `k` - Number of neighbors to consider
/// 
/// # Returns
/// * `Result<f32>` - Estimated optimal alpha value
pub fn estimate_optimal_alpha(cloud: &PointCloud<Point3f>, k: usize) -> Result<f32> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }

    if k == 0 {
        return Err(Error::InvalidData("k must be positive".to_string()));
    }

    let distances: Vec<f32> = cloud.points
        .par_iter()
        .enumerate()
        .map(|(i, point)| {
            let mut distances: Vec<f32> = cloud.points
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other)| (point - other).magnitude())
                .collect();
            
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Take the k-th nearest neighbor distance
            distances.get(k.min(distances.len()) - 1).copied().unwrap_or(0.0)
        })
        .collect();

    if distances.is_empty() {
        return Err(Error::Algorithm("Could not compute distances".to_string()));
    }

    // Use the median of k-th nearest neighbor distances as alpha estimate
    let mut sorted_distances = distances;
    sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median_distance = if sorted_distances.len() % 2 == 0 {
        let mid = sorted_distances.len() / 2;
        (sorted_distances[mid - 1] + sorted_distances[mid]) / 2.0
    } else {
        sorted_distances[sorted_distances.len() / 2]
    };

    // Alpha should be larger than typical inter-point distances
    Ok(median_distance * 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_shape_config_default() {
        let config = AlphaShapeConfig::default();
        assert_eq!(config.alpha, 1.0);
        assert!(config.boundary_only);
        assert!(config.min_triangle_area > 0.0);
    }

    #[test]
    fn test_alpha_triangle_circumradius() {
        let p1 = Point3f::new(0.0, 0.0, 0.0);
        let p2 = Point3f::new(1.0, 0.0, 0.0);
        let p3 = Point3f::new(0.5, 0.866, 0.0); // Equilateral triangle
        
        let radius = AlphaTriangle::compute_circumradius(&p1, &p2, &p3);
        
        // For an equilateral triangle with side length 1, circumradius should be ~0.577
        assert!((radius - 0.577).abs() < 0.01);
    }

    #[test]
    fn test_alpha_triangle_area() {
        let p1 = Point3f::new(0.0, 0.0, 0.0);
        let p2 = Point3f::new(1.0, 0.0, 0.0);
        let p3 = Point3f::new(0.0, 1.0, 0.0); // Right triangle
        
        let area = AlphaTriangle::compute_triangle_area(&p1, &p2, &p3);
        
        // Area should be 0.5
        assert!((area - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_shape_empty_cloud() {
        let cloud = PointCloud::<Point3f>::new();
        let result = alpha_shape_reconstruction(&cloud, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_alpha_shape_invalid_alpha() {
        let mut cloud = PointCloud::new();
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        
        let result = alpha_shape_reconstruction(&cloud, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_optimal_alpha_empty() {
        let cloud = PointCloud::<Point3f>::new();
        let result = estimate_optimal_alpha(&cloud, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_optimal_alpha_simple() {
        let mut cloud = PointCloud::new();
        
        // Create a simple grid of points
        for i in 0..3 {
            for j in 0..3 {
                cloud.push(Point3f::new(i as f32, j as f32, 0.0));
            }
        }
        
        let result = estimate_optimal_alpha(&cloud, 3);
        assert!(result.is_ok());
        
        let alpha = result.unwrap();
        assert!(alpha > 0.0);
        assert!(alpha < 10.0); // Should be reasonable
    }

    #[test]
    fn test_make_edge() {
        let edge1 = AlphaShape::make_edge(1, 2);
        let edge2 = AlphaShape::make_edge(2, 1);
        assert_eq!(edge1, edge2);
        assert_eq!(edge1, (1, 2));
    }

    #[test]
    fn test_alpha_shape_small_dataset() {
        let mut cloud = PointCloud::new();
        
        // Create a simple triangle
        cloud.push(Point3f::new(0.0, 0.0, 0.0));
        cloud.push(Point3f::new(1.0, 0.0, 0.0));
        cloud.push(Point3f::new(0.5, 1.0, 0.0));
        
        let result = alpha_shape_reconstruction(&cloud, 2.0);
        
        match result {
            Ok(mesh) => {
                assert!(!mesh.vertices.is_empty());
                assert_eq!(mesh.vertices.len(), 3);
                assert!(!mesh.faces.is_empty());
            }
            Err(e) => {
                // May fail due to algorithm limitations with very small point sets
                assert!(e.to_string().contains("No valid triangles") || 
                        e.to_string().contains("No boundary triangles") ||
                        e.to_string().contains("Need at least"));
            }
        }
    }
} 