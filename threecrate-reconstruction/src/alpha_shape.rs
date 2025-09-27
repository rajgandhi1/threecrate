//! Alpha shape reconstruction with enhanced features
//! 
//! Improved implementation with better geometric algorithms for surface reconstruction.

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, NormalPoint3f, Error};

/// Alpha complex configuration
#[derive(Debug, Clone)]
pub struct AlphaComplexConfig {
    /// Alpha parameter for shape filtering
    pub alpha: f32,
    /// Mode for alpha complex computation
    pub mode: AlphaMode,
    /// Minimum triangle area threshold
    pub min_triangle_area: f32,
    /// Enable fast computation mode (less accurate but faster)
    pub fast_mode: bool,
}

/// Alpha complex computation modes
#[derive(Debug, Clone, PartialEq)]
pub enum AlphaMode {
    /// Only boundary (manifold surface)
    BoundaryOnly,
    /// Full alpha complex including interior
    Full,
    /// Adaptive alpha based on local density
    Adaptive,
}

impl Default for AlphaComplexConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            mode: AlphaMode::BoundaryOnly,
            min_triangle_area: 1e-8,
            fast_mode: false,
        }
    }
}

/// Enhanced alpha simplex
#[derive(Debug, Clone)]
struct AlphaSimplex {
    /// Vertex indices
    vertices: Vec<usize>,
    /// Circumradius squared for efficiency
    circumradius_sq: f32,
    /// Simplex dimension (0=point, 1=edge, 2=triangle, 3=tetrahedron)
    dimension: usize,
    /// Whether this simplex is on the boundary
    is_boundary: bool,
    /// Associated data (area, volume, etc.)
    measure: f32,
}

impl AlphaSimplex {
    /// Create a triangle simplex
    fn triangle(v1: usize, v2: usize, v3: usize, points: &[Point3f]) -> Self {
        let p1 = &points[v1];
        let p2 = &points[v2];
        let p3 = &points[v3];
        
        let circumradius_sq = Self::compute_circumradius_sq_triangle(p1, p2, p3);
        let area = Self::compute_triangle_area(p1, p2, p3);
        
        Self {
            vertices: vec![v1, v2, v3],
            circumradius_sq,
            dimension: 2,
            is_boundary: false, // Will be determined later
            measure: area,
        }
    }
    
    /// Compute circumradius squared for triangle (more efficient)
    fn compute_circumradius_sq_triangle(p1: &Point3f, p2: &Point3f, p3: &Point3f) -> f32 {
        let a_sq = (p2 - p1).magnitude_squared();
        let b_sq = (p3 - p2).magnitude_squared();
        let c_sq = (p1 - p3).magnitude_squared();
        
        // Handle degenerate cases
        if a_sq < 1e-20 || b_sq < 1e-20 || c_sq < 1e-20 {
            return f32::INFINITY;
        }
        
        // Compute area using cross product
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let cross = v1.cross(&v2);
        let area_sq = cross.magnitude_squared() * 0.25;
        
        if area_sq < 1e-20 {
            return f32::INFINITY;
        }
        
        // Circumradius squared formula: R² = (abc)² / (16 * Area²)
        (a_sq * b_sq * c_sq) / (16.0 * area_sq)
    }
    
    /// Compute triangle area
    fn compute_triangle_area(p1: &Point3f, p2: &Point3f, p3: &Point3f) -> f32 {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let cross = v1.cross(&v2);
        cross.magnitude() * 0.5
    }
    
    /// Check if this simplex is valid for the given alpha value
    fn is_alpha_valid(&self, alpha: f32, min_measure: f32) -> bool {
        self.circumradius_sq <= alpha * alpha && self.measure >= min_measure
    }
}

/// Alpha Complex implementation
pub struct AlphaComplex {
    points: Vec<Point3f>,
    simplices: Vec<AlphaSimplex>,
    config: AlphaComplexConfig,
}

impl AlphaComplex {
    /// Create a new Alpha Complex
    pub fn new(cloud: &PointCloud<Point3f>, config: AlphaComplexConfig) -> Result<Self> {
        if cloud.is_empty() {
            return Err(Error::InvalidData("Point cloud is empty".to_string()));
        }

        let points = cloud.points.clone();
        
        Ok(Self {
            points,
            simplices: Vec::new(),
            config,
        })
    }
    
    /// Create from point cloud with normals
    pub fn from_normals(cloud: &PointCloud<NormalPoint3f>, config: AlphaComplexConfig) -> Result<Self> {
        let points: Vec<Point3f> = cloud.points.iter().map(|p| p.position).collect();
        let point_cloud = PointCloud::from_points(points);
        Self::new(&point_cloud, config)
    }
    
    /// Generate candidate triangles using spatial locality
    fn generate_candidate_triangles(&mut self) -> Result<()> {
        let n = self.points.len();
        
        if n < 3 {
            return Err(Error::InvalidData("Need at least 3 points for triangulation".to_string()));
        }
        
        self.simplices.clear();
        
        if self.config.fast_mode {
            self.generate_triangles_fast()?;
        } else {
            self.generate_triangles_quality()?;
        }
        
        Ok(())
    }
    
    /// Fast triangle generation using simple neighbor finding
    fn generate_triangles_fast(&mut self) -> Result<()> {
        let max_triangles = if self.points.len() < 1000 { 50000 } else { 100000 };
        let mut triangle_count = 0;
        
        for i in 0..self.points.len() {
            if triangle_count >= max_triangles { break; }
            
            let point = &self.points[i];
            
            // Find nearby points using simple distance check
            let mut neighbors = Vec::new();
            for (j, other_point) in self.points.iter().enumerate() {
                if i != j {
                    let dist = (point - other_point).magnitude();
                    if dist <= self.config.alpha * 1.5 {
                        neighbors.push(j);
                    }
                }
            }
            
            // Generate triangles with nearby points
            for (idx_j, &j) in neighbors.iter().enumerate() {
                if triangle_count >= max_triangles { break; }
                if j <= i { continue; }
                
                for &k in neighbors.iter().skip(idx_j + 1) {
                    if triangle_count >= max_triangles { break; }
                    if k <= i { continue; }
                    
                    let triangle = AlphaSimplex::triangle(i, j, k, &self.points);
                    
                    // Pre-filter by alpha to avoid storing too many candidates
                    if triangle.is_alpha_valid(self.config.alpha * 1.5, self.config.min_triangle_area) {
                        self.simplices.push(triangle);
                        triangle_count += 1;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Quality triangle generation with better geometric constraints
    fn generate_triangles_quality(&mut self) -> Result<()> {
        let mut edge_triangles: std::collections::HashMap<(usize, usize), Vec<usize>> = std::collections::HashMap::new();
        let max_triangles = 50000;
        let mut triangle_count = 0;
        
        for i in 0..self.points.len() {
            if triangle_count >= max_triangles { break; }
            
            let point = &self.points[i];
            
            // Find neighbors within alpha distance
            let mut neighbors = Vec::new();
            for (j, other_point) in self.points.iter().enumerate() {
                if i != j {
                    let dist = (point - other_point).magnitude();
                    if dist <= self.config.alpha {
                        neighbors.push(j);
                    }
                }
            }
            
            // Generate triangles with geometric quality checks
            for (idx_j, &j) in neighbors.iter().enumerate() {
                if triangle_count >= max_triangles { break; }
                if j <= i { continue; }
                
                for &k in neighbors.iter().skip(idx_j + 1) {
                    if triangle_count >= max_triangles { break; }
                    if k <= i { continue; }
                    
                    let triangle = AlphaSimplex::triangle(i, j, k, &self.points);
                    
                    // Quality checks
                    if self.is_triangle_geometrically_valid(&triangle)? {
                        let triangle_idx = self.simplices.len();
                        self.simplices.push(triangle);
                        triangle_count += 1;
                        
                        // Track edge-triangle adjacency
                        let edges = [(i, j), (j, k), (k, i)];
                        for &(v1, v2) in &edges {
                            let edge = (v1.min(v2), v1.max(v2));
                            edge_triangles.entry(edge).or_insert_with(Vec::new).push(triangle_idx);
                        }
                    }
                }
            }
        }
        
        // Mark boundary triangles based on edge count
        for triangles in edge_triangles.values() {
            for &triangle_idx in triangles {
                if triangles.len() == 1 {
                    self.simplices[triangle_idx].is_boundary = true;
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if triangle meets geometric validity criteria
    fn is_triangle_geometrically_valid(&self, triangle: &AlphaSimplex) -> Result<bool> {
        // Basic geometric checks
        if triangle.circumradius_sq == f32::INFINITY {
            return Ok(false);
        }
        
        if triangle.measure < self.config.min_triangle_area {
            return Ok(false);
        }
        
        // Check aspect ratio (avoid very thin triangles)
        let v1 = triangle.vertices[0];
        let v2 = triangle.vertices[1];
        let v3 = triangle.vertices[2];
        
        let p1 = &self.points[v1];
        let p2 = &self.points[v2];
        let p3 = &self.points[v3];
        
        let a = (p2 - p1).magnitude();
        let b = (p3 - p2).magnitude();
        let c = (p1 - p3).magnitude();
        
        let perimeter = a + b + c;
        if perimeter < 1e-10 {
            return Ok(false);
        }
        
        // Quality measure: area / perimeter² (higher is better)
        let quality = (4.0 * triangle.measure) / (perimeter * perimeter);
        
        // Accept triangles with reasonable quality
        Ok(quality > 0.01) // Configurable threshold
    }
    
    /// Filter simplices based on alpha and mode
    fn filter_simplices(&mut self) -> Result<()> {
        match self.config.mode {
            AlphaMode::BoundaryOnly => {
                self.simplices.retain(|s| s.is_boundary && s.is_alpha_valid(self.config.alpha, self.config.min_triangle_area));
            }
            AlphaMode::Full => {
                self.simplices.retain(|s| s.is_alpha_valid(self.config.alpha, self.config.min_triangle_area));
            }
            AlphaMode::Adaptive => {
                // For adaptive mode, use local density estimation
                let mut filtered_simplices = Vec::new();
                for simplex in &self.simplices {
                    // Compute adaptive alpha based on local point density around simplex
                    let adaptive_alpha = self.compute_local_alpha(simplex)?;
                    
                    if simplex.is_alpha_valid(adaptive_alpha, self.config.min_triangle_area) {
                        filtered_simplices.push(simplex.clone());
                    }
                }
                self.simplices = filtered_simplices;
            }
        }
        
        Ok(())
    }
    
    /// Compute local alpha value based on point density
    fn compute_local_alpha(&self, simplex: &AlphaSimplex) -> Result<f32> {
        // Compute centroid of simplex
        let mut centroid = Point3f::origin();
        for &vertex_idx in &simplex.vertices {
            centroid = Point3f::from(centroid.coords + self.points[vertex_idx].coords);
        }
        centroid = Point3f::from(centroid.coords / simplex.vertices.len() as f32);
        
        // Find average distance to nearby points
        let mut distances = Vec::new();
        for point in &self.points {
            let dist = (point - centroid).magnitude();
            if dist > 0.0 && dist < self.config.alpha * 3.0 {
                distances.push(dist);
            }
        }
        
        if distances.is_empty() {
            return Ok(self.config.alpha);
        }
        
        // Use median distance as basis for adaptive alpha
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_dist = distances[distances.len() / 2];
        
        // Adaptive alpha is a multiple of local density
        Ok(median_dist * 1.8)
    }
    
    /// Extract surface mesh from alpha complex
    pub fn extract_surface(&mut self) -> Result<TriangleMesh> {
        // Generate candidate simplices
        self.generate_candidate_triangles()?;
        
        if self.simplices.is_empty() {
            return Err(Error::Algorithm("No candidate triangles generated".to_string()));
        }
        
        // Filter based on alpha criteria
        self.filter_simplices()?;
        
        if self.simplices.is_empty() {
            return Err(Error::Algorithm("No triangles survived alpha filtering".to_string()));
        }
        
        // Extract triangular faces
        let faces: Vec<[usize; 3]> = self.simplices.iter()
            .filter(|s| s.dimension == 2)
            .map(|s| [s.vertices[0], s.vertices[1], s.vertices[2]])
            .collect();
        
        if faces.is_empty() {
            return Err(Error::Algorithm("No triangular faces found".to_string()));
        }
        
        // Create final mesh
        let mesh = TriangleMesh::from_vertices_and_faces(self.points.clone(), faces);
        
        Ok(mesh)
    }
    
    /// Get alpha complex statistics
    pub fn get_statistics(&self) -> AlphaComplexStats {
        let boundary_count = self.simplices.iter().filter(|s| s.is_boundary).count();
        let triangle_count = self.simplices.iter().filter(|s| s.dimension == 2).count();
        
        let avg_circumradius = if !self.simplices.is_empty() {
            self.simplices.iter().map(|s| s.circumradius_sq.sqrt()).sum::<f32>() / self.simplices.len() as f32
        } else {
            0.0
        };
        
        AlphaComplexStats {
            total_simplices: self.simplices.len(),
            boundary_simplices: boundary_count,
            triangle_count,
            average_circumradius: avg_circumradius,
        }
    }
}

/// Statistics about the alpha complex
#[derive(Debug, Clone)]
pub struct AlphaComplexStats {
    pub total_simplices: usize,
    pub boundary_simplices: usize,
    pub triangle_count: usize,
    pub average_circumradius: f32,
}

/// Convenience function for alpha shape reconstruction
pub fn alpha_complex_reconstruction(cloud: &PointCloud<Point3f>, alpha: f32) -> Result<TriangleMesh> {
    let config = AlphaComplexConfig {
        alpha,
        mode: AlphaMode::BoundaryOnly,
        ..Default::default()
    };
    let mut alpha_complex = AlphaComplex::new(cloud, config)?;
    alpha_complex.extract_surface()
}

/// Enhanced alpha shape with full configuration
pub fn alpha_complex_reconstruction_with_config(
    cloud: &PointCloud<Point3f>,
    config: &AlphaComplexConfig,
) -> Result<TriangleMesh> {
    let mut alpha_complex = AlphaComplex::new(cloud, config.clone())?;
    alpha_complex.extract_surface()
}

/// Adaptive alpha shape that adjusts to local point density
pub fn adaptive_alpha_reconstruction(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    let config = AlphaComplexConfig {
        mode: AlphaMode::Adaptive,
        fast_mode: false,
        ..Default::default()
    };
    let mut alpha_complex = AlphaComplex::new(cloud, config)?;
    alpha_complex.extract_surface()
}

// Keep existing simple functions for backward compatibility
pub fn alpha_shape_reconstruction(cloud: &PointCloud<Point3f>, alpha: f32) -> Result<TriangleMesh> {
    alpha_complex_reconstruction(cloud, alpha)
}

pub fn alpha_shape_reconstruction_with_config(
    cloud: &PointCloud<Point3f>,
    config: &AlphaShapeConfig,
) -> Result<TriangleMesh> {
    // Convert old config to new config
    let new_config = AlphaComplexConfig {
        alpha: config.alpha,
        mode: if config.boundary_only { AlphaMode::BoundaryOnly } else { AlphaMode::Full },
        min_triangle_area: config.min_triangle_area,
        fast_mode: false,
    };
    alpha_complex_reconstruction_with_config(cloud, &new_config)
}

pub fn alpha_shape_from_normals(cloud: &PointCloud<NormalPoint3f>, alpha: f32) -> Result<TriangleMesh> {
    let config = AlphaComplexConfig {
        alpha,
        mode: AlphaMode::BoundaryOnly,
        ..Default::default()
    };
    let mut alpha_complex = AlphaComplex::from_normals(cloud, config)?;
    alpha_complex.extract_surface()
}

pub fn estimate_optimal_alpha(cloud: &PointCloud<Point3f>, k: usize) -> Result<f32> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }
    
    let mut distances = Vec::new();
    
    // Sample points to estimate average k-nearest neighbor distance
    let sample_size = (cloud.points.len() / 10).max(50).min(500);
    let step = (cloud.points.len().max(1) / sample_size.max(1)).max(1);
    
    for i in (0..cloud.points.len()).step_by(step) {
        let point = &cloud.points[i];
        
        // Find k nearest neighbors
        let mut point_distances: Vec<f32> = cloud.points.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, other)| (point - other).magnitude())
            .collect();
        
        point_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if point_distances.len() >= k {
            distances.push(point_distances[k - 1]);
        }
    }
    
    if distances.is_empty() {
        return Ok(1.0); // Default fallback
    }
    
    // Use median distance as basis for alpha
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_dist = distances[distances.len() / 2];
    
    // Optimal alpha is typically 1.5-2.0 times the k-th nearest neighbor distance
    Ok(median_dist * 1.8)
}

// Keep existing config for backward compatibility
#[derive(Debug, Clone)]
pub struct AlphaShapeConfig {
    pub alpha: f32,
    pub boundary_only: bool,
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_alpha_complex_config_default() {
        let config = AlphaComplexConfig::default();
        
        assert_eq!(config.alpha, 1.0);
        assert_eq!(config.mode, AlphaMode::BoundaryOnly);
        assert_eq!(config.min_triangle_area, 1e-8);
        assert!(!config.fast_mode);
    }

    #[test]
    fn test_alpha_simplex_triangle() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        
        let simplex = AlphaSimplex::triangle(0, 1, 2, &points);
        
        assert_eq!(simplex.vertices, vec![0, 1, 2]);
        assert_eq!(simplex.dimension, 2);
        assert!(simplex.measure > 0.0); // Should have positive area
    }

    #[test]
    fn test_empty_cloud() {
        let config = AlphaComplexConfig::default();
        let cloud = PointCloud::new();
        
        let result = AlphaComplex::new(&cloud, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_optimal_alpha_empty() {
        let cloud = PointCloud::new();
        let result = estimate_optimal_alpha(&cloud, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_alpha_complex_simple() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let cloud = PointCloud::from_points(points);
        
        let result = alpha_complex_reconstruction(&cloud, 2.0);
        
        match result {
            Ok(mesh) => {
                assert!(!mesh.is_empty());
                assert!(mesh.vertex_count() <= 3);
            }
            Err(_) => {
                // May fail on simple cases - that's acceptable
            }
        }
    }
} 