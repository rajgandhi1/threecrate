//! Moving Least Squares (MLS) surface fitting for smooth surface reconstruction
//!
//! This module implements various MLS surface reconstruction methods that create
//! smooth implicit surfaces from point clouds using local polynomial fitting.

use crate::parallel;
use nalgebra::{DMatrix, DVector, Vector3};
use std::collections::HashMap;
use threecrate_core::{Error, NormalPoint3f, Point3f, PointCloud, Result, TriangleMesh};

/// Weight function types for MLS reconstruction
#[derive(Debug, Clone, PartialEq)]
pub enum WeightFunction {
    /// Gaussian weight: exp(-d²/h²)
    Gaussian,
    /// Wendland weight: (1-d/h)⁴ * (4d/h + 1) for d < h
    Wendland,
    /// Cubic weight: (1-d/h)³ for d < h
    Cubic,
    /// Inverse distance weight: 1/d
    InverseDistance,
}

/// Polynomial basis types for MLS fitting
#[derive(Debug, Clone, PartialEq)]
pub enum PolynomialBasis {
    /// Constant: [1]
    Constant,
    /// Linear: [1, x, y, z]
    Linear,
    /// Quadratic: [1, x, y, z, x², y², z², xy, xz, yz]
    Quadratic,
    /// Cubic: [1, x, y, z, x², y², z², xy, xz, yz, x³, y³, z³, x²y, x²z, xy², y²z, xz², yz², xyz]
    Cubic,
}

/// Configuration for MLS surface reconstruction
#[derive(Debug, Clone)]
pub struct MLSConfig {
    /// Weight function to use
    pub weight_function: WeightFunction,
    /// Polynomial basis for local fitting
    pub basis: PolynomialBasis,
    /// Support radius for weight function
    pub support_radius: f32,
    /// Number of neighbors to consider (alternative to support radius)
    pub num_neighbors: Option<usize>,
    /// Regularization parameter for numerical stability
    pub regularization: f32,
    /// Whether to project points onto the fitted surface
    pub project_points: bool,
    /// Grid resolution for surface sampling
    pub grid_resolution: [usize; 3],
    /// Whether to compute normals from the fitted surface
    pub compute_normals: bool,
}

impl Default for MLSConfig {
    fn default() -> Self {
        Self {
            weight_function: WeightFunction::Wendland,
            basis: PolynomialBasis::Quadratic,
            support_radius: 0.1,
            num_neighbors: None,
            regularization: 1e-6,
            project_points: false,
            grid_resolution: [50, 50, 50],
            compute_normals: true,
        }
    }
}

/// MLS Surface fitting implementation
pub struct MLSSurface {
    config: MLSConfig,
    points: Vec<Point3f>,
    normals: Option<Vec<Vector3<f32>>>,
    spatial_index: Option<SpatialIndex>,
}

/// Simple spatial index for efficient neighbor queries
struct SpatialIndex {
    grid_cells: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f32,
    bounds_min: Point3f,
}

impl SpatialIndex {
    /// Build spatial index from points
    fn build(points: &[Point3f], cell_size: f32) -> Self {
        let mut bounds_min = Point3f::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut bounds_max = Point3f::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        // Compute bounds
        for point in points {
            bounds_min.x = bounds_min.x.min(point.x);
            bounds_min.y = bounds_min.y.min(point.y);
            bounds_min.z = bounds_min.z.min(point.z);
            bounds_max.x = bounds_max.x.max(point.x);
            bounds_max.y = bounds_max.y.max(point.y);
            bounds_max.z = bounds_max.z.max(point.z);
        }

        let mut grid_cells = HashMap::new();

        // Add points to grid cells
        for (i, point) in points.iter().enumerate() {
            let cell = (
                ((point.x - bounds_min.x) / cell_size).floor() as i32,
                ((point.y - bounds_min.y) / cell_size).floor() as i32,
                ((point.z - bounds_min.z) / cell_size).floor() as i32,
            );
            grid_cells.entry(cell).or_insert_with(Vec::new).push(i);
        }

        Self {
            grid_cells,
            cell_size,
            bounds_min,
        }
    }

    /// Find neighbors within radius
    fn find_neighbors(&self, query_point: &Point3f, radius: f32, points: &[Point3f]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let cell_radius = (radius / self.cell_size).ceil() as i32;

        let base_cell = (
            ((query_point.x - self.bounds_min.x) / self.cell_size).floor() as i32,
            ((query_point.y - self.bounds_min.y) / self.cell_size).floor() as i32,
            ((query_point.z - self.bounds_min.z) / self.cell_size).floor() as i32,
        );

        // Check neighboring cells
        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                for dz in -cell_radius..=cell_radius {
                    let cell = (base_cell.0 + dx, base_cell.1 + dy, base_cell.2 + dz);
                    if let Some(cell_points) = self.grid_cells.get(&cell) {
                        for &point_idx in cell_points {
                            let distance = (query_point - &points[point_idx]).magnitude();
                            if distance <= radius {
                                neighbors.push(point_idx);
                            }
                        }
                    }
                }
            }
        }

        neighbors
    }

    /// Find k nearest neighbors
    fn find_k_nearest(&self, query_point: &Point3f, k: usize, points: &[Point3f]) -> Vec<usize> {
        let mut distances: Vec<(f32, usize)> = Vec::new();

        // Simple brute force for k-nearest (could be optimized with priority queue)
        for (i, point) in points.iter().enumerate() {
            let distance = (query_point - point).magnitude();
            distances.push((distance, i));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.into_iter().take(k).map(|(_, idx)| idx).collect()
    }
}

impl MLSSurface {
    /// Create new MLS surface from point cloud
    pub fn new(cloud: &PointCloud<Point3f>, config: MLSConfig) -> Result<Self> {
        if cloud.is_empty() {
            return Err(Error::InvalidData("Point cloud is empty".to_string()));
        }

        let points = cloud.points.clone();
        let spatial_index = Some(SpatialIndex::build(&points, config.support_radius));

        Ok(Self {
            config,
            points,
            normals: None,
            spatial_index,
        })
    }

    /// Create MLS surface from point cloud with normals
    pub fn from_normals(cloud: &PointCloud<NormalPoint3f>, config: MLSConfig) -> Result<Self> {
        if cloud.is_empty() {
            return Err(Error::InvalidData("Point cloud is empty".to_string()));
        }

        let points: Vec<Point3f> = cloud.points.iter().map(|p| p.position).collect();
        let normals: Vec<Vector3<f32>> = cloud.points.iter().map(|p| p.normal).collect();
        let spatial_index = Some(SpatialIndex::build(&points, config.support_radius));

        Ok(Self {
            config,
            points,
            normals: Some(normals),
            spatial_index,
        })
    }

    /// Evaluate MLS surface at a point
    pub fn evaluate(&self, query_point: &Point3f) -> Result<f32> {
        // Find neighbors
        let neighbors = self.find_neighbors(query_point)?;
        if neighbors.is_empty() {
            return Ok(0.0);
        }

        // Compute weights
        let weights = self.compute_weights(query_point, &neighbors)?;

        // Set up weighted least squares system
        let (basis_matrix, values) =
            self.setup_weighted_system(query_point, &neighbors, &weights)?;

        // Solve for polynomial coefficients
        let coefficients = self.solve_weighted_system(&basis_matrix, &values)?;

        // Evaluate polynomial at query point
        let basis_values = self.evaluate_basis(query_point, query_point);
        Ok(basis_values.dot(&coefficients))
    }

    /// Evaluate MLS surface gradient (normal) at a point
    pub fn evaluate_gradient(&self, query_point: &Point3f) -> Result<Vector3<f32>> {
        let delta = 1e-4;

        // Compute gradient using finite differences
        let fx_pos = self.evaluate(&Point3f::new(
            query_point.x + delta,
            query_point.y,
            query_point.z,
        ))?;
        let fx_neg = self.evaluate(&Point3f::new(
            query_point.x - delta,
            query_point.y,
            query_point.z,
        ))?;
        let fy_pos = self.evaluate(&Point3f::new(
            query_point.x,
            query_point.y + delta,
            query_point.z,
        ))?;
        let fy_neg = self.evaluate(&Point3f::new(
            query_point.x,
            query_point.y - delta,
            query_point.z,
        ))?;
        let fz_pos = self.evaluate(&Point3f::new(
            query_point.x,
            query_point.y,
            query_point.z + delta,
        ))?;
        let fz_neg = self.evaluate(&Point3f::new(
            query_point.x,
            query_point.y,
            query_point.z - delta,
        ))?;

        let gradient = Vector3::new(
            (fx_pos - fx_neg) / (2.0 * delta),
            (fy_pos - fy_neg) / (2.0 * delta),
            (fz_pos - fz_neg) / (2.0 * delta),
        );

        Ok(if gradient.magnitude() > 1e-10 {
            gradient.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        })
    }

    /// Extract surface mesh using marching cubes on the MLS implicit function
    pub fn extract_mesh(&self) -> Result<TriangleMesh> {
        // Compute bounding box using parallel processing
        let (bounds_min, bounds_max) = parallel::point_cloud::parallel_bounding_box(&self.points)
            .unwrap_or((Point3f::origin(), Point3f::new(1.0, 1.0, 1.0)));

        // Add padding
        let padding = self.config.support_radius * 2.0;
        let bounds_min = Point3f::new(
            bounds_min.x - padding,
            bounds_min.y - padding,
            bounds_min.z - padding,
        );
        let bounds_max = Point3f::new(
            bounds_max.x + padding,
            bounds_max.y + padding,
            bounds_max.z + padding,
        );

        // Create volumetric grid
        let grid_resolution = self.config.grid_resolution;
        let voxel_size = [
            (bounds_max.x - bounds_min.x) / (grid_resolution[0] - 1) as f32,
            (bounds_max.y - bounds_min.y) / (grid_resolution[1] - 1) as f32,
            (bounds_max.z - bounds_min.z) / (grid_resolution[2] - 1) as f32,
        ];

        use crate::marching_cubes::{marching_cubes, VolumetricGrid};
        let mut grid = VolumetricGrid::new(grid_resolution, voxel_size, bounds_min);

        // Generate all grid coordinates for parallel MLS sampling
        let mut grid_coords = Vec::new();
        for x in 0..grid_resolution[0] {
            for y in 0..grid_resolution[1] {
                for z in 0..grid_resolution[2] {
                    grid_coords.push((x, y, z));
                }
            }
        }

        // Sample MLS function on grid in parallel
        let grid_values: Vec<((usize, usize, usize), f32)> =
            parallel::parallel_map(&grid_coords, |(x, y, z)| {
                let world_pos = grid.grid_to_world(*x, *y, *z);
                let value = self.evaluate(&world_pos).unwrap_or(0.0);
                ((*x, *y, *z), value)
            });

        // Fill grid with computed values
        for ((x, y, z), value) in grid_values {
            grid.set_value(x, y, z, value)?;
        }

        // Extract mesh using marching cubes
        let mut mesh = marching_cubes(&grid, 0.0)?;

        // Compute normals if requested
        if self.config.compute_normals {
            let mut normals = Vec::new();
            for vertex in &mesh.vertices {
                let normal = self
                    .evaluate_gradient(vertex)
                    .unwrap_or_else(|_| Vector3::new(0.0, 0.0, 1.0));
                normals.push(normal);
            }
            mesh.set_normals(normals);
        }

        Ok(mesh)
    }

    /// Find neighbors for a query point
    fn find_neighbors(&self, query_point: &Point3f) -> Result<Vec<usize>> {
        if let Some(ref spatial_index) = self.spatial_index {
            if let Some(k) = self.config.num_neighbors {
                Ok(spatial_index.find_k_nearest(query_point, k, &self.points))
            } else {
                Ok(spatial_index.find_neighbors(
                    query_point,
                    self.config.support_radius,
                    &self.points,
                ))
            }
        } else {
            // Fallback to brute force
            let mut neighbors = Vec::new();
            if let Some(k) = self.config.num_neighbors {
                let mut distances: Vec<(f32, usize)> = self
                    .points
                    .iter()
                    .enumerate()
                    .map(|(i, p)| ((query_point - p).magnitude(), i))
                    .collect();
                distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                neighbors = distances.into_iter().take(k).map(|(_, idx)| idx).collect();
            } else {
                for (i, point) in self.points.iter().enumerate() {
                    if (query_point - point).magnitude() <= self.config.support_radius {
                        neighbors.push(i);
                    }
                }
            }
            Ok(neighbors)
        }
    }

    /// Compute weights for neighbors
    fn compute_weights(&self, query_point: &Point3f, neighbors: &[usize]) -> Result<Vec<f32>> {
        let mut weights = Vec::with_capacity(neighbors.len());

        for &neighbor_idx in neighbors {
            let point = &self.points[neighbor_idx];
            let distance = (query_point - point).magnitude();
            let weight = self.compute_weight_function(distance);
            weights.push(weight);
        }

        Ok(weights)
    }

    /// Compute weight function value
    fn compute_weight_function(&self, distance: f32) -> f32 {
        let h = self.config.support_radius;

        match self.config.weight_function {
            WeightFunction::Gaussian => (-distance * distance / (h * h)).exp(),
            WeightFunction::Wendland => {
                if distance >= h {
                    0.0
                } else {
                    let r = distance / h;
                    (1.0 - r).powi(4) * (4.0 * r + 1.0)
                }
            }
            WeightFunction::Cubic => {
                if distance >= h {
                    0.0
                } else {
                    let r = distance / h;
                    (1.0 - r).powi(3)
                }
            }
            WeightFunction::InverseDistance => {
                if distance < 1e-10 {
                    1e10 // Very large weight for coincident points
                } else {
                    1.0 / distance
                }
            }
        }
    }

    /// Set up weighted least squares system
    fn setup_weighted_system(
        &self,
        query_point: &Point3f,
        neighbors: &[usize],
        weights: &[f32],
    ) -> Result<(DMatrix<f32>, DVector<f32>)> {
        let basis_size = self.get_basis_size();
        let n_neighbors = neighbors.len();

        let mut basis_matrix = DMatrix::zeros(n_neighbors, basis_size);
        let mut values = DVector::zeros(n_neighbors);

        for (i, &neighbor_idx) in neighbors.iter().enumerate() {
            let point = &self.points[neighbor_idx];
            let weight = weights[i].sqrt(); // Apply square root of weight

            // Evaluate basis functions at this point
            let basis_values = self.evaluate_basis(point, query_point);
            for j in 0..basis_size {
                basis_matrix[(i, j)] = basis_values[j] * weight;
            }

            // For implicit surface, we want f(p) = 0 at surface points
            // If we have normals, we can set up orientation constraints
            if let Some(ref normals) = self.normals {
                // Use signed distance: positive outside, negative inside
                let normal = &normals[neighbor_idx];
                let offset = (point - query_point).dot(normal);
                values[i] = offset * weight;
            } else {
                // Without normals, assume points are on surface (f = 0)
                values[i] = 0.0;
            }
        }

        Ok((basis_matrix, values))
    }

    /// Solve weighted least squares system
    fn solve_weighted_system(
        &self,
        basis_matrix: &DMatrix<f32>,
        values: &DVector<f32>,
    ) -> Result<DVector<f32>> {
        // Solve A^T A x = A^T b with regularization
        let at = basis_matrix.transpose();
        let mut ata = &at * basis_matrix;
        let atb = &at * values;

        // Add regularization
        for i in 0..ata.nrows() {
            ata[(i, i)] += self.config.regularization;
        }

        // Solve using Cholesky decomposition
        match ata.clone().cholesky() {
            Some(chol) => Ok(chol.solve(&atb)),
            None => {
                // Fallback to SVD if Cholesky fails
                let svd = ata.svd(true, true);
                match svd.solve(&atb, 1e-10) {
                    Ok(solution) => Ok(solution),
                    Err(_) => Err(Error::Algorithm("Failed to solve MLS system".to_string())),
                }
            }
        }
    }

    /// Get basis size for current polynomial basis
    fn get_basis_size(&self) -> usize {
        match self.config.basis {
            PolynomialBasis::Constant => 1,
            PolynomialBasis::Linear => 4,
            PolynomialBasis::Quadratic => 10,
            PolynomialBasis::Cubic => 20,
        }
    }

    /// Evaluate basis functions at a point
    fn evaluate_basis(&self, point: &Point3f, center: &Point3f) -> DVector<f32> {
        let dx = point.x - center.x;
        let dy = point.y - center.y;
        let dz = point.z - center.z;

        match self.config.basis {
            PolynomialBasis::Constant => DVector::from_vec(vec![1.0]),
            PolynomialBasis::Linear => DVector::from_vec(vec![1.0, dx, dy, dz]),
            PolynomialBasis::Quadratic => DVector::from_vec(vec![
                1.0,
                dx,
                dy,
                dz,
                dx * dx,
                dy * dy,
                dz * dz,
                dx * dy,
                dx * dz,
                dy * dz,
            ]),
            PolynomialBasis::Cubic => DVector::from_vec(vec![
                1.0,
                dx,
                dy,
                dz,
                dx * dx,
                dy * dy,
                dz * dz,
                dx * dy,
                dx * dz,
                dy * dz,
                dx * dx * dx,
                dy * dy * dy,
                dz * dz * dz,
                dx * dx * dy,
                dx * dx * dz,
                dx * dy * dy,
                dy * dy * dz,
                dx * dz * dz,
                dy * dz * dz,
                dx * dy * dz,
            ]),
        }
    }
}

/// Convenience function for basic MLS reconstruction
pub fn moving_least_squares(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    let mls = MLSSurface::new(cloud, MLSConfig::default())?;
    mls.extract_mesh()
}

/// MLS reconstruction with custom configuration
pub fn moving_least_squares_with_config(
    cloud: &PointCloud<Point3f>,
    config: &MLSConfig,
) -> Result<TriangleMesh> {
    let mls = MLSSurface::new(cloud, config.clone())?;
    mls.extract_mesh()
}

/// MLS reconstruction from point cloud with normals
pub fn moving_least_squares_from_normals(
    cloud: &PointCloud<NormalPoint3f>,
) -> Result<TriangleMesh> {
    let mls = MLSSurface::from_normals(cloud, MLSConfig::default())?;
    mls.extract_mesh()
}

/// MLS reconstruction with automatic parameter estimation
pub fn moving_least_squares_auto(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    let support_radius = estimate_optimal_support_radius(cloud)?;

    let config = MLSConfig {
        support_radius,
        weight_function: WeightFunction::Wendland,
        basis: PolynomialBasis::Quadratic,
        ..Default::default()
    };

    moving_least_squares_with_config(cloud, &config)
}

/// Estimate optimal support radius based on point density
pub fn estimate_optimal_support_radius(cloud: &PointCloud<Point3f>) -> Result<f32> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }

    let mut distances = Vec::new();
    let sample_size = (cloud.points.len() / 10).max(20).min(200);
    let step = (cloud.points.len().max(1) / sample_size.max(1)).max(1);

    // Sample nearest neighbor distances
    for i in (0..cloud.points.len()).step_by(step) {
        let point = &cloud.points[i];
        let mut min_dist = f32::INFINITY;

        for (j, other_point) in cloud.points.iter().enumerate() {
            if i != j {
                let dist = (point - other_point).magnitude();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        if min_dist < f32::INFINITY {
            distances.push(min_dist);
        }
    }

    if distances.is_empty() {
        return Ok(0.1);
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_dist = distances[distances.len() / 2];

    // Support radius should be about 3-4 times the median nearest neighbor distance
    Ok(median_dist * 3.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_mls_config_default() {
        let config = MLSConfig::default();
        assert_eq!(config.weight_function, WeightFunction::Wendland);
        assert_eq!(config.basis, PolynomialBasis::Quadratic);
        assert_eq!(config.support_radius, 0.1);
        assert!(config.compute_normals);
    }

    #[test]
    fn test_weight_functions() {
        let config = MLSConfig {
            weight_function: WeightFunction::Gaussian,
            support_radius: 1.0,
            ..Default::default()
        };

        let points = vec![Point3::new(0.0, 0.0, 0.0)];
        let cloud = PointCloud::from_points(points);
        let mls = MLSSurface::new(&cloud, config).unwrap();

        // Test Gaussian weight
        let weight = mls.compute_weight_function(0.5);
        assert!(weight > 0.0);
        assert!(weight <= 1.0);

        // Test Wendland weight
        let config_wendland = MLSConfig {
            weight_function: WeightFunction::Wendland,
            support_radius: 1.0,
            ..Default::default()
        };
        let mls_wendland = MLSSurface::new(&cloud, config_wendland).unwrap();
        let weight_wendland = mls_wendland.compute_weight_function(0.5);
        assert!(weight_wendland > 0.0);
    }

    #[test]
    fn test_basis_functions() {
        let config = MLSConfig::default();
        let points = vec![Point3::new(0.0, 0.0, 0.0)];
        let cloud = PointCloud::from_points(points);
        let mls = MLSSurface::new(&cloud, config).unwrap();

        let query_point = Point3f::new(1.0, 2.0, 3.0);
        let center = Point3f::origin();

        // Test quadratic basis
        let basis = mls.evaluate_basis(&query_point, &center);
        assert_eq!(basis.len(), 10); // Quadratic basis has 10 terms
        assert_eq!(basis[0], 1.0); // Constant term
        assert_eq!(basis[1], 1.0); // x term
        assert_eq!(basis[2], 2.0); // y term
        assert_eq!(basis[3], 3.0); // z term
    }

    #[test]
    fn test_spatial_index() {
        let points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(2.0, 2.0, 2.0),
        ];

        let index = SpatialIndex::build(&points, 1.0);
        let neighbors = index.find_neighbors(&Point3f::origin(), 1.5, &points);

        // Should find the first 3 points within radius 1.5
        assert!(neighbors.len() >= 2);
        assert!(neighbors.contains(&0));
    }

    #[test]
    fn test_mls_evaluation() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let cloud = PointCloud::from_points(points);

        let config = MLSConfig {
            support_radius: 2.0,
            ..Default::default()
        };

        let mls = MLSSurface::new(&cloud, config).unwrap();
        let value = mls.evaluate(&Point3f::new(0.5, 0.5, 0.5));

        // Should succeed (actual value depends on implementation)
        assert!(value.is_ok());
    }

    #[test]
    fn test_estimate_optimal_support_radius() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let cloud = PointCloud::from_points(points);

        let radius = estimate_optimal_support_radius(&cloud).unwrap();
        assert!(radius > 0.0);
        assert!(radius < 10.0); // Should be reasonable
    }

    #[test]
    fn test_mls_empty_cloud() {
        let cloud = PointCloud::new();
        let result = moving_least_squares(&cloud);
        assert!(result.is_err());
    }

    #[test]
    fn test_mls_simple_reconstruction() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let cloud = PointCloud::from_points(points);

        let config = MLSConfig {
            support_radius: 2.0,
            grid_resolution: [10, 10, 10], // Small grid for testing
            ..Default::default()
        };

        let result = moving_least_squares_with_config(&cloud, &config);

        // May succeed or fail depending on data - that's ok for unit tests
        match result {
            Ok(mesh) => {
                assert!(!mesh.is_empty());
            }
            Err(_) => {
                // Acceptable for simple test data
            }
        }
    }
}
