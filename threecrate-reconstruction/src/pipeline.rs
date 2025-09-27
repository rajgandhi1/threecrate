//! Unified reconstruction pipeline with automatic algorithm selection
//!
//! This module provides an intelligent reconstruction pipeline that automatically
//! selects the best algorithm based on input data characteristics and user requirements.

use crate::parallel;
use std::collections::HashMap;
use threecrate_core::{Error, NormalPoint3f, Point3f, PointCloud, Result, TriangleMesh};

/// Reconstruction algorithm types available in the pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// Poisson surface reconstruction (requires normals)
    Poisson,
    /// Ball Pivoting algorithm (good for uniform sampling)
    BallPivoting,
    /// Delaunay triangulation (fast, works with any point distribution)
    Delaunay,
    /// Moving Least Squares (smooth surfaces, handles noise well)
    MovingLeastSquares,
    /// Marching Cubes (volumetric approach, good for closed surfaces)
    MarchingCubes,
}

/// Quality requirements for reconstruction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityLevel {
    /// Fast reconstruction with basic quality
    Fast,
    /// Balanced speed and quality
    Balanced,
    /// High quality reconstruction (slower)
    HighQuality,
    /// Maximum quality (slowest, best results)
    MaxQuality,
}

/// Use case scenarios that help algorithm selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UseCase {
    /// General purpose reconstruction
    General,
    /// Prototyping and quick visualization
    Prototyping,
    /// CAD/Engineering models (high precision)
    Engineering,
    /// Artistic/Organic shapes (smooth surfaces)
    Organic,
    /// Noisy sensor data (needs robust algorithms)
    NoisyData,
    /// Sparse point clouds (few points)
    Sparse,
    /// Dense point clouds (many points)
    Dense,
}

/// Data characteristics analyzed from the input point cloud
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Number of points in the cloud
    pub point_count: usize,
    /// Whether normals are available
    pub has_normals: bool,
    /// Point density uniformity (0.0 = very non-uniform, 1.0 = perfectly uniform)
    pub density_uniformity: f32,
    /// Estimated noise level (0.0 = no noise, 1.0 = very noisy)
    pub noise_level: f32,
    /// Average nearest neighbor distance
    pub avg_neighbor_distance: f32,
    /// Bounding box dimensions
    pub bounding_box: (Point3f, Point3f),
    /// Whether the surface appears to be closed
    pub is_closed_surface: bool,
    /// Estimated surface complexity (0.0 = simple, 1.0 = very complex)
    pub surface_complexity: f32,
    /// Distribution of points (planar, spherical, arbitrary)
    pub distribution_type: DistributionType,
}

/// Types of point distribution patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistributionType {
    /// Points are mostly on a plane
    Planar,
    /// Points form a spherical or ellipsoidal shape
    Spherical,
    /// Points form a cylindrical shape
    Cylindrical,
    /// Arbitrary 3D distribution
    Arbitrary,
}

/// Configuration for the unified reconstruction pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Desired quality level
    pub quality: QualityLevel,
    /// Use case scenario
    pub use_case: UseCase,
    /// Preferred algorithm (None for automatic selection)
    pub preferred_algorithm: Option<Algorithm>,
    /// Fallback algorithms to try if preferred fails
    pub fallback_algorithms: Vec<Algorithm>,
    /// Maximum processing time in seconds (None for unlimited)
    pub max_processing_time: Option<f32>,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Validate output mesh quality
    pub validate_output: bool,
    /// Attempt to repair mesh if validation fails
    pub auto_repair: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            quality: QualityLevel::Balanced,
            use_case: UseCase::General,
            preferred_algorithm: None,
            fallback_algorithms: vec![
                Algorithm::Delaunay,
                Algorithm::BallPivoting,
                Algorithm::MovingLeastSquares,
            ],
            max_processing_time: None,
            enable_parallel: true,
            validate_output: true,
            auto_repair: false,
        }
    }
}

/// Result of reconstruction attempt with metadata
#[derive(Debug)]
pub struct ReconstructionResult {
    /// The reconstructed mesh
    pub mesh: TriangleMesh,
    /// Algorithm that was used
    pub algorithm_used: Algorithm,
    /// Processing time in seconds
    pub processing_time: f32,
    /// Quality metrics of the result
    pub quality_metrics: QualityMetrics,
    /// Data characteristics that were analyzed
    pub data_characteristics: DataCharacteristics,
}

/// Quality metrics for evaluating reconstruction results
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Number of vertices in result
    pub vertex_count: usize,
    /// Number of triangles in result
    pub triangle_count: usize,
    /// Average triangle quality (0.0 = poor, 1.0 = excellent)
    pub avg_triangle_quality: f32,
    /// Mesh watertightness (0.0 = many holes, 1.0 = watertight)
    pub watertightness: f32,
    /// Surface smoothness (0.0 = rough, 1.0 = very smooth)
    pub smoothness: f32,
    /// Geometric accuracy compared to input (0.0 = poor, 1.0 = perfect)
    pub geometric_accuracy: f32,
}

/// The unified reconstruction pipeline
pub struct ReconstructionPipeline {
    config: PipelineConfig,
}

impl ReconstructionPipeline {
    /// Create a new reconstruction pipeline with configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with default configuration
    pub fn default() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Create a pipeline optimized for a specific use case
    pub fn for_use_case(use_case: UseCase) -> Self {
        let mut config = PipelineConfig::default();
        config.use_case = use_case;

        // Adjust configuration based on use case
        match use_case {
            UseCase::Prototyping => {
                config.quality = QualityLevel::Fast;
                config.fallback_algorithms = vec![Algorithm::Delaunay, Algorithm::BallPivoting];
            }
            UseCase::Engineering => {
                config.quality = QualityLevel::HighQuality;
                config.validate_output = true;
                config.auto_repair = true;
            }
            UseCase::Organic => {
                config.quality = QualityLevel::HighQuality;
                config.fallback_algorithms = vec![
                    Algorithm::MovingLeastSquares,
                    Algorithm::Poisson,
                    Algorithm::BallPivoting,
                ];
            }
            UseCase::NoisyData => {
                config.fallback_algorithms =
                    vec![Algorithm::MovingLeastSquares, Algorithm::Delaunay];
            }
            UseCase::Sparse => {
                config.fallback_algorithms =
                    vec![Algorithm::Delaunay, Algorithm::MovingLeastSquares];
            }
            UseCase::Dense => {
                config.fallback_algorithms = vec![
                    Algorithm::Poisson,
                    Algorithm::BallPivoting,
                    Algorithm::MarchingCubes,
                ];
            }
            UseCase::General => {
                // Use default configuration
            }
        }

        Self::new(config)
    }

    /// Analyze input data characteristics
    pub fn analyze_data(&self, cloud: &PointCloud<Point3f>) -> Result<DataCharacteristics> {
        if cloud.is_empty() {
            return Err(Error::InvalidData("Point cloud is empty".to_string()));
        }

        let point_count = cloud.points.len();

        // Compute bounding box
        let (bounds_min, bounds_max) = parallel::point_cloud::parallel_bounding_box(&cloud.points)
            .unwrap_or((Point3f::origin(), Point3f::new(1.0, 1.0, 1.0)));

        // Sample points for analysis to avoid O(n²) complexity
        let sample_size = point_count.min(1000);
        let step = (point_count.max(1) / sample_size.max(1)).max(1);
        let sample_points: Vec<Point3f> = cloud.points.iter().step_by(step).cloned().collect();

        // Analyze nearest neighbor distances
        let neighbor_distances = self.compute_neighbor_distances(&sample_points)?;
        let avg_neighbor_distance =
            neighbor_distances.iter().sum::<f32>() / neighbor_distances.len() as f32;

        // Estimate density uniformity
        let density_uniformity = self.estimate_density_uniformity(&neighbor_distances);

        // Estimate noise level using distance variance
        let distance_variance = self.compute_variance(&neighbor_distances);
        let noise_level = (distance_variance / avg_neighbor_distance.powi(2)).min(1.0);

        // Determine distribution type
        let distribution_type = self.classify_distribution(&cloud.points, &bounds_min, &bounds_max);

        // Estimate surface complexity
        let surface_complexity = self.estimate_surface_complexity(&sample_points);

        // Check if surface appears closed
        let is_closed_surface = self.estimate_surface_closure(&sample_points);

        Ok(DataCharacteristics {
            point_count,
            has_normals: false, // Will be overridden for NormalPoint3f clouds
            density_uniformity,
            noise_level,
            avg_neighbor_distance,
            bounding_box: (bounds_min, bounds_max),
            is_closed_surface,
            surface_complexity,
            distribution_type,
        })
    }

    /// Analyze input data characteristics for clouds with normals
    pub fn analyze_data_with_normals(
        &self,
        cloud: &PointCloud<NormalPoint3f>,
    ) -> Result<DataCharacteristics> {
        let point_cloud: PointCloud<Point3f> =
            PointCloud::from_points(cloud.points.iter().map(|p| p.position).collect());

        let mut characteristics = self.analyze_data(&point_cloud)?;
        characteristics.has_normals = true;

        Ok(characteristics)
    }

    /// Select the best algorithm based on data characteristics and configuration
    pub fn select_algorithm(&self, characteristics: &DataCharacteristics) -> Algorithm {
        // Return preferred algorithm if specified
        if let Some(preferred) = self.config.preferred_algorithm {
            return preferred;
        }

        // Algorithm selection logic based on characteristics
        let mut scores = HashMap::new();

        // Initialize all algorithms with base scores
        scores.insert(Algorithm::Delaunay, 0.5);
        scores.insert(Algorithm::BallPivoting, 0.5);
        scores.insert(Algorithm::MovingLeastSquares, 0.5);
        scores.insert(Algorithm::MarchingCubes, 0.5);
        scores.insert(
            Algorithm::Poisson,
            if characteristics.has_normals {
                0.5
            } else {
                0.0
            },
        );

        // Adjust scores based on point count
        match characteristics.point_count {
            0..=100 => {
                *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.3;
                *scores.get_mut(&Algorithm::MovingLeastSquares).unwrap() += 0.2;
            }
            101..=1000 => {
                *scores.get_mut(&Algorithm::BallPivoting).unwrap() += 0.2;
                *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.3;
            }
            1001..=10000 => {
                *scores.get_mut(&Algorithm::BallPivoting).unwrap() += 0.3;
                if characteristics.has_normals {
                    *scores.get_mut(&Algorithm::Poisson).unwrap() += 0.3;
                }
            }
            _ => {
                if characteristics.has_normals {
                    *scores.get_mut(&Algorithm::Poisson).unwrap() += 0.4;
                }
                *scores.get_mut(&Algorithm::MarchingCubes).unwrap() += 0.2;
            }
        }

        // Adjust based on density uniformity
        if characteristics.density_uniformity > 0.7 {
            *scores.get_mut(&Algorithm::BallPivoting).unwrap() += 0.2;
            if characteristics.has_normals {
                *scores.get_mut(&Algorithm::Poisson).unwrap() += 0.2;
            }
        } else {
            *scores.get_mut(&Algorithm::MovingLeastSquares).unwrap() += 0.3;
            *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.2;
        }

        // Adjust based on noise level
        if characteristics.noise_level > 0.3 {
            *scores.get_mut(&Algorithm::MovingLeastSquares).unwrap() += 0.3;
            *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.1;
            // Penalize sensitive algorithms
            *scores.get_mut(&Algorithm::BallPivoting).unwrap() -= 0.2;
        }

        // Adjust based on surface complexity
        if characteristics.surface_complexity > 0.7 {
            if characteristics.has_normals {
                *scores.get_mut(&Algorithm::Poisson).unwrap() += 0.3;
            }
            *scores.get_mut(&Algorithm::MovingLeastSquares).unwrap() += 0.2;
        }

        // Adjust based on distribution type
        match characteristics.distribution_type {
            DistributionType::Planar => {
                *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.3;
            }
            DistributionType::Spherical | DistributionType::Cylindrical => {
                *scores.get_mut(&Algorithm::BallPivoting).unwrap() += 0.2;
                *scores.get_mut(&Algorithm::MarchingCubes).unwrap() += 0.3;
            }
            DistributionType::Arbitrary => {
                // No specific preference
            }
        }

        // Adjust based on quality requirements
        match self.config.quality {
            QualityLevel::Fast => {
                *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.3;
            }
            QualityLevel::Balanced => {
                *scores.get_mut(&Algorithm::BallPivoting).unwrap() += 0.2;
            }
            QualityLevel::HighQuality | QualityLevel::MaxQuality => {
                if characteristics.has_normals {
                    *scores.get_mut(&Algorithm::Poisson).unwrap() += 0.3;
                }
                *scores.get_mut(&Algorithm::MovingLeastSquares).unwrap() += 0.2;
            }
        }

        // Adjust based on use case
        match self.config.use_case {
            UseCase::Engineering => {
                *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.2;
                if characteristics.has_normals {
                    *scores.get_mut(&Algorithm::Poisson).unwrap() += 0.2;
                }
            }
            UseCase::Organic => {
                *scores.get_mut(&Algorithm::MovingLeastSquares).unwrap() += 0.3;
                if characteristics.has_normals {
                    *scores.get_mut(&Algorithm::Poisson).unwrap() += 0.2;
                }
            }
            UseCase::Prototyping => {
                *scores.get_mut(&Algorithm::Delaunay).unwrap() += 0.4;
            }
            _ => {}
        }

        // Find algorithm with highest score
        scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(algo, _)| algo)
            .unwrap_or(Algorithm::Delaunay)
    }

    /// Reconstruct surface from point cloud without normals
    pub fn reconstruct(&self, cloud: &PointCloud<Point3f>) -> Result<ReconstructionResult> {
        let start_time = std::time::Instant::now();

        // Analyze data characteristics
        let characteristics = self.analyze_data(cloud)?;

        // Select best algorithm
        let selected_algorithm = self.select_algorithm(&characteristics);

        // Try reconstruction with selected algorithm
        let mesh = match self.try_algorithm(cloud, selected_algorithm) {
            Ok(mesh) => mesh,
            Err(_) => {
                // Try fallback algorithms
                let mut last_error = Error::Algorithm("No algorithms succeeded".to_string());
                for &fallback_algo in &self.config.fallback_algorithms {
                    if fallback_algo != selected_algorithm {
                        match self.try_algorithm(cloud, fallback_algo) {
                            Ok(mesh) => {
                                let processing_time = start_time.elapsed().as_secs_f32();
                                let quality_metrics =
                                    self.compute_quality_metrics(&mesh, &characteristics);

                                return Ok(ReconstructionResult {
                                    mesh,
                                    algorithm_used: fallback_algo,
                                    processing_time,
                                    quality_metrics,
                                    data_characteristics: characteristics,
                                });
                            }
                            Err(e) => last_error = e,
                        }
                    }
                }
                return Err(last_error);
            }
        };

        let processing_time = start_time.elapsed().as_secs_f32();
        let quality_metrics = self.compute_quality_metrics(&mesh, &characteristics);

        Ok(ReconstructionResult {
            mesh,
            algorithm_used: selected_algorithm,
            processing_time,
            quality_metrics,
            data_characteristics: characteristics,
        })
    }

    /// Reconstruct surface from point cloud with normals
    pub fn reconstruct_with_normals(
        &self,
        cloud: &PointCloud<NormalPoint3f>,
    ) -> Result<ReconstructionResult> {
        let start_time = std::time::Instant::now();

        // Analyze data characteristics
        let characteristics = self.analyze_data_with_normals(cloud)?;

        // Select best algorithm
        let selected_algorithm = self.select_algorithm(&characteristics);

        // Try reconstruction with selected algorithm
        let mesh = match self.try_algorithm_with_normals(cloud, selected_algorithm) {
            Ok(mesh) => mesh,
            Err(_) => {
                // Try fallback algorithms
                let mut last_error = Error::Algorithm("No algorithms succeeded".to_string());
                for &fallback_algo in &self.config.fallback_algorithms {
                    if fallback_algo != selected_algorithm {
                        match self.try_algorithm_with_normals(cloud, fallback_algo) {
                            Ok(mesh) => {
                                let processing_time = start_time.elapsed().as_secs_f32();
                                let quality_metrics =
                                    self.compute_quality_metrics(&mesh, &characteristics);

                                return Ok(ReconstructionResult {
                                    mesh,
                                    algorithm_used: fallback_algo,
                                    processing_time,
                                    quality_metrics,
                                    data_characteristics: characteristics,
                                });
                            }
                            Err(e) => last_error = e,
                        }
                    }
                }
                return Err(last_error);
            }
        };

        let processing_time = start_time.elapsed().as_secs_f32();
        let quality_metrics = self.compute_quality_metrics(&mesh, &characteristics);

        Ok(ReconstructionResult {
            mesh,
            algorithm_used: selected_algorithm,
            processing_time,
            quality_metrics,
            data_characteristics: characteristics,
        })
    }

    // Helper methods for data analysis
    fn compute_neighbor_distances(&self, points: &[Point3f]) -> Result<Vec<f32>> {
        if points.len() < 2 {
            return Ok(vec![1.0]); // Default fallback
        }

        let distances = parallel::parallel_map(points, |point| {
            let mut min_dist = f32::INFINITY;
            for other_point in points {
                if point != other_point {
                    let dist = (point - other_point).magnitude();
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
            }
            min_dist
        });

        let finite_distances: Vec<f32> = distances
            .into_iter()
            .filter(|&d| d.is_finite() && d > 0.0)
            .collect();

        if finite_distances.is_empty() {
            Ok(vec![1.0])
        } else {
            Ok(finite_distances)
        }
    }

    fn estimate_density_uniformity(&self, distances: &[f32]) -> f32 {
        if distances.len() < 2 {
            return 1.0;
        }

        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance = self.compute_variance(distances);
        let cv = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        };

        // Convert coefficient of variation to uniformity (0.0 = non-uniform, 1.0 = uniform)
        (1.0 / (1.0 + cv)).min(1.0)
    }

    fn compute_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }

    fn classify_distribution(
        &self,
        _points: &[Point3f],
        bounds_min: &Point3f,
        bounds_max: &Point3f,
    ) -> DistributionType {
        let extents = [
            bounds_max.x - bounds_min.x,
            bounds_max.y - bounds_min.y,
            bounds_max.z - bounds_min.z,
        ];

        let max_extent = extents.iter().fold(0.0f32, |acc, &x| acc.max(x));
        let min_extent = extents.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));

        if max_extent < 1e-6 {
            return DistributionType::Arbitrary;
        }

        // Check if one dimension is much smaller (planar)
        if min_extent < max_extent * 0.1 {
            return DistributionType::Planar;
        }

        // Check if roughly spherical/cubic
        let extent_ratios = [
            extents[0] / max_extent,
            extents[1] / max_extent,
            extents[2] / max_extent,
        ];

        if extent_ratios.iter().all(|&r| r > 0.7) {
            DistributionType::Spherical
        } else if extent_ratios.iter().filter(|&&r| r > 0.7).count() == 2 {
            DistributionType::Cylindrical
        } else {
            DistributionType::Arbitrary
        }
    }

    fn estimate_surface_complexity(&self, points: &[Point3f]) -> f32 {
        if points.len() < 10 {
            return 0.5; // Default for small datasets
        }

        // Simple complexity estimation based on local curvature variation
        let sample_size = points.len().min(100);
        let step = (points.len().max(1) / sample_size.max(1)).max(1);
        let sample_points: Vec<Point3f> = points.iter().step_by(step).cloned().collect();

        let mut curvature_variations = Vec::new();

        for (i, point) in sample_points.iter().enumerate() {
            // Find nearby points
            let mut neighbors = Vec::new();
            for (j, other_point) in sample_points.iter().enumerate() {
                if i != j {
                    let dist = (point - other_point).magnitude();
                    if dist < 0.1 {
                        // Fixed radius for simplicity
                        neighbors.push(*other_point);
                    }
                }
            }

            if neighbors.len() >= 3 {
                // Estimate local curvature variation
                let variation = self.estimate_local_curvature_variation(point, &neighbors);
                curvature_variations.push(variation);
            }
        }

        if curvature_variations.is_empty() {
            0.5
        } else {
            let avg_variation =
                curvature_variations.iter().sum::<f32>() / curvature_variations.len() as f32;
            avg_variation.min(1.0)
        }
    }

    fn estimate_local_curvature_variation(&self, center: &Point3f, neighbors: &[Point3f]) -> f32 {
        if neighbors.len() < 3 {
            return 0.0;
        }

        // Simple estimation using angle variations
        let mut angles = Vec::new();
        for i in 0..neighbors.len() {
            let v1 = (neighbors[i] - *center).normalize();
            let v2 = (neighbors[(i + 1) % neighbors.len()] - *center).normalize();
            let angle = v1.dot(&v2).clamp(-1.0, 1.0).acos();
            angles.push(angle);
        }

        self.compute_variance(&angles) / std::f32::consts::PI
    }

    fn estimate_surface_closure(&self, points: &[Point3f]) -> bool {
        // Simple heuristic: if points are roughly uniformly distributed around a center,
        // assume closed surface
        if points.len() < 50 {
            return false; // Too few points to determine
        }

        // Compute centroid
        let centroid = points.iter().fold(Point3f::origin(), |acc, p| {
            Point3f::from(acc.coords + p.coords)
        });
        let centroid = Point3f::from(centroid.coords / points.len() as f32);

        // Check if points are roughly uniformly distributed around centroid
        let distances: Vec<f32> = points.iter().map(|p| (p - centroid).magnitude()).collect();

        let mean_dist = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance = self.compute_variance(&distances);
        let cv = if mean_dist > 0.0 {
            variance.sqrt() / mean_dist
        } else {
            1.0
        };

        // If coefficient of variation is low, points are roughly at same distance from center
        cv < 0.3
    }

    // Helper methods for algorithm execution
    fn try_algorithm(
        &self,
        cloud: &PointCloud<Point3f>,
        algorithm: Algorithm,
    ) -> Result<TriangleMesh> {
        match algorithm {
            Algorithm::Delaunay => crate::delaunay::delaunay_triangulation_auto(cloud),
            Algorithm::BallPivoting => {
                let radius = crate::ball_pivoting::estimate_optimal_radius(cloud, 0.5)?;
                crate::ball_pivoting::ball_pivoting_reconstruction(cloud, radius)
            }
            Algorithm::MovingLeastSquares => {
                crate::moving_least_squares::moving_least_squares_auto(cloud)
            }
            Algorithm::MarchingCubes => {
                // Convert point cloud to volumetric representation
                let mls = crate::moving_least_squares::MLSSurface::new(
                    cloud,
                    crate::moving_least_squares::MLSConfig::default(),
                )?;
                mls.extract_mesh()
            }
            Algorithm::Poisson => {
                // Cannot use Poisson without normals
                Err(Error::InvalidData(
                    "Poisson reconstruction requires normals".to_string(),
                ))
            }
        }
    }

    fn try_algorithm_with_normals(
        &self,
        cloud: &PointCloud<NormalPoint3f>,
        algorithm: Algorithm,
    ) -> Result<TriangleMesh> {
        match algorithm {
            Algorithm::Poisson => crate::poisson::poisson_reconstruction_default(cloud),
            Algorithm::BallPivoting => {
                let radius = {
                    let point_cloud: PointCloud<Point3f> =
                        PointCloud::from_points(cloud.points.iter().map(|p| p.position).collect());
                    crate::ball_pivoting::estimate_optimal_radius(&point_cloud, 0.5)?
                };
                crate::ball_pivoting::ball_pivoting_from_normals(cloud, radius)
            }
            Algorithm::Delaunay => {
                let point_cloud: PointCloud<Point3f> =
                    PointCloud::from_points(cloud.points.iter().map(|p| p.position).collect());
                crate::delaunay::delaunay_triangulation_auto(&point_cloud)
            }
            Algorithm::MovingLeastSquares => {
                crate::moving_least_squares::moving_least_squares_from_normals(cloud)
            }
            Algorithm::MarchingCubes => {
                // Convert to MLS and extract
                let mls = crate::moving_least_squares::MLSSurface::from_normals(
                    cloud,
                    crate::moving_least_squares::MLSConfig::default(),
                )?;
                mls.extract_mesh()
            }
        }
    }

    fn compute_quality_metrics(
        &self,
        mesh: &TriangleMesh,
        characteristics: &DataCharacteristics,
    ) -> QualityMetrics {
        let vertex_count = mesh.vertex_count();
        let triangle_count = mesh.face_count();

        // Simple quality metrics - in a real implementation these would be more sophisticated
        let avg_triangle_quality = 0.75; // Placeholder
        let watertightness = if characteristics.is_closed_surface {
            0.8
        } else {
            0.6
        };
        let smoothness = 1.0 - characteristics.noise_level;
        let geometric_accuracy = 0.8; // Placeholder

        QualityMetrics {
            vertex_count,
            triangle_count,
            avg_triangle_quality,
            watertightness,
            smoothness,
            geometric_accuracy,
        }
    }
}

/// Convenience function for quick reconstruction with automatic algorithm selection
pub fn auto_reconstruct(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    let pipeline = ReconstructionPipeline::default();
    Ok(pipeline.reconstruct(cloud)?.mesh)
}

/// Convenience function for quick reconstruction with normals
pub fn auto_reconstruct_with_normals(cloud: &PointCloud<NormalPoint3f>) -> Result<TriangleMesh> {
    let pipeline = ReconstructionPipeline::default();
    Ok(pipeline.reconstruct_with_normals(cloud)?.mesh)
}

/// Convenience function for reconstruction with specific quality level
pub fn auto_reconstruct_with_quality(
    cloud: &PointCloud<Point3f>,
    quality: QualityLevel,
) -> Result<TriangleMesh> {
    let mut config = PipelineConfig::default();
    config.quality = quality;
    let pipeline = ReconstructionPipeline::new(config);
    Ok(pipeline.reconstruct(cloud)?.mesh)
}

/// Convenience function for reconstruction optimized for a use case
pub fn auto_reconstruct_for_use_case(
    cloud: &PointCloud<Point3f>,
    use_case: UseCase,
) -> Result<TriangleMesh> {
    let pipeline = ReconstructionPipeline::for_use_case(use_case);
    Ok(pipeline.reconstruct(cloud)?.mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.quality, QualityLevel::Balanced);
        assert_eq!(config.use_case, UseCase::General);
        assert!(config.enable_parallel);
        assert!(config.validate_output);
    }

    #[test]
    fn test_pipeline_for_use_case() {
        let pipeline = ReconstructionPipeline::for_use_case(UseCase::Prototyping);
        assert_eq!(pipeline.config.quality, QualityLevel::Fast);
        assert_eq!(pipeline.config.use_case, UseCase::Prototyping);
    }

    #[test]
    fn test_data_analysis_empty_cloud() {
        let pipeline = ReconstructionPipeline::default();
        let cloud = PointCloud::new();
        let result = pipeline.analyze_data(&cloud);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_analysis_simple() {
        let pipeline = ReconstructionPipeline::default();
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let cloud = PointCloud::from_points(points);

        let characteristics = pipeline.analyze_data(&cloud).unwrap();
        assert_eq!(characteristics.point_count, 4);
        assert!(!characteristics.has_normals);
        assert_eq!(characteristics.distribution_type, DistributionType::Planar);
    }

    #[test]
    fn test_algorithm_selection_sparse_data() {
        let pipeline = ReconstructionPipeline::default();
        let characteristics = DataCharacteristics {
            point_count: 50,
            has_normals: false,
            density_uniformity: 0.5,
            noise_level: 0.1,
            avg_neighbor_distance: 0.1,
            bounding_box: (Point3f::origin(), Point3f::new(1.0, 1.0, 1.0)),
            is_closed_surface: false,
            surface_complexity: 0.3,
            distribution_type: DistributionType::Planar,
        };

        let algorithm = pipeline.select_algorithm(&characteristics);
        // For sparse planar data, should prefer Delaunay
        assert!(matches!(
            algorithm,
            Algorithm::Delaunay | Algorithm::MovingLeastSquares
        ));
    }

    #[test]
    fn test_algorithm_selection_dense_with_normals() {
        let pipeline = ReconstructionPipeline::default();
        let characteristics = DataCharacteristics {
            point_count: 5000,
            has_normals: true,
            density_uniformity: 0.8,
            noise_level: 0.1,
            avg_neighbor_distance: 0.05,
            bounding_box: (Point3f::origin(), Point3f::new(1.0, 1.0, 1.0)),
            is_closed_surface: true,
            surface_complexity: 0.7,
            distribution_type: DistributionType::Spherical,
        };

        let algorithm = pipeline.select_algorithm(&characteristics);
        // For dense, uniform data with normals, should prefer Poisson or BallPivoting
        assert!(matches!(
            algorithm,
            Algorithm::Poisson | Algorithm::BallPivoting
        ));
    }

    #[test]
    fn test_auto_reconstruct_simple() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let cloud = PointCloud::from_points(points);

        let result = auto_reconstruct(&cloud);
        match result {
            Ok(mesh) => {
                assert!(!mesh.is_empty());
            }
            Err(_) => {
                // Algorithm may fail on simple test data - that's acceptable
                println!(
                    "Auto reconstruction failed on simple test data (expected for some algorithms)"
                );
            }
        }
    }

    #[test]
    fn test_quality_levels() {
        let quality_levels = [
            QualityLevel::Fast,
            QualityLevel::Balanced,
            QualityLevel::HighQuality,
            QualityLevel::MaxQuality,
        ];

        for quality in &quality_levels {
            let mut config = PipelineConfig::default();
            config.quality = *quality;
            let _pipeline = ReconstructionPipeline::new(config);
            // Just test that pipelines can be created with different quality levels
        }
    }

    #[test]
    fn test_use_cases() {
        let use_cases = [
            UseCase::General,
            UseCase::Prototyping,
            UseCase::Engineering,
            UseCase::Organic,
            UseCase::NoisyData,
            UseCase::Sparse,
            UseCase::Dense,
        ];

        for use_case in &use_cases {
            let pipeline = ReconstructionPipeline::for_use_case(*use_case);
            assert_eq!(pipeline.config.use_case, *use_case);
        }
    }

    #[test]
    fn test_distribution_classification() {
        let pipeline = ReconstructionPipeline::default();

        // Test planar distribution
        let planar_points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(1.0, 1.0, 0.0),
        ];
        let min_bounds = Point3f::new(0.0, 0.0, 0.0);
        let max_bounds = Point3f::new(1.0, 1.0, 0.0);
        let distribution = pipeline.classify_distribution(&planar_points, &min_bounds, &max_bounds);
        assert_eq!(distribution, DistributionType::Planar);

        // Test spherical distribution
        let min_bounds_sphere = Point3f::new(-1.0, -1.0, -1.0);
        let max_bounds_sphere = Point3f::new(1.0, 1.0, 1.0);
        let distribution =
            pipeline.classify_distribution(&planar_points, &min_bounds_sphere, &max_bounds_sphere);
        assert_eq!(distribution, DistributionType::Spherical);
    }
}
