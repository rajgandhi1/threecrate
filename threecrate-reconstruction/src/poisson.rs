//! Poisson surface reconstruction

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, NormalPoint3f, Error};

/// Configuration parameters for Poisson reconstruction
#[derive(Debug, Clone)]
pub struct PoissonConfig {
    /// Isosurface extraction level (default: 0.0)
    pub iso_level: f32,
    /// The maximum depth of the octree (default: 8)
    pub depth: u32,
    /// The minimum depth at which the octree will be pruned (default: 5)
    pub prune_depth: u32,
    /// The maximum depth at which the octree will be simplified (default: 10)
    pub full_depth: u32,
    /// Scale factor for the importance of point positions vs normals (default: 1.1)
    pub scale: f32,
    /// The number of samples to use for density estimation (default: 1.0)
    pub samples_per_node: f32,
    /// Whether to use confidence weighting (default: false)
    pub confidence: bool,
    /// Whether to output density information (default: false)
    pub output_density: bool,
    /// Number of gauss-seidel relaxations to be performed at each level (default: 8)
    pub cg_depth: u32,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            iso_level: 0.0,
            depth: 8,
            prune_depth: 5,
            full_depth: 10,
            scale: 1.1,
            samples_per_node: 1.0,
            confidence: false,
            output_density: false,
            cg_depth: 8,
        }
    }
}

/// Poisson surface reconstruction using the external poisson_reconstruction crate
/// 
/// # Arguments
/// * `cloud` - Point cloud with normal information
/// * `config` - Configuration parameters for the reconstruction
/// 
/// # Returns
/// * `Result<TriangleMesh>` - Reconstructed triangle mesh
pub fn poisson_reconstruction(
    cloud: &PointCloud<NormalPoint3f>, 
    _config: &PoissonConfig
) -> Result<TriangleMesh> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }

    // For now, return a placeholder until we can resolve the API issues
    Err(Error::Algorithm("Poisson reconstruction temporarily disabled due to API compatibility issues".to_string()))
}

/// Poisson surface reconstruction with default configuration
/// 
/// # Arguments
/// * `cloud` - Point cloud with normal information
/// 
/// # Returns
/// * `Result<TriangleMesh>` - Reconstructed triangle mesh
pub fn poisson_reconstruction_default(cloud: &PointCloud<NormalPoint3f>) -> Result<TriangleMesh> {
    poisson_reconstruction(cloud, &PoissonConfig::default())
}

/// Estimate normals and perform Poisson reconstruction in one step
/// 
/// # Arguments
/// * `cloud` - Point cloud without normals
/// * `k` - Number of neighbors for normal estimation
/// * `config` - Configuration parameters for the reconstruction
/// 
/// # Returns
/// * `Result<TriangleMesh>` - Reconstructed triangle mesh
pub fn poisson_reconstruction_with_normals(
    cloud: &PointCloud<Point3f>,
    k: usize,
    config: &PoissonConfig,
) -> Result<TriangleMesh> {
    // First estimate normals
    let normals_cloud = threecrate_algorithms::estimate_normals(cloud, k)?;
    
    // Then perform Poisson reconstruction
    poisson_reconstruction(&normals_cloud, config)
}

/// Wrapper function that matches the original API signature
#[deprecated(note = "Use poisson_reconstruction_default or poisson_reconstruction_with_normals instead")]
pub fn poisson_reconstruction_legacy(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    poisson_reconstruction_with_normals(cloud, 20, &PoissonConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_config_default() {
        let config = PoissonConfig::default();
        assert_eq!(config.depth, 8);
        assert_eq!(config.iso_level, 0.0);
        assert!(!config.confidence);
    }

    #[test]
    fn test_poisson_reconstruction_empty_cloud() {
        let cloud = PointCloud::<NormalPoint3f>::new();
        let result = poisson_reconstruction(&cloud, &PoissonConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_poisson_reconstruction_placeholder() {
        // Create a simple point cloud with normals
        let mut points = Vec::new();
        
        // Create points on a plane with upward normals
        for i in 0..5 {
            for j in 0..5 {
                points.push(NormalPoint3f {
                    position: Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0),
                    normal: nalgebra::Vector3::new(0.0, 0.0, 1.0),
                });
            }
        }
        
        let cloud = PointCloud::from_points(points);
        let config = PoissonConfig::default();
        
        let result = poisson_reconstruction(&cloud, &config);
        
        // Should fail with temporary placeholder message
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temporarily disabled"));
    }
} 