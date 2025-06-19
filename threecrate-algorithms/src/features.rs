//! Feature extraction algorithms

use threecrate_core::{PointCloud, Result, Point3f};

/// Extract FPFH (Fast Point Feature Histograms) features
pub fn extract_fpfh_features(_cloud: &PointCloud<Point3f>) -> Result<Vec<Vec<f32>>> {
    // TODO: Implement FPFH feature extraction
    todo!("FPFH feature extraction not yet implemented")
} 