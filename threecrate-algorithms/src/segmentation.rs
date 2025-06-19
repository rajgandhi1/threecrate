//! Segmentation algorithms

use threecrate_core::{PointCloud, Result, Point3f};

/// Plane segmentation using RANSAC
pub fn segment_plane(_cloud: &PointCloud<Point3f>, _threshold: f32) -> Result<Vec<usize>> {
    // TODO: Implement plane segmentation
    todo!("Plane segmentation not yet implemented")
}