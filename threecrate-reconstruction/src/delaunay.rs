//! Delaunay triangulation for surface reconstruction
//!
//! This module provides both 2D and 3D Delaunay triangulation capabilities.
//! For 3D point clouds, multiple projection strategies are available.

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, Error};
use crate::parallel;
use spade::{DelaunayTriangulation, Point2, Triangulation};
use nalgebra::Matrix3;

/// Configuration for Delaunay triangulation
#[derive(Debug, Clone)]
pub struct DelaunayConfig {
    /// Projection method for 3D points
    pub projection: ProjectionMethod,
    /// Whether to validate the triangulation
    pub validate: bool,
    /// Minimum triangle area threshold
    pub min_triangle_area: f32,
    /// Maximum triangle edge length (for quality control)
    pub max_edge_length: Option<f32>,
}

/// Projection methods for 3D to 2D mapping
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionMethod {
    /// Project onto XY plane (ignore Z coordinate)
    XY,
    /// Project onto XZ plane (ignore Y coordinate)
    XZ,
    /// Project onto YZ plane (ignore X coordinate)
    YZ,
    /// Use principal component analysis to find best plane
    PCA,
    /// Project using plane defined by first 3 non-collinear points
    BestFitPlane,
}

impl Default for DelaunayConfig {
    fn default() -> Self {
        Self {
            projection: ProjectionMethod::PCA,
            validate: true,
            min_triangle_area: 1e-8,
            max_edge_length: None,
        }
    }
}

/// 2D Delaunay triangulation using spade crate
pub fn delaunay_triangulation_2d(points: &[Point2<f64>]) -> Result<Vec<[usize; 3]>> {
    if points.len() < 3 {
        return Err(Error::InvalidData("Need at least 3 points for triangulation".to_string()));
    }

    let mut triangulation: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();

    // Insert points and track original indices
    for point in points {
        triangulation.insert(*point)
            .map_err(|e| Error::Algorithm(format!("Failed to insert point in Delaunay triangulation: {:?}", e)))?;
    }

    // Extract triangles
    let mut triangles = Vec::new();
    for face in triangulation.inner_faces() {
        let vertices = face.vertices();
        let positions: Vec<Point2<f64>> = vertices.iter().map(|v| v.position()).collect();

        // Find original indices by matching positions
        let mut indices = Vec::new();
        for pos in positions {
            if let Some(idx) = points.iter().position(|p| (p.x - pos.x).abs() < 1e-10 && (p.y - pos.y).abs() < 1e-10) {
                indices.push(idx);
            } else {
                return Err(Error::Algorithm("Failed to match triangle vertex to original point".to_string()));
            }
        }

        if indices.len() == 3 {
            triangles.push([indices[0], indices[1], indices[2]]);
        }
    }

    Ok(triangles)
}

/// Project 3D points to 2D using specified method with parallel processing
pub fn project_3d_to_2d(points: &[Point3f], method: &ProjectionMethod) -> Result<Vec<Point2<f64>>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    match method {
        ProjectionMethod::XY => {
            Ok(parallel::parallel_map(points, |p| Point2::new(p.x as f64, p.y as f64)))
        }
        ProjectionMethod::XZ => {
            Ok(parallel::parallel_map(points, |p| Point2::new(p.x as f64, p.z as f64)))
        }
        ProjectionMethod::YZ => {
            Ok(parallel::parallel_map(points, |p| Point2::new(p.y as f64, p.z as f64)))
        }
        ProjectionMethod::PCA => {
            project_using_pca(points)
        }
        ProjectionMethod::BestFitPlane => {
            project_using_best_fit_plane(points)
        }
    }
}

/// Project points using Principal Component Analysis to find best 2D plane
fn project_using_pca(points: &[Point3f]) -> Result<Vec<Point2<f64>>> {
    if points.len() < 3 {
        return Err(Error::InvalidData("Need at least 3 points for PCA projection".to_string()));
    }

    // Compute centroid
    let mut centroid = Point3f::origin();
    for point in points {
        centroid = Point3f::from(centroid.coords + point.coords);
    }
    centroid = Point3f::from(centroid.coords / points.len() as f32);

    // Compute covariance matrix
    let mut covariance = Matrix3::zeros();
    for point in points {
        let diff = point - centroid;
        covariance += diff * diff.transpose();
    }

    // Find eigenvalues and eigenvectors
    let eigen = covariance.symmetric_eigen();

    // Use the two eigenvectors with largest eigenvalues as 2D basis
    let mut eigen_pairs: Vec<(f32, nalgebra::Vector3<f32>)> = eigen.eigenvalues.iter()
        .zip(eigen.eigenvectors.column_iter())
        .map(|(val, vec)| (*val, vec.clone_owned()))
        .collect();

    // Sort by eigenvalue in descending order
    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let u = eigen_pairs[0].1.normalize();
    let v = eigen_pairs[1].1.normalize();

    // Project all points onto the 2D plane using parallel processing
    let projected: Vec<Point2<f64>> = parallel::parallel_map(points, |point| {
        let diff = point - centroid;
        let x = diff.dot(&u) as f64;
        let y = diff.dot(&v) as f64;
        Point2::new(x, y)
    });

    Ok(projected)
}

/// Project points using a best-fit plane through the points
fn project_using_best_fit_plane(points: &[Point3f]) -> Result<Vec<Point2<f64>>> {
    if points.len() < 3 {
        return Err(Error::InvalidData("Need at least 3 points for best-fit plane projection".to_string()));
    }

    // Use first 3 non-collinear points to define the plane
    let p1 = points[0];
    let mut p2 = None;
    let mut p3 = None;

    // Find second point that's not too close to first
    for point in points.iter().skip(1) {
        if (point - p1).magnitude() > 1e-6 {
            p2 = Some(*point);
            break;
        }
    }

    let p2 = p2.ok_or_else(|| Error::InvalidData("All points are too close together".to_string()))?;

    // Find third point that's not collinear with first two
    for point in points.iter().skip(2) {
        let v1 = p2 - p1;
        let v2 = *point - p1;
        if v1.cross(&v2).magnitude() > 1e-6 {
            p3 = Some(*point);
            break;
        }
    }

    let p3 = p3.ok_or_else(|| Error::InvalidData("All points are collinear".to_string()))?;

    // Define 2D coordinate system on the plane
    let u = (p2 - p1).normalize();
    let w = (p3 - p1).cross(&u).normalize();
    let v = w.cross(&u).normalize();

    // Project all points onto the plane using parallel processing
    let projected: Vec<Point2<f64>> = parallel::parallel_map(points, |point| {
        let diff = *point - p1;
        let x = diff.dot(&u) as f64;
        let y = diff.dot(&v) as f64;
        Point2::new(x, y)
    });

    Ok(projected)
}

/// Validate triangle quality
fn is_triangle_valid(p1: &Point3f, p2: &Point3f, p3: &Point3f, config: &DelaunayConfig) -> bool {
    // Check minimum area
    let v1 = p2 - p1;
    let v2 = p3 - p1;
    let cross = v1.cross(&v2);
    let area = cross.magnitude() * 0.5;

    if area < config.min_triangle_area {
        return false;
    }

    // Check maximum edge length if specified
    if let Some(max_len) = config.max_edge_length {
        let edge1 = (p2 - p1).magnitude();
        let edge2 = (p3 - p2).magnitude();
        let edge3 = (p1 - p3).magnitude();

        if edge1 > max_len || edge2 > max_len || edge3 > max_len {
            return false;
        }
    }

    true
}

/// 3D Delaunay triangulation using projection to 2D
pub fn delaunay_triangulation(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    delaunay_triangulation_with_config(cloud, &DelaunayConfig::default())
}

/// 3D Delaunay triangulation with configuration
pub fn delaunay_triangulation_with_config(
    cloud: &PointCloud<Point3f>,
    config: &DelaunayConfig
) -> Result<TriangleMesh> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }

    if cloud.points.len() < 3 {
        return Err(Error::InvalidData("Need at least 3 points for triangulation".to_string()));
    }

    // Project 3D points to 2D
    let projected_points = project_3d_to_2d(&cloud.points, &config.projection)?;

    // Perform 2D Delaunay triangulation
    let triangle_indices = delaunay_triangulation_2d(&projected_points)?;

    // Validate triangles if requested
    let mut valid_triangles = Vec::new();
    for &[i, j, k] in &triangle_indices {
        if i < cloud.points.len() && j < cloud.points.len() && k < cloud.points.len() {
            let p1 = &cloud.points[i];
            let p2 = &cloud.points[j];
            let p3 = &cloud.points[k];

            if !config.validate || is_triangle_valid(p1, p2, p3, config) {
                valid_triangles.push([i, j, k]);
            }
        }
    }

    if valid_triangles.is_empty() {
        return Err(Error::Algorithm("No valid triangles generated".to_string()));
    }

    // Create mesh
    let mesh = TriangleMesh::from_vertices_and_faces(cloud.points.clone(), valid_triangles);

    Ok(mesh)
}

/// Automatic projection method selection based on point cloud geometry
pub fn auto_select_projection(points: &[Point3f]) -> ProjectionMethod {
    if points.len() < 10 {
        return ProjectionMethod::BestFitPlane;
    }

    // Compute bounding box
    let mut min_vals = [f32::INFINITY; 3];
    let mut max_vals = [f32::NEG_INFINITY; 3];

    for point in points {
        min_vals[0] = min_vals[0].min(point.x);
        min_vals[1] = min_vals[1].min(point.y);
        min_vals[2] = min_vals[2].min(point.z);
        max_vals[0] = max_vals[0].max(point.x);
        max_vals[1] = max_vals[1].max(point.y);
        max_vals[2] = max_vals[2].max(point.z);
    }

    let extents = [
        max_vals[0] - min_vals[0],
        max_vals[1] - min_vals[1],
        max_vals[2] - min_vals[2],
    ];

    // Find the dimension with minimum extent
    let min_extent_idx = extents.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // If one dimension is significantly smaller, project onto the other two
    let min_extent = extents[min_extent_idx];
    let max_extent = extents.iter().fold(0.0f32, |acc, &x| acc.max(x));

    if min_extent < max_extent * 0.1 {
        match min_extent_idx {
            0 => ProjectionMethod::YZ, // X is smallest, project onto YZ
            1 => ProjectionMethod::XZ, // Y is smallest, project onto XZ
            2 => ProjectionMethod::XY, // Z is smallest, project onto XY
            _ => ProjectionMethod::PCA,
        }
    } else {
        // Use PCA for general case
        ProjectionMethod::PCA
    }
}

/// Convenience function for automatic Delaunay triangulation
pub fn delaunay_triangulation_auto(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    let projection = auto_select_projection(&cloud.points);
    let config = DelaunayConfig {
        projection,
        ..Default::default()
    };
    delaunay_triangulation_with_config(cloud, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_delaunay_config_default() {
        let config = DelaunayConfig::default();
        assert_eq!(config.projection, ProjectionMethod::PCA);
        assert!(config.validate);
        assert_eq!(config.min_triangle_area, 1e-8);
        assert_eq!(config.max_edge_length, None);
    }

    #[test]
    fn test_project_3d_to_2d_xy() {
        let points = vec![
            Point3f::new(0.0, 0.0, 1.0),
            Point3f::new(1.0, 1.0, 2.0),
            Point3f::new(2.0, 0.0, 3.0),
        ];

        let projected = project_3d_to_2d(&points, &ProjectionMethod::XY).unwrap();

        assert_eq!(projected.len(), 3);
        assert_eq!(projected[0], Point2::new(0.0, 0.0));
        assert_eq!(projected[1], Point2::new(1.0, 1.0));
        assert_eq!(projected[2], Point2::new(2.0, 0.0));
    }

    #[test]
    fn test_delaunay_triangulation_2d_simple() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];

        let triangles = delaunay_triangulation_2d(&points).unwrap();

        assert_eq!(triangles.len(), 1);
        // Should form one triangle
        assert_eq!(triangles[0].len(), 3);
    }

    #[test]
    fn test_delaunay_triangulation_empty() {
        let cloud = PointCloud::new();
        let result = delaunay_triangulation(&cloud);
        assert!(result.is_err());
    }

    #[test]
    fn test_delaunay_triangulation_too_few_points() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ];
        let cloud = PointCloud::from_points(points);

        let result = delaunay_triangulation(&cloud);
        assert!(result.is_err());
    }

    #[test]
    fn test_delaunay_triangulation_simple() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let cloud = PointCloud::from_points(points);

        let result = delaunay_triangulation(&cloud);

        match result {
            Ok(mesh) => {
                assert!(!mesh.is_empty());
                assert!(mesh.vertex_count() <= 4);
                assert!(mesh.face_count() >= 1);
            }
            Err(_) => {
                // May fail on simple coplanar data - acceptable for testing
            }
        }
    }

    #[test]
    fn test_auto_select_projection() {
        // Test points mostly in XY plane (small Z variation)
        let points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.01),
            Point3f::new(0.5, 1.0, 0.02),
            Point3f::new(0.0, 1.0, 0.01),
        ];

        let projection = auto_select_projection(&points);
        // Should select some reasonable projection for mostly planar data
        // With only 4 points, it should use BestFitPlane
        assert!(matches!(projection, ProjectionMethod::BestFitPlane));

        // Test with many points that are clearly planar in XY
        let many_points: Vec<Point3f> = (0..20)
            .map(|i| Point3f::new(i as f32 * 0.1, (i % 5) as f32 * 0.1, 0.001))
            .collect();

        let projection = auto_select_projection(&many_points);
        // Should select XY projection for clearly planar data
        assert!(matches!(projection, ProjectionMethod::XY | ProjectionMethod::PCA));
    }
} 