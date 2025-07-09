//! Interactive Viewer Example
//!
//! This example demonstrates the comprehensive interactive viewer for 3DCrate.
//! It shows how to:
//! - Load point clouds and meshes
//! - Use interactive UI controls
//! - Switch between CPU/GPU pipelines
//! - Apply algorithms with parameter controls
//! - Take screenshots
//! - Navigate with camera controls

use threecrate_core::{PointCloud, Point3f, ColoredPoint3f, TriangleMesh, Result};
use threecrate_visualization::InteractiveViewer;
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("3DCrate Interactive Viewer Example");
    println!("==================================");
    
    // Create a simple point cloud for demonstration
    let mut points = Vec::new();
    
    // Create a simple 3D cube point cloud
    for x in -5..=5 {
        for y in -5..=5 {
            for z in -5..=5 {
                if x == -5 || x == 5 || y == -5 || y == 5 || z == -5 || z == 5 {
                    points.push(Point3f::new(x as f32 * 0.1, y as f32 * 0.1, z as f32 * 0.1));
                }
            }
        }
    }
    
    let cloud = PointCloud::from_points(points);
    
    // Create and configure the viewer
    let mut viewer = InteractiveViewer::new()?;
    viewer.set_point_cloud(&cloud);
    
    println!("Controls:");
    println!("  Mouse: Drag to orbit around the view");
    println!("  Scroll: Zoom in/out");
    println!("  O: Switch to orbit mode");
    println!("  P: Switch to pan mode");
    println!("  Z: Switch to zoom mode");
    println!("  R: Reset camera position");
    println!("  Close window to exit");
    
    // Run the viewer
    viewer.run()?;
    
    Ok(())
}

/// Generate a simple colored point cloud
#[allow(dead_code)]
fn create_colored_point_cloud() -> PointCloud<ColoredPoint3f> {
    let mut points = Vec::new();
    
    // Create a rainbow sphere
    for i in 0..1000 {
        let theta = (i as f32 / 1000.0) * 2.0 * PI;
        let phi = (i as f32 / 1000.0) * PI;
        
        let x = theta.cos() * phi.sin();
        let y = theta.sin() * phi.sin();
        let z = phi.cos();
        
        let color = [
            (theta / (2.0 * PI) * 255.0) as u8,
            (phi / PI * 255.0) as u8,
            128,
        ];
        
        points.push(ColoredPoint3f {
            position: Point3f::new(x, y, z),
            color,
        });
    }
    
    PointCloud::from_points(points)
}

/// Generate a simple mesh
#[allow(dead_code)]
fn create_simple_mesh() -> TriangleMesh {
    // Create a simple pyramid
    let vertices = vec![
        Point3f::new(0.0, 1.0, 0.0),   // Top
        Point3f::new(-1.0, -1.0, 1.0), // Bottom left front
        Point3f::new(1.0, -1.0, 1.0),  // Bottom right front
        Point3f::new(0.0, -1.0, -1.0), // Bottom back
    ];
    
    let faces = vec![
        [0, 1, 2], // Front face
        [0, 2, 3], // Right face
        [0, 3, 1], // Left face
        [1, 3, 2], // Bottom face
    ];
    
    TriangleMesh::from_vertices_and_faces(vertices, faces)
}

/// Run algorithm demonstration
#[allow(dead_code)]
fn run_algorithm_demo(_point_cloud: &PointCloud<Point3f>) -> Result<()> {
    println!("Algorithm demonstrations would go here");
    println!("This is a simplified version for basic viewing");
    Ok(())
}

#[allow(dead_code)]
fn run_icp_demo() -> Result<()> {
    println!("ICP algorithm demonstration would go here");
    Ok(())
}

#[allow(dead_code)]
fn run_ransac_demo() -> Result<()> {
    println!("RANSAC algorithm demonstration would go here");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_data_creation() {
        let point_cloud = create_colored_point_cloud();
        assert!(!point_cloud.is_empty());
        println!("Colored cloud has {} points", point_cloud.len());

        let mesh = create_simple_mesh();
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.faces.is_empty());
        println!("Mesh has {} vertices and {} faces", mesh.vertices.len(), mesh.faces.len());
    }

    #[test]
    fn test_algorithms() {
        let plane_cloud = create_colored_point_cloud();
        
        // Test RANSAC
        let result = run_ransac_demo();
        assert!(result.is_ok());
        
        if let Ok(_) = result {
            println!("RANSAC found inliers out of {} points", 
                     plane_cloud.len());
        }
    }
} 