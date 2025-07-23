//! Basic usage examples for threecrate
//!
//! This example demonstrates basic usage of the threecrate library for point cloud processing.

use threecrate_core::{PointCloud, Point3f, Vector3f};
use threecrate_algorithms::{icp_point_to_point, icp_point_to_point_default};
use nalgebra::{Isometry3, UnitQuaternion, Translation3};
use std::f32::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ThreeCrate Basic Usage Example");
    println!("=============================");

    // Example 1: Basic ICP registration
    println!("\n1. Basic ICP Registration");
    println!("-------------------------");
    basic_icp_example()?;

    // Example 2: ICP with known transformation
    println!("\n2. ICP with Known Transformation");
    println!("--------------------------------");
    known_transform_example()?;

    // Example 3: ICP with noise
    println!("\n3. ICP with Noise");
    println!("-----------------");
    noisy_icp_example()?;

    // Example 4: ICP with convergence criteria
    println!("\n4. ICP with Convergence Criteria");
    println!("---------------------------------");
    convergence_example()?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

fn basic_icp_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create simple point clouds
    let mut source = PointCloud::new();
    let mut target = PointCloud::new();
    
    // Create a simple cube pattern
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                let point = Point3f::new(x as f32, y as f32, z as f32);
                source.push(point);
                // Target is translated by (1, 0.5, 0.25)
                target.push(point + Vector3f::new(1.0, 0.5, 0.25));
            }
        }
    }

    println!("  Source points: {}", source.len());
    println!("  Target points: {}", target.len());

    let init = Isometry3::identity();
    let result = icp_point_to_point(&source, &target, init, 50, 1e-6, None)?;

    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final MSE: {:.6}", result.mse);
    println!("  Translation: {:?}", result.transformation.translation.vector);
    println!("  Rotation angle: {:.6} radians", result.transformation.rotation.angle());

    Ok(())
}

fn known_transform_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create point clouds with a known transformation
    let mut source = PointCloud::new();
    let mut target = PointCloud::new();
    
    // Known transformation
    let known_translation = Vector3f::new(2.0, -1.0, 0.5);
    let known_rotation = UnitQuaternion::from_axis_angle(&Vector3f::z_axis(), PI / 4.0);
    let known_transform = Isometry3::from_parts(
        Translation3::new(known_translation.x, known_translation.y, known_translation.z),
        known_rotation,
    );

    println!("  Known translation: {:?}", known_translation);
    println!("  Known rotation: {:.6} radians", known_rotation.angle());

    // Create source points in a grid
    for x in -2..=2 {
        for y in -2..=2 {
            for z in -1..=1 {
                let point = Point3f::new(x as f32, y as f32, z as f32);
                source.push(point);
                target.push(known_transform * point);
            }
        }
    }

    println!("  Source points: {}", source.len());
    println!("  Target points: {}", target.len());

    let init = Isometry3::identity();
    let result = icp_point_to_point(&source, &target, init, 50, 1e-6, None)?;

    // Compare with known transformation
    let computed_translation = result.transformation.translation.vector;
    let translation_error = (computed_translation - known_translation).magnitude();
    let rotation_error = (result.transformation.rotation.angle() - known_rotation.angle()).abs();

    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final MSE: {:.6}", result.mse);
    println!("  Translation error: {:.6}", translation_error);
    println!("  Rotation error: {:.6} radians", rotation_error);

    Ok(())
}

fn noisy_icp_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create point clouds with noise
    let mut source = PointCloud::new();
    let mut target = PointCloud::new();
    
    let translation = Vector3f::new(1.5, 0.8, 0.3);
    let rotation = UnitQuaternion::from_axis_angle(&Vector3f::y_axis(), 0.2);
    let transform = Isometry3::from_parts(
        Translation3::new(translation.x, translation.y, translation.z),
        rotation,
    );

    println!("  True translation: {:?}", translation);
    println!("  True rotation: {:.6} radians", rotation.angle());

    // Create source points in a spiral pattern
    for i in 0..200 {
        let angle = (i as f32) * 0.1;
        let radius = 2.0 + (i % 20) as f32 * 0.1;
        let source_point = Point3f::new(
            radius * angle.cos(),
            radius * angle.sin(),
            (i % 10) as f32 * 0.2,
        );
        source.push(source_point);
    }

    // Create target points with known transformation + noise
    for point in &source.points {
        let transformed = transform * point;
        // Add Gaussian noise
        let noise = Vector3f::new(
            (rand::random::<f32>() - 0.5) * 0.05,
            (rand::random::<f32>() - 0.5) * 0.05,
            (rand::random::<f32>() - 0.5) * 0.05,
        );
        target.push(transformed + noise);
    }

    println!("  Source points: {}", source.len());
    println!("  Target points: {}", target.len());
    println!("  Noise level: ±0.025 units");

    let init = Isometry3::identity();
    let result = icp_point_to_point(&source, &target, init, 100, 1e-5, None)?;

    let computed_translation = result.transformation.translation.vector;
    let translation_error = (computed_translation - translation).magnitude();
    let rotation_error = (result.transformation.rotation.angle() - rotation.angle()).abs();

    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final MSE: {:.6}", result.mse);
    println!("  Translation error: {:.6}", translation_error);
    println!("  Rotation error: {:.6} radians", rotation_error);

    Ok(())
}

fn convergence_example() -> Result<(), Box<dyn std::error::Error>> {
    // Test different convergence thresholds
    let mut source = PointCloud::new();
    let mut target = PointCloud::new();
    
    // Create point clouds that should converge quickly
    for i in 0..100 {
        let point = Point3f::new(i as f32 * 0.1, (i * 2) as f32 * 0.1, 0.0);
        source.push(point);
        target.push(point + Vector3f::new(0.5, 0.0, 0.0));
    }

    println!("  Source points: {}", source.len());
    println!("  Target points: {}", target.len());

    let init = Isometry3::identity();

    // Test with different convergence thresholds
    let thresholds = [1e-3, 1e-4, 1e-5, 1e-6];
    
    for &threshold in &thresholds {
        let result = icp_point_to_point(&source, &target, init, 50, threshold, None)?;
        println!("  Threshold {:.0e}: {} iterations, MSE: {:.6}, Converged: {}", 
                threshold, result.iterations, result.mse, result.converged);
    }

    // Test with default parameters
    let result = icp_point_to_point_default(&source, &target, init, 50)?;
    println!("  Default params: {} iterations, MSE: {:.6}, Converged: {}", 
            result.iterations, result.mse, result.converged);

    Ok(())
} 