//! Normal Estimation Example
//!
//! This example demonstrates the enhanced normal estimation functionality
//! including k-NN, radius-based search, and orientation consistency.

use threecrate_core::{PointCloud, Point3f};
use threecrate_algorithms::{
    estimate_normals, 
    estimate_normals_with_config, 
    estimate_normals_radius,
    NormalEstimationConfig
};
use std::time::Instant;

fn main() -> threecrate_core::Result<()> {
    println!("🚀 Enhanced Normal Estimation Example");
    println!("=====================================");

    // Create a complex point cloud with multiple surfaces
    let cloud = create_sample_point_cloud();
    println!("✓ Created sample point cloud with {} points", cloud.len());

    // 1. Basic k-NN normal estimation
    println!("\n1. Basic k-NN Normal Estimation");
    println!("--------------------------------");
    let start = Instant::now();
    let normals_knn = estimate_normals(&cloud, 10)?;
    let knn_time = start.elapsed();
    println!("✓ Estimated normals using k-NN in {:?}", knn_time);
    println!("  - Result: {} points with normals", normals_knn.len());

    // 2. Radius-based normal estimation
    println!("\n2. Radius-based Normal Estimation");
    println!("----------------------------------");
    let start = Instant::now();
    let normals_radius = estimate_normals_radius(&cloud, 0.3, true)?;
    let radius_time = start.elapsed();
    println!("✓ Estimated normals using radius search in {:?}", radius_time);
    println!("  - Result: {} points with normals", normals_radius.len());

    // 3. Advanced configuration
    println!("\n3. Advanced Configuration");
    println!("-------------------------");
    let config = NormalEstimationConfig {
        k_neighbors: 15,
        radius: Some(0.25),
        consistent_orientation: true,
        viewpoint: Some(Point3f::new(0.0, 0.0, 5.0)),
    };
    
    let start = Instant::now();
    let normals_advanced = estimate_normals_with_config(&cloud, &config)?;
    let advanced_time = start.elapsed();
    println!("✓ Estimated normals with advanced config in {:?}", advanced_time);
    println!("  - Result: {} points with normals", normals_advanced.len());

    // 4. Compare results
    println!("\n4. Results Comparison");
    println!("--------------------");
    compare_normal_quality(&normals_knn, "k-NN");
    compare_normal_quality(&normals_radius, "Radius-based");
    compare_normal_quality(&normals_advanced, "Advanced config");

    // 5. Test with different shapes
    println!("\n5. Shape-specific Tests");
    println!("----------------------");
    test_planar_surface();
    test_cylindrical_surface();
    test_spherical_surface();

    println!("\n✅ All normal estimation examples completed successfully!");
    Ok(())
}

fn create_sample_point_cloud() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    
    // Add planar surface (XY plane)
    for i in 0..20 {
        for j in 0..20 {
            let x = (i as f32) * 0.1 - 1.0;
            let y = (j as f32) * 0.1 - 1.0;
            let z = 0.0;
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    // Add cylindrical surface
    for i in 0..15 {
        for j in 0..10 {
            let angle = (i as f32) * 0.4;
            let height = (j as f32) * 0.2 - 1.0;
            let x = 2.0 + angle.cos();
            let y = angle.sin();
            let z = height;
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    // Add spherical surface
    for i in 0..20 {
        for j in 0..10 {
            let phi = (i as f32) * 0.3;
            let theta = (j as f32) * 0.2;
            let x = -2.0 + phi.cos() * theta.cos();
            let y = phi.sin() * theta.cos();
            let z = theta.sin();
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    cloud
}

fn compare_normal_quality(normals: &PointCloud<threecrate_core::NormalPoint3f>, method: &str) {
    let mut unit_vector_count = 0;
    let mut z_direction_count = 0;
    
    for point in normals.iter() {
        let magnitude = point.normal.magnitude();
        if (magnitude - 1.0).abs() < 0.1 {
            unit_vector_count += 1;
        }
        
        if point.normal.z.abs() > 0.8 {
            z_direction_count += 1;
        }
    }
    
    let unit_percentage = (unit_vector_count as f32 / normals.len() as f32) * 100.0;
    let z_percentage = (z_direction_count as f32 / normals.len() as f32) * 100.0;
    
    println!("  {}: {:.1}% unit vectors, {:.1}% in Z direction", 
             method, unit_percentage, z_percentage);
}

fn test_planar_surface() {
    println!("  Testing planar surface...");
    let mut cloud = PointCloud::new();
    
    // Create a planar surface
    for i in 0..15 {
        for j in 0..15 {
            let x = (i as f32) * 0.1;
            let y = (j as f32) * 0.1;
            let z = 0.0;
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    let normals = estimate_normals(&cloud, 8).unwrap();
    let mut z_direction_count = 0;
    
    for point in normals.iter() {
        if point.normal.z.abs() > 0.8 {
            z_direction_count += 1;
        }
    }
    
    let percentage = (z_direction_count as f32 / normals.len() as f32) * 100.0;
    println!("    Planar surface: {:.1}% normals in Z direction", percentage);
}

fn test_cylindrical_surface() {
    println!("  Testing cylindrical surface...");
    let mut cloud = PointCloud::new();
    
    // Create a cylindrical surface
    for i in 0..12 {
        for j in 0..8 {
            let angle = (i as f32) * 0.5;
            let height = (j as f32) * 0.25 - 1.0;
            let x = angle.cos();
            let y = angle.sin();
            let z = height;
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    let normals = estimate_normals(&cloud, 8).unwrap();
    let mut perpendicular_count = 0;
    
    for point in normals.iter() {
        if point.normal.z.abs() < 0.8 {
            perpendicular_count += 1;
        }
    }
    
    let percentage = (perpendicular_count as f32 / normals.len() as f32) * 100.0;
    println!("    Cylindrical surface: {:.1}% normals perpendicular to Z", percentage);
}

fn test_spherical_surface() {
    println!("  Testing spherical surface...");
    let mut cloud = PointCloud::new();
    
    // Create a spherical surface
    for i in 0..15 {
        for j in 0..8 {
            let phi = (i as f32) * 0.4;
            let theta = (j as f32) * 0.25;
            let x = phi.cos() * theta.cos();
            let y = phi.sin() * theta.cos();
            let z = theta.sin();
            cloud.push(Point3f::new(x, y, z));
        }
    }
    
    let normals = estimate_normals(&cloud, 8).unwrap();
    let mut outward_count = 0;
    
    for point in normals.iter() {
        let to_center = -point.position.coords.normalize();
        let dot_product = point.normal.dot(&to_center);
        if dot_product.abs() > 0.5 {
            outward_count += 1;
        }
    }
    
    let percentage = (outward_count as f32 / normals.len() as f32) * 100.0;
    println!("    Spherical surface: {:.1}% normals pointing outward", percentage);
} 