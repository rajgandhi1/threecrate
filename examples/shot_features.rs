//! SHOT (Signature of Histograms of OrienTations) feature extraction example.
//!
//! Demonstrates computing SHOT and USC descriptors and comparing them to FPFH.

use threecrate_core::{PointCloud, NormalPoint3f, Point3f, Vector3f};
use threecrate_algorithms::features::{
    extract_shot_features_with_normals, ShotConfig, ShotVariant, SHOT_DIM, USC_DIM,
    extract_fpfh_features_with_normals, FpfhConfig, FPFH_DIM,
};

fn make_plane(nx: usize, ny: usize, spacing: f32) -> PointCloud<NormalPoint3f> {
    let mut cloud = PointCloud::new();
    for i in 0..nx {
        for j in 0..ny {
            cloud.push(NormalPoint3f {
                position: Point3f::new(i as f32 * spacing, j as f32 * spacing, 0.0),
                normal: Vector3f::new(0.0, 0.0, 1.0),
            });
        }
    }
    cloud
}

fn make_sphere(steps: usize) -> PointCloud<NormalPoint3f> {
    let mut cloud = PointCloud::new();
    for i in 0..steps {
        for j in 0..steps {
            let theta = std::f32::consts::PI * i as f32 / (steps - 1) as f32;
            let phi   = 2.0 * std::f32::consts::PI * j as f32 / steps as f32;
            let x = theta.sin() * phi.cos();
            let y = theta.sin() * phi.sin();
            let z = theta.cos();
            cloud.push(NormalPoint3f {
                position: Point3f::new(x, y, z),
                normal:   Vector3f::new(x, y, z),
            });
        }
    }
    cloud
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SHOT Feature Extraction Examples");
    println!("=================================\n");

    // -------------------------------------------------------------------------
    // Example 1: SHOT on a flat plane
    // -------------------------------------------------------------------------
    println!("1. SHOT descriptors on a 7×7 plane");
    println!("------------------------------------");
    let plane = make_plane(7, 7, 0.1);
    let shot_config = ShotConfig {
        search_radius: 0.25,
        k_neighbors: 8,
        variant: ShotVariant::Standard,
    };
    let shot_descs = extract_shot_features_with_normals(&plane, &shot_config)?;
    println!("  Points          : {}", plane.points.len());
    println!("  Descriptor dim  : {} (expected {})", shot_descs[0].len(), SHOT_DIM);
    let interior_norm: f32 = shot_descs[24].iter().map(|&v| v * v).sum::<f32>().sqrt();
    println!("  L2 norm (centre): {:.4}", interior_norm);

    // -------------------------------------------------------------------------
    // Example 2: USC on the same plane
    // -------------------------------------------------------------------------
    println!("\n2. USC descriptors on the same plane");
    println!("--------------------------------------");
    let usc_config = ShotConfig {
        search_radius: 0.25,
        k_neighbors: 8,
        variant: ShotVariant::UniqueShapeContext,
    };
    let usc_descs = extract_shot_features_with_normals(&plane, &usc_config)?;
    println!("  Descriptor dim  : {} (expected {})", usc_descs[0].len(), USC_DIM);
    let usc_norm: f32 = usc_descs[24].iter().map(|&v| v * v).sum::<f32>().sqrt();
    println!("  L2 norm (centre): {:.4}", usc_norm);

    // -------------------------------------------------------------------------
    // Example 3: Discriminability — plane vs sphere
    // -------------------------------------------------------------------------
    println!("\n3. Plane vs sphere discriminability");
    println!("-------------------------------------");
    let sphere = make_sphere(7);
    let sphere_shot = extract_shot_features_with_normals(&sphere, &shot_config)?;

    // Compare a central plane descriptor against all sphere descriptors
    let plane_mid = &shot_descs[24];
    let min_dist = sphere_shot.iter()
        .map(|sd| l2_distance(plane_mid, sd))
        .fold(f32::INFINITY, f32::min);
    println!("  Min L2 distance (plane→sphere SHOT): {:.4}", min_dist);
    println!("  {} (descriptors are {})",
        if min_dist > 0.05 { "DISCRIMINATIVE" } else { "similar — try larger radius" },
        if min_dist > 0.05 { "different" } else { "very close" }
    );

    // -------------------------------------------------------------------------
    // Example 4: SHOT vs FPFH comparison
    // -------------------------------------------------------------------------
    println!("\n4. SHOT vs FPFH descriptor comparison on a 6×6 plane");
    println!("-------------------------------------------------------");
    let plane2 = make_plane(6, 6, 0.15);

    let fpfh_config = FpfhConfig { search_radius: 0.4, k_neighbors: 8 };
    let fpfh_descs = extract_fpfh_features_with_normals(&plane2, &fpfh_config)?;

    let shot_config2 = ShotConfig { search_radius: 0.4, k_neighbors: 8, variant: ShotVariant::Standard };
    let shot_descs2 = extract_shot_features_with_normals(&plane2, &shot_config2)?;

    println!("  Points : {}", plane2.points.len());
    println!("  FPFH   : {} dims, example norm = {:.4}",
        FPFH_DIM,
        fpfh_descs[0].iter().map(|&v| v * v).sum::<f32>().sqrt());
    println!("  SHOT   : {} dims, example norm = {:.4}",
        SHOT_DIM,
        shot_descs2[0].iter().map(|&v| v * v).sum::<f32>().sqrt());

    println!("\nAll SHOT examples completed successfully.");
    Ok(())
}
