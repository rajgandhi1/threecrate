//! Global registration example: FPFH + RANSAC + ICP.
//!
//! Demonstrates the coarse-to-fine registration pipeline on synthetic point clouds
//! with known ground-truth transformations.

use threecrate_core::{PointCloud, Point3f};
use threecrate_algorithms::{
    global_registration, GlobalRegistrationConfig,
    icp_point_to_point,
};
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use std::f32::consts::PI;

fn grid_cloud(nx: usize, ny: usize, nz: usize, spacing: f32) -> PointCloud<Point3f> {
    let mut pts = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                pts.push(Point3f::new(
                    ix as f32 * spacing,
                    iy as f32 * spacing,
                    iz as f32 * spacing,
                ));
            }
        }
    }
    PointCloud { points: pts }
}

fn apply_transform(cloud: &PointCloud<Point3f>, iso: &Isometry3<f32>) -> PointCloud<Point3f> {
    PointCloud { points: cloud.points.iter().map(|p| iso * p).collect() }
}

fn translation_error(a: &Isometry3<f32>, b: &Isometry3<f32>) -> f32 {
    (a.translation.vector - b.translation.vector).magnitude()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Global Registration: FPFH + RANSAC + ICP");
    println!("==========================================\n");

    // -------------------------------------------------------------------------
    // Example 1: Coarse alignment with a large known translation
    // -------------------------------------------------------------------------
    println!("1. Large translation (no good initial guess)");
    println!("--------------------------------------------");

    let target = grid_cloud(5, 5, 5, 1.0);
    let known_t = Isometry3::from_parts(
        Translation3::new(2.0, 1.5, 1.0),
        UnitQuaternion::identity(),
    );
    let source = apply_transform(&target, &known_t);

    let config = GlobalRegistrationConfig {
        ransac_iterations: 2_000,
        distance_threshold: 0.8,
        inlier_ratio: 0.3,
        fpfh_radius: 3.0,
        fpfh_k_neighbors: 8,
        normal_k_neighbors: 8,
        refine_with_icp: true,
        icp_max_iterations: 50,
        icp_distance_threshold: Some(2.0),
    };

    println!("  Source points  : {}", source.points.len());
    println!("  Target points  : {}", target.points.len());
    println!("  Known transform: t=({:.2}, {:.2}, {:.2})",
             known_t.translation.x, known_t.translation.y, known_t.translation.z);

    let result = global_registration(&source, &target, &config)?;
    let t = result.transformation.translation.vector;
    println!("  Found transform: t=({:.3}, {:.3}, {:.3})", t.x, t.y, t.z);
    println!("  Inliers        : {}/{} ({:.1}%)",
             result.inlier_count,
             source.points.len(),
             result.inlier_ratio * 100.0);
    if let Some(ref icp) = result.icp_result {
        println!("  ICP MSE        : {:.6} (converged={})", icp.mse, icp.converged);
    }
    println!("  Translation err: {:.4}", translation_error(&result.transformation, &known_t));

    // -------------------------------------------------------------------------
    // Example 2: Coarse alignment only (no ICP), then refine separately
    // -------------------------------------------------------------------------
    println!("\n2. RANSAC-only → then explicit ICP refinement");
    println!("---------------------------------------------");

    let target2 = grid_cloud(5, 5, 5, 1.0);
    let known_t2 = Isometry3::from_parts(
        Translation3::new(1.0, 0.5, 0.25),
        UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1),
    );
    let source2 = apply_transform(&target2, &known_t2);

    let coarse_config = GlobalRegistrationConfig {
        ransac_iterations: 2_000,
        distance_threshold: 0.8,
        fpfh_radius: 3.0,
        normal_k_neighbors: 8,
        refine_with_icp: false,    // coarse only
        ..Default::default()
    };

    let coarse = global_registration(&source2, &target2, &coarse_config)?;
    let ct = coarse.transformation.translation.vector;
    println!("  Coarse transform: t=({:.3}, {:.3}, {:.3}), inliers={}/{}",
             ct.x, ct.y, ct.z, coarse.inlier_count, source2.points.len());

    // Hand off to ICP
    let icp_result = icp_point_to_point(
        &source2, &target2,
        coarse.transformation,
        100, 1e-6, Some(1.5),
    )?;
    let ft = icp_result.transformation.translation.vector;
    println!("  After ICP      : t=({:.4}, {:.4}, {:.4})", ft.x, ft.y, ft.z);
    println!("  ICP MSE        : {:.6} (converged={})", icp_result.mse, icp_result.converged);
    println!("  Translation err: {:.4}", translation_error(&icp_result.transformation, &known_t2));

    // -------------------------------------------------------------------------
    // Example 3: Config explanation — expected use
    // -------------------------------------------------------------------------
    println!("\n3. GlobalRegistrationConfig fields");
    println!("-----------------------------------");
    let cfg = GlobalRegistrationConfig::default();
    println!("  ransac_iterations     = {} (more = robust, slower)", cfg.ransac_iterations);
    println!("  distance_threshold    = {} (Euclidean inlier cutoff, model units)", cfg.distance_threshold);
    println!("  inlier_ratio          = {} (early exit when this fraction are inliers)", cfg.inlier_ratio);
    println!("  fpfh_radius           = {} (feature extraction support radius)", cfg.fpfh_radius);
    println!("  refine_with_icp       = {} (automatic ICP polish after RANSAC)", cfg.refine_with_icp);
    println!("  icp_max_iterations    = {}", cfg.icp_max_iterations);

    println!("\nAll global registration examples completed successfully.");
    Ok(())
}
