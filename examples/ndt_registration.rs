//! NDT (Normal Distributions Transform) registration example
//!
//! Demonstrates using NDT to align point clouds and compares results with ICP.

use threecrate_core::{PointCloud, Point3f};
use threecrate_algorithms::{
    ndt_registration, ndt_registration_default, NdtConfig,
    icp_point_to_point_default,
};
use nalgebra::{Isometry3, UnitQuaternion, Translation3};
use std::f32::consts::PI;

fn make_grid_cloud(nx: usize, ny: usize, nz: usize, scale: f32) -> PointCloud<Point3f> {
    let mut points = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                points.push(Point3f::new(
                    ix as f32 * scale,
                    iy as f32 * scale,
                    iz as f32 * scale,
                ));
            }
        }
    }
    PointCloud { points }
}

fn apply_transform(cloud: &PointCloud<Point3f>, iso: &Isometry3<f32>) -> PointCloud<Point3f> {
    PointCloud {
        points: cloud.points.iter().map(|p| iso * p).collect(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NDT Registration Examples");
    println!("=========================\n");

    // -------------------------------------------------------------------------
    // Example 1: NDT with a known translation
    // -------------------------------------------------------------------------
    println!("1. NDT — Known Translation");
    println!("--------------------------");

    let target = make_grid_cloud(8, 8, 8, 1.0);
    let known_translation = Isometry3::from_parts(
        Translation3::new(0.5, 0.3, 0.2),
        UnitQuaternion::identity(),
    );
    let source = apply_transform(&target, &known_translation);

    println!("  Target points : {}", target.points.len());
    println!("  Source points : {}", source.points.len());
    println!("  Known translation: ({:.2}, {:.2}, {:.2})",
             known_translation.translation.x,
             known_translation.translation.y,
             known_translation.translation.z);

    let config = NdtConfig {
        resolution: 2.0,
        step_size: 0.5,
        max_iterations: 50,
        epsilon: 1e-5,
        min_points_per_voxel: 3,
    };

    let result = ndt_registration(&source, &target, Isometry3::identity(), &config)?;

    let t = result.transformation.translation.vector;
    println!("  Result translation: ({:.4}, {:.4}, {:.4})", t.x, t.y, t.z);
    println!("  Score      : {:.4}", result.score);
    println!("  Iterations : {}", result.iterations);
    println!("  Converged  : {}", result.converged);

    // -------------------------------------------------------------------------
    // Example 2: NDT with a small rotation
    // -------------------------------------------------------------------------
    println!("\n2. NDT — Small Rotation");
    println!("-----------------------");

    let target2 = make_grid_cloud(6, 6, 6, 1.0);
    let known_rotation = Isometry3::from_parts(
        Translation3::new(0.0, 0.0, 0.0),
        UnitQuaternion::from_euler_angles(0.1, 0.05, 0.08),
    );
    let source2 = apply_transform(&target2, &known_rotation);

    let config2 = NdtConfig {
        resolution: 2.0,
        step_size: 0.2,
        max_iterations: 60,
        epsilon: 1e-5,
        min_points_per_voxel: 3,
    };

    let result2 = ndt_registration(&source2, &target2, Isometry3::identity(), &config2)?;

    let angle = result2.transformation.rotation.angle();
    println!("  Score      : {:.4}", result2.score);
    println!("  Iterations : {}", result2.iterations);
    println!("  Converged  : {}", result2.converged);
    println!("  Rotation angle recovered: {:.4} rad (expected ~{:.4} rad)",
             angle,
             known_rotation.rotation.angle());

    // -------------------------------------------------------------------------
    // Example 3: Default config convenience wrapper
    // -------------------------------------------------------------------------
    println!("\n3. NDT — Default Config");
    println!("-----------------------");

    let target3 = make_grid_cloud(5, 5, 5, 1.5);
    let source3 = apply_transform(
        &target3,
        &Isometry3::from_parts(Translation3::new(0.2, 0.1, 0.3), UnitQuaternion::identity()),
    );

    // Use a config that produces enough voxels
    let config3 = NdtConfig {
        resolution: 3.0,
        min_points_per_voxel: 2,
        ..NdtConfig::default()
    };
    let result3 = ndt_registration(&source3, &target3, Isometry3::identity(), &config3)?;
    println!("  Score      : {:.4}", result3.score);
    println!("  Iterations : {}", result3.iterations);
    println!("  Converged  : {}", result3.converged);

    // -------------------------------------------------------------------------
    // Example 4: Compare NDT vs ICP on the same data
    // -------------------------------------------------------------------------
    println!("\n4. NDT vs ICP Comparison");
    println!("------------------------");

    let target4 = make_grid_cloud(7, 7, 7, 1.0);
    let transform4 = Isometry3::from_parts(
        Translation3::new(0.4, 0.2, 0.15),
        UnitQuaternion::identity(),
    );
    let source4 = apply_transform(&target4, &transform4);

    // NDT
    let ndt_config = NdtConfig {
        resolution: 2.0,
        step_size: 0.5,
        max_iterations: 50,
        epsilon: 1e-5,
        min_points_per_voxel: 3,
    };
    let ndt_result = ndt_registration(&source4, &target4, Isometry3::identity(), &ndt_config)?;

    // ICP
    let icp_result = icp_point_to_point_default(&source4, &target4)?;

    let ndt_t = ndt_result.transformation.translation.vector;
    let icp_t = icp_result.transformation.translation.vector;
    let known_t = transform4.translation.vector;

    println!("  Known translation : ({:.3}, {:.3}, {:.3})", known_t.x, known_t.y, known_t.z);
    println!("  NDT result        : ({:.4}, {:.4}, {:.4})  iters={} score={:.2}",
             ndt_t.x, ndt_t.y, ndt_t.z, ndt_result.iterations, ndt_result.score);
    println!("  ICP result        : ({:.4}, {:.4}, {:.4})  iters={} mse={:.6}",
             icp_t.x, icp_t.y, icp_t.z, icp_result.iterations, icp_result.mse);

    println!("\nAll NDT examples completed successfully.");
    Ok(())
}
