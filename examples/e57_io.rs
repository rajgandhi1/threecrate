//! E57 file I/O example
//!
//! Demonstrates reading and writing E57 point cloud files using threecrate-io.
//! E57 (ASTM E2807) is the ISO standard format for 3D imaging data, widely used
//! by terrestrial laser scanners from Faro, Leica, and Trimble.
//!
//! Run with:
//!   cargo run --bin e57_io --features e57
//!
//! Optional: pass a path to an existing E57 file as the first argument to read it.

use std::env;
use std::path::PathBuf;
use threecrate_core::{Point3f, PointCloud};
use threecrate_io::e57::{E57WriteOptions, RobustE57Reader, RobustE57Writer};
use threecrate_io::{read_point_cloud, write_point_cloud};

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(input_path) = args.get(1) {
        // ── Read an existing E57 file ──────────────────────────────────────────
        println!("Reading E57 file: {input_path}");
        println!("{}", "=".repeat(50));

        match RobustE57Reader::read_point_cloud(input_path) {
            Ok(cloud) => {
                print_cloud_info(&cloud);
            }
            Err(e) => {
                eprintln!("Error reading E57 file: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // ── Round-trip demo with a synthetic point cloud ───────────────────────
        println!("E57 I/O Round-trip Demo");
        println!("{}", "=".repeat(50));

        let cloud = build_demo_cloud();
        println!("Created synthetic cloud with {} points", cloud.len());
        print_cloud_info(&cloud);
        println!();

        // Write using the low-level API with custom options
        let out_path = PathBuf::from(env::temp_dir()).join("threecrate_demo.e57");
        let options = E57WriteOptions::default()
            .with_file_guid("{AABBCCDD-1111-2222-3333-000000000001}")
            .with_cloud_guid("{AABBCCDD-1111-2222-3333-000000000002}");

        println!("Writing E57 file: {}", out_path.display());
        RobustE57Writer::write_point_cloud(&cloud, &out_path, &options)
            .expect("Failed to write E57 file");
        println!("Write successful.");
        println!();

        // Read back using the high-level registry API
        println!("Reading back via registry API...");
        let loaded = read_point_cloud(&out_path).expect("Failed to read E57 file");
        println!("Loaded {} points.", loaded.len());
        assert_eq!(
            loaded.len(),
            cloud.len(),
            "Point count mismatch after round-trip"
        );
        println!("Round-trip OK — point counts match.");

        // Round-trip write via registry (high-level)
        let out2 = PathBuf::from(env::temp_dir()).join("threecrate_demo2.e57");
        write_point_cloud(&loaded, &out2).expect("Failed to write via registry");
        println!(
            "Registry write successful: {}",
            out2.display()
        );

        let _ = std::fs::remove_file(&out_path);
        let _ = std::fs::remove_file(&out2);
    }
}

fn build_demo_cloud() -> PointCloud<Point3f> {
    let mut cloud = PointCloud::new();
    // Simple grid of 100 points
    for i in 0..10i32 {
        for j in 0..10i32 {
            cloud.push(Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
        }
    }
    cloud
}

fn print_cloud_info(cloud: &PointCloud<Point3f>) {
    println!("  Total points: {}", cloud.len());

    if cloud.is_empty() {
        return;
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for p in cloud.iter() {
        min[0] = min[0].min(p.x);
        min[1] = min[1].min(p.y);
        min[2] = min[2].min(p.z);
        max[0] = max[0].max(p.x);
        max[1] = max[1].max(p.y);
        max[2] = max[2].max(p.z);
    }

    println!(
        "  Bounding box  X: [{:.3}, {:.3}]  Y: [{:.3}, {:.3}]  Z: [{:.3}, {:.3}]",
        min[0], max[0], min[1], max[1], min[2], max[2]
    );

    println!("  First 5 points:");
    for (i, p) in cloud.iter().take(5).enumerate() {
        println!("    [{i}] ({:.4}, {:.4}, {:.4})", p.x, p.y, p.z);
    }
}
