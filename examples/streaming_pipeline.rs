//! Streaming Point Cloud Processing Pipeline
//!
//! Demonstrates the chunk-based streaming architecture added for issue #101.
//!
//! Key properties of the streaming approach:
//!  - Only `chunk_size` points reside in RAM at any time while reading.
//!  - Pipeline state (e.g. the voxel map) is bounded by the number of
//!    *unique voxels*, not the total point count.
//!  - The same pipeline stages work over file iterators (PLY, XYZ/CSV) or
//!    over in-memory point clouds via `cloud_as_stream`.

use std::time::Instant;
use threecrate_algorithms::streaming::{
    cloud_as_stream, run_pipeline, run_pipeline_with_options, RunOptions, StreamingCollector,
    StreamingPipeline, StreamingStatistics, StreamingVoxelFilter, StreamingVoxelFilterConfig,
};
use threecrate_core::{Point3f, PointCloud};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Point Cloud Pipeline Example (#101) ===\n");

    // ------------------------------------------------------------------
    // 1. Generate a synthetic "large" dataset.
    //    In production this would come from read_point_cloud_iter("file.ply").
    // ------------------------------------------------------------------
    const N: usize = 500_000;
    let cloud = make_synthetic_cloud(N);
    println!("Synthetic dataset: {N} points");
    println!("In-memory size: ~{:.1} MB\n", (N * 12) as f64 / 1e6);

    // ------------------------------------------------------------------
    // 2. Statistics collection — O(1) memory, single pass.
    // ------------------------------------------------------------------
    println!("--- Pass 1: Streaming statistics (chunk_size=10 000) ---");
    let mut stats_pipe = StreamingStatistics::new();
    let t = Instant::now();
    let run_stats = run_pipeline(&mut stats_pipe, cloud_as_stream(&cloud), 10_000)?;
    let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;

    let stats = stats_pipe.finalize()?;
    println!("  Points processed : {}", stats.point_count);
    println!(
        "  Bounding box     : ({:.2},{:.2},{:.2}) – ({:.2},{:.2},{:.2})",
        stats.min.x, stats.min.y, stats.min.z, stats.max.x, stats.max.y, stats.max.z,
    );
    println!(
        "  Mean position    : ({:.2},{:.2},{:.2})",
        stats.mean.x, stats.mean.y, stats.mean.z
    );
    println!(
        "  Chunks processed : {}  ({:.2} ms total)",
        run_stats.chunks_processed, elapsed_ms
    );
    println!(
        "  Throughput       : {:.1} Mpts/s\n",
        stats.point_count as f64 / elapsed_ms / 1000.0
    );

    // ------------------------------------------------------------------
    // 3. Streaming voxel-grid downsampling.
    //    Memory is proportional to occupied voxels, NOT to total point count.
    // ------------------------------------------------------------------
    println!("--- Pass 2: Streaming voxel filter (voxel_size=0.5, chunk_size=20 000) ---");
    let config = StreamingVoxelFilterConfig { voxel_size: 0.5 };
    let mut voxel_filter = StreamingVoxelFilter::new(config);
    let t = Instant::now();
    let filter_stats = run_pipeline(&mut voxel_filter, cloud_as_stream(&cloud), 20_000)?;
    let filter_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  Occupied voxels during filtering : {}",
        voxel_filter.voxel_count()
    );
    println!(
        "  Pipeline memory use              : ~{:.1} KB",
        voxel_filter.memory_bytes() as f64 / 1024.0
    );

    let downsampled = voxel_filter.finalize()?;
    let reduction = 100.0 * (1.0 - downsampled.len() as f64 / N as f64);
    println!("  Input points  : {N}");
    println!("  Output points : {}", downsampled.len());
    println!("  Reduction     : {:.1}%", reduction);
    println!(
        "  Chunks        : {}  ({:.2} ms total)\n",
        filter_stats.chunks_processed, filter_ms
    );

    // ------------------------------------------------------------------
    // 4. Chunk-size sensitivity.
    //    Smaller chunks use less peak RAM but more function-call overhead.
    // ------------------------------------------------------------------
    println!("--- Chunk-size sensitivity (voxel_size=1.0) ---");
    for &cs in &[500usize, 5_000, 50_000, 500_000] {
        let config = StreamingVoxelFilterConfig { voxel_size: 1.0 };
        let mut f = StreamingVoxelFilter::new(config);
        let t = Instant::now();
        let s = run_pipeline(&mut f, cloud_as_stream(&cloud), cs)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let out = f.finalize()?;
        println!(
            "  chunk_size={cs:>7}  chunks={:>4}  out_pts={:>6}  {:.2} ms",
            s.chunks_processed,
            out.len(),
            ms
        );
    }

    // ------------------------------------------------------------------
    // 5. Error handling: skip bad points and continue.
    // ------------------------------------------------------------------
    println!("\n--- Error handling: skip_errors mode ---");
    use threecrate_core::Error;
    let mixed_source: Vec<threecrate_core::Result<Point3f>> = vec![
        Ok(Point3f::new(0.0, 0.0, 0.0)),
        Err(Error::InvalidData("corrupted record".into())),
        Ok(Point3f::new(1.0, 0.0, 0.0)),
        Err(Error::InvalidData("another bad record".into())),
        Ok(Point3f::new(2.0, 0.0, 0.0)),
    ];
    let mut collector = StreamingCollector::new();
    let mixed_stats = run_pipeline_with_options(
        &mut collector,
        mixed_source.into_iter(),
        10,
        &RunOptions { skip_errors: true },
    )?;
    let recovered = collector.finalize()?;
    println!(
        "  Source: 5 items (3 ok, 2 errors)  →  recovered {} points, skipped {} errors",
        recovered.len(),
        mixed_stats.errors_skipped
    );

    // ------------------------------------------------------------------
    // 6. Show that a file-backed iterator plugs in identically.
    //    (PLY / XYZ / CSV via threecrate_io::read_point_cloud_iter)
    // ------------------------------------------------------------------
    println!("\n--- File-backed streaming (PLY) ---");
    let tmp_path = std::env::temp_dir().join("streaming_demo.ply");
    write_demo_ply(&tmp_path, 1_000)?;
    println!("  Wrote demo PLY: {} (1 000 points)", tmp_path.display());

    match threecrate_io::read_point_cloud_iter(&tmp_path, Some(200)) {
        Ok(iter) => {
            let mut file_stats = StreamingStatistics::new();
            let rs = run_pipeline(&mut file_stats, iter, 200)?;
            let s = file_stats.finalize()?;
            println!(
                "  Streamed {} points from PLY in {} chunks",
                s.point_count, rs.chunks_processed
            );
            println!(
                "  Bounding box: ({:.2},{:.2},{:.2}) – ({:.2},{:.2},{:.2})",
                s.min.x, s.min.y, s.min.z, s.max.x, s.max.y, s.max.z,
            );
        }
        Err(e) => println!("  Note: PLY streaming unavailable – {e}"),
    }

    let _ = std::fs::remove_file(&tmp_path);

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic point cloud spread over a unit cube.
fn make_synthetic_cloud(n: usize) -> PointCloud<Point3f> {
    // Deterministic Halton-like sequence to avoid requiring `rand`.
    let pts: Vec<Point3f> = (0..n)
        .map(|i| {
            let t = i as f32;
            let x = (t * 0.6180339887).fract() * 10.0;
            let y = (t * 0.7548776662).fract() * 10.0;
            let z = (t * 0.5698402910).fract() * 10.0;
            Point3f::new(x, y, z)
        })
        .collect();
    PointCloud::from_points(pts)
}

/// Write a minimal ASCII PLY file with `n` points for the file-streaming demo.
fn write_demo_ply(path: &std::path::Path, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "ply")?;
    writeln!(f, "format ascii 1.0")?;
    writeln!(f, "element vertex {n}")?;
    writeln!(f, "property float x")?;
    writeln!(f, "property float y")?;
    writeln!(f, "property float z")?;
    writeln!(f, "end_header")?;
    for i in 0..n {
        let t = i as f32;
        let x = (t * 0.6180339887).fract();
        let y = (t * 0.7548776662).fract();
        let z = (t * 0.5698402910).fract();
        writeln!(f, "{x:.6} {y:.6} {z:.6}")?;
    }
    Ok(())
}
