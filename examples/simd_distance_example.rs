//! SIMD Distance Computation Example
//!
//! Demonstrates the SIMD-accelerated nearest-neighbor search added for issue
//! #100.  It shows:
//!
//! 1. How to build a `SimdBruteForceSearch` and query it.
//! 2. A wall-clock timing comparison between the scalar `BruteForceSearch` and
//!    the SIMD variant so you can see the throughput improvement.
//! 3. The `batch_distances_squared` helper for computing all distances in one
//!    SIMD pass.

use std::time::Instant;
use threecrate_algorithms::{
    batch_distances_squared, scalar_distances_squared, BruteForceSearch, SimdBruteForceSearch,
    SoaPoints,
};
use threecrate_core::{NearestNeighborSearch, Point3f};

fn main() {
    println!("=== SIMD Distance Computation Example (#100) ===\n");

    detect_and_print_simd_support();

    // ------------------------------------------------------------------
    // 1. Basic API demonstration
    // ------------------------------------------------------------------
    let dataset: Vec<Point3f> = (0..16)
        .map(|i| Point3f::new(i as f32, (i * 2) as f32, (i * 3) as f32))
        .collect();

    let soa = SoaPoints::from_points(&dataset);
    let query = Point3f::new(5.0, 10.0, 15.0);

    println!("--- Batch distance computation (16 points) ---");
    let mut dist_sq_simd = vec![0.0f32; dataset.len()];
    let mut dist_sq_scalar = vec![0.0f32; dataset.len()];

    batch_distances_squared(&query, &soa, &mut dist_sq_simd);
    scalar_distances_squared(&query, &soa, &mut dist_sq_scalar);

    let max_err = dist_sq_simd
        .iter()
        .zip(&dist_sq_scalar)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("Max SIMD vs scalar error: {max_err:.2e}  (should be ~0)");
    println!("First 4 squared distances: {:?}\n", &dist_sq_simd[..4]);

    // ------------------------------------------------------------------
    // 2. SimdBruteForceSearch vs BruteForceSearch – k-NN
    // ------------------------------------------------------------------
    println!("--- k-NN search comparison ---");
    let searcher = SimdBruteForceSearch::new(&dataset);
    let scalar_searcher = BruteForceSearch::new(&dataset);

    let k = 3;
    let mut simd_knn = searcher.find_k_nearest(&query, k);
    let mut scalar_knn = scalar_searcher.find_k_nearest(&query, k);

    simd_knn.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scalar_knn.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("SIMD  k={k} nearest: {:?}", simd_knn);
    println!("Scalar k={k} nearest: {:?}", scalar_knn);

    let distances_agree = simd_knn
        .iter()
        .zip(&scalar_knn)
        .all(|((_, d1), (_, d2))| (d1 - d2).abs() < 1e-4);
    println!("Results match: {}\n", distances_agree);

    // ------------------------------------------------------------------
    // 3. Radius search
    // ------------------------------------------------------------------
    println!("--- Radius search ---");
    let radius = 10.0f32;
    let mut simd_radius = searcher.find_radius_neighbors(&query, radius);
    let mut scalar_radius = scalar_searcher.find_radius_neighbors(&query, radius);
    simd_radius.sort_by(|a, b| a.0.cmp(&b.0));
    scalar_radius.sort_by(|a, b| a.0.cmp(&b.0));
    println!(
        "Radius={radius}: SIMD found {}, scalar found {}",
        simd_radius.len(),
        scalar_radius.len()
    );
    println!(
        "Counts match: {}\n",
        simd_radius.len() == scalar_radius.len()
    );

    // ------------------------------------------------------------------
    // 4. Throughput benchmark (N = 100 000)
    // ------------------------------------------------------------------
    println!("--- Throughput benchmark (N=100 000, 100 queries) ---");
    let n = 100_000usize;
    let large_pts: Vec<Point3f> = (0..n)
        .map(|i| {
            let f = i as f32;
            Point3f::new(f.sin() * 100.0, f.cos() * 100.0, f * 0.001)
        })
        .collect();

    let large_soa = SoaPoints::from_points(&large_pts);
    let large_simd = SimdBruteForceSearch::new(&large_pts);
    let large_scalar = BruteForceSearch::new(&large_pts);

    let queries: Vec<Point3f> = (0..100)
        .map(|i| Point3f::new(i as f32, i as f32, i as f32))
        .collect();

    // SIMD batch distance
    let t = Instant::now();
    let mut scratch = vec![0.0f32; n];
    for q in &queries {
        batch_distances_squared(q, &large_soa, &mut scratch);
    }
    let simd_batch_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Scalar batch distance
    let t = Instant::now();
    for q in &queries {
        scalar_distances_squared(q, &large_soa, &mut scratch);
    }
    let scalar_batch_ms = t.elapsed().as_secs_f64() * 1000.0;

    // SIMD k-NN (k=10)
    let t = Instant::now();
    for q in &queries {
        let _ = large_simd.find_k_nearest(q, 10);
    }
    let simd_knn_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Scalar k-NN (k=10)
    let t = Instant::now();
    for q in &queries {
        let _ = large_scalar.find_k_nearest(q, 10);
    }
    let scalar_knn_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("  batch_distances_squared  — SIMD: {simd_batch_ms:.2} ms  |  scalar: {scalar_batch_ms:.2} ms  |  speedup: {:.2}x",
        scalar_batch_ms / simd_batch_ms.max(0.001));
    println!("  find_k_nearest (k=10)    — SIMD: {simd_knn_ms:.2} ms  |  scalar: {scalar_knn_ms:.2} ms  |  speedup: {:.2}x",
        scalar_knn_ms / simd_knn_ms.max(0.001));

    println!("\n=== Example completed successfully! ===");
}

fn detect_and_print_simd_support() {
    print!("CPU SIMD support: ");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("AVX2 (8-wide f32)");
        } else if is_x86_feature_detected!("sse4.1") {
            println!("SSE4.1 (4-wide f32)");
        } else if is_x86_feature_detected!("sse2") {
            println!("SSE2 (4-wide f32)");
        } else {
            println!("scalar fallback");
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    println!("scalar fallback (non-x86 target)");
    println!();
}
