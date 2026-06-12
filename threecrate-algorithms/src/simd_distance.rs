//! SIMD-accelerated distance computations for nearest-neighbor search.
//!
//! # Strategy
//!
//! Points are stored in **Structure-of-Arrays (SoA)** layout — three separate
//! `f32` slices for X, Y, and Z — so that SIMD lanes map directly onto
//! multiple points at once.
//!
//! ```text
//! AoS (cache-unfriendly for distance):   [(x0,y0,z0), (x1,y1,z1), …]
//! SoA (SIMD-friendly):                   xs=[x0,x1,…]  ys=[y0,y1,…]  zs=[z0,z1,…]
//! ```
//!
//! Given a query `q = (qx, qy, qz)` the squared distance to point `i` is:
//!
//! ```text
//! d²ᵢ = (xᵢ−qx)² + (yᵢ−qy)² + (zᵢ−qz)²
//! ```
//!
//! SIMD processes 4 (SSE2) or 8 (AVX2) distances per instruction cycle.
//!
//! # Dispatch
//!
//! ```text
//! has AVX2?  → avx2_distances_squared  (8-wide f32)
//! else SSE2? → sse2_distances_squared  (4-wide f32, always present on x86-64)
//! else       → scalar_distances_squared (portable fallback)
//! ```
//!
//! The dispatch is resolved **at runtime** using `is_x86_feature_detected!`, so
//! the same binary runs correctly on all hardware while using the widest SIMD
//! available.

use std::cmp::Ordering;
use threecrate_core::{NearestNeighborSearch, Point3f};

// ---------------------------------------------------------------------------
// SoA point store
// ---------------------------------------------------------------------------

/// Point cloud stored in Structure-of-Arrays format for SIMD-friendly access.
///
/// ```
/// # use threecrate_algorithms::SoaPoints;
/// # use threecrate_core::Point3f;
/// let pts = vec![Point3f::new(1.0, 2.0, 3.0), Point3f::new(4.0, 5.0, 6.0)];
/// let soa = SoaPoints::from_points(&pts);
/// assert_eq!(soa.xs(), &[1.0, 4.0]);
/// ```
#[derive(Debug, Clone)]
pub struct SoaPoints {
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
}

impl SoaPoints {
    /// Build an SoA store from an AoS slice.
    pub fn from_points(points: &[Point3f]) -> Self {
        let mut xs = Vec::with_capacity(points.len());
        let mut ys = Vec::with_capacity(points.len());
        let mut zs = Vec::with_capacity(points.len());
        for p in points {
            xs.push(p.x);
            ys.push(p.y);
            zs.push(p.z);
        }
        Self { xs, ys, zs }
    }

    /// Number of stored points.
    #[inline]
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Returns `true` if there are no stored points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// X coordinates slice.
    #[inline]
    pub fn xs(&self) -> &[f32] {
        &self.xs
    }

    /// Y coordinates slice.
    #[inline]
    pub fn ys(&self) -> &[f32] {
        &self.ys
    }

    /// Z coordinates slice.
    #[inline]
    pub fn zs(&self) -> &[f32] {
        &self.zs
    }
}

// ---------------------------------------------------------------------------
// Public batch-distance API
// ---------------------------------------------------------------------------

/// Compute squared Euclidean distances from `query` to every point in `pts`.
///
/// Results are written into `out`, which must have the same length as `pts`.
/// Uses the widest SIMD available (AVX2 → SSE2 → scalar).
pub fn batch_distances_squared(query: &Point3f, pts: &SoaPoints, out: &mut [f32]) {
    debug_assert_eq!(out.len(), pts.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature was detected at runtime.
            return unsafe { avx2_distances_squared(query, pts, out) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { sse2_distances_squared(query, pts, out) };
        }
    }

    scalar_distances_squared(query, pts, out);
}

/// Portable, vectorisation-friendly fallback (also the reference implementation).
///
/// This loop is written so that LLVM can auto-vectorise it on any target.
#[inline]
pub fn scalar_distances_squared(query: &Point3f, pts: &SoaPoints, out: &mut [f32]) {
    let (qx, qy, qz) = (query.x, query.y, query.z);
    let n = pts.len();
    let xs = pts.xs();
    let ys = pts.ys();
    let zs = pts.zs();
    for i in 0..n {
        let dx = xs[i] - qx;
        let dy = ys[i] - qy;
        let dz = zs[i] - qz;
        out[i] = dx * dx + dy * dy + dz * dz;
    }
}

// ---------------------------------------------------------------------------
// SSE2 implementation (4-wide, always available on x86-64)
// ---------------------------------------------------------------------------

/// Compute distances using SSE2 (4 × f32 per cycle).
///
/// # Safety
/// Caller must ensure the `sse2` CPU feature is present.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sse2_distances_squared(query: &Point3f, pts: &SoaPoints, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = pts.len();
    let xs = pts.xs();
    let ys = pts.ys();
    let zs = pts.zs();

    let qx_v = _mm_set1_ps(query.x);
    let qy_v = _mm_set1_ps(query.y);
    let qz_v = _mm_set1_ps(query.z);

    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let base = c * 4;
        let xs_v = _mm_loadu_ps(xs.as_ptr().add(base));
        let ys_v = _mm_loadu_ps(ys.as_ptr().add(base));
        let zs_v = _mm_loadu_ps(zs.as_ptr().add(base));

        let dx = _mm_sub_ps(xs_v, qx_v);
        let dy = _mm_sub_ps(ys_v, qy_v);
        let dz = _mm_sub_ps(zs_v, qz_v);

        let d2 = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)),
            _mm_mul_ps(dz, dz),
        );

        _mm_storeu_ps(out.as_mut_ptr().add(base), d2);
    }

    // Handle remainder with scalar code.
    let rem_start = chunks * 4;
    scalar_remainder(query, xs, ys, zs, out, rem_start, remainder);
}

// ---------------------------------------------------------------------------
// AVX2 implementation (8-wide)
// ---------------------------------------------------------------------------

/// Compute distances using AVX2 (8 × f32 per cycle).
///
/// # Safety
/// Caller must ensure the `avx2` CPU feature is present.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn avx2_distances_squared(query: &Point3f, pts: &SoaPoints, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = pts.len();
    let xs = pts.xs();
    let ys = pts.ys();
    let zs = pts.zs();

    let qx_v = _mm256_set1_ps(query.x);
    let qy_v = _mm256_set1_ps(query.y);
    let qz_v = _mm256_set1_ps(query.z);

    let chunks = n / 8;
    let remainder_start = chunks * 8;
    let remainder = n - remainder_start;

    for c in 0..chunks {
        let base = c * 8;
        let xs_v = _mm256_loadu_ps(xs.as_ptr().add(base));
        let ys_v = _mm256_loadu_ps(ys.as_ptr().add(base));
        let zs_v = _mm256_loadu_ps(zs.as_ptr().add(base));

        let dx = _mm256_sub_ps(xs_v, qx_v);
        let dy = _mm256_sub_ps(ys_v, qy_v);
        let dz = _mm256_sub_ps(zs_v, qz_v);

        let d2 = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
            _mm256_mul_ps(dz, dz),
        );

        _mm256_storeu_ps(out.as_mut_ptr().add(base), d2);
    }

    // Use SSE2 for the leftover 4-element block (if any), then scalar for the rest.
    let mut rem = remainder;
    let mut rem_base = remainder_start;

    if rem >= 4 {
        // We already ensured avx2 implies sse2 on x86-64, so this is safe.
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        let qx_s = _mm_set1_ps(query.x);
        let qy_s = _mm_set1_ps(query.y);
        let qz_s = _mm_set1_ps(query.z);

        let xs_v = _mm_loadu_ps(xs.as_ptr().add(rem_base));
        let ys_v = _mm_loadu_ps(ys.as_ptr().add(rem_base));
        let zs_v = _mm_loadu_ps(zs.as_ptr().add(rem_base));

        let dx = _mm_sub_ps(xs_v, qx_s);
        let dy = _mm_sub_ps(ys_v, qy_s);
        let dz = _mm_sub_ps(zs_v, qz_s);

        let d2 = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)),
            _mm_mul_ps(dz, dz),
        );
        _mm_storeu_ps(out.as_mut_ptr().add(rem_base), d2);

        rem_base += 4;
        rem -= 4;
    }

    scalar_remainder(query, xs, ys, zs, out, rem_base, rem);
}

/// Scalar tail processing for SIMD functions.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    allow(dead_code)
)]
#[inline(always)]
fn scalar_remainder(
    query: &Point3f,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    out: &mut [f32],
    start: usize,
    count: usize,
) {
    let (qx, qy, qz) = (query.x, query.y, query.z);
    for i in 0..count {
        let idx = start + i;
        let dx = xs[idx] - qx;
        let dy = ys[idx] - qy;
        let dz = zs[idx] - qz;
        out[idx] = dx * dx + dy * dy + dz * dz;
    }
}

// ---------------------------------------------------------------------------
// SimdBruteForceSearch
// ---------------------------------------------------------------------------

/// Brute-force nearest-neighbor search with SIMD-accelerated distance computation.
///
/// All N squared distances are computed in a single SIMD-vectorised pass before
/// selection, which is more cache-friendly than repeated point-by-point comparison
/// and exploits the full width of the CPU's SIMD units.
///
/// | Method            | Complexity | Notes                                |
/// |-------------------|------------|--------------------------------------|
/// | `find_k_nearest`  | O(N + k log k) | compute-all-then-partial-select  |
/// | `find_radius_neighbors` | O(N) | compute-all-then-filter           |
///
/// # Example
/// ```
/// # use threecrate_algorithms::SimdBruteForceSearch;
/// # use threecrate_core::{Point3f, NearestNeighborSearch};
/// let pts = vec![Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 0.0, 0.0)];
/// let searcher = SimdBruteForceSearch::new(&pts);
/// let result = searcher.find_k_nearest(&Point3f::new(0.1, 0.0, 0.0), 1);
/// assert_eq!(result[0].0, 0); // index of nearest point
/// ```
pub struct SimdBruteForceSearch {
    soa: SoaPoints,
}

impl SimdBruteForceSearch {
    /// Construct a new searcher from a point slice (O(N) time and space).
    pub fn new(points: &[Point3f]) -> Self {
        Self {
            soa: SoaPoints::from_points(points),
        }
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.soa.len()
    }

    /// Returns `true` if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.soa.is_empty()
    }

    /// Return the SoA representation (useful for benchmarks / inspection).
    pub fn soa(&self) -> &SoaPoints {
        &self.soa
    }
}

impl NearestNeighborSearch for SimdBruteForceSearch {
    fn find_k_nearest(&self, query: &Point3f, k: usize) -> Vec<(usize, f32)> {
        if k == 0 || self.soa.is_empty() {
            return Vec::new();
        }

        let n = self.soa.len();
        let k = k.min(n);

        // ---- 1. Compute all squared distances in one SIMD pass ----
        let mut dist_sq = vec![0.0f32; n];
        batch_distances_squared(query, &self.soa, &mut dist_sq);

        // ---- 2. Partial sort: find the k smallest using a max-heap ----
        let mut heap: std::collections::BinaryHeap<DistEntry> =
            std::collections::BinaryHeap::with_capacity(k + 1);

        for (idx, &d2) in dist_sq.iter().enumerate() {
            if heap.len() < k {
                heap.push(DistEntry {
                    dist_sq: d2,
                    index: idx,
                });
            } else if let Some(farthest) = heap.peek() {
                if d2 < farthest.dist_sq {
                    heap.pop();
                    heap.push(DistEntry {
                        dist_sq: d2,
                        index: idx,
                    });
                }
            }
        }

        // ---- 3. Extract and sort ascending by distance ----
        let mut result: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|e| (e.index, e.dist_sq.sqrt()))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result
    }

    fn find_radius_neighbors(&self, query: &Point3f, radius: f32) -> Vec<(usize, f32)> {
        if radius <= 0.0 || self.soa.is_empty() {
            return Vec::new();
        }

        let n = self.soa.len();
        let radius_sq = radius * radius;

        // ---- 1. Compute all squared distances ----
        let mut dist_sq = vec![0.0f32; n];
        batch_distances_squared(query, &self.soa, &mut dist_sq);

        // ---- 2. Filter and convert ----
        let mut result: Vec<(usize, f32)> = dist_sq
            .iter()
            .enumerate()
            .filter_map(|(idx, &d2)| {
                if d2 <= radius_sq {
                    Some((idx, d2.sqrt()))
                } else {
                    None
                }
            })
            .collect();

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result
    }
}

// ---------------------------------------------------------------------------
// Internal helper: max-heap entry ordered by dist_sq (largest = top of heap)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct DistEntry {
    dist_sq: f32,
    index: usize,
}

impl Eq for DistEntry {}

impl PartialOrd for DistEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Larger dist_sq → higher priority in max-heap → root = farthest of the k candidates.
        // `total_cmp` handles NaN consistently and avoids floating-point comparison UB.
        self.dist_sq
            .total_cmp(&other.dist_sq)
            .then(self.index.cmp(&other.index))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::Point3f;

    fn cube_points() -> Vec<Point3f> {
        vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0),
            Point3f::new(1.0, 1.0, 0.0),
            Point3f::new(1.0, 0.0, 1.0),
            Point3f::new(0.0, 1.0, 1.0),
            Point3f::new(1.0, 1.0, 1.0),
        ]
    }

    // ---- SoaPoints -------------------------------------------------------

    #[test]
    fn test_soa_layout() {
        let pts = cube_points();
        let soa = SoaPoints::from_points(&pts);
        assert_eq!(soa.len(), pts.len());
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(soa.xs()[i], p.x);
            assert_eq!(soa.ys()[i], p.y);
            assert_eq!(soa.zs()[i], p.z);
        }
    }

    // ---- batch_distances_squared -----------------------------------------

    fn reference_dist_sq(query: &Point3f, pts: &[Point3f]) -> Vec<f32> {
        pts.iter()
            .map(|p| {
                let dx = p.x - query.x;
                let dy = p.y - query.y;
                let dz = p.z - query.z;
                dx * dx + dy * dy + dz * dz
            })
            .collect()
    }

    #[test]
    fn test_scalar_distances_match_reference() {
        let pts = cube_points();
        let soa = SoaPoints::from_points(&pts);
        let query = Point3f::new(0.5, 0.5, 0.5);
        let reference = reference_dist_sq(&query, &pts);
        let mut out = vec![0.0f32; pts.len()];
        scalar_distances_squared(&query, &soa, &mut out);
        for (got, expected) in out.iter().zip(reference.iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_batch_distances_match_scalar() {
        let pts = cube_points();
        let soa = SoaPoints::from_points(&pts);
        let query = Point3f::new(0.3, 0.7, 0.2);

        let mut scalar_out = vec![0.0f32; pts.len()];
        scalar_distances_squared(&query, &soa, &mut scalar_out);

        let mut simd_out = vec![0.0f32; pts.len()];
        batch_distances_squared(&query, &soa, &mut simd_out);

        for (got, expected) in simd_out.iter().zip(scalar_out.iter()) {
            assert!(
                (got - expected).abs() < 1e-5,
                "SIMD={got}, scalar={expected}"
            );
        }
    }

    /// Exhaustively test various point counts including non-multiples of 4 and 8
    /// to verify that the remainder handling is correct.
    #[test]
    fn test_batch_distances_various_sizes() {
        for n in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100] {
            let pts: Vec<Point3f> = (0..n)
                .map(|i| Point3f::new(i as f32, (i * 2) as f32, (i * 3) as f32))
                .collect();
            let soa = SoaPoints::from_points(&pts);
            let query = Point3f::new(5.0, 10.0, 15.0);
            let reference = reference_dist_sq(&query, &pts);

            let mut simd_out = vec![0.0f32; n];
            batch_distances_squared(&query, &soa, &mut simd_out);

            for (i, (got, expected)) in simd_out.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-4,
                    "n={n} i={i}: SIMD={got}, ref={expected}"
                );
            }
        }
    }

    // ---- SimdBruteForceSearch --------------------------------------------

    #[test]
    fn test_simd_knn_matches_brute_force() {
        use crate::nearest_neighbor::BruteForceSearch;
        let pts = cube_points();
        let query = Point3f::new(0.5, 0.5, 0.5);
        let k = 3;

        let simd = SimdBruteForceSearch::new(&pts);
        let scalar = BruteForceSearch::new(&pts);

        let mut simd_res = simd.find_k_nearest(&query, k);
        let mut scalar_res = scalar.find_k_nearest(&query, k);

        simd_res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        scalar_res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));

        assert_eq!(simd_res.len(), k);
        for ((si, sd), (_, bd)) in simd_res.iter().zip(scalar_res.iter()) {
            assert!(
                (sd - bd).abs() < 1e-5,
                "dist mismatch: simd={sd} scalar={bd}"
            );
            let _ = si; // index ties are allowed at equal distance
        }
    }

    #[test]
    fn test_simd_radius_matches_brute_force() {
        use crate::nearest_neighbor::BruteForceSearch;
        let pts = cube_points();
        let query = Point3f::new(0.5, 0.5, 0.5);
        let radius = 1.0;

        let simd = SimdBruteForceSearch::new(&pts);
        let scalar = BruteForceSearch::new(&pts);

        let mut simd_res = simd.find_radius_neighbors(&query, radius);
        let mut scalar_res = scalar.find_radius_neighbors(&query, radius);

        simd_res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        scalar_res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));

        assert_eq!(simd_res.len(), scalar_res.len(), "result count mismatch");
        for ((_, sd), (_, bd)) in simd_res.iter().zip(scalar_res.iter()) {
            assert!((sd - bd).abs() < 1e-5);
        }
    }

    #[test]
    fn test_empty_cloud() {
        let simd = SimdBruteForceSearch::new(&[]);
        let q = Point3f::new(0.0, 0.0, 0.0);
        assert!(simd.find_k_nearest(&q, 5).is_empty());
        assert!(simd.find_radius_neighbors(&q, 10.0).is_empty());
    }

    #[test]
    fn test_k_larger_than_cloud() {
        let pts = cube_points();
        let simd = SimdBruteForceSearch::new(&pts);
        let q = Point3f::new(0.0, 0.0, 0.0);
        let result = simd.find_k_nearest(&q, 100);
        assert_eq!(result.len(), pts.len());
    }

    #[test]
    fn test_exact_origin_distance() {
        let pts = vec![Point3f::new(3.0, 4.0, 0.0)]; // dist from origin = 5
        let soa = SoaPoints::from_points(&pts);
        let query = Point3f::origin();
        let mut out = vec![0.0f32; 1];
        batch_distances_squared(&query, &soa, &mut out);
        assert!((out[0] - 25.0).abs() < 1e-6);
    }
}
