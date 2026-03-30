//! Feature extraction algorithms

use threecrate_core::{PointCloud, Result, Point3f, NormalPoint3f, Vector3f, Error};
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::cmp::Ordering;

/// Number of bins per angular feature sub-histogram
const FPFH_BINS: usize = 11;

/// Total FPFH descriptor dimensionality (3 sub-histograms × 11 bins)
pub const FPFH_DIM: usize = 33;

/// Configuration for FPFH feature extraction
#[derive(Debug, Clone)]
pub struct FpfhConfig {
    /// Radius for neighbor search. Points within this distance contribute to the descriptor.
    pub search_radius: f32,
    /// Fallback number of nearest neighbors when radius search yields fewer than this many points.
    pub k_neighbors: usize,
}

impl Default for FpfhConfig {
    fn default() -> Self {
        Self {
            search_radius: 0.1,
            k_neighbors: 10,
        }
    }
}

/// Compute the Darboux frame angular features (α, φ, θ) for a point pair.
///
/// Based on Rusu et al. (2009) "Fast Point Feature Histograms (FPFH) for 3D Registration"
fn compute_pair_features(
    p_s: Point3f,
    n_s: threecrate_core::Vector3f,
    p_t: Point3f,
    n_t: threecrate_core::Vector3f,
) -> Option<(f32, f32, f32)> {
    let delta = p_t - p_s;
    let dist = delta.magnitude();

    if dist < 1e-10 {
        return None;
    }

    let d = delta / dist; // unit direction vector from p_s to p_t

    let u = n_s; // u = n_s
    let v_unnorm = u.cross(&d); // v = u × d
    let v_mag = v_unnorm.magnitude();

    if v_mag < 1e-10 {
        // n_s and d are parallel — degenerate pair, skip
        return None;
    }

    let v = v_unnorm / v_mag;
    let w = u.cross(&v);

    let alpha = v.dot(&n_t); // α = v · n_t  ∈ [-1, 1]
    let phi = u.dot(&d); // φ = u · d    ∈ [-1, 1]
    let theta = w.dot(&n_t).atan2(u.dot(&n_t)); // θ = atan2(w·n_t, u·n_t) ∈ [-π, π]

    Some((alpha, phi, theta))
}

/// Map a value in `[lo, hi]` to a bin index in `[0, n_bins)`.
#[inline]
fn to_bin(value: f32, lo: f32, hi: f32, n_bins: usize) -> usize {
    let normalised = (value - lo) / (hi - lo);
    let bin = (normalised * n_bins as f32) as usize;
    bin.min(n_bins - 1)
}

/// Compute the SPFH (Simplified Point Feature Histogram) for a single point.
fn compute_spfh(
    query_idx: usize,
    points: &[NormalPoint3f],
    neighbor_indices: &[usize],
) -> [f32; FPFH_DIM] {
    let mut histogram = [0.0f32; FPFH_DIM];
    let mut count = 0usize;

    let p_s = points[query_idx].position;
    let n_s = points[query_idx].normal;

    for &nb_idx in neighbor_indices {
        if nb_idx == query_idx {
            continue;
        }
        let p_t = points[nb_idx].position;
        let n_t = points[nb_idx].normal;

        if let Some((alpha, phi, theta)) = compute_pair_features(p_s, n_s, p_t, n_t) {
            let bin_alpha = to_bin(alpha, -1.0, 1.0, FPFH_BINS);
            let bin_phi = to_bin(phi, -1.0, 1.0, FPFH_BINS);
            let bin_theta =
                to_bin(theta, -std::f32::consts::PI, std::f32::consts::PI, FPFH_BINS);

            histogram[bin_alpha] += 1.0;
            histogram[FPFH_BINS + bin_phi] += 1.0;
            histogram[2 * FPFH_BINS + bin_theta] += 1.0;
            count += 1;
        }
    }

    // Normalise each sub-histogram by the number of valid pairs
    if count > 0 {
        let scale = 1.0 / count as f32;
        for h in histogram.iter_mut() {
            *h *= scale;
        }
    }

    histogram
}

/// Find indices of neighbors for a given query point.
///
/// First tries radius-based search. Falls back to k-NN when radius yields
/// fewer than `config.k_neighbors` points.
fn find_neighbors(points: &[NormalPoint3f], query_idx: usize, config: &FpfhConfig) -> Vec<usize> {
    let query = &points[query_idx].position;
    let radius_sq = config.search_radius * config.search_radius;

    let mut within_radius: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if i == query_idx {
                return None;
            }
            let d_sq = (p.position - query).magnitude_squared();
            if d_sq <= radius_sq {
                Some((i, d_sq))
            } else {
                None
            }
        })
        .collect();

    if within_radius.len() >= config.k_neighbors {
        within_radius
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        return within_radius.into_iter().map(|(i, _)| i).collect();
    }

    // Fallback: k-NN over all points
    let mut all: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if i == query_idx {
                return None;
            }
            Some((i, (p.position - query).magnitude_squared()))
        })
        .collect();

    all.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    all.truncate(config.k_neighbors);
    all.into_iter().map(|(i, _)| i).collect()
}

/// Extract FPFH (Fast Point Feature Histograms) features from a point cloud
/// with pre-computed normals.
///
/// Returns a 33-dimensional descriptor `[f32; 33]` per point consisting of
/// three 11-bin sub-histograms for the angular features α, φ and θ between
/// each point and its neighbours.
///
/// Reference: Rusu et al. (2009) "Fast Point Feature Histograms (FPFH) for 3D Registration"
///
/// # Arguments
/// * `cloud`  - Point cloud with pre-computed unit normals.
/// * `config` - Search radius and k-NN fallback configuration.
pub fn extract_fpfh_features_with_normals(
    cloud: &PointCloud<NormalPoint3f>,
    config: &FpfhConfig,
) -> Result<Vec<[f32; FPFH_DIM]>> {
    if cloud.is_empty() {
        return Ok(Vec::new());
    }

    if config.search_radius <= 0.0 {
        return Err(Error::InvalidData(
            "search_radius must be positive".to_string(),
        ));
    }

    let points = &cloud.points;
    let n = points.len();

    // Step 1: collect neighbours for every point
    let neighbors_per_point: Vec<Vec<usize>> =
        (0..n).map(|i| find_neighbors(points, i, config)).collect();

    // Step 2: compute SPFH for every point in parallel
    let spfh: Vec<[f32; FPFH_DIM]> = (0..n)
        .into_par_iter()
        .map(|i| compute_spfh(i, points, &neighbors_per_point[i]))
        .collect();

    // Step 3: FPFH(p) = SPFH(p) + (1/k) Σ (1/dist_i · SPFH(p_i))
    let fpfh: Vec<[f32; FPFH_DIM]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let query_pos = &points[i].position;
            let mut descriptor = spfh[i];

            let neighbors = &neighbors_per_point[i];
            if neighbors.is_empty() {
                return descriptor;
            }

            let mut weight_sum = 0.0f32;
            let mut weighted = [0.0f32; FPFH_DIM];

            for &nb_idx in neighbors {
                let dist = (points[nb_idx].position - query_pos).magnitude();
                if dist < 1e-10 {
                    continue;
                }
                let w = 1.0 / dist;
                weight_sum += w;
                for (j, val) in spfh[nb_idx].iter().enumerate() {
                    weighted[j] += w * val;
                }
            }

            if weight_sum > 0.0 {
                let inv_w = 1.0 / weight_sum;
                for j in 0..FPFH_DIM {
                    descriptor[j] += inv_w * weighted[j];
                }

                // Renormalise each 11-bin sub-histogram so values sum to 1
                for part in 0..3usize {
                    let start = part * FPFH_BINS;
                    let end = start + FPFH_BINS;
                    let sum: f32 = descriptor[start..end].iter().sum();
                    if sum > 0.0 {
                        for h in &mut descriptor[start..end] {
                            *h /= sum;
                        }
                    }
                }
            }

            descriptor
        })
        .collect();

    Ok(fpfh)
}

/// Extract FPFH features from a plain point cloud, estimating normals first.
///
/// Normals are estimated using k = 10 nearest neighbours. For better control,
/// pre-compute normals with [`crate::normals::estimate_normals`] and call
/// [`extract_fpfh_features_with_normals`] directly.
///
/// Returns a 33-element `Vec<f32>` descriptor per point.
pub fn extract_fpfh_features(cloud: &PointCloud<Point3f>) -> Result<Vec<Vec<f32>>> {
    use crate::normals::estimate_normals;

    if cloud.is_empty() {
        return Ok(Vec::new());
    }

    if cloud.len() < 3 {
        return Err(Error::InvalidData(
            "At least 3 points are required to estimate normals for FPFH".to_string(),
        ));
    }

    let cloud_with_normals = estimate_normals(cloud, 10)?;
    let config = FpfhConfig::default();
    let features = extract_fpfh_features_with_normals(&cloud_with_normals, &config)?;
    Ok(features.into_iter().map(|f| f.to_vec()).collect())
}

// =============================================================================
// SHOT (Signature of Histograms of OrienTations)
// Reference: Tombari, Salti & Di Stefano (2010)
// "Unique Signatures of Histograms for Local Surface Description"
// =============================================================================

/// Azimuth sectors in SHOT's spherical partition
const SHOT_N_AZIMUTH: usize = 8;
/// Elevation hemispheres (south / north of LRF z-axis)
const SHOT_N_ELEVATION: usize = 2;
/// Radial shells (inner / outer half of support sphere)
const SHOT_N_RADIAL: usize = 2;
/// Normal-orientation histogram bins per SHOT volume
const SHOT_N_BINS: usize = 11;
/// Total number of SHOT volumes: 8 × 2 × 2 = 32
const SHOT_N_VOLUMES: usize = SHOT_N_AZIMUTH * SHOT_N_ELEVATION * SHOT_N_RADIAL;
/// Total SHOT descriptor dimensionality: 32 volumes × 11 bins = 352
pub const SHOT_DIM: usize = SHOT_N_VOLUMES * SHOT_N_BINS;

/// Azimuth sectors in USC's spatial histogram
const USC_N_AZIMUTH: usize = 8;
/// Elevation (cosine-uniform) divisions in USC
const USC_N_ELEVATION: usize = 4;
/// Radial shells in USC
const USC_N_RADIAL: usize = 4;
/// Total USC descriptor dimensionality: 8 × 4 × 4 = 128
pub const USC_DIM: usize = USC_N_AZIMUTH * USC_N_ELEVATION * USC_N_RADIAL;

/// Which SHOT-family descriptor to compute
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ShotVariant {
    /// Standard SHOT — 352-dimensional histogram of surface-normal orientations per volume
    #[default]
    Standard,
    /// Unique Shape Context — 128-dimensional spatial density histogram sharing the same LRF
    UniqueShapeContext,
}

/// Configuration for SHOT / USC feature extraction
#[derive(Debug, Clone)]
pub struct ShotConfig {
    /// Support sphere radius (same role as `FpfhConfig::search_radius`)
    pub search_radius: f32,
    /// Minimum neighbors from radius search; falls back to k-NN when fewer found
    pub k_neighbors: usize,
    /// Which descriptor variant to compute
    pub variant: ShotVariant,
}

impl Default for ShotConfig {
    fn default() -> Self {
        Self {
            search_radius: 0.2,
            k_neighbors: 10,
            variant: ShotVariant::Standard,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Find neighbor indices for SHOT (radius search with k-NN fallback).
fn find_shot_neighbors(
    points: &[NormalPoint3f],
    query_idx: usize,
    config: &ShotConfig,
) -> Vec<usize> {
    let query = &points[query_idx].position;
    let r_sq = config.search_radius * config.search_radius;

    let mut within: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if i == query_idx {
                return None;
            }
            let d_sq = (p.position - query).magnitude_squared();
            if d_sq <= r_sq { Some((i, d_sq)) } else { None }
        })
        .collect();

    if within.len() >= config.k_neighbors {
        within.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        return within.into_iter().map(|(i, _)| i).collect();
    }

    // k-NN fallback
    let mut all: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if i == query_idx { None }
            else { Some((i, (p.position - query).magnitude_squared())) }
        })
        .collect();
    all.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    all.truncate(config.k_neighbors);
    all.into_iter().map(|(i, _)| i).collect()
}

/// Compute the SHOT Local Reference Frame (LRF).
///
/// Returns (x_axis, y_axis, z_axis) forming a right-handed coordinate frame.
/// The z-axis is the disambiguated surface normal; x-axis is derived from the
/// weighted covariance of the neighborhood.
fn compute_shot_lrf(
    query_pos: threecrate_core::Point3f,
    query_normal: Vector3f,
    neighbors: &[usize],
    points: &[NormalPoint3f],
    radius: f32,
) -> (Vector3f, Vector3f, Vector3f) {
    // --- z-axis: use provided normal, then disambiguate direction ---
    let mut z_axis = if query_normal.magnitude() > 1e-10 {
        query_normal.normalize()
    } else {
        Vector3f::new(0.0, 0.0, 1.0)
    };

    let n_pos_z = neighbors.iter()
        .filter(|&&i| z_axis.dot(&(points[i].position - query_pos)) >= 0.0)
        .count();
    if n_pos_z * 2 < neighbors.len() {
        z_axis = -z_axis;
    }

    // --- x-axis: eigenvector of largest eigenvalue of the weighted covariance ---
    let mut cov = Matrix3::<f32>::zeros();
    for &i in neighbors {
        let dv = points[i].position - query_pos; // Vector3f
        let dist = dv.magnitude();
        let w = (radius - dist).max(0.0);
        cov += w * dv * dv.transpose();
    }

    let eig = cov.symmetric_eigen();
    let (max_idx, _) = eig.eigenvalues.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .unwrap_or((0, &0.0));
    let col = eig.eigenvectors.column(max_idx);
    let mut x_axis = Vector3f::new(col[0], col[1], col[2]);

    // Disambiguate
    let n_pos_x = neighbors.iter()
        .filter(|&&i| x_axis.dot(&(points[i].position - query_pos)) >= 0.0)
        .count();
    if n_pos_x * 2 < neighbors.len() {
        x_axis = -x_axis;
    }

    // Project onto tangent plane and renormalize
    let x_proj = x_axis - z_axis * z_axis.dot(&x_axis);
    let x_axis = if x_proj.magnitude() > 1e-10 {
        x_proj.normalize()
    } else {
        let c = Vector3f::new(1.0, 0.0, 0.0);
        let p = c - z_axis * z_axis.dot(&c);
        if p.magnitude() > 1e-10 {
            p.normalize()
        } else {
            let c2 = Vector3f::new(0.0, 1.0, 0.0);
            (c2 - z_axis * z_axis.dot(&c2)).normalize()
        }
    };

    let y_axis = z_axis.cross(&x_axis);
    (x_axis, y_axis, z_axis)
}

/// Compute a single 352-dimensional SHOT descriptor.
fn compute_shot_impl(
    query_idx: usize,
    points: &[NormalPoint3f],
    neighbors: &[usize],
    radius: f32,
) -> [f32; SHOT_DIM] {
    let mut desc = [0.0f32; SHOT_DIM];
    if neighbors.is_empty() {
        return desc;
    }

    let qp = points[query_idx].position;
    let (x_axis, y_axis, z_axis) =
        compute_shot_lrf(qp, points[query_idx].normal, neighbors, points, radius);

    let mut vol_counts = [0u32; SHOT_N_VOLUMES];

    for &ni in neighbors {
        if ni == query_idx { continue; }
        let dv = points[ni].position - qp;
        let dist = dv.magnitude();
        if dist < 1e-10 || dist > radius { continue; }

        let lx = x_axis.dot(&dv);
        let ly = y_axis.dot(&dv);
        let lz = z_axis.dot(&dv);

        let r_bin = if dist <= radius * 0.5 { 0 } else { 1 };
        let e_bin = if lz < 0.0 { 0 } else { 1 };
        let az_norm = (ly.atan2(lx) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
        let a_bin = ((az_norm * SHOT_N_AZIMUTH as f32) as usize).min(SHOT_N_AZIMUTH - 1);

        let vol = r_bin * (SHOT_N_ELEVATION * SHOT_N_AZIMUTH) + e_bin * SHOT_N_AZIMUTH + a_bin;

        let cos_theta = z_axis.dot(&points[ni].normal).clamp(-1.0, 1.0);
        let n_bin = to_bin(cos_theta, -1.0, 1.0, SHOT_N_BINS);

        desc[vol * SHOT_N_BINS + n_bin] += 1.0;
        vol_counts[vol] += 1;
    }

    // Normalize each sub-histogram by its volume's point count
    for vol in 0..SHOT_N_VOLUMES {
        let c = vol_counts[vol] as f32;
        if c > 0.0 {
            for v in &mut desc[vol * SHOT_N_BINS..(vol + 1) * SHOT_N_BINS] {
                *v /= c;
            }
        }
    }

    // L2-normalize the full descriptor
    let norm: f32 = desc.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut desc { *v /= norm; }
    }
    desc
}

/// Compute a single 128-dimensional USC descriptor.
///
/// USC encodes spatial point density in a 3-D histogram (azimuth × elevation × radial)
/// using the same LRF as SHOT, giving a rotationally unique but normal-free descriptor.
fn compute_usc_impl(
    query_idx: usize,
    points: &[NormalPoint3f],
    neighbors: &[usize],
    radius: f32,
) -> [f32; USC_DIM] {
    let mut desc = [0.0f32; USC_DIM];
    if neighbors.is_empty() {
        return desc;
    }

    let qp = points[query_idx].position;
    let (x_axis, y_axis, z_axis) =
        compute_shot_lrf(qp, points[query_idx].normal, neighbors, points, radius);

    let mut total = 0usize;
    for &ni in neighbors {
        if ni == query_idx { continue; }
        let dv = points[ni].position - qp;
        let dist = dv.magnitude();
        if dist < 1e-10 || dist > radius { continue; }

        let lx = x_axis.dot(&dv);
        let ly = y_axis.dot(&dv);
        let lz = z_axis.dot(&dv);

        // Azimuth: 8 uniform sectors in the tangent plane
        let az_norm = (ly.atan2(lx) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
        let a_bin = ((az_norm * USC_N_AZIMUTH as f32) as usize).min(USC_N_AZIMUTH - 1);

        // Elevation: cosine-uniform over [-1, 1] → 4 equal-area polar bands
        let cos_el = (lz / dist).clamp(-1.0, 1.0);
        let e_bin = to_bin(cos_el, -1.0, 1.0, USC_N_ELEVATION);

        // Radial: 4 equal-width shells over [0, radius]
        let r_bin = ((dist / radius * USC_N_RADIAL as f32) as usize).min(USC_N_RADIAL - 1);

        let bin = a_bin * (USC_N_ELEVATION * USC_N_RADIAL) + e_bin * USC_N_RADIAL + r_bin;
        desc[bin] += 1.0;
        total += 1;
    }

    if total > 0 {
        let scale = 1.0 / total as f32;
        for v in &mut desc { *v *= scale; }
    }

    // L2-normalize
    let norm: f32 = desc.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut desc { *v /= norm; }
    }
    desc
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract SHOT or USC features from a point cloud with pre-computed normals.
///
/// Selects the descriptor variant via `config.variant`:
/// - [`ShotVariant::Standard`] → 352-dimensional SHOT descriptor per point
/// - [`ShotVariant::UniqueShapeContext`] → 128-dimensional USC descriptor per point
///
/// Computation is parallelised over points with rayon.
///
/// Reference: Tombari, Salti & Di Stefano (2010)
/// "Unique Signatures of Histograms for Local Surface Description", ECCV 2010.
///
/// # Arguments
/// * `cloud`  – Point cloud with pre-computed unit normals.
/// * `config` – Search radius, k-NN fallback, and variant selection.
pub fn extract_shot_features_with_normals(
    cloud: &PointCloud<NormalPoint3f>,
    config: &ShotConfig,
) -> Result<Vec<Vec<f32>>> {
    if cloud.is_empty() {
        return Ok(Vec::new());
    }
    if config.search_radius <= 0.0 {
        return Err(Error::InvalidData("search_radius must be positive".into()));
    }

    let points = &cloud.points;
    let n = points.len();

    // Gather neighbors (sequential — avoids borrowing issues with par_iter)
    let neighbors: Vec<Vec<usize>> =
        (0..n).map(|i| find_shot_neighbors(points, i, config)).collect();

    // Compute descriptors in parallel
    let descriptors: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|i| match config.variant {
            ShotVariant::Standard =>
                compute_shot_impl(i, points, &neighbors[i], config.search_radius).to_vec(),
            ShotVariant::UniqueShapeContext =>
                compute_usc_impl(i, points, &neighbors[i], config.search_radius).to_vec(),
        })
        .collect();

    Ok(descriptors)
}

#[cfg(test)]
mod shot_tests {
    use super::*;
    use threecrate_core::{NormalPoint3f, Vector3f, Point3f, PointCloud};

    fn make_plane(n: usize) -> PointCloud<NormalPoint3f> {
        let side = (n as f64).sqrt().ceil() as usize;
        let step = 1.0 / side as f32;
        let mut cloud = PointCloud::new();
        'outer: for i in 0..side {
            for j in 0..side {
                if cloud.len() == n { break 'outer; }
                cloud.push(NormalPoint3f {
                    position: Point3f::new(i as f32 * step, j as f32 * step, 0.0),
                    normal: Vector3f::new(0.0, 0.0, 1.0),
                });
            }
        }
        cloud
    }

    #[test]
    fn test_shot_empty_cloud() {
        let cloud = PointCloud::<NormalPoint3f>::new();
        let result = extract_shot_features_with_normals(&cloud, &ShotConfig::default()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_shot_descriptor_dim() {
        let cloud = make_plane(25);
        let config = ShotConfig { search_radius: 0.5, k_neighbors: 5, variant: ShotVariant::Standard };
        let result = extract_shot_features_with_normals(&cloud, &config).unwrap();
        assert_eq!(result.len(), 25);
        for d in &result { assert_eq!(d.len(), SHOT_DIM); }
    }

    #[test]
    fn test_usc_descriptor_dim() {
        let cloud = make_plane(25);
        let config = ShotConfig { search_radius: 0.5, k_neighbors: 5, variant: ShotVariant::UniqueShapeContext };
        let result = extract_shot_features_with_normals(&cloud, &config).unwrap();
        assert_eq!(result.len(), 25);
        for d in &result { assert_eq!(d.len(), USC_DIM); }
    }

    #[test]
    fn test_shot_non_negative() {
        let cloud = make_plane(25);
        let config = ShotConfig { search_radius: 0.5, k_neighbors: 5, variant: ShotVariant::Standard };
        let result = extract_shot_features_with_normals(&cloud, &config).unwrap();
        for d in &result {
            for &v in d { assert!(v >= 0.0, "negative value: {v}"); }
        }
    }

    #[test]
    fn test_shot_l2_normalized_or_zero() {
        let cloud = make_plane(25);
        let config = ShotConfig { search_radius: 0.5, k_neighbors: 5, variant: ShotVariant::Standard };
        let result = extract_shot_features_with_normals(&cloud, &config).unwrap();
        for d in &result {
            let norm: f32 = d.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!(norm < 1e-6 || (norm - 1.0).abs() < 1e-4, "norm={norm}");
        }
    }

    #[test]
    fn test_shot_invalid_radius() {
        let cloud = make_plane(9);
        let config = ShotConfig { search_radius: -0.1, k_neighbors: 5, variant: ShotVariant::Standard };
        assert!(extract_shot_features_with_normals(&cloud, &config).is_err());
    }

    #[test]
    fn test_shot_deterministic() {
        let cloud = make_plane(25);
        let config = ShotConfig { search_radius: 0.5, k_neighbors: 5, variant: ShotVariant::Standard };
        let r1 = extract_shot_features_with_normals(&cloud, &config).unwrap();
        let r2 = extract_shot_features_with_normals(&cloud, &config).unwrap();
        for (d1, d2) in r1.iter().zip(&r2) {
            for (&v1, &v2) in d1.iter().zip(d2) {
                assert!((v1 - v2).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_shot_plane_vs_sphere_differ() {
        let plane = make_plane(25);
        let config = ShotConfig { search_radius: 0.6, k_neighbors: 8, variant: ShotVariant::Standard };
        let plane_descs = extract_shot_features_with_normals(&plane, &config).unwrap();

        let mut sphere = PointCloud::<NormalPoint3f>::new();
        for i in 0..5usize {
            for j in 0..5usize {
                let theta = std::f32::consts::PI * i as f32 / 4.0;
                let phi   = 2.0 * std::f32::consts::PI * j as f32 / 5.0;
                let x = theta.sin() * phi.cos();
                let y = theta.sin() * phi.sin();
                let z = theta.cos();
                sphere.push(NormalPoint3f {
                    position: Point3f::new(x, y, z),
                    normal:   Vector3f::new(x, y, z),
                });
            }
        }
        let sphere_descs = extract_shot_features_with_normals(&sphere, &config).unwrap();

        let any_different = plane_descs.iter().any(|pd| {
            sphere_descs.iter().any(|sd| {
                pd.iter().zip(sd).map(|(a, b)| (a - b).abs()).sum::<f32>() > 0.05
            })
        });
        assert!(any_different, "plane and sphere descriptors should differ");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{NormalPoint3f, Vector3f};

    /// Build a flat XY-plane point cloud with upward normals.
    fn make_plane_cloud(n: usize) -> PointCloud<NormalPoint3f> {
        let side = (n as f64).sqrt().ceil() as usize;
        let step = 1.0 / side as f32;
        let mut cloud = PointCloud::new();
        'outer: for i in 0..side {
            for j in 0..side {
                if cloud.len() == n {
                    break 'outer;
                }
                cloud.push(NormalPoint3f {
                    position: Point3f::new(i as f32 * step, j as f32 * step, 0.0),
                    normal: Vector3f::new(0.0, 0.0, 1.0),
                });
            }
        }
        cloud
    }

    #[test]
    fn test_fpfh_empty_cloud() {
        let cloud = PointCloud::<NormalPoint3f>::new();
        let config = FpfhConfig::default();
        let result = extract_fpfh_features_with_normals(&cloud, &config).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_fpfh_descriptor_dimension() {
        let cloud = make_plane_cloud(25);
        let config = FpfhConfig {
            search_radius: 0.5,
            k_neighbors: 5,
        };
        let result = extract_fpfh_features_with_normals(&cloud, &config).unwrap();
        assert_eq!(result.len(), cloud.len());
        for desc in &result {
            assert_eq!(desc.len(), FPFH_DIM);
        }
    }

    #[test]
    fn test_fpfh_descriptor_non_negative() {
        let cloud = make_plane_cloud(25);
        let config = FpfhConfig {
            search_radius: 0.5,
            k_neighbors: 5,
        };
        let result = extract_fpfh_features_with_normals(&cloud, &config).unwrap();
        for desc in &result {
            for &v in desc.iter() {
                assert!(v >= 0.0, "Descriptor value must be non-negative, got {}", v);
            }
        }
    }

    #[test]
    fn test_fpfh_sub_histograms_normalised() {
        let cloud = make_plane_cloud(36);
        let config = FpfhConfig {
            search_radius: 0.5,
            k_neighbors: 8,
        };
        let result = extract_fpfh_features_with_normals(&cloud, &config).unwrap();
        for desc in &result {
            for part in 0..3 {
                let start = part * FPFH_BINS;
                let end = start + FPFH_BINS;
                let sum: f32 = desc[start..end].iter().sum();
                assert!(
                    sum < 1e-6 || (sum - 1.0).abs() < 1e-4,
                    "Sub-histogram {} sum = {}, expected 0 or ~1.0",
                    part,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_fpfh_identical_clouds_same_descriptors() {
        let cloud = make_plane_cloud(25);
        let config = FpfhConfig {
            search_radius: 0.5,
            k_neighbors: 5,
        };
        let r1 = extract_fpfh_features_with_normals(&cloud, &config).unwrap();
        let r2 = extract_fpfh_features_with_normals(&cloud, &config).unwrap();
        for (d1, d2) in r1.iter().zip(r2.iter()) {
            for (&v1, &v2) in d1.iter().zip(d2.iter()) {
                assert!((v1 - v2).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_fpfh_plane_vs_sphere_differ() {
        let plane = make_plane_cloud(25);
        let config = FpfhConfig {
            search_radius: 0.5,
            k_neighbors: 5,
        };
        let plane_desc = extract_fpfh_features_with_normals(&plane, &config).unwrap();

        // Build a spherical cloud
        let mut sphere = PointCloud::<NormalPoint3f>::new();
        let steps = 5usize;
        for i in 0..steps {
            for j in 0..steps {
                let theta = std::f32::consts::PI * i as f32 / (steps - 1) as f32;
                let phi = 2.0 * std::f32::consts::PI * j as f32 / steps as f32;
                let x = theta.sin() * phi.cos();
                let y = theta.sin() * phi.sin();
                let z = theta.cos();
                sphere.push(NormalPoint3f {
                    position: Point3f::new(x, y, z),
                    normal: Vector3f::new(x, y, z),
                });
            }
        }

        let sphere_desc = extract_fpfh_features_with_normals(&sphere, &config).unwrap();

        let mut any_different = false;
        'outer: for pd in &plane_desc {
            for sd in &sphere_desc {
                let diff: f32 = pd.iter().zip(sd.iter()).map(|(a, b)| (a - b).abs()).sum();
                if diff > 0.1 {
                    any_different = true;
                    break 'outer;
                }
            }
        }
        assert!(
            any_different,
            "Plane and sphere descriptors should differ significantly"
        );
    }

    #[test]
    fn test_fpfh_from_xyz() {
        let mut cloud = PointCloud::<Point3f>::new();
        for i in 0..5 {
            for j in 0..5 {
                cloud.push(Point3f::new(i as f32 * 0.1, j as f32 * 0.1, 0.0));
            }
        }
        let result = extract_fpfh_features(&cloud).unwrap();
        assert_eq!(result.len(), cloud.len());
        for desc in &result {
            assert_eq!(desc.len(), FPFH_DIM);
        }
    }

    #[test]
    fn test_fpfh_invalid_radius() {
        let cloud = make_plane_cloud(9);
        let config = FpfhConfig {
            search_radius: -1.0,
            k_neighbors: 5,
        };
        let result = extract_fpfh_features_with_normals(&cloud, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_fpfh_single_point_all_zero() {
        let mut cloud = PointCloud::<NormalPoint3f>::new();
        cloud.push(NormalPoint3f {
            position: Point3f::new(0.0, 0.0, 0.0),
            normal: Vector3f::new(0.0, 0.0, 1.0),
        });
        let config = FpfhConfig {
            search_radius: 1.0,
            k_neighbors: 1,
        };
        let result = extract_fpfh_features_with_normals(&cloud, &config).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            result[0].iter().all(|&v| v == 0.0),
            "Single-point descriptor should be all zeros"
        );
    }
}
