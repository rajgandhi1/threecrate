//! Feature extraction algorithms

use threecrate_core::{PointCloud, Result, Point3f, NormalPoint3f, Error};
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
