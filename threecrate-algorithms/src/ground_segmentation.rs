//! Ground segmentation for outdoor LiDAR point clouds.
//!
//! Implements a Patchwork++-style algorithm that splits a point cloud into
//! ground and non-ground subsets. The approach uses a Concentric Zone Model
//! (CZM): the XY plane around the sensor is divided into concentric rings and
//! angular sectors, producing a grid of patches whose size adapts with range.
//! Each patch is fit with a Region-wise Ground Plane Fit (R-GPF) — seed points
//! near the patch minimum are selected, a plane is fit by PCA, then inliers
//! are extracted and the plane is refit iteratively. Each candidate plane is
//! then validated by three criteria: **uprightness** (normal nearly vertical),
//! **elevation** (within an allowed height band per zone), and **flatness**
//! (smallest eigenvalue ratio).
//!
//! Reference: Lee et al., "Patchwork++: Fast and Robust Ground Segmentation
//! Solving Partial Under-Segmentation Using 3D Point Cloud", IROS 2022.

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::f32::consts::PI;
use threecrate_core::{Error, Point3f, PointCloud, Result, Vector3f};

/// Configuration for the Patchwork++ ground segmentation algorithm.
#[derive(Debug, Clone)]
pub struct PatchworkConfig {
    /// Height of the LiDAR sensor above the expected ground plane (meters).
    pub sensor_height: f32,
    /// Concentric zone boundary radii in meters, length = num_zones + 1.
    /// Each adjacent pair defines one zone (inner_radius, outer_radius].
    pub zone_radii: Vec<f32>,
    /// Number of concentric rings per zone (length = num_zones).
    pub num_rings_per_zone: Vec<usize>,
    /// Number of angular sectors per zone (length = num_zones).
    pub num_sectors_per_zone: Vec<usize>,
    /// Maximum sensing range (meters); points beyond are treated as non-ground.
    pub max_range: f32,
    /// Minimum number of points required in a patch to attempt ground fitting.
    pub min_points_per_patch: usize,
    /// Minimum number of seed points used in the initial PCA fit.
    pub num_seed_points: usize,
    /// A point is a seed candidate if its z is within this distance of the
    /// patch's minimum z value.
    pub seed_selection_threshold: f32,
    /// Inlier distance threshold (meters) used when refining the plane fit.
    pub dist_threshold: f32,
    /// Number of refit iterations of the R-GPF inner loop.
    pub num_iterations: usize,
    /// Minimum |n_z| required for a patch's plane to count as ground
    /// (cos of maximum allowed slope; 0.707 ≈ 45°).
    pub uprightness_threshold: f32,
    /// Maximum allowed flatness ratio = lambda_min / (lambda_0 + lambda_1 + lambda_min);
    /// smaller means the patch must be flatter.
    pub flatness_threshold: f32,
    /// Maximum signed deviation of the patch mean z from `-sensor_height`
    /// (meters). Patches sitting unreasonably high or low are rejected.
    pub elevation_threshold: f32,
}

impl Default for PatchworkConfig {
    fn default() -> Self {
        // Defaults follow the Patchwork++ reference implementation.
        Self {
            sensor_height: 1.723,
            zone_radii: vec![0.0, 2.7, 12.3625, 22.025, 80.0],
            num_rings_per_zone: vec![2, 4, 4, 4],
            num_sectors_per_zone: vec![16, 32, 54, 32],
            max_range: 80.0,
            min_points_per_patch: 10,
            num_seed_points: 20,
            seed_selection_threshold: 0.5,
            dist_threshold: 0.125,
            num_iterations: 3,
            uprightness_threshold: 0.707,
            flatness_threshold: 0.05,
            elevation_threshold: 1.0,
        }
    }
}

/// Result of ground segmentation.
#[derive(Debug, Clone)]
pub struct GroundSegmentationResult {
    /// Points classified as ground.
    pub ground: PointCloud<Point3f>,
    /// Points classified as non-ground (obstacles, vegetation above ground, etc.).
    pub nonground: PointCloud<Point3f>,
    /// Per-input-point labels: true if classified as ground.
    pub labels: Vec<bool>,
}

fn validate_config(cfg: &PatchworkConfig) -> Result<()> {
    let nz = cfg.num_rings_per_zone.len();
    if nz == 0 {
        return Err(Error::InvalidData("num_rings_per_zone must be non-empty".into()));
    }
    if cfg.zone_radii.len() != nz + 1 {
        return Err(Error::InvalidData(
            "zone_radii.len() must equal num_rings_per_zone.len() + 1".into(),
        ));
    }
    if cfg.num_sectors_per_zone.len() != nz {
        return Err(Error::InvalidData(
            "num_sectors_per_zone.len() must equal num_rings_per_zone.len()".into(),
        ));
    }
    if cfg.zone_radii.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::InvalidData("zone_radii must be strictly increasing".into()));
    }
    if cfg.dist_threshold <= 0.0 {
        return Err(Error::InvalidData("dist_threshold must be positive".into()));
    }
    if cfg.num_seed_points == 0 {
        return Err(Error::InvalidData("num_seed_points must be at least 1".into()));
    }
    if cfg.uprightness_threshold <= 0.0 || cfg.uprightness_threshold > 1.0 {
        return Err(Error::InvalidData("uprightness_threshold must be in (0, 1]".into()));
    }
    Ok(())
}

/// Decide which zone a (range) value falls into; returns `None` if it's
/// outside the configured zone range.
fn find_zone(radius: f32, zone_radii: &[f32]) -> Option<usize> {
    if radius < zone_radii[0] || radius >= *zone_radii.last().unwrap() {
        return None;
    }
    for i in 0..zone_radii.len() - 1 {
        if radius >= zone_radii[i] && radius < zone_radii[i + 1] {
            return Some(i);
        }
    }
    None
}

/// Group point indices into CZM patches keyed by (zone, ring, sector).
fn bucket_points(
    points: &[Point3f],
    cfg: &PatchworkConfig,
) -> (Vec<Vec<Vec<Vec<usize>>>>, Vec<bool>) {
    // patches[zone][ring][sector] = Vec<point_index>
    let mut patches: Vec<Vec<Vec<Vec<usize>>>> = (0..cfg.num_rings_per_zone.len())
        .map(|z| {
            (0..cfg.num_rings_per_zone[z])
                .map(|_| (0..cfg.num_sectors_per_zone[z]).map(|_| Vec::new()).collect())
                .collect()
        })
        .collect();
    let mut out_of_range = vec![false; points.len()];

    for (idx, p) in points.iter().enumerate() {
        let r = (p.x * p.x + p.y * p.y).sqrt();
        if r > cfg.max_range {
            out_of_range[idx] = true;
            continue;
        }
        let zone = match find_zone(r, &cfg.zone_radii) {
            Some(z) => z,
            None => {
                out_of_range[idx] = true;
                continue;
            }
        };
        let r_inner = cfg.zone_radii[zone];
        let r_outer = cfg.zone_radii[zone + 1];
        let ring_width = (r_outer - r_inner) / cfg.num_rings_per_zone[zone] as f32;
        let ring = (((r - r_inner) / ring_width) as usize)
            .min(cfg.num_rings_per_zone[zone] - 1);

        let mut theta = p.y.atan2(p.x);
        if theta < 0.0 {
            theta += 2.0 * PI;
        }
        let sector_width = 2.0 * PI / cfg.num_sectors_per_zone[zone] as f32;
        let sector = ((theta / sector_width) as usize)
            .min(cfg.num_sectors_per_zone[zone] - 1);

        patches[zone][ring][sector].push(idx);
    }

    (patches, out_of_range)
}

/// PCA on a set of points; returns (mean, eigenvalues sorted ascending,
/// corresponding eigenvectors as columns of the matrix).
fn pca(points: &[Point3f], indices: &[usize]) -> Option<(Vector3<f32>, [f32; 3], Matrix3<f32>)> {
    if indices.len() < 3 {
        return None;
    }
    let n = indices.len() as f32;
    let mut mean = Vector3::<f32>::zeros();
    for &i in indices {
        mean += points[i].coords;
    }
    mean /= n;

    let mut cov = Matrix3::<f32>::zeros();
    for &i in indices {
        let d = points[i].coords - mean;
        cov += d * d.transpose();
    }
    cov /= n;

    // Symmetric eigendecomposition.
    let eig = cov.symmetric_eigen();
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap());
    let vals = [
        eig.eigenvalues[idx[0]],
        eig.eigenvalues[idx[1]],
        eig.eigenvalues[idx[2]],
    ];
    let mut vecs = Matrix3::<f32>::zeros();
    for k in 0..3 {
        vecs.set_column(k, &eig.eigenvectors.column(idx[k]));
    }
    Some((mean, vals, vecs))
}

/// Fit a ground plane to a single patch using R-GPF; returns plane parameters
/// (normal, d) and the inlier indices (into the original `points` slice) if a
/// valid plane was found.
fn fit_patch(
    points: &[Point3f],
    patch: &[usize],
    cfg: &PatchworkConfig,
) -> Option<(Vector3<f32>, f32, Vec<usize>)> {
    if patch.len() < cfg.min_points_per_patch {
        return None;
    }

    // Initial seeds: points whose z is within seed_selection_threshold of min z.
    let mut sorted_by_z: Vec<usize> = patch.to_vec();
    sorted_by_z.sort_by(|&a, &b| points[a].z.partial_cmp(&points[b].z).unwrap());

    let seed_count = cfg.num_seed_points.min(sorted_by_z.len());
    let z_min_mean = {
        let n = seed_count.min(sorted_by_z.len());
        if n == 0 {
            return None;
        }
        let mut s = 0.0;
        for &i in &sorted_by_z[..n] {
            s += points[i].z;
        }
        s / n as f32
    };
    let cutoff = z_min_mean + cfg.seed_selection_threshold;
    let mut current: Vec<usize> = sorted_by_z
        .iter()
        .copied()
        .take_while(|&i| points[i].z <= cutoff)
        .collect();
    if current.len() < 3 {
        return None;
    }

    let mut last: Option<(Vector3<f32>, f32)> = None;
    for _ in 0..cfg.num_iterations {
        let (mean, _vals, vecs) = pca(points, &current)?;
        // Smallest-eigenvalue eigenvector = surface normal.
        let mut normal = Vector3::new(vecs[(0, 0)], vecs[(1, 0)], vecs[(2, 0)]);
        if normal.z < 0.0 {
            normal = -normal;
        }
        let d = -normal.dot(&mean);

        // Re-collect inliers from the full patch.
        let mut new_inliers = Vec::with_capacity(patch.len());
        for &i in patch {
            let dist = (normal.dot(&points[i].coords) + d).abs();
            if dist <= cfg.dist_threshold {
                new_inliers.push(i);
            }
        }
        if new_inliers.len() < 3 {
            return None;
        }
        last = Some((normal, d));
        if new_inliers.len() == current.len() {
            current = new_inliers;
            break;
        }
        current = new_inliers;
    }

    let (normal, d) = last?;
    Some((normal, d, current))
}

/// Validate a fitted patch under uprightness, elevation, and flatness criteria.
fn validate_patch(
    points: &[Point3f],
    inliers: &[usize],
    normal: Vector3<f32>,
    cfg: &PatchworkConfig,
) -> bool {
    // Uprightness: normal must be close to +z.
    if normal.z.abs() < cfg.uprightness_threshold {
        return false;
    }

    // Mean z elevation: expected near -sensor_height.
    let mut mean_z = 0.0;
    for &i in inliers {
        mean_z += points[i].z;
    }
    mean_z /= inliers.len() as f32;
    if (mean_z + cfg.sensor_height).abs() > cfg.elevation_threshold {
        return false;
    }

    // Flatness: smallest eigenvalue should be small compared to the others.
    if let Some((_mean, vals, _vecs)) = pca(points, inliers) {
        let sum = vals[0] + vals[1] + vals[2];
        if sum > 0.0 {
            let ratio = vals[0] / sum;
            if ratio > cfg.flatness_threshold {
                return false;
            }
        }
    }

    true
}

/// Run Patchwork++-style ground segmentation on a point cloud.
pub fn patchwork_plus_plus(
    cloud: &PointCloud<Point3f>,
    config: PatchworkConfig,
) -> Result<GroundSegmentationResult> {
    validate_config(&config)?;
    let points = &cloud.points;
    let mut labels = vec![false; points.len()];

    if points.is_empty() {
        return Ok(GroundSegmentationResult {
            ground: PointCloud::new(),
            nonground: PointCloud::new(),
            labels,
        });
    }

    let (patches, out_of_range) = bucket_points(points, &config);

    // Flatten patches so we can process them in parallel.
    let mut flat: Vec<&Vec<usize>> = Vec::new();
    for zone in &patches {
        for ring in zone {
            for sector in ring {
                if !sector.is_empty() {
                    flat.push(sector);
                }
            }
        }
    }

    let cfg_ref = &config;
    let ground_index_sets: Vec<Vec<usize>> = flat
        .par_iter()
        .filter_map(|patch| {
            let (normal, _d, inliers) = fit_patch(points, patch, cfg_ref)?;
            if validate_patch(points, &inliers, normal, cfg_ref) {
                Some(inliers)
            } else {
                None
            }
        })
        .collect();

    for set in ground_index_sets {
        for i in set {
            labels[i] = true;
        }
    }

    // Anything out of range is non-ground.
    for (i, oor) in out_of_range.iter().enumerate() {
        if *oor {
            labels[i] = false;
        }
    }

    let mut ground = PointCloud::with_capacity(labels.iter().filter(|b| **b).count());
    let mut nonground = PointCloud::with_capacity(labels.iter().filter(|b| !**b).count());
    for (i, p) in points.iter().enumerate() {
        if labels[i] {
            ground.push(*p);
        } else {
            nonground.push(*p);
        }
    }

    Ok(GroundSegmentationResult { ground, nonground, labels })
}

/// Convenience wrapper using default configuration with a given sensor height.
pub fn segment_ground(
    cloud: &PointCloud<Point3f>,
    sensor_height: f32,
) -> Result<GroundSegmentationResult> {
    let config = PatchworkConfig { sensor_height, ..Default::default() };
    patchwork_plus_plus(cloud, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn build_scene(sensor_height: f32, with_obstacles: bool) -> PointCloud<Point3f> {
        let mut cloud = PointCloud::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Flat ground at z = -sensor_height over a 60×60 area, with mild noise.
        let z_ground = -sensor_height;
        for _ in 0..8000 {
            let x: f32 = rng.gen_range(-30.0..30.0);
            let y: f32 = rng.gen_range(-30.0..30.0);
            let z = z_ground + rng.gen_range(-0.02..0.02);
            // Skip points right under the sensor (too close to origin).
            if x * x + y * y < 0.25 {
                continue;
            }
            cloud.push(Point3f::new(x, y, z));
        }

        if with_obstacles {
            // Tall vertical "wall" / obstacle cluster.
            for _ in 0..1500 {
                let x = 8.0 + rng.gen_range(-0.4..0.4);
                let y = rng.gen_range(-3.0..3.0);
                let z = z_ground + rng.gen_range(0.5..3.0);
                cloud.push(Point3f::new(x, y, z));
            }
            // A pole.
            for _ in 0..400 {
                let x = -5.0 + rng.gen_range(-0.1..0.1);
                let y = -5.0 + rng.gen_range(-0.1..0.1);
                let z = z_ground + rng.gen_range(0.0..4.0);
                cloud.push(Point3f::new(x, y, z));
            }
        }

        cloud
    }

    #[test]
    fn flat_ground_is_mostly_ground() {
        let sensor_h = 1.8;
        let cloud = build_scene(sensor_h, false);
        let n = cloud.len();
        let result = segment_ground(&cloud, sensor_h).unwrap();
        let ground_frac = result.ground.len() as f32 / n as f32;
        assert!(
            ground_frac > 0.85,
            "expected >85% ground on a flat scene, got {:.2}%",
            ground_frac * 100.0
        );
    }

    #[test]
    fn obstacles_are_separated() {
        let sensor_h = 1.8;
        let cloud = build_scene(sensor_h, true);
        let result = segment_ground(&cloud, sensor_h).unwrap();
        let ground_z_mean = mean_z(&result.ground.points);
        let nonground_z_mean = mean_z(&result.nonground.points);
        assert!(
            nonground_z_mean > ground_z_mean + 0.3,
            "obstacles should sit above the ground band: ng={:.3} g={:.3}",
            nonground_z_mean,
            ground_z_mean
        );
        // Both classes should be non-empty.
        assert!(result.ground.len() > 0);
        assert!(result.nonground.len() > 0);
        // ground + nonground must equal input.
        assert_eq!(result.ground.len() + result.nonground.len(), cloud.len());
    }

    fn mean_z(pts: &[Point3f]) -> f32 {
        if pts.is_empty() {
            return 0.0;
        }
        pts.iter().map(|p| p.z).sum::<f32>() / pts.len() as f32
    }

    #[test]
    fn empty_cloud_is_handled() {
        let cloud: PointCloud<Point3f> = PointCloud::new();
        let result = segment_ground(&cloud, 1.8).unwrap();
        assert_eq!(result.ground.len(), 0);
        assert_eq!(result.nonground.len(), 0);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let cloud = build_scene(1.8, false);
        let bad = PatchworkConfig {
            zone_radii: vec![0.0, 10.0],
            num_rings_per_zone: vec![2, 2],
            num_sectors_per_zone: vec![8, 8],
            ..Default::default()
        };
        assert!(patchwork_plus_plus(&cloud, bad).is_err());
    }

    #[test]
    fn labels_match_partition() {
        let cloud = build_scene(1.8, true);
        let result = segment_ground(&cloud, 1.8).unwrap();
        assert_eq!(result.labels.len(), cloud.len());
        let ground_count = result.labels.iter().filter(|b| **b).count();
        assert_eq!(ground_count, result.ground.len());
    }
}
