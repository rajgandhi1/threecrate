//! Organized (grid-structured) point clouds.
//!
//! Most LiDAR sensors and depth cameras produce points laid out in a 2D
//! `width × height` grid: rotating LiDARs give one row per laser ring, depth
//! cameras give one row per pixel row. Preserving that structure makes a
//! number of algorithms trivial — scan-line normal estimation, range-image
//! segmentation, ring-based ground segmentation, O(1) neighbor lookup by
//! `(row±1, col±1)`. This module provides [`OrganizedPointCloud`], a
//! row-major grid of `Option<T>` where `None` represents missing/invalid
//! returns (e.g. NaN points in a `sensor_msgs/PointCloud2`).
//!
//! `OrganizedPointCloud` is additive — the existing [`PointCloud`] remains
//! the primary container for unorganized data, and you can drop down to it
//! via [`OrganizedPointCloud::to_unorganized`].

use crate::point::Point3f;
use crate::point_cloud::PointCloud;
use serde::{Deserialize, Serialize};

/// A point cloud whose points sit in a `width × height` grid in row-major order.
///
/// Cells holding `None` represent missing returns. Iteration via
/// [`OrganizedPointCloud::iter_valid`] yields only the populated cells.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizedPointCloud<T> {
    /// Points per row.
    pub width: usize,
    /// Number of rows (LiDAR rings or image rows).
    pub height: usize,
    /// Row-major storage of length `width * height`.
    pub points: Vec<Option<T>>,
    /// `true` iff every cell is `Some`. Mirrors `sensor_msgs/PointCloud2::is_dense`.
    pub is_dense: bool,
}

/// Pinhole camera intrinsics for depth-image projection.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    /// Focal length in pixels (x).
    pub fx: f32,
    /// Focal length in pixels (y).
    pub fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
}

impl CameraIntrinsics {
    /// Construct from four values.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self { fx, fy, cx, cy }
    }
}

impl<T> OrganizedPointCloud<T> {
    /// Create an empty organized cloud with the given dimensions; every cell
    /// starts as `None` and `is_dense = false`.
    pub fn new(width: usize, height: usize) -> Self
    where
        T: Clone,
    {
        let mut points = Vec::with_capacity(width * height);
        points.resize_with(width * height, || None);
        Self { width, height, points, is_dense: false }
    }

    /// Construct from a flat row-major buffer of `Option<T>`.
    ///
    /// Returns `None` if `points.len() != width * height`.
    pub fn from_points(width: usize, height: usize, points: Vec<Option<T>>) -> Option<Self> {
        if points.len() != width * height {
            return None;
        }
        let is_dense = points.iter().all(|p| p.is_some());
        Some(Self { width, height, points, is_dense })
    }

    /// Total number of grid cells (`width * height`), including empty ones.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `true` iff the grid has no cells.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Number of `Some` cells.
    pub fn valid_count(&self) -> usize {
        self.points.iter().filter(|p| p.is_some()).count()
    }

    /// Convert a `(row, col)` pair into a linear row-major index.
    #[inline]
    fn idx(&self, row: usize, col: usize) -> Option<usize> {
        if row < self.height && col < self.width {
            Some(row * self.width + col)
        } else {
            None
        }
    }

    /// Borrow the point at `(row, col)`, if any. Returns `None` for both
    /// out-of-bounds indices and empty cells.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.idx(row, col).and_then(|i| self.points[i].as_ref())
    }

    /// Mutable borrow of the point at `(row, col)`.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        let i = self.idx(row, col)?;
        self.points[i].as_mut()
    }

    /// Set a point at `(row, col)`. Returns `false` if the indices are out of bounds.
    /// Updates `is_dense` conservatively (set to `false` whenever a `None` is written).
    pub fn set(&mut self, row: usize, col: usize, value: Option<T>) -> bool {
        let Some(i) = self.idx(row, col) else { return false };
        if value.is_none() {
            self.is_dense = false;
        }
        self.points[i] = value;
        true
    }

    /// Slice of all cells in the given row (out-of-bounds → empty slice).
    pub fn row(&self, row: usize) -> &[Option<T>] {
        if row >= self.height {
            return &[];
        }
        let start = row * self.width;
        &self.points[start..start + self.width]
    }

    /// Alias for [`row`] — LiDAR users typically call rows "rings".
    pub fn ring(&self, ring_index: usize) -> &[Option<T>] {
        self.row(ring_index)
    }

    /// Iterator over `(row, col, &T)` for all populated cells.
    pub fn iter_valid(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        let width = self.width;
        self.points.iter().enumerate().filter_map(move |(i, opt)| {
            opt.as_ref().map(|p| (i / width, i % width, p))
        })
    }

    /// Recompute `is_dense` from the current cell contents.
    pub fn refresh_dense_flag(&mut self) {
        self.is_dense = self.points.iter().all(|p| p.is_some());
    }
}

impl<T: Clone> OrganizedPointCloud<T> {
    /// Drop `None` cells and return an unorganized cloud preserving point order.
    pub fn to_unorganized(&self) -> PointCloud<T> {
        let pts: Vec<T> = self.points.iter().filter_map(|p| p.clone()).collect();
        PointCloud::from_points(pts)
    }
}

impl OrganizedPointCloud<Point3f> {
    /// Build an organized cloud by back-projecting a row-major depth image
    /// through pinhole intrinsics. Pixels with a depth of `0` (or NaN after
    /// scaling) are stored as `None`.
    ///
    /// Returns `None` if `depth.len() != width * height`.
    pub fn from_depth_image(
        depth: &[u16],
        width: usize,
        height: usize,
        intrinsics: &CameraIntrinsics,
        depth_scale: f32,
    ) -> Option<Self> {
        if depth.len() != width * height {
            return None;
        }
        let mut points: Vec<Option<Point3f>> = Vec::with_capacity(width * height);
        let mut any_invalid = false;
        for row in 0..height {
            for col in 0..width {
                let d = depth[row * width + col];
                if d == 0 {
                    points.push(None);
                    any_invalid = true;
                    continue;
                }
                let z = d as f32 * depth_scale;
                if !z.is_finite() || z <= 0.0 {
                    points.push(None);
                    any_invalid = true;
                    continue;
                }
                let x = (col as f32 - intrinsics.cx) * z / intrinsics.fx;
                let y = (row as f32 - intrinsics.cy) * z / intrinsics.fy;
                points.push(Some(Point3f::new(x, y, z)));
            }
        }
        Some(Self { width, height, points, is_dense: !any_invalid })
    }
}

impl<T> Default for OrganizedPointCloud<T> {
    fn default() -> Self {
        Self { width: 0, height: 0, points: Vec::new(), is_dense: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_grid_is_empty_and_not_dense() {
        let cloud: OrganizedPointCloud<Point3f> = OrganizedPointCloud::new(4, 3);
        assert_eq!(cloud.width, 4);
        assert_eq!(cloud.height, 3);
        assert_eq!(cloud.len(), 12);
        assert_eq!(cloud.valid_count(), 0);
        assert!(!cloud.is_dense);
    }

    #[test]
    fn get_set_round_trip() {
        let mut cloud: OrganizedPointCloud<Point3f> = OrganizedPointCloud::new(3, 2);
        assert!(cloud.set(1, 2, Some(Point3f::new(1.0, 2.0, 3.0))));
        let p = cloud.get(1, 2).unwrap();
        assert_eq!(*p, Point3f::new(1.0, 2.0, 3.0));
        // Out of bounds:
        assert!(cloud.get(2, 0).is_none());
        assert!(!cloud.set(5, 5, Some(Point3f::new(0.0, 0.0, 0.0))));
    }

    #[test]
    fn row_alias_ring() {
        let mut cloud: OrganizedPointCloud<Point3f> = OrganizedPointCloud::new(3, 2);
        cloud.set(0, 0, Some(Point3f::new(0.0, 0.0, 0.0)));
        cloud.set(0, 1, Some(Point3f::new(1.0, 0.0, 0.0)));
        cloud.set(0, 2, Some(Point3f::new(2.0, 0.0, 0.0)));
        let r = cloud.row(0);
        assert_eq!(r.len(), 3);
        assert!(r.iter().all(|p| p.is_some()));
        let ring = cloud.ring(0);
        assert_eq!(ring.len(), 3);
        // Out of bounds row → empty slice.
        assert_eq!(cloud.row(99).len(), 0);
    }

    #[test]
    fn to_unorganized_drops_none() {
        let mut cloud: OrganizedPointCloud<Point3f> = OrganizedPointCloud::new(2, 2);
        cloud.set(0, 0, Some(Point3f::new(1.0, 0.0, 0.0)));
        cloud.set(1, 1, Some(Point3f::new(0.0, 1.0, 0.0)));
        let flat = cloud.to_unorganized();
        assert_eq!(flat.len(), 2);
    }

    #[test]
    fn from_points_dense_flag() {
        let pts = vec![
            Some(Point3f::new(0.0, 0.0, 0.0)),
            Some(Point3f::new(1.0, 0.0, 0.0)),
            Some(Point3f::new(0.0, 1.0, 0.0)),
            Some(Point3f::new(0.0, 0.0, 1.0)),
        ];
        let cloud = OrganizedPointCloud::from_points(2, 2, pts).unwrap();
        assert!(cloud.is_dense);
        assert_eq!(cloud.valid_count(), 4);

        let bad = OrganizedPointCloud::<Point3f>::from_points(2, 2, vec![None; 3]);
        assert!(bad.is_none());
    }

    #[test]
    fn from_depth_image_basic() {
        // 2×2 depth image with one missing pixel.
        let depth: Vec<u16> = vec![1000, 0, 2000, 1500];
        let intr = CameraIntrinsics::new(525.0, 525.0, 0.5, 0.5);
        let cloud = OrganizedPointCloud::from_depth_image(&depth, 2, 2, &intr, 0.001).unwrap();
        assert_eq!(cloud.width, 2);
        assert_eq!(cloud.height, 2);
        assert!(!cloud.is_dense);
        // Pixel (0, 1) had depth 0 → None.
        assert!(cloud.get(0, 1).is_none());
        // Pixel (0, 0) depth 1.0 m, x = (0-0.5)*1/525, y = (0-0.5)*1/525, z = 1.0
        let p = cloud.get(0, 0).unwrap();
        assert!((p.z - 1.0).abs() < 1e-6);
        assert!((p.x + 0.5 / 525.0).abs() < 1e-6);
    }

    #[test]
    fn iter_valid_yields_indexed_points() {
        let mut cloud: OrganizedPointCloud<Point3f> = OrganizedPointCloud::new(2, 2);
        cloud.set(1, 0, Some(Point3f::new(7.0, 0.0, 0.0)));
        let collected: Vec<(usize, usize)> = cloud.iter_valid().map(|(r, c, _)| (r, c)).collect();
        assert_eq!(collected, vec![(1, 0)]);
    }
}
