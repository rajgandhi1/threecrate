//! Point cloud data structures and functionality

use crate::point::*;
use crate::transform::Transform3D;
// use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

/// A generic point cloud container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud<T> {
    pub points: Vec<T>,
}

/// A point cloud with 3D points
pub type PointCloud3f = PointCloud<Point3f>;

/// A point cloud with colored points
pub type ColoredPointCloud3f = PointCloud<ColoredPoint3f>;

/// A point cloud with normal vectors
pub type NormalPointCloud3f = PointCloud<NormalPoint3f>;

/// A point cloud with colors and normals
pub type ColoredNormalPointCloud3f = PointCloud<ColoredNormalPoint3f>;

impl<T> PointCloud<T> {
    /// Create a new empty point cloud
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
        }
    }

    /// Create a new point cloud with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Create a point cloud from a vector of points
    pub fn from_points(points: Vec<T>) -> Self {
        Self { points }
    }

    /// Get the number of points in the cloud
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the point cloud is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Add a point to the cloud
    pub fn push(&mut self, point: T) {
        self.points.push(point);
    }

    /// Get an iterator over the points
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.points.iter()
    }

    /// Get a mutable iterator over the points
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.points.iter_mut()
    }

    /// Clear all points from the cloud
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Reserve capacity for additional points
    pub fn reserve(&mut self, additional: usize) {
        self.points.reserve(additional);
    }
}

impl<T> Default for PointCloud<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Index<usize> for PointCloud<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl<T> IndexMut<usize> for PointCloud<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.points[index]
    }
}

impl<T> IntoIterator for PointCloud<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a PointCloud<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut PointCloud<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.iter_mut()
    }
}

impl<T> Extend<T> for PointCloud<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.points.extend(iter);
    }
}

impl<T> FromIterator<T> for PointCloud<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            points: Vec::from_iter(iter),
        }
    }
}

impl PointCloud<Point3f> {
    /// Apply a transformation to all points in the cloud
    pub fn transform(&mut self, transform: &Transform3D) {
        for point in &mut self.points {
            *point = transform.transform_point(point);
        }
    }
} 