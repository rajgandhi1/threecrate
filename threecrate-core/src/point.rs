//! Point types and related functionality

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};

/// A 3D point with floating point coordinates
pub type Point3f = Point3<f32>;

/// A 3D point with double precision coordinates
pub type Point3d = Point3<f64>;

/// A 3D vector with floating point components
pub type Vector3f = Vector3<f32>;

/// A 3D vector with double precision components
pub type Vector3d = Vector3<f64>;

/// A point with color information
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct ColoredPoint3f {
    pub position: Point3f,
    pub color: [u8; 3],
}

unsafe impl Pod for ColoredPoint3f {}
unsafe impl Zeroable for ColoredPoint3f {}

/// A point with normal vector
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct NormalPoint3f {
    pub position: Point3f,
    pub normal: Vector3f,
}

unsafe impl Pod for NormalPoint3f {}
unsafe impl Zeroable for NormalPoint3f {}

/// A point with color and normal information
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct ColoredNormalPoint3f {
    pub position: Point3f,
    pub normal: Vector3f,
    pub color: [u8; 3],
}

unsafe impl Pod for ColoredNormalPoint3f {}
unsafe impl Zeroable for ColoredNormalPoint3f {}

impl Default for ColoredPoint3f {
    fn default() -> Self {
        Self {
            position: Point3f::origin(),
            color: [255, 255, 255],
        }
    }
}

impl Default for NormalPoint3f {
    fn default() -> Self {
        Self {
            position: Point3f::origin(),
            normal: Vector3f::new(0.0, 0.0, 1.0),
        }
    }
}

impl Default for ColoredNormalPoint3f {
    fn default() -> Self {
        Self {
            position: Point3f::origin(),
            normal: Vector3f::new(0.0, 0.0, 1.0),
            color: [255, 255, 255],
        }
    }
} 