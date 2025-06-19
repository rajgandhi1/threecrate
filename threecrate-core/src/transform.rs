//! 3D transformation utilities

use nalgebra::{Point3, Vector3, Matrix4, Isometry3, Transform3, UnitQuaternion};
use serde::{Deserialize, Serialize};

/// A 3D transformation that can be applied to points and point clouds
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform3D {
    pub matrix: Matrix4<f32>,
}

impl Transform3D {
    /// Create an identity transformation
    pub fn identity() -> Self {
        Self {
            matrix: Matrix4::identity(),
        }
    }

    /// Create a translation transformation
    pub fn translation(translation: Vector3<f32>) -> Self {
        Self {
            matrix: Matrix4::new_translation(&translation),
        }
    }

    /// Create a rotation transformation from a quaternion
    pub fn rotation(rotation: UnitQuaternion<f32>) -> Self {
        Self {
            matrix: rotation.to_homogeneous(),
        }
    }

    /// Create a scaling transformation
    pub fn scaling(scale: Vector3<f32>) -> Self {
        Self {
            matrix: Matrix4::new_nonuniform_scaling(&scale),
        }
    }

    /// Create a uniform scaling transformation
    pub fn uniform_scaling(scale: f32) -> Self {
        Self {
            matrix: Matrix4::new_scaling(scale),
        }
    }

    /// Create a transformation from translation and rotation
    pub fn from_translation_rotation(
        translation: Vector3<f32>,
        rotation: UnitQuaternion<f32>,
    ) -> Self {
        let isometry = Isometry3::from_parts(translation.into(), rotation);
        Self {
            matrix: isometry.to_homogeneous(),
        }
    }

    /// Apply the transformation to a point
    pub fn transform_point(&self, point: &Point3<f32>) -> Point3<f32> {
        let homogeneous = self.matrix * point.to_homogeneous();
        Point3::from_homogeneous(homogeneous).unwrap_or(*point)
    }

    /// Apply the transformation to a vector
    pub fn transform_vector(&self, vector: &Vector3<f32>) -> Vector3<f32> {
        let homogeneous = self.matrix.fixed_view::<3, 3>(0, 0) * vector;
        homogeneous
    }

    /// Compose this transformation with another
    pub fn compose(self, other: Self) -> Self {
        Self {
            matrix: self.matrix * other.matrix,
        }
    }

    /// Get the inverse transformation
    pub fn inverse(self) -> Option<Self> {
        self.matrix.try_inverse().map(|inv_matrix| Self {
            matrix: inv_matrix,
        })
    }

    /// Check if this is approximately the identity transformation
    pub fn is_identity(&self, epsilon: f32) -> bool {
        let identity = Matrix4::identity();
        (self.matrix - identity).norm() < epsilon
    }
}

impl Default for Transform3D {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::ops::Mul for Transform3D {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(rhs)
    }
}

impl From<Matrix4<f32>> for Transform3D {
    fn from(matrix: Matrix4<f32>) -> Self {
        Self { matrix }
    }
}

impl From<Isometry3<f32>> for Transform3D {
    fn from(isometry: Isometry3<f32>) -> Self {
        Self {
            matrix: isometry.to_homogeneous(),
        }
    }
}

impl From<Transform3<f32>> for Transform3D {
    fn from(transform: Transform3<f32>) -> Self {
        Self {
            matrix: transform.into_inner(),
        }
    }
} 