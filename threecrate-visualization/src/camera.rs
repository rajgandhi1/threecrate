//! Camera utilities for 3D visualization

use nalgebra::{Point3, Vector3, Matrix4, Perspective3};

/// A 3D camera for viewing point clouds and meshes
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// Create a new camera
    pub fn new(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        fov: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            target,
            up,
            fov,
            aspect_ratio,
            near,
            far,
        }
    }

    /// Get the view matrix
    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.position, &self.target, &self.up)
    }

    /// Get the projection matrix
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        let perspective = Perspective3::new(self.aspect_ratio, self.fov, self.near, self.far);
        perspective.into_inner()
    }

    /// Move the camera forward
    pub fn move_forward(&mut self, distance: f32) {
        let direction = (self.target - self.position).normalize();
        self.position += direction * distance;
        self.target += direction * distance;
    }

    /// Rotate the camera around the target
    pub fn orbit(&mut self, _horizontal: f32, _vertical: f32) {
        let radius = (self.position - self.target).norm();
        let direction = (self.position - self.target).normalize();
        
        // Simple orbit implementation - could be improved
        let new_direction = direction; // Placeholder for actual rotation
        self.position = self.target + new_direction * radius;
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            100.0,
        )
    }
} 