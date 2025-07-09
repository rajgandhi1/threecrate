//! Camera utilities for 3D visualization

use nalgebra::{Point3, Vector3, Matrix4, Perspective3};
use std::f32::consts::PI;

/// A 3D camera for viewing point clouds and meshes
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub fov: f32,
    pub fovy_degrees: f32,  // FOV in degrees for UI display
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
    
    // Spherical coordinates for orbit control
    pub radius: f32,
    pub theta: f32,  // Azimuth angle
    pub phi: f32,    // Polar angle
    
    // Default position for reset
    default_position: Point3<f32>,
    default_target: Point3<f32>,
    default_up: Vector3<f32>,
}

impl Camera {
    /// Create a new camera
    pub fn new(
        position: Point3<f32>,
        target: Point3<f32>,
        up: Vector3<f32>,
        fovy_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let direction = position - target;
        let radius = direction.magnitude();
        let theta = direction.z.atan2(direction.x);
        let phi = (direction.y / radius).asin();
        let fov = fovy_degrees.to_radians();
        
        Self {
            position,
            target,
            up,
            fov,
            fovy_degrees,
            aspect_ratio,
            near,
            far,
            radius,
            theta,
            phi,
            default_position: position,
            default_target: target,
            default_up: up,
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

    /// Move the camera forward/backward
    pub fn move_forward(&mut self, distance: f32) {
        self.radius = (self.radius - distance).max(0.1);
        self.update_position_from_spherical();
    }

    /// Rotate the camera around the target (orbit control)
    pub fn orbit(&mut self, delta_theta: f32, delta_phi: f32) {
        self.theta += delta_theta;
        self.phi = (self.phi + delta_phi).clamp(-PI/2.0 + 0.1, PI/2.0 - 0.1);
        self.update_position_from_spherical();
    }

    /// Pan the camera (translate both position and target)
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        let up = right.cross(&forward).normalize();
        
        let pan_vector = right * delta_x + up * delta_y;
        self.position += pan_vector;
        self.target += pan_vector;
    }

    /// Zoom by changing the field of view
    pub fn zoom_fov(&mut self, delta: f32) {
        self.fov = (self.fov + delta).clamp(0.1, PI - 0.1);
        self.fovy_degrees = self.fov.to_degrees();
    }

    /// Zoom by moving closer/farther from target
    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius - delta).max(0.1);
        self.update_position_from_spherical();
    }

    /// Reset camera to default position
    pub fn reset(&mut self) {
        self.position = self.default_position;
        self.target = self.default_target;
        self.up = self.default_up;
        
        let direction = self.position - self.target;
        self.radius = direction.magnitude();
        self.theta = direction.z.atan2(direction.x);
        self.phi = (direction.y / self.radius).asin();
    }

    /// Update position based on spherical coordinates
    fn update_position_from_spherical(&mut self) {
        let x = self.radius * self.phi.cos() * self.theta.cos();
        let y = self.radius * self.phi.sin();
        let z = self.radius * self.phi.cos() * self.theta.sin();
        
        self.position = self.target + Vector3::new(x, y, z);
    }

    /// Reset camera to default position looking at target
    pub fn reset_to_target(&mut self, target: Point3<f32>, radius: f32) {
        self.target = target;
        self.radius = radius;
        self.theta = 0.0;
        self.phi = 0.0;
        self.update_position_from_spherical();
    }

    /// Get the forward direction vector
    pub fn forward(&self) -> Vector3<f32> {
        (self.target - self.position).normalize()
    }

    /// Get the right direction vector
    pub fn right(&self) -> Vector3<f32> {
        self.forward().cross(&self.up).normalize()
    }

    /// Get the up direction vector (camera local up)
    pub fn camera_up(&self) -> Vector3<f32> {
        self.right().cross(&self.forward()).normalize()
    }

    /// Set aspect ratio
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    /// Get camera distance from target
    pub fn distance_to_target(&self) -> f32 {
        (self.position - self.target).magnitude()
    }

    /// Set field of view in degrees
    pub fn set_fov_degrees(&mut self, fovy_degrees: f32) {
        self.fovy_degrees = fovy_degrees.clamp(1.0, 179.0);
        self.fov = self.fovy_degrees.to_radians();
    }

    /// Get field of view in degrees
    pub fn fov_degrees(&self) -> f32 {
        self.fovy_degrees
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            45.0,  // 45 degrees FOV
            16.0 / 9.0,
            0.1,
            100.0,
        )
    }
} 