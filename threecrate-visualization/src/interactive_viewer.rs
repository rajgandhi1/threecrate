//! Interactive 3D viewer with UI controls
//! 
//! This module provides a simplified interactive viewer for 3D data

use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent, ElementState, MouseButton},
    event_loop::{EventLoop, ControlFlow},
    window::WindowBuilder,
    keyboard::Key,
    dpi::PhysicalPosition,
};

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, ColoredPoint3f, Error};
use threecrate_gpu::{
    PointCloudRenderer, RenderConfig, PointVertex,
    MeshRenderer, MeshRenderConfig, ShadingMode, mesh_to_gpu_mesh,
};
use threecrate_algorithms::{ICPResult, PlaneSegmentationResult};
use crate::camera::Camera;

use nalgebra::{Vector3, Point3};

/// Types of data that can be displayed
#[derive(Debug, Clone)]
pub enum ViewData {
    Empty,
    PointCloud(PointCloud<Point3f>),
    ColoredPointCloud(PointCloud<ColoredPoint3f>),
    Mesh(TriangleMesh),
}

/// Camera control modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMode {
    Orbit,
    Pan,
    Zoom,
}

/// Pipeline processing type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineType {
    Cpu,
    Gpu,
}

/// ICP algorithm parameters
#[derive(Debug, Clone)]
pub struct ICPParams {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub max_correspondence_distance: f32,
}

impl Default for ICPParams {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 0.001,
            max_correspondence_distance: 1.0,
        }
    }
}

/// RANSAC algorithm parameters
#[derive(Debug, Clone)]
pub struct RANSACParams {
    pub max_iterations: usize,
    pub distance_threshold: f32,
}

impl Default for RANSACParams {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            distance_threshold: 0.1,
        }
    }
}

/// UI state for all panels and controls (kept for future use)
#[derive(Debug)]
pub struct UIState {
    pub render_panel_open: bool,
    pub algorithm_panel_open: bool,
    pub camera_panel_open: bool,
    pub stats_panel_open: bool,
    pub icp_params: ICPParams,
    pub ransac_params: RANSACParams,
    pub source_cloud: Option<PointCloud<Point3f>>,
    pub target_cloud: Option<PointCloud<Point3f>>,
    pub icp_result: Option<ICPResult>,
    pub ransac_result: Option<PlaneSegmentationResult>,
}

impl Default for UIState {
    fn default() -> Self {
        Self {
            render_panel_open: false,
            algorithm_panel_open: false,
            camera_panel_open: false,
            stats_panel_open: false,
            icp_params: ICPParams::default(),
            ransac_params: RANSACParams::default(),
            source_cloud: None,
            target_cloud: None,
            icp_result: None,
            ransac_result: None,
        }
    }
}

/// Interactive 3D viewer with comprehensive UI controls
pub struct InteractiveViewer {
    current_data: ViewData,
    camera: Camera,
    camera_mode: CameraMode,
    last_mouse_pos: Option<PhysicalPosition<f64>>,
    mouse_pressed: bool,
    right_mouse_pressed: bool,
    debug_frame_count: usize,
}

impl InteractiveViewer {
    /// Create a new interactive viewer
    pub fn new() -> Result<Self> {
        let camera = Camera::new(
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            45.0,
            1.0,
            0.1,
            100.0,
        );

        Ok(Self {
            current_data: ViewData::Empty,
            camera,
            camera_mode: CameraMode::Orbit,
            last_mouse_pos: None,
            mouse_pressed: false,
            right_mouse_pressed: false,
            debug_frame_count: 0,
        })
    }

    /// Set point cloud data
    pub fn set_point_cloud(&mut self, cloud: &PointCloud<Point3f>) {
        self.current_data = ViewData::PointCloud(cloud.clone());
        println!("Set point cloud with {} points", cloud.len());
    }

    /// Set colored point cloud data
    pub fn set_colored_point_cloud(&mut self, cloud: &PointCloud<ColoredPoint3f>) {
        self.current_data = ViewData::ColoredPointCloud(cloud.clone());
        println!("Set colored point cloud with {} points", cloud.len());
    }

    /// Set mesh data
    pub fn set_mesh(&mut self, mesh: &TriangleMesh) {
        self.current_data = ViewData::Mesh(mesh.clone());
        println!("Set mesh with {} vertices and {} faces", mesh.vertices.len(), mesh.faces.len());
    }

    /// Run the interactive viewer
    pub fn run(mut self) -> Result<()> {
        println!("Starting threecrate Interactive Viewer...");
        
        // Create event loop and window
        let event_loop = EventLoop::new().map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to create event loop: {}", e))))?;
        let window = Arc::new(
            WindowBuilder::new()
                .with_title("threecrate Interactive Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0))
                .build(&event_loop)
                .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to create window: {}", e))))?
        );

        // Initialize renderers
        let pc_config = RenderConfig::default();
        let window_clone = window.clone();
        let mut point_renderer = pollster::block_on(PointCloudRenderer::new(&window_clone, pc_config))?;

        let mesh_config = MeshRenderConfig::default();
        let mut mesh_renderer = pollster::block_on(MeshRenderer::new(&window_clone, mesh_config))?;

        // Update camera aspect ratio
        let size = window.inner_size();
        self.camera.aspect_ratio = size.width as f32 / size.height as f32;

        println!("Viewer initialized successfully. Window should now be visible.");

        // Main event loop
        event_loop.run(move |event, target| {
            target.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::CloseRequested => {
                            target.exit();
                        }
                        WindowEvent::Resized(new_size) => {
                            point_renderer.resize(new_size);
                            mesh_renderer.resize(new_size);
                            self.camera.aspect_ratio = new_size.width as f32 / new_size.height as f32;
                        }
                        WindowEvent::MouseInput { state, button, .. } => {
                            match button {
                                MouseButton::Left => {
                                    self.mouse_pressed = state == ElementState::Pressed;
                                }
                                MouseButton::Right => {
                                    self.right_mouse_pressed = state == ElementState::Pressed;
                                }
                                _ => {}
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            if let Some(last_pos) = self.last_mouse_pos {
                                let delta_x = position.x - last_pos.x;
                                let delta_y = position.y - last_pos.y;

                                if self.mouse_pressed {
                                    match self.camera_mode {
                                        CameraMode::Orbit => {
                                            self.camera.orbit(delta_x as f32 * 0.01, delta_y as f32 * 0.01);
                                        }
                                        CameraMode::Pan => {
                                            self.camera.pan(delta_x as f32 * 0.01, delta_y as f32 * 0.01);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            self.last_mouse_pos = Some(position);
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            let scroll_delta = match delta {
                                winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                                winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                            };
                            self.camera.zoom(scroll_delta * 0.1);
                        }
                        WindowEvent::KeyboardInput { event, .. } => {
                            if event.state == ElementState::Pressed {
                                match &event.logical_key {
                                    Key::Character(c) => {
                                        match c.as_str() {
                                            "o" | "O" => {
                                                self.camera_mode = CameraMode::Orbit;
                                                println!("Switched to Orbit mode");
                                            }
                                            "p" | "P" => {
                                                self.camera_mode = CameraMode::Pan;
                                                println!("Switched to Pan mode");
                                            }
                                            "z" | "Z" => {
                                                self.camera_mode = CameraMode::Zoom;
                                                println!("Switched to Zoom mode");
                                            }
                                            "r" | "R" => {
                                                self.camera.reset();
                                                println!("Reset camera");
                                            }
                                            _ => {}
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            // Update camera matrices
                            let view_matrix = self.camera.view_matrix();
                            let proj_matrix = self.camera.projection_matrix();
                            let camera_pos = self.camera.position.coords;
                            point_renderer.update_camera(view_matrix, proj_matrix, camera_pos);
                            mesh_renderer.update_camera(view_matrix, proj_matrix, camera_pos);

                            // Convert current data
                            let vertices = match &self.current_data {
                                ViewData::PointCloud(cloud) => {
                                    let mut vertices = Vec::new();
                                    for point in cloud.iter() {
                                        // Create a quad (2 triangles) for each point
                                        let size = 0.02; // Size of each point quad
                                        let pos = [point.x, point.y, point.z];
                                        let color = [1.0, 1.0, 1.0];
                                        let normal = [0.0, 0.0, 1.0];
                                        
                                        // Create 4 vertices for a quad
                                        let v1 = PointVertex::from_point(&Point3f::new(pos[0] - size, pos[1] - size, pos[2]), color, 16.0, normal);
                                        let v2 = PointVertex::from_point(&Point3f::new(pos[0] + size, pos[1] - size, pos[2]), color, 16.0, normal);
                                        let v3 = PointVertex::from_point(&Point3f::new(pos[0] + size, pos[1] + size, pos[2]), color, 16.0, normal);
                                        let v4 = PointVertex::from_point(&Point3f::new(pos[0] - size, pos[1] + size, pos[2]), color, 16.0, normal);
                                        
                                        // First triangle (v1, v2, v3)
                                        vertices.push(v1);
                                        vertices.push(v2);
                                        vertices.push(v3);
                                        
                                        // Second triangle (v1, v3, v4)
                                        vertices.push(v1);
                                        vertices.push(v3);
                                        vertices.push(v4);
                                    }
                                    vertices
                                }
                                ViewData::ColoredPointCloud(cloud) => {
                                    let mut vertices = Vec::new();
                                    for point in cloud.iter() {
                                        // Create a quad (2 triangles) for each point
                                        let size = 0.02; // Size of each point quad
                                        let pos = [point.position.x, point.position.y, point.position.z];
                                        let color = [
                                            point.color[0] as f32 / 255.0,
                                            point.color[1] as f32 / 255.0,
                                            point.color[2] as f32 / 255.0,
                                        ];
                                        let normal = [0.0, 0.0, 1.0];
                                        
                                        // Create 4 vertices for a quad
                                        let v1 = PointVertex::from_point(&Point3f::new(pos[0] - size, pos[1] - size, pos[2]), color, 16.0, normal);
                                        let v2 = PointVertex::from_point(&Point3f::new(pos[0] + size, pos[1] - size, pos[2]), color, 16.0, normal);
                                        let v3 = PointVertex::from_point(&Point3f::new(pos[0] + size, pos[1] + size, pos[2]), color, 16.0, normal);
                                        let v4 = PointVertex::from_point(&Point3f::new(pos[0] - size, pos[1] + size, pos[2]), color, 16.0, normal);
                                        
                                        // First triangle (v1, v2, v3)
                                        vertices.push(v1);
                                        vertices.push(v2);
                                        vertices.push(v3);
                                        
                                        // Second triangle (v1, v3, v4)
                                        vertices.push(v1);
                                        vertices.push(v3);
                                        vertices.push(v4);
                                    }
                                    vertices
                                }
                                ViewData::Mesh(_) => {
                                    // Mesh rendering handled separately
                                    vec![]
                                }
                                ViewData::Empty => {
                                    println!("No data to render - ViewData is Empty");
                                    vec![]
                                }
                            };

                            // Debug: Print vertex count periodically
                            if !vertices.is_empty() {
                                if self.debug_frame_count % 60 == 0 {  // Print every 60 frames
                                    println!("Rendering {} vertices", vertices.len());
                                }
                            }
                            self.debug_frame_count += 1;

                            match &self.current_data {
                                ViewData::PointCloud(_) | ViewData::ColoredPointCloud(_) => {
                                    if !vertices.is_empty() {
                                        if let Err(e) = point_renderer.render(&vertices) {
                                            eprintln!("Render error: {}", e);
                                        }
                                    }
                                }
                                ViewData::Mesh(mesh) => {
                                    if !mesh.vertices.is_empty() && !mesh.faces.is_empty() {
                                        // Build index buffer from faces
                                        let indices: Vec<u32> = mesh
                                            .faces
                                            .iter()
                                            .flat_map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
                                            .collect();

                                        // Prepare optional normals
                                        let normals_opt = mesh.normals.as_ref().map(|n| n.as_slice());

                                        // Prepare optional colors as f32
                                        let colors_f32: Option<Vec<[f32; 3]>> = mesh.colors.as_ref().map(|cols| {
                                            cols.iter()
                                                .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
                                                .collect()
                                        });
                                        let colors_opt = colors_f32.as_ref().map(|c| c.as_slice());

                                        // Convert to GPU mesh
                                        let gpu_mesh = mesh_to_gpu_mesh(
                                            &mesh.vertices,
                                            &indices,
                                            normals_opt,
                                            colors_opt,
                                            None,
                                        );

                                        if let Err(e) = mesh_renderer.render(&gpu_mesh, ShadingMode::Flat) {
                                            eprintln!("Mesh render error: {}", e);
                                        }
                                    }
                                }
                                ViewData::Empty => {}
                            }

                            // Request next frame
                            window.request_redraw();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }).map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Event loop error: {}", e))))?;

        Ok(())
    }
}

impl Default for InteractiveViewer {
    fn default() -> Self {
        Self::new().expect("Failed to create InteractiveViewer")
    }
}

 