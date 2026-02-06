//! Interactive 3D viewer with UI controls
//! 
//! This module provides a simplified interactive viewer for 3D data

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton},
    event_loop::{EventLoop, ActiveEventLoop},
    window::{Window, WindowId},
    keyboard::Key,
    dpi::PhysicalPosition,
};

use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, ColoredPoint3f, Error};
use threecrate_gpu::{
    PointCloudRenderer, RenderConfig, RenderParams,
    point_cloud_to_instance_data, colored_point_cloud_to_instance_data,
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
    pub fn run(self) -> Result<()> {
        println!("Starting threecrate Interactive Viewer...");

        // Create event loop
        let event_loop = EventLoop::new().map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to create event loop: {}", e))))?;

        // Create application handler
        let mut app = ViewerApp {
            viewer: self,
            window: None,
            point_renderer: None,
            mesh_renderer: None,
        };

        // Run the event loop
        event_loop.run_app(&mut app).map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("Event loop error: {}", e))))?;

        Ok(())
    }
}

impl Default for InteractiveViewer {
    fn default() -> Self {
        Self::new().expect("Failed to create InteractiveViewer")
    }
}

/// Application handler for the viewer
struct ViewerApp {
    viewer: InteractiveViewer,
    window: Option<Arc<Window>>,
    point_renderer: Option<PointCloudRenderer<'static>>,
    mesh_renderer: Option<MeshRenderer<'static>>,
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            println!("Creating window and initializing renderers...");

            // Create window with attributes
            let window_attrs = Window::default_attributes()
                .with_title("threecrate Interactive Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0));

            let window = match event_loop.create_window(window_attrs) {
                Ok(w) => Arc::new(w),
                Err(e) => {
                    eprintln!("Failed to create window: {}", e);
                    event_loop.exit();
                    return;
                }
            };

            // Update camera aspect ratio
            let size = window.inner_size();
            self.viewer.camera.aspect_ratio = size.width as f32 / size.height as f32;

            // Leak the Arc to get a 'static reference
            // This is safe because the window will live for the duration of the program
            let window_ref: &'static Window = unsafe {
                std::mem::transmute::<&Window, &'static Window>(window.as_ref())
            };

            // Initialize renderers: use small point size for instanced rendering
            let mut pc_config = RenderConfig::default();
            pc_config.render_params.point_size = 0.04;
            let point_renderer = match pollster::block_on(PointCloudRenderer::new(window_ref, pc_config)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to create point cloud renderer: {}", e);
                    event_loop.exit();
                    return;
                }
            };

            let mesh_config = MeshRenderConfig::default();
            let mesh_renderer = match pollster::block_on(MeshRenderer::new(window_ref, mesh_config)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to create mesh renderer: {}", e);
                    event_loop.exit();
                    return;
                }
            };

            self.window = Some(window);
            self.point_renderer = Some(point_renderer);
            self.mesh_renderer = Some(mesh_renderer);

            println!("Viewer initialized successfully. Window should now be visible.");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = &self.window else { return; };
        let Some(point_renderer) = &mut self.point_renderer else { return; };
        let Some(mesh_renderer) = &mut self.mesh_renderer else { return; };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                point_renderer.resize(new_size);
                mesh_renderer.resize(new_size);
                self.viewer.camera.aspect_ratio = new_size.width as f32 / new_size.height as f32;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        self.viewer.mouse_pressed = state == ElementState::Pressed;
                    }
                    MouseButton::Right => {
                        self.viewer.right_mouse_pressed = state == ElementState::Pressed;
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(last_pos) = self.viewer.last_mouse_pos {
                    let delta_x = position.x - last_pos.x;
                    let delta_y = position.y - last_pos.y;

                    if self.viewer.mouse_pressed {
                        match self.viewer.camera_mode {
                            CameraMode::Orbit => {
                                self.viewer.camera.orbit(delta_x as f32 * 0.01, delta_y as f32 * 0.01);
                            }
                            CameraMode::Pan => {
                                self.viewer.camera.pan(delta_x as f32 * 0.01, delta_y as f32 * 0.01);
                            }
                            _ => {}
                        }
                    }
                }
                self.viewer.last_mouse_pos = Some(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                };
                self.viewer.camera.zoom(scroll_delta * 0.1);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match &event.logical_key {
                        Key::Character(c) => {
                            match c.as_str() {
                                "o" | "O" => {
                                    self.viewer.camera_mode = CameraMode::Orbit;
                                    println!("Switched to Orbit mode");
                                }
                                "p" | "P" => {
                                    self.viewer.camera_mode = CameraMode::Pan;
                                    println!("Switched to Pan mode");
                                }
                                "z" | "Z" => {
                                    self.viewer.camera_mode = CameraMode::Zoom;
                                    println!("Switched to Zoom mode");
                                }
                                "r" | "R" => {
                                    self.viewer.camera.reset();
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
                let view_matrix = self.viewer.camera.view_matrix();
                let proj_matrix = self.viewer.camera.projection_matrix();
                let camera_pos = self.viewer.camera.position.coords;
                point_renderer.update_camera(view_matrix, proj_matrix, camera_pos);
                mesh_renderer.update_camera(view_matrix, proj_matrix, camera_pos);

                // Convert point cloud to instance data for instanced rendering
                let instance_data = match &self.viewer.current_data {
                    ViewData::PointCloud(cloud) => {
                        point_cloud_to_instance_data(cloud, [1.0, 1.0, 1.0])
                    }
                    ViewData::ColoredPointCloud(cloud) => {
                        colored_point_cloud_to_instance_data(cloud)
                    }
                    ViewData::Mesh(_) | ViewData::Empty => Vec::new(),
                };

                if self.viewer.debug_frame_count % 60 == 0 && !instance_data.is_empty() {
                    println!("Rendering {} points (instanced)", instance_data.len());
                }
                self.viewer.debug_frame_count += 1;

                match &self.viewer.current_data {
                    ViewData::PointCloud(_) | ViewData::ColoredPointCloud(_) => {
                        if !instance_data.is_empty() {
                            if let Err(e) = point_renderer.render_instanced(&instance_data) {
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

