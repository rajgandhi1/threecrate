//! GPU Mesh Rendering Example

use std::sync::Arc;
use threecrate_core::TriangleMesh;
use threecrate_gpu::{MeshRenderer, MeshRenderConfig, ShadingMode, PbrMaterial, mesh_to_gpu_mesh};
use winit::{
    application::ApplicationHandler,
    event_loop::{EventLoop, ActiveEventLoop},
    event::WindowEvent,
    window::{Window, WindowId},
};
use nalgebra::{Point3, Vector3};

fn build_cube_mesh() -> TriangleMesh {
    let vertices = vec![
        // Front face
        Point3::new(-1.0, -1.0,  1.0),
        Point3::new( 1.0, -1.0,  1.0),
        Point3::new( 1.0,  1.0,  1.0),
        Point3::new(-1.0,  1.0,  1.0),
        // Back face
        Point3::new( 1.0, -1.0, -1.0),
        Point3::new(-1.0, -1.0, -1.0),
        Point3::new(-1.0,  1.0, -1.0),
        Point3::new( 1.0,  1.0, -1.0),
    ];
    let faces = vec![
        [0,1,2], [2,3,0], // front
        [4,5,6], [6,7,4], // back
        [3,2,7], [7,6,3], // top
        [5,4,1], [1,0,5], // bottom
        [1,4,7], [7,2,1], // right
        [5,0,3], [3,6,5], // left
    ];
    TriangleMesh::from_vertices_and_faces(vertices, faces)
}

struct App {
    mesh: TriangleMesh,
    window: Option<Arc<Window>>,
    renderer: Option<MeshRenderer<'static>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // Create window
            let window_attrs = Window::default_attributes()
                .with_title("threecrate GPU Mesh Rendering Example");

            let window = match event_loop.create_window(window_attrs) {
                Ok(w) => Arc::new(w),
                Err(e) => {
                    eprintln!("Failed to create window: {}", e);
                    event_loop.exit();
                    return;
                }
            };

            // Use transmute to get a 'static reference
            // This is safe because the window Arc will live for the duration of the program
            let window_ref: &'static Window = unsafe {
                std::mem::transmute::<&Window, &'static Window>(window.as_ref())
            };

            // Create mesh renderer
            let renderer = match pollster::block_on(MeshRenderer::new(window_ref, MeshRenderConfig::default())) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to create renderer: {}", e);
                    event_loop.exit();
                    return;
                }
            };

            self.window = Some(window);
            self.renderer = Some(renderer);

            // Set up camera
            if let Some(renderer) = &mut self.renderer {
                let eye = nalgebra::Point3::new(3.0, 3.0, 3.0);
                let center = nalgebra::Point3::origin();
                let up = Vector3::y_axis().into_inner();
                let view = nalgebra::Isometry3::look_at_rh(&eye, &center, &up).to_homogeneous();
                let proj = nalgebra::Perspective3::new(16.0/9.0, 45.0_f32.to_radians(), 0.1, 100.0).to_homogeneous();
                renderer.update_camera(view, proj, eye.coords);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = &self.window else { return; };
        let Some(renderer) = &mut self.renderer else { return; };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                renderer.resize(size);
            }
            WindowEvent::RedrawRequested => {
                // Build indices and convert to GPU mesh
                let indices: Vec<u32> = self.mesh.faces.iter().flat_map(|f| [f[0] as u32, f[1] as u32, f[2] as u32]).collect();
                let gpu_mesh = mesh_to_gpu_mesh(&self.mesh.vertices, &indices, self.mesh.normals.as_deref(), None, Some(PbrMaterial::default()));
                let _ = renderer.render(&gpu_mesh, ShadingMode::Flat);
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a simple cube mesh
    let mesh = build_cube_mesh();

    // Event loop
    let event_loop = EventLoop::new()?;

    // Create app
    let mut app = App {
        mesh,
        window: None,
        renderer: None,
    };

    // Run event loop
    event_loop.run_app(&mut app)?;

    Ok(())
}