//! GPU Mesh Rendering Example

use std::sync::Arc;
use threecrate_core::TriangleMesh;
use threecrate_gpu::{MeshRenderer, MeshRenderConfig, ShadingMode, PbrMaterial, mesh_to_gpu_mesh};
use winit::{event_loop::EventLoop, window::WindowBuilder, event::Event, event_loop::ControlFlow};
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a simple cube mesh
    let mesh = build_cube_mesh();

    // Event loop and window
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("threecrate GPU Mesh Rendering Example")
        .build(&event_loop)?);

    // Create mesh renderer
    let window_for_renderer = window.clone();
    let mut renderer = pollster::block_on(MeshRenderer::new(&window_for_renderer, MeshRenderConfig::default()))?;

    // Camera matrices (simple static camera)
    let eye = nalgebra::Point3::new(3.0, 3.0, 3.0);
    let center = nalgebra::Point3::origin();
    let up = Vector3::y_axis().into_inner();
    let view = nalgebra::Isometry3::look_at_rh(&eye, &center, &up).to_homogeneous();
    let proj = nalgebra::Perspective3::new(16.0/9.0, 45.0_f32.to_radians(), 0.1, 100.0).to_homogeneous();

    renderer.update_camera(view, proj, eye.coords);

    // Build indices and convert to GPU mesh
    let indices: Vec<u32> = mesh.faces.iter().flat_map(|f| [f[0] as u32, f[1] as u32, f[2] as u32]).collect();
    let gpu_mesh = mesh_to_gpu_mesh(&mesh.vertices, &indices, mesh.normals.as_deref(), None, Some(PbrMaterial::default()));

    // Run loop
    let window_for_loop = window.clone();
    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => {
                if let winit::event::WindowEvent::CloseRequested = event { target.exit(); }
                if let winit::event::WindowEvent::Resized(size) = event { renderer.resize(size); }
                if let winit::event::WindowEvent::RedrawRequested = event {
                    let _ = renderer.render(&gpu_mesh, ShadingMode::Flat);
                }
            }
            Event::AboutToWait => {
                window_for_loop.request_redraw();
            }
            _ => {}
        }
    })?;

    // Unreachable
    #[allow(unreachable_code)]
    Ok(())
}