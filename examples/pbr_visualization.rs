//! PBR Visualization Example — issue #98
//!
//! Demonstrates the expanded material system:
//!   - Expose ShadingMode::Pbr through InteractiveViewer
//!   - Per-mesh PBR material (albedo, metallic, roughness)
//!   - Ambient / light-intensity controls via keyboard
//!   - Screenshot export with the S key
//!
//! Controls (printed to the console on startup):
//!   M          — toggle Flat / PBR shading
//!   S          — save screenshot_<timestamp>.png
//!   [ / ]      — decrease / increase ambient strength
//!   - / =      — decrease / increase light intensity
//!   O / P / Z  — Orbit / Pan / Zoom camera
//!   R          — reset camera

use nalgebra::Point3;
use threecrate_core::{Point3f, TriangleMesh};
use threecrate_gpu::{MeshLightingParams, PbrMaterial};
use threecrate_visualization::InteractiveViewer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PBR Visualization Example (issue #98) ===\n");

    // Build a simple sphere-like mesh so the PBR shading is clearly visible
    let mesh = build_uv_sphere(32, 16);
    println!(
        "Mesh: {} vertices, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    let mut viewer = InteractiveViewer::new()?;

    // --- Set a "polished gold" PBR material ---
    viewer.set_material(PbrMaterial {
        albedo: [1.0, 0.766, 0.336], // gold colour
        metallic: 1.0,
        roughness: 0.2,
        ao: 1.0,
        _padding1: [0.0, 0.0],
        emission: [0.0, 0.0, 0.0],
        _padding2: 0.0,
    });

    // --- Enable PBR shading ---
    viewer.set_shading_mode(threecrate_gpu::ShadingMode::Pbr);

    // --- Adjust lighting for a nice result ---
    viewer.set_lighting_params(MeshLightingParams {
        light_position: [5.0, 8.0, 6.0],
        light_intensity: 2.0,
        light_color: [1.0, 1.0, 1.0],
        ambient_strength: 0.05,
        gamma: 2.2,
        exposure: 1.0,
        _padding: [0.0, 0.0],
    });

    viewer.set_mesh(&mesh);
    viewer.run()?;

    Ok(())
}

/// Build a UV sphere with `h_segs` horizontal segments and `v_segs` vertical segments.
fn build_uv_sphere(h_segs: usize, v_segs: usize) -> TriangleMesh {
    use std::f32::consts::PI;

    let mut vertices: Vec<Point3f> = Vec::new();
    let mut normals: Vec<nalgebra::Vector3<f32>> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    // Generate vertices row by row
    for v in 0..=v_segs {
        let phi = PI * v as f32 / v_segs as f32; // 0 … π
        for h in 0..=h_segs {
            let theta = 2.0 * PI * h as f32 / h_segs as f32; // 0 … 2π
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            vertices.push(Point3f::new(x, y, z));
            normals.push(nalgebra::Vector3::new(x, y, z));
        }
    }

    let stride = h_segs + 1;

    // Generate triangles
    for v in 0..v_segs {
        for h in 0..h_segs {
            let tl = v * stride + h;
            let tr = tl + 1;
            let bl = tl + stride;
            let br = bl + 1;
            faces.push([tl, bl, tr]);
            faces.push([tr, bl, br]);
        }
    }

    let mut mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
    mesh.normals = Some(normals);
    mesh
}
