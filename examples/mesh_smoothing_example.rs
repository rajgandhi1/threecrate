//! Mesh Smoothing Example
//!
//! Demonstrates Laplacian, Taubin, and HC smoothing on a triangle mesh,
//! as implemented for issue #97.

use threecrate_core::{TriangleMesh, Point3f};
use threecrate_algorithms::{
    smooth_laplacian, smooth_taubin, smooth_hc,
    LaplacianSmoothingConfig, TaubinSmoothingConfig, HcSmoothingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mesh Smoothing Example ===\n");

    let mesh = make_noisy_grid();
    println!("Input mesh: {} vertices, {} faces", mesh.vertex_count(), mesh.face_count());
    println!("Spike vertex z before smoothing: {:.4}\n", mesh.vertices[12].z);

    // 1. Laplacian
    let lap = smooth_laplacian(&mesh, &LaplacianSmoothingConfig { iterations: 10, lambda: 0.5 })?;
    println!("1. Laplacian (λ=0.5, 10 iters):");
    println!("   Spike vertex z: {:.4}", lap.vertices[12].z);
    println!("   Spread: {:.4}", avg_spread(&lap));

    // 2. Taubin
    let tau = smooth_taubin(&mesh, &TaubinSmoothingConfig { iterations: 10, lambda: 0.5, mu: -0.53 })?;
    println!("\n2. Taubin (λ=0.5, μ=-0.53, 10 iters):");
    println!("   Spike vertex z: {:.4}", tau.vertices[12].z);
    println!("   Spread: {:.4} (larger = less shrinkage than Laplacian)", avg_spread(&tau));

    // 3. HC
    let hc = smooth_hc(&mesh, &HcSmoothingConfig { iterations: 10, alpha: 0.0, beta: 0.5 })?;
    println!("\n3. HC (α=0.0, β=0.5, 10 iters):");
    println!("   Spike vertex z: {:.4}", hc.vertices[12].z);
    println!("   Spread: {:.4}", avg_spread(&hc));

    // 4. Compare shrinkage at high iteration count
    println!("\n4. Volume preservation (50 iterations):");
    let original_spread = avg_spread(&mesh);
    let lap50 = smooth_laplacian(&mesh, &LaplacianSmoothingConfig { iterations: 50, lambda: 0.5 })?;
    let tau50 = smooth_taubin(&mesh, &TaubinSmoothingConfig { iterations: 50, lambda: 0.5, mu: -0.53 })?;
    let hc50  = smooth_hc(&mesh,  &HcSmoothingConfig  { iterations: 50, alpha: 0.0, beta: 0.5 })?;
    println!("   Original spread : {:.4}", original_spread);
    println!("   Laplacian spread: {:.4}", avg_spread(&lap50));
    println!("   Taubin spread   : {:.4}", avg_spread(&tau50));
    println!("   HC spread       : {:.4}", avg_spread(&hc50));

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// 5×5 grid with the centre vertex raised to z=1 (a "spike").
fn make_noisy_grid() -> TriangleMesh {
    let mut verts = Vec::new();
    for row in 0..5i32 {
        for col in 0..5i32 {
            let z = if row == 2 && col == 2 { 1.0 } else { 0.0 };
            verts.push(Point3f::new(col as f32, row as f32, z));
        }
    }
    let mut faces = Vec::new();
    for row in 0..4usize {
        for col in 0..4usize {
            let tl = row * 5 + col;
            let tr = tl + 1;
            let bl = tl + 5;
            let br = bl + 1;
            faces.push([tl, tr, bl]);
            faces.push([tr, br, bl]);
        }
    }
    TriangleMesh::from_vertices_and_faces(verts, faces)
}

/// Average distance of vertices from their centroid.
fn avg_spread(mesh: &TriangleMesh) -> f32 {
    use threecrate_core::Vector3f;
    let c: Vector3f = mesh.vertices.iter()
        .fold(Vector3f::zeros(), |acc, v| acc + v.coords)
        / mesh.vertices.len() as f32;
    mesh.vertices.iter().map(|v| (v.coords - c).magnitude()).sum::<f32>()
        / mesh.vertices.len() as f32
}
