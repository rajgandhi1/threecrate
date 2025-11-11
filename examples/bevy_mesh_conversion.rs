//! Example demonstrating conversion between threecrate TriangleMesh and Bevy Mesh
//!
//! This example shows how to:
//! 1. Generate a mesh using marching cubes
//! 2. Convert it to a Bevy mesh
//! 3. Convert it back to a TriangleMesh
//!
//! Run with: cargo run --package threecrate-examples --bin bevy_mesh_conversion --features bevy_interop

#[cfg(feature = "bevy_interop")]
fn main() {
    use threecrate_core::Point3f;
    use threecrate_reconstruction::marching_cubes::{create_sphere_volume, marching_cubes};

    println!("Bevy Mesh Conversion Example");
    println!("============================\n");

    // Step 1: Create a mesh using marching cubes
    println!("1. Generating mesh with marching cubes...");
    let sphere_grid = create_sphere_volume(
        Point3f::new(0.0, 0.0, 0.0),
        1.0,
        [15, 15, 15],
        [3.0, 3.0, 3.0],
    );

    let triangle_mesh = marching_cubes(&sphere_grid, 0.0)
        .expect("Failed to generate mesh");

    println!("   Original TriangleMesh:");
    println!("   - Vertices: {}", triangle_mesh.vertex_count());
    println!("   - Faces: {}", triangle_mesh.face_count());
    println!("   - Has normals: {}", triangle_mesh.normals.is_some());
    println!();

    // Step 2: Convert to Bevy mesh
    println!("2. Converting to Bevy Mesh...");
    let bevy_mesh = triangle_mesh.to_bevy_mesh()
        .expect("Failed to convert to Bevy mesh");

    println!("   Bevy Mesh created successfully!");
    println!("   - Has positions: {}", bevy_mesh.attribute(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION).is_some());
    println!("   - Has normals: {}", bevy_mesh.attribute(bevy::render::mesh::Mesh::ATTRIBUTE_NORMAL).is_some());
    println!("   - Has indices: {}", bevy_mesh.indices().is_some());
    println!();

    // Step 3: Convert back to TriangleMesh
    println!("3. Converting back to TriangleMesh...");
    let reconstructed_mesh = threecrate_core::TriangleMesh::from_bevy_mesh(&bevy_mesh)
        .expect("Failed to convert from Bevy mesh");

    println!("   Reconstructed TriangleMesh:");
    println!("   - Vertices: {}", reconstructed_mesh.vertex_count());
    println!("   - Faces: {}", reconstructed_mesh.face_count());
    println!("   - Has normals: {}", reconstructed_mesh.normals.is_some());
    println!();

    // Verify round-trip conversion
    println!("4. Verifying round-trip conversion...");
    if triangle_mesh.vertex_count() == reconstructed_mesh.vertex_count() &&
       triangle_mesh.face_count() == reconstructed_mesh.face_count() {
        println!("   ✓ Round-trip conversion successful!");
        println!("   Vertex and face counts match.");
    } else {
        println!("   ✗ Round-trip conversion failed!");
        println!("   Counts do not match.");
    }
    println!();

    // Example with colored mesh
    println!("5. Testing conversion with vertex colors...");
    let mut colored_mesh = triangle_mesh;
    let colors = vec![[255u8, 0, 0]; colored_mesh.vertex_count()]; // Red vertices
    colored_mesh.set_colors(colors);

    let bevy_colored = colored_mesh.to_bevy_mesh()
        .expect("Failed to convert colored mesh");

    println!("   - Has colors: {}", bevy_colored.attribute(bevy::render::mesh::Mesh::ATTRIBUTE_COLOR).is_some());
    println!();

    println!("============================");
    println!("Example completed successfully!");
    println!("\nThis example demonstrates that TriangleMesh can be seamlessly");
    println!("converted to and from Bevy's mesh format, enabling integration");
    println!("with Bevy's rendering pipeline.");
}

#[cfg(not(feature = "bevy_interop"))]
fn main() {
    println!("This example requires the 'bevy_interop' feature.");
    println!("Run with: cargo run --package threecrate-examples --bin bevy_mesh_conversion --features bevy_interop");
}
