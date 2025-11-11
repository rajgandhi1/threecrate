// Test script to verify the complete marching cubes implementation
use threecrate_core::Point3f;
use threecrate_reconstruction::marching_cubes::{create_sphere_volume, marching_cubes, MarchingCubes, MarchingCubesConfig};

fn main() {
    println!("Testing Complete Marching Cubes Implementation");
    println!("==============================================\n");

    // Test 1: Simple sphere
    println!("Test 1: Extracting isosurface from a sphere...");
    let sphere_grid = create_sphere_volume(
        Point3f::new(0.0, 0.0, 0.0),
        1.0,
        [20, 20, 20],
        [4.0, 4.0, 4.0],
    );

    let config = MarchingCubesConfig {
        iso_level: 0.0,
        compute_normals: true,
        smooth_mesh: false,
        smoothing_iterations: 0,
    };

    let mc = MarchingCubes::new(config);
    match mc.extract_isosurface(&sphere_grid) {
        Ok(mesh) => {
            println!("  ✓ Successfully extracted mesh from sphere");
            println!("  - Vertices: {}", mesh.vertex_count());
            println!("  - Faces: {}", mesh.face_count());
            println!("  - Has normals: {}", mesh.normals.is_some());

            if mesh.vertex_count() == 0 || mesh.face_count() == 0 {
                println!("  ✗ ERROR: Mesh is empty!");
            } else {
                println!("  ✓ Mesh contains geometry");
            }
        }
        Err(e) => {
            println!("  ✗ Failed to extract mesh: {}", e);
        }
    }

    println!();

    // Test 2: Higher resolution sphere
    println!("Test 2: Higher resolution sphere (30x30x30)...");
    let sphere_grid_hires = create_sphere_volume(
        Point3f::new(0.0, 0.0, 0.0),
        1.5,
        [30, 30, 30],
        [6.0, 6.0, 6.0],
    );

    match marching_cubes(&sphere_grid_hires, 0.0) {
        Ok(mesh) => {
            println!("  ✓ Successfully extracted high-res mesh");
            println!("  - Vertices: {}", mesh.vertex_count());
            println!("  - Faces: {}", mesh.face_count());

            // Verify mesh topology
            let expected_min_vertices = 100; // Should have at least some vertices
            if mesh.vertex_count() >= expected_min_vertices {
                println!("  ✓ Mesh has sufficient detail");
            } else {
                println!("  ⚠ Mesh has fewer vertices than expected");
            }
        }
        Err(e) => {
            println!("  ✗ Failed: {}", e);
        }
    }

    println!();

    // Test 3: Different iso-levels
    println!("Test 3: Testing different iso-levels...");
    for iso_level in [-0.2, -0.1, 0.0, 0.1, 0.2] {
        match marching_cubes(&sphere_grid, iso_level) {
            Ok(mesh) => {
                println!("  - iso_level {:.1}: {} vertices, {} faces",
                    iso_level, mesh.vertex_count(), mesh.face_count());
            }
            Err(_) => {
                println!("  - iso_level {:.1}: No surface found", iso_level);
            }
        }
    }

    println!();

    // Test 4: Mesh with smoothing
    println!("Test 4: Testing mesh smoothing...");
    let config_smooth = MarchingCubesConfig {
        iso_level: 0.0,
        compute_normals: true,
        smooth_mesh: true,
        smoothing_iterations: 3,
    };

    let mc_smooth = MarchingCubes::new(config_smooth);
    match mc_smooth.extract_isosurface(&sphere_grid) {
        Ok(mesh) => {
            println!("  ✓ Successfully extracted smoothed mesh");
            println!("  - Vertices: {}", mesh.vertex_count());
            println!("  - Faces: {}", mesh.face_count());
        }
        Err(e) => {
            println!("  ✗ Failed: {}", e);
        }
    }

    println!();
    println!("==============================================");
    println!("All tests completed!");
}
