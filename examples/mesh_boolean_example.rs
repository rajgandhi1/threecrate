//! Mesh Boolean Operations Example
//!
//! Demonstrates union, intersection, and difference on closed triangle meshes
//! using BSP-tree CSG, as implemented for issue #96.

use threecrate_core::{TriangleMesh, Point3f};
use threecrate_algorithms::{mesh_union, mesh_intersection, mesh_difference, mesh_boolean, BooleanOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mesh Boolean Operations Example ===\n");

    let cube_a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(2.0, 2.0, 2.0));
    let cube_b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(3.0, 3.0, 3.0));

    println!("Cube A: {} verts, {} faces", cube_a.vertex_count(), cube_a.face_count());
    println!("Cube B: {} verts, {} faces", cube_b.vertex_count(), cube_b.face_count());
    println!("(The two cubes overlap in the region [1,2]^3)\n");

    // 1. Union
    let union_mesh = mesh_union(&cube_a, &cube_b)?;
    println!("1. Union (A ∪ B):");
    println!("   {} verts, {} faces", union_mesh.vertex_count(), union_mesh.face_count());

    // 2. Intersection
    let isect_mesh = mesh_intersection(&cube_a, &cube_b)?;
    println!("\n2. Intersection (A ∩ B):");
    println!("   {} verts, {} faces", isect_mesh.vertex_count(), isect_mesh.face_count());

    // 3. Difference
    let diff_mesh = mesh_difference(&cube_a, &cube_b)?;
    println!("\n3. Difference (A − B):");
    println!("   {} verts, {} faces", diff_mesh.vertex_count(), diff_mesh.face_count());

    // 4. Via the unified BooleanOp dispatch
    println!("\n4. Via mesh_boolean() dispatch:");
    for op in [BooleanOp::Union, BooleanOp::Intersection, BooleanOp::Difference] {
        let result = mesh_boolean(&cube_a, &cube_b, op)?;
        println!("   {:?}: {} faces", op, result.face_count());
    }

    // 5. Non-overlapping meshes
    println!("\n5. Non-overlapping cubes:");
    let far_b = make_box(Point3f::new(10.0, 0.0, 0.0), Point3f::new(11.0, 1.0, 1.0));
    let u = mesh_union(&cube_a, &far_b)?;
    let i = mesh_intersection(&cube_a, &far_b)?;
    let d = mesh_difference(&cube_a, &far_b)?;
    println!("   Union:        {} faces (both boxes preserved)", u.face_count());
    println!("   Intersection: {} faces (empty – no overlap)", i.face_count());
    println!("   Difference:   {} faces (A unchanged)", d.face_count());

    // 6. Empty mesh edge cases
    println!("\n6. Empty mesh edge cases:");
    let empty = TriangleMesh::new();
    println!("   union(empty, A):        {} faces", mesh_union(&empty, &cube_a)?.face_count());
    println!("   intersection(empty, A): {} faces", mesh_intersection(&empty, &cube_a)?.face_count());
    println!("   difference(A, empty):   {} faces", mesh_difference(&cube_a, &empty)?.face_count());

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

/// Build a closed, outward-normal box mesh spanning [min, max]^3.
fn make_box(min: Point3f, max: Point3f) -> TriangleMesh {
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);
    let verts = vec![
        Point3f::new(x0, y0, z0), // 0
        Point3f::new(x1, y0, z0), // 1
        Point3f::new(x1, y1, z0), // 2
        Point3f::new(x0, y1, z0), // 3
        Point3f::new(x0, y0, z1), // 4
        Point3f::new(x1, y0, z1), // 5
        Point3f::new(x1, y1, z1), // 6
        Point3f::new(x0, y1, z1), // 7
    ];
    let faces = vec![
        [0, 2, 1], [0, 3, 2], // bottom
        [4, 5, 6], [4, 6, 7], // top
        [0, 1, 5], [0, 5, 4], // front
        [3, 7, 6], [3, 6, 2], // back
        [0, 4, 7], [0, 7, 3], // left
        [1, 2, 6], [1, 6, 5], // right
    ];
    TriangleMesh::from_vertices_and_faces(verts, faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_operations() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(2.0, 2.0, 2.0));
        let b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(3.0, 3.0, 3.0));

        assert!(!mesh_union(&a, &b).unwrap().is_empty());
        assert!(!mesh_intersection(&a, &b).unwrap().is_empty());
        assert!(!mesh_difference(&a, &b).unwrap().is_empty());
    }
}
