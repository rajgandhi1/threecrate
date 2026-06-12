//! Mesh boolean operations: union, intersection, and difference
//!
//! Implements CSG (Constructive Solid Geometry) using a BSP-tree approach.
//! Each closed `TriangleMesh` is converted into a set of `Polygon`s, which are
//! inserted into a BSP tree.  The classic three-step clip/invert/build sequences
//! then produce the desired solid.
//!
//! **Limitations**
//! - Meshes must be closed (watertight) and have outward-facing normals.
//! - Very thin or near-degenerate triangles may be dropped (area < `EPSILON²`).
//! - Shared planar faces are handled via coplanar classification but exact
//!   floating-point coincidence can produce small artefacts at seam edges.

use threecrate_core::{Point3f, Result, TriangleMesh, Vector3f};

/// Numeric tolerance used for plane-side classification and degenerate checks.
const EPSILON: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Internal geometry types
// ---------------------------------------------------------------------------

/// A plane in the form `normal · x = w`.
#[derive(Debug, Clone)]
struct Plane {
    normal: Vector3f,
    w: f32,
}

impl Plane {
    fn from_points(a: &Point3f, b: &Point3f, c: &Point3f) -> Option<Self> {
        let n = (b - a).cross(&(c - a));
        let len = n.magnitude();
        if len < EPSILON * EPSILON {
            return None; // degenerate triangle
        }
        let normal = n / len;
        let w = normal.dot(&a.coords);
        Some(Plane { normal, w })
    }

    fn flip(&mut self) {
        self.normal = -self.normal;
        self.w = -self.w;
    }

    #[inline]
    fn signed_distance(&self, p: &Point3f) -> f32 {
        self.normal.dot(&p.coords) - self.w
    }

    /// Split `poly` into up to four output lists: coplanar-front, coplanar-back,
    /// front, and back.  Spanning polygons are clipped and triangulated.
    fn split_polygon(
        &self,
        poly: &Polygon,
        coplanar_front: &mut Vec<Polygon>,
        coplanar_back: &mut Vec<Polygon>,
        front: &mut Vec<Polygon>,
        back: &mut Vec<Polygon>,
    ) {
        let verts = &poly.vertices;
        let n = verts.len();

        // Classify every vertex
        let dists: Vec<f32> = verts.iter().map(|v| self.signed_distance(v)).collect();
        let mut poly_mask: u8 = 0;
        for &d in &dists {
            if d > EPSILON {
                poly_mask |= 1; // front
            } else if d < -EPSILON {
                poly_mask |= 2; // back
            }
            // else on-plane → 0
        }

        match poly_mask {
            0 => {
                // Coplanar – classify by normal alignment
                if self.normal.dot(&poly.plane.normal) > 0.0 {
                    coplanar_front.push(poly.clone());
                } else {
                    coplanar_back.push(poly.clone());
                }
            }
            1 => front.push(poly.clone()),
            2 => back.push(poly.clone()),
            _ => {
                // Spanning – clip and re-triangulate
                let mut f_verts: Vec<Point3f> = Vec::new();
                let mut b_verts: Vec<Point3f> = Vec::new();

                for i in 0..n {
                    let j = (i + 1) % n;
                    let di = dists[i];
                    let dj = dists[j];
                    let vi = verts[i];
                    let vj = verts[j];

                    let i_front = di > EPSILON;
                    let i_back = di < -EPSILON;
                    let j_front = dj > EPSILON;
                    let j_back = dj < -EPSILON;

                    if !i_back {
                        f_verts.push(vi);
                    }
                    if !i_front {
                        b_verts.push(vi);
                    }

                    // Edge crosses the plane?
                    if (i_front && j_back) || (i_back && j_front) {
                        let t = di / (di - dj);
                        let mid = vi + (vj - vi) * t;
                        f_verts.push(mid);
                        b_verts.push(mid);
                    }
                }

                front.extend(Polygon::fan_triangulate(f_verts));
                back.extend(Polygon::fan_triangulate(b_verts));
            }
        }
    }
}

/// A convex polygon stored as an ordered vertex list plus its precomputed plane.
#[derive(Debug, Clone)]
struct Polygon {
    vertices: Vec<Point3f>,
    plane: Plane,
}

impl Polygon {
    fn new(vertices: Vec<Point3f>) -> Option<Self> {
        if vertices.len() < 3 {
            return None;
        }
        let plane = Plane::from_points(&vertices[0], &vertices[1], &vertices[2])?;
        Some(Polygon { vertices, plane })
    }

    fn flip(&mut self) {
        self.vertices.reverse();
        self.plane.flip();
    }

    /// Fan-triangulate a convex vertex list into `Polygon` triangles.
    fn fan_triangulate(verts: Vec<Point3f>) -> Vec<Polygon> {
        if verts.len() < 3 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(verts.len() - 2);
        for i in 1..verts.len() - 1 {
            if let Some(p) = Polygon::new(vec![verts[0], verts[i], verts[i + 1]]) {
                out.push(p);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// BSP tree
// ---------------------------------------------------------------------------

struct BspNode {
    plane: Option<Plane>,
    front: Option<Box<BspNode>>,
    back: Option<Box<BspNode>>,
    polygons: Vec<Polygon>,
}

impl BspNode {
    fn new() -> Self {
        BspNode {
            plane: None,
            front: None,
            back: None,
            polygons: Vec::new(),
        }
    }

    fn invert(&mut self) {
        for p in &mut self.polygons {
            p.flip();
        }
        if let Some(ref mut plane) = self.plane {
            plane.flip();
        }
        if let Some(ref mut f) = self.front {
            f.invert();
        }
        if let Some(ref mut b) = self.back {
            b.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    /// Recursively discard polygons inside this BSP tree, keeping those outside.
    fn clip_polygons(&self, polygons: Vec<Polygon>) -> Vec<Polygon> {
        let plane = match &self.plane {
            Some(p) => p,
            None => return polygons,
        };

        let mut f: Vec<Polygon> = Vec::new();
        let mut b: Vec<Polygon> = Vec::new();

        for poly in polygons {
            let mut cf = Vec::new();
            let mut cb = Vec::new();
            let mut pf = Vec::new();
            let mut pb = Vec::new();
            plane.split_polygon(&poly, &mut cf, &mut cb, &mut pf, &mut pb);
            f.extend(cf);
            f.extend(pf);
            b.extend(cb);
            b.extend(pb);
        }

        let mut result = match &self.front {
            Some(node) => node.clip_polygons(f),
            None => f,
        };
        let back_result = match &self.back {
            Some(node) => node.clip_polygons(b),
            None => Vec::new(), // discard: these are inside the solid
        };
        result.extend(back_result);
        result
    }

    /// Remove all polygons in this tree that are inside `other`.
    fn clip_to(&mut self, other: &BspNode) {
        self.polygons = other.clip_polygons(std::mem::take(&mut self.polygons));
        if let Some(ref mut f) = self.front {
            f.clip_to(other);
        }
        if let Some(ref mut b) = self.back {
            b.clip_to(other);
        }
    }

    /// Collect all polygons stored in this tree.
    fn all_polygons(&self) -> Vec<Polygon> {
        let mut out = self.polygons.clone();
        if let Some(ref f) = self.front {
            out.extend(f.all_polygons());
        }
        if let Some(ref b) = self.back {
            out.extend(b.all_polygons());
        }
        out
    }

    /// Insert `polygons` into the BSP tree (building the tree as needed).
    fn build(&mut self, polygons: Vec<Polygon>) {
        if polygons.is_empty() {
            return;
        }

        if self.plane.is_none() {
            self.plane = Some(polygons[0].plane.clone());
        }
        let plane = self.plane.clone().unwrap();

        let mut front = Vec::new();
        let mut back = Vec::new();

        for poly in polygons {
            let mut cf = Vec::new();
            let mut cb = Vec::new();
            let mut pf = Vec::new();
            let mut pb = Vec::new();
            plane.split_polygon(&poly, &mut cf, &mut cb, &mut pf, &mut pb);
            self.polygons.extend(cf);
            self.polygons.extend(cb);
            front.extend(pf);
            back.extend(pb);
        }

        if !front.is_empty() {
            if self.front.is_none() {
                self.front = Some(Box::new(BspNode::new()));
            }
            self.front.as_mut().unwrap().build(front);
        }
        if !back.is_empty() {
            if self.back.is_none() {
                self.back = Some(Box::new(BspNode::new()));
            }
            self.back.as_mut().unwrap().build(back);
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh ↔ polygon conversion helpers
// ---------------------------------------------------------------------------

fn mesh_to_polygons(mesh: &TriangleMesh) -> Vec<Polygon> {
    mesh.faces
        .iter()
        .filter_map(|face| {
            Polygon::new(vec![
                mesh.vertices[face[0]],
                mesh.vertices[face[1]],
                mesh.vertices[face[2]],
            ])
        })
        .collect()
}

fn polygons_to_mesh(polygons: Vec<Polygon>) -> TriangleMesh {
    // Each Polygon is already a triangle (fan-triangulation keeps them as triangles),
    // so we can flatten directly.
    let capacity = polygons
        .iter()
        .map(|p| p.vertices.len().saturating_sub(2))
        .sum::<usize>();
    let mut vertices = Vec::with_capacity(capacity * 3);
    let mut faces = Vec::with_capacity(capacity);

    for poly in &polygons {
        let n = poly.vertices.len();
        if n < 3 {
            continue;
        }
        // Fan triangulation
        for i in 1..n - 1 {
            let base = vertices.len();
            vertices.push(poly.vertices[0]);
            vertices.push(poly.vertices[i]);
            vertices.push(poly.vertices[i + 1]);
            faces.push([base, base + 1, base + 2]);
        }
    }

    TriangleMesh::from_vertices_and_faces(vertices, faces)
}

fn build_bsp(mesh: &TriangleMesh) -> BspNode {
    let mut node = BspNode::new();
    node.build(mesh_to_polygons(mesh));
    node
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Which boolean operation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// All regions inside A **or** B.
    Union,
    /// All regions inside both A **and** B.
    Intersection,
    /// All regions inside A but **not** B.
    Difference,
}

/// Perform a mesh boolean operation on two closed triangle meshes.
///
/// # Arguments
/// * `a` - First operand mesh (must be closed / watertight)
/// * `b` - Second operand mesh (must be closed / watertight)
/// * `op` - The boolean operation to apply
///
/// # Returns
/// A new `TriangleMesh` representing the result of the operation.
pub fn mesh_boolean(a: &TriangleMesh, b: &TriangleMesh, op: BooleanOp) -> Result<TriangleMesh> {
    match op {
        BooleanOp::Union => mesh_union(a, b),
        BooleanOp::Intersection => mesh_intersection(a, b),
        BooleanOp::Difference => mesh_difference(a, b),
    }
}

/// Compute the boolean **union** of two closed triangle meshes.
///
/// Returns a mesh containing all regions inside either `a` or `b`.
///
/// ```text
///     A.union(B)
///
///     +-------+            +-------+
///     |       |            |       |
///     |   A   |            |       |
///     |    +--+----+   =   |       +----+
///     +----+--+    |       +----+       |
///          |   B   |            |       |
///          |       |            |       |
///          +-------+            +-------+
/// ```
pub fn mesh_union(a: &TriangleMesh, b: &TriangleMesh) -> Result<TriangleMesh> {
    if a.is_empty() {
        return Ok(b.clone());
    }
    if b.is_empty() {
        return Ok(a.clone());
    }

    let mut a_bsp = build_bsp(a);
    let mut b_bsp = build_bsp(b);

    a_bsp.clip_to(&b_bsp);
    b_bsp.clip_to(&a_bsp);
    b_bsp.invert();
    b_bsp.clip_to(&a_bsp);
    b_bsp.invert();
    a_bsp.build(b_bsp.all_polygons());

    Ok(polygons_to_mesh(a_bsp.all_polygons()))
}

/// Compute the boolean **intersection** of two closed triangle meshes.
///
/// Returns a mesh containing only the regions inside both `a` and `b`.
///
/// ```text
///     A.intersection(B)
///
///     +-------+
///     |       |
///     |   A   |
///     |    +--+----+   =   +--+
///     +----+--+    |       +--+
///          |   B   |
///          |       |
///          +-------+
/// ```
pub fn mesh_intersection(a: &TriangleMesh, b: &TriangleMesh) -> Result<TriangleMesh> {
    if a.is_empty() || b.is_empty() {
        return Ok(TriangleMesh::new());
    }

    let mut a_bsp = build_bsp(a);
    let mut b_bsp = build_bsp(b);

    a_bsp.invert();
    b_bsp.clip_to(&a_bsp);
    b_bsp.invert();
    a_bsp.clip_to(&b_bsp);
    b_bsp.clip_to(&a_bsp);
    a_bsp.build(b_bsp.all_polygons());
    a_bsp.invert();

    Ok(polygons_to_mesh(a_bsp.all_polygons()))
}

/// Compute the boolean **difference** of two closed triangle meshes (A minus B).
///
/// Returns a mesh containing only the regions inside `a` but not `b`.
///
/// ```text
///     A.difference(B)
///
///     +-------+            +-------+
///     |       |            |       |
///     |   A   |            |   A   |
///     |    +--+----+   =   |    +--+
///     +----+--+    |       +----+
///          |   B   |
///          |       |
///          +-------+
/// ```
pub fn mesh_difference(a: &TriangleMesh, b: &TriangleMesh) -> Result<TriangleMesh> {
    if a.is_empty() {
        return Ok(TriangleMesh::new());
    }
    if b.is_empty() {
        return Ok(a.clone());
    }

    let mut a_bsp = build_bsp(a);
    let mut b_bsp = build_bsp(b);

    a_bsp.invert();
    a_bsp.clip_to(&b_bsp);
    b_bsp.clip_to(&a_bsp);
    b_bsp.invert();
    b_bsp.clip_to(&a_bsp);
    b_bsp.invert();
    a_bsp.build(b_bsp.all_polygons());
    a_bsp.invert();

    Ok(polygons_to_mesh(a_bsp.all_polygons()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a closed box mesh with outward-facing normals.
    /// Vertices are the 8 corners of [min, max]^3.
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
        // Each face is CCW when viewed from outside → outward normal
        let faces = vec![
            // Bottom (z-): normal (0,0,-1)
            [0, 2, 1],
            [0, 3, 2],
            // Top (z+): normal (0,0,+1)
            [4, 5, 6],
            [4, 6, 7],
            // Front (y-): normal (0,-1,0)
            [0, 1, 5],
            [0, 5, 4],
            // Back (y+): normal (0,+1,0)
            [3, 7, 6],
            [3, 6, 2],
            // Left (x-): normal (-1,0,0)
            [0, 4, 7],
            [0, 7, 3],
            // Right (x+): normal (+1,0,0)
            [1, 2, 6],
            [1, 6, 5],
        ];
        TriangleMesh::from_vertices_and_faces(verts, faces)
    }

    // ---- empty-mesh edge cases ----

    #[test]
    fn test_union_empty_a() {
        let empty = TriangleMesh::new();
        let b = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let result = mesh_union(&empty, &b).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_union_empty_b() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let empty = TriangleMesh::new();
        let result = mesh_union(&a, &empty).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_intersection_empty() {
        let empty = TriangleMesh::new();
        let b = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let result = mesh_intersection(&empty, &b).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_difference_empty_a() {
        let empty = TriangleMesh::new();
        let b = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let result = mesh_difference(&empty, &b).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_difference_empty_b() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let empty = TriangleMesh::new();
        let result = mesh_difference(&a, &empty).unwrap();
        assert!(!result.is_empty());
    }

    // ---- non-overlapping meshes ----

    #[test]
    fn test_union_non_overlapping() {
        // Two unit cubes separated along X
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let b = make_box(Point3f::new(5.0, 0.0, 0.0), Point3f::new(6.0, 1.0, 1.0));
        let result = mesh_union(&a, &b).unwrap();
        // Both boxes should be fully preserved
        assert!(!result.is_empty());
        assert!(
            result.face_count() >= 24,
            "expected ≥24 faces, got {}",
            result.face_count()
        );
    }

    #[test]
    fn test_intersection_non_overlapping() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let b = make_box(Point3f::new(5.0, 0.0, 0.0), Point3f::new(6.0, 1.0, 1.0));
        let result = mesh_intersection(&a, &b).unwrap();
        // No overlap → empty result
        assert!(
            result.is_empty(),
            "non-overlapping intersection should be empty, got {} faces",
            result.face_count()
        );
    }

    #[test]
    fn test_difference_non_overlapping() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 1.0));
        let b = make_box(Point3f::new(5.0, 0.0, 0.0), Point3f::new(6.0, 1.0, 1.0));
        let result = mesh_difference(&a, &b).unwrap();
        // B doesn't touch A → result should be A unchanged
        assert!(!result.is_empty());
        assert!(
            result.face_count() >= 12,
            "expected ≥12 faces, got {}",
            result.face_count()
        );
    }

    // ---- overlapping meshes ----

    #[test]
    fn test_union_overlapping() {
        // Two unit cubes that partially overlap
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(2.0, 2.0, 2.0));
        let b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(3.0, 3.0, 3.0));
        let result = mesh_union(&a, &b).unwrap();
        assert!(
            !result.is_empty(),
            "union of overlapping boxes must not be empty"
        );
    }

    #[test]
    fn test_intersection_overlapping() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(2.0, 2.0, 2.0));
        let b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(3.0, 3.0, 3.0));
        let result = mesh_intersection(&a, &b).unwrap();
        assert!(
            !result.is_empty(),
            "intersection of overlapping boxes must not be empty"
        );
        // The intersection is the unit cube [1,2]^3 → should have at most as many faces as either input
        assert!(result.face_count() <= a.face_count() + b.face_count() + 50);
    }

    #[test]
    fn test_difference_overlapping() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(2.0, 2.0, 2.0));
        let b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(3.0, 3.0, 3.0));
        let result = mesh_difference(&a, &b).unwrap();
        assert!(
            !result.is_empty(),
            "A minus partially-overlapping B must not be empty"
        );
    }

    #[test]
    fn test_difference_a_fully_contains_b() {
        // B is entirely inside A → difference cuts a hole
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(4.0, 4.0, 4.0));
        let b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(2.0, 2.0, 2.0));
        let result = mesh_difference(&a, &b).unwrap();
        assert!(
            !result.is_empty(),
            "difference with interior hole must not be empty"
        );
    }

    #[test]
    fn test_intersection_a_fully_contains_b() {
        // B fully inside A → intersection == B
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(4.0, 4.0, 4.0));
        let b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(2.0, 2.0, 2.0));
        let result = mesh_intersection(&a, &b).unwrap();
        assert!(
            !result.is_empty(),
            "intersection when B inside A must not be empty"
        );
    }

    // ---- BooleanOp dispatch ----

    #[test]
    fn test_mesh_boolean_dispatch() {
        let a = make_box(Point3f::new(0.0, 0.0, 0.0), Point3f::new(2.0, 2.0, 2.0));
        let b = make_box(Point3f::new(1.0, 1.0, 1.0), Point3f::new(3.0, 3.0, 3.0));

        let u = mesh_boolean(&a, &b, BooleanOp::Union).unwrap();
        let i = mesh_boolean(&a, &b, BooleanOp::Intersection).unwrap();
        let d = mesh_boolean(&a, &b, BooleanOp::Difference).unwrap();

        assert!(!u.is_empty());
        assert!(!i.is_empty());
        assert!(!d.is_empty());
    }
}
