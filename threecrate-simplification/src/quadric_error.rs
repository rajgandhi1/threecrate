//! Quadric error decimation
//!
//! Implementation of the Garland-Heckbert algorithm for mesh simplification
//! based on quadric error metrics.

use crate::MeshSimplifier;
use nalgebra::{Matrix4, Vector4};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use threecrate_core::{Error, Point3f, Result, TriangleMesh, Vector3f};

/// Quadric error decimation simplifier
pub struct QuadricErrorSimplifier {
    /// Configuration parameters
    pub max_edge_length: Option<f32>,
    pub preserve_boundary: bool,
    /// Not used by the current algorithm; kept for API compatibility.
    pub feature_angle_threshold: f32,
}

impl Default for QuadricErrorSimplifier {
    fn default() -> Self {
        Self {
            max_edge_length: None,
            preserve_boundary: true,
            feature_angle_threshold: 45.0_f32.to_radians(),
        }
    }
}

/// A candidate edge collapse. Ordered by cost, cheapest first in the heap.
/// `stamps` are the endpoints' versions when the cost was computed; if either
/// endpoint has changed since, the entry is stale and skipped.
struct Candidate {
    cost: f64,
    keep: usize,
    remove: usize,
    position: Point3f,
    stamps: [u32; 2],
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost) // min-heap
    }
}

/// Mesh state while collapsing. `vertex_faces` may list faces that have since
/// died or no longer use the vertex; `faces_of` filters them.
struct Working {
    positions: Vec<Point3f>,
    quadrics: Vec<Matrix4<f64>>,
    stamps: Vec<u32>,
    alive: Vec<bool>,
    faces: Vec<[usize; 3]>,
    face_alive: Vec<bool>,
    vertex_faces: Vec<Vec<usize>>,
    live_faces: usize,
}

impl Working {
    fn faces_of(&self, v: usize) -> impl Iterator<Item = usize> + '_ {
        self.vertex_faces[v]
            .iter()
            .copied()
            .filter(move |&f| self.face_alive[f] && self.faces[f].contains(&v))
    }

    fn neighbors(&self, v: usize) -> HashSet<usize> {
        self.faces_of(v)
            .flat_map(|f| self.faces[f])
            .filter(|&u| u != v)
            .collect()
    }

    /// Normal of `face`, with any vertex in `moved` placed at `to`.
    fn normal(&self, face: [usize; 3], moved: &[usize], to: Point3f) -> Vector3f {
        let p = face.map(|v| {
            if moved.contains(&v) {
                to
            } else {
                self.positions[v]
            }
        });
        (p[1] - p[0]).cross(&(p[2] - p[0]))
    }

    /// Whether collapsing `keep`-`remove` into `position` keeps the mesh a
    /// valid surface: the endpoints share exactly the neighbours of their
    /// shared faces (link condition), and no surrounding face flips over.
    fn can_collapse(&self, keep: usize, remove: usize, position: Point3f) -> bool {
        let shared = self
            .faces_of(keep)
            .filter(|&f| self.faces[f].contains(&remove))
            .count();
        let common = self
            .neighbors(keep)
            .intersection(&self.neighbors(remove))
            .count();
        if shared == 0 || common != shared {
            return false;
        }
        let ends = [keep, remove];
        self.faces_of(keep)
            .chain(self.faces_of(remove))
            .filter(|&f| !(self.faces[f].contains(&keep) && self.faces[f].contains(&remove)))
            .all(|f| {
                let before = self.normal(self.faces[f], &[], position);
                let after = self.normal(self.faces[f], &ends, position);
                // Reject flips, and faces squashed to (almost) no area.
                let (b, a) = (before.norm(), after.norm());
                a > 1e-6 * b && before.dot(&after) > 0.01 * b * a
            })
    }

    fn collapse(&mut self, keep: usize, remove: usize, position: Point3f) {
        self.positions[keep] = position;
        self.quadrics[keep] = self.quadrics[keep] + self.quadrics[remove];
        let moved_faces: Vec<usize> = self.faces_of(remove).collect();
        for f in moved_faces {
            if self.faces[f].contains(&keep) {
                self.face_alive[f] = false;
                self.live_faces -= 1;
            } else {
                for v in self.faces[f].iter_mut() {
                    if *v == remove {
                        *v = keep;
                    }
                }
                self.vertex_faces[keep].push(f);
            }
        }
        self.alive[remove] = false;
        self.stamps[keep] += 1;
        self.stamps[remove] += 1;
    }
}

impl QuadricErrorSimplifier {
    /// Create new quadric error simplifier with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom parameters
    pub fn with_params(
        max_edge_length: Option<f32>,
        preserve_boundary: bool,
        feature_angle_threshold: f32,
    ) -> Self {
        Self {
            max_edge_length,
            preserve_boundary,
            feature_angle_threshold,
        }
    }

    /// Compute plane equation from triangle vertices
    fn compute_plane(&self, v0: &Point3f, v1: &Point3f, v2: &Point3f) -> Vector4<f64> {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(&edge2).normalize();

        // Handle degenerate triangles
        if !normal.iter().all(|x| x.is_finite()) {
            return Vector4::new(0.0, 0.0, 1.0, 0.0);
        }

        let d = -normal.dot(&v0.coords);
        Vector4::new(normal.x as f64, normal.y as f64, normal.z as f64, d as f64)
    }

    /// Compute quadric matrix from plane equation
    fn plane_to_quadric(&self, plane: &Vector4<f64>) -> Matrix4<f64> {
        plane * plane.transpose()
    }

    /// Find boundary edges in the mesh
    fn find_boundary_edges(&self, mesh: &TriangleMesh) -> HashSet<(usize, usize)> {
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for face in &mesh.faces {
            for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        edge_count
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(edge, _)| edge)
            .collect()
    }

    /// Get all boundary vertices from boundary edges
    fn get_boundary_vertices(&self, boundary_edges: &HashSet<(usize, usize)>) -> HashSet<usize> {
        boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect()
    }

    /// Cost and best position for collapsing edge `a`-`b`, or `None` when the
    /// edge may not be collapsed.
    fn candidate(
        &self,
        w: &Working,
        a: usize,
        b: usize,
        fixed: &HashSet<usize>,
    ) -> Option<Candidate> {
        if fixed.contains(&a) || fixed.contains(&b) {
            return None;
        }
        let (pa, pb) = (w.positions[a], w.positions[b]);
        let length = (pa - pb).magnitude();
        if self.max_edge_length.is_some_and(|max| length > max) {
            return None;
        }
        let q = w.quadrics[a] + w.quadrics[b];
        let error = |p: &Point3f| {
            let h = Vector4::new(p.x as f64, p.y as f64, p.z as f64, 1.0);
            (h.transpose() * q * h)[0]
        };
        // Best point: where the quadric is smallest, when that is well defined
        // and near the edge; otherwise the cheapest of the ends and midpoint.
        let solved = q
            .fixed_view::<3, 3>(0, 0)
            .into_owned()
            .try_inverse()
            .map(|inv| -inv * q.fixed_view::<3, 1>(0, 3))
            .map(|v| Point3f::new(v[0] as f32, v[1] as f32, v[2] as f32))
            .filter(|p| {
                p.coords.iter().all(|c| c.is_finite())
                    && (p - pa).magnitude() <= 2.0 * length.max(f32::EPSILON)
            });
        let position = solved.unwrap_or_else(|| {
            [pa, pb, Point3f::from((pa.coords + pb.coords) * 0.5)]
                .into_iter()
                .min_by(|x, y| error(x).total_cmp(&error(y)))
                .unwrap()
        });
        Some(Candidate {
            cost: error(&position),
            keep: a,
            remove: b,
            position,
            stamps: [w.stamps[a], w.stamps[b]],
        })
    }
}

impl MeshSimplifier for QuadricErrorSimplifier {
    /// Simplify mesh with target reduction ratio
    fn simplify(&self, mesh: &TriangleMesh, reduction_ratio: f32) -> Result<TriangleMesh> {
        if mesh.is_empty() {
            return Err(Error::InvalidData("Mesh is empty".to_string()));
        }

        if !(0.0..=1.0).contains(&reduction_ratio) {
            return Err(Error::InvalidData(
                "Reduction ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        if reduction_ratio == 0.0 {
            return Ok(mesh.clone());
        }
        mesh.validate_faces()?;

        let target_face_count = ((1.0 - reduction_ratio) * mesh.faces.len() as f32) as usize;
        let n = mesh.vertices.len();

        let mut w = Working {
            positions: mesh.vertices.clone(),
            quadrics: vec![Matrix4::zeros(); n],
            stamps: vec![0; n],
            alive: vec![true; n],
            faces: mesh.faces.clone(),
            face_alive: vec![true; mesh.faces.len()],
            vertex_faces: vec![Vec::new(); n],
            live_faces: mesh.faces.len(),
        };
        for (f, face) in mesh.faces.iter().enumerate() {
            let [a, b, c] = face.map(|v| mesh.vertices[v]);
            let quadric = self.plane_to_quadric(&self.compute_plane(&a, &b, &c));
            for &v in face {
                w.quadrics[v] += quadric;
                w.vertex_faces[v].push(f);
            }
        }

        // Boundary vertices stay where they are when preserving the boundary.
        let boundary_edges = self.find_boundary_edges(mesh);
        let fixed = if self.preserve_boundary {
            self.get_boundary_vertices(&boundary_edges)
        } else {
            HashSet::new()
        };

        let mut heap = BinaryHeap::new();
        for face in &mesh.faces {
            for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                if a < b {
                    heap.extend(self.candidate(&w, a, b, &fixed));
                }
            }
        }

        let mut collapses = 0;
        while w.live_faces > target_face_count {
            let Some(c) = heap.pop() else { break };
            let current = c.stamps == [w.stamps[c.keep], w.stamps[c.remove]];
            if !current || !w.alive[c.keep] || !w.alive[c.remove] {
                continue;
            }
            if !w.can_collapse(c.keep, c.remove, c.position) {
                continue;
            }
            w.collapse(c.keep, c.remove, c.position);
            collapses += 1;
            // Only edges at the kept vertex changed cost.
            for neighbor in w.neighbors(c.keep) {
                heap.extend(self.candidate(&w, c.keep, neighbor, &fixed));
            }
        }

        if collapses == 0 && w.live_faces > target_face_count {
            return Err(self.no_collapse_error(mesh, &boundary_edges));
        }

        // Compact: keep live faces and the vertices they use.
        let mut new_index = vec![usize::MAX; n];
        let mut vertices = Vec::new();
        let faces = w
            .faces
            .iter()
            .zip(&w.face_alive)
            .filter(|(_, &alive)| alive)
            .map(|(face, _)| {
                face.map(|v| {
                    if new_index[v] == usize::MAX {
                        new_index[v] = vertices.len();
                        vertices.push(w.positions[v]);
                    }
                    new_index[v]
                })
            })
            .collect();
        Ok(TriangleMesh::from_vertices_and_faces(vertices, faces))
    }
}

impl QuadricErrorSimplifier {
    /// Explain why not a single edge could be collapsed, instead of quietly
    /// returning the mesh unchanged.
    fn no_collapse_error(
        &self,
        mesh: &TriangleMesh,
        boundary_edges: &HashSet<(usize, usize)>,
    ) -> Error {
        let on_boundary = self.get_boundary_vertices(boundary_edges).len();
        if self.preserve_boundary && on_boundary > 0 {
            Error::Algorithm(format!(
                "Could not simplify: every edge touches an open edge of the mesh \
                 ({on_boundary} of {} vertices are on a hole or border), and \
                 preserve_boundary keeps those vertices fixed. Fill the holes or \
                 set preserve_boundary to false.",
                mesh.vertices.len()
            ))
        } else {
            Error::Algorithm(
                "Could not simplify: no edge can be collapsed (check max_edge_length)".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    /// An n x n vertex grid on a tilted plane, wound so normals point up.
    fn tilted_grid(n: usize) -> TriangleMesh {
        let vertices = (0..n * n)
            .map(|i| {
                let (x, y) = (
                    (i % n) as f32 / (n - 1) as f32,
                    (i / n) as f32 / (n - 1) as f32,
                );
                Point3f::new(x, y, 0.37 * x + 0.21 * y)
            })
            .collect();
        let mut faces = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let v = j * n + i;
                faces.push([v, v + 1, v + n]);
                faces.push([v + 1, v + n + 1, v + n]);
            }
        }
        TriangleMesh::from_vertices_and_faces(vertices, faces)
    }

    /// Edges used by more than two faces, and edges used twice in the same
    /// direction (two faces with opposite winding).
    fn bad_edges(mesh: &TriangleMesh) -> (usize, usize) {
        let mut uses: HashMap<(usize, usize), Vec<bool>> = HashMap::new();
        for f in &mesh.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                uses.entry((a.min(b), a.max(b))).or_default().push(a < b);
            }
        }
        let non_manifold = uses.values().filter(|u| u.len() > 2).count();
        let same_direction = uses
            .values()
            .filter(|u| u.len() == 2 && u[0] == u[1])
            .count();
        (non_manifold, same_direction)
    }

    #[test]
    fn test_simplified_tilted_grid_has_no_flipped_faces() {
        let mesh = tilted_grid(30);
        let simplifier = QuadricErrorSimplifier::with_params(None, false, 0.5);
        let out = simplifier.simplify(&mesh, 0.5).unwrap();
        // One collapse removes two faces, so the count can land one below.
        assert!((mesh.faces.len() / 2 - 1..=mesh.faces.len() / 2).contains(&out.faces.len()));
        let flipped = out
            .faces
            .iter()
            .filter(|f| {
                let [a, b, c] = f.map(|v| out.vertices[v]);
                (b - a).cross(&(c - a)).z <= 0.0
            })
            .count();
        assert_eq!(flipped, 0, "faces flipped over");
        assert_eq!(
            bad_edges(&out),
            (0, 0),
            "non-manifold or inconsistent edges"
        );
        // The grid is planar, so every kept vertex stays on the plane.
        assert!(out
            .vertices
            .iter()
            .all(|p| (p.z - 0.37 * p.x - 0.21 * p.y).abs() < 1e-4));
    }

    #[test]
    fn test_closed_mesh_stays_closed() {
        // Octahedron subdivided twice: a closed sphere-like mesh.
        let mut mesh = TriangleMesh::from_vertices_and_faces(
            vec![
                Point3f::new(1.0, 0.0, 0.0),
                Point3f::new(-1.0, 0.0, 0.0),
                Point3f::new(0.0, 1.0, 0.0),
                Point3f::new(0.0, -1.0, 0.0),
                Point3f::new(0.0, 0.0, 1.0),
                Point3f::new(0.0, 0.0, -1.0),
            ],
            vec![
                [0, 2, 4],
                [2, 1, 4],
                [1, 3, 4],
                [3, 0, 4],
                [2, 0, 5],
                [1, 2, 5],
                [3, 1, 5],
                [0, 3, 5],
            ],
        );
        for _ in 0..3 {
            let mut midpoint: HashMap<(usize, usize), usize> = HashMap::new();
            let mut faces = Vec::new();
            for f in mesh.faces.clone() {
                let mut mid = |a: usize, b: usize| {
                    *midpoint.entry((a.min(b), a.max(b))).or_insert_with(|| {
                        let p = Point3f::from(
                            (mesh.vertices[a].coords + mesh.vertices[b].coords).normalize(),
                        );
                        mesh.vertices.push(p);
                        mesh.vertices.len() - 1
                    })
                };
                let (ab, bc, ca) = (mid(f[0], f[1]), mid(f[1], f[2]), mid(f[2], f[0]));
                faces.extend([[f[0], ab, ca], [ab, f[1], bc], [ca, bc, f[2]], [ab, bc, ca]]);
            }
            mesh.faces = faces;
        }
        for ratio in [0.5, 0.9] {
            let out = QuadricErrorSimplifier::new()
                .simplify(&mesh, ratio)
                .unwrap();
            let mut count: HashMap<(usize, usize), u32> = HashMap::new();
            for f in &out.faces {
                for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                    *count.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
            assert!(
                count.values().all(|&c| c == 2),
                "ratio {ratio}: mesh opened up"
            );
            assert_eq!(bad_edges(&out), (0, 0), "ratio {ratio}");
        }
    }

    #[test]
    fn test_quadric_error_simplifier_creation() {
        let simplifier = QuadricErrorSimplifier::new();
        assert!(simplifier.preserve_boundary);
        assert!(simplifier.max_edge_length.is_none());
    }

    #[test]
    fn test_plane_computation() {
        let simplifier = QuadricErrorSimplifier::new();
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        let plane = simplifier.compute_plane(&v0, &v1, &v2);

        // Should be z = 0 plane: 0x + 0y + 1z + 0 = 0
        assert!((plane[0]).abs() < 1e-6);
        assert!((plane[1]).abs() < 1e-6);
        assert!((plane[2] - 1.0).abs() < 1e-6);
        assert!((plane[3]).abs() < 1e-6);
    }

    #[test]
    fn test_empty_mesh() {
        let simplifier = QuadricErrorSimplifier::new();
        let mesh = TriangleMesh::new();

        let result = simplifier.simplify(&mesh, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_reduction() {
        let simplifier = QuadricErrorSimplifier::new();
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);

        let result = simplifier.simplify(&mesh, 0.0).unwrap();
        assert_eq!(result.vertex_count(), 3);
        assert_eq!(result.face_count(), 1);
    }

    #[test]
    fn test_invalid_reduction_ratio() {
        let simplifier = QuadricErrorSimplifier::new();
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);

        assert!(simplifier.simplify(&mesh, -0.1).is_err());
        assert!(simplifier.simplify(&mesh, 1.1).is_err());
    }

    #[test]
    fn test_quadric_matrix_computation() {
        let simplifier = QuadricErrorSimplifier::new();
        let plane = Vector4::new(1.0, 0.0, 0.0, -1.0); // x = 1 plane

        let quadric = simplifier.plane_to_quadric(&plane);

        // Check diagonal elements
        assert!((quadric[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((quadric[(1, 1)] - 0.0).abs() < 1e-10);
        assert!((quadric[(2, 2)] - 0.0).abs() < 1e-10);
        assert!((quadric[(3, 3)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tetrahedron_simplification() {
        let simplifier = QuadricErrorSimplifier::new();
        // Create a tetrahedron
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        let simplified = simplifier.simplify(&mesh, 0.5).unwrap();
        // Should have fewer faces
        assert!(simplified.face_count() <= mesh.face_count());
        assert!(simplified.vertex_count() <= mesh.vertex_count());
    }

    #[test]
    fn test_grid_boundary_preservation() {
        // Reproduce issue #52: 11x11 grid mesh simplification
        let simplifier = QuadricErrorSimplifier::new();

        // Create 11x11 grid at Z=-2000, spacing 400 units
        let grid_size = 11;
        let spacing = 400.0;
        let z = -2000.0;

        let mut vertices = Vec::new();
        for y in 0..grid_size {
            for x in 0..grid_size {
                vertices.push(Point3::new(x as f32 * spacing, y as f32 * spacing, z));
            }
        }

        // Generate triangle faces for the grid
        let mut faces = Vec::new();
        for y in 0..(grid_size - 1) {
            for x in 0..(grid_size - 1) {
                let top_left = y * grid_size + x;
                let top_right = top_left + 1;
                let bottom_left = (y + 1) * grid_size + x;
                let bottom_right = bottom_left + 1;

                // Two triangles per quad
                faces.push([top_left, bottom_left, top_right]);
                faces.push([top_right, bottom_left, bottom_right]);
            }
        }

        let mesh = TriangleMesh::from_vertices_and_faces(vertices.clone(), faces.clone());

        println!(
            "Original mesh: {} vertices, {} faces",
            mesh.vertex_count(),
            mesh.face_count()
        );

        // Identify original boundary vertices (edges of the grid)
        let mut original_boundary_verts = std::collections::HashSet::new();
        for i in 0..grid_size {
            // Top edge
            original_boundary_verts.insert(i);
            // Bottom edge
            original_boundary_verts.insert((grid_size - 1) * grid_size + i);
            // Left edge
            original_boundary_verts.insert(i * grid_size);
            // Right edge
            original_boundary_verts.insert(i * grid_size + (grid_size - 1));
        }

        println!(
            "Original boundary vertices: {}",
            original_boundary_verts.len()
        );

        // Debug: Check boundary edge detection
        let boundary_edges = simplifier.find_boundary_edges(&mesh);
        println!("Detected boundary edges: {}", boundary_edges.len());

        // Count unique vertices in boundary edges
        let mut detected_boundary_verts = std::collections::HashSet::new();
        for &(v1, v2) in &boundary_edges {
            detected_boundary_verts.insert(v1);
            detected_boundary_verts.insert(v2);
        }
        println!(
            "Detected boundary vertices: {}",
            detected_boundary_verts.len()
        );

        // Simplify with 50% reduction
        let simplified = simplifier.simplify(&mesh, 0.5).unwrap();

        println!(
            "Simplified mesh: {} vertices, {} faces",
            simplified.vertex_count(),
            simplified.face_count()
        );

        // Check if boundary vertices are preserved
        // We need to check if the boundary vertices' positions are still in the simplified mesh
        let mut boundary_positions = std::collections::HashSet::new();
        for &idx in &original_boundary_verts {
            let pos = vertices[idx];
            boundary_positions.insert((
                (pos.x * 100.0) as i32,
                (pos.y * 100.0) as i32,
                (pos.z * 100.0) as i32,
            ));
        }

        let mut simplified_positions = std::collections::HashSet::new();
        for &pos in &simplified.vertices {
            simplified_positions.insert((
                (pos.x * 100.0) as i32,
                (pos.y * 100.0) as i32,
                (pos.z * 100.0) as i32,
            ));
        }

        let preserved_boundary_count = boundary_positions
            .intersection(&simplified_positions)
            .count();
        println!(
            "Preserved boundary vertices: {}/{}",
            preserved_boundary_count,
            boundary_positions.len()
        );

        // Boundary vertices should be preserved when preserve_boundary is true
        // We expect all or most boundary vertices to remain
        assert!(preserved_boundary_count as f32 / boundary_positions.len() as f32 > 0.9,
            "Expected at least 90% of boundary vertices to be preserved, but only {}% were preserved",
            (preserved_boundary_count as f32 / boundary_positions.len() as f32 * 100.0));
    }

    /// A strip of triangles: every vertex lies on the outer border.
    fn strip(n: usize) -> TriangleMesh {
        let vertices = (0..n)
            .flat_map(|i| {
                [
                    Point3::new(i as f32, 0.0, 0.0),
                    Point3::new(i as f32, 1.0, 0.0),
                ]
            })
            .collect();
        let faces = (0..n - 1)
            .flat_map(|i| {
                let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
                [[a, c, b], [b, c, d]]
            })
            .collect();
        TriangleMesh::from_vertices_and_faces(vertices, faces)
    }

    #[test]
    fn test_all_boundary_mesh_reports_why_it_cannot_simplify() {
        let err = QuadricErrorSimplifier::new()
            .simplify(&strip(20), 0.5)
            .unwrap_err()
            .to_string();
        assert!(err.contains("preserve_boundary"), "{err}");
        assert!(err.contains("40 of 40 vertices"), "{err}");
        assert!(!err.contains("  "), "stray spaces: {err}");
    }

    #[test]
    fn test_all_boundary_mesh_simplifies_without_boundary_preservation() {
        let mesh = strip(20);
        let simplifier = QuadricErrorSimplifier::with_params(None, false, 45.0_f32.to_radians());
        let result = simplifier.simplify(&mesh, 0.5).unwrap();
        assert!(result.faces.len() < mesh.faces.len());
    }
}
