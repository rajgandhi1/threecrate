//! Mesh health and measurement: is a mesh watertight and printable, and how
//! big is it.

use crate::{Point3f, TriangleMesh};
use std::collections::{HashMap, VecDeque};

/// Health and measurement report for a triangle mesh, from [`analyze_mesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct MeshReport {
    pub vertices: usize,
    pub faces: usize,
    /// Vertices no face uses.
    pub unused_vertices: usize,
    /// Faces with zero area (repeated or collinear corners).
    pub degenerate_faces: usize,
    /// Faces with an index past the vertex list; such faces are otherwise
    /// ignored by this report.
    pub invalid_faces: usize,
    /// Distinct edges.
    pub edges: usize,
    /// Edges used by exactly one face: the rims of holes.
    pub boundary_edges: usize,
    /// Edges used by three or more faces.
    pub non_manifold_edges: usize,
    /// Edges used twice in the same direction: neighbouring faces wound
    /// opposite ways.
    pub inconsistent_edges: usize,
    /// Number of edges around each hole (closed loop of boundary edges),
    /// largest first.
    pub holes: Vec<usize>,
    /// Face count of each separate piece, largest first.
    pub components: Vec<usize>,
    /// No holes and no non-manifold edges.
    pub watertight: bool,
    /// Bounding box corners and size (all zero for an empty mesh).
    pub min: Point3f,
    pub max: Point3f,
    pub size: [f32; 3],
    /// Total area of all faces.
    pub surface_area: f64,
    /// Enclosed volume. Only defined for a watertight, consistently wound
    /// mesh; `None` otherwise.
    pub volume: Option<f64>,
    /// Watertight, manifold, consistently wound and facing outward.
    pub printable: bool,
    /// Why the mesh is not printable, in plain words (empty when it is).
    pub problems: Vec<String>,
}

/// Inspect `mesh`: topology problems (holes, non-manifold edges, flipped
/// faces, separate pieces), size, area, volume, and whether it can be 3D
/// printed as is.
pub fn analyze_mesh(mesh: &TriangleMesh) -> MeshReport {
    let n = mesh.vertices.len();
    let valid: Vec<[usize; 3]> = mesh
        .faces
        .iter()
        .copied()
        .filter(|f| f.iter().all(|&v| v < n))
        .collect();

    // Directed uses of every undirected edge: (a, b) with a < b, counted as
    // forward when the face runs a -> b.
    let mut edge_uses: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
    let mut used = vec![false; n];
    let mut degenerate = 0;
    let mut area = 0.0f64;
    let mut signed_volume = 0.0f64;
    for f in &valid {
        let [a, b, c] = f.map(|v| mesh.vertices[v].coords.cast::<f64>());
        let cross = (b - a).cross(&(c - a));
        let face_area = 0.5 * cross.norm();
        if face_area <= f64::EPSILON * (b - a).norm().max((c - a).norm()).powi(2) {
            degenerate += 1;
        }
        area += face_area;
        signed_volume += a.dot(&b.cross(&c)) / 6.0;
        f.iter().for_each(|&v| used[v] = true);
        // A face with a repeated corner is not a triangle; keep it out of the
        // edge topology so it doesn't make real edges look non-manifold.
        if f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
            continue;
        }
        for (u, v) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            let entry = edge_uses.entry((u.min(v), u.max(v))).or_default();
            if u < v {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
    }

    let mut boundary = 0;
    let mut non_manifold = 0;
    let mut inconsistent = 0;
    // Boundary edges in face direction, for tracing holes.
    let mut next: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&(a, b), &(forward, backward)) in &edge_uses {
        match forward + backward {
            1 => {
                boundary += 1;
                let (from, to) = if forward == 1 { (a, b) } else { (b, a) };
                next.entry(from).or_default().push(to);
            }
            2 if forward != 1 => inconsistent += 1,
            2 => {}
            _ => non_manifold += 1,
        }
    }

    // Walk boundary edges into closed loops, one per hole.
    let mut holes = Vec::new();
    let mut starts: Vec<usize> = next.keys().copied().collect();
    starts.sort_unstable();
    for start in starts {
        while next.get(&start).is_some_and(|out| !out.is_empty()) {
            let mut length = 0;
            let mut at = start;
            while let Some(to) = next.get_mut(&at).and_then(Vec::pop) {
                length += 1;
                at = to;
                if at == start {
                    break;
                }
            }
            holes.push(length);
        }
    }
    holes.sort_unstable_by(|a, b| b.cmp(a));

    let mut components: Vec<usize> = mesh_pieces(&valid).iter().map(Vec::len).collect();
    components.sort_unstable_by(|a, b| b.cmp(a));

    let (min, max) = bounds(mesh, &used);
    let size = [max.x - min.x, max.y - min.y, max.z - min.z];

    let watertight = !valid.is_empty() && boundary == 0 && non_manifold == 0;
    let oriented = inconsistent == 0;
    let volume = (watertight && oriented).then_some(signed_volume.abs());

    let mut problems = Vec::new();
    if valid.is_empty() {
        problems.push("the mesh has no faces".to_string());
    }
    if !holes.is_empty() {
        problems.push(format!(
            "{} hole(s) in the surface ({} open edges)",
            holes.len(),
            boundary
        ));
    }
    if non_manifold > 0 {
        problems.push(format!(
            "{non_manifold} edge(s) shared by more than two faces"
        ));
    }
    if !oriented {
        problems.push(format!(
            "{inconsistent} edge(s) between faces that face opposite ways"
        ));
    } else if watertight && signed_volume < 0.0 {
        problems.push("all faces point inward (the mesh is inside out)".to_string());
    }
    if mesh.faces.len() != valid.len() {
        problems.push(format!(
            "{} face(s) use vertex indices past the vertex list",
            mesh.faces.len() - valid.len()
        ));
    }

    MeshReport {
        vertices: n,
        faces: mesh.faces.len(),
        unused_vertices: used.iter().filter(|&&u| !u).count(),
        degenerate_faces: degenerate,
        invalid_faces: mesh.faces.len() - valid.len(),
        edges: edge_uses.len(),
        boundary_edges: boundary,
        non_manifold_edges: non_manifold,
        inconsistent_edges: inconsistent,
        holes,
        components,
        watertight,
        min,
        max,
        size,
        surface_area: area,
        volume,
        printable: problems.is_empty(),
        problems,
    }
}

/// Faces grouped into pieces connected through shared edges, as face indices.
/// Faces must index real vertices. `analyze_mesh` and `repair_for_print` use
/// this one definition of a piece, so their counts agree.
pub fn mesh_pieces(faces: &[[usize; 3]]) -> Vec<Vec<usize>> {
    let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, f) in faces.iter().enumerate() {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            by_edge.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let mut piece_of = vec![usize::MAX; faces.len()];
    let mut out: Vec<Vec<usize>> = Vec::new();
    for start in 0..faces.len() {
        if piece_of[start] != usize::MAX {
            continue;
        }
        let id = out.len();
        let mut members = vec![start];
        piece_of[start] = id;
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            let face = faces[f];
            for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                for &g in &by_edge[&(a.min(b), a.max(b))] {
                    if piece_of[g] == usize::MAX {
                        piece_of[g] = id;
                        members.push(g);
                        queue.push_back(g);
                    }
                }
            }
        }
        out.push(members);
    }
    out
}

/// Bounding box of the vertices faces use (all zero when there are none).
fn bounds(mesh: &TriangleMesh, used: &[bool]) -> (Point3f, Point3f) {
    let mut points = mesh
        .vertices
        .iter()
        .zip(used)
        .filter(|(p, &u)| u && p.coords.iter().all(|c| c.is_finite()))
        .map(|(p, _)| *p);
    let Some(first) = points.next() else {
        return (Point3f::origin(), Point3f::origin());
    };
    points.fold((first, first), |(lo, hi), p| {
        (
            Point3f::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z)),
            Point3f::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z)),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube, 12 triangles, wound so normals point outward.
    fn cube() -> TriangleMesh {
        let v = (0..8)
            .map(|i| Point3f::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32))
            .collect();
        let faces = vec![
            [0, 2, 1],
            [1, 2, 3], // z = 0, facing -z
            [4, 5, 6],
            [5, 7, 6], // z = 1, facing +z
            [0, 1, 4],
            [1, 5, 4], // y = 0
            [2, 6, 3],
            [3, 6, 7], // y = 1
            [0, 4, 2],
            [2, 4, 6], // x = 0
            [1, 3, 5],
            [3, 7, 5], // x = 1
        ];
        TriangleMesh::from_vertices_and_faces(v, faces)
    }

    #[test]
    fn closed_cube_is_printable_with_exact_measurements() {
        let r = analyze_mesh(&cube());
        assert!(r.watertight && r.printable, "{:?}", r.problems);
        assert_eq!(
            (r.edges, r.boundary_edges, r.non_manifold_edges),
            (18, 0, 0)
        );
        assert!(r.holes.is_empty());
        assert_eq!(r.components, vec![12]);
        assert!((r.surface_area - 6.0).abs() < 1e-9);
        assert!((r.volume.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(r.size, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn missing_faces_are_reported_as_one_hole() {
        let mut mesh = cube();
        mesh.faces.truncate(10); // drop the x = 1 side: one square hole
        let r = analyze_mesh(&mesh);
        assert!(!r.watertight && !r.printable);
        assert_eq!(r.holes, vec![4]);
        assert_eq!(r.volume, None);
        assert!(r.problems[0].contains("1 hole"), "{:?}", r.problems);
    }

    #[test]
    fn inside_out_and_flipped_faces_are_caught() {
        let mut inside_out = cube();
        for f in inside_out.faces.iter_mut() {
            f.swap(1, 2);
        }
        let r = analyze_mesh(&inside_out);
        assert!(r.watertight && !r.printable);
        assert!(
            (r.volume.unwrap() - 1.0).abs() < 1e-9,
            "volume is still the size"
        );
        assert!(r.problems.iter().any(|p| p.contains("inside out")));

        let mut one_flipped = cube();
        one_flipped.faces[0].swap(1, 2);
        let r = analyze_mesh(&one_flipped);
        assert_eq!(r.inconsistent_edges, 3);
        assert_eq!(r.volume, None);
        assert!(!r.printable);
    }

    #[test]
    fn non_manifold_pieces_degenerate_and_bad_indices_are_counted() {
        let mut mesh = cube();
        // A fin on an existing edge (0-1), a separate floating triangle, a
        // zero-area face and a face pointing past the vertex list.
        mesh.vertices.extend([
            Point3f::new(0.5, -1.0, 0.0),
            Point3f::new(5.0, 5.0, 5.0),
            Point3f::new(6.0, 5.0, 5.0),
            Point3f::new(5.0, 6.0, 5.0),
            Point3f::new(9.0, 9.0, 9.0), // unused
        ]);
        mesh.faces
            .extend([[0, 1, 8], [9, 10, 11], [9, 9, 10], [0, 1, 99]]);
        let r = analyze_mesh(&mesh);
        assert_eq!(r.non_manifold_edges, 1);
        assert_eq!(r.components, vec![13, 2]);
        assert_eq!(r.degenerate_faces, 1);
        assert_eq!(r.invalid_faces, 1);
        assert_eq!(r.unused_vertices, 1);
        assert!(!r.watertight && !r.printable);
    }

    #[test]
    fn empty_mesh_is_not_printable() {
        let r = analyze_mesh(&TriangleMesh::new());
        assert!(!r.printable && !r.watertight);
        assert_eq!(r.volume, None);
        assert_eq!(r.size, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn faces_touching_at_one_corner_are_separate_pieces() {
        // Two triangles share only vertex 0: two pieces, the same as repair sees.
        let v = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(-1.0, 0.0, 0.0),
            Point3f::new(0.0, -1.0, 0.0),
        ];
        let mesh = TriangleMesh::from_vertices_and_faces(v, vec![[0, 1, 2], [0, 3, 4]]);
        assert_eq!(analyze_mesh(&mesh).components, vec![1, 1]);
        assert_eq!(mesh_pieces(&mesh.faces), vec![vec![0], vec![1]]);
    }
}
