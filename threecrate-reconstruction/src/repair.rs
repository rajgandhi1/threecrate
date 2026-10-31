//! Turn a scan into a mesh that can be 3D printed.
//!
//! [`repair_for_print`] takes a point cloud or a mesh and returns a watertight,
//! consistently oriented mesh plus a report of what it did. Small, local fixes
//! come first (weld duplicate vertices, drop loose pieces, fix winding, close
//! small holes); when the mesh is still not watertight, or the input is a
//! point cloud, the surface is rebuilt with Poisson, which always closes it.

use crate::poisson::{poisson_reconstruction_with_normals, PoissonConfig};
use std::collections::{HashMap, HashSet, VecDeque};
use threecrate_core::{
    analyze_mesh, mesh_pieces, Error, MeshReport, Point3f, PointCloud, Result, TriangleMesh,
};
use threecrate_simplification::{MeshSimplifier, QuadricErrorSimplifier};

/// What to repair.
#[derive(Clone, Copy)]
pub enum RepairInput<'a> {
    Points(&'a PointCloud<Point3f>),
    Mesh(&'a TriangleMesh),
}

/// Options for [`repair_for_print`].
#[derive(Debug, Clone)]
pub struct RepairOptions {
    /// Remove the largest flat plane (a floor or turntable) from a point
    /// cloud before meshing. Ignored for meshes.
    pub remove_floor: bool,
    /// Loose pieces with fewer faces than this share of the largest piece are
    /// removed (default 0.05).
    pub min_piece_fraction: f32,
    /// Holes with at most this many edges are closed in place; larger
    /// openings make the surface be rebuilt (default 200).
    pub max_hole_edges: usize,
    /// Poisson grid depth when rebuilding (3 to 8, default 7).
    pub depth: u32,
    /// Reduce the result to about this many triangles. `None` keeps up to
    /// 250,000 and reduces anything larger to that.
    pub target_faces: Option<usize>,
    /// Units of the input. The result is always in millimetres, the unit
    /// slicers assume for STL (default [`Units::Auto`]).
    pub units: Units,
}

/// Units of the input to [`repair_for_print`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Units {
    /// Metres when the result's longest side is under 2 (a phone or LiDAR
    /// scan of an object), otherwise millimetres.
    #[default]
    Auto,
    Millimetres,
    Centimetres,
    Metres,
    Inches,
}

impl Units {
    /// Parse `"auto"`, `"mm"`, `"cm"`, `"m"` or `"in"`.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "mm" => Self::Millimetres,
            "cm" => Self::Centimetres,
            "m" => Self::Metres,
            "in" => Self::Inches,
            _ => return None,
        })
    }
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            remove_floor: false,
            min_piece_fraction: 0.05,
            max_hole_edges: 200,
            depth: 7,
            target_faces: None,
            units: Units::Auto,
        }
    }
}

/// What [`repair_for_print`] did.
#[derive(Debug, Clone)]
pub struct RepairReport {
    /// Health of the input mesh (`None` when the input was a point cloud).
    pub before: Option<MeshReport>,
    /// Health of the result.
    pub after: MeshReport,
    /// Each step taken, in plain words.
    pub steps: Vec<String>,
    /// What the input was multiplied by to reach millimetres (1 when it
    /// already was). `before` is in the input's units, `after` in mm.
    pub scale: f32,
}

const DEFAULT_MAX_FACES: usize = 250_000;

/// Make `input` printable: watertight, manifold, consistently wound and
/// facing outward, with loose junk removed and at a manageable size.
pub fn repair_for_print(
    input: RepairInput,
    options: &RepairOptions,
) -> Result<(TriangleMesh, RepairReport)> {
    let mut steps = Vec::new();
    let (mesh, before) = match input {
        RepairInput::Mesh(mesh) => {
            mesh.validate_faces()?;
            let before = analyze_mesh(mesh);
            let fixed = fix_mesh_in_place(mesh, options, &mut steps);
            let mesh = if analyze_mesh(&fixed).printable {
                fixed
            } else {
                steps.push(
                    "local fixes were not enough; rebuilt the surface from the mesh's points"
                        .to_string(),
                );
                let cloud = PointCloud::from_points(fixed.vertices.clone());
                rebuild(&cloud, options, &mut steps)?
            };
            (mesh, Some(before))
        }
        RepairInput::Points(cloud) => {
            let cloud = clean_points(cloud, options, &mut steps)?;
            (rebuild(&cloud, options, &mut steps)?, None)
        }
    };
    let mesh = decimate(mesh, options, &mut steps)?;
    let (mesh, scale) = to_millimetres(mesh, options.units, &mut steps);
    let after = analyze_mesh(&mesh);
    Ok((
        mesh,
        RepairReport {
            before,
            after,
            steps,
            scale,
        },
    ))
}

/// Scale `mesh` to millimetres; returns it and the factor used.
fn to_millimetres(
    mut mesh: TriangleMesh,
    units: Units,
    steps: &mut Vec<String>,
) -> (TriangleMesh, f32) {
    let (factor, name) = match units {
        Units::Millimetres => (1.0, "millimetres"),
        Units::Centimetres => (10.0, "centimetres"),
        Units::Metres => (1000.0, "metres"),
        Units::Inches => (25.4, "inches"),
        Units::Auto => {
            let longest = analyze_mesh(&mesh).size.into_iter().fold(0.0, f32::max);
            if longest > 0.0 && longest < 2.0 {
                (1000.0, "metres (the longest side was under 2)")
            } else {
                (1.0, "millimetres")
            }
        }
    };
    if factor != 1.0 {
        for v in &mut mesh.vertices {
            *v = Point3f::from(v.coords * factor);
        }
        steps.push(format!(
            "read the input as {name} and scaled it to millimetres (x{factor})"
        ));
    }
    (mesh, factor)
}

/// Weld, drop junk, fix winding and close small holes, without moving the
/// surface.
fn fix_mesh_in_place(
    mesh: &TriangleMesh,
    options: &RepairOptions,
    steps: &mut Vec<String>,
) -> TriangleMesh {
    let mut mesh = weld(mesh, steps);
    mesh = keep_large_pieces(mesh, options.min_piece_fraction, steps);
    orient(&mut mesh, steps);
    fill_holes(&mut mesh, options.max_hole_edges, steps);
    orient(&mut mesh, steps); // new faces may join pieces that were apart
    mesh
}

/// Merge vertices at (almost) the same position, then drop faces that
/// collapse or repeat and vertices nothing uses. Scans often store each
/// triangle's corners separately, which looks like a hole at every edge.
fn weld(mesh: &TriangleMesh, steps: &mut Vec<String>) -> TriangleMesh {
    let (lo, hi) = mesh.vertices.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(lo, hi), p| {
            (
                [lo[0].min(p.x), lo[1].min(p.y), lo[2].min(p.z)],
                [hi[0].max(p.x), hi[1].max(p.y), hi[2].max(p.z)],
            )
        },
    );
    let diagonal = (0..3).map(|i| (hi[i] - lo[i]).powi(2)).sum::<f32>().sqrt();
    let tolerance = (diagonal as f64 * 1e-7).max(1e-12);
    let mut index: HashMap<[i64; 3], usize> = HashMap::new();
    let mut vertices = Vec::new();
    let remap: Vec<usize> = mesh
        .vertices
        .iter()
        .map(|p| {
            let key = [p.x, p.y, p.z].map(|c| (c as f64 / tolerance).round() as i64);
            *index.entry(key).or_insert_with(|| {
                vertices.push(*p);
                vertices.len() - 1
            })
        })
        .collect();
    let mut seen = HashSet::new();
    let faces: Vec<[usize; 3]> = mesh
        .faces
        .iter()
        .map(|f| f.map(|v| remap[v]))
        .filter(|f| f[0] != f[1] && f[1] != f[2] && f[0] != f[2])
        .filter(|f| {
            let mut key = *f;
            key.sort_unstable();
            seen.insert(key)
        })
        .collect();
    let merged = mesh.vertices.len() - vertices.len();
    let dropped = mesh.faces.len() - faces.len();
    if merged > 0 || dropped > 0 {
        steps.push(format!(
            "merged {merged} duplicate vertices and dropped {dropped} degenerate or repeated faces"
        ));
    }
    compact(vertices, faces)
}

/// Keep only the vertices the faces use.
fn compact(vertices: Vec<Point3f>, faces: Vec<[usize; 3]>) -> TriangleMesh {
    let mut new_index = vec![usize::MAX; vertices.len()];
    let mut kept = Vec::new();
    let faces = faces
        .into_iter()
        .map(|f| {
            f.map(|v| {
                if new_index[v] == usize::MAX {
                    new_index[v] = kept.len();
                    kept.push(vertices[v]);
                }
                new_index[v]
            })
        })
        .collect();
    TriangleMesh::from_vertices_and_faces(kept, faces)
}

/// Drop pieces smaller than `fraction` of the largest one (floating junk).
fn keep_large_pieces(mesh: TriangleMesh, fraction: f32, steps: &mut Vec<String>) -> TriangleMesh {
    let groups = mesh_pieces(&mesh.faces);
    let largest = groups.iter().map(Vec::len).max().unwrap_or(0);
    let min = ((largest as f32 * fraction).ceil() as usize).max(1);
    let keep: Vec<&Vec<usize>> = groups.iter().filter(|g| g.len() >= min).collect();
    let removed = groups.len() - keep.len();
    if removed == 0 {
        return mesh;
    }
    steps.push(format!("removed {removed} small loose piece(s)"));
    let faces = keep
        .into_iter()
        .flat_map(|g| g.iter().map(|&f| mesh.faces[f]))
        .collect();
    compact(mesh.vertices, faces)
}

/// Wind every piece consistently, then make closed pieces face outward.
fn orient(mesh: &mut TriangleMesh, steps: &mut Vec<String>) {
    let mut by_edge: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, f) in mesh.faces.iter().enumerate() {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            by_edge.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let runs_forward =
        |f: &[usize; 3], a: usize, b: usize| (0..3).any(|k| f[k] == a && f[(k + 1) % 3] == b);
    let mut flipped = 0;
    let mut done = vec![false; mesh.faces.len()];
    for group in mesh_pieces(&mesh.faces) {
        let start = group[0];
        done[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            let face = mesh.faces[f];
            for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                let neighbours = &by_edge[&(a.min(b), a.max(b))];
                if neighbours.len() != 2 {
                    continue; // boundary or non-manifold: no single answer
                }
                for &g in neighbours {
                    if done[g] {
                        continue;
                    }
                    // A consistent neighbour runs the shared edge the other way.
                    if runs_forward(&mesh.faces[g], a, b) {
                        mesh.faces[g].swap(1, 2);
                        flipped += 1;
                    }
                    done[g] = true;
                    queue.push_back(g);
                }
            }
        }
        // Outward: a closed piece should enclose positive volume.
        let volume: f64 = group
            .iter()
            .map(|&f| {
                let [a, b, c] = mesh.faces[f].map(|v| mesh.vertices[v].coords.cast::<f64>());
                a.dot(&b.cross(&c))
            })
            .sum();
        if volume < 0.0 {
            for &f in &group {
                mesh.faces[f].swap(1, 2);
            }
            flipped += group.len();
        }
    }
    if flipped > 0 {
        steps.push(format!(
            "turned {flipped} face(s) so all of them face outward"
        ));
    }
}

/// Close each hole of up to `max_edges` edges with a fan of triangles around
/// its centre point.
fn fill_holes(mesh: &mut TriangleMesh, max_edges: usize, steps: &mut Vec<String>) {
    let mut uses: HashMap<(usize, usize), u32> = HashMap::new();
    for f in &mesh.faces {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            *uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    // Boundary edges in the direction their face runs them.
    let mut next: HashMap<usize, Vec<usize>> = HashMap::new();
    for f in &mesh.faces {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            if uses[&(a.min(b), a.max(b))] == 1 {
                next.entry(a).or_default().push(b);
            }
        }
    }
    let mut starts: Vec<usize> = next.keys().copied().collect();
    starts.sort_unstable();
    let (mut filled, mut skipped) = (0, 0);
    for start in starts {
        while next.get(&start).is_some_and(|out| !out.is_empty()) {
            let mut lp = vec![start];
            let mut at = start;
            while let Some(to) = next.get_mut(&at).and_then(Vec::pop) {
                if to == start {
                    break;
                }
                lp.push(to);
                at = to;
            }
            if lp.len() < 3 || lp.len() > max_edges {
                skipped += 1;
                continue;
            }
            let centre = lp
                .iter()
                .map(|&v| mesh.vertices[v].coords)
                .sum::<threecrate_core::Vector3f>()
                / lp.len() as f32;
            let c = mesh.vertices.len();
            mesh.vertices.push(Point3f::from(centre));
            // The hole's faces must run each rim edge the other way round.
            for i in 0..lp.len() {
                mesh.faces.push([lp[(i + 1) % lp.len()], lp[i], c]);
            }
            filled += 1;
        }
    }
    if filled > 0 {
        steps.push(format!("closed {filled} hole(s)"));
    }
    if skipped > 0 {
        steps.push(format!(
            "left {skipped} opening(s) too large to close in place"
        ));
    }
}

/// Clean a point cloud before meshing: drop stray points and, if asked, the
/// floor.
fn clean_points(
    cloud: &PointCloud<Point3f>,
    options: &RepairOptions,
    steps: &mut Vec<String>,
) -> Result<PointCloud<Point3f>> {
    let finite = PointCloud::from_points(
        cloud
            .points
            .iter()
            .copied()
            .filter(|p| p.coords.iter().all(|c| c.is_finite()))
            .collect(),
    );
    if finite.len() < 10 {
        return Err(Error::InvalidData(
            "too few points to build a printable mesh (need at least 10)".to_string(),
        ));
    }
    let mut cloud = threecrate_algorithms::statistical_outlier_removal(&finite, 20, 2.0)?;
    let stray = finite.len() - cloud.len();
    if stray > 0 {
        steps.push(format!("removed {stray} stray point(s)"));
    }
    if options.remove_floor {
        let spacing = median_spacing(&cloud.points);
        let plane = threecrate_algorithms::segment_plane(&cloud, 1.5 * spacing, 1000)?;
        // Only a plane holding a real share of the points is a floor.
        if plane.inliers.len() * 10 >= cloud.len() {
            let floor: HashSet<usize> = plane.inliers.iter().copied().collect();
            let kept: Vec<Point3f> = cloud
                .points
                .iter()
                .enumerate()
                .filter(|(i, _)| !floor.contains(i))
                .map(|(_, p)| *p)
                .collect();
            steps.push(format!("removed the floor ({} points)", floor.len()));
            cloud = PointCloud::from_points(kept);
        } else {
            steps.push("no floor found to remove".to_string());
        }
    }
    Ok(cloud)
}

/// Rebuild a closed surface with Poisson (normals estimated and oriented),
/// then drop loose pieces it may leave.
fn rebuild(
    cloud: &PointCloud<Point3f>,
    options: &RepairOptions,
    steps: &mut Vec<String>,
) -> Result<TriangleMesh> {
    let config = PoissonConfig {
        depth: options.depth,
        ..Default::default()
    };
    let mesh = poisson_reconstruction_with_normals(cloud, 10, &config)?;
    steps.push(format!(
        "built a closed surface with Poisson (depth {})",
        options.depth
    ));
    let mut mesh = keep_large_pieces(mesh, options.min_piece_fraction, steps);
    orient(&mut mesh, &mut Vec::new());
    Ok(mesh)
}

/// Reduce to the target size with quadric simplification.
fn decimate(
    mesh: TriangleMesh,
    options: &RepairOptions,
    steps: &mut Vec<String>,
) -> Result<TriangleMesh> {
    let target = options.target_faces.unwrap_or(DEFAULT_MAX_FACES).max(4);
    if mesh.faces.len() <= target {
        return Ok(mesh);
    }
    let before = mesh.faces.len();
    let ratio = 1.0 - target as f32 / before as f32;
    let simplifier = QuadricErrorSimplifier::with_params(None, false, 0.0);
    let reduced = simplifier.simplify(&mesh, ratio)?;
    steps.push(format!(
        "reduced from {before} to {} triangles",
        reduced.faces.len()
    ));
    Ok(reduced)
}

/// Median distance to the nearest other point, on a sample.
fn median_spacing(points: &[Point3f]) -> f32 {
    use threecrate_core::NearestNeighborSearch;
    let Ok(tree) = threecrate_algorithms::KdTree::new(points) else {
        return 0.01;
    };
    let step = (points.len() / 1000).max(1);
    let mut d: Vec<f32> = points
        .iter()
        .step_by(step)
        .filter_map(|p| {
            tree.find_k_nearest(p, 2)
                .into_iter()
                .map(|(_, d)| d)
                .find(|&d| d > 0.0)
        })
        .collect();
    if d.is_empty() {
        return 0.01;
    }
    let mid = d.len() / 2;
    d.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    d[mid]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test shapes are a few units across: read them as millimetres.
    const MM: RepairOptions = RepairOptions {
        remove_floor: false,
        min_piece_fraction: 0.05,
        max_hole_edges: 200,
        depth: 7,
        target_faces: None,
        units: Units::Millimetres,
    };

    fn sphere_points(n: usize, z_min: f32) -> Vec<Point3f> {
        let golden = std::f32::consts::PI * (1.0 + 5f32.sqrt());
        (0..n)
            .map(|i| {
                let t = i as f32 + 0.5;
                let phi = (1.0 - 2.0 * t / n as f32).acos();
                let theta = golden * t;
                Point3f::new(theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos())
            })
            .filter(|p| p.z >= z_min)
            .collect()
    }

    /// Unit cube, wound outward.
    fn cube() -> TriangleMesh {
        let v = (0..8)
            .map(|i| Point3f::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32))
            .collect();
        let faces = vec![
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ];
        TriangleMesh::from_vertices_and_faces(v, faces)
    }

    #[test]
    fn a_good_mesh_comes_back_unchanged() {
        let (mesh, report) = repair_for_print(RepairInput::Mesh(&cube()), &MM).unwrap();
        assert!(report.after.printable);
        assert!(report.steps.is_empty(), "{:?}", report.steps);
        assert_eq!(mesh.faces.len(), 12);
        assert!((report.after.volume.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn holes_flips_junk_and_unwelded_corners_are_fixed_in_place() {
        let cube = cube();
        // Every face gets its own three corners (as many scan exports do).
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for (i, f) in cube.faces.iter().enumerate() {
            if i == 10 {
                continue; // a hole
            }
            let mut f = *f;
            if i == 3 {
                f.swap(1, 2); // a flipped face
            }
            faces.push([0, 1, 2].map(|k| vertices.len() + k));
            vertices.extend(f.map(|v| cube.vertices[v]));
        }
        // A tiny floating triangle far away.
        let n = vertices.len();
        vertices.extend([
            Point3f::new(9.0, 9.0, 9.0),
            Point3f::new(9.1, 9.0, 9.0),
            Point3f::new(9.0, 9.1, 9.0),
        ]);
        faces.push([n, n + 1, n + 2]);
        let broken = TriangleMesh::from_vertices_and_faces(vertices, faces);
        assert!(!analyze_mesh(&broken).printable);

        // The cube has only 11 faces, so the 1-face floater is 9% of it; real
        // scans have thousands of faces and the 5% default fits them.
        let options = RepairOptions {
            min_piece_fraction: 0.2,
            ..MM
        };
        let (_, report) = repair_for_print(RepairInput::Mesh(&broken), &options).unwrap();
        assert!(
            report.after.printable,
            "{:?} / {:?}",
            report.after.problems, report.steps
        );
        assert!(
            (report.after.volume.unwrap() - 1.0).abs() < 1e-6,
            "kept the cube's shape"
        );
        let text = report.steps.join("; ");
        for expected in ["merged", "loose piece", "face outward", "closed 1 hole"] {
            assert!(text.contains(expected), "missing '{expected}' in: {text}");
        }
    }

    #[test]
    fn open_scan_points_become_a_closed_printable_mesh() {
        // A dome missing its bottom, like a scan of an object on a table.
        let cloud = PointCloud::from_points(sphere_points(3000, -0.4));
        let options = RepairOptions { depth: 6, ..MM };
        let (mesh, report) = repair_for_print(RepairInput::Points(&cloud), &options).unwrap();
        assert!(report.before.is_none());
        assert!(report.after.printable, "{:?}", report.after.problems);
        assert!(mesh.vertices.iter().all(|p| p.coords.norm() < 1.1));
    }

    #[test]
    fn floor_is_removed_when_asked_and_size_is_capped() {
        let mut points = sphere_points(3000, -1.0)
            .into_iter()
            .map(|p| Point3f::new(p.x, p.y, p.z + 1.2))
            .collect::<Vec<_>>();
        for i in 0..2500 {
            points.push(Point3f::new(
                (i % 50) as f32 * 0.08 - 2.0,
                (i / 50) as f32 * 0.08 - 2.0,
                0.0,
            ));
        }
        let cloud = PointCloud::from_points(points);
        let options = RepairOptions {
            remove_floor: true,
            depth: 6,
            target_faces: Some(2000),
            ..MM
        };
        let (mesh, report) = repair_for_print(RepairInput::Points(&cloud), &options).unwrap();
        assert!(
            report
                .steps
                .iter()
                .any(|s| s.starts_with("removed the floor")),
            "{:?}",
            report.steps
        );
        assert!(report.after.printable, "{:?}", report.after.problems);
        assert!(mesh.faces.len() <= 2000);
        // The floor sat at z = 0 and spread to +-2; only the ball is left.
        assert!(mesh
            .vertices
            .iter()
            .all(|p| p.x.abs() < 1.2 && p.y.abs() < 1.2));
    }

    #[test]
    fn small_scans_are_read_as_metres_and_written_in_millimetres() {
        // A 12 cm dome, as a phone scan app exports it.
        let points = sphere_points(3000, -0.4)
            .into_iter()
            .map(|p| Point3f::from(p.coords * 0.06))
            .collect();
        let cloud = PointCloud::from_points(points);
        let options = RepairOptions {
            depth: 6,
            ..Default::default()
        };
        let (_, report) = repair_for_print(RepairInput::Points(&cloud), &options).unwrap();
        assert_eq!(report.scale, 1000.0);
        let longest = report.after.size.into_iter().fold(0.0, f32::max);
        assert!((100.0..130.0).contains(&longest), "{longest}");
        assert!(report.steps.iter().any(|s| s.contains("metres")));

        let inches = RepairOptions {
            units: Units::Inches,
            ..MM
        };
        let (_, report) = repair_for_print(RepairInput::Mesh(&cube()), &inches).unwrap();
        assert_eq!(report.after.size, [25.4, 25.4, 25.4]);
        assert_eq!(Units::parse("CM"), Some(Units::Centimetres));
        assert_eq!(Units::parse("feet"), None);
    }

    #[test]
    fn too_few_points_is_a_clear_error() {
        let cloud = PointCloud::from_points(vec![Point3f::origin(); 5]);
        let err =
            repair_for_print(RepairInput::Points(&cloud), &RepairOptions::default()).unwrap_err();
        assert!(err.to_string().contains("too few points"));
    }
}
