//! Poisson surface reconstruction (Kazhdan, Bolitho & Hoppe, 2006).
//!
//! Oriented points are turned into a smooth vector field on a regular grid, and
//! the surface is the level set of the scalar field whose gradient best matches
//! it, i.e. the solution of the Poisson equation `Δχ = ∇·V`:
//!
//! 1. Spread each point's normal onto the grid (trilinear weights), then blur it
//!    by about one point spacing so sparse clouds still give a continuous field.
//! 2. Take the divergence and solve `Δχ = ∇·V` with conjugate gradients,
//!    starting from the solution at half the resolution (far fewer iterations).
//! 3. Extract the level set through the input points with marching cubes
//!    (one shared vertex per grid edge) and orient faces along the normals.
//!
//! Optionally, triangles far from every input point can be trimmed, since
//! Poisson closes open scans into a watertight blob.

use crate::marching_cubes::TRIANGLE_TABLE;
use rayon::prelude::*;
use std::collections::HashMap;
use threecrate_algorithms::KdTree;
use threecrate_core::{
    Error, NearestNeighborSearch, NormalPoint3f, Point3f, PointCloud, Result, TriangleMesh,
    Vector3f,
};

/// Configuration parameters for Poisson reconstruction
#[derive(Debug, Clone)]
pub struct PoissonConfig {
    /// The grid has 2^depth nodes per side (default 7, i.e. 128³). Each step up
    /// doubles the detail and costs about 8x the time and memory. Allowed: 3 to 8.
    pub depth: u32,
    /// How far each normal is spread, in point spacings (default 1.0). Larger
    /// values fill bigger holes and give a smoother surface.
    pub smoothing: f32,
    /// Added to the automatically chosen level (the average field value at the
    /// input points). Positive values shrink the surface, negative grow it.
    pub iso_offset: f32,
    /// Remove triangles farther than this from every input point (default
    /// `None`, keep the watertight surface). Useful for open scans.
    pub trim_distance: Option<f32>,
    /// Conjugate-gradient iteration cap per grid level (default 1000).
    pub max_iterations: usize,
    /// Stop when the residual falls below this fraction of the right-hand side
    /// (default 1e-4).
    pub tolerance: f64,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            depth: 7,
            smoothing: 1.0,
            iso_offset: 0.0,
            trim_distance: None,
            max_iterations: 1000,
            tolerance: 1e-4,
        }
    }
}

/// Points below this resolution are solved directly from zero; above it, from
/// the half-resolution solution.
const COARSEST_DEPTH: u32 = 4;
/// Empty space around the points, as a fraction of their largest extent.
const PADDING: f64 = 0.1;
/// Largest blur, in grid nodes, so very sparse clouds stay affordable.
const MAX_BLUR_SIGMA: f64 = 6.0;

/// A cubic grid of `n³` nodes, `h` apart, starting at `origin`. Values are
/// stored x-major: index `(i * n + j) * n + k`.
#[derive(Clone, Copy)]
struct Grid {
    n: usize,
    origin: [f64; 3],
    h: f64,
}

impl Grid {
    fn new(origin: [f64; 3], size: f64, depth: u32) -> Self {
        let n = 1usize << depth;
        Self {
            n,
            origin,
            h: size / (n - 1) as f64,
        }
    }

    fn len(&self) -> usize {
        self.n * self.n * self.n
    }

    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.n + j) * self.n + k
    }

    fn coords(&self, index: usize) -> (usize, usize, usize) {
        (
            index / (self.n * self.n),
            (index / self.n) % self.n,
            index % self.n,
        )
    }

    fn is_boundary(&self, i: usize, j: usize, k: usize) -> bool {
        let last = self.n - 1;
        i == 0 || j == 0 || k == 0 || i == last || j == last || k == last
    }

    /// The 8 surrounding nodes of a world position, with trilinear weights.
    fn trilinear(&self, p: [f64; 3]) -> [(usize, f64); 8] {
        let mut base = [0usize; 3];
        let mut frac = [0f64; 3];
        for axis in 0..3 {
            let u = ((p[axis] - self.origin[axis]) / self.h).clamp(0.0, (self.n - 1) as f64);
            let cell = (u.floor() as usize).min(self.n - 2);
            base[axis] = cell;
            frac[axis] = u - cell as f64;
        }
        let mut out = [(0usize, 0f64); 8];
        for (c, slot) in out.iter_mut().enumerate() {
            let (di, dj, dk) = (c >> 2 & 1, c >> 1 & 1, c & 1);
            let w = (if di == 1 { frac[0] } else { 1.0 - frac[0] })
                * (if dj == 1 { frac[1] } else { 1.0 - frac[1] })
                * (if dk == 1 { frac[2] } else { 1.0 - frac[2] });
            *slot = (self.index(base[0] + di, base[1] + dj, base[2] + dk), w);
        }
        out
    }

    fn sample(&self, values: &[f32], p: [f64; 3]) -> f64 {
        self.trilinear(p)
            .iter()
            .map(|&(index, w)| w * values[index] as f64)
            .sum()
    }
}

/// Oriented input points in f64, with unit normals.
struct Samples {
    positions: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
}

impl Samples {
    fn from_cloud(cloud: &PointCloud<NormalPoint3f>) -> Result<Self> {
        let mut positions = Vec::with_capacity(cloud.points.len());
        let mut normals = Vec::with_capacity(cloud.points.len());
        for point in &cloud.points {
            let p = point.position;
            let n = point.normal;
            let length = n.norm();
            if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite() && length.is_finite())
                || length < 1e-12
            {
                continue; // no usable position or direction
            }
            positions.push([p.x as f64, p.y as f64, p.z as f64]);
            normals.push([
                (n.x / length) as f64,
                (n.y / length) as f64,
                (n.z / length) as f64,
            ]);
        }
        Ok(Self { positions, normals })
    }
}

/// Spread normals onto the grid and blur them (one array per axis).
fn vector_field(grid: &Grid, samples: &Samples, spacing: f64, smoothing: f32) -> [Vec<f32>; 3] {
    let mut field = [
        vec![0f32; grid.len()],
        vec![0f32; grid.len()],
        vec![0f32; grid.len()],
    ];
    for (p, n) in samples.positions.iter().zip(&samples.normals) {
        for (index, w) in grid.trilinear(*p) {
            for axis in 0..3 {
                field[axis][index] += (w * n[axis]) as f32;
            }
        }
    }
    let sigma = (smoothing as f64 * spacing / grid.h).clamp(0.5, MAX_BLUR_SIGMA);
    for component in field.iter_mut() {
        for axis in 0..3 {
            *component = blur_axis(grid, component, axis, sigma);
        }
    }
    field
}

/// One pass of a Gaussian blur along `axis`.
fn blur_axis(grid: &Grid, values: &[f32], axis: usize, sigma: f64) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|d| (-(d * d) as f64 / (2.0 * sigma * sigma)).exp() as f32)
        .collect();
    let n = grid.n as isize;
    let stride = [grid.n * grid.n, grid.n, 1][axis] as isize;
    (0..values.len())
        .into_par_iter()
        .map(|index| {
            let (i, j, k) = grid.coords(index);
            let pos = [i, j, k][axis] as isize;
            let mut sum = 0f32;
            for (offset, w) in (-radius..=radius).zip(&kernel) {
                let q = pos + offset;
                if q >= 0 && q < n {
                    sum += w * values[(index as isize + offset * stride) as usize];
                }
            }
            sum
        })
        .collect()
}

/// Right-hand side of the scaled system: `-h² ∇·V` at interior nodes.
fn rhs(grid: &Grid, field: &[Vec<f32>; 3]) -> Vec<f32> {
    let strides = [grid.n * grid.n, grid.n, 1];
    let h = grid.h as f32;
    (0..grid.len())
        .into_par_iter()
        .map(|index| {
            let (i, j, k) = grid.coords(index);
            if grid.is_boundary(i, j, k) {
                return 0.0;
            }
            let divergence: f32 = (0..3)
                .map(|axis| {
                    (field[axis][index + strides[axis]] - field[axis][index - strides[axis]])
                        / (2.0 * h)
                })
                .sum();
            -h * h * divergence
        })
        .collect()
}

/// `y = A x` for the scaled negative Laplacian `A x = 6x - Σ neighbours`, with
/// `x = 0` held on the boundary.
fn apply(grid: &Grid, x: &[f32], y: &mut [f32]) {
    let s = [grid.n * grid.n, grid.n, 1];
    y.par_iter_mut().enumerate().for_each(|(index, out)| {
        let (i, j, k) = grid.coords(index);
        *out = if grid.is_boundary(i, j, k) {
            0.0
        } else {
            6.0 * x[index]
                - x[index - s[0]]
                - x[index + s[0]]
                - x[index - s[1]]
                - x[index + s[1]]
                - x[index - s[2]]
                - x[index + s[2]]
        };
    });
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.par_iter()
        .zip(b)
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum()
}

/// Conjugate gradients for `A x = b`, starting from `x`.
fn conjugate_gradient(grid: &Grid, b: &[f32], x: &mut [f32], config: &PoissonConfig) {
    let b_norm = dot(b, b).sqrt();
    if b_norm == 0.0 {
        return;
    }
    let mut ax = vec![0f32; x.len()];
    apply(grid, x, &mut ax);
    let mut r: Vec<f32> = b.par_iter().zip(&ax).map(|(&bi, &ai)| bi - ai).collect();
    let mut p = r.clone();
    let mut rr = dot(&r, &r);
    let mut ap = ax;

    for _ in 0..config.max_iterations {
        if rr.sqrt() <= config.tolerance * b_norm {
            break;
        }
        apply(grid, &p, &mut ap);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break;
        }
        let alpha = (rr / pap) as f32;
        x.par_iter_mut()
            .zip(&p)
            .for_each(|(xi, &pi)| *xi += alpha * pi);
        r.par_iter_mut()
            .zip(&ap)
            .for_each(|(ri, &api)| *ri -= alpha * api);
        let rr_new = dot(&r, &r);
        let beta = (rr_new / rr) as f32;
        rr = rr_new;
        p.par_iter_mut()
            .zip(&r)
            .for_each(|(pi, &ri)| *pi = ri + beta * *pi);
    }
}

/// Solve on a grid of 2^depth nodes per side, starting from the solution at
/// half the resolution.
fn solve(
    origin: [f64; 3],
    size: f64,
    depth: u32,
    samples: &Samples,
    spacing: f64,
    config: &PoissonConfig,
) -> (Grid, Vec<f32>) {
    let grid = Grid::new(origin, size, depth);
    let field = vector_field(&grid, samples, spacing, config.smoothing);
    let b = rhs(&grid, &field);
    drop(field);

    let mut x = if depth > COARSEST_DEPTH {
        let (coarse, coarse_x) = solve(origin, size, depth - 1, samples, spacing, config);
        (0..grid.len())
            .into_par_iter()
            .map(|index| {
                let (i, j, k) = grid.coords(index);
                if grid.is_boundary(i, j, k) {
                    return 0.0;
                }
                let p = [
                    origin[0] + i as f64 * grid.h,
                    origin[1] + j as f64 * grid.h,
                    origin[2] + k as f64 * grid.h,
                ];
                coarse.sample(&coarse_x, p) as f32
            })
            .collect()
    } else {
        vec![0f32; grid.len()]
    };
    conjugate_gradient(&grid, &b, &mut x, config);
    (grid, x)
}

/// Median distance from a point to its nearest neighbour, on a sample.
fn median_spacing(samples: &Samples) -> f64 {
    let points: Vec<Point3f> = samples
        .positions
        .iter()
        .map(|p| Point3f::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect();
    let Ok(tree) = KdTree::new(&points) else {
        return 0.0;
    };
    let step = (points.len() / 1000).max(1);
    let mut distances: Vec<f32> = points
        .iter()
        .step_by(step)
        .filter_map(|p| {
            tree.find_k_nearest(p, 2)
                .into_iter()
                .map(|(_, d)| d)
                .find(|&d| d > 0.0)
        })
        .collect();
    if distances.is_empty() {
        return 0.0;
    }
    let mid = distances.len() / 2;
    distances.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    distances[mid] as f64
}

/// Marching cubes over the solved field. Each grid edge gets one vertex, shared
/// by every cube that touches it, so the mesh has no cracks.
fn extract_surface(grid: &Grid, x: &[f32], iso: f32) -> TriangleMesh {
    // Cube corners as (di, dj, dk), and the 12 cube edges as corner pairs, in
    // the order the lookup table uses.
    const CORNERS: [[usize; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ];
    const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];
    let n = grid.n;
    let inside = |index: usize| x[index] < iso;

    // Triangles per cube slab, as (grid edge id) triples; a grid edge id is
    // `node * 3 + axis`, naming the edge from `node` towards +axis.
    let triangles: Vec<[usize; 3]> = (0..n - 1)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut out = Vec::new();
            for j in 0..n - 1 {
                for k in 0..n - 1 {
                    let corner = |c: usize| {
                        let [di, dj, dk] = CORNERS[c];
                        grid.index(i + di, j + dj, k + dk)
                    };
                    let mut case = 0;
                    for c in 0..8 {
                        if inside(corner(c)) {
                            case |= 1 << c;
                        }
                    }
                    if case == 0 || case == 255 {
                        continue;
                    }
                    let edge_id = |e: usize| {
                        let [a, b] = EDGES[e];
                        let (lo, hi) = (CORNERS[a], CORNERS[b]);
                        let axis = (0..3).find(|&d| lo[d] != hi[d]).unwrap();
                        let start = if lo[axis] < hi[axis] { a } else { b };
                        corner(start) * 3 + axis
                    };
                    for tri in TRIANGLE_TABLE[case].chunks(3) {
                        if tri[0] < 0 {
                            break;
                        }
                        out.push([
                            edge_id(tri[0] as usize),
                            edge_id(tri[1] as usize),
                            edge_id(tri[2] as usize),
                        ]);
                    }
                }
            }
            out
        })
        .collect();

    // One vertex per crossed grid edge, interpolated from its lower node.
    let strides = [n * n, n, 1];
    let mut vertex_of: HashMap<usize, usize> = HashMap::new();
    let mut vertices = Vec::new();
    let faces = triangles
        .into_iter()
        .map(|tri| {
            tri.map(|edge| {
                *vertex_of.entry(edge).or_insert_with(|| {
                    let (node, axis) = (edge / 3, edge % 3);
                    let (v0, v1) = (x[node], x[node + strides[axis]]);
                    let t = ((iso - v0) / (v1 - v0)).clamp(0.0, 1.0) as f64;
                    let (i, j, k) = grid.coords(node);
                    let mut p = [i as f64, j as f64, k as f64];
                    p[axis] += t;
                    vertices.push(Point3f::new(
                        (grid.origin[0] + p[0] * grid.h) as f32,
                        (grid.origin[1] + p[1] * grid.h) as f32,
                        (grid.origin[2] + p[2] * grid.h) as f32,
                    ));
                    vertices.len() - 1
                })
            })
        })
        .filter(|f: &[usize; 3]| f[0] != f[1] && f[1] != f[2] && f[0] != f[2])
        .collect();
    TriangleMesh::from_vertices_and_faces(vertices, faces)
}

fn face_normal(mesh: &TriangleMesh, face: &[usize; 3]) -> Vector3f {
    let [a, b, c] = face.map(|i| mesh.vertices[i]);
    (b - a).cross(&(c - a))
}

/// Make faces point the way the input normals point (along `∇χ`).
fn orient(mesh: &mut TriangleMesh, grid: &Grid, x: &[f32]) {
    let step = (mesh.faces.len() / 500).max(1);
    let agreement: f64 = mesh
        .faces
        .iter()
        .step_by(step)
        .map(|face| {
            let n = face_normal(mesh, face);
            let c = (mesh.vertices[face[0]].coords
                + mesh.vertices[face[1]].coords
                + mesh.vertices[face[2]].coords)
                / 3.0;
            let p = [c.x as f64, c.y as f64, c.z as f64];
            let mut gradient = [0f64; 3];
            for (axis, g) in gradient.iter_mut().enumerate() {
                let (mut hi, mut lo) = (p, p);
                hi[axis] += grid.h * 0.5;
                lo[axis] -= grid.h * 0.5;
                *g = grid.sample(x, hi) - grid.sample(x, lo);
            }
            (n.x as f64 * gradient[0] + n.y as f64 * gradient[1] + n.z as f64 * gradient[2])
                .signum()
        })
        .sum();
    if agreement < 0.0 {
        for face in mesh.faces.iter_mut() {
            face.swap(1, 2);
        }
    }
}

/// Keep only triangles whose centre is within `distance` of an input point.
fn trim(mesh: TriangleMesh, samples: &Samples, distance: f32) -> Result<TriangleMesh> {
    let points: Vec<Point3f> = samples
        .positions
        .iter()
        .map(|p| Point3f::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect();
    let tree = KdTree::new(&points)?;
    let limit = distance * distance;
    let keep: Vec<bool> = mesh
        .faces
        .par_iter()
        .map(|face| {
            let c = Point3f::from(
                (mesh.vertices[face[0]].coords
                    + mesh.vertices[face[1]].coords
                    + mesh.vertices[face[2]].coords)
                    / 3.0,
            );
            tree.find_nearest_bounded(&c, limit, None).is_some()
        })
        .collect();
    let faces: Vec<[usize; 3]> = mesh
        .faces
        .iter()
        .zip(&keep)
        .filter(|(_, &k)| k)
        .map(|(f, _)| *f)
        .collect();
    // Drop vertices no longer used.
    let mut new_index = vec![usize::MAX; mesh.vertices.len()];
    let mut vertices = Vec::new();
    let faces = faces
        .into_iter()
        .map(|f| {
            f.map(|v| {
                if new_index[v] == usize::MAX {
                    new_index[v] = vertices.len();
                    vertices.push(mesh.vertices[v]);
                }
                new_index[v]
            })
        })
        .collect();
    Ok(TriangleMesh::from_vertices_and_faces(vertices, faces))
}

/// Poisson surface reconstruction from points with normals.
///
/// Normals should point out of the surface; they are normalised, and points
/// with a zero or non-finite normal are ignored. Returns a closed surface
/// unless `config.trim_distance` is set.
pub fn poisson_reconstruction(
    cloud: &PointCloud<NormalPoint3f>,
    config: &PoissonConfig,
) -> Result<TriangleMesh> {
    if cloud.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }
    if !(3..=8).contains(&config.depth) {
        return Err(Error::InvalidData(format!(
            "Poisson depth must be between 3 and 8, got {}",
            config.depth
        )));
    }
    if config.smoothing.is_nan() || config.smoothing <= 0.0 {
        return Err(Error::InvalidData(
            "Poisson smoothing must be positive".to_string(),
        ));
    }

    let samples = Samples::from_cloud(cloud)?;
    if samples.positions.len() < 10 {
        return Err(Error::InvalidData(
            "Point cloud too small for Poisson reconstruction (need at least 10 points with normals)"
                .to_string(),
        ));
    }

    // Cubic box around the points, with padding so the surface stays inside.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &samples.positions {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    let extent = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
    if extent.is_nan() || extent <= 0.0 {
        return Err(Error::InvalidData(
            "Poisson reconstruction needs points that are not all in one spot".to_string(),
        ));
    }
    let size = extent * (1.0 + 2.0 * PADDING);
    let origin = [0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0 - size / 2.0);

    let spacing = median_spacing(&samples).max(extent * 1e-6);
    let (grid, x) = solve(origin, size, config.depth, &samples, spacing, config);

    // The surface is the level set through the input points.
    let iso = samples
        .positions
        .iter()
        .map(|p| grid.sample(&x, *p))
        .sum::<f64>()
        / samples.positions.len() as f64
        + config.iso_offset as f64;

    let mut mesh = extract_surface(&grid, &x, iso as f32);
    orient(&mut mesh, &grid, &x);

    if let Some(distance) = config.trim_distance {
        mesh = trim(mesh, &samples, distance)?;
    }
    if mesh.faces.is_empty() {
        return Err(Error::Algorithm(
            "Poisson reconstruction produced no triangles".to_string(),
        ));
    }
    Ok(mesh)
}

/// Poisson surface reconstruction with default configuration
pub fn poisson_reconstruction_default(cloud: &PointCloud<NormalPoint3f>) -> Result<TriangleMesh> {
    poisson_reconstruction(cloud, &PoissonConfig::default())
}

/// Estimate normals with `k` neighbours, orient them consistently, then run
/// Poisson reconstruction.
pub fn poisson_reconstruction_with_normals(
    cloud: &PointCloud<Point3f>,
    k: usize,
    config: &PoissonConfig,
) -> Result<TriangleMesh> {
    let mut normals_cloud = threecrate_algorithms::estimate_normals(cloud, k)?;
    threecrate_algorithms::orient_normals_consistent(&mut normals_cloud, k)?;
    poisson_reconstruction(&normals_cloud, config)
}

/// Wrapper function that matches the original API signature
#[deprecated(
    note = "Use poisson_reconstruction_default or poisson_reconstruction_with_normals instead"
)]
pub fn poisson_reconstruction_legacy(cloud: &PointCloud<Point3f>) -> Result<TriangleMesh> {
    poisson_reconstruction_with_normals(cloud, 20, &PoissonConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evenly spread points on a sphere with outward normals.
    fn sphere(n: usize, radius: f32, center: Point3f) -> PointCloud<NormalPoint3f> {
        let golden = std::f32::consts::PI * (1.0 + 5f32.sqrt());
        PointCloud::from_points(
            (0..n)
                .map(|i| {
                    let t = i as f32 + 0.5;
                    let phi = (1.0 - 2.0 * t / n as f32).acos();
                    let theta = golden * t;
                    let normal =
                        Vector3f::new(theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos());
                    NormalPoint3f {
                        position: center + normal * radius,
                        normal,
                    }
                })
                .collect(),
        )
    }

    /// Mean distance of the mesh's vertices from the sphere, relative to its radius.
    fn sphere_error(mesh: &TriangleMesh, radius: f32, center: Point3f) -> f32 {
        mesh.vertices
            .iter()
            .map(|v| ((v - center).norm() - radius).abs())
            .sum::<f32>()
            / mesh.vertices.len() as f32
            / radius
    }

    #[test]
    fn test_poisson_config_default() {
        let config = PoissonConfig::default();
        assert_eq!(config.depth, 7);
        assert!(config.trim_distance.is_none());
    }

    #[test]
    fn test_poisson_reconstruction_empty_cloud() {
        let cloud = PointCloud::<NormalPoint3f>::new();
        assert!(poisson_reconstruction(&cloud, &PoissonConfig::default()).is_err());
    }

    #[test]
    fn test_too_few_points_is_an_error() {
        let cloud = sphere(5, 1.0, Point3f::origin());
        let err = poisson_reconstruction(&cloud, &PoissonConfig::default()).unwrap_err();
        assert!(err.to_string().contains("too small"));
    }

    #[test]
    fn test_invalid_depth_is_an_error() {
        let cloud = sphere(200, 1.0, Point3f::origin());
        let config = PoissonConfig {
            depth: 12,
            ..Default::default()
        };
        assert!(poisson_reconstruction(&cloud, &config).is_err());
    }

    #[test]
    fn test_sphere_is_accurate_closed_and_outward() {
        let center = Point3f::new(10.0, -5.0, 2.0);
        let radius = 2.0;
        let cloud = sphere(2000, radius, center);
        let config = PoissonConfig {
            depth: 6,
            ..Default::default()
        };
        let mesh = poisson_reconstruction(&cloud, &config).unwrap();

        let error = sphere_error(&mesh, radius, center);
        assert!(error < 0.02, "mesh is {:.2}% off the sphere", error * 100.0);

        // Closed: every edge is shared by exactly two faces.
        let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
        for f in &mesh.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(
            edges.values().all(|&count| count == 2),
            "mesh has holes or open edges"
        );

        // Faces point outward.
        let outward = mesh
            .faces
            .iter()
            .filter(|f| {
                let c = (mesh.vertices[f[0]].coords
                    + mesh.vertices[f[1]].coords
                    + mesh.vertices[f[2]].coords)
                    / 3.0;
                face_normal(&mesh, f).dot(&(c - center.coords)) > 0.0
            })
            .count();
        assert!(outward as f32 > 0.99 * mesh.faces.len() as f32);
    }

    #[test]
    fn test_sparse_sphere_still_reconstructs() {
        // 300 points is sparse for a 128³ grid; the blur keeps the field continuous.
        let cloud = sphere(300, 1.0, Point3f::origin());
        let mesh = poisson_reconstruction_default(&cloud).unwrap();
        assert!(sphere_error(&mesh, 1.0, Point3f::origin()) < 0.03);
    }

    #[test]
    fn test_trim_removes_surface_far_from_points() {
        // Only the top half of the sphere is sampled. Poisson closes the bottom;
        // trimming removes it again.
        let full = sphere(3000, 1.0, Point3f::origin());
        let top = PointCloud::from_points(
            full.points
                .iter()
                .copied()
                .filter(|p| p.position.z > 0.0)
                .collect(),
        );
        let closed = poisson_reconstruction(
            &top,
            &PoissonConfig {
                depth: 6,
                ..Default::default()
            },
        )
        .unwrap();
        let trimmed = poisson_reconstruction(
            &top,
            &PoissonConfig {
                depth: 6,
                trim_distance: Some(0.1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(trimmed.faces.len() < closed.faces.len());
        assert!(trimmed.vertices.iter().all(|v| v.z > -0.15));
    }

    #[test]
    fn test_unnormalised_and_bad_normals_are_handled() {
        let mut cloud = sphere(1000, 1.0, Point3f::origin());
        for p in cloud.points.iter_mut().step_by(3) {
            p.normal *= 5.0; // not unit length
        }
        cloud.points[0].normal = Vector3f::zeros(); // unusable, skipped
        let mesh = poisson_reconstruction(
            &cloud,
            &PoissonConfig {
                depth: 6,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(sphere_error(&mesh, 1.0, Point3f::origin()) < 0.02);
    }

    #[test]
    fn test_plain_points_get_consistent_normals() {
        // estimate_normals alone leaves the lower half pointing inward; the
        // orientation step must fix that or the surface comes out wrong.
        let cloud = sphere(2000, 1.0, Point3f::origin());
        let plain = PointCloud::from_points(cloud.points.iter().map(|p| p.position).collect());
        let mesh = poisson_reconstruction_with_normals(
            &plain,
            10,
            &PoissonConfig {
                depth: 6,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(sphere_error(&mesh, 1.0, Point3f::origin()) < 0.02);
    }
}
