//! Mesh smoothing algorithms
//!
//! Three progressively more sophisticated approaches are provided:
//!
//! | Algorithm | Volume preservation | Speed |
//! |-----------|---------------------|-------|
//! | Laplacian | Poor (shrinks mesh) | Fast  |
//! | Taubin    | Good               | Fast  |
//! | HC        | Good               | Fast  |
//!
//! All algorithms share the same pattern: build a one-ring adjacency list from
//! the mesh faces, then iteratively update vertex positions while keeping the
//! face connectivity unchanged.

use threecrate_core::{TriangleMesh, Result, Point3f, Vector3f, Error};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build a one-ring adjacency list from the mesh faces.
/// `adj[i]` contains the indices of all vertices directly connected to `i` by an edge.
fn build_adjacency(mesh: &TriangleMesh) -> Vec<Vec<usize>> {
    let n = mesh.vertices.len();
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for face in &mesh.faces {
        let [a, b, c] = *face;
        adj[a].insert(b);
        adj[a].insert(c);
        adj[b].insert(a);
        adj[b].insert(c);
        adj[c].insert(a);
        adj[c].insert(b);
    }
    adj.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Apply one uniform Laplacian displacement step.
/// Each vertex moves by `factor × (centroid_of_neighbours − vertex)`.
/// Isolated vertices (no neighbours) are left unchanged.
fn laplacian_step(vertices: &[Point3f], adj: &[Vec<usize>], factor: f32) -> Vec<Point3f> {
    vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let nbrs = &adj[i];
            if nbrs.is_empty() {
                return v;
            }
            let sum = nbrs
                .iter()
                .fold(Vector3f::zeros(), |acc, &j| acc + vertices[j].coords);
            let centroid = Point3f::from(sum / nbrs.len() as f32);
            v + (centroid - v) * factor
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Laplacian smoothing
// ---------------------------------------------------------------------------

/// Configuration for Laplacian smoothing.
#[derive(Debug, Clone)]
pub struct LaplacianSmoothingConfig {
    /// Number of smoothing iterations.
    pub iterations: usize,
    /// Per-iteration blend factor `λ ∈ (0, 1]`.
    /// Larger values = more aggressive smoothing per iteration.
    pub lambda: f32,
}

impl Default for LaplacianSmoothingConfig {
    fn default() -> Self {
        Self { iterations: 10, lambda: 0.5 }
    }
}

/// Laplacian mesh smoothing.
///
/// Each vertex is iteratively moved towards the average position of its
/// one-ring neighbours.  Simple and fast, but causes mesh shrinkage over many
/// iterations.
///
/// # Arguments
/// * `mesh`   - Input triangle mesh (connectivity unchanged in output)
/// * `config` - Smoothing parameters
///
/// # Returns
/// A new `TriangleMesh` with smoothed vertex positions and identical faces.
pub fn smooth_laplacian(
    mesh: &TriangleMesh,
    config: &LaplacianSmoothingConfig,
) -> Result<TriangleMesh> {
    if mesh.is_empty() {
        return Err(Error::InvalidData("Mesh is empty".to_string()));
    }
    if config.lambda <= 0.0 || config.lambda > 1.0 {
        return Err(Error::InvalidData("lambda must be in (0, 1]".to_string()));
    }
    if config.iterations == 0 {
        return Ok(mesh.clone());
    }

    let adj = build_adjacency(mesh);
    let mut verts = mesh.vertices.clone();
    for _ in 0..config.iterations {
        verts = laplacian_step(&verts, &adj, config.lambda);
    }
    Ok(TriangleMesh::from_vertices_and_faces(verts, mesh.faces.clone()))
}

// ---------------------------------------------------------------------------
// Taubin (μ|λ) smoothing
// ---------------------------------------------------------------------------

/// Configuration for Taubin (μ|λ) smoothing.
#[derive(Debug, Clone)]
pub struct TaubinSmoothingConfig {
    /// Number of full (λ + μ) iterations.
    pub iterations: usize,
    /// Positive step factor `λ ∈ (0, 1)`.
    pub lambda: f32,
    /// Negative step factor `μ < 0` (typically `μ ≈ −λ/(1 − λ·K)` for pass-band K).
    /// A safe default: `μ = −0.53` when `λ = 0.5`.
    pub mu: f32,
}

impl Default for TaubinSmoothingConfig {
    fn default() -> Self {
        Self { iterations: 10, lambda: 0.5, mu: -0.53 }
    }
}

/// Taubin (μ|λ) mesh smoothing.
///
/// Two alternating Laplacian passes per iteration: a positive (shrinking) λ
/// pass followed by a negative (expanding) μ pass.  The combination
/// suppresses low-frequency noise while avoiding the volume shrinkage of
/// plain Laplacian smoothing.
///
/// Reference: G. Taubin, "A Signal Processing Approach To Fair Surface Design" (1995).
///
/// # Arguments
/// * `mesh`   - Input triangle mesh
/// * `config` - Smoothing parameters (λ > 0, μ < 0, |μ| > λ recommended)
pub fn smooth_taubin(
    mesh: &TriangleMesh,
    config: &TaubinSmoothingConfig,
) -> Result<TriangleMesh> {
    if mesh.is_empty() {
        return Err(Error::InvalidData("Mesh is empty".to_string()));
    }
    if config.lambda <= 0.0 || config.lambda >= 1.0 {
        return Err(Error::InvalidData("lambda must be in (0, 1)".to_string()));
    }
    if config.mu >= 0.0 {
        return Err(Error::InvalidData("mu must be negative".to_string()));
    }
    if config.iterations == 0 {
        return Ok(mesh.clone());
    }

    let adj = build_adjacency(mesh);
    let mut verts = mesh.vertices.clone();
    for _ in 0..config.iterations {
        verts = laplacian_step(&verts, &adj, config.lambda);
        verts = laplacian_step(&verts, &adj, config.mu);
    }
    Ok(TriangleMesh::from_vertices_and_faces(verts, mesh.faces.clone()))
}

// ---------------------------------------------------------------------------
// HC (Humphrey's Classes) smoothing
// ---------------------------------------------------------------------------

/// Configuration for HC (Humphrey's Classes) smoothing.
#[derive(Debug, Clone)]
pub struct HcSmoothingConfig {
    /// Number of smoothing iterations.
    pub iterations: usize,
    /// Blend weight toward the *original* positions in the backward step.
    /// `α = 0`: correction relative to current positions (more smoothing).
    /// `α = 1`: correction relative to original positions (less drift).
    pub alpha: f32,
    /// Balance between per-vertex correction (`β = 1`) and
    /// neighbour-averaged correction (`β = 0`).  Values near 0.5 work well.
    pub beta: f32,
}

impl Default for HcSmoothingConfig {
    fn default() -> Self {
        Self { iterations: 10, alpha: 0.0, beta: 0.5 }
    }
}

/// HC (Humphrey's Classes) mesh smoothing.
///
/// A two-phase algorithm: first a standard Laplacian step, then a backward
/// correction that biases each vertex back toward a blend of its original and
/// Laplacian position.  Produces less shrinkage than plain Laplacian while
/// still effectively reducing noise.
///
/// Reference: J. Vollmer, R. Mencl, H. Müller,
/// "Improved Laplacian Smoothing of Noisy Surface Meshes" (1999).
///
/// # Arguments
/// * `mesh`   - Input triangle mesh
/// * `config` - Smoothing parameters (α, β both in [0, 1])
pub fn smooth_hc(
    mesh: &TriangleMesh,
    config: &HcSmoothingConfig,
) -> Result<TriangleMesh> {
    if mesh.is_empty() {
        return Err(Error::InvalidData("Mesh is empty".to_string()));
    }
    if !(0.0..=1.0).contains(&config.alpha) {
        return Err(Error::InvalidData("alpha must be in [0, 1]".to_string()));
    }
    if !(0.0..=1.0).contains(&config.beta) {
        return Err(Error::InvalidData("beta must be in [0, 1]".to_string()));
    }
    if config.iterations == 0 {
        return Ok(mesh.clone());
    }

    let adj = build_adjacency(mesh);
    let original: Vec<Point3f> = mesh.vertices.clone();
    let mut q: Vec<Point3f> = original.clone();

    for _ in 0..config.iterations {
        // --- Phase 1: Laplacian step → q_bar ---
        let q_bar: Vec<Point3f> = q
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let nbrs = &adj[i];
                if nbrs.is_empty() {
                    return v;
                }
                let sum = nbrs.iter().fold(Vector3f::zeros(), |acc, &j| acc + q[j].coords);
                Point3f::from(sum / nbrs.len() as f32)
            })
            .collect();

        // --- Phase 2: Backward difference vector ---
        // b[i] = q_bar[i] − (α·p[i] + (1−α)·q[i])
        let b: Vec<Vector3f> = (0..q.len())
            .map(|i| {
                let blend =
                    original[i].coords * config.alpha + q[i].coords * (1.0 - config.alpha);
                q_bar[i].coords - blend
            })
            .collect();

        // --- Phase 3: HC correction ---
        // q[i] ← q_bar[i] − (β·b[i] + (1−β) · avg_neighbour(b))
        q = (0..q.len())
            .map(|i| {
                let nbrs = &adj[i];
                let avg_b = if nbrs.is_empty() {
                    Vector3f::zeros()
                } else {
                    nbrs.iter().fold(Vector3f::zeros(), |acc, &j| acc + b[j])
                        / nbrs.len() as f32
                };
                let correction = b[i] * config.beta + avg_b * (1.0 - config.beta);
                Point3f::from(q_bar[i].coords - correction)
            })
            .collect();
    }

    Ok(TriangleMesh::from_vertices_and_faces(q, mesh.faces.clone()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3×3 grid mesh (8 triangles) with the centre vertex raised to z = 1.
    /// All boundary vertices sit at z = 0.
    fn make_spike_mesh() -> TriangleMesh {
        let verts = vec![
            Point3f::new(0.0, 0.0, 0.0), // 0
            Point3f::new(1.0, 0.0, 0.0), // 1
            Point3f::new(2.0, 0.0, 0.0), // 2
            Point3f::new(0.0, 1.0, 0.0), // 3
            Point3f::new(1.0, 1.0, 1.0), // 4  ← spike
            Point3f::new(2.0, 1.0, 0.0), // 5
            Point3f::new(0.0, 2.0, 0.0), // 6
            Point3f::new(1.0, 2.0, 0.0), // 7
            Point3f::new(2.0, 2.0, 0.0), // 8
        ];
        let faces = vec![
            [0, 1, 3], [1, 4, 3],
            [1, 2, 4], [2, 5, 4],
            [3, 4, 6], [4, 7, 6],
            [4, 5, 7], [5, 8, 7],
        ];
        TriangleMesh::from_vertices_and_faces(verts, faces)
    }

    /// Compute the centroid z of all vertices.
    fn centroid_z(mesh: &TriangleMesh) -> f32 {
        mesh.vertices.iter().map(|v| v.z).sum::<f32>() / mesh.vertices.len() as f32
    }

    // ---- topology preservation ----

    #[test]
    fn test_laplacian_preserves_topology() {
        let mesh = make_spike_mesh();
        let result = smooth_laplacian(&mesh, &LaplacianSmoothingConfig::default()).unwrap();
        assert_eq!(result.face_count(), mesh.face_count());
        assert_eq!(result.vertex_count(), mesh.vertex_count());
        assert_eq!(result.faces, mesh.faces);
    }

    #[test]
    fn test_taubin_preserves_topology() {
        let mesh = make_spike_mesh();
        let result = smooth_taubin(&mesh, &TaubinSmoothingConfig::default()).unwrap();
        assert_eq!(result.faces, mesh.faces);
    }

    #[test]
    fn test_hc_preserves_topology() {
        let mesh = make_spike_mesh();
        let result = smooth_hc(&mesh, &HcSmoothingConfig::default()).unwrap();
        assert_eq!(result.faces, mesh.faces);
    }

    // ---- spike is reduced ----

    #[test]
    fn test_laplacian_reduces_spike() {
        let mesh = make_spike_mesh();
        let before_z = mesh.vertices[4].z; // 1.0
        let result = smooth_laplacian(&mesh, &LaplacianSmoothingConfig::default()).unwrap();
        let after_z = result.vertices[4].z;
        assert!(
            after_z < before_z,
            "spike z should decrease: before={before_z}, after={after_z}"
        );
    }

    #[test]
    fn test_taubin_reduces_spike() {
        let mesh = make_spike_mesh();
        let before_z = mesh.vertices[4].z;
        let result = smooth_taubin(&mesh, &TaubinSmoothingConfig::default()).unwrap();
        let after_z = result.vertices[4].z;
        assert!(after_z < before_z, "Taubin should reduce spike: {before_z} → {after_z}");
    }

    #[test]
    fn test_hc_reduces_spike() {
        let mesh = make_spike_mesh();
        let before_z = mesh.vertices[4].z;
        let result = smooth_hc(&mesh, &HcSmoothingConfig::default()).unwrap();
        let after_z = result.vertices[4].z;
        assert!(after_z < before_z, "HC should reduce spike: {before_z} → {after_z}");
    }

    // ---- Taubin shrinks less than pure Laplacian ----

    #[test]
    fn test_taubin_less_shrinkage_than_laplacian() {
        // Build a larger, closed box mesh so that the global volume metric is
        // well-defined and the boundary effects of the small spike mesh do not
        // dominate the result.
        let verts = vec![
            Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(1.0, 1.0, 0.0), Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0), Point3f::new(1.0, 0.0, 1.0),
            Point3f::new(1.0, 1.0, 1.0), Point3f::new(0.0, 1.0, 1.0),
        ];
        let faces = vec![
            [0,2,1],[0,3,2], [4,5,6],[4,6,7],
            [0,1,5],[0,5,4], [3,7,6],[3,6,2],
            [0,4,7],[0,7,3], [1,2,6],[1,6,5],
        ];
        let mesh = TriangleMesh::from_vertices_and_faces(verts, faces);

        // Compute average distance of vertices from their centroid (proxy for "size")
        let avg_dist = |m: &TriangleMesh| {
            let c: Vector3f = m.vertices.iter().fold(Vector3f::zeros(), |acc, v| acc + v.coords)
                / m.vertices.len() as f32;
            m.vertices.iter().map(|v| (v.coords - c).magnitude()).sum::<f32>()
                / m.vertices.len() as f32
        };

        let original_spread = avg_dist(&mesh);

        let lap = smooth_laplacian(
            &mesh,
            &LaplacianSmoothingConfig { iterations: 50, lambda: 0.5 },
        ).unwrap();
        let tau = smooth_taubin(
            &mesh,
            &TaubinSmoothingConfig { iterations: 50, lambda: 0.5, mu: -0.53 },
        ).unwrap();

        let lap_spread = avg_dist(&lap);
        let tau_spread = avg_dist(&tau);

        // Laplacian should shrink the mesh more (lower spread) than Taubin
        assert!(
            tau_spread > lap_spread,
            "Taubin should preserve spread better than Laplacian: \
             original={original_spread:.4}, lap={lap_spread:.4}, tau={tau_spread:.4}"
        );
    }

    // ---- zero iterations returns clone ----

    #[test]
    fn test_laplacian_zero_iterations() {
        let mesh = make_spike_mesh();
        let result = smooth_laplacian(
            &mesh,
            &LaplacianSmoothingConfig { iterations: 0, lambda: 0.5 },
        )
        .unwrap();
        for (a, b) in mesh.vertices.iter().zip(result.vertices.iter()) {
            assert!((a - b).magnitude() < 1e-6);
        }
    }

    #[test]
    fn test_taubin_zero_iterations() {
        let mesh = make_spike_mesh();
        let result = smooth_taubin(
            &mesh,
            &TaubinSmoothingConfig { iterations: 0, lambda: 0.5, mu: -0.53 },
        )
        .unwrap();
        for (a, b) in mesh.vertices.iter().zip(result.vertices.iter()) {
            assert!((a - b).magnitude() < 1e-6);
        }
    }

    #[test]
    fn test_hc_zero_iterations() {
        let mesh = make_spike_mesh();
        let result =
            smooth_hc(&mesh, &HcSmoothingConfig { iterations: 0, alpha: 0.0, beta: 0.5 })
                .unwrap();
        for (a, b) in mesh.vertices.iter().zip(result.vertices.iter()) {
            assert!((a - b).magnitude() < 1e-6);
        }
    }

    // ---- error cases ----

    #[test]
    fn test_laplacian_empty_mesh() {
        let empty = TriangleMesh::new();
        assert!(smooth_laplacian(&empty, &LaplacianSmoothingConfig::default()).is_err());
    }

    #[test]
    fn test_laplacian_invalid_lambda() {
        let mesh = make_spike_mesh();
        assert!(smooth_laplacian(&mesh, &LaplacianSmoothingConfig { iterations: 1, lambda: 0.0 }).is_err());
        assert!(smooth_laplacian(&mesh, &LaplacianSmoothingConfig { iterations: 1, lambda: 1.5 }).is_err());
    }

    #[test]
    fn test_taubin_invalid_params() {
        let mesh = make_spike_mesh();
        // lambda out of range
        assert!(smooth_taubin(&mesh, &TaubinSmoothingConfig { iterations: 1, lambda: 0.0, mu: -0.53 }).is_err());
        // mu non-negative
        assert!(smooth_taubin(&mesh, &TaubinSmoothingConfig { iterations: 1, lambda: 0.5, mu: 0.1 }).is_err());
    }

    #[test]
    fn test_hc_invalid_params() {
        let mesh = make_spike_mesh();
        assert!(smooth_hc(&mesh, &HcSmoothingConfig { iterations: 1, alpha: -0.1, beta: 0.5 }).is_err());
        assert!(smooth_hc(&mesh, &HcSmoothingConfig { iterations: 1, alpha: 0.5, beta: 1.5 }).is_err());
    }

    #[test]
    fn test_taubin_empty_mesh() {
        let empty = TriangleMesh::new();
        assert!(smooth_taubin(&empty, &TaubinSmoothingConfig::default()).is_err());
    }

    #[test]
    fn test_hc_empty_mesh() {
        let empty = TriangleMesh::new();
        assert!(smooth_hc(&empty, &HcSmoothingConfig::default()).is_err());
    }

    // ---- more iterations = more smoothing ----

    #[test]
    fn test_laplacian_more_iterations_smoother() {
        let mesh = make_spike_mesh();
        let r1 = smooth_laplacian(&mesh, &LaplacianSmoothingConfig { iterations: 1, lambda: 0.5 }).unwrap();
        let r5 = smooth_laplacian(&mesh, &LaplacianSmoothingConfig { iterations: 5, lambda: 0.5 }).unwrap();
        // More iterations → spike vertex should be lower
        assert!(r5.vertices[4].z < r1.vertices[4].z);
    }
}
