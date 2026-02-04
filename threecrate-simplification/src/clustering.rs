//! Clustering-based mesh simplification
//!
//! Implements the Rossignac & Borrel (1993) vertex clustering algorithm with
//! extensions for adaptive octree-based clustering, multiple representative
//! selection strategies, boundary-aware clustering, and sharp feature preservation.

use crate::MeshSimplifier;
use nalgebra::{Matrix4, Vector4};
use std::collections::{HashMap, HashSet};
use threecrate_core::{Error, Point3f, Result, TriangleMesh, Vector3f};

// ============================================================
// Configuration Types
// ============================================================

/// Strategy for selecting the representative vertex within a cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepresentativeStrategy {
    /// Arithmetic mean of all vertex positions in the cluster.
    Centroid,
    /// Weighted average using vertex valence (number of adjacent faces).
    WeightedAverage,
    /// Position that minimizes the summed quadric error for the cluster.
    MinimumError,
}

/// Clustering grid mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusteringMode {
    /// Uniform grid with automatically computed cell size from reduction ratio.
    Uniform,
    /// Adaptive octree that subdivides cells exceeding the error threshold.
    Adaptive {
        max_depth: u32,
        error_threshold: f64,
    },
}

// ============================================================
// Bounding Box
// ============================================================

#[derive(Debug, Clone, Copy)]
struct BBox {
    min: [f64; 3],
    max: [f64; 3],
}

impl BBox {
    fn from_vertices(vertices: &[Point3f]) -> Self {
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for v in vertices {
            for i in 0..3 {
                let c = v[i] as f64;
                if c < min[i] {
                    min[i] = c;
                }
                if c > max[i] {
                    max[i] = c;
                }
            }
        }
        BBox { min, max }
    }

    fn size(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    fn max_extent(&self) -> f64 {
        let s = self.size();
        s[0].max(s[1]).max(s[2])
    }

    fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
}

// ============================================================
// Octree Node (for adaptive clustering)
// ============================================================

#[derive(Debug)]
struct OctreeNode {
    bbox: BBox,
    children: Option<Box<[OctreeNode; 8]>>,
    vertex_indices: Vec<usize>,
    depth: u32,
}

impl OctreeNode {
    fn new(bbox: BBox, depth: u32) -> Self {
        OctreeNode {
            bbox,
            children: None,
            vertex_indices: Vec::new(),
            depth,
        }
    }

    fn contains(&self, p: &Point3f) -> bool {
        let eps = 1e-6;
        (p.x as f64) >= self.bbox.min[0] - eps
            && (p.x as f64) <= self.bbox.max[0] + eps
            && (p.y as f64) >= self.bbox.min[1] - eps
            && (p.y as f64) <= self.bbox.max[1] + eps
            && (p.z as f64) >= self.bbox.min[2] - eps
            && (p.z as f64) <= self.bbox.max[2] + eps
    }

    fn subdivide(&mut self) {
        let c = self.bbox.center();
        let mn = self.bbox.min;
        let mx = self.bbox.max;
        let d = self.depth + 1;

        let children = [
            OctreeNode::new(BBox { min: [mn[0], mn[1], mn[2]], max: [c[0], c[1], c[2]] }, d),
            OctreeNode::new(BBox { min: [c[0], mn[1], mn[2]], max: [mx[0], c[1], c[2]] }, d),
            OctreeNode::new(BBox { min: [mn[0], c[1], mn[2]], max: [c[0], mx[1], c[2]] }, d),
            OctreeNode::new(BBox { min: [c[0], c[1], mn[2]], max: [mx[0], mx[1], c[2]] }, d),
            OctreeNode::new(BBox { min: [mn[0], mn[1], c[2]], max: [c[0], c[1], mx[2]] }, d),
            OctreeNode::new(BBox { min: [c[0], mn[1], c[2]], max: [mx[0], c[1], mx[2]] }, d),
            OctreeNode::new(BBox { min: [mn[0], c[1], c[2]], max: [c[0], mx[1], mx[2]] }, d),
            OctreeNode::new(BBox { min: [c[0], c[1], c[2]], max: [mx[0], mx[1], mx[2]] }, d),
        ];
        self.children = Some(Box::new(children));
    }

    /// Insert a vertex index; if adaptive, subdivide when the cluster error exceeds
    /// the threshold and depth < max_depth.
    fn insert(
        &mut self,
        vi: usize,
        positions: &[Point3f],
        quadrics: &[Matrix4<f64>],
        max_depth: u32,
        error_threshold: f64,
    ) {
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                if child.contains(&positions[vi]) {
                    child.insert(vi, positions, quadrics, max_depth, error_threshold);
                    return;
                }
            }
            // Fallback: add to self if no child contains the point
            self.vertex_indices.push(vi);
            return;
        }

        self.vertex_indices.push(vi);

        // Consider subdividing
        if self.vertex_indices.len() > 1 && self.depth < max_depth {
            let cluster_error = compute_cluster_quadric_error(
                &self.vertex_indices,
                positions,
                quadrics,
            );
            if cluster_error > error_threshold {
                self.subdivide();
                let verts = std::mem::take(&mut self.vertex_indices);
                for v in verts {
                    self.insert(v, positions, quadrics, max_depth, error_threshold);
                }
            }
        }
    }

    /// Collect all leaf clusters (non-empty vertex lists).
    fn collect_clusters(&self, out: &mut Vec<Vec<usize>>) {
        if let Some(ref children) = self.children {
            for child in children.iter() {
                child.collect_clusters(out);
            }
        }
        if !self.vertex_indices.is_empty() {
            out.push(self.vertex_indices.clone());
        }
    }
}

// ============================================================
// Helpers
// ============================================================

fn compute_plane(v0: &Point3f, v1: &Point3f, v2: &Point3f) -> Vector4<f64> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let n = e1.cross(&e2).normalize();
    if !n.iter().all(|x| x.is_finite()) {
        return Vector4::new(0.0, 0.0, 1.0, 0.0);
    }
    let d = -n.dot(&v0.coords);
    Vector4::new(n.x as f64, n.y as f64, n.z as f64, d as f64)
}

fn plane_to_quadric(p: &Vector4<f64>) -> Matrix4<f64> {
    let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
    Matrix4::new(
        a * a, a * b, a * c, a * d,
        a * b, b * b, b * c, b * d,
        a * c, b * c, c * c, c * d,
        a * d, b * d, c * d, d * d,
    )
}

fn compute_quadrics(mesh: &TriangleMesh) -> Vec<Matrix4<f64>> {
    let mut quadrics = vec![Matrix4::zeros(); mesh.vertices.len()];
    for face in &mesh.faces {
        let plane = compute_plane(
            &mesh.vertices[face[0]],
            &mesh.vertices[face[1]],
            &mesh.vertices[face[2]],
        );
        let q = plane_to_quadric(&plane);
        for &vi in face {
            quadrics[vi] += q;
        }
    }
    quadrics
}

fn compute_vertex_valence(mesh: &TriangleMesh) -> Vec<usize> {
    let mut valence = vec![0usize; mesh.vertices.len()];
    for face in &mesh.faces {
        for &vi in face {
            valence[vi] += 1;
        }
    }
    valence
}

fn quadric_error_at(pos: &Point3f, q: &Matrix4<f64>) -> f64 {
    let v = Vector4::new(pos.x as f64, pos.y as f64, pos.z as f64, 1.0);
    (v.transpose() * q * v)[0].max(0.0)
}

fn compute_cluster_quadric_error(
    indices: &[usize],
    positions: &[Point3f],
    quadrics: &[Matrix4<f64>],
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    // Compute centroid
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut cz = 0.0f64;
    for &vi in indices {
        cx += positions[vi].x as f64;
        cy += positions[vi].y as f64;
        cz += positions[vi].z as f64;
    }
    let n = indices.len() as f64;
    let centroid = Point3f::new((cx / n) as f32, (cy / n) as f32, (cz / n) as f32);

    // Sum quadric errors at the centroid
    let mut total = 0.0;
    for &vi in indices {
        total += quadric_error_at(&centroid, &quadrics[vi]);
    }
    total
}

fn find_boundary_vertices(mesh: &TriangleMesh) -> HashSet<usize> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for face in &mesh.faces {
        let edges = [
            (face[0].min(face[1]), face[0].max(face[1])),
            (face[1].min(face[2]), face[1].max(face[2])),
            (face[2].min(face[0]), face[2].max(face[0])),
        ];
        for &e in &edges {
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }
    let mut boundary = HashSet::new();
    for ((v1, v2), count) in &edge_count {
        if *count == 1 {
            boundary.insert(*v1);
            boundary.insert(*v2);
        }
    }
    boundary
}

/// Detect vertices on sharp features using the dihedral angle between
/// adjacent face normals.
fn find_feature_vertices(mesh: &TriangleMesh, angle_threshold: f32) -> HashSet<usize> {
    let cos_threshold = angle_threshold.cos();

    // Compute face normals
    let face_normals: Vec<Vector3f> = mesh
        .faces
        .iter()
        .map(|f| {
            let e1 = mesh.vertices[f[1]] - mesh.vertices[f[0]];
            let e2 = mesh.vertices[f[2]] - mesh.vertices[f[0]];
            let n = e1.cross(&e2);
            let len = n.magnitude();
            if len > 1e-12 {
                n / len
            } else {
                Vector3f::new(0.0, 0.0, 1.0)
            }
        })
        .collect();

    // Build edge -> face adjacency
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (fi, face) in mesh.faces.iter().enumerate() {
        let edges = [
            (face[0].min(face[1]), face[0].max(face[1])),
            (face[1].min(face[2]), face[1].max(face[2])),
            (face[2].min(face[0]), face[2].max(face[0])),
        ];
        for &e in &edges {
            edge_faces.entry(e).or_default().push(fi);
        }
    }

    let mut feature_verts = HashSet::new();
    for ((v1, v2), faces) in &edge_faces {
        if faces.len() == 2 {
            let dot = face_normals[faces[0]].dot(&face_normals[faces[1]]);
            if dot < cos_threshold {
                feature_verts.insert(*v1);
                feature_verts.insert(*v2);
            }
        }
    }
    feature_verts
}

// ============================================================
// Representative Selection
// ============================================================

fn select_representative(
    cluster: &[usize],
    positions: &[Point3f],
    quadrics: &[Matrix4<f64>],
    valence: &[usize],
    strategy: RepresentativeStrategy,
) -> Point3f {
    match strategy {
        RepresentativeStrategy::Centroid => {
            let mut cx = 0.0f64;
            let mut cy = 0.0f64;
            let mut cz = 0.0f64;
            for &vi in cluster {
                cx += positions[vi].x as f64;
                cy += positions[vi].y as f64;
                cz += positions[vi].z as f64;
            }
            let n = cluster.len() as f64;
            Point3f::new((cx / n) as f32, (cy / n) as f32, (cz / n) as f32)
        }
        RepresentativeStrategy::WeightedAverage => {
            let mut wx = 0.0f64;
            let mut wy = 0.0f64;
            let mut wz = 0.0f64;
            let mut w_total = 0.0f64;
            for &vi in cluster {
                let w = (valence[vi].max(1)) as f64;
                wx += positions[vi].x as f64 * w;
                wy += positions[vi].y as f64 * w;
                wz += positions[vi].z as f64 * w;
                w_total += w;
            }
            if w_total > 0.0 {
                Point3f::new(
                    (wx / w_total) as f32,
                    (wy / w_total) as f32,
                    (wz / w_total) as f32,
                )
            } else {
                positions[cluster[0]]
            }
        }
        RepresentativeStrategy::MinimumError => {
            // Sum the quadrics for all vertices in the cluster
            let mut q_sum = Matrix4::zeros();
            for &vi in cluster {
                q_sum += quadrics[vi];
            }

            // Try to solve for the optimal position via the quadric
            let q3 = q_sum.fixed_view::<3, 3>(0, 0);
            let q1 = q_sum.fixed_view::<3, 1>(0, 3);

            if let Some(inv) = q3.try_inverse() {
                let p = -inv * q1;
                let candidate = Point3f::new(p[0] as f32, p[1] as f32, p[2] as f32);
                if candidate.x.is_finite() && candidate.y.is_finite() && candidate.z.is_finite() {
                    return candidate;
                }
            }

            // Fallback: pick the vertex with minimum quadric error
            let mut best_vi = cluster[0];
            let mut best_err = f64::MAX;
            for &vi in cluster {
                let err = quadric_error_at(&positions[vi], &q_sum);
                if err < best_err {
                    best_err = err;
                    best_vi = vi;
                }
            }
            positions[best_vi]
        }
    }
}

// ============================================================
// Clustering Simplifier
// ============================================================

/// Clustering-based mesh simplifier.
///
/// Uses vertex clustering (Rossignac & Borrel 1993) to rapidly simplify meshes.
/// Supports uniform grid and adaptive octree clustering modes, multiple
/// representative selection strategies, boundary preservation, and sharp
/// feature maintenance.
pub struct ClusteringSimplifier {
    /// Clustering grid mode (uniform or adaptive octree).
    pub mode: ClusteringMode,
    /// Strategy for choosing the representative position of each cluster.
    pub representative_strategy: RepresentativeStrategy,
    /// If true, boundary vertices are clustered only with other boundary
    /// vertices in the same cell, preventing boundary drift.
    pub preserve_boundary: bool,
    /// Dihedral angle threshold (in radians) for sharp feature detection.
    /// Edges with dihedral angles exceeding this are treated as features.
    pub feature_angle_threshold: f32,
}

impl Default for ClusteringSimplifier {
    fn default() -> Self {
        Self {
            mode: ClusteringMode::Uniform,
            representative_strategy: RepresentativeStrategy::Centroid,
            preserve_boundary: true,
            feature_angle_threshold: 45.0_f32.to_radians(),
        }
    }
}

impl ClusteringSimplifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(
        mode: ClusteringMode,
        representative_strategy: RepresentativeStrategy,
        preserve_boundary: bool,
        feature_angle_threshold: f32,
    ) -> Self {
        Self {
            mode,
            representative_strategy,
            preserve_boundary,
            feature_angle_threshold,
        }
    }

    /// Compute uniform grid cell size from the desired reduction ratio and mesh bounding box.
    /// Handles degenerate (planar/linear) meshes by using only non-degenerate dimensions.
    fn compute_cell_size(bbox: &BBox, num_vertices: usize, reduction_ratio: f32) -> f64 {
        let target_clusters = ((1.0 - reduction_ratio) * num_vertices as f32).max(1.0) as f64;
        let s = bbox.size();
        let eps = 1e-6;

        // Use only non-degenerate dimensions for cell size calculation
        let extents: Vec<f64> = s.iter().filter(|&&d| d > eps).copied().collect();
        let dim = extents.len();

        if dim == 0 {
            // All vertices at same point; any cell size works
            return 1.0;
        }

        // product(extents) / cell_size^dim ≈ target_clusters
        let product: f64 = extents.iter().product();
        (product / target_clusters).powf(1.0 / dim as f64)
    }

    /// Assign vertices to uniform grid cells, returning a map of cell key -> vertex indices.
    fn uniform_clustering(
        &self,
        mesh: &TriangleMesh,
        cell_size: f64,
        bbox: &BBox,
        boundary_verts: &HashSet<usize>,
        feature_verts: &HashSet<usize>,
    ) -> Vec<Vec<usize>> {
        // Cell key: (ix, iy, iz, class) where class separates boundary/feature/interior
        let mut cells: HashMap<(i64, i64, i64, u8), Vec<usize>> = HashMap::new();

        for (vi, v) in mesh.vertices.iter().enumerate() {
            let ix = ((v.x as f64 - bbox.min[0]) / cell_size).floor() as i64;
            let iy = ((v.y as f64 - bbox.min[1]) / cell_size).floor() as i64;
            let iz = ((v.z as f64 - bbox.min[2]) / cell_size).floor() as i64;

            let class = if self.preserve_boundary && boundary_verts.contains(&vi) {
                1u8
            } else if feature_verts.contains(&vi) {
                2u8
            } else {
                0u8
            };

            cells.entry((ix, iy, iz, class)).or_default().push(vi);
        }

        cells.into_values().collect()
    }

    /// Assign vertices using adaptive octree clustering.
    fn adaptive_clustering(
        &self,
        mesh: &TriangleMesh,
        bbox: &BBox,
        quadrics: &[Matrix4<f64>],
        max_depth: u32,
        error_threshold: f64,
        boundary_verts: &HashSet<usize>,
        feature_verts: &HashSet<usize>,
    ) -> Vec<Vec<usize>> {
        // Separate protected vertices (boundary/feature) from interior
        let mut protected: HashMap<(i64, i64, i64, u8), Vec<usize>> = HashMap::new();
        let mut interior_indices: Vec<usize> = Vec::new();

        // Use a coarse grid for protected vertex grouping
        let extent = bbox.max_extent().max(1e-6);
        let coarse_size = extent / (1 << max_depth.min(6)) as f64;

        for vi in 0..mesh.vertices.len() {
            let is_boundary = self.preserve_boundary && boundary_verts.contains(&vi);
            let is_feature = feature_verts.contains(&vi);

            if is_boundary || is_feature {
                let v = &mesh.vertices[vi];
                let ix = ((v.x as f64 - bbox.min[0]) / coarse_size).floor() as i64;
                let iy = ((v.y as f64 - bbox.min[1]) / coarse_size).floor() as i64;
                let iz = ((v.z as f64 - bbox.min[2]) / coarse_size).floor() as i64;
                let class = if is_boundary { 1u8 } else { 2u8 };
                protected.entry((ix, iy, iz, class)).or_default().push(vi);
            } else {
                interior_indices.push(vi);
            }
        }

        // Build octree for interior vertices
        // Make bbox slightly larger to ensure all points are inside
        let padded_bbox = BBox {
            min: [bbox.min[0] - 1e-4, bbox.min[1] - 1e-4, bbox.min[2] - 1e-4],
            max: [bbox.max[0] + 1e-4, bbox.max[1] + 1e-4, bbox.max[2] + 1e-4],
        };
        let mut root = OctreeNode::new(padded_bbox, 0);

        for &vi in &interior_indices {
            root.insert(vi, &mesh.vertices, quadrics, max_depth, error_threshold);
        }

        let mut clusters: Vec<Vec<usize>> = Vec::new();
        root.collect_clusters(&mut clusters);

        // Add protected clusters
        for (_, verts) in protected {
            if !verts.is_empty() {
                clusters.push(verts);
            }
        }

        clusters
    }

    /// Build the simplified mesh from clusters.
    fn build_simplified_mesh(
        &self,
        mesh: &TriangleMesh,
        clusters: &[Vec<usize>],
        quadrics: &[Matrix4<f64>],
        valence: &[usize],
    ) -> TriangleMesh {
        // Map each original vertex to its cluster index
        let mut vertex_to_cluster: Vec<usize> = vec![0; mesh.vertices.len()];
        for (ci, cluster) in clusters.iter().enumerate() {
            for &vi in cluster {
                vertex_to_cluster[vi] = ci;
            }
        }

        // Compute representative positions
        let representatives: Vec<Point3f> = clusters
            .iter()
            .map(|cluster| {
                if cluster.len() == 1 {
                    mesh.vertices[cluster[0]]
                } else {
                    select_representative(
                        cluster,
                        &mesh.vertices,
                        quadrics,
                        valence,
                        self.representative_strategy,
                    )
                }
            })
            .collect();

        // Remap faces, filtering degenerate triangles
        let mut new_faces: Vec<[usize; 3]> = Vec::new();
        let mut seen_faces: HashSet<[usize; 3]> = HashSet::new();

        for face in &mesh.faces {
            let nv0 = vertex_to_cluster[face[0]];
            let nv1 = vertex_to_cluster[face[1]];
            let nv2 = vertex_to_cluster[face[2]];

            // Skip degenerate triangles
            if nv0 == nv1 || nv1 == nv2 || nv2 == nv0 {
                continue;
            }

            // Canonical ordering to deduplicate
            let mut sorted = [nv0, nv1, nv2];
            sorted.sort();
            if seen_faces.insert(sorted) {
                new_faces.push([nv0, nv1, nv2]);
            }
        }

        // Compact: only include clusters that appear in at least one face
        let mut used_clusters: HashSet<usize> = HashSet::new();
        for face in &new_faces {
            for &vi in face {
                used_clusters.insert(vi);
            }
        }

        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_vertices: Vec<Point3f> = Vec::new();
        let mut new_normals: Option<Vec<Vector3f>> = mesh.normals.as_ref().map(|_| Vec::new());
        let mut new_colors: Option<Vec<[u8; 3]>> = mesh.colors.as_ref().map(|_| Vec::new());

        for (ci, cluster) in clusters.iter().enumerate() {
            if !used_clusters.contains(&ci) {
                continue;
            }
            let new_idx = new_vertices.len();
            old_to_new.insert(ci, new_idx);
            new_vertices.push(representatives[ci]);

            // Interpolate normals: average normals of cluster members
            if let Some(ref normals) = mesh.normals {
                let mut avg = Vector3f::new(0.0, 0.0, 0.0);
                for &vi in cluster {
                    avg += normals[vi];
                }
                let len = avg.magnitude();
                if len > 1e-12 {
                    avg /= len;
                }
                new_normals.as_mut().unwrap().push(avg);
            }

            // Interpolate colors: average colors of cluster members
            if let Some(ref colors) = mesh.colors {
                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;
                for &vi in cluster {
                    r += colors[vi][0] as u32;
                    g += colors[vi][1] as u32;
                    b += colors[vi][2] as u32;
                }
                let n = cluster.len() as u32;
                new_colors
                    .as_mut()
                    .unwrap()
                    .push([(r / n) as u8, (g / n) as u8, (b / n) as u8]);
            }
        }

        // Remap face indices
        let remapped_faces: Vec<[usize; 3]> = new_faces
            .iter()
            .filter_map(|f| {
                match (old_to_new.get(&f[0]), old_to_new.get(&f[1]), old_to_new.get(&f[2])) {
                    (Some(&a), Some(&b), Some(&c)) if a != b && b != c && c != a => {
                        Some([a, b, c])
                    }
                    _ => None,
                }
            })
            .collect();

        let mut result = TriangleMesh::from_vertices_and_faces(new_vertices, remapped_faces);
        if let Some(normals) = new_normals {
            result.set_normals(normals);
        }
        if let Some(colors) = new_colors {
            result.set_colors(colors);
        }
        result
    }
}

impl MeshSimplifier for ClusteringSimplifier {
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

        let bbox = BBox::from_vertices(&mesh.vertices);
        let quadrics = compute_quadrics(mesh);
        let valence = compute_vertex_valence(mesh);
        let boundary_verts = if self.preserve_boundary {
            find_boundary_vertices(mesh)
        } else {
            HashSet::new()
        };
        let feature_verts = find_feature_vertices(mesh, self.feature_angle_threshold);

        let clusters = match self.mode {
            ClusteringMode::Uniform => {
                let cell_size =
                    Self::compute_cell_size(&bbox, mesh.vertices.len(), reduction_ratio);
                self.uniform_clustering(mesh, cell_size, &bbox, &boundary_verts, &feature_verts)
            }
            ClusteringMode::Adaptive {
                max_depth,
                error_threshold,
            } => self.adaptive_clustering(
                mesh,
                &bbox,
                &quadrics,
                max_depth,
                error_threshold,
                &boundary_verts,
                &feature_verts,
            ),
        };

        Ok(self.build_simplified_mesh(mesh, &clusters, &quadrics, &valence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    fn make_single_triangle() -> TriangleMesh {
        TriangleMesh::from_vertices_and_faces(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    fn make_tetrahedron() -> TriangleMesh {
        TriangleMesh::from_vertices_and_faces(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(0.5, 0.5, 1.0),
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
    }

    fn make_plane_grid(size: usize) -> TriangleMesh {
        let mut vertices = Vec::new();
        for y in 0..size {
            for x in 0..size {
                vertices.push(Point3::new(x as f32, y as f32, 0.0));
            }
        }
        let mut faces = Vec::new();
        for y in 0..(size - 1) {
            for x in 0..(size - 1) {
                let tl = y * size + x;
                let tr = tl + 1;
                let bl = (y + 1) * size + x;
                let br = bl + 1;
                faces.push([tl, bl, tr]);
                faces.push([tr, bl, br]);
            }
        }
        TriangleMesh::from_vertices_and_faces(vertices, faces)
    }

    fn make_curved_surface(size: usize) -> TriangleMesh {
        let mut vertices = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let fx = x as f32 / (size - 1) as f32 * std::f32::consts::PI;
                let fy = y as f32 / (size - 1) as f32 * std::f32::consts::PI;
                vertices.push(Point3::new(
                    x as f32,
                    y as f32,
                    (fx.sin() * fy.sin()) * 2.0,
                ));
            }
        }
        let mut faces = Vec::new();
        for y in 0..(size - 1) {
            for x in 0..(size - 1) {
                let tl = y * size + x;
                let tr = tl + 1;
                let bl = (y + 1) * size + x;
                let br = bl + 1;
                faces.push([tl, bl, tr]);
                faces.push([tr, bl, br]);
            }
        }
        TriangleMesh::from_vertices_and_faces(vertices, faces)
    }

    fn make_sharp_edge_mesh() -> TriangleMesh {
        // Two planes meeting at a 90-degree angle along the x-axis
        TriangleMesh::from_vertices_and_faces(
            vec![
                // Bottom plane (z=0)
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(2.0, 1.0, 0.0),
                // Top plane (going up at 90 degrees from the y=1 edge)
                Point3::new(0.0, 1.0, 1.0),
                Point3::new(1.0, 1.0, 1.0),
                Point3::new(2.0, 1.0, 1.0),
            ],
            vec![
                // Bottom plane faces
                [0, 1, 3],
                [1, 4, 3],
                [1, 2, 4],
                [2, 5, 4],
                // Top plane faces
                [3, 4, 6],
                [4, 7, 6],
                [4, 5, 7],
                [5, 8, 7],
            ],
        )
    }

    // ---- Construction tests ----

    #[test]
    fn test_creation() {
        let s = ClusteringSimplifier::new();
        assert!(s.preserve_boundary);
        assert_eq!(s.mode, ClusteringMode::Uniform);
        assert_eq!(s.representative_strategy, RepresentativeStrategy::Centroid);
    }

    #[test]
    fn test_with_params() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Adaptive {
                max_depth: 6,
                error_threshold: 0.01,
            },
            RepresentativeStrategy::MinimumError,
            false,
            30.0_f32.to_radians(),
        );
        assert!(!s.preserve_boundary);
        assert_eq!(
            s.mode,
            ClusteringMode::Adaptive {
                max_depth: 6,
                error_threshold: 0.01,
            }
        );
        assert_eq!(
            s.representative_strategy,
            RepresentativeStrategy::MinimumError
        );
    }

    // ---- Validation tests ----

    #[test]
    fn test_empty_mesh() {
        let s = ClusteringSimplifier::new();
        let mesh = TriangleMesh::new();
        assert!(s.simplify(&mesh, 0.5).is_err());
    }

    #[test]
    fn test_invalid_reduction_ratio() {
        let s = ClusteringSimplifier::new();
        let mesh = make_single_triangle();
        assert!(s.simplify(&mesh, -0.1).is_err());
        assert!(s.simplify(&mesh, 1.1).is_err());
    }

    #[test]
    fn test_zero_reduction() {
        let s = ClusteringSimplifier::new();
        let mesh = make_single_triangle();
        let result = s.simplify(&mesh, 0.0).unwrap();
        assert_eq!(result.vertex_count(), 3);
        assert_eq!(result.face_count(), 1);
    }

    // ---- Uniform clustering tests ----

    #[test]
    fn test_single_triangle_uniform() {
        let s = ClusteringSimplifier::new();
        let mesh = make_single_triangle();
        let result = s.simplify(&mesh, 0.5).unwrap();
        // Should still produce valid output
        assert!(result.vertex_count() > 0);
    }

    #[test]
    fn test_tetrahedron_uniform() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Uniform,
            RepresentativeStrategy::Centroid,
            false,
            std::f32::consts::PI,
        );
        let mesh = make_tetrahedron();
        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.vertex_count() <= mesh.vertex_count());
    }

    #[test]
    fn test_planar_grid_uniform() {
        let s = ClusteringSimplifier::new();
        let mesh = make_plane_grid(6);
        let original_faces = mesh.face_count();
        assert_eq!(original_faces, 50);

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() < original_faces);
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_curved_surface_uniform() {
        let s = ClusteringSimplifier::new();
        let mesh = make_curved_surface(8);
        let original_faces = mesh.face_count();

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() < original_faces);
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_large_grid_uniform() {
        let s = ClusteringSimplifier::new();
        let mesh = make_plane_grid(11);
        let original = mesh.face_count(); // 200 faces

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() < original);
        assert!(result.face_count() > 0);
        assert!(result.vertex_count() > 0);
    }

    // ---- Adaptive clustering tests ----

    #[test]
    fn test_planar_grid_adaptive() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Adaptive {
                max_depth: 4,
                error_threshold: 0.01,
            },
            RepresentativeStrategy::Centroid,
            true,
            45.0_f32.to_radians(),
        );
        let mesh = make_plane_grid(6);
        let original_faces = mesh.face_count();

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() <= original_faces);
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_curved_surface_adaptive() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Adaptive {
                max_depth: 5,
                error_threshold: 0.1,
            },
            RepresentativeStrategy::MinimumError,
            true,
            45.0_f32.to_radians(),
        );
        let mesh = make_curved_surface(8);
        let original_faces = mesh.face_count();

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() <= original_faces);
        assert!(result.face_count() > 0);
    }

    // ---- Representative strategy tests ----

    #[test]
    fn test_centroid_strategy() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Uniform,
            RepresentativeStrategy::Centroid,
            false,
            std::f32::consts::PI,
        );
        let mesh = make_plane_grid(6);
        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_weighted_average_strategy() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Uniform,
            RepresentativeStrategy::WeightedAverage,
            false,
            std::f32::consts::PI,
        );
        let mesh = make_plane_grid(6);
        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_minimum_error_strategy() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Uniform,
            RepresentativeStrategy::MinimumError,
            false,
            std::f32::consts::PI,
        );
        let mesh = make_plane_grid(6);
        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() > 0);
    }

    // ---- Boundary preservation tests ----

    #[test]
    fn test_boundary_preservation_uniform() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Uniform,
            RepresentativeStrategy::Centroid,
            true,
            45.0_f32.to_radians(),
        );
        let mesh = make_plane_grid(6);

        let original_boundary: HashSet<(i32, i32, i32)> = {
            let size = 6;
            let mut set = HashSet::new();
            for i in 0..size {
                for j in 0..size {
                    if i == 0 || i == size - 1 || j == 0 || j == size - 1 {
                        let idx = i * size + j;
                        let p = mesh.vertices[idx];
                        set.insert((
                            (p.x * 100.0) as i32,
                            (p.y * 100.0) as i32,
                            (p.z * 100.0) as i32,
                        ));
                    }
                }
            }
            set
        };

        let result = s.simplify(&mesh, 0.5).unwrap();
        let result_positions: HashSet<(i32, i32, i32)> = result
            .vertices
            .iter()
            .map(|p| {
                (
                    (p.x * 100.0) as i32,
                    (p.y * 100.0) as i32,
                    (p.z * 100.0) as i32,
                )
            })
            .collect();

        // Boundary vertices should be mostly preserved (clustered separately)
        let preserved = original_boundary.intersection(&result_positions).count();
        let ratio = preserved as f32 / original_boundary.len() as f32;
        assert!(
            ratio > 0.5,
            "Expected >50% boundary preservation, got {:.1}%",
            ratio * 100.0
        );
    }

    // ---- Sharp feature tests ----

    #[test]
    fn test_sharp_feature_detection() {
        let mesh = make_sharp_edge_mesh();
        let feature_verts = find_feature_vertices(&mesh, 45.0_f32.to_radians());
        // Vertices along the sharp edge (indices 3, 4, 5) should be detected
        assert!(
            !feature_verts.is_empty(),
            "Should detect feature vertices at the 90-degree edge"
        );
    }

    #[test]
    fn test_sharp_feature_preservation() {
        let s = ClusteringSimplifier::with_params(
            ClusteringMode::Uniform,
            RepresentativeStrategy::Centroid,
            true,
            45.0_f32.to_radians(),
        );
        let mesh = make_sharp_edge_mesh();
        let result = s.simplify(&mesh, 0.3).unwrap();
        assert!(result.face_count() > 0);
        assert!(result.vertex_count() > 0);
    }

    // ---- Attribute preservation tests ----

    #[test]
    fn test_attribute_preservation_normals() {
        let mut mesh = make_plane_grid(5);
        let normals: Vec<Vector3f> = (0..mesh.vertex_count())
            .map(|_| Vector3f::new(0.0, 0.0, 1.0))
            .collect();
        mesh.set_normals(normals);

        let s = ClusteringSimplifier::new();
        let result = s.simplify(&mesh, 0.3).unwrap();
        assert!(result.normals.is_some(), "normals should be preserved");
        let result_normals = result.normals.as_ref().unwrap();
        assert_eq!(result_normals.len(), result.vertex_count());
        for n in result_normals {
            assert!(n.z > 0.9, "normal z should be close to 1.0, got {}", n.z);
        }
    }

    #[test]
    fn test_attribute_preservation_colors() {
        let mut mesh = make_plane_grid(5);
        let colors: Vec<[u8; 3]> = (0..mesh.vertex_count()).map(|_| [128, 64, 200]).collect();
        mesh.set_colors(colors);

        let s = ClusteringSimplifier::new();
        let result = s.simplify(&mesh, 0.3).unwrap();
        assert!(result.colors.is_some(), "colors should be preserved");
        assert_eq!(result.colors.as_ref().unwrap().len(), result.vertex_count());
    }

    // ---- Comparison tests (clustering vs other methods produce valid output) ----

    #[test]
    fn test_all_strategies_produce_valid_output() {
        let mesh = make_curved_surface(8);
        let strategies = [
            RepresentativeStrategy::Centroid,
            RepresentativeStrategy::WeightedAverage,
            RepresentativeStrategy::MinimumError,
        ];

        for strategy in &strategies {
            let s = ClusteringSimplifier::with_params(
                ClusteringMode::Uniform,
                *strategy,
                true,
                45.0_f32.to_radians(),
            );
            let result = s.simplify(&mesh, 0.5).unwrap();
            assert!(
                result.face_count() > 0,
                "Strategy {:?} produced empty mesh",
                strategy
            );
            assert!(
                result.vertex_count() > 0,
                "Strategy {:?} produced no vertices",
                strategy
            );
            assert!(
                result.face_count() < mesh.face_count(),
                "Strategy {:?} did not reduce faces",
                strategy
            );
        }
    }

    #[test]
    fn test_both_modes_produce_valid_output() {
        let mesh = make_curved_surface(8);

        let uniform = ClusteringSimplifier::new();
        let adaptive = ClusteringSimplifier::with_params(
            ClusteringMode::Adaptive {
                max_depth: 4,
                error_threshold: 0.1,
            },
            RepresentativeStrategy::Centroid,
            true,
            45.0_f32.to_radians(),
        );

        let r1 = uniform.simplify(&mesh, 0.5).unwrap();
        let r2 = adaptive.simplify(&mesh, 0.5).unwrap();

        assert!(r1.face_count() > 0);
        assert!(r2.face_count() > 0);
        assert!(r1.face_count() < mesh.face_count());
        assert!(r2.face_count() < mesh.face_count());
    }
}
