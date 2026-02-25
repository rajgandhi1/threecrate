//! Progressive mesh implementation based on Hoppe (1996)
//!
//! Provides a progressive mesh representation that encodes a mesh as a coarse
//! base mesh plus a sequence of vertex split operations. This allows
//! reconstructing the mesh at any level of detail between the base and the
//! original, enabling LOD rendering and streaming.

use crate::edge_collapse::{EdgeCost, HalfEdgeMesh, INVALID};
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use threecrate_core::{Error, Point3f, Result, TriangleMesh, Vector3f};

/// A single vertex split operation (inverse of an edge collapse).
///
/// Records how to split vertex `vertex_s` into `vertex_s` (at updated position)
/// and `vertex_t` (a restored vertex), along with the face connectivity changes
/// needed to reverse the collapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexSplit {
    /// Vertex being split (the survivor of the original collapse)
    pub vertex_s: usize,
    /// Restored vertex index (the vertex that was removed in the collapse)
    pub vertex_t: usize,
    /// Position of vertex_s after the split
    pub position_s: Point3f,
    /// Position of vertex_t (restored)
    pub position_t: Point3f,
    /// Normal of vertex_s after split
    pub normal_s: Option<Vector3f>,
    /// Normal of vertex_t (restored)
    pub normal_t: Option<Vector3f>,
    /// Color of vertex_s after split
    pub color_s: Option<[u8; 3]>,
    /// Color of vertex_t (restored)
    pub color_t: Option<[u8; 3]>,
    /// Restored face on left side of the edge (if any)
    pub face_left: Option<[usize; 3]>,
    /// Restored face on right side of the edge (if any)
    pub face_right: Option<[usize; 3]>,
    /// Faces whose connectivity changes: (face_index, new vertex indices)
    pub modified_faces: Vec<(usize, [usize; 3])>,
}

/// Progressive mesh: a base mesh plus a sequence of vertex splits.
///
/// The base mesh is the coarsest representation. Applying vertex splits
/// in order progressively refines the mesh back toward the original.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveMesh {
    /// The coarsest level of detail
    pub base_mesh: TriangleMesh,
    /// Ordered sequence of vertex split operations (apply in order to refine)
    pub vertex_splits: Vec<VertexSplit>,
    /// Total vertex count of the original (fully refined) mesh
    pub full_vertex_count: usize,
    /// Total face count of the original (fully refined) mesh
    pub full_face_count: usize,
}

/// Record of a single edge collapse for later inversion into a vertex split.
struct CollapseRecord {
    /// The vertex that survived (v1)
    vertex_s: usize,
    /// The vertex that was removed (v2)
    vertex_t: usize,
    /// Position of v1 before collapse (so we can restore it on split)
    position_s_before: Point3f,
    /// Position of v2 before collapse
    position_t_before: Point3f,
    /// Normal of v1 before collapse
    normal_s_before: Option<Vector3f>,
    /// Normal of v2 before collapse
    normal_t_before: Option<Vector3f>,
    /// Color of v1 before collapse
    color_s_before: Option<[u8; 3]>,
    /// Color of v2 before collapse
    color_t_before: Option<[u8; 3]>,
    /// Faces that were removed (degenerate after collapse)
    removed_faces: Vec<(usize, [usize; 3])>,
    /// Faces that had v2 replaced with v1 (face_idx, old verts before rewrite)
    rewritten_faces: Vec<(usize, [usize; 3])>,
}

impl ProgressiveMesh {
    /// Generate a progressive mesh by simplifying the input mesh down to a
    /// base level, recording each edge collapse as a reversible vertex split.
    ///
    /// `base_face_ratio` controls how much to simplify: 0.1 means the base
    /// mesh will have ~10% of the original faces.
    pub fn from_mesh(mesh: &TriangleMesh, base_face_ratio: f32) -> Result<Self> {
        if mesh.is_empty() {
            return Err(Error::InvalidData("Mesh is empty".to_string()));
        }
        let base_face_ratio = base_face_ratio.clamp(0.01, 1.0);
        let target_faces = (base_face_ratio * mesh.faces.len() as f32).max(1.0) as usize;

        let full_vertex_count = mesh.vertex_count();
        let full_face_count = mesh.face_count();

        let mut hem = HalfEdgeMesh::from_triangle_mesh(mesh);
        let mut collapse_records: Vec<CollapseRecord> = Vec::new();

        let mut queue = build_queue(&hem);

        while hem.active_face_count > target_faces && !queue.is_empty() {
            let (_, edge_cost) = match queue.pop() {
                Some(item) => item,
                None => break,
            };

            let v1 = edge_cost.v1;
            let v2 = edge_cost.v2;

            if hem.vertex_removed[v1]
                || hem.vertex_removed[v2]
                || hem.vertex_edge[v1] == INVALID
                || hem.vertex_edge[v2] == INVALID
            {
                continue;
            }

            if hem.find_half_edge(v1, v2).is_none() {
                continue;
            }

            if !hem.check_link_condition(v1, v2) {
                continue;
            }

            // Snapshot state before collapse
            let position_s_before = hem.positions[v1];
            let position_t_before = hem.positions[v2];
            let normal_s_before = hem.normals.as_ref().map(|n| n[v1]);
            let normal_t_before = hem.normals.as_ref().map(|n| n[v2]);
            let color_s_before = hem.colors.as_ref().map(|c| c[v1]);
            let color_t_before = hem.colors.as_ref().map(|c| c[v2]);

            // Snapshot face state: find faces adjacent to edge (v1,v2) that
            // will be removed, and faces incident on v2 that will be rewritten.
            let faces_before = snapshot_faces_around_edge(&hem, v1, v2);

            let (pos, _cost) = hem.compute_collapse_cost(v1, v2);

            if !hem.collapse_edge(v1, v2, pos) {
                continue;
            }

            // Determine which faces were removed vs rewritten
            let (removed_faces, rewritten_faces) =
                classify_face_changes(&hem, &faces_before, v1, v2);

            collapse_records.push(CollapseRecord {
                vertex_s: v1,
                vertex_t: v2,
                position_s_before,
                position_t_before,
                normal_s_before,
                normal_t_before,
                color_s_before,
                color_t_before,
                removed_faces,
                rewritten_faces,
            });

            // Periodically rebuild queue
            if collapse_records.len() % 100 == 0 {
                queue = rebuild_queue(&hem, collapse_records.len() * 1000);
            }
        }

        let base_mesh = hem.to_triangle_mesh();

        // Convert collapse records to vertex splits (reverse order)
        let vertex_splits: Vec<VertexSplit> = collapse_records
            .into_iter()
            .rev()
            .map(|rec| {
                let face_left = rec.removed_faces.first().map(|(_, f)| *f);
                let face_right = rec.removed_faces.get(1).map(|(_, f)| *f);

                let modified_faces: Vec<(usize, [usize; 3])> = rec
                    .rewritten_faces
                    .iter()
                    .map(|(fi, old_verts)| (*fi, *old_verts))
                    .collect();

                VertexSplit {
                    vertex_s: rec.vertex_s,
                    vertex_t: rec.vertex_t,
                    position_s: rec.position_s_before,
                    position_t: rec.position_t_before,
                    normal_s: rec.normal_s_before,
                    normal_t: rec.normal_t_before,
                    color_s: rec.color_s_before,
                    color_t: rec.color_t_before,
                    face_left,
                    face_right,
                    modified_faces,
                }
            })
            .collect();

        Ok(ProgressiveMesh {
            base_mesh,
            vertex_splits,
            full_vertex_count,
            full_face_count,
        })
    }

    /// Reconstruct the mesh at a specific refinement level.
    ///
    /// `level` is clamped to `[0, num_levels()]`. Level 0 is the base mesh,
    /// `num_levels()` is full detail.
    pub fn reconstruct_at_level(&self, level: usize) -> TriangleMesh {
        let level = level.min(self.vertex_splits.len());
        if level == 0 {
            return self.base_mesh.clone();
        }

        // Start from original mesh data and replay only the needed splits.
        // We rebuild by applying vertex splits to the base mesh.
        let mut vertices = self.base_mesh.vertices.clone();
        let mut faces = self.base_mesh.faces.clone();
        let mut normals = self.base_mesh.normals.clone();
        let mut colors = self.base_mesh.colors.clone();

        // Build index mapping: base mesh uses compacted indices, but vertex
        // splits reference original (pre-compaction) indices. We need to map.
        // Strategy: grow vertex/face arrays as splits are applied.

        for split in self.vertex_splits.iter().take(level) {
            // Ensure vertex arrays are large enough
            while vertices.len() <= split.vertex_t.max(split.vertex_s) {
                vertices.push(Point3f::origin());
                if let Some(ref mut n) = normals {
                    n.push(Vector3f::zeros());
                }
                if let Some(ref mut c) = colors {
                    c.push([0, 0, 0]);
                }
            }

            // Restore positions
            vertices[split.vertex_s] = split.position_s;
            vertices[split.vertex_t] = split.position_t;

            // Restore normals
            if let Some(ref mut n) = normals {
                if let Some(ns) = split.normal_s {
                    n[split.vertex_s] = ns;
                }
                if let Some(nt) = split.normal_t {
                    n[split.vertex_t] = nt;
                }
            }

            // Restore colors
            if let Some(ref mut c) = colors {
                if let Some(cs) = split.color_s {
                    c[split.vertex_s] = cs;
                }
                if let Some(ct) = split.color_t {
                    c[split.vertex_t] = ct;
                }
            }

            // Restore modified faces (change v_s back to v_t where needed)
            for &(fi, old_verts) in &split.modified_faces {
                while faces.len() <= fi {
                    faces.push([0, 0, 0]);
                }
                faces[fi] = old_verts;
            }

            // Restore removed faces
            if let Some(face) = split.face_left {
                faces.push(face);
            }
            if let Some(face) = split.face_right {
                faces.push(face);
            }
        }

        // Clean up: remove any degenerate faces and compact
        let faces: Vec<[usize; 3]> = faces
            .into_iter()
            .filter(|f| {
                f[0] != f[1]
                    && f[1] != f[2]
                    && f[2] != f[0]
                    && f[0] < vertices.len()
                    && f[1] < vertices.len()
                    && f[2] < vertices.len()
            })
            .collect();

        let mut mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        if let Some(n) = normals {
            mesh.set_normals(n);
        }
        if let Some(c) = colors {
            mesh.set_colors(c);
        }
        mesh
    }

    /// Reconstruct the mesh at a given detail ratio.
    ///
    /// `detail_ratio` ranges from 0.0 (base/coarsest) to 1.0 (full detail).
    pub fn reconstruct_at_ratio(&self, detail_ratio: f32) -> TriangleMesh {
        let detail_ratio = detail_ratio.clamp(0.0, 1.0);
        let level = (detail_ratio * self.vertex_splits.len() as f32).round() as usize;
        self.reconstruct_at_level(level)
    }

    /// Get a reference to the base (coarsest) mesh.
    pub fn base(&self) -> &TriangleMesh {
        &self.base_mesh
    }

    /// Number of refinement levels (vertex splits) available.
    pub fn num_levels(&self) -> usize {
        self.vertex_splits.len()
    }

    /// Serialize the progressive mesh to bytes using bincode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| Error::InvalidData(format!("Serialization failed: {}", e)))
    }

    /// Deserialize a progressive mesh from bytes.
    pub fn deserialize_from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| Error::InvalidData(format!("Deserialization failed: {}", e)))
    }
}

// ============================================================
// Helper functions
// ============================================================

/// Snapshot face data around edge (v1, v2) before collapse.
/// Returns Vec of (face_index, [v0, v1, v2]) for all faces incident on v1 or v2.
fn snapshot_faces_around_edge(
    hem: &HalfEdgeMesh,
    v1: usize,
    v2: usize,
) -> Vec<(usize, [usize; 3])> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();

    for &v in &[v1, v2] {
        for &he in &hem.outgoing_half_edges(v) {
            let face = hem.half_edges[he].face;
            if face == INVALID || !seen.insert(face) {
                continue;
            }
            let he0 = hem.face_edge[face];
            if he0 == INVALID {
                continue;
            }
            let he1 = hem.half_edges[he0].next;
            let fv0 = hem.source(he0);
            let fv1 = hem.half_edges[he0].target;
            let fv2 = hem.half_edges[he1].target;
            result.push((face, [fv0, fv1, fv2]));
        }
    }

    result
}

/// After collapse of v2 into v1, classify which faces were removed vs rewritten.
fn classify_face_changes(
    hem: &HalfEdgeMesh,
    faces_before: &[(usize, [usize; 3])],
    _v1: usize,
    v2: usize,
) -> (Vec<(usize, [usize; 3])>, Vec<(usize, [usize; 3])>) {
    let mut removed = Vec::new();
    let mut rewritten = Vec::new();

    for &(fi, verts) in faces_before {
        if hem.face_edge[fi] == INVALID {
            // Face was removed during collapse
            removed.push((fi, verts));
        } else {
            // Check if face was rewritten (had v2 replaced with v1)
            let has_v2 = verts[0] == v2 || verts[1] == v2 || verts[2] == v2;
            if has_v2 {
                rewritten.push((fi, verts));
            }
        }
    }

    (removed, rewritten)
}

/// Build priority queue of edge collapse candidates (no boundary preservation).
fn build_queue(hem: &HalfEdgeMesh) -> PriorityQueue<usize, EdgeCost> {
    let mut queue = PriorityQueue::new();
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    let mut edge_id = 0usize;

    for vi in 0..hem.positions.len() {
        if hem.vertex_removed[vi] {
            continue;
        }
        for &he in &hem.outgoing_half_edges(vi) {
            let target = hem.half_edges[he].target;
            let key = (vi.min(target), vi.max(target));
            if !seen_edges.insert(key) {
                continue;
            }

            let (pos, cost) = hem.compute_collapse_cost(vi, target);

            queue.push(
                edge_id,
                EdgeCost {
                    v1: vi,
                    v2: target,
                    position: pos,
                    cost,
                },
            );
            edge_id += 1;
        }
    }

    queue
}

/// Rebuild priority queue after many collapses.
fn rebuild_queue(hem: &HalfEdgeMesh, id_offset: usize) -> PriorityQueue<usize, EdgeCost> {
    let mut queue = PriorityQueue::new();
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    let mut edge_id = id_offset;

    for vi in 0..hem.positions.len() {
        if hem.vertex_removed[vi] || hem.vertex_edge[vi] == INVALID {
            continue;
        }
        for &he in &hem.outgoing_half_edges(vi) {
            if hem.half_edges[he].face == INVALID {
                continue;
            }
            let target = hem.half_edges[he].target;
            let key = (vi.min(target), vi.max(target));
            if !seen_edges.insert(key) {
                continue;
            }

            let (pos, cost) = hem.compute_collapse_cost(vi, target);

            queue.push(
                edge_id,
                EdgeCost {
                    v1: vi,
                    v2: target,
                    position: pos,
                    cost,
                },
            );
            edge_id += 1;
        }
    }

    queue
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

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

    #[test]
    fn test_progressive_from_empty_mesh() {
        let mesh = TriangleMesh::new();
        assert!(ProgressiveMesh::from_mesh(&mesh, 0.5).is_err());
    }

    #[test]
    fn test_progressive_from_tetrahedron() {
        let mesh = make_tetrahedron();
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.5).unwrap();
        assert_eq!(pm.full_vertex_count, 4);
        assert_eq!(pm.full_face_count, 4);
        assert!(pm.base_mesh.face_count() <= mesh.face_count());
    }

    #[test]
    fn test_progressive_from_grid() {
        let mesh = make_plane_grid(6);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();

        // Base mesh should have fewer faces than original
        assert!(
            pm.base_mesh.face_count() < mesh.face_count(),
            "base should have fewer faces: {} vs {}",
            pm.base_mesh.face_count(),
            mesh.face_count()
        );

        // Should have at least one vertex split
        assert!(
            !pm.vertex_splits.is_empty(),
            "should have vertex splits recorded"
        );
    }

    #[test]
    fn test_progressive_base_access() {
        let mesh = make_plane_grid(6);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();

        let base = pm.base();
        assert_eq!(base.face_count(), pm.base_mesh.face_count());
    }

    #[test]
    fn test_progressive_num_levels() {
        let mesh = make_plane_grid(6);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();

        assert!(pm.num_levels() > 0);
    }

    #[test]
    fn test_progressive_reconstruct_level_zero_is_base() {
        let mesh = make_plane_grid(6);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();

        let level0 = pm.reconstruct_at_level(0);
        assert_eq!(level0.vertex_count(), pm.base_mesh.vertex_count());
        assert_eq!(level0.face_count(), pm.base_mesh.face_count());
    }

    #[test]
    fn test_progressive_reconstruct_ratio_zero_is_base() {
        let mesh = make_plane_grid(6);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();

        let r0 = pm.reconstruct_at_ratio(0.0);
        assert_eq!(r0.vertex_count(), pm.base_mesh.vertex_count());
        assert_eq!(r0.face_count(), pm.base_mesh.face_count());
    }

    #[test]
    fn test_progressive_monotonic_detail() {
        let mesh = make_plane_grid(8);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.2).unwrap();

        if pm.num_levels() < 2 {
            return; // Not enough levels to test monotonicity
        }

        let mut prev_faces = pm.base_mesh.face_count();
        let step = (pm.num_levels() / 4).max(1);

        for level in (step..=pm.num_levels()).step_by(step) {
            let reconstructed = pm.reconstruct_at_level(level);
            assert!(
                reconstructed.face_count() >= prev_faces,
                "face count should monotonically increase: level {} has {} faces, prev had {}",
                level,
                reconstructed.face_count(),
                prev_faces
            );
            prev_faces = reconstructed.face_count();
        }
    }

    #[test]
    fn test_progressive_with_normals() {
        let mut mesh = make_plane_grid(5);
        let normals: Vec<Vector3f> = (0..mesh.vertex_count())
            .map(|_| Vector3f::new(0.0, 0.0, 1.0))
            .collect();
        mesh.set_normals(normals);

        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();
        assert!(pm.base_mesh.normals.is_some());
    }

    #[test]
    fn test_progressive_with_colors() {
        let mut mesh = make_plane_grid(5);
        let colors: Vec<[u8; 3]> = (0..mesh.vertex_count()).map(|_| [128, 64, 200]).collect();
        mesh.set_colors(colors);

        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();
        assert!(pm.base_mesh.colors.is_some());
    }

    #[test]
    fn test_progressive_serialization_roundtrip() {
        let mesh = make_tetrahedron();
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.5).unwrap();

        let bytes = pm.serialize_to_bytes().unwrap();
        assert!(!bytes.is_empty());

        let pm2 = ProgressiveMesh::deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(pm2.full_vertex_count, pm.full_vertex_count);
        assert_eq!(pm2.full_face_count, pm.full_face_count);
        assert_eq!(pm2.vertex_splits.len(), pm.vertex_splits.len());
        assert_eq!(pm2.base_mesh.face_count(), pm.base_mesh.face_count());
    }

    #[test]
    fn test_progressive_clamp_ratio() {
        let mesh = make_plane_grid(6);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();

        // Should not panic for out-of-range ratios
        let _ = pm.reconstruct_at_ratio(-1.0);
        let _ = pm.reconstruct_at_ratio(2.0);
    }

    #[test]
    fn test_progressive_clamp_level() {
        let mesh = make_plane_grid(6);
        let pm = ProgressiveMesh::from_mesh(&mesh, 0.3).unwrap();

        // Should not panic for out-of-range levels
        let _ = pm.reconstruct_at_level(999999);
    }
}
