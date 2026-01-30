//! Edge collapse simplification
//!
//! Implements iterative edge collapse mesh simplification using a half-edge
//! data structure for efficient topology operations and quadric error metrics
//! (QEM) for error-driven edge prioritization.

use crate::MeshSimplifier;
use nalgebra::{Matrix4, Vector4};
use priority_queue::PriorityQueue;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use threecrate_core::{Error, Point3f, Result, TriangleMesh, Vector3f};

const INVALID: usize = usize::MAX;

// ============================================================
// Half-Edge Data Structure
// ============================================================

#[derive(Debug, Clone)]
struct HalfEdge {
    target: usize,
    twin: usize,
    next: usize,
    prev: usize,
    face: usize,
}

/// Half-edge mesh for topology-aware edge collapse operations.
struct HalfEdgeMesh {
    half_edges: Vec<HalfEdge>,
    /// One outgoing half-edge per vertex (INVALID if removed)
    vertex_edge: Vec<usize>,
    /// One half-edge per face (INVALID if removed)
    face_edge: Vec<usize>,
    active_face_count: usize,
    positions: Vec<Point3f>,
    normals: Option<Vec<Vector3f>>,
    colors: Option<Vec<[u8; 3]>>,
    quadrics: Vec<Matrix4<f64>>,
    vertex_removed: Vec<bool>,
}

impl HalfEdgeMesh {
    fn from_triangle_mesh(mesh: &TriangleMesh) -> Self {
        let nv = mesh.vertices.len();
        let nf = mesh.faces.len();

        let mut half_edges = Vec::with_capacity(nf * 3);
        let mut vertex_edge = vec![INVALID; nv];
        let mut face_edge = Vec::with_capacity(nf);

        for (fi, face) in mesh.faces.iter().enumerate() {
            let base = fi * 3;
            for j in 0..3usize {
                half_edges.push(HalfEdge {
                    target: face[(j + 1) % 3],
                    twin: INVALID,
                    next: base + (j + 1) % 3,
                    prev: base + (j + 2) % 3,
                    face: fi,
                });
                if vertex_edge[face[j]] == INVALID {
                    vertex_edge[face[j]] = base + j;
                }
            }
            face_edge.push(base);
        }

        // Build twin pointers
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::with_capacity(nf * 3);
        for (he_idx, he) in half_edges.iter().enumerate() {
            let src = half_edges[he.prev].target;
            edge_map.insert((src, he.target), he_idx);
        }
        for he_idx in 0..half_edges.len() {
            if half_edges[he_idx].twin != INVALID {
                continue;
            }
            let src = half_edges[half_edges[he_idx].prev].target;
            let tgt = half_edges[he_idx].target;
            if let Some(&twin_idx) = edge_map.get(&(tgt, src)) {
                half_edges[he_idx].twin = twin_idx;
                half_edges[twin_idx].twin = he_idx;
            }
        }

        let mut hem = HalfEdgeMesh {
            half_edges,
            vertex_edge,
            face_edge,
            active_face_count: nf,
            positions: mesh.vertices.clone(),
            normals: mesh.normals.clone(),
            colors: mesh.colors.clone(),
            quadrics: vec![Matrix4::zeros(); nv],
            vertex_removed: vec![false; nv],
        };
        hem.initialize_quadrics();
        hem
    }

    #[inline]
    fn source(&self, he: usize) -> usize {
        self.half_edges[self.half_edges[he].prev].target
    }

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

    fn initialize_quadrics(&mut self) {
        for fi in 0..self.face_edge.len() {
            let he0 = self.face_edge[fi];
            if he0 == INVALID {
                continue;
            }
            let he1 = self.half_edges[he0].next;
            let v0 = self.source(he0);
            let v1 = self.half_edges[he0].target;
            let v2 = self.half_edges[he1].target;
            let plane =
                Self::compute_plane(&self.positions[v0], &self.positions[v1], &self.positions[v2]);
            let q = Self::plane_to_quadric(&plane);
            self.quadrics[v0] += q;
            self.quadrics[v1] += q;
            self.quadrics[v2] += q;
        }
    }

    /// Get all outgoing half-edges from a vertex (handles boundary vertices).
    fn outgoing_half_edges(&self, v: usize) -> Vec<usize> {
        let start = self.vertex_edge[v];
        if start == INVALID {
            return vec![];
        }

        let mut result = Vec::new();
        let mut current = start;

        // Rotate counterclockwise: current.prev.twin
        loop {
            result.push(current);
            let prev = self.half_edges[current].prev;
            let twin = self.half_edges[prev].twin;
            if twin == INVALID {
                break;
            }
            current = twin;
            if current == start {
                return result;
            }
        }

        // Boundary: also rotate clockwise from start via twin.next
        let twin_of_start = self.half_edges[start].twin;
        if twin_of_start != INVALID {
            let mut current = self.half_edges[twin_of_start].next;
            loop {
                if current == start {
                    break;
                }
                result.push(current);
                let twin = self.half_edges[current].twin;
                if twin == INVALID {
                    break;
                }
                current = self.half_edges[twin].next;
            }
        }

        result
    }

    fn neighbors(&self, v: usize) -> HashSet<usize> {
        self.outgoing_half_edges(v)
            .iter()
            .map(|&he| self.half_edges[he].target)
            .collect()
    }

    fn is_boundary_vertex(&self, v: usize) -> bool {
        for &he in &self.outgoing_half_edges(v) {
            if self.half_edges[he].twin == INVALID {
                return true;
            }
        }
        false
    }

    /// Check the link condition: common neighbors must equal exactly the
    /// face apices opposite the edge (2 for interior, 1 for boundary).
    fn check_link_condition(&self, v1: usize, v2: usize) -> bool {
        let n1 = self.neighbors(v1);
        let n2 = self.neighbors(v2);
        let common_count = n1.intersection(&n2).count();

        let h = match self.find_half_edge(v1, v2) {
            Some(h) => h,
            None => return false,
        };
        let is_boundary = self.half_edges[h].twin == INVALID;
        let expected = if is_boundary { 1 } else { 2 };
        common_count == expected
    }

    fn find_half_edge(&self, from: usize, to: usize) -> Option<usize> {
        for &he in &self.outgoing_half_edges(from) {
            if self.half_edges[he].target == to {
                return Some(he);
            }
        }
        None
    }

    fn compute_collapse_cost(&self, v1: usize, v2: usize) -> (Point3f, f64) {
        let q = self.quadrics[v1] + self.quadrics[v2];
        let q3 = q.fixed_view::<3, 3>(0, 0);
        let q1 = q.fixed_view::<3, 1>(0, 3);

        let optimal = if let Some(inv) = q3.try_inverse() {
            let p = -inv * q1;
            Point3f::new(p[0] as f32, p[1] as f32, p[2] as f32)
        } else {
            Point3f::from((self.positions[v1].coords + self.positions[v2].coords) * 0.5)
        };

        let vh = Vector4::new(
            optimal.x as f64,
            optimal.y as f64,
            optimal.z as f64,
            1.0,
        );
        let cost = (vh.transpose() * q * vh)[0].max(0.0);
        (optimal, cost)
    }

    /// Find any valid outgoing half-edge from a vertex (linear scan fallback).
    fn find_valid_outgoing(&self, v: usize) -> usize {
        for (i, he) in self.half_edges.iter().enumerate() {
            if he.face != INVALID && self.source(i) == v {
                return i;
            }
        }
        INVALID
    }

    /// Collapse edge (v1, v2), merging v2 into v1 at new_pos.
    /// Returns true on success.
    fn collapse_edge(&mut self, v1: usize, v2: usize, new_pos: Point3f) -> bool {
        let h = match self.find_half_edge(v1, v2) {
            Some(h) => h,
            None => return false,
        };

        let h_twin = self.half_edges[h].twin;
        let h_next = self.half_edges[h].next;
        let h_prev = self.half_edges[h].prev;
        let face_a = self.half_edges[h].face;
        let h_next_twin = self.half_edges[h_next].twin;
        let h_prev_twin = self.half_edges[h_prev].twin;
        let c = self.half_edges[h_next].target;

        let (face_b, ht_next, ht_prev, ht_next_twin, ht_prev_twin, d) = if h_twin != INVALID {
            let hn = self.half_edges[h_twin].next;
            let hp = self.half_edges[h_twin].prev;
            (
                self.half_edges[h_twin].face,
                hn,
                hp,
                self.half_edges[hn].twin,
                self.half_edges[hp].twin,
                self.half_edges[hn].target,
            )
        } else {
            (INVALID, INVALID, INVALID, INVALID, INVALID, INVALID)
        };

        // Collect v2 outgoing edges BEFORE any modifications
        let v2_outgoing = self.outgoing_half_edges(v2);

        // Re-pair twins for face A border edges
        if h_next_twin != INVALID {
            self.half_edges[h_next_twin].twin = h_prev_twin;
        }
        if h_prev_twin != INVALID {
            self.half_edges[h_prev_twin].twin = h_next_twin;
        }

        // Mark face A as removed
        self.half_edges[h].face = INVALID;
        self.half_edges[h_next].face = INVALID;
        self.half_edges[h_prev].face = INVALID;
        self.face_edge[face_a] = INVALID;
        self.active_face_count -= 1;

        // Handle face B
        if face_b != INVALID {
            if ht_next_twin != INVALID {
                self.half_edges[ht_next_twin].twin = ht_prev_twin;
            }
            if ht_prev_twin != INVALID {
                self.half_edges[ht_prev_twin].twin = ht_next_twin;
            }
            self.half_edges[h_twin].face = INVALID;
            self.half_edges[ht_next].face = INVALID;
            self.half_edges[ht_prev].face = INVALID;
            self.face_edge[face_b] = INVALID;
            self.active_face_count -= 1;
        }

        // Redirect all v2 references to v1
        for &he in &v2_outgoing {
            let prev = self.half_edges[he].prev;
            self.half_edges[prev].target = v1;

            let twin = self.half_edges[he].twin;
            if twin != INVALID && self.half_edges[twin].face != INVALID {
                self.half_edges[twin].target = v1;
            }
        }

        // Fix vertex_edge pointers for v1
        if self.half_edges[self.vertex_edge[v1]].face == INVALID {
            if h_prev_twin != INVALID && self.half_edges[h_prev_twin].face != INVALID {
                self.vertex_edge[v1] = h_prev_twin;
            } else {
                self.vertex_edge[v1] = self.find_valid_outgoing(v1);
            }
        }

        // Fix vertex_edge for c
        if c != INVALID
            && self.vertex_edge[c] != INVALID
            && self.half_edges[self.vertex_edge[c]].face == INVALID
        {
            if h_next_twin != INVALID && self.half_edges[h_next_twin].face != INVALID {
                self.vertex_edge[c] = h_next_twin;
            } else {
                self.vertex_edge[c] = self.find_valid_outgoing(c);
            }
        }

        // Fix vertex_edge for d
        if d != INVALID
            && d != c
            && self.vertex_edge[d] != INVALID
            && self.half_edges[self.vertex_edge[d]].face == INVALID
        {
            if ht_next_twin != INVALID && self.half_edges[ht_next_twin].face != INVALID {
                self.vertex_edge[d] = ht_next_twin;
            } else {
                self.vertex_edge[d] = self.find_valid_outgoing(d);
            }
        }

        // Mark v2 as removed
        self.vertex_edge[v2] = INVALID;
        self.vertex_removed[v2] = true;

        // Update position and quadric for v1
        let v2_quadric = self.quadrics[v2];
        self.positions[v1] = new_pos;
        self.quadrics[v1] += v2_quadric;

        // Interpolate normals
        if let Some(ref mut normals) = self.normals {
            let n1 = normals[v1];
            let n2 = normals[v2];
            let avg = (n1 + n2).normalize();
            if avg.iter().all(|x| x.is_finite()) {
                normals[v1] = avg;
            }
        }

        // Interpolate colors
        if let Some(ref mut colors) = self.colors {
            let c1 = colors[v1];
            let c2 = colors[v2];
            colors[v1] = [
                ((c1[0] as u16 + c2[0] as u16) / 2) as u8,
                ((c1[1] as u16 + c2[1] as u16) / 2) as u8,
                ((c1[2] as u16 + c2[2] as u16) / 2) as u8,
            ];
        }

        true
    }

    fn to_triangle_mesh(&self) -> TriangleMesh {
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_positions = Vec::new();
        let mut new_normals = self.normals.as_ref().map(|_| Vec::new());
        let mut new_colors = self.colors.as_ref().map(|_| Vec::new());

        for (i, &removed) in self.vertex_removed.iter().enumerate() {
            if !removed && self.vertex_edge[i] != INVALID {
                old_to_new.insert(i, new_positions.len());
                new_positions.push(self.positions[i]);
                if let Some(ref normals) = self.normals {
                    new_normals.as_mut().unwrap().push(normals[i]);
                }
                if let Some(ref colors) = self.colors {
                    new_colors.as_mut().unwrap().push(colors[i]);
                }
            }
        }

        let mut new_faces = Vec::new();
        for fi in 0..self.face_edge.len() {
            let he0 = self.face_edge[fi];
            if he0 == INVALID {
                continue;
            }
            let he1 = self.half_edges[he0].next;
            let v0 = self.source(he0);
            let v1 = self.half_edges[he0].target;
            let v2 = self.half_edges[he1].target;

            if let (Some(&nv0), Some(&nv1), Some(&nv2)) =
                (old_to_new.get(&v0), old_to_new.get(&v1), old_to_new.get(&v2))
            {
                if nv0 != nv1 && nv1 != nv2 && nv2 != nv0 {
                    new_faces.push([nv0, nv1, nv2]);
                }
            }
        }

        let mut mesh = TriangleMesh::from_vertices_and_faces(new_positions, new_faces);
        if let Some(normals) = new_normals {
            mesh.set_normals(normals);
        }
        if let Some(colors) = new_colors {
            mesh.set_colors(colors);
        }
        mesh
    }
}

// ============================================================
// Edge Cost for Priority Queue
// ============================================================

#[derive(Debug, Clone)]
struct EdgeCost {
    v1: usize,
    v2: usize,
    #[allow(dead_code)]
    position: Point3f,
    cost: f64,
}

impl PartialEq for EdgeCost {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal
    }
}
impl Eq for EdgeCost {}

impl PartialOrd for EdgeCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCost {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smallest cost first
        other.cost.total_cmp(&self.cost)
    }
}

// ============================================================
// Edge Collapse Simplifier
// ============================================================

/// Edge collapse mesh simplifier using half-edge data structure and QEM.
///
/// This simplifier builds a half-edge mesh for efficient local topology
/// queries (neighbor iteration, boundary detection, link condition checks)
/// and uses quadric error metrics to prioritize edge collapses.
pub struct EdgeCollapseSimplifier {
    /// Stop when the minimum collapse cost exceeds this threshold
    pub error_threshold: Option<f64>,
    /// Preserve mesh boundary edges
    pub preserve_boundary: bool,
    /// Extra penalty weight applied to boundary edge costs
    pub boundary_weight: f64,
}

impl Default for EdgeCollapseSimplifier {
    fn default() -> Self {
        Self {
            error_threshold: None,
            preserve_boundary: true,
            boundary_weight: 100.0,
        }
    }
}

impl EdgeCollapseSimplifier {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(
        error_threshold: Option<f64>,
        preserve_boundary: bool,
        boundary_weight: f64,
    ) -> Self {
        Self {
            error_threshold,
            preserve_boundary,
            boundary_weight,
        }
    }

    /// Build the initial priority queue of edge collapse candidates.
    fn build_queue(&self, hem: &HalfEdgeMesh) -> PriorityQueue<usize, EdgeCost> {
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

                // Skip boundary edges if preserving boundary
                if self.preserve_boundary
                    && (hem.is_boundary_vertex(vi) || hem.is_boundary_vertex(target))
                {
                    continue;
                }

                let (pos, mut cost) = hem.compute_collapse_cost(vi, target);

                // Apply boundary penalty (if not fully skipping boundary)
                if !self.preserve_boundary
                    && (hem.is_boundary_vertex(vi) || hem.is_boundary_vertex(target))
                {
                    cost += self.boundary_weight;
                }

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

    /// Rebuild the queue after many collapses to maintain accuracy.
    fn rebuild_queue(&self, hem: &HalfEdgeMesh, id_offset: usize) -> PriorityQueue<usize, EdgeCost> {
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

                if self.preserve_boundary
                    && (hem.is_boundary_vertex(vi) || hem.is_boundary_vertex(target))
                {
                    continue;
                }

                let (pos, mut cost) = hem.compute_collapse_cost(vi, target);
                if !self.preserve_boundary
                    && (hem.is_boundary_vertex(vi) || hem.is_boundary_vertex(target))
                {
                    cost += self.boundary_weight;
                }

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
}

impl MeshSimplifier for EdgeCollapseSimplifier {
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

        let target_faces = ((1.0 - reduction_ratio) * mesh.faces.len() as f32) as usize;
        let mut hem = HalfEdgeMesh::from_triangle_mesh(mesh);
        let mut queue = self.build_queue(&hem);
        let mut collapse_count = 0usize;

        while hem.active_face_count > target_faces && !queue.is_empty() {
            let (_, edge_cost) = match queue.pop() {
                Some(item) => item,
                None => break,
            };

            // Check error threshold
            if let Some(threshold) = self.error_threshold {
                if edge_cost.cost > threshold {
                    break;
                }
            }

            let v1 = edge_cost.v1;
            let v2 = edge_cost.v2;

            // Validate: both vertices still alive and still neighbors
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

            // Check link condition to avoid non-manifold topology
            if !hem.check_link_condition(v1, v2) {
                continue;
            }

            // Recompute cost (may have changed since queuing)
            let (pos, _cost) = hem.compute_collapse_cost(v1, v2);

            if hem.collapse_edge(v1, v2, pos) {
                collapse_count += 1;

                // Periodically rebuild queue for accuracy
                if collapse_count % 100 == 0 {
                    queue = self.rebuild_queue(&hem, collapse_count * 1000);
                }
            }
        }

        Ok(hem.to_triangle_mesh())
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
        // Consistently wound: each shared edge appears in opposite directions
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

    fn make_diamond() -> TriangleMesh {
        // Two tetrahedra glued at base, consistently wound (6 faces)
        TriangleMesh::from_vertices_and_faces(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(0.5, 0.5, 1.0),
                Point3::new(0.5, 0.5, -1.0),
            ],
            vec![
                [0, 1, 3],
                [1, 2, 3],
                [0, 3, 2],
                [0, 4, 1],
                [1, 4, 2],
                [0, 2, 4],
            ],
        )
    }

    // ---- Construction tests ----

    #[test]
    fn test_creation() {
        let s = EdgeCollapseSimplifier::new();
        assert!(s.preserve_boundary);
        assert!(s.error_threshold.is_none());
    }

    #[test]
    fn test_with_params() {
        let s = EdgeCollapseSimplifier::with_params(Some(0.01), false, 50.0);
        assert_eq!(s.error_threshold, Some(0.01));
        assert!(!s.preserve_boundary);
        assert_eq!(s.boundary_weight, 50.0);
    }

    // ---- Half-edge structure tests ----

    #[test]
    fn test_halfedge_construction() {
        let mesh = make_tetrahedron();
        let hem = HalfEdgeMesh::from_triangle_mesh(&mesh);
        assert_eq!(hem.half_edges.len(), 12); // 4 faces * 3
        assert_eq!(hem.active_face_count, 4);
        assert_eq!(hem.positions.len(), 4);

        // Every interior half-edge should have a twin
        for he in &hem.half_edges {
            assert_ne!(he.twin, INVALID, "interior half-edge should have twin");
        }
    }

    #[test]
    fn test_halfedge_boundary() {
        let mesh = make_single_triangle();
        let hem = HalfEdgeMesh::from_triangle_mesh(&mesh);
        // Single triangle: all 3 edges are boundary
        for he in &hem.half_edges {
            assert_eq!(he.twin, INVALID);
        }
        assert!(hem.is_boundary_vertex(0));
        assert!(hem.is_boundary_vertex(1));
        assert!(hem.is_boundary_vertex(2));
    }

    #[test]
    fn test_halfedge_neighbors() {
        let mesh = make_tetrahedron();
        let hem = HalfEdgeMesh::from_triangle_mesh(&mesh);
        // Each vertex in a tetrahedron has 3 neighbors
        for v in 0..4 {
            let nbrs = hem.neighbors(v);
            assert_eq!(nbrs.len(), 3, "tetrahedron vertex should have 3 neighbors");
        }
    }

    #[test]
    fn test_link_condition_tetrahedron() {
        let mesh = make_tetrahedron();
        let hem = HalfEdgeMesh::from_triangle_mesh(&mesh);
        // In a tetrahedron, every pair of vertices shares exactly 2 common neighbors
        // (the other 2 vertices). Link condition should be satisfied for interior edges.
        assert!(hem.check_link_condition(0, 1));
        assert!(hem.check_link_condition(1, 2));
    }

    // ---- Simplification tests ----

    #[test]
    fn test_empty_mesh() {
        let s = EdgeCollapseSimplifier::new();
        let mesh = TriangleMesh::new();
        assert!(s.simplify(&mesh, 0.5).is_err());
    }

    #[test]
    fn test_invalid_reduction_ratio() {
        let s = EdgeCollapseSimplifier::new();
        let mesh = make_single_triangle();
        assert!(s.simplify(&mesh, -0.1).is_err());
        assert!(s.simplify(&mesh, 1.1).is_err());
    }

    #[test]
    fn test_zero_reduction() {
        let s = EdgeCollapseSimplifier::new();
        let mesh = make_single_triangle();
        let result = s.simplify(&mesh, 0.0).unwrap();
        assert_eq!(result.vertex_count(), 3);
        assert_eq!(result.face_count(), 1);
    }

    #[test]
    fn test_tetrahedron_simplification() {
        let s = EdgeCollapseSimplifier::with_params(None, false, 0.0);
        let mesh = make_tetrahedron();
        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() <= mesh.face_count());
        assert!(result.vertex_count() <= mesh.vertex_count());
    }

    #[test]
    fn test_planar_grid_simplification() {
        let s = EdgeCollapseSimplifier::new();
        let mesh = make_plane_grid(6);
        let original_faces = mesh.face_count();
        assert_eq!(original_faces, 50); // 5*5*2

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() < original_faces);
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_curved_surface_simplification() {
        let s = EdgeCollapseSimplifier::new();
        let mesh = make_curved_surface(8);
        let original_faces = mesh.face_count();

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() < original_faces);
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_complex_mesh_simplification() {
        let s = EdgeCollapseSimplifier::with_params(None, false, 0.0);
        let mesh = make_diamond();
        let original_faces = mesh.face_count();

        let result = s.simplify(&mesh, 0.3).unwrap();
        assert!(result.face_count() <= original_faces);
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_boundary_preservation() {
        let s = EdgeCollapseSimplifier::new(); // preserve_boundary = true
        let mesh = make_plane_grid(6);

        // Collect original boundary vertex positions
        let original_boundary: HashSet<(i32, i32, i32)> = {
            let size = 6;
            let mut set = HashSet::new();
            for i in 0..size {
                for j in 0..size {
                    if i == 0 || i == size - 1 || j == 0 || j == size - 1 {
                        let idx = i * size + j;
                        let p = mesh.vertices[idx];
                        set.insert(((p.x * 100.0) as i32, (p.y * 100.0) as i32, (p.z * 100.0) as i32));
                    }
                }
            }
            set
        };

        let result = s.simplify(&mesh, 0.5).unwrap();
        let result_positions: HashSet<(i32, i32, i32)> = result
            .vertices
            .iter()
            .map(|p| ((p.x * 100.0) as i32, (p.y * 100.0) as i32, (p.z * 100.0) as i32))
            .collect();

        let preserved = original_boundary.intersection(&result_positions).count();
        let ratio = preserved as f32 / original_boundary.len() as f32;
        assert!(
            ratio > 0.9,
            "Expected >90% boundary preservation, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_error_threshold() {
        let s = EdgeCollapseSimplifier::with_params(Some(0.0001), false, 0.0);
        let mesh = make_plane_grid(6);
        let result = s.simplify(&mesh, 0.99).unwrap();
        // Very tight threshold should prevent most collapses on a flat grid
        // (costs are near zero for coplanar faces, so some collapses will happen)
        assert!(result.face_count() > 0);
    }

    #[test]
    fn test_attribute_preservation_normals() {
        let mut mesh = make_plane_grid(5);
        let normals: Vec<Vector3f> = (0..mesh.vertex_count())
            .map(|_| Vector3f::new(0.0, 0.0, 1.0))
            .collect();
        mesh.set_normals(normals);

        let s = EdgeCollapseSimplifier::new();
        let result = s.simplify(&mesh, 0.3).unwrap();
        assert!(result.normals.is_some(), "normals should be preserved");
        let result_normals = result.normals.as_ref().unwrap();
        assert_eq!(result_normals.len(), result.vertex_count());
        for n in result_normals {
            // Planar mesh: normals should stay close to (0, 0, 1)
            assert!(n.z > 0.9, "normal z should be close to 1.0, got {}", n.z);
        }
    }

    #[test]
    fn test_attribute_preservation_colors() {
        let mut mesh = make_plane_grid(5);
        let colors: Vec<[u8; 3]> = (0..mesh.vertex_count()).map(|_| [128, 64, 200]).collect();
        mesh.set_colors(colors);

        let s = EdgeCollapseSimplifier::new();
        let result = s.simplify(&mesh, 0.3).unwrap();
        assert!(result.colors.is_some(), "colors should be preserved");
        assert_eq!(result.colors.as_ref().unwrap().len(), result.vertex_count());
    }

    #[test]
    fn test_large_grid_simplification() {
        let s = EdgeCollapseSimplifier::new();
        let mesh = make_plane_grid(11);
        let original = mesh.face_count(); // 200 faces

        let result = s.simplify(&mesh, 0.5).unwrap();
        assert!(result.face_count() < original);
        assert!(result.face_count() > 0);
        assert!(result.vertex_count() > 0);
    }
}
