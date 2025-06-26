//! Quadric error decimation
//! 
//! Implementation of the Garland-Heckbert algorithm for mesh simplification
//! based on quadric error metrics.

use threecrate_core::{TriangleMesh, Result, Error, Point3f};
use crate::MeshSimplifier;
use nalgebra::{Matrix4, Vector4};
// use rayon::prelude::*;
use priority_queue::PriorityQueue;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

/// Edge collapse operation
#[derive(Debug, Clone)]
struct EdgeCollapse {
    /// Source vertex index
    vertex1: usize,
    /// Target vertex index  
    vertex2: usize,
    /// New position after collapse
    new_position: Point3f,
    /// Quadric error cost
    cost: f64,
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal
    }
}

impl Eq for EdgeCollapse {}

impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (smallest cost first)
        other.cost.total_cmp(&self.cost)
    }
}

/// Face adjacency information
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct FaceInfo {
    vertices: [usize; 3],
    plane: Vector4<f64>, // ax + by + cz + d = 0
}

/// Vertex information including quadric matrix
#[derive(Debug, Clone)]
struct VertexInfo {
    position: Point3f,
    quadric: Matrix4<f64>,
    faces: HashSet<usize>,
    neighbors: HashSet<usize>,
}

/// Quadric error decimation simplifier
pub struct QuadricErrorSimplifier {
    /// Configuration parameters
    pub max_edge_length: Option<f32>,
    pub preserve_boundary: bool,
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
        let a = plane[0];
        let b = plane[1]; 
        let c = plane[2];
        let d = plane[3];
        
        Matrix4::new(
            a*a, a*b, a*c, a*d,
            a*b, b*b, b*c, b*d,
            a*c, b*c, c*c, c*d,
            a*d, b*d, c*d, d*d,
        )
    }
    
    /// Initialize vertex information including quadrics
    fn initialize_vertices(&self, mesh: &TriangleMesh) -> Vec<VertexInfo> {
        let mut vertices: Vec<VertexInfo> = mesh.vertices.iter().enumerate().map(|(_i, &pos)| {
            VertexInfo {
                position: pos,
                quadric: Matrix4::zeros(),
                faces: HashSet::new(),
                neighbors: HashSet::new(),
            }
        }).collect();
        
        // Compute face planes and accumulate quadrics
        for (face_idx, face) in mesh.faces.iter().enumerate() {
            let v0 = mesh.vertices[face[0]];
            let v1 = mesh.vertices[face[1]];
            let v2 = mesh.vertices[face[2]];
            
            let plane = self.compute_plane(&v0, &v1, &v2);
            let quadric = self.plane_to_quadric(&plane);
            
            // Add quadric to each vertex of the face
            for &vertex_idx in face.iter() {
                vertices[vertex_idx].quadric += quadric;
                vertices[vertex_idx].faces.insert(face_idx);
            }
            
            // Build adjacency
            vertices[face[0]].neighbors.insert(face[1]);
            vertices[face[0]].neighbors.insert(face[2]);
            vertices[face[1]].neighbors.insert(face[0]);
            vertices[face[1]].neighbors.insert(face[2]);
            vertices[face[2]].neighbors.insert(face[0]);
            vertices[face[2]].neighbors.insert(face[1]);
        }
        
        vertices
    }
    
    /// Find boundary edges in the mesh
    fn find_boundary_edges(&self, mesh: &TriangleMesh) -> HashSet<(usize, usize)> {
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        
        for face in &mesh.faces {
            let edges = [
                (face[0].min(face[1]), face[0].max(face[1])),
                (face[1].min(face[2]), face[1].max(face[2])),
                (face[2].min(face[0]), face[2].max(face[0])),
            ];
            
            for edge in edges.iter() {
                *edge_count.entry(*edge).or_insert(0) += 1;
            }
        }
        
        edge_count.into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(edge, _)| edge)
            .collect()
    }
    
    /// Check if vertex is on boundary
    #[allow(dead_code)]
    fn is_boundary_vertex(&self, vertex_idx: usize, boundary_edges: &HashSet<(usize, usize)>) -> bool {
        boundary_edges.iter().any(|&(v1, v2)| v1 == vertex_idx || v2 == vertex_idx)
    }
    
    /// Compute optimal position and cost for edge collapse
    fn compute_collapse_cost(&self, v1_idx: usize, v2_idx: usize, vertices: &[VertexInfo]) -> Option<EdgeCollapse> {
        let v1 = &vertices[v1_idx];
        let v2 = &vertices[v2_idx];
        
        // Check edge length constraint
        if let Some(max_len) = self.max_edge_length {
            let edge_len = (v1.position - v2.position).magnitude();
            if edge_len > max_len {
                return None;
            }
        }
        
        // Combined quadric
        let q_combined = v1.quadric + v2.quadric;
        
        // Try to solve for optimal position: ∇(v^T Q v) = 0
        // This gives us: 2Qv = 0, or the 3x3 upper-left block of Q
        let q_3x3 = q_combined.fixed_view::<3, 3>(0, 0);
        let q_3x1 = q_combined.fixed_view::<3, 1>(0, 3);
        
        let optimal_pos = if let Some(inv_q) = q_3x3.try_inverse() {
            let optimal_homogeneous = -inv_q * q_3x1;
            Point3f::new(
                optimal_homogeneous[0] as f32,
                optimal_homogeneous[1] as f32,
                optimal_homogeneous[2] as f32,
            )
        } else {
                         // If not invertible, use midpoint
             Point3f::from((v1.position.coords + v2.position.coords) * 0.5)
        };
        
        // Compute quadric error at optimal position
        let pos_homogeneous = Vector4::new(
            optimal_pos.x as f64,
            optimal_pos.y as f64,
            optimal_pos.z as f64,
            1.0,
        );
        
                 let cost = (pos_homogeneous.transpose() * q_combined * pos_homogeneous)[0];
        
        Some(EdgeCollapse {
            vertex1: v1_idx,
            vertex2: v2_idx,
            new_position: optimal_pos,
            cost,
        })
    }
    
    /// Generate all valid edge collapses
    fn generate_edge_collapses(&self, vertices: &[VertexInfo], boundary_edges: &HashSet<(usize, usize)>) -> Vec<EdgeCollapse> {
        let mut collapses = Vec::new();
        
        for (v1_idx, vertex) in vertices.iter().enumerate() {
            for &v2_idx in &vertex.neighbors {
                if v1_idx < v2_idx { // Avoid duplicate edges
                    // Skip boundary edges if preservation is enabled
                    if self.preserve_boundary {
                        let edge = (v1_idx.min(v2_idx), v1_idx.max(v2_idx));
                        if boundary_edges.contains(&edge) {
                            continue;
                        }
                    }
                    
                    if let Some(collapse) = self.compute_collapse_cost(v1_idx, v2_idx, vertices) {
                        collapses.push(collapse);
                    }
                }
            }
        }
        
        collapses
    }
    
    /// Apply edge collapse to mesh data structures
    fn apply_collapse(
        &self,
        collapse: &EdgeCollapse,
        vertices: &mut Vec<VertexInfo>,
        faces: &mut Vec<[usize; 3]>,
        vertex_mapping: &mut HashMap<usize, usize>,
    ) -> Result<()> {
        let v1_idx = collapse.vertex1;
        let v2_idx = collapse.vertex2;
        
                 // Update vertex position and combine quadrics
         let v2_quadric = vertices[v2_idx].quadric.clone();
         vertices[v1_idx].position = collapse.new_position;
         vertices[v1_idx].quadric += v2_quadric;
        
        // Update faces that reference v2 to reference v1
        for face in faces.iter_mut() {
            for vertex_ref in face.iter_mut() {
                if *vertex_ref == v2_idx {
                    *vertex_ref = v1_idx;
                }
            }
        }
        
        // Remove degenerate faces (triangles with repeated vertices)
        faces.retain(|face| {
            face[0] != face[1] && face[1] != face[2] && face[2] != face[0]
        });
        
        // Update vertex mapping
        vertex_mapping.insert(v2_idx, v1_idx);
        
        // Update adjacency information
        let v2_neighbors = vertices[v2_idx].neighbors.clone();
        for &neighbor in &v2_neighbors {
            if neighbor != v1_idx {
                vertices[v1_idx].neighbors.insert(neighbor);
                vertices[neighbor].neighbors.remove(&v2_idx);
                vertices[neighbor].neighbors.insert(v1_idx);
            }
        }
        vertices[v1_idx].neighbors.remove(&v2_idx);
        
        // Mark v2 as invalid
        vertices[v2_idx].neighbors.clear();
        vertices[v2_idx].faces.clear();
        
        Ok(())
    }
    
    /// Rebuild mesh from simplified vertex and face data
    fn rebuild_mesh(&self, vertices: &[VertexInfo], faces: &[[usize; 3]]) -> TriangleMesh {
        // Create mapping from old indices to new indices (compacting)
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        let mut new_vertices = Vec::new();
        
        // Collect valid vertices
        for (old_idx, vertex) in vertices.iter().enumerate() {
            if !vertex.neighbors.is_empty() || !vertex.faces.is_empty() {
                old_to_new.insert(old_idx, new_vertices.len());
                new_vertices.push(vertex.position);
            }
        }
        
        // Update face indices
        let new_faces: Vec<[usize; 3]> = faces.iter()
            .filter_map(|face| {
                if let (Some(&new_v0), Some(&new_v1), Some(&new_v2)) = (
                    old_to_new.get(&face[0]),
                    old_to_new.get(&face[1]),
                    old_to_new.get(&face[2]),
                ) {
                    Some([new_v0, new_v1, new_v2])
                } else {
                    None
                }
            })
            .collect();
        
        TriangleMesh::from_vertices_and_faces(new_vertices, new_faces)
    }
}

impl MeshSimplifier for QuadricErrorSimplifier {
    /// Simplify mesh with target reduction ratio
    fn simplify(&self, mesh: &TriangleMesh, reduction_ratio: f32) -> Result<TriangleMesh> {
        if mesh.is_empty() {
            return Err(Error::InvalidData("Mesh is empty".to_string()));
        }
        
        if !(0.0..=1.0).contains(&reduction_ratio) {
            return Err(Error::InvalidData("Reduction ratio must be between 0.0 and 1.0".to_string()));
        }
        
        if reduction_ratio == 0.0 {
            return Ok(mesh.clone());
        }
        
        let target_face_count = ((1.0 - reduction_ratio) * mesh.faces.len() as f32) as usize;
        
        // Initialize vertex information
        let mut vertices = self.initialize_vertices(mesh);
        let mut faces = mesh.faces.clone();
        let mut vertex_mapping = HashMap::new();
        
        // Find boundary edges
        let boundary_edges = self.find_boundary_edges(mesh);
        
        // Generate initial edge collapses
        let mut collapse_queue = PriorityQueue::new();
        let initial_collapses = self.generate_edge_collapses(&vertices, &boundary_edges);
        
        for (idx, collapse) in initial_collapses.into_iter().enumerate() {
            collapse_queue.push(idx, collapse);
        }
        
        // Perform edge collapses until target is reached
        let mut collapse_counter = 0;
        while faces.len() > target_face_count && !collapse_queue.is_empty() {
            if let Some((_, collapse)) = collapse_queue.pop() {
                // Verify collapse is still valid
                if vertices[collapse.vertex1].neighbors.contains(&collapse.vertex2) {
                    self.apply_collapse(&collapse, &mut vertices, &mut faces, &mut vertex_mapping)?;
                    collapse_counter += 1;
                    
                    // Periodically regenerate collapses to maintain quality
                    if collapse_counter % 100 == 0 {
                        collapse_queue.clear();
                        let new_collapses = self.generate_edge_collapses(&vertices, &boundary_edges);
                        for (idx, collapse) in new_collapses.into_iter().enumerate() {
                            collapse_queue.push(collapse_counter * 1000 + idx, collapse);
                        }
                    }
                }
            }
        }
        
        Ok(self.rebuild_mesh(&vertices, &faces))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

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
        let faces = vec![
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ];
        let mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        
        let simplified = simplifier.simplify(&mesh, 0.5).unwrap();
        
        // Should have fewer faces
        assert!(simplified.face_count() <= mesh.face_count());
        assert!(simplified.vertex_count() <= mesh.vertex_count());
    }
} 