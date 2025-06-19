//! Mesh data structures and functionality

use crate::point::*;
// use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

/// A triangle mesh with vertices and faces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleMesh {
    pub vertices: Vec<Point3f>,
    pub faces: Vec<[usize; 3]>,
    pub normals: Option<Vec<Vector3f>>,
    pub colors: Option<Vec<[u8; 3]>>,
}

/// A mesh with colored vertices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColoredTriangleMesh {
    pub vertices: Vec<ColoredPoint3f>,
    pub faces: Vec<[usize; 3]>,
    pub normals: Option<Vec<Vector3f>>,
}

impl TriangleMesh {
    /// Create a new empty mesh
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            normals: None,
            colors: None,
        }
    }

    /// Create a mesh from vertices and faces
    pub fn from_vertices_and_faces(vertices: Vec<Point3f>, faces: Vec<[usize; 3]>) -> Self {
        Self {
            vertices,
            faces,
            normals: None,
            colors: None,
        }
    }

    /// Get the number of vertices
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of faces
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Check if the mesh is empty
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }

    /// Add a vertex to the mesh
    pub fn add_vertex(&mut self, vertex: Point3f) -> usize {
        let index = self.vertices.len();
        self.vertices.push(vertex);
        index
    }

    /// Add a face to the mesh
    pub fn add_face(&mut self, face: [usize; 3]) {
        self.faces.push(face);
    }

    /// Calculate face normals
    pub fn calculate_face_normals(&self) -> Vec<Vector3f> {
        self.faces
            .iter()
            .map(|face| {
                let v0 = self.vertices[face[0]];
                let v1 = self.vertices[face[1]];
                let v2 = self.vertices[face[2]];
                
                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                
                edge1.cross(&edge2).normalize()
            })
            .collect()
    }

    /// Set vertex normals
    pub fn set_normals(&mut self, normals: Vec<Vector3f>) {
        if normals.len() == self.vertices.len() {
            self.normals = Some(normals);
        }
    }

    /// Set vertex colors
    pub fn set_colors(&mut self, colors: Vec<[u8; 3]>) {
        if colors.len() == self.vertices.len() {
            self.colors = Some(colors);
        }
    }

    /// Clear the mesh
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.faces.clear();
        self.normals = None;
        self.colors = None;
    }
}

impl Default for TriangleMesh {
    fn default() -> Self {
        Self::new()
    }
}

impl ColoredTriangleMesh {
    /// Create a new empty colored mesh
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
            normals: None,
        }
    }

    /// Get the number of vertices
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of faces
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Check if the mesh is empty
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }
}

impl Default for ColoredTriangleMesh {
    fn default() -> Self {
        Self::new()
    }
} 