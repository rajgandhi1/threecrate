//! Bevy mesh interoperability
//!
//! This module provides conversions between threecrate mesh types and Bevy mesh types.

use crate::mesh::TriangleMesh;
use crate::Result;
use bevy::render::mesh::{Mesh, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;

impl TriangleMesh {
    /// Convert a TriangleMesh to a Bevy Mesh
    ///
    /// This creates a Bevy mesh with:
    /// - Vertex positions from the TriangleMesh vertices
    /// - Triangle indices from the TriangleMesh faces
    /// - Normals (if available in the TriangleMesh)
    /// - Colors (if available in the TriangleMesh, converted to vertex colors)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use threecrate_core::{TriangleMesh, Point3f};
    ///
    /// let mut mesh = TriangleMesh::new();
    /// mesh.add_vertex(Point3f::new(0.0, 0.0, 0.0));
    /// mesh.add_vertex(Point3f::new(1.0, 0.0, 0.0));
    /// mesh.add_vertex(Point3f::new(0.5, 1.0, 0.0));
    /// mesh.add_face([0, 1, 2]);
    ///
    /// let bevy_mesh = mesh.to_bevy_mesh().unwrap();
    /// ```
    pub fn to_bevy_mesh(&self) -> Result<Mesh> {
        if self.is_empty() {
            return Err(crate::Error::InvalidData(
                "Cannot convert empty mesh to Bevy mesh".to_string(),
            ));
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );

        // Convert vertices to Bevy format [f32; 3]
        let positions: Vec<[f32; 3]> = self
            .vertices
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect();

        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

        // Add normals if available
        if let Some(ref normals) = self.normals {
            let bevy_normals: Vec<[f32; 3]> = normals
                .iter()
                .map(|n| [n.x, n.y, n.z])
                .collect();
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, bevy_normals);
        }

        // Add colors if available (convert from [u8; 3] to [f32; 4])
        if let Some(ref colors) = self.colors {
            let bevy_colors: Vec<[f32; 4]> = colors
                .iter()
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                        1.0, // Alpha
                    ]
                })
                .collect();
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, bevy_colors);
        }

        // Convert faces to indices (flatten the triangle array)
        let indices: Vec<u32> = self
            .faces
            .iter()
            .flat_map(|face| vec![face[0] as u32, face[1] as u32, face[2] as u32])
            .collect();

        mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));

        Ok(mesh)
    }

    /// Create a TriangleMesh from a Bevy Mesh
    ///
    /// This attempts to extract triangle data from a Bevy mesh and create a TriangleMesh.
    /// Only works with triangle-based meshes (TriangleList topology).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use threecrate_core::TriangleMesh;
    /// use bevy::render::mesh::Mesh;
    ///
    /// let bevy_mesh = Mesh::new(
    ///     bevy::render::mesh::PrimitiveTopology::TriangleList,
    ///     bevy::render::render_asset::RenderAssetUsages::default()
    /// );
    /// // ... populate bevy_mesh ...
    ///
    /// let triangle_mesh = TriangleMesh::from_bevy_mesh(&bevy_mesh).unwrap();
    /// ```
    pub fn from_bevy_mesh(bevy_mesh: &Mesh) -> Result<Self> {
        // Extract positions
        let positions = bevy_mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .ok_or_else(|| {
                crate::Error::InvalidData("Bevy mesh missing position attribute".to_string())
            })?;

        let vertices: Vec<crate::Point3f> = match positions {
            bevy::render::mesh::VertexAttributeValues::Float32x3(positions) => positions
                .iter()
                .map(|p| crate::Point3f::new(p[0], p[1], p[2]))
                .collect(),
            _ => {
                return Err(crate::Error::InvalidData(
                    "Unexpected position attribute format".to_string(),
                ))
            }
        };

        // Extract indices
        let indices = bevy_mesh.indices().ok_or_else(|| {
            crate::Error::InvalidData("Bevy mesh missing indices".to_string())
        })?;

        let faces: Vec<[usize; 3]> = match indices {
            bevy::render::mesh::Indices::U16(idx) => idx
                .chunks(3)
                .map(|chunk| {
                    if chunk.len() == 3 {
                        Ok([chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
                    } else {
                        Err(crate::Error::InvalidData(
                            "Incomplete triangle in index buffer".to_string(),
                        ))
                    }
                })
                .collect::<Result<Vec<_>>>()?,
            bevy::render::mesh::Indices::U32(idx) => idx
                .chunks(3)
                .map(|chunk| {
                    if chunk.len() == 3 {
                        Ok([chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
                    } else {
                        Err(crate::Error::InvalidData(
                            "Incomplete triangle in index buffer".to_string(),
                        ))
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        };

        let mut mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);

        // Extract normals if available
        if let Some(normals_attr) = bevy_mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            if let bevy::render::mesh::VertexAttributeValues::Float32x3(normals) = normals_attr {
                let normal_vecs: Vec<crate::Vector3f> = normals
                    .iter()
                    .map(|n| crate::Vector3f::new(n[0], n[1], n[2]))
                    .collect();
                mesh.set_normals(normal_vecs);
            }
        }

        // Extract colors if available (convert from [f32; 4] to [u8; 3])
        if let Some(colors_attr) = bevy_mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
            if let bevy::render::mesh::VertexAttributeValues::Float32x4(colors) = colors_attr {
                let color_bytes: Vec<[u8; 3]> = colors
                    .iter()
                    .map(|c| {
                        [
                            (c[0] * 255.0).clamp(0.0, 255.0) as u8,
                            (c[1] * 255.0).clamp(0.0, 255.0) as u8,
                            (c[2] * 255.0).clamp(0.0, 255.0) as u8,
                        ]
                    })
                    .collect();
                mesh.set_colors(color_bytes);
            }
        }

        Ok(mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Point3f, Vector3f};

    #[test]
    fn test_to_bevy_mesh() {
        let mut mesh = TriangleMesh::new();
        mesh.add_vertex(Point3f::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Point3f::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Point3f::new(0.5, 1.0, 0.0));
        mesh.add_face([0, 1, 2]);

        let bevy_mesh = mesh.to_bevy_mesh().unwrap();

        // Verify the mesh was created
        assert!(bevy_mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
        assert!(bevy_mesh.indices().is_some());
    }

    #[test]
    fn test_to_bevy_mesh_with_normals() {
        let mut mesh = TriangleMesh::new();
        mesh.add_vertex(Point3f::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Point3f::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Point3f::new(0.5, 1.0, 0.0));
        mesh.add_face([0, 1, 2]);

        mesh.set_normals(vec![
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 0.0, 1.0),
        ]);

        let bevy_mesh = mesh.to_bevy_mesh().unwrap();

        assert!(bevy_mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
    }

    #[test]
    fn test_to_bevy_mesh_with_colors() {
        let mut mesh = TriangleMesh::new();
        mesh.add_vertex(Point3f::new(0.0, 0.0, 0.0));
        mesh.add_vertex(Point3f::new(1.0, 0.0, 0.0));
        mesh.add_vertex(Point3f::new(0.5, 1.0, 0.0));
        mesh.add_face([0, 1, 2]);

        mesh.set_colors(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]]);

        let bevy_mesh = mesh.to_bevy_mesh().unwrap();

        assert!(bevy_mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
    }

    #[test]
    fn test_empty_mesh_error() {
        let mesh = TriangleMesh::new();
        assert!(mesh.to_bevy_mesh().is_err());
    }
}
