//! Mesh attribute serialization utilities
//!
//! This module provides comprehensive mesh attribute handling for serialization,
//! including normals, tangents, and UV coordinates. It ensures that mesh attributes
//! survive round-trip across different formats with optional recomputation.

use threecrate_core::{TriangleMesh, Vector3f, Result, Error};

#[cfg(test)]
use threecrate_core::Point3f;

/// Texture coordinates (UV mapping)
pub type UV = [f32; 2];

/// Tangent vector with handedness information
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tangent {
    /// Tangent vector
    pub vector: Vector3f,
    /// Handedness (-1.0 or 1.0)
    pub handedness: f32,
}

/// Extended mesh with full attribute support
#[derive(Debug, Clone)]
pub struct ExtendedTriangleMesh {
    /// Base mesh data
    pub mesh: TriangleMesh,
    /// Texture coordinates per vertex
    pub uvs: Option<Vec<UV>>,
    /// Tangent vectors per vertex
    pub tangents: Option<Vec<Tangent>>,
    /// Metadata for tracking attribute completeness
    pub metadata: MeshMetadata,
}

/// Metadata tracking mesh attribute completeness and validation
#[derive(Debug, Clone, Default)]
pub struct MeshMetadata {
    /// Whether normals were computed or loaded
    pub normals_computed: bool,
    /// Whether tangents were computed or loaded  
    pub tangents_computed: bool,
    /// Whether UVs were loaded from file
    pub uvs_loaded: bool,
    /// Validation errors or warnings
    pub validation_messages: Vec<String>,
    /// Original format information
    pub source_format: Option<String>,
    /// Attribute completeness score (0.0 to 1.0)
    pub completeness_score: f32,
}

/// Configuration for mesh attribute processing
#[derive(Debug, Clone)]
pub struct MeshAttributeOptions {
    /// Recompute normals if missing
    pub recompute_normals: bool,
    /// Recompute tangents if missing (requires UVs)
    pub recompute_tangents: bool,
    /// Generate default UVs if missing
    pub generate_default_uvs: bool,
    /// Validate attribute consistency
    pub validate_attributes: bool,
    /// Normalize vectors after computation
    pub normalize_vectors: bool,
    /// Smooth normals across shared vertices
    pub smooth_normals: bool,
}

impl Default for MeshAttributeOptions {
    fn default() -> Self {
        Self {
            recompute_normals: true,
            recompute_tangents: false, // Requires UVs, so disabled by default
            generate_default_uvs: false,
            validate_attributes: true,
            normalize_vectors: true,
            smooth_normals: true,
        }
    }
}

impl MeshAttributeOptions {
    /// Create options with all recomputation enabled
    pub fn recompute_all() -> Self {
        Self {
            recompute_normals: true,
            recompute_tangents: true,
            generate_default_uvs: true,
            validate_attributes: true,
            normalize_vectors: true,
            smooth_normals: true,
        }
    }
    
    /// Create options for read-only validation
    pub fn validate_only() -> Self {
        Self {
            recompute_normals: false,
            recompute_tangents: false,
            generate_default_uvs: false,
            validate_attributes: true,
            normalize_vectors: false,
            smooth_normals: false,
        }
    }
}

impl Tangent {
    /// Create a new tangent with vector and handedness
    pub fn new(vector: Vector3f, handedness: f32) -> Self {
        Self { vector, handedness }
    }
    
    /// Create a tangent from a vector (handedness = 1.0)
    pub fn from_vector(vector: Vector3f) -> Self {
        Self::new(vector, 1.0)
    }
}

impl ExtendedTriangleMesh {
    /// Create from a base TriangleMesh
    pub fn from_mesh(mesh: TriangleMesh) -> Self {
        Self {
            mesh,
            uvs: None,
            tangents: None,
            metadata: MeshMetadata::default(),
        }
    }
    
    /// Create with full attributes
    pub fn new(
        mesh: TriangleMesh,
        uvs: Option<Vec<UV>>,
        tangents: Option<Vec<Tangent>>,
    ) -> Self {
        let mut extended = Self {
            mesh,
            uvs,
            tangents,
            metadata: MeshMetadata::default(),
        };
        extended.update_metadata();
        extended
    }
    
    /// Get vertex count
    pub fn vertex_count(&self) -> usize {
        self.mesh.vertex_count()
    }
    
    /// Get face count
    pub fn face_count(&self) -> usize {
        self.mesh.face_count()
    }
    
    /// Check if mesh is empty
    pub fn is_empty(&self) -> bool {
        self.mesh.is_empty()
    }
    
    /// Set UV coordinates
    pub fn set_uvs(&mut self, uvs: Vec<UV>) {
        if uvs.len() == self.vertex_count() {
            self.uvs = Some(uvs);
            self.metadata.uvs_loaded = true;
            self.update_metadata();
        }
    }
    
    /// Set tangent vectors
    pub fn set_tangents(&mut self, tangents: Vec<Tangent>) {
        if tangents.len() == self.vertex_count() {
            self.tangents = Some(tangents);
            self.metadata.tangents_computed = true;
            self.update_metadata();
        }
    }
    
    /// Process mesh attributes with given options
    pub fn process_attributes(&mut self, options: &MeshAttributeOptions) -> Result<()> {
        if options.validate_attributes {
            self.validate_attributes()?;
        }
        
        if options.recompute_normals && self.mesh.normals.is_none() {
            self.compute_normals(options.smooth_normals, options.normalize_vectors)?;
        }
        
        if options.generate_default_uvs && self.uvs.is_none() {
            self.generate_default_uvs()?;
        }
        
        if options.recompute_tangents && self.tangents.is_none() && self.uvs.is_some() {
            self.compute_tangents(options.normalize_vectors)?;
        }
        
        self.update_metadata();
        Ok(())
    }
    
    /// Validate attribute consistency
    pub fn validate_attributes(&mut self) -> Result<()> {
        let vertex_count = self.vertex_count();
        self.metadata.validation_messages.clear();
        
        // Check normals
        if let Some(ref normals) = self.mesh.normals {
            if normals.len() != vertex_count {
                let msg = format!("Normal count mismatch: {} normals for {} vertices", 
                    normals.len(), vertex_count);
                self.metadata.validation_messages.push(msg.clone());
                return Err(Error::InvalidData(msg));
            }
            
            // Check for zero-length normals
            for (i, normal) in normals.iter().enumerate() {
                let length_sq = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
                if length_sq < 1e-6 {
                    let msg = format!("Zero-length normal at vertex {}", i);
                    self.metadata.validation_messages.push(msg);
                }
            }
        }
        
        // Check UVs
        if let Some(ref uvs) = self.uvs {
            if uvs.len() != vertex_count {
                let msg = format!("UV count mismatch: {} UVs for {} vertices", 
                    uvs.len(), vertex_count);
                self.metadata.validation_messages.push(msg.clone());
                return Err(Error::InvalidData(msg));
            }
            
            // Check for invalid UV coordinates
            for (i, uv) in uvs.iter().enumerate() {
                if !uv[0].is_finite() || !uv[1].is_finite() {
                    let msg = format!("Invalid UV coordinates at vertex {}: [{}, {}]", 
                        i, uv[0], uv[1]);
                    self.metadata.validation_messages.push(msg);
                }
            }
        }
        
        // Check tangents
        if let Some(ref tangents) = self.tangents {
            if tangents.len() != vertex_count {
                let msg = format!("Tangent count mismatch: {} tangents for {} vertices", 
                    tangents.len(), vertex_count);
                self.metadata.validation_messages.push(msg.clone());
                return Err(Error::InvalidData(msg));
            }
            
            // Check for zero-length tangents and valid handedness
            for (i, tangent) in tangents.iter().enumerate() {
                let length_sq = tangent.vector.x * tangent.vector.x + 
                    tangent.vector.y * tangent.vector.y + 
                    tangent.vector.z * tangent.vector.z;
                if length_sq < 1e-6 {
                    let msg = format!("Zero-length tangent at vertex {}", i);
                    self.metadata.validation_messages.push(msg);
                }
                
                if tangent.handedness.abs() != 1.0 {
                    let msg = format!("Invalid tangent handedness at vertex {}: {}", 
                        i, tangent.handedness);
                    self.metadata.validation_messages.push(msg);
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute vertex normals
    pub fn compute_normals(&mut self, smooth: bool, normalize: bool) -> Result<()> {
        let vertex_count = self.vertex_count();
        let mut normals = vec![Vector3f::new(0.0, 0.0, 0.0); vertex_count];
        
        // Compute face normals and accumulate to vertices
        for face in &self.mesh.faces {
            let v0 = self.mesh.vertices[face[0]];
            let v1 = self.mesh.vertices[face[1]];
            let v2 = self.mesh.vertices[face[2]];
            
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let face_normal = edge1.cross(&edge2);
            
            if smooth {
                // Smooth shading: accumulate face normals to vertices
                normals[face[0]] = normals[face[0]] + face_normal;
                normals[face[1]] = normals[face[1]] + face_normal;
                normals[face[2]] = normals[face[2]] + face_normal;
            } else {
                // Flat shading: use face normal for all vertices
                normals[face[0]] = face_normal;
                normals[face[1]] = face_normal;
                normals[face[2]] = face_normal;
            }
        }
        
        // Normalize if requested
        if normalize {
            for normal in &mut normals {
                let length = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
                if length > 1e-6 {
                    *normal = Vector3f::new(
                        normal.x / length,
                        normal.y / length,
                        normal.z / length,
                    );
                } else {
                    *normal = Vector3f::new(0.0, 0.0, 1.0); // Default up vector
                }
            }
        }
        
        self.mesh.set_normals(normals);
        self.metadata.normals_computed = true;
        Ok(())
    }
    
    /// Compute tangent vectors using Lengyel's method
    pub fn compute_tangents(&mut self, normalize: bool) -> Result<()> {
        let uvs = self.uvs.as_ref()
            .ok_or_else(|| Error::InvalidData("UV coordinates required for tangent computation".to_string()))?;
        
        let vertex_count = self.vertex_count();
        let mut tan1 = vec![Vector3f::new(0.0, 0.0, 0.0); vertex_count];
        let mut tan2 = vec![Vector3f::new(0.0, 0.0, 0.0); vertex_count];
        
        // Compute tangents per face
        for face in &self.mesh.faces {
            let i1 = face[0];
            let i2 = face[1];
            let i3 = face[2];
            
            let v1 = self.mesh.vertices[i1];
            let v2 = self.mesh.vertices[i2];
            let v3 = self.mesh.vertices[i3];
            
            let w1 = uvs[i1];
            let w2 = uvs[i2];
            let w3 = uvs[i3];
            
            let x1 = v2.x - v1.x;
            let x2 = v3.x - v1.x;
            let y1 = v2.y - v1.y;
            let y2 = v3.y - v1.y;
            let z1 = v2.z - v1.z;
            let z2 = v3.z - v1.z;
            
            let s1 = w2[0] - w1[0];
            let s2 = w3[0] - w1[0];
            let t1 = w2[1] - w1[1];
            let t2 = w3[1] - w1[1];
            
            let det = s1 * t2 - s2 * t1;
            let r = if det.abs() < 1e-6 { 1.0 } else { 1.0 / det };
            
            let sdir = Vector3f::new(
                (t2 * x1 - t1 * x2) * r,
                (t2 * y1 - t1 * y2) * r,
                (t2 * z1 - t1 * z2) * r,
            );
            
            let tdir = Vector3f::new(
                (s1 * x2 - s2 * x1) * r,
                (s1 * y2 - s2 * y1) * r,
                (s1 * z2 - s2 * z1) * r,
            );
            
            tan1[i1] = tan1[i1] + sdir;
            tan1[i2] = tan1[i2] + sdir;
            tan1[i3] = tan1[i3] + sdir;
            
            tan2[i1] = tan2[i1] + tdir;
            tan2[i2] = tan2[i2] + tdir;
            tan2[i3] = tan2[i3] + tdir;
        }
        
        let normals = self.mesh.normals.as_ref()
            .ok_or_else(|| Error::InvalidData("Normals required for tangent computation".to_string()))?;
        
        // Compute final tangents with Gram-Schmidt orthogonalization
        let mut tangents = Vec::with_capacity(vertex_count);
        
        for i in 0..vertex_count {
            let n = normals[i];
            let t = tan1[i];
            
            // Gram-Schmidt orthogonalize
            let tangent_vec = t - n * (n.x * t.x + n.y * t.y + n.z * t.z);
            
            let tangent_vec = if normalize {
                let length = (tangent_vec.x * tangent_vec.x + 
                    tangent_vec.y * tangent_vec.y + 
                    tangent_vec.z * tangent_vec.z).sqrt();
                if length > 1e-6 {
                    Vector3f::new(
                        tangent_vec.x / length,
                        tangent_vec.y / length,
                        tangent_vec.z / length,
                    )
                } else {
                    Vector3f::new(1.0, 0.0, 0.0) // Default tangent
                }
            } else {
                tangent_vec
            };
            
            // Calculate handedness
            let cross = n.cross(&tangent_vec);
            let handedness = if cross.x * tan2[i].x + cross.y * tan2[i].y + cross.z * tan2[i].z < 0.0 {
                -1.0
            } else {
                1.0
            };
            
            tangents.push(Tangent::new(tangent_vec, handedness));
        }
        
        self.tangents = Some(tangents);
        self.metadata.tangents_computed = true;
        Ok(())
    }
    
    /// Generate default UV coordinates (planar projection)
    pub fn generate_default_uvs(&mut self) -> Result<()> {
        let vertex_count = self.vertex_count();
        
        // Find bounding box
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        
        for vertex in &self.mesh.vertices {
            min_x = min_x.min(vertex.x);
            max_x = max_x.max(vertex.x);
            min_y = min_y.min(vertex.y);
            max_y = max_y.max(vertex.y);
            min_z = min_z.min(vertex.z);
            max_z = max_z.max(vertex.z);
        }
        
        let size_x = max_x - min_x;
        let size_y = max_y - min_y;
        let size_z = max_z - min_z;
        
        // Choose projection plane based on largest dimension
        let mut uvs = Vec::with_capacity(vertex_count);
        
        if size_x >= size_y && size_x >= size_z {
            // Project onto YZ plane
            for vertex in &self.mesh.vertices {
                let u = if size_y > 1e-6 { (vertex.y - min_y) / size_y } else { 0.5 };
                let v = if size_z > 1e-6 { (vertex.z - min_z) / size_z } else { 0.5 };
                uvs.push([u, v]);
            }
        } else if size_y >= size_x && size_y >= size_z {
            // Project onto XZ plane
            for vertex in &self.mesh.vertices {
                let u = if size_x > 1e-6 { (vertex.x - min_x) / size_x } else { 0.5 };
                let v = if size_z > 1e-6 { (vertex.z - min_z) / size_z } else { 0.5 };
                uvs.push([u, v]);
            }
        } else {
            // Project onto XY plane
            for vertex in &self.mesh.vertices {
                let u = if size_x > 1e-6 { (vertex.x - min_x) / size_x } else { 0.5 };
                let v = if size_y > 1e-6 { (vertex.y - min_y) / size_y } else { 0.5 };
                uvs.push([u, v]);
            }
        }
        
        self.uvs = Some(uvs);
        Ok(())
    }
    
    /// Update metadata based on current state
    fn update_metadata(&mut self) {
        let vertex_count = self.vertex_count();
        if vertex_count == 0 {
            self.metadata.completeness_score = 0.0;
            return;
        }
        
        let mut score = 1.0; // Base score for having vertices
        
        // Check normals
        if let Some(ref normals) = self.mesh.normals {
            if normals.len() == vertex_count {
                score += 1.0;
            } else {
                score += 0.5; // Partial credit
            }
        }
        
        // Check UVs
        if let Some(ref uvs) = self.uvs {
            if uvs.len() == vertex_count {
                score += 1.0;
            } else {
                score += 0.5; // Partial credit
            }
        }
        
        // Check tangents
        if let Some(ref tangents) = self.tangents {
            if tangents.len() == vertex_count {
                score += 1.0;
            } else {
                score += 0.5; // Partial credit
            }
        }
        
        self.metadata.completeness_score = score / 4.0; // Normalize to 0-1 range
    }
    
    /// Convert back to base TriangleMesh (loses extended attributes)
    pub fn to_triangle_mesh(self) -> TriangleMesh {
        self.mesh
    }
}

impl MeshMetadata {
    /// Create metadata for a loaded mesh
    pub fn from_loaded(format: &str, _has_normals: bool, has_uvs: bool, _has_tangents: bool) -> Self {
        Self {
            normals_computed: false,
            tangents_computed: false,
            uvs_loaded: has_uvs,
            validation_messages: Vec::new(),
            source_format: Some(format.to_string()),
            completeness_score: 0.0, // Will be updated by mesh
        }
    }
    
    /// Check if mesh has complete attributes
    pub fn is_complete(&self) -> bool {
        self.completeness_score >= 0.75 // 75% completeness threshold
    }
    
    /// Get a summary of missing attributes
    pub fn missing_attributes(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        
        if !self.normals_computed && self.completeness_score < 0.5 {
            missing.push("normals");
        }
        if !self.uvs_loaded {
            missing.push("uvs");
        }
        if !self.tangents_computed {
            missing.push("tangents");
        }
        
        missing
    }
}

/// Utility functions for mesh attribute processing
pub mod utils {
    use super::*;
    
    /// Convert TriangleMesh to ExtendedTriangleMesh with attribute processing
    pub fn extend_mesh(
        mesh: TriangleMesh, 
        options: &MeshAttributeOptions
    ) -> Result<ExtendedTriangleMesh> {
        let mut extended = ExtendedTriangleMesh::from_mesh(mesh);
        extended.process_attributes(options)?;
        Ok(extended)
    }
    
    /// Ensure mesh has all basic attributes (normals at minimum)
    pub fn ensure_basic_attributes(mesh: &mut ExtendedTriangleMesh) -> Result<()> {
        let options = MeshAttributeOptions {
            recompute_normals: true,
            recompute_tangents: false,
            generate_default_uvs: false,
            validate_attributes: true,
            normalize_vectors: true,
            smooth_normals: true,
        };
        mesh.process_attributes(&options)
    }
    
    /// Prepare mesh for serialization with full attributes
    pub fn prepare_for_serialization(
        mesh: &mut ExtendedTriangleMesh,
        format: &str
    ) -> Result<()> {
        let options = match format.to_lowercase().as_str() {
            "obj" => MeshAttributeOptions {
                recompute_normals: true,
                recompute_tangents: false, // OBJ doesn't typically store tangents
                generate_default_uvs: true, // OBJ commonly has UVs
                validate_attributes: true,
                normalize_vectors: true,
                smooth_normals: true,
            },
            "ply" => MeshAttributeOptions {
                recompute_normals: true,
                recompute_tangents: false, // PLY can store custom attributes
                generate_default_uvs: false, // PLY is more flexible
                validate_attributes: true,
                normalize_vectors: true,
                smooth_normals: true,
            },
            _ => MeshAttributeOptions::default(),
        };
        
        mesh.process_attributes(&options)?;
        mesh.metadata.source_format = Some(format.to_string());
        Ok(())
    }
    
    /// Validate mesh for round-trip compatibility
    pub fn validate_round_trip(mesh: &ExtendedTriangleMesh) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        
        if mesh.vertex_count() == 0 {
            warnings.push("Empty mesh - no vertices".to_string());
        }
        
        if mesh.face_count() == 0 {
            warnings.push("Mesh has no faces".to_string());
        }
        
        if mesh.mesh.normals.is_none() {
            warnings.push("Missing normals - may be recomputed on load".to_string());
        }
        
        if mesh.uvs.is_none() {
            warnings.push("Missing UV coordinates - texture mapping not available".to_string());
        }
        
        if mesh.tangents.is_none() && mesh.uvs.is_some() {
            warnings.push("Missing tangents - normal mapping may not work correctly".to_string());
        }
        
        if !mesh.metadata.validation_messages.is_empty() {
            warnings.extend(mesh.metadata.validation_messages.iter().cloned());
        }
        
        Ok(warnings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_triangle() -> TriangleMesh {
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        TriangleMesh::from_vertices_and_faces(vertices, faces)
    }
    
    #[test]
    fn test_extended_mesh_creation() {
        let base_mesh = create_test_triangle();
        let extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        assert_eq!(extended.vertex_count(), 3);
        assert_eq!(extended.face_count(), 1);
        assert!(extended.uvs.is_none());
        assert!(extended.tangents.is_none());
    }
    
    #[test]
    fn test_normal_computation() {
        let base_mesh = create_test_triangle();
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        extended.compute_normals(true, true).unwrap();
        
        assert!(extended.mesh.normals.is_some());
        let normals = extended.mesh.normals.unwrap();
        assert_eq!(normals.len(), 3);
        
        // All normals should point in +Z direction for this triangle
        for normal in &normals {
            assert!((normal.z - 1.0).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_uv_generation() {
        let base_mesh = create_test_triangle();
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        extended.generate_default_uvs().unwrap();
        
        assert!(extended.uvs.is_some());
        let uvs = extended.uvs.unwrap();
        assert_eq!(uvs.len(), 3);
        
        // UVs should be in [0, 1] range
        for uv in &uvs {
            assert!(uv[0] >= 0.0 && uv[0] <= 1.0);
            assert!(uv[1] >= 0.0 && uv[1] <= 1.0);
        }
    }
    
    #[test]
    fn test_tangent_computation() {
        let base_mesh = create_test_triangle();
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Need normals and UVs for tangent computation
        extended.compute_normals(true, true).unwrap();
        extended.generate_default_uvs().unwrap();
        extended.compute_tangents(true).unwrap();
        
        assert!(extended.tangents.is_some());
        let tangents = extended.tangents.unwrap();
        assert_eq!(tangents.len(), 3);
        
        // Check tangent properties
        for tangent in &tangents {
            let length_sq = tangent.vector.x * tangent.vector.x + 
                tangent.vector.y * tangent.vector.y + 
                tangent.vector.z * tangent.vector.z;
            assert!((length_sq - 1.0).abs() < 1e-5); // Should be normalized
            assert!(tangent.handedness.abs() == 1.0); // Should be ±1
        }
    }
    
    #[test]
    fn test_attribute_validation() {
        let base_mesh = create_test_triangle();
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Should pass validation initially
        assert!(extended.validate_attributes().is_ok());
        
        // Add mismatched UVs
        extended.uvs = Some(vec![[0.0, 0.0]]); // Wrong count
        assert!(extended.validate_attributes().is_err());
    }
    
    #[test]
    fn test_process_attributes() {
        let base_mesh = create_test_triangle();
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        let options = MeshAttributeOptions::recompute_all();
        extended.process_attributes(&options).unwrap();
        
        assert!(extended.mesh.normals.is_some());
        assert!(extended.uvs.is_some());
        assert!(extended.tangents.is_some());
        assert!(extended.metadata.normals_computed);
        assert!(extended.metadata.tangents_computed);
    }
    
    #[test]
    fn test_metadata_completeness() {
        let base_mesh = create_test_triangle();
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Initially incomplete
        assert!(!extended.metadata.is_complete());
        
        // Add all attributes
        let options = MeshAttributeOptions::recompute_all();
        extended.process_attributes(&options).unwrap();
        
        // Should be complete now
        assert!(extended.metadata.is_complete());
        assert!(extended.metadata.completeness_score > 0.75);
    }
}
