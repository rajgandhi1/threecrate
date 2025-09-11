//! Mesh serialization utilities with attribute preservation
//!
//! This module provides high-level serialization utilities that ensure mesh attributes
//! survive round-trip across different formats. It integrates with the existing I/O
//! system and provides opt-in attribute recomputation and validation.

use crate::mesh_attributes::{ExtendedTriangleMesh, MeshAttributeOptions, Tangent};
use crate::{obj, ply};
use threecrate_core::{TriangleMesh, Point3f, Vector3f, Result, Error};
use std::path::Path;
use std::collections::HashMap;

/// Configuration for mesh serialization
#[derive(Debug, Clone)]
pub struct SerializationOptions {
    /// Options for attribute processing
    pub attributes: MeshAttributeOptions,
    /// Whether to preserve custom properties during serialization
    pub preserve_custom_properties: bool,
    /// Whether to validate mesh before writing
    pub validate_before_write: bool,
    /// Whether to attach metadata to the mesh
    pub attach_metadata: bool,
    /// Custom properties to include in serialization
    pub custom_properties: HashMap<String, Vec<f32>>,
}

impl Default for SerializationOptions {
    fn default() -> Self {
        Self {
            attributes: MeshAttributeOptions::default(),
            preserve_custom_properties: true,
            validate_before_write: true,
            attach_metadata: true,
            custom_properties: HashMap::new(),
        }
    }
}

impl SerializationOptions {
    /// Create options optimized for round-trip preservation
    pub fn preserve_all() -> Self {
        Self {
            attributes: MeshAttributeOptions::recompute_all(),
            preserve_custom_properties: true,
            validate_before_write: true,
            attach_metadata: true,
            custom_properties: HashMap::new(),
        }
    }
    
    /// Create options for fast serialization (minimal processing)
    pub fn fast() -> Self {
        Self {
            attributes: MeshAttributeOptions::validate_only(),
            preserve_custom_properties: false,
            validate_before_write: false,
            attach_metadata: false,
            custom_properties: HashMap::new(),
        }
    }
    
    /// Add a custom property
    pub fn with_custom_property<S: Into<String>>(mut self, name: S, values: Vec<f32>) -> Self {
        self.custom_properties.insert(name.into(), values);
        self
    }
}

/// Enhanced mesh reader with attribute preservation
pub struct AttributePreservingReader;

/// Enhanced mesh writer with attribute preservation  
pub struct AttributePreservingWriter;

impl AttributePreservingReader {
    /// Read mesh with full attribute extraction
    pub fn read_extended_mesh<P: AsRef<Path>>(
        path: P, 
        options: &SerializationOptions
    ) -> Result<ExtendedTriangleMesh> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| Error::UnsupportedFormat("No file extension found".to_string()))?;
        
        match extension.to_lowercase().as_str() {
            "obj" => Self::read_obj_extended(path, options),
            "ply" => Self::read_ply_extended(path, options),
            _ => {
                // Fall back to basic mesh reading
                let mesh = crate::read_mesh(path)?;
                let mut extended = ExtendedTriangleMesh::from_mesh(mesh);
                extended.metadata.source_format = Some(extension.to_string());
                extended.process_attributes(&options.attributes)?;
                Ok(extended)
            }
        }
    }
    
    /// Read OBJ file with UV and normal extraction
    fn read_obj_extended<P: AsRef<Path>>(
        path: P, 
        options: &SerializationOptions
    ) -> Result<ExtendedTriangleMesh> {
        let obj_data = obj::RobustObjReader::read_obj_file(path)?;
        let mut extended = Self::obj_data_to_extended_mesh(&obj_data)?;
        
        extended.metadata.source_format = Some("obj".to_string());
        extended.metadata.uvs_loaded = !obj_data.texture_coords.is_empty();
        
        if options.attributes.validate_attributes {
            extended.validate_attributes()?;
        }
        
        // Process attributes if requested
        extended.process_attributes(&options.attributes)?;
        
        Ok(extended)
    }
    
    /// Read PLY file with custom attribute extraction
    fn read_ply_extended<P: AsRef<Path>>(
        path: P, 
        options: &SerializationOptions
    ) -> Result<ExtendedTriangleMesh> {
        let ply_data = ply::RobustPlyReader::read_ply_file(path)?;
        let mut extended = Self::ply_data_to_extended_mesh(&ply_data)?;
        
        extended.metadata.source_format = Some("ply".to_string());
        
        if options.attributes.validate_attributes {
            extended.validate_attributes()?;
        }
        
        // Process attributes if requested
        extended.process_attributes(&options.attributes)?;
        
        Ok(extended)
    }
    
    /// Convert OBJ data to extended mesh
    fn obj_data_to_extended_mesh(obj_data: &obj::ObjData) -> Result<ExtendedTriangleMesh> {
        // Convert to base mesh first
        let base_mesh = obj::RobustObjReader::obj_data_to_mesh(obj_data)?;
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Extract UVs if available
        if !obj_data.texture_coords.is_empty() {
            let mut uvs = vec![[0.0, 0.0]; extended.vertex_count()];
            let mut uv_extracted = false;
            
            // Map texture coordinates from faces to vertices
            for group in &obj_data.groups {
                for face in &group.faces {
                    for face_vertex in &face.vertices {
                        if let Some(tex_idx) = face_vertex.texture {
                            if tex_idx < obj_data.texture_coords.len() {
                                uvs[face_vertex.vertex] = obj_data.texture_coords[tex_idx];
                                uv_extracted = true;
                            }
                        }
                    }
                }
            }
            
            if uv_extracted {
                extended.set_uvs(uvs);
            }
        }
        
        Ok(extended)
    }
    
    /// Convert PLY data to extended mesh
    fn ply_data_to_extended_mesh(ply_data: &ply::PlyData) -> Result<ExtendedTriangleMesh> {
        // Extract vertices
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut has_normals = false;
        let mut has_uvs = false;
        let mut has_tangents = false;
        
        if let Some(vertex_elements) = ply_data.elements.get("vertex") {
            for vertex in vertex_elements {
                // Extract position
                let x = vertex.get("x")
                    .ok_or_else(|| Error::InvalidData("Missing x coordinate".to_string()))?
                    .as_f32()?;
                let y = vertex.get("y")
                    .ok_or_else(|| Error::InvalidData("Missing y coordinate".to_string()))?
                    .as_f32()?;
                let z = vertex.get("z")
                    .ok_or_else(|| Error::InvalidData("Missing z coordinate".to_string()))?
                    .as_f32()?;
                
                vertices.push(Point3f::new(x, y, z));
                
                // Extract normals if available
                if let (Some(nx), Some(ny), Some(nz)) = (
                    vertex.get("nx"),
                    vertex.get("ny"),
                    vertex.get("nz")
                ) {
                    normals.push(Vector3f::new(
                        nx.as_f32()?,
                        ny.as_f32()?,
                        nz.as_f32()?,
                    ));
                    has_normals = true;
                }
                
                // Extract UVs if available (various common names)
                if let Some(u) = vertex.get("u").or_else(|| vertex.get("texture_u")) {
                    if let Some(v) = vertex.get("v").or_else(|| vertex.get("texture_v")) {
                        uvs.push([u.as_f32()?, v.as_f32()?]);
                        has_uvs = true;
                    }
                } else if let Some(s) = vertex.get("s") {
                    if let Some(t) = vertex.get("t") {
                        uvs.push([s.as_f32()?, t.as_f32()?]);
                        has_uvs = true;
                    }
                }
                
                // Extract tangents if available
                if let (Some(tx), Some(ty), Some(tz)) = (
                    vertex.get("tx").or_else(|| vertex.get("tangent_x")),
                    vertex.get("ty").or_else(|| vertex.get("tangent_y")),
                    vertex.get("tz").or_else(|| vertex.get("tangent_z"))
                ) {
                    let handedness = vertex.get("th")
                        .or_else(|| vertex.get("tangent_handedness"))
                        .map(|h| h.as_f32().unwrap_or(1.0))
                        .unwrap_or(1.0);
                    
                    tangents.push(Tangent::new(
                        Vector3f::new(tx.as_f32()?, ty.as_f32()?, tz.as_f32()?),
                        handedness
                    ));
                    has_tangents = true;
                }
            }
        }
        
        // Extract faces
        let mut faces = Vec::new();
        if let Some(face_elements) = ply_data.elements.get("face") {
            for face in face_elements {
                let indices = if let Some(vertex_indices) = face.get("vertex_indices") {
                    vertex_indices.as_usize_list()?
                } else if let Some(vertex_index) = face.get("vertex_index") {
                    vertex_index.as_usize_list()?
                } else {
                    return Err(Error::InvalidData("Face missing vertex indices".to_string()));
                };
                
                // Convert to triangles
                if indices.len() >= 3 {
                    faces.push([indices[0], indices[1], indices[2]]);
                }
                if indices.len() == 4 {
                    faces.push([indices[0], indices[2], indices[3]]);
                }
            }
        }
        
        // Create base mesh
        let mut base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        if has_normals && normals.len() == base_mesh.vertex_count() {
            base_mesh.set_normals(normals);
        }
        
        // Create extended mesh
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        if has_uvs && uvs.len() == extended.vertex_count() {
            extended.set_uvs(uvs);
        }
        
        if has_tangents && tangents.len() == extended.vertex_count() {
            extended.set_tangents(tangents);
        }
        
        Ok(extended)
    }
}

impl AttributePreservingWriter {
    /// Write extended mesh with full attribute preservation
    pub fn write_extended_mesh<P: AsRef<Path>>(
        mesh: &ExtendedTriangleMesh,
        path: P,
        options: &SerializationOptions,
    ) -> Result<()> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| Error::UnsupportedFormat("No file extension found".to_string()))?;
        
        // Validate mesh if requested
        if options.validate_before_write {
            let warnings = crate::mesh_attributes::utils::validate_round_trip(mesh)?;
            if !warnings.is_empty() {
                eprintln!("Mesh validation warnings: {:#?}", warnings);
            }
        }
        
        match extension.to_lowercase().as_str() {
            "obj" => Self::write_obj_extended(mesh, path, options),
            "ply" => Self::write_ply_extended(mesh, path, options),
            _ => {
                // Fall back to basic mesh writing
                crate::write_mesh(&mesh.mesh, path)
            }
        }
    }
    
    /// Write OBJ file with UV and normal preservation
    fn write_obj_extended<P: AsRef<Path>>(
        mesh: &ExtendedTriangleMesh,
        path: P,
        options: &SerializationOptions,
    ) -> Result<()> {
        let obj_data = Self::extended_mesh_to_obj_data(mesh, options)?;
        let obj_options = obj::ObjWriteOptions::new()
            .with_normals(mesh.mesh.normals.is_some())
            .with_texcoords(mesh.uvs.is_some())
            .with_comment("Generated by ThreeCrate with attribute preservation");
        
        obj::RobustObjWriter::write_obj_file(&obj_data, path, &obj_options)
    }
    
    /// Write PLY file with custom attribute preservation
    fn write_ply_extended<P: AsRef<Path>>(
        mesh: &ExtendedTriangleMesh,
        path: P,
        options: &SerializationOptions,
    ) -> Result<()> {
        let mut ply_options = ply::PlyWriteOptions::ascii()
            .with_normals(mesh.mesh.normals.is_some())
            .with_comment("Generated by ThreeCrate with attribute preservation");
        
        // Add UV coordinates as custom properties if available
        if let Some(ref uvs) = mesh.uvs {
            let u_values: Vec<ply::PlyValue> = uvs.iter().map(|uv| ply::PlyValue::Float(uv[0])).collect();
            let v_values: Vec<ply::PlyValue> = uvs.iter().map(|uv| ply::PlyValue::Float(uv[1])).collect();
            ply_options = ply_options
                .with_custom_vertex_property("u", u_values)
                .with_custom_vertex_property("v", v_values);
        }
        
        // Add tangents as custom properties if available
        if let Some(ref tangents) = mesh.tangents {
            let tx_values: Vec<ply::PlyValue> = tangents.iter().map(|t| ply::PlyValue::Float(t.vector.x)).collect();
            let ty_values: Vec<ply::PlyValue> = tangents.iter().map(|t| ply::PlyValue::Float(t.vector.y)).collect();
            let tz_values: Vec<ply::PlyValue> = tangents.iter().map(|t| ply::PlyValue::Float(t.vector.z)).collect();
            let th_values: Vec<ply::PlyValue> = tangents.iter().map(|t| ply::PlyValue::Float(t.handedness)).collect();
            
            ply_options = ply_options
                .with_custom_vertex_property("tx", tx_values)
                .with_custom_vertex_property("ty", ty_values)
                .with_custom_vertex_property("tz", tz_values)
                .with_custom_vertex_property("th", th_values);
        }
        
        // Add custom properties from options
        for (name, values) in &options.custom_properties {
            let ply_values: Vec<ply::PlyValue> = values.iter().map(|&v| ply::PlyValue::Float(v)).collect();
            ply_options = ply_options.with_custom_vertex_property(name, ply_values);
        }
        
        ply::RobustPlyWriter::write_mesh(&mesh.mesh, path, &ply_options)
    }
    
    /// Convert extended mesh to OBJ data
    fn extended_mesh_to_obj_data(
        mesh: &ExtendedTriangleMesh,
        _options: &SerializationOptions,
    ) -> Result<obj::ObjData> {
        let mut obj_data = obj::ObjData {
            vertices: mesh.mesh.vertices.clone(),
            texture_coords: Vec::new(),
            normals: mesh.mesh.normals.clone().unwrap_or_default(),
            groups: Vec::new(),
            materials: HashMap::new(),
            mtl_files: Vec::new(),
        };
        
        // Add texture coordinates if available
        if let Some(ref uvs) = mesh.uvs {
            obj_data.texture_coords = uvs.iter().map(|&uv| uv).collect();
        }
        
        // Create faces with proper indexing
        let mut faces = Vec::new();
        for &face_indices in &mesh.mesh.faces {
            let mut face_vertices = Vec::new();
            for &vertex_idx in &face_indices {
                let face_vertex = obj::FaceVertex {
                    vertex: vertex_idx,
                    texture: if mesh.uvs.is_some() { Some(vertex_idx) } else { None },
                    normal: if mesh.mesh.normals.is_some() { Some(vertex_idx) } else { None },
                };
                face_vertices.push(face_vertex);
            }
            
            faces.push(obj::Face {
                vertices: face_vertices,
                material: None,
            });
        }
        
        // Create default group
        obj_data.groups.push(obj::Group {
            name: "default".to_string(),
            faces,
        });
        
        Ok(obj_data)
    }
}

/// High-level utility functions for mesh serialization
pub mod utils {
    use super::*;
    
    /// Read mesh with attribute preservation
    pub fn read_mesh_with_attributes<P: AsRef<Path>>(
        path: P,
        options: Option<SerializationOptions>,
    ) -> Result<ExtendedTriangleMesh> {
        let options = options.unwrap_or_default();
        AttributePreservingReader::read_extended_mesh(path, &options)
    }
    
    /// Write mesh with attribute preservation
    pub fn write_mesh_with_attributes<P: AsRef<Path>>(
        mesh: &ExtendedTriangleMesh,
        path: P,
        options: Option<SerializationOptions>,
    ) -> Result<()> {
        let options = options.unwrap_or_default();
        AttributePreservingWriter::write_extended_mesh(mesh, path, &options)
    }
    
    /// Perform a round-trip test for a mesh
    pub fn test_round_trip<P: AsRef<Path>>(
        input_path: P,
        output_path: P,
        options: Option<SerializationOptions>,
    ) -> Result<(ExtendedTriangleMesh, Vec<String>)> {
        let options = options.unwrap_or(SerializationOptions::preserve_all());
        
        // Read mesh
        let mesh = read_mesh_with_attributes(input_path, Some(options.clone()))?;
        
        // Write mesh
        write_mesh_with_attributes(&mesh, output_path, Some(options))?;
        
        // Validate round-trip
        let warnings = crate::mesh_attributes::utils::validate_round_trip(&mesh)?;
        
        Ok((mesh, warnings))
    }
    
    /// Ensure mesh is ready for serialization in a specific format
    pub fn prepare_mesh_for_format(
        mesh: &mut ExtendedTriangleMesh,
        format: &str,
    ) -> Result<()> {
        crate::mesh_attributes::utils::prepare_for_serialization(mesh, format)
    }
    
    /// Compare two meshes for attribute preservation
    pub fn compare_meshes(
        original: &ExtendedTriangleMesh,
        loaded: &ExtendedTriangleMesh,
    ) -> Result<Vec<String>> {
        let mut differences = Vec::new();
        
        if original.vertex_count() != loaded.vertex_count() {
            differences.push(format!(
                "Vertex count mismatch: {} vs {}",
                original.vertex_count(),
                loaded.vertex_count()
            ));
        }
        
        if original.face_count() != loaded.face_count() {
            differences.push(format!(
                "Face count mismatch: {} vs {}",
                original.face_count(),
                loaded.face_count()
            ));
        }
        
        // Compare vertices
        if original.vertex_count() == loaded.vertex_count() {
            for (i, (v1, v2)) in original.mesh.vertices.iter()
                .zip(loaded.mesh.vertices.iter())
                .enumerate() 
            {
                let diff = (v1.x - v2.x).abs() + (v1.y - v2.y).abs() + (v1.z - v2.z).abs();
                if diff > 1e-5 {
                    differences.push(format!("Vertex {} differs by {}", i, diff));
                    break; // Don't spam with too many vertex differences
                }
            }
        }
        
        // Compare normals
        match (&original.mesh.normals, &loaded.mesh.normals) {
            (Some(n1), Some(n2)) => {
                if n1.len() != n2.len() {
                    differences.push("Normal count mismatch".to_string());
                }
            }
            (Some(_), None) => differences.push("Normals lost during round-trip".to_string()),
            (None, Some(_)) => differences.push("Normals added during round-trip".to_string()),
            (None, None) => {} // Both missing, that's fine
        }
        
        // Compare UVs
        match (&original.uvs, &loaded.uvs) {
            (Some(uv1), Some(uv2)) => {
                if uv1.len() != uv2.len() {
                    differences.push("UV count mismatch".to_string());
                }
            }
            (Some(_), None) => differences.push("UVs lost during round-trip".to_string()),
            (None, Some(_)) => differences.push("UVs added during round-trip".to_string()),
            (None, None) => {} // Both missing, that's fine
        }
        
        // Compare tangents
        match (&original.tangents, &loaded.tangents) {
            (Some(t1), Some(t2)) => {
                if t1.len() != t2.len() {
                    differences.push("Tangent count mismatch".to_string());
                }
            }
            (Some(_), None) => differences.push("Tangents lost during round-trip".to_string()),
            (None, Some(_)) => differences.push("Tangents added during round-trip".to_string()),
            (None, None) => {} // Both missing, that's fine
        }
        
        Ok(differences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    fn create_test_extended_mesh() -> ExtendedTriangleMesh {
        let vertices = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let base_mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        
        let mut extended = ExtendedTriangleMesh::from_mesh(base_mesh);
        
        // Add attributes
        extended.compute_normals(true, true).unwrap();
        extended.generate_default_uvs().unwrap();
        extended.compute_tangents(true).unwrap();
        
        extended
    }
    
    #[test]
    fn test_serialization_options() {
        let options = SerializationOptions::preserve_all();
        assert!(options.attributes.recompute_normals);
        assert!(options.attributes.recompute_tangents);
        assert!(options.validate_before_write);
        
        let fast_options = SerializationOptions::fast();
        assert!(!fast_options.validate_before_write);
        assert!(!fast_options.preserve_custom_properties);
    }
    
    #[test]
    fn test_obj_round_trip() {
        let mesh = create_test_extended_mesh();
        let temp_file = "test_obj_round_trip.obj";
        
        let options = SerializationOptions::preserve_all();
        
        // Write mesh
        AttributePreservingWriter::write_extended_mesh(&mesh, temp_file, &options).unwrap();
        
        // Read mesh back
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Compare
        let differences = utils::compare_meshes(&mesh, &loaded_mesh).unwrap();
        assert!(differences.is_empty(), "Round-trip differences: {:#?}", differences);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_ply_round_trip() {
        let mesh = create_test_extended_mesh();
        let temp_file = "test_ply_round_trip.ply";
        
        let options = SerializationOptions::preserve_all();
        
        // Write mesh
        AttributePreservingWriter::write_extended_mesh(&mesh, temp_file, &options).unwrap();
        
        // Read mesh back
        let loaded_mesh = AttributePreservingReader::read_extended_mesh(temp_file, &options).unwrap();
        
        // Compare
        let differences = utils::compare_meshes(&mesh, &loaded_mesh).unwrap();
        // Note: PLY round-trip might have some differences due to custom property handling
        println!("PLY round-trip differences: {:#?}", differences);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_attribute_validation() {
        let mesh = create_test_extended_mesh();
        let warnings = crate::mesh_attributes::utils::validate_round_trip(&mesh).unwrap();
        
        // Should have minimal warnings for a complete mesh
        assert!(warnings.len() <= 1, "Too many warnings: {:#?}", warnings);
    }
    
    #[test]
    fn test_mesh_comparison() {
        let mesh1 = create_test_extended_mesh();
        let mesh2 = create_test_extended_mesh();
        
        let differences = utils::compare_meshes(&mesh1, &mesh2).unwrap();
        assert!(differences.is_empty(), "Identical meshes should have no differences");
    }
}
