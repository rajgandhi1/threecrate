//! PLY format support

use crate::{PointCloudReader, PointCloudWriter, MeshReader, MeshWriter};
use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, Vector3f};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use ply_rs::{
    parser::Parser,
    writer::Writer,
    ply::{Property, PropertyDef, PropertyType, ScalarType, ElementDef, Ply, Addable, DefaultElement},
};

pub struct PlyReader;
pub struct PlyWriter;

impl PointCloudReader for PlyReader {
    fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Parse PLY header
        let parser = Parser::<DefaultElement>::new();
        let ply = parser.read_ply(&mut reader)?;
        
        // Extract vertex elements
        let mut points = Vec::new();
        
        if let Some(vertex_element) = ply.payload.get("vertex") {
            for vertex in vertex_element {
                let x = extract_property_value(vertex, "x")?;
                let y = extract_property_value(vertex, "y")?;
                let z = extract_property_value(vertex, "z")?;
                
                points.push(Point3f::new(x, y, z));
            }
        }
        
        Ok(PointCloud::from_points(points))
    }
}

impl PointCloudWriter for PlyWriter {
    fn write_point_cloud<P: AsRef<Path>>(cloud: &PointCloud<Point3f>, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Create PLY structure
        let mut ply = Ply::<DefaultElement>::new();
        
        // Define vertex element
        let mut vertex_element = ElementDef::new("vertex".to_string());
        vertex_element.count = cloud.len();
        vertex_element.properties.add(PropertyDef::new(
            "x".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_element.properties.add(PropertyDef::new(
            "y".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_element.properties.add(PropertyDef::new(
            "z".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        
        ply.header.elements.add(vertex_element);
        
        // Add vertex data
        let mut vertices = Vec::new();
        for point in &cloud.points {
            let mut vertex = DefaultElement::new();
            vertex.insert("x".to_string(), Property::Float(point.x));
            vertex.insert("y".to_string(), Property::Float(point.y));
            vertex.insert("z".to_string(), Property::Float(point.z));
            vertices.push(vertex);
        }
        ply.payload.insert("vertex".to_string(), vertices);
        
        // Write PLY file
        let writer_instance = Writer::new();
        writer_instance.write_ply(&mut writer, &mut ply)?;
        
        Ok(())
    }
}

impl MeshReader for PlyReader {
    fn read_mesh<P: AsRef<Path>>(path: P) -> Result<TriangleMesh> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Parse PLY header
        let parser = Parser::<DefaultElement>::new();
        let ply = parser.read_ply(&mut reader)?;
        
        // Extract vertices
        let mut vertices = Vec::new();
        if let Some(vertex_element) = ply.payload.get("vertex") {
            for vertex in vertex_element {
                let x = extract_property_value(vertex, "x")?;
                let y = extract_property_value(vertex, "y")?;
                let z = extract_property_value(vertex, "z")?;
                
                vertices.push(Point3f::new(x, y, z));
            }
        }
        
        // Extract faces
        let mut faces = Vec::new();
        if let Some(face_element) = ply.payload.get("face") {
            for face in face_element {
                let indices = extract_face_indices(face)?;
                if indices.len() >= 3 {
                    faces.push([indices[0], indices[1], indices[2]]);
                }
            }
        }
        
        // Extract normals if available
        let normals = if let Some(vertex_element) = ply.payload.get("vertex") {
            let mut normals = Vec::new();
            let mut has_normals = true;
            
            for vertex in vertex_element {
                if let (Ok(nx), Ok(ny), Ok(nz)) = (
                    extract_property_value(vertex, "nx"),
                    extract_property_value(vertex, "ny"),
                    extract_property_value(vertex, "nz"),
                ) {
                    normals.push(Vector3f::new(nx, ny, nz));
                } else {
                    has_normals = false;
                    break;
                }
            }
            
            if has_normals && !normals.is_empty() {
                Some(normals)
            } else {
                None
            }
        } else {
            None
        };
        
        let mut mesh = TriangleMesh::from_vertices_and_faces(vertices, faces);
        if let Some(normals) = normals {
            mesh.set_normals(normals);
        }
        
        Ok(mesh)
    }
}

impl MeshWriter for PlyWriter {
    fn write_mesh<P: AsRef<Path>>(mesh: &TriangleMesh, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Create PLY structure
        let mut ply = Ply::<DefaultElement>::new();
        
        // Define vertex element
        let mut vertex_element = ElementDef::new("vertex".to_string());
        vertex_element.count = mesh.vertices.len();
        vertex_element.properties.add(PropertyDef::new(
            "x".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_element.properties.add(PropertyDef::new(
            "y".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_element.properties.add(PropertyDef::new(
            "z".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        
        // Add normal properties if available
        if mesh.normals.is_some() {
            vertex_element.properties.add(PropertyDef::new(
                "nx".to_string(),
                PropertyType::Scalar(ScalarType::Float),
            ));
            vertex_element.properties.add(PropertyDef::new(
                "ny".to_string(),
                PropertyType::Scalar(ScalarType::Float),
            ));
            vertex_element.properties.add(PropertyDef::new(
                "nz".to_string(),
                PropertyType::Scalar(ScalarType::Float),
            ));
        }
        
        ply.header.elements.add(vertex_element);
        
        // Define face element
        let mut face_element = ElementDef::new("face".to_string());
        face_element.count = mesh.faces.len();
        face_element.properties.add(PropertyDef::new(
            "vertex_indices".to_string(),
            PropertyType::List(ScalarType::UChar, ScalarType::Int),
        ));
        
        ply.header.elements.add(face_element);
        
        // Add vertex data
        let mut vertices = Vec::new();
        for (i, vertex) in mesh.vertices.iter().enumerate() {
            let mut vertex_element = DefaultElement::new();
            vertex_element.insert("x".to_string(), Property::Float(vertex.x));
            vertex_element.insert("y".to_string(), Property::Float(vertex.y));
            vertex_element.insert("z".to_string(), Property::Float(vertex.z));
            
            // Add normals if available
            if let Some(normals) = &mesh.normals {
                if i < normals.len() {
                    vertex_element.insert("nx".to_string(), Property::Float(normals[i].x));
                    vertex_element.insert("ny".to_string(), Property::Float(normals[i].y));
                    vertex_element.insert("nz".to_string(), Property::Float(normals[i].z));
                }
            }
            
            vertices.push(vertex_element);
        }
        ply.payload.insert("vertex".to_string(), vertices);
        
        // Add face data
        let mut faces = Vec::new();
        for face in &mesh.faces {
            let mut face_element = DefaultElement::new();
            let indices = vec![
                face[0] as i32,
                face[1] as i32,
                face[2] as i32,
            ];
            face_element.insert("vertex_indices".to_string(), Property::ListInt(indices));
            faces.push(face_element);
        }
        ply.payload.insert("face".to_string(), faces);
        
        // Write PLY file
        let writer_instance = Writer::new();
        writer_instance.write_ply(&mut writer, &mut ply)?;
        
        Ok(())
    }
}

/// Extract a property value as f32 from a PLY element
fn extract_property_value(element: &DefaultElement, name: &str) -> Result<f32> {
    match element.get(name) {
        Some(Property::Float(val)) => Ok(*val),
        Some(Property::Double(val)) => Ok(*val as f32),
        Some(Property::Int(val)) => Ok(*val as f32),
        Some(Property::UInt(val)) => Ok(*val as f32),
        _ => Err(threecrate_core::Error::InvalidData(
            format!("Property '{}' not found or invalid type", name)
        )),
    }
}

/// Extract face indices from a PLY face element
fn extract_face_indices(element: &DefaultElement) -> Result<Vec<usize>> {
    match element.get("vertex_indices").or_else(|| element.get("vertex_index")) {
        Some(Property::ListInt(indices)) => {
            Ok(indices.iter().map(|&idx| idx as usize).collect())
        }
        Some(Property::ListUInt(indices)) => {
            Ok(indices.iter().map(|&idx| idx as usize).collect())
        }
        _ => Err(threecrate_core::Error::InvalidData(
            "Face indices not found".to_string()
        )),
    }
}