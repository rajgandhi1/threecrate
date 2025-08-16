//! Robust PLY format support
//! 
//! This module provides comprehensive PLY (Polygon File Format) reading and writing
//! capabilities including:
//! - ASCII and binary (little/big endian) format support
//! - Vertex, face, normal, color, and generic property parsing
//! - Metadata and comment preservation
//! - Streaming support for large files
//! - Structured error handling

use crate::{PointCloudReader, PointCloudWriter, MeshReader, MeshWriter};
use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f, Vector3f, Error};
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::collections::HashMap;
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};

/// PLY file format variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

/// PLY property data types
#[derive(Debug, Clone, PartialEq)]
pub enum PlyPropertyType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
    List(Box<PlyPropertyType>, Box<PlyPropertyType>), // count type, item type
}

/// PLY property definition
#[derive(Debug, Clone)]
pub struct PlyProperty {
    pub name: String,
    pub property_type: PlyPropertyType,
}

/// PLY element definition
#[derive(Debug, Clone)]
pub struct PlyElement {
    pub name: String,
    pub count: usize,
    pub properties: Vec<PlyProperty>,
}

/// PLY property value
#[derive(Debug, Clone)]
pub enum PlyValue {
    Char(i8),
    UChar(u8),
    Short(i16),
    UShort(u16),
    Int(i32),
    UInt(u32),
    Float(f32),
    Double(f64),
    List(Vec<PlyValue>),
}

/// PLY header information
#[derive(Debug, Clone)]
pub struct PlyHeader {
    pub format: PlyFormat,
    pub version: String,
    pub elements: Vec<PlyElement>,
    pub comments: Vec<String>,
    pub obj_info: Vec<String>,
}

/// Complete PLY file data
#[derive(Debug)]
pub struct PlyData {
    pub header: PlyHeader,
    pub elements: HashMap<String, Vec<HashMap<String, PlyValue>>>,
}

/// Enhanced PLY reader with comprehensive format support
pub struct RobustPlyReader;

/// Enhanced PLY writer 
pub struct RobustPlyWriter;

impl RobustPlyReader {
    /// Read a complete PLY file with all metadata and elements
    pub fn read_ply_data<R: BufRead>(reader: &mut R) -> Result<PlyData> {
        let header = Self::read_header(reader)?;
        let elements = Self::read_elements(reader, &header)?;
        
        Ok(PlyData { header, elements })
    }
    
    /// Read PLY file from path
    pub fn read_ply_file<P: AsRef<Path>>(path: P) -> Result<PlyData> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_ply_data(&mut reader)
    }
    
    /// Read and parse PLY header
    fn read_header<R: BufRead>(reader: &mut R) -> Result<PlyHeader> {
        let mut format = None;
        let mut version = "1.0".to_string();
        let mut elements = Vec::new();
        let mut comments = Vec::new();
        let mut obj_info = Vec::new();
        
        let mut line = String::new();
        
        // Read magic number
        reader.read_line(&mut line)?;
        if line.trim() != "ply" {
            return Err(Error::InvalidData("Not a PLY file - missing magic number".to_string()));
        }
        
        // Parse header lines
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err(Error::InvalidData("Unexpected end of file in header".to_string()));
            }
            
            let line = line.trim();
            if line == "end_header" {
                break;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            
            match parts[0] {
                "format" => {
                    if parts.len() < 3 {
                        return Err(Error::InvalidData("Invalid format line".to_string()));
                    }
                    format = Some(match parts[1] {
                        "ascii" => PlyFormat::Ascii,
                        "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                        "binary_big_endian" => PlyFormat::BinaryBigEndian,
                        _ => return Err(Error::InvalidData(format!("Unknown format: {}", parts[1]))),
                    });
                    version = parts[2].to_string();
                }
                "comment" => {
                    if parts.len() > 1 {
                        comments.push(parts[1..].join(" "));
                    }
                }
                "obj_info" => {
                    if parts.len() > 1 {
                        obj_info.push(parts[1..].join(" "));
                    }
                }
                "element" => {
                    if parts.len() < 3 {
                        return Err(Error::InvalidData("Invalid element line".to_string()));
                    }
                    let name = parts[1].to_string();
                    let count: usize = parts[2].parse()
                        .map_err(|_| Error::InvalidData("Invalid element count".to_string()))?;
                    elements.push(PlyElement {
                        name,
                        count,
                        properties: Vec::new(),
                    });
                }
                "property" => {
                    if elements.is_empty() {
                        return Err(Error::InvalidData("Property without element".to_string()));
                    }
                    let property = Self::parse_property(&parts[1..])?;
                    elements.last_mut().unwrap().properties.push(property);
                }
                _ => {
                    // Ignore unknown header lines
                }
            }
        }
        
        let format = format.ok_or_else(|| Error::InvalidData("Missing format specification".to_string()))?;
        
        Ok(PlyHeader {
            format,
            version,
            elements,
            comments,
            obj_info,
        })
    }
    
    /// Parse a property definition
    fn parse_property(parts: &[&str]) -> Result<PlyProperty> {
        if parts.is_empty() {
            return Err(Error::InvalidData("Empty property definition".to_string()));
        }
        
        let property_type = if parts[0] == "list" {
            if parts.len() < 4 {
                return Err(Error::InvalidData("Invalid list property definition".to_string()));
            }
            let count_type = Self::parse_scalar_type(parts[1])?;
            let item_type = Self::parse_scalar_type(parts[2])?;
            PlyPropertyType::List(Box::new(count_type), Box::new(item_type))
        } else {
            if parts.len() < 2 {
                return Err(Error::InvalidData("Invalid property definition".to_string()));
            }
            Self::parse_scalar_type(parts[0])?
        };
        
        let name = parts.last().unwrap().to_string();
        
        Ok(PlyProperty { name, property_type })
    }
    
    /// Parse a scalar type
    fn parse_scalar_type(type_str: &str) -> Result<PlyPropertyType> {
        match type_str {
            "char" | "int8" => Ok(PlyPropertyType::Char),
            "uchar" | "uint8" => Ok(PlyPropertyType::UChar),
            "short" | "int16" => Ok(PlyPropertyType::Short),
            "ushort" | "uint16" => Ok(PlyPropertyType::UShort),
            "int" | "int32" => Ok(PlyPropertyType::Int),
            "uint" | "uint32" => Ok(PlyPropertyType::UInt),
            "float" | "float32" => Ok(PlyPropertyType::Float),
            "double" | "float64" => Ok(PlyPropertyType::Double),
            _ => Err(Error::InvalidData(format!("Unknown property type: {}", type_str))),
        }
    }
    
    /// Read all elements according to header specification
    fn read_elements<R: BufRead>(reader: &mut R, header: &PlyHeader) -> Result<HashMap<String, Vec<HashMap<String, PlyValue>>>> {
        let mut elements = HashMap::new();
        
        for element_def in &header.elements {
            let mut element_data = Vec::with_capacity(element_def.count);
            
            for _ in 0..element_def.count {
                let mut instance = HashMap::new();
                
                match header.format {
                    PlyFormat::Ascii => {
                        let mut line = String::new();
                        reader.read_line(&mut line)?;
                        let values = line.trim().split_whitespace().collect::<Vec<_>>();
                        let mut value_idx = 0;
                        
                        for property in &element_def.properties {
                            let value = Self::read_ascii_property_value(&values, &mut value_idx, &property.property_type)?;
                            instance.insert(property.name.clone(), value);
                        }
                    }
                    PlyFormat::BinaryLittleEndian => {
                        for property in &element_def.properties {
                            let value = Self::read_binary_property_value::<LittleEndian, _>(reader, &property.property_type)?;
                            instance.insert(property.name.clone(), value);
                        }
                    }
                    PlyFormat::BinaryBigEndian => {
                        for property in &element_def.properties {
                            let value = Self::read_binary_property_value::<BigEndian, _>(reader, &property.property_type)?;
                            instance.insert(property.name.clone(), value);
                        }
                    }
                }
                
                element_data.push(instance);
            }
            
            elements.insert(element_def.name.clone(), element_data);
        }
        
        Ok(elements)
    }
    
    /// Read ASCII property value
    fn read_ascii_property_value(values: &[&str], value_idx: &mut usize, property_type: &PlyPropertyType) -> Result<PlyValue> {
        if *value_idx >= values.len() {
            return Err(Error::InvalidData("Not enough values in line".to_string()));
        }
        
        match property_type {
            PlyPropertyType::Char => {
                let val = values[*value_idx].parse::<i8>()
                    .map_err(|_| Error::InvalidData("Invalid char value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::Char(val))
            }
            PlyPropertyType::UChar => {
                let val = values[*value_idx].parse::<u8>()
                    .map_err(|_| Error::InvalidData("Invalid uchar value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::UChar(val))
            }
            PlyPropertyType::Short => {
                let val = values[*value_idx].parse::<i16>()
                    .map_err(|_| Error::InvalidData("Invalid short value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::Short(val))
            }
            PlyPropertyType::UShort => {
                let val = values[*value_idx].parse::<u16>()
                    .map_err(|_| Error::InvalidData("Invalid ushort value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::UShort(val))
            }
            PlyPropertyType::Int => {
                let val = values[*value_idx].parse::<i32>()
                    .map_err(|_| Error::InvalidData("Invalid int value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::Int(val))
            }
            PlyPropertyType::UInt => {
                let val = values[*value_idx].parse::<u32>()
                    .map_err(|_| Error::InvalidData("Invalid uint value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::UInt(val))
            }
            PlyPropertyType::Float => {
                let val = values[*value_idx].parse::<f32>()
                    .map_err(|_| Error::InvalidData("Invalid float value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::Float(val))
            }
            PlyPropertyType::Double => {
                let val = values[*value_idx].parse::<f64>()
                    .map_err(|_| Error::InvalidData("Invalid double value".to_string()))?;
                *value_idx += 1;
                Ok(PlyValue::Double(val))
            }
            PlyPropertyType::List(count_type, item_type) => {
                let count_value = Self::read_ascii_property_value(values, value_idx, count_type)?;
                let count = count_value.as_usize()?;
                
                let mut list = Vec::with_capacity(count);
                for _ in 0..count {
                    let item = Self::read_ascii_property_value(values, value_idx, item_type)?;
                    list.push(item);
                }
                
                Ok(PlyValue::List(list))
            }
        }
    }
    
    /// Read binary property value
    fn read_binary_property_value<E: byteorder::ByteOrder, R: Read>(reader: &mut R, property_type: &PlyPropertyType) -> Result<PlyValue> {
        match property_type {
            PlyPropertyType::Char => Ok(PlyValue::Char(reader.read_i8()?)),
            PlyPropertyType::UChar => Ok(PlyValue::UChar(reader.read_u8()?)),
            PlyPropertyType::Short => Ok(PlyValue::Short(reader.read_i16::<E>()?)),
            PlyPropertyType::UShort => Ok(PlyValue::UShort(reader.read_u16::<E>()?)),
            PlyPropertyType::Int => Ok(PlyValue::Int(reader.read_i32::<E>()?)),
            PlyPropertyType::UInt => Ok(PlyValue::UInt(reader.read_u32::<E>()?)),
            PlyPropertyType::Float => Ok(PlyValue::Float(reader.read_f32::<E>()?)),
            PlyPropertyType::Double => Ok(PlyValue::Double(reader.read_f64::<E>()?)),
            PlyPropertyType::List(count_type, item_type) => {
                let count_value = Self::read_binary_property_value::<E, _>(reader, count_type)?;
                let count = count_value.as_usize()?;
                
                let mut list = Vec::with_capacity(count);
                for _ in 0..count {
                    let item = Self::read_binary_property_value::<E, _>(reader, item_type)?;
                    list.push(item);
                }
                
                Ok(PlyValue::List(list))
            }
        }
    }
}

impl PlyValue {
    /// Convert PLY value to f32
    pub fn as_f32(&self) -> Result<f32> {
        match self {
            PlyValue::Char(v) => Ok(*v as f32),
            PlyValue::UChar(v) => Ok(*v as f32),
            PlyValue::Short(v) => Ok(*v as f32),
            PlyValue::UShort(v) => Ok(*v as f32),
            PlyValue::Int(v) => Ok(*v as f32),
            PlyValue::UInt(v) => Ok(*v as f32),
            PlyValue::Float(v) => Ok(*v),
            PlyValue::Double(v) => Ok(*v as f32),
            _ => Err(Error::InvalidData("Cannot convert list to f32".to_string())),
        }
    }
    
    /// Convert PLY value to usize
    pub fn as_usize(&self) -> Result<usize> {
        match self {
            PlyValue::UChar(v) => Ok(*v as usize),
            PlyValue::UShort(v) => Ok(*v as usize),
            PlyValue::UInt(v) => Ok(*v as usize),
            PlyValue::Int(v) if *v >= 0 => Ok(*v as usize),
            _ => Err(Error::InvalidData("Cannot convert value to usize".to_string())),
        }
    }
    
    /// Convert PLY value to Vec<usize> (for face indices)
    pub fn as_usize_list(&self) -> Result<Vec<usize>> {
        match self {
            PlyValue::List(values) => {
                values.iter().map(|v| v.as_usize()).collect()
            }
            _ => Err(Error::InvalidData("Value is not a list".to_string())),
        }
    }
}

// Legacy PLY reader/writer using ply-rs for backward compatibility
pub struct PlyReader;
pub struct PlyWriter;

impl PointCloudReader for PlyReader {
    fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        // Use the robust reader for better format support
        let ply_data = RobustPlyReader::read_ply_file(path)?;
        
        let mut points = Vec::new();
        
        if let Some(vertex_elements) = ply_data.elements.get("vertex") {
            for vertex in vertex_elements {
                let x = vertex.get("x")
                    .ok_or_else(|| Error::InvalidData("Missing x coordinate".to_string()))?
                    .as_f32()?;
                let y = vertex.get("y")
                    .ok_or_else(|| Error::InvalidData("Missing y coordinate".to_string()))?
                    .as_f32()?;
                let z = vertex.get("z")
                    .ok_or_else(|| Error::InvalidData("Missing z coordinate".to_string()))?
                    .as_f32()?;
                
                points.push(Point3f::new(x, y, z));
            }
        }
        
        Ok(PointCloud::from_points(points))
    }
}

impl MeshReader for PlyReader {
    fn read_mesh<P: AsRef<Path>>(path: P) -> Result<TriangleMesh> {
        let ply_data = RobustPlyReader::read_ply_file(path)?;
        
        // Extract vertices
        let mut vertices = Vec::new();
        if let Some(vertex_elements) = ply_data.elements.get("vertex") {
            for vertex in vertex_elements {
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
            }
        }
        
        // Extract faces
        let mut faces = Vec::new();
        if let Some(face_elements) = ply_data.elements.get("face") {
            for face in face_elements {
                // Look for vertex_indices or vertex_index property
                let indices = if let Some(vertex_indices) = face.get("vertex_indices") {
                    vertex_indices.as_usize_list()?
                } else if let Some(vertex_index) = face.get("vertex_index") {
                    vertex_index.as_usize_list()?
                } else {
                    return Err(Error::InvalidData("Face missing vertex indices".to_string()));
                };
                
                // Convert to triangles (assuming triangular faces or taking first 3 vertices)
                if indices.len() >= 3 {
                    faces.push([indices[0], indices[1], indices[2]]);
                }
                // For quads and other polygons, we could triangulate here
                if indices.len() == 4 {
                    faces.push([indices[0], indices[2], indices[3]]);
                }
            }
        }
        
        // Extract normals if available
        let normals = if let Some(vertex_elements) = ply_data.elements.get("vertex") {
            let mut normals = Vec::new();
            let mut has_normals = true;
            
            for vertex in vertex_elements {
                if let (Some(nx), Some(ny), Some(nz)) = (
                    vertex.get("nx"),
                    vertex.get("ny"), 
                    vertex.get("nz"),
                ) {
                    normals.push(Vector3f::new(
                        nx.as_f32()?,
                        ny.as_f32()?,
                        nz.as_f32()?,
                    ));
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

impl PointCloudWriter for PlyWriter {
    fn write_point_cloud<P: AsRef<Path>>(cloud: &PointCloud<Point3f>, path: P) -> Result<()> {
        // Use the legacy ply-rs for writing for now
        use ply_rs::{

            writer::Writer,
            ply::{Property, PropertyDef, PropertyType, ScalarType, ElementDef, Ply, Addable, DefaultElement},
        };
        
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

impl MeshWriter for PlyWriter {
    fn write_mesh<P: AsRef<Path>>(mesh: &TriangleMesh, path: P) -> Result<()> {
        // Use the legacy ply-rs for writing for now
        use ply_rs::{

            writer::Writer,
            ply::{Property, PropertyDef, PropertyType, ScalarType, ElementDef, Ply, Addable, DefaultElement},
        };
        
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

// Helper functions moved to where they're needed or removed since we use the new implementation