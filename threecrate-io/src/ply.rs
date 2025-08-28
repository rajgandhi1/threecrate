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

/// PLY writer configuration options
#[derive(Debug, Clone)]
pub struct PlyWriteOptions {
    /// Output format (ASCII or binary)
    pub format: PlyFormat,
    /// Comments to include in the header
    pub comments: Vec<String>,
    /// Object info to include in the header
    pub obj_info: Vec<String>,
    /// Custom properties to include for vertices
    pub custom_vertex_properties: Vec<(String, Vec<PlyValue>)>,
    /// Custom properties to include for faces
    pub custom_face_properties: Vec<(String, Vec<PlyValue>)>,
    /// Whether to include normals if available
    pub include_normals: bool,
    /// Whether to include colors if available
    pub include_colors: bool,
    /// Custom property ordering for vertices
    pub vertex_property_order: Option<Vec<String>>,
}

impl Default for PlyWriteOptions {
    fn default() -> Self {
        Self {
            format: PlyFormat::Ascii,
            comments: Vec::new(),
            obj_info: Vec::new(),
            custom_vertex_properties: Vec::new(),
            custom_face_properties: Vec::new(),
            include_normals: true,
            include_colors: false,
            vertex_property_order: None,
        }
    }
}

impl PlyWriteOptions {
    /// Create new options with ASCII format
    pub fn ascii() -> Self {
        Self {
            format: PlyFormat::Ascii,
            ..Default::default()
        }
    }
    
    /// Create new options with binary little endian format
    pub fn binary_little_endian() -> Self {
        Self {
            format: PlyFormat::BinaryLittleEndian,
            ..Default::default()
        }
    }
    
    /// Create new options with binary big endian format
    pub fn binary_big_endian() -> Self {
        Self {
            format: PlyFormat::BinaryBigEndian,
            ..Default::default()
        }
    }
    
    /// Add a comment to the header
    pub fn with_comment<S: Into<String>>(mut self, comment: S) -> Self {
        self.comments.push(comment.into());
        self
    }
    
    /// Add object info to the header
    pub fn with_obj_info<S: Into<String>>(mut self, info: S) -> Self {
        self.obj_info.push(info.into());
        self
    }
    
    /// Include normals in output
    pub fn with_normals(mut self, include: bool) -> Self {
        self.include_normals = include;
        self
    }
    
    /// Include colors in output
    pub fn with_colors(mut self, include: bool) -> Self {
        self.include_colors = include;
        self
    }
    
    /// Set custom vertex property ordering
    pub fn with_vertex_property_order(mut self, order: Vec<String>) -> Self {
        self.vertex_property_order = Some(order);
        self
    }
    
    /// Add custom vertex property
    pub fn with_custom_vertex_property<S: Into<String>>(mut self, name: S, values: Vec<PlyValue>) -> Self {
        self.custom_vertex_properties.push((name.into(), values));
        self
    }
}

/// Enhanced PLY writer with comprehensive format support
pub struct RobustPlyWriter;

impl RobustPlyWriter {
    /// Write point cloud to PLY file with options
    pub fn write_point_cloud<P: AsRef<Path>>(
        cloud: &PointCloud<Point3f>, 
        path: P, 
        options: &PlyWriteOptions
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        Self::write_point_cloud_to_writer(cloud, &mut writer, options)
    }
    
    /// Write point cloud to writer with options
    pub fn write_point_cloud_to_writer<W: std::io::Write>(
        cloud: &PointCloud<Point3f>,
        writer: &mut W,
        options: &PlyWriteOptions,
    ) -> Result<()> {
        // Build PLY data structure
        let mut ply_data = PlyData {
            header: PlyHeader {
                format: options.format,
                version: "1.0".to_string(),
                elements: Vec::new(),
                comments: options.comments.clone(),
                obj_info: options.obj_info.clone(),
            },
            elements: HashMap::new(),
        };
        
        // Build vertex element definition
        let mut vertex_properties = Vec::new();
        let mut vertex_data = Vec::new();
        
        // Always include x, y, z
        vertex_properties.push(PlyProperty {
            name: "x".to_string(),
            property_type: PlyPropertyType::Float,
        });
        vertex_properties.push(PlyProperty {
            name: "y".to_string(),
            property_type: PlyPropertyType::Float,
        });
        vertex_properties.push(PlyProperty {
            name: "z".to_string(),
            property_type: PlyPropertyType::Float,
        });
        
        // Add custom properties if specified
        for (prop_name, _) in &options.custom_vertex_properties {
            // Determine property type from first value
            if let Some(first_values) = options.custom_vertex_properties.iter()
                .find(|(name, _)| name == prop_name)
                .map(|(_, values)| values)
            {
                if let Some(first_value) = first_values.first() {
                    let prop_type = Self::value_to_property_type(first_value);
                    vertex_properties.push(PlyProperty {
                        name: prop_name.clone(),
                        property_type: prop_type,
                    });
                }
            }
        }
        
        // Reorder properties if custom order is specified
        if let Some(order) = &options.vertex_property_order {
            vertex_properties.sort_by_key(|prop| {
                order.iter().position(|name| name == &prop.name)
                    .unwrap_or(order.len())
            });
        }
        
        // Build vertex data
        for (i, point) in cloud.iter().enumerate() {
            let mut vertex_instance = HashMap::new();
            vertex_instance.insert("x".to_string(), PlyValue::Float(point.x));
            vertex_instance.insert("y".to_string(), PlyValue::Float(point.y));
            vertex_instance.insert("z".to_string(), PlyValue::Float(point.z));
            
            // Add custom properties
            for (prop_name, values) in &options.custom_vertex_properties {
                if i < values.len() {
                    vertex_instance.insert(prop_name.clone(), values[i].clone());
                }
            }
            
            vertex_data.push(vertex_instance);
        }
        
        // Add vertex element
        ply_data.header.elements.push(PlyElement {
            name: "vertex".to_string(),
            count: cloud.len(),
            properties: vertex_properties,
        });
        ply_data.elements.insert("vertex".to_string(), vertex_data);
        
        // Write PLY data
        Self::write_ply_data(writer, &ply_data)
    }
    
    /// Write triangle mesh to PLY file with options
    pub fn write_mesh<P: AsRef<Path>>(
        mesh: &TriangleMesh,
        path: P,
        options: &PlyWriteOptions,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        Self::write_mesh_to_writer(mesh, &mut writer, options)
    }
    
    /// Write triangle mesh to writer with options
    pub fn write_mesh_to_writer<W: std::io::Write>(
        mesh: &TriangleMesh,
        writer: &mut W,
        options: &PlyWriteOptions,
    ) -> Result<()> {
        // Build PLY data structure
        let mut ply_data = PlyData {
            header: PlyHeader {
                format: options.format,
                version: "1.0".to_string(),
                elements: Vec::new(),
                comments: options.comments.clone(),
                obj_info: options.obj_info.clone(),
            },
            elements: HashMap::new(),
        };
        
        // Build vertex element
        let mut vertex_properties = Vec::new();
        let mut vertex_data = Vec::new();
        
        // Always include x, y, z
        vertex_properties.push(PlyProperty {
            name: "x".to_string(),
            property_type: PlyPropertyType::Float,
        });
        vertex_properties.push(PlyProperty {
            name: "y".to_string(),
            property_type: PlyPropertyType::Float,
        });
        vertex_properties.push(PlyProperty {
            name: "z".to_string(),
            property_type: PlyPropertyType::Float,
        });
        
        // Add normals if requested and available
        if options.include_normals && mesh.normals.is_some() {
            vertex_properties.push(PlyProperty {
                name: "nx".to_string(),
                property_type: PlyPropertyType::Float,
            });
            vertex_properties.push(PlyProperty {
                name: "ny".to_string(),
                property_type: PlyPropertyType::Float,
            });
            vertex_properties.push(PlyProperty {
                name: "nz".to_string(),
                property_type: PlyPropertyType::Float,
            });
        }
        
        // Add custom vertex properties
        for (prop_name, _) in &options.custom_vertex_properties {
            if let Some(first_values) = options.custom_vertex_properties.iter()
                .find(|(name, _)| name == prop_name)
                .map(|(_, values)| values)
            {
                if let Some(first_value) = first_values.first() {
                    let prop_type = Self::value_to_property_type(first_value);
                    vertex_properties.push(PlyProperty {
                        name: prop_name.clone(),
                        property_type: prop_type,
                    });
                }
            }
        }
        
        // Reorder properties if specified
        if let Some(order) = &options.vertex_property_order {
            vertex_properties.sort_by_key(|prop| {
                order.iter().position(|name| name == &prop.name)
                    .unwrap_or(order.len())
            });
        }
        
        // Build vertex data
        for (i, vertex) in mesh.vertices.iter().enumerate() {
            let mut vertex_instance = HashMap::new();
            vertex_instance.insert("x".to_string(), PlyValue::Float(vertex.x));
            vertex_instance.insert("y".to_string(), PlyValue::Float(vertex.y));
            vertex_instance.insert("z".to_string(), PlyValue::Float(vertex.z));
            
            // Add normals if available and requested
            if options.include_normals {
                if let Some(normals) = &mesh.normals {
                    if i < normals.len() {
                        vertex_instance.insert("nx".to_string(), PlyValue::Float(normals[i].x));
                        vertex_instance.insert("ny".to_string(), PlyValue::Float(normals[i].y));
                        vertex_instance.insert("nz".to_string(), PlyValue::Float(normals[i].z));
                    }
                }
            }
            
            // Add custom properties
            for (prop_name, values) in &options.custom_vertex_properties {
                if i < values.len() {
                    vertex_instance.insert(prop_name.clone(), values[i].clone());
                }
            }
            
            vertex_data.push(vertex_instance);
        }
        
        // Add vertex element
        ply_data.header.elements.push(PlyElement {
            name: "vertex".to_string(),
            count: mesh.vertices.len(),
            properties: vertex_properties,
        });
        ply_data.elements.insert("vertex".to_string(), vertex_data);
        
        // Build face element
        if !mesh.faces.is_empty() {
            let face_properties = vec![
                PlyProperty {
                    name: "vertex_indices".to_string(),
                    property_type: PlyPropertyType::List(
                        Box::new(PlyPropertyType::UChar),
                        Box::new(PlyPropertyType::Int),
                    ),
                },
            ];
            
            let mut face_data = Vec::new();
            for face in &mesh.faces {
                let mut face_instance = HashMap::new();
                let indices = vec![
                    PlyValue::Int(face[0] as i32),
                    PlyValue::Int(face[1] as i32),
                    PlyValue::Int(face[2] as i32),
                ];
                face_instance.insert("vertex_indices".to_string(), PlyValue::List(indices));
                face_data.push(face_instance);
            }
            
            ply_data.header.elements.push(PlyElement {
                name: "face".to_string(),
                count: mesh.faces.len(),
                properties: face_properties,
            });
            ply_data.elements.insert("face".to_string(), face_data);
        }
        
        // Write PLY data
        Self::write_ply_data(writer, &ply_data)
    }
    
    /// Write PLY data to writer
    fn write_ply_data<W: std::io::Write>(writer: &mut W, ply_data: &PlyData) -> Result<()> {
        // Write header
        Self::write_header(writer, &ply_data.header)?;
        
        // Write element data
        match ply_data.header.format {
            PlyFormat::Ascii => Self::write_ascii_data(writer, ply_data)?,
            PlyFormat::BinaryLittleEndian => Self::write_binary_data::<LittleEndian, _>(writer, ply_data)?,
            PlyFormat::BinaryBigEndian => Self::write_binary_data::<BigEndian, _>(writer, ply_data)?,
        }
        
        Ok(())
    }
    
    /// Write PLY header
    fn write_header<W: std::io::Write>(writer: &mut W, header: &PlyHeader) -> Result<()> {
        writeln!(writer, "ply")?;
        
        let format_str = match header.format {
            PlyFormat::Ascii => "ascii",
            PlyFormat::BinaryLittleEndian => "binary_little_endian",
            PlyFormat::BinaryBigEndian => "binary_big_endian",
        };
        writeln!(writer, "format {} {}", format_str, header.version)?;
        
        // Write comments
        for comment in &header.comments {
            writeln!(writer, "comment {}", comment)?;
        }
        
        // Write obj_info
        for info in &header.obj_info {
            writeln!(writer, "obj_info {}", info)?;
        }
        
        // Write elements and properties
        for element in &header.elements {
            writeln!(writer, "element {} {}", element.name, element.count)?;
            for property in &element.properties {
                Self::write_property_definition(writer, property)?;
            }
        }
        
        writeln!(writer, "end_header")?;
        Ok(())
    }
    
    /// Write property definition
    fn write_property_definition<W: std::io::Write>(writer: &mut W, property: &PlyProperty) -> Result<()> {
        match &property.property_type {
            PlyPropertyType::List(count_type, item_type) => {
                let count_str = Self::property_type_to_string(count_type);
                let item_str = Self::property_type_to_string(item_type);
                writeln!(writer, "property list {} {} {}", count_str, item_str, property.name)?;
            }
            _ => {
                let type_str = Self::property_type_to_string(&property.property_type);
                writeln!(writer, "property {} {}", type_str, property.name)?;
            }
        }
        Ok(())
    }
    
    /// Convert property type to string
    fn property_type_to_string(prop_type: &PlyPropertyType) -> &'static str {
        match prop_type {
            PlyPropertyType::Char => "char",
            PlyPropertyType::UChar => "uchar",
            PlyPropertyType::Short => "short",
            PlyPropertyType::UShort => "ushort",
            PlyPropertyType::Int => "int",
            PlyPropertyType::UInt => "uint",
            PlyPropertyType::Float => "float",
            PlyPropertyType::Double => "double",
            PlyPropertyType::List(_, _) => "list", // Should not be called directly for lists
        }
    }
    
    /// Write ASCII format data
    fn write_ascii_data<W: std::io::Write>(writer: &mut W, ply_data: &PlyData) -> Result<()> {
        for element_def in &ply_data.header.elements {
            if let Some(element_data) = ply_data.elements.get(&element_def.name) {
                for instance in element_data {
                    let mut values = Vec::new();
                    
                    for property in &element_def.properties {
                        if let Some(value) = instance.get(&property.name) {
                            Self::format_ascii_value(value, &mut values)?;
                        }
                    }
                    
                    writeln!(writer, "{}", values.join(" "))?;
                }
            }
        }
        Ok(())
    }
    
    /// Format a value for ASCII output
    fn format_ascii_value(value: &PlyValue, output: &mut Vec<String>) -> Result<()> {
        match value {
            PlyValue::Char(v) => output.push(v.to_string()),
            PlyValue::UChar(v) => output.push(v.to_string()),
            PlyValue::Short(v) => output.push(v.to_string()),
            PlyValue::UShort(v) => output.push(v.to_string()),
            PlyValue::Int(v) => output.push(v.to_string()),
            PlyValue::UInt(v) => output.push(v.to_string()),
            PlyValue::Float(v) => output.push(v.to_string()),
            PlyValue::Double(v) => output.push(v.to_string()),
            PlyValue::List(values) => {
                output.push(values.len().to_string());
                for item in values {
                    Self::format_ascii_value(item, output)?;
                }
            }
        }
        Ok(())
    }
    
    /// Write binary format data
    fn write_binary_data<E: byteorder::ByteOrder, W: std::io::Write>(
        writer: &mut W,
        ply_data: &PlyData,
    ) -> Result<()> {
        
        for element_def in &ply_data.header.elements {
            if let Some(element_data) = ply_data.elements.get(&element_def.name) {
                for instance in element_data {
                    for property in &element_def.properties {
                        if let Some(value) = instance.get(&property.name) {
                            Self::write_binary_value::<E, _>(writer, value)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Write a binary value
    fn write_binary_value<E: byteorder::ByteOrder, W: std::io::Write>(
        writer: &mut W,
        value: &PlyValue,
    ) -> Result<()> {
        use byteorder::WriteBytesExt;
        
        match value {
            PlyValue::Char(v) => writer.write_i8(*v)?,
            PlyValue::UChar(v) => writer.write_u8(*v)?,
            PlyValue::Short(v) => writer.write_i16::<E>(*v)?,
            PlyValue::UShort(v) => writer.write_u16::<E>(*v)?,
            PlyValue::Int(v) => writer.write_i32::<E>(*v)?,
            PlyValue::UInt(v) => writer.write_u32::<E>(*v)?,
            PlyValue::Float(v) => writer.write_f32::<E>(*v)?,
            PlyValue::Double(v) => writer.write_f64::<E>(*v)?,
            PlyValue::List(values) => {
                // Write count as uchar (assuming list count type is uchar)
                writer.write_u8(values.len() as u8)?;
                for item in values {
                    Self::write_binary_value::<E, _>(writer, item)?;
                }
            }
        }
        Ok(())
    }
    
    /// Determine property type from PLY value
    fn value_to_property_type(value: &PlyValue) -> PlyPropertyType {
        match value {
            PlyValue::Char(_) => PlyPropertyType::Char,
            PlyValue::UChar(_) => PlyPropertyType::UChar,
            PlyValue::Short(_) => PlyPropertyType::Short,
            PlyValue::UShort(_) => PlyPropertyType::UShort,
            PlyValue::Int(_) => PlyPropertyType::Int,
            PlyValue::UInt(_) => PlyPropertyType::UInt,
            PlyValue::Float(_) => PlyPropertyType::Float,
            PlyValue::Double(_) => PlyPropertyType::Double,
            PlyValue::List(values) => {
                let item_type = if let Some(first_item) = values.first() {
                    Self::value_to_property_type(first_item)
                } else {
                    PlyPropertyType::Int
                };
                PlyPropertyType::List(Box::new(PlyPropertyType::UChar), Box::new(item_type))
            }
        }
    }
}

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

// Implement the new unified traits
impl crate::registry::PointCloudReader for PlyReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
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
    
    fn can_read(&self, path: &Path) -> bool {
        // Check if file starts with "ply"
        if let Ok(mut file) = File::open(path) {
            let mut header = [0u8; 4];
            if let Ok(_) = file.read(&mut header) {
                return header.starts_with(b"ply");
            }
        }
        false
    }
    
    fn format_name(&self) -> &'static str {
        "ply"
    }
}

impl crate::registry::MeshReader for PlyReader {
    fn read_mesh(&self, path: &Path) -> Result<TriangleMesh> {
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
            
            if has_normals {
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
    
    fn can_read(&self, path: &Path) -> bool {
        // Check if file starts with "ply"
        if let Ok(mut file) = File::open(path) {
            let mut header = [0u8; 4];
            if let Ok(_) = file.read(&mut header) {
                return header.starts_with(b"ply");
            }
        }
        false
    }
    
    fn format_name(&self) -> &'static str {
        "ply"
    }
}

impl crate::registry::PointCloudWriter for PlyWriter {
    fn write_point_cloud(&self, cloud: &PointCloud<Point3f>, path: &Path) -> Result<()> {
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
    
    fn format_name(&self) -> &'static str {
        "ply"
    }
}

impl crate::registry::MeshWriter for PlyWriter {
    fn write_mesh(&self, mesh: &TriangleMesh, path: &Path) -> Result<()> {
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
        
        // Add vertex data
        let mut vertices = Vec::new();
        for (i, point) in mesh.vertices.iter().enumerate() {
            let mut vertex = DefaultElement::new();
            vertex.insert("x".to_string(), Property::Float(point.x));
            vertex.insert("y".to_string(), Property::Float(point.y));
            vertex.insert("z".to_string(), Property::Float(point.z));
            
            // Add normals if available
            if let Some(ref normals) = mesh.normals {
                if i < normals.len() {
                    vertex.insert("nx".to_string(), Property::Float(normals[i].x));
                    vertex.insert("ny".to_string(), Property::Float(normals[i].y));
                    vertex.insert("nz".to_string(), Property::Float(normals[i].z));
                }
            }
            
            vertices.push(vertex);
        }
        ply.payload.insert("vertex".to_string(), vertices);
        
        // Define face element if we have faces
        if !mesh.faces.is_empty() {
            let mut face_element = ElementDef::new("face".to_string());
            face_element.count = mesh.faces.len();
            face_element.properties.add(PropertyDef::new(
                "vertex_indices".to_string(),
                PropertyType::List(ScalarType::UChar, ScalarType::Int),
            ));
            
            ply.header.elements.add(face_element);
            
            // Add face data
            let mut faces = Vec::new();
            for face in &mesh.faces {
                let mut face_data = DefaultElement::new();
                let indices = vec![
                    face[0] as i32,
                    face[1] as i32,
                    face[2] as i32,
                ];
                face_data.insert("vertex_indices".to_string(), Property::ListInt(indices));
                faces.push(face_data);
            }
            ply.payload.insert("face".to_string(), faces);
        }
        
        // Write PLY file
        let writer_instance = Writer::new();
        writer_instance.write_ply(&mut writer, &mut ply)?;
        
        Ok(())
    }
    
    fn format_name(&self) -> &'static str {
        "ply"
    }
}

// Keep the legacy trait implementations for backward compatibility
impl PointCloudReader for PlyReader {
    fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let reader = PlyReader;
        crate::registry::PointCloudReader::read_point_cloud(&reader, path.as_ref())
    }
}

impl MeshReader for PlyReader {
    fn read_mesh<P: AsRef<Path>>(path: P) -> Result<TriangleMesh> {
        let reader = PlyReader;
        crate::registry::MeshReader::read_mesh(&reader, path.as_ref())
    }
}

impl PointCloudWriter for PlyWriter {
    fn write_point_cloud<P: AsRef<Path>>(cloud: &PointCloud<Point3f>, path: P) -> Result<()> {
        let writer = PlyWriter;
        crate::registry::PointCloudWriter::write_point_cloud(&writer, cloud, path.as_ref())
    }
}

impl MeshWriter for PlyWriter {
    fn write_mesh<P: AsRef<Path>>(mesh: &TriangleMesh, path: P) -> Result<()> {
        let writer = PlyWriter;
        crate::registry::MeshWriter::write_mesh(&writer, mesh, path.as_ref())
    }
}