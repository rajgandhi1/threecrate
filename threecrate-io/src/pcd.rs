//! PCD (Point Cloud Data) format support
//!
//! This module provides comprehensive PCD format reading and writing capabilities
//! including ASCII and binary formats, with support for various field types.

use crate::{PointCloudReader, PointCloudWriter};
use crate::registry::{PointCloudReader as RegistryPointCloudReader, PointCloudWriter as RegistryPointCloudWriter};
use threecrate_core::{PointCloud, Point3f, Result, Error};
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::collections::HashMap;

/// PCD data format variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcdDataFormat {
    Ascii,
    Binary,
    BinaryCompressed,
}

/// PCD field data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcdFieldType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

/// PCD field definition
#[derive(Debug, Clone)]
pub struct PcdField {
    pub name: String,
    pub field_type: PcdFieldType,
    pub count: usize,
}

/// PCD header information
#[derive(Debug, Clone)]
pub struct PcdHeader {
    pub version: String,
    pub fields: Vec<PcdField>,
    pub width: usize,
    pub height: usize,
    pub viewpoint: [f64; 7], // tx, ty, tz, qw, qx, qy, qz
    pub data_format: PcdDataFormat,
}

/// PCD field value
#[derive(Debug, Clone)]
pub enum PcdValue {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
}

/// PCD point data (all fields for a single point)
pub type PcdPoint = HashMap<String, Vec<PcdValue>>;

/// PCD write options
#[derive(Debug, Clone)]
pub struct PcdWriteOptions {
    pub data_format: PcdDataFormat,
    pub version: String,
    pub viewpoint: Option<[f64; 7]>,
    pub additional_fields: Vec<PcdField>,
}

impl Default for PcdWriteOptions {
    fn default() -> Self {
        Self {
            data_format: PcdDataFormat::Binary,
            version: "0.7".to_string(),
            viewpoint: None,
            additional_fields: Vec::new(),
        }
    }
}

/// Enhanced PCD reader with comprehensive format support
pub struct RobustPcdReader;

impl RobustPcdReader {
    /// Read PCD file and return header and point data
    pub fn read_pcd_file<P: AsRef<Path>>(path: P) -> Result<(PcdHeader, Vec<PcdPoint>)> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_pcd_data(&mut reader)
    }

    /// Read PCD data from a reader
    pub fn read_pcd_data<R: BufRead>(reader: &mut R) -> Result<(PcdHeader, Vec<PcdPoint>)> {
        let header = Self::read_header(reader)?;
        let points = Self::read_points(reader, &header)?;
        Ok((header, points))
    }

    /// Read PCD header
    fn read_header<R: BufRead>(reader: &mut R) -> Result<PcdHeader> {
        let mut version = None;
        let mut fields = Vec::new();
        let mut size = Vec::new();
        let mut field_types = Vec::new();
        let mut count = Vec::new();
        let mut width = None;
        let mut height = None;
        let mut viewpoint = [0.0; 7];
        let mut points = None;
        let mut _data_format = None;

        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(Error::InvalidData("Unexpected end of file in PCD header".to_string()));
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with('#') {
                continue; // Skip comments
            }

            if line == "DATA ascii" {
                _data_format = Some(PcdDataFormat::Ascii);
                break;
            } else if line == "DATA binary" {
                _data_format = Some(PcdDataFormat::Binary);
                break;
            } else if line == "DATA binary_compressed" {
                _data_format = Some(PcdDataFormat::BinaryCompressed);
                break;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }


            match parts[0] {
                "VERSION" => {
                    if parts.len() >= 2 {
                        version = Some(parts[1].to_string());
                    }
                }
                "FIELDS" => {
                    if parts.len() >= 2 {
                        for &field_name in &parts[1..] {
                            fields.push(PcdField {
                                name: field_name.to_string(),
                                field_type: PcdFieldType::F32, // Will be updated by TYPE
                                count: 1, // Will be updated by COUNT
                            });
                        }
                    }
                }
                "SIZE" => {
                    if parts.len() >= 2 {
                        for &size_str in &parts[1..] {
                            size.push(size_str.parse::<usize>()
                                .map_err(|_| Error::InvalidData(format!("Invalid SIZE value: {}", size_str)))?);
                        }
                    }
                }
                "TYPE" => {
                    if parts.len() >= 2 {
                        for (i, &type_str) in parts[1..].iter().enumerate() {
                            let size = if i < size.len() { size[i] } else { 4 }; // Default to 4 if SIZE not specified
                            let field_type = match (type_str, size) {
                                ("I", 1) => PcdFieldType::I8,
                                ("I", 2) => PcdFieldType::I16,
                                ("I", 4) | ("I", _) => PcdFieldType::I32, // Default to I32 for unknown sizes
                                ("U", 1) => PcdFieldType::U8,
                                ("U", 2) => PcdFieldType::U16,
                                ("U", 4) | ("U", _) => PcdFieldType::U32,
                                ("F", 4) => PcdFieldType::F32,
                                ("F", 8) | ("F", _) => PcdFieldType::F64,
                                _ => return Err(Error::InvalidData(format!("Unknown field type/size combination: {}/{}", type_str, size))),
                            };
                            field_types.push(field_type);
                        }
                    }
                }
                "COUNT" => {
                    if parts.len() >= 2 {
                        for &count_str in &parts[1..] {
                            count.push(count_str.parse::<usize>()
                                .map_err(|_| Error::InvalidData(format!("Invalid COUNT value: {}", count_str)))?);
                        }
                    }
                }
                "WIDTH" => {
                    if parts.len() >= 2 {
                        width = Some(parts[1].parse::<usize>()
                            .map_err(|_| Error::InvalidData(format!("Invalid WIDTH value: {}", parts[1])))?);
                    }
                }
                "HEIGHT" => {
                    if parts.len() >= 2 {
                        height = Some(parts[1].parse::<usize>()
                            .map_err(|_| Error::InvalidData(format!("Invalid HEIGHT value: {}", parts[1])))?);
                    }
                }
                "VIEWPOINT" => {
                    if parts.len() >= 8 {
                        for i in 0..7 {
                            viewpoint[i] = parts[i + 1].parse::<f64>()
                                .map_err(|_| Error::InvalidData(format!("Invalid VIEWPOINT value: {}", parts[i + 1])))?;
                        }
                    }
                }
                "POINTS" => {
                    if parts.len() >= 2 {
                        points = Some(parts[1].parse::<usize>()
                            .map_err(|_| Error::InvalidData(format!("Invalid POINTS value: {}", parts[1])))?);
                    }
                }
                _ => {
                    // Ignore unknown header fields
                }
            }
        }

        let version = version.ok_or_else(|| Error::InvalidData("Missing VERSION in PCD header".to_string()))?;
        let width = width.ok_or_else(|| Error::InvalidData("Missing WIDTH in PCD header".to_string()))?;
        let height = height.ok_or_else(|| Error::InvalidData("Missing HEIGHT in PCD header".to_string()))?;
        let data_format = _data_format.ok_or_else(|| Error::InvalidData("Missing DATA format in PCD header".to_string()))?;

        // Update field definitions with type and count information
        if fields.len() == field_types.len() && fields.len() == count.len() {
            for (i, field) in fields.iter_mut().enumerate() {
                field.field_type = field_types[i];
                field.count = count[i];
            }
        } else {
            return Err(Error::InvalidData("Mismatch between FIELDS, TYPE, and COUNT declarations".to_string()));
        }

        // If POINTS is specified and different from WIDTH * HEIGHT, validate it
        if let Some(points) = points {
            if points != width * height {
                return Err(Error::InvalidData(format!("POINTS ({}) doesn't match WIDTH * HEIGHT ({})", points, width * height)));
            }
        }

        Ok(PcdHeader {
            version,
            fields,
            width,
            height,
            viewpoint,
            data_format,
        })
    }

    /// Read point data based on header format
    fn read_points<R: BufRead>(reader: &mut R, header: &PcdHeader) -> Result<Vec<PcdPoint>> {
        match header.data_format {
            PcdDataFormat::Ascii => Self::read_ascii_points(reader, header),
            PcdDataFormat::Binary => Self::read_binary_points(reader, header),
            PcdDataFormat::BinaryCompressed => {
                Err(Error::Unsupported("Binary compressed PCD format not yet supported".to_string()))
            }
        }
    }

    /// Read ASCII format points
    fn read_ascii_points<R: BufRead>(reader: &mut R, header: &PcdHeader) -> Result<Vec<PcdPoint>> {
        let mut points = Vec::with_capacity(header.width * header.height);

        for _ in 0..(header.width * header.height) {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let values: Vec<&str> = line.split_whitespace().collect();
            let mut value_idx = 0;
            let mut point = PcdPoint::new();

            for field in &header.fields {
                let field_values = Self::read_ascii_field_values(&values, &mut value_idx, field)?;
                point.insert(field.name.clone(), field_values);
            }

            points.push(point);
        }

        Ok(points)
    }

    /// Read binary format points
    fn read_binary_points<R: Read>(reader: &mut R, header: &PcdHeader) -> Result<Vec<PcdPoint>> {
        let mut points = Vec::with_capacity(header.width * header.height);

        for _ in 0..(header.width * header.height) {
            let mut point = PcdPoint::new();

            for field in &header.fields {
                let field_values = Self::read_binary_field_values(reader, field)?;
                point.insert(field.name.clone(), field_values);
            }

            points.push(point);
        }

        Ok(points)
    }

    /// Read ASCII field values
    fn read_ascii_field_values(values: &[&str], value_idx: &mut usize, field: &PcdField) -> Result<Vec<PcdValue>> {
        let mut field_values = Vec::with_capacity(field.count);

        for _ in 0..field.count {
            if *value_idx >= values.len() {
                return Err(Error::InvalidData("Not enough values in ASCII PCD line".to_string()));
            }

            let value = match field.field_type {
                PcdFieldType::I8 => PcdValue::I8(values[*value_idx].parse::<i8>()
                    .map_err(|_| Error::InvalidData(format!("Invalid I8 value: {}", values[*value_idx])))?),
                PcdFieldType::U8 => PcdValue::U8(values[*value_idx].parse::<u8>()
                    .map_err(|_| Error::InvalidData(format!("Invalid U8 value: {}", values[*value_idx])))?),
                PcdFieldType::I16 => PcdValue::I16(values[*value_idx].parse::<i16>()
                    .map_err(|_| Error::InvalidData(format!("Invalid I16 value: {}", values[*value_idx])))?),
                PcdFieldType::U16 => PcdValue::U16(values[*value_idx].parse::<u16>()
                    .map_err(|_| Error::InvalidData(format!("Invalid U16 value: {}", values[*value_idx])))?),
                PcdFieldType::I32 => PcdValue::I32(values[*value_idx].parse::<i32>()
                    .map_err(|_| Error::InvalidData(format!("Invalid I32 value: {}", values[*value_idx])))?),
                PcdFieldType::U32 => PcdValue::U32(values[*value_idx].parse::<u32>()
                    .map_err(|_| Error::InvalidData(format!("Invalid U32 value: {}", values[*value_idx])))?),
                PcdFieldType::F32 => PcdValue::F32(values[*value_idx].parse::<f32>()
                    .map_err(|_| Error::InvalidData(format!("Invalid F32 value: {}", values[*value_idx])))?),
                PcdFieldType::F64 => PcdValue::F64(values[*value_idx].parse::<f64>()
                    .map_err(|_| Error::InvalidData(format!("Invalid F64 value: {}", values[*value_idx])))?),
            };

            field_values.push(value);
            *value_idx += 1;
        }

        Ok(field_values)
    }

    /// Read binary field values
    fn read_binary_field_values<R: Read>(reader: &mut R, field: &PcdField) -> Result<Vec<PcdValue>> {
        let mut field_values = Vec::with_capacity(field.count);

        for _ in 0..field.count {
            let value = match field.field_type {
                PcdFieldType::I8 => {
                    let mut buf = [0u8; 1];
                    reader.read_exact(&mut buf)?;
                    PcdValue::I8(buf[0] as i8)
                }
                PcdFieldType::U8 => {
                    let mut buf = [0u8; 1];
                    reader.read_exact(&mut buf)?;
                    PcdValue::U8(buf[0])
                }
                PcdFieldType::I16 => {
                    let mut buf = [0u8; 2];
                    reader.read_exact(&mut buf)?;
                    PcdValue::I16(i16::from_le_bytes(buf))
                }
                PcdFieldType::U16 => {
                    let mut buf = [0u8; 2];
                    reader.read_exact(&mut buf)?;
                    PcdValue::U16(u16::from_le_bytes(buf))
                }
                PcdFieldType::I32 => {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    PcdValue::I32(i32::from_le_bytes(buf))
                }
                PcdFieldType::U32 => {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    PcdValue::U32(u32::from_le_bytes(buf))
                }
                PcdFieldType::F32 => {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    PcdValue::F32(f32::from_le_bytes(buf))
                }
                PcdFieldType::F64 => {
                    let mut buf = [0u8; 8];
                    reader.read_exact(&mut buf)?;
                    PcdValue::F64(f64::from_le_bytes(buf))
                }
            };

            field_values.push(value);
        }

        Ok(field_values)
    }

    /// Convert PCD data to PointCloud
    pub fn pcd_to_point_cloud(_header: &PcdHeader, points: &[PcdPoint]) -> Result<PointCloud<Point3f>> {
        let mut cloud_points = Vec::with_capacity(points.len());

        for point in points {
            // Extract x, y, z coordinates
            let x_values = point.get("x")
                .ok_or_else(|| Error::InvalidData("Missing x coordinate in PCD point".to_string()))?;
            let y_values = point.get("y")
                .ok_or_else(|| Error::InvalidData("Missing y coordinate in PCD point".to_string()))?;
            let z_values = point.get("z")
                .ok_or_else(|| Error::InvalidData("Missing z coordinate in PCD point".to_string()))?;

            if x_values.is_empty() || y_values.is_empty() || z_values.is_empty() {
                return Err(Error::InvalidData("Empty coordinate values in PCD point".to_string()));
            }

            let x = Self::pcd_value_to_f64(&x_values[0])?;
            let y = Self::pcd_value_to_f64(&y_values[0])?;
            let z = Self::pcd_value_to_f64(&z_values[0])?;

            cloud_points.push(Point3f::new(x as f32, y as f32, z as f32));
        }

        Ok(PointCloud::from_points(cloud_points))
    }

    /// Convert PcdValue to f64
    fn pcd_value_to_f64(value: &PcdValue) -> Result<f64> {
        match value {
            PcdValue::I8(v) => Ok(*v as f64),
            PcdValue::U8(v) => Ok(*v as f64),
            PcdValue::I16(v) => Ok(*v as f64),
            PcdValue::U16(v) => Ok(*v as f64),
            PcdValue::I32(v) => Ok(*v as f64),
            PcdValue::U32(v) => Ok(*v as f64),
            PcdValue::F32(v) => Ok(*v as f64),
            PcdValue::F64(v) => Ok(*v),
        }
    }
}

/// Enhanced PCD writer with comprehensive format support
pub struct RobustPcdWriter;

impl RobustPcdWriter {
    /// Write point cloud to PCD file with options
    pub fn write_point_cloud<P: AsRef<Path>>(
        cloud: &PointCloud<Point3f>,
        path: P,
        options: &PcdWriteOptions
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        Self::write_point_cloud_to_writer(cloud, &mut writer, options)
    }

    /// Write point cloud to writer with options
    pub fn write_point_cloud_to_writer<W: Write>(
        cloud: &PointCloud<Point3f>,
        writer: &mut W,
        options: &PcdWriteOptions
    ) -> Result<()> {
        // Build PCD header
        let mut fields = vec![
            PcdField {
                name: "x".to_string(),
                field_type: PcdFieldType::F32,
                count: 1,
            },
            PcdField {
                name: "y".to_string(),
                field_type: PcdFieldType::F32,
                count: 1,
            },
            PcdField {
                name: "z".to_string(),
                field_type: PcdFieldType::F32,
                count: 1,
            },
        ];

        // Add additional fields
        fields.extend(options.additional_fields.clone());

        let header = PcdHeader {
            version: options.version.clone(),
            fields,
            width: cloud.len(),
            height: 1,
            viewpoint: options.viewpoint.unwrap_or([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            data_format: options.data_format,
        };

        // Write header
        Self::write_header(writer, &header)?;

        // Write data
        match options.data_format {
            PcdDataFormat::Ascii => Self::write_ascii_data(writer, cloud, &header),
            PcdDataFormat::Binary => Self::write_binary_data(writer, cloud, &header),
            PcdDataFormat::BinaryCompressed => {
                Err(Error::Unsupported("Binary compressed PCD format not yet supported".to_string()))
            }
        }
    }

    /// Write PCD header
    fn write_header<W: Write>(writer: &mut W, header: &PcdHeader) -> Result<()> {
        writeln!(writer, "# .PCD v{} - Point Cloud Data file format", header.version)?;
        writeln!(writer, "VERSION {}", header.version)?;
        write!(writer, "FIELDS")?;
        for field in &header.fields {
            write!(writer, " {}", field.name)?;
        }
        writeln!(writer)?;

        write!(writer, "SIZE")?;
        for field in &header.fields {
            let size = match field.field_type {
                PcdFieldType::I8 | PcdFieldType::U8 => 1,
                PcdFieldType::I16 | PcdFieldType::U16 => 2,
                PcdFieldType::I32 | PcdFieldType::U32 | PcdFieldType::F32 => 4,
                PcdFieldType::F64 => 8,
            };
            write!(writer, " {}", size)?;
        }
        writeln!(writer)?;

        write!(writer, "TYPE")?;
        for field in &header.fields {
            let type_char = match field.field_type {
                PcdFieldType::I8 | PcdFieldType::I16 | PcdFieldType::I32 => "I",
                PcdFieldType::U8 | PcdFieldType::U16 | PcdFieldType::U32 => "U",
                PcdFieldType::F32 | PcdFieldType::F64 => "F",
            };
            write!(writer, " {}", type_char)?;
        }
        writeln!(writer)?;

        write!(writer, "COUNT")?;
        for field in &header.fields {
            write!(writer, " {}", field.count)?;
        }
        writeln!(writer)?;

        writeln!(writer, "WIDTH {}", header.width)?;
        writeln!(writer, "HEIGHT {}", header.height)?;
        writeln!(writer, "VIEWPOINT {} {} {} {} {} {} {}",
                 header.viewpoint[0], header.viewpoint[1], header.viewpoint[2],
                 header.viewpoint[3], header.viewpoint[4], header.viewpoint[5], header.viewpoint[6])?;
        writeln!(writer, "POINTS {}", header.width * header.height)?;

        let data_str = match header.data_format {
            PcdDataFormat::Ascii => "ascii",
            PcdDataFormat::Binary => "binary",
            PcdDataFormat::BinaryCompressed => "binary_compressed",
        };
        writeln!(writer, "DATA {}", data_str)?;

        Ok(())
    }

    /// Write ASCII format data
    fn write_ascii_data<W: Write>(
        writer: &mut W,
        cloud: &PointCloud<Point3f>,
        _header: &PcdHeader
    ) -> Result<()> {
        for point in cloud.iter() {
            // Write x, y, z
            write!(writer, "{} {} {}", point.x, point.y, point.z)?;

            // Write additional fields (all zeros for now)
            // TODO: Implement additional field writing when needed
            // for field in &header.fields[3..] {
            //     for _ in 0..field.count {
            //         write!(writer, " 0")?;
            //     }
            // }

            writeln!(writer)?;
        }

        Ok(())
    }

    /// Write binary format data
    fn write_binary_data<W: Write>(
        writer: &mut W,
        cloud: &PointCloud<Point3f>,
        _header: &PcdHeader
    ) -> Result<()> {
        for point in cloud.iter() {
            // Write x, y, z as f32 little endian
            writer.write_all(&point.x.to_le_bytes())?;
            writer.write_all(&point.y.to_le_bytes())?;
            writer.write_all(&point.z.to_le_bytes())?;

            // Write additional fields (all zeros for now)
            // TODO: Implement additional field writing when needed
            // for field in &header.fields[3..] {
            //     for _ in 0..field.count {
            //         match field.field_type {
            //             PcdFieldType::I8 | PcdFieldType::U8 => writer.write_all(&[0u8])?,
            //             PcdFieldType::I16 | PcdFieldType::U16 => writer.write_all(&[0u8, 0u8])?,
            //             PcdFieldType::I32 | PcdFieldType::U32 | PcdFieldType::F32 => writer.write_all(&[0u8, 0u8, 0u8, 0u8])?,
            //             PcdFieldType::F64 => writer.write_all(&[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8])?,
            //         }
            //     }
            // }
        }

        Ok(())
    }
}

/// PCD reader implementing the registry trait
pub struct PcdReader;

impl RegistryPointCloudReader for PcdReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        let (header, points) = RobustPcdReader::read_pcd_file(path)?;
        RobustPcdReader::pcd_to_point_cloud(&header, &points)
    }

    fn can_read(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase() == "pcd")
            .unwrap_or(false)
    }

    fn format_name(&self) -> &'static str {
        "pcd"
    }
}

/// PCD writer implementing the registry trait
pub struct PcdWriter;

impl RegistryPointCloudWriter for PcdWriter {
    fn write_point_cloud(&self, cloud: &PointCloud<Point3f>, path: &Path) -> Result<()> {
        let options = PcdWriteOptions::default();
        RobustPcdWriter::write_point_cloud(cloud, path, &options)
    }

    fn format_name(&self) -> &'static str {
        "pcd"
    }
}

// Keep the legacy trait implementations for backward compatibility
impl PointCloudReader for PcdReader {
    fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let reader = PcdReader;
        RegistryPointCloudReader::read_point_cloud(&reader, path.as_ref())
    }
}

impl PointCloudWriter for PcdWriter {
    fn write_point_cloud<P: AsRef<Path>>(cloud: &PointCloud<Point3f>, path: P) -> Result<()> {
        let writer = PcdWriter;
        RegistryPointCloudWriter::write_point_cloud(&writer, cloud, path.as_ref())
    }
}
