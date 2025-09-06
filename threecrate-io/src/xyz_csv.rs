//! XYZ/CSV point cloud format support
//! 
//! This module provides comprehensive XYZ and CSV point cloud reading capabilities including:
//! - Auto-detection of delimiters (comma, space, tab)
//! - Header detection and parsing
//! - Support for x,y,z coordinates (required) and optional intensity, r,g,b, nx,ny,nz
//! - Schema hints for flexible parsing
//! - Streaming support for large files

use threecrate_core::{PointCloud, Result, Point3f, Vector3f, Error};
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Supported delimiters for CSV/XYZ files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    Comma,
    Space,
    Tab,
    Semicolon,
}

impl Delimiter {
    /// Get the character representation of the delimiter
    pub fn as_char(&self) -> char {
        match self {
            Delimiter::Comma => ',',
            Delimiter::Space => ' ',
            Delimiter::Tab => '\t',
            Delimiter::Semicolon => ';',
        }
    }
    
    /// Detect delimiter from a line of text
    pub fn detect_from_line(line: &str) -> Option<Self> {
        let comma_count = line.matches(',').count();
        let space_count = line.matches(' ').count();
        let tab_count = line.matches('\t').count();
        let semicolon_count = line.matches(';').count();
        
        // Find the delimiter with the highest count
        let counts = [
            (comma_count, Delimiter::Comma),
            (space_count, Delimiter::Space),
            (tab_count, Delimiter::Tab),
            (semicolon_count, Delimiter::Semicolon),
        ];
        
        counts.iter()
            .max_by_key(|(count, _)| count)
            .filter(|(count, _)| *count > 0)
            .map(|(_, delimiter)| *delimiter)
    }
}

/// Column types that can be parsed from XYZ/CSV files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    X,
    Y,
    Z,
    Intensity,
    Red,
    Green,
    Blue,
    NormalX,
    NormalY,
    NormalZ,
    Unknown,
}

impl ColumnType {
    /// Parse column type from header name
    pub fn from_header(header: &str) -> Self {
        let header_lower = header.to_lowercase();
        let header_trimmed = header_lower.trim();
        match header_trimmed {
            "x" | "px" | "pos_x" | "position_x" => ColumnType::X,
            "y" | "py" | "pos_y" | "position_y" => ColumnType::Y,
            "z" | "pz" | "pos_z" | "position_z" => ColumnType::Z,
            "i" | "intensity" | "int" => ColumnType::Intensity,
            "r" | "red" | "color_r" => ColumnType::Red,
            "g" | "green" | "color_g" => ColumnType::Green,
            "b" | "blue" | "color_b" => ColumnType::Blue,
            "nx" | "normal_x" | "n_x" => ColumnType::NormalX,
            "ny" | "normal_y" | "n_y" => ColumnType::NormalY,
            "nz" | "normal_z" | "n_z" => ColumnType::NormalZ,
            _ => ColumnType::Unknown,
        }
    }
}

/// Schema definition for parsing XYZ/CSV files
#[derive(Debug, Clone)]
pub struct XyzCsvSchema {
    pub columns: Vec<ColumnType>,
    pub has_header: bool,
    pub delimiter: Delimiter,
}

impl XyzCsvSchema {
    /// Create a new schema
    pub fn new(columns: Vec<ColumnType>, has_header: bool, delimiter: Delimiter) -> Self {
        Self {
            columns,
            has_header,
            delimiter,
        }
    }
    
    /// Auto-detect schema from file content
    pub fn detect_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let mut reader = BufReader::new(file);
        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;
        
        // Detect delimiter
        let delimiter = Delimiter::detect_from_line(&first_line)
            .ok_or_else(|| Error::InvalidData("Could not detect delimiter".to_string()))?;
        
        // Determine if this is a header by checking if first line contains non-numeric data
        let has_header = Self::is_header_line(&first_line, &[], delimiter);
        
        let columns = if has_header {
            // Parse columns from header
            Self::parse_columns(&first_line, delimiter)?
        } else {
            // For files without headers, assume x,y,z format
            vec![ColumnType::X, ColumnType::Y, ColumnType::Z]
        };
        
        Ok(Self::new(columns, has_header, delimiter))
    }
    
    /// Parse columns from a header line
    fn parse_columns(line: &str, delimiter: Delimiter) -> Result<Vec<ColumnType>> {
        let parts: Vec<&str> = line.split(delimiter.as_char())
            .map(|s| s.trim())
            .collect();
        
        let columns: Vec<ColumnType> = parts.iter()
            .map(|header| ColumnType::from_header(header))
            .collect();
        
        // Validate that we have at least x, y, z columns
        let has_x = columns.contains(&ColumnType::X);
        let has_y = columns.contains(&ColumnType::Y);
        let has_z = columns.contains(&ColumnType::Z);
        
        if !has_x || !has_y || !has_z {
            return Err(Error::InvalidData(
                "XYZ/CSV file must contain x, y, z coordinates".to_string()
            ));
        }
        
        Ok(columns)
    }
    
    /// Determine if a line is a header by checking for non-numeric content
    fn is_header_line(line: &str, columns: &[ColumnType], delimiter: Delimiter) -> bool {
        let parts: Vec<&str> = line.split(delimiter.as_char())
            .map(|s| s.trim())
            .collect();
        
        // If we have fewer than 3 parts, it's likely not a valid point cloud line
        if parts.len() < 3 {
            return false;
        }
        
        // If we have columns defined, check against them
        if !columns.is_empty() {
            // If we have fewer parts than expected columns, it's likely not a header
            if parts.len() < columns.len() {
                return false;
            }
            
            // Check if any part that should be numeric is not numeric
            for (i, part) in parts.iter().enumerate() {
                if i < columns.len() {
                    match columns[i] {
                        ColumnType::X | ColumnType::Y | ColumnType::Z |
                        ColumnType::Intensity | ColumnType::Red | ColumnType::Green | ColumnType::Blue |
                        ColumnType::NormalX | ColumnType::NormalY | ColumnType::NormalZ => {
                            if part.parse::<f32>().is_err() {
                                return true; // Non-numeric data suggests this is a header
                            }
                        }
                        ColumnType::Unknown => {
                            // For unknown columns, try to parse as float
                            if part.parse::<f32>().is_err() {
                                return true;
                            }
                        }
                    }
                }
            }
        } else {
            // No columns defined, check if first 3 parts are numeric
            for (_i, part) in parts.iter().enumerate().take(3) {
                if part.parse::<f32>().is_err() {
                    return true; // Non-numeric data suggests this is a header
                }
            }
        }
        
        false
    }
}

/// Point data parsed from XYZ/CSV file
#[derive(Debug, Clone)]
pub struct XyzCsvPoint {
    pub position: Point3f,
    pub intensity: Option<f32>,
    pub color: Option<[u8; 3]>,
    pub normal: Option<Vector3f>,
}

impl XyzCsvPoint {
    /// Create a new point with only position
    pub fn new(position: Point3f) -> Self {
        Self {
            position,
            intensity: None,
            color: None,
            normal: None,
        }
    }
    
    /// Create a point with position and intensity
    pub fn with_intensity(position: Point3f, intensity: f32) -> Self {
        Self {
            position,
            intensity: Some(intensity),
            color: None,
            normal: None,
        }
    }
    
    /// Create a point with position and color
    pub fn with_color(position: Point3f, color: [u8; 3]) -> Self {
        Self {
            position,
            intensity: None,
            color: Some(color),
            normal: None,
        }
    }
    
    /// Create a point with position and normal
    pub fn with_normal(position: Point3f, normal: Vector3f) -> Self {
        Self {
            position,
            intensity: None,
            color: None,
            normal: Some(normal),
        }
    }
}

/// XYZ/CSV reader implementation
pub struct XyzCsvReader;

impl XyzCsvReader {
    /// Read a point cloud from an XYZ/CSV file with auto-detection
    pub fn read_point_cloud<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let schema = XyzCsvSchema::detect_from_file(&path)?;
        Self::read_point_cloud_with_schema(path, &schema)
    }
    
    /// Read a point cloud with a specific schema
    pub fn read_point_cloud_with_schema<P: AsRef<Path>>(
        path: P, 
        schema: &XyzCsvSchema
    ) -> Result<PointCloud<Point3f>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header if present
        if schema.has_header {
            lines.next();
        }
        
        let mut cloud = PointCloud::new();
        
        for line_result in lines {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }
            
            let point = Self::parse_line(&line, schema)?;
            cloud.push(point.position);
        }
        
        Ok(cloud)
    }
    
    /// Read detailed point data (with colors, normals, etc.)
    pub fn read_detailed_points<P: AsRef<Path>>(path: P) -> Result<Vec<XyzCsvPoint>> {
        let schema = XyzCsvSchema::detect_from_file(&path)?;
        Self::read_detailed_points_with_schema(path, &schema)
    }
    
    /// Read detailed point data with a specific schema
    pub fn read_detailed_points_with_schema<P: AsRef<Path>>(
        path: P,
        schema: &XyzCsvSchema
    ) -> Result<Vec<XyzCsvPoint>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header if present
        if schema.has_header {
            lines.next();
        }
        
        let mut points = Vec::new();
        
        for line_result in lines {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }
            
            let point = Self::parse_line(&line, schema)?;
            points.push(point);
        }
        
        Ok(points)
    }
    
    /// Parse a single line into a point
    fn parse_line(line: &str, schema: &XyzCsvSchema) -> Result<XyzCsvPoint> {
        let parts: Vec<&str> = line.split(schema.delimiter.as_char())
            .map(|s| s.trim())
            .collect();
        
        if parts.len() < 3 {
            return Err(Error::InvalidData(
                "Line must have at least 3 columns (x, y, z)".to_string()
            ));
        }
        
        // Find x, y, z indices
        let mut x_idx = None;
        let mut y_idx = None;
        let mut z_idx = None;
        let mut intensity_idx = None;
        let mut red_idx = None;
        let mut green_idx = None;
        let mut blue_idx = None;
        let mut nx_idx = None;
        let mut ny_idx = None;
        let mut nz_idx = None;
        
        for (i, col_type) in schema.columns.iter().enumerate() {
            match *col_type {
                ColumnType::X => x_idx = Some(i),
                ColumnType::Y => y_idx = Some(i),
                ColumnType::Z => z_idx = Some(i),
                ColumnType::Intensity => intensity_idx = Some(i),
                ColumnType::Red => red_idx = Some(i),
                ColumnType::Green => green_idx = Some(i),
                ColumnType::Blue => blue_idx = Some(i),
                ColumnType::NormalX => nx_idx = Some(i),
                ColumnType::NormalY => ny_idx = Some(i),
                ColumnType::NormalZ => nz_idx = Some(i),
                ColumnType::Unknown => {}
            }
        }
        
        // Parse position (required)
        let x = parts[x_idx.ok_or_else(|| Error::InvalidData("Missing x coordinate".to_string()))?]
            .parse::<f32>()
            .map_err(|_| Error::InvalidData("Invalid x coordinate".to_string()))?;
        let y = parts[y_idx.ok_or_else(|| Error::InvalidData("Missing y coordinate".to_string()))?]
            .parse::<f32>()
            .map_err(|_| Error::InvalidData("Invalid y coordinate".to_string()))?;
        let z = parts[z_idx.ok_or_else(|| Error::InvalidData("Missing z coordinate".to_string()))?]
            .parse::<f32>()
            .map_err(|_| Error::InvalidData("Invalid z coordinate".to_string()))?;
        
        let position = Point3f::new(x, y, z);
        
        // Parse optional attributes
        let intensity = if let Some(idx) = intensity_idx {
            parts.get(idx).and_then(|s| s.parse::<f32>().ok())
        } else {
            None
        };
        
        let color = if let (Some(r_idx), Some(g_idx), Some(b_idx)) = (red_idx, green_idx, blue_idx) {
            if let (Some(r), Some(g), Some(b)) = (
                parts.get(r_idx).and_then(|s| s.parse::<f32>().ok()),
                parts.get(g_idx).and_then(|s| s.parse::<f32>().ok()),
                parts.get(b_idx).and_then(|s| s.parse::<f32>().ok()),
            ) {
                Some([
                    (r.clamp(0.0, 255.0) as u8),
                    (g.clamp(0.0, 255.0) as u8),
                    (b.clamp(0.0, 255.0) as u8),
                ])
            } else {
                None
            }
        } else {
            None
        };
        
        let normal = if let (Some(nx_idx), Some(ny_idx), Some(nz_idx)) = (nx_idx, ny_idx, nz_idx) {
            if let (Some(nx), Some(ny), Some(nz)) = (
                parts.get(nx_idx).and_then(|s| s.parse::<f32>().ok()),
                parts.get(ny_idx).and_then(|s| s.parse::<f32>().ok()),
                parts.get(nz_idx).and_then(|s| s.parse::<f32>().ok()),
            ) {
                Some(Vector3f::new(nx, ny, nz))
            } else {
                None
            }
        } else {
            None
        };
        
        Ok(XyzCsvPoint {
            position,
            intensity,
            color,
            normal,
        })
    }
}

/// XYZ/CSV streaming reader for large files
pub struct XyzCsvStreamingReader {
    reader: BufReader<File>,
    schema: XyzCsvSchema,
    buffer: Vec<String>,
    buffer_index: usize,
    header_skipped: bool,
}

impl XyzCsvStreamingReader {
    /// Create a new streaming reader
    pub fn new<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let reader = BufReader::with_capacity(chunk_size, file);
        let schema = XyzCsvSchema::detect_from_file(path_ref)?;
        
        Ok(Self {
            reader,
            schema,
            buffer: Vec::new(),
            buffer_index: 0,
            header_skipped: false,
        })
    }
    
    /// Fill the buffer with the next chunk of lines
    fn fill_buffer(&mut self) -> Result<bool> {
        self.buffer.clear();
        self.buffer_index = 0;
        
        for _ in 0..1000 { // Read up to 1000 lines at a time
            let mut line = String::new();
            match self.reader.read_line(&mut line)? {
                0 => break, // EOF
                _ => {
                    if !line.trim().is_empty() {
                        self.buffer.push(line);
                    }
                }
            }
        }
        
        Ok(!self.buffer.is_empty())
    }
}

impl Iterator for XyzCsvStreamingReader {
    type Item = Result<Point3f>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Skip header on first read
        if !self.header_skipped && self.schema.has_header {
            let mut line = String::new();
            if self.reader.read_line(&mut line).is_err() {
                return None;
            }
            self.header_skipped = true;
        }
        
        // Fill buffer if needed
        if self.buffer_index >= self.buffer.len() {
            match self.fill_buffer() {
                Ok(true) => {}, // Buffer filled successfully
                Ok(false) => return None, // EOF
                Err(e) => return Some(Err(e)),
            }
        }
        
        // Get next line from buffer
        if self.buffer_index < self.buffer.len() {
            let line = &self.buffer[self.buffer_index];
            self.buffer_index += 1;
            
            match XyzCsvReader::parse_line(line, &self.schema) {
                Ok(point) => Some(Ok(point.position)),
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        }
    }
}

/// XYZ/CSV writer implementation
pub struct XyzCsvWriter;

impl XyzCsvWriter {
    /// Write a point cloud to an XYZ/CSV file
    pub fn write_point_cloud<P: AsRef<Path>>(
        cloud: &PointCloud<Point3f>, 
        path: P,
        options: &XyzCsvWriteOptions
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header if requested
        if options.include_header {
            let header = Self::generate_header(&options.schema);
            writeln!(writer, "{}", header)?;
        }
        
        // Write points
        for point in cloud.iter() {
            let line = Self::format_point(point, &options.schema);
            writeln!(writer, "{}", line)?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Write detailed points to an XYZ/CSV file
    pub fn write_detailed_points<P: AsRef<Path>>(
        points: &[XyzCsvPoint],
        path: P,
        options: &XyzCsvWriteOptions
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header if requested
        if options.include_header {
            let header = Self::generate_header(&options.schema);
            writeln!(writer, "{}", header)?;
        }
        
        // Write points
        for point in points {
            let line = Self::format_detailed_point(point, &options.schema);
            writeln!(writer, "{}", line)?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Generate header line
    fn generate_header(schema: &XyzCsvSchema) -> String {
        let headers: Vec<&str> = schema.columns.iter().map(|col| match col {
            ColumnType::X => "x",
            ColumnType::Y => "y",
            ColumnType::Z => "z",
            ColumnType::Intensity => "intensity",
            ColumnType::Red => "r",
            ColumnType::Green => "g",
            ColumnType::Blue => "b",
            ColumnType::NormalX => "nx",
            ColumnType::NormalY => "ny",
            ColumnType::NormalZ => "nz",
            ColumnType::Unknown => "unknown",
        }).collect();
        
        headers.join(&schema.delimiter.as_char().to_string())
    }
    
    /// Format a basic point
    fn format_point(point: &Point3f, schema: &XyzCsvSchema) -> String {
        let mut values = Vec::new();
        
        for col_type in &schema.columns {
            match col_type {
                ColumnType::X => values.push(point.x.to_string()),
                ColumnType::Y => values.push(point.y.to_string()),
                ColumnType::Z => values.push(point.z.to_string()),
                _ => values.push("0".to_string()), // Default value for missing columns
            }
        }
        
        values.join(&schema.delimiter.as_char().to_string())
    }
    
    /// Format a detailed point
    fn format_detailed_point(point: &XyzCsvPoint, schema: &XyzCsvSchema) -> String {
        let mut values = Vec::new();
        
        for col_type in &schema.columns {
            let value = match col_type {
                ColumnType::X => point.position.x.to_string(),
                ColumnType::Y => point.position.y.to_string(),
                ColumnType::Z => point.position.z.to_string(),
                ColumnType::Intensity => point.intensity.unwrap_or(0.0).to_string(),
                ColumnType::Red => point.color.map(|c| c[0] as f32).unwrap_or(0.0).to_string(),
                ColumnType::Green => point.color.map(|c| c[1] as f32).unwrap_or(0.0).to_string(),
                ColumnType::Blue => point.color.map(|c| c[2] as f32).unwrap_or(0.0).to_string(),
                ColumnType::NormalX => point.normal.map(|n| n.x).unwrap_or(0.0).to_string(),
                ColumnType::NormalY => point.normal.map(|n| n.y).unwrap_or(0.0).to_string(),
                ColumnType::NormalZ => point.normal.map(|n| n.z).unwrap_or(0.0).to_string(),
                ColumnType::Unknown => "0".to_string(),
            };
            values.push(value);
        }
        
        values.join(&schema.delimiter.as_char().to_string())
    }
}

/// Write options for XYZ/CSV files
#[derive(Debug, Clone)]
pub struct XyzCsvWriteOptions {
    pub schema: XyzCsvSchema,
    pub include_header: bool,
}

impl XyzCsvWriteOptions {
    /// Create basic options for XYZ format
    pub fn xyz() -> Self {
        Self {
            schema: XyzCsvSchema::new(
                vec![ColumnType::X, ColumnType::Y, ColumnType::Z],
                false,
                Delimiter::Space,
            ),
            include_header: false,
        }
    }
    
    /// Create options for CSV format with header
    pub fn csv_with_header() -> Self {
        Self {
            schema: XyzCsvSchema::new(
                vec![ColumnType::X, ColumnType::Y, ColumnType::Z],
                true,
                Delimiter::Comma,
            ),
            include_header: true,
        }
    }
    
    /// Create options for CSV with colors
    pub fn csv_with_colors() -> Self {
        Self {
            schema: XyzCsvSchema::new(
                vec![
                    ColumnType::X, ColumnType::Y, ColumnType::Z,
                    ColumnType::Red, ColumnType::Green, ColumnType::Blue,
                ],
                true,
                Delimiter::Comma,
            ),
            include_header: true,
        }
    }
    
    /// Create options for CSV with normals
    pub fn csv_with_normals() -> Self {
        Self {
            schema: XyzCsvSchema::new(
                vec![
                    ColumnType::X, ColumnType::Y, ColumnType::Z,
                    ColumnType::NormalX, ColumnType::NormalY, ColumnType::NormalZ,
                ],
                true,
                Delimiter::Comma,
            ),
            include_header: true,
        }
    }
    
    /// Create options for CSV with all attributes
    pub fn csv_complete() -> Self {
        Self {
            schema: XyzCsvSchema::new(
                vec![
                    ColumnType::X, ColumnType::Y, ColumnType::Z,
                    ColumnType::Intensity,
                    ColumnType::Red, ColumnType::Green, ColumnType::Blue,
                    ColumnType::NormalX, ColumnType::NormalY, ColumnType::NormalZ,
                ],
                true,
                Delimiter::Comma,
            ),
            include_header: true,
        }
    }
}

// Implement the registry traits
impl crate::registry::PointCloudReader for XyzCsvReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        Self::read_point_cloud(path)
    }
    
    fn can_read(&self, path: &Path) -> bool {
        // Check file extension
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            matches!(ext.to_lowercase().as_str(), "xyz" | "csv" | "txt")
        } else {
            false
        }
    }
    
    fn format_name(&self) -> &'static str {
        "xyz_csv"
    }
}

impl crate::registry::PointCloudWriter for XyzCsvWriter {
    fn write_point_cloud(&self, cloud: &PointCloud<Point3f>, path: &Path) -> Result<()> {
        // Auto-detect format based on extension
        let options = if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext.to_lowercase().as_str() {
                "xyz" => XyzCsvWriteOptions::xyz(),
                "csv" => XyzCsvWriteOptions::csv_with_header(),
                _ => XyzCsvWriteOptions::xyz(),
            }
        } else {
            XyzCsvWriteOptions::xyz()
        };
        
        Self::write_point_cloud(cloud, path, &options)
    }
    
    fn format_name(&self) -> &'static str {
        "xyz_csv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_delimiter_detection() {
        assert_eq!(Delimiter::detect_from_line("1,2,3"), Some(Delimiter::Comma));
        assert_eq!(Delimiter::detect_from_line("1 2 3"), Some(Delimiter::Space));
        assert_eq!(Delimiter::detect_from_line("1\t2\t3"), Some(Delimiter::Tab));
        assert_eq!(Delimiter::detect_from_line("1;2;3"), Some(Delimiter::Semicolon));
    }
    
    #[test]
    fn test_column_type_detection() {
        assert_eq!(ColumnType::from_header("x"), ColumnType::X);
        assert_eq!(ColumnType::from_header("X"), ColumnType::X);
        assert_eq!(ColumnType::from_header("position_x"), ColumnType::X);
        assert_eq!(ColumnType::from_header("intensity"), ColumnType::Intensity);
        assert_eq!(ColumnType::from_header("red"), ColumnType::Red);
        assert_eq!(ColumnType::from_header("nx"), ColumnType::NormalX);
        assert_eq!(ColumnType::from_header("unknown"), ColumnType::Unknown);
    }
    
    #[test]
    fn test_xyz_reader_basic() {
        let temp_file = "test_basic.xyz";
        let content = "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n";
        fs::write(temp_file, content).unwrap();
        
        let cloud = XyzCsvReader::read_point_cloud(temp_file).unwrap();
        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud[0], Point3f::new(1.0, 2.0, 3.0));
        assert_eq!(cloud[1], Point3f::new(4.0, 5.0, 6.0));
        assert_eq!(cloud[2], Point3f::new(7.0, 8.0, 9.0));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_csv_reader_with_header() {
        let temp_file = "test_header.csv";
        let content = "x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n";
        fs::write(temp_file, content).unwrap();
        
        let cloud = XyzCsvReader::read_point_cloud(temp_file).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud[0], Point3f::new(1.0, 2.0, 3.0));
        assert_eq!(cloud[1], Point3f::new(4.0, 5.0, 6.0));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_csv_reader_with_colors() {
        let temp_file = "test_colors.csv";
        let content = "x,y,z,r,g,b\n1.0,2.0,3.0,255,0,0\n4.0,5.0,6.0,0,255,0\n";
        fs::write(temp_file, content).unwrap();
        
        let points = XyzCsvReader::read_detailed_points(temp_file).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].position, Point3f::new(1.0, 2.0, 3.0));
        assert_eq!(points[0].color, Some([255, 0, 0]));
        assert_eq!(points[1].position, Point3f::new(4.0, 5.0, 6.0));
        assert_eq!(points[1].color, Some([0, 255, 0]));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_csv_reader_with_normals() {
        let temp_file = "test_normals.csv";
        let content = "x,y,z,nx,ny,nz\n1.0,2.0,3.0,0.0,0.0,1.0\n4.0,5.0,6.0,0.0,1.0,0.0\n";
        fs::write(temp_file, content).unwrap();
        
        let points = XyzCsvReader::read_detailed_points(temp_file).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].position, Point3f::new(1.0, 2.0, 3.0));
        assert_eq!(points[0].normal, Some(Vector3f::new(0.0, 0.0, 1.0)));
        assert_eq!(points[1].position, Point3f::new(4.0, 5.0, 6.0));
        assert_eq!(points[1].normal, Some(Vector3f::new(0.0, 1.0, 0.0)));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_xyz_writer() {
        let temp_file = "test_write.xyz";
        let cloud = PointCloud::from_points(vec![
            Point3f::new(1.0, 2.0, 3.0),
            Point3f::new(4.0, 5.0, 6.0),
        ]);
        
        let options = XyzCsvWriteOptions::xyz();
        XyzCsvWriter::write_point_cloud(&cloud, temp_file, &options).unwrap();
        
        let content = fs::read_to_string(temp_file).unwrap();
        assert!(content.contains("1 2 3"));
        assert!(content.contains("4 5 6"));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_csv_writer_with_header() {
        let temp_file = "test_write_header.csv";
        let cloud = PointCloud::from_points(vec![
            Point3f::new(1.0, 2.0, 3.0),
            Point3f::new(4.0, 5.0, 6.0),
        ]);
        
        let options = XyzCsvWriteOptions::csv_with_header();
        XyzCsvWriter::write_point_cloud(&cloud, temp_file, &options).unwrap();
        
        let content = fs::read_to_string(temp_file).unwrap();
        assert!(content.starts_with("x,y,z"));
        assert!(content.contains("1,2,3"));
        assert!(content.contains("4,5,6"));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_detailed_points_writer() {
        let temp_file = "test_detailed.csv";
        let points = vec![
            XyzCsvPoint::with_color(Point3f::new(1.0, 2.0, 3.0), [255, 0, 0]),
            XyzCsvPoint::with_intensity(Point3f::new(4.0, 5.0, 6.0), 0.8),
        ];
        
        let options = XyzCsvWriteOptions::csv_complete();
        XyzCsvWriter::write_detailed_points(&points, temp_file, &options).unwrap();
        
        let content = fs::read_to_string(temp_file).unwrap();
        assert!(content.starts_with("x,y,z,intensity,r,g,b,nx,ny,nz"));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_schema_detection() {
        let temp_file = "test_schema.csv";
        let content = "x,y,z,intensity\n1.0,2.0,3.0,0.5\n4.0,5.0,6.0,0.8\n";
        fs::write(temp_file, content).unwrap();
        
        let schema = XyzCsvSchema::detect_from_file(temp_file).unwrap();
        assert_eq!(schema.delimiter, Delimiter::Comma);
        assert!(schema.has_header);
        assert!(schema.columns.contains(&ColumnType::X));
        assert!(schema.columns.contains(&ColumnType::Y));
        assert!(schema.columns.contains(&ColumnType::Z));
        assert!(schema.columns.contains(&ColumnType::Intensity));
        
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_error_handling() {
        // Test missing coordinates
        let temp_file = "test_error.xyz";
        let content = "1.0 2.0\n"; // Missing z coordinate
        fs::write(temp_file, content).unwrap();
        
        let result = XyzCsvReader::read_point_cloud(temp_file);
        assert!(result.is_err());
        
        let _ = fs::remove_file(temp_file);
    }
    
    #[test]
    fn test_registry_traits() {
        use crate::registry::{PointCloudReader, PointCloudWriter};
        
        let reader = XyzCsvReader;
        let writer = XyzCsvWriter;
        
        assert_eq!(reader.format_name(), "xyz_csv");
        assert_eq!(writer.format_name(), "xyz_csv");
        
        // Test can_read
        assert!(reader.can_read(Path::new("test.xyz")));
        assert!(reader.can_read(Path::new("test.csv")));
        assert!(reader.can_read(Path::new("test.txt")));
        assert!(!reader.can_read(Path::new("test.ply")));
    }
}
