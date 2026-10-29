//! Unified IO registry for format-agnostic reading and writing
//!
//! This module provides a registry-based approach to IO operations,
//! allowing downstream crates to work with any supported format
//! without knowing the specific implementation details.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use threecrate_core::{Point3f, PointCloud, Result, TriangleMesh};

/// Trait for reading point clouds from files
pub trait PointCloudReader: Send + Sync {
    /// Read a point cloud from the given path
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>>;

    /// Check if this reader can handle the given file by examining its header
    fn can_read(&self, path: &Path) -> bool;

    /// Get the format name this reader handles
    fn format_name(&self) -> &'static str;
}

/// Trait for writing point clouds to files
pub trait PointCloudWriter: Send + Sync {
    /// Write a point cloud to the given path
    fn write_point_cloud(&self, cloud: &PointCloud<Point3f>, path: &Path) -> Result<()>;

    /// Get the format name this writer handles
    fn format_name(&self) -> &'static str;
}

/// Trait for reading meshes from files
pub trait MeshReader: Send + Sync {
    /// Read a mesh from the given path
    fn read_mesh(&self, path: &Path) -> Result<TriangleMesh>;

    /// Check if this reader can handle the given file by examining its header
    fn can_read(&self, path: &Path) -> bool;

    /// Get the format name this reader handles
    fn format_name(&self) -> &'static str;
}

/// Trait for writing meshes to files
pub trait MeshWriter: Send + Sync {
    /// Write a mesh to the given path
    fn write_mesh(&self, mesh: &TriangleMesh, path: &Path) -> Result<()>;

    /// Get the format name this writer handles
    fn format_name(&self) -> &'static str;
}

/// Format handler that can read and write both point clouds and meshes
pub trait FormatHandler:
    PointCloudReader + PointCloudWriter + MeshReader + MeshWriter + Send + Sync
{
    /// Get the file extensions this handler supports
    fn supported_extensions(&self) -> &[&'static str];

    /// Get the magic bytes/header signature for format detection
    fn magic_bytes(&self) -> &[u8];
}

/// IO registry that manages format handlers and provides unified access
pub struct IoRegistry {
    point_cloud_readers: HashMap<String, Box<dyn PointCloudReader>>,
    point_cloud_writers: HashMap<String, Box<dyn PointCloudWriter>>,
    mesh_readers: HashMap<String, Box<dyn MeshReader>>,
    mesh_writers: HashMap<String, Box<dyn MeshWriter>>,
}

impl IoRegistry {
    /// Create a new empty IO registry
    pub fn new() -> Self {
        Self {
            point_cloud_readers: HashMap::new(),
            point_cloud_writers: HashMap::new(),
            mesh_readers: HashMap::new(),
            mesh_writers: HashMap::new(),
        }
    }

    /// Register a point cloud reader for a specific format
    pub fn register_point_cloud_handler(
        &mut self,
        format: &str,
        handler: Box<dyn PointCloudReader>,
    ) {
        self.point_cloud_readers
            .insert(format.to_lowercase(), handler);
    }

    /// Register a point cloud writer for a specific format
    pub fn register_point_cloud_writer(
        &mut self,
        format: &str,
        handler: Box<dyn PointCloudWriter>,
    ) {
        self.point_cloud_writers
            .insert(format.to_lowercase(), handler);
    }

    /// Register a mesh reader for a specific format
    pub fn register_mesh_handler(&mut self, format: &str, handler: Box<dyn MeshReader>) {
        self.mesh_readers.insert(format.to_lowercase(), handler);
    }

    /// Register a mesh writer for a specific format
    pub fn register_mesh_writer(&mut self, format: &str, handler: Box<dyn MeshWriter>) {
        self.mesh_writers.insert(format.to_lowercase(), handler);
    }

    /// Register a handler as the reader and writer of points and meshes for
    /// each of its `supported_extensions`.
    pub fn register_format_handler(&mut self, handler: Box<dyn FormatHandler>) {
        let shared: Arc<dyn FormatHandler> = Arc::from(handler);
        for ext in shared.supported_extensions() {
            self.register_point_cloud_handler(ext, Box::new(Shared(shared.clone())));
            self.register_point_cloud_writer(ext, Box::new(Shared(shared.clone())));
            self.register_mesh_handler(ext, Box::new(Shared(shared.clone())));
            self.register_mesh_writer(ext, Box::new(Shared(shared.clone())));
        }
    }

    /// Read a point cloud, auto-detecting the format
    pub fn read_point_cloud(&self, path: &Path, format_hint: &str) -> Result<PointCloud<Point3f>> {
        // First try the format hint
        if let Some(reader) = self.point_cloud_readers.get(&format_hint.to_lowercase()) {
            if reader.can_read(path) {
                return reader.read_point_cloud(path);
            }
        }

        // Try to detect format by header signature
        if let Some(detected_format) = self.detect_format_by_header(path) {
            if let Some(reader) = self.point_cloud_readers.get(&detected_format) {
                return reader.read_point_cloud(path);
            }
        }

        // Fall back to extension-based detection
        if let Some(reader) = self.point_cloud_readers.get(&format_hint.to_lowercase()) {
            return reader.read_point_cloud(path);
        }

        Err(threecrate_core::Error::UnsupportedFormat(format!(
            "No point cloud reader found for format: {}",
            format_hint
        )))
    }

    /// Read a mesh, auto-detecting the format. Faces that point at missing
    /// vertices are an error, whichever reader ran.
    pub fn read_mesh(&self, path: &Path, format_hint: &str) -> Result<TriangleMesh> {
        let mesh = self.read_mesh_unchecked(path, format_hint)?;
        mesh.validate_faces()?;
        Ok(mesh)
    }

    fn read_mesh_unchecked(&self, path: &Path, format_hint: &str) -> Result<TriangleMesh> {
        // First try the format hint
        if let Some(reader) = self.mesh_readers.get(&format_hint.to_lowercase()) {
            if reader.can_read(path) {
                return reader.read_mesh(path);
            }
        }

        // Try to detect format by header signature
        if let Some(detected_format) = self.detect_format_by_header(path) {
            if let Some(reader) = self.mesh_readers.get(&detected_format) {
                return reader.read_mesh(path);
            }
        }

        // Fall back to extension-based detection
        if let Some(reader) = self.mesh_readers.get(&format_hint.to_lowercase()) {
            return reader.read_mesh(path);
        }

        Err(threecrate_core::Error::UnsupportedFormat(format!(
            "No mesh reader found for format: {}",
            format_hint
        )))
    }

    /// Write a point cloud, auto-detecting the format
    pub fn write_point_cloud(
        &self,
        cloud: &PointCloud<Point3f>,
        path: &Path,
        format_hint: &str,
    ) -> Result<()> {
        if let Some(writer) = self.point_cloud_writers.get(&format_hint.to_lowercase()) {
            return writer.write_point_cloud(cloud, path);
        }

        Err(threecrate_core::Error::UnsupportedFormat(format!(
            "No point cloud writer found for format: {}",
            format_hint
        )))
    }

    /// Write a mesh, auto-detecting the format
    pub fn write_mesh(&self, mesh: &TriangleMesh, path: &Path, format_hint: &str) -> Result<()> {
        if let Some(writer) = self.mesh_writers.get(&format_hint.to_lowercase()) {
            return writer.write_mesh(mesh, path);
        }

        Err(threecrate_core::Error::UnsupportedFormat(format!(
            "No mesh writer found for format: {}",
            format_hint
        )))
    }

    /// Detect file format by examining the header/magic bytes
    fn detect_format_by_header(&self, path: &Path) -> Option<String> {
        use std::fs::File;
        use std::io::Read;

        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(_) => return None,
        };

        let mut header = [0u8; 16];
        if let Ok(bytes_read) = file.read(&mut header) {
            if bytes_read < 4 {
                return None;
            }

            // Check magic bytes for various formats
            if header.starts_with(b"ply") {
                return Some("ply".to_string());
            } else if header.starts_with(b"#") || header.starts_with(b"v ") {
                // Check if it's an OBJ file by looking for vertex definitions
                let mut file = File::open(path).ok()?;
                let mut content = String::new();
                if file.read_to_string(&mut content).is_ok() {
                    if content.lines().any(|line| line.trim().starts_with("v ")) {
                        return Some("obj".to_string());
                    }
                }
            } else if header.starts_with(b"LASF") {
                return Some("las".to_string());
            } else if header.starts_with(b"# .PCD") {
                return Some("pcd".to_string());
            } else if header.starts_with(b"ASTM-E57") {
                return Some("e57".to_string());
            }
        }

        None
    }

    /// Get a list of supported formats for point clouds
    pub fn supported_point_cloud_formats(&self) -> Vec<String> {
        self.point_cloud_readers.keys().cloned().collect()
    }

    /// Get a list of supported formats for meshes
    pub fn supported_mesh_formats(&self) -> Vec<String> {
        self.mesh_readers.keys().cloned().collect()
    }

    /// Check if a format is supported for reading point clouds
    pub fn supports_point_cloud_reading(&self, format: &str) -> bool {
        self.point_cloud_readers
            .contains_key(&format.to_lowercase())
    }

    /// Check if a format is supported for writing point clouds
    pub fn supports_point_cloud_writing(&self, format: &str) -> bool {
        self.point_cloud_writers
            .contains_key(&format.to_lowercase())
    }

    /// Check if a format is supported for reading meshes
    pub fn supports_mesh_reading(&self, format: &str) -> bool {
        self.mesh_readers.contains_key(&format.to_lowercase())
    }

    /// Check if a format is supported for writing meshes
    pub fn supports_mesh_writing(&self, format: &str) -> bool {
        self.mesh_writers.contains_key(&format.to_lowercase())
    }
}

impl Default for IoRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// One registered `FormatHandler` shared by its four capability slots.
struct Shared(Arc<dyn FormatHandler>);

impl PointCloudReader for Shared {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        self.0.read_point_cloud(path)
    }
    fn can_read(&self, path: &Path) -> bool {
        PointCloudReader::can_read(&*self.0, path)
    }
    fn format_name(&self) -> &'static str {
        PointCloudReader::format_name(&*self.0)
    }
}

impl PointCloudWriter for Shared {
    fn write_point_cloud(&self, cloud: &PointCloud<Point3f>, path: &Path) -> Result<()> {
        self.0.write_point_cloud(cloud, path)
    }
    fn format_name(&self) -> &'static str {
        PointCloudWriter::format_name(&*self.0)
    }
}

impl MeshReader for Shared {
    fn read_mesh(&self, path: &Path) -> Result<TriangleMesh> {
        self.0.read_mesh(path)
    }
    fn can_read(&self, path: &Path) -> bool {
        MeshReader::can_read(&*self.0, path)
    }
    fn format_name(&self) -> &'static str {
        MeshReader::format_name(&*self.0)
    }
}

impl MeshWriter for Shared {
    fn write_mesh(&self, mesh: &TriangleMesh, path: &Path) -> Result<()> {
        self.0.write_mesh(mesh, path)
    }
    fn format_name(&self) -> &'static str {
        MeshWriter::format_name(&*self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use threecrate_core::Point3f;

    // Mock implementations for testing
    struct MockPlyHandler;

    impl PointCloudReader for MockPlyHandler {
        fn read_point_cloud(&self, _path: &Path) -> Result<PointCloud<Point3f>> {
            let mut cloud = PointCloud::new();
            cloud.push(Point3f::new(0.0, 0.0, 0.0));
            Ok(cloud)
        }

        fn can_read(&self, _path: &Path) -> bool {
            true
        }

        fn format_name(&self) -> &'static str {
            "ply"
        }
    }

    impl PointCloudWriter for MockPlyHandler {
        fn write_point_cloud(&self, _cloud: &PointCloud<Point3f>, _path: &Path) -> Result<()> {
            Ok(())
        }

        fn format_name(&self) -> &'static str {
            "ply"
        }
    }

    impl MeshReader for MockPlyHandler {
        fn read_mesh(&self, _path: &Path) -> Result<TriangleMesh> {
            let vertices = vec![Point3f::new(0.0, 0.0, 0.0)];
            let faces = vec![];
            Ok(TriangleMesh::from_vertices_and_faces(vertices, faces))
        }

        fn can_read(&self, _path: &Path) -> bool {
            true
        }

        fn format_name(&self) -> &'static str {
            "ply"
        }
    }

    impl MeshWriter for MockPlyHandler {
        fn write_mesh(&self, _mesh: &TriangleMesh, _path: &Path) -> Result<()> {
            Ok(())
        }

        fn format_name(&self) -> &'static str {
            "ply"
        }
    }

    impl FormatHandler for MockPlyHandler {
        fn supported_extensions(&self) -> &[&'static str] {
            &["ply"]
        }

        fn magic_bytes(&self) -> &[u8] {
            b"ply"
        }
    }

    impl Clone for MockPlyHandler {
        fn clone(&self) -> Self {
            Self
        }
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = IoRegistry::new();

        // Register handlers
        registry.register_point_cloud_handler("ply", Box::new(MockPlyHandler));
        registry.register_mesh_handler("ply", Box::new(MockPlyHandler));
        registry.register_point_cloud_writer("ply", Box::new(MockPlyHandler));
        registry.register_mesh_writer("ply", Box::new(MockPlyHandler));

        // Check support
        assert!(registry.supports_point_cloud_reading("ply"));
        assert!(registry.supports_mesh_reading("ply"));
        assert!(registry.supports_point_cloud_writing("ply"));
        assert!(registry.supports_mesh_writing("ply"));

        // Check unsupported formats
        assert!(!registry.supports_point_cloud_reading("obj"));
        assert!(!registry.supports_mesh_reading("xyz"));
    }

    #[test]
    fn test_format_detection() {
        let mut registry = IoRegistry::new();
        registry.register_point_cloud_handler("ply", Box::new(MockPlyHandler));
        registry.register_mesh_handler("ply", Box::new(MockPlyHandler));

        // Create a test PLY file
        let temp_file = "test_detection.ply";
        let ply_content = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0.0 0.0 0.0\n";
        fs::write(temp_file, ply_content).unwrap();

        // Test reading with format detection
        let cloud = registry
            .read_point_cloud(Path::new(temp_file), "ply")
            .unwrap();
        assert_eq!(cloud.len(), 1);

        let mesh = registry.read_mesh(Path::new(temp_file), "ply").unwrap();
        assert_eq!(mesh.vertex_count(), 1);

        // Cleanup
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn format_handler_registers_every_capability() {
        let mut registry = IoRegistry::new();
        registry.register_format_handler(Box::new(MockPlyHandler));
        assert!(registry.supports_point_cloud_reading("PLY"));
        assert!(registry.supports_point_cloud_writing("ply"));
        assert!(registry.supports_mesh_reading("ply"));
        assert!(registry.supports_mesh_writing("ply"));
    }

    #[test]
    fn read_mesh_rejects_faces_past_the_vertices() {
        let temp_file = std::env::temp_dir().join("threecrate_registry_bad_face.ply");
        let ply = "ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 9
";
        fs::write(&temp_file, ply).unwrap();
        let mut registry = IoRegistry::new();
        registry.register_mesh_handler("ply", Box::new(crate::ply::PlyReader));
        assert!(registry.read_mesh(&temp_file, "ply").is_err());
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_unsupported_format() {
        let registry = IoRegistry::new();

        let result = registry.read_point_cloud(Path::new("test.xyz"), "xyz");
        assert!(result.is_err());

        let result = registry.read_mesh(Path::new("test.xyz"), "xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_supported_formats_list() {
        let mut registry = IoRegistry::new();
        registry.register_point_cloud_handler("ply", Box::new(MockPlyHandler));
        registry.register_mesh_handler("obj", Box::new(MockPlyHandler));

        let pc_formats = registry.supported_point_cloud_formats();
        let mesh_formats = registry.supported_mesh_formats();

        assert!(pc_formats.contains(&"ply".to_string()));
        assert!(mesh_formats.contains(&"obj".to_string()));
    }
}
