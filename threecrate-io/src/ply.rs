//! PLY format support

use crate::{PointCloudReader, PointCloudWriter, MeshReader, MeshWriter};
use threecrate_core::{PointCloud, TriangleMesh, Result, Point3f};
use std::path::Path;

pub struct PlyReader;
pub struct PlyWriter;

impl PointCloudReader for PlyReader {
    fn read_point_cloud<P: AsRef<Path>>(_path: P) -> Result<PointCloud<Point3f>> {
        // TODO: Implement PLY point cloud reading
        todo!("PLY point cloud reading not yet implemented")
    }
}

impl PointCloudWriter for PlyWriter {
    fn write_point_cloud<P: AsRef<Path>>(_cloud: &PointCloud<Point3f>, _path: P) -> Result<()> {
        // TODO: Implement PLY point cloud writing
        todo!("PLY point cloud writing not yet implemented")
    }
}

impl MeshReader for PlyReader {
    fn read_mesh<P: AsRef<Path>>(_path: P) -> Result<TriangleMesh> {
        // TODO: Implement PLY mesh reading
        todo!("PLY mesh reading not yet implemented")
    }
}

impl MeshWriter for PlyWriter {
    fn write_mesh<P: AsRef<Path>>(_mesh: &TriangleMesh, _path: P) -> Result<()> {
        // TODO: Implement PLY mesh writing
        todo!("PLY mesh writing not yet implemented")
    }
}