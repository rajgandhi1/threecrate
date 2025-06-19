//! OBJ format support

use crate::{MeshReader, MeshWriter};
use threecrate_core::{TriangleMesh, Result};
use std::path::Path;

pub struct ObjReader;
pub struct ObjWriter;

impl MeshReader for ObjReader {
    fn read_mesh<P: AsRef<Path>>(_path: P) -> Result<TriangleMesh> {
        // TODO: Implement OBJ mesh reading
        todo!("OBJ mesh reading not yet implemented")
    }
}

impl MeshWriter for ObjWriter {
    fn write_mesh<P: AsRef<Path>>(_mesh: &TriangleMesh, _path: P) -> Result<()> {
        // TODO: Implement OBJ mesh writing
        todo!("OBJ mesh writing not yet implemented")
    }
} 