//! OBJ format support

use crate::{MeshReader, MeshWriter};
use threecrate_core::{TriangleMesh, Result, Point3f, Vector3f};
use std::path::Path;
use std::fs::File;
use std::io::{BufWriter, Write};
use obj::Obj;

pub struct ObjReader;
pub struct ObjWriter;

impl MeshReader for ObjReader {
    fn read_mesh<P: AsRef<Path>>(path: P) -> Result<TriangleMesh> {
        // Parse OBJ file directly from path
        let obj: Obj = Obj::load(path)
            .map_err(|e| threecrate_core::Error::InvalidData(
                format!("Failed to parse OBJ file: {:?}", e)
            ))?;
        
        let obj_data = &obj.data;
        
        // Extract vertices
        let vertices: Vec<Point3f> = obj_data.position
            .iter()
            .map(|pos| Point3f::new(pos[0], pos[1], pos[2]))
            .collect();
        
        // Extract faces (triangulate if necessary)
        let mut faces = Vec::new();
        for object in &obj_data.objects {
            for group in &object.groups {
                for poly in &group.polys {
                    match poly.0.len() {
                        3 => {
                            // Triangle - direct conversion
                            let face = [
                                poly.0[0].0,
                                poly.0[1].0,
                                poly.0[2].0,
                            ];
                            faces.push(face);
                        }
                        4 => {
                            // Quad - split into two triangles
                            let face1 = [
                                poly.0[0].0,
                                poly.0[1].0,
                                poly.0[2].0,
                            ];
                            let face2 = [
                                poly.0[0].0,
                                poly.0[2].0,
                                poly.0[3].0,
                            ];
                            faces.push(face1);
                            faces.push(face2);
                        }
                        n if n > 4 => {
                            // N-gon - fan triangulation
                            for i in 1..(n - 1) {
                                let face = [
                                    poly.0[0].0,
                                    poly.0[i].0,
                                    poly.0[i + 1].0,
                                ];
                                faces.push(face);
                            }
                        }
                        _ => {
                            // Skip degenerate polygons
                            continue;
                        }
                    }
                }
            }
        }
        
        // Extract normals if available
        let normals = if !obj_data.normal.is_empty() {
            let normals: Vec<Vector3f> = obj_data.normal
                .iter()
                .map(|norm| Vector3f::new(norm[0], norm[1], norm[2]))
                .collect();
            Some(normals)
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

impl MeshWriter for ObjWriter {
    fn write_mesh<P: AsRef<Path>>(mesh: &TriangleMesh, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        writeln!(writer, "# OBJ file generated by ThreeCrate")?;
        writeln!(writer, "# Vertices: {}", mesh.vertices.len())?;
        writeln!(writer, "# Faces: {}", mesh.faces.len())?;
        writeln!(writer)?;
        
        // Write vertices
        for vertex in &mesh.vertices {
            writeln!(writer, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
        }
        writeln!(writer)?;
        
        // Write normals if available
        if let Some(normals) = &mesh.normals {
            for normal in normals {
                writeln!(writer, "vn {} {} {}", normal.x, normal.y, normal.z)?;
            }
            writeln!(writer)?;
        }
        
        // Write faces
        if mesh.normals.is_some() {
            // Write faces with normals
            for face in &mesh.faces {
                writeln!(
                    writer,
                    "f {}//{} {}//{} {}//{}",
                    face[0] + 1, face[0] + 1,
                    face[1] + 1, face[1] + 1,
                    face[2] + 1, face[2] + 1
                )?;
            }
        } else {
            // Write faces without normals
            for face in &mesh.faces {
                writeln!(
                    writer,
                    "f {} {} {}",
                    face[0] + 1,
                    face[1] + 1,
                    face[2] + 1
                )?;
            }
        }
        
        Ok(())
    }
}

/// Read an OBJ file and return vertex positions only (useful for point clouds)
pub fn read_obj_vertices<P: AsRef<Path>>(path: P) -> Result<Vec<Point3f>> {
    let obj: Obj = Obj::load(path)
        .map_err(|e| threecrate_core::Error::InvalidData(
            format!("Failed to parse OBJ file: {:?}", e)
        ))?;
    
    let vertices: Vec<Point3f> = obj.data.position
        .iter()
        .map(|pos| Point3f::new(pos[0], pos[1], pos[2]))
        .collect();
    
    Ok(vertices)
}

/// Write vertices as an OBJ file (useful for point clouds)
pub fn write_obj_vertices<P: AsRef<Path>>(vertices: &[Point3f], path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "# OBJ vertices file generated by ThreeCrate")?;
    writeln!(writer, "# Vertices: {}", vertices.len())?;
    writeln!(writer)?;
    
    for vertex in vertices {
        writeln!(writer, "v {} {} {}", vertex.x, vertex.y, vertex.z)?;
    }
    
    Ok(())
} 