//! Example demonstrating OBJ file reading with material support
//! 
//! This example shows how to:
//! - Read OBJ files with vertices, normals, texture coordinates
//! - Parse MTL material files with texture maps
//! - Handle groups and materials
//! - Convert to triangle meshes
//! - Access detailed OBJ data structures

use threecrate_io::{RobustObjReader, ObjData, read_mesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample OBJ file for demonstration
    create_sample_obj_files()?;
    
    println!("=== OBJ Reader Example ===\n");
    
    // Method 1: Simple mesh reading using the standard interface
    println!("1. Reading OBJ as TriangleMesh using standard interface:");
    let mesh = read_mesh("sample.obj")?;
    println!("   Loaded mesh with {} vertices and {} faces", mesh.vertex_count(), mesh.face_count());
    if mesh.normals.is_some() {
        println!("   Mesh includes vertex normals");
    }
    println!();
    
    // Method 2: Detailed OBJ reading with full access to materials and groups
    println!("2. Reading OBJ with detailed access using RobustObjReader:");
    let obj_data = RobustObjReader::read_obj_file("sample.obj")?;
    
    print_obj_summary(&obj_data);
    print_material_details(&obj_data);
    print_group_details(&obj_data);
    
    // Method 3: Converting detailed OBJ data to mesh
    println!("3. Converting ObjData to TriangleMesh:");
    let converted_mesh = RobustObjReader::obj_data_to_mesh(&obj_data)?;
    println!("   Converted mesh: {} vertices, {} faces", 
             converted_mesh.vertex_count(), converted_mesh.face_count());
    println!();
    
    // Method 4: Reading just vertices (useful for point clouds)
    println!("4. Reading vertices only:");
    let vertices = threecrate_io::obj::read_obj_vertices("sample.obj")?;
    println!("   Loaded {} vertices", vertices.len());
    for (i, vertex) in vertices.iter().take(3).enumerate() {
        println!("   Vertex {}: ({:.2}, {:.2}, {:.2})", i, vertex.x, vertex.y, vertex.z);
    }
    println!();
    
    // Method 5: Reading MTL file directly
    println!("5. Reading MTL file directly:");
    let materials = RobustObjReader::read_mtl_file("sample.mtl")?;
    println!("   Loaded {} materials:", materials.len());
    for (name, material) in &materials {
        println!("   - {}: {:?}", name, material.diffuse);
    }
    println!();
    
    // Cleanup
    cleanup_sample_files()?;
    
    println!("Example completed successfully!");
    Ok(())
}

fn create_sample_obj_files() -> Result<(), Box<dyn std::error::Error>> {
    // Create MTL file
    let mtl_content = r#"# Sample MTL file
newmtl red_plastic
Ka 0.2 0.0 0.0
Kd 0.8 0.1 0.1
Ks 0.9 0.9 0.9
Ns 32.0
d 1.0
illum 2
map_Kd red_texture.jpg

newmtl blue_metal
Ka 0.0 0.1 0.2
Kd 0.1 0.3 0.8
Ks 0.9 0.9 0.9
Ns 96.0
d 1.0
illum 2

newmtl green_glass
Ka 0.0 0.2 0.0
Kd 0.0 0.6 0.2
Ks 1.0 1.0 1.0
Ns 128.0
d 0.7
illum 4
"#;
    
    std::fs::write("sample.mtl", mtl_content)?;
    
    // Create OBJ file with various features
    let obj_content = r#"# Sample OBJ file demonstrating various features
mtllib sample.mtl

# Vertices
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 0.866 0.0
v 0.5 0.289 0.816
v 2.0 0.0 0.0
v 2.5 0.866 0.0
v 1.0 1.0 0.0
v 1.5 1.0 0.0

# Texture coordinates
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 1.0
vt 0.5 0.5
vt 0.0 1.0
vt 1.0 1.0

# Vertex normals
vn 0.0 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 1.0
vn 0.577 0.577 0.577
vn -0.577 0.577 0.577
vn 0.0 1.0 0.0

# Group 1: Red triangular pyramid
g pyramid
usemtl red_plastic
f 1/1/1 2/2/2 3/3/3
f 1/1/4 4/4/4 2/2/4
f 2/2/4 4/4/4 3/3/4
f 3/3/4 4/4/4 1/1/4

# Group 2: Blue quad (will be triangulated)
g quad
usemtl blue_metal
f 5/1/1 6/2/2 8/5/6 7/4/6

# Group 3: Green triangle with transparency
g glass_triangle
usemtl green_glass
f 2/2/1 5/1/1 7/4/6
"#;
    
    std::fs::write("sample.obj", obj_content)?;
    
    Ok(())
}

fn print_obj_summary(obj_data: &ObjData) {
    println!("   OBJ Summary:");
    println!("   - {} vertices", obj_data.vertices.len());
    println!("   - {} texture coordinates", obj_data.texture_coords.len());
    println!("   - {} normals", obj_data.normals.len());
    println!("   - {} groups", obj_data.groups.len());
    println!("   - {} materials", obj_data.materials.len());
    println!("   - MTL files: {:?}", obj_data.mtl_files);
    println!();
}

fn print_material_details(obj_data: &ObjData) {
    if !obj_data.materials.is_empty() {
        println!("   Material Details:");
        for (name, material) in &obj_data.materials {
            println!("   - {}:", name);
            if let Some(diffuse) = material.diffuse {
                println!("     Diffuse: ({:.2}, {:.2}, {:.2})", diffuse[0], diffuse[1], diffuse[2]);
            }
            if let Some(specular) = material.specular {
                println!("     Specular: ({:.2}, {:.2}, {:.2})", specular[0], specular[1], specular[2]);
            }
            if let Some(shininess) = material.shininess {
                println!("     Shininess: {:.1}", shininess);
            }
            if let Some(transparency) = material.transparency {
                println!("     Transparency: {:.2}", transparency);
            }
            if let Some(ref diffuse_map) = material.diffuse_map {
                println!("     Diffuse Map: {}", diffuse_map);
            }
            if let Some(ref normal_map) = material.normal_map {
                println!("     Normal Map: {}", normal_map);
            }
        }
        println!();
    }
}

fn print_group_details(obj_data: &ObjData) {
    println!("   Group Details:");
    for (i, group) in obj_data.groups.iter().enumerate() {
        println!("   - Group {}: '{}' ({} faces)", i, group.name, group.faces.len());
        
        // Show material usage in this group
        let mut materials_used = std::collections::HashSet::new();
        for face in &group.faces {
            if let Some(ref material) = face.material {
                materials_used.insert(material);
            }
        }
        
        if !materials_used.is_empty() {
            println!("     Materials used: {:?}", materials_used.iter().collect::<Vec<_>>());
        }
        
        // Show face details for first few faces
        for (j, face) in group.faces.iter().take(2).enumerate() {
            println!("     Face {}: {} vertices", j, face.vertices.len());
            for (k, vertex) in face.vertices.iter().enumerate() {
                print!("       Vertex {}: pos={}", k, vertex.vertex);
                if let Some(tex) = vertex.texture {
                    print!(", tex={}", tex);
                }
                if let Some(norm) = vertex.normal {
                    print!(", norm={}", norm);
                }
                println!();
            }
        }
        
        if group.faces.len() > 2 {
            println!("     ... and {} more faces", group.faces.len() - 2);
        }
    }
    println!();
}

fn cleanup_sample_files() -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::fs::remove_file("sample.obj");
    let _ = std::fs::remove_file("sample.mtl");
    Ok(())
}
