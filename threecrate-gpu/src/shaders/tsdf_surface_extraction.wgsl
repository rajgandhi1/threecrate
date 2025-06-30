struct TsdfVoxel {
    tsdf_value: f32,
    weight: f32,
    color_r: u32,
    color_g: u32,
    color_b: u32,
    _padding: u32,
}

struct Point3f {
    x: f32,
    y: f32,
    z: f32,
}

struct TsdfParams {
    voxel_size: f32,
    truncation_distance: f32,
    max_weight: f32,
    _padding: f32,
    resolution: vec3<u32>,
    _padding2: u32,
    origin: vec3<f32>,
    _padding3: f32,
}

@group(0) @binding(0) var<storage, read> voxels: array<TsdfVoxel>;
@group(0) @binding(1) var<storage, read_write> vertices: array<Point3f>;
@group(0) @binding(2) var<storage, read_write> vertex_counter: atomic<u32>;
@group(0) @binding(3) var<uniform> params: TsdfParams;
@group(0) @binding(4) var<uniform> iso_value: f32;

fn get_voxel_index(coord: vec3<u32>) -> u32 {
    return coord.z * params.resolution.x * params.resolution.y + 
           coord.y * params.resolution.x + 
           coord.x;
}

fn get_voxel_value(coord: vec3<u32>) -> f32 {
    if coord.x >= params.resolution.x || 
       coord.y >= params.resolution.y || 
       coord.z >= params.resolution.z {
        return 1.0; // Outside bounds, assume positive (outside surface)
    }
    
    let index = get_voxel_index(coord);
    if index >= arrayLength(&voxels) {
        return 1.0;
    }
    
    let voxel = voxels[index];
    if voxel.weight < 1.0 {
        return 1.0; // Insufficient weight, assume outside surface
    }
    
    return voxel.tsdf_value;
}

fn voxel_to_world(voxel_coord: vec3<f32>) -> vec3<f32> {
    return params.origin + voxel_coord * params.voxel_size;
}

fn interpolate_vertex(p1: vec3<f32>, p2: vec3<f32>, val1: f32, val2: f32) -> vec3<f32> {
    if abs(val1 - val2) < 0.00001 {
        return p1;
    }
    
    let t = (iso_value - val1) / (val2 - val1);
    return p1 + t * (p2 - p1);
}

fn add_vertex(vertex: vec3<f32>) {
    let index = atomicAdd(&vertex_counter, 1u);
    if index < arrayLength(&vertices) {
        vertices[index] = Point3f(vertex.x, vertex.y, vertex.z);
    }
}

// Simplified marching cubes - check for zero crossings and generate vertices
@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_coord = global_id;
    
    // Check bounds - we need to check a cube, so stop one voxel before the edge
    if voxel_coord.x >= params.resolution.x - 1u || 
       voxel_coord.y >= params.resolution.y - 1u || 
       voxel_coord.z >= params.resolution.z - 1u {
        return;
    }
    
    // Get the 8 corner values of the current voxel cube
    let v000 = get_voxel_value(voxel_coord + vec3<u32>(0u, 0u, 0u));
    let v001 = get_voxel_value(voxel_coord + vec3<u32>(0u, 0u, 1u));
    let v010 = get_voxel_value(voxel_coord + vec3<u32>(0u, 1u, 0u));
    let v011 = get_voxel_value(voxel_coord + vec3<u32>(0u, 1u, 1u));
    let v100 = get_voxel_value(voxel_coord + vec3<u32>(1u, 0u, 0u));
    let v101 = get_voxel_value(voxel_coord + vec3<u32>(1u, 0u, 1u));
    let v110 = get_voxel_value(voxel_coord + vec3<u32>(1u, 1u, 0u));
    let v111 = get_voxel_value(voxel_coord + vec3<u32>(1u, 1u, 1u));
    
    // World positions of the 8 corners
    let base_coord = vec3<f32>(voxel_coord);
    let p000 = voxel_to_world(base_coord + vec3<f32>(0.0, 0.0, 0.0));
    let p001 = voxel_to_world(base_coord + vec3<f32>(0.0, 0.0, 1.0));
    let p010 = voxel_to_world(base_coord + vec3<f32>(0.0, 1.0, 0.0));
    let p011 = voxel_to_world(base_coord + vec3<f32>(0.0, 1.0, 1.0));
    let p100 = voxel_to_world(base_coord + vec3<f32>(1.0, 0.0, 0.0));
    let p101 = voxel_to_world(base_coord + vec3<f32>(1.0, 0.0, 1.0));
    let p110 = voxel_to_world(base_coord + vec3<f32>(1.0, 1.0, 0.0));
    let p111 = voxel_to_world(base_coord + vec3<f32>(1.0, 1.0, 1.0));
    
    // Check for zero crossings along the 12 edges of the cube
    // and create vertices where the surface intersects the edges
    
    // Edge 0-1 (bottom face, front edge)
    if (v000 <= iso_value) != (v100 <= iso_value) {
        let vertex = interpolate_vertex(p000, p100, v000, v100);
        add_vertex(vertex);
    }
    
    // Edge 1-3 (bottom face, right edge)
    if (v100 <= iso_value) != (v110 <= iso_value) {
        let vertex = interpolate_vertex(p100, p110, v100, v110);
        add_vertex(vertex);
    }
    
    // Edge 3-2 (bottom face, back edge)
    if (v110 <= iso_value) != (v010 <= iso_value) {
        let vertex = interpolate_vertex(p110, p010, v110, v010);
        add_vertex(vertex);
    }
    
    // Edge 2-0 (bottom face, left edge)
    if (v010 <= iso_value) != (v000 <= iso_value) {
        let vertex = interpolate_vertex(p010, p000, v010, v000);
        add_vertex(vertex);
    }
    
    // Edge 4-5 (top face, front edge)
    if (v001 <= iso_value) != (v101 <= iso_value) {
        let vertex = interpolate_vertex(p001, p101, v001, v101);
        add_vertex(vertex);
    }
    
    // Edge 5-7 (top face, right edge)
    if (v101 <= iso_value) != (v111 <= iso_value) {
        let vertex = interpolate_vertex(p101, p111, v101, v111);
        add_vertex(vertex);
    }
    
    // Edge 7-6 (top face, back edge)
    if (v111 <= iso_value) != (v011 <= iso_value) {
        let vertex = interpolate_vertex(p111, p011, v111, v011);
        add_vertex(vertex);
    }
    
    // Edge 6-4 (top face, left edge)
    if (v011 <= iso_value) != (v001 <= iso_value) {
        let vertex = interpolate_vertex(p011, p001, v011, v001);
        add_vertex(vertex);
    }
    
    // Vertical edges
    // Edge 0-4 (front-left vertical)
    if (v000 <= iso_value) != (v001 <= iso_value) {
        let vertex = interpolate_vertex(p000, p001, v000, v001);
        add_vertex(vertex);
    }
    
    // Edge 1-5 (front-right vertical)
    if (v100 <= iso_value) != (v101 <= iso_value) {
        let vertex = interpolate_vertex(p100, p101, v100, v101);
        add_vertex(vertex);
    }
    
    // Edge 2-6 (back-left vertical)
    if (v010 <= iso_value) != (v011 <= iso_value) {
        let vertex = interpolate_vertex(p010, p011, v010, v011);
        add_vertex(vertex);
    }
    
    // Edge 3-7 (back-right vertical)
    if (v110 <= iso_value) != (v111 <= iso_value) {
        let vertex = interpolate_vertex(p110, p111, v110, v111);
        add_vertex(vertex);
    }
} 