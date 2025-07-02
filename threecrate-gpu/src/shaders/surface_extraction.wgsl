struct TsdfVoxel {
    tsdf_value: f32,
    weight: f32,
    color_r: u32,
    color_g: u32,
    color_b: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

struct Point3f {
    x: f32,
    y: f32,
    z: f32,
    r: u32,
    g: u32,
    b: u32,
    _padding1: u32,
    _padding2: u32,
}

struct TsdfParams {
    voxel_size: f32,
    truncation_distance: f32,
    max_weight: f32,
    iso_value: f32,
    resolution: vec3<u32>,
    _padding2: u32,
    origin: vec3<f32>,
    _padding3: f32,
}

@group(0) @binding(0) var<storage, read> voxels: array<TsdfVoxel>;
@group(0) @binding(1) var<storage, read_write> points: array<Point3f>;
@group(0) @binding(2) var<uniform> params: TsdfParams;
@group(0) @binding(3) var<storage, read_write> point_count: array<atomic<u32>>;

fn get_voxel_index(coord: vec3<u32>) -> u32 {
    return coord.z * params.resolution.x * params.resolution.y + 
           coord.y * params.resolution.x + 
           coord.x;
}

fn get_voxel_value(coord: vec3<u32>) -> f32 {
    if coord.x >= params.resolution.x || 
       coord.y >= params.resolution.y || 
       coord.z >= params.resolution.z {
        return params.truncation_distance; // Outside bounds, assume outside surface
    }
    
    let index = get_voxel_index(coord);
    if index >= arrayLength(&voxels) {
        return params.truncation_distance;
    }
    
    let voxel = voxels[index];
    if voxel.weight < 0.001 { // More lenient weight threshold
        return params.truncation_distance; // Insufficient weight, assume outside surface
    }
    
    // Return TSDF value relative to iso-value
    // Note: TSDF values are positive outside, negative inside
    return voxel.tsdf_value - params.iso_value;
}

fn voxel_to_world(voxel_coord: vec3<f32>) -> vec3<f32> {
    return params.origin + voxel_coord * params.voxel_size;
}

fn add_point(position: vec3<f32>, color: vec3<u32>) {
    let index = atomicAdd(&point_count[0], 1u);
    if index < arrayLength(&points) {
        points[index] = Point3f(
            position.x, position.y, position.z,
            color.x, color.y, color.z,
            0u, 0u
        );
    }
}

fn interpolate_zero_crossing(p1: vec3<f32>, v1: f32, p2: vec3<f32>, v2: f32) -> vec3<f32> {
    // Ensure we don't divide by zero
    if abs(v1 - v2) < 0.00001 {
        return 0.5 * (p1 + p2);
    }
    
    // Linear interpolation to find zero crossing
    // Note: v1 and v2 are TSDF values relative to iso-value
    // Positive outside, negative inside
    let t = v1 / (v1 - v2);
    return p1 + clamp(t, 0.0, 1.0) * (p2 - p1);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_coord = global_id;

    // Skip voxels at volume boundaries
    if voxel_coord.x >= params.resolution.x - 1u ||
       voxel_coord.y >= params.resolution.y - 1u ||
       voxel_coord.z >= params.resolution.z - 1u {
        return;
    }

    let index = get_voxel_index(voxel_coord);
    if index >= arrayLength(&voxels) {
        return;
    }

    let voxel = voxels[index];
    if voxel.weight < 0.001 {
        return;
    }

    // Get values at cube corners (TSDF values relative to iso-value)
    let v000 = get_voxel_value(voxel_coord);
    let v100 = get_voxel_value(voxel_coord + vec3<u32>(1u, 0u, 0u));
    let v010 = get_voxel_value(voxel_coord + vec3<u32>(0u, 1u, 0u));
    let v110 = get_voxel_value(voxel_coord + vec3<u32>(1u, 1u, 0u));
    let v001 = get_voxel_value(voxel_coord + vec3<u32>(0u, 0u, 1u));
    let v101 = get_voxel_value(voxel_coord + vec3<u32>(1u, 0u, 1u));
    let v011 = get_voxel_value(voxel_coord + vec3<u32>(0u, 1u, 1u));
    let v111 = get_voxel_value(voxel_coord + vec3<u32>(1u, 1u, 1u));

    // Check if cube contains surface (zero crossing)
    // Note: Positive outside, negative inside, so multiply values to check for sign change
    let has_surface = (v000 * v100 <= 0.0) || (v010 * v110 <= 0.0) ||
                     (v001 * v101 <= 0.0) || (v011 * v111 <= 0.0) ||
                     (v000 * v010 <= 0.0) || (v100 * v110 <= 0.0) ||
                     (v001 * v011 <= 0.0) || (v101 * v111 <= 0.0) ||
                     (v000 * v001 <= 0.0) || (v100 * v101 <= 0.0) ||
                     (v010 * v011 <= 0.0) || (v110 * v111 <= 0.0);

    if !has_surface {
        return;
    }

    // Get world space positions of cube corners
    let p000 = voxel_to_world(vec3<f32>(voxel_coord));
    let p100 = voxel_to_world(vec3<f32>(voxel_coord) + vec3<f32>(1.0, 0.0, 0.0));
    let p010 = voxel_to_world(vec3<f32>(voxel_coord) + vec3<f32>(0.0, 1.0, 0.0));
    let p110 = voxel_to_world(vec3<f32>(voxel_coord) + vec3<f32>(1.0, 1.0, 0.0));
    let p001 = voxel_to_world(vec3<f32>(voxel_coord) + vec3<f32>(0.0, 0.0, 1.0));
    let p101 = voxel_to_world(vec3<f32>(voxel_coord) + vec3<f32>(1.0, 0.0, 1.0));
    let p011 = voxel_to_world(vec3<f32>(voxel_coord) + vec3<f32>(0.0, 1.0, 1.0));
    let p111 = voxel_to_world(vec3<f32>(voxel_coord) + vec3<f32>(1.0, 1.0, 1.0));

    // Add points at zero crossings
    if v000 * v100 <= 0.0 {
        let p = interpolate_zero_crossing(p000, v000, p100, v100);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v010 * v110 <= 0.0 {
        let p = interpolate_zero_crossing(p010, v010, p110, v110);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v001 * v101 <= 0.0 {
        let p = interpolate_zero_crossing(p001, v001, p101, v101);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v011 * v111 <= 0.0 {
        let p = interpolate_zero_crossing(p011, v011, p111, v111);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v000 * v010 <= 0.0 {
        let p = interpolate_zero_crossing(p000, v000, p010, v010);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v100 * v110 <= 0.0 {
        let p = interpolate_zero_crossing(p100, v100, p110, v110);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v001 * v011 <= 0.0 {
        let p = interpolate_zero_crossing(p001, v001, p011, v011);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v101 * v111 <= 0.0 {
        let p = interpolate_zero_crossing(p101, v101, p111, v111);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v000 * v001 <= 0.0 {
        let p = interpolate_zero_crossing(p000, v000, p001, v001);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v100 * v101 <= 0.0 {
        let p = interpolate_zero_crossing(p100, v100, p101, v101);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v010 * v011 <= 0.0 {
        let p = interpolate_zero_crossing(p010, v010, p011, v011);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
    if v110 * v111 <= 0.0 {
        let p = interpolate_zero_crossing(p110, v110, p111, v111);
        add_point(p, vec3<u32>(voxel.color_r, voxel.color_g, voxel.color_b));
    }
} 