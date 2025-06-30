struct TsdfVoxel {
    tsdf_value: f32,
    weight: f32,
    color_r: u32,
    color_g: u32,
    color_b: u32,
    _padding: u32,
}

struct CameraIntrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: u32,
    height: u32,
    depth_scale: f32,
    _padding: f32,
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

@group(0) @binding(0) var<storage, read_write> voxels: array<TsdfVoxel>;
@group(0) @binding(1) var<storage, read> depth_image: array<f32>;
@group(0) @binding(2) var<uniform> camera_transform: mat4x4<f32>;
@group(0) @binding(3) var<uniform> intrinsics: CameraIntrinsics;
@group(0) @binding(4) var<uniform> params: TsdfParams;
@group(0) @binding(5) var<storage, read> color_image: array<u32>; // Optional RGB color data

fn voxel_to_world(voxel_coord: vec3<u32>) -> vec3<f32> {
    let world_coord = params.origin + vec3<f32>(voxel_coord) * params.voxel_size;
    return world_coord;
}

fn world_to_camera(world_pos: vec3<f32>) -> vec3<f32> {
    let homogeneous_pos = vec4<f32>(world_pos, 1.0);
    let camera_pos = camera_transform * homogeneous_pos;
    return camera_pos.xyz;
}

fn camera_to_image(camera_pos: vec3<f32>) -> vec2<i32> {
    if camera_pos.z <= 0.0 {
        return vec2<i32>(-1, -1); // Behind camera
    }
    
    let x = (camera_pos.x * intrinsics.fx / camera_pos.z) + intrinsics.cx;
    let y = (camera_pos.y * intrinsics.fy / camera_pos.z) + intrinsics.cy;
    
    let pixel_x = i32(x);
    let pixel_y = i32(y);
    
    if pixel_x < 0 || pixel_x >= i32(intrinsics.width) || 
       pixel_y < 0 || pixel_y >= i32(intrinsics.height) {
        return vec2<i32>(-1, -1); // Outside image bounds
    }
    
    return vec2<i32>(pixel_x, pixel_y);
}

fn get_depth_value(pixel: vec2<i32>) -> f32 {
    if pixel.x < 0 || pixel.y < 0 {
        return 0.0;
    }
    
    let index = u32(pixel.y) * intrinsics.width + u32(pixel.x);
    if index >= arrayLength(&depth_image) {
        return 0.0;
    }
    
    return depth_image[index] * intrinsics.depth_scale;
}

fn get_color_value(pixel: vec2<i32>) -> vec3<u32> {
    if pixel.x < 0 || pixel.y < 0 {
        return vec3<u32>(0u, 0u, 0u);
    }
    
    let index = (u32(pixel.y) * intrinsics.width + u32(pixel.x)) * 3u;
    if index + 2u >= arrayLength(&color_image) {
        return vec3<u32>(0u, 0u, 0u);
    }
    
    return vec3<u32>(
        color_image[index],
        color_image[index + 1u],
        color_image[index + 2u]
    );
}

fn compute_tsdf_value(world_pos: vec3<f32>, depth_value: f32, camera_pos: vec3<f32>) -> f32 {
    if depth_value <= 0.0 {
        return 1.0; // No measurement, assume outside surface
    }
    
    let camera_depth = length(camera_pos);
    let sdf = depth_value - camera_depth;
    
    // Truncate the SDF
    let truncated_sdf = clamp(sdf, -params.truncation_distance, params.truncation_distance);
    return truncated_sdf / params.truncation_distance;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_coord = global_id;
    
    // Check bounds
    if voxel_coord.x >= params.resolution.x || 
       voxel_coord.y >= params.resolution.y || 
       voxel_coord.z >= params.resolution.z {
        return;
    }
    
    let voxel_index = voxel_coord.z * params.resolution.x * params.resolution.y + 
                      voxel_coord.y * params.resolution.x + 
                      voxel_coord.x;
    
    if voxel_index >= arrayLength(&voxels) {
        return;
    }
    
    // Convert voxel coordinate to world space
    let world_pos = voxel_to_world(voxel_coord);
    
    // Transform to camera space
    let camera_pos = world_to_camera(world_pos);
    
    // Project to image space
    let pixel = camera_to_image(camera_pos);
    
    // Get depth measurement
    let depth_value = get_depth_value(pixel);
    
    if depth_value <= 0.0 {
        return; // No valid depth measurement
    }
    
    // Compute TSDF value
    let tsdf_value = compute_tsdf_value(world_pos, depth_value, camera_pos);
    
    // Update voxel with weighted average
    let current_voxel = voxels[voxel_index];
    let current_weight = current_voxel.weight;
    let new_weight = min(current_weight + 1.0, params.max_weight);
    
    let updated_tsdf = (current_voxel.tsdf_value * current_weight + tsdf_value) / new_weight;
    
    // Update color if available
    var updated_color = vec3<u32>(
        current_voxel.color_r,
        current_voxel.color_g, 
        current_voxel.color_b
    );
    
    if arrayLength(&color_image) > 0u {
        let color_value = get_color_value(pixel);
        if color_value.x > 0u || color_value.y > 0u || color_value.z > 0u {
            updated_color = vec3<u32>(
                u32((f32(current_voxel.color_r) * current_weight + f32(color_value.x)) / new_weight),
                u32((f32(current_voxel.color_g) * current_weight + f32(color_value.y)) / new_weight),
                u32((f32(current_voxel.color_b) * current_weight + f32(color_value.z)) / new_weight)
            );
        }
    }
    
    // Write updated voxel
    voxels[voxel_index] = TsdfVoxel(
        updated_tsdf,
        new_weight,
        updated_color.x,
        updated_color.y,
        updated_color.z,
        0u
    );
} 