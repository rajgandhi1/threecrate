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
@group(0) @binding(5) var<storage, read> color_image: array<u32>; // RGB color data as bytes

fn voxel_to_world(voxel_coord: vec3<u32>) -> vec3<f32> {
    // Convert voxel coordinate to world space
    return vec3<f32>(voxel_coord) * params.voxel_size + params.origin;
}

fn world_to_camera(world_pos: vec3<f32>) -> vec3<f32> {
    let homogeneous_pos = vec4<f32>(world_pos, 1.0);
    // Need to use inverse of camera_transform to go from world to camera space
    // For now, assume camera_transform is already the world-to-camera matrix
    // (This should be passed as the inverse of the camera pose from Rust)
    let camera_pos = camera_transform * homogeneous_pos;
    return camera_pos.xyz;
}

fn camera_to_image(camera_pos: vec3<f32>) -> vec2<u32> {
    // Project point to image space
    let px = (camera_pos.x / camera_pos.z) * intrinsics.fx + intrinsics.cx;
    let py = (camera_pos.y / camera_pos.z) * intrinsics.fy + intrinsics.cy;
    
    // Round to nearest pixel
    return vec2<u32>(u32(px + 0.5), u32(py + 0.5));
}

fn get_depth_value(pixel: vec2<u32>) -> f32 {
    // Check image bounds
    if pixel.x >= intrinsics.width || pixel.y >= intrinsics.height {
        return 0.0;
    }
    
    // Get depth value from image
    let index = pixel.y * intrinsics.width + pixel.x;
    if index >= arrayLength(&depth_image) {
        return 0.0;
    }
    
    return depth_image[index];
}

fn get_color_value(pixel: vec2<u32>) -> vec3<u32> {
    // Check image bounds
    if pixel.x >= intrinsics.width || pixel.y >= intrinsics.height {
        return vec3<u32>(0u);
    }
    
    // Get color value from image (packed RGB format from Rust)
    let index = pixel.y * intrinsics.width + pixel.x;
    if index >= arrayLength(&color_image) {
        return vec3<u32>(0u);
    }
    
    // Unpack RGB from the packed u32 value: (r << 16) | (g << 8) | b
    let packed_color = color_image[index];
    return vec3<u32>(
        (packed_color >> 16u) & 0xFFu,  // Extract red
        (packed_color >> 8u) & 0xFFu,   // Extract green
        packed_color & 0xFFu             // Extract blue
    );
}

fn compute_tsdf_value(world_pos: vec3<f32>, depth_value: f32, camera_pos: vec3<f32>) -> f32 {
    if depth_value <= 0.0 {
        return params.truncation_distance;
    }
    
    // Use camera-space Z depth
    let voxel_depth = camera_pos.z;
    
    // Compute SDF with object-relative sign convention:
    // - Negative values = inside object (closer to camera than measured surface)
    // - Positive values = outside object (further from camera than measured surface)
    let sdf = depth_value - voxel_depth;
    
    // Truncate the SDF to the specified range
    return clamp(sdf, -params.truncation_distance, params.truncation_distance);
}

@compute @workgroup_size(4, 4, 4)
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
    
    // Use exponential moving average for better stability
    let alpha = 1.0 / new_weight;
    let updated_tsdf = (1.0 - alpha) * current_voxel.tsdf_value + alpha * tsdf_value;
    
    // Update color if available
    var updated_color = vec3<u32>(
        current_voxel.color_r,
        current_voxel.color_g, 
        current_voxel.color_b
    );
    
    if arrayLength(&color_image) > 0u {
        let color_value = get_color_value(pixel);
        if color_value.x > 0u || color_value.y > 0u || color_value.z > 0u {
            // Use exponential moving average for color too
            updated_color = vec3<u32>(
                u32(clamp((1.0 - alpha) * f32(current_voxel.color_r) + alpha * f32(color_value.x), 0.0, 255.0)),
                u32(clamp((1.0 - alpha) * f32(current_voxel.color_g) + alpha * f32(color_value.y), 0.0, 255.0)),
                u32(clamp((1.0 - alpha) * f32(current_voxel.color_b) + alpha * f32(color_value.z), 0.0, 255.0))
            );
        }
    }
    
    // Update voxel
    voxels[voxel_index] = TsdfVoxel(
        updated_tsdf,
        new_weight,
        updated_color.x,
        updated_color.y,
        updated_color.z,
        0u,
        0u,
        0u
    );
} 