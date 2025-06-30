struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) size: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) view_pos: vec3<f32>,
    @builtin(point_size) point_size: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform position to clip space
    out.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    
    // Pass through color and world position
    out.color = input.color;
    out.world_pos = input.position;
    out.view_pos = camera.view_pos;
    
    // Set point size based on distance to camera for depth perception
    let distance = length(input.position - camera.view_pos);
    let base_size = input.size;
    let distance_attenuation = 1.0 / (1.0 + distance * 0.01);
    out.point_size = max(base_size * distance_attenuation, 1.0);
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from center of point for circular points
    let center = vec2<f32>(0.5, 0.5);
    let coord = fract(input.clip_position.xy);
    let dist = distance(coord, center);
    
    // Create circular points with smooth edges
    let radius = 0.5;
    let smoothness = 0.1;
    let alpha = 1.0 - smoothstep(radius - smoothness, radius, dist);
    
    if alpha < 0.1 {
        discard;
    }
    
    // Calculate basic lighting based on distance from camera
    let distance_to_camera = length(input.world_pos - input.view_pos);
    let distance_factor = 1.0 / (1.0 + distance_to_camera * 0.01);
    
    // Apply distance-based brightness
    let lit_color = input.color * (0.7 + 0.3 * distance_factor);
    
    return vec4<f32>(lit_color, alpha);
} 