struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

struct RenderParams {
    point_size: f32,
    alpha_threshold: f32,
    enable_splatting: f32,
    enable_lighting: f32,
    ambient_strength: f32,
    diffuse_strength: f32,
    specular_strength: f32,
    shininess: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) size: f32,
    @location(3) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) view_pos: vec3<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) point_size: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> render_params: RenderParams;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform position to clip space
    out.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    
    // Pass through color and positions
    out.color = input.color;
    out.world_pos = input.position;
    out.view_pos = camera.view_pos;
    out.normal = normalize(input.normal);
    
    // Calculate point size based on distance and settings
    let distance = length(input.position - camera.view_pos);
    let base_size = max(input.size, render_params.point_size);
    
    // Distance-based size scaling with better attenuation
    let distance_attenuation = 1.0 / (1.0 + distance * distance * 0.001);
    let final_size = base_size * distance_attenuation;
    
    // Clamp size to reasonable bounds
    out.point_size = clamp(final_size, 1.0, 64.0);
    
    return out;
}

// Phong lighting calculation
fn calculate_phong_lighting(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_pos: vec3<f32>,
    color: vec3<f32>
) -> vec3<f32> {
    let light_pos = view_pos + vec3<f32>(2.0, 2.0, 2.0);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    
    // Ambient
    let ambient = render_params.ambient_strength * color;
    
    // Diffuse
    let light_dir = normalize(light_pos - world_pos);
    let diff = max(dot(normal, light_dir), 0.0);
    let diffuse = render_params.diffuse_strength * diff * color * light_color;
    
    // Specular
    let view_dir = normalize(view_pos - world_pos);
    let reflect_dir = reflect(-light_dir, normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), render_params.shininess);
    let specular = render_params.specular_strength * spec * light_color;
    
    return ambient + diffuse + specular;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var final_color: vec3<f32>;
    
    // Apply lighting if enabled
    if render_params.enable_lighting > 0.5 {
        final_color = calculate_phong_lighting(
            input.world_pos,
            input.normal,
            input.view_pos,
            input.color
        );
    } else {
        // Simple distance-based brightness
        let distance_to_camera = length(input.world_pos - input.view_pos);
        let distance_factor = 1.0 / (1.0 + distance_to_camera * 0.01);
        final_color = input.color * (0.7 + 0.3 * distance_factor);
    }
    
    // Use full alpha for points (no fancy shapes for now)
    return vec4<f32>(final_color, 1.0);
} 