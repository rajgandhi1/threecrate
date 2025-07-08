// Flat Shading Mesh Shader
// Simple, fast shading with face normals and Lambert lighting

struct MeshCameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

struct FlatMaterial {
    color: vec3<f32>,
    _padding: f32,
}

struct MeshLightingParams {
    light_position: vec3<f32>,
    light_intensity: f32,
    light_color: vec3<f32>,
    ambient_strength: f32,
    gamma: f32,
    exposure: f32,
    _padding: vec2<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) view_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: MeshCameraUniform;

@group(0) @binding(1)
var<uniform> material: FlatMaterial;

@group(0) @binding(2)
var<uniform> lighting: MeshLightingParams;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    out.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    out.world_pos = input.position;
    out.normal = normalize(input.normal);
    out.color = input.color;
    out.view_pos = camera.view_pos;
    
    return out;
}

// Simple Lambert lighting for flat shading
fn calculate_flat_lighting(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>
) -> vec3<f32> {
    // Light direction
    let light_dir = normalize(lighting.light_position - world_pos);
    
    // Distance and attenuation
    let distance = length(lighting.light_position - world_pos);
    let attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
    
    // Lambert diffuse
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = color * n_dot_l * lighting.light_color * lighting.light_intensity * attenuation;
    
    // Ambient lighting
    let ambient = lighting.ambient_strength * color;
    
    return ambient + diffuse;
}

// Simple gamma correction
fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / lighting.gamma));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Use vertex color blended with material color
    let base_color = mix(material.color, input.color, 0.5);
    
    // Calculate flat lighting
    let lit_color = calculate_flat_lighting(
        input.world_pos,
        input.normal,
        base_color
    );
    
    // Gamma correction
    let final_color = gamma_correct(lit_color);
    
    return vec4<f32>(final_color, 1.0);
} 