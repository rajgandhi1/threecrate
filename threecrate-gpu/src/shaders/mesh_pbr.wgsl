// PBR Mesh Shader with Metallic-Roughness Workflow
// Supports albedo, metallic, roughness, ambient occlusion, emission, and normal mapping

struct MeshCameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    _padding: f32,
}

struct PbrMaterial {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    emission: vec3<f32>,
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
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) color: vec3<f32>,
    @location(6) view_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: MeshCameraUniform;

@group(0) @binding(1)
var<uniform> material: PbrMaterial;

@group(0) @binding(2)
var<uniform> lighting: MeshLightingParams;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    out.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    out.world_pos = input.position;
    out.normal = normalize(input.normal);
    out.tangent = normalize(input.tangent);
    out.bitangent = normalize(input.bitangent);
    out.uv = input.uv;
    out.color = input.color;
    out.view_pos = camera.view_pos;
    
    return out;
}

// === PBR BRDF Functions ===

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;
    
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    
    let denom = n_dot_v * (1.0 - k) + k;
    return n_dot_v / denom;
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    
    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn calculate_pbr_lighting(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32
) -> vec3<f32> {
    // Light direction
    let light_dir = normalize(lighting.light_position - world_pos);
    let half_dir = normalize(view_dir + light_dir);
    
    // Distance and attenuation
    let distance = length(lighting.light_position - world_pos);
    let attenuation = 1.0 / (distance * distance);
    let radiance = lighting.light_color * lighting.light_intensity * attenuation;
    
    // F0 for dielectric and metallic materials
    var f0 = vec3<f32>(0.04);
    f0 = mix(f0, albedo, metallic);
    
    // Calculate BRDF
    let ndf = distribution_ggx(normal, half_dir, roughness);
    let g = geometry_smith(normal, view_dir, light_dir, roughness);
    let f = fresnel_schlick(max(dot(half_dir, view_dir), 0.0), f0);
    
    let numerator = ndf * g * f;
    let denominator = 4.0 * max(dot(normal, view_dir), 0.0) * max(dot(normal, light_dir), 0.0) + 0.0001;
    let specular = numerator / denominator;
    
    // Energy conservation
    let ks = f;
    var kd = vec3<f32>(1.0) - ks;
    kd *= 1.0 - metallic;
    
    // Lambertian diffuse
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = kd * albedo / 3.14159265;
    
    // Outgoing radiance
    let lo = (diffuse + specular) * radiance * n_dot_l;
    
    // Ambient lighting
    let ambient = lighting.ambient_strength * albedo * ao;
    
    return ambient + lo;
}

// HDR tone mapping
fn tone_map_reinhard(color: vec3<f32>) -> vec3<f32> {
    // Exposure
    let exposed = color * lighting.exposure;
    
    // Reinhard tone mapping
    let mapped = exposed / (exposed + vec3<f32>(1.0));
    
    // Gamma correction
    return pow(mapped, vec3<f32>(1.0 / lighting.gamma));
}

// Advanced tone mapping (ACES)
fn tone_map_aces(color: vec3<f32>) -> vec3<f32> {
    let exposed = color * lighting.exposure;
    
    // ACES tone mapping curve
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    
    let tone_mapped = (exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e);
    
    // Gamma correction
    return pow(clamp(tone_mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / lighting.gamma));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(input.view_pos - input.world_pos);
    
    // Use vertex color blended with material albedo
    let albedo = mix(material.albedo, input.color, 0.5);
    
    // Calculate PBR lighting
    let color = calculate_pbr_lighting(
        input.world_pos,
        input.normal,
        view_dir,
        albedo,
        material.metallic,
        material.roughness,
        material.ao
    );
    
    // Add emission
    let final_color = color + material.emission;
    
    // HDR tone mapping
    let tone_mapped = tone_map_aces(final_color);
    
    return vec4<f32>(tone_mapped, 1.0);
} 