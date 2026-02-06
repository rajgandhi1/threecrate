// Instanced point cloud rendering: one quad per point, no CPU quad generation.
// Unit quad vertices (slot 0-3) + instance buffer (position, color per point).

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

// Unit quad vertex: offset from center (-0.5 to 0.5 in XY plane)
struct QuadVertexInput {
    @location(0) offset: vec3<f32>,
}

// Instance data: one per point
struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> render_params: RenderParams;

@vertex
fn vs_main(
    quad: QuadVertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    // Scale quad by point_size and place at instance position
    let point_size = render_params.point_size;
    let world_pos = instance.position + quad.offset * point_size;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = instance.color;
    out.world_pos = world_pos;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple shading: slight falloff by distance
    let distance_to_camera = length(input.world_pos - camera.view_pos);
    let distance_factor = 1.0 / (1.0 + distance_to_camera * 0.01);
    let final_color = input.color * (0.8 + 0.2 * distance_factor);

    return vec4<f32>(final_color, 1.0);
}
