//! Shader utilities for 3D visualization

/// Vertex shader for point cloud rendering
pub const POINT_VERTEX_SHADER: &str = r#"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 view_proj;
};

layout(location = 0) out vec3 v_color;

void main() {
    gl_Position = view_proj * vec4(position, 1.0);
    v_color = color;
}
"#;

/// Fragment shader for point cloud rendering
pub const POINT_FRAGMENT_SHADER: &str = r#"
#version 450

layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(v_color, 1.0);
}
"#;

/// Vertex shader for mesh rendering
pub const MESH_VERTEX_SHADER: &str = r#"
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 view_proj;
    mat4 model;
    vec3 light_dir;
};

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;

void main() {
    gl_Position = view_proj * model * vec4(position, 1.0);
    v_color = color;
    v_normal = normal;
}
"#;

/// Fragment shader for mesh rendering
pub const MESH_FRAGMENT_SHADER: &str = r#"
#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 view_proj;
    mat4 model;
    vec3 light_dir;
};

layout(location = 0) out vec4 f_color;

void main() {
    float light = max(0.1, dot(normalize(v_normal), normalize(light_dir)));
    f_color = vec4(v_color * light, 1.0);
}
"#; 