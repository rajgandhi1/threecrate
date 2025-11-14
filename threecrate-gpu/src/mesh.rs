//! GPU-accelerated mesh rendering with PBR and flat shading

use threecrate_core::{Result, Error};
use crate::GpuContext;
use nalgebra::{Matrix4, Vector3, Point3};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Vertex data for mesh rendering with full PBR attributes
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
    pub _padding: f32,
}

impl MeshVertex {
    /// Create a new mesh vertex
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        color: [f32; 3],
    ) -> Self {
        // Calculate tangent and bitangent vectors (simplified)
        let tangent = if normal[0].abs() > 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        
        let bitangent = [
            normal[1] * tangent[2] - normal[2] * tangent[1],
            normal[2] * tangent[0] - normal[0] * tangent[2],
            normal[0] * tangent[1] - normal[1] * tangent[0],
        ];
        
        Self {
            position,
            normal,
            tangent,
            bitangent,
            uv,
            color,
            _padding: 0.0,
        }
    }
    
    /// Create vertex from position and normal with default color
    pub fn from_pos_normal(position: [f32; 3], normal: [f32; 3]) -> Self {
        Self::new(position, normal, [0.0, 0.0], [0.8, 0.8, 0.8])
    }
    
    /// Vertex buffer layout descriptor
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Tangent
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Bitangent
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // UV
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 14]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Camera uniform data for mesh rendering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MeshCameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_pos: [f32; 3],
    pub _padding: f32,
}

/// PBR material properties
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PbrMaterial {
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub emission: [f32; 3],
    pub _padding: f32,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            albedo: [0.7, 0.7, 0.7],
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            emission: [0.0, 0.0, 0.0],
            _padding: 0.0,
        }
    }
}

/// Flat shading material properties
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FlatMaterial {
    pub color: [f32; 3],
    pub _padding: f32,
}

impl Default for FlatMaterial {
    fn default() -> Self {
        Self {
            color: [0.8, 0.8, 0.8],
            _padding: 0.0,
        }
    }
}

/// Lighting parameters for mesh rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshLightingParams {
    pub light_position: [f32; 3],
    pub light_intensity: f32,
    pub light_color: [f32; 3],
    pub ambient_strength: f32,
    pub gamma: f32,
    pub exposure: f32,
    pub _padding: [f32; 2],
}

impl Default for MeshLightingParams {
    fn default() -> Self {
        Self {
            light_position: [10.0, 10.0, 10.0],
            light_intensity: 1.0,
            light_color: [1.0, 1.0, 1.0],
            ambient_strength: 0.03,
            gamma: 2.2,
            exposure: 1.0,
            _padding: [0.0, 0.0],
        }
    }
}

/// Mesh rendering configuration
#[derive(Debug, Clone)]
pub struct MeshRenderConfig {
    pub lighting_params: MeshLightingParams,
    pub background_color: [f64; 4],
    pub enable_depth_test: bool,
    pub enable_backface_culling: bool,
    pub enable_multisampling: bool,
    pub wireframe_mode: bool,
}

impl Default for MeshRenderConfig {
    fn default() -> Self {
        Self {
            lighting_params: MeshLightingParams::default(),
            background_color: [0.1, 0.1, 0.1, 1.0],
            enable_depth_test: true,
            enable_backface_culling: true,
            enable_multisampling: true,
            wireframe_mode: false,
        }
    }
}

/// Mesh data structure for GPU rendering
#[derive(Debug, Clone)]
pub struct GpuMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub material: PbrMaterial,
}

impl GpuMesh {
    /// Create a new GPU mesh
    pub fn new(vertices: Vec<MeshVertex>, indices: Vec<u32>, material: PbrMaterial) -> Self {
        Self {
            vertices,
            indices,
            material,
        }
    }
    
    /// Create a simple triangle mesh for testing
    pub fn triangle() -> Self {
        let vertices = vec![
            MeshVertex::new([-0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], [1.0, 0.0, 0.0]),
            MeshVertex::new([0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], [0.0, 1.0, 0.0]),
            MeshVertex::new([0.0, 0.5, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0], [0.0, 0.0, 1.0]),
        ];
        
        let indices = vec![0, 1, 2];
        
        Self::new(vertices, indices, PbrMaterial::default())
    }
    
    /// Create a cube mesh for testing
    pub fn cube() -> Self {
        let vertices = vec![
            // Front face
            MeshVertex::new([-1.0, -1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0], [0.8, 0.2, 0.2]),
            MeshVertex::new([1.0, -1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0], [0.8, 0.2, 0.2]),
            MeshVertex::new([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0], [0.8, 0.2, 0.2]),
            MeshVertex::new([-1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0], [0.8, 0.2, 0.2]),
            
            // Back face
            MeshVertex::new([1.0, -1.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0], [0.2, 0.8, 0.2]),
            MeshVertex::new([-1.0, -1.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0], [0.2, 0.8, 0.2]),
            MeshVertex::new([-1.0, 1.0, -1.0], [0.0, 0.0, -1.0], [1.0, 1.0], [0.2, 0.8, 0.2]),
            MeshVertex::new([1.0, 1.0, -1.0], [0.0, 0.0, -1.0], [0.0, 1.0], [0.2, 0.8, 0.2]),
            
            // Top face
            MeshVertex::new([-1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0], [0.2, 0.2, 0.8]),
            MeshVertex::new([1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0], [0.2, 0.2, 0.8]),
            MeshVertex::new([1.0, 1.0, -1.0], [0.0, 1.0, 0.0], [1.0, 1.0], [0.2, 0.2, 0.8]),
            MeshVertex::new([-1.0, 1.0, -1.0], [0.0, 1.0, 0.0], [0.0, 1.0], [0.2, 0.2, 0.8]),
            
            // Bottom face
            MeshVertex::new([-1.0, -1.0, -1.0], [0.0, -1.0, 0.0], [0.0, 0.0], [0.8, 0.8, 0.2]),
            MeshVertex::new([1.0, -1.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0], [0.8, 0.8, 0.2]),
            MeshVertex::new([1.0, -1.0, 1.0], [0.0, -1.0, 0.0], [1.0, 1.0], [0.8, 0.8, 0.2]),
            MeshVertex::new([-1.0, -1.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0], [0.8, 0.8, 0.2]),
            
            // Right face
            MeshVertex::new([1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0], [0.8, 0.2, 0.8]),
            MeshVertex::new([1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0], [0.8, 0.2, 0.8]),
            MeshVertex::new([1.0, 1.0, -1.0], [1.0, 0.0, 0.0], [1.0, 1.0], [0.8, 0.2, 0.8]),
            MeshVertex::new([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0], [0.8, 0.2, 0.8]),
            
            // Left face
            MeshVertex::new([-1.0, -1.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 0.0], [0.2, 0.8, 0.8]),
            MeshVertex::new([-1.0, -1.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0], [0.2, 0.8, 0.8]),
            MeshVertex::new([-1.0, 1.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 1.0], [0.2, 0.8, 0.8]),
            MeshVertex::new([-1.0, 1.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0], [0.2, 0.8, 0.8]),
        ];
        
        let indices = vec![
            // Front face
            0, 1, 2, 2, 3, 0,
            // Back face
            4, 5, 6, 6, 7, 4,
            // Top face
            8, 9, 10, 10, 11, 8,
            // Bottom face
            12, 13, 14, 14, 15, 12,
            // Right face
            16, 17, 18, 18, 19, 16,
            // Left face
            20, 21, 22, 22, 23, 20,
        ];
        
        Self::new(vertices, indices, PbrMaterial::default())
    }
    
    /// Create a mesh from point cloud with estimated normals
    pub fn from_point_cloud(points: &[Point3<f32>], color: [f32; 3]) -> Self {
        let vertices: Vec<MeshVertex> = points.iter().map(|p| {
            // Simple normal estimation (could use the GPU normal computation)
            let normal = [0.0, 0.0, 1.0]; // Default normal
            MeshVertex::new([p.x, p.y, p.z], normal, [0.0, 0.0], color)
        }).collect();
        
        // Create indices for point rendering (each point is a degenerate triangle)
        let indices: Vec<u32> = (0..vertices.len() as u32).collect();
        
        Self::new(vertices, indices, PbrMaterial::default())
    }
}

/// Shading mode for mesh rendering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShadingMode {
    Flat,
    Pbr,
}

/// GPU-accelerated mesh renderer with PBR and flat shading
pub struct MeshRenderer<'window> {
    pub gpu_context: GpuContext,
    pub surface: wgpu::Surface<'window>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub pbr_pipeline: wgpu::RenderPipeline,
    pub flat_pipeline: wgpu::RenderPipeline,
    pub camera_uniform: MeshCameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub lighting_params: MeshLightingParams,
    pub lighting_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub config: MeshRenderConfig,
    pub msaa_texture: Option<wgpu::Texture>,
    pub msaa_view: Option<wgpu::TextureView>,
}

impl<'window> MeshRenderer<'window> {
    /// Create new mesh renderer with PBR and flat shading support
    pub async fn new(window: &'window Window, config: MeshRenderConfig) -> Result<Self> {
        let gpu_context = GpuContext::new().await?;
        
        let surface = gpu_context.instance.create_surface(window)
            .map_err(|e| Error::Gpu(format!("Failed to create surface: {:?}", e)))?;

        let surface_caps = surface.get_capabilities(&gpu_context.adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let sample_count = if config.enable_multisampling { 4 } else { 1 };
        
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu_context.device, &surface_config);

        // Create MSAA texture if enabled
        let (msaa_texture, msaa_view) = if config.enable_multisampling {
            let msaa_texture = gpu_context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("MSAA Texture"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(msaa_texture), Some(msaa_view))
        } else {
            (None, None)
        };

        // Create camera uniform
        let camera_uniform = MeshCameraUniform {
            view_proj: Matrix4::identity().into(),
            view_pos: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };

        let camera_buffer = gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create lighting parameters buffer
        let lighting_params = config.lighting_params;
        let lighting_buffer = gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lighting Buffer"),
            contents: bytemuck::bytes_of(&lighting_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = gpu_context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("mesh_bind_group_layout"),
        });

        // Create PBR pipeline
        let pbr_shader = gpu_context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh_pbr.wgsl").into()),
        });

        let pbr_pipeline = Self::create_render_pipeline(
            &gpu_context.device,
            &bind_group_layout,
            &pbr_shader,
            surface_format,
            sample_count,
            &config,
            "PBR",
        );

        // Create flat pipeline
        let flat_shader = gpu_context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flat Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh_flat.wgsl").into()),
        });

        let flat_pipeline = Self::create_render_pipeline(
            &gpu_context.device,
            &bind_group_layout,
            &flat_shader,
            surface_format,
            sample_count,
            &config,
            "Flat",
        );

        Ok(Self {
            gpu_context,
            surface,
            surface_config,
            pbr_pipeline,
            flat_pipeline,
            camera_uniform,
            camera_buffer,
            lighting_params,
            lighting_buffer,
            bind_group_layout,
            config,
            msaa_texture,
            msaa_view,
        })
    }

    /// Create a render pipeline for mesh rendering
    fn create_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        config: &MeshRenderConfig,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Mesh Render Pipeline Layout", label)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{} Mesh Render Pipeline", label)),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[MeshVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: if config.wireframe_mode {
                    wgpu::PrimitiveTopology::LineList
                } else {
                    wgpu::PrimitiveTopology::TriangleList
                },
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: if config.enable_backface_culling {
                    Some(wgpu::Face::Back)
                } else {
                    None
                },
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: if config.enable_depth_test {
                Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                })
            } else {
                None
            },
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }

    /// Update camera matrices and position
    pub fn update_camera(&mut self, view_matrix: Matrix4<f32>, proj_matrix: Matrix4<f32>, camera_pos: Vector3<f32>) {
        self.camera_uniform.view_proj = (proj_matrix * view_matrix).into();
        self.camera_uniform.view_pos = camera_pos.into();
        
        self.gpu_context.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );
    }

    /// Update lighting parameters
    pub fn update_lighting(&mut self, params: MeshLightingParams) {
        self.lighting_params = params;
        self.gpu_context.queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::bytes_of(&self.lighting_params),
        );
    }

    /// Resize renderer
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.gpu_context.device, &self.surface_config);
            
            // Recreate MSAA texture if needed
            if self.config.enable_multisampling {
                let msaa_texture = self.gpu_context.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("MSAA Texture"),
                    size: wgpu::Extent3d {
                        width: new_size.width,
                        height: new_size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 4,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.surface_config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
                self.msaa_texture = Some(msaa_texture);
                self.msaa_view = Some(msaa_view);
            }
        }
    }

    /// Create vertex buffer
    pub fn create_vertex_buffer(&self, vertices: &[MeshVertex]) -> wgpu::Buffer {
        self.gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    /// Create index buffer
    pub fn create_index_buffer(&self, indices: &[u32]) -> wgpu::Buffer {
        self.gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        })
    }

    /// Create depth texture
    pub fn create_depth_texture(&self) -> wgpu::Texture {
        let sample_count = if self.config.enable_multisampling { 4 } else { 1 };
        
        self.gpu_context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    /// Render mesh with specified shading mode
    pub fn render(&self, mesh: &GpuMesh, shading_mode: ShadingMode) -> Result<()> {
        let vertex_buffer = self.create_vertex_buffer(&mesh.vertices);
        let index_buffer = self.create_index_buffer(&mesh.indices);
        let depth_texture = self.create_depth_texture();
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let output = self.surface.get_current_texture()
            .map_err(|e| Error::Gpu(format!("Failed to get surface texture: {:?}", e)))?;
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create material buffer based on shading mode
        let material_buffer = match shading_mode {
            ShadingMode::Pbr => {
                self.gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("PBR Material Buffer"),
                    contents: bytemuck::bytes_of(&mesh.material),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            }
            ShadingMode::Flat => {
                let flat_material = FlatMaterial {
                    color: mesh.material.albedo,
                    _padding: 0.0,
                };
                self.gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Flat Material Buffer"),
                    contents: bytemuck::bytes_of(&flat_material),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            }
        };

        let bind_group = self.gpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.lighting_buffer.as_entire_binding(),
                },
            ],
            label: Some("mesh_bind_group"),
        });

        let mut encoder = self.gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mesh Render Encoder"),
        });

        // Determine render target
        let (color_attachment, resolve_target) = if let Some(ref msaa_view) = self.msaa_view {
            (msaa_view, Some(&view))
        } else {
            (&view, None)
        };

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_attachment,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.config.background_color[0],
                            g: self.config.background_color[1],
                            b: self.config.background_color[2],
                            a: self.config.background_color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: if self.config.enable_depth_test {
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    })
                } else {
                    None
                },
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let pipeline = match shading_mode {
                ShadingMode::Pbr => &self.pbr_pipeline,
                ShadingMode::Flat => &self.flat_pipeline,
            };

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }

        self.gpu_context.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Convert threecrate mesh to GPU mesh format
pub fn mesh_to_gpu_mesh(
    vertices: &[Point3<f32>],
    indices: &[u32],
    normals: Option<&[Vector3<f32>]>,
    colors: Option<&[[f32; 3]]>,
    material: Option<PbrMaterial>,
) -> GpuMesh {
    let gpu_vertices: Vec<MeshVertex> = vertices
        .iter()
        .enumerate()
        .map(|(i, vertex)| {
            let normal = normals
                .and_then(|n| n.get(i))
                .map(|n| [n.x, n.y, n.z])
                .unwrap_or([0.0, 0.0, 1.0]);
            
            let color = colors
                .and_then(|c| c.get(i))
                .copied()
                .unwrap_or([0.8, 0.8, 0.8]);
            
            MeshVertex::new([vertex.x, vertex.y, vertex.z], normal, [0.0, 0.0], color)
        })
        .collect();

    GpuMesh::new(
        gpu_vertices,
        indices.to_vec(),
        material.unwrap_or_default(),
    )
} 