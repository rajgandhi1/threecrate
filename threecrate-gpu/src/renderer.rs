use crate::device::GpuContext;
use threecrate_core::{PointCloud, Point3f, Error, Result};
use nalgebra::{Matrix4, Vector3};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Vertex data for point cloud rendering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PointVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub size: f32,
}

impl PointVertex {
    /// Create vertex from Point3f with default color and size
    pub fn from_point(point: &Point3f, color: [f32; 3], size: f32) -> Self {
        Self {
            position: [point.x, point.y, point.z],
            color,
            size,
        }
    }

    /// Vertex buffer layout descriptor
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Size
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Camera uniform data
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_pos: [f32; 3],
    pub _padding: f32,
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub point_size: f32,
    pub background_color: [f64; 4],
    pub enable_depth_test: bool,
    pub enable_alpha_blending: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            point_size: 2.0,
            background_color: [0.1, 0.1, 0.1, 1.0],
            enable_depth_test: true,
            enable_alpha_blending: true,
        }
    }
}

/// GPU-accelerated point cloud renderer
pub struct PointCloudRenderer<'window> {
    pub gpu_context: GpuContext,
    pub surface: wgpu::Surface<'window>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub render_pipeline: wgpu::RenderPipeline,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub config: RenderConfig,
}

impl<'window> PointCloudRenderer<'window> {
    /// Create new point cloud renderer
    pub async fn new(window: &'window Window, config: RenderConfig) -> Result<Self> {
        let gpu_context = GpuContext::new().await?;
        
        let surface = gpu_context.instance.create_surface(window)
            .map_err(|e| Error::Gpu(format!("Failed to create surface: {:?}", e)))?;

        let surface_caps = surface.get_capabilities(&gpu_context.adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
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

        // Create camera uniform
        let camera_uniform = CameraUniform {
            view_proj: Matrix4::identity().into(),
            view_pos: [0.0, 0.0, 0.0],
            _padding: 0.0,
        };

        let camera_buffer = gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout = gpu_context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = gpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Create render pipeline
        let shader = gpu_context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/point_cloud.wgsl").into()),
        });

        let render_pipeline_layout = gpu_context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Point Cloud Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = gpu_context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Point Cloud Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[PointVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: if config.enable_alpha_blending {
                        Some(wgpu::BlendState::ALPHA_BLENDING)
                    } else {
                        Some(wgpu::BlendState::REPLACE)
                    },
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
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
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Ok(Self {
            gpu_context,
            surface,
            surface_config,
            render_pipeline,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            config,
        })
    }

    /// Update camera view and projection matrices
    pub fn update_camera(&mut self, view_matrix: Matrix4<f32>, proj_matrix: Matrix4<f32>, camera_pos: Vector3<f32>) {
        let view_proj = proj_matrix * view_matrix;
        self.camera_uniform.view_proj = view_proj.into();
        self.camera_uniform.view_pos = [camera_pos.x, camera_pos.y, camera_pos.z];

        self.gpu_context.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );
    }

    /// Resize renderer surface
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.gpu_context.device, &self.surface_config);
        }
    }

    /// Create vertex buffer from point cloud
    pub fn create_vertex_buffer(&self, vertices: &[PointVertex]) -> wgpu::Buffer {
        self.gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Cloud Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    /// Create depth texture for depth testing
    pub fn create_depth_texture(&self) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: self.surface_config.width,
            height: self.surface_config.height,
            depth_or_array_layers: 1,
        };

        self.gpu_context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    /// Render point cloud
    pub fn render(&self, vertices: &[PointVertex]) -> Result<()> {
        let vertex_buffer = self.create_vertex_buffer(vertices);
        
        let output = self.surface.get_current_texture()
            .map_err(|e| Error::Gpu(format!("Failed to get surface texture: {:?}", e)))?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.create_depth_texture();
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Point Cloud Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Point Cloud Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.config.background_color[0],
                            g: self.config.background_color[1],
                            b: self.config.background_color[2],
                            a: self.config.background_color[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..vertices.len() as u32, 0..1);
        }

        self.gpu_context.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Convert point cloud to render vertices with default colors
pub fn point_cloud_to_vertices(point_cloud: &PointCloud<Point3f>, color: [f32; 3], size: f32) -> Vec<PointVertex> {
    point_cloud.points.iter()
        .map(|point| PointVertex::from_point(point, color, size))
        .collect()
}

/// Convert point cloud to render vertices with height-based coloring
pub fn point_cloud_to_vertices_colored(point_cloud: &PointCloud<Point3f>, size: f32) -> Vec<PointVertex> {
    if point_cloud.points.is_empty() {
        return Vec::new();
    }

    // Find height range for coloring
    let min_y = point_cloud.points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let max_y = point_cloud.points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
    let y_range = max_y - min_y;

    point_cloud.points.iter()
        .map(|point| {
            // Height-based color mapping (blue to red)
            let normalized_height = if y_range > 0.0 {
                (point.y - min_y) / y_range
            } else {
                0.5
            };
            
            let color = [
                normalized_height,         // Red component
                0.5,                      // Green component
                1.0 - normalized_height,  // Blue component
            ];
            
            PointVertex::from_point(point, color, size)
        })
        .collect()
}

// Note: Interactive rendering function commented out due to lifetime complexity
// Users can create their own rendering loop using PointCloudRenderer::new()
// 
// /// GPU-accelerated point cloud rendering function  
// pub async fn gpu_render_point_cloud(...) -> Result<()> {
//     // Implementation requires complex lifetime management
//     // Use PointCloudRenderer directly for custom rendering loops
// } 