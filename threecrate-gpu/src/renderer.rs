use crate::device::GpuContext;
use threecrate_core::{PointCloud, Point3f, ColoredPoint3f, Error, Result};
use nalgebra::{Matrix4, Vector3};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Vertex data for point cloud rendering with normals
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PointVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub size: f32,
    pub normal: [f32; 3],
}

impl PointVertex {
    /// Create vertex from Point3f with default color, size, and normal
    pub fn from_point(point: &Point3f, color: [f32; 3], size: f32, normal: [f32; 3]) -> Self {
        Self {
            position: [point.x, point.y, point.z],
            color,
            size,
            normal,
        }
    }

    /// Create vertex from ColoredPoint3f with default size and normal
    pub fn from_colored_point(point: &ColoredPoint3f, size: f32, normal: [f32; 3]) -> Self {
        Self {
            position: [point.position.x, point.position.y, point.position.z],
            color: [
                point.color[0] as f32 / 255.0,
                point.color[1] as f32 / 255.0,
                point.color[2] as f32 / 255.0,
            ],
            size,
            normal,
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
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
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

/// Render parameters for point cloud splatting
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderParams {
    pub point_size: f32,
    pub alpha_threshold: f32,
    pub enable_splatting: f32,
    pub enable_lighting: f32,
    pub ambient_strength: f32,
    pub diffuse_strength: f32,
    pub specular_strength: f32,
    pub shininess: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            point_size: 4.0,
            alpha_threshold: 0.1,
            enable_splatting: 1.0,
            enable_lighting: 1.0,
            ambient_strength: 0.3,
            diffuse_strength: 0.7,
            specular_strength: 0.5,
            shininess: 32.0,
        }
    }
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub render_params: RenderParams,
    pub background_color: [f64; 4],
    pub enable_depth_test: bool,
    pub enable_alpha_blending: bool,
    pub enable_multisampling: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            render_params: RenderParams::default(),
            background_color: [0.1, 0.1, 0.1, 1.0],
            enable_depth_test: true,
            enable_alpha_blending: true,
            enable_multisampling: true,
        }
    }
}

/// GPU-accelerated point cloud renderer with splatting support
pub struct PointCloudRenderer<'window> {
    pub gpu_context: GpuContext,
    pub surface: wgpu::Surface<'window>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub render_pipeline: wgpu::RenderPipeline,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub render_params: RenderParams,
    pub render_params_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub config: RenderConfig,
    pub msaa_texture: Option<wgpu::Texture>,
    pub msaa_view: Option<wgpu::TextureView>,
}

impl<'window> PointCloudRenderer<'window> {
    /// Create new point cloud renderer with splatting support
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

        // Create render parameters buffer
        let render_params = config.render_params;
        let render_params_buffer = gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: bytemuck::bytes_of(&render_params),
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
            ],
            label: Some("point_cloud_bind_group_layout"),
        });

        let bind_group = gpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_params_buffer.as_entire_binding(),
                },
            ],
            label: Some("point_cloud_bind_group"),
        });

        // Create render pipeline
        let shader = gpu_context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/point_cloud.wgsl").into()),
        });

        let render_pipeline_layout = gpu_context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Point Cloud Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = gpu_context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Point Cloud Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[PointVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
                topology: wgpu::PrimitiveTopology::TriangleList,
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
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            gpu_context,
            surface,
            surface_config,
            render_pipeline,
            camera_uniform,
            camera_buffer,
            render_params,
            render_params_buffer,
            bind_group,
            bind_group_layout,
            config,
            msaa_texture,
            msaa_view,
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

    /// Update render parameters
    pub fn update_render_params(&mut self, params: RenderParams) {
        self.render_params = params;
        self.gpu_context.queue.write_buffer(
            &self.render_params_buffer,
            0,
            bytemuck::bytes_of(&self.render_params),
        );
    }

    /// Resize renderer
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
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

    /// Create vertex buffer from point vertices
    pub fn create_vertex_buffer(&self, vertices: &[PointVertex]) -> wgpu::Buffer {
        self.gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Cloud Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
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

    /// Render point cloud with splatting
    pub fn render(&self, vertices: &[PointVertex]) -> Result<()> {
        let vertex_buffer = self.create_vertex_buffer(vertices);
        let depth_texture = self.create_depth_texture();
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let output = self.surface.get_current_texture()
            .map_err(|e| Error::Gpu(format!("Failed to get surface texture: {:?}", e)))?;
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Point Cloud Render Encoder"),
        });

        // Determine render target
        let (color_attachment, resolve_target) = if let Some(ref msaa_view) = self.msaa_view {
            (msaa_view, Some(&view))
        } else {
            (&view, None)
        };

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Point Cloud Render Pass"),
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
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..vertices.len() as u32, 0..1);
        }

        self.gpu_context.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

/// Convert point cloud to vertices with estimated normals
pub fn point_cloud_to_vertices(point_cloud: &PointCloud<Point3f>, color: [f32; 3], size: f32) -> Vec<PointVertex> {
    let normals = estimate_point_normals(&point_cloud.points);
    point_cloud.points.iter()
        .zip(normals.iter())
        .map(|(point, normal)| PointVertex::from_point(point, color, size, *normal))
        .collect()
}

/// Convert colored point cloud to vertices with estimated normals
pub fn point_cloud_to_vertices_colored(point_cloud: &PointCloud<Point3f>, size: f32) -> Vec<PointVertex> {
    let normals = estimate_point_normals(&point_cloud.points);
    point_cloud.points.iter()
        .zip(normals.iter())
        .map(|(point, normal)| {
            // Generate a color based on point position for visualization
            let color = [
                (point.x * 0.5 + 0.5).clamp(0.0, 1.0),
                (point.y * 0.5 + 0.5).clamp(0.0, 1.0),
                (point.z * 0.5 + 0.5).clamp(0.0, 1.0),
            ];
            
            PointVertex::from_point(point, color, size, *normal)
        })
        .collect()
}

/// Convert colored point cloud to vertices with estimated normals
pub fn colored_point_cloud_to_vertices(point_cloud: &PointCloud<ColoredPoint3f>, size: f32) -> Vec<PointVertex> {
    let positions: Vec<Point3f> = point_cloud.points.iter()
        .map(|p| Point3f::new(p.position.x, p.position.y, p.position.z))
        .collect();
    let normals = estimate_point_normals(&positions);
    
    point_cloud.points.iter()
        .zip(normals.iter())
        .map(|(point, normal)| PointVertex::from_colored_point(point, size, *normal))
        .collect()
}

/// Simple normal estimation using local point neighborhood
fn estimate_point_normals(points: &[Point3f]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0, 0.0, 1.0]; points.len()];
    
    for (i, point) in points.iter().enumerate() {
        // Find nearby points for normal estimation
        let mut neighbors = Vec::new();
        let search_radius = 0.1;
        
        for (j, other_point) in points.iter().enumerate() {
            if i != j {
                let distance = ((point.x - other_point.x).powi(2) + 
                               (point.y - other_point.y).powi(2) + 
                               (point.z - other_point.z).powi(2)).sqrt();
                
                if distance < search_radius && neighbors.len() < 10 {
                    neighbors.push(*other_point);
                }
            }
        }
        
        if neighbors.len() >= 3 {
            // Compute normal using PCA-like approach
            let mut normal = estimate_normal_from_neighbors(point, &neighbors);
            
            // Normalize the normal
            let length = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
            if length > 0.0 {
                normal[0] /= length;
                normal[1] /= length;
                normal[2] /= length;
            }
            
            normals[i] = normal;
        }
    }
    
    normals
}

/// Estimate normal from neighboring points using cross product
fn estimate_normal_from_neighbors(center: &Point3f, neighbors: &[Point3f]) -> [f32; 3] {
    if neighbors.len() < 2 {
        return [0.0, 0.0, 1.0];
    }
    
    // Use first two neighbors to compute normal
    let v1 = [
        neighbors[0].x - center.x,
        neighbors[0].y - center.y,
        neighbors[0].z - center.z,
    ];
    
    let v2 = [
        neighbors[1].x - center.x,
        neighbors[1].y - center.y,
        neighbors[1].z - center.z,
    ];
    
    // Cross product
    let normal = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    
    normal
}

// Note: Interactive rendering function commented out due to lifetime complexity
// Users can create their own rendering loop using PointCloudRenderer::new()
// 
// /// GPU-accelerated point cloud rendering function  
// pub async fn gpu_render_point_cloud(...) -> Result<()> {
//     // Implementation requires complex lifetime management
//     // Use PointCloudRenderer directly for custom rendering loops
// } 