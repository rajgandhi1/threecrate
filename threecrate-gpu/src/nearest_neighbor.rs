//! GPU-accelerated nearest neighbor search

use threecrate_core::{Point3f, Result, Error};
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};

/// Parameters for nearest neighbor search
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NearestNeighborParams {
    pub num_points: u32,
    pub k_neighbors: u32,
    pub max_distance: f32,
    pub _padding: u32,
}

/// GPU representation of a point for nearest neighbor search
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(align(16))]
pub struct GpuPoint {
    pub position: [f32; 3],
    pub _padding: f32,
}

/// Result of nearest neighbor search
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct NeighborResult {
    pub index: u32,
    pub distance: f32,
    pub _padding: [u32; 2],
}

const NEAREST_NEIGHBOR_SHADER: &str = r#"
struct GpuPoint {
    position: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> input_points: array<GpuPoint>;
@group(0) @binding(1) var<storage, read> query_points: array<GpuPoint>;
@group(0) @binding(2) var<storage, read_write> output_neighbors: array<array<vec2<f32>, MAX_K>>;
@group(0) @binding(3) var<uniform> params: NearestNeighborParams;

struct NearestNeighborParams {
    num_points: u32,
    k_neighbors: u32,
    max_distance: f32,
    _padding: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_idx = global_id.x;
    if (query_idx >= arrayLength(&query_points)) {
        return;
    }
    
    let query_point = query_points[query_idx].position;
    
    // Initialize neighbors with maximum distance
    var neighbors: array<vec2<f32>, MAX_K>;
    for (var i = 0u; i < params.k_neighbors; i++) {
        neighbors[i] = vec2<f32>(f32(params.num_points), params.max_distance);
    }
    
    // Find k nearest neighbors
    for (var i = 0u; i < params.num_points; i++) {
        let diff = input_points[i].position - query_point;
        let distance = length(diff);
        
        if (distance < params.max_distance) {
            // Insert into sorted neighbors array
            let neighbor = vec2<f32>(f32(i), distance);
            
            // Find insertion point
            var insert_idx = params.k_neighbors;
            for (var j = 0u; j < params.k_neighbors; j++) {
                if (distance < neighbors[j].y) {
                    insert_idx = j;
                    break;
                }
            }
            
            // Shift and insert
            if (insert_idx < params.k_neighbors) {
                for (var j = params.k_neighbors - 1u; j > insert_idx; j--) {
                    neighbors[j] = neighbors[j - 1u];
                }
                neighbors[insert_idx] = neighbor;
            }
        }
    }
    
    // Write results
    for (var i = 0u; i < params.k_neighbors; i++) {
        output_neighbors[query_idx][i] = neighbors[i];
    }
}
"#;

impl GpuContext {
    /// GPU-accelerated k-nearest neighbor search
    pub async fn find_k_nearest_neighbors(
        &self,
        points: &[Point3f],
        query_points: &[Point3f],
        k: usize,
        max_distance: f32,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        if points.is_empty() || query_points.is_empty() {
            return Ok(vec![Vec::new(); query_points.len()]);
        }
        
        let k = k.min(32).max(1); // Limit k to reasonable bounds
        
        // Convert points to GPU format with proper alignment
        let gpu_points: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint { position: [p.x, p.y, p.z], _padding: 0.0 })
            .collect();
            
        let gpu_query_points: Vec<GpuPoint> = query_points
            .iter()
            .map(|p| GpuPoint { position: [p.x, p.y, p.z], _padding: 0.0 })
            .collect();

        // Create buffers
        let points_buffer = self.create_buffer_init(
            "Points Buffer",
            &gpu_points,
            wgpu::BufferUsages::STORAGE,
        );

        let query_buffer = self.create_buffer_init(
            "Query Points Buffer",
            &gpu_query_points,
            wgpu::BufferUsages::STORAGE,
        );

        let output_buffer = self.create_buffer(
            "Output Buffer",
            (query_points.len() * k * std::mem::size_of::<[f32; 2]>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = NearestNeighborParams {
            num_points: points.len() as u32,
            k_neighbors: k as u32,
            max_distance,
            _padding: 0,
        };

        let params_buffer = self.create_buffer_init(
            "Params Buffer",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create shader with MAX_K constant
        let shader_source = NEAREST_NEIGHBOR_SHADER.replace("MAX_K", &k.to_string());
        let shader = self.create_shader_module("Nearest Neighbor Shader", &shader_source);

        // Create bind group layout
        let bind_group_layout = self.create_bind_group_layout(
            "Nearest Neighbor Layout",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nearest Neighbor Pipeline"),
            layout: Some(&self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Nearest Neighbor Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Create bind group
        let bind_group = self.create_bind_group(
            "Nearest Neighbor Bind Group",
            &bind_group_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Nearest Neighbor Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nearest Neighbor Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (query_points.len() + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
        }

        // Read back results
        let staging_buffer = self.create_buffer(
            "Staging Buffer",
            (query_points.len() * k * std::mem::size_of::<[f32; 2]>()) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (query_points.len() * k * std::mem::size_of::<[f32; 2]>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let raw_neighbors: Vec<[f32; 2]> = bytemuck::cast_slice(&data).to_vec();
            
            let mut results = Vec::with_capacity(query_points.len());
            for i in 0..query_points.len() {
                let mut neighbors = Vec::with_capacity(k);
                for j in 0..k {
                    let idx = i * k + j;
                    if idx < raw_neighbors.len() {
                        let neighbor = raw_neighbors[idx];
                        let point_idx = neighbor[0] as usize;
                        let distance = neighbor[1];
                        
                        if point_idx < points.len() && distance < max_distance {
                            neighbors.push((point_idx, distance));
                        }
                    }
                }
                results.push(neighbors);
            }
            
            drop(data);
            staging_buffer.unmap();
            
            Ok(results)
        } else {
            Err(Error::Gpu("Failed to read GPU results".to_string()))
        }
    }
}

/// GPU-accelerated nearest neighbor search for single query point
pub async fn gpu_find_k_nearest(
    gpu_context: &GpuContext,
    points: &[Point3f],
    query: &Point3f,
    k: usize,
) -> Result<Vec<(usize, f32)>> {
    let results = gpu_context.find_k_nearest_neighbors(points, &[*query], k, f32::INFINITY).await?;
    Ok(results.into_iter().next().unwrap_or_default())
}

/// GPU-accelerated nearest neighbor search for multiple query points
pub async fn gpu_find_k_nearest_batch(
    gpu_context: &GpuContext,
    points: &[Point3f],
    query_points: &[Point3f],
    k: usize,
) -> Result<Vec<Vec<(usize, f32)>>> {
    gpu_context.find_k_nearest_neighbors(points, query_points, k, f32::INFINITY).await
}

/// GPU-accelerated radius-based nearest neighbor search
pub async fn gpu_find_radius_neighbors(
    gpu_context: &GpuContext,
    points: &[Point3f],
    query: &Point3f,
    radius: f32,
) -> Result<Vec<(usize, f32)>> {
    let results = gpu_context.find_k_nearest_neighbors(points, &[*query], 32, radius).await?;
    Ok(results.into_iter().next().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuContext;
    use threecrate_core::Point3f;
    use approx::assert_relative_eq;

    /// Try to create a GPU context, return None if not available
    async fn try_create_gpu_context() -> Option<GpuContext> {
        match GpuContext::new().await {
            Ok(gpu) => Some(gpu),
            Err(_) => {
                println!("⚠️  GPU not available, skipping GPU-dependent test");
                None
            }
        }
    }

    /// Create a simple test point cloud
    fn create_test_points() -> Vec<Point3f> {
        vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(0.0, 1.0, 0.0),
            Point3f::new(0.0, 0.0, 1.0),
            Point3f::new(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn test_gpu_nearest_neighbor_single() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let query = Point3f::new(0.1, 0.1, 0.1);
            
            let neighbors = gpu_find_k_nearest(&gpu, &points, &query, 3).await.unwrap();
            
            assert_eq!(neighbors.len(), 3);
            assert_eq!(neighbors[0].0, 0); // Closest should be origin
            assert!(neighbors[0].1 < 0.2); // Distance should be small
            
            println!("✓ GPU single nearest neighbor test passed");
        });
    }

    #[test]
    fn test_gpu_nearest_neighbor_batch() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let queries = vec![
                Point3f::new(0.1, 0.1, 0.1),
                Point3f::new(0.9, 0.1, 0.1),
            ];
            
            let results = gpu_find_k_nearest_batch(&gpu, &points, &queries, 2).await.unwrap();
            
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].len(), 2);
            assert_eq!(results[1].len(), 2);
            
            // First query should find origin as closest
            assert_eq!(results[0][0].0, 0);
            
            // Second query should find (1,0,0) as closest
            assert_eq!(results[1][0].0, 1);
            
            println!("✓ GPU batch nearest neighbor test passed");
        });
    }

    #[test]
    fn test_gpu_radius_neighbors() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let query = Point3f::new(0.0, 0.0, 0.0);
            let radius = 1.5;
            
            let neighbors = gpu_find_radius_neighbors(&gpu, &points, &query, radius).await.unwrap();
            
            // Should find points within radius
            assert!(!neighbors.is_empty());
            
            // All distances should be within radius
            for (_, distance) in &neighbors {
                assert!(*distance <= radius);
            }
            
            println!("✓ GPU radius neighbors test passed: {} neighbors found", neighbors.len());
        });
    }

    #[test]
    fn test_gpu_nearest_neighbor_accuracy() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let points = create_test_points();
            let query = Point3f::new(0.5, 0.5, 0.5);
            
            let neighbors = gpu_find_k_nearest(&gpu, &points, &query, 1).await.unwrap();
            
            assert_eq!(neighbors.len(), 1);
            
            // Manually verify the nearest neighbor
            let mut min_dist = f32::INFINITY;
            let mut min_idx = 0;
            
            for (i, point) in points.iter().enumerate() {
                let dist = (query - *point).magnitude();
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = i;
                }
            }
            
            assert_eq!(neighbors[0].0, min_idx);
            assert_relative_eq!(neighbors[0].1, min_dist, epsilon = 0.001);
            
            println!("✓ GPU nearest neighbor accuracy test passed");
        });
    }
} 