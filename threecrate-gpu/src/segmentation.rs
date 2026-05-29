//! GPU-accelerated point cloud segmentation.

use std::collections::HashMap;

use nalgebra::{Vector3, Vector4};
use threecrate_core::{Error, Point3f, PointCloud, Result};

use crate::GpuContext;

const RANSAC_SCORE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> samples: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> candidates: array<PlaneCandidate>;
@group(0) @binding(3) var<uniform> params: RansacParams;

struct PlaneCandidate {
    coefficients: vec4<f32>,
    inlier_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct RansacParams {
    num_points: u32,
    num_samples: u32,
    threshold: f32,
    _pad: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    if (sample_idx >= params.num_samples) {
        return;
    }

    let sample = samples[sample_idx];
    let p1 = points[sample.x].xyz;
    let p2 = points[sample.y].xyz;
    let p3 = points[sample.z].xyz;

    let normal_raw = cross(p2 - p1, p3 - p1);
    let normal_len = length(normal_raw);
    if (normal_len < 1e-8) {
        candidates[sample_idx].coefficients = vec4<f32>(0.0);
        candidates[sample_idx].inlier_count = 0u;
        return;
    }

    let normal = normal_raw / normal_len;
    let d = -dot(normal, p1);
    var count = 0u;

    for (var i = 0u; i < params.num_points; i++) {
        let distance = abs(dot(normal, points[i].xyz) + d);
        if (distance <= params.threshold) {
            count++;
        }
    }

    candidates[sample_idx].coefficients = vec4<f32>(normal, d);
    candidates[sample_idx].inlier_count = count;
}
"#;

const RANSAC_INLIER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> flags: array<u32>;
@group(0) @binding(2) var<uniform> params: InlierParams;

struct InlierParams {
    num_points: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    plane: vec4<f32>,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_points) {
        return;
    }

    let normal_len = length(params.plane.xyz);
    if (normal_len < 1e-8) {
        flags[index] = 0u;
        return;
    }

    let distance = abs(dot(params.plane.xyz, points[index].xyz) + params.plane.w) / normal_len;
    flags[index] = select(0u, 1u, distance <= params.threshold);
}
"#;

const RADIUS_NEIGHBOR_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> candidate_counts: array<u32>;
@group(0) @binding(2) var<storage, read> candidates: array<array<u32, MAX_CANDIDATES>>;
@group(0) @binding(3) var<storage, read_write> neighbor_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> neighbors: array<array<u32, MAX_NEIGHBORS>>;
@group(0) @binding(5) var<uniform> params: ClusterParams;

struct ClusterParams {
    num_points: u32,
    max_neighbors: u32,
    max_candidates: u32,
    _pad0: u32,
    tolerance: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_points) {
        return;
    }

    let center = points[index].xyz;
    var stored = 0u;
    let candidate_count = min(candidate_counts[index], params.max_candidates);

    for (var slot = 0u; slot < candidate_count; slot++) {
        let candidate_index = candidates[index][slot];
        if (candidate_index >= params.num_points || candidate_index == index) {
            continue;
        }

        let distance = length(points[candidate_index].xyz - center);
        if (distance <= params.tolerance) {
            if (stored < params.max_neighbors) {
                neighbors[index][stored] = candidate_index;
                stored++;
            }
        }
    }

    neighbor_counts[index] = stored;
}
"#;

/// A 3D plane model defined by `ax + by + cz + d = 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuPlaneModel {
    /// Plane coefficients `[a, b, c, d]`.
    pub coefficients: Vector4<f32>,
}

impl GpuPlaneModel {
    /// Create a model from normalized coefficients.
    pub fn new(coefficients: Vector4<f32>) -> Self {
        Self { coefficients }
    }

    /// Return the plane normal.
    pub fn normal(&self) -> Vector3<f32> {
        Vector3::new(
            self.coefficients.x,
            self.coefficients.y,
            self.coefficients.z,
        )
    }

    /// Distance from `point` to this plane.
    pub fn distance_to_point(&self, point: &Point3f) -> f32 {
        let normal = self.normal();
        let normal_len = normal.magnitude();
        if normal_len < 1e-8 {
            return f32::INFINITY;
        }

        (normal.dot(&point.coords) + self.coefficients.w).abs() / normal_len
    }
}

/// Result of GPU RANSAC plane segmentation.
#[derive(Debug, Clone)]
pub struct GpuPlaneSegmentationResult {
    /// Best-scoring plane model.
    pub plane: GpuPlaneModel,
    /// Best-scoring plane model.
    pub model: GpuPlaneModel,
    /// Indices of points within the distance threshold.
    pub inliers: Vec<u32>,
    /// Number of RANSAC candidates evaluated.
    pub iterations: usize,
}

/// Configuration for GPU RANSAC plane segmentation.
#[derive(Debug, Clone, Copy)]
pub struct GpuPlaneSegmentationConfig {
    /// Maximum RANSAC candidates to evaluate.
    pub max_iterations: usize,
    /// Maximum point-to-plane distance for an inlier.
    pub distance_threshold: f32,
    /// Minimum inliers required for a valid result.
    pub min_inliers: usize,
}

impl Default for GpuPlaneSegmentationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            distance_threshold: 0.02,
            min_inliers: 1,
        }
    }
}

/// Configuration for GPU Euclidean cluster extraction.
#[derive(Debug, Clone)]
pub struct GpuEuclideanClusterConfig {
    /// Maximum distance between neighboring points in the same cluster.
    pub tolerance: f32,
    /// Minimum number of points for a valid cluster.
    pub min_cluster_size: usize,
    /// Maximum number of points allowed in a valid cluster.
    pub max_cluster_size: usize,
    /// Maximum radius neighbors retained per point from the GPU adjacency pass.
    pub max_neighbors: usize,
}

impl Default for GpuEuclideanClusterConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.02,
            min_cluster_size: 100,
            max_cluster_size: 25_000,
            max_neighbors: 64,
        }
    }
}

impl GpuEuclideanClusterConfig {
    /// Create a config using the default `max_neighbors` cap.
    pub fn new(tolerance: f32, min_cluster_size: usize, max_cluster_size: usize) -> Self {
        Self {
            tolerance,
            min_cluster_size,
            max_cluster_size,
            ..Self::default()
        }
    }

    /// Create a config with an explicit GPU neighbor cap.
    pub fn with_max_neighbors(
        tolerance: f32,
        min_cluster_size: usize,
        max_cluster_size: usize,
        max_neighbors: usize,
    ) -> Self {
        Self {
            tolerance,
            min_cluster_size,
            max_cluster_size,
            max_neighbors,
        }
    }
}

/// Issue-compatible alias for GPU Euclidean cluster extraction config.
pub type GpuClusterConfig = GpuEuclideanClusterConfig;

/// Result of GPU-accelerated Euclidean cluster extraction.
#[derive(Debug, Clone)]
pub struct GpuClusterExtractionResult {
    /// Each inner vector contains point indices for one cluster, largest first.
    pub clusters: Vec<Vec<usize>>,
}

impl GpuClusterExtractionResult {
    /// Number of clusters found.
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Extract a sub-cloud for the cluster at `index`.
    pub fn get_cluster_cloud(
        &self,
        cloud: &PointCloud<Point3f>,
        index: usize,
    ) -> Option<PointCloud<Point3f>> {
        self.clusters.get(index).map(|indices| {
            PointCloud::from_points(indices.iter().map(|&i| cloud.points[i]).collect())
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PlaneCandidate {
    coefficients: [f32; 4],
    inlier_count: u32,
    _padding: [u32; 3],
}

impl GpuContext {
    /// Segment the dominant plane using GPU-scored RANSAC candidates.
    pub async fn segment_plane(
        &self,
        points: &[Point3f],
        config: GpuPlaneSegmentationConfig,
    ) -> Result<GpuPlaneSegmentationResult> {
        validate_ransac_config(points, config)?;

        let mut result = self
            .segment_plane_ransac(points, config.distance_threshold, config.max_iterations)
            .await?;
        if result.inliers.len() < config.min_inliers {
            return Err(Error::Algorithm(format!(
                "Plane model has {} inliers, below required minimum {}",
                result.inliers.len(),
                config.min_inliers
            )));
        }

        result.plane = result.model.clone();
        Ok(result)
    }

    /// Segment the dominant plane using GPU-scored RANSAC candidates.
    pub async fn segment_plane_ransac(
        &self,
        points: &[Point3f],
        threshold: f32,
        max_iters: usize,
    ) -> Result<GpuPlaneSegmentationResult> {
        validate_ransac_inputs(points, threshold, max_iters)?;

        let point_data = points_to_vec4(points);
        let sample_count = max_iters.min(u32::MAX as usize);
        let samples = generate_ransac_samples(points.len(), sample_count);

        let input_buffer =
            self.create_buffer_init("RANSAC Points", &point_data, wgpu::BufferUsages::STORAGE);
        let samples_buffer =
            self.create_buffer_init("RANSAC Samples", &samples, wgpu::BufferUsages::STORAGE);
        let candidates_buffer = self.create_buffer(
            "RANSAC Candidates",
            (sample_count * std::mem::size_of::<PlaneCandidate>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RansacParams {
            num_points: u32,
            num_samples: u32,
            threshold: f32,
            _padding: u32,
        }

        let params = RansacParams {
            num_points: points.len() as u32,
            num_samples: sample_count as u32,
            threshold,
            _padding: 0,
        };
        let params_buffer =
            self.create_buffer_init("RANSAC Params", &[params], wgpu::BufferUsages::UNIFORM);

        let shader = self.create_shader_module("RANSAC Score Shader", RANSAC_SCORE_SHADER);
        let layout = self.create_bind_group_layout(
            "RANSAC Score Layout",
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        );
        let pipeline = self.create_pipeline_with_layout("RANSAC Score Pipeline", &shader, &layout);
        let bind_group = self.create_bind_group(
            "RANSAC Score Bind Group",
            &layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: samples_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: candidates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RANSAC Score Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RANSAC Score Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(sample_count, 64) as u32, 1, 1);
        }

        let candidates = self
            .read_storage_buffer::<PlaneCandidate>(
                encoder,
                &candidates_buffer,
                sample_count,
                "RANSAC Candidate Staging",
            )
            .await?;

        let best = candidates
            .iter()
            .max_by_key(|candidate| candidate.inlier_count)
            .ok_or_else(|| Error::Algorithm("Failed to evaluate RANSAC candidates".to_string()))?;

        if best.inlier_count == 0 {
            return Err(Error::Algorithm(
                "Failed to find valid plane model".to_string(),
            ));
        }

        let coefficients = Vector4::new(
            best.coefficients[0],
            best.coefficients[1],
            best.coefficients[2],
            best.coefficients[3],
        );
        let inliers = self
            .plane_inlier_indices(points, coefficients, threshold)
            .await?;

        Ok(GpuPlaneSegmentationResult {
            plane: GpuPlaneModel::new(coefficients),
            model: GpuPlaneModel::new(coefficients),
            inliers,
            iterations: sample_count,
        })
    }

    /// Extract Euclidean clusters as point-cloud values.
    pub async fn extract_clusters(
        &self,
        cloud: &PointCloud<Point3f>,
        config: GpuClusterConfig,
    ) -> Result<Vec<PointCloud<Point3f>>> {
        let result = self
            .extract_euclidean_clusters(&cloud.points, &config)
            .await?;
        Ok(result
            .clusters
            .iter()
            .map(|indices| {
                PointCloud::from_points(indices.iter().map(|&i| cloud.points[i]).collect())
            })
            .collect())
    }

    /// Extract Euclidean clusters using GPU-computed radius adjacency.
    pub async fn extract_euclidean_clusters(
        &self,
        points: &[Point3f],
        config: &GpuEuclideanClusterConfig,
    ) -> Result<GpuClusterExtractionResult> {
        validate_cluster_inputs(points, config)?;

        let max_neighbors = config.max_neighbors.min(256).max(1);
        let max_candidates = max_neighbors.saturating_mul(8).clamp(max_neighbors, 1024);
        let point_data = points_to_vec4(points);
        let (candidate_counts, candidate_indices) =
            build_voxel_candidate_neighbors(points, config.tolerance, max_candidates)?;
        let input_buffer =
            self.create_buffer_init("Cluster Points", &point_data, wgpu::BufferUsages::STORAGE);
        let candidate_counts_buffer = self.create_buffer_init(
            "Cluster Candidate Counts",
            &candidate_counts,
            wgpu::BufferUsages::STORAGE,
        );
        let candidates_buffer = self.create_buffer_init(
            "Cluster Candidates",
            &candidate_indices,
            wgpu::BufferUsages::STORAGE,
        );
        let counts_buffer = self.create_buffer(
            "Cluster Neighbor Counts",
            (points.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let neighbors_buffer = self.create_buffer(
            "Cluster Neighbors",
            (points.len() * max_neighbors * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ClusterParams {
            num_points: u32,
            max_neighbors: u32,
            max_candidates: u32,
            _padding0: u32,
            tolerance: f32,
            _padding1: [u32; 3],
        }

        let params = ClusterParams {
            num_points: points.len() as u32,
            max_neighbors: max_neighbors as u32,
            max_candidates: max_candidates as u32,
            _padding0: 0,
            tolerance: config.tolerance,
            _padding1: [0; 3],
        };
        let params_buffer =
            self.create_buffer_init("Cluster Params", &[params], wgpu::BufferUsages::UNIFORM);

        let shader_source = RADIUS_NEIGHBOR_SHADER
            .replace("MAX_NEIGHBORS", &max_neighbors.to_string())
            .replace("MAX_CANDIDATES", &max_candidates.to_string());
        let shader = self.create_shader_module("Cluster Radius Neighbor Shader", &shader_source);
        let layout = self.create_bind_group_layout(
            "Cluster Radius Neighbor Layout",
            &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                uniform_entry(5),
            ],
        );
        let pipeline =
            self.create_pipeline_with_layout("Cluster Radius Neighbor Pipeline", &shader, &layout);
        let bind_group = self.create_bind_group(
            "Cluster Radius Neighbor Bind Group",
            &layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidate_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: candidates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: neighbors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cluster Radius Neighbor Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cluster Radius Neighbor Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(points.len(), 64) as u32, 1, 1);
        }

        let counts = self
            .read_storage_buffer::<u32>(
                encoder,
                &counts_buffer,
                points.len(),
                "Cluster Count Staging",
            )
            .await?;

        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cluster Neighbor Read Encoder"),
            });
        let neighbors = self
            .read_storage_buffer::<u32>(
                encoder,
                &neighbors_buffer,
                points.len() * max_neighbors,
                "Cluster Neighbor Staging",
            )
            .await?;

        let mut disjoint_set = DisjointSet::new(points.len());
        for point_idx in 0..points.len() {
            let count = counts[point_idx].min(max_neighbors as u32) as usize;
            let base = point_idx * max_neighbors;

            for &neighbor in &neighbors[base..base + count] {
                let neighbor = neighbor as usize;
                if neighbor < points.len() {
                    disjoint_set.union(point_idx, neighbor);
                }
            }
        }

        let mut by_root: HashMap<usize, usize> = HashMap::new();
        let mut clusters = Vec::new();
        for point_idx in 0..points.len() {
            let root = disjoint_set.find(point_idx);
            let cluster_idx = *by_root.entry(root).or_insert_with(|| {
                clusters.push(Vec::new());
                clusters.len() - 1
            });
            clusters[cluster_idx].push(point_idx);
        }

        clusters.retain(|cluster| {
            cluster.len() >= config.min_cluster_size && cluster.len() <= config.max_cluster_size
        });
        clusters.sort_by(|a, b| b.len().cmp(&a.len()));
        Ok(GpuClusterExtractionResult { clusters })
    }

    async fn plane_inlier_indices(
        &self,
        points: &[Point3f],
        coefficients: Vector4<f32>,
        threshold: f32,
    ) -> Result<Vec<u32>> {
        let point_data = points_to_vec4(points);
        let input_buffer = self.create_buffer_init(
            "Plane Inlier Points",
            &point_data,
            wgpu::BufferUsages::STORAGE,
        );
        let flags_buffer = self.create_buffer(
            "Plane Inlier Flags",
            (points.len() * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct InlierParams {
            num_points: u32,
            threshold: f32,
            _padding: [u32; 2],
            plane: [f32; 4],
        }

        let params = InlierParams {
            num_points: points.len() as u32,
            threshold,
            _padding: [0; 2],
            plane: [
                coefficients.x,
                coefficients.y,
                coefficients.z,
                coefficients.w,
            ],
        };
        let params_buffer = self.create_buffer_init(
            "Plane Inlier Params",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );
        let shader = self.create_shader_module("Plane Inlier Shader", RANSAC_INLIER_SHADER);
        let layout = self.create_bind_group_layout(
            "Plane Inlier Layout",
            &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
            ],
        );
        let pipeline = self.create_pipeline_with_layout("Plane Inlier Pipeline", &shader, &layout);
        let bind_group = self.create_bind_group(
            "Plane Inlier Bind Group",
            &layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Plane Inlier Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Plane Inlier Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(div_ceil(points.len(), 64) as u32, 1, 1);
        }

        let flags = self
            .read_storage_buffer::<u32>(
                encoder,
                &flags_buffer,
                points.len(),
                "Plane Inlier Staging",
            )
            .await?;
        Ok(flags
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 1).then_some(idx as u32))
            .collect())
    }

    fn create_pipeline_with_layout(
        &self,
        label: &str,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some(label),
                        bind_group_layouts: &[Some(layout)],
                        immediate_size: 0,
                    },
                )),
                module: shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }

    async fn read_storage_buffer<T: bytemuck::Pod>(
        &self,
        mut encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        len: usize,
        label: &str,
    ) -> Result<Vec<T>> {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let staging = self.create_buffer(
            label,
            size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Some(Ok(())) = receiver.receive().await {
            let data = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging.unmap();
            Ok(result)
        } else {
            Err(Error::Gpu(
                "Failed to read GPU segmentation results".to_string(),
            ))
        }
    }
}

/// GPU-accelerated RANSAC plane segmentation for a point cloud.
pub async fn gpu_segment_plane(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    config: GpuPlaneSegmentationConfig,
) -> Result<GpuPlaneSegmentationResult> {
    gpu_context.segment_plane(&cloud.points, config).await
}

/// GPU-accelerated RANSAC plane segmentation for a point cloud.
pub async fn gpu_segment_plane_ransac(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    threshold: f32,
    max_iters: usize,
) -> Result<GpuPlaneSegmentationResult> {
    gpu_context
        .segment_plane_ransac(&cloud.points, threshold, max_iters)
        .await
}

/// GPU-accelerated Euclidean cluster extraction for a point cloud.
pub async fn gpu_extract_clusters(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    config: GpuClusterConfig,
) -> Result<Vec<PointCloud<Point3f>>> {
    gpu_context.extract_clusters(cloud, config).await
}

/// GPU-accelerated Euclidean cluster extraction for a point cloud.
pub async fn gpu_extract_euclidean_clusters(
    gpu_context: &GpuContext,
    cloud: &PointCloud<Point3f>,
    config: &GpuEuclideanClusterConfig,
) -> Result<GpuClusterExtractionResult> {
    gpu_context
        .extract_euclidean_clusters(&cloud.points, config)
        .await
}

fn validate_ransac_config(points: &[Point3f], config: GpuPlaneSegmentationConfig) -> Result<()> {
    validate_ransac_inputs(points, config.distance_threshold, config.max_iterations)?;
    if config.min_inliers == 0 {
        return Err(Error::InvalidData(
            "min_inliers must be at least 1".to_string(),
        ));
    }
    Ok(())
}

fn validate_ransac_inputs(points: &[Point3f], threshold: f32, max_iters: usize) -> Result<()> {
    if points.len() < 3 {
        return Err(Error::InvalidData(
            "Need at least 3 points for plane segmentation".to_string(),
        ));
    }
    if threshold <= 0.0 {
        return Err(Error::InvalidData("Threshold must be positive".to_string()));
    }
    if max_iters == 0 {
        return Err(Error::InvalidData(
            "Max iterations must be positive".to_string(),
        ));
    }
    if points.len() > u32::MAX as usize {
        return Err(Error::InvalidData(
            "Point cloud is too large for GPU segmentation".to_string(),
        ));
    }
    Ok(())
}

fn validate_cluster_inputs(points: &[Point3f], config: &GpuEuclideanClusterConfig) -> Result<()> {
    if points.is_empty() {
        return Err(Error::InvalidData("Point cloud is empty".to_string()));
    }
    if config.tolerance <= 0.0 {
        return Err(Error::InvalidData("Tolerance must be positive".to_string()));
    }
    if config.min_cluster_size == 0 {
        return Err(Error::InvalidData(
            "min_cluster_size must be at least 1".to_string(),
        ));
    }
    if config.min_cluster_size > config.max_cluster_size {
        return Err(Error::InvalidData(
            "min_cluster_size must not exceed max_cluster_size".to_string(),
        ));
    }
    if config.max_neighbors == 0 {
        return Err(Error::InvalidData(
            "max_neighbors must be at least 1".to_string(),
        ));
    }
    if points.len() > u32::MAX as usize {
        return Err(Error::InvalidData(
            "Point cloud is too large for GPU clustering".to_string(),
        ));
    }
    Ok(())
}

fn points_to_vec4(points: &[Point3f]) -> Vec<[f32; 4]> {
    points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect()
}

fn build_voxel_candidate_neighbors(
    points: &[Point3f],
    tolerance: f32,
    max_candidates: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let total_candidates = points
        .len()
        .checked_mul(max_candidates)
        .ok_or_else(|| Error::InvalidData("Cluster candidate buffer is too large".to_string()))?;
    let mut bins: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();

    for (idx, point) in points.iter().enumerate() {
        bins.entry(voxel_key(point, tolerance))
            .or_default()
            .push(idx as u32);
    }

    let mut counts = vec![0u32; points.len()];
    let mut candidates = vec![u32::MAX; total_candidates];
    for (idx, point) in points.iter().enumerate() {
        let (vx, vy, vz) = voxel_key(point, tolerance);
        let base = idx * max_candidates;
        let mut stored = 0usize;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (vx + dx, vy + dy, vz + dz);
                    let Some(bucket) = bins.get(&key) else {
                        continue;
                    };

                    for &candidate in bucket {
                        if candidate as usize == idx {
                            continue;
                        }
                        if stored == max_candidates {
                            break;
                        }
                        candidates[base + stored] = candidate;
                        stored += 1;
                    }
                }
            }
        }

        counts[idx] = stored as u32;
    }

    Ok((counts, candidates))
}

fn voxel_key(point: &Point3f, tolerance: f32) -> (i32, i32, i32) {
    (
        (point.x / tolerance).floor() as i32,
        (point.y / tolerance).floor() as i32,
        (point.z / tolerance).floor() as i32,
    )
}

fn generate_ransac_samples(num_points: usize, sample_count: usize) -> Vec<[u32; 4]> {
    let mut state = ((num_points as u64) << 32) ^ sample_count as u64 ^ 0x9E37_79B9_7F4A_7C15;
    let mut samples = Vec::with_capacity(sample_count);

    for iteration in 0..sample_count {
        let mut a = next_index(&mut state, num_points);
        let mut b = next_index(&mut state, num_points);
        let mut c = next_index(&mut state, num_points);

        if a == b || a == c || b == c {
            a = iteration % num_points;
            b = (iteration.wrapping_mul(37) + 1) % num_points;
            c = (iteration.wrapping_mul(101) + 2) % num_points;
            while b == a {
                b = (b + 1) % num_points;
            }
            while c == a || c == b {
                c = (c + 1) % num_points;
            }
        }

        samples.push([a as u32, b as u32, c as u32, 0]);
    }

    samples
}

fn next_index(state: &mut u64, len: usize) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 32) as usize) % len
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return;
        }

        match self.rank[root_a].cmp(&self.rank[root_b]) {
            std::cmp::Ordering::Less => self.parent[root_a] = root_b,
            std::cmp::Ordering::Greater => self.parent[root_b] = root_a,
            std::cmp::Ordering::Equal => {
                self.parent[root_b] = root_a;
                self.rank[root_a] += 1;
            }
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn try_create_gpu_context() -> Option<GpuContext> {
        match GpuContext::new().await {
            Ok(gpu) => Some(gpu),
            Err(_) => {
                println!("GPU not available, skipping GPU-dependent test");
                None
            }
        }
    }

    fn planar_cloud() -> PointCloud<Point3f> {
        let mut cloud = PointCloud::new();
        for x in 0..12 {
            for y in 0..12 {
                cloud.push(Point3f::new(x as f32 * 0.1, y as f32 * 0.1, 0.0));
            }
        }
        cloud.push(Point3f::new(0.0, 0.0, 3.0));
        cloud.push(Point3f::new(1.0, 1.0, -3.0));
        cloud
    }

    fn clustered_cloud() -> PointCloud<Point3f> {
        let mut cloud = PointCloud::new();
        for i in 0..30 {
            let x = (i % 5) as f32 * 0.04;
            let y = (i / 5) as f32 * 0.04;
            cloud.push(Point3f::new(x, y, 0.0));
        }
        for i in 0..20 {
            let x = 5.0 + (i % 5) as f32 * 0.04;
            let y = (i / 5) as f32 * 0.04;
            cloud.push(Point3f::new(x, y, 0.0));
        }
        cloud
    }

    #[test]
    fn test_gpu_ransac_plane() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let cloud = planar_cloud();
            let config = GpuPlaneSegmentationConfig {
                max_iterations: 256,
                distance_threshold: 0.01,
                min_inliers: 140,
            };
            let result = gpu_segment_plane(&gpu, &cloud, config).await.unwrap();

            assert!(result.inliers.len() >= 140);
            assert!(result.plane.normal().z.abs() > 0.9);
            assert_eq!(result.model, result.plane);
            assert_eq!(result.iterations, 256);
        });
    }

    #[test]
    fn test_gpu_clusters() {
        pollster::block_on(async {
            let Some(gpu) = try_create_gpu_context().await else {
                return;
            };

            let cloud = clustered_cloud();
            let config = GpuEuclideanClusterConfig::with_max_neighbors(0.09, 5, 100, 16);
            let result = gpu_extract_euclidean_clusters(&gpu, &cloud, &config)
                .await
                .unwrap();
            let cluster_clouds = gpu_extract_clusters(&gpu, &cloud, config).await.unwrap();

            assert_eq!(result.num_clusters(), 2);
            assert_eq!(result.clusters[0].len(), 30);
            assert_eq!(result.clusters[1].len(), 20);
            assert_eq!(cluster_clouds.len(), 2);
            assert_eq!(cluster_clouds[0].len(), 30);
            assert_eq!(cluster_clouds[1].len(), 20);
        });
    }

    #[test]
    fn test_invalid_inputs() {
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        assert!(validate_ransac_inputs(&cloud.points, 0.1, 10).is_err());
        assert!(validate_ransac_inputs(&planar_cloud().points, -0.1, 10).is_err());
        assert!(validate_ransac_inputs(&planar_cloud().points, 0.1, 0).is_err());
        assert!(validate_ransac_config(
            &planar_cloud().points,
            GpuPlaneSegmentationConfig {
                min_inliers: 0,
                ..GpuPlaneSegmentationConfig::default()
            }
        )
        .is_err());

        let config = GpuEuclideanClusterConfig::new(-1.0, 1, 10);
        assert!(validate_cluster_inputs(&clustered_cloud().points, &config).is_err());
    }
}
