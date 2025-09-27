//! Parallel processing utilities for surface reconstruction algorithms
//!
//! This module provides configurable thread pool management and parallel processing
//! optimizations for all reconstruction algorithms in the crate.

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, Mutex, OnceLock};
use threecrate_core::Result;

/// Global thread pool configuration for reconstruction algorithms
static GLOBAL_THREAD_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();
static THREAD_POOL_CONFIG: Mutex<ThreadPoolConfig> = Mutex::new(ThreadPoolConfig::new());

/// Thread pool configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of threads to use (None = automatic)
    pub num_threads: Option<usize>,
    /// Thread stack size in bytes
    pub stack_size: Option<usize>,
    /// Thread name prefix
    pub thread_name_prefix: String,
    /// Enable parallel processing (can be disabled for debugging)
    pub enabled: bool,
    /// Minimum chunk size for parallel iteration
    pub min_chunk_size: usize,
    /// Maximum chunk size for parallel iteration
    pub max_chunk_size: usize,
    /// Adaptive chunk sizing based on workload
    pub adaptive_chunks: bool,
}

impl ThreadPoolConfig {
    /// Create a new thread pool configuration with defaults
    const fn new() -> Self {
        Self {
            num_threads: None,
            stack_size: None,
            thread_name_prefix: String::new(),
            enabled: true,
            min_chunk_size: 100,
            max_chunk_size: 10000,
            adaptive_chunks: true,
        }
    }

    /// Create default configuration
    pub fn default() -> Self {
        Self {
            num_threads: None,
            stack_size: Some(8 * 1024 * 1024), // 8MB stack
            thread_name_prefix: "threecrate-recon".to_string(),
            enabled: true,
            min_chunk_size: 100,
            max_chunk_size: 10000,
            adaptive_chunks: true,
        }
    }

    /// Set number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Set stack size
    pub fn with_stack_size(mut self, stack_size: usize) -> Self {
        self.stack_size = Some(stack_size);
        self
    }

    /// Enable or disable parallel processing
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set chunk size range
    pub fn with_chunk_size_range(mut self, min: usize, max: usize) -> Self {
        self.min_chunk_size = min;
        self.max_chunk_size = max;
        self
    }

    /// Enable adaptive chunk sizing
    pub fn with_adaptive_chunks(mut self, adaptive: bool) -> Self {
        self.adaptive_chunks = adaptive;
        self
    }
}

/// Initialize the global thread pool with custom configuration
pub fn init_thread_pool(config: ThreadPoolConfig) -> Result<()> {
    if GLOBAL_THREAD_POOL.get().is_some() {
        return Ok(()); // Already initialized
    }

    let mut builder = ThreadPoolBuilder::new();

    if let Some(num_threads) = config.num_threads {
        builder = builder.num_threads(num_threads);
    }

    if let Some(stack_size) = config.stack_size {
        builder = builder.stack_size(stack_size);
    }

    if !config.thread_name_prefix.is_empty() {
        let prefix = config.thread_name_prefix.clone();
        builder = builder.thread_name(move |index| format!("{}-{}", prefix, index));
    }

    let pool = builder.build().map_err(|e| {
        threecrate_core::Error::Algorithm(format!("Failed to create thread pool: {}", e))
    })?;

    // Store configuration
    if let Ok(mut global_config) = THREAD_POOL_CONFIG.lock() {
        *global_config = config;
    }

    GLOBAL_THREAD_POOL.set(Arc::new(pool)).map_err(|_| {
        threecrate_core::Error::Algorithm("Thread pool already initialized".to_string())
    })?;

    Ok(())
}

/// Get the global thread pool, initializing with defaults if needed
pub fn get_thread_pool() -> Arc<ThreadPool> {
    GLOBAL_THREAD_POOL
        .get_or_init(|| {
            let config = ThreadPoolConfig::default();
            let pool = ThreadPoolBuilder::new()
                .num_threads(config.num_threads.unwrap_or_else(num_cpus::get))
                .stack_size(config.stack_size.unwrap_or(8 * 1024 * 1024))
                .thread_name(|index| format!("threecrate-recon-{}", index))
                .build()
                .expect("Failed to create default thread pool");
            Arc::new(pool)
        })
        .clone()
}

/// Get current thread pool configuration
pub fn get_config() -> ThreadPoolConfig {
    THREAD_POOL_CONFIG
        .lock()
        .map(|config| config.clone())
        .unwrap_or_else(|_| ThreadPoolConfig::default())
}

/// Check if parallel processing is enabled
pub fn is_parallel_enabled() -> bool {
    get_config().enabled
}

/// Compute optimal chunk size based on data size and configuration
pub fn compute_chunk_size(data_size: usize) -> usize {
    let config = get_config();

    if !config.adaptive_chunks {
        return config
            .min_chunk_size
            .max(data_size / num_cpus::get())
            .min(config.max_chunk_size);
    }

    let num_threads = get_thread_pool().current_num_threads();
    let base_chunk_size = data_size / (num_threads * 4); // Aim for 4 chunks per thread

    base_chunk_size
        .max(config.min_chunk_size)
        .min(config.max_chunk_size)
}

/// Execute a parallel operation with the global thread pool
pub fn execute_parallel<F, R>(op: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if is_parallel_enabled() {
        get_thread_pool().install(op)
    } else {
        op()
    }
}

/// Parallel map operation with optimal chunking
pub fn parallel_map<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    if !is_parallel_enabled() || data.len() < get_config().min_chunk_size {
        return data.iter().map(f).collect();
    }

    execute_parallel(|| data.par_iter().map(f).collect())
}

/// Parallel map with index
pub fn parallel_map_indexed<T, U, F>(data: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(usize, &T) -> U + Sync + Send,
{
    if !is_parallel_enabled() || data.len() < get_config().min_chunk_size {
        return data.iter().enumerate().map(|(i, x)| f(i, x)).collect();
    }

    execute_parallel(|| data.par_iter().enumerate().map(|(i, x)| f(i, x)).collect())
}

/// Parallel filter operation
pub fn parallel_filter<T, F>(data: &[T], predicate: F) -> Vec<T>
where
    T: Clone + Sync + Send,
    F: Fn(&T) -> bool + Sync + Send,
{
    if !is_parallel_enabled() || data.len() < get_config().min_chunk_size {
        return data.iter().filter(|x| predicate(*x)).cloned().collect();
    }

    execute_parallel(|| data.par_iter().filter(|x| predicate(*x)).cloned().collect())
}

/// Parallel reduce operation
pub fn parallel_reduce<T, U, F, R>(data: &[T], identity: U, map_op: F, reduce_op: R) -> U
where
    T: Sync,
    U: Clone + Send + Sync,
    F: Fn(&T) -> U + Sync + Send,
    R: Fn(U, U) -> U + Sync + Send,
{
    if !is_parallel_enabled() || data.len() < get_config().min_chunk_size {
        return data.iter().map(map_op).fold(identity, reduce_op);
    }

    execute_parallel(|| {
        data.par_iter()
            .map(map_op)
            .reduce(|| identity.clone(), reduce_op)
    })
}

/// Parallel processing for point cloud operations
pub mod point_cloud {
    use super::*;
    use threecrate_core::Point3f;

    /// Parallel normal estimation for point clouds
    pub fn parallel_compute_normals<F>(
        points: &[Point3f],
        radius: f32,
        compute_normal: F,
    ) -> Vec<Point3f>
    where
        F: Fn(&Point3f, &[Point3f], f32) -> Point3f + Sync + Send,
    {
        parallel_map(points, |point| {
            // Find neighbors within radius (simplified for parallel processing)
            let neighbors: Vec<Point3f> = points
                .iter()
                .filter(|p| (*p - *point).magnitude() <= radius)
                .cloned()
                .collect();

            compute_normal(point, &neighbors, radius)
        })
    }

    /// Parallel distance computation between point clouds
    pub fn parallel_point_distances(points1: &[Point3f], points2: &[Point3f]) -> Vec<f32> {
        parallel_map(points1, |p1| {
            points2
                .iter()
                .map(|p2| (p1 - p2).magnitude())
                .fold(f32::INFINITY, f32::min)
        })
    }

    /// Parallel bounding box computation
    pub fn parallel_bounding_box(points: &[Point3f]) -> Option<(Point3f, Point3f)> {
        if points.is_empty() {
            return None;
        }

        let (min_vals, max_vals) = parallel_reduce(
            points,
            (
                Point3f::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
                Point3f::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
            ),
            |point| (*point, *point),
            |(min1, max1), (min2, max2)| {
                (
                    Point3f::new(min1.x.min(min2.x), min1.y.min(min2.y), min1.z.min(min2.z)),
                    Point3f::new(max1.x.max(max2.x), max1.y.max(max2.y), max1.z.max(max2.z)),
                )
            },
        );

        Some((min_vals, max_vals))
    }
}

/// Parallel processing for mesh operations
pub mod mesh {
    use super::*;
    use threecrate_core::Point3f;

    /// Parallel triangle normal computation
    pub fn parallel_triangle_normals(
        vertices: &[Point3f],
        faces: &[[usize; 3]],
    ) -> Vec<nalgebra::Vector3<f32>> {
        parallel_map(faces, |face| {
            let v1 = &vertices[face[0]];
            let v2 = &vertices[face[1]];
            let v3 = &vertices[face[2]];

            let edge1 = v2 - v1;
            let edge2 = v3 - v1;
            let normal = edge1.cross(&edge2).normalize();

            nalgebra::Vector3::new(normal.x, normal.y, normal.z)
        })
    }

    /// Parallel vertex normal computation (angle-weighted)
    pub fn parallel_vertex_normals(
        vertices: &[Point3f],
        faces: &[[usize; 3]],
    ) -> Vec<nalgebra::Vector3<f32>> {
        let triangle_normals = parallel_triangle_normals(vertices, faces);

        parallel_map_indexed(vertices, |vertex_idx, _vertex| {
            let mut normal = nalgebra::Vector3::zeros();
            let mut weight_sum = 0.0f32;

            for (face_idx, face) in faces.iter().enumerate() {
                if face.contains(&vertex_idx) {
                    // Compute angle weight for this vertex in this triangle
                    let face_normal = triangle_normals[face_idx];

                    // Find the position of vertex_idx in the face
                    let local_idx = face.iter().position(|&v| v == vertex_idx).unwrap();
                    let v1_idx = face[local_idx];
                    let v2_idx = face[(local_idx + 1) % 3];
                    let v3_idx = face[(local_idx + 2) % 3];

                    let v1 = &vertices[v1_idx];
                    let v2 = &vertices[v2_idx];
                    let v3 = &vertices[v3_idx];

                    // Compute angle at vertex
                    let edge1 = (v2 - v1).normalize();
                    let edge2 = (v3 - v1).normalize();
                    let angle = edge1.dot(&edge2).clamp(-1.0, 1.0).acos();

                    normal += face_normal * angle;
                    weight_sum += angle;
                }
            }

            if weight_sum > 1e-6 {
                normal / weight_sum
            } else {
                nalgebra::Vector3::new(0.0, 0.0, 1.0)
            }
        })
    }
}

/// Example usage of parallel processing configuration
///
/// ```rust
/// use threecrate_reconstruction::parallel::{init_thread_pool, ThreadPoolConfig};
///
/// // Configure thread pool with 4 threads and larger stack
/// let config = ThreadPoolConfig::default()
///     .with_threads(4)
///     .with_stack_size(16 * 1024 * 1024)
///     .with_chunk_size_range(200, 5000);
///
/// init_thread_pool(config).expect("Failed to initialize thread pool");
///
/// // Now all reconstruction algorithms will use parallel processing
/// ```
///
/// For performance optimization, you can also disable parallel processing for small datasets:
///
/// ```rust
/// use threecrate_reconstruction::parallel::ThreadPoolConfig;
///
/// let config = ThreadPoolConfig::default()
///     .with_enabled(false); // Disable for debugging or small datasets
/// ```

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::Point3f;

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig::default()
            .with_threads(4)
            .with_stack_size(16 * 1024 * 1024)
            .with_enabled(true);

        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.stack_size, Some(16 * 1024 * 1024));
        assert!(config.enabled);
    }

    #[test]
    fn test_chunk_size_computation() {
        let data_size = 10000;
        let chunk_size = compute_chunk_size(data_size);
        let config = get_config();

        assert!(chunk_size >= config.min_chunk_size);
        assert!(chunk_size <= config.max_chunk_size);
    }

    #[test]
    fn test_parallel_map() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_map(&data, |x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_filter() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let result = parallel_filter(&data, |x| *x % 2 == 0);
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn test_parallel_reduce() {
        let data = vec![1, 2, 3, 4, 5];
        let sum = parallel_reduce(&data, 0, |x| *x, |a, b| a + b);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_point_cloud_bounding_box() {
        let points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 1.0, 1.0),
            Point3f::new(-1.0, -1.0, -1.0),
            Point3f::new(2.0, 0.5, -0.5),
        ];

        let (min_pt, max_pt) = point_cloud::parallel_bounding_box(&points).unwrap();

        assert_eq!(min_pt.x, -1.0);
        assert_eq!(min_pt.y, -1.0);
        assert_eq!(min_pt.z, -1.0);
        assert_eq!(max_pt.x, 2.0);
        assert_eq!(max_pt.y, 1.0);
        assert_eq!(max_pt.z, 1.0);
    }
}
