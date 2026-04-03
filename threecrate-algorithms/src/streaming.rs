//! Streaming point cloud processing pipeline.
//!
//! Enables out-of-core processing of arbitrarily large point clouds by reading
//! and processing data in bounded-size chunks.  Only one chunk resides in RAM at
//! a time; the pipeline accumulates lightweight per-chunk state (e.g. a voxel map)
//! that is orders of magnitude smaller than the full dataset.
//!
//! # Architecture
//!
//! ```text
//! Source iterator                Pipeline stage           Output
//! (file / network / …)  ──►  process_chunk(&[T])  ──►  finalize()
//!      chunk 0                  accumulate state
//!      chunk 1                  accumulate state
//!      …                        …
//!      chunk N                  accumulate state
//! ```
//!
//! # Provided pipelines
//!
//! | Type | Description |
//! |---|---|
//! | [`StreamingVoxelFilter`] | Downsamples via a voxel grid; O(voxels) memory |
//! | [`StreamingStatistics`] | Accumulates bounding-box and point count |
//! | [`StreamingCollector`] | Collects all points (useful for testing) |
//!
//! # Example
//!
//! ```rust
//! use threecrate_algorithms::streaming::{
//!     StreamingPipeline, StreamingVoxelFilter, StreamingVoxelFilterConfig, run_pipeline,
//! };
//! use threecrate_core::Point3f;
//!
//! let points: Vec<Result<Point3f, _>> = vec![
//!     Ok(Point3f::new(0.0, 0.0, 0.0)),
//!     Ok(Point3f::new(0.05, 0.0, 0.0)),
//!     Ok(Point3f::new(1.0, 1.0, 1.0)),
//! ];
//! let mut filter = StreamingVoxelFilter::new(StreamingVoxelFilterConfig { voxel_size: 0.1 });
//! let stats = run_pipeline(&mut filter, points.into_iter(), 2).unwrap();
//! let cloud = filter.finalize().unwrap();
//! println!("Downsampled to {} points", cloud.len());
//! ```

use std::collections::HashMap;
use threecrate_core::{Error, Point3f, PointCloud, Result};

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// Trait for chunk-based streaming processors.
///
/// Implementations accumulate state across calls to [`process_chunk`] and
/// produce a final result via [`finalize`].  The chunk size controls peak RAM
/// usage: smaller chunks use less memory at the cost of more function-call
/// overhead.
///
/// [`process_chunk`]: StreamingPipeline::process_chunk
/// [`finalize`]: StreamingPipeline::finalize
pub trait StreamingPipeline<T> {
    /// The type produced after all chunks have been processed.
    type Output;

    /// Ingest one chunk of items.  Called repeatedly until the source is
    /// exhausted.  `chunk` will never be empty.
    fn process_chunk(&mut self, chunk: &[T]) -> Result<()>;

    /// Consume the pipeline and return the accumulated output.
    fn finalize(self) -> Result<Self::Output>;

    /// Estimated number of bytes currently held by this pipeline stage.
    /// Default returns `0`; override to expose real memory usage.
    fn memory_bytes(&self) -> usize { 0 }
}

// ---------------------------------------------------------------------------
// Pipeline runner
// ---------------------------------------------------------------------------

/// Statistics reported by [`run_pipeline`].
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    /// Total number of successfully processed items.
    pub items_processed: usize,
    /// Number of chunks delivered to the pipeline.
    pub chunks_processed: usize,
    /// Number of items skipped due to errors (if `skip_errors` is set).
    pub errors_skipped: usize,
}

/// Options for [`run_pipeline`].
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// If `true`, item-level errors from the source iterator are counted and
    /// skipped rather than causing [`run_pipeline`] to return early.
    /// Default: `false`.
    pub skip_errors: bool,
}

impl Default for RunOptions {
    fn default() -> Self { Self { skip_errors: false } }
}

/// Drive `pipeline` by reading from `source` in chunks of `chunk_size` items.
///
/// Returns [`RunStats`] on success.  The iterator's item errors are propagated
/// unless [`RunOptions::skip_errors`] is set.
///
/// # Arguments
/// * `pipeline`   – A mutable reference to a [`StreamingPipeline`].
/// * `source`     – Any iterator whose items are `Result<T>`.
/// * `chunk_size` – Number of items to accumulate before calling
///                  [`StreamingPipeline::process_chunk`].  Must be ≥ 1.
pub fn run_pipeline<T, P>(
    pipeline: &mut P,
    source: impl Iterator<Item = Result<T>>,
    chunk_size: usize,
) -> Result<RunStats>
where
    P: StreamingPipeline<T>,
{
    run_pipeline_with_options(pipeline, source, chunk_size, &RunOptions::default())
}

/// Like [`run_pipeline`] but accepts explicit options.
pub fn run_pipeline_with_options<T, P>(
    pipeline: &mut P,
    source: impl Iterator<Item = Result<T>>,
    chunk_size: usize,
    opts: &RunOptions,
) -> Result<RunStats>
where
    P: StreamingPipeline<T>,
{
    if chunk_size == 0 {
        return Err(Error::InvalidData("chunk_size must be ≥ 1".into()));
    }

    let mut stats = RunStats::default();
    let mut chunk: Vec<T> = Vec::with_capacity(chunk_size);

    for item in source {
        match item {
            Ok(point) => {
                chunk.push(point);
                if chunk.len() == chunk_size {
                    pipeline.process_chunk(&chunk)?;
                    stats.items_processed += chunk.len();
                    stats.chunks_processed += 1;
                    chunk.clear();
                }
            }
            Err(e) => {
                if opts.skip_errors {
                    stats.errors_skipped += 1;
                } else {
                    return Err(e);
                }
            }
        }
    }

    // Flush any remaining items.
    if !chunk.is_empty() {
        pipeline.process_chunk(&chunk)?;
        stats.items_processed += chunk.len();
        stats.chunks_processed += 1;
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// StreamingVoxelFilter
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingVoxelFilter`].
#[derive(Debug, Clone)]
pub struct StreamingVoxelFilterConfig {
    /// Side length of each cubic voxel (same units as the point coordinates).
    /// Must be positive.
    pub voxel_size: f32,
}

/// Streaming voxel-grid downsampler.
///
/// Maintains a [`HashMap`] from voxel coordinates to a representative point.
/// Peak memory is `O(V)` where `V` is the number of occupied voxels in the
/// entire dataset — typically far smaller than N points.
///
/// Unlike the in-memory [`voxel_grid_filter`](crate::filtering::voxel_grid_filter),
/// no bounding-box pre-scan is required; voxel keys are derived by dividing
/// each coordinate by `voxel_size` and rounding toward negative infinity, so
/// they are consistent across all chunks.
///
/// The representative point for each voxel is the **centroid** of all points
/// assigned to that voxel, giving a smoother result than first-point selection.
pub struct StreamingVoxelFilter {
    config: StreamingVoxelFilterConfig,
    /// Accumulated sum and count for centroid computation.
    voxels: HashMap<(i32, i32, i32), ([f64; 3], u32)>,
}

impl StreamingVoxelFilter {
    /// Create a new streaming voxel filter.
    pub fn new(config: StreamingVoxelFilterConfig) -> Self {
        Self { config, voxels: HashMap::new() }
    }

    #[inline]
    fn voxel_key(&self, p: &Point3f) -> (i32, i32, i32) {
        let inv = 1.0 / self.config.voxel_size;
        (
            (p.x * inv).floor() as i32,
            (p.y * inv).floor() as i32,
            (p.z * inv).floor() as i32,
        )
    }

    /// Number of occupied voxels accumulated so far.
    pub fn voxel_count(&self) -> usize { self.voxels.len() }
}

impl StreamingPipeline<Point3f> for StreamingVoxelFilter {
    type Output = PointCloud<Point3f>;

    fn process_chunk(&mut self, chunk: &[Point3f]) -> Result<()> {
        if self.config.voxel_size <= 0.0 {
            return Err(Error::InvalidData("voxel_size must be positive".into()));
        }
        for p in chunk {
            let key = self.voxel_key(p);
            let entry = self.voxels.entry(key).or_insert(([0.0; 3], 0));
            entry.0[0] += p.x as f64;
            entry.0[1] += p.y as f64;
            entry.0[2] += p.z as f64;
            entry.1 += 1;
        }
        Ok(())
    }

    fn finalize(self) -> Result<PointCloud<Point3f>> {
        let points: Vec<Point3f> = self
            .voxels
            .values()
            .map(|(sum, count)| {
                let n = *count as f64;
                Point3f::new((sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32)
            })
            .collect();
        Ok(PointCloud::from_points(points))
    }

    fn memory_bytes(&self) -> usize {
        // Each entry: key (12 bytes) + value (28 bytes) + HashMap overhead (~50 bytes).
        self.voxels.len() * 90
    }
}

// ---------------------------------------------------------------------------
// StreamingStatistics
// ---------------------------------------------------------------------------

/// Accumulated statistics produced by [`StreamingStatistics`].
#[derive(Debug, Clone)]
pub struct PointCloudStats {
    /// Total number of points processed.
    pub point_count: u64,
    /// Minimum coordinate (axis-aligned bounding box corner).
    pub min: Point3f,
    /// Maximum coordinate (axis-aligned bounding box corner).
    pub max: Point3f,
    /// Per-axis mean coordinates.
    pub mean: Point3f,
}

/// Streaming statistics collector.
///
/// Computes bounding box, point count, and mean position in a single pass
/// without retaining any individual points.  Peak memory is `O(1)`.
pub struct StreamingStatistics {
    count: u64,
    min: [f32; 3],
    max: [f32; 3],
    sum: [f64; 3],
}

impl StreamingStatistics {
    /// Create a new statistics collector.
    pub fn new() -> Self {
        Self {
            count: 0,
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            sum: [0.0; 3],
        }
    }
}

impl Default for StreamingStatistics {
    fn default() -> Self { Self::new() }
}

impl StreamingPipeline<Point3f> for StreamingStatistics {
    type Output = PointCloudStats;

    fn process_chunk(&mut self, chunk: &[Point3f]) -> Result<()> {
        for p in chunk {
            self.count += 1;
            self.min[0] = self.min[0].min(p.x);
            self.min[1] = self.min[1].min(p.y);
            self.min[2] = self.min[2].min(p.z);
            self.max[0] = self.max[0].max(p.x);
            self.max[1] = self.max[1].max(p.y);
            self.max[2] = self.max[2].max(p.z);
            self.sum[0] += p.x as f64;
            self.sum[1] += p.y as f64;
            self.sum[2] += p.z as f64;
        }
        Ok(())
    }

    fn finalize(self) -> Result<PointCloudStats> {
        if self.count == 0 {
            return Err(Error::InvalidData("no points were processed".into()));
        }
        let n = self.count as f64;
        Ok(PointCloudStats {
            point_count: self.count,
            min: Point3f::new(self.min[0], self.min[1], self.min[2]),
            max: Point3f::new(self.max[0], self.max[1], self.max[2]),
            mean: Point3f::new(
                (self.sum[0] / n) as f32,
                (self.sum[1] / n) as f32,
                (self.sum[2] / n) as f32,
            ),
        })
    }

    fn memory_bytes(&self) -> usize { std::mem::size_of::<Self>() }
}

// ---------------------------------------------------------------------------
// StreamingCollector
// ---------------------------------------------------------------------------

/// Streaming pipeline stage that collects all points into a `PointCloud`.
///
/// Useful for testing or as a terminal stage when the full cloud must
/// eventually be materialized (e.g. after prior stages have filtered it down).
pub struct StreamingCollector {
    points: Vec<Point3f>,
}

impl StreamingCollector {
    /// Create a new collector.
    pub fn new() -> Self { Self { points: Vec::new() } }

    /// Create a collector with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self { points: Vec::with_capacity(cap) }
    }
}

impl Default for StreamingCollector {
    fn default() -> Self { Self::new() }
}

impl StreamingPipeline<Point3f> for StreamingCollector {
    type Output = PointCloud<Point3f>;

    fn process_chunk(&mut self, chunk: &[Point3f]) -> Result<()> {
        self.points.extend_from_slice(chunk);
        Ok(())
    }

    fn finalize(self) -> Result<PointCloud<Point3f>> {
        Ok(PointCloud::from_points(self.points))
    }

    fn memory_bytes(&self) -> usize {
        self.points.len() * std::mem::size_of::<Point3f>()
    }
}

// ---------------------------------------------------------------------------
// Streaming source helpers
// ---------------------------------------------------------------------------

/// Wrap a `PointCloud` as a streaming source of `Result<Point3f>`.
///
/// Useful for testing pipelines without a file on disk.
pub fn cloud_as_stream(
    cloud: &PointCloud<Point3f>,
) -> impl Iterator<Item = Result<Point3f>> + '_ {
    cloud.points.iter().copied().map(Ok)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_cloud(n: usize) -> PointCloud<Point3f> {
        let pts: Vec<Point3f> = (0..n)
            .map(|i| Point3f::new(i as f32 * 0.1, 0.0, 0.0))
            .collect();
        PointCloud::from_points(pts)
    }

    // ---- StreamingCollector -----------------------------------------------

    #[test]
    fn test_collector_round_trip() {
        let cloud = grid_cloud(25);
        let mut collector = StreamingCollector::new();
        let stats = run_pipeline(&mut collector, cloud_as_stream(&cloud), 8).unwrap();
        let out = collector.finalize().unwrap();

        assert_eq!(stats.items_processed, 25);
        assert_eq!(stats.chunks_processed, 4); // 8+8+8+1
        assert_eq!(out.len(), 25);
    }

    // ---- StreamingStatistics ----------------------------------------------

    #[test]
    fn test_statistics_correctness() {
        // Three points: (0,0,0), (1,0,0), (2,0,0)
        let cloud = PointCloud::from_points(vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
            Point3f::new(2.0, 0.0, 0.0),
        ]);
        let mut stats_pipe = StreamingStatistics::new();
        run_pipeline(&mut stats_pipe, cloud_as_stream(&cloud), 2).unwrap();
        let s = stats_pipe.finalize().unwrap();

        assert_eq!(s.point_count, 3);
        assert!((s.min.x - 0.0).abs() < 1e-6);
        assert!((s.max.x - 2.0).abs() < 1e-6);
        assert!((s.mean.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_statistics_empty_fails() {
        let mut stats_pipe = StreamingStatistics::new();
        // No chunks processed — finalize should error.
        assert!(stats_pipe.finalize().is_err());
    }

    // ---- StreamingVoxelFilter ---------------------------------------------

    #[test]
    fn test_voxel_filter_reduces_density() {
        // 100 points at x=0..9.9 in steps of 0.1 → should collapse to ~10 voxels
        // with voxel_size=1.0.
        let cloud = grid_cloud(100);
        let config = StreamingVoxelFilterConfig { voxel_size: 1.0 };
        let mut filter = StreamingVoxelFilter::new(config);
        run_pipeline(&mut filter, cloud_as_stream(&cloud), 32).unwrap();
        let out = filter.finalize().unwrap();

        assert!(out.len() <= 10, "expected ≤10 voxels, got {}", out.len());
        assert!(!out.is_empty());
    }

    #[test]
    fn test_voxel_filter_centroid() {
        // Two points in the same voxel → centroid should be their midpoint.
        let cloud = PointCloud::from_points(vec![
            Point3f::new(0.1, 0.0, 0.0),
            Point3f::new(0.3, 0.0, 0.0),
        ]);
        let config = StreamingVoxelFilterConfig { voxel_size: 1.0 };
        let mut filter = StreamingVoxelFilter::new(config);
        run_pipeline(&mut filter, cloud_as_stream(&cloud), 10).unwrap();
        let out = filter.finalize().unwrap();

        assert_eq!(out.len(), 1);
        assert!((out.points[0].x - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_voxel_filter_across_chunk_boundary() {
        // The two points that belong to the same voxel are split across chunks.
        // The filter must still merge them.
        let cloud = PointCloud::from_points(vec![
            Point3f::new(0.1, 0.0, 0.0),
            Point3f::new(0.9, 0.0, 0.0),
        ]);
        let config = StreamingVoxelFilterConfig { voxel_size: 1.0 };
        let mut filter = StreamingVoxelFilter::new(config);
        // chunk_size=1 forces each point into its own chunk.
        run_pipeline(&mut filter, cloud_as_stream(&cloud), 1).unwrap();
        let out = filter.finalize().unwrap();

        assert_eq!(out.len(), 1, "points in the same voxel across chunks should merge");
        assert!((out.points[0].x - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_invalid_voxel_size() {
        let config = StreamingVoxelFilterConfig { voxel_size: -1.0 };
        let mut filter = StreamingVoxelFilter::new(config);
        let cloud = PointCloud::from_points(vec![Point3f::new(0.0, 0.0, 0.0)]);
        let result = run_pipeline(&mut filter, cloud_as_stream(&cloud), 1);
        assert!(result.is_err());
    }

    // ---- run_pipeline options --------------------------------------------

    #[test]
    fn test_skip_errors() {
        let source: Vec<Result<Point3f>> = vec![
            Ok(Point3f::new(0.0, 0.0, 0.0)),
            Err(Error::InvalidData("bad point".into())),
            Ok(Point3f::new(1.0, 0.0, 0.0)),
        ];
        let mut collector = StreamingCollector::new();
        let run_stats = run_pipeline_with_options(
            &mut collector,
            source.into_iter(),
            10,
            &RunOptions { skip_errors: true },
        )
        .unwrap();
        let out = collector.finalize().unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(run_stats.errors_skipped, 1);
    }

    #[test]
    fn test_error_propagation() {
        let source: Vec<Result<Point3f>> = vec![
            Ok(Point3f::new(0.0, 0.0, 0.0)),
            Err(Error::InvalidData("bad point".into())),
        ];
        let mut collector = StreamingCollector::new();
        // Default options: errors propagate.
        assert!(run_pipeline(&mut collector, source.into_iter(), 10).is_err());
    }

    #[test]
    fn test_chunk_size_zero_fails() {
        let mut collector = StreamingCollector::new();
        let result = run_pipeline(
            &mut collector,
            std::iter::empty::<Result<Point3f>>(),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_bytes() {
        let mut filter = StreamingVoxelFilter::new(StreamingVoxelFilterConfig { voxel_size: 0.5 });
        let cloud = grid_cloud(20);
        run_pipeline(&mut filter, cloud_as_stream(&cloud), 5).unwrap();
        // At least some voxels should be occupied and memory should be > 0.
        assert!(filter.memory_bytes() > 0);
    }
}
