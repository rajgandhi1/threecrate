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
//! For real-time sources (sensors, network streams), [`RealtimePipeline`] wraps
//! any [`StreamingPipeline`] with a bounded input queue and a background worker
//! thread.  When the queue is full the producer is automatically throttled
//! (backpressure); [`RealtimePipeline::try_send`] provides a non-blocking
//! variant that drops items instead of blocking.
//!
//! # Provided pipelines
//!
//! | Type | Description |
//! |---|---|
//! | [`StreamingVoxelFilter`] | Downsamples via a voxel grid; O(voxels) memory |
//! | [`StreamingStatistics`] | Accumulates bounding-box and point count |
//! | [`StreamingCollector`] | Collects all points (useful for testing) |
//! | [`RealtimePipeline`] | Drives any pipeline from a background thread with bounded-queue backpressure |
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Duration;
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
// Real-time streaming pipeline with backpressure
// ---------------------------------------------------------------------------

/// Configuration for [`RealtimePipeline`].
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum number of items buffered in the pipeline queue before backpressure
    /// kicks in.  [`RealtimePipeline::send`] blocks; [`RealtimePipeline::try_send`]
    /// drops the item and increments [`RealtimeMetrics::items_dropped`].
    pub max_queue_depth: usize,
    /// Number of items accumulated before calling [`StreamingPipeline::process_chunk`].
    pub chunk_size: usize,
    /// How long the worker waits for the next item before flushing a partial chunk.
    ///
    /// `Some(d)` bounds end-to-end latency at the cost of occasional small chunk
    /// calls.  `None` flushes only when a full chunk is available or the input
    /// is closed, which is more efficient for bulk/batch workloads.
    pub flush_timeout: Option<Duration>,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 1024,
            chunk_size: 256,
            flush_timeout: Some(Duration::from_millis(10)),
        }
    }
}

/// Real-time metrics snapshot from a [`RealtimePipeline`].
#[derive(Debug, Clone, Default)]
pub struct RealtimeMetrics {
    /// Total items successfully placed in the queue (accepted by `send` or `try_send`).
    pub items_queued: u64,
    /// Total items dequeued and processed by the background worker.
    pub items_processed: u64,
    /// Items dropped because the queue was full (only incremented by `try_send`).
    pub items_dropped: u64,
    /// Estimated current queue depth (`items_queued − items_processed`).
    pub estimated_queue_depth: u64,
}

struct SharedMetrics {
    items_queued: AtomicU64,
    items_processed: AtomicU64,
    items_dropped: AtomicU64,
}

impl SharedMetrics {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            items_queued: AtomicU64::new(0),
            items_processed: AtomicU64::new(0),
            items_dropped: AtomicU64::new(0),
        })
    }

    fn snapshot(&self) -> RealtimeMetrics {
        let queued = self.items_queued.load(Ordering::Relaxed);
        let processed = self.items_processed.load(Ordering::Relaxed);
        let dropped = self.items_dropped.load(Ordering::Relaxed);
        RealtimeMetrics {
            items_queued: queued,
            items_processed: processed,
            items_dropped: dropped,
            estimated_queue_depth: queued.saturating_sub(processed),
        }
    }
}

/// Real-time streaming pipeline with backpressure.
///
/// Wraps any [`StreamingPipeline`] with a bounded input queue and drives it
/// from a background worker thread.  Flow control works as follows:
///
/// - **[`send`]** blocks the caller when the queue is full — the producer is
///   naturally throttled without any explicit rate-limiter code.
/// - **[`try_send`]** never blocks; it drops the item and increments
///   [`RealtimeMetrics::items_dropped`] when the queue is full.
/// - Call **[`finish`]** to close the input, drain remaining items, join the
///   worker, and retrieve the pipeline's final output.
/// - Dropping the pipeline without calling `finish` is safe: the channel is
///   closed and the worker is joined in the `Drop` impl (result discarded).
///
/// [`send`]: RealtimePipeline::send
/// [`try_send`]: RealtimePipeline::try_send
/// [`finish`]: RealtimePipeline::finish
///
/// # Example
///
/// ```rust
/// use threecrate_algorithms::streaming::{
///     StreamingCollector, BackpressureConfig, RealtimePipeline,
/// };
/// use threecrate_core::Point3f;
///
/// let config = BackpressureConfig { max_queue_depth: 64, chunk_size: 16, ..Default::default() };
/// let rt = RealtimePipeline::new(StreamingCollector::new(), config);
/// for i in 0..50_u32 {
///     rt.send(Point3f::new(i as f32, 0.0, 0.0)).unwrap();
/// }
/// let cloud = rt.finish().unwrap();
/// assert_eq!(cloud.len(), 50);
/// ```
pub struct RealtimePipeline<T: Send + 'static, O: Send + 'static> {
    sender: Option<SyncSender<T>>,
    metrics: Arc<SharedMetrics>,
    join_handle: Option<JoinHandle<Result<O>>>,
}

impl<T: Send + 'static, O: Send + 'static> RealtimePipeline<T, O> {
    /// Create a new real-time pipeline backed by `pipeline`.
    ///
    /// The worker thread starts immediately and is ready to receive items.
    pub fn new<P>(pipeline: P, config: BackpressureConfig) -> Self
    where
        P: StreamingPipeline<T, Output = O> + Send + 'static,
    {
        assert!(config.chunk_size >= 1, "chunk_size must be ≥ 1");
        assert!(config.max_queue_depth >= 1, "max_queue_depth must be ≥ 1");

        let (sender, receiver) = sync_channel::<T>(config.max_queue_depth);
        let metrics = SharedMetrics::new();
        let metrics_worker = Arc::clone(&metrics);
        let chunk_size = config.chunk_size;
        let flush_timeout = config.flush_timeout;

        let join_handle = thread::spawn(move || {
            realtime_worker(receiver, pipeline, chunk_size, flush_timeout, metrics_worker)
        });

        Self { sender: Some(sender), metrics, join_handle: Some(join_handle) }
    }

    /// Send an item, blocking until queue space is available (backpressure).
    ///
    /// Returns `Err` if the worker thread has unexpectedly terminated.
    pub fn send(&self, item: T) -> Result<()> {
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| Error::InvalidData("pipeline already finished".into()))?;
        sender
            .send(item)
            .map_err(|_| Error::InvalidData("pipeline worker has terminated".into()))?;
        self.metrics.items_queued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Try to send without blocking.
    ///
    /// Returns `Ok(true)` when the item was queued, `Ok(false)` when the queue
    /// was full and the item was dropped (counted in
    /// [`RealtimeMetrics::items_dropped`]).  Returns `Err` if the worker has
    /// terminated unexpectedly.
    pub fn try_send(&self, item: T) -> Result<bool> {
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| Error::InvalidData("pipeline already finished".into()))?;
        match sender.try_send(item) {
            Ok(()) => {
                self.metrics.items_queued.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(TrySendError::Full(_)) => {
                self.metrics.items_dropped.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(Error::InvalidData("pipeline worker has terminated".into()))
            }
        }
    }

    /// Snapshot current pipeline metrics.
    pub fn metrics(&self) -> RealtimeMetrics {
        self.metrics.snapshot()
    }

    /// Close the input queue, wait for the worker to drain all buffered items,
    /// and return the pipeline's final output.
    pub fn finish(mut self) -> Result<O> {
        self.sender = None;
        self.join_handle
            .take()
            .expect("pipeline already finished")
            .join()
            .map_err(|_| Error::InvalidData("pipeline worker panicked".into()))?
    }
}

impl<T: Send + 'static, O: Send + 'static> Drop for RealtimePipeline<T, O> {
    fn drop(&mut self) {
        self.sender = None;
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

fn realtime_worker<T, P>(
    receiver: std::sync::mpsc::Receiver<T>,
    mut pipeline: P,
    chunk_size: usize,
    flush_timeout: Option<Duration>,
    metrics: Arc<SharedMetrics>,
) -> Result<P::Output>
where
    P: StreamingPipeline<T>,
{
    let mut chunk: Vec<T> = Vec::with_capacity(chunk_size);

    match flush_timeout {
        None => {
            for item in receiver {
                metrics.items_processed.fetch_add(1, Ordering::Relaxed);
                chunk.push(item);
                if chunk.len() >= chunk_size {
                    pipeline.process_chunk(&chunk)?;
                    chunk.clear();
                }
            }
        }
        Some(timeout) => {
            use std::sync::mpsc::RecvTimeoutError;
            loop {
                match receiver.recv_timeout(timeout) {
                    Ok(item) => {
                        metrics.items_processed.fetch_add(1, Ordering::Relaxed);
                        chunk.push(item);
                        if chunk.len() >= chunk_size {
                            pipeline.process_chunk(&chunk)?;
                            chunk.clear();
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if !chunk.is_empty() {
                            pipeline.process_chunk(&chunk)?;
                            chunk.clear();
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        }
    }

    if !chunk.is_empty() {
        pipeline.process_chunk(&chunk)?;
    }

    pipeline.finalize()
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

    // ---- RealtimePipeline ---------------------------------------------------

    #[test]
    fn test_realtime_basic_round_trip() {
        let config = BackpressureConfig { max_queue_depth: 32, chunk_size: 8, flush_timeout: None };
        let rt = RealtimePipeline::new(StreamingCollector::new(), config);
        for i in 0..50_u32 {
            rt.send(Point3f::new(i as f32, 0.0, 0.0)).unwrap();
        }
        let cloud = rt.finish().unwrap();
        assert_eq!(cloud.len(), 50);
    }

    #[test]
    fn test_realtime_with_flush_timeout() {
        let config = BackpressureConfig {
            max_queue_depth: 64,
            chunk_size: 100,
            flush_timeout: Some(Duration::from_millis(5)),
        };
        let rt = RealtimePipeline::new(StreamingCollector::new(), config);
        for i in 0..20_u32 {
            rt.send(Point3f::new(i as f32, 0.0, 0.0)).unwrap();
        }
        // finish() waits for worker; flush_timeout ensures partial chunk is processed.
        let cloud = rt.finish().unwrap();
        assert_eq!(cloud.len(), 20);
    }

    #[test]
    fn test_realtime_voxel_filter() {
        let filter = StreamingVoxelFilter::new(StreamingVoxelFilterConfig { voxel_size: 1.0 });
        let config = BackpressureConfig { max_queue_depth: 64, chunk_size: 16, flush_timeout: None };
        let rt = RealtimePipeline::new(filter, config);
        for i in 0..100_u32 {
            rt.send(Point3f::new(i as f32 * 0.1, 0.0, 0.0)).unwrap();
        }
        let cloud = rt.finish().unwrap();
        assert!(cloud.len() <= 10, "expected ≤10 voxels, got {}", cloud.len());
        assert!(!cloud.is_empty());
    }

    #[test]
    fn test_realtime_metrics_queued_count() {
        let config = BackpressureConfig { max_queue_depth: 64, chunk_size: 32, flush_timeout: None };
        let rt = RealtimePipeline::new(StreamingCollector::new(), config);
        for i in 0..20_u32 {
            rt.send(Point3f::new(i as f32, 0.0, 0.0)).unwrap();
        }
        let m = rt.metrics();
        assert_eq!(m.items_queued, 20);
        assert_eq!(m.items_dropped, 0);
        rt.finish().unwrap();
    }

    #[test]
    fn test_realtime_try_send_accepts_when_space() {
        let config = BackpressureConfig { max_queue_depth: 16, chunk_size: 8, flush_timeout: None };
        let rt = RealtimePipeline::new(StreamingCollector::new(), config);
        let accepted = rt.try_send(Point3f::new(1.0, 0.0, 0.0)).unwrap();
        assert!(accepted, "should accept item when queue has space");
        let m = rt.metrics();
        assert_eq!(m.items_queued, 1);
        assert_eq!(m.items_dropped, 0);
        let cloud = rt.finish().unwrap();
        assert_eq!(cloud.len(), 1);
    }

    #[test]
    fn test_realtime_try_send_drops_when_full() {
        use std::sync::{Condvar, Mutex};

        // A pipeline whose first process_chunk blocks until we release a latch,
        // so we can fill the bounded channel before the worker drains it.
        struct LatchedCollector {
            latch: Arc<(Mutex<bool>, Condvar)>,
            inner: StreamingCollector,
            blocked: bool,
        }
        impl StreamingPipeline<Point3f> for LatchedCollector {
            type Output = PointCloud<Point3f>;
            fn process_chunk(&mut self, chunk: &[Point3f]) -> Result<()> {
                if !self.blocked {
                    self.blocked = true;
                    let (lock, cv) = &*self.latch;
                    let mut released = lock.lock().unwrap();
                    while !*released {
                        released = cv.wait(released).unwrap();
                    }
                }
                self.inner.process_chunk(chunk)
            }
            fn finalize(self) -> Result<PointCloud<Point3f>> {
                self.inner.finalize()
            }
        }

        let latch = Arc::new((Mutex::new(false), Condvar::new()));
        let latch_release = Arc::clone(&latch);

        // chunk_size=1: every item triggers process_chunk → first one blocks.
        let config = BackpressureConfig { max_queue_depth: 1, chunk_size: 1, flush_timeout: None };
        let rt = RealtimePipeline::new(
            LatchedCollector { latch, inner: StreamingCollector::new(), blocked: false },
            config,
        );

        // First item: worker picks it up and blocks inside process_chunk.
        rt.send(Point3f::new(0.0, 0.0, 0.0)).unwrap();
        // Give worker time to dequeue and enter process_chunk.
        std::thread::sleep(Duration::from_millis(20));

        // Queue depth is 1; worker is blocked → channel is full.
        let mut accepted = 0usize;
        let mut dropped = 0usize;
        for i in 1..=8_u32 {
            if rt.try_send(Point3f::new(i as f32, 0.0, 0.0)).unwrap() {
                accepted += 1;
            } else {
                dropped += 1;
            }
        }
        assert!(dropped > 0, "expected at least one drop with max_queue_depth=1");

        // Release the latch so the worker can finish processing.
        let (lock, cv) = &*latch_release;
        *lock.lock().unwrap() = true;
        cv.notify_all();

        // Capture drop count before consuming rt.
        let total_dropped = rt.metrics().items_dropped;
        let cloud = rt.finish().unwrap();
        // Total accepted = 1 (from send) + accepted (from try_send).
        assert_eq!(cloud.len(), 1 + accepted);
        assert_eq!(total_dropped, dropped as u64);
    }

    #[test]
    fn test_realtime_drop_without_finish() {
        let config = BackpressureConfig { max_queue_depth: 16, chunk_size: 4, flush_timeout: None };
        let rt = RealtimePipeline::new(StreamingCollector::new(), config);
        for i in 0..10_u32 {
            rt.send(Point3f::new(i as f32, 0.0, 0.0)).unwrap();
        }
        drop(rt); // Must not panic or deadlock.
    }

    #[test]
    fn test_realtime_large_workload() {
        let config = BackpressureConfig {
            max_queue_depth: 512,
            chunk_size: 128,
            flush_timeout: None,
        };
        let rt = RealtimePipeline::new(StreamingCollector::new(), config);
        const N: u32 = 10_000;
        for i in 0..N {
            rt.send(Point3f::new(i as f32, 0.0, 0.0)).unwrap();
        }
        let cloud = rt.finish().unwrap();
        assert_eq!(cloud.len(), N as usize);
    }
}
