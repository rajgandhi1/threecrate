//! Real-time streaming pipeline with backpressure (issue #147).
//!
//! Demonstrates the bounded-queue architecture that throttles producers when
//! the consumer falls behind and tracks dropped items when using the
//! non-blocking `try_send` path.

use std::thread;
use std::time::{Duration, Instant};

use threecrate_algorithms::streaming::{
    BackpressureConfig, RealtimePipeline, StreamingCollector, StreamingStatistics,
    StreamingVoxelFilter, StreamingVoxelFilterConfig,
};
use threecrate_core::{Point3f, PointCloud};

fn main() {
    println!("=== Real-time Streaming Pipeline with Backpressure (#147) ===\n");

    demo_blocking_backpressure();
    demo_try_send_with_drops();
    demo_voxel_filter_realtime();
    demo_flush_timeout_latency();

    println!("=== All demos completed successfully ===");
}

// ---------------------------------------------------------------------------
// 1. Blocking backpressure — producer automatically throttled when queue fills
// ---------------------------------------------------------------------------

fn demo_blocking_backpressure() {
    println!("--- Demo 1: Blocking backpressure (send) ---");

    const N: usize = 100_000;
    let config = BackpressureConfig {
        max_queue_depth: 512, // bounded queue — producer blocks when full
        chunk_size: 256,
        flush_timeout: None,
    };
    let rt = RealtimePipeline::new(StreamingCollector::new(), config);

    let t = Instant::now();
    for i in 0..N as u32 {
        let p = Point3f::new(i as f32 * 0.001, 0.0, 0.0);
        rt.send(p).expect("send failed");
    }
    let cloud = rt.finish().expect("finish failed");
    let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("  Points sent     : {N}");
    println!("  Points received : {}", cloud.len());
    println!("  Elapsed         : {:.2} ms", elapsed_ms);
    println!(
        "  Throughput      : {:.1} Mpts/s\n",
        N as f64 / elapsed_ms / 1000.0
    );
    assert_eq!(cloud.len(), N);
}

// ---------------------------------------------------------------------------
// 2. Non-blocking try_send — items dropped when queue is full
// ---------------------------------------------------------------------------

fn demo_try_send_with_drops() {
    println!("--- Demo 2: Non-blocking try_send (drop-on-full) ---");

    // A deliberately slow "pipeline" that simulates a busy consumer by sleeping
    // briefly inside process_chunk.  This makes the bounded queue fill up,
    // causing try_send to drop items.
    struct SlowCollector {
        inner: StreamingCollector,
        delay: Duration,
    }
    impl threecrate_algorithms::streaming::StreamingPipeline<Point3f> for SlowCollector {
        type Output = PointCloud<Point3f>;
        fn process_chunk(&mut self, chunk: &[Point3f]) -> threecrate_core::Result<()> {
            thread::sleep(self.delay);
            self.inner.process_chunk(chunk)
        }
        fn finalize(self) -> threecrate_core::Result<PointCloud<Point3f>> {
            self.inner.finalize()
        }
    }

    let slow = SlowCollector {
        inner: StreamingCollector::new(),
        delay: Duration::from_millis(5),
    };
    let config = BackpressureConfig {
        max_queue_depth: 4,
        chunk_size: 1,
        flush_timeout: None,
    };
    let rt = RealtimePipeline::new(slow, config);

    // Give the worker time to start and block on first chunk.
    rt.send(Point3f::new(0.0, 0.0, 0.0)).unwrap();
    thread::sleep(Duration::from_millis(20));

    let mut accepted = 0usize;
    let mut dropped = 0usize;
    for i in 1..=20_u32 {
        if rt.try_send(Point3f::new(i as f32, 0.0, 0.0)).unwrap() {
            accepted += 1;
        } else {
            dropped += 1;
        }
    }

    let pre_finish_metrics = rt.metrics();
    let cloud = rt.finish().unwrap();

    println!("  try_send attempts : 20");
    println!("  Accepted          : {}", accepted);
    println!("  Dropped (full)    : {}", dropped);
    println!(
        "  items_dropped metric: {}",
        pre_finish_metrics.items_dropped
    );
    println!("  Output cloud size : {}\n", cloud.len());

    assert_eq!(pre_finish_metrics.items_dropped, dropped as u64);
    assert_eq!(cloud.len(), 1 + accepted); // 1 from initial send + accepted from try_send
}

// ---------------------------------------------------------------------------
// 3. Voxel-grid downsampling in real-time via backpressure pipeline
// ---------------------------------------------------------------------------

fn demo_voxel_filter_realtime() {
    println!("--- Demo 3: Real-time voxel filter ---");

    let filter = StreamingVoxelFilter::new(StreamingVoxelFilterConfig { voxel_size: 1.0 });
    let config = BackpressureConfig {
        max_queue_depth: 256,
        chunk_size: 64,
        flush_timeout: None,
    };
    let rt = RealtimePipeline::new(filter, config);

    // 1000 points spread over 10 units → ≤10 voxels at voxel_size=1.0.
    const N: usize = 1_000;
    for i in 0..N as u32 {
        rt.send(Point3f::new(i as f32 * 0.01, 0.0, 0.0)).unwrap();
    }
    let cloud = rt.finish().unwrap();

    println!("  Input points  : {N}");
    println!("  Output voxels : {}", cloud.len());
    println!(
        "  Compression   : {:.1}%\n",
        100.0 * (1.0 - cloud.len() as f64 / N as f64)
    );
    assert!(cloud.len() <= 10);
    assert!(!cloud.is_empty());
}

// ---------------------------------------------------------------------------
// 4. flush_timeout — low-latency partial-chunk processing
// ---------------------------------------------------------------------------

fn demo_flush_timeout_latency() {
    println!("--- Demo 4: flush_timeout for low-latency processing ---");

    // chunk_size is large (1000), but flush_timeout=5ms ensures the worker
    // processes partial batches promptly without waiting for a full chunk.
    let config = BackpressureConfig {
        max_queue_depth: 128,
        chunk_size: 1000,
        flush_timeout: Some(Duration::from_millis(5)),
    };
    let rt = RealtimePipeline::new(StreamingStatistics::new(), config);

    let t = Instant::now();
    // Send only 50 items — far fewer than chunk_size.
    for i in 0..50_u32 {
        rt.send(Point3f::new(i as f32, 0.0, 0.0)).unwrap();
    }
    // finish() closes the channel; worker drains via flush_timeout.
    let stats = rt.finish().unwrap();
    let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("  Items sent       : 50  (chunk_size=1000)");
    println!("  Items processed  : {}", stats.point_count);
    println!("  Elapsed          : {:.2} ms", elapsed_ms);
    println!(
        "  Bounding box X   : [{:.0}, {:.0}]\n",
        stats.min.x, stats.max.x
    );

    assert_eq!(stats.point_count, 50);
}
