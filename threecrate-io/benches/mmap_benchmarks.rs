//! Benchmarks comparing memory-mapped vs standard buffered I/O for PLY/PCD files
//!
//! These benchmarks demonstrate the performance improvements of memory-mapped I/O
//! for large binary point cloud files.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::fs::File;
use tempfile::NamedTempFile;
use threecrate_core::{PointCloud, Point3f};
use threecrate_io::{RobustPlyReader, RobustPlyWriter, PlyWriteOptions};
use threecrate_io::{RobustPcdReader, RobustPcdWriter, PcdWriteOptions, PcdDataFormat};

/// Generate a test point cloud with the specified number of points
fn generate_test_point_cloud(num_points: usize) -> PointCloud<Point3f> {
    let points: Vec<Point3f> = (0..num_points)
        .map(|i| {
            let i = i as f32;
            Point3f::new(
                (i * 0.1).sin(),
                (i * 0.1).cos(),
                i * 0.001,
            )
        })
        .collect();
    
    PointCloud::from_points(points)
}

/// Create a temporary binary PLY file for benchmarking
fn create_binary_ply_file(num_points: usize) -> NamedTempFile {
    let cloud = generate_test_point_cloud(num_points);
    let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
    
    let options = PlyWriteOptions::binary_little_endian();
    RobustPlyWriter::write_point_cloud(&cloud, temp_file.path(), &options)
        .expect("Failed to write PLY file");
    
    temp_file
}

/// Create a temporary binary PCD file for benchmarking
fn create_binary_pcd_file(num_points: usize) -> NamedTempFile {
    let cloud = generate_test_point_cloud(num_points);
    let temp_file = NamedTempFile::new().expect("Failed to create temporary file");
    
    let options = PcdWriteOptions {
        data_format: PcdDataFormat::Binary,
        ..Default::default()
    };
    RobustPcdWriter::write_point_cloud(&cloud, temp_file.path(), &options)
        .expect("Failed to write PCD file");
    
    temp_file
}

/// Benchmark PLY reading with different methods
fn benchmark_ply_reading(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000, 500000];
    
    let mut group = c.benchmark_group("ply_reading");
    
    for size in sizes {
        let temp_file = create_binary_ply_file(size);
        let file_size = std::fs::metadata(temp_file.path())
            .expect("Failed to get file metadata")
            .len();
        
        group.throughput(Throughput::Bytes(file_size));
        
        // Benchmark with mmap feature enabled
        #[cfg(feature = "io-mmap")]
        {
            group.bench_with_input(
                BenchmarkId::new("mmap_enabled", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let ply_data = RobustPlyReader::read_ply_file(temp_file.path())
                            .expect("Failed to read PLY file");
                        black_box(ply_data);
                    });
                },
            );
        }
        
        // Benchmark with mmap feature disabled (simulate fallback)
        group.bench_with_input(
            BenchmarkId::new("buffered_io", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let file = File::open(temp_file.path()).expect("Failed to open file");
                    let mut reader = std::io::BufReader::new(file);
                    let ply_data = RobustPlyReader::read_ply_data(&mut reader)
                        .expect("Failed to read PLY file");
                    black_box(ply_data);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark PCD reading with different methods
fn benchmark_pcd_reading(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000, 500000];
    
    let mut group = c.benchmark_group("pcd_reading");
    
    for size in sizes {
        let temp_file = create_binary_pcd_file(size);
        let file_size = std::fs::metadata(temp_file.path())
            .expect("Failed to get file metadata")
            .len();
        
        group.throughput(Throughput::Bytes(file_size));
        
        // Benchmark with mmap feature enabled
        #[cfg(feature = "io-mmap")]
        {
            group.bench_with_input(
                BenchmarkId::new("mmap_enabled", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let (header, points) = RobustPcdReader::read_pcd_file(temp_file.path())
                            .expect("Failed to read PCD file");
                        black_box((header, points));
                    });
                },
            );
        }
        
        // Benchmark with mmap feature disabled (simulate fallback)
        group.bench_with_input(
            BenchmarkId::new("buffered_io", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let file = File::open(temp_file.path()).expect("Failed to open file");
                    let mut reader = std::io::BufReader::new(file);
                    let (header, points) = RobustPcdReader::read_pcd_data(&mut reader)
                        .expect("Failed to read PCD file");
                    black_box((header, points));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory mapping overhead for small files
fn benchmark_small_file_overhead(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000]; // Small files where mmap might have overhead
    
    let mut group = c.benchmark_group("small_file_overhead");
    
    for size in sizes {
        let temp_file = create_binary_ply_file(size);
        let file_size = std::fs::metadata(temp_file.path())
            .expect("Failed to get file metadata")
            .len();
        
        group.throughput(Throughput::Bytes(file_size));
        
        // Test if mmap is actually used for small files
        #[cfg(feature = "io-mmap")]
        {
            group.bench_with_input(
                BenchmarkId::new("mmap_small", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let should_use = threecrate_io::mmap::should_use_mmap(temp_file.path());
                        if should_use {
                            let ply_data = RobustPlyReader::read_ply_file(temp_file.path())
                                .expect("Failed to read PLY file");
                            black_box(ply_data);
                        } else {
                            // Should fall back to buffered I/O for small files
                            let file = File::open(temp_file.path()).expect("Failed to open file");
                            let mut reader = std::io::BufReader::new(file);
                            let ply_data = RobustPlyReader::read_ply_data(&mut reader)
                                .expect("Failed to read PLY file");
                            black_box(ply_data);
                        }
                    });
                },
            );
        }
        
        group.bench_with_input(
            BenchmarkId::new("buffered_small", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let file = File::open(temp_file.path()).expect("Failed to open file");
                    let mut reader = std::io::BufReader::new(file);
                    let ply_data = RobustPlyReader::read_ply_data(&mut reader)
                        .expect("Failed to read PLY file");
                    black_box(ply_data);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different endianness handling
fn benchmark_endianness(c: &mut Criterion) {
    let size = 50000;
    
    let mut group = c.benchmark_group("endianness");
    
    // Create little endian PLY file
    let cloud = generate_test_point_cloud(size);
    let le_temp_file = NamedTempFile::new().expect("Failed to create temporary file");
    let le_options = PlyWriteOptions::binary_little_endian();
    RobustPlyWriter::write_point_cloud(&cloud, le_temp_file.path(), &le_options)
        .expect("Failed to write LE PLY file");
    
    // Create big endian PLY file
    let be_temp_file = NamedTempFile::new().expect("Failed to create temporary file");
    let be_options = PlyWriteOptions::binary_big_endian();
    RobustPlyWriter::write_point_cloud(&cloud, be_temp_file.path(), &be_options)
        .expect("Failed to write BE PLY file");
    
    let file_size = std::fs::metadata(le_temp_file.path())
        .expect("Failed to get file metadata")
        .len();
    
    group.throughput(Throughput::Bytes(file_size));
    
    #[cfg(feature = "io-mmap")]
    {
        group.bench_function("little_endian_mmap", |b| {
            b.iter(|| {
                let ply_data = RobustPlyReader::read_ply_file(le_temp_file.path())
                    .expect("Failed to read LE PLY file");
                black_box(ply_data);
            });
        });
        
        group.bench_function("big_endian_mmap", |b| {
            b.iter(|| {
                let ply_data = RobustPlyReader::read_ply_file(be_temp_file.path())
                    .expect("Failed to read BE PLY file");
                black_box(ply_data);
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_ply_reading,
    benchmark_pcd_reading,
    benchmark_small_file_overhead,
    benchmark_endianness
);

criterion_main!(benches);
