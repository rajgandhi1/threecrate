use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use threecrate_algorithms::features::{extract_shot_features_with_normals, ShotConfig, ShotVariant};
use threecrate_core::{NormalPoint3f, Point3f, PointCloud, Vector3f};

fn make_plane_cloud(n: usize) -> PointCloud<NormalPoint3f> {
    let side = (n as f64).sqrt().ceil() as usize;
    let step = 1.0 / side as f32;
    let mut cloud = PointCloud::new();
    'outer: for i in 0..side {
        for j in 0..side {
            if cloud.len() == n {
                break 'outer;
            }
            cloud.push(NormalPoint3f {
                position: Point3f::new(i as f32 * step, j as f32 * step, 0.0),
                normal: Vector3f::new(0.0, 0.0, 1.0),
            });
        }
    }
    cloud
}

fn bench_shot_by_cloud_size(c: &mut Criterion) {
    let config = ShotConfig {
        search_radius: 0.2,
        k_neighbors: 10,
        variant: ShotVariant::Standard,
    };

    let mut group = c.benchmark_group("shot_cloud_size");
    for &n in &[100usize, 500, 1000] {
        let cloud = make_plane_cloud(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &cloud, |b, cloud| {
            b.iter(|| extract_shot_features_with_normals(black_box(cloud), black_box(&config)));
        });
    }
    group.finish();
}

fn bench_shot_vs_usc(c: &mut Criterion) {
    let cloud = make_plane_cloud(500);

    let mut group = c.benchmark_group("shot_vs_usc");
    let shot_config = ShotConfig {
        search_radius: 0.2,
        k_neighbors: 10,
        variant: ShotVariant::Standard,
    };
    let usc_config = ShotConfig {
        search_radius: 0.2,
        k_neighbors: 10,
        variant: ShotVariant::UniqueShapeContext,
    };

    group.bench_function("shot", |b| {
        b.iter(|| extract_shot_features_with_normals(black_box(&cloud), black_box(&shot_config)));
    });
    group.bench_function("usc", |b| {
        b.iter(|| extract_shot_features_with_normals(black_box(&cloud), black_box(&usc_config)));
    });
    group.finish();
}

fn bench_shot_by_radius(c: &mut Criterion) {
    let cloud = make_plane_cloud(500);

    let mut group = c.benchmark_group("shot_search_radius");
    for &r in &[0.05f32, 0.1, 0.2, 0.5] {
        let config = ShotConfig {
            search_radius: r,
            k_neighbors: 10,
            variant: ShotVariant::Standard,
        };
        group.bench_with_input(BenchmarkId::from_parameter(r), &config, |b, config| {
            b.iter(|| extract_shot_features_with_normals(black_box(&cloud), black_box(config)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_shot_by_cloud_size, bench_shot_vs_usc, bench_shot_by_radius);
criterion_main!(benches);
