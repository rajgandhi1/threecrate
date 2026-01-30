//! Benchmarks comparing EdgeCollapseSimplifier vs QuadricErrorSimplifier

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use threecrate_core::{Point3f, TriangleMesh};
use threecrate_simplification::{EdgeCollapseSimplifier, MeshSimplifier, QuadricErrorSimplifier};

fn generate_grid_mesh(size: usize) -> TriangleMesh {
    let mut vertices = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 / (size - 1) as f32 * std::f32::consts::PI;
            let fy = y as f32 / (size - 1) as f32 * std::f32::consts::PI;
            vertices.push(Point3f::new(
                x as f32,
                y as f32,
                (fx.sin() * fy.sin()) * 2.0,
            ));
        }
    }
    let mut faces = Vec::with_capacity((size - 1) * (size - 1) * 2);
    for y in 0..(size - 1) {
        for x in 0..(size - 1) {
            let tl = y * size + x;
            let tr = tl + 1;
            let bl = (y + 1) * size + x;
            let br = bl + 1;
            faces.push([tl, bl, tr]);
            faces.push([tr, bl, br]);
        }
    }
    TriangleMesh::from_vertices_and_faces(vertices, faces)
}

fn bench_simplification(c: &mut Criterion) {
    let sizes = [10, 20, 40];
    let ratios = [0.3, 0.5, 0.7];

    let mut group = c.benchmark_group("simplification");

    for &size in &sizes {
        let mesh = generate_grid_mesh(size);
        let face_count = mesh.face_count();

        for &ratio in &ratios {
            group.bench_with_input(
                BenchmarkId::new(
                    "edge_collapse",
                    format!("{}f_r{}", face_count, (ratio * 100.0) as u32),
                ),
                &(&mesh, ratio),
                |b, &(mesh, ratio)| {
                    let simplifier = EdgeCollapseSimplifier::new();
                    b.iter(|| {
                        let result = simplifier.simplify(black_box(mesh), ratio).unwrap();
                        black_box(result);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    "qem",
                    format!("{}f_r{}", face_count, (ratio * 100.0) as u32),
                ),
                &(&mesh, ratio),
                |b, &(mesh, ratio)| {
                    let simplifier = QuadricErrorSimplifier::new();
                    b.iter(|| {
                        let result = simplifier.simplify(black_box(mesh), ratio).unwrap();
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_simplification);
criterion_main!(benches);
