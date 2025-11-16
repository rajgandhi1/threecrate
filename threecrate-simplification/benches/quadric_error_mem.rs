use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use threecrate_bench::{ThreecrateMeasurment, mem::{AllocationSize, Allocations, INSTRUMENTED_SYSTEM, InstrumentedSystem}};
use threecrate_io::{AttributePreservingReader, ExtendedTriangleMesh, SerializationOptions};
use threecrate_simplification::{MeshSimplifier, QuadricErrorSimplifier};

const ASSET_PATH: &str = "../assets/bunny.obj";
const REDUCTION_RATIOS: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];

#[global_allocator]
static GLOBAL_ALLOCATOR: &InstrumentedSystem = &INSTRUMENTED_SYSTEM;

fn quadric_error_memory<M: ThreecrateMeasurment>(c: &mut Criterion<M>) {
    let ExtendedTriangleMesh { mesh, .. } = AttributePreservingReader::read_extended_mesh(ASSET_PATH, &SerializationOptions::default()).unwrap();
    let simplifier = QuadricErrorSimplifier::default();
    
    let mut g = c.benchmark_group(format!("quadric error {}", M::NAME));
    g.sample_size(10);

    for ratio in REDUCTION_RATIOS {
        g.bench_with_input(BenchmarkId::from_parameter(ratio), &(&simplifier, &mesh, ratio), |b, &(simplifier, mesh, ratio)| {
            b.iter(|| std::hint::black_box(simplifier).simplify(std::hint::black_box(mesh), std::hint::black_box(ratio)));
        });
    }

    g.finish();
}

criterion_group!{
    name = allocations;
    config = Criterion::default().with_measurement(Allocations);
    targets = quadric_error_memory
}

criterion_group!{
    name = allocation_size;
    config = Criterion::default().with_measurement(AllocationSize);
    targets = quadric_error_memory
}

criterion_main!(
    allocations, 
    allocation_size,
);