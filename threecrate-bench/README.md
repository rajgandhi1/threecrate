# ThreeCrate Bench

[![Crates.io](https://img.shields.io/crates/v/threecrate-bench.svg)](https://crates.io/crates/threecrate-bench)
[![Documentation](https://docs.rs/threecrate-bench/badge.svg)](https://docs.rs/threecrate-bench)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/rajgandhi1/threecrate#license)

Benchmarks utilities for ThreeCrate.

## Usage

1. Add this crate to dev-dependencies
2. Create benchmark
   1. Create directory `benches`
   2. Create file `benches/<bench_name>.rs`
   3. Add bench with name `<bench_name>` to `Cargo.toml` with `harness=false`

    ```toml
    [[bench]]
    name = "<bench_name>"
    harness = false
    ```

   4. Add boilerplate code to `benches/<bench_name>.rs`

   ```rust
    use criterion::{Criterion, criterion_group, criterion_main};
    use threecrate_bench::{ThreecrateMeasurement, mem::{AllocationSize, Allocations, INSTRUMENTED_SYSTEM, InstrumentedSystem}};

    #[global_allocator]
    static GLOBAL_ALLOCATOR: &InstrumentedSystem = &INSTRUMENTED_SYSTEM;

    fn benchmark<M: ThreecrateMeasurement>(c: &mut Criterion<M>) {
        let benchmark_name = format!("benchmark {}", M::NAME);
        // Your benchmark code here
    }

    criterion_group!{
        name = allocations;
        config = Criterion::default().with_measurement(Allocations);
        targets = 
            benchmark,
            // any bench functions here
    }

    criterion_group!{
        name = allocation_size;
        config = Criterion::default().with_measurement(AllocationSize);
        targets = 
            benchmark,
            // any bench functions here
    }

    criterion_main!(
        allocations, 
        allocation_size,
    );
   ```
   5. Write [criterion benchmark code](https://bheisler.github.io/criterion.rs/book/getting_started.html)
3. Run benchmark **before** optimizing code
4. Optimize code
5. Run benchmark **after** code optimizations
6. Compare results

## Results of benchmark

It's difficult to obtain absolute values ​​for performance, allocation counts, and memory footprint, so it's best to focus only on the difference between measurements before and after optimizations.

- Performance - the lower the better
- Allocation count - the lower the better
- Allocation size - the lower the better


## Sidenotes

This crate uses a “hack” criterion to measure memory, so on some labels you can see something like “time: 10 MiB”, you should pay attention to the units of measurement. In addition, there are graphs that the criterion creates for time measurements and they have nothing to do with measuring the amount or size of memory.

It is worth paying more attention to the readings of the minimum measurement values, since noise when measuring a specific function can only be positive
