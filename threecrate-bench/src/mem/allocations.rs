use criterion::measurement::{Measurement, ValueFormatter};
use stats_alloc::INSTRUMENTED_SYSTEM;

use crate::ThreecrateMeasurment;

pub struct Allocations;

impl ThreecrateMeasurment for Allocations {
    const NAME: &'static str = "allocations";
}

impl Measurement for Allocations {
    type Intermediate = usize;

    type Value = usize;

    fn start(&self) -> Self::Intermediate {
        INSTRUMENTED_SYSTEM.stats().allocations
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        INSTRUMENTED_SYSTEM.stats().allocations - i
    }

    fn add(&self, &v1: &Self::Value, &v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, &value: &Self::Value) -> f64 {
        value as f64
    }

    fn formatter(&self) -> &dyn criterion::measurement::ValueFormatter {
        &AllocationsFormatter
    }
}

struct AllocationsFormatter;

impl ValueFormatter for AllocationsFormatter {
    fn scale_values(&self, _: f64, _: &mut [f64]) -> &'static str {
        "allocation times"
    }

    fn scale_throughputs(
        &self,
        _: f64,
        _: &criterion::Throughput,
        _: &mut [f64],
    ) -> &'static str {
        "allocation times"
    }

    fn scale_for_machines(&self, _: &mut [f64]) -> &'static str {
        "allocation times"
    }
}
