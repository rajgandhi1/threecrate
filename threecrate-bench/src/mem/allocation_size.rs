use criterion::measurement::{Measurement, ValueFormatter};
use stats_alloc::INSTRUMENTED_SYSTEM;

use crate::ThreecrateMeasurment;

pub struct AllocationSize;

impl ThreecrateMeasurment for AllocationSize {
    const NAME: &'static str = "allocation size";
}

impl Measurement for AllocationSize {
    type Intermediate = usize;

    type Value = usize;

    fn start(&self) -> Self::Intermediate {
        let stats = INSTRUMENTED_SYSTEM.stats();
        stats.bytes_allocated - stats.bytes_deallocated
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        INSTRUMENTED_SYSTEM.stats().bytes_allocated - i
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
        &AllocationSizeFormatter
    }
}

struct AllocationSizeFormatter;


impl ValueFormatter for AllocationSizeFormatter {
    fn scale_values(&self, typycal_value: f64, values: &mut [f64]) -> &'static str {
        let magnitude = typycal_value.log(2.0).floor() as i32 / 10;
        let factor = 1.0 / 1024.0_f64.powi(magnitude);

        for value in values {
            *value *= factor;
        }
        match magnitude {
            ..=1 => "B",
            2 => "KiB",
            3 => "MiB",
            4 => "GiB",
            5 => "TiB",
            _ => "a lot of bytes"
        }
    }

    fn scale_throughputs(
        &self,
        _: f64,
        _: &criterion::Throughput,
        _: &mut [f64],
    ) -> &'static str {
        "B"
    }

    fn scale_for_machines(&self, _: &mut [f64]) -> &'static str {
        "B"
    }
}
