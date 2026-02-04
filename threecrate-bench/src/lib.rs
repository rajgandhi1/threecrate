use criterion::measurement::Measurement;

pub mod mem;

pub trait ThreecrateMeasurement: Measurement {
    const NAME: &'static str;
} 
