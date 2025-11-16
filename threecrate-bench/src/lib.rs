use criterion::measurement::Measurement;

pub mod mem;

pub trait ThreecrateMeasurment: Measurement {
    const NAME: &'static str;
} 
