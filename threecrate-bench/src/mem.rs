use stats_alloc::StatsAlloc;

pub type InstrumentedSystem = StatsAlloc<std::alloc::System>;
pub use stats_alloc::INSTRUMENTED_SYSTEM;

mod allocations;
pub use allocations::Allocations;



