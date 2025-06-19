# Contributing to Rust3D

Thank you for your interest in contributing! This project is in early development with many `todo!()` algorithms waiting for implementation.

## Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/rust3D.git
   cd rust3D
   git remote add upstream https://github.com/ORIGINAL_OWNER/rust3D.git
   ```

2. **Build and Test**
   ```bash
   cargo build --workspace
   cargo test --workspace
   cargo run --bin basic_usage
   ```

## What We Need

- **Algorithm Implementation**: Replace `todo!()` in `threecrate-algorithms`
- **File I/O**: PLY/OBJ readers/writers in `threecrate-io`  
- **GPU Computing**: WGPU acceleration in `threecrate-gpu`
- **Testing**: Unit tests for all modules
- **Documentation**: API docs and examples

## Project Structure

```
rust3D/
├── threecrate-core/           # Core data structures
├── threecrate-algorithms/     # Point cloud algorithms (filtering, normals, ICP)
├── threecrate-gpu/           # GPU acceleration
├── threecrate-io/            # File I/O (PLY, OBJ)
├── threecrate-reconstruction/ # Surface reconstruction
├── threecrate-simplification/ # Mesh simplification
└── threecrate-visualization/  # 3D visualization
```

## Implementation Best Practices

### Algorithm Implementation Pattern

```rust
/// Brief description of the algorithm
/// 
/// # Arguments
/// * `input` - Description
/// 
/// # Returns
/// Description of return value
/// 
/// # Complexity
/// Time/space complexity
pub fn algorithm_name<T>(input: &T) -> Result<Output, ProcessingError> {
    // 1. Validate input
    // 2. Implement core algorithm
    // 3. Return result
    todo!("Implement [algorithm name]")
}
```

### Implementation Steps

1. **Research**: Understand the algorithm's mathematical foundation
2. **Test First**: Write tests before implementation
3. **Document**: Include complexity notes and references
4. **Validate**: Check edge cases and error conditions

## Priority Tasks

### High Priority
- [ ] Statistical outlier removal (`filtering.rs`)
- [ ] Surface normal estimation (`normals.rs`) 
- [ ] PLY file reader/writer (`ply.rs`)
- [ ] Basic point cloud rendering

### Medium Priority  
- [ ] ICP registration algorithm (`registration.rs`)
- [ ] RANSAC plane segmentation (`segmentation.rs`)
- [ ] GPU buffer management

## Testing Requirements

**All new algorithms must include tests.** Example:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::{PointCloud, Point3f};

    #[test]
    fn test_algorithm_name() {
        let points = vec![
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 0.0, 0.0),
        ];
        let cloud = PointCloud::from_points(points);
        
        let result = your_algorithm(&cloud);
        assert!(result.is_ok());
    }
}
```

**Run tests**: `cargo test --workspace`

## Pull Request Process

### Before Submitting
```bash
git fetch upstream && git rebase upstream/main
cargo fmt && cargo clippy && cargo test --workspace
```

### PR Requirements
- **Descriptive title**: "Implement statistical outlier removal filter"
- **Include tests** for new functionality
- **Update documentation** if adding major features
- **Keep PRs focused** - one feature per PR

## Code Style

**Formatting**: `cargo fmt` (required)
**Linting**: `cargo clippy` (required)
**Error Handling**: Use `Result<T, ProcessingError>` for fallible operations

```rust
use threecrate_core::ProcessingError;

pub fn process_point_cloud(
    cloud: &PointCloud<Point3f>
) -> Result<PointCloud<Point3f>, ProcessingError> {
    // Implementation
}
```

---

Questions? Open an issue or check existing documentation in `examples/`. 