# Changelog

All notable changes to the threecrate-reconstruction crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-01-27

### Added

#### New Algorithms
- **Poisson Surface Reconstruction**: Complete implementation with API compatibility fixes for poisson_reconstruction crate v0.4.0
- **Delaunay Triangulation**: Comprehensive 2D/3D triangulation with 5 projection methods (XY, XZ, YZ, PCA, BestFitPlane)
- **Marching Cubes**: Volumetric surface reconstruction with isosurface extraction and trilinear interpolation
- **Moving Least Squares (MLS)**: Smooth surface fitting with 4 weight functions and polynomial basis options
- **Enhanced Ball Pivoting**: Multi-scale capabilities with adaptive radius selection and quality metrics

#### Unified Reconstruction Pipeline
- **Intelligent Algorithm Selection**: Automatic algorithm selection based on input data characteristics
- **Data Analysis Engine**: Analyzes point count, density uniformity, noise level, and surface complexity
- **Quality Levels**: 4 quality presets (Fast, Balanced, HighQuality, MaxQuality)
- **Use Case Optimization**: 7 specialized use cases (General, Prototyping, Engineering, Organic, NoisyData, Sparse, Dense)
- **Fallback Strategies**: Automatic fallback to alternative algorithms when preferred methods fail

#### Parallel Processing
- **Rayon Integration**: Comprehensive parallel processing across all reconstruction algorithms
- **Thread Pool Management**: Configurable thread pools with adaptive chunk sizing
- **Performance Optimizations**: Spatial acceleration structures and efficient parallel iterators
- **Scalability**: Automatic parallel/sequential fallback based on data size

#### Advanced Features
- **Quality Metrics**: Triangle quality assessment, watertightness analysis, and geometric accuracy
- **Spatial Indexing**: Grid-based spatial acceleration for neighbor queries
- **Multi-Scale Processing**: Support for different detail levels and adaptive parameters
- **Robust Error Handling**: Comprehensive error handling with graceful degradation

### Enhanced

#### Algorithm Improvements
- **Ball Pivoting**: Added multi-scale capabilities, adaptive radius selection, and improved quality metrics
- **All Algorithms**: Integrated parallel processing for significant performance improvements
- **Robustness**: Enhanced input validation and error handling across all methods

#### API Improvements
- **Unified Interface**: Consistent API across all reconstruction algorithms
- **Convenience Functions**: Simple auto-reconstruction functions for common use cases
- **Configuration**: Flexible configuration system with sensible defaults
- **Documentation**: Comprehensive documentation with usage examples

### Fixed
- **Poisson Reconstruction**: Resolved API compatibility issues with external poisson_reconstruction crate
- **Step Size Validation**: Fixed step_by(0) panics in parameter estimation across all algorithms
- **Matrix Operations**: Fixed nalgebra matrix ownership issues with proper cloning
- **Memory Management**: Improved memory efficiency in parallel processing

### Technical Details

#### New Dependencies
- `num_cpus`: For automatic thread count detection
- Enhanced `rayon` integration for parallel processing

#### Performance Improvements
- **Parallel Processing**: 2-4x speedup on multi-core systems for large point clouds
- **Spatial Acceleration**: O(log n) neighbor queries instead of O(n²)
- **Memory Efficiency**: Reduced memory allocations in critical paths

#### Comprehensive Testing
- **65+ Tests**: Comprehensive test suite covering all algorithms and edge cases
- **Integration Tests**: 12 integration tests for full pipeline workflows
- **Quality Assurance**: Automated testing for algorithm selection logic and fallback mechanisms

### Migration Guide

#### From 0.4.x to 0.5.0
- **New API**: Use the unified pipeline for automatic algorithm selection:
  ```rust
  // Old way (still supported)
  let mesh = some_specific_algorithm(&point_cloud)?;

  // New way (recommended)
  let mesh = auto_reconstruct(&point_cloud)?;
  let result = ReconstructionPipeline::default().reconstruct(&point_cloud)?;
  ```

- **Enhanced Configuration**: Take advantage of new quality and use case presets:
  ```rust
  let mesh = auto_reconstruct_with_quality(&point_cloud, QualityLevel::HighQuality)?;
  let mesh = auto_reconstruct_for_use_case(&point_cloud, UseCase::Engineering)?;
  ```

- **Parallel Processing**: Optionally configure thread pools for optimal performance:
  ```rust
  use threecrate_reconstruction::parallel::{init_thread_pool, ThreadPoolConfig};

  let config = ThreadPoolConfig::default().with_threads(8);
  init_thread_pool(config)?;
  ```

### Breaking Changes
- Minimum Rust version requirement may have increased due to new dependencies
- Some internal APIs have changed (public API remains backward compatible)

---

## [0.4.0] and earlier

Previous versions focused on basic reconstruction algorithms. See git history for detailed changes.