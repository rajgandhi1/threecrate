//! Mesh simplification and decimation algorithms
//! 
//! This crate provides algorithms for reducing mesh complexity while
//! preserving important geometric features:
//! - Quadric error decimation
//! - Edge collapse algorithms
//! - Clustering-based simplification

pub mod quadric_error;
pub mod edge_collapse;
pub mod clustering;

pub use quadric_error::*;
pub use edge_collapse::*;
pub use clustering::*;

use threecrate_core::{TriangleMesh, Result};

/// Simplify a mesh by reducing the number of faces/vertices
pub trait MeshSimplifier {
    /// Simplify mesh with target reduction ratio (0.0 = no reduction, 1.0 = maximum reduction)
    fn simplify(&self, mesh: &TriangleMesh, reduction_ratio: f32) -> Result<TriangleMesh>;
} 