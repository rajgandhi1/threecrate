//! Quadric error decimation

use threecrate_core::{TriangleMesh, Result};
use crate::MeshSimplifier;

/// Quadric error decimation simplifier
pub struct QuadricErrorSimplifier;

impl MeshSimplifier for QuadricErrorSimplifier {
    fn simplify(&self, _mesh: &TriangleMesh, _reduction_ratio: f32) -> Result<TriangleMesh> {
        // TODO: Implement quadric error decimation
        todo!("Quadric error decimation not yet implemented")
    }
} 