//! Edge collapse simplification

use threecrate_core::{TriangleMesh, Result};
use crate::MeshSimplifier;

/// Edge collapse simplifier
pub struct EdgeCollapseSimplifier;

impl MeshSimplifier for EdgeCollapseSimplifier {
    fn simplify(&self, _mesh: &TriangleMesh, _reduction_ratio: f32) -> Result<TriangleMesh> {
        // TODO: Implement edge collapse simplification
        todo!("Edge collapse simplification not yet implemented")
    }
} 