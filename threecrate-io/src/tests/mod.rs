//! Test modules for threecrate-io
//!
//! This module organizes comprehensive tests for the I/O functionality,
//! including mesh serialization utilities and round-trip validation.

pub mod mesh_serialization_tests;

#[test]
fn read_mesh_rejects_face_index_past_vertex_count() {
    let path = std::env::temp_dir().join(format!("tc_bad_face_{}.ply", std::process::id()));
    std::fs::write(
        &path,
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n\
         property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n\
         0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 99\n",
    )
    .unwrap();
    let result = crate::read_mesh(&path);
    std::fs::remove_file(&path).ok();
    assert!(result.unwrap_err().to_string().contains("vertex index"));
}
