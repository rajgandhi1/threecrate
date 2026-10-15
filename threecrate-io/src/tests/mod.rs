//! Test modules for threecrate-io
//!
//! This module organizes comprehensive tests for the I/O functionality,
//! including mesh serialization utilities and round-trip validation.

pub mod mesh_serialization_tests;

/// Header counts are untrusted: a tiny file claiming billions of points must
/// give an error, not reserve terabytes and abort the process.
#[test]
fn huge_header_counts_are_errors_not_aborts() {
    let dir = std::env::temp_dir();
    let id = std::process::id();
    let cases = [
        (
            "ply",
            "ply\nformat binary_little_endian 1.0\nelement vertex 4000000000\n\
             property float x\nproperty float y\nproperty float z\nend_header\n",
        ),
        (
            "ply",
            "ply\nformat ascii 1.0\nelement vertex 18446744073709551615\n\
             property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
        ),
        (
            "pcd",
            "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
             WIDTH 4000000000\nHEIGHT 1\nPOINTS 4000000000\nDATA ascii\n0 0 0\n",
        ),
        (
            "pcd",
            "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n\
             WIDTH 4294967296\nHEIGHT 4294967296\nDATA ascii\n0 0 0\n",
        ),
    ];
    for (i, (ext, text)) in cases.iter().enumerate() {
        let path = dir.join(format!("tc_huge_{id}_{i}.{ext}"));
        std::fs::write(&path, text).unwrap();
        let result = crate::read_point_cloud(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "case {i} should be an error");
    }
}
