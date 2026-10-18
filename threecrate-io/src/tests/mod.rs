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

/// The fast position readers must return exactly what the general readers do,
/// across encodings, types, field order and extra fields.
#[test]
fn fast_position_readers_match_the_general_readers() {
    use crate::pcd::RobustPcdReader;
    use crate::ply::RobustPlyReader;
    use threecrate_core::Point3f;

    let points: Vec<[f64; 3]> = (0..50)
        .map(|i| {
            [
                i as f64 * 0.5 - 3.0,
                (i * 7 % 11) as f64 * 0.25,
                -(i as f64) * 0.125,
            ]
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("tc_fast_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // PLY: (format, header properties, writer for one vertex)
    type Row = fn(&[f64; 3], &mut Vec<u8>);
    let ply_cases: [(&str, &str, Row); 4] = [
        ("ascii", "property float x\nproperty float y\nproperty float z\n", |p, out| {
            out.extend(format!("{} {} {}\n", p[0], p[1], p[2]).bytes())
        }),
        (
            "ascii",
            "property uchar red\nproperty float z\nproperty float nx\nproperty float x\nproperty float y\n",
            |p, out| out.extend(format!("200 {} 0.5 {} {}\n", p[2], p[0], p[1]).bytes()),
        ),
        (
            "binary_little_endian",
            "property double x\nproperty uchar tag\nproperty double y\nproperty double z\nproperty float nx\n",
            |p, out| {
                out.extend(p[0].to_le_bytes());
                out.push(7);
                out.extend(p[1].to_le_bytes());
                out.extend(p[2].to_le_bytes());
                out.extend(1.0f32.to_le_bytes());
            },
        ),
        (
            "binary_big_endian",
            "property float x\nproperty float y\nproperty float z\nproperty short intensity\n",
            |p, out| {
                for v in p {
                    out.extend((*v as f32).to_be_bytes());
                }
                out.extend(3i16.to_be_bytes());
            },
        ),
    ];
    for (i, (format, properties, row)) in ply_cases.iter().enumerate() {
        let mut bytes = format!(
            "ply\nformat {format} 1.0\nelement vertex {}\n{properties}end_header\n",
            points.len()
        )
        .into_bytes();
        for p in &points {
            row(p, &mut bytes);
        }
        let path = dir.join(format!("case{i}.ply"));
        std::fs::write(&path, &bytes).unwrap();
        let fast = crate::read_point_cloud(&path).unwrap();
        let general: Vec<Point3f> = RobustPlyReader::read_ply_file(&path).unwrap().elements
            ["vertex"]
            .iter()
            .map(|v| {
                Point3f::new(
                    v["x"].as_f32().unwrap(),
                    v["y"].as_f32().unwrap(),
                    v["z"].as_f32().unwrap(),
                )
            })
            .collect();
        assert_eq!(fast.points, general, "PLY case {i} ({format})");
    }

    // PCD: ascii and binary, with a multi-value field before x.
    for (i, data) in ["ascii", "binary"].iter().enumerate() {
        let mut bytes = format!(
            "VERSION .7\nFIELDS hist x y z rgb\nSIZE 4 4 4 4 4\nTYPE F F F F U\nCOUNT 2 1 1 1 1\n\
             WIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA {data}\n",
            points.len(),
            points.len()
        )
        .into_bytes();
        for p in &points {
            if *data == "ascii" {
                bytes.extend(format!("0.5 0.25 {} {} {} 255\n", p[0], p[1], p[2]).bytes());
            } else {
                for v in [0.5f32, 0.25, p[0] as f32, p[1] as f32, p[2] as f32] {
                    bytes.extend(v.to_le_bytes());
                }
                bytes.extend(255u32.to_le_bytes());
            }
        }
        let path = dir.join(format!("case{i}.pcd"));
        std::fs::write(&path, &bytes).unwrap();
        let fast = crate::read_point_cloud(&path).unwrap();
        let (header, raw) = RobustPcdReader::read_pcd_file(&path).unwrap();
        let general = RobustPcdReader::pcd_to_point_cloud(&header, &raw).unwrap();
        assert_eq!(fast.points, general.points, "PCD case {i} ({data})");
        assert_eq!(fast.points.len(), points.len());
    }
    std::fs::remove_dir_all(&dir).ok();
}
