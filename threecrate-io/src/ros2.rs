//! ROS 2 `sensor_msgs/PointCloud2` serialization and deserialization.
//!
//! This module converts between the raw binary layout of a `PointCloud2` message
//! and the threecrate point-cloud types.  **No ROS installation is required** —
//! the conversion operates purely on bytes and the field-descriptor metadata that
//! any ROS 2 driver or bag file exposes.
//!
//! # PointField datatype constants
//!
//! | Value | Type    | Size |
//! |-------|---------|------|
//! | 1     | int8    | 1 B  |
//! | 2     | uint8   | 1 B  |
//! | 3     | int16   | 2 B  |
//! | 4     | uint16  | 2 B  |
//! | 5     | int32   | 4 B  |
//! | 6     | uint32  | 4 B  |
//! | 7     | float32 | 4 B  |
//! | 8     | float64 | 8 B  |
//!
//! # RGB / RGBA packing
//!
//! ROS 2 stores RGB colour in a single `float32` field named `"rgb"` (or `"rgba"`).
//! The four bytes of that float are reinterpreted as a packed `uint32`:
//! `0xAARRGGBB`.  Alpha is silently dropped when storing into `[u8; 3]`.

use threecrate_core::{
    ColoredNormalPoint3f, ColoredPoint3f, NormalPoint3f, Point3f, PointCloud, Result, Vector3f,
    Error,
};

// ---------------------------------------------------------------------------
// Public metadata types
// ---------------------------------------------------------------------------

/// Mirrors `sensor_msgs/PointField`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointField {
    /// Field name, e.g. `"x"`, `"y"`, `"z"`, `"rgb"`, `"normal_x"`, `"intensity"`.
    pub name: String,
    /// Byte offset of this field within a single point row.
    pub offset: u32,
    /// Datatype constant (1–8, see module-level table).
    pub datatype: u8,
    /// Number of elements (almost always `1`).
    pub count: u32,
}

impl PointField {
    /// Byte size of one element of this field.
    pub fn element_size(&self) -> usize {
        match self.datatype {
            1 | 2 => 1,
            3 | 4 => 2,
            5 | 6 | 7 => 4,
            8 => 8,
            _ => 0,
        }
    }
}

/// Layout descriptor for a `PointCloud2` message (header-free).
#[derive(Debug, Clone)]
pub struct PointCloud2Info {
    /// Field descriptors in declaration order.
    pub fields: Vec<PointField>,
    /// Byte stride between consecutive points.
    pub point_step: u32,
    /// Byte stride between consecutive rows (`point_step * width`).
    pub row_step: u32,
    /// Points per row (for unorganised clouds: total point count).
    pub width: u32,
    /// Number of rows (for unorganised clouds: `1`).
    pub height: u32,
    /// `true` if the sensor produces big-endian multi-byte values.
    pub is_bigendian: bool,
    /// `false` if the data may contain NaN / Inf points.
    pub is_dense: bool,
}

impl PointCloud2Info {
    /// Total number of points described by this layout.
    #[inline]
    pub fn num_points(&self) -> usize {
        (self.width * self.height) as usize
    }
}

/// A self-contained, serialised PointCloud2 payload (no ROS header).
#[derive(Debug, Clone)]
pub struct PointCloud2Data {
    /// Layout metadata.
    pub info: PointCloud2Info,
    /// Raw point bytes, row-major, length == `height * row_step`.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn find_field<'a>(fields: &'a [PointField], name: &str) -> Option<&'a PointField> {
    fields.iter().find(|f| f.name == name)
}

/// Read a field value as `f64`, regardless of the on-wire datatype.
fn read_field_f64(data: &[u8], base: usize, field: &PointField, big: bool) -> Result<f64> {
    let off = base + field.offset as usize;
    match field.datatype {
        1 => Ok(data[off] as i8 as f64),
        2 => Ok(data[off] as f64),
        3 => {
            let b: [u8; 2] = data[off..off + 2].try_into().unwrap();
            Ok(if big { i16::from_be_bytes(b) } else { i16::from_le_bytes(b) } as f64)
        }
        4 => {
            let b: [u8; 2] = data[off..off + 2].try_into().unwrap();
            Ok(if big { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) } as f64)
        }
        5 => {
            let b: [u8; 4] = data[off..off + 4].try_into().unwrap();
            Ok(if big { i32::from_be_bytes(b) } else { i32::from_le_bytes(b) } as f64)
        }
        6 => {
            let b: [u8; 4] = data[off..off + 4].try_into().unwrap();
            Ok(if big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) } as f64)
        }
        7 => {
            let b: [u8; 4] = data[off..off + 4].try_into().unwrap();
            Ok(if big { f32::from_be_bytes(b) } else { f32::from_le_bytes(b) } as f64)
        }
        8 => {
            let b: [u8; 8] = data[off..off + 8].try_into().unwrap();
            Ok(if big { f64::from_be_bytes(b) } else { f64::from_le_bytes(b) })
        }
        d => Err(Error::InvalidData(format!("unknown PointField datatype {d}"))),
    }
}

/// Extract the packed RGB `uint32` from an `"rgb"` or `"rgba"` field.
/// The field may have datatype float32 (7) or uint32 (6).
fn read_rgb_packed(data: &[u8], base: usize, field: &PointField, big: bool) -> Result<u32> {
    let off = base + field.offset as usize;
    match field.datatype {
        7 => {
            let b: [u8; 4] = data[off..off + 4].try_into().unwrap();
            // Reinterpret the float32 bits as uint32 — do NOT do arithmetic conversion.
            let raw = if big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) };
            Ok(raw)
        }
        6 => {
            let b: [u8; 4] = data[off..off + 4].try_into().unwrap();
            Ok(if big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
        }
        d => Err(Error::InvalidData(format!(
            "rgb/rgba field has unsupported datatype {d} (expected 6=uint32 or 7=float32)"
        ))),
    }
}

/// Validate that the data buffer is large enough for the declared layout.
fn check_buffer(data: &[u8], info: &PointCloud2Info) -> Result<()> {
    let required = info.height as usize * info.row_step as usize;
    if data.len() < required {
        return Err(Error::InvalidData(format!(
            "PointCloud2 data too short: need {required} bytes, got {}",
            data.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Deserialization
// ---------------------------------------------------------------------------

/// Parse raw PointCloud2 bytes into `PointCloud<Point3f>` (XYZ only).
///
/// Points with NaN/Inf coordinates are skipped when `info.is_dense == false`.
pub fn pointcloud2_to_xyz(data: &[u8], info: &PointCloud2Info) -> Result<PointCloud<Point3f>> {
    check_buffer(data, info)?;

    let xf = find_field(&info.fields, "x")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'x'".into()))?;
    let yf = find_field(&info.fields, "y")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'y'".into()))?;
    let zf = find_field(&info.fields, "z")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'z'".into()))?;

    let ps = info.point_step as usize;
    let big = info.is_bigendian;
    let mut points = Vec::with_capacity(info.num_points());

    for row in 0..info.height as usize {
        let row_base = row * info.row_step as usize;
        for col in 0..info.width as usize {
            let base = row_base + col * ps;
            let x = read_field_f64(data, base, xf, big)? as f32;
            let y = read_field_f64(data, base, yf, big)? as f32;
            let z = read_field_f64(data, base, zf, big)? as f32;
            if !info.is_dense && (x.is_nan() || y.is_nan() || z.is_nan()) {
                continue;
            }
            points.push(Point3f::new(x, y, z));
        }
    }
    Ok(PointCloud::from_points(points))
}

/// Parse raw PointCloud2 bytes into `PointCloud<ColoredPoint3f>`.
///
/// Requires an `"rgb"` or `"rgba"` field.  Alpha is discarded.
pub fn pointcloud2_to_colored(
    data: &[u8],
    info: &PointCloud2Info,
) -> Result<PointCloud<ColoredPoint3f>> {
    check_buffer(data, info)?;

    let xf = find_field(&info.fields, "x")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'x'".into()))?;
    let yf = find_field(&info.fields, "y")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'y'".into()))?;
    let zf = find_field(&info.fields, "z")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'z'".into()))?;
    let cf = find_field(&info.fields, "rgb")
        .or_else(|| find_field(&info.fields, "rgba"))
        .ok_or_else(|| {
            Error::InvalidData("PointCloud2 missing 'rgb' or 'rgba' field".into())
        })?;

    let ps = info.point_step as usize;
    let big = info.is_bigendian;
    let mut points = Vec::with_capacity(info.num_points());

    for row in 0..info.height as usize {
        let row_base = row * info.row_step as usize;
        for col in 0..info.width as usize {
            let base = row_base + col * ps;
            let x = read_field_f64(data, base, xf, big)? as f32;
            let y = read_field_f64(data, base, yf, big)? as f32;
            let z = read_field_f64(data, base, zf, big)? as f32;
            if !info.is_dense && (x.is_nan() || y.is_nan() || z.is_nan()) {
                continue;
            }
            let packed = read_rgb_packed(data, base, cf, big)?;
            let r = ((packed >> 16) & 0xFF) as u8;
            let g = ((packed >> 8) & 0xFF) as u8;
            let b = (packed & 0xFF) as u8;
            points.push(ColoredPoint3f {
                position: Point3f::new(x, y, z),
                color: [r, g, b],
            });
        }
    }
    Ok(PointCloud::from_points(points))
}

/// Parse raw PointCloud2 bytes into `PointCloud<NormalPoint3f>`.
///
/// Requires `"normal_x"`, `"normal_y"`, `"normal_z"` fields in addition to `"x"`, `"y"`, `"z"`.
pub fn pointcloud2_to_normals(
    data: &[u8],
    info: &PointCloud2Info,
) -> Result<PointCloud<NormalPoint3f>> {
    check_buffer(data, info)?;

    let xf = find_field(&info.fields, "x")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'x'".into()))?;
    let yf = find_field(&info.fields, "y")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'y'".into()))?;
    let zf = find_field(&info.fields, "z")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'z'".into()))?;
    let nxf = find_field(&info.fields, "normal_x")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'normal_x'".into()))?;
    let nyf = find_field(&info.fields, "normal_y")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'normal_y'".into()))?;
    let nzf = find_field(&info.fields, "normal_z")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'normal_z'".into()))?;

    let ps = info.point_step as usize;
    let big = info.is_bigendian;
    let mut points = Vec::with_capacity(info.num_points());

    for row in 0..info.height as usize {
        let row_base = row * info.row_step as usize;
        for col in 0..info.width as usize {
            let base = row_base + col * ps;
            let x = read_field_f64(data, base, xf, big)? as f32;
            let y = read_field_f64(data, base, yf, big)? as f32;
            let z = read_field_f64(data, base, zf, big)? as f32;
            if !info.is_dense && (x.is_nan() || y.is_nan() || z.is_nan()) {
                continue;
            }
            let nx = read_field_f64(data, base, nxf, big)? as f32;
            let ny = read_field_f64(data, base, nyf, big)? as f32;
            let nz = read_field_f64(data, base, nzf, big)? as f32;
            points.push(NormalPoint3f {
                position: Point3f::new(x, y, z),
                normal: Vector3f::new(nx, ny, nz),
            });
        }
    }
    Ok(PointCloud::from_points(points))
}

/// Parse raw PointCloud2 bytes into `PointCloud<ColoredNormalPoint3f>`.
///
/// Requires `"x"`, `"y"`, `"z"`, `"normal_x"`, `"normal_y"`, `"normal_z"`, and `"rgb"`/`"rgba"`.
pub fn pointcloud2_to_colored_normals(
    data: &[u8],
    info: &PointCloud2Info,
) -> Result<PointCloud<ColoredNormalPoint3f>> {
    check_buffer(data, info)?;

    let xf = find_field(&info.fields, "x")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'x'".into()))?;
    let yf = find_field(&info.fields, "y")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'y'".into()))?;
    let zf = find_field(&info.fields, "z")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'z'".into()))?;
    let nxf = find_field(&info.fields, "normal_x")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'normal_x'".into()))?;
    let nyf = find_field(&info.fields, "normal_y")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'normal_y'".into()))?;
    let nzf = find_field(&info.fields, "normal_z")
        .ok_or_else(|| Error::InvalidData("PointCloud2 missing field 'normal_z'".into()))?;
    let cf = find_field(&info.fields, "rgb")
        .or_else(|| find_field(&info.fields, "rgba"))
        .ok_or_else(|| {
            Error::InvalidData("PointCloud2 missing 'rgb' or 'rgba' field".into())
        })?;

    let ps = info.point_step as usize;
    let big = info.is_bigendian;
    let mut points = Vec::with_capacity(info.num_points());

    for row in 0..info.height as usize {
        let row_base = row * info.row_step as usize;
        for col in 0..info.width as usize {
            let base = row_base + col * ps;
            let x = read_field_f64(data, base, xf, big)? as f32;
            let y = read_field_f64(data, base, yf, big)? as f32;
            let z = read_field_f64(data, base, zf, big)? as f32;
            if !info.is_dense && (x.is_nan() || y.is_nan() || z.is_nan()) {
                continue;
            }
            let nx = read_field_f64(data, base, nxf, big)? as f32;
            let ny = read_field_f64(data, base, nyf, big)? as f32;
            let nz = read_field_f64(data, base, nzf, big)? as f32;
            let packed = read_rgb_packed(data, base, cf, big)?;
            let r = ((packed >> 16) & 0xFF) as u8;
            let g = ((packed >> 8) & 0xFF) as u8;
            let b = (packed & 0xFF) as u8;
            points.push(ColoredNormalPoint3f {
                position: Point3f::new(x, y, z),
                normal: Vector3f::new(nx, ny, nz),
                color: [r, g, b],
            });
        }
    }
    Ok(PointCloud::from_points(points))
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn make_field(name: &str, offset: u32) -> PointField {
    PointField { name: name.into(), offset, datatype: 7, count: 1 }
}

fn make_info(fields: Vec<PointField>, point_step: u32, n: usize) -> PointCloud2Info {
    PointCloud2Info {
        fields,
        point_step,
        row_step: point_step * n as u32,
        width: n as u32,
        height: 1,
        is_bigendian: false,
        is_dense: true,
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serialise `PointCloud<Point3f>` to PointCloud2 format.
///
/// Output fields: `x` (offset 0), `y` (4), `z` (8) — all `float32`, little-endian.
/// `point_step` = 12.
pub fn xyz_to_pointcloud2(cloud: &PointCloud<Point3f>) -> PointCloud2Data {
    let n = cloud.len();
    let point_step: u32 = 12;
    let mut data = Vec::with_capacity(n * point_step as usize);
    for p in cloud.iter() {
        data.extend_from_slice(&p.x.to_le_bytes());
        data.extend_from_slice(&p.y.to_le_bytes());
        data.extend_from_slice(&p.z.to_le_bytes());
    }
    PointCloud2Data {
        info: make_info(
            vec![make_field("x", 0), make_field("y", 4), make_field("z", 8)],
            point_step,
            n,
        ),
        data,
    }
}

/// Serialise `PointCloud<ColoredPoint3f>` to PointCloud2 format.
///
/// Output fields: `x`(0), `y`(4), `z`(8), `rgb`(12) — `rgb` is `float32`
/// whose bits encode `0x00RRGGBB`.  `point_step` = 16.
pub fn colored_to_pointcloud2(cloud: &PointCloud<ColoredPoint3f>) -> PointCloud2Data {
    let n = cloud.len();
    let point_step: u32 = 16;
    let mut data = Vec::with_capacity(n * point_step as usize);
    for p in cloud.iter() {
        data.extend_from_slice(&p.position.x.to_le_bytes());
        data.extend_from_slice(&p.position.y.to_le_bytes());
        data.extend_from_slice(&p.position.z.to_le_bytes());
        let packed: u32 =
            ((p.color[0] as u32) << 16) | ((p.color[1] as u32) << 8) | (p.color[2] as u32);
        // Store the uint32 bit-pattern as a float32 field (ROS 2 convention).
        let rgb_f32 = f32::from_bits(packed);
        data.extend_from_slice(&rgb_f32.to_le_bytes());
    }
    PointCloud2Data {
        info: make_info(
            vec![
                make_field("x", 0),
                make_field("y", 4),
                make_field("z", 8),
                make_field("rgb", 12),
            ],
            point_step,
            n,
        ),
        data,
    }
}

/// Serialise `PointCloud<NormalPoint3f>` to PointCloud2 format.
///
/// Output fields: `x`(0), `y`(4), `z`(8), `normal_x`(12), `normal_y`(16), `normal_z`(20).
/// `point_step` = 24.
pub fn normals_to_pointcloud2(cloud: &PointCloud<NormalPoint3f>) -> PointCloud2Data {
    let n = cloud.len();
    let point_step: u32 = 24;
    let mut data = Vec::with_capacity(n * point_step as usize);
    for p in cloud.iter() {
        data.extend_from_slice(&p.position.x.to_le_bytes());
        data.extend_from_slice(&p.position.y.to_le_bytes());
        data.extend_from_slice(&p.position.z.to_le_bytes());
        data.extend_from_slice(&p.normal.x.to_le_bytes());
        data.extend_from_slice(&p.normal.y.to_le_bytes());
        data.extend_from_slice(&p.normal.z.to_le_bytes());
    }
    PointCloud2Data {
        info: make_info(
            vec![
                make_field("x", 0),
                make_field("y", 4),
                make_field("z", 8),
                make_field("normal_x", 12),
                make_field("normal_y", 16),
                make_field("normal_z", 20),
            ],
            point_step,
            n,
        ),
        data,
    }
}

/// Serialise `PointCloud<ColoredNormalPoint3f>` to PointCloud2 format.
///
/// Output fields: `x`(0), `y`(4), `z`(8), `normal_x`(12), `normal_y`(16),
/// `normal_z`(20), `rgb`(24).  `point_step` = 28.
pub fn colored_normals_to_pointcloud2(
    cloud: &PointCloud<ColoredNormalPoint3f>,
) -> PointCloud2Data {
    let n = cloud.len();
    let point_step: u32 = 28;
    let mut data = Vec::with_capacity(n * point_step as usize);
    for p in cloud.iter() {
        data.extend_from_slice(&p.position.x.to_le_bytes());
        data.extend_from_slice(&p.position.y.to_le_bytes());
        data.extend_from_slice(&p.position.z.to_le_bytes());
        data.extend_from_slice(&p.normal.x.to_le_bytes());
        data.extend_from_slice(&p.normal.y.to_le_bytes());
        data.extend_from_slice(&p.normal.z.to_le_bytes());
        let packed: u32 =
            ((p.color[0] as u32) << 16) | ((p.color[1] as u32) << 8) | (p.color[2] as u32);
        data.extend_from_slice(&f32::from_bits(packed).to_le_bytes());
    }
    PointCloud2Data {
        info: make_info(
            vec![
                make_field("x", 0),
                make_field("y", 4),
                make_field("z", 8),
                make_field("normal_x", 12),
                make_field("normal_y", 16),
                make_field("normal_z", 20),
                make_field("rgb", 24),
            ],
            point_step,
            n,
        ),
        data,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn xyz_fields(point_step: u32) -> PointCloud2Info {
        PointCloud2Info {
            fields: vec![
                PointField { name: "x".into(), offset: 0, datatype: 7, count: 1 },
                PointField { name: "y".into(), offset: 4, datatype: 7, count: 1 },
                PointField { name: "z".into(), offset: 8, datatype: 7, count: 1 },
            ],
            point_step,
            row_step: point_step * 3,
            width: 3,
            height: 1,
            is_bigendian: false,
            is_dense: true,
        }
    }

    // ---- round-trip XYZ ----

    #[test]
    fn xyz_round_trip() {
        let pts = vec![
            Point3f::new(1.0, 2.0, 3.0),
            Point3f::new(-0.5, 0.0, 100.0),
            Point3f::new(0.001, -99.9, 0.5),
        ];
        let cloud = PointCloud::from_points(pts.clone());
        let msg = xyz_to_pointcloud2(&cloud);

        assert_eq!(msg.info.point_step, 12);
        assert_eq!(msg.data.len(), 3 * 12);

        let back = pointcloud2_to_xyz(&msg.data, &msg.info).unwrap();
        assert_eq!(back.len(), 3);
        for (orig, got) in pts.iter().zip(back.iter()) {
            assert_relative_eq!(got.x, orig.x, epsilon = 1e-6);
            assert_relative_eq!(got.y, orig.y, epsilon = 1e-6);
            assert_relative_eq!(got.z, orig.z, epsilon = 1e-6);
        }
    }

    // ---- big-endian decoding ----

    #[test]
    fn bigendian_xyz() {
        // Manually write big-endian f32 bytes for (1.0, 2.0, 3.0).
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_be_bytes());
        data.extend_from_slice(&2.0f32.to_be_bytes());
        data.extend_from_slice(&3.0f32.to_be_bytes());

        let info = PointCloud2Info {
            fields: vec![
                PointField { name: "x".into(), offset: 0, datatype: 7, count: 1 },
                PointField { name: "y".into(), offset: 4, datatype: 7, count: 1 },
                PointField { name: "z".into(), offset: 8, datatype: 7, count: 1 },
            ],
            point_step: 12,
            row_step: 12,
            width: 1,
            height: 1,
            is_bigendian: true,
            is_dense: true,
        };
        let cloud = pointcloud2_to_xyz(&data, &info).unwrap();
        assert_eq!(cloud.len(), 1);
        assert_relative_eq!(cloud.points[0].x, 1.0f32, epsilon = 1e-6);
        assert_relative_eq!(cloud.points[0].y, 2.0f32, epsilon = 1e-6);
        assert_relative_eq!(cloud.points[0].z, 3.0f32, epsilon = 1e-6);
    }

    // ---- NaN skip when is_dense == false ----

    #[test]
    fn nan_points_skipped() {
        let mut data = Vec::new();
        // Point 0: valid
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        // Point 1: NaN x
        data.extend_from_slice(&f32::NAN.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // Point 2: valid
        data.extend_from_slice(&4.0f32.to_le_bytes());
        data.extend_from_slice(&5.0f32.to_le_bytes());
        data.extend_from_slice(&6.0f32.to_le_bytes());

        let mut info = xyz_fields(12);
        info.width = 3;
        info.row_step = 36;
        info.is_dense = false;

        let cloud = pointcloud2_to_xyz(&data, &info).unwrap();
        assert_eq!(cloud.len(), 2, "NaN point should be dropped");
        assert_relative_eq!(cloud.points[0].x, 1.0f32, epsilon = 1e-6);
        assert_relative_eq!(cloud.points[1].x, 4.0f32, epsilon = 1e-6);
    }

    // ---- RGB packing round-trip ----

    #[test]
    fn rgb_round_trip() {
        let pts = vec![
            ColoredPoint3f { position: Point3f::new(0.0, 0.0, 0.0), color: [255, 128, 0] },
            ColoredPoint3f { position: Point3f::new(1.0, 1.0, 1.0), color: [0, 64, 200] },
        ];
        let cloud = PointCloud::from_points(pts.clone());
        let msg = colored_to_pointcloud2(&cloud);

        assert_eq!(msg.info.point_step, 16);
        let back = pointcloud2_to_colored(&msg.data, &msg.info).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.points[0].color, [255, 128, 0]);
        assert_eq!(back.points[1].color, [0, 64, 200]);
    }

    // ---- normals round-trip ----

    #[test]
    fn normals_round_trip() {
        let pts = vec![NormalPoint3f {
            position: Point3f::new(1.0, 0.0, 0.0),
            normal: Vector3f::new(0.0, 1.0, 0.0),
        }];
        let cloud = PointCloud::from_points(pts);
        let msg = normals_to_pointcloud2(&cloud);

        assert_eq!(msg.info.point_step, 24);
        let back = pointcloud2_to_normals(&msg.data, &msg.info).unwrap();
        assert_eq!(back.len(), 1);
        assert_relative_eq!(back.points[0].normal.y, 1.0f32, epsilon = 1e-6);
    }

    // ---- missing field returns error ----

    #[test]
    fn missing_rgb_field_errors() {
        // Provide enough bytes for 3 points (width=3, point_step=12) so that
        // check_buffer passes and the missing-field error is the one surfaced.
        let data = vec![0u8; 36];
        let info = xyz_fields(12);
        let result = pointcloud2_to_colored(&data, &info);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("rgb") || msg.contains("rgba"), "unexpected error: {msg}");
    }

    // ---- float64 xyz field ----

    #[test]
    fn float64_xyz_field() {
        let mut data = Vec::new();
        data.extend_from_slice(&7.5f64.to_le_bytes()); // x (float64)
        data.extend_from_slice(&(-3.0f64).to_le_bytes()); // y
        data.extend_from_slice(&0.25f64.to_le_bytes()); // z

        let info = PointCloud2Info {
            fields: vec![
                PointField { name: "x".into(), offset: 0, datatype: 8, count: 1 },
                PointField { name: "y".into(), offset: 8, datatype: 8, count: 1 },
                PointField { name: "z".into(), offset: 16, datatype: 8, count: 1 },
            ],
            point_step: 24,
            row_step: 24,
            width: 1,
            height: 1,
            is_bigendian: false,
            is_dense: true,
        };
        let cloud = pointcloud2_to_xyz(&data, &info).unwrap();
        assert_eq!(cloud.len(), 1);
        assert_relative_eq!(cloud.points[0].x, 7.5f32, epsilon = 1e-4);
        assert_relative_eq!(cloud.points[0].y, -3.0f32, epsilon = 1e-4);
    }

    // ---- colored normals round-trip ----

    #[test]
    fn colored_normals_round_trip() {
        let pts = vec![ColoredNormalPoint3f {
            position: Point3f::new(1.0, 2.0, 3.0),
            normal: Vector3f::new(0.0, 0.0, 1.0),
            color: [10, 20, 30],
        }];
        let cloud = PointCloud::from_points(pts);
        let msg = colored_normals_to_pointcloud2(&cloud);
        assert_eq!(msg.info.point_step, 28);
        let back = pointcloud2_to_colored_normals(&msg.data, &msg.info).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back.points[0].color, [10, 20, 30]);
        assert_relative_eq!(back.points[0].normal.z, 1.0f32, epsilon = 1e-6);
    }

    // ---- buffer too short ----

    #[test]
    fn buffer_too_short_errors() {
        let data = vec![0u8; 8]; // needs 12
        let info = xyz_fields(12);
        assert!(pointcloud2_to_xyz(&data, &info).is_err());
    }

    // ---- padding between fields is handled correctly ----

    #[test]
    fn padded_layout() {
        // x at 0, y at 4, z at 8, then 4 bytes padding, intensity at 16 — point_step = 20
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes()); // x
        data.extend_from_slice(&2.0f32.to_le_bytes()); // y
        data.extend_from_slice(&3.0f32.to_le_bytes()); // z
        data.extend_from_slice(&0u32.to_le_bytes()); // padding
        data.extend_from_slice(&42.0f32.to_le_bytes()); // intensity (ignored by xyz parser)

        let info = PointCloud2Info {
            fields: vec![
                PointField { name: "x".into(), offset: 0, datatype: 7, count: 1 },
                PointField { name: "y".into(), offset: 4, datatype: 7, count: 1 },
                PointField { name: "z".into(), offset: 8, datatype: 7, count: 1 },
                PointField { name: "intensity".into(), offset: 16, datatype: 7, count: 1 },
            ],
            point_step: 20,
            row_step: 20,
            width: 1,
            height: 1,
            is_bigendian: false,
            is_dense: true,
        };
        let cloud = pointcloud2_to_xyz(&data, &info).unwrap();
        assert_eq!(cloud.len(), 1);
        assert_relative_eq!(cloud.points[0].z, 3.0f32, epsilon = 1e-6);
    }
}
