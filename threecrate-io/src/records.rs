//! Fast x/y/z extraction for record-based formats (PLY, PCD).
//!
//! The general readers build a map of every property for every point, which
//! costs about 320 bytes per point. When only positions are wanted and every
//! record has a fixed layout, the header says where x, y and z are, so they can
//! be read straight into `Point3f`s and everything else skipped.

use std::io::{BufRead, Read};
use threecrate_core::{Error, Point3f, Result};

/// How one coordinate is stored on disk.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    pub(crate) fn size(self) -> usize {
        match self {
            Scalar::I8 | Scalar::U8 => 1,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::F64 => 8,
        }
    }

    fn decode(self, bytes: &[u8], big_endian: bool) -> f32 {
        macro_rules! num {
            ($t:ty, $n:expr) => {{
                let raw: [u8; $n] = bytes[..$n].try_into().unwrap();
                if big_endian {
                    <$t>::from_be_bytes(raw)
                } else {
                    <$t>::from_le_bytes(raw)
                }
            }};
        }
        match self {
            Scalar::I8 => bytes[0] as i8 as f32,
            Scalar::U8 => bytes[0] as f32,
            Scalar::I16 => num!(i16, 2) as f32,
            Scalar::U16 => num!(u16, 2) as f32,
            Scalar::I32 => num!(i32, 4) as f32,
            Scalar::U32 => num!(u32, 4) as f32,
            Scalar::F32 => num!(f32, 4),
            Scalar::F64 => num!(f64, 8) as f32,
        }
    }
}

/// Where x, y and z sit inside one fixed-size binary record.
pub(crate) struct BinaryXyz {
    pub record_size: usize,
    /// (byte offset, type) of x, y and z.
    pub xyz: [(usize, Scalar); 3],
    pub big_endian: bool,
}

/// Read `count` binary records and keep only their positions.
pub(crate) fn read_binary_xyz<R: Read>(
    reader: &mut R,
    layout: &BinaryXyz,
    count: usize,
) -> Result<Vec<Point3f>> {
    const CHUNK_RECORDS: usize = 64 * 1024;
    if layout.record_size == 0 {
        return Err(Error::InvalidData("record size is 0".to_string()));
    }
    let mut points = Vec::with_capacity(crate::capacity_hint(count));
    let mut buffer = vec![0u8; CHUNK_RECORDS.min(count.max(1)) * layout.record_size];
    while points.len() < count {
        let records = (count - points.len()).min(CHUNK_RECORDS);
        let chunk = &mut buffer[..records * layout.record_size];
        reader.read_exact(chunk).map_err(|_| {
            Error::InvalidData(format!(
                "data ended after about {} of {count} points",
                points.len()
            ))
        })?;
        points.extend(chunk.chunks_exact(layout.record_size).map(|record| {
            let [x, y, z] = layout
                .xyz
                .map(|(offset, ty)| ty.decode(&record[offset..], layout.big_endian));
            Point3f::new(x, y, z)
        }));
    }
    Ok(points)
}

/// Read `count` text records, one per non-blank line, taking x, y and z from
/// the whitespace-separated tokens at `columns`.
pub(crate) fn read_ascii_xyz<R: BufRead>(
    reader: &mut R,
    columns: [usize; 3],
    count: usize,
) -> Result<Vec<Point3f>> {
    let mut points = Vec::with_capacity(crate::capacity_hint(count));
    let mut line = String::new();
    while points.len() < count {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(Error::InvalidData(format!(
                "data ended after {} of {count} points",
                points.len()
            )));
        }
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let [x, y, z] = columns.map(|c| {
            tokens
                .get(c)
                .and_then(|t| t.parse::<f32>().ok())
                .ok_or_else(|| {
                    Error::InvalidData(format!("bad or missing value in row: {}", line.trim()))
                })
        });
        points.push(Point3f::new(x?, y?, z?));
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_records_skip_other_fields_and_respect_endianness() {
        // record: u8 tag, f32 x, f64 y, i16 z  (15 bytes)
        let mut le = Vec::new();
        let mut be = Vec::new();
        for (x, y, z) in [(1.5f32, -2.25f64, 7i16), (0.0, 3.0, -4)] {
            le.push(9u8);
            le.extend(x.to_le_bytes());
            le.extend(y.to_le_bytes());
            le.extend(z.to_le_bytes());
            be.push(9u8);
            be.extend(x.to_be_bytes());
            be.extend(y.to_be_bytes());
            be.extend(z.to_be_bytes());
        }
        let layout = |big_endian| BinaryXyz {
            record_size: 15,
            xyz: [(1, Scalar::F32), (5, Scalar::F64), (13, Scalar::I16)],
            big_endian,
        };
        for (bytes, big) in [(le, false), (be, true)] {
            let points = read_binary_xyz(&mut bytes.as_slice(), &layout(big), 2).unwrap();
            assert_eq!(
                points,
                vec![Point3f::new(1.5, -2.25, 7.0), Point3f::new(0.0, 3.0, -4.0)]
            );
        }
        // Too few records is an error, not made-up points.
        assert!(read_binary_xyz(&mut [0u8; 20].as_slice(), &layout(false), 2).is_err());
    }

    #[test]
    fn ascii_records_pick_columns_and_skip_blank_lines() {
        let text = "9 1 2 3\n\n9 4 5 6\n";
        let points = read_ascii_xyz(&mut text.as_bytes(), [1, 2, 3], 2).unwrap();
        assert_eq!(
            points,
            vec![Point3f::new(1.0, 2.0, 3.0), Point3f::new(4.0, 5.0, 6.0)]
        );
        assert!(read_ascii_xyz(&mut "1 2 3\n".as_bytes(), [0, 1, 2], 2).is_err());
        assert!(read_ascii_xyz(&mut "1 2\n".as_bytes(), [0, 1, 2], 1).is_err());
    }
}
