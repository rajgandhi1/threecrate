//! Readers for ROS 2 bag formats: MCAP (`.mcap`) and rosbag2 SQLite (`.db3`).
//!
//! Both readers expose the same [`BagMessage`] type, which wraps the raw
//! serialized payload (CDR-encoded for ROS 2). Pull a `PointCloud<Point3f>`
//! out of a PointCloud2 message with [`BagMessage::as_pointcloud2`].
//!
//! This module is gated by the `rosbag` feature so that downstream users
//! who don't need `mcap` / `rusqlite` don't pay for them.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use threecrate_core::{Error, Point3f, PointCloud, Result};

use crate::ros2::{pointcloud2_to_xyz, PointCloud2Info, PointField as Pc2PointField};

/// One message from a bag: a topic, ROS 2 message type, timestamp (ns since epoch),
/// and the raw CDR-encoded payload.
#[derive(Debug, Clone)]
pub struct BagMessage {
    /// ROS topic, e.g. `/lidar/points`.
    pub topic: String,
    /// ROS message type, e.g. `sensor_msgs/msg/PointCloud2`.
    pub message_type: String,
    /// Log timestamp in nanoseconds since the Unix epoch.
    pub timestamp: u64,
    /// Raw CDR-encoded payload (with the 4-byte encapsulation header).
    pub data: Vec<u8>,
}

impl BagMessage {
    /// Deserialize this message as a `sensor_msgs/PointCloud2` and return the
    /// XYZ points. NaN points are dropped (matching the existing PointCloud2
    /// conversion semantics).
    pub fn as_pointcloud2(&self) -> Result<PointCloud<Point3f>> {
        let (info, body) = parse_pointcloud2_cdr(&self.data)?;
        pointcloud2_to_xyz(body, &info)
    }
}

/// Topic descriptor (name + message type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicInfo {
    /// Topic name.
    pub name: String,
    /// ROS message type string.
    pub message_type: String,
}

// ---------------------------------------------------------------------------
// CDR parsing of sensor_msgs/PointCloud2
// ---------------------------------------------------------------------------

/// A minimal CDR reader tracking position within the encapsulation body so it
/// can honour CDR's natural-alignment rules.
struct CdrReader<'a> {
    buf: &'a [u8],
    pos: usize,
    little_endian: bool,
}

impl<'a> CdrReader<'a> {
    /// Construct from a CDR payload starting with the 4-byte encapsulation header.
    fn from_payload(payload: &'a [u8]) -> Result<(Self, &'a [u8])> {
        if payload.len() < 4 {
            return Err(Error::InvalidData(
                "CDR payload too short for encapsulation header".into(),
            ));
        }
        // Byte 0: representation id MSB (always 0 for plain CDR).
        // Byte 1: 0=BE, 1=LE.
        let little_endian = match payload[1] {
            0 => false,
            1 => true,
            other => {
                return Err(Error::InvalidData(format!(
                    "Unsupported CDR encapsulation kind: 0x00{other:02x}"
                )))
            }
        };
        let body = &payload[4..];
        Ok((
            Self {
                buf: body,
                pos: 0,
                little_endian,
            },
            body,
        ))
    }

    fn align(&mut self, n: usize) {
        let r = self.pos % n;
        if r != 0 {
            self.pos += n - r;
        }
    }

    fn ensure(&self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            Err(Error::InvalidData("Unexpected end of CDR payload".into()))
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.align(4);
        self.ensure(4)?;
        let b: [u8; 4] = self.buf[self.pos..self.pos + 4].try_into().unwrap();
        self.pos += 4;
        Ok(if self.little_endian {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        if len == 0 {
            return Err(Error::InvalidData("CDR string length cannot be 0".into()));
        }
        self.ensure(len)?;
        // Length includes the trailing NUL.
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len - 1])
            .map_err(|_| Error::InvalidData("invalid UTF-8 in CDR string".into()))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn skip_string(&mut self) -> Result<()> {
        let len = self.read_u32()? as usize;
        self.ensure(len)?;
        self.pos += len;
        Ok(())
    }
}

/// Parse a CDR-encoded `sensor_msgs/PointCloud2` payload into a
/// [`PointCloud2Info`] descriptor plus the raw point bytes (suitable for
/// passing to [`pointcloud2_to_xyz`]).
fn parse_pointcloud2_cdr(payload: &[u8]) -> Result<(PointCloud2Info, &[u8])> {
    let (mut r, _) = CdrReader::from_payload(payload)?;

    // std_msgs/Header { stamp { sec: i32, nanosec: u32 }, frame_id: string }
    let _sec = r.read_i32()?;
    let _nanosec = r.read_u32()?;
    r.skip_string()?; // frame_id

    let height = r.read_u32()?;
    let width = r.read_u32()?;

    // fields: sequence<PointField>
    let n_fields = r.read_u32()? as usize;
    let mut fields = Vec::with_capacity(n_fields);
    for _ in 0..n_fields {
        // PointField { name: string, offset: u32, datatype: u8, count: u32 }
        let name = r.read_string()?;
        let offset = r.read_u32()?;
        let datatype = r.read_u8()?;
        let count = r.read_u32()?;
        fields.push(Pc2PointField {
            name,
            offset,
            datatype,
            count,
        });
    }

    let is_bigendian = r.read_bool()?;
    let point_step = r.read_u32()?;
    let row_step = r.read_u32()?;

    // data: sequence<u8>
    let n_data = r.read_u32()? as usize;
    r.ensure(n_data)?;
    let start = r.pos;
    let data_slice = &r.buf[start..start + n_data];
    r.pos += n_data;

    let is_dense = r.read_bool()?;

    let info = PointCloud2Info {
        fields,
        point_step,
        row_step,
        width,
        height,
        is_bigendian,
        is_dense,
    };
    Ok((info, data_slice))
}

// ---------------------------------------------------------------------------
// MCAP reader
// ---------------------------------------------------------------------------

/// Reader over an MCAP file. Holds the file bytes in memory.
pub struct McapReader {
    bytes: Vec<u8>,
}

impl McapReader {
    /// Open an MCAP file and read it into memory.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        // Quick magic-byte check.
        if !bytes.starts_with(&mcap::MAGIC[..4]) {
            return Err(Error::InvalidData(
                "File does not start with MCAP magic bytes".into(),
            ));
        }
        Ok(Self { bytes })
    }

    /// All topics found in the file (deduplicated by name).
    pub fn topics(&self) -> Result<Vec<TopicInfo>> {
        let mut seen: HashMap<String, String> = HashMap::new();
        let stream = mcap::MessageStream::new(&self.bytes)
            .map_err(|e| Error::InvalidData(format!("MCAP open failed: {e}")))?;
        for msg in stream {
            let msg = msg.map_err(|e| Error::InvalidData(format!("MCAP read failed: {e}")))?;
            let topic = msg.channel.topic.clone();
            if seen.contains_key(&topic) {
                continue;
            }
            let mt = msg
                .channel
                .schema
                .as_ref()
                .map(|s| s.name.clone())
                .unwrap_or_default();
            seen.insert(topic, mt);
        }
        Ok(seen
            .into_iter()
            .map(|(name, message_type)| TopicInfo { name, message_type })
            .collect())
    }

    /// Iterator over every message in the file, in log order.
    pub fn messages(&self) -> Result<impl Iterator<Item = Result<BagMessage>> + '_> {
        let stream = mcap::MessageStream::new(&self.bytes)
            .map_err(|e| Error::InvalidData(format!("MCAP open failed: {e}")))?;
        Ok(stream.map(|res| {
            res.map(|msg| {
                let message_type = msg
                    .channel
                    .schema
                    .as_ref()
                    .map(|s| s.name.clone())
                    .unwrap_or_default();
                BagMessage {
                    topic: msg.channel.topic.clone(),
                    message_type,
                    timestamp: msg.log_time,
                    data: msg.data.into_owned(),
                }
            })
            .map_err(|e| Error::InvalidData(format!("MCAP read failed: {e}")))
        }))
    }

    /// Iterator restricted to a single topic.
    pub fn messages_on_topic<'a>(
        &'a self,
        topic: &'a str,
    ) -> Result<impl Iterator<Item = Result<BagMessage>> + 'a> {
        Ok(self.messages()?.filter(move |m| match m {
            Ok(m) => m.topic == topic,
            Err(_) => true,
        }))
    }

    /// Iterator restricted to a `[start_ns, end_ns]` time window (inclusive).
    pub fn messages_in_range(
        &self,
        start_ns: u64,
        end_ns: u64,
    ) -> Result<impl Iterator<Item = Result<BagMessage>> + '_> {
        Ok(self.messages()?.filter(move |m| match m {
            Ok(m) => m.timestamp >= start_ns && m.timestamp <= end_ns,
            Err(_) => true,
        }))
    }
}

// ---------------------------------------------------------------------------
// rosbag2 SQLite reader
// ---------------------------------------------------------------------------

/// Reader over a rosbag2 SQLite (`.db3`) recording.
///
/// Accepts either the recording directory (looks for the first `.db3` inside
/// it) or a `.db3` file path directly.
pub struct Rosbag2Reader {
    conn: rusqlite::Connection,
    topics: Vec<(i64, TopicInfo)>, // (topic_id, info)
}

impl Rosbag2Reader {
    /// Open a rosbag2 recording.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let db_path: PathBuf = if path.is_dir() {
            let entry = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .find(|e| {
                    e.path()
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.eq_ignore_ascii_case("db3"))
                        .unwrap_or(false)
                })
                .ok_or_else(|| {
                    Error::InvalidData(format!(
                        "no .db3 file found in rosbag2 directory: {}",
                        path.display()
                    ))
                })?;
            entry.path()
        } else {
            path.to_path_buf()
        };

        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| Error::InvalidData(format!("failed to open rosbag2 db: {e}")))?;

        // Topics table schema is "topics(id, name, type, ...)" in rosbag2 v1+.
        let mut stmt = conn
            .prepare("SELECT id, name, type FROM topics")
            .map_err(|e| Error::InvalidData(format!("failed to read topics: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| Error::InvalidData(format!("topics query failed: {e}")))?;
        let mut topics = Vec::new();
        for row in rows {
            let (id, name, ty) =
                row.map_err(|e| Error::InvalidData(format!("topics row error: {e}")))?;
            topics.push((
                id,
                TopicInfo {
                    name,
                    message_type: ty,
                },
            ));
        }
        drop(stmt);

        Ok(Self { conn, topics })
    }

    /// List the topics in this bag.
    pub fn topics(&self) -> Vec<TopicInfo> {
        self.topics.iter().map(|(_, t)| t.clone()).collect()
    }

    fn topic_id(&self, topic: &str) -> Option<i64> {
        self.topics
            .iter()
            .find(|(_, t)| t.name == topic)
            .map(|(id, _)| *id)
    }

    /// Read every message on a single topic, in timestamp order.
    pub fn read_topic(&self, topic: &str) -> Result<Vec<BagMessage>> {
        let topic_id = self
            .topic_id(topic)
            .ok_or_else(|| Error::InvalidData(format!("topic not found in bag: {topic}")))?;
        let info = self
            .topics
            .iter()
            .find(|(id, _)| *id == topic_id)
            .map(|(_, t)| t.clone())
            .unwrap();

        let mut stmt = self
            .conn
            .prepare("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp")
            .map_err(|e| Error::InvalidData(format!("messages query failed: {e}")))?;
        let rows = stmt
            .query_map([topic_id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| Error::InvalidData(format!("messages query failed: {e}")))?;

        let mut out = Vec::new();
        for row in rows {
            let (ts, data) = row.map_err(|e| Error::InvalidData(format!("row error: {e}")))?;
            out.push(BagMessage {
                topic: info.name.clone(),
                message_type: info.message_type.clone(),
                timestamp: ts as u64,
                data,
            });
        }
        Ok(out)
    }

    /// Read every message whose timestamp is in `[start_ns, end_ns]`.
    pub fn messages_in_range(&self, start_ns: u64, end_ns: u64) -> Result<Vec<BagMessage>> {
        let topic_map: HashMap<i64, TopicInfo> = self.topics.iter().cloned().collect();
        let mut stmt = self
            .conn
            .prepare("SELECT topic_id, timestamp, data FROM messages WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp")
            .map_err(|e| Error::InvalidData(format!("messages query failed: {e}")))?;
        let rows = stmt
            .query_map(rusqlite::params![start_ns as i64, end_ns as i64], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })
            .map_err(|e| Error::InvalidData(format!("range query failed: {e}")))?;

        let mut out = Vec::new();
        for row in rows {
            let (tid, ts, data) = row.map_err(|e| Error::InvalidData(format!("row error: {e}")))?;
            let info = topic_map.get(&tid).cloned().unwrap_or(TopicInfo {
                name: String::new(),
                message_type: String::new(),
            });
            out.push(BagMessage {
                topic: info.name,
                message_type: info.message_type,
                timestamp: ts as u64,
                data,
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CDR-encoded PointCloud2 byte buffer carrying three XYZ float32 points.
    fn build_pc2_cdr(points: &[(f32, f32, f32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        // CDR encapsulation header: PLAIN_CDR, little-endian.
        buf.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]);

        let body_start = buf.len();
        let align = |buf: &mut Vec<u8>, n: usize| {
            let r = (buf.len() - body_start) % n;
            if r != 0 {
                buf.extend(std::iter::repeat(0u8).take(n - r));
            }
        };

        // header.stamp.sec (i32) + nanosec (u32).
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        // header.frame_id = "map" (length=4 with NUL).
        align(&mut buf, 4);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(b"map\0");

        // height = 1, width = N.
        align(&mut buf, 4);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(points.len() as u32).to_le_bytes());

        // fields: x, y, z — each PointField.
        align(&mut buf, 4);
        buf.extend_from_slice(&3u32.to_le_bytes()); // sequence length

        let mut add_field = |buf: &mut Vec<u8>, name: &str, offset: u32| {
            // name: string
            let bytes = name.as_bytes();
            let len = (bytes.len() + 1) as u32;
            // String length is a u32 -> align 4.
            let r = (buf.len() - body_start) % 4;
            if r != 0 {
                buf.extend(std::iter::repeat(0u8).take(4 - r));
            }
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(bytes);
            buf.push(0);
            // offset: u32
            let r = (buf.len() - body_start) % 4;
            if r != 0 {
                buf.extend(std::iter::repeat(0u8).take(4 - r));
            }
            buf.extend_from_slice(&offset.to_le_bytes());
            // datatype: u8 (7=float32)
            buf.push(7);
            // count: u32
            let r = (buf.len() - body_start) % 4;
            if r != 0 {
                buf.extend(std::iter::repeat(0u8).take(4 - r));
            }
            buf.extend_from_slice(&1u32.to_le_bytes());
        };

        add_field(&mut buf, "x", 0);
        add_field(&mut buf, "y", 4);
        add_field(&mut buf, "z", 8);

        // is_bigendian: bool (u8 = 0).
        buf.push(0);
        // point_step: u32, row_step: u32.
        align(&mut buf, 4);
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend_from_slice(&((12 * points.len()) as u32).to_le_bytes());

        // data: sequence<u8>.
        align(&mut buf, 4);
        buf.extend_from_slice(&((12 * points.len()) as u32).to_le_bytes());
        for (x, y, z) in points {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&z.to_le_bytes());
        }

        // is_dense: bool (true).
        buf.push(1);

        buf
    }

    #[test]
    fn bag_message_decodes_pointcloud2() {
        let points = vec![(1.0_f32, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)];
        let cdr = build_pc2_cdr(&points);
        let msg = BagMessage {
            topic: "/test".into(),
            message_type: "sensor_msgs/msg/PointCloud2".into(),
            timestamp: 1_000_000_000,
            data: cdr,
        };
        let cloud = msg.as_pointcloud2().unwrap();
        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud.points[0], Point3f::new(1.0, 2.0, 3.0));
        assert_eq!(cloud.points[1], Point3f::new(4.0, 5.0, 6.0));
        assert_eq!(cloud.points[2], Point3f::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn mcap_round_trip() {
        use mcap::{records::MessageHeader, WriteOptions};
        use std::io::Cursor;

        let points = vec![(1.0_f32, 2.0, 3.0), (4.0, 5.0, 6.0)];
        let cdr = build_pc2_cdr(&points);

        // Write an MCAP into an in-memory buffer.
        let mut out: Vec<u8> = Vec::new();
        {
            let mut writer = WriteOptions::new().create(Cursor::new(&mut out)).unwrap();
            let schema_id = writer
                .add_schema("sensor_msgs/msg/PointCloud2", "ros2msg", &[])
                .unwrap();
            let channel_id = writer
                .add_channel(schema_id, "/lidar/points", "cdr", &Default::default())
                .unwrap();
            for (seq, t) in [(0u32, 100u64), (1, 200)] {
                writer
                    .write_to_known_channel(
                        &MessageHeader {
                            channel_id,
                            sequence: seq,
                            log_time: t,
                            publish_time: t,
                        },
                        &cdr,
                    )
                    .unwrap();
            }
            writer.finish().unwrap();
        }

        // Persist to a tempfile.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.mcap");
        std::fs::write(&path, &out).unwrap();

        let reader = McapReader::open(&path).unwrap();
        let collected: Vec<_> = reader
            .messages_on_topic("/lidar/points")
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(collected.len(), 2);
        let cloud = collected[0].as_pointcloud2().unwrap();
        assert_eq!(cloud.len(), 2);
    }

    #[test]
    fn rosbag2_sqlite_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("recording.db3");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, \
                                  serialization_format TEXT, offered_qos_profiles TEXT);
             CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, \
                                    timestamp INTEGER, data BLOB);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO topics (id, name, type, serialization_format, offered_qos_profiles) \
             VALUES (1, '/lidar/points', 'sensor_msgs/msg/PointCloud2', 'cdr', '')",
            [],
        )
        .unwrap();

        let points = vec![(1.0_f32, 2.0, 3.0)];
        let cdr = build_pc2_cdr(&points);
        conn.execute(
            "INSERT INTO messages (topic_id, timestamp, data) VALUES (1, 1000, ?)",
            rusqlite::params![cdr],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (topic_id, timestamp, data) VALUES (1, 2000, ?)",
            rusqlite::params![cdr],
        )
        .unwrap();
        drop(conn);

        // Open via directory:
        let reader = Rosbag2Reader::open(dir.path()).unwrap();
        let topics = reader.topics();
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0].name, "/lidar/points");

        let msgs = reader.read_topic("/lidar/points").unwrap();
        assert_eq!(msgs.len(), 2);
        let cloud = msgs[0].as_pointcloud2().unwrap();
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud.points[0], Point3f::new(1.0, 2.0, 3.0));

        let range = reader.messages_in_range(0, 1500).unwrap();
        assert_eq!(range.len(), 1);
    }
}
