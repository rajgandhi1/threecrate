//! LiDAR sensor raw format support for Velodyne, Ouster, and Livox sensors.
//!
//! Supported formats:
//! - **Velodyne**: KITTI binary (`.bin`) and PCAP (`.pcap`) for VLP-16, VLP-32C, HDL-32E
//! - **Ouster**: PCAP (`.pcap`) for OS0, OS1, OS2 series (configurable beam count)
//! - **Livox**: LVX (`.lvx`) and LVX2 (`.lvx2`) proprietary recording formats
//!
//! # Notes on PCAP files
//! Both Velodyne and Ouster produce PCAP packet capture files, distinguished by
//! their UDP port numbers (Velodyne: 2368, Ouster: 7502). The global `.pcap`
//! registry entry defaults to Velodyne. For Ouster PCAP files use
//! [`OusterPcapReader`] directly.

use crate::registry::PointCloudReader as RegistryPointCloudReader;
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use threecrate_core::{Error, Point3f, PointCloud, Result};

// ---------------------------------------------------------------------------
// Byte-slice helpers
// ---------------------------------------------------------------------------

fn read_u16_le(data: &[u8], off: usize) -> Option<u16> {
    data.get(off..off + 2)
        .and_then(|s| s.try_into().ok())
        .map(u16::from_le_bytes)
}

fn read_u32_le(data: &[u8], off: usize) -> Option<u32> {
    data.get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes)
}

fn read_u64_le(data: &[u8], off: usize) -> Option<u64> {
    data.get(off..off + 8)
        .and_then(|s| s.try_into().ok())
        .map(u64::from_le_bytes)
}

fn read_i32_le(data: &[u8], off: usize) -> Option<i32> {
    data.get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .map(i32::from_le_bytes)
}

fn read_i16_le(data: &[u8], off: usize) -> Option<i16> {
    data.get(off..off + 2)
        .and_then(|s| s.try_into().ok())
        .map(i16::from_le_bytes)
}

fn read_f32_le(data: &[u8], off: usize) -> Option<f32> {
    data.get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .map(f32::from_le_bytes)
}

// ---------------------------------------------------------------------------
// PCAP Parsing
// ---------------------------------------------------------------------------

const PCAP_MAGIC_LE: u32 = 0xd4c3b2a1;
const PCAP_MAGIC_BE: u32 = 0xa1b2c3d4;
const PCAP_MAGIC_NS_LE: u32 = 0x4d3cb2a1;
const PCAP_MAGIC_NS_BE: u32 = 0xa1b23c4d;

/// Reads a PCAP file and returns all UDP payloads directed to `target_port`.
/// Pass `target_port = 0` to collect every UDP payload regardless of port.
fn pcap_extract_udp_payloads(path: &Path, target_port: u16) -> Result<Vec<Vec<u8>>> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    if data.len() < 24 {
        return Err(Error::InvalidData(
            "PCAP file is too small to be valid".to_string(),
        ));
    }

    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let is_le = match magic {
        PCAP_MAGIC_LE | PCAP_MAGIC_NS_LE => true,
        PCAP_MAGIC_BE | PCAP_MAGIC_NS_BE => false,
        _ => {
            return Err(Error::InvalidData(format!(
                "Not a valid PCAP file (magic = 0x{:08x})",
                magic
            )))
        }
    };

    let mut offset = 24usize; // skip 24-byte global header
    let mut payloads = Vec::new();

    while offset + 16 <= data.len() {
        // Per-packet record header (16 bytes): ts_sec, ts_usec, incl_len, orig_len
        let incl_len = if is_le {
            u32::from_le_bytes([
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
            ])
        } else {
            u32::from_be_bytes([
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
            ])
        } as usize;

        offset += 16;
        if offset + incl_len > data.len() {
            break;
        }

        let pkt = &data[offset..offset + incl_len];
        offset += incl_len;

        // Need Ethernet (14) + IPv4 (20) + UDP (8) = 42 bytes minimum
        if pkt.len() < 42 {
            continue;
        }

        // EtherType, handling optional 802.1Q VLAN tag (4 extra bytes)
        let mut etype_off = 12usize;
        let mut ethertype = u16::from_be_bytes([pkt[etype_off], pkt[etype_off + 1]]);
        if ethertype == 0x8100 {
            etype_off += 4;
            if etype_off + 2 > pkt.len() {
                continue;
            }
            ethertype = u16::from_be_bytes([pkt[etype_off], pkt[etype_off + 1]]);
        }
        if ethertype != 0x0800 {
            continue; // IPv4 only
        }

        let ip_start = etype_off + 2;
        if ip_start + 20 > pkt.len() {
            continue;
        }
        let ip = &pkt[ip_start..];
        if (ip[0] >> 4) != 4 {
            continue; // not IPv4
        }
        if ip[9] != 17 {
            continue; // not UDP
        }
        let ihl = ((ip[0] & 0xF) as usize) * 4;

        let udp_start = ip_start + ihl;
        if udp_start + 8 > pkt.len() {
            continue;
        }
        let dst_port = u16::from_be_bytes([pkt[udp_start + 2], pkt[udp_start + 3]]);
        if target_port != 0 && dst_port != target_port {
            continue;
        }

        let payload_start = udp_start + 8;
        if payload_start < pkt.len() {
            payloads.push(pkt[payload_start..].to_vec());
        }
    }

    Ok(payloads)
}

// ---------------------------------------------------------------------------
// Velodyne
// ---------------------------------------------------------------------------

const VELODYNE_DATA_PORT: u16 = 2368;
const VELODYNE_BLOCK_FLAG: u16 = 0xFFEE;
const VELODYNE_BLOCKS_PER_PKT: usize = 12;
const VELODYNE_CHANNELS_PER_BLOCK: usize = 32;

/// Elevation angles (degrees) for VLP-16 by laser index 0-15.
const VLP16_VERT: [f32; 16] = [
    -15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0, -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0,
];

/// Elevation angles (degrees) for HDL-32E by laser index 0-31.
const HDL32E_VERT: [f32; 32] = [
    -30.67, -9.33, -29.33, -8.00, -28.00, -6.67, -26.67, -5.33, -25.33, -4.00, -24.00, -2.67,
    -22.67, -1.33, -21.33, 0.00, -20.00, 1.33, -18.67, 2.67, -17.33, 4.00, -16.00, 5.33, -14.67,
    6.67, -13.33, 8.00, -12.00, 9.33, -10.67, 10.67,
];

/// Velodyne sensor model inferred from the factory byte at packet offset 1205.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelodyneModel {
    VLP16,
    HDL32E,
    VLP32C,
    Unknown,
}

fn detect_velodyne_model(factory_byte: u8) -> VelodyneModel {
    match factory_byte {
        0x22 => VelodyneModel::VLP16,
        0x28 => VelodyneModel::HDL32E,
        0x21 | 0x35 => VelodyneModel::VLP32C,
        _ => VelodyneModel::Unknown,
    }
}

fn decode_velodyne_packet(payload: &[u8], points: &mut Vec<Point3f>) {
    if payload.len() < 1206 {
        return;
    }
    let model = detect_velodyne_model(payload[1205]);

    for blk in 0..VELODYNE_BLOCKS_PER_PKT {
        let base = blk * 100;
        let flag = u16::from_le_bytes([payload[base], payload[base + 1]]);
        if flag != VELODYNE_BLOCK_FLAG {
            continue;
        }

        let az_raw = u16::from_le_bytes([payload[base + 2], payload[base + 3]]);
        let azimuth = az_raw as f32 / 100.0; // hundredths of degrees → degrees

        // Half-step used for VLP-16 second firing-sequence azimuth interpolation
        let half_step = if blk + 1 < VELODYNE_BLOCKS_PER_PKT {
            let nb = (blk + 1) * 100;
            let nf = u16::from_le_bytes([payload[nb], payload[nb + 1]]);
            if nf == VELODYNE_BLOCK_FLAG {
                let na = u16::from_le_bytes([payload[nb + 2], payload[nb + 3]]) as f32 / 100.0;
                let mut step = na - azimuth;
                if step < 0.0 {
                    step += 360.0;
                }
                step / 2.0
            } else {
                1.0
            }
        } else {
            1.0
        };

        for ch in 0..VELODYNE_CHANNELS_PER_BLOCK {
            let cb = base + 4 + ch * 3;
            let dist_raw = u16::from_le_bytes([payload[cb], payload[cb + 1]]);
            if dist_raw == 0 {
                continue;
            }
            let distance = dist_raw as f32 * 0.002; // 2 mm units → meters

            let (el_deg, az_deg) = match model {
                VelodyneModel::VLP16 => {
                    let laser = ch % 16;
                    let firing = ch / 16;
                    let mut az = azimuth;
                    if firing == 1 {
                        az += half_step;
                        if az >= 360.0 {
                            az -= 360.0;
                        }
                    }
                    (VLP16_VERT[laser], az)
                }
                VelodyneModel::HDL32E => (HDL32E_VERT[ch], azimuth),
                // VLP-32C / Unknown: linear elevation approximation
                VelodyneModel::VLP32C | VelodyneModel::Unknown => {
                    (-15.0 + (ch as f32 / 31.0) * 30.0, azimuth)
                }
            };

            let el = el_deg.to_radians();
            let az = az_deg.to_radians();
            let cos_el = el.cos();
            points.push(Point3f::new(
                distance * cos_el * az.sin(),
                distance * cos_el * az.cos(),
                distance * el.sin(),
            ));
        }
    }
}

/// Reads Velodyne LiDAR point clouds from PCAP packet capture files.
///
/// Supports VLP-16, HDL-32E, and VLP-32C sensors. The sensor model is
/// auto-detected from the factory byte embedded in each data packet.
/// Packets are extracted from UDP port 2368 (the Velodyne data port).
pub struct VelodynePcapReader;

impl VelodynePcapReader {
    pub fn read<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let payloads = pcap_extract_udp_payloads(path.as_ref(), VELODYNE_DATA_PORT)?;
        if payloads.is_empty() {
            return Err(Error::InvalidData(
                "No Velodyne data packets found on port 2368".to_string(),
            ));
        }
        let mut points = Vec::new();
        for p in &payloads {
            decode_velodyne_packet(p, &mut points);
        }
        Ok(PointCloud::from_points(points))
    }
}

/// Reads Velodyne LiDAR point clouds in KITTI binary format (`.bin`).
///
/// Each point is stored as four consecutive `f32` values: `x`, `y`, `z`
/// (in metres) followed by `intensity` (0.0–1.0). The intensity channel is
/// discarded when converting to [`PointCloud<Point3f>`].
pub struct VelodyneKittiBinReader;

impl VelodyneKittiBinReader {
    pub fn read<P: AsRef<Path>>(path: P) -> Result<PointCloud<Point3f>> {
        let mut file = File::open(path.as_ref())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() % 16 != 0 {
            return Err(Error::InvalidData(format!(
                "Velodyne KITTI binary: file size {} is not a multiple of 16 bytes",
                data.len()
            )));
        }

        let num_points = data.len() / 16;
        let mut points = Vec::with_capacity(num_points);
        let mut cur = Cursor::new(&data);

        for _ in 0..num_points {
            let x = cur.read_f32::<LittleEndian>()?;
            let y = cur.read_f32::<LittleEndian>()?;
            let z = cur.read_f32::<LittleEndian>()?;
            let _ = cur.read_f32::<LittleEndian>()?; // intensity, not needed
            points.push(Point3f::new(x, y, z));
        }

        Ok(PointCloud::from_points(points))
    }
}

// ---------------------------------------------------------------------------
// Ouster
// ---------------------------------------------------------------------------

const OUSTER_LIDAR_PORT: u16 = 7502;
const OUSTER_ENCODER_TICKS: f32 = 90112.0; // encoder ticks per full revolution
const OUSTER_COLUMNS_PER_PKT: usize = 16;
const OUSTER_COL_HEADER_BYTES: usize = 16; // timestamp(8)+meas_id(2)+frame_id(2)+encoder(4)
const OUSTER_COL_FOOTER_BYTES: usize = 4; // status u32
const OUSTER_CHANNEL_BYTES: usize = 12;

/// Default beam altitude angles (degrees) for Ouster OS1-64, top to bottom.
const OS1_64_ALTITUDES: [f32; 64] = [
    16.611, 16.084, 15.557, 15.029, 14.502, 13.975, 13.447, 12.920, 12.393, 11.865, 11.338, 10.811,
    10.283, 9.756, 9.229, 8.701, 8.174, 7.647, 7.119, 6.592, 6.065, 5.537, 5.010, 4.483, 3.955,
    3.428, 2.901, 2.373, 1.846, 1.319, 0.791, 0.264, -0.264, -0.791, -1.319, -1.846, -2.373,
    -2.901, -3.428, -3.955, -4.483, -5.010, -5.537, -6.065, -6.592, -7.119, -7.647, -8.174, -8.701,
    -9.229, -9.756, -10.283, -10.811, -11.338, -11.865, -12.393, -12.920, -13.447, -13.975,
    -14.502, -15.029, -15.557, -16.084, -16.611,
];

/// Reads Ouster OS-series LiDAR data from PCAP packet capture files.
///
/// Uses the LEGACY UDP packet profile (port 7502). The sensor parameters
/// (beam count, columns per packet) must match the recording configuration.
/// Defaults to OS1-64 at 1024 azimuth resolution.
///
/// For sensors not covered by the defaults, build a custom reader:
/// ```rust,ignore
/// let reader = OusterPcapReader {
///     columns_per_packet: 16,
///     pixels_per_column: 128,
///     beam_altitudes: my_beam_angles,
///     lidar_port: 7502,
/// };
/// ```
pub struct OusterPcapReader {
    /// Azimuth columns per UDP packet (16 at 1024 resolution).
    pub columns_per_packet: usize,
    /// Beams (channels) per column: 16, 32, 64, or 128.
    pub pixels_per_column: usize,
    /// Beam altitude angles in degrees, one entry per channel index.
    pub beam_altitudes: Vec<f32>,
    /// UDP destination port for lidar data packets (default: 7502).
    pub lidar_port: u16,
}

impl Default for OusterPcapReader {
    fn default() -> Self {
        Self {
            columns_per_packet: OUSTER_COLUMNS_PER_PKT,
            pixels_per_column: 64,
            beam_altitudes: OS1_64_ALTITUDES.to_vec(),
            lidar_port: OUSTER_LIDAR_PORT,
        }
    }
}

impl OusterPcapReader {
    /// Create a reader for OS0-64, OS1-64, or OS2-64 (default configuration).
    pub fn os1_64() -> Self {
        Self::default()
    }

    /// Create a reader with linearly-spaced altitude angles for a 128-beam sensor.
    pub fn os_128() -> Self {
        let altitudes = (0..128).map(|i| 22.5 - (i as f32 / 127.0) * 45.0).collect();
        Self {
            columns_per_packet: OUSTER_COLUMNS_PER_PKT,
            pixels_per_column: 128,
            beam_altitudes: altitudes,
            lidar_port: OUSTER_LIDAR_PORT,
        }
    }

    /// Read all point cloud data from an Ouster PCAP file.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<PointCloud<Point3f>> {
        let payloads = pcap_extract_udp_payloads(path.as_ref(), self.lidar_port)?;
        if payloads.is_empty() {
            return Err(Error::InvalidData(format!(
                "No Ouster lidar packets found on port {}",
                self.lidar_port
            )));
        }

        let col_size = OUSTER_COL_HEADER_BYTES
            + self.pixels_per_column * OUSTER_CHANNEL_BYTES
            + OUSTER_COL_FOOTER_BYTES;
        let expected_pkt_size = self.columns_per_packet * col_size;

        let mut points = Vec::new();

        for payload in &payloads {
            if payload.len() < expected_pkt_size {
                continue;
            }

            for col in 0..self.columns_per_packet {
                let col_base = col * col_size;

                // encoder_count is at bytes 12-15 of the column header
                let encoder = match read_u32_le(payload, col_base + 12) {
                    Some(v) => v,
                    None => continue,
                };

                let azimuth = (encoder as f32 / OUSTER_ENCODER_TICKS) * std::f32::consts::TAU;
                let cos_az = azimuth.cos();
                let sin_az = azimuth.sin();

                for ch in 0..self.pixels_per_column {
                    let ch_base = col_base + OUSTER_COL_HEADER_BYTES + ch * OUSTER_CHANNEL_BYTES;

                    // Lower 20 bits of the first u32 give range in mm
                    let raw = match read_u32_le(payload, ch_base) {
                        Some(v) => v,
                        None => break,
                    };
                    let range_mm = raw & 0x000F_FFFF;
                    if range_mm == 0 {
                        continue;
                    }

                    let r = range_mm as f32 / 1000.0;
                    let alt = self.beam_altitudes[ch].to_radians();
                    let cos_alt = alt.cos();

                    points.push(Point3f::new(
                        r * cos_alt * cos_az,
                        r * cos_alt * sin_az,
                        r * alt.sin(),
                    ));
                }
            }
        }

        Ok(PointCloud::from_points(points))
    }
}

// ---------------------------------------------------------------------------
// Livox LVX (v1)
// ---------------------------------------------------------------------------

const LVX_SIGNATURE: &[u8] = b"livox_tech";
const LVX_MAGIC_CODE: u32 = 0xAC0EA767;
const LVX_FILE_HEADER_SIZE: usize = 24;
const LVX_PRIV_HEADER_BASE: usize = 5; // frame_duration(4) + device_count(1)
const LVX_DEVICE_INFO_SIZE: usize = 59;
const LVX_FRAME_HEADER_SIZE: usize = 24; // current_offset(8) + next_offset(8) + frame_index(8)
const LVX_PKG_HEADER_SIZE: usize = 27;

// Package header field offsets (within a package):
// [0]  device_index : u8
// [1]  version      : u8
// [2]  slot_id      : u8
// [3]  lidar_id     : u8
// [4]  reserved     : u8
// [5]  error_code   : u32 (LE)
// [9]  timestamp_type : u8
// [10] data_type    : u8
// [11] timestamp    : u64 (LE)
// [19] udp_counter  : u16 (LE)
// [21] length       : u16 (LE) — bytes of point data that follow
// [23] frame_counter: u32 (LE)
// total: 27 bytes
const LVX_PKG_DATA_TYPE_OFF: usize = 10;
const LVX_PKG_LENGTH_OFF: usize = 21;

/// Livox LVX1 point data type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LvxDataType {
    CartesianInt32 = 1, // x, y, z: i32 (mm) + reflectivity: u8 + tag: u8 = 14 bytes
    Spherical = 2, // depth: u32 (mm) + theta: u16 + phi: u16 + reflectivity: u8 + tag: u8 = 10 bytes
    CartesianFloat = 3, // x, y, z: f32 (m) + intensity: u8 + tag: u8 = 14 bytes
}

impl LvxDataType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::CartesianInt32),
            2 => Some(Self::Spherical),
            3 => Some(Self::CartesianFloat),
            _ => None,
        }
    }

    fn point_size(self) -> usize {
        match self {
            Self::CartesianInt32 | Self::CartesianFloat => 14,
            Self::Spherical => 10,
        }
    }
}

fn lvx_spherical_to_xyz(depth_mm: u32, theta_cdeg: u16, phi_cdeg: u16) -> Point3f {
    // theta = zenith angle (0° = up), phi = azimuth angle; both in hundredths of degrees
    let r = depth_mm as f32 / 1000.0;
    let theta = (theta_cdeg as f32 * 0.01_f32).to_radians();
    let phi = (phi_cdeg as f32 * 0.01_f32).to_radians();
    let sin_t = theta.sin();
    Point3f::new(
        r * sin_t * phi.cos(),
        r * sin_t * phi.sin(),
        r * theta.cos(),
    )
}

fn parse_lvx_point(dtype: LvxDataType, pt: &[u8]) -> Option<Point3f> {
    match dtype {
        LvxDataType::CartesianInt32 => {
            let x = read_i32_le(pt, 0)? as f32 / 1000.0;
            let y = read_i32_le(pt, 4)? as f32 / 1000.0;
            let z = read_i32_le(pt, 8)? as f32 / 1000.0;
            Some(Point3f::new(x, y, z))
        }
        LvxDataType::Spherical => {
            let depth = read_u32_le(pt, 0)?;
            let theta = read_u16_le(pt, 4)?;
            let phi = read_u16_le(pt, 6)?;
            Some(lvx_spherical_to_xyz(depth, theta, phi))
        }
        LvxDataType::CartesianFloat => {
            let x = read_f32_le(pt, 0)?;
            let y = read_f32_le(pt, 4)?;
            let z = read_f32_le(pt, 8)?;
            Some(Point3f::new(x, y, z))
        }
    }
}

/// Reads Livox LiDAR data from LVX (`.lvx`) recording files.
///
/// The LVX format is Livox's proprietary format produced by Livox Viewer and
/// their ROS driver. Supports cartesian (int32 mm, float32 m) and spherical
/// point layouts.
pub struct LivoxLvxReader;

impl LivoxLvxReader {
    pub fn new() -> Self {
        Self
    }

    /// Read all point cloud data from an LVX file.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<PointCloud<Point3f>> {
        let mut file = File::open(path.as_ref())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() < LVX_FILE_HEADER_SIZE {
            return Err(Error::InvalidData("LVX file is too small".to_string()));
        }
        if !data.starts_with(LVX_SIGNATURE) {
            return Err(Error::InvalidData(
                "Not a valid LVX file: missing 'livox_tech' signature".to_string(),
            ));
        }

        let magic = match read_u32_le(&data, 20) {
            Some(m) => m,
            None => return Err(Error::InvalidData("LVX header truncated".to_string())),
        };
        if magic != LVX_MAGIC_CODE {
            return Err(Error::InvalidData(format!(
                "Unexpected LVX magic code: 0x{:08x}",
                magic
            )));
        }

        // Private header: frame_duration(4) + device_count(1) + device_infos
        let ph = LVX_FILE_HEADER_SIZE;
        if ph + LVX_PRIV_HEADER_BASE > data.len() {
            return Err(Error::InvalidData(
                "LVX private header truncated".to_string(),
            ));
        }
        let device_count = data[ph + 4] as usize;
        let data_block_start = ph + LVX_PRIV_HEADER_BASE + device_count * LVX_DEVICE_INFO_SIZE;

        if data_block_start > data.len() {
            return Err(Error::InvalidData(
                "LVX device info section truncated".to_string(),
            ));
        }

        let mut points = Vec::new();
        let mut pos = data_block_start;

        while pos + LVX_FRAME_HEADER_SIZE <= data.len() {
            let next_offset = match read_u64_le(&data, pos + 8) {
                Some(v) => v as usize,
                None => break,
            };

            let frame_end = if next_offset == 0 {
                data.len()
            } else {
                (data_block_start + next_offset).min(data.len())
            };

            let mut pkg = pos + LVX_FRAME_HEADER_SIZE;

            while pkg + LVX_PKG_HEADER_SIZE <= frame_end {
                let data_type_byte = data[pkg + LVX_PKG_DATA_TYPE_OFF];
                let length = match read_u16_le(&data, pkg + LVX_PKG_LENGTH_OFF) {
                    Some(v) => v as usize,
                    None => break,
                };

                let body_start = pkg + LVX_PKG_HEADER_SIZE;
                let body_end = body_start + length;
                if body_end > data.len() {
                    break;
                }

                if let Some(dtype) = LvxDataType::from_u8(data_type_byte) {
                    let psz = dtype.point_size();
                    if psz > 0 && length >= psz {
                        let n = length / psz;
                        for i in 0..n {
                            let p0 = body_start + i * psz;
                            if p0 + psz > body_end {
                                break;
                            }
                            if let Some(pt) = parse_lvx_point(dtype, &data[p0..p0 + psz]) {
                                points.push(pt);
                            }
                        }
                    }
                }

                pkg = body_end;
            }

            if next_offset == 0 || data_block_start + next_offset <= pos {
                break;
            }
            pos = data_block_start + next_offset;
        }

        Ok(PointCloud::from_points(points))
    }
}

// ---------------------------------------------------------------------------
// Livox LVX2
// ---------------------------------------------------------------------------

const LVX2_MAGIC_CODE: u32 = 0x20200903;

// LVX2 public header layout:
// [0]  magic_code    : u32
// [4]  version       : [u8; 4]
// [8]  header_size   : u32  — byte offset from file start where device infos begin
// [12] file_size     : u64
// [20] frame_duration: u32 (ms)
// [24] device_count  : u8
// [25] data_type     : u8
// (remaining bytes up to header_size are reserved)
const LVX2_DEVICE_COUNT_OFF: usize = 24;
const LVX2_HEADER_SIZE_OFF: usize = 8;

// LVX2 device info: lidar_sn(16) + extrinsic_enable(1) + roll(4)+pitch(4)+yaw(4)+x(4)+y(4)+z(4) = 41 bytes
const LVX2_DEVICE_INFO_SIZE: usize = 41;
const LVX2_FRAME_HEADER_SIZE: usize = 24;

// LVX2 packet header layout (11 bytes):
// [0]  device_index: u8
// [1]  lidar_type  : u8
// [2]  point_num   : u32 (LE)
// [6]  data_type   : u8
// [7]  data_length : u32 (LE)
const LVX2_PKT_HEADER_SIZE: usize = 11;
const LVX2_PKT_DATA_TYPE_OFF: usize = 6;
const LVX2_PKT_DATA_LENGTH_OFF: usize = 7;

/// Livox LVX2 point data type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lvx2DataType {
    CartesianInt16 = 0, // x, y, z: i16 (10 mm units) + reflectivity: u8 + tag: u8 = 8 bytes
    CartesianInt32 = 1, // x, y, z: i32 (mm) + reflectivity: u8 + tag: u8 = 14 bytes
    Spherical = 2, // depth: u32 (mm) + theta: u16 + phi: u16 + reflectivity: u8 + tag: u8 = 10 bytes
}

impl Lvx2DataType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::CartesianInt16),
            1 => Some(Self::CartesianInt32),
            2 => Some(Self::Spherical),
            _ => None,
        }
    }

    fn point_size(self) -> usize {
        match self {
            Self::CartesianInt16 => 8,
            Self::CartesianInt32 => 14,
            Self::Spherical => 10,
        }
    }
}

fn parse_lvx2_point(dtype: Lvx2DataType, pt: &[u8]) -> Option<Point3f> {
    match dtype {
        Lvx2DataType::CartesianInt16 => {
            let x = read_i16_le(pt, 0)? as f32 * 0.01; // 10 mm → m
            let y = read_i16_le(pt, 2)? as f32 * 0.01;
            let z = read_i16_le(pt, 4)? as f32 * 0.01;
            Some(Point3f::new(x, y, z))
        }
        Lvx2DataType::CartesianInt32 => {
            let x = read_i32_le(pt, 0)? as f32 / 1000.0;
            let y = read_i32_le(pt, 4)? as f32 / 1000.0;
            let z = read_i32_le(pt, 8)? as f32 / 1000.0;
            Some(Point3f::new(x, y, z))
        }
        Lvx2DataType::Spherical => {
            let depth = read_u32_le(pt, 0)?;
            let theta = read_u16_le(pt, 4)?;
            let phi = read_u16_le(pt, 6)?;
            Some(lvx_spherical_to_xyz(depth, theta, phi))
        }
    }
}

/// Reads Livox LiDAR data from LVX2 (`.lvx2`) recording files.
///
/// LVX2 is the updated Livox recording format used with Avia, HAP, and
/// Mid-360 sensors. It differs from LVX v1 in its file header layout and
/// packet structure.
pub struct LivoxLvx2Reader;

impl LivoxLvx2Reader {
    pub fn new() -> Self {
        Self
    }

    /// Read all point cloud data from an LVX2 file.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<PointCloud<Point3f>> {
        let mut file = File::open(path.as_ref())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() < 28 {
            return Err(Error::InvalidData("LVX2 file is too small".to_string()));
        }

        let magic = match read_u32_le(&data, 0) {
            Some(m) => m,
            None => return Err(Error::InvalidData("LVX2 header truncated".to_string())),
        };
        if magic != LVX2_MAGIC_CODE {
            return Err(Error::InvalidData(format!(
                "Not a valid LVX2 file (magic = 0x{:08x})",
                magic
            )));
        }

        let header_size = match read_u32_le(&data, LVX2_HEADER_SIZE_OFF) {
            Some(v) => v as usize,
            None => {
                return Err(Error::InvalidData(
                    "LVX2 header_size field missing".to_string(),
                ))
            }
        };

        let device_count = if LVX2_DEVICE_COUNT_OFF < data.len() {
            data[LVX2_DEVICE_COUNT_OFF] as usize
        } else {
            return Err(Error::InvalidData(
                "LVX2 device_count field missing".to_string(),
            ));
        };

        let data_block_start = header_size + device_count * LVX2_DEVICE_INFO_SIZE;
        if data_block_start > data.len() {
            return Err(Error::InvalidData(
                "LVX2 device info section extends past end of file".to_string(),
            ));
        }

        let mut points = Vec::new();
        let mut pos = data_block_start;

        while pos + LVX2_FRAME_HEADER_SIZE <= data.len() {
            let next_offset = match read_u64_le(&data, pos + 8) {
                Some(v) => v as usize,
                None => break,
            };

            let frame_end = if next_offset == 0 {
                data.len()
            } else {
                (data_block_start + next_offset).min(data.len())
            };

            let mut pkg = pos + LVX2_FRAME_HEADER_SIZE;

            while pkg + LVX2_PKT_HEADER_SIZE <= frame_end {
                let data_type_byte = data[pkg + LVX2_PKT_DATA_TYPE_OFF];
                let data_length = match read_u32_le(&data, pkg + LVX2_PKT_DATA_LENGTH_OFF) {
                    Some(v) => v as usize,
                    None => break,
                };

                let body_start = pkg + LVX2_PKT_HEADER_SIZE;
                let body_end = body_start + data_length;
                if body_end > data.len() {
                    break;
                }

                if let Some(dtype) = Lvx2DataType::from_u8(data_type_byte) {
                    let psz = dtype.point_size();
                    if psz > 0 && data_length >= psz {
                        let n = data_length / psz;
                        for i in 0..n {
                            let p0 = body_start + i * psz;
                            if p0 + psz > body_end {
                                break;
                            }
                            if let Some(pt) = parse_lvx2_point(dtype, &data[p0..p0 + psz]) {
                                points.push(pt);
                            }
                        }
                    }
                }

                pkg = body_end;
            }

            if next_offset == 0 || data_block_start + next_offset <= pos {
                break;
            }
            pos = data_block_start + next_offset;
        }

        Ok(PointCloud::from_points(points))
    }
}

// ---------------------------------------------------------------------------
// IO Registry Adapters
// ---------------------------------------------------------------------------

/// Registry adapter: Velodyne KITTI binary (`.bin`).
pub struct VelodyneBinRegistryReader;

impl RegistryPointCloudReader for VelodyneBinRegistryReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        VelodyneKittiBinReader::read(path)
    }

    fn can_read(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("bin"))
            .unwrap_or(false)
    }

    fn format_name(&self) -> &'static str {
        "velodyne-bin"
    }
}

/// Registry adapter: Velodyne PCAP (`.pcap`).
pub struct VelodynePcapRegistryReader;

impl RegistryPointCloudReader for VelodynePcapRegistryReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        VelodynePcapReader::read(path)
    }

    fn can_read(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("pcap"))
            .unwrap_or(false)
    }

    fn format_name(&self) -> &'static str {
        "velodyne-pcap"
    }
}

/// Registry adapter: Livox LVX (`.lvx`).
pub struct LivoxLvxRegistryReader;

impl RegistryPointCloudReader for LivoxLvxRegistryReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        LivoxLvxReader::new().read(path)
    }

    fn can_read(&self, path: &Path) -> bool {
        if let Ok(mut f) = File::open(path) {
            let mut buf = [0u8; 10];
            if f.read_exact(&mut buf).is_ok() {
                return buf.starts_with(LVX_SIGNATURE);
            }
        }
        false
    }

    fn format_name(&self) -> &'static str {
        "livox-lvx"
    }
}

/// Registry adapter: Livox LVX2 (`.lvx2`).
pub struct LivoxLvx2RegistryReader;

impl RegistryPointCloudReader for LivoxLvx2RegistryReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        LivoxLvx2Reader::new().read(path)
    }

    fn can_read(&self, path: &Path) -> bool {
        if let Ok(mut f) = File::open(path) {
            let mut buf = [0u8; 4];
            if f.read_exact(&mut buf).is_ok() {
                return u32::from_le_bytes(buf) == LVX2_MAGIC_CODE;
            }
        }
        false
    }

    fn format_name(&self) -> &'static str {
        "livox-lvx2"
    }
}
