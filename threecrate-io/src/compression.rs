//! Draco point cloud compression and decompression.
//!
//! Wraps the Google Draco codec (`spatial_codec_draco`) behind the `compression`
//! feature flag.  Draco uses lossy quantization so decoded positions will not
//! be bit-for-bit identical to the originals; increase `quantization_bits` for
//! higher fidelity at the cost of a larger compressed payload.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use threecrate_io::compression::{draco_encode, draco_decode, DracoConfig};
//! use threecrate_core::{PointCloud, Point3f};
//!
//! let cloud: PointCloud<Point3f> = PointCloud::new(); // your cloud here
//! let config = DracoConfig { quantization_bits: 14, compression_level: 7, encode_colors: false };
//! let bytes: Vec<u8> = draco_encode(&cloud, config).unwrap();
//! let restored: PointCloud<Point3f> = draco_decode(&bytes).unwrap();
//! ```

use std::fs;
use std::path::Path;

use spatial_codec_draco::{
    decode_draco, encode_draco_with_config, EncodeConfig, PointCloudEncodingMethod,
};
use threecrate_core::{Error, Point3f, PointCloud, Result};

use crate::registry;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Draco point cloud compression.
#[derive(Debug, Clone)]
pub struct DracoConfig {
    /// Position quantization precision in bits (1–31).  Higher means more
    /// precision and a larger compressed payload.  14 is a good default.
    pub quantization_bits: u32,
    /// Compression effort (0 = fastest / least compression, 10 = slowest /
    /// best compression).  This is the inverse of Draco's internal
    /// `encoding_speed` parameter.
    pub compression_level: u8,
    /// When `true`, RGB color attributes are preserved in the stream.
    /// `PointCloud<Point3f>` has no color, so this field is reserved for
    /// future use with `ColoredPoint3f` clouds and currently has no effect.
    pub encode_colors: bool,
}

impl Default for DracoConfig {
    fn default() -> Self {
        Self {
            quantization_bits: 14,
            compression_level: 7,
            encode_colors: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Core encode / decode
// ---------------------------------------------------------------------------

/// Compress a point cloud to Draco-encoded bytes.
///
/// Returns `Err` if the cloud is empty or the encoder fails.
pub fn draco_encode(cloud: &PointCloud<Point3f>, config: DracoConfig) -> Result<Vec<u8>> {
    if cloud.is_empty() {
        return Err(Error::InvalidData(
            "cannot encode an empty point cloud".into(),
        ));
    }

    let coords: Vec<[f32; 3]> = cloud.points.iter().map(|p| [p.x, p.y, p.z]).collect();
    // spatial_codec_draco requires colours even for position-only clouds.
    let colors: Vec<[u8; 3]> = vec![[0u8; 3]; coords.len()];

    // DracoConfig.compression_level uses "higher = better"; Draco's
    // encoding_speed uses "higher = faster (less compression)".
    let encoding_speed = 10u8.saturating_sub(config.compression_level);

    let encode_config = EncodeConfig {
        position_quantization_bits: config.quantization_bits,
        color_quantization_bits: 8,
        encoding_speed,
        decoding_speed: 5,
    };

    // KdTree gives ~5% better compression but reorders points; Sequential
    // preserves insertion order, which callers generally expect.
    encode_draco_with_config(&coords, &colors, PointCloudEncodingMethod::Sequential, &encode_config)
        .map_err(|e| Error::InvalidData(format!("Draco encode failed: {e}")))
}

/// Decompress Draco-encoded bytes back into a point cloud.
///
/// Returns `Err` if the buffer is empty or the decoder fails.
pub fn draco_decode(compressed: &[u8]) -> Result<PointCloud<Point3f>> {
    if compressed.is_empty() {
        return Err(Error::InvalidData(
            "cannot decode an empty buffer".into(),
        ));
    }

    let (coords, _colors) = decode_draco(compressed)
        .map_err(|e| Error::InvalidData(format!("Draco decode failed: {e}")))?;

    // spatial_codec_draco returns a flat interleaved Vec<f32>:
    // [x0, y0, z0, x1, y1, z1, …]
    if coords.len() % 3 != 0 || coords.is_empty() {
        return Err(Error::InvalidData(
            format!("Draco decode returned {} floats, expected a multiple of 3", coords.len()),
        ));
    }
    let points = coords
        .chunks(3)
        .map(|c| Point3f::new(c[0], c[1], c[2]))
        .collect();

    Ok(PointCloud::from_points(points))
}

// ---------------------------------------------------------------------------
// Registry handlers — read / write .drc files
// ---------------------------------------------------------------------------

/// Registry handler for reading `.drc` files.
pub struct DracoReader;

/// Registry handler for writing `.drc` files.
pub struct DracoWriter;

impl registry::PointCloudReader for DracoReader {
    fn read_point_cloud(&self, path: &Path) -> Result<PointCloud<Point3f>> {
        let bytes = fs::read(path).map_err(Error::Io)?;
        draco_decode(&bytes)
    }

    fn can_read(&self, path: &Path) -> bool {
        path.extension().and_then(|e| e.to_str()) == Some("drc")
    }

    fn format_name(&self) -> &'static str {
        "draco"
    }
}

impl registry::PointCloudWriter for DracoWriter {
    fn write_point_cloud(&self, cloud: &PointCloud<Point3f>, path: &Path) -> Result<()> {
        let bytes = draco_encode(cloud, DracoConfig::default())?;
        fs::write(path, bytes).map_err(Error::Io)
    }

    fn format_name(&self) -> &'static str {
        "draco"
    }
}

// ---------------------------------------------------------------------------
// Streaming compressor
// ---------------------------------------------------------------------------

/// Accumulates point chunks and produces a single Draco-compressed `Vec<u8>`
/// on [`finalize`].
///
/// Mirrors the [`StreamingPipeline`] interface from `threecrate-algorithms`
/// without requiring a cross-crate dependency.  Use with
/// [`read_point_cloud_iter`] to compress large files in bounded memory:
///
/// ```rust,no_run
/// use threecrate_io::{read_point_cloud_iter, compression::{DracoCompressorPipeline, DracoConfig}};
///
/// let mut compressor = DracoCompressorPipeline::new(DracoConfig::default());
/// let iter = read_point_cloud_iter("large.ply", Some(4096)).unwrap();
/// for result in iter {
///     compressor.process_chunk(&[result.unwrap()]).unwrap();
/// }
/// let compressed: Vec<u8> = compressor.finalize().unwrap();
/// ```
///
/// [`read_point_cloud_iter`]: crate::read_point_cloud_iter
/// [`StreamingPipeline`]: https://docs.rs/threecrate-algorithms/latest/threecrate_algorithms/streaming/trait.StreamingPipeline.html
pub struct DracoCompressorPipeline {
    config: DracoConfig,
    buffer: Vec<Point3f>,
}

impl DracoCompressorPipeline {
    /// Create a new compressor with the given configuration.
    pub fn new(config: DracoConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
        }
    }

    /// Ingest one chunk of points.  May be called multiple times.
    pub fn process_chunk(&mut self, chunk: &[Point3f]) -> Result<()> {
        self.buffer.extend_from_slice(chunk);
        Ok(())
    }

    /// Compress all accumulated points and return the encoded bytes.
    pub fn finalize(self) -> Result<Vec<u8>> {
        let cloud = PointCloud::from_points(self.buffer);
        draco_encode(&cloud, self.config)
    }

    /// Current number of bytes held in the in-memory point buffer.
    pub fn memory_bytes(&self) -> usize {
        self.buffer.len() * std::mem::size_of::<Point3f>()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use threecrate_core::Point3f;

    fn sample_cloud(n: usize) -> PointCloud<Point3f> {
        let points = (0..n)
            .map(|i| Point3f::new(i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3))
            .collect();
        PointCloud::from_points(points)
    }

    #[test]
    fn roundtrip_basic() {
        let cloud = sample_cloud(100);
        let config = DracoConfig::default();
        let bytes = draco_encode(&cloud, config).expect("encode failed");
        assert!(!bytes.is_empty());
        let decoded = draco_decode(&bytes).expect("decode failed");
        assert_eq!(decoded.len(), cloud.len());
        // Lossy codec: positions are close but not exact.
        for (orig, dec) in cloud.points.iter().zip(decoded.points.iter()) {
            assert!((orig.x - dec.x).abs() < 0.01, "x drift too large");
            assert!((orig.y - dec.y).abs() < 0.01, "y drift too large");
            assert!((orig.z - dec.z).abs() < 0.01, "z drift too large");
        }
    }

    #[test]
    fn encode_empty_cloud_returns_error() {
        let cloud: PointCloud<Point3f> = PointCloud::new();
        assert!(draco_encode(&cloud, DracoConfig::default()).is_err());
    }

    #[test]
    fn decode_empty_buffer_returns_error() {
        assert!(draco_decode(&[]).is_err());
    }

    #[test]
    fn streaming_compressor_roundtrip() {
        let cloud = sample_cloud(50);
        let mut compressor = DracoCompressorPipeline::new(DracoConfig::default());
        compressor.process_chunk(&cloud.points).expect("process_chunk failed");
        let bytes = compressor.finalize().expect("finalize failed");
        let decoded = draco_decode(&bytes).expect("decode failed");
        assert_eq!(decoded.len(), cloud.len());
    }
}
