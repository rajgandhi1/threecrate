//! Memory-mapped file I/O for high-performance reading of binary PLY/PCD files
//!
//! This module provides OS-gated memory-mapped file support for fast reading of
//! large binary point cloud and mesh files. Falls back to standard buffered I/O
//! on unsupported platforms or when the feature is disabled.

#[cfg(feature = "io-mmap")]
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use threecrate_core::{Result, Error};

/// Memory-mapped file reader for binary data
pub struct MmapReader {
    #[cfg(feature = "io-mmap")]
    mmap: Option<Mmap>,
    #[cfg(feature = "io-mmap")]
    position: usize,
    // Fallback for when mmap is not available
    #[cfg(not(feature = "io-mmap"))]
    _phantom: std::marker::PhantomData<()>,
}

impl MmapReader {
    /// Create a new memory-mapped reader for the given file
    /// 
    /// This will attempt to use memory mapping if available and supported,
    /// otherwise returns None to indicate fallback should be used.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Option<Self>> {
        #[cfg(feature = "io-mmap")]
        {
            // Check if we're on a supported platform
            if !Self::is_supported() {
                return Ok(None);
            }

            let file = File::open(path)?;
            let metadata = file.metadata()?;
            
            // Only use mmap for files larger than a threshold (e.g., 64KB)
            // For small files, regular I/O is often faster due to mmap overhead
            const MIN_MMAP_SIZE: u64 = 64 * 1024; // 64KB
            if metadata.len() < MIN_MMAP_SIZE {
                return Ok(None);
            }

            // Create memory mapping
            let mmap = unsafe {
                match Mmap::map(&file) {
                    Ok(mmap) => mmap,
                    Err(_) => return Ok(None), // Fall back to regular I/O
                }
            };

            Ok(Some(Self {
                mmap: Some(mmap),
                position: 0,
            }))
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            let _ = path; // Suppress unused variable warning
            Ok(None)
        }
    }

    /// Check if memory mapping is supported on this platform
    pub fn is_supported() -> bool {
        #[cfg(feature = "io-mmap")]
        {
            // Memory mapping is generally supported on Unix-like systems and Windows
            cfg!(any(unix, windows))
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            false
        }
    }

    /// Get the total size of the mapped file
    pub fn len(&self) -> usize {
        #[cfg(feature = "io-mmap")]
        {
            self.mmap.as_ref().map(|m| m.len()).unwrap_or(0)
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            0
        }
    }

    /// Get the current position in the file
    pub fn position(&self) -> usize {
        #[cfg(feature = "io-mmap")]
        {
            self.position
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            0
        }
    }

    /// Seek to a specific position in the file
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        #[cfg(feature = "io-mmap")]
        {
            if pos > self.len() {
                return Err(Error::InvalidData("Seek position beyond file end".to_string()));
            }
            self.position = pos;
            Ok(())
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            let _ = pos;
            Err(Error::Unsupported("Memory mapping not available".to_string()))
        }
    }

    /// Read a slice of bytes from the current position
    pub fn read_slice(&mut self, len: usize) -> Result<&[u8]> {
        #[cfg(feature = "io-mmap")]
        {
            let mmap = self.mmap.as_ref().ok_or_else(|| 
                Error::InvalidData("No memory mapping available".to_string()))?;
            
            if self.position + len > mmap.len() {
                return Err(Error::InvalidData("Read beyond file end".to_string()));
            }
            
            let slice = &mmap[self.position..self.position + len];
            self.position += len;
            Ok(slice)
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            let _ = len;
            Err(Error::Unsupported("Memory mapping not available".to_string()))
        }
    }

    /// Read a single byte from the current position
    pub fn read_u8(&mut self) -> Result<u8> {
        let slice = self.read_slice(1)?;
        Ok(slice[0])
    }

    /// Read a little-endian u16 from the current position
    pub fn read_u16_le(&mut self) -> Result<u16> {
        let slice = self.read_slice(2)?;
        Ok(u16::from_le_bytes([slice[0], slice[1]]))
    }

    /// Read a little-endian u32 from the current position
    pub fn read_u32_le(&mut self) -> Result<u32> {
        let slice = self.read_slice(4)?;
        Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// Read a little-endian f32 from the current position
    pub fn read_f32_le(&mut self) -> Result<f32> {
        let slice = self.read_slice(4)?;
        Ok(f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// Read a little-endian f64 from the current position
    pub fn read_f64_le(&mut self) -> Result<f64> {
        let slice = self.read_slice(8)?;
        Ok(f64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3],
            slice[4], slice[5], slice[6], slice[7]
        ]))
    }

    /// Read a big-endian u16 from the current position
    pub fn read_u16_be(&mut self) -> Result<u16> {
        let slice = self.read_slice(2)?;
        Ok(u16::from_be_bytes([slice[0], slice[1]]))
    }

    /// Read a big-endian u32 from the current position
    pub fn read_u32_be(&mut self) -> Result<u32> {
        let slice = self.read_slice(4)?;
        Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// Read a big-endian f32 from the current position
    pub fn read_f32_be(&mut self) -> Result<f32> {
        let slice = self.read_slice(4)?;
        Ok(f32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// Read a big-endian f64 from the current position
    pub fn read_f64_be(&mut self) -> Result<f64> {
        let slice = self.read_slice(8)?;
        Ok(f64::from_be_bytes([
            slice[0], slice[1], slice[2], slice[3],
            slice[4], slice[5], slice[6], slice[7]
        ]))
    }

    /// Skip ahead by the specified number of bytes
    pub fn skip(&mut self, bytes: usize) -> Result<()> {
        #[cfg(feature = "io-mmap")]
        {
            let new_pos = self.position + bytes;
            if new_pos > self.len() {
                return Err(Error::InvalidData("Skip beyond file end".to_string()));
            }
            self.position = new_pos;
            Ok(())
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            let _ = bytes;
            Err(Error::Unsupported("Memory mapping not available".to_string()))
        }
    }

    /// Check if we've reached the end of the file
    pub fn is_at_end(&self) -> bool {
        #[cfg(feature = "io-mmap")]
        {
            self.position >= self.len()
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            true
        }
    }

    /// Get the remaining bytes in the file
    pub fn remaining(&self) -> usize {
        #[cfg(feature = "io-mmap")]
        {
            self.len().saturating_sub(self.position)
        }
        
        #[cfg(not(feature = "io-mmap"))]
        {
            0
        }
    }
}

/// Utility function to check if a file should use memory mapping
/// based on size and platform support
pub fn should_use_mmap<P: AsRef<Path>>(path: P) -> bool {
    if !MmapReader::is_supported() {
        return false;
    }

    // Check file size
    if let Ok(metadata) = std::fs::metadata(path) {
        const MIN_MMAP_SIZE: u64 = 64 * 1024; // 64KB
        metadata.len() >= MIN_MMAP_SIZE
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_availability() {
        // This test just checks that the API doesn't panic
        let supported = MmapReader::is_supported();
        println!("Memory mapping supported: {}", supported);
    }

    #[cfg(feature = "io-mmap")]
    #[test]
    fn test_mmap_basic_operations() -> Result<()> {
        // Create a temporary file with test data
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        
        // Write enough data to trigger mmap (> 64KB)
        for _ in 0..10000 {
            temp_file.write_all(&test_data)?;
        }
        temp_file.flush()?;

        // Test memory mapping
        if let Some(mut reader) = MmapReader::new(temp_file.path())? {
            assert_eq!(reader.position(), 0);
            assert!(reader.len() > 64 * 1024);
            
            // Test reading
            let byte = reader.read_u8()?;
            assert_eq!(byte, 0x01);
            assert_eq!(reader.position(), 1);

            // Test seeking
            reader.seek(4)?;
            assert_eq!(reader.position(), 4);
            
            let byte = reader.read_u8()?;
            assert_eq!(byte, 0x05);
            
            // Test reading multi-byte values
            reader.seek(0)?;
            let u32_val = reader.read_u32_le()?;
            assert_eq!(u32_val, 0x04030201); // little-endian

            reader.seek(0)?;
            let u32_val = reader.read_u32_be()?;
            assert_eq!(u32_val, 0x01020304); // big-endian
        }

        Ok(())
    }

    #[test]
    fn test_should_use_mmap() {
        // Create a small temporary file
        let temp_file = NamedTempFile::new().unwrap();
        
        // Small file should not use mmap
        let should_mmap = should_use_mmap(temp_file.path());
        // This might be false due to size or platform support
        println!("Should use mmap for small file: {}", should_mmap);
    }
}
