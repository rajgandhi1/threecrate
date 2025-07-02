//! Error types for 3DCrate

use thiserror::Error;

/// Main error type for 3DCrate operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Invalid data: {0}")]
    InvalidData(String),
    
    #[error("Algorithm error: {0}")]
    Algorithm(String),
    
    #[error("GPU error: {0}")]
    Gpu(String),
    
    #[error("Visualization error: {0}")]
    Visualization(String),
    
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// Result type alias for 3DCrate operations
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "gpu")]
impl From<wgpu::BufferAsyncError> for Error {
    fn from(e: wgpu::BufferAsyncError) -> Self {
        Error::Gpu(e.to_string())
    }
}