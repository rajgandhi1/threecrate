//! Error types for I/O operations

use thiserror::Error;

/// Errors that can occur during I/O operations
#[derive(Error, Debug)]
pub enum IoError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },
    
    #[error("Invalid file format: {format}")]
    InvalidFormat { format: String },
    
    #[error("Parse error: {message}")]
    ParseError { message: String },
    
    #[error("Write error: {message}")]
    WriteError { message: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
} 