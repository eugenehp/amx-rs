//! Error types for high-level AMX API

use core::fmt;

/// Result type for AMX operations
pub type AmxResult<T> = Result<T, AmxError>;

/// High-level AMX API error type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmxError {
    /// Dimension mismatch in operation
    DimensionMismatch { expected: usize, got: usize },
    
    /// Index out of bounds
    IndexOutOfBounds { index: usize, max: usize },
    
    /// Invalid register index
    InvalidRegisterIndex { got: usize },
    
    /// Memory allocation failed
    AllocationFailed,
    
    /// Operation not supported for this type
    UnsupportedType,
    
    /// Register file full
    RegisterFileFull,
}

impl fmt::Display for AmxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AmxError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            AmxError::IndexOutOfBounds { index, max } => {
                write!(f, "index {} out of bounds [0, {})", index, max)
            }
            AmxError::InvalidRegisterIndex { got } => {
                write!(f, "invalid register index {}", got)
            }
            AmxError::AllocationFailed => {
                write!(f, "memory allocation failed")
            }
            AmxError::UnsupportedType => {
                write!(f, "operation not supported for this type")
            }
            AmxError::RegisterFileFull => {
                write!(f, "register file is full")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AmxError {}

