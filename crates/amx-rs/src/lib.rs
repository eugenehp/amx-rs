//! High-Level AMX API
//!
//! This crate provides ergonomic, type-safe abstractions over the low-level AMX
//! instruction set provided by `amx-sys`.
//!
//! # Core Types
//!
//! - [`Matrix<T>`] - 2D matrix operations with generic element types
//! - [`Vector<T>`] - 1D vector operations with iterators
//! - [`MatMulBuilder`] - Fluent API for matrix multiplication
//! - [`ConvBuilder`] - Fluent API for convolution operations
//!
//! # Example
//!
//! ```ignore
//! use amx::Matrix;
//!
//! let a = Matrix::<f32>::zeros(8, 8);
//! let b = Matrix::<f32>::zeros(8, 8);
//! let c = a.matmul(&b);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod error;
pub mod matrix;
pub mod vector;
pub mod algorithms;
pub mod macros;

pub use error::{AmxError, AmxResult};
pub use matrix::Matrix;
pub use vector::Vector;
pub use algorithms::{MatMulBuilder, ConvBuilder};

/// AMX-rs version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
