//! Vector type and operations

use alloc::{vec, vec::Vec};
use core::fmt;
use crate::error::{AmxError, AmxResult};

/// Generic vector type with support for AMX operations
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T: Clone + Default> Vector<T> {
    /// Create a new vector of zeros
    pub fn zeros(len: usize) -> AmxResult<Self> {
        let data = vec![T::default(); len];
        Ok(Vector { data })
    }

    /// Create a vector from data
    pub fn from_data(data: Vec<T>) -> Self {
        Vector { data }
    }

    /// Get vector length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> AmxResult<&T> {
        self.data.get(index).ok_or(AmxError::IndexOutOfBounds {
            index,
            max: self.data.len(),
        })
    }

    /// Get mutable element at index
    pub fn get_mut(&mut self, index: usize) -> AmxResult<&mut T> {
        let len = self.data.len();
        self.data.get_mut(index).ok_or(AmxError::IndexOutOfBounds {
            index,
            max: len,
        })
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) -> AmxResult<()> {
        if index >= self.data.len() {
            return Err(AmxError::IndexOutOfBounds {
                index,
                max: self.data.len(),
            });
        }
        self.data[index] = value;
        Ok(())
    }

    /// Get raw data slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable raw data slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert to owned vector
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Create an iterator over elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Create a mutable iterator over elements
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }
}

// ---------------------------------------------------------------------------
// f32 dot product — scalar and AMX implementations
// ---------------------------------------------------------------------------

impl Vector<f32> {
    /// Dot product using the best available backend.
    ///
    /// Uses AMX hardware on `aarch64` Apple Silicon if available, scalar otherwise.
    pub fn dot(&self, other: &Vector<f32>) -> AmxResult<f32> {
        #[cfg(target_arch = "aarch64")]
        {
            if amx_sys::is_amx_available() {
                return self.dot_amx(other);
            }
        }
        self.dot_scalar(other)
    }

    /// Scalar (pure-Rust) dot product.
    ///
    /// Portable, no hardware dependencies.  Useful as a reference baseline
    /// for benchmarking against the AMX path.
    pub fn dot_scalar(&self, other: &Vector<f32>) -> AmxResult<f32> {
        if self.len() != other.len() {
            return Err(AmxError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        Ok(self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum())
    }

    /// AMX-accelerated dot product using `fma32` vector mode.
    ///
    /// Processes 16 f32 lanes at a time, accumulates in Z row 0, then reduces.
    /// Only available on `aarch64`.
    #[cfg(target_arch = "aarch64")]
    pub fn dot_amx(&self, other: &Vector<f32>) -> AmxResult<f32> {
        use amx_sys::*;

        if self.len() != other.len() {
            return Err(AmxError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        const CHUNK: usize = 16;

        #[repr(align(128))]
        struct Buf64([u8; 64]);

        let a = &self.data;
        let b = &other.data;
        let n = a.len();

        unsafe {
            amx_set();

            let zero = Buf64([0u8; 64]);
            amx_ldz(ptr_row_flags(zero.0.as_ptr(), 0, 0));

            for start in (0..n).step_by(CHUNK) {
                let count = CHUNK.min(n - start);

                let mut xb = Buf64([0u8; 64]);
                let mut yb = Buf64([0u8; 64]);
                for i in 0..count {
                    xb.0[i * 4..(i + 1) * 4].copy_from_slice(&a[start + i].to_le_bytes());
                    yb.0[i * 4..(i + 1) * 4].copy_from_slice(&b[start + i].to_le_bytes());
                }

                amx_ldx(ptr_row_flags(xb.0.as_ptr(), 0, 0));
                amx_ldy(ptr_row_flags(yb.0.as_ptr(), 0, 0));
                amx_fma32(1u64 << 63); // vector mode
            }

            let mut zb = Buf64([0u8; 64]);
            amx_stz(ptr_row_flags(zb.0.as_mut_ptr(), 0, 0));
            amx_clr();

            let mut sum = 0.0f32;
            for i in 0..CHUNK {
                let off = i * 4;
                sum += f32::from_le_bytes([
                    zb.0[off], zb.0[off + 1], zb.0[off + 2], zb.0[off + 3],
                ]);
            }

            Ok(sum)
        }
    }
}

impl<T: fmt::Display + Clone + Default> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector<{}>[", self.len())?;
        for (i, item) in self.data.iter().enumerate() {
            write!(f, "{}", item)?;
            if i < self.data.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<T: Clone + Default> Clone for Vector<T> {
    fn clone(&self) -> Self {
        Vector {
            data: self.data.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_dot_scalar_basic() {
        let a = Vector::from_data(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::from_data(vec![4.0f32, 5.0, 6.0]);
        let d = a.dot_scalar(&b).unwrap();
        assert!((d - 32.0).abs() < 1e-4, "got {d}");
    }

    #[test]
    fn test_dot_basic() {
        let a = Vector::from_data(vec![1.0f32, 2.0, 3.0]);
        let b = Vector::from_data(vec![4.0f32, 5.0, 6.0]);
        let d = a.dot(&b).unwrap();
        assert!((d - 32.0).abs() < 1e-4, "got {d}");
    }

    #[test]
    fn test_dot_large() {
        let n = 37;
        let a = Vector::from_data((0..n).map(|i| i as f32).collect());
        let b = Vector::from_data((0..n).map(|i| (i as f32) * 0.5).collect());
        let got = a.dot(&b).unwrap();
        let want = a.dot_scalar(&b).unwrap();
        assert!((got - want).abs() < 1e-1, "got={got}, want={want}");
    }

    #[test]
    fn test_dot_dimension_mismatch() {
        let a = Vector::<f32>::zeros(3).unwrap();
        let b = Vector::<f32>::zeros(5).unwrap();
        assert!(a.dot(&b).is_err());
    }
}
