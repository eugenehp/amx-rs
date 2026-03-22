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
    /// Uses NEON hardware acceleration on `aarch64` Apple Silicon (faster than AMX
    /// for dot products due to lower setup overhead), Kahan-compensated scalar
    /// otherwise for best precision.
    pub fn dot(&self, other: &Vector<f32>) -> AmxResult<f32> {
        #[cfg(target_arch = "aarch64")]
        {
            if amx_sys::is_amx_available() {
                return self.dot_neon(other);
            }
        }
        self.dot_kahan(other)
    }

    /// NEON-accelerated dot product.
    ///
    /// Uses NEON SIMD with 4-way parallel accumulators and 16-float unrolling.
    /// Significantly faster than AMX for dot products due to lower setup overhead.
    ///
    /// Only available on `aarch64`.
    #[cfg(target_arch = "aarch64")]
    pub fn dot_neon(&self, other: &Vector<f32>) -> AmxResult<f32> {
        if self.len() != other.len() {
            return Err(AmxError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        let a = self.as_slice();
        let b = other.as_slice();
        let n = a.len();

        let result = unsafe {
            amx_sys::neon_f32_dot(a.as_ptr(), b.as_ptr(), n as i32)
        };

        Ok(result)
    }

    /// Scalar (pure-Rust) dot product.
    ///
    /// Simple sequential sum — fast, auto-vectorisable, but accumulates
    /// rounding error for long vectors.  Use [`dot_kahan`] or [`dot_f64`]
    /// when precision matters.
    pub fn dot_scalar(&self, other: &Vector<f32>) -> AmxResult<f32> {
        if self.len() != other.len() {
            return Err(AmxError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        Ok(self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum())
    }

    /// Kahan-compensated dot product for improved precision.
    ///
    /// Uses 8 parallel Kahan accumulators so the inner loop remains
    /// vectorisable, then reduces with pairwise summation.  Precision
    /// is close to f64 accumulation while keeping f32 throughput.
    pub fn dot_kahan(&self, other: &Vector<f32>) -> AmxResult<f32> {
        if self.len() != other.len() {
            return Err(AmxError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let n = a.len();

        const LANES: usize = 8;
        let mut sums = [0.0f32; LANES];
        let mut comp = [0.0f32; LANES];
        let full = n / LANES * LANES;

        for i in (0..full).step_by(LANES) {
            for l in 0..LANES {
                let prod = a[i + l] * b[i + l];
                let y = prod - comp[l];
                let t = sums[l] + y;
                comp[l] = (t - sums[l]) - y;
                sums[l] = t;
            }
        }
        // Tail
        for i in full..n {
            let prod = a[i] * b[i];
            let y = prod - comp[0];
            let t = sums[0] + y;
            comp[0] = (t - sums[0]) - y;
            sums[0] = t;
        }
        // Pairwise reduction in f64 for final sum
        let total: f64 = sums.iter().map(|&s| s as f64).sum();
        Ok(total as f32)
    }

    /// High-precision dot product using f64 accumulation throughout.
    ///
    /// Every product is computed in f64 and accumulated in f64.
    /// The result is rounded to f32 at the end.
    pub fn dot_f64(&self, other: &Vector<f32>) -> AmxResult<f32> {
        if self.len() != other.len() {
            return Err(AmxError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let mut sum = 0.0f64;
        for i in 0..a.len() {
            sum += (a[i] as f64) * (b[i] as f64);
        }
        Ok(sum as f32)
    }

    /// AMX-accelerated dot product.
    ///
    /// Delegates to a single C kernel call (`amx_f32_dot`) that handles
    /// AMX set/clr, vectorised accumulation (16 f32 lanes at a time with
    /// 8× unrolling), and pairwise reduction — all without per-chunk FFI
    /// overhead.
    ///
    /// Only available on `aarch64`.
    #[cfg(target_arch = "aarch64")]
    pub fn dot_amx(&self, other: &Vector<f32>) -> AmxResult<f32> {
        if self.len() != other.len() {
            return Err(AmxError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }

        let a = self.as_slice();
        let b = other.as_slice();
        let n = a.len();

        let result = unsafe {
            amx_sys::amx_f32_dot(a.as_ptr(), b.as_ptr(), n as i32)
        };

        Ok(result)
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
        let want = a.dot_f64(&b).unwrap();
        assert!((got - want).abs() < 1e-1, "got={got}, want={want}");
    }

    #[test]
    fn test_dot_dimension_mismatch() {
        let a = Vector::<f32>::zeros(3).unwrap();
        let b = Vector::<f32>::zeros(5).unwrap();
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn test_dot_kahan_precision() {
        // Test that Kahan is more precise than naive for pathological input
        let n = 10000;
        let a = Vector::from_data(vec![1.0f32; n]);
        let b = Vector::from_data((0..n).map(|i| {
            if i % 2 == 0 { 1e-7_f32 } else { -1e-7_f32 }
        }).collect());

        let kahan = a.dot_kahan(&b).unwrap();
        let f64_ref = a.dot_f64(&b).unwrap();

        // Kahan should be close to the f64 reference
        let err = (kahan - f64_ref).abs();
        assert!(err < 1e-6, "kahan={kahan}, f64={f64_ref}, err={err}");
    }

    #[test]
    fn test_dot_f64_precision() {
        let n = 1000;
        let a = Vector::from_data((0..n).map(|i| (i as f32) * 0.001).collect());
        let b = Vector::from_data((0..n).map(|i| ((n - i) as f32) * 0.001).collect());

        let f64_result = a.dot_f64(&b).unwrap();
        let scalar_result = a.dot_scalar(&b).unwrap();

        // Both should be close but f64 should be at least as good
        let diff = (f64_result - scalar_result).abs();
        assert!(diff < 1.0, "results diverge too much: f64={f64_result}, scalar={scalar_result}");
    }
}
