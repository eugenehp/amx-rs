//! Algorithm builders for complex operations

use crate::error::AmxResult;
use crate::matrix::Matrix;

/// Precision mode for matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Default f32 accumulation — fastest, hardware-native precision.
    Fast,
    /// f64 accumulation for scalar fallback — ~2× slower but near-f64 precision.
    /// On AMX hardware, this is equivalent to `Fast` (hardware only supports f32).
    High,
}

impl Default for Precision {
    fn default() -> Self { Precision::Fast }
}

/// Builder for matrix multiplication operations
pub struct MatMulBuilder {
    /// Transpose A before multiplication
    pub transpose_a: bool,
    /// Transpose B before multiplication
    pub transpose_b: bool,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Beta scaling factor for accumulation
    pub beta: f32,
    /// Precision mode
    pub precision: Precision,
}

impl MatMulBuilder {
    /// Create a new matrix multiplication builder with defaults
    pub fn new() -> Self {
        MatMulBuilder {
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
            precision: Precision::Fast,
        }
    }

    /// Set whether to transpose A
    pub fn transpose_a(mut self, transpose: bool) -> Self {
        self.transpose_a = transpose;
        self
    }

    /// Set whether to transpose B
    pub fn transpose_b(mut self, transpose: bool) -> Self {
        self.transpose_b = transpose;
        self
    }

    /// Set alpha scaling factor
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set beta scaling factor (for accumulation)
    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Set precision mode
    pub fn precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// Execute the matrix multiplication using AMX hardware acceleration.
    ///
    /// Computes `C = alpha * A * B`.  If `transpose_a` or `transpose_b` are set
    /// the corresponding matrix is transposed before multiplication.
    ///
    /// When `precision` is `High` and AMX is not available, uses f64
    /// accumulation for better numerical stability.
    pub fn execute(&self, a: &Matrix<f32>, b: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let a_eff = if self.transpose_a { a.transpose()? } else { a.clone() };
        let b_eff = if self.transpose_b { b.transpose()? } else { b.clone() };

        let mut c = match self.precision {
            Precision::Fast => a_eff.matmul(&b_eff)?,
            Precision::High => {
                // Try AMX first (hardware precision), fall back to f64 scalar
                #[cfg(target_arch = "aarch64")]
                {
                    if amx_sys::is_amx_available() {
                        a_eff.matmul(&b_eff)?
                    } else {
                        a_eff.matmul_f64(&b_eff)?
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    a_eff.matmul_f64(&b_eff)?
                }
            }
        };

        // Apply alpha scaling
        if (self.alpha - 1.0).abs() > f32::EPSILON {
            for v in c.as_mut_slice().iter_mut() {
                *v *= self.alpha;
            }
        }

        Ok(c)
    }
}

impl Default for MatMulBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for convolution operations
pub struct ConvBuilder {
    /// Kernel height
    pub kernel_h: usize,
    /// Kernel width
    pub kernel_w: usize,
    /// Stride height
    pub stride_h: usize,
    /// Stride width
    pub stride_w: usize,
    /// Padding height
    pub pad_h: usize,
    /// Padding width
    pub pad_w: usize,
}

impl ConvBuilder {
    /// Create a new convolution builder
    pub fn new(kernel_h: usize, kernel_w: usize) -> Self {
        ConvBuilder {
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        }
    }

    /// Set stride
    pub fn stride(mut self, stride_h: usize, stride_w: usize) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    /// Set padding
    pub fn padding(mut self, pad_h: usize, pad_w: usize) -> Self {
        self.pad_h = pad_h;
        self.pad_w = pad_w;
        self
    }

    /// Get output dimensions given input dimensions
    pub fn output_dims(&self, in_h: usize, in_w: usize) -> (usize, usize) {
        let out_h = (in_h + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1;
        (out_h, out_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_builder() {
        let builder = MatMulBuilder::new()
            .transpose_a(true)
            .alpha(2.0);
        
        assert!(builder.transpose_a);
        assert!(!builder.transpose_b);
        assert_eq!(builder.alpha, 2.0);
    }

    #[test]
    fn test_matmul_builder_precision() {
        let builder = MatMulBuilder::new()
            .precision(Precision::High);
        assert_eq!(builder.precision, Precision::High);
    }

    #[test]
    fn test_matmul_builder_execute_high_precision() {
        let a = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Matrix::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let c = MatMulBuilder::new()
            .precision(Precision::High)
            .execute(&a, &b)
            .unwrap();
        assert_eq!(c.dims(), (2, 2));
        let s = c.as_slice();
        assert!((s[0] - 19.0).abs() < 1e-4);
        assert!((s[1] - 22.0).abs() < 1e-4);
        assert!((s[2] - 43.0).abs() < 1e-4);
        assert!((s[3] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_conv_builder() {
        let conv = ConvBuilder::new(3, 3)
            .stride(2, 2)
            .padding(1, 1);
        
        assert_eq!(conv.kernel_h, 3);
        assert_eq!(conv.stride_h, 2);
        assert_eq!(conv.pad_h, 1);
    }

    #[test]
    fn test_conv_output_dims() {
        let conv = ConvBuilder::new(3, 3)
            .stride(1, 1)
            .padding(1, 1);
        
        let (h, w) = conv.output_dims(224, 224);
        assert_eq!(h, 224);
        assert_eq!(w, 224);
    }
}
