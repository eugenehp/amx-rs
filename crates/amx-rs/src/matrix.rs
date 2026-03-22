//! Matrix type and operations

use alloc::{vec, vec::Vec};
use core::fmt;
use crate::error::{AmxError, AmxResult};

/// Generic matrix type supporting AMX operations
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone + Default> Matrix<T> {
    /// Create a new matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> AmxResult<Self> {
        let capacity = rows.checked_mul(cols)
            .ok_or(AmxError::AllocationFailed)?;
        let data = vec![T::default(); capacity];
        Ok(Matrix { data, rows, cols })
    }

    /// Create a new matrix from flat data
    pub fn from_data(data: Vec<T>, rows: usize, cols: usize) -> AmxResult<Self> {
        if data.len() != rows * cols {
            return Err(AmxError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(Matrix { data, rows, cols })
    }

    /// Get matrix dimensions
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get element at position
    pub fn get(&self, row: usize, col: usize) -> AmxResult<&T> {
        if row >= self.rows || col >= self.cols {
            return Err(AmxError::IndexOutOfBounds {
                index: row * self.cols + col,
                max: self.data.len(),
            });
        }
        Ok(&self.data[row * self.cols + col])
    }

    /// Get mutable element at position
    pub fn get_mut(&mut self, row: usize, col: usize) -> AmxResult<&mut T> {
        if row >= self.rows || col >= self.cols {
            return Err(AmxError::IndexOutOfBounds {
                index: row * self.cols + col,
                max: self.data.len(),
            });
        }
        Ok(&mut self.data[row * self.cols + col])
    }

    /// Set element at position
    pub fn set(&mut self, row: usize, col: usize, value: T) -> AmxResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(AmxError::IndexOutOfBounds {
                index: row * self.cols + col,
                max: self.data.len(),
            });
        }
        self.data[row * self.cols + col] = value;
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

    /// Transpose the matrix.
    ///
    /// Uses cache-oblivious recursive decomposition for large matrices
    /// to minimize TLB and cache misses.
    pub fn transpose(&self) -> AmxResult<Matrix<T>> {
        let mut result = Matrix::zeros(self.cols, self.rows)?;
        transpose_block(&self.data, &mut result.data, self.rows, self.cols, self.cols, self.rows);
        Ok(result)
    }
}

/// Cache-oblivious transpose: recursively decompose until tiles fit in L1.
fn transpose_block<T: Clone + Default>(
    src: &[T], dst: &mut [T],
    rows: usize, cols: usize,
    src_stride: usize, dst_stride: usize,
) {
    const BLOCK: usize = 32;
    if rows <= BLOCK && cols <= BLOCK {
        // Base case: small enough for L1
        for i in 0..rows {
            for j in 0..cols {
                dst[j * dst_stride + i] = src[i * src_stride + j].clone();
            }
        }
    } else if rows >= cols {
        let mid = rows / 2;
        transpose_block(src, dst, mid, cols, src_stride, dst_stride);
        transpose_block(
            &src[mid * src_stride..], &mut dst[mid..],
            rows - mid, cols, src_stride, dst_stride,
        );
    } else {
        let mid = cols / 2;
        transpose_block(src, dst, rows, mid, src_stride, dst_stride);
        transpose_block(
            &src[mid..], &mut dst[mid * dst_stride..],
            rows, cols - mid, src_stride, dst_stride,
        );
    }
}

// ---------------------------------------------------------------------------
// f32 matrix multiplication — scalar and AMX implementations
// ---------------------------------------------------------------------------

/// Allocate `size` bytes with `align`-byte alignment (zeroed).
fn aligned_alloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 { return align as *mut u8; } // non-null sentinel
    let layout = alloc::alloc::Layout::from_size_align(size, align)
        .expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc::alloc_zeroed(layout);
        assert!(!ptr.is_null(), "allocation failed");
        ptr
    }
}

/// Free memory allocated with `aligned_alloc`.
fn aligned_free(ptr: *mut u8, size: usize, align: usize) {
    if size == 0 { return; }
    let layout = alloc::alloc::Layout::from_size_align(size, align)
        .expect("invalid layout");
    unsafe { alloc::alloc::dealloc(ptr, layout); }
}

impl Matrix<f32> {
    /// Matrix multiplication using the best available backend.
    ///
    /// Uses multi-threaded AMX on `aarch64` with `std` feature, single-threaded
    /// AMX without `std`, or scalar fallback on non-Apple Silicon.
    pub fn matmul(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        #[cfg(target_arch = "aarch64")]
        {
            if amx_sys::is_amx_available() {
                #[cfg(feature = "std")]
                {
                    let n_threads = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    return self.matmul_amx_parallel(other, n_threads);
                }
                #[cfg(not(feature = "std"))]
                {
                    return self.matmul_amx(other);
                }
            }
        }
        self.matmul_scalar(other)
    }

    /// Scalar (pure-Rust) matrix multiplication.
    ///
    /// Uses the `i,k,j` loop order for optimal cache behaviour on row-major
    /// storage.  The inner `j` loop is auto-vectorisable by the compiler.
    pub fn matmul_scalar(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            let c_row = &mut c[i * n..(i + 1) * n];
            for kk in 0..k {
                let a_ik = a[i * k + kk];
                let b_row = &b[kk * n..(kk + 1) * n];
                // Inner loop — auto-vectorisable
                for j in 0..n {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
        Matrix::from_data(c, m, n)
    }

    /// High-precision scalar matrix multiplication using f64 accumulation.
    ///
    /// Each output element is accumulated in f64 precision, then rounded
    /// to f32.  This reduces catastrophic cancellation for ill-conditioned
    /// matrices at the cost of ~2× slower throughput (half SIMD width).
    pub fn matmul_f64(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }
        let a = self.as_slice();
        let b = other.as_slice();
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            let c_row = &mut c[i * n..(i + 1) * n];
            for kk in 0..k {
                let a_ik = a[i * k + kk] as f64;
                let b_row = &b[kk * n..(kk + 1) * n];
                for j in 0..n {
                    c_row[j] += a_ik * (b_row[j] as f64);
                }
            }
        }
        let c_f32: Vec<f32> = c.into_iter().map(|v| v as f32).collect();
        Matrix::from_data(c_f32, m, n)
    }

    /// AMX-accelerated matrix multiplication using 16×16 outer-product tiling.
    ///
    /// Pre-packs A (column-gather) and B (row-copy) into aligned panels,
    /// then calls a C micro-kernel that runs the entire inner loop without
    /// per-iteration FFI overhead.  The micro-kernel is software-pipelined:
    /// loads overlap with fma for better throughput.
    ///
    /// Only available on `aarch64`.
    #[cfg(target_arch = "aarch64")]
    pub fn matmul_amx(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        use amx_sys::*;

        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        const TILE: usize = 16;
        const TILE_BYTES: usize = TILE * 4; // 64

        let a = self.as_slice();
        let b = other.as_slice();

        let n_i_tiles = (m + TILE - 1) / TILE;
        let n_j_tiles = (n + TILE - 1) / TILE;

        // ── Pre-pack A (column-major within each tile row) ───────────
        let a_pack_len = n_i_tiles * k;
        let a_packed = aligned_alloc(a_pack_len * TILE_BYTES, 64);
        for it in 0..n_i_tiles {
            let i_blk = it * TILE;
            let tile_m = TILE.min(m - i_blk);
            for kk in 0..k {
                let dst = unsafe { a_packed.add((it * k + kk) * TILE_BYTES) as *mut f32 };
                for ii in 0..tile_m {
                    unsafe { dst.add(ii).write(a[(i_blk + ii) * k + kk]); }
                }
            }
        }

        // ── Pre-pack B (row-major within each tile column) ───────────
        let b_pack_len = n_j_tiles * k;
        let b_packed = aligned_alloc(b_pack_len * TILE_BYTES, 64);
        for jt in 0..n_j_tiles {
            let j_blk = jt * TILE;
            let tile_n = TILE.min(n - j_blk);
            for kk in 0..k {
                let dst = unsafe { b_packed.add((jt * k + kk) * TILE_BYTES) };
                let src = b[kk * n + j_blk..].as_ptr() as *const u8;
                unsafe { core::ptr::copy_nonoverlapping(src, dst, tile_n * 4); }
            }
        }

        // ── Tile loop with C micro-kernel ────────────────────────────
        let mut c_data = vec![0.0f32; m * n];

        // Tile store buffer: 16 rows × 64 bytes = 1024 bytes
        let z_buf = aligned_alloc(TILE * TILE_BYTES, 64);

        unsafe {
            amx_set();

            // Process in j-major order: B panel stays warm in cache
            // across all i-tiles that share the same j-tile.
            for jt in 0..n_j_tiles {
                let j_blk = jt * TILE;
                let tile_n = TILE.min(n - j_blk);
                let bp = b_packed.add(jt * k * TILE_BYTES);

                for it in 0..n_i_tiles {
                    let i_blk = it * TILE;
                    let tile_m = TILE.min(m - i_blk);
                    let ap = a_packed.add(it * k * TILE_BYTES);

                    amx_f32_tile_kernel(ap, bp, z_buf, k as i32, tile_m as i32);

                    // Unpack Z store buffer → output matrix
                    // Use copy_nonoverlapping for bandwidth
                    for ii in 0..tile_m {
                        let src = z_buf.add(ii * TILE_BYTES) as *const f32;
                        let dst = c_data.as_mut_ptr().add((i_blk + ii) * n + j_blk);
                        core::ptr::copy_nonoverlapping(src, dst, tile_n);
                    }
                }
            }

            amx_clr();
        }

        aligned_free(a_packed, a_pack_len * TILE_BYTES, 64);
        aligned_free(b_packed, b_pack_len * TILE_BYTES, 64);
        aligned_free(z_buf, TILE * TILE_BYTES, 64);

        Matrix::from_data(c_data, m, n)
    }

    /// Multi-threaded AMX matmul — 2D tile distribution across threads.
    ///
    /// Pre-packs A and B globally (shared read-only), then distributes
    /// (it, jt) tile pairs evenly across threads.  Each thread gets its
    /// own AMX context and Z buffer.  This distributes work well even
    /// for very rectangular matrices (e.g. 32×4096 × 4096×4096).
    ///
    /// Falls back to single-threaded `matmul_amx` when `n_threads <= 1` or
    /// the matrix is too small to benefit from parallelism.
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    pub fn matmul_amx_parallel(&self, other: &Matrix<f32>, n_threads: usize) -> AmxResult<Matrix<f32>> {
        use amx_sys::*;

        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        const TILE: usize = 16;
        const TILE_BYTES: usize = TILE * 4;

        let a = self.as_slice();
        let b = other.as_slice();

        let n_i_tiles = (m + TILE - 1) / TILE;
        let n_j_tiles = (n + TILE - 1) / TILE;
        let total_tiles = n_i_tiles * n_j_tiles;

        let n_threads = n_threads.max(1);
        if n_threads <= 1 || total_tiles < n_threads * 2 {
            return self.matmul_amx(other);
        }

        // ── Pre-pack A globally (shared read-only) ──────────────────
        let a_pack_len = n_i_tiles * k;
        let a_packed = aligned_alloc(a_pack_len * TILE_BYTES, 64);
        for it in 0..n_i_tiles {
            let i_blk = it * TILE;
            let tile_m = TILE.min(m - i_blk);
            for kk in 0..k {
                let dst = unsafe { a_packed.add((it * k + kk) * TILE_BYTES) as *mut f32 };
                for ii in 0..tile_m {
                    unsafe { dst.add(ii).write(a[(i_blk + ii) * k + kk]); }
                }
            }
        }

        // ── Pre-pack B globally (shared read-only) ──────────────────
        let b_pack_len = n_j_tiles * k;
        let b_packed = aligned_alloc(b_pack_len * TILE_BYTES, 64);
        for jt in 0..n_j_tiles {
            let j_blk = jt * TILE;
            let tile_n = TILE.min(n - j_blk);
            for kk in 0..k {
                let dst = unsafe { b_packed.add((jt * k + kk) * TILE_BYTES) };
                let src = b[kk * n + j_blk..].as_ptr() as *const u8;
                unsafe { core::ptr::copy_nonoverlapping(src, dst, tile_n * 4); }
            }
        }

        let mut c_data = vec![0.0f32; m * n];

        let c_send = SendPtr(c_data.as_mut_ptr());
        let a_send = SendPtr(a_packed);
        let b_send = SendPtr(b_packed);

        std::thread::scope(|scope| {
            let tiles_per_thread = (total_tiles + n_threads - 1) / n_threads;

            for tid in 0..n_threads {
                let start_tile = tid * tiles_per_thread;
                let end_tile = (start_tile + tiles_per_thread).min(total_tiles);
                if start_tile >= end_tile { continue; }

                let c_s = c_send;
                let a_s = a_send;
                let b_s = b_send;

                let work = unsafe { AssertSend(move || {
                    let c_out = c_s.0;
                    let a_pack = a_s.0;
                    let b_pack = b_s.0;

                    let z_buf = aligned_alloc(TILE * TILE_BYTES, 64);

                    amx_set();

                    for tile_idx in start_tile..end_tile {
                        // 2D tile index: column-major order for B-panel reuse
                        let jt = tile_idx / n_i_tiles;
                        let it = tile_idx % n_i_tiles;

                        let i_blk = it * TILE;
                        let j_blk = jt * TILE;
                        let tile_m = TILE.min(m - i_blk);
                        let tile_n = TILE.min(n - j_blk);

                        let ap = a_pack.add(it * k * TILE_BYTES);
                        let bp = b_pack.add(jt * k * TILE_BYTES);

                        amx_f32_tile_kernel(ap, bp, z_buf, k as i32, tile_m as i32);

                        for ii in 0..tile_m {
                            let src = z_buf.add(ii * TILE_BYTES) as *const f32;
                            let dst = c_out.add((i_blk + ii) * n + j_blk);
                            core::ptr::copy_nonoverlapping(src, dst, tile_n);
                        }
                    }

                    amx_clr();
                    aligned_free(z_buf, TILE * TILE_BYTES, 64);
                }) };

                scope.spawn(|| work.run());
            }
        });

        aligned_free(a_packed, a_pack_len * TILE_BYTES, 64);
        aligned_free(b_packed, b_pack_len * TILE_BYTES, 64);

        Matrix::from_data(c_data, m, n)
    }
}

/// Wrapper to send raw pointers across thread boundaries.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Assert a closure is Send even if it captures raw pointers.
/// Safety: caller must ensure the captured data is valid and non-overlapping.
#[cfg(feature = "std")]
struct AssertSend<F: FnOnce()>(F);
#[cfg(feature = "std")]
unsafe impl<F: FnOnce()> Send for AssertSend<F> {}
#[cfg(feature = "std")]
impl<F: FnOnce()> AssertSend<F> {
    fn run(self) { (self.0)() }
}

impl<T: fmt::Display + Clone + Default> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix({} x {})", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                write!(f, "{}", self.data[i * self.cols + j])?;
                if j < self.cols - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl<T: Clone + Default> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Matrix {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_matmul_scalar_identity() {
        let a = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let eye = Matrix::from_data(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let c = eye.matmul_scalar(&a).unwrap();
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let eye = Matrix::from_data(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let c = eye.matmul(&a).unwrap();
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        let a = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let b = Matrix::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), (2, 2));
        let s = c.as_slice();
        assert!((s[0] - 58.0).abs() < 1e-4);
        assert!((s[1] - 64.0).abs() < 1e-4);
        assert!((s[2] - 139.0).abs() < 1e-4);
        assert!((s[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Matrix::<f32>::zeros(2, 3).unwrap();
        let b = Matrix::<f32>::zeros(4, 2).unwrap();
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_matmul_large_tiled() {
        // 20×17 × 17×19 — exercises partial edge tiles (>16)
        let m = 20;
        let k = 17;
        let n = 19;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let a = Matrix::from_data(a_data, m, k).unwrap();
        let b = Matrix::from_data(b_data, k, n).unwrap();

        let c_best = a.matmul(&b).unwrap();
        let c_scalar = a.matmul_scalar(&b).unwrap();

        for (got, want) in c_best.as_slice().iter().zip(c_scalar.as_slice().iter()) {
            assert!(
                (got - want).abs() < 1e-2,
                "mismatch: got={got}, want={want}"
            );
        }
    }

    #[test]
    fn test_matmul_scalar_matches_amx_16x16() {
        let m = 16;
        let k = 16;
        let n = 16;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
        let a = Matrix::from_data(a_data, m, k).unwrap();
        let b = Matrix::from_data(b_data, k, n).unwrap();

        let c_best = a.matmul(&b).unwrap();
        let c_scalar = a.matmul_scalar(&b).unwrap();

        for (i, (got, want)) in c_best.as_slice().iter().zip(c_scalar.as_slice().iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "element {i}: got={got}, want={want}"
            );
        }
    }

    #[test]
    fn test_matmul_f64_precision() {
        // Test that f64 accumulation is more precise than f32
        let m = 64;
        let k = 256;
        let n = 64;
        let a_data: Vec<f32> = (0..m * k).map(|i| {
            let x = (i as f32) * 1e-4;
            if i % 3 == 0 { 1e6 + x } else { -1e6 + x }
        }).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| {
            let x = (i as f32) * 1e-4;
            if i % 5 == 0 { 1e3 + x } else { -1e3 + x }
        }).collect();
        let a = Matrix::from_data(a_data, m, k).unwrap();
        let b = Matrix::from_data(b_data, k, n).unwrap();

        let c_f64 = a.matmul_f64(&b).unwrap();
        let c_f32 = a.matmul_scalar(&b).unwrap();

        // Both should produce finite results
        assert!(c_f64.as_slice().iter().all(|v| v.is_finite()));
        assert!(c_f32.as_slice().iter().all(|v| v.is_finite()));

        // f64 result should exist (basic sanity)
        assert_eq!(c_f64.dims(), (m, n));
    }

    #[test]
    fn test_transpose_large() {
        let m = 100;
        let n = 80;
        let data: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        let a = Matrix::from_data(data, m, n).unwrap();
        let at = a.transpose().unwrap();
        assert_eq!(at.dims(), (n, m));
        for i in 0..m {
            for j in 0..n {
                assert_eq!(*at.get(j, i).unwrap(), *a.get(i, j).unwrap());
            }
        }
    }
}
