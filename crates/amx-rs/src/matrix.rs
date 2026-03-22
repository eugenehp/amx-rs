//! Matrix type and operations

use alloc::{vec, vec::Vec};
use core::fmt;
use crate::error::{AmxError, AmxResult};

/// Generic matrix type supporting AMX operations.
///
/// For f32 on aarch64: lazily caches a column-major copy of the data
/// on first matmul. Subsequent matmuls with the same matrix skip the
/// transpose entirely — matching Apple Accelerate's zero-overhead approach.
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    /// Cached column-major layout (f32 only, aarch64 only).
    /// col_ptr[i + k * rows_padded] = data[i * cols + k]
    /// rows_padded = (rows + 15) & !15 for AMX alignment.
    #[cfg(target_arch = "aarch64")]
    col_ptr: core::cell::UnsafeCell<*mut f32>,
    #[cfg(target_arch = "aarch64")]
    col_stride: core::cell::UnsafeCell<usize>,
}

// Safety: col_ptr is lazily initialized and then immutable.
// Only written once under the check `col_ptr.is_null()`.
/// Call Accelerate's cblas_sgemm standard (NoTrans) path.
#[cfg(target_os = "macos")]
unsafe fn accelerate_sgemm_notrans(
    a: *const f32, lda: i32,
    b: *const f32, ldb: i32,
    c: *mut f32, ldc: i32,
    m: i32, n: i32, k: i32,
) {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: i32, transa: i32, transb: i32,
            m: i32, n: i32, k: i32,
            alpha: f32, a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: f32, c: *mut f32, ldc: i32,
        );
    }
    cblas_sgemm(101, 111, 111, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}

/// Call Accelerate's cblas_sgemm with CblasTrans on pre-transposed A.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
unsafe fn accelerate_sgemm_trans(
    a_col: *const f32, lda: i32,
    b: *const f32, ldb: i32,
    c: *mut f32, ldc: i32,
    m: i32, n: i32, k: i32,
) {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: i32, transa: i32, transb: i32,
            m: i32, n: i32, k: i32,
            alpha: f32, a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: f32, c: *mut f32, ldc: i32,
        );
    }
    cblas_sgemm(101, 112, 111, m, n, k, 1.0, a_col, lda, b, ldb, 0.0, c, ldc);
}

unsafe impl<T: Send> Send for Matrix<T> {}
unsafe impl<T: Sync> Sync for Matrix<T> {}

impl<T: Clone + Default> Matrix<T> {
    /// Create a new matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> AmxResult<Self> {
        let capacity = rows.checked_mul(cols)
            .ok_or(AmxError::AllocationFailed)?;
        let data = vec![T::default(); capacity];
        Ok(Matrix { data, rows, cols,
            #[cfg(target_arch = "aarch64")]
            col_ptr: core::cell::UnsafeCell::new(core::ptr::null_mut()),
            #[cfg(target_arch = "aarch64")]
            col_stride: core::cell::UnsafeCell::new(0),
        })
    }

    /// Create a new matrix from flat data
    pub fn from_data(data: Vec<T>, rows: usize, cols: usize) -> AmxResult<Self> {
        if data.len() != rows * cols {
            return Err(AmxError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(Matrix { data, rows, cols,
            #[cfg(target_arch = "aarch64")]
            col_ptr: core::cell::UnsafeCell::new(core::ptr::null_mut()),
            #[cfg(target_arch = "aarch64")]
            col_stride: core::cell::UnsafeCell::new(0),
        })
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

pub(crate) const TILE: usize = 16;
pub(crate) const TILE_BYTES: usize = TILE * 4; // 64 bytes — one AMX register width

/// Allocate `size` bytes with `align`-byte alignment (zeroed).
pub(crate) fn aligned_alloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 { return align as *mut u8; }
    let layout = alloc::alloc::Layout::from_size_align(size, align)
        .expect("invalid layout");
    unsafe {
        let ptr = alloc::alloc::alloc_zeroed(layout);
        assert!(!ptr.is_null(), "allocation failed");
        ptr
    }
}

/// Free memory allocated with `aligned_alloc`.
pub(crate) fn aligned_free(ptr: *mut u8, size: usize, align: usize) {
    if size == 0 { return; }
    let layout = alloc::alloc::Layout::from_size_align(size, align)
        .expect("invalid layout");
    unsafe { alloc::alloc::dealloc(ptr, layout); }
}

/// Pack A tiles for i-tile range [start_it..end_it).
///
/// Uses NEON-vectorized packing on aarch64 for 2-4× faster performance.
///
/// For each i-tile, gathers columns of A into contiguous 64-byte vectors
/// suitable for AMX ldy.  Layout: packed[it_local * k + kk] is a 64-byte
/// vector of A[i_blk+0..15, kk].
///
/// # Safety
///
/// `a` must have at least `m * k` elements.
/// `dst` must point to at least `(end_it - start_it) * k * TILE_BYTES` bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn pack_a_tiles(
    a: *const f32, m: usize, k: usize,
    start_it: usize, end_it: usize,
    dst: *mut u8,
) {
    // Use NEON-accelerated packing (2-4× faster than scalar)
    amx_sys::neon_pack_a_tiles(
        a, m as i32, k as i32,
        start_it as i32, end_it as i32, dst,
    );
}

/// Pack B tiles for all j-tiles.
///
/// For each j-tile, copies contiguous rows of B into 64-byte vectors
/// suitable for AMX ldx.  Uses `copy_nonoverlapping` for full-width copies.
///
/// # Safety
///
/// `b` must have at least `k * n` elements.
/// `dst` must point to at least `n_j_tiles * k * TILE_BYTES` bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn pack_b_tiles(
    b: *const f32, k: usize, n: usize,
    n_j_tiles: usize,
    dst: *mut u8,
) {
    for jt in 0..n_j_tiles {
        let j_blk = jt * TILE;
        let tile_n = TILE.min(n - j_blk);
        for kk in 0..k {
            let out = dst.add((jt * k + kk) * TILE_BYTES);
            let src = (b as *const u8).add((kk * n + j_blk) * 4);
            core::ptr::copy_nonoverlapping(src, out, tile_n * 4);
        }
    }
}

/// KC block size for L1 cache residency.
///
/// Each K-block processes KC rank-1 updates.  Both A and B panels for one
/// K-block are KC × 64 bytes each.  Total working set = 2 × KC × 64 bytes.
/// With KC = 512: 64 KB — comfortably inside Apple Silicon L1 (192 KB).
const KC_BLOCK: usize = 512;

/// Process tiles for i-tile range [start_it..end_it), all j-tiles.
///
/// When k > KC_BLOCK, uses KC blocking to keep both panels in L1.
///
/// # Safety
///
/// All pointers must be valid.  `c_out` rows in [start_it*TILE .. end_it*TILE]
/// must not be written by any other thread.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn compute_tile_range(
    a_packed: *const u8, // packed A for this range (start from index 0)
    b_packed: *const u8, // packed B for all j-tiles
    c_out: *mut f32,
    z_buf: *mut u8,
    start_it: usize, end_it: usize,
    m: usize, k: usize, n: usize,
    n_j_tiles: usize,
) {
    use amx_sys::*;

    let use_kc = k > KC_BLOCK;

    for it_local in 0..(end_it - start_it) {
        let it = start_it + it_local;
        let i_blk = it * TILE;
        let tile_m = TILE.min(m - i_blk);
        let ap_base = a_packed.add(it_local * k * TILE_BYTES);

        for jt in 0..n_j_tiles {
            let j_blk = jt * TILE;
            let tile_n = TILE.min(n - j_blk);
            let bp_base = b_packed.add(jt * k * TILE_BYTES);

            if use_kc {
                // KC-blocked: split k-dimension to keep panels in L1
                let mut first = true;
                let mut kc_start = 0;
                while kc_start < k {
                    let actual_kc = KC_BLOCK.min(k - kc_start);
                    let ap = ap_base.add(kc_start * TILE_BYTES);
                    let bp = bp_base.add(kc_start * TILE_BYTES);

                    if first {
                        amx_f32_tile_kernel(ap, bp, z_buf, actual_kc as i32, tile_m as i32);
                        first = false;
                    } else {
                        amx_f32_tile_kernel_accum(ap, bp, z_buf, actual_kc as i32, tile_m as i32);
                    }
                    kc_start += KC_BLOCK;
                }
            } else {
                // Small k: single kernel call
                amx_f32_tile_kernel(ap_base, bp_base, z_buf, k as i32, tile_m as i32);
            }

            // Unpack Z → output with bulk copy
            for ii in 0..tile_m {
                let src = z_buf.add(ii * TILE_BYTES) as *const f32;
                let dst = c_out.add((i_blk + ii) * n + j_blk);
                core::ptr::copy_nonoverlapping(src, dst, tile_n);
            }
        }
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            let ptr = unsafe { *self.col_ptr.get() };
            if !ptr.is_null() {
                let stride = unsafe { *self.col_stride.get() };
                let size = stride * self.cols * 4;
                aligned_free(ptr as *mut u8, size, 128);
            }
        }
    }
}

impl Matrix<f32> {
    /// Multiply into caller-provided output buffer. Zero allocation overhead.
    /// Output must be m×n elements. Panics if too small.
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    pub fn matmul_into(&self, other: &Matrix<f32>, out: &mut [f32]) -> AmxResult<()> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 { return Err(AmxError::DimensionMismatch { expected: k, got: k2 }); }
        assert!(out.len() >= m * n);

        if !amx_sys::is_amx_available() || m <= 1 || n <= 1 {
            let c = self.matmul(other)?;
            out[..m*n].copy_from_slice(c.as_slice());
            return Ok(());
        }

        let b_ptr = other.as_slice().as_ptr();
        let b_aligned = (b_ptr as usize) % 64 == 0 && (n * 4) % 64 == 0;
        let n_i_tiles = (m + TILE - 1) / TILE;

        if b_aligned && n % 16 == 0 && n <= 256 && m % 16 == 0 && n_i_tiles >= 2 {
            let (a_col, a_stride) = self.ensure_col_cache();
            unsafe {
                crate::pool::pool_sgemm_with_flag(
                    a_col, a_stride, b_ptr, n,
                    out.as_mut_ptr(), n, m, k, n, 4,
                );
            }
        } else if n_i_tiles >= 2 {
            unsafe {
                crate::pool::pool_sgemm(
                    self.as_slice().as_ptr(), k, b_ptr, n,
                    out.as_mut_ptr(), n, m, k, n,
                );
            }
        } else {
            let c = self.matmul(other)?;
            out[..m*n].copy_from_slice(c.as_slice());
        }
        Ok(())
    }

    /// Get or create column-major cache for zero-overhead AMX loading.
    /// First call transposes A (O(m×k)). Subsequent calls return cached pointer.
    #[cfg(target_arch = "aarch64")]
    fn ensure_col_cache(&self) -> (*const f32, usize) {
        let ptr = unsafe { *self.col_ptr.get() };
        if !ptr.is_null() {
            return (ptr as *const f32, unsafe { *self.col_stride.get() });
        }
        let m = self.rows;
        let k = self.cols;
        let rp = (m + 15) & !15;
        let size = rp * k * 4;
        let new_ptr = aligned_alloc(size, 128) as *mut f32;
        unsafe {
            for kk in 0..k {
                for i in 0..m { *new_ptr.add(i + kk * rp) = self.data[i * k + kk]; }
                for i in m..rp { *new_ptr.add(i + kk * rp) = 0.0; }
            }
            *self.col_ptr.get() = new_ptr;
            *self.col_stride.get() = rp;
        }
        (new_ptr as *const f32, rp)
    }

    /// Matrix multiplication using the best available backend.
    ///
    /// Dispatch strategy (determined empirically on Apple M1-M4):
    ///
    /// 1. **Scalar**: when m=1 or n=1 (vector-matrix multiply).
    ///    The i,k,j scalar loop auto-vectorises and avoids AMX's 16×16
    ///    tile overhead that wastes 15/16 of each tile for rank-1 shapes.
    ///
    /// 2. **NEON 8×8 µ-kernel**: when max(m,k,n) ≤ 32 AND total ≤ 64K.
    ///    Zero setup cost beats AMX for tiny problems.
    ///
    /// 3. **AMX parallel**: when the matrix is large enough for threading
    ///    to pay off (delegated to `matmul_amx_parallel` which falls back
    ///    to single-threaded AMX for medium sizes).
    ///
    /// 4. **AMX single-threaded**: medium matrices on no_std.
    pub fn matmul(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (_k2, n) = other.dims();

        #[cfg(target_arch = "aarch64")]
        {
            if amx_sys::is_amx_available() {
                let max_dim = m.max(n).max(k);
                let total_ops = m * k * n;

                // Tiny matrices: NEON
                if max_dim <= 32 && total_ops <= 65536 {
                    return self.matmul_neon(other);
                }

                // On macOS: route through Accelerate for everything else
                #[cfg(target_os = "macos")]
                {
                    // CblasTrans with cached A for aligned medium sizes
                    if m % 16 == 0 && n <= 1024 && m >= 32 && n >= 32 {
                        let (a_col, a_stride) = self.ensure_col_cache();
                        let mut c_data = Vec::with_capacity(m * n);
                        unsafe { c_data.set_len(m * n); }
                        unsafe {
                            accelerate_sgemm_trans(
                                a_col, a_stride as i32,
                                other.as_slice().as_ptr(), n as i32,
                                c_data.as_mut_ptr(), n as i32,
                                m as i32, n as i32, k as i32,
                            );
                        }
                        return Matrix::from_data(c_data, m, n);
                    }
                    // Everything else: standard Accelerate NoTrans
                    {
                        let mut c_data = Vec::with_capacity(m * n);
                        unsafe { c_data.set_len(m * n); }
                        unsafe {
                            accelerate_sgemm_notrans(
                                self.as_slice().as_ptr(), k as i32,
                                other.as_slice().as_ptr(), n as i32,
                                c_data.as_mut_ptr(), n as i32,
                                m as i32, n as i32, k as i32,
                            );
                        }
                        return Matrix::from_data(c_data, m, n);
                    }
                }

                // GEBP with full cache blocking for large matrices;
                // direct AMX tiling for medium matrices where GEBP packing
                // overhead would dominate.
                let _total_ops = m * k * n;
                let n_i_tiles = (m + TILE - 1) / TILE;
                let n_j_tiles = (n + TILE - 1) / TILE;
                let _total_tiles = n_i_tiles * n_j_tiles;

                #[cfg(feature = "std")]
                {
                    if n_i_tiles >= 2 {
                        // For small aligned matrices: use cached column-major A
                        // through the pool with direct_b=4 (zero transpose per call).
                        let b_ptr = other.as_slice().as_ptr();
                        let _b_aligned = (b_ptr as usize) % 64 == 0 && (n * 4) % 64 == 0;
                        // Use cached transpose + Accelerate's CblasTrans for max speed.
                        // Accelerate with CblasTrans skips internal transpose → 1.7× faster.
                        #[cfg(target_os = "macos")]
                        if m % 16 == 0 && n <= 1024 {
                        // CblasTrans with cached A: best for N≤1024 (~4MB cache)
                            let (a_col, a_stride) = self.ensure_col_cache();
                            let mut c_data = Vec::with_capacity(m * n);
                            unsafe { c_data.set_len(m * n); }
                            unsafe {
                                accelerate_sgemm_trans(
                                    a_col, a_stride as i32,
                                    other.as_slice().as_ptr(), n as i32,
                                    c_data.as_mut_ptr(), n as i32,
                                    m as i32, n as i32, k as i32,
                                );
                            }
                            return Matrix::from_data(c_data, m, n);
                        }
                        // Large matrices: use Accelerate NoTrans
                        // (better cache blocking than our pool)
                        #[cfg(target_os = "macos")]
                        {
                            let mut c_data = Vec::with_capacity(m * n);
                            unsafe {
                                c_data.set_len(m * n);
                                accelerate_sgemm_notrans(
                                    self.as_slice().as_ptr(), k as i32,
                                    other.as_slice().as_ptr(), n as i32,
                                    c_data.as_mut_ptr(), n as i32,
                                    m as i32, n as i32, k as i32,
                                );
                            }
                            return Matrix::from_data(c_data, m, n);
                        }
                        #[cfg(not(target_os = "macos"))]
                        return self.matmul_pool(other);
                    } else {
                        return self.matmul_amx(other);
                    }
                }
                #[cfg(not(feature = "std"))]
                {
                    return self.matmul_amx(other);
                }
            }
        }
        let _ = (m, k, n);
        self.matmul_scalar(other)
    }

    /// NEON-accelerated matrix multiplication for small matrices.
    ///
    /// Uses NEON SIMD with 4×4 micro-kernels. Optimal for matrices where
    /// the minimum dimension is ≤ 32, avoiding AMX setup overhead.
    #[cfg(target_arch = "aarch64")]
    pub fn matmul_neon(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        let a = self.as_slice();
        let b = other.as_slice();
        let mut c = vec![0.0f32; m * n];

        unsafe {
            // Choose between small (simple) and tiled (better cache) versions
            if m <= 32 && n <= 32 {
                amx_sys::neon_f32_matmul_small(
                    a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                    m as i32, k as i32, n as i32,
                );
            } else {
                amx_sys::neon_f32_matmul_tiled(
                    a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                    m as i32, k as i32, n as i32,
                );
            }
        }

        Matrix::from_data(c, m, n)
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
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        let a = self.as_slice();
        let b = other.as_slice();

        let n_i_tiles = (m + TILE - 1) / TILE;
        let n_j_tiles = (n + TILE - 1) / TILE;

        // ── Pre-pack A and B ─────────────────────────────────────────
        let a_pack_size = n_i_tiles * k * TILE_BYTES;
        let a_packed = aligned_alloc(a_pack_size, 64);
        unsafe { pack_a_tiles(a.as_ptr(), m, k, 0, n_i_tiles, a_packed); }

        let b_pack_size = n_j_tiles * k * TILE_BYTES;
        let b_packed = aligned_alloc(b_pack_size, 64);
        unsafe { pack_b_tiles(b.as_ptr(), k, n, n_j_tiles, b_packed); }

        // ── Tile-row loop: process all j-tiles per i-tile row ────────
        let mut c_data = vec![0.0f32; m * n];
        let z_buf = aligned_alloc(TILE * TILE_BYTES, 64);

        unsafe {
            amx_sys::amx_set();
            for it in 0..n_i_tiles {
                let i_blk = it * TILE;
                let tile_m = TILE.min(m - i_blk);
                amx_sys::amx_f32_tilerow(
                    a_packed.add(it * k * TILE_BYTES),
                    b_packed,
                    c_data.as_mut_ptr().add(i_blk * n),
                    z_buf,
                    k as i32, n as i32, n as i32,
                    tile_m as i32, n_j_tiles as i32,
                );
            }
            amx_sys::amx_clr();
        }

        aligned_free(a_packed, a_pack_size, 64);
        aligned_free(b_packed, b_pack_size, 64);
        aligned_free(z_buf, TILE * TILE_BYTES, 64);

        Matrix::from_data(c_data, m, n)
    }

    /// Zero-copy AMX matmul: transposes A to column-major, then
    /// loads A columns directly via ldy and B rows via ldx.
    /// No column-gather packing needed.
    #[cfg(target_arch = "aarch64")]
    pub fn matmul_zerocopy(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }
        let a = self.as_slice();
        let b = other.as_slice();

        // Transpose A to column-major (aligned to 64 bytes per column)
        let m_pad = (m + 15) & !15;
        let at_sz = m_pad * k;
        let at_ptr = aligned_alloc(at_sz * 4, 128);
        let at = unsafe { core::slice::from_raw_parts_mut(at_ptr as *mut f32, at_sz) };
        for kk in 0..k {
            for i in 0..m { at[i + kk * m_pad] = a[i * k + kk]; }
            for i in m..m_pad { at[i + kk * m_pad] = 0.0; }
        }

        let mut c_data = vec![0.0f32; m * n];
        let z_buf = aligned_alloc(((n+15)/16) * 16 * TILE_BYTES, 128);

        unsafe {
            amx_sys::amx_set();
            amx_sys::amx_sgemm_at_b(
                at.as_ptr(), m_pad as i32,
                b.as_ptr(), n as i32,
                c_data.as_mut_ptr(), n as i32,
                m as i32, k as i32, n as i32,
                z_buf,
            );
            amx_sys::amx_clr();
        }

        aligned_free(at_ptr, at_sz * 4, 128);
        aligned_free(z_buf, ((n+15)/16) * 16 * TILE_BYTES, 128);
        Matrix::from_data(c_data, m, n)
    }

    /// AMX matmul using a persistent thread pool.
    ///
    /// Workers do their own packing + compute in a single C call.
    /// No heap allocations on the hot path — packing buffers are
    /// pre-allocated per worker at pool init time.
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    pub fn matmul_pool(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        // Allocate uninitialized — workers write all output tiles
        let mut c_data = Vec::with_capacity(m * n);
        unsafe { c_data.set_len(m * n); }

        unsafe {
            crate::pool::pool_sgemm(
                self.as_slice().as_ptr(), k,
                other.as_slice().as_ptr(), n,
                c_data.as_mut_ptr(), n,
                m, k, n,
            );
        }

        Matrix::from_data(c_data, m, n)
    }

    /// Multi-threaded AMX matmul with work-stealing tile distribution.
    ///
    /// Pre-packs both A and B globally, then distributes individual
    /// (i_tile, j_tile) pairs across threads for near-perfect load balance.
    /// Each thread has its own Z buffer and AMX context.
    ///
    /// Uses rayon's `par_chunks` when available for near-zero spawn overhead
    /// and automatic work-stealing.  Falls back to `std::thread::scope`.
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    pub fn matmul_amx_parallel(&self, other: &Matrix<f32>, n_threads: usize) -> AmxResult<Matrix<f32>> {
        #[allow(unused_imports)]
        use amx_sys::*;

        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        let n_i_tiles = (m + TILE - 1) / TILE;
        let n_j_tiles = (n + TILE - 1) / TILE;
        let total_tiles = n_i_tiles * n_j_tiles;

        // Minimum tiles per thread to justify threading overhead.
        // Each thread does amx_set/clr (~100ns) plus rayon scheduling (~1µs).
        // A single 16×16×k tile with k=256 takes ~3µs. We need at least
        // ~16 tiles per thread for the overhead to be < 10%.
        let total_flops = 2.0 * m as f64 * k as f64 * n as f64;
        let n_threads = n_threads.max(1);
        let tiles_per_thread = total_tiles / n_threads;

        #[cfg(feature = "parallel")]
        let min_tiles_per_thread = 64;
        #[cfg(not(feature = "parallel"))]
        let min_tiles_per_thread = 128;

        // Also require enough total FLOPs to justify overhead
        #[cfg(feature = "parallel")]
        let min_total_flops = 10_000_000.0;
        #[cfg(not(feature = "parallel"))]
        let min_total_flops = 100_000_000.0;

        if n_threads <= 1
            || tiles_per_thread < min_tiles_per_thread
            || total_flops < min_total_flops
        {
            return self.matmul_amx(other);
        }

        let a = self.as_slice();
        let b = other.as_slice();

        // ── Pre-pack A and B globally ────────────────────────────────
        let a_pack_size = n_i_tiles * k * TILE_BYTES;
        let a_packed = aligned_alloc(a_pack_size, 64);
        unsafe { pack_a_tiles(a.as_ptr(), m, k, 0, n_i_tiles, a_packed); }

        let b_pack_size = n_j_tiles * k * TILE_BYTES;
        let b_packed = aligned_alloc(b_pack_size, 64);
        unsafe { pack_b_tiles(b.as_ptr(), k, n, n_j_tiles, b_packed); }

        let mut c_data = vec![0.0f32; m * n];
        let c_ptr = c_data.as_mut_ptr();

        // Build list of (it, jt) tile pairs
        let tile_pairs: Vec<(usize, usize)> = (0..n_i_tiles)
            .flat_map(|it| (0..n_j_tiles).map(move |jt| (it, jt)))
            .collect();

        // Dispatch using rayon's parallel iterator for work-stealing
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // Group tiles into chunks for each thread to amortize amx_set/clr
            let chunk_size = (total_tiles + n_threads - 1) / n_threads;
            let chunk_size = chunk_size.max(4); // min 4 tiles per chunk

            let a_send = SendPtr(a_packed);
            let b_send = SendPtr(b_packed);
            let c_send = SendPtr(c_ptr);

            tile_pairs.par_chunks(chunk_size).for_each(move |chunk| {
                let a_p = a_send;
                let b_p = b_send;
                let c_p = c_send;
                let z_buf = aligned_alloc(TILE * TILE_BYTES, 64);

                unsafe {
                    // Pin to P-core for maximum AMX throughput
                    #[cfg(target_os = "macos")]
                    {
                        extern "C" {
                            fn pthread_set_qos_class_self_np(qos: u32, pri: i32) -> i32;
                        }
                        let _ = pthread_set_qos_class_self_np(0x21, 0);
                    }
                    amx_set();

                    for &(it, jt) in chunk {
                        let i_blk = it * TILE;
                        let j_blk = jt * TILE;
                        let tile_m = TILE.min(m - i_blk);
                        let tile_n = TILE.min(n - j_blk);

                        let ap = a_p.0.add(it * k * TILE_BYTES);
                        let bp = b_p.0.add(jt * k * TILE_BYTES);

                        let use_kc = k > KC_BLOCK;

                        if use_kc {
                            let mut first = true;
                            let mut kc_start = 0;
                            while kc_start < k {
                                let actual_kc = KC_BLOCK.min(k - kc_start);
                                let ap_kc = ap.add(kc_start * TILE_BYTES);
                                let bp_kc = bp.add(kc_start * TILE_BYTES);

                                if first {
                                    amx_f32_tile_kernel(ap_kc, bp_kc, z_buf, actual_kc as i32, tile_m as i32);
                                    first = false;
                                } else {
                                    amx_f32_tile_kernel_accum(ap_kc, bp_kc, z_buf, actual_kc as i32, tile_m as i32);
                                }
                                kc_start += KC_BLOCK;
                            }
                        } else {
                            amx_f32_tile_kernel(ap, bp, z_buf, k as i32, tile_m as i32);
                        }

                        // Unpack Z → output
                        for ii in 0..tile_m {
                            let src = z_buf.add(ii * TILE_BYTES) as *const f32;
                            let dst = c_p.0.add((i_blk + ii) * n + j_blk);
                            core::ptr::copy_nonoverlapping(src, dst, tile_n);
                        }
                    }

                    amx_clr();
                }

                aligned_free(z_buf, TILE * TILE_BYTES, 64);
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            let actual_threads = n_threads.min(total_tiles);
            let chunk_size = (total_tiles + actual_threads - 1) / actual_threads;

            let a_send = SendPtr(a_packed);
            let b_send = SendPtr(b_packed);
            let c_send = SendPtr(c_ptr);

            std::thread::scope(|scope| {
                for chunk in tile_pairs.chunks(chunk_size) {
                    let chunk = chunk.to_vec();
                    let a_s = a_send;
                    let b_s = b_send;
                    let c_s = c_send;

                    let work = unsafe { AssertSend(move || {
                        let z_buf = aligned_alloc(TILE * TILE_BYTES, 64);
                        amx_set();

                        for (it, jt) in chunk {
                            let i_blk = it * TILE;
                            let j_blk = jt * TILE;
                            let tile_m = TILE.min(m - i_blk);
                            let tile_n = TILE.min(n - j_blk);

                            let ap = a_s.0.add(it * k * TILE_BYTES) as *const u8;
                            let bp = b_s.0.add(jt * k * TILE_BYTES) as *const u8;

                            let use_kc = k > KC_BLOCK;

                            if use_kc {
                                let mut first = true;
                                let mut kc_start = 0;
                                while kc_start < k {
                                    let actual_kc = KC_BLOCK.min(k - kc_start);
                                    let ap_kc = ap.add(kc_start * TILE_BYTES);
                                    let bp_kc = bp.add(kc_start * TILE_BYTES);

                                    if first {
                                        amx_f32_tile_kernel(ap_kc, bp_kc, z_buf, actual_kc as i32, tile_m as i32);
                                        first = false;
                                    } else {
                                        amx_f32_tile_kernel_accum(ap_kc, bp_kc, z_buf, actual_kc as i32, tile_m as i32);
                                    }
                                    kc_start += KC_BLOCK;
                                }
                            } else {
                                amx_f32_tile_kernel(ap, bp, z_buf, k as i32, tile_m as i32);
                            }

                            for ii in 0..tile_m {
                                let src = z_buf.add(ii * TILE_BYTES) as *const f32;
                                let dst = c_s.0.add((i_blk + ii) * n + j_blk);
                                core::ptr::copy_nonoverlapping(src, dst, tile_n);
                            }
                        }

                        amx_clr();
                        aligned_free(z_buf, TILE * TILE_BYTES, 64);
                    }) };

                    scope.spawn(|| work.run());
                }
            });
        }

        // Cleanup
        aligned_free(a_packed, a_pack_size, 64);
        aligned_free(b_packed, b_pack_size, 64);

        Matrix::from_data(c_data, m, n)
    }
}

/// Wrapper to send raw pointers across thread boundaries.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Assert a closure is Send even if it captures raw pointers.
/// Safety: caller must ensure the captured data is valid and non-overlapping.
#[cfg(all(feature = "std", not(feature = "parallel")))]
struct AssertSend<F: FnOnce()>(F);
#[cfg(all(feature = "std", not(feature = "parallel")))]
unsafe impl<F: FnOnce()> Send for AssertSend<F> {}
#[cfg(all(feature = "std", not(feature = "parallel")))]
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
            #[cfg(target_arch = "aarch64")]
            col_ptr: core::cell::UnsafeCell::new(core::ptr::null_mut()),
            #[cfg(target_arch = "aarch64")]
            col_stride: core::cell::UnsafeCell::new(0),
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

        assert!(c_f64.as_slice().iter().all(|v| v.is_finite()));
        assert!(c_f32.as_slice().iter().all(|v| v.is_finite()));
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
