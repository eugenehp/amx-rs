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

const TILE: usize = 16;
const TILE_BYTES: usize = TILE * 4; // 64 bytes — one AMX register width

/// Allocate `size` bytes with `align`-byte alignment (zeroed).
fn aligned_alloc(size: usize, align: usize) -> *mut u8 {
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
fn aligned_free(ptr: *mut u8, size: usize, align: usize) {
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
unsafe fn pack_a_tiles(
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
unsafe fn pack_b_tiles(
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

impl Matrix<f32> {
    /// Matrix multiplication using the best available backend.
    ///
    /// Uses NEON for small matrices (N ≤ 32 AND total ops ≤ 32768) where 
    /// AMX setup overhead dominates. Uses multi-threaded AMX on `aarch64` 
    /// with `std` feature for larger matrices, single-threaded AMX without 
    /// `std`, or scalar fallback on non-Apple Silicon.
    pub fn matmul(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (_k2, n) = other.dims();

        #[cfg(target_arch = "aarch64")]
        {
            if amx_sys::is_amx_available() {
                // For small matrices, NEON is faster than AMX due to lower overhead
                let min_dim = m.min(n);
                let total_ops = m * k * n;
                
                // Use NEON for truly small matrices only:
                // - min dimension ≤ 32 AND total operations ≤ 32K
                // This avoids NEON for large thin matrices where AMX is better
                let use_neon = min_dim <= 32 && total_ops <= 32768;
                
                if use_neon {
                    return self.matmul_neon(other);
                }
                
                // Use AMX for larger matrices
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

        // ── Tile loop ────────────────────────────────────────────────
        let mut c_data = vec![0.0f32; m * n];
        let z_buf = aligned_alloc(TILE * TILE_BYTES, 64);

        unsafe {
            amx_sys::amx_set();
            compute_tile_range(
                a_packed, b_packed, c_data.as_mut_ptr(), z_buf,
                0, n_i_tiles, m, k, n, n_j_tiles,
            );
            amx_sys::amx_clr();
        }

        aligned_free(a_packed, a_pack_size, 64);
        aligned_free(b_packed, b_pack_size, 64);
        aligned_free(z_buf, TILE * TILE_BYTES, 64);

        Matrix::from_data(c_data, m, n)
    }

    /// Multi-threaded AMX matmul with parallel packing.
    ///
    /// Distributes i-tile rows across workers.  Each worker:
    ///   1. Packs its own A tiles (parallel — eliminates sequential bottleneck)
    ///   2. Computes all j-tiles for its i-tiles using AMX
    ///
    /// B is pre-packed once (fast sequential reads) and shared read-only.
    ///
    /// When the `parallel` feature is enabled (default), uses rayon's
    /// persistent work-stealing thread pool for near-zero spawn overhead.
    /// Without rayon, falls back to `std::thread::scope`.
    ///
    /// Falls back to single-threaded when the matrix is too small for the
    /// overhead to pay off.
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

        // Minimum FLOPs per thread to justify threading overhead.
        // With rayon (~1µs spawn), threshold is lower.
        // With std::thread (~20µs spawn), threshold is higher.
        let total_flops = 2.0 * m as f64 * k as f64 * n as f64;
        let n_threads = n_threads.max(1);

        #[cfg(feature = "parallel")]
        let min_flops_per_thread = 500_000.0; // rayon: low overhead
        #[cfg(not(feature = "parallel"))]
        let min_flops_per_thread = 10_000_000.0; // std::thread: high overhead

        if n_threads <= 1
            || n_i_tiles < 2
            || total_flops / (n_threads as f64) < min_flops_per_thread
        {
            return self.matmul_amx(other);
        }

        let a = self.as_slice();
        let b = other.as_slice();

        // ── Pre-pack B globally (fast — sequential source reads) ─────
        let b_pack_size = n_j_tiles * k * TILE_BYTES;
        let b_packed = aligned_alloc(b_pack_size, 64);
        unsafe { pack_b_tiles(b.as_ptr(), k, n, n_j_tiles, b_packed); }

        // ── Pre-allocate z_bufs for each thread ──────────────────────
        let actual_threads = n_threads.min(n_i_tiles);
        let z_bufs: Vec<*mut u8> = (0..actual_threads)
            .map(|_| aligned_alloc(TILE * TILE_BYTES, 64))
            .collect();

        let mut c_data = vec![0.0f32; m * n];

        let c_ptr = c_data.as_mut_ptr();
        let a_ptr = a.as_ptr();
        let b_ptr = b_packed;
        let i_tiles_per_thread = (n_i_tiles + actual_threads - 1) / actual_threads;

        // Dispatch workers using the best available parallelism backend
        #[cfg(feature = "parallel")]
        {
            // Wrap all raw-pointer state into a Send struct for rayon
            struct TileWork {
                c_ptr: SendPtr<f32>,
                a_ptr: SendPtr<f32>,
                b_ptr: SendPtr<u8>,
                z_buf: SendPtr<u8>,
                start_it: usize,
                end_it: usize,
                m: usize,
                k: usize,
                n: usize,
                n_j_tiles: usize,
            }
            unsafe impl Send for TileWork {}

            impl TileWork {
                unsafe fn run(&self) {
                    use amx_sys::*;
                    let my_tiles = self.end_it - self.start_it;
                    let a_pack_size = my_tiles * self.k * TILE_BYTES;
                    let a_packed = aligned_alloc(a_pack_size, 64);
                    pack_a_tiles(
                        self.a_ptr.0 as *const f32, self.m, self.k,
                        self.start_it, self.end_it, a_packed,
                    );

                    amx_set();
                    compute_tile_range(
                        a_packed, self.b_ptr.0 as *const u8,
                        self.c_ptr.0, self.z_buf.0,
                        self.start_it, self.end_it,
                        self.m, self.k, self.n, self.n_j_tiles,
                    );
                    amx_clr();

                    aligned_free(a_packed, a_pack_size, 64);
                }
            }

            let works: Vec<TileWork> = (0..actual_threads)
                .filter_map(|tid| {
                    let start_it = tid * i_tiles_per_thread;
                    let end_it = (start_it + i_tiles_per_thread).min(n_i_tiles);
                    if start_it >= end_it { return None; }
                    Some(TileWork {
                        c_ptr: SendPtr(c_ptr),
                        a_ptr: SendPtr(a_ptr as *mut f32),
                        b_ptr: SendPtr(b_ptr),
                        z_buf: SendPtr(z_bufs[tid]),
                        start_it, end_it, m, k, n, n_j_tiles,
                    })
                })
                .collect();

            rayon::scope(|s| {
                for work in &works {
                    s.spawn(move |_| unsafe { work.run() });
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            let c_send = SendPtr(c_ptr);
            let a_send = SendPtr(a_ptr as *mut f32);
            let b_send = SendPtr(b_ptr);

            std::thread::scope(|scope| {
                for tid in 0..actual_threads {
                    let c_s = c_send;
                    let a_s = a_send;
                    let b_s = b_send;
                    let z_buf = z_bufs[tid];
                    let start_it = tid * i_tiles_per_thread;
                    let end_it = (start_it + i_tiles_per_thread).min(n_i_tiles);

                    if start_it >= end_it { continue; }

                    let work = unsafe { AssertSend(move || {
                        let my_tiles = end_it - start_it;
                        let a_pack_size = my_tiles * k * TILE_BYTES;
                        let a_packed = aligned_alloc(a_pack_size, 64);
                        pack_a_tiles(a_s.0 as *const f32, m, k, start_it, end_it, a_packed);

                        amx_set();
                        compute_tile_range(
                            a_packed, b_s.0 as *const u8, c_s.0, z_buf,
                            start_it, end_it, m, k, n, n_j_tiles,
                        );
                        amx_clr();

                        aligned_free(a_packed, a_pack_size, 64);
                    }) };

                    scope.spawn(|| work.run());
                }
            });
        }

        // Cleanup
        for &z in &z_bufs {
            aligned_free(z, TILE * TILE_BYTES, 64);
        }
        aligned_free(b_packed, b_pack_size, 64);

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
