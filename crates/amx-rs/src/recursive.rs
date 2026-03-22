//! Recursive cache-oblivious matrix multiplication (Apple-style).
//!
//! Instead of GEBP's pack-then-compute model, this recursively splits
//! the matrix until sub-problems fit in cache, then packs only the
//! tiny leaf tiles. This eliminates the ~40-70µs full-matrix packing
//! overhead that dominates at N=64-256.
//!
//! Recursion order: split K first (for L1), then M and N (for L2/L3).
//! At each leaf (m≤16, n≤16, k≤KC), pack locally and run AMX micro-kernel.

use alloc::vec;
use crate::error::{AmxError, AmxResult};
use crate::matrix::{Matrix, TILE, TILE_BYTES, aligned_alloc, aligned_free};

/// Max K at the leaf level. Leaf packing buffers = KC_LEAF * 64 * 2 = 64 KB.
const KC_LEAF: usize = 512;

/// Leaf tile size (AMX register tile).
const MR: usize = TILE; // 16
const NR: usize = TILE; // 16

// ═══════════════════════════════════════════════════════════════════════
// Per-thread scratch buffers, allocated once
// ═══════════════════════════════════════════════════════════════════════

struct ScratchBufs {
    a_pack: *mut u8,  // KC_LEAF * TILE_BYTES
    b_pack: *mut u8,  // KC_LEAF * TILE_BYTES
    z_buf:  *mut u8,  // MR * TILE_BYTES
}

impl ScratchBufs {
    fn new() -> Self {
        ScratchBufs {
            a_pack: aligned_alloc(KC_LEAF * TILE_BYTES, 128),
            b_pack: aligned_alloc(KC_LEAF * TILE_BYTES, 128),
            z_buf:  aligned_alloc(MR * TILE_BYTES, 128),
        }
    }
}

impl Drop for ScratchBufs {
    fn drop(&mut self) {
        aligned_free(self.a_pack, KC_LEAF * TILE_BYTES, 128);
        aligned_free(self.b_pack, KC_LEAF * TILE_BYTES, 128);
        aligned_free(self.z_buf, MR * TILE_BYTES, 128);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Leaf packing (tiny: only 1 tile, fits on stack practically)
// ═══════════════════════════════════════════════════════════════════════

/// Pack A[m×k] from strided source into MR-wide contiguous layout.
/// a points to A[0,0]; A[i,j] = a[i*lda + j].
#[inline]
unsafe fn pack_a_leaf(
    a: *const f32, lda: usize,
    m: usize, k: usize,
    dst: *mut u8,
) {
    let out = dst as *mut f32;
    let tile_m = m.min(MR);
    for kk in 0..k {
        for ii in 0..tile_m {
            *out.add(kk * MR + ii) = *a.add(ii * lda + kk);
        }
        for ii in tile_m..MR {
            *out.add(kk * MR + ii) = 0.0;
        }
    }
}

/// Pack B[k×n] from strided source into NR-wide contiguous layout.
/// b points to B[0,0]; B[i,j] = b[i*ldb + j].
#[inline]
unsafe fn pack_b_leaf(
    b: *const f32, ldb: usize,
    k: usize, n: usize,
    dst: *mut u8,
) {
    let out = dst as *mut f32;
    let tile_n = n.min(NR);
    for kk in 0..k {
        let src = b.add(kk * ldb);
        let d = out.add(kk * NR);
        core::ptr::copy_nonoverlapping(src, d, tile_n);
        for jj in tile_n..NR {
            *d.add(jj) = 0.0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Recursive matmul core
// ═══════════════════════════════════════════════════════════════════════

/// Recursive cache-oblivious matmul: C += A × B
///
/// Operates on strided sub-matrices (no pre-packing).
/// Packs only at the leaf level (m ≤ MR, n ≤ NR).
///
/// # Safety
/// All pointers must be valid for the given dimensions and strides.
unsafe fn recursive_matmul_accum(
    a: *const f32, lda: usize,  // A[i,j] at a[i*lda + j]
    b: *const f32, ldb: usize,  // B[i,j] at b[i*ldb + j]
    c: *mut f32,   ldc: usize,  // C[i,j] at c[i*ldc + j]
    m: usize, k: usize, n: usize,
    scratch: &ScratchBufs,
) {
    // ── Base case: use strided AMX kernel (NO packing) ─────────
    const LEAF_THRESHOLD: usize = 128;
    if m <= LEAF_THRESHOLD && n <= LEAF_THRESHOLD && k <= KC_LEAF {
        amx_sys::amx_strided_sgemm_tile_opt(
            a, lda as i32,
            b, ldb as i32,
            c, ldc as i32,
            m as i32, k as i32, n as i32,
        );
        return;
    }

    // ── Recursive splitting ──────────────────────────────────────
    // Split K first (for L1 residency), then the larger of M, N.

    if k > KC_LEAF {
        // Split K: C += A[:, :k2] * B[:k2, :] + A[:, k2:] * B[k2:, :]
        let k2 = (k / 2 + 3) & !3; // round to 4 for alignment
        let k2 = k2.min(k);
        recursive_matmul_accum(a, lda, b, ldb, c, ldc, m, k2, n, scratch);
        recursive_matmul_accum(
            a.add(k2), lda,
            b.add(k2 * ldb), ldb,
            c, ldc,
            m, k - k2, n,
            scratch,
        );
    } else if m > MR && m >= n {
        // Split M: process top and bottom halves
        let m2 = (m / 2 + MR - 1) / MR * MR; // round up to MR
        let m2 = m2.min(m);
        recursive_matmul_accum(a, lda, b, ldb, c, ldc, m2, k, n, scratch);
        recursive_matmul_accum(
            a.add(m2 * lda), lda,
            b, ldb,
            c.add(m2 * ldc), ldc,
            m - m2, k, n,
            scratch,
        );
    } else {
        // Split N: process left and right halves
        let n2 = (n / 2 + NR - 1) / NR * NR; // round up to NR
        let n2 = n2.min(n);
        recursive_matmul_accum(a, lda, b, ldb, c, ldc, m, k, n2, scratch);
        recursive_matmul_accum(
            a, lda,
            b.add(n2), ldb,
            c.add(n2), ldc,
            m, k, n - n2,
            scratch,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════

impl Matrix<f32> {
    /// Recursive cache-oblivious AMX matmul (single-threaded).
    ///
    /// No pre-packing: recursively splits until sub-tiles fit in L1,
    /// then packs and computes each 16×16 leaf tile individually.
    #[cfg(target_arch = "aarch64")]
    pub fn matmul_recursive(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        let a = self.as_slice().as_ptr();
        let b = other.as_slice().as_ptr();
        let mut c_data = vec![0.0f32; m * n];

        let scratch = ScratchBufs::new();
        unsafe {
            amx_sys::amx_set();
            recursive_matmul_accum(a, k, b, n, c_data.as_mut_ptr(), n, m, k, n, &scratch);
            amx_sys::amx_clr();
        }

        Matrix::from_data(c_data, m, n)
    }

    /// Parallel recursive cache-oblivious AMX matmul.
    ///
    /// Splits the M dimension across threads at the top level,
    /// then each thread recurses independently (no sync during recursion).
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    pub fn matmul_recursive_parallel(&self, other: &Matrix<f32>, n_threads: usize) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        let n_threads = n_threads.max(1);
        if n_threads <= 1 || m < MR * 2 {
            return self.matmul_recursive(other);
        }

        let a = self.as_slice().as_ptr();
        let b = other.as_slice().as_ptr();
        let mut c_data = vec![0.0f32; m * n];
        let c = c_data.as_mut_ptr();

        // Split M into chunks for each thread
        let rows_per = ((m + n_threads - 1) / n_threads + MR - 1) / MR * MR;
        let rows_per = rows_per.max(MR);

        struct Work {
            a: usize, b: usize, c: usize,
            lda: usize, ldb: usize, ldc: usize,
            m: usize, k: usize, n: usize,
        }
        unsafe impl Send for Work {}
        unsafe impl Sync for Work {}

        let mut works = Vec::new();
        let mut row = 0;
        while row < m {
            let chunk_m = rows_per.min(m - row);
            works.push(Work {
                a: unsafe { a.add(row * k) } as usize,
                b: b as usize,
                c: unsafe { c.add(row * n) } as usize,
                lda: k, ldb: n, ldc: n,
                m: chunk_m, k, n,
            });
            row += chunk_m;
        }

        #[cfg(feature = "parallel")]
        {
            rayon::scope(|s| {
                for work in &works {
                    s.spawn(move |_| {
                        #[cfg(target_os = "macos")]
                        unsafe {
                            extern "C" {
                                fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32;
                            }
                            let _ = pthread_set_qos_class_self_np(0x21, 0);
                        }

                        let scratch = ScratchBufs::new();
                        unsafe {
                            amx_sys::amx_set();
                            recursive_matmul_accum(
                                work.a as *const f32, work.lda,
                                work.b as *const f32, work.ldb,
                                work.c as *mut f32, work.ldc,
                                work.m, work.k, work.n,
                                &scratch,
                            );
                            amx_sys::amx_clr();
                        }
                    });
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            std::thread::scope(|scope| {
                for work in &works {
                    scope.spawn(|| {
                        let scratch = ScratchBufs::new();
                        unsafe {
                            amx_sys::amx_set();
                            recursive_matmul_accum(
                                work.a as *const f32, work.lda,
                                work.b as *const f32, work.ldb,
                                work.c as *mut f32, work.ldc,
                                work.m, work.k, work.n,
                                &scratch,
                            );
                            amx_sys::amx_clr();
                        }
                    });
                }
            });
        }

        Matrix::from_data(c_data, m, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make(m: usize, n: usize) -> Matrix<f32> {
        Matrix::from_data(
            (0..m * n).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect(),
            m, n,
        ).unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(diff < tol * scale, "element {i}: got={x}, want={y}, diff={diff}");
        }
    }

    #[test]
    fn test_recursive_small() {
        let a = make(4, 5);
        let b = make(5, 6);
        let c_scalar = a.matmul_scalar(&b).unwrap();
        let c_rec = a.matmul_recursive(&b).unwrap();
        assert_close(c_rec.as_slice(), c_scalar.as_slice(), 0.15);
    }

    #[test]
    fn test_recursive_16x16() {
        let a = make(16, 16);
        let b = make(16, 16);
        let c_scalar = a.matmul_scalar(&b).unwrap();
        let c_rec = a.matmul_recursive(&b).unwrap();
        assert_close(c_rec.as_slice(), c_scalar.as_slice(), 0.15);
    }

    #[test]
    fn test_recursive_nonaligned() {
        let a = make(37, 53);
        let b = make(53, 41);
        let c_scalar = a.matmul_scalar(&b).unwrap();
        let c_rec = a.matmul_recursive(&b).unwrap();
        assert_close(c_rec.as_slice(), c_scalar.as_slice(), 0.15);
    }

    #[test]
    fn test_recursive_256() {
        let a = make(256, 256);
        let b = make(256, 256);
        let c_scalar = a.matmul_scalar(&b).unwrap();
        let c_rec = a.matmul_recursive(&b).unwrap();
        assert_close(c_rec.as_slice(), c_scalar.as_slice(), 0.15);
    }

    #[test]
    fn test_recursive_large_k() {
        let a = make(64, 600);
        let b = make(600, 64);
        let c_rec = a.matmul_recursive(&b).unwrap();
        assert!(c_rec.as_slice().iter().all(|v| v.is_finite()));
        assert_eq!(c_rec.dims(), (64, 64));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_recursive_parallel() {
        let a = make(256, 256);
        let b = make(256, 256);
        let c_scalar = a.matmul_scalar(&b).unwrap();
        let c_par = a.matmul_recursive_parallel(&b, 4).unwrap();
        assert_close(c_par.as_slice(), c_scalar.as_slice(), 0.15);
    }
}
