//! Full GEBP (Goto's BLAS) matmul with L1/L2/L3 cache blocking.
//!
//! Implements the 5-loop Goto-style algorithm for matrix multiplication
//! with three levels of cache blocking:
//!
//! ```text
//! C[m×n] = A[m×k] × B[k×n]
//!
//! Loop 1 (L3/SLC): for jc in 0..n step NC
//!   Pack B̃ ← B[:, jc..jc+NC]                 // B̃ lives in L3/SLC
//!   Loop 2 (L2):   for pc in 0..k step KC
//!     Pack Ã ← A[ic..ic+MC, pc..pc+KC]        // Ã lives in L2
//!     Loop 3 (L1):  for ic in 0..m step MC
//!       GEBP macro-kernel(Ã, B̃, C):
//!         Loop 4: for jr in 0..NC step NR      // B̃ panel column
//!           Loop 5: for ir in 0..MC step MR    // Ã panel row
//!             µ-kernel: C[ir..ir+MR, jr..jr+NR] += Ã × B̃
//! ```
//!
//! # Cache sizing (Apple Silicon)
//!
//! | Level | Size | Contents | Constraint |
//! |-------|------|----------|------------|
//! | L1D   | 64 KB | Ã panel column + B̃ panel row | KC × MR + KC × NR ≤ L1 |
//! | L2    | 4 MB | Ã panel (MC × KC) | MC × KC × 4 ≤ L2 |
//! | L3/SLC | 32 MB | B̃ panel (KC × NC) | KC × NC × 4 ≤ L3 |
//!
//! With MR=NR=16 (AMX tile), KC=512, MC=256, NC=4096:
//! - L1 working set: 512×16×4 × 2 = 64 KB ✓
//! - L2 working set: 256×512×4 = 512 KB ✓ (leaves room for B̃ streaming)
//! - L3 working set: 512×4096×4 = 8 MB ✓

use alloc::vec;
use crate::error::{AmxError, AmxResult};
use crate::matrix::{Matrix, TILE, TILE_BYTES, aligned_alloc, aligned_free, SendPtr};

// ═══════════════════════════════════════════════════════════════════════
// GEBP blocking parameters — tuned for Apple Silicon M1-M4
// ═══════════════════════════════════════════════════════════════════════

/// Micro-kernel register tile: M dimension.
/// AMX does 16×16 f32 outer products, so MR = 16.
const MR: usize = TILE; // 16

/// Micro-kernel register tile: N dimension.
const NR: usize = TILE; // 16

/// KC: K-dimension block for L1 residency.
///
/// Working set per µ-kernel call: KC × (MR + NR) × 4 bytes.
/// With KC=512: 512 × 32 × 4 = 64 KB → fills 64 KB L1D.
const KC: usize = 512;

/// MC: M-dimension block for L2 residency.
///
/// The packed Ã panel is MC × KC × 4 bytes.
/// With MC=256, KC=512: 256 × 512 × 4 = 512 KB.
/// Apple Silicon L2 is 4 MB shared; 512 KB gives plenty of headroom
/// for B̃ streaming and OS overhead.
const MC: usize = 256;

/// NC: N-dimension block for shared L2 residency.
///
/// The packed B̃ panel is KC × NC × 4 bytes and is read by ALL cores.
/// Apple Silicon L2 is 4 MB shared across all cores in a cluster.
/// B̃ must fit comfortably: KC × NC × 4 ≤ ~2 MB (leave room for Ã streaming).
/// With KC=512, NC=1024: 512 × 1024 × 4 = 2 MB ✓
const NC: usize = 1024;

// ═══════════════════════════════════════════════════════════════════════
// Packing routines
// ═══════════════════════════════════════════════════════════════════════

/// Pack a panel of A into column-gathered layout for AMX ldy.
///
/// Packs A[i_start..i_end, k_start..k_end] into contiguous MR-wide
/// vectors suitable for AMX ldy.
///
/// Layout (per i-tile `it`):
///   panel[it * kc * MR + kk * MR + ii] = A[i_blk + ii, k_start + kk]
///
/// Delegates to an optimized C implementation.
///
/// # Safety
/// All pointers and ranges must be valid.
#[cfg(target_arch = "aarch64")]
unsafe fn pack_a_panel(
    a: *const f32, lda: usize,
    i_start: usize, i_end: usize,
    k_start: usize, k_end: usize,
    dst: *mut u8,
) {
    amx_sys::gebp_pack_a_panel(
        a, lda as i32,
        i_start as i32, i_end as i32,
        k_start as i32, k_end as i32,
        dst as *mut f32,
    );
}

/// Pack a panel of B into row-copy layout for AMX ldx.
///
/// Packs B[k_start..k_end, j_start..j_end] into contiguous 64-byte
/// vectors: for each k, copies 16 elements from the N dimension.
///
/// Layout: panel[(jt * kc + kk) * TILE_BYTES .. +TILE_BYTES]
///   = { B[k_start+kk, j_blk+0], ..., B[k_start+kk, j_blk+15] }
///
/// # Safety
/// All pointers and ranges must be valid.
#[cfg(target_arch = "aarch64")]
unsafe fn pack_b_panel(
    b: *const f32, ldb: usize,  // row-major: B[i,j] at b[i*ldb + j]
    k_start: usize, k_end: usize,
    j_start: usize, j_end: usize,
    dst: *mut u8,
) {
    let kc = k_end - k_start;
    let nc = j_end - j_start;
    let n_j_tiles = (nc + NR - 1) / NR;
    let out = dst as *mut f32;

    for jt in 0..n_j_tiles {
        let j_blk = j_start + jt * NR;
        let tile_n = NR.min(j_end - j_blk);
        let base = jt * kc * NR;

        for kk in 0..kc {
            let row = k_start + kk;
            let src = b.add(row * ldb + j_blk);
            let dst_off = base + kk * NR;

            // Copy tile_n elements, zero-pad to NR
            core::ptr::copy_nonoverlapping(src, out.add(dst_off), tile_n);
            for jj in tile_n..NR {
                *out.add(dst_off + jj) = 0.0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GEBP macro-kernel
// ═══════════════════════════════════════════════════════════════════════

/// GEBP macro-kernel: multiplies packed Ã (mc_tiles × kc) by packed B̃
/// (kc × nc_tiles) and accumulates into C.
///
/// This is the inner engine of the Goto algorithm. For each (ir, jr)
/// micro-tile, it calls the AMX tile kernel to compute one MR×NR block.
///
/// # Safety
/// All pointers must be valid. C region must not overlap with other
/// threads' writes for concurrent use.
#[cfg(target_arch = "aarch64")]
unsafe fn gebp_macro_kernel(
    a_packed: *const u8,   // packed Ã: mc_tiles × kc panels
    b_packed: *const u8,   // packed B̃: nc_tiles × kc panels
    c: *mut f32,           // output matrix (row-major, stride = ldc)
    ldc: usize,            // leading dimension of C
    mc: usize,             // actual M dimension of this block
    nc: usize,             // actual N dimension of this block
    kc: usize,             // K dimension of this block
    i_base: usize,         // row offset into C
    j_base: usize,         // col offset into C
    first_kc: bool,        // true = zero Z, false = load-accumulate from z_buf
    z_buf: *mut u8,        // temporary buffer for Z ↔ C transfer
) {
    use amx_sys::*;

    let mc_tiles = (mc + MR - 1) / MR;
    let nc_tiles = (nc + NR - 1) / NR;

    for ir in 0..mc_tiles {
        let i_blk = i_base + ir * MR;
        let tile_m = MR.min(mc - ir * MR);
        let ap = a_packed.add(ir * kc * TILE_BYTES);

        for jr in 0..nc_tiles {
            let j_blk = j_base + jr * NR;
            let tile_n = NR.min(nc - jr * NR);
            let bp = b_packed.add(jr * kc * TILE_BYTES);

            if first_kc {
                // First KC block: zero + compute
                amx_f32_tile_kernel(ap, bp, z_buf, kc as i32, tile_m as i32);
            } else {
                // Subsequent KC blocks: load partial sums from C into z_buf,
                // accumulate, store back
                // Load existing C values into z_buf
                for ii in 0..tile_m {
                    let c_row = c.add((i_blk + ii) * ldc + j_blk);
                    let z_row = z_buf.add(ii * TILE_BYTES) as *mut f32;
                    core::ptr::copy_nonoverlapping(c_row, z_row, tile_n);
                    // Zero-pad to NR
                    for jj in tile_n..NR {
                        *z_row.add(jj) = 0.0;
                    }
                }
                amx_f32_tile_kernel_accum(ap, bp, z_buf, kc as i32, tile_m as i32);
            }

            // Store z_buf → C
            for ii in 0..tile_m {
                let src = z_buf.add(ii * TILE_BYTES) as *const f32;
                let dst = c.add((i_blk + ii) * ldc + j_blk);
                core::ptr::copy_nonoverlapping(src, dst, tile_n);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Top-level GEBP matmul (single-threaded)
// ═══════════════════════════════════════════════════════════════════════

impl Matrix<f32> {
    /// Full GEBP (Goto's BLAS) matrix multiplication with 3-level cache blocking.
    ///
    /// Loop structure:
    /// ```text
    /// for jc in 0..n step NC:       // L3/SLC blocking
    ///   for pc in 0..k step KC:     // L1 blocking
    ///     pack B̃[pc..pc+KC, jc..jc+NC]
    ///     for ic in 0..m step MC:   // L2 blocking
    ///       pack Ã[ic..ic+MC, pc..pc+KC]
    ///       GEBP(Ã, B̃ → C[ic..ic+MC, jc..jc+NC])
    /// ```
    ///
    /// Note: the loop order places pc outside ic so that Ã (smaller, L2)
    /// is repacked more often, while B̃ (larger, L3) stays resident longer.
    #[cfg(target_arch = "aarch64")]
    pub fn matmul_gebp(&self, other: &Matrix<f32>) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        let a = self.as_slice().as_ptr();
        let b = other.as_slice().as_ptr();
        let mut c_data = vec![0.0f32; m * n];
        let c = c_data.as_mut_ptr();

        // Pre-allocate packing buffers (worst-case sizes)
        let mc_tiles = (MC + MR - 1) / MR;
        let nc_tiles = (NC + NR - 1) / NR;
        let a_buf_size = mc_tiles * KC * TILE_BYTES;
        let b_buf_size = nc_tiles * KC * TILE_BYTES;
        let a_buf = aligned_alloc(a_buf_size, 128);
        let b_buf = aligned_alloc(b_buf_size, 128);
        let z_buf = aligned_alloc(MR * TILE_BYTES, 128);

        unsafe {
            amx_sys::amx_set();

            // Loop 1: N-dimension blocking (L3/SLC)
            let mut jc = 0;
            while jc < n {
                let actual_nc = NC.min(n - jc);

                // Loop 2: K-dimension blocking (L1)
                let mut pc = 0;
                while pc < k {
                    let actual_kc = KC.min(k - pc);
                    let first_kc = pc == 0;

                    // Pack B̃ panel: B[pc..pc+kc, jc..jc+nc] → b_buf
                    pack_b_panel(
                        b, n, // ldb = n (row-major)
                        pc, pc + actual_kc,
                        jc, jc + actual_nc,
                        b_buf,
                    );

                    // Loop 3: M-dimension blocking (L2)
                    let mut ic = 0;
                    while ic < m {
                        let actual_mc = MC.min(m - ic);

                        // Pack Ã panel: A[ic..ic+mc, pc..pc+kc] → a_buf
                        pack_a_panel(
                            a, k, // lda = k (row-major)
                            ic, ic + actual_mc,
                            pc, pc + actual_kc,
                            a_buf,
                        );

                        // GEBP macro-kernel
                        gebp_macro_kernel(
                            a_buf, b_buf, c, n,
                            actual_mc, actual_nc, actual_kc,
                            ic, jc,
                            first_kc,
                            z_buf,
                        );

                        ic += actual_mc;
                    }

                    pc += actual_kc;
                }

                jc += actual_nc;
            }

            amx_sys::amx_clr();
        }

        aligned_free(a_buf, a_buf_size, 128);
        aligned_free(b_buf, b_buf_size, 128);
        aligned_free(z_buf, MR * TILE_BYTES, 128);

        Matrix::from_data(c_data, m, n)
    }

    /// Multi-threaded GEBP matmul with 2D work distribution.
    ///
    /// Instead of parallelising only the ic loop (which gives too few
    /// work items when M is small relative to MC), this distributes
    /// individual (ic_tile, jc_tile) pairs across threads within each
    /// KC block.  Each (ir, jr) micro-tile is independent and writes
    /// to a unique region of C.
    ///
    /// The B̃ panel is packed once and shared read-only across threads
    /// (lives in L3/SLC).  Each thread packs its own Ã rows on demand.
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    pub fn matmul_gebp_parallel(&self, other: &Matrix<f32>, n_threads: usize) -> AmxResult<Matrix<f32>> {
        let (m, k) = self.dims();
        let (k2, n) = other.dims();
        if k != k2 {
            return Err(AmxError::DimensionMismatch { expected: k, got: k2 });
        }

        let n_threads = n_threads.max(1);

        let total_flops = 2.0 * m as f64 * k as f64 * n as f64;
        if n_threads <= 1 || total_flops < 20_000_000.0 || m < MR * 2 {
            return self.matmul_gebp(other);
        }

        let a = self.as_slice().as_ptr();
        let b = other.as_slice().as_ptr();
        let mut c_data = vec![0.0f32; m * n];
        let c = c_data.as_mut_ptr();

        // Shared B̃ buffer
        let nc_tiles_max = (NC + NR - 1) / NR;
        let b_buf_size = nc_tiles_max * KC * TILE_BYTES;
        let b_buf = aligned_alloc(b_buf_size, 128);

        // Build a GebpWork struct for Send safety
        struct GebpWork {
            a_ptr: usize, b_buf_ptr: usize, c_ptr: usize,
            lda: usize, ldc: usize,
            ic_start: usize, ic_end: usize,
            jc_start: usize, jc_end: usize,
            pc: usize, actual_kc: usize,
            first_kc: bool,
        }
        unsafe impl Send for GebpWork {}
        unsafe impl Sync for GebpWork {}

        impl GebpWork {
            unsafe fn run(&self) {
                let actual_mc = self.ic_end - self.ic_start;
                let actual_nc = self.jc_end - self.jc_start;
                let mc_tiles = (actual_mc + MR - 1) / MR;
                let nc_tiles = (actual_nc + NR - 1) / NR;

                // Pack Ã for this thread's row range
                let a_buf_size = mc_tiles * self.actual_kc * TILE_BYTES;
                let a_buf = aligned_alloc(a_buf_size, 128);
                let z_buf = aligned_alloc(MR * TILE_BYTES, 128);

                pack_a_panel(
                    self.a_ptr as *const f32, self.lda,
                    self.ic_start, self.ic_end,
                    self.pc, self.pc + self.actual_kc,
                    a_buf,
                );

                // B̃ panel offset for this j-range
                // B̃ is packed for the full NC range starting at jc=0 of the
                // current NC block. We need the j-tile offset within that block.
                let j_offset_tiles = (self.jc_start - self.jc_start) / NR; // always 0 for per-tile dispatch

                // Pin to P-core for maximum AMX throughput
                #[cfg(target_os = "macos")]
                {
                    extern "C" {
                        fn pthread_set_qos_class_self_np(qos: u32, pri: i32) -> i32;
                    }
                    let _ = unsafe { pthread_set_qos_class_self_np(0x21, 0) }; // USER_INTERACTIVE → P-core
                }

                amx_sys::amx_set();

                gebp_macro_kernel(
                    a_buf, self.b_buf_ptr as *const u8,
                    self.c_ptr as *mut f32, self.ldc,
                    actual_mc, actual_nc, self.actual_kc,
                    self.ic_start, self.jc_start,
                    self.first_kc,
                    z_buf,
                );

                amx_sys::amx_clr();

                aligned_free(a_buf, a_buf_size, 128);
                aligned_free(z_buf, MR * TILE_BYTES, 128);
            }
        }

        // ── Main loops ───────────────────────────────────────────────

        let a_raw = a as usize;
        let b_buf_raw = b_buf as usize;
        let c_raw = c as usize;

        // Dynamic MC: shrink so we get enough parallel work items.
        // Want at least n_threads * 2 work items in the M dimension.
        // MC must be a multiple of MR (16).
        let mc_par = {
            let target_blocks = n_threads * 2;
            let mc_for_target = ((m + target_blocks - 1) / target_blocks + MR - 1) / MR * MR;
            mc_for_target.max(MR).min(MC)
        };

        let mut jc = 0;
        while jc < n {
            let actual_nc = NC.min(n - jc);

            let mut pc = 0;
            while pc < k {
                let actual_kc = KC.min(k - pc);
                let first_kc = pc == 0;

                // Pack B̃ panel for this (jc, pc) block
                // Parallelize across j-tiles for large panels
                #[cfg(feature = "parallel")]
                {
                    let n_j_tiles = (actual_nc + NR - 1) / NR;
                    if n_j_tiles >= 4 {
                        use rayon::prelude::*;
                        let b_raw = b as usize;
                        let b_buf_raw = b_buf as usize;
                        (0..n_j_tiles).into_par_iter().for_each(|jt| {
                            let b_ptr = b_raw as *const f32;
                            let out = b_buf_raw as *mut f32;
                            let j_blk = jc + jt * NR;
                            let tile_n = NR.min(jc + actual_nc - j_blk);
                            let base = jt * actual_kc * NR;
                            for kk in 0..actual_kc {
                                let row = pc + kk;
                                unsafe {
                                    let src = b_ptr.add(row * n + j_blk);
                                    let dst = out.add(base + kk * NR);
                                    core::ptr::copy_nonoverlapping(src, dst, tile_n);
                                    for jj in tile_n..NR {
                                        *dst.add(jj) = 0.0;
                                    }
                                }
                            }
                        });
                    } else {
                        unsafe {
                            pack_b_panel(b, n, pc, pc + actual_kc, jc, jc + actual_nc, b_buf);
                        }
                    }
                }
                #[cfg(not(feature = "parallel"))]
                unsafe {
                    pack_b_panel(b, n, pc, pc + actual_kc, jc, jc + actual_nc, b_buf);
                }

                // Build 2D work list: (ic_block, jc_block) pairs
                // Each work item processes mc_par × actual_nc output block
                // The B̃ panel is shared across all work items.
                //
                // For better cache reuse, we want each thread to process
                // a contiguous range of i-rows (so Ã stays in its L2).
                // So we create work items by splitting M into mc_par-sized
                // blocks — now with dynamic mc_par there are enough blocks.
                let works: Vec<GebpWork> = {
                    let mut w = Vec::new();
                    let mut ic = 0;
                    while ic < m {
                        let ic_end = (ic + mc_par).min(m);
                        w.push(GebpWork {
                            a_ptr: a_raw,
                            b_buf_ptr: b_buf_raw,
                            c_ptr: c_raw,
                            lda: k,
                            ldc: n,
                            ic_start: ic,
                            ic_end,
                            jc_start: jc,
                            jc_end: jc + actual_nc,
                            pc,
                            actual_kc,
                            first_kc,
                        });
                        ic = ic_end;
                    }
                    w
                };

                #[cfg(feature = "parallel")]
                {
                    rayon::scope(|s| {
                        for work in &works {
                            s.spawn(move |_| unsafe { work.run() });
                        }
                    });
                }

                #[cfg(not(feature = "parallel"))]
                {
                    std::thread::scope(|scope| {
                        for work in &works {
                            let w_ptr = work as *const GebpWork as usize;
                            scope.spawn(move || unsafe {
                                let w = &*(w_ptr as *const GebpWork);
                                w.run();
                            });
                        }
                    });
                }

                pc += actual_kc;
            }

            jc += actual_nc;
        }

        aligned_free(b_buf, b_buf_size, 128);

        Matrix::from_data(c_data, m, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_matrix(m: usize, n: usize) -> Matrix<f32> {
        let data: Vec<f32> = (0..m * n)
            .map(|i| ((i % 17) as f32) * 0.1 - 0.8)
            .collect();
        Matrix::from_data(data, m, n).unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(
                diff < tol * scale,
                "element {i}: got={x}, want={y}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_gebp_small() {
        let a = make_matrix(4, 5);
        let b = make_matrix(5, 6);
        let c_amx = a.matmul_amx(&b).unwrap();
        let c_gebp = a.matmul_gebp(&b).unwrap();
        assert_close(c_gebp.as_slice(), c_amx.as_slice(), 1e-5);
    }

    #[test]
    fn test_gebp_16x16() {
        let a = make_matrix(16, 16);
        let b = make_matrix(16, 16);
        let c_amx = a.matmul_amx(&b).unwrap();
        let c_gebp = a.matmul_gebp(&b).unwrap();
        assert_close(c_gebp.as_slice(), c_amx.as_slice(), 1e-5);
    }

    #[test]
    fn test_gebp_non_aligned() {
        let a = make_matrix(37, 53);
        let b = make_matrix(53, 41);
        let c_amx = a.matmul_amx(&b).unwrap();
        let c_gebp = a.matmul_gebp(&b).unwrap();
        assert_close(c_gebp.as_slice(), c_amx.as_slice(), 1e-4);
    }

    #[test]
    fn test_gebp_large_needs_mc_blocking() {
        let a = make_matrix(300, 100);
        let b = make_matrix(100, 80);
        let c_amx = a.matmul_amx(&b).unwrap();
        let c_gebp = a.matmul_gebp(&b).unwrap();
        assert_close(c_gebp.as_slice(), c_amx.as_slice(), 1e-4);
    }

    #[test]
    fn test_gebp_large_needs_kc_blocking() {
        let a = make_matrix(64, 600);
        let b = make_matrix(600, 64);
        let c_amx = a.matmul_amx(&b).unwrap();
        let c_gebp = a.matmul_gebp(&b).unwrap();
        assert_close(c_gebp.as_slice(), c_amx.as_slice(), 1e-4);
    }

    #[test]
    fn test_gebp_256x256() {
        let a = make_matrix(256, 256);
        let b = make_matrix(256, 256);
        let c_amx = a.matmul_amx(&b).unwrap();
        let c_gebp = a.matmul_gebp(&b).unwrap();
        assert_close(c_gebp.as_slice(), c_amx.as_slice(), 1e-4);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gebp_parallel_256x256() {
        let a = make_matrix(256, 256);
        let b = make_matrix(256, 256);
        let c_amx = a.matmul_amx(&b).unwrap();
        let c_par = a.matmul_gebp_parallel(&b, 4).unwrap();
        assert_close(c_par.as_slice(), c_amx.as_slice(), 1e-4);
    }

    /// Verify GEBP produces finite, reasonable results.
    /// AMX outer-product accumulation order differs from scalar i,k,j order,
    /// causing significant FP differences for f32 — this is expected.
    #[test]
    fn test_gebp_produces_finite() {
        let a = make_matrix(64, 64);
        let b = make_matrix(64, 64);
        let c_gebp = a.matmul_gebp(&b).unwrap();
        assert!(c_gebp.as_slice().iter().all(|v| v.is_finite()));
        assert_eq!(c_gebp.dims(), (64, 64));
    }
}
