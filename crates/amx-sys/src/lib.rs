//! Low-level bindings to Apple's undocumented AMX (Apple Matrix eXtensions) instructions.
//!
//! This crate uses the **exact same inline assembly** as the reference C implementation
//! (`aarch64.h`). The C code is compiled by the system compiler (clang) via `build.rs`,
//! giving us the proven `AMX_OP_GPR` / `AMX_NOP_OP_IMM5` encoding that works on every
//! Apple Silicon chip.
//!
//! # Safety
//!
//! All instruction functions are unsafe because they execute undocumented CPU instructions
//! that are only available on Apple Silicon (M1/M2/M3/M4). Calling these on non-Apple
//! hardware will crash.
//!
//! # Usage
//!
//! ```no_run
//! use amx_sys::*;
//!
//! unsafe {
//!     amx_set();
//!     // ... load, compute, store ...
//!     amx_clr();
//! }
//! ```
//!
//! # Instruction encoding
//!
//! All AMX instructions live in the reserved A64 instruction space:
//!
//! ```text
//! .word (0x201000 + (op << 5) + reg_or_imm5)
//! ```
//!
//! The C compiler handles register allocation and the `0%1` encoding trick
//! that converts the GPR name to a 5-bit register index.

#![no_std]

extern "C" {
    pub fn amx_set();
    pub fn amx_clr();
    pub fn amx_ldx(operand: u64);
    pub fn amx_ldy(operand: u64);
    pub fn amx_stx(operand: u64);
    pub fn amx_sty(operand: u64);
    pub fn amx_ldz(operand: u64);
    pub fn amx_stz(operand: u64);
    pub fn amx_ldzi(operand: u64);
    pub fn amx_stzi(operand: u64);
    pub fn amx_extrx(operand: u64);
    pub fn amx_extry(operand: u64);
    pub fn amx_fma64(operand: u64);
    pub fn amx_fms64(operand: u64);
    pub fn amx_fma32(operand: u64);
    pub fn amx_fms32(operand: u64);
    pub fn amx_mac16(operand: u64);
    pub fn amx_fma16(operand: u64);
    pub fn amx_fms16(operand: u64);
    pub fn amx_vecint(operand: u64);
    pub fn amx_vecfp(operand: u64);
    pub fn amx_matint(operand: u64);
    pub fn amx_matfp(operand: u64);
    pub fn amx_genlut(operand: u64);

    // Runtime AMX availability probe (safe, catches SIGILL)
    pub fn amx_available() -> i32;

    // f32 micro-kernel helpers (batched, no per-iteration FFI overhead)
    pub fn amx_f32_zero_z();
    pub fn amx_f32_mac_tile(a_panel: *const u8, b_panel: *const u8, k: i32);
    pub fn amx_f32_store_z(dst: *mut u8, rows: i32);
    pub fn amx_f32_load_z(src: *const u8, rows: i32);
    pub fn amx_f32_tile_kernel(
        a_panel: *const u8, b_panel: *const u8,
        dst: *mut u8, k: i32, tile_m: i32,
    );
    pub fn amx_f32_tile_kernel_accum(
        a_panel: *const u8, b_panel: *const u8,
        dst: *mut u8, k: i32, tile_m: i32,
    );

    // Complete dot product: set, zero, accumulate, reduce, clr.
    // Single FFI call for the entire operation.
    pub fn amx_f32_dot(a: *const f32, b: *const f32, n: i32) -> f32;

    // NEON-accelerated functions (faster than AMX for small operations)

    /// NEON-vectorized A packing: significantly faster than scalar column-gather.
    pub fn neon_pack_a_tiles(
        a: *const f32, m: i32, k: i32,
        start_it: i32, end_it: i32, dst: *mut u8,
    );

    /// NEON small matrix multiply: M×K × K×N → M×N.
    /// Optimal for N ≤ 32, bypasses AMX entirely.
    pub fn neon_f32_matmul_small(
        a: *const f32, b: *const f32, c: *mut f32,
        m: i32, k: i32, n: i32,
    );

    /// NEON tiled matrix multiply with 8×8 tiling.
    /// Good for N ≤ 64.
    pub fn neon_f32_matmul_tiled(
        a: *const f32, b: *const f32, c: *mut f32,
        m: i32, k: i32, n: i32,
    );

    /// NEON f32 dot product — much faster than AMX for this operation.
    pub fn neon_f32_dot(a: *const f32, b: *const f32, n: i32) -> f32;

    /// GEBP A-panel packing with column-gather into MR=16 vectors.
    pub fn gebp_pack_a_panel(
        a: *const f32, lda: i32,
        i_start: i32, i_end: i32,
        k_start: i32, k_end: i32,
        dst: *mut f32,
    );
}

/// Returns `true` if AMX instructions are available on this CPU.
///
/// The result is probed once (via a SIGILL handler) and cached.
#[cfg(target_arch = "aarch64")]
pub fn is_amx_available() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};
    // 0 = unknown, 1 = available, 2 = unavailable
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let avail = unsafe { amx_available() } != 0;
            CACHE.store(if avail { 1 } else { 2 }, Ordering::Relaxed);
            avail
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn is_amx_available() -> bool {
    false
}

// The extern "C" functions above are the public API.
// Users call: unsafe { amx_sys::amx_set(); amx_sys::amx_ldx(op); ... }

// ---------------------------------------------------------------------------
// Operand construction helpers
// ---------------------------------------------------------------------------

/// Build an operand for load/store instructions.
///
/// Equivalent to the C macro:
/// ```c
/// #define PTR_ROW_FLAGS(ptr, row, flags) \
///     (((uint64_t)&*(ptr)) + (((uint64_t)((row) + (flags) * 64)) << 56))
/// ```
#[inline(always)]
pub fn ptr_row_flags(ptr: *const u8, row: u8, flags: u8) -> u64 {
    (ptr as u64) + (((row as u64) + (flags as u64) * 64) << 56)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptr_row_flags() {
        let buf = [0u8; 128];
        let op = ptr_row_flags(buf.as_ptr(), 3, 1);
        let ptr_bits = op & ((1u64 << 56) - 1);
        assert_eq!(ptr_bits, buf.as_ptr() as u64);
        assert_eq!((op >> 56) as u8, 67); // 3 + 1*64
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_set_clr() {
        if !is_amx_available() { return; }
        unsafe {
            amx_set();
            amx_clr();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_ldx_stx_roundtrip() {
        if !is_amx_available() { return; }
        let mut src = [0u8; 64];
        for i in 0..64 {
            src[i] = i as u8;
        }
        let mut dst = [0u8; 64];

        unsafe {
            amx_set();
            amx_ldx(ptr_row_flags(src.as_ptr(), 0, 0));
            amx_stx(ptr_row_flags(dst.as_mut_ptr(), 0, 0));
            amx_clr();
        }

        assert_eq!(src, dst);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fma32_basic() {
        if !is_amx_available() { return; }
        #[repr(align(128))]
        struct A([u8; 128]);

        let mut x_buf = A([0u8; 128]);
        let mut y_buf = A([0u8; 128]);
        let mut z_buf = A([0u8; 128]);

        for chunk in x_buf.0[..64].chunks_exact_mut(4) {
            chunk.copy_from_slice(&1.0f32.to_le_bytes());
        }
        for chunk in y_buf.0[..64].chunks_exact_mut(4) {
            chunk.copy_from_slice(&2.0f32.to_le_bytes());
        }

        unsafe {
            amx_set();
            amx_ldx(ptr_row_flags(x_buf.0.as_ptr(), 0, 0));
            amx_ldy(ptr_row_flags(y_buf.0.as_ptr(), 0, 0));
            amx_fma32(1u64 << 63); // vector product
            amx_stz(ptr_row_flags(z_buf.0.as_mut_ptr(), 0, 0));
            amx_clr();
        }

        for chunk in z_buf.0[..64].chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert_eq!(v, 2.0);
        }
    }
}
