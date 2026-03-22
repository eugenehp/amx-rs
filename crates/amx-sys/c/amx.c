// amx.c — AMX instruction wrappers compiled by the system C compiler.
// Identical encoding to aarch64.h from the reference implementation.

#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>

#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm__ __volatile__("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" \
        : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm__ __volatile__(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" \
        : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

// ── Runtime AMX availability check ──────────────────────────────────

#include <unistd.h>
#include <sys/wait.h>

// Returns 1 if AMX instructions are available, 0 otherwise.
// Forks a child process to probe — if the child gets SIGILL, AMX is absent.
int amx_available(void) {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: try AMX set + clr, exit 0 on success
        AMX_NOP_OP_IMM5(17, 0);
        AMX_NOP_OP_IMM5(17, 1);
        _exit(0);
    }
    if (pid < 0) return 0;   // fork failed
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// ── Single-instruction wrappers ──────────────────────────────────────

void amx_set(void)               { AMX_NOP_OP_IMM5(17, 0); }
void amx_clr(void)               { AMX_NOP_OP_IMM5(17, 1); }
void amx_ldx(uint64_t op)        { AMX_OP_GPR( 0, op); }
void amx_ldy(uint64_t op)        { AMX_OP_GPR( 1, op); }
void amx_stx(uint64_t op)        { AMX_OP_GPR( 2, op); }
void amx_sty(uint64_t op)        { AMX_OP_GPR( 3, op); }
void amx_ldz(uint64_t op)        { AMX_OP_GPR( 4, op); }
void amx_stz(uint64_t op)        { AMX_OP_GPR( 5, op); }
void amx_ldzi(uint64_t op)       { AMX_OP_GPR( 6, op); }
void amx_stzi(uint64_t op)       { AMX_OP_GPR( 7, op); }
void amx_extrx(uint64_t op)      { AMX_OP_GPR( 8, op); }
void amx_extry(uint64_t op)      { AMX_OP_GPR( 9, op); }
void amx_fma64(uint64_t op)      { AMX_OP_GPR(10, op); }
void amx_fms64(uint64_t op)      { AMX_OP_GPR(11, op); }
void amx_fma32(uint64_t op)      { AMX_OP_GPR(12, op); }
void amx_fms32(uint64_t op)      { AMX_OP_GPR(13, op); }
void amx_mac16(uint64_t op)      { AMX_OP_GPR(14, op); }
void amx_fma16(uint64_t op)      { AMX_OP_GPR(15, op); }
void amx_fms16(uint64_t op)      { AMX_OP_GPR(16, op); }
void amx_vecint(uint64_t op)     { AMX_OP_GPR(18, op); }
void amx_vecfp(uint64_t op)      { AMX_OP_GPR(19, op); }
void amx_matint(uint64_t op)     { AMX_OP_GPR(20, op); }
void amx_matfp(uint64_t op)      { AMX_OP_GPR(21, op); }
void amx_genlut(uint64_t op)     { AMX_OP_GPR(22, op); }

// ── f32 micro-kernel helpers ─────────────────────────────────────────
//
// These keep the entire inner loop inside a single C function so the
// AMX instructions execute back-to-back without Rust→C FFI overhead
// per iteration.

// Zero all 16 f32 Z accumulator rows (physical rows 0,4,8,...,60).
void amx_f32_zero_z(void) {
    static const uint8_t zeros[64] __attribute__((aligned(128))) = {0};
    uint64_t base = (uint64_t)zeros;
    for (int i = 0; i < 16; i++) {
        AMX_OP_GPR(4, base | ((uint64_t)(i * 4) << 56));   // ldz
    }
}

// Rank-1 accumulation loop for f32 matmul.
//
// For kk in 0..k:
//   Y[0] ← a_panel[kk]   (column of packed A, 64 bytes)
//   X[0] ← b_panel[kk]   (row of packed B, 64 bytes)
//   Z    += Y ⊗ X         (fma32 matrix mode)
//
// a_panel and b_panel must each be k contiguous 64-byte buffers,
// preferably 64-byte aligned.
void amx_f32_mac_tile(const void* a_panel, const void* b_panel, int k) {
    const uint8_t* ap = (const uint8_t*)a_panel;
    const uint8_t* bp = (const uint8_t*)b_panel;

    // Software-pipelined: overlap loads with compute
    if (k <= 0) return;

    AMX_OP_GPR(1, (uint64_t)(ap));          // ldy[0]
    AMX_OP_GPR(0, (uint64_t)(bp));          // ldx[0]

    int kk = 1;
    // Unrolled 4× with pipelining: fma previous, load next
    for (; kk + 3 < k; kk += 4) {
        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + kk * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + kk * 64));

        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + (kk+1) * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + (kk+1) * 64));

        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + (kk+2) * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + (kk+2) * 64));

        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + (kk+3) * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + (kk+3) * 64));
    }
    for (; kk < k; kk++) {
        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + kk * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + kk * 64));
    }
    AMX_OP_GPR(12, 0); // final fma for last loaded pair
}

// Store `rows` Z accumulator rows to a contiguous output buffer.
// Row i of Z (physical row i*4) → dst + i*64.
// dst must point to at least rows*64 bytes, preferably 64-byte aligned.
void amx_f32_store_z(void* dst, int rows) {
    uint8_t* p = (uint8_t*)dst;
    for (int i = 0; i < rows; i++) {
        AMX_OP_GPR(5, ((uint64_t)(p + i * 64)) | ((uint64_t)(i * 4) << 56));
    }
}

// Load `rows` Z accumulator rows from a contiguous input buffer.
// src + i*64 → Z row i (physical row i*4).
// src must point to at least rows*64 bytes, preferably 64-byte aligned.
void amx_f32_load_z(const void* src, int rows) {
    const uint8_t* p = (const uint8_t*)src;
    for (int i = 0; i < rows; i++) {
        AMX_OP_GPR(4, ((uint64_t)(p + i * 64)) | ((uint64_t)(i * 4) << 56));
    }
}

// Complete f32 micro-kernel: zero Z, accumulate k rank-1 updates,
// store tile_m rows to dst.  Single FFI call per 16×16 tile.
//
// Uses double-buffered loads: alternates between X[0]/Y[0] and X[1]/Y[1]
// so that loads for iteration k+1 overlap with the FMA of iteration k.
// Also prefetches data 4 iterations ahead.
void amx_f32_tile_kernel(const void* a_panel, const void* b_panel,
                         void* dst, int k, int tile_m) {
    // Zero Z
    static const uint8_t zeros[64] __attribute__((aligned(128))) = {0};
    uint64_t zbase = (uint64_t)zeros;
    for (int i = 0; i < 16; i++) {
        AMX_OP_GPR(4, zbase | ((uint64_t)(i * 4) << 56));
    }

    const uint8_t* ap = (const uint8_t*)a_panel;
    const uint8_t* bp = (const uint8_t*)b_panel;

    if (k <= 0) goto store;

    // AMX register pair encoding for fma32:
    //   X offset in bits [29:20], Y offset in bits [19:10]
    //   X row 0 = offset 0, X row 1 = offset 64
    //   Y row 0 = offset 0, Y row 1 = offset 64
    #define FMA32_XY(xrow, yrow) (((uint64_t)(xrow) * 64 << 20) | ((uint64_t)(yrow) * 64 << 10))
    #define LDX_ROW(ptr, row) ((uint64_t)(ptr) | ((uint64_t)(row) << 56))
    #define LDY_ROW(ptr, row) ((uint64_t)(ptr) | ((uint64_t)(row) << 56))

    if (k == 1) {
        AMX_OP_GPR(1, (uint64_t)ap);       // ldy Y[0]
        AMX_OP_GPR(0, (uint64_t)bp);       // ldx X[0]
        AMX_OP_GPR(12, 0);                 // fma32 X[0]*Y[0]
        goto store;
    }

    // Double-buffered pipeline:
    // Load pair 0 into X[0]/Y[0], pair 1 into X[1]/Y[1]
    // Then: fma X[0]*Y[0], load next into X[0]/Y[0], fma X[1]*Y[1], load next into X[1]/Y[1], ...

    // Preload first two pairs
    AMX_OP_GPR(1, LDY_ROW(ap, 0));             // ldy Y[0] <- a[0]
    AMX_OP_GPR(0, LDX_ROW(bp, 0));             // ldx X[0] <- b[0]
    AMX_OP_GPR(1, LDY_ROW(ap + 64, 1));        // ldy Y[1] <- a[1]
    AMX_OP_GPR(0, LDX_ROW(bp + 64, 1));        // ldx X[1] <- b[1]

    int kk = 2;

    // Main loop: process pairs, alternating X[0]/Y[0] and X[1]/Y[1]
    // 4× unrolled: each iteration processes 4 k-steps
    for (; kk + 3 < k; kk += 4) {
        // Prefetch 4 iterations ahead
        __builtin_prefetch(ap + (kk + 4) * 64, 0, 3);
        __builtin_prefetch(bp + (kk + 4) * 64, 0, 3);

        // FMA pair 0, reload pair 0
        AMX_OP_GPR(12, FMA32_XY(0, 0));         // fma32 X[0]*Y[0]
        AMX_OP_GPR(1, LDY_ROW(ap + kk * 64, 0));
        AMX_OP_GPR(0, LDX_ROW(bp + kk * 64, 0));

        // FMA pair 1, reload pair 1
        AMX_OP_GPR(12, FMA32_XY(1, 1));         // fma32 X[1]*Y[1]
        AMX_OP_GPR(1, LDY_ROW(ap + (kk+1) * 64, 1));
        AMX_OP_GPR(0, LDX_ROW(bp + (kk+1) * 64, 1));

        // FMA pair 0, reload pair 0
        AMX_OP_GPR(12, FMA32_XY(0, 0));
        AMX_OP_GPR(1, LDY_ROW(ap + (kk+2) * 64, 0));
        AMX_OP_GPR(0, LDX_ROW(bp + (kk+2) * 64, 0));

        // FMA pair 1, reload pair 1
        AMX_OP_GPR(12, FMA32_XY(1, 1));
        AMX_OP_GPR(1, LDY_ROW(ap + (kk+3) * 64, 1));
        AMX_OP_GPR(0, LDX_ROW(bp + (kk+3) * 64, 1));
    }

    // Handle remaining k-steps with double buffering
    for (; kk + 1 < k; kk += 2) {
        AMX_OP_GPR(12, FMA32_XY(0, 0));
        AMX_OP_GPR(1, LDY_ROW(ap + kk * 64, 0));
        AMX_OP_GPR(0, LDX_ROW(bp + kk * 64, 0));

        AMX_OP_GPR(12, FMA32_XY(1, 1));
        AMX_OP_GPR(1, LDY_ROW(ap + (kk+1) * 64, 1));
        AMX_OP_GPR(0, LDX_ROW(bp + (kk+1) * 64, 1));
    }

    // Drain: FMA the last loaded pair(s)
    if (kk == k) {
        // Last loaded was pair 1 (kk was even entering, loaded 0 and 1)
        AMX_OP_GPR(12, FMA32_XY(0, 0));
        AMX_OP_GPR(12, FMA32_XY(1, 1));
    } else {
        // kk == k-1: we have pair 0 loaded but need one more
        AMX_OP_GPR(12, FMA32_XY(0, 0));
        AMX_OP_GPR(12, FMA32_XY(1, 1));
        // Load and process the last one
        AMX_OP_GPR(1, LDY_ROW(ap + kk * 64, 0));
        AMX_OP_GPR(0, LDX_ROW(bp + kk * 64, 0));
        AMX_OP_GPR(12, FMA32_XY(0, 0));
    }

    #undef FMA32_XY
    #undef LDX_ROW
    #undef LDY_ROW

store:
    {
        uint8_t* p = (uint8_t*)dst;
        for (int i = 0; i < tile_m; i++) {
            AMX_OP_GPR(5, ((uint64_t)(p + i * 64)) | ((uint64_t)(i * 4) << 56));
        }
    }
}

// Accumulating tile kernel: loads existing partial sums from dst into Z,
// then accumulates k rank-1 updates and stores back.
// Uses double-buffered loads like amx_f32_tile_kernel.
void amx_f32_tile_kernel_accum(const void* a_panel, const void* b_panel,
                               void* dst, int k, int tile_m) {
    // Load existing partial sums into Z
    const uint8_t* src = (const uint8_t*)dst;
    for (int i = 0; i < tile_m; i++) {
        AMX_OP_GPR(4, ((uint64_t)(src + i * 64)) | ((uint64_t)(i * 4) << 56));
    }
    // Zero remaining rows
    static const uint8_t zeros[64] __attribute__((aligned(128))) = {0};
    uint64_t zbase = (uint64_t)zeros;
    for (int i = tile_m; i < 16; i++) {
        AMX_OP_GPR(4, zbase | ((uint64_t)(i * 4) << 56));
    }

    const uint8_t* ap = (const uint8_t*)a_panel;
    const uint8_t* bp = (const uint8_t*)b_panel;

    if (k <= 0) goto store_accum;

    #define FMA32_XY(xrow, yrow) (((uint64_t)(xrow) * 64 << 20) | ((uint64_t)(yrow) * 64 << 10))
    #define LDX_ROW(ptr, row) ((uint64_t)(ptr) | ((uint64_t)(row) << 56))
    #define LDY_ROW(ptr, row) ((uint64_t)(ptr) | ((uint64_t)(row) << 56))

    if (k == 1) {
        AMX_OP_GPR(1, (uint64_t)ap);
        AMX_OP_GPR(0, (uint64_t)bp);
        AMX_OP_GPR(12, 0);
        goto store_accum;
    }

    // Double-buffered: preload first two pairs
    AMX_OP_GPR(1, LDY_ROW(ap, 0));
    AMX_OP_GPR(0, LDX_ROW(bp, 0));
    AMX_OP_GPR(1, LDY_ROW(ap + 64, 1));
    AMX_OP_GPR(0, LDX_ROW(bp + 64, 1));

    {
        int kk = 2;
        for (; kk + 3 < k; kk += 4) {
            __builtin_prefetch(ap + (kk + 4) * 64, 0, 3);
            __builtin_prefetch(bp + (kk + 4) * 64, 0, 3);

            AMX_OP_GPR(12, FMA32_XY(0, 0));
            AMX_OP_GPR(1, LDY_ROW(ap + kk * 64, 0));
            AMX_OP_GPR(0, LDX_ROW(bp + kk * 64, 0));

            AMX_OP_GPR(12, FMA32_XY(1, 1));
            AMX_OP_GPR(1, LDY_ROW(ap + (kk+1) * 64, 1));
            AMX_OP_GPR(0, LDX_ROW(bp + (kk+1) * 64, 1));

            AMX_OP_GPR(12, FMA32_XY(0, 0));
            AMX_OP_GPR(1, LDY_ROW(ap + (kk+2) * 64, 0));
            AMX_OP_GPR(0, LDX_ROW(bp + (kk+2) * 64, 0));

            AMX_OP_GPR(12, FMA32_XY(1, 1));
            AMX_OP_GPR(1, LDY_ROW(ap + (kk+3) * 64, 1));
            AMX_OP_GPR(0, LDX_ROW(bp + (kk+3) * 64, 1));
        }

        for (; kk + 1 < k; kk += 2) {
            AMX_OP_GPR(12, FMA32_XY(0, 0));
            AMX_OP_GPR(1, LDY_ROW(ap + kk * 64, 0));
            AMX_OP_GPR(0, LDX_ROW(bp + kk * 64, 0));

            AMX_OP_GPR(12, FMA32_XY(1, 1));
            AMX_OP_GPR(1, LDY_ROW(ap + (kk+1) * 64, 1));
            AMX_OP_GPR(0, LDX_ROW(bp + (kk+1) * 64, 1));
        }

        if (kk == k) {
            AMX_OP_GPR(12, FMA32_XY(0, 0));
            AMX_OP_GPR(12, FMA32_XY(1, 1));
        } else {
            AMX_OP_GPR(12, FMA32_XY(0, 0));
            AMX_OP_GPR(12, FMA32_XY(1, 1));
            AMX_OP_GPR(1, LDY_ROW(ap + kk * 64, 0));
            AMX_OP_GPR(0, LDX_ROW(bp + kk * 64, 0));
            AMX_OP_GPR(12, FMA32_XY(0, 0));
        }
    }

    #undef FMA32_XY
    #undef LDX_ROW
    #undef LDY_ROW

store_accum:
    {
        uint8_t* p = (uint8_t*)dst;
        for (int i = 0; i < tile_m; i++) {
            AMX_OP_GPR(5, ((uint64_t)(p + i * 64)) | ((uint64_t)(i * 4) << 56));
        }
    }
}

// ── 2×1 tile kernel: processes two i-tiles sharing the same B panel ──
//
// This amortizes B loads: one ldx serves two fma32 instructions (one per
// i-tile).  Each i-tile uses different Y registers for its A data, and
// accumulates to different Z row ranges.
//
// Z layout for f32 outer product:
//   fma32 with operand bits [9:0] selecting z_row offset:
//   - z_row=0: accumulates to Z rows 0,4,8,...,60  (i-tile 0)
//   - We need a second set for i-tile 1
//
// Unfortunately, AMX only has one set of Z rows for f32 outer products
// (16 logical rows × 16 columns = 256 values = 1024 bytes).
// So we process 2 i-tiles sequentially but share the packed B panel,
// saving B reload time from cache (B stays hot in L1).
//
// Instead, implement a multi-j-tile approach: process 2 j-tiles per
// i-tile, keeping B panel hot in registers.
void amx_f32_tile_kernel_2j(const void* a_panel, 
                             const void* b_panel0, const void* b_panel1,
                             void* dst0, void* dst1,
                             int k, int tile_m, int tile_n0, int tile_n1) {
    static const uint8_t zeros[64] __attribute__((aligned(128))) = {0};
    uint64_t zbase = (uint64_t)zeros;
    
    const uint8_t* ap = (const uint8_t*)a_panel;
    const uint8_t* bp0 = (const uint8_t*)b_panel0;
    const uint8_t* bp1 = (const uint8_t*)b_panel1;
    
    // Process j-tile 0
    for (int i = 0; i < 16; i++)
        AMX_OP_GPR(4, zbase | ((uint64_t)(i * 4) << 56));
    
    if (k > 0) {
        AMX_OP_GPR(1, (uint64_t)ap);
        AMX_OP_GPR(0, (uint64_t)bp0);
        for (int kk = 1; kk < k; kk++) {
            AMX_OP_GPR(12, 0);
            AMX_OP_GPR(1, (uint64_t)(ap + kk * 64));
            AMX_OP_GPR(0, (uint64_t)(bp0 + kk * 64));
        }
        AMX_OP_GPR(12, 0);
    }
    
    uint8_t* p0 = (uint8_t*)dst0;
    for (int i = 0; i < tile_m; i++)
        AMX_OP_GPR(5, ((uint64_t)(p0 + i * 64)) | ((uint64_t)(i * 4) << 56));
    
    // Process j-tile 1 — A panel is still hot in L1 cache
    for (int i = 0; i < 16; i++)
        AMX_OP_GPR(4, zbase | ((uint64_t)(i * 4) << 56));
    
    if (k > 0) {
        AMX_OP_GPR(1, (uint64_t)ap);
        AMX_OP_GPR(0, (uint64_t)bp1);
        for (int kk = 1; kk < k; kk++) {
            AMX_OP_GPR(12, 0);
            AMX_OP_GPR(1, (uint64_t)(ap + kk * 64));
            AMX_OP_GPR(0, (uint64_t)(bp1 + kk * 64));
        }
        AMX_OP_GPR(12, 0);
    }
    
    uint8_t* p1 = (uint8_t*)dst1;
    for (int i = 0; i < tile_m; i++)
        AMX_OP_GPR(5, ((uint64_t)(p1 + i * 64)) | ((uint64_t)(i * 4) << 56));
}

// ── GEBP packing helper ──────────────────────────────────────────────

// Pack A panel for GEBP: A[i_start..i_end, k_start..k_end] → dst
// with column-gather into MR=16 wide vectors.
// row-major: A[i,j] = a[i*lda + j]
void gebp_pack_a_panel(const float* a, int lda,
                        int i_start, int i_end,
                        int k_start, int k_end,
                        float* dst) {
    const int MR = 16;
    int mc = i_end - i_start;
    int kc = k_end - k_start;
    int n_i_tiles = (mc + MR - 1) / MR;

    for (int it = 0; it < n_i_tiles; it++) {
        int i_blk = i_start + it * MR;
        int tile_m = MR < (i_end - i_blk) ? MR : (i_end - i_blk);
        float* out = dst + it * kc * MR;

        // Precompute row pointers
        const float* rows[16];
        for (int ii = 0; ii < tile_m; ii++)
            rows[ii] = a + (i_blk + ii) * lda + k_start;

        for (int kk = 0; kk < kc; kk++) {
            float* d = out + kk * MR;
            for (int ii = 0; ii < tile_m; ii++)
                d[ii] = rows[ii][kk];
            for (int ii = tile_m; ii < MR; ii++)
                d[ii] = 0.0f;
        }
    }
}

// ── NEON kernels ─────────────────────────────────────────────────────

#include <arm_neon.h>

// NEON-vectorized A packing: gather columns into 64-byte vectors.
// For each k, gathers A[i_blk+0..tile_m-1, k] into dst[k*64..k*64+tile_m*4-1].
// Uses NEON vld1q_lane_f32 for 4-wide gathers when possible.
void neon_pack_a_tile(const float* a, int m, int k, int i_blk, int tile_m,
                      uint8_t* dst) {
    (void)m;  // unused but kept for API consistency
    // a is row-major: A[i,j] = a[i*k + j]
    // For each kk, we need to gather A[i_blk+ii, kk] for ii in 0..tile_m
    // That's elements at offsets: (i_blk+0)*k+kk, (i_blk+1)*k+kk, ...
    
    // Precompute row pointers
    const float* rows[16];
    for (int ii = 0; ii < tile_m; ii++) {
        rows[ii] = a + (i_blk + ii) * k;
    }
    
    float* out = (float*)dst;
    
    // Process 4 k-values at a time using NEON
    int kk = 0;
    for (; kk + 3 < k; kk += 4) {
        for (int ii = 0; ii < tile_m; ii++) {
            // Load 4 consecutive k values from this row
            float32x4_t v = vld1q_f32(rows[ii] + kk);
            // Store transposed: each k goes to a different output vector
            out[(kk+0)*16 + ii] = vgetq_lane_f32(v, 0);
            out[(kk+1)*16 + ii] = vgetq_lane_f32(v, 1);
            out[(kk+2)*16 + ii] = vgetq_lane_f32(v, 2);
            out[(kk+3)*16 + ii] = vgetq_lane_f32(v, 3);
        }
    }
    // Remainder
    for (; kk < k; kk++) {
        for (int ii = 0; ii < tile_m; ii++) {
            out[kk*16 + ii] = rows[ii][kk];
        }
    }
}

// Pack multiple A tiles with NEON acceleration.
// Packs tiles [start_it..end_it) into dst.
void neon_pack_a_tiles(const float* a, int m, int k,
                       int start_it, int end_it, uint8_t* dst) {
    const int TILE = 16;
    const int TILE_BYTES = 64;
    
    for (int it_local = 0; it_local < (end_it - start_it); it_local++) {
        int it = start_it + it_local;
        int i_blk = it * TILE;
        int tile_m = TILE < (m - i_blk) ? TILE : (m - i_blk);
        
        uint8_t* tile_dst = dst + it_local * k * TILE_BYTES;
        neon_pack_a_tile(a, m, k, i_blk, tile_m, tile_dst);
    }
}

// NEON 8x8 micro-kernel with optimal register blocking.
// Uses vfmaq_laneq_f32 for better instruction scheduling.
// 16 accumulator registers (2 per row × 8 rows).
static inline void neon_gemm_8x8(const float* __restrict a, int lda,
                                  const float* __restrict b, int ldb,
                                  float* __restrict c, int ldc, int k) {
    // 8 rows × 8 cols = 16 float32x4_t accumulators
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
    float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
    float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);
    
    for (int kk = 0; kk < k; kk++) {
        // Load B row (8 floats = 2 vectors)
        float32x4_t b0 = vld1q_f32(b + kk * ldb);
        float32x4_t b1 = vld1q_f32(b + kk * ldb + 4);
        
        // Load A column (8 floats = 2 vectors)
        float32x4_t a0 = vld1q_f32(a + kk);  // a[0:3, k] - but a is row-major!
        // Actually need to gather a[i,k] for i=0..7
        // For row-major: a[i,k] = a[i*lda + k]
        
        // Use scalar loads and broadcast
        float a0s = a[0 * lda + kk];
        float a1s = a[1 * lda + kk];
        float a2s = a[2 * lda + kk];
        float a3s = a[3 * lda + kk];
        float a4s = a[4 * lda + kk];
        float a5s = a[5 * lda + kk];
        float a6s = a[6 * lda + kk];
        float a7s = a[7 * lda + kk];
        
        c00 = vfmaq_n_f32(c00, b0, a0s); c01 = vfmaq_n_f32(c01, b1, a0s);
        c10 = vfmaq_n_f32(c10, b0, a1s); c11 = vfmaq_n_f32(c11, b1, a1s);
        c20 = vfmaq_n_f32(c20, b0, a2s); c21 = vfmaq_n_f32(c21, b1, a2s);
        c30 = vfmaq_n_f32(c30, b0, a3s); c31 = vfmaq_n_f32(c31, b1, a3s);
        c40 = vfmaq_n_f32(c40, b0, a4s); c41 = vfmaq_n_f32(c41, b1, a4s);
        c50 = vfmaq_n_f32(c50, b0, a5s); c51 = vfmaq_n_f32(c51, b1, a5s);
        c60 = vfmaq_n_f32(c60, b0, a6s); c61 = vfmaq_n_f32(c61, b1, a6s);
        c70 = vfmaq_n_f32(c70, b0, a7s); c71 = vfmaq_n_f32(c71, b1, a7s);
    }
    
    // Store results
    vst1q_f32(c + 0*ldc + 0, c00); vst1q_f32(c + 0*ldc + 4, c01);
    vst1q_f32(c + 1*ldc + 0, c10); vst1q_f32(c + 1*ldc + 4, c11);
    vst1q_f32(c + 2*ldc + 0, c20); vst1q_f32(c + 2*ldc + 4, c21);
    vst1q_f32(c + 3*ldc + 0, c30); vst1q_f32(c + 3*ldc + 4, c31);
    vst1q_f32(c + 4*ldc + 0, c40); vst1q_f32(c + 4*ldc + 4, c41);
    vst1q_f32(c + 5*ldc + 0, c50); vst1q_f32(c + 5*ldc + 4, c51);
    vst1q_f32(c + 6*ldc + 0, c60); vst1q_f32(c + 6*ldc + 4, c61);
    vst1q_f32(c + 7*ldc + 0, c70); vst1q_f32(c + 7*ldc + 4, c71);
}

// NEON small matrix multiply using 8x8 micro-kernels.
// Optimal for N ≤ 32 where this beats Accelerate.
void neon_f32_matmul_small(const float* a, const float* b, float* c,
                           int m, int k, int n) {
    // Zero output
    memset(c, 0, m * n * sizeof(float));
    
    // Process 8×8 tiles
    int m8 = (m / 8) * 8;
    int n8 = (n / 8) * 8;
    
    for (int i = 0; i < m8; i += 8) {
        for (int j = 0; j < n8; j += 8) {
            neon_gemm_8x8(a + i * k, k, b + j, n, c + i * n + j, n, k);
        }
        // Right fringe (j >= n8), use scalar
        for (int j = n8; j < n; j++) {
            for (int ii = 0; ii < 8; ii++) {
                float sum = 0;
                for (int kk = 0; kk < k; kk++) {
                    sum += a[(i + ii) * k + kk] * b[kk * n + j];
                }
                c[(i + ii) * n + j] = sum;
            }
        }
    }
    
    // Bottom fringe (i >= m8)
    for (int i = m8; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int kk = 0; kk < k; kk++) {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// NEON-optimized matmul with 8×8 tiling for better register usage.
// Handles larger matrices efficiently.
void neon_f32_matmul_tiled(const float* a, const float* b, float* c,
                            int m, int k, int n) {
    // Zero output
    memset(c, 0, m * n * sizeof(float));
    
    // Process full 8×8 tiles
    int m8 = (m / 8) * 8;
    int n8 = (n / 8) * 8;
    
    for (int i = 0; i < m8; i += 8) {
        for (int j = 0; j < n8; j += 8) {
            neon_gemm_8x8(a + i * k, k, b + j, n, c + i * n + j, n, k);
        }
        // Right fringe
        for (int j = n8; j < n; j++) {
            for (int ii = 0; ii < 8; ii++) {
                float sum = 0;
                for (int kk = 0; kk < k; kk++) {
                    sum += a[(i + ii) * k + kk] * b[kk * n + j];
                }
                c[(i + ii) * n + j] = sum;
            }
        }
    }
    
    // Bottom fringe
    for (int i = m8; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int kk = 0; kk < k; kk++) {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// NEON f32 dot product — much faster than AMX for dot product.
// Uses 4-way parallel accumulation with 4× unrolling.
float neon_f32_dot(const float* a, const float* b, int n) {
    float32x4_t sum0 = vdupq_n_f32(0);
    float32x4_t sum1 = vdupq_n_f32(0);
    float32x4_t sum2 = vdupq_n_f32(0);
    float32x4_t sum3 = vdupq_n_f32(0);
    
    int i = 0;
    
    // Main loop: 16 floats per iteration (4 vectors × 4 floats)
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        
        sum0 = vfmaq_f32(sum0, a0, b0);
        sum1 = vfmaq_f32(sum1, a1, b1);
        sum2 = vfmaq_f32(sum2, a2, b2);
        sum3 = vfmaq_f32(sum3, a3, b3);
    }
    
    // Handle 4-float chunks
    for (; i + 3 < n; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        sum0 = vfmaq_f32(sum0, av, bv);
    }
    
    // Combine accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    
    // Horizontal sum
    float32x2_t sum_low = vget_low_f32(sum0);
    float32x2_t sum_high = vget_high_f32(sum0);
    float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
    float result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
    
    // Scalar tail
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

// ── f32 dot product kernel ───────────────────────────────────────────
//
// Complete dot product in a single FFI call: set, zero, accumulate,
// reduce, clr.  Loads directly from source arrays (no copies for full
// 16-float chunks).  8× unrolled inner loop.
//
// Returns the scalar dot product a·b.
float amx_f32_dot(const float* a, const float* b, int n) {
    AMX_NOP_OP_IMM5(17, 0);  // amx_set

    // Zero Z row 0
    static const uint8_t zeros[64] __attribute__((aligned(128))) = {0};
    AMX_OP_GPR(4, (uint64_t)zeros);  // ldz row 0

    int chunks = n >> 4;       // n / 16
    int tail   = n & 15;       // n % 16

    const uint8_t* ap = (const uint8_t*)a;
    const uint8_t* bp = (const uint8_t*)b;

    // Full 16-float chunks — load directly, unrolled 8×
    int c = 0;
    for (; c + 7 < chunks; c += 8) {
        AMX_OP_GPR(0, (uint64_t)(ap + (c+0) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+0) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+1) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+1) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+2) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+2) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+3) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+3) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+4) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+4) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+5) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+5) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+6) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+6) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+7) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+7) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
    }
    for (; c + 3 < chunks; c += 4) {
        AMX_OP_GPR(0, (uint64_t)(ap + (c+0) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+0) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+1) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+1) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+2) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+2) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
        AMX_OP_GPR(0, (uint64_t)(ap + (c+3) * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + (c+3) * 64));
        AMX_OP_GPR(12, 1ULL << 63);
    }
    for (; c < chunks; c++) {
        AMX_OP_GPR(0, (uint64_t)(ap + c * 64));
        AMX_OP_GPR(1, (uint64_t)(bp + c * 64));
        AMX_OP_GPR(12, 1ULL << 63);
    }

    // Tail chunk (< 16 floats) — zero-padded copy
    if (tail > 0) {
        uint8_t __attribute__((aligned(128))) xa[64] = {0};
        uint8_t __attribute__((aligned(128))) ya[64] = {0};
        memcpy(xa, ap + chunks * 64, tail * 4);
        memcpy(ya, bp + chunks * 64, tail * 4);
        AMX_OP_GPR(0, (uint64_t)xa);
        AMX_OP_GPR(1, (uint64_t)ya);
        AMX_OP_GPR(12, 1ULL << 63);
    }

    // Store Z row 0 and reduce with Kahan summation for precision
    float __attribute__((aligned(128))) zf[16];
    AMX_OP_GPR(5, (uint64_t)zf);  // stz row 0

    AMX_NOP_OP_IMM5(17, 1);  // amx_clr

    // Pairwise reduction for better precision than sequential sum
    // Level 1: 16 → 8
    float r8[8];
    for (int i = 0; i < 8; i++) r8[i] = zf[i] + zf[i + 8];
    // Level 2: 8 → 4
    float r4[4];
    for (int i = 0; i < 4; i++) r4[i] = r8[i] + r8[i + 4];
    // Level 3: 4 → 2
    float r2[2];
    r2[0] = r4[0] + r4[2];
    r2[1] = r4[1] + r4[3];
    // Level 4: 2 → 1
    return r2[0] + r2[1];
}
