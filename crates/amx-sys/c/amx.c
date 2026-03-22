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
// Software-pipelined: loads overlap with compute for better throughput.
void amx_f32_tile_kernel(const void* a_panel, const void* b_panel,
                         void* dst, int k, int tile_m) {
    // Zero Z
    static const uint8_t zeros[64] __attribute__((aligned(128))) = {0};
    uint64_t zbase = (uint64_t)zeros;
    for (int i = 0; i < 16; i++) {
        AMX_OP_GPR(4, zbase | ((uint64_t)(i * 4) << 56));
    }

    // Software-pipelined accumulate
    const uint8_t* ap = (const uint8_t*)a_panel;
    const uint8_t* bp = (const uint8_t*)b_panel;

    if (k > 0) {
        // Preload first pair
        AMX_OP_GPR(1, (uint64_t)(ap));
        AMX_OP_GPR(0, (uint64_t)(bp));

        int kk = 1;
        // Unrolled 4× with pipelining
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
        AMX_OP_GPR(12, 0); // final fma
    }

    // Store
    uint8_t* p = (uint8_t*)dst;
    for (int i = 0; i < tile_m; i++) {
        AMX_OP_GPR(5, ((uint64_t)(p + i * 64)) | ((uint64_t)(i * 4) << 56));
    }
}

// Accumulating tile kernel: loads existing partial sums from dst into Z,
// then accumulates k rank-1 updates and stores back.
// Used for KC-blocked matmul where the k-dimension is split into blocks.
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

    // Software-pipelined accumulate
    const uint8_t* ap = (const uint8_t*)a_panel;
    const uint8_t* bp = (const uint8_t*)b_panel;

    if (k > 0) {
        AMX_OP_GPR(1, (uint64_t)(ap));
        AMX_OP_GPR(0, (uint64_t)(bp));

        int kk = 1;
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
        AMX_OP_GPR(12, 0);
    }

    // Store back
    uint8_t* p = (uint8_t*)dst;
    for (int i = 0; i < tile_m; i++) {
        AMX_OP_GPR(5, ((uint64_t)(p + i * 64)) | ((uint64_t)(i * 4) << 56));
    }
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
