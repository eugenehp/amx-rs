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
    for (int kk = 0; kk < k; kk++) {
        AMX_OP_GPR(1, (uint64_t)(ap + kk * 64));   // ldy
        AMX_OP_GPR(0, (uint64_t)(bp + kk * 64));   // ldx
        AMX_OP_GPR(12, 0);                          // fma32 matrix mode
    }
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

// Complete f32 micro-kernel: zero Z, accumulate k rank-1 updates,
// store tile_m rows to dst.  Single FFI call per 16×16 tile.
// Inner loop unrolled 4× for better instruction scheduling.
void amx_f32_tile_kernel(const void* a_panel, const void* b_panel,
                         void* dst, int k, int tile_m) {
    // Zero Z
    static const uint8_t zeros[64] __attribute__((aligned(128))) = {0};
    uint64_t zbase = (uint64_t)zeros;
    for (int i = 0; i < 16; i++) {
        AMX_OP_GPR(4, zbase | ((uint64_t)(i * 4) << 56));
    }

    // Accumulate — unrolled 4×
    const uint8_t* ap = (const uint8_t*)a_panel;
    const uint8_t* bp = (const uint8_t*)b_panel;
    int kk = 0;
    for (; kk + 3 < k; kk += 4) {
        AMX_OP_GPR(1, (uint64_t)(ap + (kk+0) * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + (kk+0) * 64));
        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + (kk+1) * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + (kk+1) * 64));
        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + (kk+2) * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + (kk+2) * 64));
        AMX_OP_GPR(12, 0);
        AMX_OP_GPR(1, (uint64_t)(ap + (kk+3) * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + (kk+3) * 64));
        AMX_OP_GPR(12, 0);
    }
    for (; kk < k; kk++) {
        AMX_OP_GPR(1, (uint64_t)(ap + kk * 64));
        AMX_OP_GPR(0, (uint64_t)(bp + kk * 64));
        AMX_OP_GPR(12, 0);
    }

    // Store
    uint8_t* p = (uint8_t*)dst;
    for (int i = 0; i < tile_m; i++) {
        AMX_OP_GPR(5, ((uint64_t)(p + i * 64)) | ((uint64_t)(i * 4) << 56));
    }
}
