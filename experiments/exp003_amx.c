// EXP-003: AMX Optimization Strategies
// Tests different AMX configurations and packing strategies
//
// Compile: clang -O3 -march=native -o exp003 exp003_amx.c -framework Accelerate
// Run: ./exp003

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Accelerate/Accelerate.h>
#include <stdint.h>

#define WARMUP 5
#define TILE 16
#define TILE_BYTES 64

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// AMX instruction wrappers
// ============================================================================

#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm__ __volatile__("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" \
        : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm__ __volatile__(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" \
        : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

static inline void amx_set(void) { AMX_NOP_OP_IMM5(17, 0); }
static inline void amx_clr(void) { AMX_NOP_OP_IMM5(17, 1); }
static inline void amx_ldx(uint64_t op) { AMX_OP_GPR(0, op); }
static inline void amx_ldy(uint64_t op) { AMX_OP_GPR(1, op); }
static inline void amx_stz(uint64_t op) { AMX_OP_GPR(5, op); }
static inline void amx_ldz(uint64_t op) { AMX_OP_GPR(4, op); }
static inline void amx_fma32(uint64_t op) { AMX_OP_GPR(12, op); }

// ============================================================================
// Packing strategies
// ============================================================================

// Strategy 1: Current scalar packing (baseline)
void pack_a_scalar(const float* A, uint8_t* Ap, int m, int k, int i_start, int i_end) {
    for (int it = i_start; it < i_end; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        for (int kk = 0; kk < k; kk++) {
            float* out = (float*)(Ap + (it - i_start) * k * TILE_BYTES + kk * TILE_BYTES);
            for (int ii = 0; ii < tile_m; ii++) {
                out[ii] = A[(i_blk + ii) * k + kk];
            }
            for (int ii = tile_m; ii < TILE; ii++) {
                out[ii] = 0;
            }
        }
    }
}

// Strategy 2: NEON 4-wide packing
void pack_a_neon4(const float* A, uint8_t* Ap, int m, int k, int i_start, int i_end) {
    for (int it = i_start; it < i_end; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        
        // Precompute row pointers
        const float* rows[16];
        for (int ii = 0; ii < tile_m; ii++) {
            rows[ii] = A + (i_blk + ii) * k;
        }
        for (int ii = tile_m; ii < 16; ii++) {
            rows[ii] = NULL;
        }
        
        float* out = (float*)(Ap + (it - i_start) * k * TILE_BYTES);
        
        // Process 4 k values at a time
        int kk = 0;
        for (; kk + 3 < k; kk += 4) {
            for (int ii = 0; ii < tile_m; ii++) {
                float32x4_t v = vld1q_f32(rows[ii] + kk);
                out[(kk+0)*16 + ii] = vgetq_lane_f32(v, 0);
                out[(kk+1)*16 + ii] = vgetq_lane_f32(v, 1);
                out[(kk+2)*16 + ii] = vgetq_lane_f32(v, 2);
                out[(kk+3)*16 + ii] = vgetq_lane_f32(v, 3);
            }
            for (int ii = tile_m; ii < 16; ii++) {
                out[(kk+0)*16 + ii] = 0;
                out[(kk+1)*16 + ii] = 0;
                out[(kk+2)*16 + ii] = 0;
                out[(kk+3)*16 + ii] = 0;
            }
        }
        for (; kk < k; kk++) {
            for (int ii = 0; ii < tile_m; ii++) {
                out[kk*16 + ii] = rows[ii][kk];
            }
            for (int ii = tile_m; ii < 16; ii++) {
                out[kk*16 + ii] = 0;
            }
        }
    }
}

// Strategy 3: NEON with prefetch
void pack_a_neon_prefetch(const float* A, uint8_t* Ap, int m, int k, int i_start, int i_end) {
    for (int it = i_start; it < i_end; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        
        // Prefetch next tile's rows
        if (it + 1 < i_end) {
            int next_blk = (it + 1) * TILE;
            for (int ii = 0; ii < TILE && next_blk + ii < m; ii++) {
                __builtin_prefetch(A + (next_blk + ii) * k, 0, 3);
            }
        }
        
        const float* rows[16];
        for (int ii = 0; ii < tile_m; ii++) {
            rows[ii] = A + (i_blk + ii) * k;
        }
        
        float* out = (float*)(Ap + (it - i_start) * k * TILE_BYTES);
        
        int kk = 0;
        for (; kk + 3 < k; kk += 4) {
            // Prefetch ahead in k dimension
            if (kk + 64 < k) {
                for (int ii = 0; ii < tile_m; ii += 4) {
                    __builtin_prefetch(rows[ii] + kk + 64, 0, 1);
                }
            }
            
            for (int ii = 0; ii < tile_m; ii++) {
                float32x4_t v = vld1q_f32(rows[ii] + kk);
                out[(kk+0)*16 + ii] = vgetq_lane_f32(v, 0);
                out[(kk+1)*16 + ii] = vgetq_lane_f32(v, 1);
                out[(kk+2)*16 + ii] = vgetq_lane_f32(v, 2);
                out[(kk+3)*16 + ii] = vgetq_lane_f32(v, 3);
            }
            for (int ii = tile_m; ii < 16; ii++) {
                out[(kk+0)*16 + ii] = 0;
                out[(kk+1)*16 + ii] = 0;
                out[(kk+2)*16 + ii] = 0;
                out[(kk+3)*16 + ii] = 0;
            }
        }
        for (; kk < k; kk++) {
            for (int ii = 0; ii < tile_m; ii++) {
                out[kk*16 + ii] = rows[ii][kk];
            }
            for (int ii = tile_m; ii < 16; ii++) {
                out[kk*16 + ii] = 0;
            }
        }
    }
}

// Pack B (same for all - it's already sequential)
void pack_b(const float* B, uint8_t* Bp, int k, int n, int n_j_tiles) {
    for (int jt = 0; jt < n_j_tiles; jt++) {
        int j_blk = jt * TILE;
        int tile_n = (j_blk + TILE <= n) ? TILE : (n - j_blk);
        for (int kk = 0; kk < k; kk++) {
            uint8_t* out = Bp + jt * k * TILE_BYTES + kk * TILE_BYTES;
            memcpy(out, B + kk * n + j_blk, tile_n * sizeof(float));
            memset(out + tile_n * 4, 0, (TILE - tile_n) * sizeof(float));
        }
    }
}

// ============================================================================
// AMX Kernels
// ============================================================================

// Kernel 1: Basic (current implementation)
void amx_kernel_basic(
    const uint8_t* Ap, const uint8_t* Bp, float* C,
    int m, int k, int n, int n_i_tiles, int n_j_tiles
) {
    static __attribute__((aligned(128))) uint8_t zeros[64] = {0};
    __attribute__((aligned(128))) uint8_t z_buf[TILE * TILE_BYTES];
    
    amx_set();
    
    for (int it = 0; it < n_i_tiles; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        const uint8_t* ap_base = Ap + it * k * TILE_BYTES;
        
        for (int jt = 0; jt < n_j_tiles; jt++) {
            int j_blk = jt * TILE;
            int tile_n = (j_blk + TILE <= n) ? TILE : (n - j_blk);
            const uint8_t* bp_base = Bp + jt * k * TILE_BYTES;
            
            // Zero Z
            for (int i = 0; i < 16; i++) {
                amx_ldz((uint64_t)zeros | ((uint64_t)(i * 4) << 56));
            }
            
            // Accumulate
            for (int kk = 0; kk < k; kk++) {
                amx_ldy((uint64_t)(ap_base + kk * TILE_BYTES));
                amx_ldx((uint64_t)(bp_base + kk * TILE_BYTES));
                amx_fma32(0);
            }
            
            // Store
            for (int i = 0; i < tile_m; i++) {
                amx_stz((uint64_t)(z_buf + i * TILE_BYTES) | ((uint64_t)(i * 4) << 56));
            }
            for (int ii = 0; ii < tile_m; ii++) {
                memcpy(C + (i_blk + ii) * n + j_blk, z_buf + ii * TILE_BYTES, tile_n * sizeof(float));
            }
        }
    }
    
    amx_clr();
}

// Kernel 2: Software pipelined (overlap load and compute)
void amx_kernel_pipelined(
    const uint8_t* Ap, const uint8_t* Bp, float* C,
    int m, int k, int n, int n_i_tiles, int n_j_tiles
) {
    static __attribute__((aligned(128))) uint8_t zeros[64] = {0};
    __attribute__((aligned(128))) uint8_t z_buf[TILE * TILE_BYTES];
    
    amx_set();
    
    for (int it = 0; it < n_i_tiles; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        const uint8_t* ap_base = Ap + it * k * TILE_BYTES;
        
        for (int jt = 0; jt < n_j_tiles; jt++) {
            int j_blk = jt * TILE;
            int tile_n = (j_blk + TILE <= n) ? TILE : (n - j_blk);
            const uint8_t* bp_base = Bp + jt * k * TILE_BYTES;
            
            // Zero Z
            for (int i = 0; i < 16; i++) {
                amx_ldz((uint64_t)zeros | ((uint64_t)(i * 4) << 56));
            }
            
            // Pipelined: load first, then fma+load next
            if (k > 0) {
                amx_ldy((uint64_t)(ap_base));
                amx_ldx((uint64_t)(bp_base));
                
                int kk = 1;
                // Unroll 4x
                for (; kk + 3 < k; kk += 4) {
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + kk * TILE_BYTES));
                    amx_ldx((uint64_t)(bp_base + kk * TILE_BYTES));
                    
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + (kk+1) * TILE_BYTES));
                    amx_ldx((uint64_t)(bp_base + (kk+1) * TILE_BYTES));
                    
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + (kk+2) * TILE_BYTES));
                    amx_ldx((uint64_t)(bp_base + (kk+2) * TILE_BYTES));
                    
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + (kk+3) * TILE_BYTES));
                    amx_ldx((uint64_t)(bp_base + (kk+3) * TILE_BYTES));
                }
                for (; kk < k; kk++) {
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + kk * TILE_BYTES));
                    amx_ldx((uint64_t)(bp_base + kk * TILE_BYTES));
                }
                amx_fma32(0);  // Final fma
            }
            
            // Store
            for (int i = 0; i < tile_m; i++) {
                amx_stz((uint64_t)(z_buf + i * TILE_BYTES) | ((uint64_t)(i * 4) << 56));
            }
            for (int ii = 0; ii < tile_m; ii++) {
                memcpy(C + (i_blk + ii) * n + j_blk, z_buf + ii * TILE_BYTES, tile_n * sizeof(float));
            }
        }
    }
    
    amx_clr();
}

// Kernel 3: KC-blocked for L1 residency
#define KC_BLOCK 256

void amx_kernel_kc_blocked(
    const uint8_t* Ap, const uint8_t* Bp, float* C,
    int m, int k, int n, int n_i_tiles, int n_j_tiles
) {
    static __attribute__((aligned(128))) uint8_t zeros[64] = {0};
    __attribute__((aligned(128))) uint8_t z_buf[TILE * TILE_BYTES];
    
    amx_set();
    
    for (int it = 0; it < n_i_tiles; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        const uint8_t* ap_base = Ap + it * k * TILE_BYTES;
        
        for (int jt = 0; jt < n_j_tiles; jt++) {
            int j_blk = jt * TILE;
            int tile_n = (j_blk + TILE <= n) ? TILE : (n - j_blk);
            const uint8_t* bp_base = Bp + jt * k * TILE_BYTES;
            
            int first = 1;
            for (int kc_start = 0; kc_start < k; kc_start += KC_BLOCK) {
                int kc_end = (kc_start + KC_BLOCK < k) ? (kc_start + KC_BLOCK) : k;
                
                if (first) {
                    // Zero Z on first block
                    for (int i = 0; i < 16; i++) {
                        amx_ldz((uint64_t)zeros | ((uint64_t)(i * 4) << 56));
                    }
                    first = 0;
                } else {
                    // Reload partial sums
                    for (int i = 0; i < tile_m; i++) {
                        amx_ldz((uint64_t)(z_buf + i * TILE_BYTES) | ((uint64_t)(i * 4) << 56));
                    }
                }
                
                // Pipelined accumulate for this KC block
                if (kc_start < kc_end) {
                    amx_ldy((uint64_t)(ap_base + kc_start * TILE_BYTES));
                    amx_ldx((uint64_t)(bp_base + kc_start * TILE_BYTES));
                    
                    for (int kk = kc_start + 1; kk < kc_end; kk++) {
                        amx_fma32(0);
                        amx_ldy((uint64_t)(ap_base + kk * TILE_BYTES));
                        amx_ldx((uint64_t)(bp_base + kk * TILE_BYTES));
                    }
                    amx_fma32(0);
                }
                
                // Store partial sums
                for (int i = 0; i < tile_m; i++) {
                    amx_stz((uint64_t)(z_buf + i * TILE_BYTES) | ((uint64_t)(i * 4) << 56));
                }
            }
            
            // Copy to output
            for (int ii = 0; ii < tile_m; ii++) {
                memcpy(C + (i_blk + ii) * n + j_blk, z_buf + ii * TILE_BYTES, tile_n * sizeof(float));
            }
        }
    }
    
    amx_clr();
}

// ============================================================================
// Benchmark harness
// ============================================================================

typedef void (*pack_fn)(const float*, uint8_t*, int, int, int, int);
typedef void (*kernel_fn)(const uint8_t*, const uint8_t*, float*, int, int, int, int, int);

double benchmark(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    pack_fn pack_a, kernel_fn kernel,
    int iters
) {
    int n_i_tiles = (M + TILE - 1) / TILE;
    int n_j_tiles = (N + TILE - 1) / TILE;
    
    uint8_t* Ap = aligned_alloc(128, n_i_tiles * K * TILE_BYTES);
    uint8_t* Bp = aligned_alloc(128, n_j_tiles * K * TILE_BYTES);
    
    // Warmup
    for (int w = 0; w < WARMUP; w++) {
        pack_a(A, Ap, M, K, 0, n_i_tiles);
        pack_b(B, Bp, K, N, n_j_tiles);
        kernel(Ap, Bp, C, M, K, N, n_i_tiles, n_j_tiles);
    }
    
    double start = now_sec();
    for (int i = 0; i < iters; i++) {
        pack_a(A, Ap, M, K, 0, n_i_tiles);
        pack_b(B, Bp, K, N, n_j_tiles);
        kernel(Ap, Bp, C, M, K, N, n_i_tiles, n_j_tiles);
    }
    double elapsed = now_sec() - start;
    
    free(Ap);
    free(Bp);
    
    return elapsed / iters;
}

int main() {
    printf("EXP-003: AMX Optimization Strategies\n");
    printf("====================================\n\n");
    
    int sizes[] = {64, 128, 256, 512};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int M = N, K = N;
        
        float* A = aligned_alloc(128, M * K * sizeof(float));
        float* B = aligned_alloc(128, K * N * sizeof(float));
        float* C = aligned_alloc(128, M * N * sizeof(float));
        
        for (int i = 0; i < M * K; i++) A[i] = (float)(i % 17) * 0.1f - 0.8f;
        for (int i = 0; i < K * N; i++) B[i] = (float)(i % 13) * 0.1f - 0.6f;
        
        double flops = 2.0 * M * N * K;
        int iters = (N <= 128) ? 200 : (N <= 256) ? 100 : 50;
        
        printf("N=%d (%.2f GFLOP, %d iters)\n", N, flops / 1e9, iters);
        printf("  %-28s  %8s  %8s  %8s\n", "Strategy", "Time(ms)", "GFLOPS", "vs Accel");
        printf("  %-28s  %8s  %8s  %8s\n", "--------", "--------", "------", "--------");
        
        // Accelerate baseline
        for (int w = 0; w < WARMUP; w++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        }
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        }
        double t_accel = (now_sec() - t0) / iters;
        double gf_accel = flops / t_accel / 1e9;
        printf("  %-28s  %8.3f  %8.1f  %8s\n", "Accelerate", t_accel * 1e3, gf_accel, "100%");
        
        // Test combinations
        struct {
            const char* name;
            pack_fn pack;
            kernel_fn kernel;
        } strategies[] = {
            {"scalar + basic",     pack_a_scalar, amx_kernel_basic},
            {"scalar + pipelined", pack_a_scalar, amx_kernel_pipelined},
            {"scalar + kc-block",  pack_a_scalar, amx_kernel_kc_blocked},
            {"neon4 + basic",      pack_a_neon4,  amx_kernel_basic},
            {"neon4 + pipelined",  pack_a_neon4,  amx_kernel_pipelined},
            {"neon4 + kc-block",   pack_a_neon4,  amx_kernel_kc_blocked},
            {"prefetch + pipelined", pack_a_neon_prefetch, amx_kernel_pipelined},
        };
        int nstrategies = sizeof(strategies) / sizeof(strategies[0]);
        
        for (int s = 0; s < nstrategies; s++) {
            double t = benchmark(A, B, C, M, N, K, 
                                strategies[s].pack, strategies[s].kernel, iters);
            double gf = flops / t / 1e9;
            printf("  %-28s  %8.3f  %8.1f  %7.1f%%\n",
                   strategies[s].name, t * 1e3, gf, 100.0 * gf / gf_accel);
        }
        
        printf("\n");
        free(A); free(B); free(C);
    }
    
    return 0;
}
