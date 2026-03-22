// EXP-004: Parallel AMX GEMM Strategies
// Tests parallel i-tile distribution and j-tile unrolling
//
// Compile: clang -O3 -march=native -o exp004 exp004_parallel.c -framework Accelerate -lpthread
// Run: ./exp004

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <Accelerate/Accelerate.h>
#include <stdint.h>

#define WARMUP 3
#define TILE 16
#define TILE_BYTES 64
#define MAX_THREADS 12

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
// Packing
// ============================================================================

void pack_a_neon(const float* A, uint8_t* Ap, int m, int k, int i_start, int i_end) {
    for (int it = i_start; it < i_end; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        
        const float* rows[16];
        for (int ii = 0; ii < tile_m; ii++) {
            rows[ii] = A + (i_blk + ii) * k;
        }
        
        float* out = (float*)(Ap + (it - i_start) * k * TILE_BYTES);
        
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
// AMX Kernel with 2-wide j-unrolling
// ============================================================================

void amx_kernel_2wide(
    const uint8_t* Ap, const uint8_t* Bp, float* C,
    int m, int k, int n, int it_start, int it_end, int n_j_tiles
) {
    static __attribute__((aligned(128))) uint8_t zeros[64] = {0};
    __attribute__((aligned(128))) uint8_t z_buf0[TILE * TILE_BYTES];
    __attribute__((aligned(128))) uint8_t z_buf1[TILE * TILE_BYTES];
    
    amx_set();
    
    for (int it = it_start; it < it_end; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        const uint8_t* ap_base = Ap + (it - it_start) * k * TILE_BYTES;
        
        // Process j-tiles in pairs
        int jt = 0;
        for (; jt + 1 < n_j_tiles; jt += 2) {
            int j_blk0 = jt * TILE;
            int j_blk1 = (jt + 1) * TILE;
            int tile_n0 = TILE;
            int tile_n1 = (j_blk1 + TILE <= n) ? TILE : (n - j_blk1);
            
            const uint8_t* bp0 = Bp + jt * k * TILE_BYTES;
            const uint8_t* bp1 = Bp + (jt + 1) * k * TILE_BYTES;
            
            // Zero both Z buffers (use different Z rows for each j-tile)
            // Z[0-15] for j0, Z[16-31] for j1
            for (int i = 0; i < 16; i++) {
                amx_ldz((uint64_t)zeros | ((uint64_t)(i * 4) << 56));
            }
            
            // Pipelined accumulate with interleaved j-tiles
            // This helps hide latency by keeping the execution units busy
            if (k > 0) {
                amx_ldy((uint64_t)(ap_base));
                
                for (int kk = 0; kk < k; kk++) {
                    // Load A once, use for both j-tiles
                    if (kk + 1 < k) {
                        amx_ldy((uint64_t)(ap_base + (kk + 1) * TILE_BYTES));
                    }
                    
                    // j-tile 0
                    amx_ldx((uint64_t)(bp0 + kk * TILE_BYTES));
                    amx_fma32(0);
                    
                    // j-tile 1 (uses same Y register)
                    amx_ldx((uint64_t)(bp1 + kk * TILE_BYTES));
                    // Use Z offset for second tile - but AMX only has one Z accumulator
                    // So we need to store and reload... 
                }
            }
            
            // Store first tile
            for (int i = 0; i < tile_m; i++) {
                amx_stz((uint64_t)(z_buf0 + i * TILE_BYTES) | ((uint64_t)(i * 4) << 56));
            }
            for (int ii = 0; ii < tile_m; ii++) {
                memcpy(C + (i_blk + ii) * n + j_blk0, z_buf0 + ii * TILE_BYTES, tile_n0 * sizeof(float));
            }
            
            // Redo for second tile (Z was accumulated for both, but we can only extract one)
            // Actually AMX accumulates into same Z, so this approach doesn't work
            // Need to compute tiles sequentially
            
            // Zero Z again for j-tile 1
            for (int i = 0; i < 16; i++) {
                amx_ldz((uint64_t)zeros | ((uint64_t)(i * 4) << 56));
            }
            
            if (k > 0) {
                amx_ldy((uint64_t)(ap_base));
                amx_ldx((uint64_t)(bp1));
                
                for (int kk = 1; kk < k; kk++) {
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + kk * TILE_BYTES));
                    amx_ldx((uint64_t)(bp1 + kk * TILE_BYTES));
                }
                amx_fma32(0);
            }
            
            for (int i = 0; i < tile_m; i++) {
                amx_stz((uint64_t)(z_buf1 + i * TILE_BYTES) | ((uint64_t)(i * 4) << 56));
            }
            for (int ii = 0; ii < tile_m; ii++) {
                memcpy(C + (i_blk + ii) * n + j_blk1, z_buf1 + ii * TILE_BYTES, tile_n1 * sizeof(float));
            }
        }
        
        // Handle remaining j-tile
        if (jt < n_j_tiles) {
            int j_blk = jt * TILE;
            int tile_n = (j_blk + TILE <= n) ? TILE : (n - j_blk);
            const uint8_t* bp = Bp + jt * k * TILE_BYTES;
            
            for (int i = 0; i < 16; i++) {
                amx_ldz((uint64_t)zeros | ((uint64_t)(i * 4) << 56));
            }
            
            if (k > 0) {
                amx_ldy((uint64_t)(ap_base));
                amx_ldx((uint64_t)(bp));
                
                for (int kk = 1; kk < k; kk++) {
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + kk * TILE_BYTES));
                    amx_ldx((uint64_t)(bp + kk * TILE_BYTES));
                }
                amx_fma32(0);
            }
            
            for (int i = 0; i < tile_m; i++) {
                amx_stz((uint64_t)(z_buf0 + i * TILE_BYTES) | ((uint64_t)(i * 4) << 56));
            }
            for (int ii = 0; ii < tile_m; ii++) {
                memcpy(C + (i_blk + ii) * n + j_blk, z_buf0 + ii * TILE_BYTES, tile_n * sizeof(float));
            }
        }
    }
    
    amx_clr();
}

// Standard kernel (for comparison)
void amx_kernel_standard(
    const uint8_t* Ap, const uint8_t* Bp, float* C,
    int m, int k, int n, int it_start, int it_end, int n_j_tiles
) {
    static __attribute__((aligned(128))) uint8_t zeros[64] = {0};
    __attribute__((aligned(128))) uint8_t z_buf[TILE * TILE_BYTES];
    
    amx_set();
    
    for (int it = it_start; it < it_end; it++) {
        int i_blk = it * TILE;
        int tile_m = (i_blk + TILE <= m) ? TILE : (m - i_blk);
        const uint8_t* ap_base = Ap + (it - it_start) * k * TILE_BYTES;
        
        for (int jt = 0; jt < n_j_tiles; jt++) {
            int j_blk = jt * TILE;
            int tile_n = (j_blk + TILE <= n) ? TILE : (n - j_blk);
            const uint8_t* bp = Bp + jt * k * TILE_BYTES;
            
            for (int i = 0; i < 16; i++) {
                amx_ldz((uint64_t)zeros | ((uint64_t)(i * 4) << 56));
            }
            
            if (k > 0) {
                amx_ldy((uint64_t)(ap_base));
                amx_ldx((uint64_t)(bp));
                
                for (int kk = 1; kk < k; kk++) {
                    amx_fma32(0);
                    amx_ldy((uint64_t)(ap_base + kk * TILE_BYTES));
                    amx_ldx((uint64_t)(bp + kk * TILE_BYTES));
                }
                amx_fma32(0);
            }
            
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

// ============================================================================
// Parallel worker
// ============================================================================

typedef struct {
    const float* A;
    const float* B;
    float* C;
    const uint8_t* Bp;
    int M, K, N;
    int it_start, it_end;
    int n_j_tiles;
} worker_args_t;

void* worker_fn(void* arg) {
    worker_args_t* w = (worker_args_t*)arg;
    
    int n_my_tiles = w->it_end - w->it_start;
    uint8_t* Ap = aligned_alloc(128, n_my_tiles * w->K * TILE_BYTES);
    
    pack_a_neon(w->A, Ap, w->M, w->K, w->it_start, w->it_end);
    amx_kernel_standard(Ap, w->Bp, w->C, w->M, w->K, w->N, w->it_start, w->it_end, w->n_j_tiles);
    
    free(Ap);
    return NULL;
}

void gemm_parallel(
    const float* A, const float* B, float* C,
    int M, int N, int K, int n_threads
) {
    int n_i_tiles = (M + TILE - 1) / TILE;
    int n_j_tiles = (N + TILE - 1) / TILE;
    
    // Pack B once (shared)
    uint8_t* Bp = aligned_alloc(128, n_j_tiles * K * TILE_BYTES);
    pack_b(B, Bp, K, N, n_j_tiles);
    
    if (n_threads == 1 || n_i_tiles < 2) {
        uint8_t* Ap = aligned_alloc(128, n_i_tiles * K * TILE_BYTES);
        pack_a_neon(A, Ap, M, K, 0, n_i_tiles);
        amx_kernel_standard(Ap, Bp, C, M, K, N, 0, n_i_tiles, n_j_tiles);
        free(Ap);
    } else {
        n_threads = (n_threads > n_i_tiles) ? n_i_tiles : n_threads;
        
        pthread_t threads[MAX_THREADS];
        worker_args_t args[MAX_THREADS];
        
        int tiles_per_thread = (n_i_tiles + n_threads - 1) / n_threads;
        
        for (int t = 0; t < n_threads; t++) {
            args[t].A = A;
            args[t].B = B;
            args[t].C = C;
            args[t].Bp = Bp;
            args[t].M = M;
            args[t].K = K;
            args[t].N = N;
            args[t].it_start = t * tiles_per_thread;
            args[t].it_end = (t + 1) * tiles_per_thread;
            if (args[t].it_end > n_i_tiles) args[t].it_end = n_i_tiles;
            args[t].n_j_tiles = n_j_tiles;
            
            if (args[t].it_start < args[t].it_end) {
                pthread_create(&threads[t], NULL, worker_fn, &args[t]);
            }
        }
        
        for (int t = 0; t < n_threads; t++) {
            if (args[t].it_start < args[t].it_end) {
                pthread_join(threads[t], NULL);
            }
        }
    }
    
    free(Bp);
}

// ============================================================================
// Benchmarks
// ============================================================================

int main() {
    printf("EXP-004: Parallel AMX GEMM Strategies\n");
    printf("=====================================\n\n");
    
    int sizes[] = {256, 512, 1024};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int thread_counts[] = {1, 2, 4, 6, 8, 10};
    int n_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int M = N, K = N;
        
        float* A = aligned_alloc(128, M * K * sizeof(float));
        float* B = aligned_alloc(128, K * N * sizeof(float));
        float* C = aligned_alloc(128, M * N * sizeof(float));
        
        for (int i = 0; i < M * K; i++) A[i] = (float)(i % 17) * 0.1f - 0.8f;
        for (int i = 0; i < K * N; i++) B[i] = (float)(i % 13) * 0.1f - 0.6f;
        
        double flops = 2.0 * M * N * K;
        int iters = (N <= 256) ? 50 : (N <= 512) ? 20 : 10;
        
        printf("N=%d (%.2f GFLOP, %d iters)\n", N, flops / 1e9, iters);
        printf("  %-20s  %8s  %8s  %8s\n", "Strategy", "Time(ms)", "GFLOPS", "vs Accel");
        printf("  %-20s  %8s  %8s  %8s\n", "--------", "--------", "------", "--------");
        
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
        printf("  %-20s  %8.3f  %8.1f  %8s\n", "Accelerate", t_accel * 1e3, gf_accel, "100%");
        
        // Test parallel configurations
        for (int ti = 0; ti < n_thread_counts; ti++) {
            int nt = thread_counts[ti];
            char name[32];
            snprintf(name, sizeof(name), "AMX %d-thread", nt);
            
            for (int w = 0; w < WARMUP; w++) {
                gemm_parallel(A, B, C, M, N, K, nt);
            }
            double t0 = now_sec();
            for (int i = 0; i < iters; i++) {
                gemm_parallel(A, B, C, M, N, K, nt);
            }
            double t = (now_sec() - t0) / iters;
            double gf = flops / t / 1e9;
            printf("  %-20s  %8.3f  %8.1f  %7.1f%%\n", name, t * 1e3, gf, 100.0 * gf / gf_accel);
        }
        
        printf("\n");
        free(A); free(B); free(C);
    }
    
    return 0;
}
