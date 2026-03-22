// EXP-002: Cache-Blocked GEMM with Packed Panels
// Tests MC/KC/NC blocking parameters with packed data layouts
//
// Compile: clang -O3 -march=native -o exp002 exp002_blocked.c -framework Accelerate
// Run: ./exp002

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#define WARMUP 5
#define ALIGN 128

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// Micro-kernel: 8x8 with packed panels
// A panel: MR×KC packed column-major (8 × KC)
// B panel: KC×NR packed row-major (KC × 8)
// ============================================================================
#define MR 8
#define NR 8

static inline void microkernel_8x8_packed(
    const float* __restrict Ap,  // MR×KC packed
    const float* __restrict Bp,  // KC×NR packed  
    float* __restrict C, int ldc,
    int kc
) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
    float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
    float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);
    
    for (int k = 0; k < kc; k++) {
        // A panel is packed column-major: A[i,k] = Ap[k*MR + i]
        float32x4_t a0 = vld1q_f32(Ap + k * MR);      // a[0:3, k]
        float32x4_t a1 = vld1q_f32(Ap + k * MR + 4);  // a[4:7, k]
        
        // B panel is packed row-major: B[k,j] = Bp[k*NR + j]
        float32x4_t b0 = vld1q_f32(Bp + k * NR);      // b[k, 0:3]
        float32x4_t b1 = vld1q_f32(Bp + k * NR + 4);  // b[k, 4:7]
        
        // Outer product
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0); c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1); c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2); c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3); c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0); c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1); c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2); c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3); c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
    }
    
    // Store (accumulate)
    vst1q_f32(C + 0*ldc + 0, vaddq_f32(vld1q_f32(C + 0*ldc + 0), c00));
    vst1q_f32(C + 0*ldc + 4, vaddq_f32(vld1q_f32(C + 0*ldc + 4), c01));
    vst1q_f32(C + 1*ldc + 0, vaddq_f32(vld1q_f32(C + 1*ldc + 0), c10));
    vst1q_f32(C + 1*ldc + 4, vaddq_f32(vld1q_f32(C + 1*ldc + 4), c11));
    vst1q_f32(C + 2*ldc + 0, vaddq_f32(vld1q_f32(C + 2*ldc + 0), c20));
    vst1q_f32(C + 2*ldc + 4, vaddq_f32(vld1q_f32(C + 2*ldc + 4), c21));
    vst1q_f32(C + 3*ldc + 0, vaddq_f32(vld1q_f32(C + 3*ldc + 0), c30));
    vst1q_f32(C + 3*ldc + 4, vaddq_f32(vld1q_f32(C + 3*ldc + 4), c31));
    vst1q_f32(C + 4*ldc + 0, vaddq_f32(vld1q_f32(C + 4*ldc + 0), c40));
    vst1q_f32(C + 4*ldc + 4, vaddq_f32(vld1q_f32(C + 4*ldc + 4), c41));
    vst1q_f32(C + 5*ldc + 0, vaddq_f32(vld1q_f32(C + 5*ldc + 0), c50));
    vst1q_f32(C + 5*ldc + 4, vaddq_f32(vld1q_f32(C + 5*ldc + 4), c51));
    vst1q_f32(C + 6*ldc + 0, vaddq_f32(vld1q_f32(C + 6*ldc + 0), c60));
    vst1q_f32(C + 6*ldc + 4, vaddq_f32(vld1q_f32(C + 6*ldc + 4), c61));
    vst1q_f32(C + 7*ldc + 0, vaddq_f32(vld1q_f32(C + 7*ldc + 0), c70));
    vst1q_f32(C + 7*ldc + 4, vaddq_f32(vld1q_f32(C + 7*ldc + 4), c71));
}

// ============================================================================
// Packing functions
// ============================================================================

// Pack A panel: MC×KC block into column-major panels of MR×KC each
void pack_a(const float* A, float* Ap, int lda, int mc, int kc) {
    for (int i = 0; i < mc; i += MR) {
        int mr = (i + MR <= mc) ? MR : (mc - i);
        for (int k = 0; k < kc; k++) {
            for (int ii = 0; ii < mr; ii++) {
                Ap[(i/MR) * MR * kc + k * MR + ii] = A[(i + ii) * lda + k];
            }
            for (int ii = mr; ii < MR; ii++) {
                Ap[(i/MR) * MR * kc + k * MR + ii] = 0;
            }
        }
    }
}

// Pack B panel: KC×NC block into row-major panels of KC×NR each
void pack_b(const float* B, float* Bp, int ldb, int kc, int nc) {
    for (int j = 0; j < nc; j += NR) {
        int nr = (j + NR <= nc) ? NR : (nc - j);
        for (int k = 0; k < kc; k++) {
            for (int jj = 0; jj < nr; jj++) {
                Bp[(j/NR) * NR * kc + k * NR + jj] = B[k * ldb + j + jj];
            }
            for (int jj = nr; jj < NR; jj++) {
                Bp[(j/NR) * NR * kc + k * NR + jj] = 0;
            }
        }
    }
}

// ============================================================================
// BLIS-style 5-loop GEMM
// ============================================================================
void gemm_blocked(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int mc, int nc, int kc
) {
    // Allocate packed panels
    float* Ap = aligned_alloc(ALIGN, ((mc + MR - 1) / MR) * MR * kc * sizeof(float));
    float* Bp = aligned_alloc(ALIGN, ((nc + NR - 1) / NR) * NR * kc * sizeof(float));
    
    memset(C, 0, M * N * sizeof(float));
    
    // 5-loop structure
    for (int jc = 0; jc < N; jc += nc) {
        int nc_eff = (jc + nc <= N) ? nc : (N - jc);
        
        for (int pc = 0; pc < K; pc += kc) {
            int kc_eff = (pc + kc <= K) ? kc : (K - pc);
            
            // Pack B panel: B[pc:pc+kc, jc:jc+nc]
            pack_b(B + pc * N + jc, Bp, N, kc_eff, nc_eff);
            
            for (int ic = 0; ic < M; ic += mc) {
                int mc_eff = (ic + mc <= M) ? mc : (M - ic);
                
                // Pack A panel: A[ic:ic+mc, pc:pc+kc]
                pack_a(A + ic * K + pc, Ap, K, mc_eff, kc_eff);
                
                // Micro-kernel loops
                for (int jr = 0; jr < nc_eff; jr += NR) {
                    for (int ir = 0; ir < mc_eff; ir += MR) {
                        int mr_eff = (ir + MR <= mc_eff) ? MR : (mc_eff - ir);
                        int nr_eff = (jr + NR <= nc_eff) ? NR : (nc_eff - jr);
                        
                        if (mr_eff == MR && nr_eff == NR) {
                            microkernel_8x8_packed(
                                Ap + (ir / MR) * MR * kc_eff,
                                Bp + (jr / NR) * NR * kc_eff,
                                C + (ic + ir) * N + jc + jr, N,
                                kc_eff
                            );
                        } else {
                            // Scalar fringe
                            const float* ap = Ap + (ir / MR) * MR * kc_eff;
                            const float* bp = Bp + (jr / NR) * NR * kc_eff;
                            for (int ii = 0; ii < mr_eff; ii++) {
                                for (int jj = 0; jj < nr_eff; jj++) {
                                    float sum = 0;
                                    for (int kk = 0; kk < kc_eff; kk++) {
                                        sum += ap[kk * MR + ii] * bp[kk * NR + jj];
                                    }
                                    C[(ic + ir + ii) * N + jc + jr + jj] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    free(Ap);
    free(Bp);
}

// ============================================================================
// Benchmarks
// ============================================================================

int main() {
    printf("EXP-002: Cache-Blocked GEMM with Packed Panels\n");
    printf("==============================================\n\n");
    
    // Test different blocking parameters
    typedef struct { int mc, nc, kc; const char* name; } config_t;
    config_t configs[] = {
        {64,  64,  64,  "64/64/64"},
        {64,  128, 128, "64/128/128"},
        {128, 128, 128, "128/128/128"},
        {128, 256, 256, "128/256/256"},
        {256, 256, 256, "256/256/256"},
        {128, 512, 256, "128/512/256"},
        {256, 512, 128, "256/512/128"},
        {128, 256, 512, "128/256/512"},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);
    
    int sizes[] = {64, 128, 256, 512, 1024};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int M = N, K = N;
        
        float* A = aligned_alloc(ALIGN, M * K * sizeof(float));
        float* B = aligned_alloc(ALIGN, K * N * sizeof(float));
        float* C = aligned_alloc(ALIGN, M * N * sizeof(float));
        
        for (int i = 0; i < M * K; i++) A[i] = (float)(i % 17) * 0.1f - 0.8f;
        for (int i = 0; i < K * N; i++) B[i] = (float)(i % 13) * 0.1f - 0.6f;
        
        double flops = 2.0 * M * N * K;
        int iters = (N <= 128) ? 100 : (N <= 256) ? 50 : (N <= 512) ? 20 : 5;
        
        printf("N=%d (%.1f GFLOP, %d iters)\n", N, flops / 1e9, iters);
        printf("  %-16s  %8s  %8s  %8s\n", "Config", "Time(ms)", "GFLOPS", "vs Accel");
        printf("  %-16s  %8s  %8s  %8s\n", "------", "--------", "------", "--------");
        
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
        printf("  %-16s  %8.3f  %8.1f  %8s\n", "Accelerate", t_accel * 1e3, gf_accel, "100%");
        
        // Test each config
        for (int ci = 0; ci < nconfigs; ci++) {
            int mc = configs[ci].mc;
            int nc = configs[ci].nc;
            int kc = configs[ci].kc;
            
            // Skip configs that are too big for small matrices
            if (mc > M || nc > N || kc > K) continue;
            
            for (int w = 0; w < WARMUP; w++) {
                gemm_blocked(A, B, C, M, N, K, mc, nc, kc);
            }
            double t0 = now_sec();
            for (int i = 0; i < iters; i++) {
                gemm_blocked(A, B, C, M, N, K, mc, nc, kc);
            }
            double t = (now_sec() - t0) / iters;
            double gf = flops / t / 1e9;
            printf("  %-16s  %8.3f  %8.1f  %7.1f%%\n", 
                   configs[ci].name, t * 1e3, gf, 100.0 * gf / gf_accel);
        }
        
        printf("\n");
        free(A); free(B); free(C);
    }
    
    return 0;
}
