// EXP-001: NEON Micro-Kernel Size Comparison
// Tests 4x4, 6x8, 8x8, and 4x16 kernels to find optimal register blocking
//
// Compile: clang -O3 -march=native -o exp001 exp001_microkernel.c -framework Accelerate
// Run: ./exp001

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#define WARMUP 10
#define ITERS 1000

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// 4x4 Micro-Kernel (current implementation)
// ============================================================================
static inline void kernel_4x4(
    const float* __restrict A, int lda,
    const float* __restrict B, int ldb,
    float* __restrict C, int ldc,
    int K
) {
    float32x4_t c0 = vdupq_n_f32(0);
    float32x4_t c1 = vdupq_n_f32(0);
    float32x4_t c2 = vdupq_n_f32(0);
    float32x4_t c3 = vdupq_n_f32(0);
    
    for (int k = 0; k < K; k++) {
        float32x4_t b = vld1q_f32(B + k * ldb);
        c0 = vfmaq_n_f32(c0, b, A[0 * lda + k]);
        c1 = vfmaq_n_f32(c1, b, A[1 * lda + k]);
        c2 = vfmaq_n_f32(c2, b, A[2 * lda + k]);
        c3 = vfmaq_n_f32(c3, b, A[3 * lda + k]);
    }
    
    vst1q_f32(C + 0 * ldc, vaddq_f32(vld1q_f32(C + 0 * ldc), c0));
    vst1q_f32(C + 1 * ldc, vaddq_f32(vld1q_f32(C + 1 * ldc), c1));
    vst1q_f32(C + 2 * ldc, vaddq_f32(vld1q_f32(C + 2 * ldc), c2));
    vst1q_f32(C + 3 * ldc, vaddq_f32(vld1q_f32(C + 3 * ldc), c3));
}

// ============================================================================
// 8x8 Micro-Kernel (more register pressure, better compute density)
// ============================================================================
static inline void kernel_8x8(
    const float* __restrict A, int lda,
    const float* __restrict B, int ldb,
    float* __restrict C, int ldc,
    int K
) {
    // 8 rows of C, each row is 8 floats = 2 NEON registers
    // Total: 16 accumulators (c00-c07, c10-c17, ... c70-c77)
    // Uses 16 of 32 NEON registers for accumulators
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
    float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
    float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);
    
    for (int k = 0; k < K; k++) {
        float32x4_t b0 = vld1q_f32(B + k * ldb);
        float32x4_t b1 = vld1q_f32(B + k * ldb + 4);
        
        float a0 = A[0 * lda + k];
        float a1 = A[1 * lda + k];
        float a2 = A[2 * lda + k];
        float a3 = A[3 * lda + k];
        float a4 = A[4 * lda + k];
        float a5 = A[5 * lda + k];
        float a6 = A[6 * lda + k];
        float a7 = A[7 * lda + k];
        
        c00 = vfmaq_n_f32(c00, b0, a0); c01 = vfmaq_n_f32(c01, b1, a0);
        c10 = vfmaq_n_f32(c10, b0, a1); c11 = vfmaq_n_f32(c11, b1, a1);
        c20 = vfmaq_n_f32(c20, b0, a2); c21 = vfmaq_n_f32(c21, b1, a2);
        c30 = vfmaq_n_f32(c30, b0, a3); c31 = vfmaq_n_f32(c31, b1, a3);
        c40 = vfmaq_n_f32(c40, b0, a4); c41 = vfmaq_n_f32(c41, b1, a4);
        c50 = vfmaq_n_f32(c50, b0, a5); c51 = vfmaq_n_f32(c51, b1, a5);
        c60 = vfmaq_n_f32(c60, b0, a6); c61 = vfmaq_n_f32(c61, b1, a6);
        c70 = vfmaq_n_f32(c70, b0, a7); c71 = vfmaq_n_f32(c71, b1, a7);
    }
    
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
// 4x16 Micro-Kernel (wide B panel, good for skinny matrices)
// ============================================================================
static inline void kernel_4x16(
    const float* __restrict A, int lda,
    const float* __restrict B, int ldb,
    float* __restrict C, int ldc,
    int K
) {
    // 4 rows × 16 cols = 16 accumulators (4 per row)
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c32 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);
    
    for (int k = 0; k < K; k++) {
        float32x4_t b0 = vld1q_f32(B + k * ldb + 0);
        float32x4_t b1 = vld1q_f32(B + k * ldb + 4);
        float32x4_t b2 = vld1q_f32(B + k * ldb + 8);
        float32x4_t b3 = vld1q_f32(B + k * ldb + 12);
        
        float a0 = A[0 * lda + k];
        float a1 = A[1 * lda + k];
        float a2 = A[2 * lda + k];
        float a3 = A[3 * lda + k];
        
        c00 = vfmaq_n_f32(c00, b0, a0); c01 = vfmaq_n_f32(c01, b1, a0);
        c02 = vfmaq_n_f32(c02, b2, a0); c03 = vfmaq_n_f32(c03, b3, a0);
        c10 = vfmaq_n_f32(c10, b0, a1); c11 = vfmaq_n_f32(c11, b1, a1);
        c12 = vfmaq_n_f32(c12, b2, a1); c13 = vfmaq_n_f32(c13, b3, a1);
        c20 = vfmaq_n_f32(c20, b0, a2); c21 = vfmaq_n_f32(c21, b1, a2);
        c22 = vfmaq_n_f32(c22, b2, a2); c23 = vfmaq_n_f32(c23, b3, a2);
        c30 = vfmaq_n_f32(c30, b0, a3); c31 = vfmaq_n_f32(c31, b1, a3);
        c32 = vfmaq_n_f32(c32, b2, a3); c33 = vfmaq_n_f32(c33, b3, a3);
    }
    
    vst1q_f32(C + 0*ldc +  0, vaddq_f32(vld1q_f32(C + 0*ldc +  0), c00));
    vst1q_f32(C + 0*ldc +  4, vaddq_f32(vld1q_f32(C + 0*ldc +  4), c01));
    vst1q_f32(C + 0*ldc +  8, vaddq_f32(vld1q_f32(C + 0*ldc +  8), c02));
    vst1q_f32(C + 0*ldc + 12, vaddq_f32(vld1q_f32(C + 0*ldc + 12), c03));
    vst1q_f32(C + 1*ldc +  0, vaddq_f32(vld1q_f32(C + 1*ldc +  0), c10));
    vst1q_f32(C + 1*ldc +  4, vaddq_f32(vld1q_f32(C + 1*ldc +  4), c11));
    vst1q_f32(C + 1*ldc +  8, vaddq_f32(vld1q_f32(C + 1*ldc +  8), c12));
    vst1q_f32(C + 1*ldc + 12, vaddq_f32(vld1q_f32(C + 1*ldc + 12), c13));
    vst1q_f32(C + 2*ldc +  0, vaddq_f32(vld1q_f32(C + 2*ldc +  0), c20));
    vst1q_f32(C + 2*ldc +  4, vaddq_f32(vld1q_f32(C + 2*ldc +  4), c21));
    vst1q_f32(C + 2*ldc +  8, vaddq_f32(vld1q_f32(C + 2*ldc +  8), c22));
    vst1q_f32(C + 2*ldc + 12, vaddq_f32(vld1q_f32(C + 2*ldc + 12), c23));
    vst1q_f32(C + 3*ldc +  0, vaddq_f32(vld1q_f32(C + 3*ldc +  0), c30));
    vst1q_f32(C + 3*ldc +  4, vaddq_f32(vld1q_f32(C + 3*ldc +  4), c31));
    vst1q_f32(C + 3*ldc +  8, vaddq_f32(vld1q_f32(C + 3*ldc +  8), c32));
    vst1q_f32(C + 3*ldc + 12, vaddq_f32(vld1q_f32(C + 3*ldc + 12), c33));
}

// ============================================================================
// 8x4 Micro-Kernel (tall A panel)
// ============================================================================
static inline void kernel_8x4(
    const float* __restrict A, int lda,
    const float* __restrict B, int ldb,
    float* __restrict C, int ldc,
    int K
) {
    float32x4_t c0 = vdupq_n_f32(0);
    float32x4_t c1 = vdupq_n_f32(0);
    float32x4_t c2 = vdupq_n_f32(0);
    float32x4_t c3 = vdupq_n_f32(0);
    float32x4_t c4 = vdupq_n_f32(0);
    float32x4_t c5 = vdupq_n_f32(0);
    float32x4_t c6 = vdupq_n_f32(0);
    float32x4_t c7 = vdupq_n_f32(0);
    
    for (int k = 0; k < K; k++) {
        float32x4_t b = vld1q_f32(B + k * ldb);
        c0 = vfmaq_n_f32(c0, b, A[0 * lda + k]);
        c1 = vfmaq_n_f32(c1, b, A[1 * lda + k]);
        c2 = vfmaq_n_f32(c2, b, A[2 * lda + k]);
        c3 = vfmaq_n_f32(c3, b, A[3 * lda + k]);
        c4 = vfmaq_n_f32(c4, b, A[4 * lda + k]);
        c5 = vfmaq_n_f32(c5, b, A[5 * lda + k]);
        c6 = vfmaq_n_f32(c6, b, A[6 * lda + k]);
        c7 = vfmaq_n_f32(c7, b, A[7 * lda + k]);
    }
    
    vst1q_f32(C + 0*ldc, vaddq_f32(vld1q_f32(C + 0*ldc), c0));
    vst1q_f32(C + 1*ldc, vaddq_f32(vld1q_f32(C + 1*ldc), c1));
    vst1q_f32(C + 2*ldc, vaddq_f32(vld1q_f32(C + 2*ldc), c2));
    vst1q_f32(C + 3*ldc, vaddq_f32(vld1q_f32(C + 3*ldc), c3));
    vst1q_f32(C + 4*ldc, vaddq_f32(vld1q_f32(C + 4*ldc), c4));
    vst1q_f32(C + 5*ldc, vaddq_f32(vld1q_f32(C + 5*ldc), c5));
    vst1q_f32(C + 6*ldc, vaddq_f32(vld1q_f32(C + 6*ldc), c6));
    vst1q_f32(C + 7*ldc, vaddq_f32(vld1q_f32(C + 7*ldc), c7));
}

// ============================================================================
// Full GEMM using micro-kernels
// ============================================================================

typedef void (*kernel_fn)(const float*, int, const float*, int, float*, int, int);

void gemm_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int MR, int NR,
    kernel_fn kernel
) {
    memset(C, 0, M * N * sizeof(float));
    
    for (int i = 0; i < M; i += MR) {
        int mb = (i + MR <= M) ? MR : (M - i);
        for (int j = 0; j < N; j += NR) {
            int nb = (j + NR <= N) ? NR : (N - j);
            if (mb == MR && nb == NR) {
                kernel(A + i * K, K, B + j, N, C + i * N + j, N, K);
            } else {
                // Scalar fallback for fringe
                for (int ii = 0; ii < mb; ii++) {
                    for (int jj = 0; jj < nb; jj++) {
                        float sum = 0;
                        for (int kk = 0; kk < K; kk++) {
                            sum += A[(i + ii) * K + kk] * B[kk * N + j + jj];
                        }
                        C[(i + ii) * N + j + jj] += sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Benchmark harness
// ============================================================================

double benchmark_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int MR, int NR,
    kernel_fn kernel,
    int iters
) {
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        gemm_tiled(A, B, C, M, N, K, MR, NR, kernel);
    }
    
    double start = now_sec();
    for (int i = 0; i < iters; i++) {
        gemm_tiled(A, B, C, M, N, K, MR, NR, kernel);
    }
    double elapsed = now_sec() - start;
    
    return elapsed / iters;
}

double benchmark_accelerate(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int iters
) {
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    
    double start = now_sec();
    for (int i = 0; i < iters; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double elapsed = now_sec() - start;
    
    return elapsed / iters;
}

int main() {
    printf("EXP-001: NEON Micro-Kernel Size Comparison\n");
    printf("==========================================\n\n");
    
    int sizes[] = {16, 32, 48, 64, 96, 128, 192, 256};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int M = N, K = N;
        
        float* A = aligned_alloc(64, M * K * sizeof(float));
        float* B = aligned_alloc(64, K * N * sizeof(float));
        float* C = aligned_alloc(64, M * N * sizeof(float));
        
        // Initialize
        for (int i = 0; i < M * K; i++) A[i] = (float)(i % 17) * 0.1f - 0.8f;
        for (int i = 0; i < K * N; i++) B[i] = (float)(i % 13) * 0.1f - 0.6f;
        
        double flops = 2.0 * M * N * K;
        int iters = (N <= 64) ? 5000 : (N <= 128) ? 1000 : 200;
        
        printf("N=%d (%.1f MFLOP, %d iters)\n", N, flops / 1e6, iters);
        printf("  %-12s  %8s  %8s  %8s\n", "Kernel", "Time(µs)", "GFLOPS", "vs Accel");
        printf("  %-12s  %8s  %8s  %8s\n", "------", "--------", "------", "--------");
        
        // Accelerate baseline
        double t_accel = benchmark_accelerate(A, B, C, M, N, K, iters);
        double gf_accel = flops / t_accel / 1e9;
        printf("  %-12s  %8.2f  %8.1f  %8s\n", "Accelerate", t_accel * 1e6, gf_accel, "100%");
        
        // 4x4 kernel
        double t_4x4 = benchmark_kernel(A, B, C, M, N, K, 4, 4, kernel_4x4, iters);
        double gf_4x4 = flops / t_4x4 / 1e9;
        printf("  %-12s  %8.2f  %8.1f  %7.1f%%\n", "4x4", t_4x4 * 1e6, gf_4x4, 100.0 * gf_4x4 / gf_accel);
        
        // 8x4 kernel
        double t_8x4 = benchmark_kernel(A, B, C, M, N, K, 8, 4, kernel_8x4, iters);
        double gf_8x4 = flops / t_8x4 / 1e9;
        printf("  %-12s  %8.2f  %8.1f  %7.1f%%\n", "8x4", t_8x4 * 1e6, gf_8x4, 100.0 * gf_8x4 / gf_accel);
        
        // 4x16 kernel (only if N >= 16)
        if (N >= 16) {
            double t_4x16 = benchmark_kernel(A, B, C, M, N, K, 4, 16, kernel_4x16, iters);
            double gf_4x16 = flops / t_4x16 / 1e9;
            printf("  %-12s  %8.2f  %8.1f  %7.1f%%\n", "4x16", t_4x16 * 1e6, gf_4x16, 100.0 * gf_4x16 / gf_accel);
        }
        
        // 8x8 kernel (only if N >= 8)
        if (N >= 8) {
            double t_8x8 = benchmark_kernel(A, B, C, M, N, K, 8, 8, kernel_8x8, iters);
            double gf_8x8 = flops / t_8x8 / 1e9;
            printf("  %-12s  %8.2f  %8.1f  %7.1f%%\n", "8x8", t_8x8 * 1e6, gf_8x8, 100.0 * gf_8x8 / gf_accel);
        }
        
        printf("\n");
        
        free(A);
        free(B);
        free(C);
    }
    
    return 0;
}
