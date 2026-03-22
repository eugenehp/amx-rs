# Optimization Plan: Beating Apple Accelerate

## Goal
Achieve competitive or superior performance to Apple Accelerate for matrix multiplication and dot products across all sizes.

## Current State (2026-03-22)

| Operation | amx-rs | Accelerate | Gap |
|-----------|-------:|-----------:|----:|
| Matmul N=16 | 57 GF | 15 GF | **+280%** ✓ |
| Matmul N=32 | 58 GF | 95 GF | -39% |
| Matmul N=64 | 101 GF | 471 GF | -79% |
| Matmul N=128 | 168 GF | 1020 GF | -84% |
| Matmul N=512 | 678 GF | 1624 GF | -58% |
| Dot N=1024 | 34 GF | 38 GF | -11% |
| Dot N=1M | 32 GF | 224 GF | -86% |

## Experiment Results

### EXP-001: NEON Micro-Kernel Sizes ✅

| N | 4x4 | 8x4 | 4x16 | 8x8 | Accelerate |
|---|-----|-----|------|-----|------------|
| 16 | 54.8 GF | 65.7 GF | 90.0 GF | **97.8 GF** | 39.8 GF |
| 32 | 64.5 GF | 72.2 GF | 111 GF | **115 GF** | 280 GF |
| 64 | 54.8 GF | 67.8 GF | 114 GF | **116 GF** | 855 GF |

**Finding**: 8x8 kernel is best. Pure NEON hits ~115 GFLOPS ceiling (memory-bound).

### EXP-002: Cache-Blocked NEON GEMM ✅

Even with BLIS-style 5-loop blocking:
- N=512: Best config = 115 GFLOPS (7% of Accelerate)
- **Conclusion**: Pure NEON can't compete for N > 64. Must use AMX.

### EXP-003: AMX Optimization Strategies ✅

| N | scalar+basic | neon4+pipelined | Accelerate |
|---|-------------|-----------------|------------|
| 64 | 89.7 GF | 104 GF | 788 GF |
| 128 | 153 GF | 179 GF | 1344 GF |
| 256 | 254 GF | 270 GF | 1731 GF |
| 512 | 353 GF | **370 GF** | 1670 GF |

**Finding**: NEON packing gives 5-15% improvement. Single-threaded AMX maxes at ~370 GFLOPS.

### EXP-004: Parallel AMX ✅ 🎯

| N | 1-thread | 4-thread | 8-thread | Accelerate | % of Accel |
|---|----------|----------|----------|------------|------------|
| 256 | 265 GF | 327 GF | 247 GF | 1691 GF | 19% |
| 512 | 369 GF | 865 GF | **1000 GF** | 1671 GF | **60%** |
| 1024 | 418 GF | 1014 GF | **1469 GF** | 1619 GF | **91%** |

**Finding**: 8-thread parallel AMX achieves 91% of Accelerate at N=1024!

## Optimal Strategy

```
Size        Backend              Expected Performance
─────────────────────────────────────────────────────
N ≤ 16      NEON 8x8 µkernel    >100% of Accelerate ✓
N = 17-32   NEON 8x8 µkernel    ~60% of Accelerate
N = 33-63   AMX single-thread   ~15% of Accelerate  
N ≥ 64      AMX 8-thread        60-91% of Accelerate
```

## Next Steps

1. [x] EXP-001: Micro-kernel comparison
2. [x] EXP-002: Cache blocking 
3. [x] EXP-003: AMX strategies
4. [x] EXP-004: Parallel scaling
5. [ ] Implement optimized 8x8 NEON kernel in library
6. [ ] Improve parallel AMX in library with findings
7. [ ] Tune thresholds based on experiment data
8. [ ] Benchmark dot product optimizations

## Implementation Priority

1. **HIGH**: Implement 8x8 NEON micro-kernel (beats Accelerate for N≤16)
2. **HIGH**: Fix parallel AMX scaling (currently hitting ~700 GF, should get ~1000)
3. **MEDIUM**: Tune dispatch thresholds
4. **LOW**: Dot product (already 85% of Accelerate with NEON)

