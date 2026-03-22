<!-- Auto-generated benchmark results -->
<!-- Last updated: 2026-03-22 -->
<!-- Chip: Apple M4 Pro (Virtual) -->

# Benchmark Results

> Measured on **Apple M4 Pro (Virtual)** · f32 · 2026-03-22

## Performance Summary

| Size | amx-rs Smart | Accelerate | Ratio |
|------|-------------:|-----------:|------:|
| N=8 | **24.8 GF** | 15.2 GF | **163%** ✓ |
| N=16 | **66.2 GF** | 15.0 GF | **441%** ✓ |
| N=32 | **69.8 GF** | 100 GF | 70% |
| N=64 | 91 GF | 469 GF | 19% |
| N=128 | 167 GF | 1032 GF | 16% |
| N=512 | **715 GF** | 1645 GF | 43% |
| Dot 1K | **49.2 GF** | 33.7 GF | **146%** ✓ |

**Key wins**: Small matrices (N≤16) and dot products beat Accelerate!

## Optimizations Implemented

1. **NEON 8×8 micro-kernel** for small matrices (N ≤ 32)
2. **NEON-vectorized A packing** for AMX path
3. **NEON dot product** with 4-way parallel accumulators
4. **Smart dispatch** - routes to optimal backend by size

## Square Matmul (N×N × N×N)

| N | Scalar (µs) | Smart (µs) | AMX-par (µs) | Accel (µs) | Smart GF | Accel GF | Ratio |
|--:|------------:|-----------:|-------------:|-----------:|---------:|---------:|------:|
| 8 | 0.18 | 0.04 | 0.53 | 0.07 | **24.8** | 15.2 | **163%** |
| 16 | 0.50 | 0.12 | 0.52 | 0.55 | **66.2** | 15.0 | **441%** |
| 32 | 3.1 | 0.94 | 1.4 | 0.65 | **69.8** | 100 | 70% |
| 64 | 16.9 | 5.8 | 5.2 | 1.1 | 91.0 | 469 | 19% |
| 128 | 111 | 25.1 | 24.6 | 4.1 | 167 | 1032 | 16% |
| 256 | 968 | 174 | 148 | 24.2 | 192 | 1385 | 14% |
| 512 | 7823 | 392 | 376 | 163 | 684 | 1645 | 42% |

## Dot Product (a · b)

| Length | Scalar (µs) | NEON (µs) | Accel (µs) | NEON GF | Accel GF | Ratio |
|-------:|------------:|----------:|-----------:|--------:|---------:|------:|
| 64 | 0.013 | 0.004 | 0.007 | **34.1** | 18.1 | **188%** |
| 256 | 0.100 | 0.014 | 0.015 | **37.2** | 33.2 | **112%** |
| 1024 | 0.432 | 0.042 | 0.061 | **49.2** | 33.7 | **146%** |
| 16384 | 8.1 | 0.85 | 0.23 | 38.5 | 145 | 27% |
| 1M | 516 | 65 | 9.4 | 32 | 224 | 14% |

## Architecture

```
matmul() dispatch:
├── N ≤ 32 && M×K×N ≤ 32K  →  NEON 8×8 µkernels (beats Accelerate!)
└── Otherwise              →  AMX 16×16 tiles + parallel

dot() dispatch:
└── Always                 →  NEON 4-way parallel (beats Accelerate for N≤1K)
```

## Experimental Results

See `experiments/` directory and `PLAN.md` for detailed optimization experiments.

Key findings from experiments:
- **EXP-001**: 8×8 NEON kernel achieves 98 GFLOPS (best of tested sizes)
- **EXP-003**: NEON packing gives 5-15% improvement over scalar
- **EXP-004**: 8-thread parallel AMX achieves 91% of Accelerate at N=1024
