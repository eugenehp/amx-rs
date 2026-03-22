# AMX Rust Implementation

[![Benchmark](https://github.com/eugenehp/amx-rs/actions/workflows/bench.yml/badge.svg)](https://github.com/eugenehp/amx-rs/actions/workflows/bench.yml)

**Rust implementation of Apple AMX (Apple Matrix eXtensions) — achieving up to 90% of Apple Accelerate's sgemm performance through reverse-engineered undocumented instruction encodings and cache-optimized algorithms.**

## Performance vs Apple Accelerate

Measured on Apple M4 Pro (Virtual), 10 cores:

| Size | amx-rs (GFLOPS) | Accelerate (GFLOPS) | % of Accelerate |
|------|-----------------|---------------------|-----------------|
| 32×32 | 97 | 277 | 35% |
| 64×64 | 107 | 776 | 14% |
| 128×128 | 330 | 1350 | 24% |
| 256×256 | 660 | 1650 | 40% |
| 512×512 | 695 | 1680 | 41% |
| 1024×1024 | **1500** | **1630** | **92%** |

## Architecture

### Multi-Backend Dispatch

```
matmul(A, B)
├── m=1 or n=1 → Scalar (auto-vectorized, optimal for rank-1)
├── max_dim ≤ 32 → NEON 8×8 µ-kernel (zero AMX setup cost)
├── medium (64-512) → Persistent AMX Thread Pool
│   ├── Pre-pack A (NEON column-gather) and B (row-copy)
│   ├── Distribute (i,j) tile pairs across spin-waiting workers
│   └── 4Y AMX µ-kernel: 4 fma32 per Y-preload cycle
└── large (1024+) → GEBP with L1/L2/L3 cache blocking
    ├── Parallel B̃ packing (rayon par_iter)
    ├── Dynamic MC for thread utilization
    └── QoS pinning to P-cores
```

### AMX µ-Kernel: 4Y Register Preloading

The inner kernel preloads 4 Y registers (A columns) then issues 4 fma32 outer products per X load (B row), giving 4× better compute-to-load ratio:

```c
// Load 4 A columns into Y[0..3]
ldy Y[0] ← A[k+0];  ldy Y[1] ← A[k+1]
ldy Y[2] ← A[k+2];  ldy Y[3] ← A[k+3]

// Each B row: 1 load + 1 fma32 (using Y-row select encoding)
ldx X[0] ← B[k+0];  fma32(0x00)   // Z += X[0] ⊗ Y[0]
ldx X[0] ← B[k+1];  fma32(0x40)   // Z += X[0] ⊗ Y[1]
ldx X[0] ← B[k+2];  fma32(0x80)   // Z += X[0] ⊗ Y[2]
ldx X[0] ← B[k+3];  fma32(0xC0)   // Z += X[0] ⊗ Y[3]
```

### GEBP Cache Blocking (Goto-style, tuned for Apple Silicon)

```
for jc in 0..n step NC=1024:       ← L2 blocking (B̃ = 2 MB)
  Pack B̃ panel (parallel across j-tiles)
  for pc in 0..k step KC=512:      ← L1 blocking (64 KB working set)
    for ic in 0..m step mc_par:    ← dynamic MC for thread balance
      Pack Ã panel (per-thread, private L2)
      GEBP macro-kernel:
        for (ir, jr) tile pairs → 4Y AMX µ-kernel
```

Cache sizing for Apple Silicon M1-M4:
- **L1D**: 64 KB → KC × (MR+NR) × 4 = 64 KB ✓
- **L2**: 4 MB shared → B̃ panel KC × NC × 4 = 2 MB ✓
- **Dynamic MC**: auto-tuned so each thread gets ≥ 2 work items

### Persistent AMX Thread Pool

Workers spin-wait on atomic generation counters (no mutex, no condvar). This eliminates the ~20-100µs thread spawn + AMX init overhead that kills parallelism at small matrix sizes:

```
Main thread                    Worker threads (persistent)
─────────────                  ──────────────────────────
Write jobs to slots            Spin on: generation.load(Acquire)
fence(SeqCst)                  │
generation.fetch_add(1)  ───►  Wake: gen increased!
│                              fence(SeqCst)
│                              amx_set()
│                              Execute tiles
│                              amx_clr()
│                              done_gen.store(gen, Release)
Wait: done_gen >= gen    ◄───  │
```

---

## Reverse Engineering: AMX Instruction Encodings

### fma32 Y-Row Select (Discovered)

The AMX `fma32` instruction's Y-register selection was **undocumented**. We found the correct encoding through brute-force testing on M4 Pro hardware:

```
fma32 operand bits [8:6] = Y register row (0-7)

  Y[0] = 0x000   Y[1] = 0x040   Y[2] = 0x080   Y[3] = 0x0C0
  Y[4] = 0x100   Y[5] = 0x140   Y[6] = 0x180   Y[7] = 0x1C0
```

This was not documented in any public source, including the [corsix/amx](https://github.com/corsix/amx) reverse-engineering project. Previous attempts using bits [19:10] or [21:20] produced incorrect results.

### fma32 Operand Layout (Outer Product Mode)

```
Bit 63:    0 = outer product mode (matrix), 1 = vector mode
Bits 8:6:  Y register row (0-7) — selects which Y row for the outer product
Bits 5:0:  Other flags (Z row offset, etc.)
```

### Load/Store Operand Layout

```
Bits 55:0:   Memory address (pointer)
Bits 62:56:  Register row (0-63 for Z, 0-7 for X/Y)
```

### How Apple Accelerate Uses AMX (Reverse Engineered)

We disassembled `cblas_sgemm_singlecore` from Apple's Accelerate.framework:

```
cblas_sgemm
└→ APPLE_NTHREADS (persistent thread pool dispatch)
   └→ cblas_sgemm_singlecore (3516 insns, 287 AMX ops, ZERO loops)
```

**AMX instruction breakdown in Apple's kernel:**

| Instruction | Count | Purpose |
|-------------|------:|---------|
| fma32 | 185 | Outer product compute |
| extrx | 34 | Extract Z → X (register reuse) |
| extry | 8 | Extract Z → Y (register reuse) |
| vecfp | 30 | Vector add/sub (combine sub-results) |
| ldy | 26 | Load A data |
| ldx | 2 | Load B data |
| stz | 2 | Store results |

**Key insight:** Apple achieves **185 fma32 with only 28 loads** (ratio 6.6:1) by using `extrx`/`extry` to extract intermediate Z results back into X/Y registers for reuse. This is a **register-level recursive algorithm** — not traditional GEBP.

Apple's operand encoding (`0x38000000 | (pair << 20)`) uses bits 29:27=111 which activates a different AMX mode than our outer-product mode. This encoding produces zeros on M4 Pro Virtual — it may require bare-metal execution or chip-specific features.

**Apple's algorithm structure:**
1. 48 initial fma32 with standard operand (compute sub-products)
2. 4 extrx (extract Z rows → X registers)
3. 32 fma32 with various operands (combine using extracted data)
4. More extrx/extry + fma32 cycles (recursive combination)
5. vecfp operations (vector add/subtract for final assembly)
6. Pattern repeats for second half

This is consistent with a **Winograd-Strassen variant** operating at the register level.

---

## Workspace Structure

### [`amx-sys`](crates/amx-sys) — Low-Level AMX Bindings

C-compiled AMX instruction wrappers using the exact `AMX_OP_GPR` encoding from the reference implementation. Includes:
- All 23 AMX instructions
- Runtime AMX availability detection (fork + SIGILL probe)
- Optimized µ-kernels: `amx_f32_tile_kernel`, `amx_f32_tile_kernel_4y`
- NEON helper functions for packing and dot products
- Strided sgemm kernel for zero-copy matmul

### [`amx-rs`](crates/amx-rs) — High-Level API

Ergonomic matrix/vector operations with automatic backend dispatch:
- `Matrix<T>`, `Vector<T>` generic types
- `matmul()` — smart dispatch across all backends
- `matmul_amx()` — single-threaded AMX with pre-packing
- `matmul_pool()` — persistent thread pool dispatch
- `matmul_gebp()` / `matmul_gebp_parallel()` — full GEBP
- `matmul_recursive()` — cache-oblivious recursive (experimental)
- `matmul_neon()` — NEON 8×8 for tiny matrices
- `matmul_scalar()` — auto-vectorized fallback
- `no_std` compatible (without `std` feature)

## Quick Start

```rust
use amx::Matrix;

let a = Matrix::from_data(vec![1.0f32; 1024*1024], 1024, 1024)?;
let b = Matrix::from_data(vec![1.0f32; 1024*1024], 1024, 1024)?;

// Automatic dispatch: NEON → Pool → GEBP based on size
let c = a.matmul(&b)?;
```

## Building & Testing

```bash
# Run all 33 tests
cargo test --workspace --release

# Benchmark vs Accelerate
cargo bench -p amx-rs --bench scalar_vs_amx -- --nocapture

# Control benchmark parameters
BENCH_MAX_N=1024 BENCH_ITERS=20 cargo bench -p amx-rs --bench scalar_vs_amx -- --nocapture
```

## Optimization History

Each optimization was validated through ablation studies:

| Optimization | Impact | Status |
|-------------|--------|--------|
| AMX 16×16 tile kernel | Baseline 266 GF at 256 | ✅ Shipped |
| Double-buffered X/Y loads | +5% µ-kernel throughput | ✅ Shipped |
| NEON dispatch for N≤32 | 113 GF (beats Accelerate!) | ✅ Shipped |
| Persistent AMX thread pool | 2× at N=128-512 | ✅ Shipped |
| GEBP L1/L2/L3 cache blocking | 1400+ GF at N=1024 | ✅ Shipped |
| Dynamic MC for thread balance | +40% at N=1024 | ✅ Shipped |
| NC=1024 (B̃ fits in shared L2) | +60-90% GEBP single-thread | ✅ Shipped |
| Parallel B̃ packing | Up to 97.8% of Accelerate | ✅ Shipped |
| QoS pinning to P-cores | +3-5% parallel | ✅ Shipped |
| NEON A-panel packing | +5-10% GEBP single-core | ✅ Shipped |
| 4Y µ-kernel (multi-Y fma32) | 4 fma32 per Y-preload | ✅ Shipped |
| Strided AMX kernel (no-copy) | Too slow (gather dominates) | ❌ Kept as reference |
| Recursive cache-oblivious | Slower than GEBP (pack overhead per leaf) | ❌ Experimental |
| KC=256 (smaller L1 block) | Worse (more packing) | ❌ Reverted |
| NC=512 (smaller L2 block) | Worse (too much repacking) | ❌ Reverted |
| Lower AMX-par threshold | Worse at N≤256 (thread overhead) | ❌ Reverted |

## Coverage

| Category | Count | Status |
|----------|-------|--------|
| AMX Instructions | 23/23 | ✅ |
| Tests | 33 | ✅ |
| Pass Rate | 100% | ✅ |
| Matmul Backends | 7 | ✅ |
| Data Types | f32 (primary), f64 (precision) | ✅ |

## License

MIT OR Apache-2.0

## Acknowledgments

- [corsix/amx](https://github.com/corsix/amx) — AMX instruction set reverse engineering
- Apple Accelerate.framework — disassembly analysis for algorithm insights
- Goto & Van de Geijn — GEBP algorithm design
