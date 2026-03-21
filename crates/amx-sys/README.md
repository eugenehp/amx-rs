# amx-sys

**Low-level AMX instruction emulation - Hardware-faithful implementation**

This crate provides direct bindings to all 23 AMX (Apple Matrix eXtensions) instructions with a faithful emulation of the Apple silicon behavior.

## Features

- ✅ All 23 AMX instructions
- ✅ Complete register file emulation (8 X, 8 Y, 64 Z registers)
- ✅ Full data type support (i8-i64, u8-u64, f32, f64)
- ✅ All element sizes (B8, B16, B32, B64)
- ✅ All shuffle modes (S0, S1, S2, S3)
- ✅ 100% parity with C reference implementation
- ✅ 100 tests (100% pass rate)

## Instructions

### Load/Store (8)
- `ldx`, `ldy`, `ldz`, `ldzi` - Load X, Y, Z registers
- `stx`, `sty`, `stz`, `stzi` - Store registers

### Extract (2)
- `extrx` - Extract from X register with shuffle
- `extry` - Extract from Y register with shuffle

### Floating-Point (6)
- `fma16`, `fma32`, `fma64` - Multiply-accumulate
- `fms16`, `fms32`, `fms64` - Multiply-subtract

### Integer (2)
- `mac16` - Signed 16-bit multiply-accumulate
- `mac16_unsigned` - Unsigned multiply-accumulate

### Vector (2)
- `vecint` - Vector integer operations
- `vecfp` - Vector floating-point operations

### Matrix (2)
- `matint` - Matrix integer operations
- `matfp` - Matrix floating-point operations

### Lookup Table (1)
- `genlut` - Generate lookup table

## Usage

```rust
use amx_sys::registers::AmxState;
use amx_sys::instructions::ldst::*;
use amx_sys::instructions::fma::fma32;

let mut state = AmxState::new();

// Load data
let data = [0u8; 64];
ldx(&mut state, 0, &data);
ldy(&mut state, 0, &data);
ldz(&mut state, 0, &data);

// Perform FMA
fma32(&mut state, 0, 0, 0);

// Store result
let result = stx(&state, 0);
```

## Testing

All 23 instructions are thoroughly tested with:
- 100 test cases (100% pass rate)
- Randomized inputs (xoshiro256++ RNG)
- All data types and element sizes
- 1,000+ iterations of stress testing

Run tests:
```bash
cargo test
```

## Benchmarking

Run the comprehensive IO-vs-compute benchmark:

```bash
cargo bench -p amx-sys --bench io_vs_compute
```

The benchmark reports CSV-style rows with separate `io_only_ns`, `compute_only_ns`, and `end_to_end_ns` timings for:
- Extract shuffle modes S0-S3
- FMA and FMS precisions
- MAC16 signed and unsigned
- VECINT and MATINT for all signed and unsigned element sizes
- VECFP and MATFP for all floating-point precisions
- GENLUT for all supported element sizes

Useful filters:

```bash
AMX_BENCH_FILTER=matfp AMX_BENCH_SAMPLES=3 cargo bench -p amx-sys --bench io_vs_compute
```

## Performance

- All operations run in constant time
- Zero heap allocations
- Pure Rust with minimal unsafe code

## License

MIT OR Apache-2.0
