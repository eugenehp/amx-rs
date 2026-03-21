# amx-rs

**High-level ergonomic API for AMX (Apple Matrix eXtensions)**

This crate provides type-safe, ergonomic abstractions over the low-level `amx-sys` instruction set. It includes:

- [`Matrix<T>`] - Generic matrix type with operations
- [`Vector<T>`] - Generic vector type with iterators  
- [`MatMulBuilder`] - Fluent API for matrix multiplication
- [`ConvBuilder`] - Fluent API for convolution operations

## Features

- ✅ Generic over data types
- ✅ Type-safe bounds checking
- ✅ Iterator support
- ✅ Fluent builder API
- ✅ Zero-copy operations where possible
- ✅ `no_std` compatible (with `alloc`)

## Quick Start

```rust
use amx::Matrix;

// Create matrices
let a = Matrix::<f32>::zeros(8, 8)?;
let b = Matrix::<f32>::zeros(8, 8)?;

// Matrix operations
let c = a.transpose()?;
println!("{}", c);
```

## Vectors

```rust
use amx::Vector;

let mut v = Vector::<f32>::zeros(10)?;
v.set(0, 3.14)?;
v.set(1, 2.71)?;

// Iterate
for val in v.iter() {
    println!("{}", val);
}
```

## Algorithms

### Matrix Multiplication

```rust
use amx::algorithms::MatMulBuilder;

let builder = MatMulBuilder::new()
    .transpose_a(false)
    .transpose_b(false)
    .alpha(1.0)
    .beta(0.0);

let c = builder.execute(&a, &b)?;
```

### Convolution

```rust
use amx::algorithms::ConvBuilder;

let conv = ConvBuilder::new(3, 3)  // 3x3 kernel
    .stride(1, 1)
    .padding(1, 1);

let (out_h, out_w) = conv.output_dims(224, 224);
```

## Examples

Run examples:
```bash
cargo run --example matrix_ops
cargo run --example vector_ops
```

## Architecture

- **amx-sys** - Low-level instruction emulation (23 instructions)
- **amx-rs** - High-level API (this crate)

The split enables:
- Low-level research and education (amx-sys)
- Production usage with ergonomic API (amx-rs)
- Independent versioning and updates

## Testing

All algorithms include unit tests:
```bash
cargo test
```

## Performance

- Dimension checking at construction time (when possible)
- Runtime bounds checking for safety
- Zero-copy operations via references
- Minimal memory overhead

## License

MIT OR Apache-2.0
