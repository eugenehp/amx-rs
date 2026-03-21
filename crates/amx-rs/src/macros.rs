//! Convenience macros for common operations

/// Create a matrix from dimensions
/// 
/// # Example
/// ```ignore
/// let m = matrix!(4, 4, f32);
/// ```
#[macro_export]
macro_rules! matrix {
    ($rows:expr, $cols:expr, $type:ty) => {
        <Matrix<$type>>::zeros($rows, $cols)
    };
}

/// Create a vector
/// 
/// # Example
/// ```ignore
/// let v = vector!(10, f32);
/// ```
#[macro_export]
macro_rules! vector {
    ($len:expr, $type:ty) => {
        <Vector<$type>>::zeros($len)
    };
}

/// Quick matrix multiplication
/// 
/// # Example
/// ```ignore
/// let c = matmul!(a, b);
/// ```
#[macro_export]
macro_rules! matmul {
    ($a:expr, $b:expr) => {
        MatMulBuilder::new().execute(&$a, &$b)
    };
}
