//! Matrix operations example

use amx::Matrix;

fn main() {
    // Create matrices
    let m1 = Matrix::<f32>::zeros(4, 4).expect("Failed to create matrix");
    let m2 = Matrix::<f32>::zeros(4, 4).expect("Failed to create matrix");

    println!("Matrix 1: {} x {}", m1.dims().0, m1.dims().1);
    println!("Matrix 2: {} x {}", m2.dims().0, m2.dims().1);

    // Transpose
    let m1_t = m1.transpose().expect("Failed to transpose");
    println!("Matrix 1 transposed: {} x {}", m1_t.dims().0, m1_t.dims().1);
}
