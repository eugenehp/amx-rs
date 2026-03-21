//! Vector operations example

use amx::Vector;

fn main() {
    // Create vector
    let mut v = Vector::<f32>::zeros(10).expect("Failed to create vector");

    // Fill with values
    for i in 0..v.len() {
        v.set(i, i as f32).expect("Failed to set value");
    }

    println!("Vector: {}", v);

    // Iterate
    let sum: f32 = v.iter().sum();
    println!("Sum: {}", sum);

    // Get element
    let first = v.get(0).expect("Failed to get element");
    println!("First element: {}", first);
}
