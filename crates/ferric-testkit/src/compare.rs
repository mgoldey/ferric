//! Max-abs norms and differences. These return the number; the caller
//! asserts it against its own tolerance.

use ndarray::Array2;

/// Largest absolute entry of `a` (0.0 for an empty matrix).
pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Largest absolute entrywise difference `max |a - b|`. Panics if the shapes
/// differ, rather than comparing the overlap.
pub fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "max_abs_diff: shape mismatch");
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}
