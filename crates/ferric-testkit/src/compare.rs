//! Max-abs norms and differences. These return the number; the caller
//! asserts it against its own tolerance.
//!
//! # NaN
//!
//! Both functions return `f64::NAN` if any entry (or difference) is NaN.
//! `f64::max` would silently drop it, and an all-NaN gradient would compare
//! as `0.0` and pass a `< tol` check. With NaN returned, both `< tol` and
//! `> tol` are false, so a NaN fails whichever way the assertion points.
//! An empty matrix still gives `0.0`.

use ndarray::Array2;

fn fold_max<I: Iterator<Item = f64>>(it: I) -> f64 {
    let mut m = 0.0_f64;
    for v in it {
        if v.is_nan() {
            return f64::NAN;
        }
        m = m.max(v);
    }
    m
}

/// Largest absolute entry of `a` (0.0 for an empty matrix, NaN if any entry
/// is NaN).
pub fn max_abs(a: &Array2<f64>) -> f64 {
    fold_max(a.iter().map(|v| v.abs()))
}

/// Largest absolute entrywise difference `max |a - b|` (NaN if any difference
/// is NaN). Panics if the shapes differ, rather than comparing the overlap.
pub fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "max_abs_diff: shape mismatch");
    fold_max(a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn values() {
        let a = array![[1.0, -3.0], [2.0, 0.5]];
        let b = array![[1.5, -3.0], [2.0, 0.5]];
        assert_eq!(max_abs(&a), 3.0);
        assert_eq!(max_abs_diff(&a, &b), 0.5);
        assert_eq!(max_abs(&Array2::<f64>::zeros((0, 3))), 0.0);
    }

    #[test]
    fn nan_propagates() {
        let nan = array![[f64::NAN, f64::NAN]];
        assert!(max_abs(&nan).is_nan());
        let mixed = array![[1.0, f64::NAN, 5.0]];
        assert!(max_abs(&mixed).is_nan());
        assert!(max_abs_diff(&mixed, &array![[1.0, 2.0, 5.0]]).is_nan());
        // Neither direction of a tolerance check passes on NaN.
        let r = max_abs(&nan);
        assert!(!(r < 1e-3) && !(r > 1e-3));
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn shape_mismatch_panics() {
        max_abs_diff(&Array2::<f64>::zeros((2, 2)), &Array2::<f64>::zeros((2, 3)));
    }
}
