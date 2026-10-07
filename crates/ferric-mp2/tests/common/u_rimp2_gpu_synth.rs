//! A random but well-posed spin channel without an SCF (occupied eps < 0 <
//! virtual eps, so every denominator is negative).
use crate::fixture::Chan;
use ndarray::Array2;

pub fn synthetic(naux: usize, nocc: usize, nvir: usize, first_occ: usize, seed: u64) -> Chan {
    let mut st = seed;
    let mut next = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 11) as f64 / (1u64 << 53) as f64
    };
    let b = Array2::from_shape_fn((naux, nocc * nvir), |_| 0.1 * (next() - 0.5));
    let nocc_total = first_occ + nocc;
    let mut eps = Vec::new();
    for _ in 0..nocc_total {
        eps.push(-1.5 + 1.2 * next());
    }
    for _ in 0..nvir {
        eps.push(0.2 + 1.8 * next());
    }
    Chan {
        b,
        eps,
        nocc,
        nvir,
        first_occ,
        nocc_total,
    }
}
