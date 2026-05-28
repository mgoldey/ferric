//! Physical-anchor tests for the TS dispersion C6 path.

use ferric_rpa::{casimir_polder_c6, ts_dynamic_polarizability};

/// Fine trapezoid imaginary-frequency grid; integrates the Casimir-Polder
/// α(iω)α(iω) product to <1% for the single-pole London model.
fn freq_grid() -> (Vec<f64>, Vec<f64>) {
    let n = 20000usize;
    let wmax = 200.0_f64;
    let dw = wmax / (n as f64);
    let mut f = Vec::with_capacity(n + 1);
    let mut w = Vec::with_capacity(n + 1);
    for k in 0..=n {
        f.push(k as f64 * dw);
        w.push(if k == 0 || k == n { 0.5 * dw } else { dw });
    }
    (f, w)
}

#[test]
fn free_atom_c6_matches_ts_reference() {
    let (freqs, weights) = freq_grid();
    // Free H and free O at ratio 1.0, isotropic static α = α_free.
    let z = vec![1usize, 8usize];
    let ratio = vec![1.0, 1.0];
    let alpha_static = vec![
        [[4.5, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]],
        [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]],
    ];
    let dp = ts_dynamic_polarizability(&z, &ratio, &alpha_static, &freqs, &weights);
    let res = casimir_polder_c6(&dp);

    // Homonuclear C6 reproduces the table by construction.
    let c6_hh = res.c6_iso_pair[(0, 0)];
    let c6_oo = res.c6_iso_pair[(1, 1)];
    assert!((c6_hh - 6.5).abs() / 6.5 < 3e-3, "C6(H-H)={c6_hh}");
    assert!((c6_oo - 15.6).abs() / 15.6 < 3e-3, "C6(O-O)={c6_oo}");

    // Pair-matrix symmetry.
    let c6_ho = res.c6_iso_pair[(0, 1)];
    let c6_oh = res.c6_iso_pair[(1, 0)];
    assert!((c6_ho - c6_oh).abs() / c6_ho < 1e-12, "asymmetric C6 matrix");

    // Heteronuclear C6(H-O) finite, positive, between the two homonuclear scales.
    assert!(c6_ho > 0.0 && c6_ho.is_finite());
    assert!(c6_ho > c6_hh.min(c6_oo) * 0.5, "C6(H-O)={c6_ho} too small");
    assert!(c6_ho < c6_hh.max(c6_oo) * 1.1, "C6(H-O)={c6_ho} too large");
}
