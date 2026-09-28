//! Exp-scaled modified spherical Bessel functions `k̃_n(z) = e^{−z} i_n(z)`.
//!
//! `k̃_n` is bounded (`0 ≤ k̃_n ≤ 1`), `k̃_n(0) = δ_{n0}`, and every regime below
//! is either a positive-term sum or a recurrence run in its stable direction,
//! so there is no cancellation at any `z` (FINDINGS "Iteration 25", Choices).
//!
//! Regimes (identical to the prototype `ecpq.bessel_ktil`):
//! * `z = 0`: `δ_{n0}` (the on-centre case is exact, not a limit).
//! * `0 < z < 1e-8`: `e^{−z} zⁿ/(2n+1)!! (1 + z²/(2(2n+3)))` — avoids the `zⁿ`
//!   underflow that would poison the recurrence.
//! * `1e-8 ≤ z < max(16, 4 n_max)`: the all-positive power series for `n_max`
//!   and `n_max + 1` only, then the DOWNWARD recurrence
//!   `k̃_{n−1} = k̃_{n+1} + (2n+1)/z k̃_n` (`i_n` is the minimal solution in n).
//! * `z ≥ max(16, 4 n_max)`: the UPWARD recurrence
//!   `k̃_{n+1} = k̃_{n−1} − (2n+1)/z k̃_n` from the closed forms of `k̃_0`, `k̃_1`.
//!
//! The series runs on reciprocal tables (`1/(k(2n+2k+1))`, `1/(2n+1)!!`), so the
//! hot path has no divisions (the C++ stand-in measured 150–270 → ~100 ns/node).

use std::sync::OnceLock;

/// Largest order `n` [`ktil`] supports (type-2 `λ ≤ l_ECP + l_shell + 1`,
/// type-1 `λ ≤ l_a + l_b + 1`; the validated ranges stay well inside).
pub const KTIL_NMAX: usize = 16;

/// Series terms available (the series converges by k ≈ z/2 + O(√z) terms;
/// z < 4·16 = 64 needs < 100).
const SERIES_KMAX: usize = 192;

struct Tables {
    /// `recip[n][k] = 1/(k (2n + 2k + 1))`, `n ≤ KTIL_NMAX + 1`.
    recip: Vec<[f64; SERIES_KMAX]>,
    /// `idf[n] = 1/(2n+1)!!`.
    idf: [f64; KTIL_NMAX + 2],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut idf = [0.0; KTIL_NMAX + 2];
        let mut recip = Vec::with_capacity(KTIL_NMAX + 2);
        for (n, slot) in idf.iter_mut().enumerate() {
            let mut d = 1.0_f64;
            let mut j = 1;
            while j <= 2 * n + 1 {
                d *= j as f64;
                j += 2;
            }
            *slot = 1.0 / d;
            let mut row = [0.0; SERIES_KMAX];
            for (k, v) in row.iter_mut().enumerate().skip(1) {
                *v = 1.0 / (k as f64 * (2.0 * n as f64 + 2.0 * k as f64 + 1.0));
            }
            recip.push(row);
        }
        Tables { recip, idf }
    })
}

/// Upward/downward switch point for a set up to order `nmax`.
#[inline]
pub fn z_switch(nmax: usize) -> f64 {
    (4.0 * nmax as f64).max(16.0)
}

/// `zn · Σ_k (z²/2)^k / (k! (2n+3)(2n+5)⋯(2n+2k+1)) / (2n+1)!!` — `k̃_n(z)` when
/// `zn = e^{−z} zⁿ`.
#[inline]
fn series(t: &Tables, n: usize, z: f64, zn: f64) -> f64 {
    let h = 0.5 * z * z;
    let r = &t.recip[n];
    let mut term = 1.0;
    let mut s = 1.0;
    for &rk in r.iter().skip(1) {
        term *= h * rk;
        s += term;
        if term <= 1e-17 * s {
            break;
        }
    }
    zn * t.idf[n] * s
}

/// `out[n] = k̃_n(z)` for `n = 0..=nmax`, `z ≥ 0`.
///
/// # Panics
/// If `nmax > KTIL_NMAX`, `out.len() <= nmax` or `z` is negative / NaN (all
/// internal-contract violations: callers validate angular momenta upstream).
pub fn ktil(nmax: usize, z: f64, out: &mut [f64]) {
    assert!(nmax <= KTIL_NMAX, "ktil: nmax {nmax} > {KTIL_NMAX}");
    assert!(out.len() > nmax, "ktil: output too short");
    assert!(z >= 0.0, "ktil: z = {z} must be >= 0");
    let out = &mut out[..=nmax];
    if z == 0.0 {
        out.fill(0.0);
        out[0] = 1.0;
        return;
    }
    let t = tables();
    if z < z_switch(nmax) {
        let ez = (-z).exp();
        if z < 1e-8 {
            let mut zn = ez;
            for (n, o) in out.iter_mut().enumerate() {
                *o = zn * t.idf[n] * (1.0 + z * z / (2.0 * (2.0 * n as f64 + 3.0)));
                zn *= z;
            }
            return;
        }
        let mut zn = ez;
        for _ in 0..nmax {
            zn *= z;
        }
        let mut kn = series(t, nmax, z, zn);
        let mut kp1 = series(t, nmax + 1, z, zn * z);
        let iz = 1.0 / z;
        out[nmax] = kn;
        for n in (1..=nmax).rev() {
            let km1 = kp1 + (2 * n + 1) as f64 * iz * kn;
            out[n - 1] = km1;
            kp1 = kn;
            kn = km1;
        }
        return;
    }
    let e2 = (-2.0 * z).exp();
    let iz = 1.0 / z;
    out[0] = 0.5 * (1.0 - e2) * iz;
    if nmax >= 1 {
        out[1] = 0.5 * (1.0 + e2) * iz - 0.5 * (1.0 - e2) * iz * iz;
    }
    for n in 1..nmax {
        out[n + 1] = out[n - 1] - (2 * n + 1) as f64 * iz * out[n];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed forms at moderate z (the pinned mpmath table lives in
    /// tests/ecp_quadrature.rs): k̃_0 = (1 − e^{−2z})/(2z).
    #[test]
    fn ktil0_closed_form_all_regimes() {
        let mut out = [0.0; KTIL_NMAX + 1];
        for &z in &[1e-9, 1e-6, 0.3, 2.0, 15.9, 16.1, 63.0, 70.0, 1e4] {
            for nmax in [0, 3, 12, 16] {
                ktil(nmax, z, &mut out);
                let want = -(-2.0 * z).exp_m1() / (2.0 * z);
                assert!(
                    (out[0] - want).abs() <= 4e-15 * want,
                    "z {z} nmax {nmax}: {} vs {want}",
                    out[0]
                );
            }
        }
        ktil(5, 0.0, &mut out);
        assert_eq!(&out[..6], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
