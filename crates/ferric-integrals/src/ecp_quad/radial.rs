//! Radial rule of the ECP quadrature (FINDINGS "Iteration 25", Choices).
//!
//! Every (primitive pair, ECP term) integrand is `e^{−p(r − r0)²}` times a
//! slowly varying analytic factor of polynomial degree `deg`:
//! * interior windows (`r0 √p ≥ RAD_TAU`): Gauss–Hermite, `RAD_NH` nodes,
//!   `r = r0 + x/√p` (every node at r > 0; the neglected mass below r = 0 is
//!   ≤ e^{−42});
//! * all others: Gauss–Legendre on `[max(0, r0 − T/√p), max(r0, r_pk) + T/√p]`
//!   (`r_pk` = argmax of `r^deg e^{−p(r−r0)²}`) with
//!   `clamp(⌈RAD_PER_SIGMA · (hi − lo) · √p⌉, RAD_NMIN, RAD_N)` nodes.
//!
//! Measured per window over 42.8k windows: max relative error 5.2e-14, mean
//! 27.7 nodes (the prototype's `radial_scheme.py`). The constants live here
//! and nowhere else.

use std::sync::OnceLock;

/// Window half-width in units of `1/√p` (Gauss–Legendre windows).
pub const RAD_T: f64 = 6.0;
/// Maximum Gauss–Legendre nodes per window.
pub const RAD_N: usize = 40;
/// Minimum Gauss–Legendre nodes per window.
pub const RAD_NMIN: usize = 16;
/// Gauss–Legendre nodes per unit of `(window length) · √p`.
pub const RAD_PER_SIGMA: f64 = 3.4;
/// Gauss–Hermite nodes for interior windows.
pub const RAD_NH: usize = 20;
/// Interior iff `r0 √p ≥ RAD_TAU`.
pub const RAD_TAU: f64 = 6.5;
/// Largest rule the tables hold (node buffers are this long).
pub const MAX_NODES: usize = 64;

/// The node-count policy of one call (production = [`RadialRule::default`];
/// the under-resolved negative control lowers it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RadialRule {
    /// Gauss–Legendre maximum (`RAD_N`).
    pub gl_max: usize,
    /// Gauss–Legendre minimum (`RAD_NMIN`).
    pub gl_min: usize,
    /// Gauss–Hermite count (`RAD_NH`).
    pub gh: usize,
}

impl Default for RadialRule {
    fn default() -> Self {
        Self {
            gl_max: RAD_N,
            gl_min: RAD_NMIN,
            gh: RAD_NH,
        }
    }
}

type Rule = (Vec<f64>, Vec<f64>);

/// Gauss–Legendre on [−1, 1] (Newton on the three-term recurrence, then one
/// polishing step).
fn gauss_legendre(n: usize) -> Rule {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut pp = 0.0;
        let mut polish = false;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0_f64, 0.0_f64);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            pp = nf * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if polish {
                break;
            }
            if (z - z1).abs() < 1e-15 {
                polish = true;
            }
        }
        x[i] = -z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
    (x, w)
}

/// Gauss–Hermite (weight e^{−x²}) nodes, weights MULTIPLIED by e^{x²} (the
/// envelope stays explicit in the integrand). Numerical Recipes `gauher`
/// initial guesses, Newton on the orthonormal recurrence, one polishing step.
fn gauss_hermite(n: usize) -> Rule {
    const PIM4: f64 = 0.751_125_544_464_942_5; // π^{−1/4}
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    let m = n.div_ceil(2);
    let mut z = 0.0_f64;
    for i in 0..m {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-0.16667),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 0.0;
        let mut polish = false;
        for _ in 0..100 {
            let (mut p1, mut p2) = (PIM4, 0.0_f64);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = z * (2.0 / (j + 1) as f64).sqrt() * p2
                    - (j as f64 / (j + 1) as f64).sqrt() * p3;
            }
            pp = (2.0 * nf).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if polish {
                break;
            }
            if (z - z1).abs() < 1e-15 {
                polish = true;
            }
        }
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp) * (z * z).exp();
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

fn gl_table(n: usize) -> &'static Rule {
    static T: OnceLock<Vec<Rule>> = OnceLock::new();
    assert!((1..=MAX_NODES).contains(&n), "Gauss-Legendre n = {n}");
    &T.get_or_init(|| {
        (0..=MAX_NODES)
            .map(|k| {
                if k == 0 {
                    (vec![], vec![])
                } else {
                    gauss_legendre(k)
                }
            })
            .collect()
    })[n]
}

fn gh_table(n: usize) -> &'static Rule {
    static T: OnceLock<Vec<OnceLock<Rule>>> = OnceLock::new();
    assert!((1..=MAX_NODES).contains(&n), "Gauss-Hermite n = {n}");
    T.get_or_init(|| (0..=MAX_NODES).map(|_| OnceLock::new()).collect())[n]
        .get_or_init(|| gauss_hermite(n))
}

/// Nodes/weights (weights include the Jacobian) for one window of the
/// envelope `e^{−p (r − r0)²}` with polynomial degree `deg`; writes the first
/// `k` entries of `r`/`w` and returns `k`.
#[inline]
pub fn nodes(
    p: f64,
    r0: f64,
    deg: i32,
    rule: &RadialRule,
    r: &mut [f64; MAX_NODES],
    w: &mut [f64; MAX_NODES],
) -> usize {
    let sp = p.sqrt();
    if r0 * sp >= RAD_TAU {
        let (x, wt) = gh_table(rule.gh);
        let isp = 1.0 / sp;
        for i in 0..rule.gh {
            r[i] = r0 + x[i] * isp;
            w[i] = wt[i] * isp;
        }
        return rule.gh;
    }
    let lo = (r0 - RAD_T / sp).max(0.0);
    let rpk = 0.5 * (r0 + (r0 * r0 + 2.0 * f64::from(deg.max(0)) / p).sqrt());
    let hi = r0.max(rpk) + RAD_T / sp;
    let want = (RAD_PER_SIGMA * (hi - lo) * sp).ceil();
    let n = if want.is_finite() {
        (want as usize).clamp(rule.gl_min, rule.gl_max)
    } else {
        rule.gl_max
    };
    let (x, wt) = gl_table(n);
    let (a0, a1) = (0.5 * (hi + lo), 0.5 * (hi - lo));
    for i in 0..n {
        r[i] = a0 + a1 * x[i];
        w[i] = a1 * wt[i];
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Moments: GL-n integrates x^k exactly for k < 2n; GH-n (with e^{x²}
    /// folded into the weights) integrates x^k e^{−x²} for k < 2n.
    #[test]
    fn rules_integrate_their_polynomial_space() {
        for n in [8usize, 16, 20, 27, 40] {
            let (x, w) = gauss_legendre(n);
            for k in 0..2 * n {
                let got: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(k as i32)).sum();
                let want = if k % 2 == 1 {
                    0.0
                } else {
                    2.0 / (k + 1) as f64
                };
                assert!((got - want).abs() < 2e-14, "GL{n} x^{k}: {got} vs {want}");
            }
            let (x, w) = gauss_hermite(n);
            for k in (0..2 * n).step_by(2).take(12) {
                let got: f64 = x
                    .iter()
                    .zip(&w)
                    .map(|(x, w)| w * (-x * x).exp() * x.powi(k as i32))
                    .sum();
                // Γ((k+1)/2) = (k−1)!! √π / 2^{k/2}
                let mut df = 1.0;
                let mut j = k as i64 - 1;
                while j > 1 {
                    df *= j as f64;
                    j -= 2;
                }
                let want = df * std::f64::consts::PI.sqrt() / 2f64.powi(k as i32 / 2);
                assert!(
                    (got - want).abs() < 1e-13 * want,
                    "GH{n} x^{k}: {got} vs {want}"
                );
            }
        }
    }
}
