//! Vidberg–Serene Padé continued-fraction analytic continuation
//! (J. Low Temp. Phys. 29, 179, 1977).
//!
//! Given function values f(z_k) at N support points {z_k} (typically on the
//! imaginary axis, z_k = iω_k), construct a continued-fraction
//! representation
//!
//!   C_N(z) = a_1 / (1 + a_2 (z − z_1) / (1 + a_3 (z − z_2) / (... )))
//!
//! and evaluate at arbitrary complex z. The {a_k} are computed by the
//! Thiele recursion on a doubly-indexed g table:
//!
//!   g_1(z_k)     = f(z_k)
//!   g_p(z_k)     = (g_{p-1}(z_{p-1}) − g_{p-1}(z_k)) /
//!                  ((z_k − z_{p-1}) · g_{p-1}(z_k))
//!   a_p          = g_p(z_p)

use ferric_core::FerricError;
use num_complex::Complex64;

/// Padé continued-fraction model.
#[derive(Debug, Clone)]
pub struct PadeCF {
    /// Support nodes (must equal length of `coeffs`).
    pub z: Vec<Complex64>,
    /// Thiele coefficients a_k, length N.
    pub coeffs: Vec<Complex64>,
}

impl PadeCF {
    /// Construct from N support pairs (z_k, f(z_k)).
    ///
    /// The Thiele recursion divides by `(z_k - z_{p-1}) * g_prev[k]`. A
    /// degenerate/repeated support point (`z_k == z_{p-1}`) or a sample that
    /// has converged onto a pole of the running continued fraction
    /// (`g_prev[k] == 0`) drives that denominator to zero, and the quotient
    /// silently becomes NaN/Inf — which would otherwise flow uncaught into
    /// downstream GW quasiparticle energies. Fail loudly instead.
    pub fn fit(z: Vec<Complex64>, f: &[Complex64]) -> Result<Self, FerricError> {
        let n = z.len();
        assert_eq!(f.len(), n, "Padé fit: z and f length mismatch");
        // g[p][k] for p,k in 0..n. We only need the diagonal g[p][p] as a_p
        // and the previous row to advance. Use rolling buffers.
        let mut g_prev: Vec<Complex64> = f.to_vec();
        let mut coeffs = vec![Complex64::new(0.0, 0.0); n];
        coeffs[0] = g_prev[0];
        for p in 1..n {
            let mut g_curr = vec![Complex64::new(0.0, 0.0); n];
            for k in p..n {
                let num = g_prev[p - 1] - g_prev[k];
                let den = (z[k] - z[p - 1]) * g_prev[k];
                if den == Complex64::new(0.0, 0.0) {
                    return Err(FerricError::Convergence(format!(
                        "Padé fit: degenerate Thiele recursion at order p={p}, sample k={k} \
                         (den=0 — repeated support point z[{k}]==z[{}], or g_prev[{k}] hit a \
                         continued-fraction pole); cannot continue analytic continuation",
                        p - 1
                    )));
                }
                let quotient = num / den;
                if !quotient.is_finite() {
                    return Err(FerricError::Convergence(format!(
                        "Padé fit: non-finite Thiele coefficient at order p={p}, sample k={k} \
                         (got {quotient:?}); refusing to propagate NaN/Inf into GW self-energy"
                    )));
                }
                g_curr[k] = quotient;
            }
            coeffs[p] = g_curr[p];
            g_prev = g_curr;
        }
        Ok(Self { z, coeffs })
    }

    /// Evaluate C_N(z) using the bottom-up recursion.
    pub fn eval(&self, z: Complex64) -> Complex64 {
        let n = self.coeffs.len();
        // Start from the innermost: A = 1.
        let mut acc = Complex64::new(1.0, 0.0);
        for p in (1..n).rev() {
            // acc <- 1 + a_p (z - z_{p-1}) / acc
            acc = Complex64::new(1.0, 0.0) + self.coeffs[p] * (z - self.z[p - 1]) / acc;
        }
        self.coeffs[0] / acc
    }

    /// Numerical derivative dC/dz via 4-point central differences on the
    /// imaginary axis (real z evaluations near a small offset). For Σ_c
    /// we evaluate at real ω, so use real-axis central differences.
    pub fn deriv_real(&self, omega: f64, h: f64) -> Complex64 {
        let fp1 = self.eval(Complex64::new(omega + h, 0.0));
        let fm1 = self.eval(Complex64::new(omega - h, 0.0));
        let fp2 = self.eval(Complex64::new(omega + 2.0 * h, 0.0));
        let fm2 = self.eval(Complex64::new(omega - 2.0 * h, 0.0));
        (Complex64::new(-1.0, 0.0) * fp2 + Complex64::new(8.0, 0.0) * fp1
            - Complex64::new(8.0, 0.0) * fm1
            + Complex64::new(1.0, 0.0) * fm2)
            / (12.0 * h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Padé must interpolate exactly at the support points.
    #[test]
    fn interpolation_identity() {
        // Test function: f(z) = 1 / (z² + 1) -- has poles at ±i, real on
        // the imaginary axis above |z| > 1.
        let omegas = [0.1_f64, 0.3, 0.7, 1.5, 3.0, 8.0];
        let z: Vec<Complex64> = omegas.iter().map(|&w| Complex64::new(0.0, w)).collect();
        let f: Vec<Complex64> = z
            .iter()
            .map(|&zi| Complex64::new(1.0, 0.0) / (zi * zi + Complex64::new(1.0, 0.0)))
            .collect();
        let pade = PadeCF::fit(z.clone(), &f).expect("well-conditioned Padé fit must not error");
        for (zi, fi) in z.iter().zip(f.iter()) {
            let pi = pade.eval(*zi);
            let err = (pi - *fi).norm();
            assert!(err < 1e-8, "interpolation err at z={zi:?}: {err}");
        }
    }

    /// GW-realistic regime: ef-shifted nodes (Re = ef ≠ 0), evaluate far away on
    /// the real axis. f(z) = 1/(z−0.5) + 0.3/(z+1.2); 18 geometric nodes at
    /// ef + iω; eval at −0.17. PySCF pade_thiele gives −1.2012751775 (= exact).
    #[test]
    fn shifted_nodes_far_eval_matches_exact() {
        let f = |z: Complex64| {
            Complex64::new(1.0, 0.0) / (z - Complex64::new(0.5, 0.0))
                + Complex64::new(0.3, 0.0) / (z + Complex64::new(1.2, 0.0))
        };
        let ef = -0.0954_f64;
        let n = 18;
        let z: Vec<Complex64> = (0..n)
            .map(|k| {
                let t = k as f64 / (n as f64 - 1.0);
                let w = (0.01_f64.ln() + t * (5.0_f64.ln() - 0.01_f64.ln())).exp();
                Complex64::new(ef, w)
            })
            .collect();
        let fv: Vec<Complex64> = z.iter().map(|&zi| f(zi)).collect();
        let pade = PadeCF::fit(z, &fv).expect("well-conditioned Padé fit must not error");
        let val = pade.eval(Complex64::new(-0.17, 0.0));
        let exact = f(Complex64::new(-0.17, 0.0)).re; // −1.2012751775
        assert!(
            (val.re - exact).abs() < 1e-6,
            "shifted-node far-eval Padé: got {:.10}, exact {:.10}", val.re, exact
        );
    }

    /// On the real axis, f(ω) = 1/(1−ω²) → diverges at ω = ±1. Padé from
    /// imag-axis data should reproduce the simple pole structure.
    #[test]
    fn pole_reproduction() {
        let omegas = [0.2_f64, 0.5, 1.0, 2.0, 4.0, 10.0];
        let z: Vec<Complex64> = omegas.iter().map(|&w| Complex64::new(0.0, w)).collect();
        // f(z) = 1 / (1 - z²)  (so f(iω) = 1/(1+ω²) is real)
        let f: Vec<Complex64> = z
            .iter()
            .map(|&zi| Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) - zi * zi))
            .collect();
        let pade = PadeCF::fit(z, &f).expect("well-conditioned Padé fit must not error");
        // Evaluate at real ω = 0.5 → exact = 1/(1−0.25) = 4/3.
        let val = pade.eval(Complex64::new(0.5, 0.0));
        let exact = 1.0 / (1.0 - 0.25);
        assert!(
            (val.re - exact).abs() < 1e-6 && val.im.abs() < 1e-6,
            "Padé real-axis eval failed: got {val:?}, exact {exact}"
        );
    }

    /// A repeated support node (z_1 == z_0) drives the Thiele recursion's
    /// first denominator, (z_1 − z_0) · g_prev[1], to zero. Before the
    /// guard this silently produced NaN/Inf coefficients that would flow
    /// uncaught into `eval`/`deriv_real`; now `fit` must return `Err`.
    #[test]
    fn fit_errors_on_repeated_support_node() {
        let z = vec![
            Complex64::new(0.0, 0.5),
            Complex64::new(0.0, 0.5), // duplicate of z[0]
            Complex64::new(0.0, 1.0),
        ];
        let f: Vec<Complex64> = z
            .iter()
            .map(|&zi| Complex64::new(1.0, 0.0) / (zi * zi + Complex64::new(1.0, 0.0)))
            .collect();
        let result = PadeCF::fit(z, &f);
        assert!(
            result.is_err(),
            "repeated support node must return Err, not silently propagate NaN/Inf coefficients"
        );
    }
}
