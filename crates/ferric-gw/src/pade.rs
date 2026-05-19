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
    pub fn fit(z: Vec<Complex64>, f: &[Complex64]) -> Self {
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
                g_curr[k] = num / den;
            }
            coeffs[p] = g_curr[p];
            g_prev = g_curr;
        }
        Self { z, coeffs }
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
        let pade = PadeCF::fit(z.clone(), &f);
        for (zi, fi) in z.iter().zip(f.iter()) {
            let pi = pade.eval(*zi);
            let err = (pi - *fi).norm();
            assert!(err < 1e-8, "interpolation err at z={zi:?}: {err}");
        }
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
        let pade = PadeCF::fit(z, &f);
        // Evaluate at real ω = 0.5 → exact = 1/(1−0.25) = 4/3.
        let val = pade.eval(Complex64::new(0.5, 0.0));
        let exact = 1.0 / (1.0 - 0.25);
        assert!(
            (val.re - exact).abs() < 1e-6 && val.im.abs() < 1e-6,
            "Padé real-axis eval failed: got {val:?}, exact {exact}"
        );
    }
}
