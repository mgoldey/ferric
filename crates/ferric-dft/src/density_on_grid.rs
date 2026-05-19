//! Compute ρ, ∇ρ, σ = |∇ρ|² on a grid from D and (χ, ∇χ).
//!
//! Closed-shell only this round. The input density matrix is expected to be
//! the **total** density (trace = N_e), as produced by ferric's `ScfResult::density_total`.

use ndarray::{Array1, Array2, Array3};

#[derive(Debug, Clone)]
pub struct DensityGrid {
    /// ρ(r_g), shape (npts,)
    pub rho: Array1<f64>,
    /// ∇ρ(r_g), shape (3, npts) with axis order [x, y, z]
    pub grad: Array2<f64>,
    /// σ(r_g) = |∇ρ(r_g)|², shape (npts,)
    pub sigma: Array1<f64>,
}

/// Closed-shell density on a grid, from the total density matrix.
///
/// Given a total D (tr D = N_e), χ_μ(r_g), and ∇χ_μ(r_g):
///
///   ρ(r)   = Σ_μν D_μν χ_μ(r) χ_ν(r)
///   ∇ρ(r)  = 2 Σ_μν D_μν χ_μ(r) ∇χ_ν(r)
///   σ(r)   = |∇ρ(r)|²
///
/// The ∇ρ expression uses μ↔ν symmetry of D: the full sum
/// Σ_μν D_μν (χ_μ ∇χ_ν + χ_ν ∇χ_μ) equals 2 Σ_μν D_μν χ_μ ∇χ_ν when D = Dᵀ.
pub fn eval_density_closed(
    d: &Array2<f64>,
    chi: &Array2<f64>,        // (nbf, npts)
    dchi: &Array3<f64>,       // (3, nbf, npts)
) -> DensityGrid {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(d.dim(), (nbf, nbf));
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    // Phi_{μg} = Σ_ν D_μν χ_νg  (one GEMM)
    let phi: Array2<f64> = d.dot(chi);

    let mut rho = Array1::<f64>::zeros(npts);
    for g in 0..npts {
        let mut s = 0.0_f64;
        for mu in 0..nbf {
            s += chi[(mu, g)] * phi[(mu, g)];
        }
        rho[g] = s;
    }

    let mut grad = Array2::<f64>::zeros((3, npts));
    for axis in 0..3 {
        for g in 0..npts {
            let mut s = 0.0_f64;
            for mu in 0..nbf {
                s += phi[(mu, g)] * dchi[(axis, mu, g)];
            }
            grad[(axis, g)] = 2.0 * s;
        }
    }

    let mut sigma = Array1::<f64>::zeros(npts);
    for g in 0..npts {
        let gx = grad[(0, g)];
        let gy = grad[(1, g)];
        let gz = grad[(2, g)];
        sigma[g] = gx * gx + gy * gy + gz * gz;
    }

    DensityGrid { rho, grad, sigma }
}
