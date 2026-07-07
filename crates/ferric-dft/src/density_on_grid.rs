//! Compute ρ, ∇ρ, σ = |∇ρ|² on a grid from D and (χ, ∇χ).
//!
//! Closed-shell only this round. The input density matrix is expected to be
//! the **total** density (trace = N_e), as produced by ferric's `ScfResult::density_total`.

use ndarray::{Array1, Array2, Array3, Axis, Zip};

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
///
/// The reductions over μ run row-wise (contiguous in the C-order (nbf, npts)
/// layout) with per-point accumulation order identical to the naive
/// point-major loops.
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

    // ρ_g = Σ_μ χ_μg · Φ_μg
    let mut rho = Array1::<f64>::zeros(npts);
    for mu in 0..nbf {
        Zip::from(&mut rho)
            .and(chi.row(mu))
            .and(phi.row(mu))
            .for_each(|r, &c, &p| *r += c * p);
    }

    // ∇ρ_ag = 2 Σ_μ Φ_μg · ∂_a χ_μg
    let mut grad = Array2::<f64>::zeros((3, npts));
    for axis in 0..3 {
        let dchi_axis = dchi.index_axis(Axis(0), axis);
        let mut grow = grad.row_mut(axis);
        for mu in 0..nbf {
            Zip::from(&mut grow)
                .and(phi.row(mu))
                .and(dchi_axis.row(mu))
                .for_each(|g, &p, &dc| *g += p * dc);
        }
        grow.mapv_inplace(|x| 2.0 * x);
    }

    let mut sigma = Array1::<f64>::zeros(npts);
    Zip::from(&mut sigma)
        .and(grad.row(0))
        .and(grad.row(1))
        .and(grad.row(2))
        .for_each(|s, &gx, &gy, &gz| *s = gx * gx + gy * gy + gz * gz);

    DensityGrid { rho, grad, sigma }
}

/// Open-shell density on a grid, from separate α/β density matrices.
///
/// Each `D_σ` should have `tr(D_σ) = N_σ`. Returns:
///   `rho_α`, `rho_β`  : per-spin densities (each integrating to N_σ)
///   `grad_α`, `grad_β`: per-spin density gradients, shape (3, npts)
///   `sigma`           : (3, npts) — rows are σ_αα, σ_αβ, σ_ββ
#[derive(Debug, Clone)]
pub struct UksDensityGrid {
    pub rho_a: Array1<f64>,
    pub rho_b: Array1<f64>,
    pub grad_a: Array2<f64>,
    pub grad_b: Array2<f64>,
    /// (3, npts) — sigma[(0, g)] = σ_αα, sigma[(1, g)] = σ_αβ, sigma[(2, g)] = σ_ββ.
    pub sigma: Array2<f64>,
}

/// Fused α/β evaluation: two GEMMs (one per spin — unavoidable, D differs),
/// then a single pass over χ / ∇χ accumulating both spins at once, and all
/// three σ channels (αα, αβ, ββ) computed together. χ and ∇χ are read once
/// instead of twice, and σ_αα/σ_ββ are not computed twice as they were when
/// this delegated to `eval_density_closed` per spin.
pub fn eval_density_uks(
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
) -> UksDensityGrid {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(d_a.dim(), (nbf, nbf));
    debug_assert_eq!(d_b.dim(), (nbf, nbf));
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    let phi_a: Array2<f64> = d_a.dot(chi);
    let phi_b: Array2<f64> = d_b.dot(chi);

    let mut rho_a = Array1::<f64>::zeros(npts);
    let mut rho_b = Array1::<f64>::zeros(npts);
    for mu in 0..nbf {
        Zip::from(&mut rho_a)
            .and(&mut rho_b)
            .and(chi.row(mu))
            .and(phi_a.row(mu))
            .and(phi_b.row(mu))
            .for_each(|ra, rb, &c, &pa, &pb| {
                *ra += c * pa;
                *rb += c * pb;
            });
    }

    let mut grad_a = Array2::<f64>::zeros((3, npts));
    let mut grad_b = Array2::<f64>::zeros((3, npts));
    for axis in 0..3 {
        let dchi_axis = dchi.index_axis(Axis(0), axis);
        let mut ga = grad_a.row_mut(axis);
        let mut gb = grad_b.row_mut(axis);
        for mu in 0..nbf {
            Zip::from(&mut ga)
                .and(&mut gb)
                .and(phi_a.row(mu))
                .and(phi_b.row(mu))
                .and(dchi_axis.row(mu))
                .for_each(|ga, gb, &pa, &pb, &dc| {
                    *ga += pa * dc;
                    *gb += pb * dc;
                });
        }
        ga.mapv_inplace(|x| 2.0 * x);
        gb.mapv_inplace(|x| 2.0 * x);
    }

    let mut sigma = Array2::<f64>::zeros((3, npts));
    for g in 0..npts {
        let (ax, ay, az) = (grad_a[(0, g)], grad_a[(1, g)], grad_a[(2, g)]);
        let (bx, by, bz) = (grad_b[(0, g)], grad_b[(1, g)], grad_b[(2, g)]);
        sigma[(0, g)] = ax * ax + ay * ay + az * az;
        sigma[(1, g)] = ax * bx + ay * by + az * bz;
        sigma[(2, g)] = bx * bx + by * by + bz * bz;
    }
    UksDensityGrid { rho_a, rho_b, grad_a, grad_b, sigma }
}
