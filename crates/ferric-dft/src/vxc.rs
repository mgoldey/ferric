//! Semilocal V_xc and E_xc assembly on a Becke-Lebedev grid.
//!
//! For an LDA functional:
//!   V_xc_μν = Σ_g w_g · v_ρ(r_g) · χ_μ(r_g) · χ_ν(r_g)
//!   E_xc   = Σ_g w_g · ρ(r_g) · ε_xc(r_g)
//!
//! For a GGA functional, add a v_σ-coupled term:
//!   V_xc_μν += Σ_g w_g · 2 · v_σ(r_g) · Σ_a ∇ρ_a(r_g) ·
//!                [χ_μ(r_g) · ∂χ_ν/∂a(r_g) + χ_ν(r_g) · ∂χ_μ/∂a(r_g)]
//!
//! The implementation uses one GEMM per term — χ (or ∇χ) is pre-scaled
//! by a per-grid-point factor, then contracted with itself.
//!
//! Hybrid GGA and range-separated GGA functionals use the same semilocal
//! eval path as plain GGA — the exact-exchange mixing is the SCF's job.

use ndarray::{Array1, Array2, Array3, Axis};

/// Below this density, libxc-returned v_ρ / v_σ may diverge; skip grid points
/// to keep V_xc well-conditioned. Matches libxc's internal `dens_threshold` default.
const DENSITY_FLOOR: f64 = 1e-10;

use crate::density_on_grid::{DensityGrid, UksDensityGrid};
use crate::grid::GridPoint;
use crate::libxc::{FunctionalFamily, XcDef};

/// Closed-shell semilocal exchange-correlation energy and potential.
///
/// Returns (E_xc, V_xc). V_xc is symmetrized before return.
pub fn semilocal_vxc_closed(
    grid: &[GridPoint],
    chi: &Array2<f64>,         // (nbf, npts)
    dchi: &Array3<f64>,        // (3, nbf, npts)
    dens: &DensityGrid,
    xc: &XcDef,
) -> (f64, Array2<f64>) {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    let w: Array1<f64> = grid.iter().map(|g| g.weight).collect();

    let mut exc_total    = Array1::<f64>::zeros(npts);
    let mut vrho_total   = Array1::<f64>::zeros(npts);
    let mut vsigma_total = Array1::<f64>::zeros(npts);

    let rho_slice   = dens.rho.as_slice().expect("rho is contiguous");
    let sigma_slice = dens.sigma.as_slice().expect("sigma is contiguous");

    // Accumulate ε_xc, v_ρ, v_σ across all component functionals.
    for func in &xc.funcs {
        let mut exc  = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_unpolarized(rho_slice, &mut exc, &mut vrho);
            }
            FunctionalFamily::Gga
            | FunctionalFamily::HybridGga
            | FunctionalFamily::RangeSepGga => {
                let mut vsigma = vec![0.0_f64; npts];
                func.eval_gga_unpolarized(
                    rho_slice, sigma_slice,
                    &mut exc, &mut vrho, &mut vsigma,
                );
                for g in 0..npts {
                    vsigma_total[g] += vsigma[g];
                }
            }
        }
        for g in 0..npts {
            exc_total[g]  += exc[g];
            vrho_total[g] += vrho[g];
        }
    }

    // E_xc = Σ_g w_g · ρ(r_g) · ε_xc(r_g)
    let e_xc: f64 = (0..npts)
        .map(|g| w[g] * dens.rho[g] * exc_total[g])
        .sum();

    // ──────────────────────────────────────────────────────────────────────
    // LDA piece: V_lda_μν = Σ_g (w_g v_ρ_g) · χ_μg · χ_νg
    //
    // Pre-scale χ by (w · v_ρ) per grid point, then GEMM: chi_scaled @ chiᵀ.
    // ──────────────────────────────────────────────────────────────────────
    let mut chi_scaled = chi.clone();
    for g in 0..npts {
        let s = if dens.rho[g] > DENSITY_FLOOR { w[g] * vrho_total[g] } else { 0.0 };
        for mu in 0..nbf {
            chi_scaled[(mu, g)] *= s;
        }
    }
    let mut vxc: Array2<f64> = chi_scaled.dot(&chi.t());

    // ──────────────────────────────────────────────────────────────────────
    // GGA piece: V_gga_μν = Σ_g (2 w_g v_σ_g) ·
    //              Σ_axis ∇ρ_axis_g · [χ_μg (∂_axis χ_ν)_g + χ_νg (∂_axis χ_μ)_g]
    //
    // Define f_axis_g = 2 · w_g · v_σ_g · ∇ρ_axis(r_g). Then for each axis:
    //   M_axis = (f_axis ⊙ χ) · (∂_axis χ)ᵀ
    //   V_gga += M_axis + M_axisᵀ
    // ──────────────────────────────────────────────────────────────────────
    let has_gga = xc.funcs.iter().any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    if has_gga {
        let mut chi_scaled_axis = chi.clone(); // allocate once; refilled each axis
        for axis in 0..3 {
            chi_scaled_axis.assign(chi); // refill (no allocation)
            let dchi_axis = dchi.index_axis(Axis(0), axis);
            // Pre-scale χ by f_axis per grid point.
            for g in 0..npts {
                let f_ag = if dens.rho[g] > DENSITY_FLOOR {
                    2.0 * w[g] * vsigma_total[g] * dens.grad[(axis, g)]
                } else {
                    0.0
                };
                for mu in 0..nbf {
                    chi_scaled_axis[(mu, g)] *= f_ag;
                }
            }
            let m_axis: Array2<f64> = chi_scaled_axis.dot(&dchi_axis.t());
            vxc = vxc + &m_axis + &m_axis.t();
        }
    }

    // Symmetrize: V_xc ← ½(V_xc + V_xcᵀ)
    let vxc_sym = 0.5 * (&vxc + &vxc.t());

    (e_xc, vxc_sym)
}

/// Spin-polarized (UKS) semilocal exchange-correlation energy and potentials.
///
/// Returns `(E_xc, V_α, V_β)`. Each `V_σ` is symmetrized before return.
///
/// libxc polarized interleaved layouts:
///   `rho_in[2g+0]   = ρ_α`,  `rho_in[2g+1]   = ρ_β`
///   `sigma_in[3g+0] = σ_αα`, `sigma_in[3g+1] = σ_αβ`, `sigma_in[3g+2] = σ_ββ`
///   `vrho[2g+0]     = v_α`,  `vrho[2g+1]     = v_β`
///   `vsigma[3g+0]   = v_σαα`,`vsigma[3g+1]   = v_σαβ`, `vsigma[3g+2]   = v_σββ`
///
/// V^α_μν includes a σ_αβ cross-term proportional to ∇ρ_β (and vice versa).
pub fn semilocal_vxc_polarized(
    grid: &[GridPoint],
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
    dens: &UksDensityGrid,
    xc: &XcDef,
) -> (f64, Array2<f64>, Array2<f64>) {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    let w: Array1<f64> = grid.iter().map(|g| g.weight).collect();

    // Build interleaved rho / sigma input for libxc.
    let mut rho_in = vec![0.0_f64; 2 * npts];
    let mut sigma_in = vec![0.0_f64; 3 * npts];
    for g in 0..npts {
        rho_in[2 * g + 0]     = dens.rho_a[g];
        rho_in[2 * g + 1]     = dens.rho_b[g];
        sigma_in[3 * g + 0]   = dens.sigma[(0, g)];
        sigma_in[3 * g + 1]   = dens.sigma[(1, g)];
        sigma_in[3 * g + 2]   = dens.sigma[(2, g)];
    }

    let mut exc_total    = Array1::<f64>::zeros(npts);
    let mut vrho_a_total = Array1::<f64>::zeros(npts);
    let mut vrho_b_total = Array1::<f64>::zeros(npts);
    let mut vsigma_aa_total = Array1::<f64>::zeros(npts);
    let mut vsigma_ab_total = Array1::<f64>::zeros(npts);
    let mut vsigma_bb_total = Array1::<f64>::zeros(npts);

    for func in &xc.funcs {
        let mut exc = vec![0.0_f64; npts];
        let mut vrho = vec![0.0_f64; 2 * npts];
        match func.family() {
            FunctionalFamily::Lda => {
                func.eval_lda_polarized(&rho_in, &mut exc, &mut vrho);
            }
            FunctionalFamily::Gga
            | FunctionalFamily::HybridGga
            | FunctionalFamily::RangeSepGga => {
                let mut vsigma = vec![0.0_f64; 3 * npts];
                func.eval_gga_polarized(
                    &rho_in, &sigma_in,
                    &mut exc, &mut vrho, &mut vsigma,
                );
                for g in 0..npts {
                    vsigma_aa_total[g] += vsigma[3 * g + 0];
                    vsigma_ab_total[g] += vsigma[3 * g + 1];
                    vsigma_bb_total[g] += vsigma[3 * g + 2];
                }
            }
        }
        for g in 0..npts {
            exc_total[g]    += exc[g];
            vrho_a_total[g] += vrho[2 * g + 0];
            vrho_b_total[g] += vrho[2 * g + 1];
        }
    }

    // E_xc = Σ_g w_g · (ρ_α + ρ_β) · ε_xc
    let e_xc: f64 = (0..npts)
        .map(|g| w[g] * (dens.rho_a[g] + dens.rho_b[g]) * exc_total[g])
        .sum();

    let has_gga = xc.funcs.iter().any(|f| !matches!(f.family(), FunctionalFamily::Lda));

    // Build V^σ for σ ∈ {α, β}.
    let build = |vrho_sigma: &Array1<f64>,
                     vsigma_self: &Array1<f64>,
                     vsigma_cross: &Array1<f64>,
                     grad_self: &Array2<f64>,
                     grad_cross: &Array2<f64>,
                     rho_floor_ref: &Array1<f64>| -> Array2<f64> {
        // LDA piece: V^σ_μν = Σ_g (w v_ρσ) · χ_μg · χ_νg
        let mut chi_scaled = chi.clone();
        for g in 0..npts {
            let s = if rho_floor_ref[g] > DENSITY_FLOOR { w[g] * vrho_sigma[g] } else { 0.0 };
            for mu in 0..nbf {
                chi_scaled[(mu, g)] *= s;
            }
        }
        let mut v: Array2<f64> = chi_scaled.dot(&chi.t());

        if has_gga {
            // GGA piece for spin σ:
            //   V^σ_μν += Σ_g (2 w_g v_σσσ) · ∇ρ_σ · (χ_μ ∇χ_ν + χ_ν ∇χ_μ)
            //           + Σ_g (  w_g v_σαβ) · ∇ρ_other · (χ_μ ∇χ_ν + χ_ν ∇χ_μ)
            // (the αβ cross-coupling enters with coefficient 1, not 2, because
            // ∂σ_αβ/∂(∇ρ_α) = ∇ρ_β rather than 2∇ρ_β.)
            let mut chi_scaled_axis = chi.clone();
            for axis in 0..3 {
                chi_scaled_axis.assign(chi);
                let dchi_axis = dchi.index_axis(Axis(0), axis);
                for g in 0..npts {
                    let f_ag = if rho_floor_ref[g] > DENSITY_FLOOR {
                        2.0 * w[g] * vsigma_self[g] * grad_self[(axis, g)]
                            + w[g] * vsigma_cross[g] * grad_cross[(axis, g)]
                    } else {
                        0.0
                    };
                    for mu in 0..nbf {
                        chi_scaled_axis[(mu, g)] *= f_ag;
                    }
                }
                let m_axis: Array2<f64> = chi_scaled_axis.dot(&dchi_axis.t());
                v = v + &m_axis + &m_axis.t();
            }
        }

        0.5 * (&v + &v.t())
    };

    // Floor each spin block on its own ρ_σ (libxc treats v_ρσ as ill-defined
    // where ρ_σ → 0). For the αβ cross-term, gate on the smaller of the two.
    let v_a = build(
        &vrho_a_total, &vsigma_aa_total, &vsigma_ab_total,
        &dens.grad_a, &dens.grad_b, &dens.rho_a,
    );
    let v_b = build(
        &vrho_b_total, &vsigma_bb_total, &vsigma_ab_total,
        &dens.grad_b, &dens.grad_a, &dens.rho_b,
    );

    (e_xc, v_a, v_b)
}
