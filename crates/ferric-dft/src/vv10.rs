//! VV10 nonlocal correlation (Vydrov-Van Voorhis JCP 133, 244103 (2010)).
//!
//! Closed-shell energy:
//! ```text
//!     E_nl = ∫ ρ(r) · [β + ½ · ∫ ρ(r') Φ(r,r') dr'] dr
//!     Φ(r,r') = −3 / [2 g(r) g(r') (g(r) + g(r'))]
//!     g(r)    = ω₀(r) · |r − r'|² + κ(r)
//!     ω₀²(r)  = C · (|∇ρ|/ρ)⁴ + (4π/3) · ρ
//!     κ(r)    = b · (3π/2) · (ρ/(9π))^(1/6)
//!     β       = (1/32) · (3/b²)^(3/4)
//! ```
//!
//! V_nl potential is the functional derivative δE_nl/δρ + δE_nl/δσ via the
//! chain rule through ω₀ and κ — see PySCF `_vv10nlc` for the exact form.
//! Algorithm direct from PySCF's `pyscf/dft/numint.py::_vv10nlc`.

use ndarray::{Array2, Array3, Axis};

use crate::density_on_grid::DensityGrid;
use crate::grid::GridPoint;
use crate::libxc::Vv10Params;

/// Density threshold below which a grid point is skipped — keeps κ, ω₀, and
/// their derivatives well-conditioned. Matches PySCF's `_vv10nlc` thresh=1e-8.
const RHO_THRESH: f64 = 1e-8;

/// Compute the VV10 energy contribution and add the matrix V_nl to `f`.
///
/// Single-grid implementation: the NLC grid serves as both outer (integration)
/// and inner (kernel partner) points. O(N²) in grid size.
pub fn add_vv10(
    grid: &[GridPoint],
    chi: &Array2<f64>,        // (nbf, npts)
    dchi: &Array3<f64>,       // (3, nbf, npts)
    dens: &DensityGrid,
    params: &Vv10Params,
    f: &mut Array2<f64>,
) -> f64 {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));
    debug_assert_eq!(dens.rho.len(), npts);

    let b_vv = params.b;
    let c_vv = params.c;

    let pi = std::f64::consts::PI;
    let pi43 = 4.0 * pi / 3.0;
    let k_vv = b_vv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
    let beta = (3.0 / (b_vv * b_vv)).powf(0.75) / 32.0;

    // Per-point precompute: ρ, σ = |∇ρ|², ω₀, κ, and the derivatives needed
    // for V_nl. Mask low-density points.
    let mut rho = vec![0.0_f64; npts];
    let mut sig = vec![0.0_f64; npts];
    let mut w0  = vec![0.0_f64; npts];
    let mut kp  = vec![0.0_f64; npts];
    let mut dw0_dr = vec![0.0_f64; npts];
    let mut dw0_dg = vec![0.0_f64; npts];
    let mut dk_dr  = vec![0.0_f64; npts];
    let mut active = vec![false; npts];
    let mut xyz = vec![[0.0_f64; 3]; npts];
    let mut rho_w = vec![0.0_f64; npts]; // ρ · w for kernel sums

    for g in 0..npts {
        let r = dens.rho[g];
        let s = dens.sigma[g];
        if r < RHO_THRESH {
            continue;
        }
        active[g] = true;
        rho[g] = r;
        sig[g] = s;
        xyz[g] = grid[g].xyz;
        rho_w[g] = r * grid[g].weight;

        // ω₀² = C·(σ/ρ²)² + (4π/3)·ρ
        let w0tmp = c_vv * (s / (r * r)).powi(2);  // = C·(σ/ρ²)²
        let w0sq  = w0tmp + pi43 * r;
        w0[g]    = w0sq.sqrt();
        // κ = K_vv · ρ^(1/6)
        kp[g] = k_vv * r.powf(1.0 / 6.0);
        // dκ/dρ = κ / (6 ρ)·ρ = κ/6 — but PySCF uses dKdR = K/6 then later
        // multiplies by ρ inside vxc; we follow PySCF exactly.
        dk_dr[g] = kp[g] / 6.0;
        // dω₀/dρ = (½·(4π/3)·ρ − 2·w0tmp) / ω₀
        dw0_dr[g] = (0.5 * pi43 * r - 2.0 * w0tmp) / w0[g];
        // dω₀/dσ = w0tmp · ρ / (σ · ω₀) — derived from d(W0tmp)/dG with
        //   W0tmp = C·σ²/ρ⁴ ⟹ ∂W0tmp/∂σ = 2·W0tmp/σ
        //   ∂ω₀/∂σ = (½/ω₀)·∂(ω₀²)/∂σ = (½/ω₀)·2·W0tmp/σ = W0tmp/(σ·ω₀)
        // PySCF stores dW0dG = W0tmp·R/(G·W0) — same when G is σ and R is ρ.
        // We follow PySCF: dw0_dg[g] = W0tmp * ρ / (σ * ω₀).
        if s > RHO_THRESH {
            dw0_dg[g] = w0tmp * r / (s * w0[g]);
        }
    }

    // O(N²) kernel sums.
    //
    //   F[i] = -1.5 · Σ_p RpW_p / [g_i · g_p · (g_i + g_p)]
    //   U[i] = Σ_p RpW_p / [g_i · g_p · (g_i + g_p)] · (1/g_i + 1/(g_i+g_p))
    //   W[i] = Σ_p RpW_p / [g_i · g_p · (g_i + g_p)] · (1/g_i + 1/(g_i+g_p)) · R²
    //
    // where R² = |r_i − r_p|², g_i = R²·ω₀_i + κ_i, g_p = R²·ω₀_p + κ_p.
    let mut f_arr = vec![0.0_f64; npts];
    let mut u_arr = vec![0.0_f64; npts];
    let mut w_arr = vec![0.0_f64; npts];

    for i in 0..npts {
        if !active[i] { continue; }
        let xi = xyz[i];
        let w0i = w0[i];
        let ki  = kp[i];
        let mut fi = 0.0_f64;
        let mut ui = 0.0_f64;
        let mut wi = 0.0_f64;
        for p in 0..npts {
            if !active[p] { continue; }
            let dx = xyz[p][0] - xi[0];
            let dy = xyz[p][1] - xi[1];
            let dz = xyz[p][2] - xi[2];
            let r2 = dx*dx + dy*dy + dz*dz;
            let gp = r2 * w0[p] + kp[p];
            let gi = r2 * w0i  + ki;
            let gt = gi + gp;
            // Guard against divide-by-zero at i == p with R² = 0 (g_i = κ_i ≠ 0
            // already, so this is just defensive).
            if gi < 1e-30 || gp < 1e-30 || gt < 1e-30 { continue; }
            let t = rho_w[p] / (gi * gp * gt);
            fi += t;
            let t_u = t * (1.0 / gi + 1.0 / gt);
            ui += t_u;
            wi += t_u * r2;
        }
        f_arr[i] = -1.5 * fi;
        u_arr[i] = ui;
        w_arr[i] = wi;
    }

    // Per-point exchange-correlation:
    //   exc[i]   = β + 0.5·F[i]                                  (multiplied by ρ for E)
    //   vrho[i]  = β + F[i] + 1.5·(U·dKdR + W·dW0dR)
    //   vsig[i]  = 1.5·W·dW0dG
    let mut exc  = vec![0.0_f64; npts];
    let mut vrho = vec![0.0_f64; npts];
    let mut vsig = vec![0.0_f64; npts];
    for g in 0..npts {
        if !active[g] { continue; }
        exc[g]  = beta + 0.5 * f_arr[g];
        vrho[g] = beta + f_arr[g] + 1.5 * (u_arr[g] * dk_dr[g] + w_arr[g] * dw0_dr[g]);
        vsig[g] = 1.5 * w_arr[g] * dw0_dg[g];
    }

    // Energy: E_nl = Σ_g w_g · ρ_g · exc_g
    let e_nl: f64 = (0..npts).map(|g| grid[g].weight * rho[g] * exc[g]).sum();

    // V_nl matrix contribution — same GEMM pattern as semilocal V_xc.
    //   LDA-like piece: V_μν += Σ_g w_g · vrho_g · χ_μg · χ_νg
    //   GGA-like piece: V_μν += Σ_g 2·w_g · vsig_g · Σ_axis ∇ρ_axis_g ·
    //                           [χ_μg · ∂_axis χ_νg + χ_νg · ∂_axis χ_μg]
    let mut chi_scaled = chi.clone();
    for g in 0..npts {
        let s = if active[g] { grid[g].weight * vrho[g] } else { 0.0 };
        for mu in 0..nbf {
            chi_scaled[(mu, g)] *= s;
        }
    }
    let mut v_nl: Array2<f64> = chi_scaled.dot(&chi.t());

    let mut chi_scaled_axis = chi.clone();
    for axis in 0..3 {
        chi_scaled_axis.assign(chi);
        let dchi_axis = dchi.index_axis(Axis(0), axis);
        for g in 0..npts {
            let f_ag = if active[g] {
                2.0 * grid[g].weight * vsig[g] * dens.grad[(axis, g)]
            } else {
                0.0
            };
            for mu in 0..nbf {
                chi_scaled_axis[(mu, g)] *= f_ag;
            }
        }
        let m_axis: Array2<f64> = chi_scaled_axis.dot(&dchi_axis.t());
        v_nl = v_nl + &m_axis + &m_axis.t();
    }

    // Symmetrize and accumulate.
    let v_nl_sym = 0.5 * (&v_nl + &v_nl.t());
    *f += &v_nl_sym;

    e_nl
}
