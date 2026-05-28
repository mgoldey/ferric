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

/// Compute per-grid-point VV10 potentials `(v_ρ, v_σ)` plus the energy E_nl.
///
/// Factored out of `add_vv10` so the nuclear-gradient path can reuse the
/// pair sum without going through the matrix assembly.
pub fn compute_vv10_potentials(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
) -> (Vec<f64>, Vec<f64>) {
    let (e, vr, vs, _) = vv10_internal(grid, dens, params);
    let _ = e;
    (vr, vs)
}

/// Like `compute_vv10_potentials` but also returns E_nl.
pub fn compute_vv10_energy_and_potentials(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
) -> (f64, Vec<f64>, Vec<f64>) {
    let (e, vr, vs, _) = vv10_internal(grid, dens, params);
    (e, vr, vs)
}

/// Compute the per-grid-point VV10 energy density ε_nl(g) = β + ½ · f(g)
/// alongside the potentials. Needed by the gradient path that wants weight
/// response Σ_g w1[g, B, α] · ε_nl(g) · ρ(g).
pub fn compute_vv10_full(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (_e, vr, vs, active) = vv10_internal(grid, dens, params);
    // Re-derive ε(g) from the same pair sum used internally. `vv10_internal`
    // already integrated this; re-doing it here keeps the function shape simple
    // at the cost of a single extra pair-sum traversal. For typical grids the
    // pair-sum dominates the cost regardless.
    let (exc_per_point, _) = vv10_exc_per_point(grid, dens, params, &active);
    (vr, vs, exc_per_point)
}

/// Per-grid-point ε_nl(g) = β + ½ · f(g) on the union NLC grid. Returns
/// `(exc, active_mask)`. Skips grid points whose density is below threshold.
fn vv10_exc_per_point(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
    active: &[bool],
) -> (Vec<f64>, ()) {
    let npts = dens.rho.len();
    let b_vv = params.b;
    let c_vv = params.c;
    let pi = std::f64::consts::PI;
    let pi43 = 4.0 * pi / 3.0;
    let k_vv = b_vv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
    let beta = (3.0 / (b_vv * b_vv)).powf(0.75) / 32.0;

    let mut w0 = vec![0.0f64; npts];
    let mut kp = vec![0.0f64; npts];
    let mut rho_w = vec![0.0f64; npts];
    let mut xyz = vec![[0.0f64; 3]; npts];
    for g in 0..npts {
        if !active[g] {
            continue;
        }
        let r = dens.rho[g];
        let s = dens.sigma[g];
        let w0sq = c_vv * (s / (r * r)).powi(2) + pi43 * r;
        w0[g] = w0sq.sqrt();
        kp[g] = k_vv * r.powf(1.0 / 6.0);
        rho_w[g] = r * grid[g].weight;
        xyz[g] = grid[g].xyz;
    }

    let mut exc = vec![0.0f64; npts];
    for i in 0..npts {
        if !active[i] {
            continue;
        }
        let xi = xyz[i];
        let w0i = w0[i];
        let ki = kp[i];
        let mut fi = 0.0f64;
        for p in 0..npts {
            if !active[p] {
                continue;
            }
            let dx = xyz[p][0] - xi[0];
            let dy = xyz[p][1] - xi[1];
            let dz = xyz[p][2] - xi[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let gp_val = r2 * w0[p] + kp[p];
            let gi_val = r2 * w0i + ki;
            let gt_val = gi_val + gp_val;
            if gi_val < 1e-30 || gp_val < 1e-30 || gt_val < 1e-30 {
                continue;
            }
            fi += rho_w[p] / (gi_val * gp_val * gt_val);
        }
        exc[i] = beta + 0.5 * (-1.5 * fi);
    }
    (exc, ())
}

/// Per-grid-point VV10 egrad `F[g, axis] = -3 · Σ_p RpW_p · Q[g,p] · DR[g,p]`
/// where `Q[g,p] = (1/(g_i·g_p·g_t)) · (ω₀_i/g_i + ω₀_p/g_p + (ω₀_i+ω₀_p)/g_t)`
/// and `DR = r_p − r_g`.
///
/// This is the gradient of the VV10 pair integrand with respect to the
/// **outer** grid coordinate r_g. The outer grid (`outer`) and the inner
/// partner grid (`inner`) may be the same (canonical full-grid) or distinct
/// (e.g. PySCF's `vvrho_sub` / `vvcoords_sub` which excludes the atom whose
/// gradient is being computed, avoiding self-coupling). The energy double-
/// integral factor of ½ is absorbed at the use site via the `ρ·w·F` outer
/// sum — matches PySCF's `excsum[atm_id] += einsum('r,rx->x', rho*weight, F)`.
pub fn vv10_egrad(
    outer_grid: &[GridPoint],
    outer_dens: &DensityGrid,
    inner_grid: &[GridPoint],
    inner_dens: &DensityGrid,
    params: &Vv10Params,
) -> ndarray::Array2<f64> {
    let n_out = outer_grid.len();
    let n_in = inner_grid.len();
    let b_vv = params.b;
    let c_vv = params.c;
    let pi = std::f64::consts::PI;
    let pi43 = 4.0 * pi / 3.0;
    let k_vv = b_vv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);

    // Cache outer ω₀, κ, coords (active outer points only).
    let mut w0_out = vec![0.0f64; n_out];
    let mut k_out = vec![0.0f64; n_out];
    let mut active_out = vec![false; n_out];
    let mut xyz_out = vec![[0.0f64; 3]; n_out];
    for i in 0..n_out {
        let r = outer_dens.rho[i];
        if r < RHO_THRESH {
            continue;
        }
        let s = outer_dens.sigma[i];
        let w0sq = c_vv * (s / (r * r)).powi(2) + pi43 * r;
        w0_out[i] = w0sq.sqrt();
        k_out[i] = k_vv * r.powf(1.0 / 6.0);
        xyz_out[i] = outer_grid[i].xyz;
        active_out[i] = true;
    }

    // Cache inner ω₀, κ, RpW, coords.
    let mut w0_in = vec![0.0f64; n_in];
    let mut k_in = vec![0.0f64; n_in];
    let mut rpw = vec![0.0f64; n_in];
    let mut xyz_in = vec![[0.0f64; 3]; n_in];
    let mut active_in = vec![false; n_in];
    for j in 0..n_in {
        let r = inner_dens.rho[j];
        if r < RHO_THRESH {
            continue;
        }
        let s = inner_dens.sigma[j];
        let w0sq = c_vv * (s / (r * r)).powi(2) + pi43 * r;
        w0_in[j] = w0sq.sqrt();
        k_in[j] = k_vv * r.powf(1.0 / 6.0);
        rpw[j] = r * inner_grid[j].weight;
        xyz_in[j] = inner_grid[j].xyz;
        active_in[j] = true;
    }

    let mut f = ndarray::Array2::<f64>::zeros((n_out, 3));
    for i in 0..n_out {
        if !active_out[i] {
            continue;
        }
        let xi = xyz_out[i];
        let w0i = w0_out[i];
        let ki = k_out[i];
        let mut fx = 0.0f64;
        let mut fy = 0.0f64;
        let mut fz = 0.0f64;
        for j in 0..n_in {
            if !active_in[j] {
                continue;
            }
            let dx = xyz_in[j][0] - xi[0];
            let dy = xyz_in[j][1] - xi[1];
            let dz = xyz_in[j][2] - xi[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let g_i = r2 * w0i + ki;
            let g_p = r2 * w0_in[j] + k_in[j];
            let g_t = g_i + g_p;
            if g_i < 1e-30 || g_p < 1e-30 || g_t < 1e-30 {
                continue;
            }
            let t = rpw[j] / (g_i * g_p * g_t);
            let q = t * (w0i / g_i + w0_in[j] / g_p + (w0i + w0_in[j]) / g_t);
            fx += q * dx;
            fy += q * dy;
            fz += q * dz;
        }
        f[(i, 0)] = -3.0 * fx;
        f[(i, 1)] = -3.0 * fy;
        f[(i, 2)] = -3.0 * fz;
    }
    f
}

/// Internal: compute (E_nl, vrho, vsig, active) on a single grid.
fn vv10_internal(
    grid: &[GridPoint],
    dens: &DensityGrid,
    params: &Vv10Params,
) -> (f64, Vec<f64>, Vec<f64>, Vec<bool>) {
    let npts = dens.rho.len();
    let b_vv = params.b;
    let c_vv = params.c;
    let pi = std::f64::consts::PI;
    let pi43 = 4.0 * pi / 3.0;
    let k_vv = b_vv * 1.5 * pi * (9.0 * pi).powf(-1.0 / 6.0);
    let beta = (3.0 / (b_vv * b_vv)).powf(0.75) / 32.0;

    let mut rho = vec![0.0_f64; npts];
    let mut w0  = vec![0.0_f64; npts];
    let mut kp  = vec![0.0_f64; npts];
    let mut dw0_dr = vec![0.0_f64; npts];
    let mut dw0_dg = vec![0.0_f64; npts];
    let mut dk_dr  = vec![0.0_f64; npts];
    let mut active = vec![false; npts];
    let mut xyz = vec![[0.0_f64; 3]; npts];
    let mut rho_w = vec![0.0_f64; npts];

    for g in 0..npts {
        let r = dens.rho[g];
        let s = dens.sigma[g];
        if r < RHO_THRESH { continue; }
        active[g] = true;
        rho[g] = r;
        xyz[g] = grid[g].xyz;
        rho_w[g] = r * grid[g].weight;
        let w0tmp = c_vv * (s / (r * r)).powi(2);
        let w0sq = w0tmp + pi43 * r;
        w0[g] = w0sq.sqrt();
        kp[g] = k_vv * r.powf(1.0 / 6.0);
        dk_dr[g] = kp[g] / 6.0;
        dw0_dr[g] = (0.5 * pi43 * r - 2.0 * w0tmp) / w0[g];
        if s > RHO_THRESH {
            dw0_dg[g] = w0tmp * r / (s * w0[g]);
        }
    }

    let mut f_arr = vec![0.0_f64; npts];
    let mut u_arr = vec![0.0_f64; npts];
    let mut w_arr = vec![0.0_f64; npts];
    for i in 0..npts {
        if !active[i] { continue; }
        let xi = xyz[i];
        let w0i = w0[i];
        let ki = kp[i];
        let mut fi = 0.0_f64;
        let mut ui = 0.0_f64;
        let mut wi = 0.0_f64;
        for p in 0..npts {
            if !active[p] { continue; }
            let dx = xyz[p][0] - xi[0];
            let dy = xyz[p][1] - xi[1];
            let dz = xyz[p][2] - xi[2];
            let r2 = dx*dx + dy*dy + dz*dz;
            let gp_val = r2 * w0[p] + kp[p];
            let gi_val = r2 * w0i + ki;
            let gt_val = gi_val + gp_val;
            if gi_val < 1e-30 || gp_val < 1e-30 || gt_val < 1e-30 { continue; }
            let t = rho_w[p] / (gi_val * gp_val * gt_val);
            fi += t;
            let t_u = t * (1.0 / gi_val + 1.0 / gt_val);
            ui += t_u;
            wi += t_u * r2;
        }
        f_arr[i] = -1.5 * fi;
        u_arr[i] = ui;
        w_arr[i] = wi;
    }

    let mut vrho = vec![0.0_f64; npts];
    let mut vsig = vec![0.0_f64; npts];
    let mut e_nl = 0.0_f64;
    for g in 0..npts {
        if !active[g] { continue; }
        let exc_g = beta + 0.5 * f_arr[g];
        vrho[g] = beta + f_arr[g] + 1.5 * (u_arr[g] * dk_dr[g] + w_arr[g] * dw0_dr[g]);
        vsig[g] = 1.5 * w_arr[g] * dw0_dg[g];
        e_nl += grid[g].weight * rho[g] * exc_g;
    }
    (e_nl, vrho, vsig, active)
}

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

    // Compute energy + per-point potentials via the shared pair-sum routine.
    let (e_nl, vrho, vsig, active) = vv10_internal(grid, dens, params);

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
