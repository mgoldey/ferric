//! G0W0 correlation self-energy via imaginary-axis quadrature + Padé AC.
//!
//! Working formula (closed-shell, RHF reference, MO basis, PDEP-W
//! representation):
//!
//!   Σ_c(m, z) = − (1/π) Σ_n Σ_α  M²_{mn,α}
//!                 ∫₀^∞ dω' w_α(iω')
//!                  ·  (z − ε_n) / ((z − ε_n)² + ω'²)
//!
//! where w_α(iω') = 1/λ_α(iω') − 1. The integrand is even in ω', so the
//! ∫₋∞^∞ → 2·∫₀^∞ factor of 2 has been absorbed; the prefactor is then
//! 1/(2π) · 2 = 1/π.
//!
//! For z = iω (imaginary axis), the integrand is purely imaginary in the
//! conventional split, and the resulting Σ_c(m, iω) is real-imaginary
//! — for our purposes we evaluate at complex z and let num-complex do
//! the bookkeeping.
//!
//! The quadrature {(ω_k, w_k)} is reused verbatim from PDEP
//! (`pdep.quad_freqs`, `pdep.quad_weights`). This guarantees that
//! Σ_c at z=0 is consistent with the inverse-dielectric values used in
//! COHSEX.
//!
//! After evaluating Σ_c(m, iω_k) at the PDEP quadrature nodes, we fit a
//! Padé continued-fraction model (`crate::pade::PadeCF`) on the sampled
//! values and evaluate at real ω = ε^QP_m via Newton iteration.

use crate::cohsex::{project_b_into_pdep, sigma_x_diag};
use crate::mo_b::MoB;
use crate::pade::PadeCF;
use crate::w_pdep;
use crate::{GwConfig, GwResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_rpa::PdepRpaResult;
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Fermi level (mid-gap) of the active-space spectrum: midpoint between the
/// active HOMO and active LUMO. Used to center the analytic-continuation
/// support line `z = ef + iω`, matching PySCF gw_ac. For an all-occupied or
/// all-virtual active block (degenerate edge cases) we fall back to the band
/// edge so `ef` stays finite.
pub(crate) fn fermi_level(eps_act: &[f64], n_occ_act: usize) -> f64 {
    let n = eps_act.len();
    if n == 0 {
        return 0.0;
    }
    if n_occ_act == 0 {
        return eps_act[0];
    }
    if n_occ_act >= n {
        return eps_act[n - 1];
    }
    0.5 * (eps_act[n_occ_act - 1] + eps_act[n_occ_act])
}

/// Evaluate Σ_c(m, z) at a single (possibly complex) z given the projected
/// matrix elements M[(α,m,n)], inverse-dielectric weights w_α(iω_k), and
/// quadrature data. `n_occ_act` is used only for completeness (sum runs
/// over all n).
pub(crate) fn sigma_c_at_z(
    m_idx: usize,
    z: Complex64,
    m_proj: &ndarray::Array3<f64>,
    inv_diel_freq: &[Array2<f64>], // length N_quad, each (M, M): W̃_d(iω_k) = ε̃⁻¹ − I
    quad_weights: &[f64],
    quad_freqs: &[f64],
    eps_act: &[f64],
) -> Complex64 {
    let m_modes = m_proj.shape()[0];
    let n_quad = quad_freqs.len();
    let n_act = eps_act.len();

    // Precompute the screened matrix element W_{mn}(iω_k) = Mᵀ_{mn} · W̃_d(iω_k) · M_{mn}
    // using the FULL inverse-dielectric matrix in the PDEP basis. The earlier
    // diagonal-only form  Σ_α M²_α (1/λ_α − 1)  was wrong: the scalar
    // eigenvalues λ_α(iω) live in a per-ω rotated eigenbasis, inconsistent with
    // the static B̃ projection M, which corrupts the per-(m,n) matrix elements
    // (only the trace survives, which is why the RPA energy was unaffected).
    let mut sigma = Complex64::new(0.0, 0.0);
    for n_idx in 0..n_act {
        let eps_n = eps_act[n_idx];
        let diff = z - Complex64::new(eps_n, 0.0);
        let mut inner = Complex64::new(0.0, 0.0);
        for k in 0..n_quad {
            let omk = quad_freqs[k];
            let den = diff * diff + Complex64::new(omk * omk, 0.0);
            let wd = &inv_diel_freq[k];
            // w_mn = Σ_αβ M_α D_αβ M_β  (real, symmetric D).
            let mut w_mn = 0.0_f64;
            for a in 0..m_modes {
                let ma = m_proj[(a, m_idx, n_idx)];
                if ma == 0.0 {
                    continue;
                }
                let mut row = 0.0_f64;
                for b in 0..m_modes {
                    row += wd[(a, b)] * m_proj[(b, m_idx, n_idx)];
                }
                w_mn += ma * row;
            }
            inner += Complex64::new(quad_weights[k] * w_mn, 0.0) / den;
        }
        sigma += diff * inner;
    }
    -Complex64::new(1.0 / std::f64::consts::PI, 0.0) * sigma
}

/// Per-MO QP solve given a fixed (m_proj, w_α(iω), quadrature, propagator-ε)
/// snapshot. Returns (ε_qp, Σ_c at ε_qp, Z, was Newton-converged).
pub(crate) fn solve_qp_for_mo(
    m_loc: usize,
    eps_m_mf: f64,
    m_proj: &ndarray::Array3<f64>,
    inv_diel_freq: &[Array2<f64>],
    quad_weights: &[f64],
    quad_freqs: &[f64],
    eps_prop: &[f64],
    pade_npts: usize,
    newton_damp: f64,
    ef: f64,
    // Static (energy-independent) self-energy shift Σ_x − v_xc, added INSIDE the
    // QP self-consistency. Zero for HF references (Σ_x is already in ε_mf); for a
    // KS reference it is Σ_x − v_xc and is large (~−7 eV), moving the QP root by
    // several eV — so Σ_c must be evaluated at the shifted energy, not post-hoc.
    static_shift: f64,
) -> (f64, f64, f64) {
    let n_pade = if pade_npts == 0 {
        quad_freqs.len().min(16)
    } else {
        pade_npts
    };
    let pade_omegas: Vec<f64> = (0..n_pade)
        .map(|k| {
            let t = (k as f64 + 0.5) / n_pade as f64;
            let lo = 0.05_f64.ln();
            let hi = (3.0 * quad_freqs.last().copied().unwrap_or(5.0)).ln();
            (lo + t * (hi - lo)).exp()
        })
        .collect();
    // Sample Σ_c on the Fermi-shifted imaginary axis z = ef + iω'.
    let z_nodes: Vec<Complex64> = pade_omegas
        .iter()
        .map(|&w| Complex64::new(ef, w))
        .collect();
    let f_vals: Vec<Complex64> = z_nodes
        .iter()
        .map(|&z| {
            sigma_c_at_z(
                m_loc, z, m_proj, inv_diel_freq, quad_weights, quad_freqs, eps_prop,
            )
        })
        .collect();
    let pade = PadeCF::fit(z_nodes, &f_vals);
    let h = 0.05;
    let sc_at_ref = pade.eval(Complex64::new(eps_m_mf, 0.0)).re;
    let dsig = pade.deriv_real(eps_m_mf, h).re;
    let z_renorm = (1.0 / (1.0 - dsig)).clamp(0.0, 1.5);
    // Linearized: ε_qp = ε_mf + Z·(Σc(ε_mf) + static_shift). The static shift
    // (Σx−vxc, KS only) enters the QP equation alongside Σc (PySCF gw_ac form
    // zn·(sigmaR + vk − v_mf)); it is large for KS and moves the root by eV, so
    // the self-consistent Σc below is evaluated at the correctly-shifted energy.
    let eps_qp_lin = eps_m_mf + z_renorm * (sc_at_ref + static_shift);
    let mut eps_curr = eps_qp_lin;
    let damp = newton_damp.clamp(0.1, 1.0);
    for _ in 0..30 {
        let sc = pade.eval(Complex64::new(eps_curr, 0.0)).re;
        // QP residual: ε − ε_mf − static_shift − Σc(ε) = 0.
        let f = eps_curr - eps_m_mf - static_shift - sc;
        let dsc = pade.deriv_real(eps_curr, h).re;
        let fprime = 1.0 - dsc;
        if fprime.abs() < 1e-3 {
            break;
        }
        let step = -damp * f / fprime;
        eps_curr += step;
        if step.abs() < 1e-7 {
            break;
        }
    }
    let sc_final = pade.eval(Complex64::new(eps_curr, 0.0)).re;
    (eps_curr, sc_final, z_renorm)
}

/// Run G0W0. `vxc_diag`, if given (KS reference), is the absolute-MO-indexed
/// diagonal v_xc; the QP equation then includes Σ_x − v_xc inside the
/// self-consistency. `None` ⇒ HF reference (no shift, unchanged behavior).
pub fn run_g0w0(
    mol: &Molecule,
    rhf: &ScfResult,
    mo_b: &MoB,
    v_dressed: &Array2<f64>,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
    vxc_diag: Option<&Array1<f64>>,
) -> Result<GwResult, FerricError> {
    let _ = (mol, rhf);
    let _n_act = mo_b.n_act;
    let first_act = mo_b.first_act;
    let m_modes = v_dressed.ncols();
    if pdep.eigenvalues_freq.ncols() != m_modes {
        return Err(FerricError::General(
            "pdep eigenvalues_freq mode count does not match dressed eigenpotentials".into(),
        ));
    }

    // Σ_x for all active MOs.
    let sigma_x_all = sigma_x_diag(mo_b);

    // Project B̃ → M[(α,m,n)] (shape M × n_act × n_act).
    let m_proj = project_b_into_pdep(mo_b, v_dressed);

    // Full per-frequency inverse-dielectric matrices W̃_d(iω_k) in the PDEP basis.
    let inv_diel_freq = pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
        FerricError::General(
            "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
        )
    })?;

    let quad_freqs = pdep.quad_freqs.clone();
    let quad_weights = pdep.quad_weights.clone();

    // Fermi level: mid-gap of the active-space mean-field spectrum.
    let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);

    // For each MO in qp_range, sample Σ_c on the imaginary axis at the PDEP
    // quadrature nodes, fit Padé, solve QP via Newton.
    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let mut z_out = Array1::<f64>::ones(mo_indices.len());

    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        if mo_abs < first_act {
            return Err(FerricError::General(format!(
                "qp_mos index {mo_abs} is in the frozen-core block"
            )));
        }
        let m_loc = mo_abs - first_act;
        let eps_m = mo_b.eps_act[m_loc];
        eps_mf[idx] = eps_m;
        sx_out[idx] = sigma_x_all[m_loc];

        // KS reference: shift = Σ_x − v_xc (inside the QP self-consistency).
        let shift = vxc_diag.map(|v| sigma_x_all[m_loc] - v[mo_abs]).unwrap_or(0.0);
        let (eps_qp_m, sc_final, z_renorm) = solve_qp_for_mo(
            m_loc,
            eps_m,
            &m_proj,
            inv_diel_freq,
            &quad_weights,
            &quad_freqs,
            &mo_b.eps_act,
            gw_cfg.pade_npts,
            gw_cfg.qp_newton_damp,
            ef,
            shift,
        );
        sc_out[idx] = sc_final;
        z_out[idx] = z_renorm;
        eps_qp[idx] = eps_qp_m;
    }

    Ok(GwResult {
        mo_indices,
        eps_mf,
        eps_qp,
        sigma_x: sx_out,
        sigma_c: sc_out,
        z_factor: z_out,
        n_ev_iter: 0,
        pdep,
    })
}

/// evGW₀: eigenvalue self-consistency on G only.
///
/// Iterate: at each step k, replace ε_n in the propagator denominator by
/// the current QP estimate. W stays frozen at iteration 0 (so PDEP is not
/// re-run). Converges in a few iterations for closed-shell molecules.
pub fn run_evgw0(
    mol: &Molecule,
    rhf: &ScfResult,
    mo_b: &MoB,
    v_dressed: &Array2<f64>,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
) -> Result<GwResult, FerricError> {
    let _ = (mol, rhf);
    let first_act = mo_b.first_act;
    let m_modes = v_dressed.ncols();
    if pdep.eigenvalues_freq.ncols() != m_modes {
        return Err(FerricError::General(
            "pdep eigenvalues_freq mode count does not match dressed eigenpotentials".into(),
        ));
    }
    let sigma_x_all = sigma_x_diag(mo_b);
    let m_proj = project_b_into_pdep(mo_b, v_dressed);
    let inv_diel_freq = pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
        FerricError::General(
            "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
        )
    })?;

    let quad_freqs = pdep.quad_freqs.clone();
    let quad_weights = pdep.quad_weights.clone();

    let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);

    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let mut z_out = Array1::<f64>::ones(mo_indices.len());
    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        let m_loc = mo_abs - first_act;
        eps_mf[idx] = mo_b.eps_act[m_loc];
        sx_out[idx] = sigma_x_all[m_loc];
        eps_qp[idx] = eps_mf[idx];
    }

    // Outer loop.
    let mut eps_prop = mo_b.eps_act.clone();
    let mut iter_done = 0usize;
    for it in 0..gw_cfg.max_ev_iter {
        let mut max_dev = 0.0_f64;
        // First update eps_prop using the *previous* QP energies.
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            let m_loc = mo_abs - first_act;
            eps_prop[m_loc] = eps_qp[idx];
        }
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            let m_loc = mo_abs - first_act;
            let eps_m_mf = mo_b.eps_act[m_loc];
            let (eps_new, sc_new, z_new) = solve_qp_for_mo(
                m_loc,
                eps_m_mf,
                &m_proj,
                inv_diel_freq,
                &quad_weights,
                &quad_freqs,
                &eps_prop,
                gw_cfg.pade_npts,
                gw_cfg.qp_newton_damp,
                ef,
                0.0,
            );
            max_dev = max_dev.max((eps_new - eps_qp[idx]).abs());
            eps_qp[idx] = eps_new;
            sc_out[idx] = sc_new;
            z_out[idx] = z_new;
        }
        iter_done = it + 1;
        if max_dev < gw_cfg.ev_conv_thresh {
            break;
        }
    }

    Ok(GwResult {
        mo_indices,
        eps_mf,
        eps_qp,
        sigma_x: sx_out,
        sigma_c: sc_out,
        z_factor: z_out,
        n_ev_iter: iter_done,
        pdep,
    })
}

/// evGW: eigenvalue self-consistency on G *and* W.
///
/// Outer loop re-runs PDEP-RPA with QP-shifted ε's on every iteration.
/// Inside each PDEP rebuild we override the occupied/virtual orbital
/// energies in the χ₀ denominators using the current QP estimates for the
/// QP block and the mean-field ε's elsewhere.
///
/// **Spike note**: the cleanest way to override the ε's used by PDEP is
/// to shift the RHF eigenvalue vector before re-calling `run_pdep_rpa`.
/// We mutate a `ScfResult` clone so it presents the QP-shifted ε's. All
/// other RHF data (orbitals, density, energy) stays.
pub fn run_evgw(
    mol: &Molecule,
    obs: &ferric_integrals::basis_bridge::PreparedBasis,
    dfbs: &ferric_integrals::basis_bridge::PreparedBasis,
    op: ferric_integrals::operator::Operator,
    rhf: &ScfResult,
    pdep_cfg: &ferric_rpa::config::PdepRpaConfig,
    mo_b: &MoB,
    pdep0: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
) -> Result<GwResult, FerricError> {
    // We rebuild PDEP from a *modified* ScfResult that has QP-shifted
    // eigenvalues for the QP block. mo_b.eps_act stays as the original
    // MF eigenvalues — only the PDEP χ₀ denominators see the shifted ε's.
    let mut shifted_rhf = rhf.clone();
    let mut current_pdep = pdep0;
    let mut current_v_dressed =
        w_pdep::redress_eigenpotentials(&mo_b.v_inv_sqrt, &current_pdep.eigenpotentials)?;

    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let mut z_out = Array1::<f64>::ones(mo_indices.len());

    let first_act = mo_b.first_act;
    let sigma_x_all = sigma_x_diag(mo_b);
    let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);
    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        let m_loc = mo_abs - first_act;
        eps_mf[idx] = mo_b.eps_act[m_loc];
        sx_out[idx] = sigma_x_all[m_loc];
        eps_qp[idx] = eps_mf[idx];
    }
    let mut iter_done = 0usize;
    for it in 0..gw_cfg.max_ev_iter {
        // Update shifted_rhf.eps_alpha by overlaying QP energies for the
        // QP block (in absolute MO indices).
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            shifted_rhf.eps_alpha[mo_abs] = eps_qp[idx];
        }
        // Rebuild PDEP on iterations > 0.
        if it > 0 {
            current_pdep = ferric_rpa::run_pdep_rpa(mol, obs, dfbs, op, &shifted_rhf, pdep_cfg)?;
            current_v_dressed = w_pdep::redress_eigenpotentials(
                &mo_b.v_inv_sqrt,
                &current_pdep.eigenpotentials,
            )?;
        }
        let m_proj = project_b_into_pdep(mo_b, &current_v_dressed);
        let inv_diel_freq = current_pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
            FerricError::General(
                "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
            )
        })?;

        // For the propagator denominator we also use QP-shifted ε's
        // (consistent with the rebuilt W).
        let mut eps_prop = mo_b.eps_act.clone();
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            let m_loc = mo_abs - first_act;
            eps_prop[m_loc] = eps_qp[idx];
        }
        let mut max_dev = 0.0_f64;
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            let m_loc = mo_abs - first_act;
            let (eps_new, sc_new, z_new) = solve_qp_for_mo(
                m_loc,
                mo_b.eps_act[m_loc],
                &m_proj,
                inv_diel_freq,
                &current_pdep.quad_weights,
                &current_pdep.quad_freqs,
                &eps_prop,
                gw_cfg.pade_npts,
                gw_cfg.qp_newton_damp,
                ef,
                0.0,
            );
            max_dev = max_dev.max((eps_new - eps_qp[idx]).abs());
            eps_qp[idx] = eps_new;
            sc_out[idx] = sc_new;
            z_out[idx] = z_new;
        }
        iter_done = it + 1;
        if max_dev < gw_cfg.ev_conv_thresh && it > 0 {
            break;
        }
    }

    Ok(GwResult {
        mo_indices,
        eps_mf,
        eps_qp,
        sigma_x: sx_out,
        sigma_c: sc_out,
        z_factor: z_out,
        n_ev_iter: iter_done,
        pdep: current_pdep,
    })
}
