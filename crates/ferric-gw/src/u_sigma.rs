//! Spin-unrestricted GW self-energy: U-G0W0, U-evGW₀, U-evGW.
//!
//! W is built from the spin-summed dielectric (ε̃ = I + Π_α + Π_β) via
//! `run_u_pdep_rpa`, so the PDEP eigenpotentials V_α and frequency-dependent
//! eigenvalues λ_α(iω) are spin-independent.
//!
//! Per-spin Σ_c is then:
//!
//!   Σ_c,σ(m, z) = − (1/π) Σ_{n in σ} Σ_α  M_σ²_{mn,α}
//!                   ∫₀^∞ dω' w_α(iω')
//!                    ·  (z − ε_{n,σ}) / ((z − ε_{n,σ})² + ω'²)
//!
//! where M_σ_α^{mn} = Σ_P V_α^P B̃_σ^P_{mn} (per-spin projection of B̃ onto
//! the shared eigenpotentials), and w_α(iω) = 1/λ_α(iω) − 1.

use crate::cohsex::{project_b_into_pdep, sigma_x_diag};
use crate::mo_b::MoB;
use crate::sigma::{fermi_level, solve_qp_for_mo};
use crate::w_pdep;
use crate::{GwConfig, UGwResult};
use ferric_core::FerricError;
use ferric_rpa::PdepRpaResult;
use ndarray::Array1;
use rayon::prelude::*;

/// Run U-G0W0 given pre-built per-spin MoB and a shared U-PDEP-RPA result.
pub fn run_u_g0w0(
    mo_b_a: &MoB,
    mo_b_b: &MoB,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
    v_dressed: &ndarray::Array2<f64>,
) -> Result<UGwResult, FerricError> {
    let m_modes = v_dressed.ncols();
    if pdep.eigenvalues_freq.ncols() != m_modes {
        return Err(FerricError::General(
            "pdep eigenvalues_freq mode count does not match dressed eigenpotentials".into(),
        ));
    }
    let m_proj_a = project_b_into_pdep(mo_b_a, v_dressed);
    let m_proj_b = project_b_into_pdep(mo_b_b, v_dressed);
    let sigma_x_a_all = sigma_x_diag(mo_b_a);
    let sigma_x_b_all = sigma_x_diag(mo_b_b);
    let inv_diel_freq = pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
        FerricError::General(
            "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
        )
    })?;
    let quad_freqs = pdep.quad_freqs.clone();
    let quad_weights = pdep.quad_weights.clone();

    let (eps_qp_a, eps_mf_a, sx_a, sc_a, z_a) = qp_per_spin_g0w0(
        mo_b_a, &m_proj_a, &sigma_x_a_all, inv_diel_freq, &quad_weights, &quad_freqs,
        &qp_range, gw_cfg,
    )?;
    let (eps_qp_b, eps_mf_b, sx_b, sc_b, z_b) = qp_per_spin_g0w0(
        mo_b_b, &m_proj_b, &sigma_x_b_all, inv_diel_freq, &quad_weights, &quad_freqs,
        &qp_range, gw_cfg,
    )?;

    Ok(UGwResult {
        mo_indices: qp_range.collect(),
        eps_mf_a, eps_qp_a, sigma_x_a: sx_a, sigma_c_a: sc_a, z_factor_a: z_a,
        eps_mf_b, eps_qp_b, sigma_x_b: sx_b, sigma_c_b: sc_b, z_factor_b: z_b,
        n_ev_iter: 0,
        pdep,
    })
}

/// Helper: per-spin G0W0 QP loop.
fn qp_per_spin_g0w0(
    mo_b: &MoB,
    m_proj: &ndarray::Array3<f64>,
    sigma_x_all: &Array1<f64>,
    inv_diel_freq: &[ndarray::Array2<f64>],
    quad_weights: &[f64],
    quad_freqs: &[f64],
    qp_range: &std::ops::Range<usize>,
    gw_cfg: &GwConfig,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), FerricError> {
    let first_act = mo_b.first_act;
    let ef = fermi_level(&mo_b.eps_act, mo_b.n_occ_act);
    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let mut z_out = Array1::<f64>::ones(mo_indices.len());
    // Independent per-state QP solves (scalar math only) — parallelize.
    let qp_rows = mo_indices
        .par_iter()
        .map(|&mo_abs| {
            if mo_abs < first_act {
                return Err(FerricError::General(format!(
                    "qp_mos index {mo_abs} is in the frozen-core block"
                )));
            }
            let m_loc = mo_abs - first_act;
            let eps_m = mo_b.eps_act[m_loc];
            let (eps_qp_m, sc_final, z_renorm) = solve_qp_for_mo(
                m_loc, eps_m, m_proj, inv_diel_freq, quad_weights, quad_freqs,
                &mo_b.eps_act, gw_cfg.pade_npts, gw_cfg.qp_newton_damp, ef, 0.0,
            );
            Ok((eps_m, sigma_x_all[m_loc], eps_qp_m, sc_final, z_renorm))
        })
        .collect::<Result<Vec<_>, FerricError>>()?;
    for (idx, &(eps_m, sx, eps_qp_m, sc_final, z_renorm)) in qp_rows.iter().enumerate() {
        eps_mf[idx] = eps_m;
        sx_out[idx] = sx;
        eps_qp[idx] = eps_qp_m;
        sc_out[idx] = sc_final;
        z_out[idx] = z_renorm;
    }
    Ok((eps_qp, eps_mf, sx_out, sc_out, z_out))
}

/// U-evGW₀: per-spin eigenvalue self-consistency on G; W frozen at iter 0.
pub fn run_u_evgw0(
    mo_b_a: &MoB,
    mo_b_b: &MoB,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
    v_dressed: &ndarray::Array2<f64>,
) -> Result<UGwResult, FerricError> {
    let m_modes = v_dressed.ncols();
    if pdep.eigenvalues_freq.ncols() != m_modes {
        return Err(FerricError::General(
            "pdep eigenvalues_freq mode count does not match dressed eigenpotentials".into(),
        ));
    }
    let m_proj_a = project_b_into_pdep(mo_b_a, v_dressed);
    let m_proj_b = project_b_into_pdep(mo_b_b, v_dressed);
    let sigma_x_a_all = sigma_x_diag(mo_b_a);
    let sigma_x_b_all = sigma_x_diag(mo_b_b);
    let inv_diel_freq = pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
        FerricError::General(
            "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
        )
    })?;

    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let first_act_a = mo_b_a.first_act;
    let first_act_b = mo_b_b.first_act;
    let mut eps_qp_a = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_qp_b = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf_a = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf_b = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_a = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_b = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_a = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_b = Array1::<f64>::zeros(mo_indices.len());
    let mut z_a = Array1::<f64>::ones(mo_indices.len());
    let mut z_b = Array1::<f64>::ones(mo_indices.len());
    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        let mla = mo_abs - first_act_a;
        let mlb = mo_abs - first_act_b;
        eps_mf_a[idx] = mo_b_a.eps_act[mla];
        eps_mf_b[idx] = mo_b_b.eps_act[mlb];
        sx_a[idx] = sigma_x_a_all[mla];
        sx_b[idx] = sigma_x_b_all[mlb];
        eps_qp_a[idx] = eps_mf_a[idx];
        eps_qp_b[idx] = eps_mf_b[idx];
    }

    let ef_a = fermi_level(&mo_b_a.eps_act, mo_b_a.n_occ_act);
    let ef_b = fermi_level(&mo_b_b.eps_act, mo_b_b.n_occ_act);
    let mut eps_prop_a = mo_b_a.eps_act.clone();
    let mut eps_prop_b = mo_b_b.eps_act.clone();
    let mut iter_done = 0usize;
    for it in 0..gw_cfg.max_ev_iter {
        // Update propagator ε's per spin from previous QP estimates.
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            eps_prop_a[mo_abs - first_act_a] = eps_qp_a[idx];
            eps_prop_b[mo_abs - first_act_b] = eps_qp_b[idx];
        }
        let mut max_dev = 0.0_f64;
        // Frozen per-iteration eps_prop snapshots ⇒ independent per-state solves.
        let qp_new: Vec<((f64, f64, f64), (f64, f64, f64))> = mo_indices
            .par_iter()
            .map(|&mo_abs| {
                let mla = mo_abs - first_act_a;
                let mlb = mo_abs - first_act_b;
                let ra = solve_qp_for_mo(
                    mla, mo_b_a.eps_act[mla], &m_proj_a, inv_diel_freq,
                    &pdep.quad_weights, &pdep.quad_freqs, &eps_prop_a,
                    gw_cfg.pade_npts, gw_cfg.qp_newton_damp, ef_a, 0.0,
                );
                let rb = solve_qp_for_mo(
                    mlb, mo_b_b.eps_act[mlb], &m_proj_b, inv_diel_freq,
                    &pdep.quad_weights, &pdep.quad_freqs, &eps_prop_b,
                    gw_cfg.pade_npts, gw_cfg.qp_newton_damp, ef_b, 0.0,
                );
                (ra, rb)
            })
            .collect();
        for (idx, &((ena, sca, za), (enb, scb, zb))) in qp_new.iter().enumerate() {
            max_dev = max_dev.max((ena - eps_qp_a[idx]).abs()).max((enb - eps_qp_b[idx]).abs());
            eps_qp_a[idx] = ena; sc_a[idx] = sca; z_a[idx] = za;
            eps_qp_b[idx] = enb; sc_b[idx] = scb; z_b[idx] = zb;
        }
        iter_done = it + 1;
        if max_dev < gw_cfg.ev_conv_thresh { break; }
    }

    Ok(UGwResult {
        mo_indices,
        eps_mf_a, eps_qp_a, sigma_x_a: sx_a, sigma_c_a: sc_a, z_factor_a: z_a,
        eps_mf_b, eps_qp_b, sigma_x_b: sx_b, sigma_c_b: sc_b, z_factor_b: z_b,
        n_ev_iter: iter_done,
        pdep,
    })
}

/// U-evGW: rebuild W every iteration with QP-shifted ε's per spin.
pub fn run_u_evgw(
    mol: &ferric_core::mol::Molecule,
    obs: &ferric_integrals::basis_bridge::PreparedBasis,
    dfbs: &ferric_integrals::basis_bridge::PreparedBasis,
    op: ferric_integrals::operator::Operator,
    scf: &ferric_scf::ScfResult,
    pdep_cfg: &ferric_rpa::config::PdepRpaConfig,
    mo_b_a: &MoB,
    mo_b_b: &MoB,
    pdep0: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
) -> Result<UGwResult, FerricError> {
    let mut shifted_scf = scf.clone();
    let mut current_pdep = pdep0;
    let mut current_v_dressed = w_pdep::redress_eigenpotentials(
        &mo_b_a.v_inv_sqrt, &current_pdep.eigenpotentials,
    )?;

    let first_act_a = mo_b_a.first_act;
    let first_act_b = mo_b_b.first_act;
    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let sigma_x_a_all = sigma_x_diag(mo_b_a);
    let sigma_x_b_all = sigma_x_diag(mo_b_b);

    let mut eps_qp_a = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_qp_b = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf_a = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf_b = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_a = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_b = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_a = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_b = Array1::<f64>::zeros(mo_indices.len());
    let mut z_a = Array1::<f64>::ones(mo_indices.len());
    let mut z_b = Array1::<f64>::ones(mo_indices.len());
    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        let mla = mo_abs - first_act_a;
        let mlb = mo_abs - first_act_b;
        eps_mf_a[idx] = mo_b_a.eps_act[mla];
        eps_mf_b[idx] = mo_b_b.eps_act[mlb];
        sx_a[idx] = sigma_x_a_all[mla];
        sx_b[idx] = sigma_x_b_all[mlb];
        eps_qp_a[idx] = eps_mf_a[idx];
        eps_qp_b[idx] = eps_mf_b[idx];
    }

    let ef_a = fermi_level(&mo_b_a.eps_act, mo_b_a.n_occ_act);
    let ef_b = fermi_level(&mo_b_b.eps_act, mo_b_b.n_occ_act);
    let mut iter_done = 0usize;
    for it in 0..gw_cfg.max_ev_iter {
        // Overlay current QP energies on shifted_scf so PDEP χ₀ denominators
        // see the QP gaps. For ROHF, β reuses α — we still write both arrays
        // since the underlying compute_rpa_intermediates_spin reads eps_a()
        // for β when spin == RestrictedOpen.
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            shifted_scf.eps_alpha[mo_abs] = eps_qp_a[idx];
        }
        if let Some(eps_b_vec) = shifted_scf.eps_beta.as_mut() {
            for (idx, &mo_abs) in mo_indices.iter().enumerate() {
                eps_b_vec[mo_abs] = eps_qp_b[idx];
            }
        }
        if it > 0 {
            current_pdep = ferric_rpa::run_u_pdep_rpa(mol, obs, dfbs, op, &shifted_scf, pdep_cfg)?;
            current_v_dressed = w_pdep::redress_eigenpotentials(
                &mo_b_a.v_inv_sqrt, &current_pdep.eigenpotentials,
            )?;
        }
        let m_proj_a = project_b_into_pdep(mo_b_a, &current_v_dressed);
        let m_proj_b = project_b_into_pdep(mo_b_b, &current_v_dressed);
        let inv_diel_freq = current_pdep.inv_dielectric_freq.as_ref().ok_or_else(|| {
            FerricError::General(
                "PDEP result missing inv_dielectric_freq (GW requires the dense χ₀ path)".into(),
            )
        })?;

        let mut eps_prop_a = mo_b_a.eps_act.clone();
        let mut eps_prop_b = mo_b_b.eps_act.clone();
        for (idx, &mo_abs) in mo_indices.iter().enumerate() {
            eps_prop_a[mo_abs - first_act_a] = eps_qp_a[idx];
            eps_prop_b[mo_abs - first_act_b] = eps_qp_b[idx];
        }
        let mut max_dev = 0.0_f64;
        // Frozen (m_proj, W, eps_prop) snapshot ⇒ independent per-state solves.
        let qp_new: Vec<((f64, f64, f64), (f64, f64, f64))> = mo_indices
            .par_iter()
            .map(|&mo_abs| {
                let mla = mo_abs - first_act_a;
                let mlb = mo_abs - first_act_b;
                let ra = solve_qp_for_mo(
                    mla, mo_b_a.eps_act[mla], &m_proj_a, inv_diel_freq,
                    &current_pdep.quad_weights, &current_pdep.quad_freqs, &eps_prop_a,
                    gw_cfg.pade_npts, gw_cfg.qp_newton_damp, ef_a, 0.0,
                );
                let rb = solve_qp_for_mo(
                    mlb, mo_b_b.eps_act[mlb], &m_proj_b, inv_diel_freq,
                    &current_pdep.quad_weights, &current_pdep.quad_freqs, &eps_prop_b,
                    gw_cfg.pade_npts, gw_cfg.qp_newton_damp, ef_b, 0.0,
                );
                (ra, rb)
            })
            .collect();
        for (idx, &((ena, sca, za), (enb, scb, zb))) in qp_new.iter().enumerate() {
            max_dev = max_dev.max((ena - eps_qp_a[idx]).abs()).max((enb - eps_qp_b[idx]).abs());
            eps_qp_a[idx] = ena; sc_a[idx] = sca; z_a[idx] = za;
            eps_qp_b[idx] = enb; sc_b[idx] = scb; z_b[idx] = zb;
        }
        iter_done = it + 1;
        if max_dev < gw_cfg.ev_conv_thresh && it > 0 { break; }
    }

    Ok(UGwResult {
        mo_indices,
        eps_mf_a, eps_qp_a, sigma_x_a: sx_a, sigma_c_a: sc_a, z_factor_a: z_a,
        eps_mf_b, eps_qp_b, sigma_x_b: sx_b, sigma_c_b: sc_b, z_factor_b: z_b,
        n_ev_iter: iter_done,
        pdep: current_pdep,
    })
}
