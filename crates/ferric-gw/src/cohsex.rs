//! COHSEX (static screened-exchange + Coulomb hole).
//!
//! Σ_SEX(m) = − Σ_i Σ_α [1/λ_α(0)]              M^α_{mi}²
//! Σ_COH(m) = +½ Σ_p Σ_α [1/λ_α(0) − 1]         M^α_{mp}²  (sum over ALL active MOs p)
//! ε^QP_m   = ε_m + Σ_x(m) + Σ_SEX(m) + Σ_COH(m)    (no v_xc since HF reference)
//!
//! Note: with the convention that W̃_total(0) = ε̃⁻¹(0) in the dressed basis,
//! the static *full* W̃ has eigenvalues 1/λ_α(0). The exchange piece of W̃(0)
//! (i.e., 1/λ_α = 1 modes) reproduces bare Σ_x; modes with 1/λ_α < 1 give the
//! screened exchange shift. We compute Σ_x separately (from bare B̃) and then
//! Σ_SEX uses the *reduced* weight (1/λ_α). Hedin's original convention is
//! equivalent up to a relabeling.

use crate::mo_b::MoB;
use crate::w_pdep;
use crate::{GwConfig, GwResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::three_index_source::env_budget_bytes;
use ferric_rpa::PdepRpaResult;
use ferric_scf::ScfResult;
use ndarray::{Array1, Array2};
use std::sync::atomic::{AtomicBool, Ordering};

/// Warn (once) if the projected M tensor (m_modes, n_act, n_act) would exceed
/// `FERRIC_ERI3_BUDGET_GB`. `project_b_into_pdep` is infallible and called from
/// many sites (including per-iteration evGW rebuilds and both U-GW spins), so we
/// surface the concrete GB number rather than change the signature to `Result`.
/// M2 owns the hard allocation guards; this keeps the documented formula in sync.
static M_PROJ_WARNED: AtomicBool = AtomicBool::new(false);
fn guard_m_proj(m_modes: usize, n_act: usize) {
    let budget = env_budget_bytes();
    let need = m_modes
        .saturating_mul(n_act)
        .saturating_mul(n_act)
        .saturating_mul(8);
    if need > budget && !M_PROJ_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "ferric-gw WARNING: projected M tensor ({m_modes}×{n_act}×{n_act} f64 = \
             {:.2} GB) exceeds FERRIC_ERI3_BUDGET_GB ({:.2} GB) and is rebuilt every \
             evGW iteration (×2 for U-GW). Full-rank GW at this scale needs \
             trunc_thresh > 0 (rank truncation shrinks this quadratically), a smaller \
             active space, or a larger budget.",
            need as f64 / 1e9,
            budget as f64 / 1e9,
        );
    }
}

/// Σ_x(m) = −Σ_i Σ_P B̃^P_{mi}²  (closed-shell RHF exchange diagonal, MO basis).
pub fn sigma_x_diag(mo_b: &MoB) -> Array1<f64> {
    let n_act = mo_b.n_act;
    let naux = mo_b.naux;
    let n_occ = mo_b.n_occ_act;
    let mut sx = Array1::<f64>::zeros(n_act);
    // B̃[(P,m,i)] for i in 0..n_occ.
    for m in 0..n_act {
        let mut acc = 0.0;
        for i in 0..n_occ {
            for p in 0..naux {
                let b = mo_b.b_full[(p, m, i)];
                acc += b * b;
            }
        }
        sx[m] = -acc;
    }
    sx
}

/// Project B̃^P_{mn} onto PDEP dressed eigenpotentials V_α^P:
///   M[(α, m, n)] = Σ_P V_α^P · B̃^P_{mn}
///
/// Peak: a single (m_modes, n_act, n_act) f64 tensor. At full rank
/// (`trunc_thresh = 0`, the default) m_modes = naux, so this is the same
/// ~12.5 GB footprint as `b_full` and it is REBUILT every evGW outer iteration
/// (and once per spin for U-GW). Rank truncation (`trunc_thresh > 0`) shrinks
/// m_modes and this tensor quadratically. We warn (once) with the concrete GB
/// number if the projection would exceed `FERRIC_ERI3_BUDGET_GB`.
pub fn project_b_into_pdep(mo_b: &MoB, v_dressed: &Array2<f64>) -> ndarray::Array3<f64> {
    let naux = mo_b.naux;
    let n_act = mo_b.n_act;
    let m_modes = v_dressed.ncols();
    guard_m_proj(m_modes, n_act);
    // Reshape b_full (naux, n_act, n_act) → (naux, n_act*n_act) for one GEMM.
    let b_flat = mo_b
        .b_full
        .view()
        .into_shape_with_order((naux, n_act * n_act))
        .expect("reshape b_full");
    // M_flat[(α, mn)] = Σ_P V[(P,α)] · b_flat[(P, mn)] = V^T · b_flat
    let m_flat: Array2<f64> = v_dressed.t().dot(&b_flat);
    m_flat
        .into_shape_with_order((m_modes, n_act, n_act))
        .expect("reshape M")
}

/// Run COHSEX.
pub fn run_cohsex(
    mol: &Molecule,
    rhf: &ScfResult,
    mo_b: &MoB,
    v_dressed: &Array2<f64>,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    gw_cfg: &GwConfig,
) -> Result<GwResult, FerricError> {
    let _ = (mol, gw_cfg, rhf);
    let n_act = mo_b.n_act;
    let n_occ = mo_b.n_occ_act;
    let first_act = mo_b.first_act;
    let m_modes = v_dressed.ncols();

    // Σ_x for all active MOs (cheap), then we select qp_range at the end.
    let sigma_x_all = sigma_x_diag(mo_b);

    // Static reduced weights w_α = 1/λ_α(0) − 1.
    let w_static = w_pdep::static_weights(&pdep.eigenvalues_static);
    assert_eq!(w_static.len(), m_modes);

    // Project B̃ → M[(α,m,n)].
    let m_proj = project_b_into_pdep(mo_b, v_dressed);

    // Σ_SEX(m) = −Σ_i Σ_α [1/λ_α(0)] M[(α,m,i)]² = −Σ_i Σ_α (w_α + 1) M²
    // Σ_COH(m) = +½ Σ_p Σ_α  w_α     M[(α,m,p)]²
    let mut sigma_sex = Array1::<f64>::zeros(n_act);
    let mut sigma_coh = Array1::<f64>::zeros(n_act);
    for m_idx in 0..n_act {
        for alpha in 0..m_modes {
            let inv_lam = w_static[alpha] + 1.0; // = 1/λ_α(0)
            // SEX: sum over occupied i.
            let mut sex_acc = 0.0;
            for i in 0..n_occ {
                let v = m_proj[(alpha, m_idx, i)];
                sex_acc += v * v;
            }
            sigma_sex[m_idx] -= inv_lam * sex_acc;
            // COH: sum over ALL p (occ + vir).
            let mut coh_acc = 0.0;
            for p in 0..n_act {
                let v = m_proj[(alpha, m_idx, p)];
                coh_acc += v * v;
            }
            sigma_coh[m_idx] += 0.5 * w_static[alpha] * coh_acc;
        }
    }

    // Hedin convention reconciliation: Σ_x already equals the bare-exchange
    // diagonal (=−Σ_i (mi|im) RI form). Σ_SEX as written above equals the
    // *full* screened exchange (W = ε⁻¹·v); subtract the bare piece so that
    // the total exchange counted in the QP equation is Σ_x + ΔΣ_SEX.
    //
    // ΔΣ_SEX(m) = −Σ_i Σ_α [1/λ_α(0) − 1] M_{mi}² = −Σ_i Σ_α w_α M_{mi}²
    // and Σ_x + ΔΣ_SEX = Σ_SEX_full (by linearity), so we can either use
    // (Σ_x + ΔΣ_SEX) OR (just Σ_SEX_full). Both equivalent; we keep the
    // explicit decomposition for the report.
    let mut delta_sigma_sex = Array1::<f64>::zeros(n_act);
    for m_idx in 0..n_act {
        for alpha in 0..m_modes {
            let mut acc = 0.0;
            for i in 0..n_occ {
                let v = m_proj[(alpha, m_idx, i)];
                acc += v * v;
            }
            delta_sigma_sex[m_idx] -= w_static[alpha] * acc;
        }
    }

    // HF reference: Σ_x^GW − v_xc^HF = 0 exactly (the HF Fock already
    // contains Σ_x). So the QP correction is just the correlation piece
    // ΔΣ_SEX + Σ_COH.
    //
    // We still report Σ_x for diagnostics — it equals the diagonal of the
    // HF exchange in MO basis, used by `sigma_x_matches_rhf_exchange` test.
    let mo_indices: Vec<usize> = qp_range.clone().collect();
    let mut eps_qp = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_out = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_out = Array1::<f64>::zeros(mo_indices.len());
    let z_out = Array1::<f64>::ones(mo_indices.len()); // Z=1 for static.

    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        if mo_abs < first_act {
            return Err(FerricError::General(format!(
                "qp_mos index {mo_abs} is in the frozen-core block"
            )));
        }
        let m_loc = mo_abs - first_act;
        eps_mf[idx] = mo_b.eps_act[m_loc];
        sx_out[idx] = sigma_x_all[m_loc];
        let dsex = delta_sigma_sex[m_loc];
        let scoh = sigma_coh[m_loc];
        sc_out[idx] = dsex + scoh;
        eps_qp[idx] = eps_mf[idx] + sc_out[idx];
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
