//! Spin-unrestricted static COHSEX. Mirrors `cohsex.rs` per spin, using a
//! shared W (PDEP eigenvalues at iω=0 are spin-independent).
//!
//! Σ_SEX,σ(m) = − Σ_{i in σ} Σ_α [1/λ_α(0)] M_σ_α^{mi}²
//! Σ_COH,σ(m) = +½ Σ_{p in σ} Σ_α  w_α(0)   M_σ_α^{mp}²
//! ΔΣ_SEX,σ   = Σ_SEX,σ − Σ_x,σ = − Σ_{i in σ} Σ_α w_α(0) M_σ_α^{mi}²

use crate::cohsex::{project_b_into_pdep, sigma_x_diag};
use crate::mo_b::MoB;
use crate::w_pdep;
use crate::{GwConfig, UGwResult};
use ferric_core::FerricError;
use ferric_rpa::PdepRpaResult;
use ndarray::{Array1, Array2};

pub fn run_u_cohsex(
    mo_b_a: &MoB,
    mo_b_b: &MoB,
    pdep: PdepRpaResult,
    qp_range: std::ops::Range<usize>,
    _gw_cfg: &GwConfig,
    v_dressed: &Array2<f64>,
) -> Result<UGwResult, FerricError> {
    let m_modes = v_dressed.ncols();
    let w_static = w_pdep::static_weights(&pdep.eigenvalues_static);
    assert_eq!(w_static.len(), m_modes);

    let sigma_x_a_all = sigma_x_diag(mo_b_a);
    let sigma_x_b_all = sigma_x_diag(mo_b_b);
    let m_proj_a = project_b_into_pdep(mo_b_a, v_dressed);
    let m_proj_b = project_b_into_pdep(mo_b_b, v_dressed);

    let (dsex_a, scoh_a) = cohsex_pieces(mo_b_a, &m_proj_a, &w_static);
    let (dsex_b, scoh_b) = cohsex_pieces(mo_b_b, &m_proj_b, &w_static);

    let mo_indices: Vec<usize> = qp_range.collect();
    let mut eps_qp_a = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_qp_b = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf_a = Array1::<f64>::zeros(mo_indices.len());
    let mut eps_mf_b = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_a = Array1::<f64>::zeros(mo_indices.len());
    let mut sx_b = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_a = Array1::<f64>::zeros(mo_indices.len());
    let mut sc_b = Array1::<f64>::zeros(mo_indices.len());
    let z_a = Array1::<f64>::ones(mo_indices.len());
    let z_b = Array1::<f64>::ones(mo_indices.len());

    let first_act_a = mo_b_a.first_act;
    let first_act_b = mo_b_b.first_act;
    for (idx, &mo_abs) in mo_indices.iter().enumerate() {
        if mo_abs < first_act_a || mo_abs < first_act_b {
            return Err(FerricError::General(format!(
                "qp_mos index {mo_abs} is in the frozen-core block"
            )));
        }
        let mla = mo_abs - first_act_a;
        let mlb = mo_abs - first_act_b;
        eps_mf_a[idx] = mo_b_a.eps_act[mla];
        eps_mf_b[idx] = mo_b_b.eps_act[mlb];
        sx_a[idx] = sigma_x_a_all[mla];
        sx_b[idx] = sigma_x_b_all[mlb];
        sc_a[idx] = dsex_a[mla] + scoh_a[mla];
        sc_b[idx] = dsex_b[mlb] + scoh_b[mlb];
        eps_qp_a[idx] = eps_mf_a[idx] + sc_a[idx];
        eps_qp_b[idx] = eps_mf_b[idx] + sc_b[idx];
    }

    let n_states = mo_indices.len();
    Ok(UGwResult {
        mo_indices,
        eps_mf_a, eps_qp_a, sigma_x_a: sx_a, sigma_c_a: sc_a, z_factor_a: z_a,
        eps_mf_b, eps_qp_b, sigma_x_b: sx_b, sigma_c_b: sc_b, z_factor_b: z_b,
        qp_converged_a: vec![true; n_states], // closed-form static, no Newton solve
        qp_converged_b: vec![true; n_states],
        n_ev_iter: 0,
        outer_converged: true,
        pdep,
    })
}

fn cohsex_pieces(
    mo_b: &MoB,
    m_proj: &ndarray::Array3<f64>,
    w_static: &[f64],
) -> (Array1<f64>, Array1<f64>) {
    let n_act = mo_b.n_act;
    let n_occ = mo_b.n_occ_act;
    let m_modes = w_static.len();
    let mut delta_sex = Array1::<f64>::zeros(n_act);
    let mut coh = Array1::<f64>::zeros(n_act);
    for m_idx in 0..n_act {
        for alpha in 0..m_modes {
            let w_a = w_static[alpha];
            let mut sex_acc = 0.0;
            for i in 0..n_occ {
                let v = m_proj[(alpha, m_idx, i)];
                sex_acc += v * v;
            }
            delta_sex[m_idx] -= w_a * sex_acc;
            let mut coh_acc = 0.0;
            for p in 0..n_act {
                let v = m_proj[(alpha, m_idx, p)];
                coh_acc += v * v;
            }
            coh[m_idx] += 0.5 * w_a * coh_acc;
        }
    }
    (delta_sex, coh)
}
