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
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
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
    let budget = ferric_core::memory::resolve_budget_bytes(None);
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
///
/// Elements are independent (each `sx[m]` is written exactly once by a
/// single `m`) — parallelize over `m`, order-preserving `par_iter` +
/// `collect` into the same-shape output (no reduction, bit-identical by
/// construction). No BLAS inside (scalar accumulation), so no
/// with_blas_threads guard. Serial below PAR_ROWS_THRESHOLD.
pub fn sigma_x_diag(mo_b: &MoB) -> Array1<f64> {
    let n_act = mo_b.n_act;
    let naux = mo_b.naux;
    let n_occ = mo_b.n_occ_act;
    let compute_one = |m: usize| -> f64 {
        let mut acc = 0.0;
        for i in 0..n_occ {
            for p in 0..naux {
                let b = mo_b.b_full[(p, m, i)];
                acc += b * b;
            }
        }
        -acc
    };
    const PAR_ROWS_THRESHOLD: usize = 8;
    let sx: Vec<f64> = if n_act >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        (0..n_act).into_par_iter().map(compute_one).collect()
    } else {
        (0..n_act).map(compute_one).collect()
    };
    Array1::from_vec(sx)
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
    // M_flat[(α, mn)] = Σ_P V[(P,α)] · b_flat[(P, mn)] = V^T · b_flat.
    // This (m_modes×naux)·(naux×n_act²) product is the single largest GEMM in
    // GW (rebuilt every evGW iteration, ×2 per spin for U-GW), so raise BLAS
    // threads on it under the opt-in guard.
    //
    // Call-path proof (this must NOT run inside a rayon parallel region — the
    // rayon-worker self-guard in opt_in_blas_threads() would silently degrade
    // the raise to 1 there, AND a raise inside a par region is a design error).
    // `project_b_into_pdep` is `pub` but every caller invokes it from serial
    // driver code, before any assembly/quadrature par_iter:
    //   * cohsex.rs::run_cohsex (:132) — before the row-parallel SEX/COH loops;
    //   * u_cohsex.rs::run_u_cohsex (:30-31) — before cohsex_pieces' par loops;
    //   * sigma.rs G0W0 (:245) / evGW outer `for it` loop (:508 — a plain
    //     serial for-loop, not rayon) — before the per-MO frequency work;
    //   * u_sigma.rs (:41-42,:142-143,:316-317) — same serial-driver placement;
    //   * bse.rs run_bse_tda/run_bse_c6/run_bse_c6_ks (:134,:420,:630) — before
    //     the row-parallel A-matrix assembly.
    // None of these enclose the call in a par_iter. opt_in_blas_threads()
    // additionally defaults to 1 (bit-identical to today) and self-guards to 1
    // from any rayon worker, so this is safe even if a future caller breaks the
    // assumption.
    let m_flat: Array2<f64> =
        with_blas_threads(opt_in_blas_threads(), || v_dressed.t().dot(&b_flat));
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
    //
    // Independent per m_idx (each (sigma_sex[m_idx], sigma_coh[m_idx]) pair
    // is written exactly once) — parallelize over m_idx, order-preserving
    // par_iter + collect (no reduction). No BLAS inside. Serial below
    // PAR_ROWS_THRESHOLD.
    const PAR_ROWS_THRESHOLD: usize = 8;
    let compute_sex_coh = |m_idx: usize| -> (f64, f64) {
        let mut sex = 0.0;
        let mut coh = 0.0;
        for alpha in 0..m_modes {
            let inv_lam = w_static[alpha] + 1.0; // = 1/λ_α(0)
            // SEX: sum over occupied i.
            let mut sex_acc = 0.0;
            for i in 0..n_occ {
                let v = m_proj[(alpha, m_idx, i)];
                sex_acc += v * v;
            }
            sex -= inv_lam * sex_acc;
            // COH: sum over ALL p (occ + vir).
            let mut coh_acc = 0.0;
            for p in 0..n_act {
                let v = m_proj[(alpha, m_idx, p)];
                coh_acc += v * v;
            }
            coh += 0.5 * w_static[alpha] * coh_acc;
        }
        (sex, coh)
    };
    let sex_coh: Vec<(f64, f64)> = if n_act >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        (0..n_act).into_par_iter().map(compute_sex_coh).collect()
    } else {
        (0..n_act).map(compute_sex_coh).collect()
    };
    let mut sigma_sex = Array1::<f64>::zeros(n_act);
    let mut sigma_coh = Array1::<f64>::zeros(n_act);
    for (m_idx, (sex, coh)) in sex_coh.into_iter().enumerate() {
        sigma_sex[m_idx] = sex;
        sigma_coh[m_idx] = coh;
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
    // Independent per m_idx — same par pattern as the sex_coh loop above.
    let compute_dsex = |m_idx: usize| -> f64 {
        let mut acc_total = 0.0;
        for alpha in 0..m_modes {
            let mut acc = 0.0;
            for i in 0..n_occ {
                let v = m_proj[(alpha, m_idx, i)];
                acc += v * v;
            }
            acc_total -= w_static[alpha] * acc;
        }
        acc_total
    };
    let delta_sigma_sex: Array1<f64> = if n_act >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        Array1::from_vec((0..n_act).into_par_iter().map(compute_dsex).collect())
    } else {
        Array1::from_vec((0..n_act).map(compute_dsex).collect())
    };

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

    let n_states = mo_indices.len();
    Ok(GwResult {
        mo_indices,
        eps_mf,
        eps_qp,
        sigma_x: sx_out,
        sigma_c: sc_out,
        z_factor: z_out,
        qp_converged: vec![true; n_states], // closed-form static, no Newton solve
        n_ev_iter: 0,
        outer_converged: true,
        pdep,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Deterministic synthetic dressed tensor (no RHF/PDEP build needed) —
    /// just needs plausible shapes for `sigma_x_diag` / the SEX-COH loops.
    fn synthetic_mo_b(naux: usize, n_act: usize, n_occ_act: usize) -> MoB {
        let mut b_full = Array3::<f64>::zeros((naux, n_act, n_act));
        for p in 0..naux {
            for m in 0..n_act {
                for n in 0..n_act {
                    // Arbitrary deterministic values, symmetric in (m,n) like a
                    // real dressed tensor for a fixed p (not required for the
                    // test, just plausible).
                    let v = ((p * 7 + m * 3 + n * 5 + 1) as f64).sin() * 0.1;
                    b_full[(p, m, n)] = v;
                }
            }
        }
        MoB {
            b_full,
            v_inv_sqrt: Array2::<f64>::eye(naux),
            naux,
            n_act,
            first_act: 0,
            n_occ_act,
            eps_act: (0..n_act).map(|i| i as f64 * 0.1).collect(),
        }
    }

    #[test]
    fn sigma_x_diag_bit_identical_across_thread_counts() {
        // P4: sigma_x_diag's per-m par_iter must be bit-identical regardless
        // of RAYON_NUM_THREADS — each element is written by exactly one
        // worker (map+collect, no reduction). n_act=12 exceeds
        // PAR_ROWS_THRESHOLD=8 so the parallel branch runs at 4 threads.
        let mo_b = synthetic_mo_b(10, 12, 5);
        let run_with_threads = |n: usize| -> Array1<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            pool.install(|| sigma_x_diag(&mo_b))
        };
        let r1 = run_with_threads(1);
        let r4 = run_with_threads(4);
        assert_eq!(r1.len(), r4.len());
        for (m, (a, b)) in r1.iter().zip(r4.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "sigma_x_diag[{m}] not bit-identical: 1-thread={a:.17e}, 4-thread={b:.17e}"
            );
        }
    }

    #[test]
    fn cohsex_pieces_via_run_cohsex_bit_identical_across_thread_counts() {
        // Exercise the sex/coh and delta_sigma_sex row-parallel loops inside
        // run_cohsex end-to-end with a synthetic fixture (no RHF/PDEP-RPA
        // solve needed — those are integration-tested in
        // tests/h2o_g0w0_cohsex.rs). n_act=12 exceeds PAR_ROWS_THRESHOLD=8.
        let naux = 10;
        let n_act = 12;
        let mo_b = synthetic_mo_b(naux, n_act, 5);
        let v_dressed = Array2::<f64>::eye(naux);
        let eigenvalues_static: Vec<f64> =
            (0..naux).map(|k| 1.0 + 0.05 * (k as f64 + 1.0)).collect();
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let rhf_stub_pdep = || PdepRpaResult {
            e_rpa: 0.0,
            n_eigenpotentials: naux,
            eigenvalues_static: eigenvalues_static.clone(),
            eigenpotentials: Array2::<f64>::eye(naux),
            dressed_eigenvectors: Array2::<f64>::eye(naux),
            quad_freqs: vec![],
            quad_weights: vec![],
            eigenvalues_freq: Array2::<f64>::zeros((0, naux)),
            inv_dielectric_freq: None,
            e_rpa_dft_diag: None,
        };
        // run_cohsex needs an rhf/gw_cfg but only for logging (`let _ = (mol,
        // gw_cfg, rhf)`), so any RHF result over the fixture molecule works —
        // build the cheapest one available.
        let obs = ferric_integrals::basis_bridge::PreparedBasis::new(
            &mol,
            &ferric_core::basis::bundled("sto-3g").unwrap(),
        )
        .unwrap();
        let op = ferric_integrals::operator::Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = ferric_scf::rhf::solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &ferric_scf::rhf::RhfConfig::default(),
        )
        .unwrap();
        let gw_cfg = GwConfig::default();

        let run_with_threads = |n: usize| -> GwResult {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            pool.install(|| {
                run_cohsex(&mol, &rhf, &mo_b, &v_dressed, rhf_stub_pdep(), 0..n_act, &gw_cfg)
                    .unwrap()
            })
        };
        let r1 = run_with_threads(1);
        let r4 = run_with_threads(4);
        assert_eq!(r1.eps_qp.len(), r4.eps_qp.len());
        for (idx, (a, b)) in r1.eps_qp.iter().zip(r4.eps_qp.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "run_cohsex eps_qp[{idx}] not bit-identical: 1-thread={a:.17e}, 4-thread={b:.17e}"
            );
        }
        for (idx, (a, b)) in r1.sigma_c.iter().zip(r4.sigma_c.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "run_cohsex sigma_c[{idx}] not bit-identical: 1-thread={a:.17e}, 4-thread={b:.17e}"
            );
        }
    }
}
