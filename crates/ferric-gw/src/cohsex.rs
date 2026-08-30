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

/// Gate the projected M tensor (m_modes, n_act, n_act) against the memory
/// budget.
///
/// # Why this now returns `Err` instead of only printing
///
/// This was a warn-once `eprintln!`, with a doc comment that justified itself
/// by saying `project_b_into_pdep` "is infallible ... so we surface the
/// concrete GB number rather than change the signature to `Result`". That made
/// it a *diagnostic*, not a gate: the allocation went ahead regardless, one
/// line of stderr scrolled past in a long GW log, and the OOM killer arrived
/// seconds later — the exact failure mode the budget machinery exists to
/// retire. A guard that cannot refuse is not a guard. `project_b_into_pdep` is
/// therefore fallible now and every caller propagates.
///
/// The warn-once latch is kept for the *stderr* advice (the truncation hint is
/// worth printing once per process, not once per evGW iteration), but the
/// `Err` itself is unconditional.
///
/// # Why `budget` is a parameter
///
/// It used to read `resolve_budget_bytes(None)`, which discards
/// [`GwConfig::memory_budget_bytes`](crate::GwConfig::memory_budget_bytes) and
/// substitutes the env / cgroup / RAM-auto-detected ceiling instead — so a run
/// pinned to 4 GB could be gated against ~0.8× of a 23 GB box. Same defect,
/// same fix, as `ferric_rpa::properties::pdep_polarizability_static`; ferric-gw
/// simply never got it.
static M_PROJ_WARNED: AtomicBool = AtomicBool::new(false);
fn guard_m_proj(m_modes: usize, n_act: usize, budget: Option<usize>) -> Result<(), FerricError> {
    let budget = ferric_core::memory::resolve_budget_bytes(budget);
    let need = m_modes
        .saturating_mul(n_act)
        .saturating_mul(n_act)
        .saturating_mul(8);
    if need <= budget {
        return Ok(());
    }
    let msg = format!(
        "projected M tensor ({m_modes}×{n_act}×{n_act} f64 = {:.2} GB) exceeds the memory \
         budget ({:.2} GB) and is rebuilt every evGW iteration (×2 for U-GW). Full-rank GW \
         at this scale needs trunc_thresh > 0 (rank truncation shrinks this quadratically), \
         a smaller active space, or a larger budget.",
        need as f64 / 1e9,
        budget as f64 / 1e9,
    );
    if !M_PROJ_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("ferric-gw: {msg}");
    }
    Err(FerricError::General(format!("project_b_into_pdep: {msg}")))
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
/// m_modes and this tensor quadratically.
///
/// Gated by [`guard_m_proj`] against `memory_budget_bytes` (the caller's
/// [`GwConfig::memory_budget_bytes`](crate::GwConfig::memory_budget_bytes), or
/// `None` to fall back to the env/auto ceiling). This function is FALLIBLE for
/// that reason — see `guard_m_proj`'s docs for why the old warn-only form was
/// not a gate.
pub fn project_b_into_pdep(
    mo_b: &MoB,
    v_dressed: &Array2<f64>,
    memory_budget_bytes: Option<usize>,
) -> Result<ndarray::Array3<f64>, FerricError> {
    let naux = mo_b.naux;
    let n_act = mo_b.n_act;
    let m_modes = v_dressed.ncols();
    guard_m_proj(m_modes, n_act, memory_budget_bytes)?;
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
    Ok(m_flat
        .into_shape_with_order((m_modes, n_act, n_act))
        .expect("reshape M"))
}

/// Per-MO static COHSEX pieces from a projected M tensor (one spin channel;
/// closed-shell RHF and each U-COHSEX spin call this with their own `mo_b`
/// and `m_proj`, sharing `w_static` — W is spin-independent):
///
///   ΔΣ_SEX(m) = −Σ_{i occ} Σ_α w_α M[(α,m,i)]²   (w_α = 1/λ_α(0) − 1)
///   Σ_COH(m)  = +½ Σ_{p all} Σ_α w_α M[(α,m,p)]²
///
/// Independent per m_idx (each (delta_sex[m_idx], coh[m_idx]) pair is
/// written exactly once) — parallelize over m_idx, order-preserving
/// par_iter + collect (no reduction, bit-identical by construction). No
/// BLAS inside. Serial below PAR_ROWS_THRESHOLD.
pub(crate) fn cohsex_pieces(
    mo_b: &MoB,
    m_proj: &ndarray::Array3<f64>,
    w_static: &[f64],
) -> (Array1<f64>, Array1<f64>) {
    let n_act = mo_b.n_act;
    let n_occ = mo_b.n_occ_act;
    let m_modes = w_static.len();
    const PAR_ROWS_THRESHOLD: usize = 8;
    let compute_one = |m_idx: usize| -> (f64, f64) {
        let mut delta_sex = 0.0;
        let mut coh = 0.0;
        for alpha in 0..m_modes {
            let w_a = w_static[alpha];
            // SEX: sum over occupied i.
            let mut sex_acc = 0.0;
            for i in 0..n_occ {
                let v = m_proj[(alpha, m_idx, i)];
                sex_acc += v * v;
            }
            delta_sex -= w_a * sex_acc;
            // COH: sum over ALL p (occ + vir).
            let mut coh_acc = 0.0;
            for p in 0..n_act {
                let v = m_proj[(alpha, m_idx, p)];
                coh_acc += v * v;
            }
            coh += 0.5 * w_a * coh_acc;
        }
        (delta_sex, coh)
    };
    let pieces: Vec<(f64, f64)> = if n_act >= PAR_ROWS_THRESHOLD {
        use rayon::prelude::*;
        (0..n_act).into_par_iter().map(compute_one).collect()
    } else {
        (0..n_act).map(compute_one).collect()
    };
    let mut delta_sex = Array1::<f64>::zeros(n_act);
    let mut coh = Array1::<f64>::zeros(n_act);
    for (m_idx, (ds, c)) in pieces.into_iter().enumerate() {
        delta_sex[m_idx] = ds;
        coh[m_idx] = c;
    }
    (delta_sex, coh)
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
    let first_act = mo_b.first_act;
    let m_modes = v_dressed.ncols();

    // Σ_x for all active MOs (cheap), then we select qp_range at the end.
    let sigma_x_all = sigma_x_diag(mo_b);

    // Static reduced weights w_α = 1/λ_α(0) − 1.
    let w_static = w_pdep::static_weights(&pdep.eigenvalues_static);
    assert_eq!(w_static.len(), m_modes);

    // Project B̃ → M[(α,m,n)].
    let m_proj = project_b_into_pdep(mo_b, v_dressed, gw_cfg.memory_budget_bytes)?;

    // ΔΣ_SEX(m) = −Σ_i Σ_α [1/λ_α(0) − 1] M[(α,m,i)]² = −Σ_i Σ_α w_α M²
    // Σ_COH(m)  = +½ Σ_p Σ_α  w_α                     M[(α,m,p)]²
    //
    // Hedin convention reconciliation: Σ_x already equals the bare-exchange
    // diagonal (=−Σ_i (mi|im) RI form). The *full* screened exchange
    // Σ_SEX_full(m) = −Σ_i Σ_α [1/λ_α(0)] M_{mi}² satisfies
    // Σ_x + ΔΣ_SEX = Σ_SEX_full (by linearity), so the QP equation only
    // needs the reduced-weight ΔΣ_SEX; Σ_x is reported separately.
    let (delta_sigma_sex, sigma_coh) = cohsex_pieces(mo_b, &m_proj, &w_static);

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
            eigensolver_converged: true,
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

    /// The M-projection guard must REFUSE, not merely warn.
    ///
    /// `guard_m_proj` was a warn-once `eprintln!` returning `()`, justified in
    /// its own doc comment by `project_b_into_pdep` being infallible. A guard
    /// that cannot refuse is a diagnostic: the allocation proceeded, the line
    /// scrolled past in a long evGW log, and the OOM killer followed. This
    /// pins that the over-budget path is an `Err` reaching the caller.
    #[test]
    fn project_b_into_pdep_refuses_an_over_budget_projection() {
        let mo_b = synthetic_mo_b(8, 6, 3);
        let v_dressed = Array2::<f64>::eye(8);
        // 8 modes x 6 x 6 x 8 bytes = 2304 bytes. A 1 kB ceiling cannot hold it.
        let err = project_b_into_pdep(&mo_b, &v_dressed, Some(1_000))
            .expect_err("an over-budget M projection must be an Err, not a warning")
            .to_string();
        assert!(err.contains("project_b_into_pdep"), "must name the site: {err}");
        assert!(err.contains("projected M tensor"), "must name the term: {err}");
        assert!(err.contains("trunc_thresh"), "must keep the actionable advice: {err}");
    }

    /// An over-estimating guard is also a bug. The SAME shape under an ample
    /// ceiling must run to completion and return the correct tensor, so the
    /// refusal above is attributable to the budget and not to the shape.
    #[test]
    fn project_b_into_pdep_still_runs_under_an_ample_budget() {
        let mo_b = synthetic_mo_b(8, 6, 3);
        let v_dressed = Array2::<f64>::eye(8);
        let m = project_b_into_pdep(&mo_b, &v_dressed, Some(1_000_000_000))
            .expect("an ample 1 GB budget must not be refused for a 2.3 kB tensor");
        assert_eq!(m.shape(), &[8, 6, 6]);
        // With v_dressed = I the projection is the identity on b_full, so this
        // also pins that adding the guard did not perturb the numerics.
        for p in 0..8 {
            for i in 0..6 {
                for j in 0..6 {
                    assert_eq!(m[(p, i, j)], mo_b.b_full[(p, i, j)]);
                }
            }
        }
    }

    /// The plumbing pin, distinct from the guard tests above: the ONLY
    /// difference between these two calls is `memory_budget_bytes`. The site
    /// used to pass `resolve_budget_bytes(None)`, discarding
    /// `GwConfig::memory_budget_bytes` and substituting the env / cgroup /
    /// RAM-auto ceiling — under which both calls would have succeeded.
    #[test]
    fn project_b_into_pdep_honours_the_caller_budget_rather_than_discarding_it() {
        let mo_b = synthetic_mo_b(8, 6, 3);
        let v_dressed = Array2::<f64>::eye(8);
        assert!(project_b_into_pdep(&mo_b, &v_dressed, Some(1_000_000_000)).is_ok());
        assert!(
            project_b_into_pdep(&mo_b, &v_dressed, Some(1_000)).is_err(),
            "a caller-supplied 1 kB ceiling was ignored — the budget is not reaching the guard"
        );
    }

}
