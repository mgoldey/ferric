//! Full-MO dressed RI tensor B̃^P_{mn} = V^{-1/2}_{PQ} (Q | mn),
//! shape (naux, n_active_mo, n_active_mo).
//!
//! "Active" = MO indices `frozen_core..nmo`. The frozen-core indices are
//! excluded from the (m,n) pair list to match the PDEP intermediates that
//! were built with the same frozen_core.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex;
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2, Array3};
use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};

/// Dressed RI integrals over the full active-MO square.
#[derive(Debug, Clone)]
pub struct MoB {
    /// Shape (naux, n_act, n_act). `b_full[(P, m, n)] = Σ_Q V^{-1/2}_{PQ} (Q|mn)`.
    pub b_full: Array3<f64>,
    /// V^{-1/2}, shape (naux, naux). Same as `RpaIntermediates.v_inv_sqrt`.
    pub v_inv_sqrt: Array2<f64>,
    pub naux: usize,
    /// Number of active MOs (= n_total_mo - frozen_core).
    pub n_act: usize,
    /// Absolute MO index of the first active MO (== frozen_core).
    pub first_act: usize,
    /// Number of doubly-occupied active MOs (= nocc_total - frozen_core).
    pub n_occ_act: usize,
    /// Mean-field orbital energies on the active block, length n_act.
    pub eps_act: Vec<f64>,
}

/// Peak resident bytes of a dressed MO tensor `b_full` of shape
/// (naux, n_act, n_act). Used for the pre-allocation budget guard.
fn b_full_bytes(naux: usize, n_act: usize) -> usize {
    naux.saturating_mul(n_act)
        .saturating_mul(n_act)
        .saturating_mul(8)
}

/// Fail-fast if the requested `b_full` allocation exceeds `FERRIC_ERI3_BUDGET_GB`.
/// The MoB peak is a single (naux, n_act, n_act) f64 buffer with the aux-blocked
/// transform (the old code held eri3_mm + b_flat simultaneously → 2×). The AO
/// source is streamed aux-blocked under the same budget, so it is not the peak.
fn guard_b_full(naux: usize, n_act: usize, label: &str) -> Result<(), FerricError> {
    guard_b_full_at(ferric_core::memory::resolve_budget_bytes(None), naux, n_act, label)
}

fn guard_b_full_at(
    budget: usize,
    naux: usize,
    n_act: usize,
    label: &str,
) -> Result<(), FerricError> {
    let need = b_full_bytes(naux, n_act);
    if need > budget {
        return Err(FerricError::General(format!(
            "{label}: dressed MO tensor b_full ({naux}×{n_act}×{n_act} f64 = \
             {:.2} GB) exceeds FERRIC_ERI3_BUDGET_GB ({:.2} GB). Reduce the active \
             space (frozen_core / smaller basis) or raise the budget.",
            need as f64 / 1e9,
            budget as f64 / 1e9,
        )));
    }
    Ok(())
}

/// Metric Cholesky + triangular solve (naux×naux). Call-path proof: reached
/// only from `build_full_b_with_mos`/`build_full_b_both_spins`, which are in
/// turn reached only from top-level GW entry points (lib.rs:211,:279) and
/// BSE's top-level `build_full_b` calls (bse.rs) — none of which run inside a
/// rayon `par_iter`. `opt_in_blas_threads()` defaults to 1 (identical to
/// today) and additionally self-guards to 1 if it is ever invoked from a
/// rayon worker thread, so this raise is safe even if a future caller's
/// assumption breaks.
fn cholesky_inverse_sqrt(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = v.nrows();
    with_blas_threads(opt_in_blas_threads(), || {
        let l = v
            .cholesky(UPLO::Lower)
            .map_err(|e| FerricError::General(format!("V cholesky failed: {e}")))?;
        let eye = Array2::<f64>::eye(n);
        let v_inv_sqrt = l
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &eye)
            .map_err(|e| FerricError::General(format!("triangular solve failed: {e}")))?;
        Ok(v_inv_sqrt)
    })
}

/// Build B̃^P_{mn} over the full active-MO square (closed-shell RHF reference).
pub fn build_full_b(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
) -> Result<MoB, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "build_full_b: closed-shell only — use build_full_b_spin for U/RO/UKS".into(),
        ));
    }
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    build_full_b_with_mos(obs, dfbs, op, rhf.mos_r(), rhf.eps_r(), nocc_total, frozen_core)
}

/// Build per-spin B̃^P_{mn} for an open-shell reference (UHF, ROHF, UKS).
///
/// `is_alpha = true` uses α-MOs + α-eps; `false` uses β. For ROHF, β
/// reuses the α-MO coefficients (Guest–Saunders canonicalized α serves
/// both channels) — matches `compute_rpa_intermediates_spin` and
/// `run_u_pdep_rpa`.
///
/// `nocc_σ` is read from the molecule's nelec + 2S:
///   nocc_α = (nelec + 2S) / 2, nocc_β = (nelec − 2S) / 2.
pub fn build_full_b_spin(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    is_alpha: bool,
    frozen_core: usize,
) -> Result<MoB, FerricError> {
    if matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "build_full_b_spin: use build_full_b for Restricted results".into(),
        ));
    }
    let nelec = mol.nelec();
    let two_s = (mol.multiplicity as i32) - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    let (mos, eps_slice, nocc) = if is_alpha {
        (scf.mos_a(), scf.eps_a(), nocc_a)
    } else {
        // ROHF reuses α MOs and α energies for β; UHF has its own β block.
        match scf.spin {
            Spin::RestrictedOpen => (scf.mos_a(), scf.eps_a(), nocc_b),
            Spin::Unrestricted => (scf.mos_b(), scf.eps_b(), nocc_b),
            Spin::Restricted => unreachable!(),
        }
    };
    build_full_b_with_mos(obs, dfbs, op, mos, eps_slice, nocc, frozen_core)
}

fn build_full_b_with_mos(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    c: &Array2<f64>,
    eps_full: &[f64],
    nocc_total: usize,
    frozen_core: usize,
) -> Result<MoB, FerricError> {
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    // Stream the AO 3-index tensor aux-blocked under FERRIC_ERI3_BUDGET_GB rather
    // than materializing the dense (naux, nbf, nbf) tensor (~37 GB at dimer/aTZ).
    let mut ao_src = ThreeIndexSource::build(
        op, obs, dfbs, ferric_core::memory::resolve_budget_bytes(None),
    )?;
    build_mo_b_from_source(&mut ao_src, &v_inv_sqrt, c, eps_full, nocc_total, frozen_core)
}

/// Build MoB from an aux-blocked AO source and V^{-1/2}.
///
/// Delegates the streamed-transform-and-dress step to the shared canonical
/// streamer [`ferric_mp2::rimp2::stream_dressed_mo_band`] (promoted from
/// `oo_rimp2.rs::compute_b_full_mo_with`) with `c_left = c_right = c_act` (the
/// full active-MO square this function has always computed) and no output-band
/// restriction. That function's peak footprint is the SAME single dressed
/// `(naux, n_act, n_act)` tensor `guard_b_full` guards (the transient row-chunk
/// MO panel it streams through is `(≤256, n_act²)`, smaller than or comparable
/// to this function's former `(naux, cblk)` column-panel temp at typical
/// scales — in both cases dwarfed by the one big buffer the guard actually
/// accounts for) — so the documented peak-memory bound is unchanged.
fn build_mo_b_from_source(
    ao_src: &mut ThreeIndexSource,
    v_inv_sqrt: &Array2<f64>,
    c: &Array2<f64>,
    eps_full: &[f64],
    nocc_total: usize,
    frozen_core: usize,
) -> Result<MoB, FerricError> {
    let nbas = ao_src.nao();
    let nmo = nbas;
    if frozen_core > nocc_total {
        return Err(FerricError::General(
            "build_full_b: frozen_core exceeds occupied count".into(),
        ));
    }
    let n_act = nmo - frozen_core;
    let n_occ_act = nocc_total - frozen_core;
    let naux = v_inv_sqrt.nrows();
    if ao_src.naux() != naux {
        return Err(FerricError::General(format!(
            "build_full_b: AO source naux {} != metric naux {naux}",
            ao_src.naux()
        )));
    }

    // Guard the (single) b_full allocation against the byte budget before we
    // allocate it. The AO source is already budget-bounded (streamed).
    guard_b_full(naux, n_act, "build_full_b")?;

    let c_act = c.slice(s![.., frozen_core..nmo]).to_owned();

    let b_flat = with_blas_threads(opt_in_blas_threads(), || {
        ferric_mp2::rimp2::stream_dressed_mo_band(ao_src, v_inv_sqrt, &c_act, &c_act, None)
    })?;
    let b_full = b_flat
        .into_shape_with_order((naux, n_act, n_act))
        .map_err(|e| FerricError::General(format!("build_full_b reshape: {e}")))?;

    let eps_act = eps_full[frozen_core..nmo].to_vec();

    Ok(MoB {
        b_full,
        v_inv_sqrt: v_inv_sqrt.clone(),
        naux,
        n_act,
        first_act: frozen_core,
        n_occ_act,
        eps_act,
    })
}

/// Build both α and β MoB in one shot, sharing the AO 3-index build and V^{-1/2}.
/// Returns (MoB_α, MoB_β).
pub fn build_full_b_both_spins(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    scf: &ScfResult,
    frozen_core: usize,
) -> Result<(MoB, MoB), FerricError> {
    if matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "build_full_b_both_spins: not applicable to Restricted results".into(),
        ));
    }
    let nelec = mol.nelec();
    let two_s = (mol.multiplicity as i32) - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;

    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    // One aux-blocked AO source, streamed twice (α then β) instead of a dense
    // (naux, nbf, nbf) tensor shared across both channels. Each channel's MoB is
    // a single dressed (naux, n_act, n_act) buffer; α and β MoBs still co-reside
    // (the caller holds both), but no separate MO intermediate is co-resident.
    // The AO source stays budget-bounded across both passes.
    let mut ao_src = ThreeIndexSource::build(
        op, obs, dfbs, ferric_core::memory::resolve_budget_bytes(None),
    )?;

    let mo_b_a = build_mo_b_from_source(
        &mut ao_src, &v_inv_sqrt, scf.mos_a(), scf.eps_a(), nocc_a, frozen_core,
    )?;
    let (mos_b, eps_b_slice) = match scf.spin {
        Spin::RestrictedOpen => (scf.mos_a(), scf.eps_a()),
        Spin::Unrestricted => (scf.mos_b(), scf.eps_b()),
        Spin::Restricted => unreachable!(),
    };
    let mo_b_b = build_mo_b_from_source(
        &mut ao_src, &v_inv_sqrt, mos_b, eps_b_slice, nocc_b, frozen_core,
    )?;
    Ok((mo_b_a, mo_b_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    // NOTE: the real-env FERRIC_BLAS_THREADS=2 identity test for the wrapped
    // BLAS sites in this file lives in tests/blas_raise_identity.rs, a
    // dedicated integration-test binary: raising the process-global OpenBLAS
    // thread count inside this parallel lib test binary races other
    // BLAS-doing tests in the same process (observed as corrupted GEMM
    // output — OpenBLAS multi-thread mode is not safe against concurrent
    // callers).

    fn water_setup() -> (PreparedBasis, PreparedBasis, Operator) {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        (obs, dfbs, Operator::coulomb())
    }

    /// The aux-blocked (spilled, multi-block) transform must reproduce the
    /// in-core single-block build exactly. Uses an explicit tiny budget for the
    /// AO source (no env dependence); the MO coefficient matrix is arbitrary —
    /// both paths consume the same C so any full-rank choice validates equality.
    #[test]
    fn blocked_build_matches_incore() {
        let (obs, dfbs, op) = water_setup();
        let nbas = obs.nbasis();
        let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();

        // Arbitrary well-conditioned C: identity + small off-diagonal ramp.
        let mut c = Array2::<f64>::eye(nbas);
        for i in 0..nbas {
            for j in 0..nbas {
                c[(i, j)] += 0.01 * ((i + 2 * j) % 7) as f64;
            }
        }
        let eps: Vec<f64> = (0..nbas).map(|k| -1.0 + 0.1 * k as f64).collect();
        let nocc_total = 5;
        let frozen_core = 1; // exercise the active-block slicing too

        // In-core: single aux block.
        let mut src_full = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX).unwrap();
        assert_eq!(src_full.n_blocks(), 1);
        let full = build_mo_b_from_source(
            &mut src_full, &v_inv_sqrt, &c, &eps, nocc_total, frozen_core,
        )
        .unwrap();

        // Spilled: ~3 aux rows per block.
        let tiny = nbas * nbas * 8 * 3;
        let mut src_tiny = ThreeIndexSource::build(op, &obs, &dfbs, tiny).unwrap();
        assert!(src_tiny.n_blocks() > 1, "expected multi-block spill");
        let blocked = build_mo_b_from_source(
            &mut src_tiny, &v_inv_sqrt, &c, &eps, nocc_total, frozen_core,
        )
        .unwrap();

        assert_eq!(full.b_full.dim(), blocked.b_full.dim());
        let maxdiff = (&full.b_full - &blocked.b_full)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(
            maxdiff <= 1e-14,
            "blocked b_full != in-core b_full, maxdiff={maxdiff:e}"
        );
        assert_eq!(full.n_act, blocked.n_act);
        assert_eq!(full.n_occ_act, blocked.n_occ_act);
        assert_eq!(full.eps_act, blocked.eps_act);
    }

    /// The budget guard must reject a b_full that exceeds the byte budget and
    /// pass one that fits, with the GB numbers in the message.
    #[test]
    fn b_full_budget_guard() {
        // 100 aux × 50² act × 8 B = 2 MB.
        let need = b_full_bytes(100, 50);
        assert_eq!(need, 100 * 50 * 50 * 8);
        assert!(guard_b_full_at(need, 100, 50, "test").is_ok());
        let err = guard_b_full_at(need - 1, 100, 50, "test").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("FERRIC_ERI3_BUDGET_GB") && msg.contains("100×50×50"),
            "unexpected guard message: {msg}"
        );
    }
}
