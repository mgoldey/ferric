//! Full-MO dressed RI tensor B̃^P_{mn} = V^{-1/2}_{PQ} (Q | mn),
//! shape (naux, n_active_mo, n_active_mo).
//!
//! "Active" = MO indices `frozen_core..nmo`. The frozen-core indices are
//! excluded from the (m,n) pair list to match the PDEP intermediates that
//! were built with the same frozen_core.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::{ScfResult, Spin};
use ndarray::{s, Array2, Array3};
use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};

/// Dressed RI integrals over the full active-MO square.
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

fn cholesky_inverse_sqrt(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = v.nrows();
    let l = v
        .cholesky(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("V cholesky failed: {e}")))?;
    let eye = Array2::<f64>::eye(n);
    let v_inv_sqrt = l
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &eye)
        .map_err(|e| FerricError::General(format!("triangular solve failed: {e}")))?;
    Ok(v_inv_sqrt)
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
    let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;
    build_full_b_with_ao(&eri3_ao, &v_inv_sqrt, c, eps_full, nocc_total, frozen_core, obs.nbasis())
}

/// Build MoB given precomputed AO 3-index tensor and V^{-1/2}.
/// Used to avoid duplicating the 2c/3c builds across α and β channels.
fn build_full_b_with_ao(
    eri3_ao: &Array3<f64>,
    v_inv_sqrt: &Array2<f64>,
    c: &Array2<f64>,
    eps_full: &[f64],
    nocc_total: usize,
    frozen_core: usize,
    nbas: usize,
) -> Result<MoB, FerricError> {
    let nmo = nbas;
    if frozen_core > nocc_total {
        return Err(FerricError::General(
            "build_full_b: frozen_core exceeds occupied count".into(),
        ));
    }
    let n_act = nmo - frozen_core;
    let n_occ_act = nocc_total - frozen_core;
    let naux = v_inv_sqrt.nrows();

    let c_act = c.slice(s![.., frozen_core..nmo]).to_owned();

    let eri3_mm = ferric_mp2::mo_transform::transform_3center_ov(eri3_ao, &c_act, &c_act);
    let eri3_flat = eri3_mm
        .into_shape_with_order((naux, n_act * n_act))
        .map_err(|e| FerricError::General(format!("reshape failed: {e}")))?;
    let b_flat = v_inv_sqrt.dot(&eri3_flat);
    let b_full = b_flat
        .into_shape_with_order((naux, n_act, n_act))
        .map_err(|e| FerricError::General(format!("reshape failed: {e}")))?;

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
    let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;
    let nbas = obs.nbasis();

    let mo_b_a = build_full_b_with_ao(
        &eri3_ao, &v_inv_sqrt, scf.mos_a(), scf.eps_a(), nocc_a, frozen_core, nbas,
    )?;
    let (mos_b, eps_b_slice) = match scf.spin {
        Spin::RestrictedOpen => (scf.mos_a(), scf.eps_a()),
        Spin::Unrestricted => (scf.mos_b(), scf.eps_b()),
        Spin::Restricted => unreachable!(),
    };
    let mo_b_b = build_full_b_with_ao(
        &eri3_ao, &v_inv_sqrt, mos_b, eps_b_slice, nocc_b, frozen_core, nbas,
    )?;
    Ok((mo_b_a, mo_b_b))
}
