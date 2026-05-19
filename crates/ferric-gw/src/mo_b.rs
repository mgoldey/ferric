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
use ferric_scf::ScfResult;
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
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nmo = nbas;
    if frozen_core > nocc_total {
        return Err(FerricError::General(
            "build_full_b: frozen_core exceeds occupied count".into(),
        ));
    }
    let n_act = nmo - frozen_core;
    let n_occ_act = nocc_total - frozen_core;
    let naux = dfbs.nbasis();

    let c = rhf.mos_r();
    let c_act = c.slice(s![.., frozen_core..nmo]).to_owned();

    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;

    // (P | mn) over the active-active square.
    let eri3_mm = ferric_mp2::mo_transform::transform_3center_ov(&eri3_ao, &c_act, &c_act);
    // Contract V^{-1/2} on the auxiliary axis. eri3_mm: (naux, n_act, n_act).
    // Flatten the (n_act, n_act) inner pair into a single column index.
    let eri3_flat = eri3_mm
        .into_shape_with_order((naux, n_act * n_act))
        .map_err(|e| FerricError::General(format!("reshape failed: {e}")))?;
    let b_flat = v_inv_sqrt.dot(&eri3_flat);
    let b_full = b_flat
        .into_shape_with_order((naux, n_act, n_act))
        .map_err(|e| FerricError::General(format!("reshape failed: {e}")))?;

    let eps_full = rhf.eps_r();
    let eps_act = eps_full[frozen_core..nmo].to_vec();

    Ok(MoB {
        b_full,
        v_inv_sqrt,
        naux,
        n_act,
        first_act: frozen_core,
        n_occ_act,
        eps_act,
    })
}
