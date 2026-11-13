//! RESEARCH SPIKE (feature `spike-cholesky-seed`, off in every shipped build):
//! pivoted-Cholesky localized occupied orbitals as an alternative to
//! Foster-Boys ([`crate::boys_localize`]) for seeding the sparse PDEP-RPA
//! screening in [`crate::screen::build_screened_bov`].
//!
//! The record, the measured tables and the verdict live in
//! `reference/spikes/cholesky-vs-boys-pdep-seed.md`. Nothing here is wired to
//! the CLI, Python or `Chi0Sparsity::parse_config_str`.
//!
//! # Construction (Aquilante et al., Cholesky localization)
//!
//! For the active-occupied density `D = C_occ C_occᵀ` (AO basis, positive
//! semidefinite, rank `nocc_active`), a diagonally pivoted Cholesky
//! factorization `P D Pᵀ = L Lᵀ` stopped at the numerical rank gives
//! `nocc_active` columns `c_loc` with `c_loc c_locᵀ = D`. The pivot order
//! (largest remaining diagonal first) makes each column short-ranged.
//!
//! Two full-rank factorizations of the same matrix differ by an orthogonal
//! transformation, so `c_loc = C_occ U` with `U` orthogonal: the Cholesky
//! orbitals span the occupied space and are orthonormal in the AO overlap
//! metric (`c_locᵀ S c_loc = I`), although not in the Euclidean coefficient
//! sense (`c_locᵀ c_loc ≠ I`). `tests/spike_cholesky_seed.rs` checks both.
//!
//! The 2026-07-16 spike used `faer`'s `llt_pivoting` for the factorization.
//! This port implements the same diagonal-pivoting outer-product algorithm in
//! ndarray (stopping tolerance `n·ε·max diag(D)`, the LAPACK `dpstrf` default)
//! so the spike adds no dependency.
//!
//! Closed-shell only (uses `mos_r()`), mirroring `boys_localize.rs`.

use crate::screen::{build_screened_bov, ScreenedBov};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{dipole, overlap};
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ndarray::{s, Array2};

/// Cholesky-localized active occupied block.
#[derive(Debug, Clone)]
pub struct CholeskyOccupied {
    /// Localized occupied coefficients, shape (nbas, nocc_active).
    pub c_loc: Array2<f64>,
    /// Per-orbital centroid ⟨k|r|k⟩/⟨k|S|k⟩, `[nocc_active][3]` (Bohr).
    /// Diagnostic only: `build_screened_bov` recomputes centroids from its
    /// semicanonicalized orbitals.
    pub centroids: Vec<[f64; 3]>,
    /// AO indices in pivot order (first `rank` entries are the pivots used).
    pub pivots: Vec<usize>,
    /// Numerical rank detected by the pivoted factorization.
    pub rank: usize,
}

/// Diagonally pivoted Cholesky of a symmetric positive-semidefinite matrix.
///
/// Returns `(l, pivots)` where `l` has shape `(n, rank)` in the ORIGINAL row
/// order (so `l lᵀ ≈ a` with no permutation to undo) and `pivots[k]` is the
/// row chosen at step `k`. Stops when the largest remaining diagonal is
/// `<= tol`.
pub fn pivoted_cholesky(a: &Array2<f64>, tol: f64) -> (Array2<f64>, Vec<usize>) {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    let mut d: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
    let mut remaining = vec![true; n];
    let mut pivots = Vec::with_capacity(n);
    for k in 0..n {
        let Some(p) = (0..n)
            .filter(|&i| remaining[i])
            .max_by(|&i, &j| d[i].total_cmp(&d[j]))
        else {
            break;
        };
        if d[p] <= tol {
            break;
        }
        let lpp = d[p].sqrt();
        // Column k: (A[:, p] - L[:, :k] L[p, :k]) / L[p, k], zero on pivoted rows.
        let mut col = a.column(p).to_owned();
        if k > 0 {
            let lp = l.slice(s![p, ..k]).to_owned();
            col -= &l.slice(s![.., ..k]).dot(&lp);
        }
        for i in 0..n {
            l[(i, k)] = if i == p {
                lpp
            } else if remaining[i] {
                col[i] / lpp
            } else {
                0.0
            };
        }
        remaining[p] = false;
        pivots.push(p);
        for i in 0..n {
            if remaining[i] {
                d[i] -= l[(i, k)] * l[(i, k)];
            }
        }
    }
    let rank = pivots.len();
    (l.slice(s![.., ..rank]).to_owned(), pivots)
}

/// Cholesky-localize the active occupied block `[first_occ, first_occ +
/// nocc_active)` of an RHF result (same slicing as `boys_localize_occupied`).
///
/// Errors if the detected rank differs from `nocc_active` (a non-idempotent or
/// ill-conditioned density), rather than silently truncating or padding.
pub fn cholesky_localize_occupied(
    rhf: &ScfResult,
    obs: &PreparedBasis,
    first_occ: usize,
    nocc_active: usize,
) -> Result<CholeskyOccupied, FerricError> {
    let c_occ = rhf
        .mos_r()
        .slice(s![.., first_occ..first_occ + nocc_active])
        .to_owned();
    let n = c_occ.nrows();
    let d = c_occ.dot(&c_occ.t());
    let max_diag = (0..n).map(|i| d[(i, i)]).fold(0.0_f64, f64::max);
    let tol = n as f64 * f64::EPSILON * max_diag;
    let (c_loc, pivots) = pivoted_cholesky(&d, tol);
    let rank = c_loc.ncols();
    if rank != nocc_active {
        return Err(FerricError::General(format!(
            "cholesky localization rank mismatch: detected rank {rank} != nocc_active \
             {nocc_active} (non-idempotent / ill-conditioned occupied density)"
        )));
    }

    let s_ao = overlap(obs);
    let dip = dipole(obs, [0.0, 0.0, 0.0])?;
    let centroids = (0..rank)
        .map(|k| {
            let ck = c_loc.column(k);
            let norm = ck.dot(&s_ao.dot(&ck)).max(1e-300);
            let mut r = [0.0f64; 3];
            for (axis, m) in dip.iter().enumerate() {
                r[axis] = ck.dot(&m.dot(&ck)) / norm;
            }
            r
        })
        .collect();

    Ok(CholeskyOccupied {
        c_loc,
        centroids,
        pivots,
        rank,
    })
}

/// Cholesky-seeded analogue of [`crate::screen::build_screened_bov_boys`]:
/// the screening and tile build are the identical `build_screened_bov` path,
/// so the localization method is the only variable.
#[allow(clippy::too_many_arguments)]
pub fn build_screened_bov_cholesky(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
    thresh: f64,
    dist_cutoff: f64,
) -> Result<(ScreenedBov, CholeskyOccupied), FerricError> {
    let nocc_total = mol.nelec() as usize / 2;
    let nocc_active = ferric_mp2::rimp2::active_occ(nocc_total, frozen_core)?;
    let chol = cholesky_localize_occupied(rhf, obs, frozen_core, nocc_active)?;
    let screened = build_screened_bov(
        mol,
        obs,
        dfbs,
        op,
        rhf,
        frozen_core,
        nocc_active,
        nocc_total,
        &chol.c_loc,
        chol.centroids.clone(),
        thresh,
        dist_cutoff,
    )?;
    Ok((screened, chol))
}
