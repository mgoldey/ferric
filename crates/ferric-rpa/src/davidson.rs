//! Davidson subspace eigensolver for PDEP dielectric eigenpotentials.

use ferric_core::FerricError;
use ferric_integrals::blas_threads::with_blas_threads;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, QR, UPLO};

pub struct DavidsonResult {
    /// Converged eigenvalues λ_α, sorted descending (most significant first).
    pub eigenvalues: Vec<f64>,
    /// Eigenpotentials, shape (naux, n_converged). Columns = V_α.
    pub eigenvectors: Array2<f64>,
    /// Whether the returned Ritz pairs met `conv_thresh` (Davidson path) or
    /// the caller's residual tolerance (Lanczos path, when a
    /// [`crate::lanczos::LanczosResult`] is threaded into a `DavidsonResult`
    /// at the `lib.rs` call sites). `run_davidson_seeded_impl` only ever
    /// returns `Ok` after its own residual check passes — on `max_iter`
    /// exhaustion it hard-errors via
    /// `Err(FerricError::General("Davidson did not converge"))` rather than
    /// returning an unconverged `Ok` — so every genuine Davidson-solver
    /// construction site sets this `true`.
    pub converged: bool,
}

/// Run Davidson at ω=0 to find the leading eigenpotentials of ε̃(0).
///
/// `dielectric_fn(v_mat, omega)` returns the projected dielectric matrix
/// ε̃_αβ(iω) for trial vectors V (columns of v_mat).
///
/// `m0`: initial subspace dimension (= naux for RI-seed)
/// `n_desired`: number of eigenpairs to extract
pub fn run_davidson_static<F>(
    m0: usize,
    dielectric_fn: F,
    conv_thresh: f64,
    max_vecs: usize,
    n_desired: usize,
    find_lowest: bool,
) -> Result<DavidsonResult, FerricError>
where
    F: Fn(&Array2<f64>, f64) -> Array2<f64>,
{
    // Seed: identity subspace (unit vectors in aux space)
    let seed = Array2::eye(m0);
    run_davidson_seeded(seed, dielectric_fn, conv_thresh, max_vecs, n_desired, find_lowest)
}

/// Run Davidson with an explicit seed matrix.
///
/// `seed`: initial trial subspace, shape (naux, n_seed). Columns are orthonormal
///   trial vectors in the dressed aux basis. Davidson will grow this subspace
///   as needed until `n_desired` eigenpairs converge.
///
/// `dielectric_fn(v_mat, omega)` returns the projected dielectric matrix
/// ε̃_αβ(iω) for trial vectors V (columns of v_mat).
///
/// `n_desired`: number of eigenpairs to extract
///
/// BLAS threading: the whole iteration (dielectric-matrix matvec via the
/// caller-supplied `dielectric_fn`, `eigh`, the Ritz-vector GEMMs, and QR
/// re-orthonormalization) runs under one `with_blas_threads` scope — see
/// [`run_davidson_seeded_impl`]. Default 1 (unchanged behavior); opt-in via
/// `FERRIC_BLAS_THREADS`/`FERRIC_LANCZOS_BLAS_THREADS` closes the
/// Davidson/Lanczos gating asymmetry (Lanczos has had this since the
/// FERRIC_LANCZOS_BLAS_THREADS precedent; Davidson did not).
pub fn run_davidson_seeded<F>(
    seed: Array2<f64>,
    dielectric_fn: F,
    conv_thresh: f64,
    max_vecs: usize,
    n_desired: usize,
    find_lowest: bool,
) -> Result<DavidsonResult, FerricError>
where
    F: Fn(&Array2<f64>, f64) -> Array2<f64>,
{
    with_blas_threads(davidson_blas_threads(), || {
        run_davidson_seeded_impl(seed, dielectric_fn, conv_thresh, max_vecs, n_desired, find_lowest)
    })
}

/// BLAS thread count for the Davidson solve. Same resolver as Lanczos
/// (`lanczos::solver_blas_threads_with` — see that doc comment and
/// `blas_threads::opt_in_blas_threads` for the precedence and hazard model):
/// `FERRIC_LANCZOS_BLAS_THREADS` > `FERRIC_BLAS_THREADS` > 1, plus the
/// rayon-worker runtime guard. Sharing the Lanczos var (not a separate
/// `FERRIC_DAVIDSON_BLAS_THREADS`) is deliberate: Davidson and Lanczos are
/// alternative eigensolvers for the same call site (`config.eigensolver` in
/// lib.rs) and a caller tuning one expects the other to behave the same way
/// when swapped in.
fn davidson_blas_threads() -> usize {
    crate::lanczos::solver_blas_threads_with(|k| std::env::var(k).ok())
}

/// Serial body for [`run_davidson_seeded`]; BLAS threading managed by caller.
/// Call-path proof: `run_davidson_seeded`/`run_davidson_static` are called
/// from `run_pdep_rpa`'s body (lib.rs:369,:408,:423) and `run_u_pdep_rpa`'s
/// body (lib.rs:649) — both plain function bodies, never inside a
/// `par_iter`. The `dielectric_fn` closures passed in from lib.rs
/// (sternheimer/sternheimer_sparse/laplace_chi0) are BLAS-only with no rayon
/// region, matching the same requirement Lanczos's `matvec` already carries.
fn run_davidson_seeded_impl<F>(
    seed: Array2<f64>,
    dielectric_fn: F,
    conv_thresh: f64,
    max_vecs: usize,
    n_desired: usize,
    find_lowest: bool,
) -> Result<DavidsonResult, FerricError>
where
    F: Fn(&Array2<f64>, f64) -> Array2<f64>,
{
    // Orthonormalize the seed to get the initial subspace
    let mut v_mat = if seed.ncols() <= seed.nrows() {
        qr_orthonormalize(seed)?
    } else {
        seed
    };

    let max_iter = 200;
    for _iter in 0..max_iter {
        let m = v_mat.ncols();

        // Form projected dielectric
        let eps_proj = dielectric_fn(&v_mat, 0.0);

        // Diagonalize (symmetric)
        let (evals, evecs) = eps_proj.eigh(UPLO::Upper)
            .map_err(|e| FerricError::General(format!("Davidson diagonalization failed: {e}")))?;

        // Ritz vectors in original space: V @ evecs (naux, m)
        let ritz = v_mat.dot(&evecs);

        // Compute residuals for the n_desired most-relevant eigenvalues
        // (largest by default; smallest when find_lowest=true).
        let mut max_resid = 0.0f64;
        let mut new_vecs: Vec<Array1<f64>> = Vec::new();
        let m_check = m.min(n_desired + 2);
        let k_iter: Box<dyn Iterator<Item = usize>> = if find_lowest {
            Box::new(0..m_check)
        } else {
            Box::new((m.saturating_sub(m_check))..m)
        };
        for k in k_iter {
            let lk = evals[k];
            let vk = ritz.column(k);

            // residual = ε̃ @ vk − lk * vk
            let eps_vk = v_mat.dot(&eps_proj.dot(&evecs.column(k)));
            let resid: Array1<f64> = &eps_vk - &(lk * &vk);
            let resid_norm = resid.dot(&resid).sqrt();
            max_resid = max_resid.max(resid_norm);

            if resid_norm > conv_thresh {
                // Davidson preconditioner: divide by shift (ε̃ ≥ 1)
                let denom = (lk - 1.0).abs().max(0.1);
                let t: Array1<f64> = resid.mapv(|x| x / denom);
                new_vecs.push(t);
            }
        }

        // All checked eigenpairs have converged.
        if max_resid < conv_thresh || new_vecs.is_empty() {
            if m >= n_desired {
                // Full convergence: we have enough eigenpairs.
                let n_keep = n_desired.min(m);
                let (eigenvalues, eigenvectors) = if find_lowest {
                    // eigh returns ascending order, so first n_keep are smallest.
                    let eigenvalues: Vec<f64> = evals.slice(ndarray::s![..n_keep]).iter().copied().collect();
                    let eigenvectors = ritz.slice(ndarray::s![.., ..n_keep]).to_owned();
                    (eigenvalues, eigenvectors)
                } else {
                    let start = m - n_keep;
                    let eigenvalues: Vec<f64> = evals.slice(ndarray::s![start..]).iter().copied().rev().collect();
                    let eigenvectors = ritz.slice(ndarray::s![.., start..]).to_owned();
                    let eigenvectors = eigenvectors.slice(ndarray::s![.., ..;-1]).to_owned();
                    (eigenvalues, eigenvectors)
                };
                return Ok(DavidsonResult { eigenvalues, eigenvectors, converged: true });
            }

            // m < n_desired: current Ritz vectors all converged, but subspace is too small.
            // This happens when the seed spans only a subset of the eigenpairs.
            // Expand with unit vectors orthogonal to the current subspace.
            let naux = v_mat.nrows();
            let budget = max_vecs.saturating_sub(m);
            if budget == 0 || m >= naux {
                // No room to grow: return what we have (truncation step will handle it).
                let (eigenvalues, eigenvectors) = if find_lowest {
                    let eigenvalues: Vec<f64> = evals.iter().copied().collect();
                    let eigenvectors = ritz.clone();
                    (eigenvalues, eigenvectors)
                } else {
                    let eigenvalues: Vec<f64> = evals.iter().copied().rev().collect();
                    let eigenvectors = ritz.slice(ndarray::s![.., ..;-1]).to_owned();
                    (eigenvalues, eigenvectors)
                };
                return Ok(DavidsonResult { eigenvalues, eigenvectors, converged: true });
            }
            // Add a batch of orthogonal unit vectors to bootstrap coverage of the missing space.
            let n_to_add = (n_desired - m).min(budget).min(naux - m);
            let mut expanded = Array2::zeros((naux, m + n_to_add));
            expanded.slice_mut(ndarray::s![.., ..m]).assign(&v_mat);
            let mut added = 0;
            for ei in 0..naux {
                if added >= n_to_add { break; }
                // Gram-Schmidt: project unit vector against all current columns
                let mut e = Array1::zeros(naux);
                e[ei] = 1.0;
                for col_idx in 0..(m + added) {
                    let col = expanded.column(col_idx);
                    let proj = col.dot(&e);
                    let col_owned = col.to_owned();
                    e = e - proj * &col_owned;
                }
                let norm = e.dot(&e).sqrt();
                if norm > 1e-10 {
                    e.mapv_inplace(|x| x / norm);
                    expanded.column_mut(m + added).assign(&e);
                    added += 1;
                }
            }
            if added > 0 {
                v_mat = expanded.slice(ndarray::s![.., ..(m + added)]).to_owned();
            }
            // No need to re-QR here since we built ortho columns incrementally.
            continue;
        }

        if v_mat.ncols() + new_vecs.len() > max_vecs {
            // Restart: keep only current Ritz vectors
            let n_keep = n_desired.min(m);
            let start = m - n_keep;
            v_mat = ritz.slice(ndarray::s![.., start..]).to_owned();
        } else {
            // Expand subspace
            let naux = v_mat.nrows();
            let n_new = new_vecs.len();
            let mut expanded = Array2::zeros((naux, v_mat.ncols() + n_new));
            expanded.slice_mut(ndarray::s![.., ..v_mat.ncols()]).assign(&v_mat);
            for (j, t) in new_vecs.iter().enumerate() {
                expanded.slice_mut(ndarray::s![.., v_mat.ncols() + j]).assign(t);
            }
            v_mat = expanded;
        }

        // Orthonormalize via QR
        v_mat = qr_orthonormalize(v_mat)?;
    }

    Err(FerricError::General("Davidson did not converge".into()))
}

/// Orthonormalize columns of mat via QR decomposition.
fn qr_orthonormalize(mat: Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let (q, _r) = mat.qr()
        .map_err(|e| FerricError::General(format!("QR failed: {e}")))?;
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Resolver precedence/guard tests live in lanczos.rs (the shared
    // solver_blas_threads_with, injection-based — no env mutation). The
    // real-env FERRIC_BLAS_THREADS=2 identity test lives in
    // tests/blas_raise_identity.rs (its own process): raising the
    // process-global OpenBLAS thread count inside this parallel lib test
    // binary races other BLAS-doing tests (observed corruption, not
    // hypothetical).

    #[test]
    fn davidson_recovers_known_eigenvalues() {
        // 2D dielectric: ε̃ = [[2.0, 0.5], [0.5, 3.0]]
        // Eigenvalues: ~1.809 and ~3.191
        // We'll test that Davidson finds them given the explicit matrix.
        use ndarray::array;

        let result = run_davidson_static(
            2,      // naux = m0
            |v_mat: &Array2<f64>, _omega: f64| -> Array2<f64> {
                let fixed = array![[2.0f64, 0.5], [0.5, 3.0]];
                v_mat.t().dot(&fixed.dot(v_mat))
            },
            1e-6,  // conv_thresh
            20,    // max_vecs
            2,     // n_desired
            false, // find_lowest
        ).unwrap();

        let mut evals = result.eigenvalues.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // For 2x2 [[2,0.5],[0.5,3]]: trace=5, det=6-0.25=5.75
        // λ = (5 ± sqrt(2)) / 2
        let expected_lo = (5.0 - 2.0f64.sqrt()) / 2.0;
        let expected_hi = (5.0 + 2.0f64.sqrt()) / 2.0;
        assert!((evals[0] - expected_lo).abs() < 1e-4,
            "λ_0={} expected {}", evals[0], expected_lo);
        assert!((evals[1] - expected_hi).abs() < 1e-4,
            "λ_1={} expected {}", evals[1], expected_hi);
    }

    #[test]
    fn davidson_find_lowest_returns_smallest_eigenpair() {
        use ndarray::array;
        let result = run_davidson_seeded(
            Array2::eye(2),
            |v_mat: &Array2<f64>, _omega: f64| -> Array2<f64> {
                let fixed = array![[2.0f64, 0.5], [0.5, 3.0]];
                v_mat.t().dot(&fixed.dot(v_mat))
            },
            1e-8,
            20,
            1,
            true,
        ).unwrap();
        let expected_lo = (5.0 - 2.0f64.sqrt()) / 2.0;
        assert_eq!(result.eigenvalues.len(), 1);
        assert!(
            (result.eigenvalues[0] - expected_lo).abs() < 1e-6,
            "find_lowest returned {} expected {}",
            result.eigenvalues[0], expected_lo,
        );
    }

    /// A genuine Davidson solve only ever returns `Ok` after its own residual
    /// check passes (`run_davidson_seeded_impl` hard-errors via
    /// `Err(FerricError::General("Davidson did not converge"))` on `max_iter`
    /// exhaustion instead of returning an unconverged `Ok`) — so `converged`
    /// must be `true` at both `Ok(DavidsonResult { .. })` sites. Regression
    /// guard for the TD-CONV field-threading fix: previously `DavidsonResult`
    /// had no `converged` field at all and the flag was dropped everywhere.
    #[test]
    fn davidson_result_reports_converged_true() {
        use ndarray::array;
        let result = run_davidson_static(
            2,
            |v_mat: &Array2<f64>, _omega: f64| -> Array2<f64> {
                let fixed = array![[2.0f64, 0.5], [0.5, 3.0]];
                v_mat.t().dot(&fixed.dot(v_mat))
            },
            1e-6,
            20,
            2,
            false,
        ).unwrap();
        assert!(result.converged, "a returned Ok(DavidsonResult) must always be converged");
    }

    /// TD-CONV regression: reproduces the exact `davidson::DavidsonResult {
    /// eigenvalues: lz.eigenvalues, eigenvectors: lz.eigenvectors, converged:
    /// lz.converged }` wiring used at the three Lanczos call sites in
    /// `lib.rs` (`run_pdep_rpa_from_intermediates`'s Boys-screened Lanczos arm
    /// and both `run_lanczos_full_rank` sites). Before the fix, `converged`
    /// was silently dropped when a `LanczosResult` was folded into a
    /// `DavidsonResult`; this test drives `run_lanczos_seeded` into the same
    /// deliberately-unconverged regime as
    /// `lanczos::tests::seeded_lanczos_max_iter_one_reports_unconverged`
    /// (narrow non-spanning seed + `max_iter=1` + an unreachably tight
    /// `conv_thresh`) and asserts the `false` flag survives the
    /// `DavidsonResult` wrapping unchanged.
    #[test]
    fn davidson_result_carries_unconverged_lanczos_flag() {
        use crate::lanczos::run_lanczos_seeded;

        let naux = 40;
        let nov = 90;
        // Dense dielectric-shaped operator ε̃ = I + B diag(s²) Bᵀ (SPD, λ ≥ 1),
        // same construction as lanczos.rs's make_dielectric_op (duplicated
        // here since that helper is private to lanczos::tests).
        let mut state = 11u64.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / u32::MAX as f64) - 0.5
        };
        let mut b = Array2::<f64>::zeros((naux, nov));
        for v in b.iter_mut() {
            *v = next();
        }
        let s2: Vec<f64> = (0..nov).map(|k| 0.3 + (k as f64 % 7.0) * 0.1).collect();
        let b_mv = b.clone();
        let s2_mv = s2.clone();
        let matvec = move |v: &Array2<f64>| -> Array2<f64> {
            let mut y = b_mv.t().dot(v);
            for k in 0..y.nrows() {
                let sk = s2_mv[k];
                let row = y.row(k).to_owned() * sk;
                y.row_mut(k).assign(&row);
            }
            let mut out = v.to_owned();
            out += &b_mv.dot(&y);
            out
        };

        // Narrow seed block (4 of 40 columns): one Lanczos step cannot span
        // the ambient space, and max_iter=1 with an unreachable conv_thresh
        // forces the outer-iteration budget to exhaust.
        let mut seed = Array2::<f64>::zeros((naux, 4));
        for j in 0..4 {
            for i in 0..naux {
                seed[(i, j)] = ((i + j * 13) as f64).cos();
            }
        }
        let lz = run_lanczos_seeded(seed, matvec, naux, 1, 1e-14).unwrap();
        assert!(!lz.converged, "test setup must reproduce an unconverged Lanczos result");

        // Exact lib.rs wiring pattern under test.
        let dr = DavidsonResult {
            eigenvalues: lz.eigenvalues,
            eigenvectors: lz.eigenvectors,
            converged: lz.converged,
        };
        assert!(
            !dr.converged,
            "DavidsonResult must carry converged=false through from the Lanczos result, \
             not silently default to true"
        );
    }
}
