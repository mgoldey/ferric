//! Block Lanczos eigensolver for the PDEP dielectric matrix.
//!
//! Alternative to Davidson (`davidson.rs`): builds a Krylov subspace by
//! repeated block matvecs A·V, with full reorthogonalization. Memory footprint
//! is the stacked Krylov basis Q (naux × (k+1)·block_size) plus the small
//! block-tridiagonal T; no growing projected dielectric is retained.
//!
//! Designed to plug into the same matvec used by Davidson: callers pass a
//! closure `|V| → A·V` (typically `dielectric_matrix` applied to V, returning
//! ε̃·V, *not* the V^T·ε̃·V projection that Davidson asks for).

use ferric_core::FerricError;
use ferric_integrals::blas_threads::with_blas_threads;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, QR, UPLO};

/// Result of a block-Lanczos run.
pub struct LanczosResult {
    /// Converged eigenvalues, sorted by `|λ − 1|` descending (most significant
    /// PDEP modes first), matching the convention used downstream of Davidson.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors in the original space, shape `(naux, n_converged)`.
    pub eigenvectors: Array2<f64>,
}

/// QR-orthonormalize columns; returns Q with the same shape as the input when
/// it is "tall" (rows ≥ cols), else returns the input unchanged.
fn qr_orthonormalize(mat: Array2<f64>) -> Result<Array2<f64>, FerricError> {
    if mat.ncols() > mat.nrows() {
        return Ok(mat);
    }
    let (q, _r) = mat
        .qr()
        .map_err(|e| FerricError::General(format!("Lanczos QR failed: {e}")))?;
    Ok(q)
}

/// BLAS thread count for the Lanczos solve.
///
/// Defaults to **1** (deterministic, safe). Raising OpenBLAS above 1 thread for
/// the Lanczos `eigh`/QR is opt-in via `FERRIC_LANCZOS_BLAS_THREADS`, for two
/// reasons proven during the perf-integration verification:
///
///  1. **Stack overflow.** A multi-threaded OpenBLAS `eigh` on a large
///     block-tridiagonal T (or QR of a naux-wide Krylov block) runs on worker
///     stacks that overflow when the solve is large and/or `run_pdep_rpa` is
///     itself invoked concurrently (the test harness runs tests in parallel;
///     see the openblas-rayon-dgetrf-crash memory and blas_threads.rs). The
///     aug-cc-pV{D,T}Z PDEP-RPA tests abort with `stack overflow` when this
///     defaults to `available_parallelism()`; they pass at 1.
///  2. **Reproducibility.** Multi-threaded OpenBLAS GEMM/eigh changes the
///     reduction order run-to-run, which the crate's equivalence tests
///     (screened-vs-dense at thresh=0) are built to hold at a tight tolerance.
///
/// An explicit `FERRIC_LANCZOS_BLAS_THREADS=N` override wins (clamped ≥ 1) for
/// callers who want the wide-GEMM speedup and manage rayon/stack sizing
/// themselves. Such a caller must ensure the Lanczos solve is not running
/// inside a rayon worker, where nested OpenBLAS threads are the documented
/// segfault/oversubscription mode.
fn lanczos_blas_threads() -> usize {
    if let Ok(v) = std::env::var("FERRIC_LANCZOS_BLAS_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n.max(1);
        }
    }
    // Deterministic, stack-safe default. Speed is opt-in via the env override.
    1
}

/// Block Lanczos with full reorthogonalization.
///
/// `seed` is the initial block (naux × block_size); columns are QR'd to form
/// the first Krylov block V_0. The matvec closure must return A·V for the
/// supplied (naux × m) block, where A is the symmetric operator whose
/// eigenpairs we seek (here: the projected dielectric ε̃ at ω=0).
///
/// On convergence, returns up to `n_desired` Ritz pairs ordered by `|λ − 1|`
/// descending. If the residuals never drop below `conv_thresh` within
/// `max_iter` outer block iterations, returns the best Ritz pairs from the
/// final iteration without error (matching Davidson's "soft" fall-through
/// pattern when subspace expansion is exhausted).
///
/// The block matvec, full reorthogonalization (Qᵀ·W, Q·proj), QR, and Ritz
/// assembly are all naux-wide GEMMs with no rayon region active anywhere on
/// this call path (every dielectric_apply variant is BLAS-only), so this is
/// the one place BLAS threads are temporarily raised; the scoped guard
/// restores the prior count on exit. Because the raise covers the matvec
/// closure too, `matvec` must NOT enter a rayon parallel region — if a future
/// caller needs that, it must set `FERRIC_LANCZOS_BLAS_THREADS=1`.
pub fn run_lanczos_seeded<F>(
    seed: Array2<f64>,
    matvec: F,
    n_desired: usize,
    max_iter: usize,
    conv_thresh: f64,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    with_blas_threads(lanczos_blas_threads(), || {
        lanczos_iterations(seed, matvec, n_desired, max_iter, conv_thresh)
    })
}

/// Serial Lanczos iteration body; BLAS threading is managed by the caller
/// (`run_lanczos_seeded`).
fn lanczos_iterations<F>(
    seed: Array2<f64>,
    matvec: F,
    n_desired: usize,
    max_iter: usize,
    conv_thresh: f64,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    let naux = seed.nrows();
    if n_desired == 0 {
        return Ok(LanczosResult {
            eigenvalues: Vec::new(),
            eigenvectors: Array2::zeros((naux, 0)),
        });
    }

    // V_0: QR-orthonormalize the seed block.
    let v0 = qr_orthonormalize(seed)?;
    let block_size = v0.ncols();
    if block_size == 0 {
        return Ok(LanczosResult {
            eigenvalues: Vec::new(),
            eigenvectors: Array2::zeros((naux, 0)),
        });
    }

    // Stacked basis Q = [V_0 | V_1 | ... | V_k]. We grow this column-by-block.
    let mut q_basis: Array2<f64> = v0.clone();

    // Block tridiagonal entries. We store them as a flat Vec of blocks
    // (block_size × block_size) for α and β.
    // T has the structure (block-tridiag):
    //   diag blocks: α_0, α_1, ..., α_k
    //   off-diag:    β_0, β_1, ..., β_{k-1}    (β_j connects V_j and V_{j+1})
    let mut alphas: Vec<Array2<f64>> = Vec::new();
    let mut betas: Vec<Array2<f64>> = Vec::new();

    // Most recent V_k; previous block V_{k-1} (for the three-term recurrence).
    let mut v_curr: Array2<f64> = v0;
    let mut v_prev: Option<Array2<f64>> = None;
    // Most recent β block (β_{k-1}) used in the three-term recurrence.
    let mut beta_prev: Option<Array2<f64>> = None;

    // Cap the Krylov dimension so we don't exceed the ambient space.
    let max_krylov = (max_iter + 1).saturating_mul(block_size).min(naux);
    let outer_iters = (max_krylov / block_size).max(1);

    // Track best result across iterations to return if convergence isn't reached.
    let mut last_result: Option<LanczosResult> = None;

    for k in 0..outer_iters {
        // W = A · V_k
        let mut w: Array2<f64> = matvec(&v_curr);

        // α_k = V_k^T W   (block_size × block_size)
        let alpha_k: Array2<f64> = v_curr.t().dot(&w);
        // Use the UNSYMMETRIZED α_k for projection so V_k^T W_new = 0 exactly.
        // Store its symmetrization into T for the eigenproblem.
        let alpha_k_sym: Array2<f64> = 0.5 * (&alpha_k + &alpha_k.t());

        // W ← W − V_k α_k − V_{k-1} β_{k-1}^T
        w = &w - &v_curr.dot(&alpha_k);
        if let (Some(vp), Some(bp)) = (v_prev.as_ref(), beta_prev.as_ref()) {
            // β_{k-1} is (block_size × block_size) upper-triangular; we use β^T.
            w = &w - &vp.dot(&bp.t());
        }

        // Full reorthogonalization against the entire stacked basis Q.
        //   W ← W − Q (Q^T W)
        // Apply twice for numerical safety ("twice is enough").
        for _pass in 0..2 {
            let proj: Array2<f64> = q_basis.t().dot(&w);
            w = &w - &q_basis.dot(&proj);
        }

        alphas.push(alpha_k_sym);

        // QR decompose W = V_{k+1} β_k. If naux − cols(Q) < block_size we must
        // shrink the new block.
        let space_left = naux.saturating_sub(q_basis.ncols());
        let next_block_size = block_size.min(space_left);
        let v_next_opt: Option<Array2<f64>> = if next_block_size == 0 {
            None
        } else {
            let (q_new, r_new) = w
                .qr()
                .map_err(|e| FerricError::General(format!("Lanczos QR failed: {e}")))?;
            // q_new is (naux × block_size); r_new is (block_size × block_size).
            // If the block lost rank (norm tiny), early-terminate.
            let r_norm_sq: f64 = r_new.iter().map(|x| x * x).sum();
            if r_norm_sq.sqrt() < 1e-14 {
                None
            } else {
                betas.push(r_new);
                Some(q_new)
            }
        };

        // Assemble the block-tridiagonal T from alphas/betas built so far.
        let nb = alphas.len();
        let tdim = nb * block_size;
        let mut t = Array2::<f64>::zeros((tdim, tdim));
        for (j, a) in alphas.iter().enumerate() {
            let r0 = j * block_size;
            t.slice_mut(s![r0..r0 + block_size, r0..r0 + block_size])
                .assign(a);
        }
        for (j, b) in betas.iter().enumerate() {
            if j + 1 >= nb {
                break; // β_j connects block j and j+1; need both present in T.
            }
            let r0 = j * block_size;
            let c0 = (j + 1) * block_size;
            // T[k, k+1] = V_k^T A V_{k+1} = β_k^T  ; T[k+1, k] = β_k.
            let bt = b.t().to_owned();
            t.slice_mut(s![r0..r0 + block_size, c0..c0 + block_size])
                .assign(&bt);
            t.slice_mut(s![c0..c0 + block_size, r0..r0 + block_size])
                .assign(b);
        }
        // Symmetrize T explicitly to wash out drift.
        let t_sym: Array2<f64> = 0.5 * (&t + &t.t());

        // Diagonalize.
        let (theta, y) = t_sym
            .eigh(UPLO::Upper)
            .map_err(|e| FerricError::General(format!("Lanczos T-diag failed: {e}")))?;

        // Pick the top n_desired by |λ − 1| descending (PDEP relevance metric).
        let mut order: Vec<usize> = (0..theta.len()).collect();
        order.sort_by(|&i, &j| {
            (theta[j] - 1.0)
                .abs()
                .partial_cmp(&(theta[i] - 1.0).abs())
                .unwrap()
        });
        let n_keep = n_desired.min(order.len());
        let picks = &order[..n_keep];

        // Build Ritz vectors X = Q · Y_pick where Y_pick selects picked columns.
        // y is (tdim × tdim); we want columns indexed by picks.
        let mut y_pick = Array2::<f64>::zeros((tdim, n_keep));
        for (slot, &col) in picks.iter().enumerate() {
            y_pick.slice_mut(s![.., slot]).assign(&y.slice(s![.., col]));
        }
        let ritz: Array2<f64> = q_basis.dot(&y_pick);
        let eigenvalues: Vec<f64> = picks.iter().map(|&c| theta[c]).collect();

        // Residual estimate for block Lanczos: for each picked Ritz pair, the
        // residual norm equals ||β_k · y_i[-block_size:]|| where β_k is the
        // most recent QR R-factor and y_i[-block_size:] is the trailing block
        // of the Ritz coefficient.
        let mut max_resid = 0.0f64;
        if let Some(beta_k) = betas.last() {
            // Only meaningful when we actually have a *next* block of Q to advance to.
            if v_next_opt.is_some() {
                let last_block_start = tdim - block_size;
                for (slot, _) in picks.iter().enumerate() {
                    let y_tail = y_pick.slice(s![last_block_start.., slot]).to_owned();
                    let r_vec = beta_k.dot(&y_tail);
                    let nrm = r_vec.dot(&r_vec).sqrt();
                    max_resid = max_resid.max(nrm);
                }
            }
        } else {
            // No β computed yet (single-iteration corner case); count as converged.
            max_resid = 0.0;
        }

        let result = LanczosResult { eigenvalues, eigenvectors: ritz };
        last_result = Some(result);

        // Termination criteria:
        //   1. No further expansion possible (v_next_opt is None) — return what we have.
        //   2. Residuals below threshold and we already have enough eigenpairs.
        if v_next_opt.is_none() {
            break;
        }
        if max_resid < conv_thresh && nb * block_size >= n_desired {
            break;
        }

        // Advance the recurrence: shift V_{k-1} ← V_k, V_k ← V_{k+1}, β_{k-1} ← β_k.
        let v_next = v_next_opt.unwrap();
        v_prev = Some(v_curr);
        v_curr = v_next.clone();
        beta_prev = betas.last().cloned();

        // Append V_{k+1} to the stacked basis.
        let old_cols = q_basis.ncols();
        let new_cols = old_cols + v_next.ncols();
        let mut q_new = Array2::<f64>::zeros((naux, new_cols));
        q_new.slice_mut(s![.., ..old_cols]).assign(&q_basis);
        q_new.slice_mut(s![.., old_cols..]).assign(&v_next);
        q_basis = q_new;

        // Sanity stop: if we've already exhausted the ambient space.
        if q_basis.ncols() >= naux {
            // One more pass on the existing Krylov basis to update Ritz vectors,
            // then stop. Easiest: just break here; last_result holds the latest.
            // For consistency, run one more (k+1) iteration so α_k+1 is captured.
            let _ = k; // suppress unused if loop body completes.
        }
    }

    last_result.ok_or_else(|| FerricError::General("Lanczos produced no result".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_symmetric_with_spectrum(lambdas: &[f64], seed: u64) -> Array2<f64> {
        let n = lambdas.len();
        // Build a random orthogonal U via QR of a deterministic pseudo-random matrix.
        // Use a simple LCG so tests don't depend on rand crate.
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / u32::MAX as f64) - 0.5
        };
        let mut g = Array2::<f64>::zeros((n, n));
        for v in g.iter_mut() {
            *v = next();
        }
        let (u, _r) = g.qr().unwrap();
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += u[(i, k)] * lambdas[k] * u[(j, k)];
                }
                a[(i, j)] = s;
            }
        }
        a
    }

    #[test]
    fn lanczos_recovers_dense_spectrum() {
        // Spectrum: extreme eigenvalues should be picked up first by |λ − 1|.
        let mut lambdas = vec![0.01, 0.1, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 2.0, 5.0, 10.0];
        // ensure length 50
        lambdas.truncate(50);
        let a = make_symmetric_with_spectrum(&lambdas, 42);

        // Seed: 4 columns of unit vectors that mix the basis (pseudo-random).
        let n = 50;
        let mut seed = Array2::<f64>::zeros((n, 4));
        for j in 0..4 {
            for i in 0..n {
                seed[(i, j)] = ((i + j * 7) as f64).sin();
            }
        }

        let result = run_lanczos_seeded(
            seed,
            |v| a.dot(v),
            4,
            100,
            1e-10,
        ).unwrap();

        // Expected top 4 by |λ − 1| descending: 10.0, 5.0, 0.01, 2.0 (|9|, |4|, |0.99|, |1|)
        // Sort recovered for stable comparison by |λ − 1| descending.
        let mut got = result.eigenvalues.clone();
        got.sort_by(|a, b| (b - 1.0).abs().partial_cmp(&(a - 1.0).abs()).unwrap());
        let mut want: Vec<f64> = vec![10.0, 5.0, 2.0, 0.01];
        want.sort_by(|a, b| (b - 1.0f64).abs().partial_cmp(&(a - 1.0f64).abs()).unwrap());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-8, "Lanczos eigenvalue mismatch: got {g}, want {w}");
        }

        // Verify eigenvectors satisfy A·v ≈ λ·v.
        for (col_idx, &lam) in result.eigenvalues.iter().enumerate() {
            let v: Array1<f64> = result.eigenvectors.column(col_idx).to_owned();
            let av = a.dot(&v);
            let resid: Array1<f64> = &av - &(lam * &v);
            let nrm = resid.dot(&resid).sqrt();
            assert!(nrm < 1e-8, "residual too large: {nrm}");
        }
    }
}
