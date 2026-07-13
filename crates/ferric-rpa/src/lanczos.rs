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

/// Panel width (number of identity-seed columns processed per matvec) for the
/// full-rank paneled dielectric assembly in [`run_lanczos_full_rank`].
///
/// The full-rank identity seed drives the block Lanczos as a *single* outer
/// iteration whose only algebraic content is `A = ε̃(0) = matvec(I)` followed by
/// one dense `eigh(A)` (see [`run_lanczos_full_rank`]). Materializing the whole
/// `naux`-wide block at once forces the matvec closure to allocate its full
/// `(nov × naux)` and several `naux × naux` temporaries simultaneously — the
/// ~17 GB peak at benzene/aTZ scale (atz-benzene-rpa-memory-bound). Assembling
/// `A` in `k`-column panels caps that transient footprint at `O(nov·k + naux·k)`
/// while producing the identical `A` and identical eigenpairs.
///
/// The width is budget-derived from `FERRIC_ERI3_BUDGET_GB` (the same process-wide
/// resident-tensor ceiling used by the 3-index source), reserving budget for one
/// `(nov × k)` matvec scratch plus one `(naux × k)` output panel. An explicit
/// `FERRIC_LANCZOS_PANEL=N` override wins (clamped ≥ 1). Unset budget ⇒ a
/// conservative default panel so the win applies even without an explicit budget.
fn lanczos_panel_width(naux: usize, nov: usize) -> usize {
    // Explicit override wins, clamped to [1, naux]. The clamp target depends on
    // naux (not a constant), so this reads the raw value via ConfigVar for
    // consistent parse/validate but applies the context clamp here rather than a
    // fixed default; a malformed value warns and falls through to budget-derived.
    static PANEL: ferric_core::config::ConfigVar<usize> = ferric_core::config::ConfigVar {
        env_name: "FERRIC_LANCZOS_PANEL",
        default: 0, // sentinel: 0 means "unset" here → fall through to budget-derived
        parse: |s| s.trim().parse::<usize>().map_err(|e| e.to_string()),
        validate: ferric_core::config::accept_any,
    };
    match PANEL.get() {
        Ok(r) if r.source == ferric_core::config::ConfigSource::Env => {
            return r.value.max(1).min(naux.max(1));
        }
        Ok(_) => {} // Default sentinel → budget-derived below.
        Err(e) => eprintln!("[config] FERRIC_LANCZOS_PANEL: {e}; using budget-derived width"),
    }
    // Budget-derived: reserve one (nov × k) matvec scratch + one (naux × k)
    // output panel per panel column. Bytes per panel column ≈ (nov + naux)·8.
    //
    // Resolve via the unified budget. When no *explicit or env* budget is set
    // (the resolver falls back to auto-detect or the 2 GiB fallback), keep the
    // legacy behavior of a conservative fixed 256-column panel so concurrency on
    // memory-tight boxes is preserved — the auto/fallback figure is an OOM guard
    // for the big resident tensors, not a hint that this panel should widen.
    use ferric_core::memory::{self, BudgetSource};
    let resolution = memory::resolve_budget(None);
    let explicitly_budgeted = matches!(
        resolution.source,
        BudgetSource::UnifiedEnv | BudgetSource::LegacyOocEnv | BudgetSource::LegacyEri3Env
    );
    let budget = resolution.bytes;
    let per_col_bytes = (nov.saturating_add(naux)).saturating_mul(8).max(1);
    if !explicitly_budgeted {
        // No explicit budget: default to a panel that keeps the transient matvec
        // scratch to roughly a few hundred MB regardless of naux — enough BLAS-3
        // width to stay efficient, small enough to let benzene/aTZ jobs run
        // concurrently. 256 columns of (nov+naux) doubles ≈ 26 MB at benzene/aTZ.
        return 256usize.max(1).min(naux.max(1));
    }
    // Use ~1/2 of the budget for the panel scratch (the assembled A and its
    // eigenvectors, each naux², live for the whole solve and are counted
    // separately by the caller's footprint).
    let k = (budget / 2 / per_col_bytes).max(1);
    k.min(naux.max(1))
}

/// Full-rank PDEP eigensolve via **paneled dense dielectric assembly**.
///
/// Semantically identical to running [`run_lanczos_seeded`] with the full
/// `naux`-wide identity seed, but restructured for memory: the identity seed
/// collapses block Lanczos to a single outer iteration — `V_0 = QR(I)` is an
/// orthogonal `naux × naux`, `α_0 = V_0ᵀ A V_0` is a similarity transform of
/// `A = ε̃(0)`, and the returned Ritz pairs are the eigenpairs of `A`. This
/// function computes exactly those eigenpairs by assembling `A` column-panel by
/// column-panel through the same `matvec` closure (so `A[:, j..j+k] =
/// matvec(I[:, j..j+k])`), symmetrizing, and doing one `eigh`. Peak transient
/// memory is the panel scratch (`O((nov + naux)·k)`) instead of the full
/// `naux`-wide block's `O(nov·naux)` + several `naux²` temporaries.
///
/// `n_desired` Ritz pairs are returned ordered by `|λ − 1|` descending, matching
/// the identity-seed Lanczos and Davidson conventions. Eigenvalues match the
/// identity-seed path to LAPACK precision; eigenvectors span the same eigenspaces
/// (gauge/sign within degenerate blocks is immaterial to every downstream
/// consumer — the RPA energy trace-log is basis-invariant, and GW/property paths
/// pair each eigenvector with its own eigenvalue).
///
/// BLAS threading is managed exactly as in [`run_lanczos_seeded`] (default 1;
/// `FERRIC_LANCZOS_BLAS_THREADS` opt-in). The `matvec` closure must be BLAS-only
/// (no rayon region) for the same reason.
pub fn run_lanczos_full_rank<F>(
    naux: usize,
    nov: usize,
    matvec: F,
    n_desired: usize,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    with_blas_threads(lanczos_blas_threads(), || {
        full_rank_paneled(naux, nov, matvec, n_desired)
    })
}

/// Serial body for [`run_lanczos_full_rank`]; BLAS threading managed by caller.
fn full_rank_paneled<F>(
    naux: usize,
    nov: usize,
    matvec: F,
    n_desired: usize,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    if n_desired == 0 || naux == 0 {
        return Ok(LanczosResult {
            eigenvalues: Vec::new(),
            eigenvectors: Array2::zeros((naux, 0)),
            converged: true,
            residual_norm: 0.0,
        });
    }

    let panel = lanczos_panel_width(naux, nov);

    // Assemble A = ε̃(0) one column-panel at a time. `matvec(I_panel)` yields the
    // corresponding naux-row × k-col slab of A; scatter it into A. Only one
    // (naux × k) identity panel + the matvec's internal (nov × k)/(naux × k)
    // scratch are live at once, so peak transient memory scales by k/naux.
    let mut a = Array2::<f64>::zeros((naux, naux));
    let mut col = 0usize;
    while col < naux {
        let w = panel.min(naux - col);
        let mut e_panel = Array2::<f64>::zeros((naux, w));
        for j in 0..w {
            e_panel[(col + j, j)] = 1.0;
        }
        let a_panel = matvec(&e_panel); // (naux × w)
        a.slice_mut(s![.., col..col + w]).assign(&a_panel);
        col += w;
    }

    // Symmetrize to wash out any asymmetry in the matvec's floating-point path
    // (the operator ε̃ = I + Π is symmetric by construction).
    let a_sym: Array2<f64> = 0.5 * (&a + &a.t());
    drop(a);

    let (theta, y) = a_sym
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("Full-rank PDEP eigh failed: {e}")))?;

    // Order by |λ − 1| descending (PDEP relevance metric), matching Lanczos/Davidson.
    let mut order: Vec<usize> = (0..theta.len()).collect();
    order.sort_by(|&i, &j| {
        (theta[j] - 1.0)
            .abs()
            .partial_cmp(&(theta[i] - 1.0).abs())
            .unwrap()
    });
    let n_keep = n_desired.min(order.len());
    let picks = &order[..n_keep];

    let mut eigenvectors = Array2::<f64>::zeros((naux, n_keep));
    let mut eigenvalues = Vec::with_capacity(n_keep);
    for (slot, &c) in picks.iter().enumerate() {
        eigenvectors.slice_mut(s![.., slot]).assign(&y.slice(s![.., c]));
        eigenvalues.push(theta[c]);
    }

    // Exact dense eigh of the fully-assembled A — not an iterative subspace
    // solve, so there is no residual to converge and this path is always
    // "converged" by construction.
    Ok(LanczosResult {
        eigenvalues,
        eigenvectors,
        converged: true,
        residual_norm: 0.0,
    })
}

/// Result of a block-Lanczos run.
pub struct LanczosResult {
    /// Converged eigenvalues, sorted by `|λ − 1|` descending (most significant
    /// PDEP modes first), matching the convention used downstream of Davidson.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors in the original space, shape `(naux, n_converged)`.
    pub eigenvectors: Array2<f64>,
    /// Whether the returned Ritz pairs met the caller's residual-norm
    /// tolerance (`conv_thresh` in [`run_lanczos_seeded`]) within the
    /// iteration budget. `true` unconditionally for [`run_lanczos_full_rank`]
    /// (single dense `eigh`, no iterative residual). `false` means the block
    /// Lanczos recurrence exhausted `max_iter` (or ran out of Krylov space
    /// before reaching Ambient dimension) while `residual_norm` was still
    /// above `conv_thresh` — the returned Ritz pairs are the best available,
    /// not verified eigenpairs.
    pub converged: bool,
    /// Residual norm `max_i ||A v_i − λ_i v_i||` (block-Lanczos estimate) of
    /// the returned Ritz pairs at the point the solve stopped. `0.0` for the
    /// full-rank (exact) path.
    pub residual_norm: f64,
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
///
/// Precedence: `FERRIC_LANCZOS_BLAS_THREADS` (this var, back-compat) >
/// `FERRIC_BLAS_THREADS` (umbrella, see `ferric_integrals::blas_threads::
/// opt_in_blas_threads`) > `1`. Falls back to the umbrella resolver when this
/// var is unset, so the umbrella's rayon-worker guard (return 1 if
/// `rayon::current_thread_index().is_some()`) still applies on that path;
/// replicated here too so a caller who sets ONLY the Lanczos-specific var
/// gets the same belt-and-suspenders protection.
///
/// Shared with Davidson (`davidson.rs::davidson_blas_threads`): the two are
/// alternative eigensolvers for the same call site (`config.eigensolver` in
/// lib.rs), and a caller tuning one expects the other to behave the same way
/// when swapped in.
fn lanczos_blas_threads() -> usize {
    solver_blas_threads_with(|k| std::env::var(k).ok())
}

/// [`lanczos_blas_threads`] with an injected env lookup — the testable core,
/// also used by Davidson. Injection instead of `set_var` in tests: see
/// `ferric_integrals::blas_threads::opt_in_blas_threads_with` for why
/// env-mutating resolver tests are a data race under the parallel test
/// harness (process-global OpenBLAS state + concurrent BLAS-doing tests).
/// Real-env raise tests live in `tests/blas_raise_identity.rs` (a dedicated
/// process where nothing else runs BLAS concurrently).
pub(crate) fn solver_blas_threads_with(get: impl Fn(&str) -> Option<String> + Copy) -> usize {
    if rayon::current_thread_index().is_some() {
        return 1;
    }
    // The Lanczos-specific override wins when set to a parseable value. This
    // keeps a hand-rolled read rather than routing through ConfigVar: its
    // precedence is cross-var (a present-but-unparseable value must fall THROUGH
    // to the umbrella `FERRIC_BLAS_THREADS`, not degrade to this var's own
    // default), which `ConfigVar::resolve` — single-var, no sibling fallback —
    // does not model. The umbrella side IS a ConfigVar (see
    // `blas_threads::BLAS_THREADS`); this resolver composes on top of it.
    if let Some(v) = get("FERRIC_LANCZOS_BLAS_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n.max(1);
        }
    }
    // Unset (or unparseable): fall back to the umbrella convention (itself
    // defaulting to 1, and itself re-checking the rayon-worker guard).
    ferric_integrals::blas_threads::opt_in_blas_threads_with(get)
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
    let result = with_blas_threads(lanczos_blas_threads(), || {
        lanczos_iterations(seed, matvec, n_desired, max_iter, conv_thresh)
    })?;
    if !result.converged {
        eprintln!(
            "ferric-rpa WARNING: block Lanczos did not converge within max_iter={max_iter} \
             (conv_thresh={conv_thresh:.3e}, worst residual={:.3e}); returned Ritz pairs \
             are the best available, not verified eigenpairs of the dielectric matrix.",
            result.residual_norm,
        );
    }
    Ok(result)
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
            converged: true,
            residual_norm: 0.0,
        });
    }

    // V_0: QR-orthonormalize the seed block.
    let v0 = qr_orthonormalize(seed)?;
    let block_size = v0.ncols();
    if block_size == 0 {
        return Ok(LanczosResult {
            eigenvalues: Vec::new(),
            eigenvectors: Array2::zeros((naux, 0)),
            converged: true,
            residual_norm: 0.0,
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

        // Termination criteria (checked in order):
        //   1. No further Krylov expansion possible (v_next_opt is None) — the
        //      subspace is either invariant (lucky breakdown) or spans the
        //      full ambient naux-dim space, so the block-tridiagonal T's
        //      eigenpairs are exact for that (sub)space — converged.
        //   2. Residual below threshold with enough eigenpairs — converged.
        //   3. Neither holds and the outer-iteration budget is exhausted on
        //      this pass — NOT converged; the caller gets the best Ritz pairs
        //      found so far, flagged accordingly (mirrors Davidson's "soft"
        //      fall-through, but with the flag Davidson doesn't have).
        let no_more_expansion = v_next_opt.is_none();
        let residual_ok = max_resid < conv_thresh && nb * block_size >= n_desired;
        let is_last_iter = k + 1 == outer_iters;
        let converged = no_more_expansion || residual_ok || !is_last_iter;

        let result = LanczosResult {
            eigenvalues,
            eigenvectors: ritz,
            converged,
            residual_norm: max_resid,
        };
        last_result = Some(result);

        if no_more_expansion {
            break;
        }
        if residual_ok {
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

    // Resolver tests use the injected-lookup variant (solver_blas_threads_with)
    // so they never mutate the real process-global env — see
    // opt_in_blas_threads_with's doc comment for why set_var-based tests are a
    // data race with the other (BLAS-doing) tests in this binary.

    #[test]
    fn solver_blas_threads_defaults_to_one() {
        assert_eq!(solver_blas_threads_with(|_| None), 1);
    }

    #[test]
    fn solver_blas_threads_falls_back_to_umbrella() {
        let get = |k: &str| (k == "FERRIC_BLAS_THREADS").then(|| "5".to_string());
        assert_eq!(solver_blas_threads_with(get), 5);
    }

    #[test]
    fn solver_blas_threads_own_var_wins_over_umbrella() {
        let get = |k: &str| match k {
            "FERRIC_LANCZOS_BLAS_THREADS" => Some("2".to_string()),
            "FERRIC_BLAS_THREADS" => Some("5".to_string()),
            _ => None,
        };
        assert_eq!(solver_blas_threads_with(get), 2, "Lanczos-specific var must win over umbrella");
    }

    #[test]
    fn solver_blas_threads_forces_one_inside_rayon_worker() {
        let get = |k: &str| (k == "FERRIC_LANCZOS_BLAS_THREADS").then(|| "4".to_string());
        assert_eq!(solver_blas_threads_with(get), 4);
        let inside = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| solver_blas_threads_with(get));
        assert_eq!(inside, 1, "rayon-worker guard must force 1 even with own var set");
    }

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

    /// Build a dielectric-shaped operator ε̃ = I + B diag(s²) Bᵀ (SPD, λ ≥ 1),
    /// matching the production `sternheimer::dielectric_apply` structure, and a
    /// closure that applies it to a block. Returns (matvec, dense A) for cross-
    /// checking. `naux`×`nov` random B, positive scale factors.
    fn make_dielectric_op(
        naux: usize,
        nov: usize,
        seed: u64,
    ) -> (impl Fn(&Array2<f64>) -> Array2<f64>, Array2<f64>) {
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
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
        // Positive scale factors s² (like 2·e_ia/(ω²+e_ia²) at ω=0 → 2/e_ia > 0).
        let s2: Vec<f64> = (0..nov).map(|k| 0.3 + (k as f64 % 7.0) * 0.1).collect();

        // Dense A = I + B diag(s²) Bᵀ.
        let mut bs = b.clone();
        for k in 0..nov {
            let col = bs.column(k).to_owned() * s2[k];
            bs.column_mut(k).assign(&col);
        }
        let mut a = bs.dot(&b.t());
        for i in 0..naux {
            a[(i, i)] += 1.0;
        }

        let b_mv = b.clone();
        let s2_mv = s2.clone();
        let matvec = move |v: &Array2<f64>| -> Array2<f64> {
            // out = V + B (diag(s²) (Bᵀ V))
            let mut y = b_mv.t().dot(v); // (nov × m)
            for k in 0..y.nrows() {
                let sk = s2_mv[k];
                let row = y.row(k).to_owned() * sk;
                y.row_mut(k).assign(&row);
            }
            let mut out = v.to_owned();
            out = &out + &b_mv.dot(&y);
            out
        };
        (matvec, a)
    }

    /// The paneled full-rank path must reproduce the identity-seed block Lanczos
    /// to LAPACK precision on a production-shaped dielectric operator — the exact
    /// equivalence guarantee the run_pdep_rpa driver relies on. Also checks that
    /// the panel width (via FERRIC_LANCZOS_PANEL) does not change the answer.
    #[test]
    fn full_rank_matches_identity_seed_lanczos() {
        let naux = 60;
        let nov = 140;
        let (matvec, a) = make_dielectric_op(naux, nov, 7);

        // Reference: identity-seed block Lanczos (the old production path).
        let ref_res = run_lanczos_seeded(
            Array2::<f64>::eye(naux),
            |v| matvec(v),
            naux,
            naux + 4,
            1e-12,
        )
        .unwrap();

        // Sanity: identity-seed Lanczos itself ≡ dense eigh of A.
        let (theta_dense, _) = a.eigh(UPLO::Upper).unwrap();
        let mut dense_sorted = theta_dense.to_vec();
        dense_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

        for panel in [7usize, 16, 60, 200] {
            std::env::set_var("FERRIC_LANCZOS_PANEL", panel.to_string());
            let new_res = run_lanczos_full_rank(naux, nov, |v| matvec(v), naux).unwrap();
            std::env::remove_var("FERRIC_LANCZOS_PANEL");

            assert_eq!(new_res.eigenvalues.len(), ref_res.eigenvalues.len());

            // Eigenvalues match the identity-seed path to LAPACK precision
            // (both are ordered by |λ − 1| descending).
            for (g, r) in new_res.eigenvalues.iter().zip(ref_res.eigenvalues.iter()) {
                assert!(
                    (g - r).abs() < 1e-10,
                    "panel {panel}: eigenvalue mismatch new={g} ref={r}"
                );
            }

            // And match the dense reference spectrum (sorted ascending).
            let mut got_sorted = new_res.eigenvalues.clone();
            got_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for (g, d) in got_sorted.iter().zip(dense_sorted.iter()) {
                assert!(
                    (g - d).abs() < 1e-10,
                    "panel {panel}: vs dense eigh mismatch {g} vs {d}"
                );
            }

            // Eigenvectors satisfy A·v ≈ λ·v (each paired with its own λ).
            for (idx, &lam) in new_res.eigenvalues.iter().enumerate() {
                let v = new_res.eigenvectors.column(idx).to_owned();
                let av = a.dot(&v);
                let resid = &av - &(lam * &v);
                let nrm = resid.dot(&resid).sqrt();
                assert!(nrm < 1e-9, "panel {panel}: residual {nrm}");
            }

            // Full-rank is a single exact dense eigh — always converged.
            assert!(new_res.converged, "panel {panel}: full-rank must report converged=true");
            assert_eq!(new_res.residual_norm, 0.0, "panel {panel}: full-rank residual must be 0");
        }
    }

    /// TD-CONV: a full-width identity-seed block Lanczos with a generous
    /// iteration budget must reach its residual tolerance and report
    /// `converged: true` with a small residual — the control case for the
    /// unconverged test below.
    #[test]
    fn seeded_lanczos_converges_with_ample_budget() {
        let naux = 40;
        let nov = 90;
        let (matvec, _a) = make_dielectric_op(naux, nov, 11);
        let res = run_lanczos_seeded(
            Array2::<f64>::eye(naux),
            |v| matvec(v),
            naux,
            naux + 4, // ample outer-iteration budget
            1e-10,
        )
        .unwrap();
        assert!(res.converged, "expected convergence with a generous iteration budget");
        assert!(
            res.residual_norm < 1e-8,
            "expected a small residual, got {}",
            res.residual_norm
        );
    }

    /// TD-CONV: driving the block Lanczos with `max_iter=1` on a seed that
    /// does NOT span the ambient space forces the outer-iteration budget to
    /// exhaust before either termination criterion (no-more-expansion or
    /// residual-below-threshold) is met — the eigensolver equivalent of the
    /// "max_iter=1" non-convergence probe requested by the TD-CONV brief.
    /// Asserts the unconverged flag is set and the residual is (verifiably)
    /// above the requested tolerance, so the flag is not a false negative.
    #[test]
    fn seeded_lanczos_max_iter_one_reports_unconverged() {
        let naux = 40;
        let nov = 90;
        let (matvec, _a) = make_dielectric_op(naux, nov, 11);
        // Narrow seed block (4 columns out of 40) so one Lanczos step cannot
        // possibly span the full ambient space, and an unreachably tight
        // conv_thresh so the residual check also can't pass in one step.
        let mut seed = Array2::<f64>::zeros((naux, 4));
        for j in 0..4 {
            for i in 0..naux {
                seed[(i, j)] = ((i + j * 13) as f64).cos();
            }
        }
        let res = run_lanczos_seeded(
            seed,
            |v| matvec(v),
            naux, // n_desired = full naux, unreachable from a 4-wide block in 1 step
            1,    // max_iter = 1
            1e-14,
        )
        .unwrap();
        assert!(
            !res.converged,
            "expected max_iter=1 on a narrow, non-spanning seed to be flagged unconverged"
        );
        assert!(
            res.residual_norm > 1e-14,
            "unconverged result should carry a residual above the (unreachable) tolerance, \
             got {}",
            res.residual_norm
        );
    }
}
