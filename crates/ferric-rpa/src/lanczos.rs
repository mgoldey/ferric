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
fn lanczos_panel_width(naux: usize, nov: usize, memory_budget_bytes: Option<usize>) -> usize {
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
    // Resolve via the unified budget, now including the CALLER's own
    // `memory_budget_bytes` (a TOML/kwarg `[memory] budget_gb`) as an explicit
    // source — previously this always called `resolve_budget(None)`, so a
    // TOML-set budget was invisible here even though it fully controls the
    // 3-index AO tensor blocking elsewhere in the same run (found 2026-07-21).
    // When no explicit-OR-env budget is set at all (the resolver falls back to
    // auto-detect or the 2 GiB fallback), keep the legacy behavior of a
    // conservative fixed 256-column panel so concurrency on memory-tight boxes
    // is preserved — the auto/fallback figure is an OOM guard for the big
    // resident tensors, not a hint that this panel should widen.
    use ferric_core::memory::{self, BudgetSource};
    let resolution = memory::resolve_budget(memory_budget_bytes);
    let explicitly_budgeted = matches!(
        resolution.source,
        BudgetSource::Explicit
            | BudgetSource::UnifiedEnv
            | BudgetSource::LegacyOocEnv
            | BudgetSource::LegacyEri3Env
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
    run_lanczos_full_rank_budgeted(naux, nov, matvec, n_desired, None)
}

/// Same as [`run_lanczos_full_rank`], but accepts the caller's own resolved
/// `memory_budget_bytes` (e.g. `RsMp2RpaConfig`/`PdepRpaConfig.memory_budget_bytes`,
/// i.e. a TOML/kwarg `[memory] budget_gb`) instead of always falling back to
/// `lanczos_panel_width`'s internal `resolve_budget(None)` — which only
/// recognizes an ENV-sourced budget as "explicit" and silently ignores any
/// TOML-supplied one (found 2026-07-21 auditing a benzene aug-cc-pVQZ OOM).
pub fn run_lanczos_full_rank_budgeted<F>(
    naux: usize,
    nov: usize,
    matvec: F,
    n_desired: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    with_blas_threads(lanczos_blas_threads(), || {
        full_rank_paneled(naux, nov, matvec, n_desired, memory_budget_bytes)
    })
}

/// Serial body for [`run_lanczos_full_rank`]; BLAS threading managed by caller.
fn full_rank_paneled<F>(
    naux: usize,
    nov: usize,
    matvec: F,
    n_desired: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    if n_desired == 0 || naux == 0 {
        return Ok(LanczosResult {
            eigenvalues: Vec::new(),
            eigenvectors: Array2::zeros((naux, 0)),
            converged: true,
            max_resid: 0.0,
            subspace_dim: 0,
            n_desired,
            block_iterations: 0,
            hit_ambient_cap: false,
        });
    }

    let panel = lanczos_panel_width(naux, nov, memory_budget_bytes);
    paneled_dense_eigs(naux, panel, &matvec, n_desired)
}

/// Dense eigensolve of the `naux`-dim operator behind `matvec`: assemble
/// `A = matvec(I)` in `panel`-column slabs, symmetrize, one `eigh`, and return
/// the top `n_desired` pairs by `|λ − 1|` descending. Shared by the full-rank
/// path and by [`run_lanczos_seeded`] for a request no Krylov space can hold.
fn paneled_dense_eigs<F>(
    naux: usize,
    panel: usize,
    matvec: &F,
    n_desired: usize,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    let panel = panel.max(1);

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
    //
    // MEMORY (why this is an in-place loop and not `0.5 * (&a + &a.t())`):
    // the expression form allocated a fresh naux² for `&a + &a.t()`, which
    // `0.5 * ...` then consumed into `a_sym` — with `a` still live, because
    // the `drop(a)` came AFTER `a_sym` was materialized. So the true peak was
    // THREE naux² buffers (a, the sum temporary, and shortly after the `eigh`
    // eigenvector output `y`), while `budget::estimate_peak_bytes` charges
    // `naux*naux*2*8` for this stage. At naux=2976 that is a 71 MB shortfall
    // against a 141 MB charge — a 1.5x under-count on the Lanczos term.
    //
    // Symmetrizing in place fixes the accounting by fixing the allocation: `a`
    // becomes its own symmetrized self, so only `a` and the subsequent `eigh`
    // output are ever co-resident and the existing 2·naux² charge is exact.
    //
    // Bit-identical: the previous form computed element (i,j) as
    // `0.5 * (a[(i,j)] + a[(j,i)])`, and so does this. Reading a[(j,i)] before
    // either element is written, and writing the pair together, means the
    // in-place update sees exactly the same operands in the same order — IEEE
    // addition and the multiply by 0.5 (an exact power of two) are unchanged.
    // The diagonal is 0.5*(x+x) == x exactly, so it is left untouched.
    for i in 0..naux {
        for j in (i + 1)..naux {
            let avg = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = avg;
            a[(j, i)] = avg;
        }
    }
    let a_sym = a;

    let (theta, y) = a_sym
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("Full-rank PDEP eigh failed: {e}")))?;

    // Order by |λ − 1| descending (PDEP relevance metric), matching Lanczos/Davidson.
    let mut order: Vec<usize> = (0..theta.len()).collect();
    order.sort_by(|&i, &j| (theta[j] - 1.0).abs().total_cmp(&(theta[i] - 1.0).abs()));
    let n_keep = n_desired.min(order.len());
    let picks = &order[..n_keep];

    let mut eigenvectors = Array2::<f64>::zeros((naux, n_keep));
    let mut eigenvalues = Vec::with_capacity(n_keep);
    for (slot, &c) in picks.iter().enumerate() {
        eigenvectors
            .slice_mut(s![.., slot])
            .assign(&y.slice(s![.., c]));
        eigenvalues.push(theta[c]);
    }

    // Exact dense eigh of the fully-assembled A — not an iterative subspace
    // solve, so there is no residual to converge and this path is always
    // "converged" by construction.
    Ok(LanczosResult {
        eigenvalues,
        eigenvectors,
        converged: true,
        max_resid: 0.0,
        subspace_dim: naux,
        n_desired,
        block_iterations: 0,
        hit_ambient_cap: false,
    })
}

/// `a.max(b)`, except a NaN in either operand is returned as NaN. `f64::max`
/// drops a NaN, so a NaN Ritz residual would read as `0.0` and pass the
/// `max_resid < conv_thresh` gate.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// Block-Lanczos residual estimate over the picked Ritz pairs: the max over
/// columns `0..n_pick` of `||β_k · y_pick[tail_start.., col]||`, reduced with
/// [`nan_max`] so a NaN in any column survives. The ONE reduction both the
/// seeded and the targeted loop call, so neither can drift back to `f64::max`.
fn max_ritz_residual(
    beta_k: &Array2<f64>,
    y_pick: &Array2<f64>,
    tail_start: usize,
    n_pick: usize,
) -> f64 {
    (0..n_pick).fold(0.0f64, |acc, col| {
        let y_tail = y_pick.slice(s![tail_start.., col]).to_owned();
        let r_vec = beta_k.dot(&y_tail);
        nan_max(acc, r_vec.dot(&r_vec).sqrt())
    })
}

/// Result of a block-Lanczos run.
#[derive(Debug, Clone)]
#[must_use]
pub struct LanczosResult {
    /// Converged eigenvalues, sorted by `|λ − 1|` descending (most significant
    /// PDEP modes first), matching the convention used downstream of Davidson.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors in the original space, shape `(naux, n_converged)`.
    pub eigenvectors: Array2<f64>,
    /// Whether the returned Ritz pairs met the caller's residual-norm
    /// tolerance (`conv_thresh` in [`run_lanczos_seeded`]) within the
    /// iteration budget. `true` unconditionally for [`run_lanczos_full_rank`]
    /// and for a seeded request above the Krylov cap, which is solved densely
    /// (single dense `eigh`, no iterative residual). `false` means the block
    /// Lanczos recurrence exhausted `max_iter` or the ambient cap before both
    /// `max_resid < conv_thresh` and the pair count held — the returned Ritz
    /// pairs are the best available, not verified eigenpairs.
    pub converged: bool,
    /// Residual norm `max_i ||A v_i − λ_i v_i||` (block-Lanczos estimate) of
    /// the returned Ritz pairs at the point the solve stopped. `0.0` for the
    /// full-rank (exact) path.
    pub max_resid: f64,
    /// Krylov dimension reached (`nb * block_size`) when the solve stopped;
    /// `naux` for the full-rank path.
    ///
    /// Carried so a non-convergence warning can name the ACTUAL cause:
    /// [`run_lanczos_seeded`] needs BOTH `max_resid < conv_thresh` AND
    /// `subspace_dim >= n_desired`. Reporting only `max_resid` made a
    /// count-starved solve read as a residual failure (an alkane sweep warned
    /// "worst residual=1.172e-14" against `conv_thresh=1e-6`).
    pub subspace_dim: usize,
    /// Eigenpairs requested. For [`run_lanczos_targeted`] it is the number of
    /// modes found above `depart_thresh`, whose count criterion differs.
    pub n_desired: usize,
    /// Outer block iterations run (`0` for the full-rank path).
    pub block_iterations: usize,
    /// An UNCONVERGED solve ran its last permitted block iteration AND that
    /// bound was the ambient-space cap (`naux` rounded DOWN to a multiple of
    /// the block size), not `max_iter`. Set by the solver from its own loop
    /// bound (`outer_iters <= max_iter`, i.e. fewer than the `max_iter + 1`
    /// passes `max_iter` alone would allow), so the warning never has to infer
    /// the limit from `block_iterations`. At the cap the pair count always
    /// holds (a larger request is solved densely), so the residual is what
    /// failed. Always `false` when `converged`
    /// (including a full-space solve that converged on its last pass), and
    /// for the full-rank path.
    pub hit_ambient_cap: bool,
}

/// Why [`run_lanczos_seeded`] reported `converged == false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanczosStallCause {
    /// Residual below `conv_thresh`, but the Krylov subspace holds fewer than
    /// `n_desired` pairs: more iterations are needed, not a tighter solve.
    CountStarved,
    /// Enough pairs, but the residual is still at or above `conv_thresh`.
    ResidualAboveThreshold,
    /// Both the residual and the pair count fall short.
    ResidualAndCount,
    /// Both criteria hold yet the solve is flagged unconverged: a logic error
    /// in the iteration, reported rather than hidden.
    Inconsistent,
    /// The residual is NaN or infinite, so it can neither pass nor be
    /// compared; reported as such whatever the pair count.
    NonFiniteResidual,
}

/// Classify an unconverged seeded-Lanczos stop by which convergence criterion
/// failed. Pure, so the warning text is testable without capturing stderr.
pub fn lanczos_stall_cause(
    max_resid: f64,
    conv_thresh: f64,
    subspace_dim: usize,
    n_desired: usize,
) -> LanczosStallCause {
    if !max_resid.is_finite() {
        return LanczosStallCause::NonFiniteResidual;
    }
    let resid_ok = max_resid < conv_thresh;
    let count_ok = subspace_dim >= n_desired;
    match (resid_ok, count_ok) {
        (true, false) => LanczosStallCause::CountStarved,
        (false, true) => LanczosStallCause::ResidualAboveThreshold,
        (false, false) => LanczosStallCause::ResidualAndCount,
        (true, true) => LanczosStallCause::Inconsistent,
    }
}

/// Which limit ended an unconverged seeded solve, read from
/// [`LanczosResult::hit_ambient_cap`] (never inferred from iteration counts).
fn stop_limit(result: &LanczosResult) -> &'static str {
    if result.hit_ambient_cap {
        "the ambient Krylov cap"
    } else {
        "max_iter"
    }
}

/// Short consequence summary for a call site that has ALREADY had the full
/// [`lanczos_nonconvergence_message`] printed by [`run_lanczos_seeded`]: the
/// failed criterion and the limit that stopped the solve, with no repeated
/// numbers (those are in the solver's warning just above it).
///
/// Seeded-solver only: the full-rank path never stalls, and the targeted
/// solver's count criterion differs from the one classified here.
pub fn lanczos_stall_summary(result: &LanczosResult, conv_thresh: f64) -> String {
    debug_assert!(result.block_iterations > 0, "seeded-solver results only");
    let (resid, dim, want) = (result.max_resid, result.subspace_dim, result.n_desired);
    let what = match lanczos_stall_cause(resid, conv_thresh, dim, want) {
        LanczosStallCause::CountStarved => "count-starved, residual is fine",
        LanczosStallCause::ResidualAboveThreshold => "residual above threshold",
        LanczosStallCause::ResidualAndCount => "residual above threshold and count-starved",
        LanczosStallCause::Inconsistent => "INTERNAL INCONSISTENCY, both criteria pass",
        LanczosStallCause::NonFiniteResidual => "non-finite residual",
    };
    format!(
        "{what}; stopped by {} after {} block iterations, see the block Lanczos warning above",
        stop_limit(result),
        result.block_iterations
    )
}

/// `Some` when every eigenpair of the `naux`-dim space was requested; `naux`
/// is the row count of the returned Ritz vectors. Such a request reaches the
/// iteration only when the block size divides `naux` (otherwise
/// [`run_lanczos_seeded`] solves it densely), so a full Krylov space is
/// reachable and only `max_iter` can have starved it.
fn full_request_note(result: &LanczosResult) -> Option<String> {
    let naux = result.eigenvectors.nrows();
    if result.n_desired < naux {
        return None;
    }
    Some(format!(
        "all {naux} eigenpairs were requested; only a full Krylov space satisfies this, so the \
         solve runs to max_iter or the size cap"
    ))
}

/// What a count-starved user can do. When every eigenpair of the space was
/// requested only a full Krylov space meets the count, so raising `max_vecs`
/// is NOT the advice. A count-starved stop is never the ambient cap: a request
/// above [`krylov_cap`] is solved densely and never iterates.
fn count_starved_advice(result: &LanczosResult) -> String {
    full_request_note(result).unwrap_or_else(|| {
        "more block iterations (a larger max_iter / eigensolver_max_vecs) are needed, not a \
         tighter residual"
            .to_string()
    })
}

/// The non-convergence warning for an unconverged seeded-Lanczos result,
/// naming the failed criterion and what ended the iteration.
///
/// Seeded-solver only (see [`lanczos_stall_summary`]).
pub fn lanczos_nonconvergence_message(
    result: &LanczosResult,
    max_iter: usize,
    conv_thresh: f64,
) -> String {
    debug_assert!(result.block_iterations > 0, "seeded-solver results only");
    let (resid, dim, want) = (result.max_resid, result.subspace_dim, result.n_desired);
    let stop = if result.hit_ambient_cap {
        format!(
            "stopped by the ambient Krylov cap of {dim} after {} block iterations \
             (naux rounded down to a multiple of the block size), NOT max_iter={max_iter}",
            result.block_iterations
        )
    } else {
        format!("stopped by max_iter={max_iter}, which was exhausted")
    };
    let cause = match lanczos_stall_cause(resid, conv_thresh, dim, want) {
        LanczosStallCause::CountStarved => format!(
            "the residual is FINE ({resid:.3e} < conv_thresh {conv_thresh:.3e}) but the Krylov \
             subspace reached only {dim} of the {want} eigenpairs requested; the residual is \
             NOT the problem: {}",
            count_starved_advice(result)
        ),
        LanczosStallCause::ResidualAboveThreshold => format!(
            "residual {resid:.3e} is still above conv_thresh {conv_thresh:.3e} (subspace {dim} \
             >= {want} requested, so the count is fine)"
        ),
        LanczosStallCause::ResidualAndCount => format!(
            "residual {resid:.3e} is above conv_thresh {conv_thresh:.3e} AND the subspace \
             reached only {dim} of the {want} eigenpairs requested{}",
            full_request_note(result).map_or(String::new(), |n| format!(": {n}"))
        ),
        LanczosStallCause::Inconsistent => format!(
            "INTERNAL INCONSISTENCY: residual {resid:.3e} < {conv_thresh:.3e} and subspace \
             {dim} >= {want} both hold, yet the solve reports non-convergence; please report \
             this"
        ),
        LanczosStallCause::NonFiniteResidual => format!(
            "non-finite residual ({resid}): the Ritz pairs are numerically broken, not merely \
             unconverged (subspace {dim} of {want} eigenpairs requested)"
        ),
    };
    format!(
        "ferric-rpa WARNING: block Lanczos did not converge: {cause}; {stop}. Returned Ritz \
         pairs are the best available, not verified eigenpairs of the dielectric matrix."
    )
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
/// Defaults to **1**, opt-in via `FERRIC_LANCZOS_BLAS_THREADS`.
///
///  1. ~~**Stack overflow.**~~ **NO LONGER REPRODUCIBLE — re-verified
///     2026-07-26.** `78bc70b` (2026-07-07) introduced this default because
///     `available_parallelism()` made the aug-cc-pV{D,T}Z PDEP-RPA tests abort
///     with `stack overflow`. That crash no longer occurs: the full ferric-rpa
///     suite passes at `FERRIC_LANCZOS_BLAS_THREADS=12`, 8 concurrent CLI
///     solves at 12 BLAS threads each (96 threads on a 12-core box) all
///     complete with identical energies, and a direct probe
///     (`benchmarks/harness/examples/lanczos_stack_probe.rs`) runs `eigh` at
///     n=1200 — larger than benzene/aug-cc-pVTZ's naux=763 — on a 2 MB stack
///     and inside a rayon worker without aborting.
///
///     Most likely cause of the fix: the eigensolve moved onto
///     `ferric_core::linalg::eigh_dc`/`eigvalsh_dc` (`c284e48`), which follow
///     the standard LAPACK workspace-query convention and **heap**-allocate
///     the workspace, where the previous path put it on the caller's stack.
///     Pinned by `tests/lanczos_blas_raise_no_crash.rs`, which runs the exact
///     configuration `78bc70b` reported as aborting.
///
///     NOTE this does NOT generalize to the per-frequency loop in `energy.rs`,
///     which genuinely runs inside a rayon region — BLAS must stay at 1 there,
///     and that guard is still load-bearing (oversubscription, and the
///     openblas-rayon-dgetrf-crash footgun).
///  2. **Reproducibility — this is now the ONLY reason for the default.**
///     Multi-threaded OpenBLAS GEMM/eigh changes the reduction order, which the
///     crate's equivalence tests (screened-vs-dense at thresh=0) hold to a tight
///     tolerance. It costs bit-identity, NOT accuracy (measured separately: a
///     threaded dgemm's max error is identical to serial — see
///     `ferric_tensors::einsum::gemm_kblocked`).
///
///     Measured benefit, benzene/aug-cc-pVTZ (naux=763), eigensolve stage:
///     `=1` 1195 ms, `=4` 840 ms, **`=6` 628 ms (1.9x)**, `=12` 1175 ms
///     (regresses past the knee). RPA correlation energy identical to all 10
///     printed digits at every setting. Worth setting for a solo large job.
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
/// descending. A request the block Krylov space can never hold (`n_desired`
/// above `naux` rounded down to a multiple of the block size, e.g. every pair
/// from a block that does not divide `naux`) is solved densely through the
/// same `matvec` instead, ignoring `max_iter` and `conv_thresh`. If the residuals never drop below `conv_thresh` within
/// `max_iter` outer block iterations, returns the best Ritz pairs from the
/// final iteration without error (matching Davidson's "soft" fall-through
/// pattern when subspace expansion is exhausted).
///
/// The block matvec, full reorthogonalization (Qᵀ·W, Q·proj), QR, and Ritz
/// assembly are all naux-wide GEMMs with no *unguarded* rayon region active
/// anywhere on this call path, so this is the one place BLAS threads are
/// temporarily raised (via `lanczos_blas_threads()`, opt-in above 1); the
/// scoped guard restores the prior count on exit. Because the raise covers
/// the matvec closure too, any `matvec` implementation that internally opens
/// a rayon region (e.g. `sternheimer_sparse::dielectric_apply_screened`,
/// parallel over `i_loc` since the sternheimer-sparse-parallelize change)
/// MUST re-pin BLAS to 1 thread for the duration of that region itself
/// (`with_blas_threads(1, ...)`) — nesting rayon under a possibly-raised
/// outer BLAS thread count is exactly the oversubscription/stack-overflow
/// hazard `blas_threads.rs` warns about. A `matvec` that opens rayon without
/// its own internal re-pin is unsound here.
/// `verbose`: print one line per outer block iteration to stdout (iteration
/// number, block size, worst Ritz residual, target threshold) — live
/// progress for a long-running eigensolve, opt-in and additive. Default
/// `false` at every call site (`PdepRpaConfig::verbose`, itself default
/// `false`); mirrors `ferric_scf::rhf::RhfConfig::verbose`'s convention.
pub fn run_lanczos_seeded<F>(
    seed: Array2<f64>,
    matvec: F,
    n_desired: usize,
    max_iter: usize,
    conv_thresh: f64,
    verbose: bool,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    let result = with_blas_threads(lanczos_blas_threads(), || {
        lanczos_iterations(seed, matvec, n_desired, max_iter, conv_thresh, verbose)
    })?;
    if !result.converged {
        eprintln!(
            "{}",
            lanczos_nonconvergence_message(&result, max_iter, conv_thresh)
        );
    }
    Ok(result)
}

/// Largest Krylov dimension a `block_size`-wide block Lanczos can build in an
/// `naux`-dim space: the loop runs whole blocks only, so `naux` rounded DOWN
/// to a multiple of `block_size` (`0` for a seed wider than the space).
fn krylov_cap(naux: usize, block_size: usize) -> usize {
    naux - naux % block_size.max(1)
}

/// Serial Lanczos iteration body; BLAS threading is managed by the caller
/// (`run_lanczos_seeded`).
///
/// A request for more pairs than [`krylov_cap`] (every full request
/// `n_desired >= naux` whose block size does not divide `naux`, and any seed
/// wider than the space) could never meet the count criterion, so it is
/// solved densely instead: `A = matvec(I)` assembled in `block_size`-wide
/// panels (the matvec width the caller already sized) and one `eigh`.
fn lanczos_iterations<F>(
    seed: Array2<f64>,
    matvec: F,
    n_desired: usize,
    max_iter: usize,
    conv_thresh: f64,
    verbose: bool,
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
            max_resid: 0.0,
            subspace_dim: 0,
            n_desired,
            block_iterations: 0,
            hit_ambient_cap: false,
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
            max_resid: 0.0,
            subspace_dim: 0,
            n_desired,
            block_iterations: 0,
            hit_ambient_cap: false,
        });
    }

    if n_desired > krylov_cap(naux, block_size) {
        if verbose {
            println!(
                "Lanczos: {n_desired} of {naux} pairs exceed the block-{block_size} Krylov cap \
                 {}; solving densely",
                krylov_cap(naux, block_size)
            );
        }
        return paneled_dense_eigs(naux, block_size, &matvec, n_desired);
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
    let max_krylov = max_iter
        .saturating_add(1)
        .saturating_mul(block_size)
        .min(naux);
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
        order.sort_by(|&i, &j| (theta[j] - 1.0).abs().total_cmp(&(theta[i] - 1.0).abs()));
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
                max_resid = max_ritz_residual(beta_k, &y_pick, tdim - block_size, picks.len());
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

        // Live per-iteration progress for a user watching a long-running PDEP
        // eigensolve (opt-in via `verbose`, threaded from
        // `PdepRpaConfig::verbose`). Printed to STDOUT (normal-operation
        // progress, not a warning), one line per outer block iteration,
        // mirroring the RHF/UHF/ROHF/OO-RI-MP2 verbose-trace convention.
        if verbose {
            println!(
                "Lanczos iter={k:4}  block={block_size:4}  krylov_dim={tdim:5}  \
                 max_resid={max_resid:.3e}  conv_thresh={conv_thresh:.3e}"
            );
        }

        let result = LanczosResult {
            eigenvalues,
            eigenvectors: ritz,
            converged,
            max_resid,
            subspace_dim: tdim,
            n_desired,
            block_iterations: k + 1,
            hit_ambient_cap: !converged && outer_iters <= max_iter,
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

/// Targeted Lanczos: find only the eigenpotentials with |λ−1| > `depart_thresh`.
///
/// Identical to [`run_lanczos_seeded`] except the convergence check considers
/// only Ritz pairs whose eigenvalue departs from the identity by more than
/// `depart_thresh`. Tail modes (|λ−1| ≤ thresh) are NOT required to converge
/// and are NOT returned. This avoids growing the Krylov basis to full rank
/// when only the significant modes are needed — the dominant speedup path for
/// PDEP truncation.
///
/// Returns all converged Ritz pairs with |λ−1| > `depart_thresh`, ordered by
/// |λ−1| descending. If the iteration budget is exhausted before all
/// significant modes converge, falls back to returning whatever was found
/// (with `converged = false`).
pub fn run_lanczos_targeted<F>(
    seed: Array2<f64>,
    matvec: F,
    depart_thresh: f64,
    max_iter: usize,
    conv_thresh: f64,
    verbose: bool,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    with_blas_threads(lanczos_blas_threads(), || {
        lanczos_targeted_iterations(seed, matvec, depart_thresh, max_iter, conv_thresh, verbose)
    })
}

fn lanczos_targeted_iterations<F>(
    seed: Array2<f64>,
    matvec: F,
    depart_thresh: f64,
    max_iter: usize,
    conv_thresh: f64,
    verbose: bool,
) -> Result<LanczosResult, FerricError>
where
    F: Fn(&Array2<f64>) -> Array2<f64>,
{
    let naux = seed.nrows();

    let v0 = qr_orthonormalize(seed)?;
    let block_size = v0.ncols();
    if block_size == 0 {
        return Ok(LanczosResult {
            eigenvalues: Vec::new(),
            eigenvectors: Array2::zeros((naux, 0)),
            converged: true,
            max_resid: 0.0,
            subspace_dim: 0,
            n_desired: 0,
            block_iterations: 0,
            hit_ambient_cap: false,
        });
    }

    let mut q_basis: Array2<f64> = v0.clone();
    let mut alphas: Vec<Array2<f64>> = Vec::new();
    let mut betas: Vec<Array2<f64>> = Vec::new();
    let mut v_curr: Array2<f64> = v0;
    let mut v_prev: Option<Array2<f64>> = None;
    let mut beta_prev: Option<Array2<f64>> = None;

    let max_krylov = max_iter
        .saturating_add(1)
        .saturating_mul(block_size)
        .min(naux);
    let outer_iters = (max_krylov / block_size).max(1);
    let mut last_result: Option<LanczosResult> = None;
    let mut last_y_pick: Option<Array2<f64>> = None;

    for k in 0..outer_iters {
        let mut w: Array2<f64> = matvec(&v_curr);
        let alpha_k: Array2<f64> = v_curr.t().dot(&w);
        let alpha_k_sym: Array2<f64> = 0.5 * (&alpha_k + &alpha_k.t());

        w = &w - &v_curr.dot(&alpha_k);
        if let (Some(vp), Some(bp)) = (v_prev.as_ref(), beta_prev.as_ref()) {
            w = &w - &vp.dot(&bp.t());
        }

        for _pass in 0..2 {
            let proj: Array2<f64> = q_basis.t().dot(&w);
            w = &w - &q_basis.dot(&proj);
        }

        alphas.push(alpha_k_sym);

        let space_left = naux.saturating_sub(q_basis.ncols());
        let next_block_size = block_size.min(space_left);
        let v_next_opt: Option<Array2<f64>> = if next_block_size == 0 {
            None
        } else {
            let (q_new, r_new) = w
                .qr()
                .map_err(|e| FerricError::General(format!("Lanczos QR failed: {e}")))?;
            let r_norm_sq: f64 = r_new.iter().map(|x| x * x).sum();
            if r_norm_sq.sqrt() < 1e-14 {
                None
            } else {
                betas.push(r_new);
                Some(q_new)
            }
        };

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
                break;
            }
            let r0 = j * block_size;
            let c0 = (j + 1) * block_size;
            let bt = b.t().to_owned();
            t.slice_mut(s![r0..r0 + block_size, c0..c0 + block_size])
                .assign(&bt);
            t.slice_mut(s![c0..c0 + block_size, r0..r0 + block_size])
                .assign(b);
        }
        let t_sym: Array2<f64> = 0.5 * (&t + &t.t());

        let (theta, y) = t_sym
            .eigh(UPLO::Upper)
            .map_err(|e| FerricError::General(format!("Lanczos T-diag failed: {e}")))?;

        // Pick only modes with |λ−1| > depart_thresh, ordered by |λ−1| descending.
        let mut order: Vec<usize> = (0..theta.len())
            .filter(|&i| (theta[i] - 1.0).abs() > depart_thresh)
            .collect();
        order.sort_by(|&i, &j| (theta[j] - 1.0).abs().total_cmp(&(theta[i] - 1.0).abs()));
        let n_keep = order.len();

        let mut y_pick = Array2::<f64>::zeros((tdim, n_keep.max(1)));
        let mut eigenvalues = Vec::with_capacity(n_keep);
        for (slot, &col) in order.iter().enumerate() {
            y_pick.slice_mut(s![.., slot]).assign(&y.slice(s![.., col]));
            eigenvalues.push(theta[col]);
        }

        // Residual from the T-eigenvector tail — no full Ritz vectors needed.
        let mut max_resid = 0.0f64;
        if let Some(beta_k) = betas.last() {
            if v_next_opt.is_some() && n_keep > 0 {
                max_resid = max_ritz_residual(beta_k, &y_pick, tdim - block_size, n_keep);
            }
        }

        let no_more_expansion = v_next_opt.is_none();
        let krylov_wide_enough = tdim >= (3 * n_keep / 2).max(2 * block_size);
        let residual_ok = n_keep > 0 && max_resid < conv_thresh && krylov_wide_enough;
        let converged = no_more_expansion || residual_ok;

        if verbose {
            println!(
                "targeted Lanczos iter={k:4}  block={block_size:4}  krylov_dim={tdim:5}  \
                 n_above_thresh={n_keep:4}  max_resid={max_resid:.3e}  \
                 conv_thresh={conv_thresh:.3e}"
            );
        }

        last_y_pick = Some(y_pick.slice(s![.., ..n_keep]).to_owned());
        last_result = Some(LanczosResult {
            eigenvalues,
            eigenvectors: Array2::zeros((naux, 0)),
            converged,
            max_resid,
            subspace_dim: tdim,
            n_desired: n_keep,
            block_iterations: k + 1,
            hit_ambient_cap: !converged && k + 1 == outer_iters && outer_iters <= max_iter,
        });

        if no_more_expansion || residual_ok {
            break;
        }

        let v_next = v_next_opt.unwrap();
        v_prev = Some(v_curr);
        v_curr = v_next.clone();
        beta_prev = betas.last().cloned();

        let old_cols = q_basis.ncols();
        let new_cols = old_cols + v_next.ncols();
        let mut q_new = Array2::<f64>::zeros((naux, new_cols));
        q_new.slice_mut(s![.., ..old_cols]).assign(&q_basis);
        q_new.slice_mut(s![.., old_cols..]).assign(&v_next);
        q_basis = q_new;
    }

    // Compute Ritz vectors once after the loop (deferred from per-iteration).
    let mut result = last_result
        .ok_or_else(|| FerricError::General("targeted Lanczos produced no result".into()))?;
    if let Some(yp) = last_y_pick {
        let tdim = yp.nrows();
        if yp.ncols() > 0 && tdim > 0 {
            let qb = q_basis.slice(s![.., ..tdim]);
            result.eigenvectors = qb.dot(&yp);
        }
    }
    Ok(result)
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

    /// The in-place symmetrization in `full_rank_paneled` must be BIT-identical
    /// to the `0.5 * (&a + &a.t())` expression it replaced.
    ///
    /// That expression allocated a fresh `naux²` for `&a + &a.t()`, which
    /// `0.5 * ...` consumed into `a_sym` while `a` was still live (the
    /// `drop(a)` came AFTER), so the true peak was THREE `naux²` buffers
    /// against a `budget.rs` charge of two — a 1.5x under-count on the Lanczos
    /// term. Symmetrizing in place fixes the allocation, which is what makes
    /// the existing charge correct.
    ///
    /// The replacement is only legitimate if it is numerically inert, and
    /// "obviously equivalent" is exactly the claim that has burned this repo
    /// before — so assert exact equality, not a tolerance. Random asymmetric
    /// input, because a symmetric one would pass trivially.
    #[test]
    fn in_place_symmetrization_is_bit_identical_to_the_expression_form() {
        let n = 37;
        // Deterministic pseudo-random asymmetric fill; irrational-ish strides
        // so no two entries coincide and cancellation is exercised.
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 7 + j * 13) as f64).sin() * 1e3
                    + ((i as f64) - (j as f64) * 0.5).cos() * 1e-7;
            }
        }

        // Reference: the old expression form.
        let reference: Array2<f64> = 0.5 * (&a + &a.t());

        // Current: the in-place loop, copied verbatim from full_rank_paneled.
        let mut got = a.clone();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (got[(i, j)] + got[(j, i)]);
                got[(i, j)] = avg;
                got[(j, i)] = avg;
            }
        }

        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    got[(i, j)].to_bits(),
                    reference[(i, j)].to_bits(),
                    "element ({i},{j}) differs: in-place symmetrization must be \
                     bit-identical to 0.5 * (&a + &a.t())"
                );
            }
        }
    }

    /// The full-rank driver must still produce correct eigenpairs after the
    /// symmetrization change — the end-to-end complement to the bit-identity
    /// unit test above.
    #[test]
    fn full_rank_paneled_recovers_a_known_spectrum() {
        // A small SPD operator with a known spectrum: A = I + w wᵀ has
        // eigenvalue 1 + |w|² once and 1 with multiplicity n-1.
        let n = 9;
        let w: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64 + 1.0) * 0.25);
        let norm2: f64 = w.iter().map(|x| x * x).sum();

        let a_op = {
            let w = w.clone();
            move |v: &Array2<f64>| -> Array2<f64> {
                // (I + w wᵀ) V = V + w (wᵀ V)
                let wt_v = w.t().dot(v); // (ncols,)
                let mut out = v.clone();
                for (j, &s) in wt_v.iter().enumerate() {
                    let mut col = out.column_mut(j);
                    for i in 0..n {
                        col[i] += w[i] * s;
                    }
                }
                out
            }
        };

        let res = run_lanczos_full_rank(n, n, a_op, n).expect("full-rank solve");
        assert_eq!(res.eigenvalues.len(), n);

        // Ordered by |lambda - 1| descending, so the rank-one mode is first.
        assert!(
            (res.eigenvalues[0] - (1.0 + norm2)).abs() < 1e-10,
            "leading eigenvalue should be 1 + |w|^2 = {}; got {}",
            1.0 + norm2,
            res.eigenvalues[0]
        );
        for lam in &res.eigenvalues[1..] {
            assert!(
                (lam - 1.0).abs() < 1e-10,
                "remaining spectrum should be 1; got {lam}"
            );
        }
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
        assert_eq!(
            solver_blas_threads_with(get),
            2,
            "Lanczos-specific var must win over umbrella"
        );
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
        assert_eq!(
            inside, 1,
            "rayon-worker guard must force 1 even with own var set"
        );
    }

    fn make_symmetric_with_spectrum(lambdas: &[f64], seed: u64) -> Array2<f64> {
        let n = lambdas.len();
        // Build a random orthogonal U via QR of a deterministic pseudo-random matrix.
        // Use a simple LCG so tests don't depend on rand crate.
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
        let mut lambdas = vec![
            0.01, 0.1, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 2.0, 5.0, 10.0,
        ];
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

        let result = run_lanczos_seeded(seed, |v| a.dot(v), 4, 100, 1e-10, false).unwrap();

        // Expected top 4 by |λ − 1| descending: 10.0, 5.0, 0.01, 2.0 (|9|, |4|, |0.99|, |1|)
        // Sort recovered for stable comparison by |λ − 1| descending.
        let mut got = result.eigenvalues.clone();
        got.sort_by(|a, b| (b - 1.0).abs().total_cmp(&(a - 1.0).abs()));
        let mut want: Vec<f64> = vec![10.0, 5.0, 2.0, 0.01];
        want.sort_by(|a, b| (b - 1.0f64).abs().total_cmp(&(a - 1.0f64).abs()));
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < 1e-8,
                "Lanczos eigenvalue mismatch: got {g}, want {w}"
            );
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
        // FERRIC_LANCZOS_PANEL is process-global; serialize against every
        // other test in this crate that sets/reads a budget-family env var
        // (see TEST_BUDGET_ENV_LOCK's doc in lib.rs).
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
            false,
        )
        .unwrap();

        // Sanity: identity-seed Lanczos itself ≡ dense eigh of A.
        let (theta_dense, _) = a.eigh(UPLO::Upper).unwrap();
        let mut dense_sorted = theta_dense.to_vec();
        dense_sorted.sort_by(|x, y| x.total_cmp(y));

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
            got_sorted.sort_by(|x, y| x.total_cmp(y));
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
            assert!(
                new_res.converged,
                "panel {panel}: full-rank must report converged=true"
            );
            assert_eq!(
                new_res.max_resid, 0.0,
                "panel {panel}: full-rank residual must be 0"
            );
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
            false,
        )
        .unwrap();
        assert!(
            res.converged,
            "expected convergence with a generous iteration budget"
        );
        assert!(
            res.max_resid < 1e-8,
            "expected a small residual, got {}",
            res.max_resid
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
            false,
        )
        .unwrap();
        assert!(
            !res.converged,
            "expected max_iter=1 on a narrow, non-spanning seed to be flagged unconverged"
        );
        assert!(
            res.max_resid > 1e-14,
            "unconverged result should carry a residual above the (unreachable) tolerance, \
             got {}",
            res.max_resid
        );
    }

    /// Regression for the 2026-07-21 fix: `lanczos_panel_width` used to call
    /// `resolve_budget(None)` unconditionally, so a caller-supplied explicit
    /// budget (e.g. a TOML `[memory] budget_gb`, threaded through
    /// `run_lanczos_full_rank_budgeted`) was invisible — only an
    /// environment-variable-sourced budget counted as "explicit," so the panel
    /// always fell back to the fixed 256-column default regardless of what a
    /// TOML/kwarg budget requested. This does not read/set any process env var
    /// (would race with the other env-sensitive tests in this file) —  it
    /// tests purely that passing `Some(explicit_bytes)` changes the resolved
    /// panel width relative to `None`, at a big enough naux/nov that the
    /// legacy 256-column default and a budget-derived width can actually
    /// differ.
    #[test]
    fn lanczos_panel_width_honors_explicit_budget_argument() {
        // This test's "no explicit-or-env budget" branch asserts the
        // default 256-column panel, which requires NO budget-family env var
        // (FERRIC_MEM_BUDGET_GB / FERRIC_LANCZOS_PANEL / legacy vars) set
        // anywhere in the process. Other tests in this crate (ao_rpa,
        // full_rank_matches_identity_seed_lanczos, rs_mp2_rpa) mutate those
        // process-global vars; without this lock this test intermittently
        // saw a stale env value from a concurrently-running test and failed
        // (observed twice under parallel `cargo test`). See
        // TEST_BUDGET_ENV_LOCK's doc in lib.rs.
        let _env_guard = crate::TEST_BUDGET_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // The lock alone is NOT enough: a Mutex serialises threads within one
        // PROCESS, and nextest is process-per-test. See BudgetEnvCleared's doc
        // in lib.rs for the measurement (986 vs 256 under ci.yml's
        // FERRIC_MEM_BUDGET_GB=3) and why this must be enforced, not hoped for.
        let _restore = crate::BudgetEnvCleared::new();

        let naux = 4000;
        let nov = 200_000;

        // No explicit budget, no env var set: legacy fixed 256-column default.
        let default_panel = lanczos_panel_width(naux, nov, None);
        assert_eq!(
            default_panel, 256,
            "with no explicit-or-env budget, panel width must stay at the legacy 256 default"
        );

        // A tiny explicit budget must shrink the panel well below 256 (proves
        // the explicit value is actually reaching the budget-derived branch,
        // not silently falling through to the 256 default as it did before
        // the fix).
        let tiny_budget_bytes = 4 * 1024 * 1024; // 4 MiB
        let tiny_panel = lanczos_panel_width(naux, nov, Some(tiny_budget_bytes));
        assert!(
            tiny_panel < default_panel,
            "an explicit 4 MiB budget should yield a panel narrower than the 256-col default, \
             got {tiny_panel} (default was {default_panel})"
        );
        assert!(tiny_panel >= 1, "panel width must never be zero");

        // A generous explicit budget must widen the panel beyond 256 (proves
        // the explicit path isn't just clamping down, it genuinely scales
        // with the supplied budget in both directions).
        let generous_budget_bytes = 8usize * 1024 * 1024 * 1024; // 8 GiB
        let generous_panel = lanczos_panel_width(naux, nov, Some(generous_budget_bytes));
        assert!(
            generous_panel > default_panel,
            "an explicit 8 GiB budget should yield a panel wider than the 256-col default, \
             got {generous_panel} (default was {default_panel})"
        );
        assert!(
            generous_panel <= naux,
            "panel width must never exceed naux, got {generous_panel} for naux={naux}"
        );
    }

    /// Narrow 4-column seed on a naux=40 dielectric operator: one Lanczos
    /// step (max_iter=1, two block iterations) reaches an 8-dim subspace.
    fn narrow_seed_run(n_desired: usize, conv_thresh: f64) -> LanczosResult {
        let naux = 40;
        let (matvec, _a) = make_dielectric_op(naux, 90, 11);
        let mut seed = Array2::<f64>::zeros((naux, 4));
        for j in 0..4 {
            for i in 0..naux {
                seed[(i, j)] = ((i + j * 13) as f64).cos();
            }
        }
        run_lanczos_seeded(seed, |v| matvec(v), n_desired, 1, conv_thresh, false).unwrap()
    }

    /// The bug this guards: a COUNT-starved solve (residual already below
    /// `conv_thresh`, too few pairs) used to be reported as a residual failure.
    #[test]
    fn count_starved_solve_is_not_blamed_on_the_residual() {
        let res = narrow_seed_run(40, 1.0e3);
        assert!(!res.converged);
        assert!(res.max_resid < 1.0e3, "setup: residual must pass the bar");
        assert_eq!((res.subspace_dim, res.n_desired), (8, 40));
        assert_eq!(
            lanczos_stall_cause(res.max_resid, 1.0e3, res.subspace_dim, res.n_desired),
            LanczosStallCause::CountStarved
        );
        let msg = lanczos_nonconvergence_message(&res, 1, 1.0e3);
        assert!(msg.contains("residual is FINE"), "{msg}");
        assert!(msg.contains("only 8 of the 40 eigenpairs"), "{msg}");
        assert!(
            msg.contains("stopped by max_iter=1, which was exhausted"),
            "{msg}"
        );
        assert!(!res.hit_ambient_cap);
        let summary = lanczos_stall_summary(&res, 1.0e3);
        assert!(summary.starts_with("count-starved"), "{summary}");
        assert!(
            summary.contains("stopped by max_iter after 2 block"),
            "{summary}"
        );
    }

    /// Enough pairs (8 >= 4) but an unreachable residual bar: the residual,
    /// and only the residual, is named.
    #[test]
    fn residual_stalled_solve_names_the_residual_not_the_count() {
        let res = narrow_seed_run(4, 1e-14);
        assert!(!res.converged);
        assert_eq!(
            lanczos_stall_cause(res.max_resid, 1e-14, res.subspace_dim, res.n_desired),
            LanczosStallCause::ResidualAboveThreshold
        );
        let msg = lanczos_nonconvergence_message(&res, 1, 1e-14);
        assert!(msg.contains("still above conv_thresh"), "{msg}");
        assert!(msg.contains("count is fine"), "{msg}");
        assert!(lanczos_stall_summary(&res, 1e-14).starts_with("residual above threshold;"));
    }

    /// Pseudo-random 5-wide seed on the naux=42 dielectric operator (seed 3).
    fn seed_42x5() -> Array2<f64> {
        let mut seed = Array2::<f64>::zeros((42, 5));
        for j in 0..5 {
            for i in 0..42 {
                seed[(i, j)] = ((i * 7 + j * 13) as f64).sin();
            }
        }
        seed
    }

    /// naux=42 with a 5-wide block: the Krylov space caps at 40 after 8 block
    /// iterations. Requesting exactly the cap (40 pairs) still iterates; the
    /// unreachable 1e-300 residual then stops it AT the cap, and the message
    /// must name the cap, not the never-reached max_iter=50.
    #[test]
    fn ambient_cap_stop_is_not_blamed_on_max_iter() {
        let (matvec, _a) = make_dielectric_op(42, 90, 3);
        let res = run_lanczos_seeded(seed_42x5(), |v| matvec(v), 40, 50, 1e-300, false).unwrap();
        assert!(!res.converged);
        assert_eq!((res.block_iterations, res.subspace_dim), (8, 40));
        assert_eq!(
            lanczos_stall_cause(res.max_resid, 1e-300, res.subspace_dim, res.n_desired),
            LanczosStallCause::ResidualAboveThreshold
        );
        let msg = lanczos_nonconvergence_message(&res, 50, 1e-300);
        assert!(res.hit_ambient_cap);
        assert!(
            msg.contains("ambient Krylov cap of 40 after 8 block"),
            "{msg}"
        );
        assert!(msg.contains("NOT max_iter=50"), "{msg}");
        assert!(!msg.contains("was exhausted"), "{msg}");
        assert!(!msg.contains("eigenpairs were requested"), "{msg}");
        let summary = lanczos_stall_summary(&res, 1e-300);
        assert!(
            summary.contains("stopped by the ambient Krylov cap after 8"),
            "{summary}"
        );
    }

    /// naux=42, block 5, 40 pairs: the ambient cap allows 8 block iterations.
    /// At `max_iter = 8` that cap binds one pass before `max_iter` would
    /// (`outer_iters == max_iter`, the `<=` boundary); at `max_iter = 7`
    /// `max_iter` binds. Both stop after 8 iterations, so only the solver's
    /// own loop bound can name the limit.
    fn cap_boundary_run(max_iter: usize) -> LanczosResult {
        let (matvec, _a) = make_dielectric_op(42, 90, 3);
        run_lanczos_seeded(seed_42x5(), |v| matvec(v), 40, max_iter, 1e-300, false).unwrap()
    }

    #[test]
    fn cap_binding_at_outer_iters_equal_max_iter_names_the_cap() {
        let res = cap_boundary_run(8);
        assert!(!res.converged);
        assert_eq!((res.block_iterations, res.subspace_dim), (8, 40));
        assert!(
            res.hit_ambient_cap,
            "outer_iters 8 <= max_iter 8: the cap binds"
        );
        let msg = lanczos_nonconvergence_message(&res, 8, 1e-300);
        assert!(
            msg.contains("ambient Krylov cap of 40 after 8 block"),
            "{msg}"
        );
        assert!(msg.contains("NOT max_iter=8"), "{msg}");
        assert!(!msg.contains("was exhausted"), "{msg}");
    }

    #[test]
    fn max_iter_binding_one_below_the_cap_names_max_iter() {
        let res = cap_boundary_run(7);
        assert!(!res.converged);
        assert_eq!((res.block_iterations, res.subspace_dim), (8, 40));
        assert!(
            !res.hit_ambient_cap,
            "outer_iters 8 == max_iter + 1: max_iter binds"
        );
        let msg = lanczos_nonconvergence_message(&res, 7, 1e-300);
        assert!(
            msg.contains("stopped by max_iter=7, which was exhausted"),
            "{msg}"
        );
        assert!(!msg.contains("ambient Krylov cap"), "{msg}");
    }

    /// Ascending spectrum of the dense `a`, for comparison against a solve.
    fn dense_spectrum(a: &Array2<f64>) -> Vec<f64> {
        let (theta, _) = a.eigh(UPLO::Upper).unwrap();
        let mut v = theta.to_vec();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Max |Δλ| (sorted) and max ||A v − λ v|| of `res` against the dense `a`.
    fn errors_vs_dense(res: &LanczosResult, a: &Array2<f64>) -> (f64, f64) {
        let mut got = res.eigenvalues.clone();
        got.sort_by(f64::total_cmp);
        let dl = got
            .iter()
            .zip(dense_spectrum(a))
            .map(|(g, d)| (g - d).abs())
            .fold(0.0, f64::max);
        let r = res
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(k, &lam)| {
                let v = res.eigenvectors.column(k).to_owned();
                let r = a.dot(&v) - lam * &v;
                r.dot(&r).sqrt()
            })
            .fold(0.0, f64::max);
        (dl, r)
    }

    /// THE DEFECT: all 42 pairs from a 5-wide block. The Krylov space caps at
    /// 40, so the iteration could never meet the count at any max_iter (it
    /// returned 40 pairs, `converged = false`). The request must now be solved
    /// densely: every pair, converged, equal to a dense eigh of the operator.
    #[test]
    fn full_request_with_non_dividing_block_converges_to_dense_eigh() {
        let (matvec, a) = make_dielectric_op(42, 90, 3);
        for max_iter in [1, 50] {
            let res =
                run_lanczos_seeded(seed_42x5(), |v| matvec(v), 42, max_iter, 1e-10, false).unwrap();
            assert!(res.converged, "max_iter {max_iter}");
            assert_eq!(res.eigenvalues.len(), 42, "max_iter {max_iter}");
            assert_eq!(res.eigenvectors.dim(), (42, 42));
            assert!(!res.hit_ambient_cap);
            let (dl, r) = errors_vs_dense(&res, &a);
            assert!(dl < 1e-12, "max_iter {max_iter}: max |dλ| {dl:.3e}");
            assert!(r < 1e-10, "max_iter {max_iter}: max residual {r:.3e}");
        }
    }

    /// A partial request above the cap (41 > 40) is equally unreachable and
    /// takes the same dense route; the PDEP ordering by |λ − 1| is kept.
    #[test]
    fn partial_request_above_the_krylov_cap_is_solved_densely() {
        let (matvec, a) = make_dielectric_op(42, 90, 3);
        let res = run_lanczos_seeded(seed_42x5(), |v| matvec(v), 41, 50, 1e-10, false).unwrap();
        assert!(res.converged);
        assert_eq!(res.eigenvalues.len(), 41);
        let dev: Vec<f64> = res.eigenvalues.iter().map(|l| (l - 1.0).abs()).collect();
        assert!(dev.windows(2).all(|w| w[0] >= w[1]), "{dev:?}");
        let mut want = dense_spectrum(&a);
        want.sort_by(|x, y| (y - 1.0).abs().total_cmp(&(x - 1.0).abs()));
        for (g, w) in res.eigenvalues.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12, "{g} vs {w}");
        }
    }

    #[test]
    fn krylov_cap_rounds_naux_down_to_whole_blocks() {
        assert_eq!(krylov_cap(42, 5), 40);
        assert_eq!(krylov_cap(84, 5), 80);
        assert_eq!(krylov_cap(84, 4), 84);
        assert_eq!(krylov_cap(40, 40), 40);
        assert_eq!(krylov_cap(40, 41), 0);
        assert_eq!(krylov_cap(40, 0), 40);
    }

    /// A count-starved solve that did NOT request every pair gets the
    /// "more iterations" advice, and the full-space note does not appear.
    #[test]
    fn partial_request_count_starved_advises_more_iterations() {
        let res = narrow_seed_run(20, 1.0e3);
        assert_eq!((res.subspace_dim, res.n_desired), (8, 20));
        let msg = lanczos_nonconvergence_message(&res, 1, 1.0e3);
        assert!(msg.contains("more block iterations"), "{msg}");
        assert!(!msg.contains("eigenpairs were requested"), "{msg}");
    }

    /// When all naux pairs are requested, max_vecs advice would be wrong.
    #[test]
    fn full_request_count_starved_does_not_advise_max_vecs() {
        let msg = lanczos_nonconvergence_message(&narrow_seed_run(40, 1.0e3), 1, 1.0e3);
        assert!(
            msg.contains("all 40 eigenpairs were requested; only a full Krylov"),
            "{msg}"
        );
        assert!(!msg.contains("max_vecs"), "{msg}");
        assert!(!msg.contains("more block iterations"), "{msg}");
    }

    /// naux=40, block 5: the cap equals naux, so the full-space solve converges
    /// on its last permitted pass and must NOT read as stopped by the cap.
    #[test]
    fn converged_full_space_solve_does_not_report_the_cap() {
        let naux = 40;
        let (matvec, _a) = make_dielectric_op(naux, 90, 3);
        let mut seed = Array2::<f64>::zeros((naux, 5));
        for j in 0..5 {
            for i in 0..naux {
                seed[(i, j)] = ((i * 7 + j * 13) as f64).sin();
            }
        }
        let res = run_lanczos_seeded(seed, |v| matvec(v), naux, 50, 1e-300, false).unwrap();
        assert!(res.converged);
        assert_eq!((res.block_iterations, res.subspace_dim), (8, 40));
        assert!(!res.hit_ambient_cap);
    }

    /// A non-finite residual gets its own text, not "NaN is above conv_thresh".
    #[test]
    fn nan_residual_message_says_non_finite() {
        let res = LanczosResult {
            eigenvalues: Vec::new(),
            eigenvectors: Array2::zeros((40, 0)),
            converged: false,
            max_resid: f64::NAN,
            subspace_dim: 8,
            n_desired: 4,
            block_iterations: 2,
            hit_ambient_cap: false,
        };
        let msg = lanczos_nonconvergence_message(&res, 1, 1e-6);
        assert!(msg.contains("non-finite residual (NaN)"), "{msg}");
        assert!(!msg.contains("above conv_thresh"), "{msg}");
        let summary = lanczos_stall_summary(&res, 1e-6);
        assert!(summary.starts_with("non-finite residual;"), "{summary}");
    }

    /// The shared residual reduction (both solver loops call it): max norm of
    /// the β-projected tails, and a NaN in ANY column, first or last, survives.
    #[test]
    fn max_ritz_residual_is_the_tail_max_and_keeps_nan() {
        let beta = Array2::<f64>::eye(2);
        let mut y = Array2::<f64>::zeros((3, 3));
        y[(1, 0)] = 3.0;
        y[(2, 0)] = 4.0;
        y[(2, 1)] = 1.0;
        assert_eq!(max_ritz_residual(&beta, &y, 1, 3), 5.0);
        y[(1, 2)] = f64::NAN;
        assert!(
            max_ritz_residual(&beta, &y, 1, 3).is_nan(),
            "NaN in last column"
        );
        let mut y0 = y.clone();
        y0[(1, 2)] = 0.0;
        y0[(2, 0)] = f64::NAN;
        assert!(
            max_ritz_residual(&beta, &y0, 1, 3).is_nan(),
            "NaN in first column"
        );
        assert_eq!(max_ritz_residual(&beta, &y, 1, 0), 0.0);
    }

    /// `f64::max` drops NaN; the residual reduction must not, or a NaN Ritz
    /// residual reads as 0 and passes `max_resid < conv_thresh`. (Not driven
    /// end to end: a NaN-poisoned T makes LAPACK's eigh fail first, so the
    /// solve errors before any residual is formed.)
    #[test]
    fn nan_ritz_residual_is_propagated_not_dropped() {
        assert!(nan_max(0.0, f64::NAN).is_nan());
        assert!(nan_max(f64::NAN, 1e-12).is_nan());
        assert_eq!(nan_max(1e-9, 1e-12), 1e-9);
    }

    /// The classifier mirrors the solver gate `max_resid < conv_thresh &&
    /// subspace_dim >= n_desired` exactly, boundaries included.
    #[test]
    fn stall_cause_classifier_matches_the_convergence_gate() {
        use LanczosStallCause::*;
        assert_eq!(lanczos_stall_cause(1e-9, 1e-6, 4, 8), CountStarved);
        assert_eq!(
            lanczos_stall_cause(1e-3, 1e-6, 8, 8),
            ResidualAboveThreshold
        );
        assert_eq!(lanczos_stall_cause(1e-3, 1e-6, 4, 8), ResidualAndCount);
        assert_eq!(lanczos_stall_cause(1e-9, 1e-6, 8, 8), Inconsistent);
        // `<` on the residual: equality is NOT converged.
        assert_eq!(
            lanczos_stall_cause(1e-6, 1e-6, 8, 8),
            ResidualAboveThreshold
        );
        // `>=` on the count: equality IS enough (the Inconsistent row above),
        // one short is not.
        assert_eq!(lanczos_stall_cause(1e-9, 1e-6, 7, 8), CountStarved);
        // A NaN residual never passes.
        assert_eq!(lanczos_stall_cause(f64::NAN, 1e-6, 8, 8), NonFiniteResidual);
        assert_eq!(lanczos_stall_cause(f64::NAN, 1e-6, 4, 8), NonFiniteResidual);
        assert_eq!(
            lanczos_stall_cause(f64::INFINITY, 1e-6, 8, 8),
            NonFiniteResidual
        );
    }
}
