//! RPA correlation energy integration via trace-log formula.
//!
//! E_c^RPA = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))].

use crate::channel::RpaChannel;
use ferric_core::FerricError;
use ndarray::Array2;

/// Map an eigh/inv failure on the (projected) dielectric to a clean error.
/// A NaN/Inf dielectric — e.g. from a near-zero occ/vir gap poisoning χ₀ —
/// must surface as `Err`, not abort the process from inside a rayon worker.
/// `pub(crate)` (not private): `crate::mpi_rpa`'s MPI-distributed frequency
/// loop reuses this exact error-mapping so its error messages match the
/// serial path verbatim.
pub(crate) fn dielectric_lapack_err(what: &str, e: impl std::fmt::Display) -> FerricError {
    FerricError::Lapack(format!(
        "{what} (NaN/Inf dielectric from near-degenerate reference?): {e}"
    ))
}

/// Shared per-frequency parallel scaffold: evaluate `f` at every quadrature
/// frequency (build ε̃(iω) → eigh or inv), order-preserving `par_iter` +
/// `Result`-collect. BLAS is pinned to 1 inside the rayon region — the
/// per-frequency eigh/inv/GEMM must not nest OpenBLAS threads under rayon
/// workers (stack-overflow crash site). `f` returns `Result` so a NaN/Inf
/// dielectric surfaces as `Err`, never a panic inside a worker.
///
/// `init` builds per-WORKER scratch reused across the frequencies that worker
/// processes (`map_init`); scratch-free callers pass `|| ()`.
/// `eval_eigenvalues_at_frequencies_budgeted` also builds on this: its
/// panel-width scratch sizing happens from the memory budget BEFORE calling
/// this function, then the resulting shape is what `init` allocates per
/// worker — the sizing step doesn't need its own parallel scaffold.
fn per_frequency<S: Send, T: Send>(
    quad_freqs: &[f64],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, f64) -> Result<T, FerricError> + Sync + Send,
) -> Result<Vec<T>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use rayon::prelude::*;
    with_blas_threads(1, || {
        quad_freqs
            .par_iter()
            .map_init(&init, |scratch, &omega| f(scratch, omega))
            .collect::<Result<Vec<T>, FerricError>>()
    })
}

/// Assemble per-frequency eigenvalue rows into the (N_quad, M) tensor consumed
/// by [`rpa_correlation_energy`].
fn rows_into_array(n_quad: usize, m: usize, rows: Vec<Vec<f64>>) -> Array2<f64> {
    let mut out = Array2::zeros((n_quad, m));
    for (k, row) in rows.into_iter().enumerate() {
        for (alpha, val) in row.into_iter().enumerate() {
            out[(k, alpha)] = val;
        }
    }
    out
}

/// Below this `|x| = |λ − 1|` [`trace_log_summand_term`] uses the Taylor
/// series; at or above it, the literal `ln λ − x`.
const TRACE_LOG_SERIES_CUTOFF: f64 = 0.25;

/// Highest power of `x` kept in the series branch. The first omitted term,
/// `x³¹/31`, is `(2/31)·|x|²⁹ ≤ 2.0e-19` relative to the leading `x²/2` at the
/// cutoff, far below `f64` epsilon.
const TRACE_LOG_SERIES_ORDER: i32 = 30;

/// One RPA trace-log summand `f(λ) = ln λ + (1 − λ)`, evaluated without the
/// cancellation the literal `lam.ln() + (1.0 - lam)` suffers as `λ → 1`.
///
/// With `λ = 1 + x` the exact value is
/// `f = ln(1+x) − x = −x²/2 + x³/3 − x⁴/4 + …`, of size `x²/2`. The literal
/// form subtracts two quantities of size `x` (the rounding of `ln λ` alone is
/// `~u·|x|`), so its relative error grows as `~u/|x|` (`u = 2⁻⁵³`):
/// `~1e-8` at `x = 1e-8`, `~1e-2` at `x = 1e-14`. That regime is real: at
/// high quadrature frequency, and for the weakly-screened modes at every
/// frequency, the dielectric eigenvalues cluster at 1.
///
/// * `|x| < 0.25`: the series through `x³⁰`, by Horner. Successive terms
///   shrink by a factor `≥ 4`, so there is no subtraction of near-equal
///   quantities. `ln_1p` alone does NOT fix this regime — `ln_1p(x) − x`
///   still cancels.
/// * `|x| ≥ 0.25`: `ln λ − x`, bit-identical to the literal form
///   (`−(λ − 1) == 1 − λ` exactly under round-to-nearest). Here the result is
///   well conditioned (at most ~9 ulp of cancellation, at `λ = 1.25`). It
///   deliberately does NOT use `ln_1p(x)`: for `λ < 0.5`, `x = λ − 1` is
///   itself rounded, and `ln_1p` of the rounded `x` loses the low digits of
///   `λ` (measured rel err 4.3e-13 at `λ = 1e-5`, 3.8e-9 at `λ = 1e-10`)
///   where `ln λ` reads `λ` directly.
///
/// Measured worst relative error vs a 300-bit reference: 2.1e-16 on the
/// series branch, 3.9e-16 on the `ln λ − x` branch (the table in
/// `trace_log_summand_term_matches_high_precision_reference`).
///
/// The helper cannot restore digits the eigensolver already discarded when it
/// rounded `1 + x`; it removes the cancellation in the summand only.
#[inline]
pub fn trace_log_summand_term(lam: f64) -> f64 {
    let x = lam - 1.0;
    if x.abs() < TRACE_LOG_SERIES_CUTOFF {
        // f = x² · Σ_{k=0}^{N−2} (−1)^{k+1} x^k / (k+2), Horner from the top.
        let mut acc = 0.0f64;
        for k in (0..=TRACE_LOG_SERIES_ORDER - 2).rev() {
            let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
            acc = acc.mul_add(x, sign / f64::from(k + 2));
        }
        acc * (x * x)
    } else {
        lam.ln() - x
    }
}

/// `Σ_α [ln λ_α + (1 − λ_α)]` over one frequency's dielectric eigenvalues,
/// each summand via [`trace_log_summand_term`]. The one entry point every
/// eigenvalue-route RPA/dRPA energy uses.
pub fn trace_log_sum<'a>(eigenvalues: impl IntoIterator<Item = &'a f64>) -> f64 {
    eigenvalues
        .into_iter()
        .map(|&lam| trace_log_summand_term(lam))
        .sum()
}

/// E_c^RPA = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))].
///
/// Computes RPA correlation energy from eigenvalues of the dielectric matrix
/// evaluated at imaginary frequencies (quadrature points), via
/// [`trace_log_sum`].
///
/// # Arguments
/// * `quad_weights` - Gauss-Legendre or other quadrature weights w_k, length N_quad
/// * `eigenvalues_freq` - Eigenvalues λ_α(iω_k) of shape (N_quad, M)
///
/// # Returns
/// RPA correlation energy in Hartree.
pub fn rpa_correlation_energy(quad_weights: &[f64], eigenvalues_freq: &Array2<f64>) -> f64 {
    let n_quad = quad_weights.len();
    assert_eq!(eigenvalues_freq.nrows(), n_quad);

    let mut e_c = 0.0f64;
    for (k, &wk) in quad_weights.iter().enumerate() {
        e_c += wk * trace_log_sum(eigenvalues_freq.row(k));
    }
    e_c / (2.0 * std::f64::consts::PI)
}

/// Evaluate λ_α(iω_k) for each quadrature point given converged eigenpotentials.
///
/// Uses the diagonal of the projected dielectric matrix as λ_α(iω_k).
///
/// # Arguments
/// * `eigenvectors` - Converged eigenvectors V_α from PDEP, shape (naux, M)
/// * `b_ov` - RI-MO tensor B^P_ia, shape (naux, nocc*nvir)
/// * `eps_occ` - Occupied orbital energies
/// * `eps_vir` - Virtual orbital energies
/// * `quad_freqs` - Imaginary quadrature frequencies ω_k
///
/// # Returns
/// Array2 of shape (N_quad, M) containing eigenvalues λ_α(iω_k).
/// Laplace-separable variant of [`eval_eigenvalues_at_frequencies`].
///
/// Uses the χ₀ kernel from [`crate::laplace_chi0`] in place of the dense
/// `dielectric_matrix`. The output shape and convention are identical.
pub fn eval_eigenvalues_at_frequencies_laplace(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    laplace: &ferric_quadrature::LaplaceQuadrature,
) -> Result<Array2<f64>, FerricError> {
    use crate::laplace_chi0::dielectric_matrix_laplace_into;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();
    let nov = eps_occ.len() * eps_vir.len();

    // Per-worker (rhs_scaled, out) scratch reused across the frequencies that
    // worker processes, instead of allocating an (m×nov) and (m×m) buffer per
    // quadrature point.
    let rows = per_frequency(
        quad_freqs,
        || (Array2::<f64>::zeros((m, nov)), Array2::<f64>::zeros((m, m))),
        |(rhs_scaled, out), omega| {
            dielectric_matrix_laplace_into(
                eigenvectors,
                b_ov,
                eps_occ,
                eps_vir,
                omega,
                laplace,
                rhs_scaled,
                out,
            );
            let evals = ferric_core::linalg::eigvalsh_dc(out, ferric_core::linalg::Uplo::Upper)
                .map_err(|e| dielectric_lapack_err("Laplace dielectric eigh failed", e))?;
            Ok(evals)
        },
    )?;
    Ok(rows_into_array(n_quad, m, rows))
}

/// Unrestricted variant: per-frequency eigenvalues of ε̃_U = I + Π_α + Π_β.
///
/// Diagonalizes the projected unrestricted dielectric at each ω_k.
/// Output shape (N_quad, M) is identical to the closed-shell path so
/// `rpa_correlation_energy` consumes it unchanged.
pub fn eval_eigenvalues_at_frequencies_unrestricted(
    eigenvectors: &Array2<f64>,
    chan_a: &RpaChannel,
    chan_b: &RpaChannel,
    quad_freqs: &[f64],
) -> Result<Array2<f64>, FerricError> {
    use crate::sternheimer::dielectric_matrix_unrestricted;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();

    let rows = per_frequency(
        quad_freqs,
        || (),
        |(), omega| {
            let eps_proj = dielectric_matrix_unrestricted(eigenvectors, chan_a, chan_b, omega);
            let evals =
                ferric_core::linalg::eigvalsh_dc(&eps_proj, ferric_core::linalg::Uplo::Upper)
                    .map_err(|e| dielectric_lapack_err("unrestricted dielectric eigh failed", e))?;
            Ok(evals)
        },
    )?;
    Ok(rows_into_array(n_quad, m, rows))
}

/// Laplace + unrestricted variant of [`eval_eigenvalues_at_frequencies`].
///
/// Builds `ε̃ = I + Π_α + Π_β` at each ω_k using the Laplace-separable
/// kernel per spin, diagonalizes, returns eigenvalue tensor.
pub fn eval_eigenvalues_at_frequencies_laplace_unrestricted(
    eigenvectors: &Array2<f64>,
    chan_a: &RpaChannel,
    laplace_a: &ferric_quadrature::LaplaceQuadrature,
    chan_b: &RpaChannel,
    laplace_b: &ferric_quadrature::LaplaceQuadrature,
    quad_freqs: &[f64],
) -> Result<Array2<f64>, FerricError> {
    use crate::laplace_chi0::dielectric_matrix_laplace_unrestricted;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();

    let rows = per_frequency(
        quad_freqs,
        || (),
        |(), omega| {
            let eps_proj = dielectric_matrix_laplace_unrestricted(
                eigenvectors,
                chan_a,
                laplace_a,
                chan_b,
                laplace_b,
                omega,
            );
            let evals =
                ferric_core::linalg::eigvalsh_dc(&eps_proj, ferric_core::linalg::Uplo::Upper)
                    .map_err(|e| dielectric_lapack_err("U-Laplace dielectric eigh failed", e))?;
            Ok(evals)
        },
    )?;
    Ok(rows_into_array(n_quad, m, rows))
}

/// Evaluate PDEP eigenvalues at each quadrature frequency by projecting onto eigenvectors.
pub fn eval_eigenvalues_at_frequencies(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
) -> Result<Array2<f64>, FerricError> {
    eval_eigenvalues_at_frequencies_budgeted(eigenvectors, b_ov, eps_occ, eps_vir, quad_freqs, None)
}

/// Choose the per-worker `(m, k)` scratch panel width `k` for the frequency-
/// quadrature loop, given a resolved memory budget. `None` (no budget
/// configured) or a budget that already comfortably covers the full-width
/// `n_workers·(m·nov + m²)·8` footprint returns `nov` unchanged — the full-
/// width fast path — so every existing bit-for-bit regression test
/// (`eval_eigenvalues_at_frequencies_matches_manual_reference`,
/// `dielectric_matrix_from_projection_into_matches_allocating_version`) is
/// completely unaffected unless panelling is actually needed to fit budget.
///
/// "Comfortably covers": reserves the SAME per-worker footprint accounting
/// `budget.rs::estimate_peak_bytes` uses for this term
/// (`n_workers·(m·nov + m²)·8`), plus the shared frequency-independent
/// projection `y` (`m·nov·8`, held once for the whole loop) and the eigh
/// scratch LAPACK needs internally (folded into a small safety margin) —
/// deliberately conservative so a job that already fits is never routed onto
/// the (slower) panelled path.
///
/// k-term accounting (reconciled after `dielectric_matrix_from_projection_into_panelled`
/// stopped copying every panel — see that function's doc comment): each
/// worker's *persistent* live set is `(m·k + m²)·8` — the `rhs_scaled_panel`
/// scratch plus the `(m,m)` `out` accumulator. A full-width panel (`w ==
/// panel_width`, the common case for every panel but a possible ragged last
/// one) is passed to `syrk_aat_accumulate_into` as a view with NO extra copy,
/// so it never exceeds that persistent footprint. Only the ragged tail
/// (`w < panel_width`) transiently owns one extra `(m, w)` buffer via
/// `to_owned()`, and `w < panel_width` by construction, so that transient
/// never exceeds the persistent `m·k` term either — sizing for a single
/// (not doubled) `m·k` term is therefore exact, not an underestimate.
/// Test-only re-export of [`quad_panel_width`] for the budget-contract MWEs in
/// `tests/mwe_panel_width_budget.rs`, which sweep worker counts EXPLICITLY
/// rather than inheriting the ambient rayon pool.
///
/// Note the deliberate contrast with the two in-tree tests in this file
/// (`..._forced_panel_matches_manual_reference` and
/// `logdet_energy_matches_eigenvalue_energy_under_forced_panelling`): those
/// exercise the PRODUCTION path, which computes `n_workers` internally from
/// `rayon::current_num_threads()`, so pinning a constant there would assert
/// against a different `k` than the code actually uses — there the fix is the
/// BUDGET, not the thread count. This accessor is for the direct-call contracts,
/// where sweeping the worker count is the whole point: the defect it guards
/// against is proportional to worker count, so a test that cannot vary it
/// cannot see it.
#[doc(hidden)]
pub fn quad_panel_width_for_test(
    m: usize,
    nov: usize,
    n_workers: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    quad_panel_width(m, nov, n_workers, memory_budget_bytes)
}

fn quad_panel_width(
    m: usize,
    nov: usize,
    n_workers: usize,
    memory_budget_bytes: Option<usize>,
) -> usize {
    let Some(budget) = memory_budget_bytes else {
        return nov.max(1);
    };
    if nov == 0 {
        return 1;
    }
    let n_workers = n_workers.max(1);
    let y_bytes = m.saturating_mul(nov).saturating_mul(8);
    let full_width_scratch = n_workers
        .saturating_mul(m.saturating_mul(nov).saturating_add(m.saturating_mul(m)))
        .saturating_mul(8);
    if y_bytes.saturating_add(full_width_scratch) <= budget {
        return nov; // fast path: full width already fits
    }
    // Panelling needed: solve n_workers·(m·k + m²)·8 + y_bytes <= budget for k,
    // per worker, leaving headroom for the always-resident y and the (m,m)
    // `out` term (m² is independent of k, so it's subtracted off first). This
    // single `m·k` term is exact (not an underestimate) — see the k-term
    // accounting note above.
    let remaining = budget.saturating_sub(y_bytes);
    let per_worker_budget = remaining / n_workers.max(1);
    let out_bytes = m.saturating_mul(m).saturating_mul(8);
    let scratch_for_k = per_worker_budget.saturating_sub(out_bytes);
    let per_col_bytes = m.saturating_mul(8).max(1);
    (scratch_for_k / per_col_bytes).clamp(1, nov)
}

/// Budget-aware sibling of [`eval_eigenvalues_at_frequencies`]. Identical
/// result to the plain function (which is now a thin `None`-budget wrapper
/// around this one); the only behavioral difference is which BLAS assembly
/// path (`dielectric_matrix_from_projection_into` vs the nov-panelled
/// `..._into_panelled`) each rayon worker's `map_init` closure calls, chosen
/// once via `quad_panel_width` before the parallel region starts.
///
/// `memory_budget_bytes`: the caller's resolved `[memory] budget_gb` /
/// `PdepRpaConfig::memory_budget_bytes` (already resolved via
/// `ferric_core::memory::resolve_budget_bytes` by the caller — this function
/// takes the final byte ceiling, not an `Option` needing further resolution,
/// mirroring `lanczos::run_lanczos_full_rank_budgeted`'s convention of
/// accepting the caller's own resolved value rather than re-resolving).
pub fn eval_eigenvalues_at_frequencies_budgeted(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    memory_budget_bytes: Option<usize>,
) -> Result<Array2<f64>, FerricError> {
    use crate::sternheimer::{
        build_scale_factors, dielectric_matrix_from_projection_into,
        dielectric_matrix_from_projection_into_panelled,
    };
    use ndarray::Array2;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();
    let nov = eps_occ.len() * eps_vir.len();

    // The projection y = Vᵀ·B_ov is frequency-independent — compute it ONCE
    // instead of recomputing the (m × nov) GEMM at every quadrature point. Each
    // ω then only does the cheap column scaling + DSYRK.
    let y = eigenvectors.t().dot(b_ov);

    // Panel-width sizing happens BEFORE the parallel region (needs only
    // n_workers, not any per-frequency state), so it can be computed here and
    // captured by `per_frequency`'s `init` closure — the reason this function
    // used to hand-roll its own with_blas_threads/par_iter/map_init scaffold
    // (rather than sharing `per_frequency`) no longer holds.
    let n_workers = rayon::current_num_threads().max(1);
    let panel_width = quad_panel_width(m, nov, n_workers, memory_budget_bytes);
    let use_panelled = panel_width < nov;

    // map_init (not map): each rayon WORKER allocates its own (rhs_scaled, out)
    // scratch ONCE and reuses it across every frequency that worker processes,
    // instead of `dielectric_matrix_from_projection` cloning the full (m, nov)
    // `y` on every single call. At benzene/aug-cc-pVQZ scale (naux=m≈2976,
    // nov≈61740) that clone is ~1.4 GB; with the old `map` + per-call clone,
    // up to min(n_quad, active_threads) copies were live simultaneously —
    // multi-GB, unbounded by `[memory] budget_gb`, and scaling with core count
    // (found 2026-07-21 tracing a benzene aQZ OOM). map_init caps this at one
    // (rhs_scaled, out) pair per THREAD, not per frequency.
    //
    // When `use_panelled` is false (no budget configured, or the full-width
    // footprint already fits — see `quad_panel_width`), this is the EXACT
    // pre-existing full-width path (`rhs_scaled` sized (m, nov)), so every
    // existing bit-for-bit regression test is unaffected. Only when panelling
    // is actually needed does each worker's scratch narrow to (m, panel_width).
    let rows = per_frequency(
        quad_freqs,
        || {
            (
                Array2::<f64>::zeros((m, panel_width)),
                Array2::<f64>::zeros((m, m)),
            )
        },
        |(rhs_scaled, out), omega| {
            let scale = build_scale_factors(eps_occ, eps_vir, omega);
            if use_panelled {
                dielectric_matrix_from_projection_into_panelled(&y, &scale, rhs_scaled, out);
            } else {
                dielectric_matrix_from_projection_into(&y, &scale, rhs_scaled, out);
            }
            // Diagonalize at each frequency — eigenvectors at ω=0 don't diagonalize ε̃(iω).
            // Divide-and-conquer eigenvalue-only solver (dsyevd_): same eigenvalues
            // as `.eigh()` (validated bit-close in ferric_core::linalg tests), faster.
            let evals = ferric_core::linalg::eigvalsh_dc(out, ferric_core::linalg::Uplo::Upper)
                .map_err(|e| dielectric_lapack_err("dielectric eigh failed", e))?;
            Ok(evals)
        },
    )?;

    Ok(rows_into_array(n_quad, m, rows))
}

/// Per-frequency RPA trace-log summand `S(iω) = ln det(ε̃) + tr(I − ε̃)`,
/// computed WITHOUT diagonalizing — the LU-determinant route.
///
/// `S(iω) ≡ Σ_α [ln λ_α(iω) + (1 − λ_α(iω))]`, the exact quantity
/// [`rpa_correlation_energy`] sums over α, because both terms are
/// basis-invariant traces: `Σ_α ln λ_α = ln det(ε̃)` and
/// `Σ_α (1 − λ_α) = M − tr(ε̃)`. This is the formulation PySCF uses
/// (`pyscf/gw/rpa.py`).
///
/// Cheaper than the eigenvalue route: `dgetrf_` is a blocked BLAS3 LU
/// (~2/3·M³) while eigenvalue-only `dsyevd_` pays a memory-bound BLAS2
/// tridiagonal reduction (~4/3·M³) — measured 3.7–6.2× faster on the
/// kernel alone at M = 64…1024 (`ferric-core`'s `logdet_vs_eigvals_bench`).
///
/// `ε̃` is symmetric, but that is irrelevant here: `logdet_lu` reads the whole
/// matrix and is transpose-invariant, so the row-major buffer needs no
/// transposition (see [`ferric_core::linalg::logdet_lu`]'s layout note).
///
/// This route carries the same `λ → 1` cancellation that
/// [`trace_log_summand_term`] removes from the eigenvalue route, aggregated
/// over modes: `ln det ≈ Σ_α x_α` and `M − tr = −Σ_α x_α` are two independently
/// rounded sums whose difference is `≈ −Σ_α x_α²/2`. With no eigenvalues there
/// is no per-mode structure to expand, so it is not reformulated. The absolute
/// error per mode is `~u·x_α`, negligible in f64; a reduced-precision (f32)
/// evaluation of this route would not be.
fn dielectric_trace_log_summand(eps_proj: &Array2<f64>, what: &str) -> Result<f64, FerricError> {
    let m = eps_proj.nrows();
    let log_det =
        ferric_core::linalg::logdet_lu(eps_proj).map_err(|e| dielectric_lapack_err(what, e))?;
    let trace: f64 = (0..m).map(|i| eps_proj[(i, i)]).sum();
    Ok(log_det + (m as f64 - trace))
}

/// Integrate the RPA correlation energy from per-frequency trace-log summands.
///
/// `E_c = (1/2π) Σ_k w_k S(iω_k)`. Companion to [`rpa_correlation_energy`],
/// which takes the eigenvalue tensor instead; the two agree to round-off.
pub fn rpa_correlation_energy_from_summands(quad_weights: &[f64], summands: &[f64]) -> f64 {
    assert_eq!(quad_weights.len(), summands.len());
    let e_c: f64 = quad_weights
        .iter()
        .zip(summands)
        .map(|(&w, &s)| w * s)
        .sum();
    e_c / (2.0 * std::f64::consts::PI)
}

/// Log-det sibling of [`eval_eigenvalues_at_frequencies_budgeted`]: returns the
/// per-frequency trace-log summands `S(iω_k)` (length `N_quad`) directly,
/// skipping the eigendecomposition entirely.
///
/// Use this when ONLY the correlation energy is wanted. It is NOT a drop-in
/// replacement for the eigenvalue evaluator: `PdepRpaResult.eigenvalues_freq`
/// is a public field (Python `eigenvalues_freq` getter, MPI banding regression
/// test) and the GW paths shape-check it, so the eigenvalue route stays.
/// Everything up to the diagonalization — the frequency-independent `y`
/// projection, the panel-width sizing, the per-worker scratch reuse, the
/// `with_blas_threads(1)` guard — is identical to that sibling by construction.
pub fn eval_trace_log_summands_budgeted(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    memory_budget_bytes: Option<usize>,
) -> Result<Vec<f64>, FerricError> {
    use crate::sternheimer::{
        build_scale_factors, dielectric_matrix_from_projection_into,
        dielectric_matrix_from_projection_into_panelled,
    };

    let m = eigenvectors.ncols();
    let nov = eps_occ.len() * eps_vir.len();
    let y = eigenvectors.t().dot(b_ov);

    let n_workers = rayon::current_num_threads().max(1);
    let panel_width = quad_panel_width(m, nov, n_workers, memory_budget_bytes);
    let use_panelled = panel_width < nov;

    per_frequency(
        quad_freqs,
        || {
            (
                Array2::<f64>::zeros((m, panel_width)),
                Array2::<f64>::zeros((m, m)),
            )
        },
        |(rhs_scaled, out), omega| {
            let scale = build_scale_factors(eps_occ, eps_vir, omega);
            if use_panelled {
                dielectric_matrix_from_projection_into_panelled(&y, &scale, rhs_scaled, out);
            } else {
                dielectric_matrix_from_projection_into(&y, &scale, rhs_scaled, out);
            }
            dielectric_trace_log_summand(out, "dielectric log-det failed")
        },
    )
}

/// Per-frequency *dynamic inverse-dielectric* matrices in the PDEP basis.
///
/// Returns `W̃_d(iω_k) = ε̃_proj(iω_k)⁻¹ − I` (shape M×M) for each quadrature
/// frequency, where `ε̃_proj = Uᵀ ε̃(iω) U` is the projected dielectric in the
/// static PDEP eigenvector basis `U` (same basis used to project B̃ in GW).
///
/// This is required by the GW self-energy: the scalar `eigenvalues_freq` tensor
/// (eigenvalues of ε̃(iω) in its *own* ω-dependent eigenbasis) is only valid for
/// basis-invariant traces like the RPA correlation energy. The GW Σ_c contracts
/// the *matrix* W̃_d(iω) with specific B̃_{mn} vectors, so it needs eigenvalues
/// and eigenvectors paired consistently — i.e. the full matrix in the fixed PDEP
/// basis, not diagonal weights computed in a per-ω rotated basis.
pub fn eval_inv_dielectric_matrices(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
) -> Result<Vec<Array2<f64>>, FerricError> {
    use crate::sternheimer::{build_scale_factors, dielectric_matrix_from_projection};
    use ndarray_linalg::Inverse;

    let m = eigenvectors.ncols();
    let y = eigenvectors.t().dot(b_ov);

    per_frequency(
        quad_freqs,
        || (),
        |(), omega| {
            let scale = build_scale_factors(eps_occ, eps_vir, omega);
            let eps_proj = dielectric_matrix_from_projection(&y, &scale);
            let mut winv = eps_proj
                .inv()
                .map_err(|e| dielectric_lapack_err("PDEP-basis dielectric inversion failed", e))?;
            // Subtract identity → dynamic part W̃_d = ε̃⁻¹ − I.
            for d in 0..m {
                winv[(d, d)] -= 1.0;
            }
            Ok(winv)
        },
    )
}

/// Unrestricted variant of [`eval_inv_dielectric_matrices`]: full per-frequency
/// `ε̃_U(iω)⁻¹ − I` (M×M) in the fixed PDEP basis, where ε̃_U = I + Π_α + Π_β.
pub fn eval_inv_dielectric_matrices_unrestricted(
    eigenvectors: &Array2<f64>,
    chan_a: &RpaChannel,
    chan_b: &RpaChannel,
    quad_freqs: &[f64],
) -> Result<Vec<Array2<f64>>, FerricError> {
    use crate::sternheimer::dielectric_matrix_unrestricted;
    use ndarray_linalg::Inverse;

    let m = eigenvectors.ncols();
    per_frequency(
        quad_freqs,
        || (),
        |(), omega| {
            let eps_proj = dielectric_matrix_unrestricted(eigenvectors, chan_a, chan_b, omega);
            let mut winv = eps_proj.inv().map_err(|e| {
                dielectric_lapack_err("U PDEP-basis dielectric inversion failed", e)
            })?;
            for d in 0..m {
                winv[(d, d)] -= 1.0;
            }
            Ok(winv)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_log_two_eigenvalues() {
        // E_c = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))]
        // 1 quad point w=1, λ1=2.0, λ2=0.5:
        // contrib = (ln(2)+1-2) + (ln(0.5)+1-0.5) = (0.693-1) + (-0.693+0.5) = -0.5
        // E_c = -0.5 / (2π)
        let quad_weights = vec![1.0f64];
        let eigenvalues_freq = ndarray::array![[2.0f64, 0.5]]; // (1 quad, 2 eigs)
        let e_c = rpa_correlation_energy(&quad_weights, &eigenvalues_freq);
        let expected = -0.5 / (2.0 * std::f64::consts::PI);
        assert!(
            (e_c - expected).abs() < 1e-10,
            "E_c = {} expected {}",
            e_c,
            expected
        );
    }

    /// Regression for the 2026-07-21 memory fix: `eval_eigenvalues_at_frequencies`
    /// was rewritten from `par_iter().map(...)` (allocating a fresh `y.clone()`
    /// per quadrature frequency via `dielectric_matrix_from_projection`) to
    /// `par_iter().map_init(...)` with per-worker reusable scratch buffers via
    /// `dielectric_matrix_from_projection_into`. This must produce numerically
    /// identical eigenvalues to a hand-computed reference on a small,
    /// non-trivial (m=2, nov=2, n_quad=3) case — not just unit-test the
    /// scratch-reuse helper in isolation (sternheimer.rs already covers that),
    /// but confirm the whole rayon fan-out + eigh pipeline in THIS function
    /// still gives the right answer after the rewrite.
    #[test]
    fn eval_eigenvalues_at_frequencies_matches_manual_reference() {
        use ndarray_linalg::{Eigh, UPLO};

        // 2 aux "modes" (eigenvectors is naux x naux here, m=2), 1 occ x 2 vir
        // (nov=2). Trivial eigenvectors = identity so y = b_ov directly.
        let eigenvectors = ndarray::array![[1.0f64, 0.0], [0.0, 1.0]];
        let b_ov = ndarray::array![[1.0f64, 0.5], [0.3, -0.8]];
        let eps_occ = vec![-0.4f64];
        let eps_vir = vec![0.3f64, 0.9f64];
        let quad_freqs = vec![0.1f64, 0.5, 1.0];

        let got =
            eval_eigenvalues_at_frequencies(&eigenvectors, &b_ov, &eps_occ, &eps_vir, &quad_freqs)
                .expect("eval_eigenvalues_at_frequencies failed");

        // Manual reference: for each omega, build y = eigenvectors^T . b_ov
        // (= b_ov here since eigenvectors is identity), scale, SYRK, +I, eigh.
        let y = eigenvectors.t().dot(&b_ov);
        for (k, &omega) in quad_freqs.iter().enumerate() {
            let scale = crate::sternheimer::build_scale_factors(&eps_occ, &eps_vir, omega);
            let eps_mat = crate::sternheimer::dielectric_matrix_from_projection(&y, &scale);
            let (expected_evals, _) = eps_mat.eigh(UPLO::Upper).expect("reference eigh failed");
            for (alpha, &expected) in expected_evals.iter().enumerate() {
                let g = got[(k, alpha)];
                assert!(
                    (g - expected).abs() < 1e-12,
                    "quad point {k} (omega={omega}) eigenvalue {alpha}: got {g}, expected {expected}"
                );
            }
        }
    }

    /// `quad_panel_width` must return the full `nov` (fast path) when no
    /// budget is configured, or when the budget already comfortably covers
    /// the full-width footprint — this is what guarantees the pre-existing
    /// bit-for-bit tests above are unaffected by this change.
    #[test]
    fn quad_panel_width_fast_path_when_no_budget_or_ample_budget() {
        // No budget at all.
        assert_eq!(quad_panel_width(50, 1000, 4, None), 1000);
        // A very generous budget (way more than the full-width footprint needs).
        let ample = 64usize * 1024 * 1024 * 1024; // 64 GiB
        assert_eq!(quad_panel_width(50, 1000, 4, Some(ample)), 1000);
    }

    /// A tiny forced budget must narrow the panel well below `nov`, proving
    /// the budget-derived branch is actually reachable and shrinks the width.
    #[test]
    fn quad_panel_width_narrows_under_tiny_budget() {
        let m = 200;
        let nov = 50_000;
        let n_workers = 8;
        let tiny_budget = 8usize * 1024 * 1024; // 8 MiB — far too small for full width
        let k = quad_panel_width(m, nov, n_workers, Some(tiny_budget));
        assert!(
            k < nov,
            "expected a narrowed panel width, got {k} (nov={nov})"
        );
        assert!(k >= 1, "panel width must never be zero");
    }

    /// End-to-end: `eval_eigenvalues_at_frequencies_budgeted` with a small
    /// FORCED budget (small enough that quad_panel_width must narrow the
    /// panel well below nov) must produce the SAME eigenvalues as the manual
    /// reference used by `eval_eigenvalues_at_frequencies_matches_manual_reference`
    /// above — proving the panelled dielectric assembly path, exercised
    /// through the full rayon fan-out + eigh pipeline (not just the isolated
    /// sternheimer.rs helper), gives the right physics.
    #[test]
    fn eval_eigenvalues_at_frequencies_budgeted_forced_panel_matches_manual_reference() {
        use ndarray_linalg::{Eigh, UPLO};

        // Same small system as the manual-reference test above (m=2, nov=2),
        // but with a budget so tiny quad_panel_width is forced to k=1 (the
        // narrowest possible panel: nov=2, so any k<2 collapses to k=1).
        let eigenvectors = ndarray::array![[1.0f64, 0.0], [0.0, 1.0]];
        let b_ov = ndarray::array![[1.0f64, 0.5], [0.3, -0.8]];
        let eps_occ = vec![-0.4f64];
        let eps_vir = vec![0.3f64, 0.9f64];
        let quad_freqs = vec![0.1f64, 0.5, 1.0];

        // Sanity: confirm the tiny budget actually forces panelling for this
        // shape before trusting the result below (m=2, nov=2, n_workers>=1).
        //
        // The budget MUST be below the WORST-CASE (i.e. smallest) full-width
        // footprint, which occurs at n_workers=1: `quad_panel_width`'s fast
        // path triggers when y_bytes + n_workers·(m·nov + m²)·8 <= budget, and
        // that sum GROWS with n_workers. For m=nov=2 it is 32 + 64·n_workers,
        // so any budget >= 96 takes the fast path at n_workers=1 and returns
        // nov=2, never exercising the panelled branch this test exists to
        // cover. The old 200 did exactly that: it passed on a many-core box
        // (n_workers>=3) and failed at RAYON_NUM_THREADS=1..2 — an
        // ambient-thread-count dependency, not a physics bug. 64 < 96 forces
        // k=1 at every worker count. n_workers is NOT pinned here on purpose:
        // the production path computes it internally from
        // rayon::current_num_threads(), so a hardcoded value would assert
        // against a different k than the code under test actually uses.
        let n_workers = rayon::current_num_threads().max(1);
        let tiny_budget = 64usize; // bytes — below the n_workers=1 fast-path floor (96)
        let forced_k = quad_panel_width(2, 2, n_workers, Some(tiny_budget));
        assert_eq!(
            forced_k, 1,
            "expected the tiny budget to force k=1 for this tiny shape \
             (n_workers={n_workers}); if this fires, re-derive the budget against \
             quad_panel_width's fast-path condition"
        );

        let got = eval_eigenvalues_at_frequencies_budgeted(
            &eigenvectors,
            &b_ov,
            &eps_occ,
            &eps_vir,
            &quad_freqs,
            Some(tiny_budget),
        )
        .expect("eval_eigenvalues_at_frequencies_budgeted failed");

        // Manual reference: identical derivation to the non-budgeted test.
        let y = eigenvectors.t().dot(&b_ov);
        for (k, &omega) in quad_freqs.iter().enumerate() {
            let scale = crate::sternheimer::build_scale_factors(&eps_occ, &eps_vir, omega);
            let eps_mat = crate::sternheimer::dielectric_matrix_from_projection(&y, &scale);
            let (expected_evals, _) = eps_mat.eigh(UPLO::Upper).expect("reference eigh failed");
            for (alpha, &expected) in expected_evals.iter().enumerate() {
                let g = got[(k, alpha)];
                assert!(
                    (g - expected).abs() < 1e-12,
                    "forced-panel quad point {k} (omega={omega}) eigenvalue {alpha}: got {g}, expected {expected}"
                );
            }
        }
    }

    /// `eval_eigenvalues_at_frequencies` (the plain, `None`-budget entry
    /// point) must be byte-for-byte the same as calling
    /// `eval_eigenvalues_at_frequencies_budgeted` with `memory_budget_bytes:
    /// None` explicitly — confirming the former really is a thin wrapper and
    /// not an independent (potentially drifting) implementation.
    #[test]
    fn plain_entry_point_matches_budgeted_with_none() {
        let eigenvectors = ndarray::array![[1.0f64, 0.0], [0.0, 1.0]];
        let b_ov = ndarray::array![[1.0f64, 0.5], [0.3, -0.8]];
        let eps_occ = vec![-0.4f64];
        let eps_vir = vec![0.3f64, 0.9f64];
        let quad_freqs = vec![0.1f64, 0.5, 1.0];

        let a =
            eval_eigenvalues_at_frequencies(&eigenvectors, &b_ov, &eps_occ, &eps_vir, &quad_freqs)
                .unwrap();
        let b = eval_eigenvalues_at_frequencies_budgeted(
            &eigenvectors,
            &b_ov,
            &eps_occ,
            &eps_vir,
            &quad_freqs,
            None,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    // ──────────────── trace_log_summand_term (λ → 1 cancellation) ────────────────

    /// `(λ, f(λ) = ln λ + 1 − λ)` with `f` from mpmath at 300 bits, evaluated at
    /// the EXACT f64 `λ` listed (so the reference measures the formula, not the
    /// rounding of `1 + x`). Generated by:
    /// `x = mpf(lam) - 1; print(nstr(log1p(x) - x, 21))`.
    const TRACE_LOG_REFERENCE: &[(f64, f64)] = &[
        (1.00000000000001, -4.99201041585168207765e-29),
        (1.000000000001, -5.00088904533664359657e-25),
        (1.0000000001, -5.0000008270704108047e-21),
        (1.00000001, -4.99999990589195799987e-17),
        (1.000001, -4.9999966658465011079e-13),
        (1.00001, -4.99996666698217605181e-11),
        (1.0001, -4.9996666916635656022e-9),
        (1.001, -4.99666916466723166501e-7),
        (1.01, -4.96691468319172397231e-5),
        (1.1, -4.6898201956751480304e-3),
        (1.2, -1.76784432060453663868e-2),
        (1.2499999, -2.68564286857934325565e-2),
        (1.2500001, -2.68564686857934559109e-2),
        (0.999999999, -4.9999997205140224181e-19),
        (0.9999, -5.00033335833423204863e-9),
        (0.99, -5.03358535014412732638e-5),
        (0.8, -2.31435513142097446641e-2),
        (0.3, -5.03972804325936018528e-1),
        (1e-5, -1.05129354649702283383e+1),
        (1e-10, -2.20258509300404568037e+1),
        (1.5, -9.4534891891835618022e-2),
        (2.0, -3.06852819440054690583e-1),
        (3.0, -9.01387711331890308605e-1),
        (10.0, -6.69741490700595431598),
        (100.0, -9.4394829814011908632e+1),
        (10000.0, -9.98978965962802381726e+3),
    ];

    fn literal_trace_log(lam: f64) -> f64 {
        lam.ln() + (1.0 - lam)
    }

    /// Exactness anchor: the helper matches a 300-bit reference to a few ulp
    /// on BOTH branches and both sides of λ = 1, while the literal form loses
    /// digits as `~u/|x|` near λ = 1.
    ///
    /// Measured (glibc libm): helper max rel err 2.1e-16 on the series
    /// branch, 3.9e-16 on the `ln λ − x` branch; literal form 1.1e-2 at
    /// x = 1e-14, 8.4e-7 at 1e-10, 2.6e-9 at 1e-8, 1.6e-11 at 1e-5,
    /// 1.4e-13 at 1e-3, 7.5e-15 at 1e-2.
    /// TOL = 5e-15 is ~13x the helper's worst case; the literal form misses it
    /// at all 10 points with |x| <= 1e-2 (1.01 and 0.99 sit just above
    /// 1e-2 in f64, so they are not counted).
    ///
    /// Mutation ledger: replacing the helper's body with the literal form
    /// fails this test (first failing point x = 1e-14, rel err ~1e-2). So does
    /// `ln_1p(x) - x` alone (same point, rel err 1.1e-2): it still cancels.
    /// Using `ln_1p(x) - x` in the `|x| >= 0.25` branch fails here at
    /// λ = 1e-5 (rel err 4.3e-13; 3.8e-9 at λ = 1e-10) and fails the bitwise
    /// check in `..._matches_literal_away_from_unity`.
    #[test]
    fn trace_log_summand_term_matches_high_precision_reference() {
        const TOL: f64 = 5e-15;
        let (mut worst_series, mut worst_literal) = (0.0f64, 0.0f64);
        let mut literal_misses = 0usize;
        let mut near_one = 0usize;
        for &(lam, want) in TRACE_LOG_REFERENCE {
            let got = trace_log_summand_term(lam);
            let rel = ((got - want) / want).abs();
            if (lam - 1.0).abs() < TRACE_LOG_SERIES_CUTOFF {
                worst_series = worst_series.max(rel);
            } else {
                worst_literal = worst_literal.max(rel);
            }
            assert!(
                rel < TOL,
                "helper at lam={lam:e}: got {got:.17e}, want {want:.17e}, rel {rel:.2e}"
            );
            if (lam - 1.0).abs() <= 1e-2 {
                near_one += 1;
                let lit = literal_trace_log(lam);
                if ((lit - want) / want).abs() >= TOL {
                    literal_misses += 1;
                }
            }
        }
        // The bar must discriminate: the literal form has to fail it near 1.
        assert_eq!(
            literal_misses, near_one,
            "literal form met TOL at some |x| <= 1e-2 point; the bar is not discriminating"
        );
        eprintln!(
            "trace_log_summand_term worst rel err vs 300-bit ref: series {worst_series:.2e}, \
             ln-lambda branch {worst_literal:.2e}"
        );
    }

    /// Where the literal form is accurate (λ far from 1) the helper must agree
    /// with it bit for bit: the `|x| >= 0.25` branch IS the literal form.
    #[test]
    fn trace_log_summand_term_matches_literal_away_from_unity() {
        for &lam in &[
            1e-10f64, 1e-5, 0.3, 0.75, 1.3, 1.5, 2.0, 3.0, 7.25, 10.0, 100.0, 1e4, 1e8,
        ] {
            let h = trace_log_summand_term(lam);
            let lit = literal_trace_log(lam);
            assert_eq!(
                h.to_bits(),
                lit.to_bits(),
                "lam={lam}: helper {h:.17e} vs literal {lit:.17e} must be bit-identical"
            );
        }
    }

    /// The two branches agree across the series cutoff (no seam).
    #[test]
    fn trace_log_summand_term_is_continuous_at_the_series_cutoff() {
        for sign in [1.0f64, -1.0] {
            let below = 1.0 + sign * TRACE_LOG_SERIES_CUTOFF * (1.0 - 1e-12);
            let above = 1.0 + sign * TRACE_LOG_SERIES_CUTOFF * (1.0 + 1e-12);
            let (fb, fa) = (trace_log_summand_term(below), trace_log_summand_term(above));
            // f ~ x², so the 2e-12 relative step in x moves f by ~4e-12
            // relative; 1e-10 allows that and rejects a branch mismatch.
            let rel = ((fa - fb) / fb).abs();
            assert!(
                rel < 1e-10,
                "seam at sign {sign}: {fb:.17e} vs {fa:.17e} ({rel:.2e})"
            );
        }
    }

    // ────────────── log-det (LU) vs eigenvalue path equivalence ──────────────

    /// The core identity the LU route rests on, at the level of a single
    /// dielectric matrix: `ln det(ε) + tr(I − ε) ≡ Σ_α [ln λ_α + (1 − λ_α)]`.
    #[test]
    fn trace_log_summand_matches_eigenvalue_sum() {
        // A small SPD matrix standing in for ε̃(iω).
        let eps = ndarray::array![[3.0f64, 0.4, -0.2], [0.4, 2.5, 0.7], [-0.2, 0.7, 4.0]];
        let evals =
            ferric_core::linalg::eigvalsh_dc(&eps, ferric_core::linalg::Uplo::Upper).unwrap();
        let eig_way: f64 = evals.iter().map(|&l| l.ln() + (1.0 - l)).sum();
        let lu_way = dielectric_trace_log_summand(&eps, "test").unwrap();
        assert!(
            (lu_way - eig_way).abs() < 1e-12,
            "LU summand {lu_way} vs eigenvalue sum {eig_way}"
        );
    }

    /// End-to-end equivalence on the same tiny system the eigenvalue-path
    /// regression tests use: the RPA correlation ENERGY computed via the LU
    /// log-det evaluator must match the energy computed from the eigenvalue
    /// tensor to round-off. This is the gate that makes the substitution safe.
    #[test]
    fn logdet_energy_matches_eigenvalue_energy() {
        let eigenvectors = ndarray::array![[1.0f64, 0.0], [0.0, 1.0]];
        let b_ov = ndarray::array![[1.0f64, 0.5], [0.3, -0.8]];
        let eps_occ = vec![-0.4f64];
        let eps_vir = vec![0.3f64, 0.9f64];
        let quad_freqs = vec![0.1f64, 0.5, 1.0];
        let quad_weights = vec![0.25f64, 0.5, 0.25];

        let evals =
            eval_eigenvalues_at_frequencies(&eigenvectors, &b_ov, &eps_occ, &eps_vir, &quad_freqs)
                .unwrap();
        let e_eig = rpa_correlation_energy(&quad_weights, &evals);

        let summands = eval_trace_log_summands_budgeted(
            &eigenvectors,
            &b_ov,
            &eps_occ,
            &eps_vir,
            &quad_freqs,
            None,
        )
        .unwrap();
        let e_lu = rpa_correlation_energy_from_summands(&quad_weights, &summands);

        assert!(
            (e_eig - e_lu).abs() < 1e-14,
            "eigenvalue-path E_c = {e_eig:.18} vs log-det-path E_c = {e_lu:.18}"
        );
    }

    /// Same equivalence, but with a budget tiny enough to force the PANELLED
    /// dielectric assembly — the log-det evaluator must share that path
    /// faithfully with its eigenvalue sibling, not just the full-width one.
    #[test]
    fn logdet_energy_matches_eigenvalue_energy_under_forced_panelling() {
        let eigenvectors = ndarray::array![[1.0f64, 0.0], [0.0, 1.0]];
        let b_ov = ndarray::array![[1.0f64, 0.5], [0.3, -0.8]];
        let eps_occ = vec![-0.4f64];
        let eps_vir = vec![0.3f64, 0.9f64];
        let quad_freqs = vec![0.1f64, 0.5, 1.0];
        let quad_weights = vec![0.25f64, 0.5, 0.25];
        // Must stay below the n_workers=1 fast-path floor (96 bytes for
        // m=nov=2) so panelling is forced at EVERY ambient thread count — see
        // the sibling test `eval_eigenvalues_at_frequencies_budgeted_forced_
        // panel_matches_manual_reference` for the full derivation. The old 200
        // silently took the full-width path at RAYON_NUM_THREADS=1..2.
        let tiny_budget = 64usize;

        let n_workers = rayon::current_num_threads().max(1);
        assert_eq!(
            quad_panel_width(2, 2, n_workers, Some(tiny_budget)),
            1,
            "tiny budget must force panelling for this test to mean anything \
             (n_workers={n_workers})"
        );

        let evals = eval_eigenvalues_at_frequencies_budgeted(
            &eigenvectors,
            &b_ov,
            &eps_occ,
            &eps_vir,
            &quad_freqs,
            Some(tiny_budget),
        )
        .unwrap();
        let e_eig = rpa_correlation_energy(&quad_weights, &evals);

        let summands = eval_trace_log_summands_budgeted(
            &eigenvectors,
            &b_ov,
            &eps_occ,
            &eps_vir,
            &quad_freqs,
            Some(tiny_budget),
        )
        .unwrap();
        let e_lu = rpa_correlation_energy_from_summands(&quad_weights, &summands);

        assert!(
            (e_eig - e_lu).abs() < 1e-14,
            "panelled: eigenvalue E_c = {e_eig:.18} vs log-det E_c = {e_lu:.18}"
        );
    }
}
