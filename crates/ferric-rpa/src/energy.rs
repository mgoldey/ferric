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
    FerricError::Lapack(format!("{what} (NaN/Inf dielectric from near-degenerate reference?): {e}"))
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
/// `eval_eigenvalues_at_frequencies_budgeted` stays separate: its panel-width
/// scratch sizing is chosen from the memory budget before the parallel region.
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

/// E_c^RPA = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))].
///
/// Computes RPA correlation energy from eigenvalues of the dielectric matrix
/// evaluated at imaginary frequencies (quadrature points).
///
/// # Arguments
/// * `quad_weights` - Gauss-Legendre or other quadrature weights w_k, length N_quad
/// * `eigenvalues_freq` - Eigenvalues λ_α(iω_k) of shape (N_quad, M)
///
/// # Returns
/// RPA correlation energy in Hartree.
pub fn rpa_correlation_energy(
    quad_weights: &[f64],
    eigenvalues_freq: &Array2<f64>,
) -> f64 {
    let n_quad = quad_weights.len();
    assert_eq!(eigenvalues_freq.nrows(), n_quad);

    let mut e_c = 0.0f64;
    for (k, &wk) in quad_weights.iter().enumerate() {
        let row = eigenvalues_freq.row(k);
        let contrib: f64 = row
            .iter()
            .map(|&lam| lam.ln() + (1.0 - lam))
            .sum();
        e_c += wk * contrib;
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
    use ndarray_linalg::{Eigh, UPLO};

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
                eigenvectors, b_ov, eps_occ, eps_vir, omega, laplace, rhs_scaled, out,
            );
            let (evals, _) = out
                .eigh(UPLO::Upper)
                .map_err(|e| dielectric_lapack_err("Laplace dielectric eigh failed", e))?;
            Ok(evals.to_vec())
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
    use ndarray_linalg::{Eigh, UPLO};

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();

    let rows = per_frequency(quad_freqs, || (), |(), omega| {
        let eps_proj = dielectric_matrix_unrestricted(eigenvectors, chan_a, chan_b, omega);
        let (evals, _) = eps_proj
            .eigh(UPLO::Upper)
            .map_err(|e| dielectric_lapack_err("unrestricted dielectric eigh failed", e))?;
        Ok(evals.to_vec())
    })?;
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
    use ndarray_linalg::{Eigh, UPLO};

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();

    let rows = per_frequency(quad_freqs, || (), |(), omega| {
        let eps_proj = dielectric_matrix_laplace_unrestricted(
            eigenvectors,
            chan_a, laplace_a,
            chan_b, laplace_b,
            omega,
        );
        let (evals, _) = eps_proj
            .eigh(UPLO::Upper)
            .map_err(|e| dielectric_lapack_err("U-Laplace dielectric eigh failed", e))?;
        Ok(evals.to_vec())
    })?;
    Ok(rows_into_array(n_quad, m, rows))
}

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
fn quad_panel_width(m: usize, nov: usize, n_workers: usize, memory_budget_bytes: Option<usize>) -> usize {
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
    // `out` term (m² is independent of k, so it's subtracted off first).
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
/// once via [`quad_panel_width`] before the parallel region starts.
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
    use ferric_integrals::blas_threads::with_blas_threads;
    use ndarray::Array2;
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();
    let nov = eps_occ.len() * eps_vir.len();

    // The projection y = Vᵀ·B_ov is frequency-independent — compute it ONCE
    // instead of recomputing the (m × nov) GEMM at every quadrature point. Each
    // ω then only does the cheap column scaling + DSYRK.
    let y = eigenvectors.t().dot(b_ov);

    let n_workers = rayon::current_num_threads().max(1);
    let panel_width = quad_panel_width(m, nov, n_workers, memory_budget_bytes);
    let use_panelled = panel_width < nov;

    // Each quadrature point is fully independent — parallelize over frequencies.
    // Pin BLAS to 1 inside the rayon region (per-frequency eigh must not nest).
    //
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
    let rows: Vec<Vec<f64>> = with_blas_threads(1, || {
        quad_freqs
            .par_iter()
            .map_init(
                || (Array2::<f64>::zeros((m, panel_width)), Array2::<f64>::zeros((m, m))),
                |(rhs_scaled, out), &omega| {
                    let scale = build_scale_factors(eps_occ, eps_vir, omega);
                    if use_panelled {
                        dielectric_matrix_from_projection_into_panelled(&y, &scale, rhs_scaled, out);
                    } else {
                        dielectric_matrix_from_projection_into(&y, &scale, rhs_scaled, out);
                    }
                    // Diagonalize at each frequency — eigenvectors at ω=0 don't diagonalize ε̃(iω)
                    let (evals, _) = out
                        .eigh(UPLO::Upper)
                        .map_err(|e| dielectric_lapack_err("dielectric eigh failed", e))?;
                    Ok(evals.to_vec())
                },
            )
            .collect::<Result<Vec<Vec<f64>>, FerricError>>()
    })?;

    let mut eigenvalues_freq = Array2::zeros((n_quad, m));
    for (k, row) in rows.into_iter().enumerate() {
        for (alpha, val) in row.into_iter().enumerate() {
            eigenvalues_freq[(k, alpha)] = val;
        }
    }
    Ok(eigenvalues_freq)
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

    per_frequency(quad_freqs, || (), |(), omega| {
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
    })
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
    per_frequency(quad_freqs, || (), |(), omega| {
        let eps_proj = dielectric_matrix_unrestricted(eigenvectors, chan_a, chan_b, omega);
        let mut winv = eps_proj
            .inv()
            .map_err(|e| dielectric_lapack_err("U PDEP-basis dielectric inversion failed", e))?;
        for d in 0..m {
            winv[(d, d)] -= 1.0;
        }
        Ok(winv)
    })
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
            let (expected_evals, _) =
                eps_mat.eigh(UPLO::Upper).expect("reference eigh failed");
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
        assert!(k < nov, "expected a narrowed panel width, got {k} (nov={nov})");
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
        let n_workers = rayon::current_num_threads().max(1);
        let tiny_budget = 200usize; // bytes — absurdly small, forces k=1
        let forced_k = quad_panel_width(2, 2, n_workers, Some(tiny_budget));
        assert_eq!(forced_k, 1, "expected the tiny budget to force k=1 for this tiny shape");

        let got = eval_eigenvalues_at_frequencies_budgeted(
            &eigenvectors, &b_ov, &eps_occ, &eps_vir, &quad_freqs, Some(tiny_budget),
        )
        .expect("eval_eigenvalues_at_frequencies_budgeted failed");

        // Manual reference: identical derivation to the non-budgeted test.
        let y = eigenvectors.t().dot(&b_ov);
        for (k, &omega) in quad_freqs.iter().enumerate() {
            let scale = crate::sternheimer::build_scale_factors(&eps_occ, &eps_vir, omega);
            let eps_mat = crate::sternheimer::dielectric_matrix_from_projection(&y, &scale);
            let (expected_evals, _) =
                eps_mat.eigh(UPLO::Upper).expect("reference eigh failed");
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

        let a = eval_eigenvalues_at_frequencies(&eigenvectors, &b_ov, &eps_occ, &eps_vir, &quad_freqs).unwrap();
        let b = eval_eigenvalues_at_frequencies_budgeted(
            &eigenvectors, &b_ov, &eps_occ, &eps_vir, &quad_freqs, None,
        ).unwrap();
        assert_eq!(a, b);
    }
}
