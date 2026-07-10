//! RI-dRPA sanity checks: full-basis correlation energy and eigenvalue diagnostics.
//!
//! The RI-dRPA approximation evaluates RPA on the full RI-auxiliary basis without
//! eigenpotential truncation. Used as a sanity check against PDEP-RPA.

use ndarray::{Array2, Axis, Zip};
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;
use ferric_core::FerricError;

use crate::channel::RpaChannel;
use crate::sternheimer::build_scale_factors;

/// Per-frequency footprint (bytes) of one RI-dRPA diagnostic task and the
/// number of rayon workers the memory budget can sustain in parallel.
///
/// Each task holds a scaled `b_ov` copy (`naux·nov·8`), the `naux²` dielectric,
/// plus LAPACK `eigh` workspace (~`naux²`, conservatively 3× the matrix). With
/// `n_spin` spin channels contributing to the dielectric simultaneously the
/// `b_scaled` copy is counted per active channel.
///
/// Returns `max(1, min(nfreq, budget / footprint))` — never over-throttles below
/// one worker, never spawns more than there are frequencies. `~0.9 GB × active
/// workers` was the hidden peak this bounds (M9).
fn diag_worker_budget(naux: usize, nov: usize, n_spin: usize, nfreq: usize) -> usize {
    let bscaled = naux.saturating_mul(nov).saturating_mul(8).saturating_mul(n_spin.max(1));
    let eps = naux.saturating_mul(naux).saturating_mul(8);
    // eigh eigenvectors + workspace: treat as ~3× the dielectric matrix.
    let footprint = bscaled.saturating_add(eps.saturating_mul(4)).max(1);
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let by_budget = (budget / footprint).max(1);
    by_budget.min(nfreq.max(1))
}

/// Run `f` over the (frequency, weight) pairs with rayon parallelism capped at
/// `max_workers`. A bounded scoped pool keeps the diagnostic's peak resident
/// memory below the budget while leaving per-task numerics untouched (results
/// are order-independent — they are summed). Falls back to the ambient pool if
/// the capped pool cannot be built.
fn par_map_capped<F>(
    quad_freqs: &[f64],
    quad_weights: &[f64],
    max_workers: usize,
    f: F,
) -> Result<Vec<f64>, FerricError>
where
    F: Fn(f64, f64) -> Result<f64, FerricError> + Sync + Send,
{
    let run = || {
        quad_freqs
            .par_iter()
            .zip(quad_weights.par_iter())
            .map(|(&omega, &wk)| f(omega, wk))
            .collect::<Result<Vec<f64>, FerricError>>()
    };
    match rayon::ThreadPoolBuilder::new().num_threads(max_workers).build() {
        Ok(pool) => pool.install(run),
        Err(_) => run(),
    }
}

/// Compute RI-dRPA eigenvalues of ε̃(iω) = I − χ₀(iω) in the full RI basis.
///
/// The dielectric matrix is computed directly without truncation to eigenpotentials.
/// Returns eigenvalues in descending order.
///
/// # Arguments
/// * `b_ov` - RI-MO tensor B^P_ia, shape (naux, nocc*nvir)
/// * `eps_occ` - Occupied orbital energies
/// * `eps_vir` - Virtual orbital energies
/// * `omega` - Imaginary frequency ω (typically positive for iω)
///
/// # Returns
/// Vector of eigenvalues sorted in descending order.
pub fn ri_drpa_eigenvalues(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> Result<Vec<f64>, FerricError> {
    let naux = b_ov.nrows();
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    let nov = nocc * nvir;
    assert_eq!(b_ov.shape()[1], nov);

    // Build Π = 4 Σ_{ia} e_ia/(ω²+e_ia²) B^P B^Q (= −χ₀, RHF: factor 4 = 2 spin × 2 orb)
    //
    // SYRK path: b_scaled[p,ia] = b_ov[p,ia] * sqrt(4·e_ia/(ω²+e_ia²))
    //   chi0 = b_scaled @ b_scaled^T via DSYRK (symmetric rank-k update, ~2× DGEMM).
    let scale = build_scale_factors(eps_occ, eps_vir, omega);
    let mut b_scaled: Array2<f64> = b_ov.to_owned();
    let scale_row = scale.view().insert_axis(Axis(0)); // (1, nov)
    Zip::from(&mut b_scaled)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s);
    let mut chi0 = crate::sternheimer::syrk_aat(&b_scaled);
    let _ = nvir; // nvir factored into scale via build_scale_factors

    // ε̃ = I − χ₀ = I + Π (Π = −χ₀, positive); eigenvalues 1 + μ ≥ 1
    for p in 0..naux {
        chi0[(p, p)] += 1.0;
    }

    let (evals, _) = chi0
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("RI-dRPA diagonalization failed: {e}")))?;

    let mut result: Vec<f64> = evals.to_vec();
    result.sort_by(|a, b| b.total_cmp(a));
    Ok(result)
}

/// Compute RI-dRPA correlation energy via trace-log on full-basis eigenvalues.
///
/// E_c^dRPA = (1/2π) Σ_k w_k [ln det(I − Π(iω_k)) + tr(Π(iω_k))]
///         where Π = 4 Σ_{ia} e_ia/(ω²+e_ia²) B B^T (= −χ₀, RHF).
///
/// Uses ln|det| (real part of log determinant) so the formula stays well-defined
/// even when (I − Π) has negative eigenvalues — matching PySCF's behavior.
/// Unrestricted RI-dRPA correlation energy via full-basis dielectric.
///
/// Builds ε̃ = I + Π_α + Π_β at each ω_k, evaluates trace-log without
/// any Davidson/PDEP machinery. Used to localize bugs in the U-RPA stack
/// — if the eigensolver path disagrees with this diagnostic, the bug is
/// in eigensolving, not the spin-summed dielectric formula.
pub fn u_ri_drpa_energy(
    chan_a: &RpaChannel,
    chan_b: &RpaChannel,
    quad_freqs: &[f64],
    quad_weights: &[f64],
) -> Result<f64, FerricError> {
    use crate::sternheimer::build_scale_factors_with_prefactor;
    use crate::sternheimer::syrk_aat;

    let naux = chan_a.b_ov.nrows();
    // Two spin channels contribute a b_scaled copy each; size on the larger nov.
    let nov = chan_a.b_ov.shape()[1].max(chan_b.b_ov.shape()[1]);
    let max_workers = diag_worker_budget(naux, nov, 2, quad_freqs.len());

    let contribs = par_map_capped(quad_freqs, quad_weights, max_workers, |omega, wk| {
        let mut eps_mat = Array2::<f64>::zeros((naux, naux));
        for p in 0..naux { eps_mat[(p, p)] = 1.0; }
        for RpaChannel { b_ov: b, eps_occ: eo, eps_vir: ev } in [*chan_a, *chan_b] {
            if eo.is_empty() {
                continue; // empty spin channel adds nothing
            }
            let scale = build_scale_factors_with_prefactor(eo, ev, omega, 2.0);
            let mut bs = b.to_owned();
            let scale_row = scale.view().insert_axis(Axis(0));
            Zip::from(&mut bs)
                .and_broadcast(scale_row)
                .for_each(|x, &s| *x *= s);
            let chi_sigma = syrk_aat(&bs);
            eps_mat += &chi_sigma;
        }
        let (evals, _) = eps_mat.eigh(UPLO::Upper)
            .map_err(|e| FerricError::General(format!("U-RI-dRPA eigh: {e}")))?;
        let contrib: f64 = evals.iter().map(|&lam| lam.ln() + (1.0 - lam)).sum();
        Ok(wk * contrib)
    })?;

    let e_c: f64 = contribs.iter().sum();
    Ok(e_c / (2.0 * std::f64::consts::PI))
}

pub fn ri_drpa_energy(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    quad_weights: &[f64],
) -> Result<f64, FerricError> {
    // Each quadrature point is fully independent — parallelize over frequencies,
    // but cap the width so nfreq × (b_scaled + eps) stays within the memory
    // budget (M9: single spin channel here).
    let naux = b_ov.nrows();
    let nov = eps_occ.len() * eps_vir.len();
    let max_workers = diag_worker_budget(naux, nov, 1, quad_freqs.len());
    let contribs = par_map_capped(quad_freqs, quad_weights, max_workers, |omega, wk| {
        let evals = ri_drpa_eigenvalues(b_ov, eps_occ, eps_vir, omega)?;
        // ln det(I + Π) − tr(Π) where Π = −χ₀ ≥ 0
        // = Σ_α [ln(λ_α) + (1 − λ_α)] with λ_α = 1 + μ_α ≥ 1
        let contrib: f64 = evals
            .iter()
            .map(|&lam| lam.ln() + (1.0 - lam))
            .sum();
        Ok(wk * contrib)
    })?;
    let e_c: f64 = contribs.iter().sum();
    Ok(e_c / (2.0 * std::f64::consts::PI))
}
