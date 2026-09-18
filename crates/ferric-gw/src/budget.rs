//! Plane sizes for ferric-gw's large tensors, and the one helper that decides
//! whether a soft charge may take its bytes.
//!
//! # Why this module exists
//!
//! Before the pool migration, ferric-gw had nine sites reading a memory budget
//! and not one of them debited a shared ledger: `guard_b_full` asked "does a
//! `(naux, n_act, n_act)` tensor fit in the budget?", `guard_m_proj` asked
//! "does `m_proj + b_full` fit in the budget?", and both said yes while the
//! process held the sum of all three. That is precisely the 27-atom
//! `memory budget: 4.72 GiB -> MAXRSS 6.04 GiB` failure the pool exists to
//! retire, with GW's tensor pair in place of the DF/grid pair.
//!
//! The functions here are shape arithmetic only. They are `pub` so the gate
//! tests can derive their capacities from the SAME expressions the gates
//! charge, rather than freezing a constant that goes stale (the ferric-rpa
//! migration froze one at this box's 12 workers and broke 3 of 7 tests at
//! narrower widths).

/// Bytes per `f64`.
const F64_BYTES: usize = 8;

/// The dressed MO tensor `b_full`, shape `(naux, n_act, n_act)`.
///
/// This is GW's single largest resident allocation: one per spin channel, held
/// for the whole method (`MoB` is passed by reference into every Σ driver and
/// every evGW outer iteration).
pub fn b_full_bytes(naux: usize, n_act: usize) -> usize {
    naux.saturating_mul(n_act)
        .saturating_mul(n_act)
        .saturating_mul(F64_BYTES)
}

/// The projected tensor `m_proj`, shape `(m_modes, n_act, n_act)`.
///
/// `m_modes == naux` at the default full rank (`trunc_thresh = 0`), so this is
/// the same size as [`b_full_bytes`] unless truncation is on.
pub fn m_proj_bytes(m_modes: usize, n_act: usize) -> usize {
    m_modes
        .saturating_mul(n_act)
        .saturating_mul(n_act)
        .saturating_mul(F64_BYTES)
}

/// The raw AO 3-index tensor `(naux, nao, nao)` that `ThreeIndexSource` holds
/// in-core while `stream_dressed_mo_band` reads out of it.
///
/// ferric-gw does NOT charge this -- `ferric_integrals::three_index_source`
/// hard-charges it itself, inside `ThreeIndexSource::build`, and charging it
/// twice would double-count. But ferric-gw must still SUBTRACT it when
/// deciding whether one of its own charges may be taken, which is what
/// `b_full_fits_beside_the_ao_source` is for. The rule from the ferric-rpa
/// migration, verbatim: whatever you decline to CHARGE because another crate
/// charges it, you must also decline to SPEND.
pub fn ao_source_bytes(naux: usize, nao: usize) -> usize {
    naux.saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(F64_BYTES)
}

/// The peak of one `project_b_into_pdep` call: `m_proj` and the `b_full` it is
/// built FROM, which is live across the whole GEMM.
///
/// Summed rather than doubled because truncation makes `m_modes < naux`.
pub fn projection_peak_bytes(m_modes: usize, n_act: usize, naux: usize) -> usize {
    m_proj_bytes(m_modes, n_act).saturating_add(b_full_bytes(naux, n_act))
}

/// Both spin channels' [`projection_peak_bytes`], which U-GW holds at once.
pub fn projection_peak_bytes_both_spins(
    m_modes: usize,
    n_act_a: usize,
    n_act_b: usize,
    naux: usize,
) -> usize {
    projection_peak_bytes(m_modes, n_act_a, naux)
        .saturating_add(projection_peak_bytes(m_modes, n_act_b, naux))
}

/// The dense BSE/TDHF `A` (and, off-TDA, `B`) matrices, shape `(nov, nov)`.
///
/// `n_matrices` is how many are co-resident: 1 for TDA, 2 for the full
/// A/B problem. The eigensolve adds its own `(nov, nov)` eigenvector output,
/// which the caller counts by asking for one more matrix.
pub fn dense_ab_bytes(nov: usize, n_matrices: usize) -> usize {
    nov.saturating_mul(nov)
        .saturating_mul(F64_BYTES)
        .saturating_mul(n_matrices)
}

/// Per-worker scratch held concurrently by the QP-solve `par_iter`.
///
/// `sigma::sigma_c_at_z` allocates, per call: `v_owned` and one `wv` of shape
/// `(m_modes, n_act)`, plus `w_nk` of shape `(n_act, n_quad)`. It is called from
/// inside the `mo_indices.par_iter()` in `run_g0w0` / `run_evgw0` / `run_evgw`
/// (and both spin channels of the U- variants), so at the peak `n_workers`
/// copies are live at once.
///
/// This is the term that scales with the RAYON THREAD COUNT rather than with
/// any budget knob -- structurally the same plane as `ferric_rpa`'s
/// per-worker frequency-quadrature scratch, which was the crux of the
/// 2026-07-13 over-budget incident.
///
/// `n_channels` is 1 for closed-shell and 2 for U-GW, whose alpha and beta
/// sweeps are separate `par_iter`s but whose `m_proj_a`/`m_proj_b` are both
/// live; passing 2 charges the wider of the two shapes twice, which is
/// conservative only in the degenerate case where the channels differ.
pub fn qp_worker_scratch_bytes(
    m_modes: usize,
    n_act: usize,
    n_quad: usize,
    n_workers: usize,
    n_channels: usize,
) -> usize {
    let per_worker = m_modes
        .saturating_mul(n_act)
        .saturating_mul(2) // v_owned + one live `wv`
        .saturating_add(n_act.saturating_mul(n_quad)) // w_nk
        .saturating_mul(F64_BYTES);
    per_worker
        .saturating_mul(n_workers.max(1))
        .saturating_mul(n_channels.max(1))
}

/// May a ferric-gw plane of `want` bytes be taken, given that the AO 3-index
/// source is about to be (or already is) charged by ferric-integrals?
///
/// # The defect this shape avoids
///
/// This is the ferric-rpa starvation bug transplanted to GW's ordering. In
/// `mo_b::build_mo_b_from_source` the order is:
///
/// ```text
///   1. ThreeIndexSource::build          HARD, by ferric-integrals
///   2. b_full                           this crate's charge
/// ```
///
/// so in GW the mandatory integrals plane asks FIRST and its bytes are already
/// outstanding when `b_full` asks. The starvation risk therefore runs the
/// other way and the naive fix (subtract the AO bytes again) would DOUBLE-
/// count them: `available_bytes()` has already been debited.
///
/// `build_full_b_both_spins` is where it matters. One AO source is built and
/// streamed TWICE, so at the β pass the pool holds the AO source AND α's
/// `b_full`; a gate that re-subtracted `ao_source_bytes` there would refuse a
/// job that fits. Hence this helper takes `already_charged`: the caller passes
/// the AO bytes only when they are NOT yet outstanding.
///
/// Returns `true` when the plane fits. `None` from `global_available_bytes`
/// (no pool) is `true` unconditionally -- the trivial limit.
pub fn fits_beside_pending(want: usize, pending_hard: usize) -> bool {
    match ferric_core::memory::pool::global_available_bytes() {
        None => true,
        Some(avail) => want.saturating_add(pending_hard) <= avail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_peak_sums_the_two_co_resident_tensors() {
        // Full rank: m_modes == naux, so the peak is exactly 2x one tensor.
        let one = b_full_bytes(200, 50);
        assert_eq!(projection_peak_bytes(200, 50, 200), 2 * one);
    }

    #[test]
    fn projection_peak_is_a_sum_not_a_doubling_under_truncation() {
        // m_modes < naux: doubling b_full would OVER-charge, and an
        // over-estimating guard refuses jobs that fit.
        let truncated = projection_peak_bytes(50, 50, 200);
        let doubled = 2 * b_full_bytes(200, 50);
        assert!(
            truncated < doubled,
            "truncation must shrink the charge: {truncated} vs {doubled}"
        );
        assert_eq!(truncated, m_proj_bytes(50, 50) + b_full_bytes(200, 50));
    }

    #[test]
    fn both_spins_sums_unequal_channels() {
        let a = projection_peak_bytes(200, 50, 200);
        let b = projection_peak_bytes(200, 25, 200);
        assert_eq!(projection_peak_bytes_both_spins(200, 50, 25, 200), a + b);
    }

    #[test]
    fn shape_arithmetic_saturates_rather_than_wrapping() {
        // A nonsense shape must report usize::MAX (refused), never a small
        // wrapped number that would PASS a check.
        assert_eq!(b_full_bytes(usize::MAX, usize::MAX), usize::MAX);
        assert_eq!(dense_ab_bytes(usize::MAX, 2), usize::MAX);
    }

    #[test]
    fn fits_beside_pending_is_unconditionally_true_without_a_pool() {
        // The trivial limit: no pool installed => never refuse, never resize.
        assert!(ferric_core::memory::pool::global().is_none());
        assert!(fits_beside_pending(usize::MAX, usize::MAX));
    }
}
