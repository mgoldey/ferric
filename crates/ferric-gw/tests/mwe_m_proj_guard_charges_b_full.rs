//! MWE: the GW projection guard must charge BOTH co-resident tensors.
//!
//! # The defect
//!
//! `cohsex::guard_m_proj` charges only the tensor it is about to allocate:
//!
//! ```text
//! let need = m_modes * n_act * n_act * 8;
//! if need <= budget { return Ok(()); }
//! ```
//!
//! But `project_b_into_pdep` builds `m_proj` FROM `mo_b.b_full`, which is
//! `(naux, n_act, n_act)` and live across the whole GEMM — the guard runs at
//! cohsex.rs:128, and `b_full` is reshaped and consumed at :130-134. Both are
//! resident at the peak; one is charged.
//!
//! At full rank (`trunc_thresh = 0`, the default) `m_modes == naux`, so
//! `b_full` is exactly the same size as `m_proj`:
//!
//! ```text
//!   system                naux  n_act    m_proj    b_full     true peak
//!   benzene/aug-cc-pVTZ   1512    300   1.09 GB   1.09 GB       2.18 GB
//!   GW100-scale           2000    500   4.00 GB   4.00 GB       8.00 GB
//! ```
//!
//! A clean **2x** under-count on GW's largest tensor pair.
//!
//! # And 4x for U-GW
//!
//! `u_cohsex.rs:31-32` calls the projection once per spin:
//!
//! ```text
//! let m_proj_a = project_b_into_pdep(mo_b_a, v_dressed, gw_cfg.memory_budget_bytes)?;
//! let m_proj_b = project_b_into_pdep(mo_b_b, v_dressed, gw_cfg.memory_budget_bytes)?;
//! ```
//!
//! Both are live simultaneously — `cohsex_pieces` is called on each afterwards
//! — yet each call gates independently against the FULL budget. The same shape
//! repeats at `u_sigma.rs:41`, `:142` and `:325` (the latter inside the evGW
//! iteration loop). The guard's own message even says "(×2 for U-GW)" while
//! the arithmetic does not.
//!
//! # Scope
//!
//! Pins the guard's arithmetic through the public `project_b_into_pdep`, by
//! choosing budgets that straddle the true peak. It does not attempt to run a
//! GW job: the point is which allocations the gate counts.

use ferric_gw::cohsex::{guard_m_proj_both_spins, project_b_into_pdep};
use ferric_gw::mo_b::MoB;
use ndarray::{Array2, Array3};

/// A shape where m_proj and b_full are each ~8 MB, so a budget between 1x and
/// 2x of one tensor cleanly separates "charges one" from "charges both".
const NAUX: usize = 200;
const N_ACT: usize = 50;
const F64: usize = 8;

fn one_tensor_bytes() -> usize {
    NAUX * N_ACT * N_ACT * F64
}

/// Minimal `MoB` carrying a full-rank `b_full`.
fn mo_b() -> MoB {
    MoB {
        b_full: Array3::<f64>::zeros((NAUX, N_ACT, N_ACT)),
        v_inv_sqrt: Array2::<f64>::zeros((NAUX, NAUX)),
        naux: NAUX,
        n_act: N_ACT,
        first_act: 0,
        n_occ_act: N_ACT / 2,
        eps_act: vec![0.0; N_ACT],
    }
}

/// Full rank: `m_modes == naux`, the default (`trunc_thresh = 0`).
fn v_dressed() -> Array2<f64> {
    Array2::<f64>::zeros((NAUX, NAUX))
}

/// CONTRACT 1: a budget that fits ONLY m_proj must be REFUSED.
///
/// The defect itself. `1.5x` one tensor is comfortably above `m_proj` alone
/// and comfortably below the true `m_proj + b_full` peak, so the old guard
/// admitted it and the new one must not.
#[test]
fn a_budget_that_fits_only_m_proj_is_refused() {
    let budget = one_tensor_bytes() * 3 / 2;
    let r = project_b_into_pdep(&mo_b(), &v_dressed(), Some(budget));
    assert!(
        r.is_err(),
        "a budget of 1.5x one tensor was ACCEPTED, but the peak is m_proj + b_full = 2x — \
         b_full is live across the GEMM that produces m_proj (guard at cohsex.rs:128, \
         b_full consumed at :130-134), so charging only m_proj under-counts by 2x."
    );
}

/// CONTRACT 2 (the over-rejection guard): a budget that fits BOTH must run.
///
/// "An over-estimating guard is also a bug." A fix that simply doubled every
/// charge, or that refused everything, would be a regression dressed as a
/// correction. 3x one tensor comfortably holds the 2x peak.
#[test]
fn a_budget_that_fits_both_tensors_is_accepted() {
    let budget = one_tensor_bytes() * 3;
    let r = project_b_into_pdep(&mo_b(), &v_dressed(), Some(budget));
    assert!(
        r.is_ok(),
        "a budget of 3x one tensor must accept a 2x peak — refusing it would break GW runs \
         that fit: {:?}",
        r.err()
    );
}

/// CONTRACT 3: the charge is at least both tensors, not a fudge factor.
///
/// CONTRACT 1 would pass if the guard multiplied its old figure by any number
/// above 1.5. Bracket the threshold: just under 2x must fail, comfortably over
/// must pass, which pins the charge at the actual `m_proj + b_full` sum.
#[test]
fn the_threshold_sits_at_the_sum_of_both_tensors() {
    let one = one_tensor_bytes();
    let just_under = project_b_into_pdep(&mo_b(), &v_dressed(), Some(one * 2 - one / 10));
    assert!(
        just_under.is_err(),
        "just under 2x one tensor must be refused — the peak is exactly 2x at full rank"
    );
    let just_over = project_b_into_pdep(&mo_b(), &v_dressed(), Some(one * 2 + one / 2));
    assert!(
        just_over.is_ok(),
        "comfortably over 2x must be accepted: {:?}",
        just_over.err()
    );
}

/// CONTRACT 4: the U-GW guard charges BOTH spins.
///
/// `u_cohsex.rs` and the three `u_sigma.rs` sites build `m_proj_a` and
/// `m_proj_b` and hold them together, but each `project_b_into_pdep` gates
/// only against the whole budget on its own — so both pass while the process
/// holds the sum. `guard_m_proj_both_spins` is the spin-summed check no single
/// call can make.
///
/// This contract exists because mutation-testing found the fix UNGUARDED:
/// changing the sum to charge only the alpha spin left the entire ferric-gw
/// suite green.
#[test]
fn the_u_gw_guard_charges_both_spins() {
    let one = one_tensor_bytes();
    // Per spin the peak is m_proj + b_full = 2x one tensor, so both spins is 4x.
    // A budget of 3x holds one spin comfortably and both spins not at all.
    let r = guard_m_proj_both_spins(NAUX, N_ACT, N_ACT, NAUX, Some(one * 3));
    assert!(
        r.is_err(),
        "a budget of 3x one tensor was accepted for BOTH spins, but each spin needs 2x \
         (m_proj + b_full) so the pair needs 4x. Charging one spin is the defect: each \
         per-spin guard passes against the full budget independently."
    );
}

/// CONTRACT 5 (the over-rejection guard for CONTRACT 4): a budget that fits
/// both spins must be accepted.
///
/// Without this, CONTRACT 4 would pass on a guard that refused everything.
#[test]
fn the_u_gw_guard_accepts_a_budget_that_fits_both_spins() {
    let one = one_tensor_bytes();
    let r = guard_m_proj_both_spins(NAUX, N_ACT, N_ACT, NAUX, Some(one * 5));
    assert!(
        r.is_ok(),
        "5x one tensor must accept a 4x two-spin peak: {:?}",
        r.err()
    );
}

/// CONTRACT 6: the two spins are SUMMED, not doubled.
///
/// Under `trunc_thresh > 0` the two spin channels can have different active
/// sizes, so `2 * pair(n_act_a)` would be wrong in both directions. Give the
/// beta spin a smaller active space and check the threshold tracks the true
/// sum rather than twice the alpha.
#[test]
fn unequal_spin_channels_are_summed_not_doubled() {
    let small = N_ACT / 2;
    let per = |n: usize| (NAUX * n * n * 8) * 2; // m_proj + b_full at full rank
    let true_sum = per(N_ACT) + per(small);
    let twice_alpha = 2 * per(N_ACT);
    assert!(true_sum < twice_alpha, "fixture must make the two differ");
    // A budget between the true sum and twice-alpha must be ACCEPTED: refusing
    // it would mean the guard is doubling the larger spin instead of summing.
    let between = (true_sum + twice_alpha) / 2;
    let r = guard_m_proj_both_spins(NAUX, N_ACT, small, NAUX, Some(between));
    assert!(
        r.is_ok(),
        "a budget above the true two-spin sum ({true_sum}) but below twice-alpha \
         ({twice_alpha}) was refused, so the guard is doubling rather than summing: {:?}",
        r.err()
    );
}
