//! The Laplace-MP2 energy must be BIT-IDENTICAL across rayon worker counts.
//!
//! This is the end-to-end assertion behind
//! `mwe_laplace_width_is_thread_independent.rs`. That MWE pins the helper
//! (`per_task_budget_bytes` no longer divides by the ambient thread count);
//! this one pins the property a user actually cares about — that the reported
//! energy is a function of the configuration and nothing else.
//!
//! # What was wrong
//!
//! `compute_ao` had three independent thread dependencies, all reaching the
//! energy:
//!
//! 1. `block_mu` came from `per_task_budget_bytes`, which was
//!    `resolve_budget_bytes(..) / rayon::current_num_threads()`. It k-blocks
//!    `j_mat += &m_panel.dot(&n_panel.t())` — a partial-sum accumulation over
//!    the μ axis.
//! 2. `block_p` in `laplace_exchange_energy` came from the same value, and the
//!    number of partial sums IS `naux / block_p`.
//! 3. The quadrature-point sum used `.reduce(|| (0.0,0.0), ..)`, a rayon TREE
//!    fold whose association depends on how the work was split.
//!
//! (1) and (2) are float reassociations that this test observes: the same
//! molecule/basis/budget returned different last digits at different
//! `RAYON_NUM_THREADS`.
//!
//! (3) is different, and worth stating precisely so nobody over-claims it.
//! Mutating only that fold back to `.par_iter().reduce(..)` — with the divisor
//! left fixed — leaves every contract below GREEN at n_quad = 5 and 7. rayon
//! does not split a 5- or 7-element range into a differently-shaped tree at
//! 1/2/3/12 workers, so no reassociation occurs. The collect-then-serial-fold
//! is therefore HARDENING (the fold order becomes a pure function of the point
//! order at any n_quad and any pool) and NOT a defect this test caught.
//!
//! # Why bit-identity is the right bar here
//!
//! A looser tolerance would pass while the defect was live: the perturbation is
//! ~1e-15 relative, so any `assert_relative_eq!` with a physics-scale epsilon
//! cannot see it. And it is the correct bar on the merits — the thread count is
//! not a physical parameter, so there is no accuracy tradeoff to make. Note the
//! contrast with the BUDGET, where a change legitimately may move the last
//! digits (`rimp2::mo_stream_chunk_for`'s doc spells this out); this test holds
//! the budget FIXED and varies only the pool.
//!
//! # Choosing a regime where the mechanism is not inert
//!
//! Both widths end in a `clamp(1, ..)` at full width, so at any budget large
//! enough to reach that clamp they are full width at EVERY worker count — and a
//! thread-invariance test there passes whether or not the divisor is ambient.
//! The first version of this test did exactly that: water/cc-pVDZ at 2 GiB, on
//! which it passed with the defect deliberately restored. Full width on water
//! needs a 0.7 MiB share, far below the 64 MiB `MIN_PER_TASK` floor, so on that
//! system the widths can never bind and the defect is unreachable.
//!
//! Benzene/cc-pVDZ (nbas=114, naux=420) at a 0.5 GiB budget is the smallest
//! point where they do bind: `block_p` was 362/181/120/45 at 1/2/3/12 workers
//! under the ambient divisor, against a constant 45 now. CONTRACT 3 asserts
//! that regime holds rather than assuming it, and the mutation record below
//! shows the test failing when the defect is restored.
//!
//! # Mutation record
//!
//! * Restoring `let share = total / rayon::current_num_threads().max(1)`:
//!   CONTRACTS 1, 2 and 4 FAIL. The energy moves by 2.5e-13 Ha between 1 and 2
//!   workers (-7.98475422303975502e-1 vs -7.98475422303720928e-1) and
//!   `block_p` moves 362 -> 181. Note these passed on the ORIGINAL water/2 GiB
//!   fixture, which is why the fixture changed.
//! * Restoring the rayon tree fold over quadrature points, divisor left fixed:
//!   all contracts PASS. See the note on (3) above — that half of the change is
//!   unproven hardening, not a caught defect.
//!
//! Run with `OPENBLAS_NUM_THREADS=1` per the project's rayon/BLAS convention —
//! these paths run GEMMs under a rayon map.
//!
//! TIERS. CONTRACT 2 (the same bit identity at n_quad = 7, ~134 s on CI)
//! runs weekly in `validation_laplace_energy_is_thread_independent.rs`. It
//! catches nothing CONTRACT 1 misses: the restored ambient divisor fails
//! CONTRACT 1 here, and the rayon tree-fold mutant passes both (mutation
//! record above). CONTRACT 1 is the per-commit canary. Shared fixture:
//! `laplace_thread_support/mod.rs`.

#[path = "laplace_thread_support/mod.rs"]
mod support;
use support::*;

/// CONTRACT 1: same config, same bits, at every worker count.
///
/// Fails on the unfixed tree: `block_mu` and `block_p` differ by 12x between
/// 1 and 12 workers, and the `.reduce()` tree shape differs too.
#[test]
fn the_laplace_correlation_energy_is_bit_identical_across_worker_counts() {
    let f = fixture();
    let n_quad = 5;
    let reference = energy_at(&f, n_quad, 1);
    assert!(
        reference.is_finite() && reference < 0.0,
        "sanity: the correlation energy must be finite and negative, got {reference}"
    );
    for n in THREAD_COUNTS {
        let got = energy_at(&f, n_quad, n);
        assert_eq!(
            got.to_bits(),
            reference.to_bits(),
            "Laplace-MP2 correlation energy changed with the worker count: \
             {reference:.17e} at 1 worker vs {got:.17e} at {n}. Difference {:.3e}. \
             The thread count is not a physical parameter — a budget-derived k-blocking \
             width or a rayon tree fold has leaked the ambient pool size into the energy.",
            (got - reference).abs()
        );
    }
}

/// CONTRACT 3 (reachability guard): at this fixture and budget, at least one
/// panel width must be STRICTLY BELOW full width.
///
/// This is the assertion whose absence made the first version of this test
/// vacuous. Both widths end in `clamp(1, full)`, so if the budget reaches that
/// clamp they are full width at every worker count and CONTRACTS 1-2 pass with
/// the defect live — verified by restoring the ambient divisor and watching
/// them pass on the old water fixture.
///
/// Checking the WIDTH, not the byte ceiling, is the point: `share < budget` is
/// true at any budget and says nothing about whether the width moved.
#[test]
fn at_least_one_panel_width_is_actually_constrained_here() {
    let (block_mu, block_p) =
        ferric_mp2::laplace::laplace_panel_widths_for_test(BUDGET, NAUX, NBAS, NOCC);
    assert!(
        block_mu < NBAS || block_p < NAUX,
        "neither panel width binds at this fixture/budget (block_mu={block_mu}/{NBAS}, \
         block_p={block_p}/{NAUX}): both are clamped at full width, so CONTRACTS 1-2 \
         would pass even with an ambient divisor. Lower BUDGET or grow the fixture."
    );
}

/// CONTRACT 4: the widths themselves are thread-invariant at this shape.
///
/// The direct statement of the fix, one level below the energy. If this fails
/// but CONTRACTS 1-2 pass, the energy insensitivity is luck rather than
/// structure.
#[test]
fn the_panel_widths_do_not_move_with_the_worker_count() {
    let reference = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| ferric_mp2::laplace::laplace_panel_widths_for_test(BUDGET, NAUX, NBAS, NOCC));
    for n in THREAD_COUNTS {
        let got = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| {
                ferric_mp2::laplace::laplace_panel_widths_for_test(BUDGET, NAUX, NBAS, NOCC)
            });
        assert_eq!(
            got, reference,
            "panel widths moved from {reference:?} at 1 worker to {got:?} at {n}; both \
             k-block a float accumulation, so this is an energy change"
        );
    }
}
