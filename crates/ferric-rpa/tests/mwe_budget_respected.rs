//! MWE: does a pinned memory budget actually bound the per-atom polarizability
//! path? (At time of writing: no — contract 2 fails.)
//!
//! This is the *proof* half of the memory-limits work. Each test states a
//! contract a budget-respecting implementation must satisfy, in the smallest
//! form that distinguishes "respected" from "ignored". They are the regression
//! net for the fix; see
//! `docs/superpowers/specs/2026-07-25-memory-limits-design.md`.
//!
//! Deliberately tiny shapes and no SCF: the claims under test are structural
//! (does a knob respond to the budget? can it exceed it?), not scale-dependent.
//! That makes them both sufficient and safe on a shared, frequently-full box.
//!
//! # Why every test pins an explicit worker count
//!
//! The first version of this file called the ambient-pool `dipole_band_width`
//! under `RAYON_NUM_THREADS=2`. At 2 workers the thread floor and the byte cap
//! coincide, so all three contracts passed against the unfixed tree — the test
//! was satisfied by accident and the bug hid. The overage is proportional to
//! worker count, so a test that cannot vary it cannot see the defect.
//! Measured on the shape below (natoms=3, nbf=24, budget = 2 partials):
//!
//! ```text
//!   threads |  width | resident vs budget
//!         2 |      2 |   1.0x   <- the blind spot
//!        12 |     12 |   6.0x   <- this box
//!        64 |     64 |  32.0x
//! ```

use ferric_rpa::properties::dipole_band_width_for_test as dipole_band_width;

/// Bytes one dipole chunk-partial occupies: `natoms * 3 * nbf^2 * f64`.
///
/// Deliberately re-derived here rather than imported, so the test also catches
/// the production formula silently changing shape.
fn per_partial_bytes(natoms: usize, nbf: usize) -> usize {
    natoms * 3 * nbf * nbf * std::mem::size_of::<f64>()
}

/// Worker counts to sweep. Spans "fewer workers than the budget admits"
/// (where floor and cap coincide) through many-core boxes where they diverge
/// sharply. 12 is this development box; 64 stands in for a fat node.
const THREAD_SWEEP: [usize; 4] = [1, 2, 12, 64];

const NATOMS: usize = 3;
const NBF: usize = 24;

/// CONTRACT 1: the band width must respond to the budget.
///
/// A width constant across a 64x budget range proves the budget is ignored.
/// The cheapest observable distinguishing a budget-aware knob from a hardcoded
/// one; needs no SCF.
#[test]
fn dipole_band_width_scales_with_budget() {
    let unit = per_partial_bytes(NATOMS, NBF);

    for nthreads in THREAD_SWEEP {
        // Both budgets are large relative to the worker floor, so neither is
        // clamped by it — this isolates budget-responsiveness from the floor.
        let small = dipole_band_width(NATOMS, NBF, unit * 128, nthreads);
        let large = dipole_band_width(NATOMS, NBF, unit * 8192, nthreads);

        assert!(
            large > small,
            "at {nthreads} workers: band width must grow with the budget, but a \
             128-partial budget gave {small} and a 8192-partial budget gave \
             {large} (identical => budget ignored)"
        );
    }
}

/// CONTRACT 2: the band width must never exceed what the budget pays for.
///
/// This is the 16-17 GB incident in miniature, and the one that FAILS today.
/// `dipole_band_width` floors its result at the rayon worker count, so a tight
/// budget is silently overridden: the caller asks for N partials and gets
/// `max(N, n_workers)`. A budget that can be exceeded by adding cores is not a
/// budget — memory must not be a function of core count.
///
/// The intended post-fix behavior is that the byte cap wins and parallelism
/// degrades instead (contract 3 guarantees it never degrades to zero).
#[test]
fn dipole_band_width_never_exceeds_budget() {
    let unit = per_partial_bytes(NATOMS, NBF);

    // A budget deliberately tighter than one-partial-per-core: the floor and the
    // cap disagree, so this detects which one wins.
    let n_partials = 2usize;
    let budget = unit * n_partials;

    for nthreads in THREAD_SWEEP {
        let width = dipole_band_width(NATOMS, NBF, budget, nthreads);
        let resident = width * unit;

        assert!(
            width <= n_partials,
            "at {nthreads} workers: band width {width} implies {resident} bytes \
             resident against a {budget}-byte budget ({:.1}x over) — the \
             thread-count floor is overriding the byte cap",
            resident as f64 / budget as f64,
        );
    }
}

/// CONTRACT 3: a single partial must always be representable.
///
/// The complement of contract 2: bounding memory must never yield a width of
/// zero, which would mean "make no progress". Even a starvation budget must
/// give at least one partial, so the algorithm degrades to *slow* rather than
/// *broken*. Guards the fix against overcorrecting.
#[test]
fn dipole_band_width_floors_at_one() {
    for nthreads in THREAD_SWEEP {
        for budget in [0usize, 1, 64] {
            assert!(
                dipole_band_width(NATOMS, NBF, budget, nthreads) >= 1,
                "at {nthreads} workers with a {budget}-byte budget: must still \
                 yield >=1 partial (slow, not stuck)"
            );
        }
    }
}
