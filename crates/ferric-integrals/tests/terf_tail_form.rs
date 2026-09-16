//! Gate for the truncated-tail rearrangement of the terf/terfc auxiliary
//! function `G_m(S,s)`.
//!
//! ## The identity
//!
//! `G_m(S,s)` is defined (see `terf-tables/terf_plan.md`,
//! `terf-tables/terfc_base_derivation.py`) as
//!
//! ```text
//! G_m(S,s) = SUM_{i>=0} df(2i) * Delta^m pmf_S(i) * cdf_s(i)
//!   pmf_S(i) = e^{-S} S^i / i!                (Poisson pmf)
//!   cdf_s(i) = e^{-s} SUM_{j<=i} s^j / j!     (Poisson cdf)
//!   df(0) = 1, df(2i) = df(2i-2) * (2i)/(2i+1)
//!   Delta^m = m-th forward difference in i, Delta^1 x(i) = x(i)-x(i-1), x(-1):=0
//! ```
//!
//! At s=0, `cdf_s(i) == 1` for every i, so `G_m(S,0) == F_m(S)` (the ordinary
//! Boys function) EXACTLY -- this holds because `cdf_s(i) = 1 - tail_s(i)`
//! with `tail_s(i) = e^{-s} SUM_{j>i} s^j/j!`, and splitting the sum gives the
//! EXACT rearrangement (not an approximation of any kind):
//!
//! ```text
//! G_m(S,s) = F_m(S) - Delta_m(S,s),   Delta_m(S,s) = SUM_i df(2i) Delta^m pmf_S(i) tail_s(i)
//! ```
//!
//! `tail_s(i)` is a Poisson upper tail: for i beyond s it collapses
//! super-exponentially, so `Delta_m(S,s)` truncated at a SMALL fixed `I`
//! converges independent of how large S is -- unlike the existing series,
//! which sums `S + 12*sqrt(S) + 60` terms (driven by S itself, so it gets
//! slow precisely when S is large). `terf_G_tail` is the fast path that
//! evaluates `F_m(S) - Delta_m(S,s;I)` instead of the full series.
//!
//! ## Measured numbers (see `terf-tables/terf_tail_reference.py`, mpmath
//! 200-bit; run via `/home/matt/qc/ferric/.venv/bin/python
//! terf-tables/terf_tail_reference.py`)
//!
//! - PROOF A (exactness anchor): `max |G_m(S,0) - F_m(S)|` over S in
//!   {0..300}, m in 0..8 is 3.7e-59 -- i.e. zero within the 200-bit floor.
//!   NOTE: naively re-deriving the ground-truth series with the C++-quoted
//!   term count `S + 12*sqrt(S) + 60` gives a spurious 5.8e-45 residual at
//!   S=300 that does NOT shrink when precision is raised to 400/800 bits --
//!   that count under-converges the SERIES itself (`df(2i)` decays only as
//!   the slow power law `sqrt(pi/(4i))`, not exponentially). The reference
//!   uses `S + 20*sqrt(S) + 200` instead and lands on the 200-bit floor at
//!   every S tested. This is a note about the reference's own convergence,
//!   not a defect in the identity (PROOF A passes cleanly once converged).
//! - PROOF B (s <= 0.5, the curvature-constrained regime, S up to 100, m up
//!   to 12): worst relative error of the truncated tail form vs exact G_m,
//!   by truncation length I:
//!       I= 8 -> 1.4e-5   I=10 -> 1.4e-10  I=12 -> 5.8e-13
//!       I=14 -> 6.3e-16  I=18 -> 4.5e-22
//!   Smallest I reaching 1e-14: **I = 14**.
//! - PROOF C (large s, the `terfc_with_omega` regime where s decouples from
//!   r0 and can reach ~80): smallest I reaching 1e-14 relative error, by s:
//!       s= 2  -> I=28     s=10 -> I=50     s=20 -> I=72     s=80 -> I=170
//!   Linear fit: `I(1e-14) ~= 1.76*s + 30.7`. So a FIXED I (14-18) is only
//!   safe for s <= 0.5; a probe that only tests small s could pass while an
//!   implementation with a fixed small I silently loses precision at large s.
//!
//! ## Tolerance chosen here
//!
//! This gate exercises `scf_terf_tail_probe`'s OWN sweep (mmax up to what the
//! caller requests, `reps` samples), not a hand-picked (S,s) pair, so the
//! bound must cover whatever regime that sweep spans. We assert
//! `worst_rel < 1e-12`: two orders of magnitude looser than the I=14 number
//! measured for the s<=0.5 regime (6.3e-16) to absorb f64-vs-mpmath rounding
//! differences (the C++ side runs in f64, ~1e-16 ULP, so a bound tighter than
//! ~1e-14 would be measuring double-precision noise, not the identity), while
//! still two orders of magnitude tighter than the I=8 failure mode (1.4e-5) --
//! so a probe that silently regressed to a too-small I, or that mixed up S
//! and s, could not sneak under this bar.
//!
//! ## Anti-inertness
//!
//! A WRONG implementation would look like one of:
//!   - `terf_G_tail` computing `F_m(S) - Delta_m(S,s;I)` with `Delta_m`
//!     missing the m-th forward difference (i.e. using `pmf_S(i)` directly
//!     instead of `Delta^m pmf_S(i)`) -- this reproduces m=0 exactly (Delta^0
//!     is the identity) but diverges from the series at m>=1, so a probe that
//!     tests only m=0 would NOT catch it. `scf_terf_tail_probe` takes `mmax`
//!     explicitly so this gate requests `mmax=8`, well above 0.
//!   - `Delta_m` summing `cdf_s(i)` instead of `tail_s(i)` (dropping the
//!     `1 -` complement) -- this would produce a result of roughly
//!     `F_m(S) - G_m(S,s) = tail-of-something-else`, wildly different in
//!     magnitude from `G_m(S,s)` itself, not just off by a small relative
//!     amount; `worst_rel` would blow up past 1.0, not sit near 1e-12.
//!   - a truncation length `I` fixed too small for the s actually swept
//!     (PROOF C shows I must scale with s) -- `worst_rel` would land in the
//!     1e-3..1e-1 range at the crossing points measured above, comfortably
//!     failing the 1e-12 bar without being so large it looks like a crash.
//! Any of these is caught by comparing against the untouched exact SERIES
//! (`series_ns`/the existing path), never against the interpolation tables --
//! the tables are the LESS accurate object here (measured 1.1e-5 at m=8 on
//! the S<=20 table, 1.9e-4 on the terfc auxiliary table), so anchoring to
//! them would raise, not lower, the achievable bar and could mask exactly the
//! kind of bug this gate exists to catch.

use ferric_integrals::ffi;
use std::os::raw::{c_double, c_int};

#[test]
fn terf_tail_form_matches_series_and_is_faster() {
    let mmax: c_int = 8;
    let reps: c_int = 200;
    let (mut tail_ns, mut series_ns, mut worst_rel): (c_double, c_double, c_double) =
        (0.0, 0.0, 0.0);
    let mut worst_i: c_int = -1;
    // The probe also reports the worst relative error on DELTA itself -- the
    // quantity production consumes (the terfc auxiliary is F - G == Delta).
    // Unlike G it stays well-conditioned at large s, where F - Delta is pure
    // subtractive cancellation. See the ACCURACY DOMAIN note in shim.cc.
    let mut worst_rel_delta: c_double = -1.0;

    // SAFETY: five live out-params (three f64, one i32 sample-count-shaped
    // slot below, one i32 here); the shim wraps its body in try/catch and
    // returns a negative status on internal failure, checked before any
    // output is trusted.
    let sampled: c_int = unsafe {
        ffi::scf_terf_tail_probe(
            mmax,
            reps,
            &mut tail_ns,
            &mut series_ns,
            &mut worst_rel,
            &mut worst_i,
            &mut worst_rel_delta,
        )
    };

    assert!(
        sampled > 0,
        "scf_terf_tail_probe returned status {sampled} (negative = internal error)"
    );
    // Anti-inertness: the probe must actually have exercised a nontrivial
    // sweep and a nontrivial truncation length, or a probe that samples
    // nothing (or silently uses I=0) could report worst_rel==0 vacuously.
    assert!(
        sampled >= 50,
        "only {sampled} samples compared -- the tail-form sweep is not \
         exercising terf_G_tail broadly enough to be evidence of anything"
    );
    assert!(
        worst_i > 0,
        "worst_I={worst_i} -- the probe never reports a truncation length, \
         so a vacuous (I=0, no terms summed) implementation could pass"
    );

    eprintln!(
        "terf tail form vs series: {sampled} samples, worst rel {worst_rel:.3e} \
         (worst at truncation I={worst_i}), tail {tail_ns:.1} ns/call, \
         series {series_ns:.1} ns/call => {:.1}x faster",
        series_ns / tail_ns.max(1e-9)
    );

    // See "Tolerance chosen here" above: 1e-12 sits two orders of magnitude
    // above the measured 6.3e-16 (I=14, s<=0.5 regime) f64-rounding floor and
    // two orders below the 1.4e-5 failure mode at I=8, so it is loose enough
    // to absorb f64 ULP noise but tight enough that none of the wrong
    // implementations described above could sneak under it.
    eprintln!("  worst rel on Delta (the terfc auxiliary): {worst_rel_delta:.3e}");
    assert!(
        worst_rel_delta >= 0.0,
        "probe did not populate worst_rel_delta -- signature or sweep is wrong"
    );
    assert!(
        worst_rel_delta < 1e-9,
        "Delta -- the quantity RI-MP2 actually consumes -- disagrees with the \
         reference by {worst_rel_delta:.3e}. This is the assertion that matters \
         most: G can cancel at large s, Delta cannot."
    );
    assert!(
        worst_rel < 1e-12,
        "tail form disagrees with the exact series by {worst_rel:.3e} \
         (worst truncation I={worst_i}) -- exceeds the 1e-12 bar derived from \
         terf-tables/terf_tail_reference.py PROOF B/C; the rearrangement is \
         wrong or the truncation bound I is too small for the (S,s) swept"
    );

    // THE POINT OF THE TAIL FORM: it must be materially cheaper than the
    // exact series, or there is no reason to add it. `terf_asymptotic_crossover.rs`
    // uses a 10x bar for its (looser, single-regime) asymptotic; the tail
    // form's whole rationale is replacing an O(S) series with an O(I) one at
    // large S, so require the same 10x floor here.
    assert!(
        series_ns > tail_ns * 10.0,
        "tail form ({tail_ns:.1} ns) is not materially cheaper than the exact \
         series ({series_ns:.1} ns) -- the rationale for this rearrangement \
         has evaporated"
    );
}
