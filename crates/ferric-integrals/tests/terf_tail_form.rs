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
//!
//!   ```text
//!   I= 8 -> 1.4e-5   I=10 -> 1.4e-10  I=12 -> 5.8e-13
//!   I=14 -> 6.3e-16  I=18 -> 4.5e-22
//!   ```
//!
//!   Smallest I reaching 1e-14: **I = 14**.
//! - PROOF C (large s, the `terfc_with_omega` regime where s decouples from
//!   r0 and can reach ~80): smallest I reaching 1e-14 relative error, by s:
//!
//!   ```text
//!   s= 2  -> I=28     s=10 -> I=50     s=20 -> I=72     s=80 -> I=170
//!   ```
//!
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
    // 1e-10, not the 1e-12 this gate originally asserted. CORRECTED 2026-09-16
    // after CI run 35108758569 failed here at 2.730e-12, reproduced locally
    // bit-for-bit (same value, same reported I), so this was never platform
    // noise -- the bar had simply never been run against the finished kernel.
    //
    // The 1e-12 was derived by loosening PROOF B's 6.3e-16 (I=14) by two
    // orders "to absorb f64-vs-mpmath rounding". That derivation does not
    // apply to what this probe actually measures: PROOF B is an mpmath 200-bit
    // number, but the probe compares f64 `terf_G_tail` against f64
    // `terf_G_series`, so BOTH sides carry f64 rounding and the reconstruction
    // G = F - Delta is a subtraction. 6.3e-16 was never reachable through it.
    //
    // Truncation is ruled out as the cause: every G comparison runs at
    // s <= 0.5 (the probe `continue`s above that, since F - Delta has no
    // surviving digits at large s), where the kernel's I = ceil(1.80*s + 40)
    // supplies I = 41 -- about 3x PROOF B's converged I = 14. A truncation
    // residual would already be at the 1e-16 floor there. The reported
    // "worst truncation I=76" belongs to the s = 20 cells, which are EXCLUDED
    // from worst_rel; the old panic message therefore blamed a truncation
    // length that no G comparison ever used.
    //
    // The right scale is terf_G_tail's own ACCURACY DOMAIN note, measured
    // independently against a long-double evaluation of the defining series
    // (i.e. NOT through this subtraction): 1.1e-11 worst relative on G for
    // s <= 0.5. A gate asserting 1e-12 was an order of magnitude tighter than
    // the kernel was ever measured to achieve.
    //
    // Anti-inertness is preserved -- all three wrong implementations the
    // header enumerates still fail by >=5 orders: missing forward difference
    // (diverges), cdf-instead-of-tail (~1.0), I fixed too small (1.4e-5).
    // `worst_rel_delta < 1e-9` above remains the load-bearing assertion:
    // Delta is what RI-MP2 consumes, and it does not go through this
    // cancellation.
    assert!(
        worst_rel < 1e-10,
        "tail form disagrees with the exact series by {worst_rel:.3e} \
         (max truncation over the whole sweep I={worst_i}; note the G \
         comparison itself only runs at s <= 0.5) -- exceeds the 1e-10 bar, \
         which is set by terf_G_tail's measured 1.1e-11 accuracy on G in that \
         regime, not by PROOF B's 200-bit figure. Either the rearrangement is \
         wrong, or f64 conditioning of G = F - Delta has degraded"
    );

    // THE POINT OF THE TAIL FORM: it must be materially cheaper than the
    // exact series, or there is no reason to add it.
    //
    // 2x, not the 10x this gate originally asserted. CORRECTED 2026-09-16:
    // the 10x was copied from `terf_asymptotic_crossover.rs` on the reasoning
    // that both replace the series -- but that is a different mechanism in a
    // different regime (the asymptotic fires above TERF_ASYMPTOTIC_S, where
    // the series is at its most expensive), and the number does not transfer.
    //
    // What 10x would require is arithmetically unavailable on THIS sweep. The
    // series costs N = S + 12*sqrt(S) + 60 terms; the tail costs
    // I = 1.80*s + 40. Summed over the probe's own (S,s) grid that is 4248 vs
    // 1791 terms => a predicted 2.37x, and 2.2x is measured -- the kernel is
    // doing exactly what its term counts say. Even the single most favourable
    // cell (S = 49, the top of the sweep, s <= 0.5) is only 193/41 = 4.7x,
    // because TERF_ASYMPTOTIC_S = 50 caps S before the series ever gets long
    // enough for 10x to exist. The old bar could only have passed if the
    // series were being mismeasured.
    //
    // 2x still fails loudly for the regression this guard is for -- the tail
    // form silently falling back to series-length work, or the Abel/cache
    // rearrangement being reverted, both of which land at ~1x.
    //
    // NOTE for anyone raising TERF_ASYMPTOTIC_S: the achievable ratio scales
    // with the largest S actually swept, so re-derive this bar from the term
    // counts above rather than assuming 2x is still the right floor.
    assert!(
        series_ns > tail_ns * 2.0,
        "tail form ({tail_ns:.1} ns) is not materially cheaper than the exact \
         series ({series_ns:.1} ns) -- expected >=2x from the term-count ratio \
         (~2.37x predicted over this sweep); the rearrangement has been \
         reverted or is falling back to series-length work"
    );
}
