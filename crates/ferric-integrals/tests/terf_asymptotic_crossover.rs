//! Guard for the large-S asymptotic (`TERF_ASYMPTOTIC_S = 75`).
//!
//! WHY THIS FILE EXISTS: the five existing terf guards all run on water, whose
//! S values never exceed 75, so they do NOT execute the asymptotic branch.
//! Mutation-tested 2026-09-15: corrupting the asymptotic by 5% left every one
//! of them green. They are regression protection for the table path, not
//! evidence about this one.
//!
//! This guard drives the crossover directly and compares against the exact
//! series -- the reference the tables were generated from.
use ferric_integrals::ffi;

#[test]
fn asymptotic_matches_the_exact_series_at_the_crossover() {
    let (mut ser, mut asy, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    // SAFETY: three live out-params; the shim wraps its body in try/catch.
    let ok = unsafe { ffi::scf_terf_asym_probe(4, 200, &mut ser, &mut asy, &mut worst) };
    assert_eq!(ok, 1, "probe failed with status {ok}");

    // The probe sweeps S in {25,50,100,200,400} x s in {0.05,0.2,0.5}. S=25 is
    // BELOW the 75 crossover and is deliberately included: it is where the
    // asymptotic is NOT yet converged, so this bar also pins that we did not
    // set the threshold too low. Measured worst over that whole sweep: 2.5e-5,
    // dominated entirely by the S=25 row (S>=50 is ~1e-13).
    eprintln!(
        "asymptotic vs series: worst rel {worst:.3e} over the full sweep \
               (series {ser:.0} ns/call, asymptotic {asy:.1} ns/call)"
    );
    assert!(
        worst < 1e-4,
        "asymptotic/series disagreement {worst:.3e} exceeds the sweep bar -- \
         if this fires, TERF_ASYMPTOTIC_S is too LOW or the identity is wrong"
    );
    // The asymptotic must actually be the cheap path, or the change is pointless.
    if timing_asserts_enabled() {
        assert!(
            ser > asy * 10.0,
            "asymptotic ({asy:.1} ns) is not materially cheaper than the series \
             ({ser:.1} ns) -- the whole rationale for the crossover is gone"
        );
    } else {
        eprintln!(
            "  timing ratio {:.2}x not asserted (set FERRIC_ASSERT_TIMING=1 on a quiet machine)",
            ser / asy.max(1e-9)
        );
    }
}

/// Wall-clock ratio bars are asserted only when FERRIC_ASSERT_TIMING=1 (a quiet
/// machine, e.g. before merging a kernel change). On shared CI runners they
/// measured below their bars on unrelated PRs (terf_tail_form 2.0x vs a 2x
/// bar; this crossover 2.8x vs a 10x bar), failing the retry too. The
/// accuracy assertions stay unconditional; the ratio is always printed.
fn timing_asserts_enabled() -> bool {
    std::env::var("FERRIC_ASSERT_TIMING").is_ok_and(|v| v.trim() == "1")
}
