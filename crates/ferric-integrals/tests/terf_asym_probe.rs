//! Is the large-S asymptotic actually FASTER than the exact series, and by how
//! much do we pay in accuracy? Both numbers, measured, before committing to it.
use ferric_integrals::ffi;

#[test]
#[ignore = "micro-benchmark; run with --ignored --nocapture"]
fn asymptotic_vs_series_cost_and_accuracy() {
    let (mut ser, mut asy, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    // SAFETY: three live out-params; shim wraps in try/catch.
    let ok = unsafe { ffi::scf_terf_asym_probe(4, 20_000, &mut ser, &mut asy, &mut worst) };
    assert_eq!(ok, 1, "probe failed with status {ok}");
    eprintln!("  exact series : {ser:8.1} ns/call");
    eprintln!("  asymptotic   : {asy:8.1} ns/call   => {:.1}x faster", ser / asy.max(1e-9));
    eprintln!("  worst rel err over S in [25,400], s in [0.05,0.5]: {worst:.3e}");
}
