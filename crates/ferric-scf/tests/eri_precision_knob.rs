//! The configurable ERI precision reaches the SCF J/K build.
//!
//! `set_eri_precision` is process-wide, so this file holds a single test (its
//! own test binary) and restores the default before returning.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine_pool::{eri_precision, set_eri_precision, EnginePool, ERI_PRECISION};
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{build_jk, build_jk_with_pool, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// sum(D K) from the public `build_jk`, which reads `eri_precision()`.
fn dk_via_knob(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
) -> f64 {
    let mut j = Array2::<f64>::zeros(d.raw_dim());
    let mut k = Array2::<f64>::zeros(d.raw_dim());
    build_jk(
        ctx,
        prep,
        bounds,
        RhfConfig::default().integral_thresh,
        d,
        &mut j,
        &mut k,
    )
    .unwrap();
    (d * &k).sum()
}

#[test]
fn eri_precision_knob_reaches_build_jk() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/molecules");
    let mol = Molecule::load_xyz(&format!("{root}/cs2.xyz")).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let d = solve_rhf(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-8,
            density_conv: 1e-6,
            ..Default::default()
        },
    )
    .unwrap()
    .density_total
    .clone();

    // Unscreened reference through an explicit pool (independent of the knob).
    let pool0 = EnginePool::new(op, &prep, 0.0).unwrap();
    let mut j = Array2::<f64>::zeros(d.raw_dim());
    let mut k = Array2::<f64>::zeros(d.raw_dim());
    build_jk_with_pool(
        &ctx,
        &prep,
        &bounds,
        RhfConfig::default().integral_thresh,
        &d,
        &mut j,
        &mut k,
        &pool0,
        ferric_scf::reduce::default_band_bytes(),
    )
    .unwrap();
    let dk0 = (&d * &k).sum();

    // Out-of-range values are refused and leave the setting unchanged.
    for bad in [-1e-20, 1e-6, f64::NAN] {
        assert!(set_eri_precision(Some(bad)).is_err(), "{bad:e} accepted");
    }

    set_eri_precision(Some(1e-14)).unwrap();
    assert_eq!(eri_precision(), 1e-14);
    let loose = dk_via_knob(&ctx, &prep, &bounds, &d) - dk0;

    // Set the default explicitly: clearing the override would fall through to
    // FERRIC_ERI_PRECISION if the environment sets it.
    set_eri_precision(Some(ERI_PRECISION)).unwrap();
    assert_eq!(eri_precision(), ERI_PRECISION);
    let default = dk_via_knob(&ctx, &prep, &bounds, &d) - dk0;
    set_eri_precision(None).unwrap();

    eprintln!(
        "CS2/cc-pVDZ sum(D K) vs precision 0: knob 1e-14 {loose:+.2e}, default {default:+.2e}"
    );
    // 1e-14 measured 7.6e-10 off; 1e-20 measured exact. A knob that never reached
    // the engines would leave both equal.
    assert!(
        loose.abs() > 1e-10,
        "knob at 1e-14 had no effect: {loose:.2e}"
    );
    assert!(
        default.abs() < 1e-12,
        "default missed unscreened by {default:.2e}"
    );
}
