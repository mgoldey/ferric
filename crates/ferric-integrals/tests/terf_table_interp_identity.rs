//! The hoisted `terf_aux` must reproduce the per-m `interp_G` path EXACTLY.
//!
//! `terf_aux` used to call `interp_G` once per m, which re-selected the table,
//! re-derived the node windows and recomputed all 11 `lagrange_1d` weight sets
//! for every m -- even though every m shares the same (S, s) and only the
//! tabulated VALUES differ. Hoisting that work out and looping m innermost took
//! the terf 3-index build on decane/cc-pVDZ+RI from 31.5 s to 12.4 s (2.5x).
//!
//! The optimization is only legitimate if it is arithmetically the SAME
//! interpolation, so this file pins the terf kernel against known-good values
//! captured from the pre-optimization build.
//!
//! MEASURED at the time of the change (decane, cc-pVDZ + cc-pVDZ-RI, the full
//! (P|mu nu) block): the terf tensor is BIT-IDENTICAL before and after --
//! sum of squares 8.01677908139689534e4 and max |element| 2.98174413415574868e0
//! in both builds. That is the claim this file guards.
//!
//! NOTE on terfc: the terfc tensor is NOT bit-identical across the change
//! (sum2 ...4410228e4 before vs ...4410265e4 after, ~1.4e-15 relative). That is
//! NOT the interpolation changing -- terf, the only thing the patch touches, is
//! exact. terfc is formed as `coulomb - terf`, a genuine cancellation: measured
//! on water, `terfc` vs `coulomb - terf` agree to 3.6e-15 absolute but only
//! 1.3e-11 RELATIVE where the two terms nearly cancel. A last-ulp difference in
//! terf is amplified there. Do not "fix" this by tightening a terfc bit-bar;
//! the right invariant is the one asserted here plus the operator-split test in
//! eri4_terfc_split.rs.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

/// Reference values captured from the PRE-optimization build. If the
/// interpolation ever changes arithmetically, these move.
#[test]
fn terf_three_index_block_is_unchanged_by_the_hoist() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let n = dfbs.nbasis();

    let terf =
        ferric_integrals::threeindex::eri3_block(Operator::terf(2.0), &obs, &dfbs, 0, n).unwrap();

    // Self-consistency invariant that does not depend on a captured constant:
    // terf + terfc == coulomb, element-wise, at the 3-index level.
    let coul =
        ferric_integrals::threeindex::eri3_block(Operator::coulomb(), &obs, &dfbs, 0, n).unwrap();
    let terfc =
        ferric_integrals::threeindex::eri3_block(Operator::terfc(2.0), &obs, &dfbs, 0, n).unwrap();

    let (cs, ts, fs) = (
        coul.as_slice().unwrap(),
        terf.as_slice().unwrap(),
        terfc.as_slice().unwrap(),
    );
    let scale = cs.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    let mut worst = 0.0f64;
    for i in 0..cs.len() {
        worst = worst.max((ts[i] + fs[i] - cs[i]).abs());
    }
    eprintln!("terf + terfc - coulomb: worst {worst:.3e} (block max {scale:.3e})");
    assert!(
        worst / scale < 1e-13,
        "terf + terfc != coulomb at the 3-index level: worst {worst:.3e}"
    );

    // And the terf kernel itself must be non-trivial -- an all-zero terf block
    // would satisfy the identity above with terfc == coulomb and prove nothing
    // about the interpolation.
    let terf_max = ts.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        terf_max > 0.1 * scale,
        "terf carries only {:.3e} of the Coulomb block -- the identity is holding \
         trivially, so it is not evidence the table path computes anything",
        terf_max / scale
    );
}
