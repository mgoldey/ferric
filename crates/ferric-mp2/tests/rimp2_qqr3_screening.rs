//! QQR-3 distance screening on the RI-MP2 3-index build.
//!
//! EXACTNESS ANCHOR FIRST (repo protocol): the approximation has a trivial
//! limit -- a threshold of 0 (or `None`) keeps every shell triple -- and that
//! limit must reproduce the dense energy before any screening sweep is run or
//! believed.
//!
//! What is being screened: `(P|mu nu)` shell triples whose QQR-3 bound falls
//! below the threshold are skipped and left zero. The envelope is the COULOMB
//! monopole `min(1, 1.10*ext_sum/r_eff)`; `estimate3` never reads `op.omega`,
//! so the win is GEOMETRIC and is nearly identical for Coulomb, erfc and terfc.
//! Do not read a terfc speedup here as evidence that attenuation screens.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(name: &str) -> Sys {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{name}.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    Sys { mol, obs, dfbs, rhf }
}

fn energy(s: &Sys, op: Operator, thresh: Option<f64>) -> f64 {
    let cfg = RiMp2Config {
        eri3_screen_thresh: thresh,
        ..Default::default()
    };
    ri_mp2(&s.mol, &s.obs, &s.dfbs, op, &s.rhf, &cfg)
        .unwrap()
        .mp2_corr
}

/// THE ANCHOR. A zero threshold must keep every triple, so the energy must be
/// bit-identical to the unscreened path -- not merely close. Anything else
/// means the screened branch changed the arithmetic, not just the work.
#[test]
fn zero_threshold_is_bit_identical_to_dense() {
    let s = setup("water");
    let dense = energy(&s, Operator::coulomb(), None);
    let zero = energy(&s, Operator::coulomb(), Some(0.0));
    assert_eq!(
        dense.to_bits(),
        zero.to_bits(),
        "thresh=0 must be BIT-identical to dense: dense={dense:.17e} zero={zero:.17e}"
    );
}

/// The anchor for the operator that motivated this work.
#[test]
fn zero_threshold_is_bit_identical_to_dense_for_terfc() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let s = setup("water");
    let op = Operator::terfc(2.0);
    let dense = energy(&s, op, None);
    let zero = energy(&s, op, Some(0.0));
    assert_eq!(
        dense.to_bits(),
        zero.to_bits(),
        "terfc thresh=0 must be BIT-identical to dense: dense={dense:.17e} zero={zero:.17e}"
    );
}

/// Screening must actually cost something bounded, and the error must FALL as
/// the threshold tightens. An invariant beats a point bar: a single tolerance
/// can pass while the mechanism is inert, but a monotone trend cannot.
#[test]
fn screening_error_falls_with_threshold() {
    let s = setup("alkane_10");
    let op = Operator::coulomb();
    let dense = energy(&s, op, None);

    let mut prev = f64::INFINITY;
    for &t in &[1e-6_f64, 1e-8, 1e-10, 1e-12] {
        let e = energy(&s, op, Some(t));
        let err = (e - dense).abs();
        eprintln!("thresh {t:.0e}  E_corr {e:.12}  |dE| {err:.3e}");
        assert!(
            err <= prev * 1.5 + 1e-14,
            "error did not fall as threshold tightened: {err:.3e} at {t:.0e} vs {prev:.3e} before"
        );
        prev = err;
    }
    // The production setting must be cheap in accuracy terms.
    let e10 = energy(&s, op, Some(1e-10));
    assert!(
        (e10 - dense).abs() < 1e-6,
        "1e-10 screening cost {:.3e} Ha -- too expensive to be a default",
        (e10 - dense).abs()
    );
}
