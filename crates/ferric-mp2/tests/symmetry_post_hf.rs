//! Post-HF consumers of a point-group-blocked RHF (issue #294).
//!
//! A blocked result lives in the symmetry frame, so it is a distinct type
//! (`SymmetricRhf`) that cannot be passed to `ri_mp2` or `rhf_gradient` by
//! accident. Used as documented (`scf_in_frame()` with `molecule()`), the
//! consumers must reproduce the unblocked answer on the same frame molecule.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::symmetric::solve_rhf_symmetric;
use ferric_scf::{rhf_gradient, ScfResult};

fn nh3() -> Molecule {
    Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/nh3.xyz"
    ))
    .unwrap()
}

fn cfg(symmetry: bool) -> RhfConfig {
    RhfConfig {
        symmetry,
        density_conv: 1e-10,
        energy_conv: 1e-12,
        ..RhfConfig::default()
    }
}

#[test]
fn ri_mp2_and_gradient_on_the_frame_molecule_match_unblocked() {
    let mol = nh3();
    let bs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();

    let prep_in = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds_in = SchwarzBounds::compute(op, &prep_in).unwrap();
    let sym = solve_rhf_symmetric(&ctx, &mol, &prep_in, op, &bounds_in, &cfg(true)).unwrap();
    assert_eq!(sym.group(), "Cs");

    // Reference: the unblocked solve on the SAME (frame) molecule.
    let fm = sym.molecule();
    let prep = PreparedBasis::new(fm, &bs).unwrap();
    let dfbs = PreparedBasis::new(fm, &aux).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let off: ScfResult = solve_rhf(&ctx, fm, &prep, op, &bounds, &cfg(false)).unwrap();

    let mp2 = |r: &ScfResult| {
        ri_mp2(fm, &prep, &dfbs, op, r, &RiMp2Config::default())
            .unwrap()
            .mp2_corr
    };
    let (e_on, e_off) = (mp2(sym.scf_in_frame()), mp2(&off));
    assert!((e_on - e_off).abs() < 1e-9, "RI-MP2 {e_on} vs {e_off}");
    assert!(
        e_on < -0.01 && e_on > -0.5,
        "implausible correlation {e_on}"
    );

    let g_on = rhf_gradient(fm, &prep, op, &bounds, sym.scf_in_frame(), None).unwrap();
    let g_off = rhf_gradient(fm, &prep, op, &bounds, &off, None).unwrap();
    let worst = (&g_on - &g_off).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(worst < 1e-8, "gradient differs by {worst:e}");
}
