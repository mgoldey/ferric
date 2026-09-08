//! End-to-end RHF with `k_builder = "cosx"` (water/cc-pVDZ) vs the direct
//! builder: converged energy difference and iteration count. Also the
//! honesty checks around the knob: DF-K active => cosx ignored (with a
//! warning) and the energy is exactly the DF-JK one.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::CosxConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::result::ScfResult;
use ferric_scf::screening::SchwarzBounds;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn run(cfg: RhfConfig) -> ScfResult {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let bs = bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).expect("rhf")
}

fn tight() -> RhfConfig {
    RhfConfig { energy_conv: 1e-10, density_conv: 1e-8, cosx: cosx_default(), ..Default::default() }
}

/// `COSX_ANCHOR_HALF=dense` runs this suite on the DENSE half transform
/// (absent / "sparse" = the default sparse path); see `cosx_k_anchors.rs`.
fn cosx_default() -> CosxConfig {
    use ferric_scf::cosx_k::CosxHalfTransform;
    let half = match std::env::var("COSX_ANCHOR_HALF").ok().as_deref() {
        None | Some("sparse") => CosxHalfTransform::SPARSE_DEFAULT,
        Some("dense") => CosxHalfTransform::Dense,
        Some(other) => panic!("COSX_ANCHOR_HALF = {other:?}: expected \"sparse\" or \"dense\""),
    };
    CosxConfig { half_transform: half, ..CosxConfig::default() }
}

#[test]
fn cosx_rhf_water_ccpvdz_vs_direct() {
    let direct = run(tight());
    let cosx = run(RhfConfig { k_builder: Some("cosx".into()), ..tight() });
    let cosx_nofit = run(RhfConfig {
        k_builder: Some("cosx".into()),
        cosx: CosxConfig { overlap_fit: false, ..cosx_default() },
        ..tight()
    });
    assert!(direct.converged && cosx.converged && cosx_nofit.converged);
    let de = cosx.energy - direct.energy;
    let de_nofit = cosx_nofit.energy - direct.energy;
    println!(
        "water/cc-pVDZ RHF: direct E={:.10} ({} iters); COSX(50,110)+fit E={:.10} ({} iters) dE={de:+.3e}; \
         COSX(50,110) no fit E={:.10} ({} iters) dE={de_nofit:+.3e}",
        direct.energy, direct.iterations, cosx.energy, cosx.iterations, cosx_nofit.energy, cosx_nofit.iterations
    );
    // Grid error at the default operating point: sub-0.1 mHa on this system
    // (K element error 5.6e-5 / 4.8e-5 at (50,110), anchors). A wrong K (sign,
    // weights, missing symmetrization) is O(1) Ha off — see the anchors'
    // mutation proofs.
    assert!(de.abs() < 1e-4, "COSX+fit energy off by {de:e} Ha");
    assert!(de_nofit.abs() < 1e-4, "COSX no-fit energy off by {de_nofit:e} Ha");
    assert!(de.abs() > 1e-12, "COSX energy equals direct to 1e-12 — the COSX K is not being used");
}

/// Refinement: the SCF energy error must fall with the grid (the SCF-level
/// echo of anchor (a)).
#[test]
fn cosx_rhf_energy_error_falls_with_grid() {
    let direct = run(tight());
    let mut errs = Vec::new();
    for (nr, na) in [(25usize, 50usize), (50, 110), (99, 302)] {
        let mut cosx = cosx_default();
        cosx.grid.n_radial = nr;
        cosx.grid.n_angular = na;
        let r = run(RhfConfig { k_builder: Some("cosx".into()), cosx, ..tight() });
        assert!(r.converged);
        let e = (r.energy - direct.energy).abs();
        println!("COSX ({nr},{na}) + fit: |dE| = {e:.3e} Ha");
        errs.push(e);
    }
    for w in errs.windows(2) {
        assert!(w[1] < w[0], "SCF energy error did not fall with the grid: {errs:?}");
    }
    assert!(errs[2] < 1e-6, "COSX (99,302) SCF energy error {:.3e} >= 1e-6 Ha", errs[2]);
}

/// With DF-K active the pluggable builder is ignored (warning on stderr) and
/// the energy is EXACTLY the DF-JK energy — never a half-wired hybrid.
#[test]
fn cosx_is_ignored_and_warned_when_df_k_is_active() {
    let aux = "def2-universal-jkfit";
    let df = run(RhfConfig { df_j_aux: Some(aux.into()), df_k_aux: Some(aux.into()), ..tight() });
    let df_cosx = run(RhfConfig {
        df_j_aux: Some(aux.into()),
        df_k_aux: Some(aux.into()),
        k_builder: Some("cosx".into()),
        ..tight()
    });
    assert!(df.converged && df_cosx.converged);
    assert_eq!(df.energy.to_bits(), df_cosx.energy.to_bits(), "DF-JK energy changed when k_builder=cosx was set");
    assert_eq!(df.iterations, df_cosx.iterations);
}

/// Unknown k_builder values are a hard error, and an untabulated COSX grid
/// order is a typed error rather than a Lebedev-table panic.
#[test]
fn cosx_config_errors_are_typed() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let bs = bundled("sto-3g").expect("sto-3g");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let bad = RhfConfig { k_builder: Some("cosxx".into()), ..Default::default() };
    assert!(solve_rhf(&ctx, &mol, &prep, op, &bounds, &bad).is_err());
    let mut cosx = cosx_default();
    cosx.grid.n_angular = 194;
    let bad_grid = RhfConfig { k_builder: Some("cosx".into()), cosx, ..Default::default() };
    assert!(solve_rhf(&ctx, &mol, &prep, op, &bounds, &bad_grid).is_err());
}
