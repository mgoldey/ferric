//! `ferric_scf::scf_jk`: the one rule that maps an MP2-family entry point's
//! `scf_jk` / `jk_aux` onto `RhfConfig`, exercised through a real SCF.
//!
//! PRE-REGISTERED:
//!
//! * EXACTNESS ANCHORS. `"exact"` through the rule is the untouched default
//!   config (bitwise), and `"rijk"` / `"rijcosx"` through the rule are the
//!   hand-written `df_j_aux` / `df_k_aux` / `k_builder` configs (bitwise).
//! * SEPARATION. The fitted modes must move the energy by the fitting error
//!   (expected 1e-6..1e-4 Ha on water/6-31G), so a rule that silently mapped a
//!   fitted mode to `"exact"`, dropped a named `jk_aux`, or put the aux on the
//!   wrong side of RIJCOSX fails here. ARTIFACT HYPOTHESIS: a broken rule gives
//!   a bitwise-equal energy to the wrong reference, which these asserts reject.
//!
//! Command:
//!   OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-scf --test scf_jk_reference

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::scf_jk::{ScfJk, ScfJkKeys, ScfJkMode, DEFAULT_SCF_JK_AUX};
use ferric_scf::screening::SchwarzBounds;

const WATER: &str =
    "3\nwater\nO 0.0 0.0 0.11779\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";

fn energy(cfg: &RhfConfig) -> Result<f64, ferric_core::FerricError> {
    let mol = Molecule::parse_xyz(WATER, 0, 1)?;
    let bs = bundled("6-31g")?;
    let prep = PreparedBasis::new(&mol, &bs)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)?;
    let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, cfg)?;
    assert!(r.converged);
    Ok(r.energy)
}

fn through_rule(mode: &str, aux: Option<&str>) -> RhfConfig {
    let jk = ScfJk::resolve(Some(mode), aux, None, ScfJkMode::Exact, ScfJkKeys::PYTHON)
        .expect("valid scf_jk");
    let mut cfg = RhfConfig::default();
    jk.apply(&mut cfg);
    cfg
}

fn named(j: Option<&str>, k: Option<&str>, k_builder: Option<&str>) -> RhfConfig {
    RhfConfig {
        df_j_aux: j.map(str::to_string),
        df_k_aux: k.map(str::to_string),
        k_builder: k_builder.map(str::to_string),
        ..Default::default()
    }
}

#[test]
fn exact_through_the_rule_is_the_default_config_bitwise() {
    let default = energy(&RhfConfig::default()).unwrap();
    assert_eq!(energy(&through_rule("exact", None)).unwrap(), default);
}

#[test]
fn rijk_is_both_fits_on_one_jk_basis_and_moves_the_energy() {
    let exact = energy(&RhfConfig::default()).unwrap();
    let rijk = energy(&through_rule("rijk", None)).unwrap();
    let j = Some(DEFAULT_SCF_JK_AUX);
    assert_eq!(rijk, energy(&named(j, j, None)).unwrap());
    let d = (rijk - exact).abs();
    assert!((1e-7..1e-3).contains(&d), "RI-JK fitting error {d:e}");
}

#[test]
fn a_named_jk_aux_reaches_both_fits() {
    let default_aux = energy(&through_rule("rijk", None)).unwrap();
    let tz = energy(&through_rule("rijk", Some("cc-pvtz-jkfit"))).unwrap();
    let a = Some("cc-pvtz-jkfit");
    assert_eq!(tz, energy(&named(a, a, None)).unwrap());
    assert!(
        (tz - default_aux).abs() > 1e-8,
        "jk_aux changed nothing: {tz} vs {default_aux}"
    );
}

#[test]
fn rijcosx_is_ri_j_plus_cosx_k() {
    let rijcosx = energy(&through_rule("rijcosx", None)).unwrap();
    let j = Some(DEFAULT_SCF_JK_AUX);
    assert_eq!(rijcosx, energy(&named(j, None, Some("cosx"))).unwrap());
    let rijk = energy(&through_rule("rijk", None)).unwrap();
    let exact_j_cosx = energy(&named(None, None, Some("cosx"))).unwrap();
    assert!((rijcosx - rijk).abs() > 1e-8, "COSX K not used");
    assert!((rijcosx - exact_j_cosx).abs() > 1e-8, "RI-J not used");
}
