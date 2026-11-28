//! The DF-guess decision (`ladder::DfGuessPlan::decide` / `df_guess_blocker`)
//! and the ladder form of the pre-stage (`ladder::solve_rhf_ladder_with_df_guess`).
//!
//! The plan is the rule the Python `run_rhf` / `run_dft` and the CLI `rhf` /
//! `ksdft` ladder share: unset runs the pre-stage only where the SCF has a
//! four-centre build to save and the caller supplied no starting point of its
//! own; an explicit `true` that cannot apply is an error; `false` never runs.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{
    default_ladder_from, df_guess_blocker, solve_rhf_ladder, solve_rhf_ladder_with_df_guess,
    solve_rhf_with_df_guess, DfGuessPlan, DF_GUESS_DEFAULT_AUX, DF_GUESS_DEFAULT_ON,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
}

/// Exact four-centre HF with the given thresholds.
fn exact(e: f64, d: f64) -> RhfConfig {
    RhfConfig::default()
        .with_energy_conv(e)
        .with_density_conv(d)
}

#[test]
fn unset_follows_the_default_on_an_exact_closed_shell_scf() {
    let plan = DfGuessPlan::decide(None, &RhfConfig::default(), &water(), None, None).unwrap();
    assert_eq!(plan.run, DF_GUESS_DEFAULT_ON);
    assert_eq!(plan.aux, DF_GUESS_DEFAULT_AUX);
    assert_eq!(
        df_guess_blocker(&RhfConfig::default(), &water(), None),
        None
    );
}

#[test]
fn false_never_runs_and_true_runs_where_it_applies() {
    let mol = water();
    let off = DfGuessPlan::decide(Some(false), &RhfConfig::default(), &mol, None, None).unwrap();
    assert!(!off.run);
    assert!(off.reason.is_some());
    let on = DfGuessPlan::decide(Some(true), &RhfConfig::default(), &mol, None, None).unwrap();
    assert!(on.run);
    assert_eq!(on.reason, None);
}

/// Every gate condition: each blocks an unset request (with its reason) and
/// turns an explicit `true` into an error naming it.
#[test]
fn each_gate_condition_blocks_with_its_reason() {
    let mol = water();
    let fitted = |mut c: RhfConfig| {
        c.df_j_aux = Some(DF_GUESS_DEFAULT_AUX.into());
        c.df_k_aux = Some(DF_GUESS_DEFAULT_AUX.into());
        c
    };
    let n = 7; // water/STO-3G
    let cases: Vec<(&str, RhfConfig)> = vec![
        ("starting density", {
            let mut c = RhfConfig::default();
            c.init_guess_density = Some(ndarray::Array2::zeros((n, n)));
            c
        }),
        ("core-Hamiltonian", {
            let mut c = RhfConfig::default();
            c.use_sad_guess = false;
            c
        }),
        ("MOM", {
            let mut c = RhfConfig::default();
            c.mom_after_iter = 3;
            c
        }),
        ("symmetry", {
            let mut c = RhfConfig::default();
            c.symmetry = true;
            c
        }),
        ("no four-centre", fitted(RhfConfig::default())),
    ];
    for (needle, cfg) in cases {
        let why = df_guess_blocker(&cfg, &mol, None)
            .unwrap_or_else(|| panic!("{needle}: the gate let the pre-stage through"));
        assert!(why.contains(needle), "{needle}: reason was {why:?}");
        let plan = DfGuessPlan::decide(None, &cfg, &mol, None, None).unwrap();
        assert!(!plan.run, "{needle}: unset must stay off");
        assert_eq!(plan.reason.as_deref(), Some(why.as_str()));
        let err = DfGuessPlan::decide(Some(true), &cfg, &mol, None, None).unwrap_err();
        assert!(format!("{err}").contains(needle), "{needle}: {err}");
    }
    // A caller-level condition (a Fock hook, a gradient task) blocks the same way.
    let plan = DfGuessPlan::decide(
        None,
        &RhfConfig::default(),
        &mol,
        None,
        Some("a fock_hook is set"),
    )
    .unwrap();
    assert!(!plan.run);
    assert_eq!(plan.reason.as_deref(), Some("a fock_hook is set"));
}

/// Exact J next to fitted K, and fitted J next to exact K, both keep a
/// four-centre build: the pre-stage applies.
#[test]
fn one_four_centre_build_is_enough() {
    let mol = water();
    let mut exact_j = RhfConfig::default();
    exact_j.df_k_aux = Some(DF_GUESS_DEFAULT_AUX.into());
    assert_eq!(df_guess_blocker(&exact_j, &mol, None), None);
    let mut exact_k = RhfConfig::default();
    exact_k.df_j_aux = Some(DF_GUESS_DEFAULT_AUX.into());
    assert_eq!(df_guess_blocker(&exact_k, &mol, None), None);
}

#[test]
fn an_aux_that_misses_an_element_blocks() {
    let xe = Molecule::load_xyz("../../testdata/molecules/validation/ecp/xe.xyz").unwrap();
    let why = df_guess_blocker(&RhfConfig::default(), &xe, Some("cc-pvtz-jkfit")).unwrap();
    assert!(why.contains("no shells for Xe"), "{why}");
    assert_eq!(df_guess_blocker(&RhfConfig::default(), &xe, None), None);
}

/// The plain pre-stage on a fixed small case: the same energy as the exact SCF
/// without it, in fewer exact builds.
#[test]
fn the_pre_stage_saves_exact_builds_at_the_same_energy() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let base = exact(1e-10, 1e-8);
    let plain = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base).unwrap();
    let dfg = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &base, None).unwrap();
    assert!(plain.converged && dfg.result.converged);
    assert!(dfg.df_iterations > 0);
    assert!(
        (plain.energy - dfg.result.energy).abs() < 1e-9,
        "{} vs {}",
        plain.energy,
        dfg.result.energy
    );
    assert!(
        dfg.result.iterations < plain.iterations,
        "exact builds: {} with the pre-stage, {} without",
        dfg.result.iterations,
        plain.iterations
    );
}

/// The ladder form: rung 0 starts from the pre-stage's density, converges to
/// the plain ladder's energy, in fewer exact builds.
#[test]
fn the_ladder_pre_stage_saves_exact_builds_at_the_same_energy() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let ladder = default_ladder_from(&exact(1e-10, 1e-8));
    let plain = solve_rhf_ladder(&ctx, &mol, &prep, op, &bounds, &ladder).unwrap();
    let dfg =
        solve_rhf_ladder_with_df_guess(&ctx, &mol, &prep, op, &bounds, &ladder, None).unwrap();
    assert!(plain.converged && dfg.result.converged);
    assert_eq!((plain.rung_reached, dfg.result.rung_reached), (0, 0));
    assert!(dfg.df_iterations > 0 && !dfg.skipped && dfg.fallback.is_none());
    assert!((plain.result.energy - dfg.result.result.energy).abs() < 1e-9);
    assert!(
        dfg.result.result.iterations < plain.result.iterations,
        "exact builds: {} with the pre-stage, {} without",
        dfg.result.result.iterations,
        plain.result.iterations
    );
}
