//! Both states of an ω-tuning evaluation must use the SAME Coulomb treatment.
//!
//! The cation is range-separated UKS, and `solve_uhf` never density-fits J
//! when ω > 0. `solve_rhf` auto-selects RI-J for a functional when `df_j_aux`
//! is unset. With no resolution, the neutral ran RI-J and the cation exact J,
//! so the ΔSCF IP carried the neutral's fitting error (2.3e-5 to 2.8e-5 Ha for
//! ωB97X-V/def2-SVP H2O, ~1.4e-4 Bohr⁻¹ in ω*). `state_scf_config` therefore
//! resolves an unset `df_j_aux` to exact J and refuses a named aux basis.
//!
//! The end-to-end test carries its own negative control: the unresolved
//! default config must give DIFFERENT routes, so the equality it asserts for
//! the resolved config is not vacuous.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::omega_tuning::{state_scf_config, OmegaTuneConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;

const FUNCTIONAL: &str = "wB97X-V";

fn cfg_with(df_j_aux: Option<&str>) -> OmegaTuneConfig {
    OmegaTuneConfig {
        functional: FUNCTIONAL.into(),
        scf: RhfConfig {
            df_j_aux: df_j_aux.map(str::to_string),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn unset_resolves_to_exact_j_and_a_named_aux_is_refused() {
    let unset = state_scf_config(&cfg_with(None), 0.4).expect("unset is accepted");
    assert_eq!(
        unset.df_j_aux.as_deref(),
        Some(""),
        "unset must resolve to exact J"
    );
    assert_eq!(unset.xc_omega, Some(0.4));
    assert_eq!(
        state_scf_config(&cfg_with(Some("")), 0.4)
            .expect("explicit exact J is accepted")
            .df_j_aux
            .as_deref(),
        Some(""),
        "explicit exact J must pass through"
    );
    let err = state_scf_config(&cfg_with(Some("cc-pvdz-jkfit")), 0.4)
        .expect_err("a named RI-J aux cannot reach the RSH UKS cation, so it must be refused");
    assert!(
        err.to_string().contains("cc-pvdz-jkfit"),
        "the refusal must name the offending aux basis: {err}"
    );
}

fn j_route(r: &ScfResult) -> Option<String> {
    r.df_jk.as_ref().and_then(|d| d.j_aux.clone())
}

/// End to end on water/STO-3G, ωB97X-V at ω = 0.4: the neutral RKS and the
/// cation UKS record the same J route (exact J) under the resolved config,
/// and different routes under the unresolved one.
#[test]
fn neutral_and_cation_run_the_same_coulomb_route() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(
        "3\nwater\nO 0.000000 0.000000 0.117300\n\
         H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cation = mol.clone();
    cation.charge += 1;
    cation.multiplicity = 2;

    let resolved = state_scf_config(&cfg_with(None), 0.4).unwrap();
    let neutral = solve_rhf(&ctx, &mol, &prep, op, &bounds, &resolved).unwrap();
    let cat = solve_uhf(&ctx, &cation, &prep, &bounds, &resolved).unwrap();
    assert!(neutral.converged && cat.converged);
    assert_eq!(
        j_route(&neutral),
        j_route(&cat),
        "neutral and cation J routes differ"
    );
    assert!(j_route(&neutral).is_none(), "the resolved route is exact J");

    // Negative control: the same config with df_j_aux left unresolved.
    let unresolved = RhfConfig {
        df_j_aux: None,
        ..resolved
    };
    let neutral_raw = solve_rhf(&ctx, &mol, &prep, op, &bounds, &unresolved).unwrap();
    let cat_raw = solve_uhf(&ctx, &cation, &prep, &bounds, &unresolved).unwrap();
    assert_ne!(
        j_route(&neutral_raw),
        j_route(&cat_raw),
        "control: unresolved, the neutral auto-selects RI-J and the RSH cation \
         stays exact, so the routes must differ -- if they do not, the equality \
         above is not testing anything"
    );
}
