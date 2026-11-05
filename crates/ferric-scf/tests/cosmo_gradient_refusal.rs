//! COSMO (`RhfConfig.cosmo`) has no nuclear-gradient term. Every analytic
//! gradient entry that takes the SCF config must refuse it instead of
//! returning the vacuum-form gradient of a solvated energy. IEF-PCM
//! (`RhfConfig.pcm`) has a gradient and stays accepted.
//!
//! Mutation check: delete `refuse_cosmo_gradient(config)?` from one entry in
//! `gradient.rs` (a dispatcher, or the matching `preflight_cosx_*`) and the
//! corresponding assertion below fails.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosmo::CosmoConfig;
use ferric_scf::gradient::{
    preflight_cosx_restricted, preflight_cosx_restricted_open, preflight_cosx_unrestricted,
    refuse_cosmo_gradient, restricted_open_scf_gradient, restricted_scf_gradient,
    unrestricted_scf_gradient,
};
use ferric_scf::optimize::{
    optimize_geometry, optimize_geometry_rohf, optimize_geometry_uhf, OptimizeConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const MSG: &str = "COSMO has no analytic gradient";

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.000000 0.000000 0.117300\n\
         H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
        0,
        1,
    )
    .unwrap()
}

fn cosmo() -> RhfConfig {
    RhfConfig {
        cosmo: Some(CosmoConfig::default()),
        ..Default::default()
    }
}

fn has_msg<T>(r: Result<T, ferric_core::error::FerricError>, what: &str) {
    match r {
        Ok(_) => panic!("{what}: COSMO gradient was not refused"),
        Err(e) => assert!(e.to_string().contains(MSG), "{what}: {e}"),
    }
}

#[test]
fn the_helper_refuses_cosmo_only() {
    has_msg(refuse_cosmo_gradient(&cosmo()), "helper");
    refuse_cosmo_gradient(&RhfConfig::default()).expect("vacuum config passes");
}

#[test]
fn preflights_refuse_cosmo_before_any_scf() {
    has_msg(preflight_cosx_restricted(&cosmo()), "restricted");
    has_msg(preflight_cosx_unrestricted(&cosmo()), "unrestricted");
    has_msg(preflight_cosx_restricted_open(&cosmo()), "restricted-open");
    // Exactness anchor: no COSMO, no refusal.
    preflight_cosx_restricted(&RhfConfig::default()).unwrap();
    preflight_cosx_unrestricted(&RhfConfig::default()).unwrap();
    preflight_cosx_restricted_open(&RhfConfig::default()).unwrap();
}

#[test]
fn dispatchers_refuse_a_cosmo_config_even_with_a_converged_result() {
    // Vacuum SCF result, COSMO config: isolates the dispatcher's own guard
    // from the SCF and from the preflights.
    let mol = water();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let c = cosmo();
    has_msg(
        restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &c, &res),
        "restricted_scf_gradient",
    );
    has_msg(
        unrestricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &c, &res),
        "unrestricted_scf_gradient",
    );
    has_msg(
        restricted_open_scf_gradient(&mol, &prep, &bs, op, &bounds, &c, &res),
        "restricted_open_scf_gradient",
    );
    // Anchor: the same vacuum result with a vacuum config differentiates.
    restricted_scf_gradient(&mol, &prep, &bs, op, &bounds, &RhfConfig::default(), &res)
        .expect("vacuum gradient");
}

#[test]
fn geometry_optimizers_refuse_cosmo() {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let opt = OptimizeConfig::default();
    let mol = water();
    has_msg(
        optimize_geometry(&ctx, &mol, "sto-3g", op, &cosmo(), &opt),
        "optimize_geometry (RHF/RKS)",
    );
    has_msg(
        optimize_geometry_uhf(&ctx, &mol, "sto-3g", op, &cosmo(), &opt),
        "optimize_geometry_uhf",
    );
    has_msg(
        optimize_geometry_rohf(&ctx, &mol, "sto-3g", op, &cosmo(), &opt),
        "optimize_geometry_rohf",
    );
}
