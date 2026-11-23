//! `coordinates = "internal"` is refused, before any SCF, whenever the energy
//! is not invariant to a rigid rotation or translation of the optimized atoms.
//!
//! Physics hypothesis: a uniform field or fixed charges exert a net torque and
//! force; internal coordinates cannot rotate or translate the molecule, so the
//! search cannot converge, and the refusal is correct. Artifact hypothesis: the
//! guard is too broad (fires for an invariant energy) or too narrow (misses one
//! of the four fixed-frame terms, or one optimizer entry). The tests pin both
//! sides: every fixed-frame term is refused on every SCF entry point, and gas
//! phase, an empty `ExternalPotential` and PCM pass and converge.
//!
//! "Before any SCF" is tested by an unknown basis name: if the refusal came
//! after setup, the error would be the basis lookup, not the refusal.

use ferric_core::external_potential::{ExternalPotential, PointCharge, SmearedCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::{
    fixed_frame_cause, optimize_geometry, optimize_geometry_rohf, optimize_geometry_uhf,
    CoordSystem, OptimizeConfig,
};
use ferric_scf::polarizable::{PolarizableSite, PolarizableSites};
use ferric_scf::rhf::RhfConfig;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n",
        0,
        1,
    )
    .unwrap()
}

fn internal() -> OptimizeConfig {
    OptimizeConfig {
        coord_system: CoordSystem::RedundantInternal,
        ..Default::default()
    }
}

fn with_ext(ext: ExternalPotential) -> RhfConfig {
    let mut c = RhfConfig::default();
    c.external_potential = Some(ext);
    c
}

fn point_charge() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![PointCharge {
            q: 0.4,
            x: 4.0,
            y: 1.0,
            z: 0.5,
        }],
        ..Default::default()
    }
}

fn smeared() -> ExternalPotential {
    ExternalPotential {
        smeared_charges: vec![SmearedCharge {
            q: -0.4,
            x: 4.0,
            y: 1.0,
            z: 0.5,
            width: 1.0,
        }],
        ..Default::default()
    }
}

fn field() -> ExternalPotential {
    ExternalPotential {
        field: Some([0.02, 0.0, 0.0]),
        ..Default::default()
    }
}

fn polarizable() -> RhfConfig {
    let mut c = RhfConfig::default();
    c.polarizable = Some(PolarizableSites {
        sites: vec![PolarizableSite {
            x: 4.0,
            y: 0.0,
            z: 0.0,
            alpha: 5.0,
        }],
        ..Default::default()
    });
    c
}

/// Every fixed-frame configuration the guard must name.
fn fixed_frame_configs() -> Vec<(&'static str, RhfConfig, &'static str)> {
    vec![
        ("point charge", with_ext(point_charge()), "charges"),
        ("smeared charge", with_ext(smeared()), "charges"),
        ("field", with_ext(field()), "field"),
        ("polarizable", polarizable(), "polarizable"),
    ]
}

fn assert_refusal(tag: &str, err: &str, needle: &str) {
    assert!(err.contains("internal"), "{tag}: {err}");
    assert!(
        err.contains(needle),
        "{tag}: cause {needle:?} missing: {err}"
    );
    assert!(err.contains("cartesian"), "{tag}: no suggestion: {err}");
    assert!(
        !err.to_lowercase().contains("basis"),
        "{tag}: came after setup: {err}"
    );
}

#[test]
fn each_fixed_frame_term_is_refused_on_every_scf_entry_before_any_scf() {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    for (tag, cfg, needle) in fixed_frame_configs() {
        let e = optimize_geometry(&ctx, &water(), "no-such-basis", op, &cfg, &internal())
            .expect_err(tag);
        assert_refusal(&format!("rhf {tag}"), &e.to_string(), needle);
        let e = optimize_geometry_uhf(&ctx, &water(), "no-such-basis", op, &cfg, &internal())
            .expect_err(tag);
        assert_refusal(&format!("uhf {tag}"), &e.to_string(), needle);
        let e = optimize_geometry_rohf(&ctx, &water(), "no-such-basis", op, &cfg, &internal())
            .expect_err(tag);
        assert_refusal(&format!("rohf {tag}"), &e.to_string(), needle);
    }
}

#[test]
fn invariant_configurations_are_not_refused() {
    let mut pcm = RhfConfig::default();
    pcm.pcm = Some(ferric_pcm::PcmConfig::water());
    let cosmo = {
        let mut c = RhfConfig::default();
        c.cosmo = Some(ferric_scf::cosmo::CosmoConfig::default());
        c
    };
    for (tag, cfg) in [
        ("gas phase", RhfConfig::default()),
        (
            "empty external potential",
            with_ext(ExternalPotential::default()),
        ),
        ("pcm", pcm),
        ("cosmo", cosmo),
    ] {
        assert_eq!(fixed_frame_cause(&cfg), None, "{tag}");
        // Past the guard the unknown basis is the first thing to fail.
        let ctx = ParallelContext::default();
        let e = optimize_geometry(
            &ctx,
            &water(),
            "no-such-basis",
            Operator::coulomb(),
            &cfg,
            &internal(),
        )
        .expect_err(tag)
        .to_string();
        assert!(!e.contains("cannot optimize"), "{tag} was refused: {e}");
    }
}

#[test]
fn cartesian_in_a_field_is_not_refused_and_converges() {
    let ctx = ParallelContext::default();
    let cfg = with_ext(field());
    let opt = OptimizeConfig {
        max_steps: 60,
        ..Default::default()
    };
    let r = optimize_geometry(&ctx, &water(), "sto-3g", Operator::coulomb(), &cfg, &opt).unwrap();
    assert!(r.converged, "Cartesian water in a field: {} steps", r.steps);
}

#[test]
fn internal_with_pcm_converges() {
    let ctx = ParallelContext::default();
    let mut cfg = RhfConfig::default();
    cfg.pcm = Some(ferric_pcm::PcmConfig::water());
    let opt = OptimizeConfig {
        max_steps: 60,
        ..internal()
    };
    let r = optimize_geometry(&ctx, &water(), "sto-3g", Operator::coulomb(), &cfg, &opt).unwrap();
    assert!(r.converged, "internal water in PCM: {} steps", r.steps);
}
