//! Exactness anchors for the Fock-propagated screen of the amplitude-threshold
//! LMP2 mask (`AmplitudeLmp2Config::fock_screen`, theory.md §12.6).
//!
//! Protocol order (CLAUDE.md Experimental Protocol): the trivial limits come
//! first — screen ON at ε = 0 is canonical RI-MP2 on both assembly paths, and
//! a VACUOUS screen (κ → 0, keep every candidate) at finite ε is canonical
//! RI-MP2 on the integral-direct path with inert maps. The ragged-level
//! bitwise no-op of `fock_screen = None` is pinned in `ragged.rs`
//! (`screened_blocks_none_is_bitwise_plain_and_some_is_a_superset`).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config};
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    Setup {
        mol,
        obs,
        obs_bs,
        dfbs,
        rhf,
    }
}

/// Every map inert (the direct path's trivial limit, tests/lmp2_direct.rs).
fn trivial_maps() -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 1e6,
        virt_radius_bohr: None,
        ao_tail: 0.0,
        ..Default::default()
    }
}

#[test]
fn screen_on_at_eps_zero_matches_canonical_on_both_paths() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 0.0,
        frozen_core: 1,
        fock_screen: Some(1.0),
        ..Default::default()
    };
    let op = Operator::coulomb();
    let g = amplitude_lmp2(&su.mol, &su.obs, &su.obs_bs, &su.dfbs, op, &su.rhf, &cfg).unwrap();
    let (d, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &cfg,
        &trivial_maps(),
    )
    .unwrap();
    for (name, r) in [("global", &g), ("direct", &d)] {
        let de = (r.e_corr - r.e_corr_canonical_ri).abs();
        eprintln!("ANCHOR screen-on eps=0 {name}: dE={de:.3e}");
        assert!(r.keep_fraction == 1.0, "{name}: eps=0 must keep everything");
        assert!(
            de < 1e-9,
            "{name}: eps=0 with the screen on FAILED: {de:.3e}"
        );
    }
}

/// A vacuous screen (κ = 1e-300: every dressed integral clears κε) keeps the
/// whole candidate square of every pair; with inert maps the candidates are
/// all virtuals, so a finite-ε run IS canonical RI-MP2. Mutations seen to
/// fail (2026-11-16/17): passing `None` instead of `ctx.fock_screen` in the
/// direct path's `mask_pair` (keep < 1, the assert below), and retaining the
/// gate values instead of `g` in `pair_blocks_screened` (the run errors).
/// The ε = 0 anchor above cannot see either: the gate is not evaluated at
/// ε = 0, so both are equivalent mutants there.
#[test]
fn vacuous_screen_at_finite_eps_matches_canonical() {
    let su = setup("water.xyz");
    let cfg = AmplitudeLmp2Config {
        compute_reference: true,
        eps: 1e-3,
        frozen_core: 1,
        fock_screen: Some(1e-300),
        ..Default::default()
    };
    let (d, _) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
        &trivial_maps(),
    )
    .unwrap();
    let de = (d.e_corr - d.e_corr_canonical_ri).abs();
    eprintln!(
        "ANCHOR vacuous screen eps=1e-3: keep={} dE={de:.3e}",
        d.keep_fraction
    );
    assert!(
        d.keep_fraction == 1.0,
        "vacuous screen must keep every element"
    );
    assert!(de < 1e-9, "vacuous screen FAILED: {de:.3e}");
}
