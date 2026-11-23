//! ADIIS/EDIIS must not report convergence on a stalled extrapolation (#422).
//!
//! The energy-based DIIS variants steer by an energy functional, not by the
//! commutator. In a strong external field EDIIS can freeze its weights on a few
//! old history entries (c ~ [0.41, 0.41, 0.19] for 30+ iterations): the
//! extrapolated Fock stops changing, `dp_rms` falls to zero, and the
//! density-based gate used to report `converged = true` with `err_max` ~ 0.84
//! and an energy 56 Ha too high (sulfamethoxazole/STO-3G in 2000 point charges,
//! every history entry idempotent, so the guess veto of #411 does not apply).
//!
//! The fix refuses that exit while the commutator is above the energy-to-Pulay
//! crossover and restarts plain Pulay DIIS from a clean history.
//!
//! 2026-11-19: the stalls below were driven by an EDIIS curvature twice the
//! exact HF interpolant (`diis::tests::ediis_model_equals_hf_energy_of_the_
//! interpolated_density`). With the exact factor none of these fields stalls;
//! the veto stays as a guard and is pinned at unit level.
//!
//! Exactness anchor: the energy equals plain Pulay DIIS from the same start.
//! Default Pulay is untouched by construction (it has no veto bound;
//! `diis::tests::veto_bound_and_pulay_fallback`). Cheap witnesses: benzene/
//! STO-3G in the same kind of field (seconds), where the SCF stops at
//! iteration 5 without the fix; the original 75-atom case is `#[ignore]`d.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::diis::DiisFlavor;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// 2000 deterministic +-0.8 e charges 9-23 Bohr from the centroid (>= 5 Bohr
/// from every atom), net -3 e: the field that stalls EDIIS.
fn field(mol: &Molecule, n: usize, seed: u64, net: f64) -> ExternalPotential {
    let mut st = seed;
    let mut rnd = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let na = mol.atoms.len() as f64;
    let (cx, cy, cz) = mol.atoms.iter().fold((0.0, 0.0, 0.0), |a, at| {
        (a.0 + at.x / na, a.1 + at.y / na, a.2 + at.zpos / na)
    });
    let mut pcs = Vec::new();
    while pcs.len() < n {
        let (x, y, z) = (rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0);
        let r = (x * x + y * y + z * z).sqrt();
        if !(1e-3..=1.0).contains(&r) {
            continue;
        }
        let rad = 9.0 + rnd() * 14.0;
        let (px, py, pz) = (cx + x / r * rad, cy + y / r * rad, cz + z / r * rad);
        if mol
            .atoms
            .iter()
            .any(|a| ((a.x - px).powi(2) + (a.y - py).powi(2) + (a.zpos - pz).powi(2)).sqrt() < 5.0)
        {
            continue;
        }
        let q = if rnd() < 0.5 { 0.8 } else { -0.8 } + net / n as f64;
        pcs.push(PointCharge {
            q,
            x: px,
            y: py,
            z: pz,
        });
    }
    ExternalPotential {
        point_charges: pcs,
        ..Default::default()
    }
}

fn solve(
    mol: &Molecule,
    ext: &ExternalPotential,
    flavor: DiisFlavor,
    max_iter: usize,
) -> ferric_scf::result::ScfResult {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cfg = RhfConfig::default()
        .with_max_iter(max_iter)
        .with_external_potential(ext.clone());
    cfg.diis_flavor = flavor;
    cfg.diis_switch_thresh = 1e-1;
    solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &cfg).unwrap()
}

/// Benzene/STO-3G in 600 and in 2000 charges (seed 11): before the EDIIS
/// curvature fix (2026-11-19) EDIIS reached the density-settled exit at
/// iteration 5 with err_max 0.12 / 0.45; without the veto it reported
/// `converged` at a wrong energy. Either way the run must match Pulay.
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn ediis_does_not_report_a_stalled_extrapolation_as_converged() {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    for n in [600usize, 2000] {
        let ext = field(&mol, n, 11, -3.0);
        let plain = solve(&mol, &ext, DiisFlavor::Pulay, 200);
        assert!(plain.converged, "n={n}: reference Pulay did not converge");
        let r = solve(&mol, &ext, DiisFlavor::Ediis, 200);
        assert!(r.converged, "n={n}: EDIIS did not converge");
        assert!(
            (r.energy - plain.energy).abs() < 1e-6,
            "n={n}: EDIIS converged to {} but Pulay gives {} (stalled extrapolation accepted)",
            r.energy,
            plain.energy
        );
        // The exit is a real one: the commutator has drained.
        assert!(
            r.iterations > 5,
            "n={n}: stopped at the stalled iteration {}",
            r.iterations
        );
    }
}

/// Benzene/STO-3G in 600 charges (seed 11) was the cheap stall witness: with
/// the former EDIIS curvature (twice the exact HF interpolant) the density
/// settled at iteration 5 with err_max 0.12 and the veto restarted on Pulay
/// (16 iterations). With the exact interpolant (2026-11-19) the stall is gone:
/// no fallback, converged to the Pulay energy in 15 iterations. A search over
/// benzene/water/CH3Br in 600-3000 charges x 6 seeds found no stall for
/// EDIIS or ADIIS, so the veto and its MaxIter report are pinned at unit level
/// (`rhf::tests::veto_refuses_a_converged_exit_above_the_energy_diis_crossover`,
/// `result::tests::unmet_and_describe_name_the_commutator_when_the_veto_refused_this_exit`).
#[test]
fn exact_ediis_curvature_does_not_stall_in_the_former_witness_field() {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let ext = field(&mol, 600, 11, -3.0);
    let r = solve(&mol, &ext, DiisFlavor::Ediis, 200);
    assert!(r.converged);
    let rep = r.convergence.expect("RHF carries a convergence report");
    assert!(
        rep.diis_fallback.is_none(),
        "EDIIS stalled again: {:?}",
        rep.diis_fallback
    );
    let plain = solve(&mol, &ext, DiisFlavor::Pulay, 200);
    assert!(
        (r.energy - plain.energy).abs() < 1e-8,
        "{} vs {}",
        r.energy,
        plain.energy
    );
}

/// The original #422 case (sulfamethoxazole/STO-3G, 2000 charges, net -3 e;
/// every history entry idempotent): without the fix EDIIS stopped at
/// iteration 36, err_max 0.84, E = -1103.651 Ha. Pulay gives -1159.3076324072
/// (pinned to 1e-7, from this construction). Slow (~100 s optimized): run
/// serially on a quiet box.
#[test]
#[ignore = "slow (~100 s); run serially on a quiet box"]
fn ediis_in_the_original_422_field_matches_pulay() {
    let mol = Molecule::load_xyz("../../testdata/molecules/sulfamethoxazole.xyz").unwrap();
    let ext = field(&mol, 2000, 7, -3.0);
    let r = solve(&mol, &ext, DiisFlavor::Ediis, 200);
    assert!(r.converged, "EDIIS did not converge");
    assert!(
        (r.energy - (-1159.3076324072)).abs() < 1e-7,
        "EDIIS converged to {} (Pulay: -1159.3076324072)",
        r.energy
    );
}
