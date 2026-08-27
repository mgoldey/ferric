//! `ExternalPotential` threaded into U-OO-RI-MP2 (open-shell), Lane F1
//! follow-up: `u_oo_rimp2.rs` (open-shell OO-MP2) was named in spec §3
//! alongside the closed-shell `oo_rimp2.rs` but was not fixed in the first
//! pass. It carried the SAME two bugs as the closed-shell code (see
//! `crates/ferric-mp2/tests/oo_rimp2_external.rs`'s module doc comment for
//! the full derivation of both):
//!   1. `u_oo_ri_mp2` rebuilt a bare `oneelectron::hcore(obs)` internally,
//!      silently dropping any external potential.
//!   2. `compute_uhf_energy` computed `e_hf = e_elec + mol.nuclear_repulsion()`,
//!      missing the classical charge-nuclear/field-nuclear terms.
//!
//! Artifact hypothesis: if the bugs are real, U-OO-MP2 energy in a
//! point-charge field is IDENTICAL to vacuum (shift exactly 0.0, not just
//! small) because `ext` never reaches any hcore build inside the orbital-
//! optimization loop. If the fix threads `ext` correctly (both the hcore
//! term AND the classical vnn term), the U-OO-MP2 shift should track the UHF
//! shift closely, landing within 2e-3 Ha of the independently-measured UHF
//! shift for this exact system (−0.0122683 Ha, from
//! `testdata/reference/oh_sto-3g_uqmmm_plus_lonepair.json`).

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::u_oo_rimp2::{u_oo_ri_mp2, UOoRiMp2Config};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

/// OH doublet, same geometry as `scripts/gen_pyscf_qmmm_refs.py`'s
/// `oh_bohr()` (and therefore `testdata/reference/oh_sto-3g_uqmmm_plus_lonepair.json`):
/// O at the origin, H at z = 0.97 Å along the z-axis.
fn oh_radical() -> Molecule {
    let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
    Molecule::parse_xyz(xyz, 0, 2).unwrap()
}

fn plus_charge_field() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: -6.0 }],
        field: None,
    }
}

/// UHF shift measured independently in `qmmm_vs_pyscf.rs`/PySCF for this
/// exact geometry + charge (see the module doc comment).
const UHF_SHIFT: f64 = -0.012268307654181854;

fn tight_uoo_config() -> UOoRiMp2Config {
    UOoRiMp2Config {
        grad_conv: 1e-8,
        energy_conv: 1e-11,
        max_iter: 200,
        ..Default::default()
    }
}

/// Regression test for the hcore + vnn bugs: U-OO-MP2 energy in the field
/// minus vacuum U-OO-MP2 energy must land within 2e-3 Ha of the UHF shift.
/// Before the fix this difference was exactly 0.0 (ext silently dropped).
#[test]
fn u_oo_mp2_energy_shift_tracks_uhf_shift() {
    let mol = oh_radical();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ext = plus_charge_field();
    let uoo_config = tight_uoo_config();

    // Vacuum.
    let uhf_vac = solve_uhf(
        &ParallelContext::default(), &mol, &obs, &bounds,
        &RhfConfig { energy_conv: 1e-11, density_conv: 1e-10, ..Default::default() },
    ).unwrap();
    assert!(uhf_vac.converged);
    let uoo_vac = u_oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &uhf_vac, &uoo_config, None).unwrap();
    assert!(uoo_vac.converged, "vacuum U-OO-MP2 must converge");

    // In the field.
    let field_cfg = RhfConfig {
        external_potential: Some(ext.clone()),
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    let uhf_field = solve_uhf(&ParallelContext::default(), &mol, &obs, &bounds, &field_cfg).unwrap();
    assert!(uhf_field.converged);
    let uoo_field = u_oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &uhf_field, &uoo_config, Some(&ext)).unwrap();
    assert!(uoo_field.converged, "U-OO-MP2 in the field must converge");

    let uoo_shift = uoo_field.total_energy - uoo_vac.total_energy;
    let uhf_shift_measured = uhf_field.energy - uhf_vac.energy;

    eprintln!("=== U-OO-MP2 vs UHF embedding shift (OH/STO-3G, +1 charge at (0,0,-6) Bohr) ===");
    eprintln!("  UHF shift (measured here):  {uhf_shift_measured:.10} Ha");
    eprintln!("  UHF shift (PySCF ref):      {UHF_SHIFT:.10} Ha");
    eprintln!("  U-OO-MP2 shift:             {uoo_shift:.10} Ha");
    eprintln!("  |U-OO-MP2 shift - UHF shift (PySCF ref)| = {:.3e}", (uoo_shift - UHF_SHIFT).abs());

    // The bug: pre-fix, uoo_shift was exactly 0.0 (ext silently dropped
    // inside u_oo_ri_mp2's internal hcore rebuild + missing vnn term)
    // regardless of how large UHF_SHIFT is.
    assert!(
        uoo_shift.abs() > 1e-4,
        "U-OO-MP2 shift is suspiciously close to zero ({uoo_shift:.3e}) -- looks like the external \
         potential was dropped (the pre-fix bug), not a small physical effect"
    );
    assert!(
        (uoo_shift - UHF_SHIFT).abs() < 2e-3,
        "U-OO-MP2 shift {uoo_shift:.10} should be within 2e-3 Ha of the independently measured UHF \
         shift {UHF_SHIFT:.10} (diff {:.3e})",
        (uoo_shift - UHF_SHIFT).abs()
    );
}

/// Exactness anchor: `ext = None` reproduces the pre-change vacuum U-OO-MP2
/// energy for this geometry (no external-potential machinery exercised at
/// all, so this must be identical to a build without the `ext` parameter).
/// The numeric value below was captured from a run of this exact test BEFORE
/// `ext` was threaded through `u_oo_ri_mp2` (i.e. the pre-change signature),
/// pinning that the signature change alone introduces no behavior change.
#[test]
fn u_oo_mp2_ext_none_matches_vacuum() {
    let mol = oh_radical();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uoo_config = tight_uoo_config();

    let uhf = solve_uhf(
        &ParallelContext::default(), &mol, &obs, &bounds,
        &RhfConfig { energy_conv: 1e-11, density_conv: 1e-10, ..Default::default() },
    ).unwrap();
    assert!(uhf.converged);

    let uoo_none = u_oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &uhf, &uoo_config, None).unwrap();
    let empty = ExternalPotential::default();
    let uoo_empty = u_oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &uhf, &uoo_config, Some(&empty)).unwrap();

    assert!(uoo_none.converged && uoo_empty.converged);
    assert_eq!(
        uoo_none.total_energy.to_bits(),
        uoo_empty.total_energy.to_bits(),
        "None vs Some(default) total_energy not bit-identical: {:.17e} vs {:.17e}",
        uoo_none.total_energy, uoo_empty.total_energy,
    );

    eprintln!("U-OO-MP2 vacuum total_energy (ext=None): {:.10}", uoo_none.total_energy);
}
