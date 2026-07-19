//! COSMO implicit-solvent correctness test: water in water (eps = 78.39).
//!
//! Cross-checked against a local PySCF checkout's own COSMO implementation
//! (`pyscf.solvent.pcm.PCM(method="COSMO")`), run with matching conventions
//! (unscaled Bondi radii table x 1.17, Lebedev order -> 110 points/atom,
//! eps = 78.39) at RHF/cc-pVDZ on the same geometry
//! (`testdata/molecules/water.xyz`):
//!
//! ```text
//! E_vacuum   = -76.0267679974 Ha
//! E_solvated = -76.0416829680 Ha (PySCF SWIG-COSMO, SAME radii/scale/eps as ferric)
//! E_cosmo (E_solvated - E_vacuum) = -0.0148298 Ha = -9.30 kcal/mol
//! ```
//!
//! NOTE (2026-07-19 investigation): the ORIGINAL version of this doc-comment
//! quoted PySCF's *own default* conventions (`vdw_scale=1.2` + PySCF's
//! `modified_Bondi` table, which overrides H to 1.1 Angstrom) as the
//! cross-check target: -5.97 to -6.57 kcal/mol. That was an apples-to-oranges
//! comparison -- ferric uses `radius_scale=1.17` with the *unmodified* Bondi
//! table (H=1.20 Angstrom). Re-running PySCF's COSMO with ferric's actual
//! radii/scale gives -9.30 kcal/mol (confirmed insensitive to Lebedev grid
//! density: 110 vs 302 points/atom changes it by <1%), which is the correct
//! apples-to-apples target.
//!
//! A SWIG-style smooth switching function (Lange & Herbert 2010, matching
//! PySCF's `pcm.py`) was added to `ferric_scf::cosmo::CosmoCavity::build` to
//! replace the previous hard point-in-sphere visibility trim, on the
//! hypothesis (from the pre-existing VALIDATION.md caveat) that this
//! discretization difference was the dominant source of the gap. Measured
//! result: it is NOT. With the switching function, ferric's solvation energy
//! moved from -3.38 to -3.39 kcal/mol (<0.5% change) -- essentially no
//! effect, still ~2.7x off from the corrected -9.30 kcal/mol PySCF target.
//!
//! Follow-up isolation (same SWIG cavity points/potential fed through both
//! ferric's bare-point-charge `S`-matrix formula and PySCF's Gaussian-smeared
//! `S`-matrix formula) shows the point-charge-vs-Gaussian-smearing choice is
//! a bigger lever than the switching function: -6.35 vs -10.6 kcal/mol on
//! that isolated comparison, a ~40% swing. This does not fully close ferric's
//! gap either (there is a further factor still unaccounted for between the
//! isolated -6.35 kcal/mol and ferric's actual pipeline result of -3.39
//! kcal/mol), so the remaining discrepancy is NOT attributed to any single
//! fixed cause here -- see `ferric_scf::cosmo` module docs for the full,
//! itemized list of simplifications still in place. The correctness bar for
//! this test remains order-of-magnitude, not a tight match:
//!
//! 1. The solvation energy is negative (stabilizing) -- required by physics
//!    for a polar solute in a polar solvent, not an assumption.
//! 2. Its magnitude is the right ORDER of magnitude for a small polar
//!    molecule (textbook range): roughly 1-15 kcal/mol, generously
//!    bracketing the simplified model's expected deviation from the
//!    corrected -9.30 kcal/mol PySCF reference.
//! 3. `cosmo: None` is exactly byte-identical to a build with no COSMO
//!    support at all (regression-guards the "None is a true no-op"
//!    convention used throughout this codebase for optional physics terms).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosmo::CosmoConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HARTREE_TO_KCAL: f64 = 627.509_474_063_1;

fn water_ccpvdz_rhf(config: &RhfConfig) -> ferric_scf::ScfResult {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    solve_rhf(&ctx, &mol, &prep, op, &bounds, config).unwrap()
}

#[test]
fn cosmo_none_is_byte_identical_to_no_cosmo_support() {
    let base = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    let with_none = RhfConfig {
        cosmo: None,
        ..base.clone()
    };
    let r_base = water_ccpvdz_rhf(&base);
    let r_none = water_ccpvdz_rhf(&with_none);
    assert!(r_base.converged && r_none.converged);
    assert_eq!(r_base.energy.to_bits(), r_none.energy.to_bits());
}

#[test]
fn cosmo_water_in_water_solvation_energy_is_negative_and_right_order_of_magnitude() {
    let tight = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };

    let vacuum = water_ccpvdz_rhf(&tight);
    assert!(vacuum.converged, "vacuum RHF did not converge");

    let cosmo_cfg = CosmoConfig {
        epsilon: 78.39,
        radius_scale: 1.17,
        lebedev_order: 110,
    };
    let solvated_config = RhfConfig {
        cosmo: Some(cosmo_cfg),
        ..tight
    };
    let solvated = water_ccpvdz_rhf(&solvated_config);
    assert!(solvated.converged, "solvated RHF did not converge");

    let e_solvation_ha = solvated.energy - vacuum.energy;
    let e_solvation_kcal = e_solvation_ha * HARTREE_TO_KCAL;

    eprintln!(
        "COSMO water/water: E_vacuum={:.10} Ha, E_solvated={:.10} Ha, \
         E_solvation={:.6} Ha = {:.4} kcal/mol \
         (PySCF SWIG-COSMO cross-check with MATCHED radii/scale: -0.0148 Ha = -9.30 kcal/mol)",
        vacuum.energy, solvated.energy, e_solvation_ha, e_solvation_kcal
    );

    // 1. Stabilizing (negative) -- the core physical requirement.
    assert!(
        e_solvation_ha < 0.0,
        "COSMO solvation energy must be negative (stabilizing) for a polar \
         solute in a polar solvent, got {e_solvation_ha:.6} Ha"
    );

    // 2. Right order of magnitude vs the PySCF cross-check (-9.30 kcal/mol
    // under MATCHED radii/scale/eps conventions -- see module doc-comment
    // above for how this differs from the original, radii-mismatched
    // -5.97/-6.57 figures). Bracket generously (1-15 kcal/mol): a SWIG
    // switching function was added to the cavity construction (2026-07-19)
    // but measured to have negligible effect (<0.5%) on this gap; the
    // dominant remaining difference is the point-charge (ferric) vs
    // Gaussian-smeared-charge (PySCF) segment representation, not yet
    // implemented here -- see `ferric_scf::cosmo` module docs.
    assert!(
        e_solvation_kcal.abs() > 1.0 && e_solvation_kcal.abs() < 15.0,
        "COSMO solvation energy magnitude {e_solvation_kcal:.4} kcal/mol is \
         outside the expected textbook range for water/water (PySCF \
         cross-check, matched conventions: -9.30 kcal/mol)"
    );
}

#[test]
fn cosmo_charges_have_correct_sign_and_magnitude_for_polar_water() {
    // The apparent surface charges should be predominantly of sign opposite
    // to the local solute potential (a conductor screens the field), and
    // their sum should roughly cancel the molecule's net charge contribution
    // to the far-field potential for a neutral solute (not exactly zero,
    // since COSMO is an approximate boundary condition, but not wildly off).
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let config = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    let vacuum = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

    let cosmo_cfg = CosmoConfig {
        epsilon: 78.39,
        radius_scale: 1.17,
        lebedev_order: 110,
    };
    let cavity = ferric_scf::cosmo::CosmoCavity::build(&mol, &cosmo_cfg).unwrap();
    let cr = ferric_scf::cosmo::cosmo_reaction_field(
        &mol,
        &prep,
        &cavity,
        &cosmo_cfg,
        &vacuum.density_total,
    )
    .unwrap();

    // Neutral molecule: total apparent surface charge should be small
    // relative to the number of segments (COSMO's conductor limit makes a
    // neutral solute's surface charge integrate close to zero, since there's
    // no net enclosed charge to screen).
    let total_q: f64 = cr.charges.sum();
    assert!(
        total_q.abs() < 0.05,
        "total COSMO surface charge for neutral water should be small, got {total_q}"
    );
    // Charges should have both signs (water has an internal dipole: O side
    // is screened oppositely from the H side).
    let n_pos = cr.charges.iter().filter(|&&q| q > 0.0).count();
    let n_neg = cr.charges.iter().filter(|&&q| q < 0.0).count();
    assert!(n_pos > 0 && n_neg > 0, "expected both signs of surface charge for polar water");
}
