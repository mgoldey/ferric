//! `ExternalPotential` threaded into OO-RI-MP2 (Lane F1, task F1-3).
//!
//! `oo_ri_mp2` rebuilt a bare `oneelectron::hcore(obs)` internally, silently
//! dropping any external potential the caller's `RhfConfig` had set — even
//! though the starting-orbital `ScfResult` passed in was itself solved WITH
//! the potential. This is a bug (the OO-MP2 orbital-rotation loop, and every
//! energy/Fock rebuild inside it, quietly reverted to vacuum), not a design
//! choice: `oo_ri_mp2_gradient` had the identical defect.
//!
//! Artifact hypothesis: if the bug is real, OO-MP2 energy in a point-charge
//! field is IDENTICAL to vacuum OO-MP2 (the field-in minus field-out
//! difference is exactly 0.0, not just small) because `ext` never reaches any
//! hcore build inside the orbital-optimization loop. If the fix threads `ext`
//! correctly, the OO-MP2 shift should track the RHF shift closely (both are
//! dominated by the same one-electron charge-electron/charge-nuclear terms;
//! MP2 correlation reweights orbitals only at second order), landing within
//! the plan's 2e-3 Ha tolerance of the independently-measured RHF shift
//! (−0.018357900 Ha, from `testdata/reference/water_sto-3g_qmmm_plus_lonepair.json`,
//! same geometry/charge used here).

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::oo_rimp2_gradient::oo_ri_mp2_gradient;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Same geometry as `scripts/gen_pyscf_qmmm_refs.py`'s `water_bohr()` (and
/// therefore `testdata/reference/water_sto-3g_qmmm_plus_lonepair.json`): O at
/// the origin, r(OH) = 0.9572 Å, HOH = 104.52°, H's in the yz plane.
/// `parse_xyz` takes Å and converts to Bohr internally (the same Å->Bohr
/// constant PySCF uses), so this lands on the exact same Bohr coordinates the
/// reference JSON records.
fn water() -> Molecule {
    let r = 0.9572;
    let half = (104.52_f64).to_radians() / 2.0;
    let xyz = format!(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 {:.10} {:.10}\nH 0.0 {:.10} {:.10}\n",
        r * half.sin(), r * half.cos(),
        -r * half.sin(), r * half.cos(),
    );
    Molecule::parse_xyz(&xyz, 0, 1).unwrap()
}

fn plus_charge_field() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: -6.0 }],
        field: None,
    smeared_charges: Vec::new(),
    }
}

/// RHF shift measured independently in `qmmm_vs_pyscf.rs`/PySCF for this
/// exact geometry + charge (see the module doc comment).
const RHF_SHIFT: f64 = -0.018357900377196756;

fn tight_oo_config() -> OoRiMp2Config {
    OoRiMp2Config {
        grad_conv: 1e-8,
        energy_conv: 1e-11,
        max_iter: 200,
        ..Default::default()
    }
}

/// Regression test for the hcore bug: OO-MP2 energy in the field minus vacuum
/// OO-MP2 energy must land within 2e-3 Ha of the RHF shift. Before the fix
/// this difference was exactly 0.0 (ext silently dropped).
#[test]
fn oo_mp2_energy_shift_tracks_rhf_shift() {
    let mol = water();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ext = plus_charge_field();
    let oo_config = tight_oo_config();

    // Vacuum.
    let rhf_vac = solve_rhf(
        &ParallelContext::default(), &mol, &obs, op, &bounds,
        &RhfConfig { energy_conv: 1e-11, density_conv: 1e-10, ..Default::default() },
    ).unwrap();
    assert!(rhf_vac.converged);
    let oo_vac = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf_vac, &oo_config, None).unwrap();
    assert!(oo_vac.converged, "vacuum OO-MP2 must converge");

    // In the field.
    let field_cfg = RhfConfig {
        external_potential: Some(ext.clone()),
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    let rhf_field = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &field_cfg).unwrap();
    assert!(rhf_field.converged);
    let oo_field = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf_field, &oo_config, Some(&ext)).unwrap();
    assert!(oo_field.converged, "OO-MP2 in the field must converge");

    let oo_shift = oo_field.total_energy - oo_vac.total_energy;
    let rhf_shift_measured = rhf_field.energy - rhf_vac.energy;

    eprintln!("=== OO-MP2 vs RHF embedding shift (water/STO-3G, +1 charge at (0,0,-6) Bohr) ===");
    eprintln!("  RHF shift (measured here):  {rhf_shift_measured:.10} Ha");
    eprintln!("  RHF shift (PySCF ref):      {RHF_SHIFT:.10} Ha");
    eprintln!("  OO-MP2 shift:               {oo_shift:.10} Ha");
    eprintln!("  |OO-MP2 shift - RHF shift (PySCF ref)| = {:.3e}", (oo_shift - RHF_SHIFT).abs());

    // The bug: pre-fix, oo_shift was exactly 0.0 (ext silently dropped inside
    // oo_ri_mp2's internal hcore rebuild) regardless of how large RHF_SHIFT is.
    assert!(
        oo_shift.abs() > 1e-4,
        "OO-MP2 shift is suspiciously close to zero ({oo_shift:.3e}) -- looks like the external \
         potential was dropped (the pre-fix bug), not a small physical effect"
    );
    assert!(
        (oo_shift - RHF_SHIFT).abs() < 2e-3,
        "OO-MP2 shift {oo_shift:.10} should be within 2e-3 Ha of the independently measured RHF \
         shift {RHF_SHIFT:.10} (diff {:.3e})",
        (oo_shift - RHF_SHIFT).abs()
    );
}

/// Exactness anchor: `ext = None` reproduces the pre-change vacuum OO-MP2
/// energy for this geometry (no external-potential machinery exercised at
/// all, so this must be identical to a build without the `ext` parameter).
#[test]
fn oo_mp2_ext_none_matches_vacuum() {
    let mol = water();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let oo_config = tight_oo_config();

    let rhf = solve_rhf(
        &ParallelContext::default(), &mol, &obs, op, &bounds,
        &RhfConfig { energy_conv: 1e-11, density_conv: 1e-10, ..Default::default() },
    ).unwrap();
    assert!(rhf.converged);

    let oo_none = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_config, None).unwrap();
    let empty = ExternalPotential::default();
    let oo_empty = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_config, Some(&empty)).unwrap();

    assert!(oo_none.converged && oo_empty.converged);
    assert_eq!(
        oo_none.total_energy.to_bits(),
        oo_empty.total_energy.to_bits(),
        "None vs Some(default) total_energy not bit-identical: {:.17e} vs {:.17e}",
        oo_none.total_energy, oo_empty.total_energy,
    );
}

/// OO-MP2 analytic gradient in the field vs central FD of the TRUE
/// re-converged OO-MP2 total energy (SCF + orbital optimization re-run at
/// each displaced geometry, in the field).
#[test]
fn oo_mp2_gradient_matches_fd_in_field() {
    let mol = water();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let ext = plus_charge_field();
    let oo_config = tight_oo_config();

    let oo_total_energy_in_field = |m: &Molecule| -> f64 {
        let obs = PreparedBasis::new(m, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(m, &aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig {
            external_potential: Some(ext.clone()),
            energy_conv: 1e-11,
            density_conv: 1e-10,
            ..Default::default()
        };
        let rhf = solve_rhf(&ParallelContext::default(), m, &obs, op, &bounds, &cfg).unwrap();
        assert!(rhf.converged);
        let oo = oo_ri_mp2(m, &obs, &dfbs, op, &bounds, &rhf, &oo_config, Some(&ext)).unwrap();
        assert!(oo.converged, "OO-MP2 must converge at every displaced geometry");
        oo.total_energy
    };

    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        external_potential: Some(ext.clone()),
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged);
    let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_config, Some(&ext)).unwrap();
    assert!(oo.converged);

    let analytic = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, Some(&ext)).unwrap();

    let h = 1e-4;
    let mut fd = ndarray::Array2::<f64>::zeros((3, 3));
    for atom in 0..3 {
        for c in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match c {
                0 => { mol_p.atoms[atom].x += h; mol_m.atoms[atom].x -= h; }
                1 => { mol_p.atoms[atom].y += h; mol_m.atoms[atom].y -= h; }
                _ => { mol_p.atoms[atom].zpos += h; mol_m.atoms[atom].zpos -= h; }
            }
            let e_p = oo_total_energy_in_field(&mol_p);
            let e_m = oo_total_energy_in_field(&mol_m);
            fd[(atom, c)] = (e_p - e_m) / (2.0 * h);
        }
    }

    eprintln!("=== OO-MP2 gradient in +1 charge field (water/STO-3G) ===");
    let mut max_diff = 0.0f64;
    for atom in 0..3 {
        for c in 0..3 {
            let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
            max_diff = max_diff.max(diff);
            eprintln!(
                "  atom={atom} coord={c}: analytic={:+.8} fd={:+.8} diff={:.2e}",
                analytic[(atom, c)], fd[(atom, c)], diff
            );
        }
    }
    eprintln!("  max diff = {max_diff:.2e}");
    // Measured 9.01e-4 (water/STO-3G, +1 charge field). This is NOT a defect
    // introduced by threading `ext` through OO-MP2's gradient: it matches the
    // magnitude `oo_rimp2_gradient.rs`'s own VACUUM analytic-vs-FD water/
    // STO-3G test already carries (measured 8.71e-4 there, asserted < 1.5e-3)
    // — the documented, investigated-but-not-fully-closed OO-MP2 gradient
    // approximation (`compute_orbital_gradient`'s d(eps_p)/dkappa closed form
    // is exact only at kappa=0; see that module's doc comment). Plain
    // RI-MP2's z-vector gradient (rimp2_gradient_external.rs) is tight to
    // ~1e-7 in the SAME field on the SAME system, so this floor is intrinsic
    // to OO-MP2's existing gradient formula, not to the external-potential
    // plumbing added here.
    assert!(max_diff < 1.5e-3, "OO-MP2 analytic vs FD gradient in field max diff = {max_diff:.2e} (expected < 1.5e-3, matching the existing vacuum water/STO-3G OO-MP2 gradient bar)");
}
