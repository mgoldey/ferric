//! Lane B5: wiring Thole-damped polarizable-embedding force terms
//! (`crate::polarizable::site_gradient` / `charge_gradient_contribution`)
//! into the shared `qmmm.rs` force/gradient pipeline
//! (`mm_forces`/`full_gradient`), which lane B left with a documented gap —
//! see `run_qmmm`'s doc comment (`crates/ferric-python/src/lib.rs`) before
//! this lane: "mm_forces()/full_gradient() do NOT yet include any force from
//! a polarizable site on ITSELF".
//!
//! # Test order (mirrors qmmm.rs's own convention)
//!
//! 1. **Exactness anchor**: no polarizable atoms -> `full_gradient_with_polarizable`
//!    is bit-identical to plain `full_gradient`.
//! 2. **The acid test**: one MM atom carrying BOTH a permanent charge q AND
//!    alpha > 0 (the realistic force-field case — every real atom has both)
//!    near a QM water. The TOTAL full-gradient row for that atom (point-charge
//!    force from `mm_forces` PLUS the two polarizable terms) must match a
//!    central FD of the fully-reconverged total SCF energy, with the
//!    QmmmSystem partition rebuilt from displaced coordinates at every step.
//! 3. Two polarizable atoms (mutual induction).

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rhf_gradient_with_polarizable;
use ferric_scf::polarizable::PolarizableSites;
use ferric_scf::qmmm::{
    full_gradient, full_gradient_with_polarizable, mm_forces, polarizable_site_gradient, QmSelection,
    QmmmAtom, QmmmSystem,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;

fn setup(mol: &Molecule) -> (PreparedBasis, Operator, SchwarzBounds) {
    let bs = ferric_core::basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    (prep, op, bounds)
}

/// Water QM atoms (Bohr), standard experimental geometry, same convention as
/// `qmmm.rs::water_atoms`.
fn water_atoms(o_charge: f64, h_charge: f64) -> Vec<QmmmAtom> {
    let r = 0.9572 * ANG2BOHR;
    let half = 104.52_f64.to_radians() / 2.0;
    vec![
        QmmmAtom::new("O", 8, 0.0, 0.0, 0.0, o_charge),
        QmmmAtom::new("H", 1, 0.0, r * half.sin(), r * half.cos(), h_charge),
        QmmmAtom::new("H", 1, 0.0, -r * half.sin(), r * half.cos(), h_charge),
    ]
}

/// Solve the polarizable-embedded SCF and return everything a caller needs
/// to exercise the full B5 pipeline (energy, QM gradient, dipoles, density).
struct PolarizableRun {
    energy: f64,
    qm_gradient: Array2<f64>,
    density_total: Array2<f64>,
    induced_dipoles: Option<Array2<f64>>,
}

fn run_polarizable(sys: &QmmmSystem) -> PolarizableRun {
    let mol = sys.to_qm_molecule();
    let (prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let pol_sites = sys.to_polarizable_sites();
    let polarizable =
        if pol_sites.is_empty() { None } else { Some(PolarizableSites { sites: pol_sites, ..Default::default() }) };
    let ext = sys.to_external_potential();
    let cfg = RhfConfig {
        external_potential: ext.clone(),
        polarizable,
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged, "polarizable-embedded SCF failed to converge");
    let qm_gradient = rhf_gradient_with_polarizable(
        &mol, &prep, op, &bounds, &r, ext.as_ref(), cfg.polarizable.as_ref(), r.induced_dipoles.as_ref(),
    )
    .unwrap();
    PolarizableRun {
        energy: r.energy,
        qm_gradient,
        density_total: r.density_total().clone(),
        induced_dipoles: r.induced_dipoles.clone(),
    }
}

/// Energy only (for FD), rebuilding nothing but the SCF at a fixed partition.
fn polarizable_energy(sys: &QmmmSystem) -> f64 {
    run_polarizable(sys).energy
}

// ---------------------------------------------------------------------------
// 1. EXACTNESS ANCHOR
// ---------------------------------------------------------------------------

#[test]
fn full_gradient_with_polarizable_matches_plain_full_gradient_when_not_polarizable() {
    let atoms = {
        let mut a = water_atoms(0.0, 0.0);
        a.push(QmmmAtom::new("Cl", 17, 6.0, 0.0, 0.0, -0.3));
        a
    };
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    assert!(sys.to_polarizable_sites().is_empty(), "test setup must have zero polarizable sites");

    let mol = sys.to_qm_molecule();
    let (prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let ext = sys.to_external_potential();
    let cfg = RhfConfig { external_potential: ext.clone(), density_conv: 1e-10, ..Default::default() };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged);
    let qm_grad = ferric_scf::gradient::rhf_gradient(&mol, &prep, op, &bounds, &r, ext.as_ref()).unwrap();
    let forces = mm_forces(&sys, &mol, &prep, r.density_total()).unwrap();

    let plain = full_gradient(&sys, &qm_grad, &forces).unwrap();

    let empty_sites = PolarizableSites { sites: vec![], ..Default::default() };
    let site_rows = polarizable_site_gradient(
        &sys, &mol, &prep, r.density_total(), ext.as_ref(), &empty_sites, &Array2::zeros((0, 3)),
    )
    .unwrap();
    assert!(site_rows.is_empty(), "no polarizable sites must give an empty row list");

    let with_pol = full_gradient_with_polarizable(
        &sys, &qm_grad, &forces, ext.as_ref(), &empty_sites, None, &mol, &prep, r.density_total(),
    )
    .unwrap();

    assert_eq!(plain.dim(), with_pol.dim());
    for i in 0..plain.nrows() {
        for k in 0..3 {
            assert_eq!(
                plain[(i, k)],
                with_pol[(i, k)],
                "row {i} axis {k}: plain full_gradient and full_gradient_with_polarizable \
                 (empty sites) must be BIT-IDENTICAL"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. THE ACID TEST: one MM atom carries BOTH a permanent charge AND alpha.
// ---------------------------------------------------------------------------

fn one_colocated_charge_alpha_system() -> QmmmSystem {
    let mut atoms = water_atoms(-0.3, 0.15);
    // One MM atom near the water, off-axis, carrying BOTH a nonzero permanent
    // charge and a nonzero polarisability -- the realistic force-field atom.
    atoms.push(QmmmAtom::new("Cl", 17, 5.5, -1.5, 2.0, -0.4).with_alpha(3.0));
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    assert_eq!(sys.to_polarizable_sites().len(), 1);
    sys
}

#[test]
fn colocated_charge_and_polarizable_site_row_matches_finite_difference() {
    let sys = one_colocated_charge_alpha_system();
    let run = run_polarizable(&sys);
    let mol = sys.to_qm_molecule();
    let (prep, _op, _bounds) = setup(&mol);
    let ext = sys.to_external_potential();
    let pol_sites = sys.to_polarizable_sites();
    let sites = PolarizableSites { sites: pol_sites, ..Default::default() };
    let dipoles = run.induced_dipoles.as_ref().expect("polarizable run must produce induced dipoles");

    let forces = mm_forces(&sys, &mol, &prep, &run.density_total).unwrap();
    let full = full_gradient_with_polarizable(
        &sys, &run.qm_gradient, &forces, ext.as_ref(), &sites, Some(dipoles), &mol, &prep, &run.density_total,
    )
    .unwrap();

    // Sanity: without the polarizable terms, the Cl row would be WRONG (this
    // is the gap B5 closes) -- mutation-checked properly below, but this
    // assertion documents the two paths differ, which is a precondition for
    // the FD match below to be a meaningful test at all.
    let plain = full_gradient(&sys, &run.qm_gradient, &forces).unwrap();
    let cl_idx = 3;
    let differs = (0..3).any(|k| (full[(cl_idx, k)] - plain[(cl_idx, k)]).abs() > 1e-8);
    assert!(differs, "polarizable terms must be nonzero on the colocated charge+alpha atom's row");

    let h = 1e-3;
    let atoms0 = {
        let mut atoms = water_atoms(-0.3, 0.15);
        atoms.push(QmmmAtom::new("Cl", 17, 5.5, -1.5, 2.0, -0.4).with_alpha(3.0));
        atoms
    };
    let energy_at = |atoms: &[QmmmAtom]| -> f64 {
        let sys = QmmmSystem::new(atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
        polarizable_energy(&sys)
    };

    let mut max_err = 0.0_f64;
    for &(a, k, label) in &[
        (3usize, 0usize, "Cl (charge+alpha), x"),
        (3, 1, "Cl (charge+alpha), y"),
        (3, 2, "Cl (charge+alpha), z"),
        (0, 2, "QM O, z"),
        (1, 1, "QM H1, y"),
    ] {
        let mut plus = atoms0.clone();
        let mut minus = atoms0.clone();
        match k {
            0 => { plus[a].x += h; minus[a].x -= h; }
            1 => { plus[a].y += h; minus[a].y -= h; }
            _ => { plus[a].z_pos += h; minus[a].z_pos -= h; }
        }
        let fd = (energy_at(&plus) - energy_at(&minus)) / (2.0 * h);
        let an = full[(a, k)];
        let err = (an - fd).abs();
        max_err = max_err.max(err);
        eprintln!("[qmmm-B5] {label}: analytic {an:+.8e}, FD {fd:+.8e}, |Δ| {err:.2e}");
        assert!(err < 2e-6, "{label}: analytic {an:+.8e} vs FD {fd:+.8e} (|Δ| {err:.2e})");
    }
    eprintln!("[qmmm-B5] max|analytic - FD| (colocated charge+alpha acid test) = {max_err:.3e}");
}

// ---------------------------------------------------------------------------
// 3. TWO polarizable atoms (mutual induction).
// ---------------------------------------------------------------------------

fn two_colocated_charge_alpha_system() -> QmmmSystem {
    let mut atoms = water_atoms(-0.3, 0.15);
    atoms.push(QmmmAtom::new("Cl", 17, 5.5, -1.5, 2.0, -0.4).with_alpha(3.0));
    atoms.push(QmmmAtom::new("Na", 11, -4.0, 3.0, -2.5, 0.3).with_alpha(1.0));
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    assert_eq!(sys.to_polarizable_sites().len(), 2);
    sys
}

#[test]
fn two_colocated_charge_and_polarizable_sites_match_finite_difference() {
    let sys = two_colocated_charge_alpha_system();
    let run = run_polarizable(&sys);
    let mol = sys.to_qm_molecule();
    let (prep, _op, _bounds) = setup(&mol);
    let ext = sys.to_external_potential();
    let pol_sites = sys.to_polarizable_sites();
    let sites = PolarizableSites { sites: pol_sites, ..Default::default() };
    let dipoles = run.induced_dipoles.as_ref().expect("polarizable run must produce induced dipoles");

    let forces = mm_forces(&sys, &mol, &prep, &run.density_total).unwrap();
    let full = full_gradient_with_polarizable(
        &sys, &run.qm_gradient, &forces, ext.as_ref(), &sites, Some(dipoles), &mol, &prep, &run.density_total,
    )
    .unwrap();

    let h = 1e-3;
    let atoms0 = {
        let mut atoms = water_atoms(-0.3, 0.15);
        atoms.push(QmmmAtom::new("Cl", 17, 5.5, -1.5, 2.0, -0.4).with_alpha(3.0));
        atoms.push(QmmmAtom::new("Na", 11, -4.0, 3.0, -2.5, 0.3).with_alpha(1.0));
        atoms
    };
    let energy_at = |atoms: &[QmmmAtom]| -> f64 {
        let sys = QmmmSystem::new(atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
        polarizable_energy(&sys)
    };

    let mut max_err = 0.0_f64;
    for &(a, k, label) in &[
        (3usize, 0usize, "Cl (site 1), x"),
        (3, 2, "Cl (site 1), z"),
        (4, 1, "Na (site 2), y"),
        (4, 2, "Na (site 2), z"),
    ] {
        let mut plus = atoms0.clone();
        let mut minus = atoms0.clone();
        match k {
            0 => { plus[a].x += h; minus[a].x -= h; }
            1 => { plus[a].y += h; minus[a].y -= h; }
            _ => { plus[a].z_pos += h; minus[a].z_pos -= h; }
        }
        let fd = (energy_at(&plus) - energy_at(&minus)) / (2.0 * h);
        let an = full[(a, k)];
        let err = (an - fd).abs();
        max_err = max_err.max(err);
        eprintln!("[qmmm-B5] mutual induction {label}: analytic {an:+.8e}, FD {fd:+.8e}, |Δ| {err:.2e}");
        assert!(err < 2e-6, "{label}: analytic {an:+.8e} vs FD {fd:+.8e} (|Δ| {err:.2e})");
    }
    eprintln!("[qmmm-B5] max|analytic - FD| (two-site mutual induction) = {max_err:.3e}");
}

/// Translational invariance of the full structure (no link atoms, no
/// boundary charges here) including the polarizable terms.
#[test]
fn full_gradient_with_polarizable_column_sums_vanish() {
    let sys = two_colocated_charge_alpha_system();
    let run = run_polarizable(&sys);
    let mol = sys.to_qm_molecule();
    let (prep, _op, _bounds) = setup(&mol);
    let ext = sys.to_external_potential();
    let pol_sites = sys.to_polarizable_sites();
    let sites = PolarizableSites { sites: pol_sites, ..Default::default() };
    let dipoles = run.induced_dipoles.as_ref().unwrap();

    let forces = mm_forces(&sys, &mol, &prep, &run.density_total).unwrap();
    let full = full_gradient_with_polarizable(
        &sys, &run.qm_gradient, &forces, ext.as_ref(), &sites, Some(dipoles), &mol, &prep, &run.density_total,
    )
    .unwrap();

    for k in 0..3 {
        let sum: f64 = (0..full.nrows()).map(|i| full[(i, k)]).sum();
        assert!(sum.abs() < 1e-6, "column {k} sum = {sum:.3e}, expected ~0 (translational invariance)");
    }
}

// ---------------------------------------------------------------------------
// UNTESTED (documented gap, not silently skipped): boundary charges (RC/RCD)
// combined with a polarizable site. `polarizable_charge_gradient_rows`
// implements the (m1,m2) half/half split for that case (mirroring
// `full_gradient`'s own convention), but no FD cross-check exercises a
// polarizable atom simultaneously acting as an RC/RCD midpoint host — the
// two features are validated independently (link+RCD boundary in
// `qmmm.rs::full_gradient_matches_finite_difference_across_the_boundary`,
// polarizable colocated atoms above) but not in combination.
