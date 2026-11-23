//! Same-molecule exclusion in Thole polarizable embedding
//! (`QmmmSystem::polarizable_embedding` with molecule ids).
//!
//! A TIP3P water's charges already contain its own polarization, so a
//! polarizable site must not be polarized by the charges (or induced dipoles)
//! of its own molecule. Without exclusions an oxygen site 1.81 Bohr from two
//! +0.417 e hydrogens carries E_pol ~ -1/2 alpha E^2 ~ -0.12 Ha at alpha =
//! 1.44 A^3, with no QM region involved.
//!
//! Physics hypothesis: with exclusions, one isolated MM water has E_pol = 0
//! exactly (nothing else polarizes it). Artifact hypothesis: an exclusion
//! that misses the non-site hydrogen charges leaves the -0.12 Ha; one that
//! misses the intramolecular site pairs leaves nonzero mutual induction when
//! the hydrogens are sites too. The two variants below separate the two.
//!
//! Order: exactness anchor (every atom its own molecule = no exclusions, bit
//! for bit), the isolated-water anchor, then the gradient against a central
//! difference of the reconverged energy with the partition (and its
//! exclusion positions) rebuilt at every displaced geometry.

use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rhf_gradient_with_polarizable;
use ferric_scf::polarizable::{mm_only_polarization_energy, PolarizableSites, DEFAULT_THOLE_A};
use ferric_scf::qmmm::{
    full_gradient_with_polarizable, mm_forces, molecule_ids_from_bonds, QmSelection, QmmmAtom,
    QmmmSystem,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_903;
const A3_TO_BOHR3: f64 = ANG2BOHR * ANG2BOHR * ANG2BOHR;

/// A TIP3P-geometry water (Angstrom in, Bohr out) with its O at `o`, charges
/// O -0.834 / H +0.417 and polarizabilities `(alpha_o, alpha_h)` in A^3.
fn tip3p(o: [f64; 3], alpha_o: f64, alpha_h: f64, polar: bool) -> Vec<QmmmAtom> {
    let (qo, qh) = if polar { (-0.834, 0.417) } else { (0.0, 0.0) };
    let b = |p: [f64; 3]| [p[0] * ANG2BOHR, p[1] * ANG2BOHR, p[2] * ANG2BOHR];
    let o_b = b(o);
    let h1 = b([o[0] + 0.757, o[1] + 0.586, o[2]]);
    let h2 = b([o[0] - 0.757, o[1] + 0.586, o[2] + 0.1]);
    vec![
        QmmmAtom::new("O", 8, o_b[0], o_b[1], o_b[2], qo).with_alpha(alpha_o * A3_TO_BOHR3),
        QmmmAtom::new("H", 1, h1[0], h1[1], h1[2], qh).with_alpha(alpha_h * A3_TO_BOHR3),
        QmmmAtom::new("H", 1, h2[0], h2[1], h2[2], qh).with_alpha(alpha_h * A3_TO_BOHR3),
    ]
}

/// One QM water (no MM charge, QM atoms carry none) and the given MM atoms,
/// molecule ids: QM water 0, then one id per MM water of three atoms.
fn system(mm: &[QmmmAtom]) -> (QmmmSystem, Vec<usize>) {
    let mut atoms = tip3p([0.0, 0.0, 0.0], 0.0, 0.0, false);
    atoms.extend_from_slice(mm);
    let ids: Vec<usize> = (0..atoms.len()).map(|i| i / 3).collect();
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1)
        .unwrap()
        .with_residue_ids(ids.clone())
        .unwrap();
    (sys, ids)
}

fn mm_only(sys: &QmmmSystem, ids: Option<&[usize]>) -> f64 {
    let pol = sys
        .polarizable_embedding(Some(DEFAULT_THOLE_A), ids)
        .unwrap()
        .expect("system has polarizable sites");
    mm_only_polarization_energy(sys.to_external_potential().as_ref(), &pol).unwrap()
}

struct Run {
    energy: f64,
    full: Array2<f64>,
}

fn run(sys: &QmmmSystem, pol: Option<PolarizableSites>, with_gradient: bool) -> Run {
    let mol = sys.to_qm_molecule();
    let bs = ferric_core::basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ext = sys.to_external_potential();
    let mut cfg = RhfConfig::default()
        .with_external_potential(ext.clone())
        .with_energy_conv(1e-11)
        .with_density_conv(1e-10)
        .with_max_iter(300);
    cfg.polarizable = pol;
    let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged, "polarizable SCF did not converge");
    if !with_gradient {
        return Run {
            energy: r.energy,
            full: Array2::zeros((0, 3)),
        };
    }
    let pol = cfg.polarizable.as_ref().unwrap();
    let qm = rhf_gradient_with_polarizable(
        &mol,
        &prep,
        op,
        &bounds,
        &r,
        ext.as_ref(),
        Some(pol),
        r.induced_dipoles.as_ref(),
    )
    .unwrap();
    let forces = mm_forces(sys, &mol, &prep, r.density_total()).unwrap();
    let full = full_gradient_with_polarizable(
        sys,
        &qm,
        &forces,
        ext.as_ref(),
        pol,
        r.induced_dipoles.as_ref(),
        &mol,
        &prep,
        r.density_total(),
    )
    .unwrap();
    Run {
        energy: r.energy,
        full,
    }
}

/// Exactness anchor: when every atom is its own molecule, same-molecule
/// exclusion excludes nothing beyond a site's own charge, so the SCF is
/// bit-identical to the no-exclusion embedding.
#[test]
fn every_atom_its_own_molecule_is_bit_identical_to_no_exclusions() {
    let mm: Vec<QmmmAtom> = tip3p([0.0, -2.9, 0.0], 1.44, 0.3, true);
    let (sys, _) = system(&mm);
    let singletons: Vec<usize> = (0..sys.atoms.len()).collect();
    let none = sys
        .polarizable_embedding(Some(DEFAULT_THOLE_A), None)
        .unwrap();
    let solo = sys
        .polarizable_embedding(Some(DEFAULT_THOLE_A), Some(&singletons))
        .unwrap();
    assert!(solo.as_ref().unwrap().exclusions.is_empty());
    let e_none = run(&sys, none, false).energy;
    let e_solo = run(&sys, solo, false).energy;
    assert_eq!(
        e_none.to_bits(),
        e_solo.to_bits(),
        "{e_none} vs {e_solo}: singleton molecules must exclude nothing"
    );
}

/// The anchor of the fix: one isolated MM water, no other charge, no QM
/// field (mm_only_polarization_energy drops E^QM). With same-molecule
/// exclusion nothing polarizes it, E_pol = 0 exactly; without, the oxygen
/// is polarized by its own hydrogens.
#[test]
fn an_isolated_mm_water_does_not_polarize_itself() {
    for (label, alpha_h) in [("oxygen-only sites", 0.0), ("every atom a site", 0.496)] {
        let mm = tip3p([0.0, -2.9, 0.0], 1.44, alpha_h, true);
        let (sys, ids) = system(&mm);
        let e_excl = mm_only(&sys, Some(&ids));
        let e_none = mm_only(&sys, None);
        eprintln!("{label}: MM-only E_pol excluded {e_excl:.3e}, not excluded {e_none:.6e}");
        assert_eq!(e_excl, 0.0, "{label}: excluded MM-only E_pol must vanish");
        assert!(
            e_none < -1e-2,
            "{label}: control: without exclusion the water polarizes itself ({e_none})"
        );
    }
}

/// Molecule ids from bonds: connected components, numbered by lowest atom.
#[test]
fn molecule_ids_from_bonds_are_connected_components() {
    let ids = molecule_ids_from_bonds(7, &[(5, 3), (0, 1), (1, 2), (3, 4)]).unwrap();
    assert_eq!(ids, vec![0, 0, 0, 1, 1, 1, 2]);
    assert!(molecule_ids_from_bonds(2, &[(0, 2)]).is_err());
}

/// The gradient of the excluded functional: every row of two MM waters (one
/// oxygen-only, one with every atom a site, so both exclusion lists and
/// intermolecular mutual induction are live) against a central difference
/// of the reconverged energy, partition and exclusions rebuilt each time.
#[test]
fn excluded_embedding_gradient_matches_finite_difference() {
    let mut mm = tip3p([0.0, -2.9, 0.0], 1.44, 0.0, true);
    mm.extend(tip3p([2.9, 1.2, 0.5], 0.837, 0.496, true));
    let (sys, ids) = system(&mm);
    let pol = |s: &QmmmSystem| {
        s.polarizable_embedding(Some(DEFAULT_THOLE_A), Some(&ids))
            .unwrap()
    };
    let r = run(&sys, pol(&sys), true);

    let natoms = sys.atoms.len();
    let coords0 = Array2::from_shape_fn((natoms, 3), |(i, k)| {
        let a = &sys.atoms[i];
        [a.x, a.y, a.z_pos][k]
    });
    let h = 1e-3;
    let mut worst = 0.0_f64;
    for a in 0..natoms {
        for k in 0..3 {
            let e = |d: f64| {
                let mut c = coords0.clone();
                c[(a, k)] += d;
                let s = sys.with_coordinates(&c).unwrap();
                run(&s, pol(&s), false).energy
            };
            let fd = (e(h) - e(-h)) / (2.0 * h);
            let err = (r.full[(a, k)] - fd).abs();
            worst = worst.max(err);
            assert!(
                err < 2e-6,
                "atom {a} axis {k}: analytic {:+.8e} vs FD {fd:+.8e} (|d| {err:.1e})",
                r.full[(a, k)]
            );
        }
    }
    eprintln!("max |analytic - FD| = {worst:.2e}");
}
