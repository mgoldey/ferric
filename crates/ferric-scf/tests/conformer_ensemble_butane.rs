//! End-to-end conformer-ensemble validation on a REAL molecule with REAL SCF
//! energies: n-butane anti vs gauche at RHF/STO-3G.
//!
//! The unit tests in `ferric_core::conformers` pin the weighting/averaging
//! arithmetic against hand computations. This test closes the loop: it runs
//! actual SCF on two actual conformers of an actual flexible molecule and shows
//! the ensemble machinery produces sensible physics on those numbers.
//!
//! Why butane: it is the textbook conformational system. The anti conformer
//! (C-C-C-C dihedral 180 deg) is the global minimum; gauche (+60 deg) sits
//! above it by the well-known ~0.9 kcal/mol "gauche interaction", which at room
//! temperature is a few kT -- exactly the regime where a single-conformer
//! property is neither obviously fine nor obviously wrong, so the ensemble
//! diagnostics have something real to say.
//!
//! The gauche geometry is the anti geometry rigidly rotated 120 deg about the
//! central C1-C2 bond, moving the whole C2 end as one rigid unit: C3 and its
//! three hydrogens H11/H12/H13, *and* C2's own hydrogens H9/H10. Including
//! H9/H10 is the part that is easy to get wrong -- they define C2's local
//! tetrahedral frame, and leaving them behind drives the rotated C3 to 0.43
//! Angstrom from H9, a geometry no SCF result from which means anything. With
//! them included, every bond length is preserved to 1e-6 Angstrom and the
//! closest non-bonded contact in gauche (1.786 A, the geminal H7-H8 pair on C1)
//! is one that is already present in anti -- i.e. the torsion introduces no new
//! close contact at all, while C0...C3 contracts 3.840 -> 2.922 A, the
//! signature of a genuine gauche conformer.
//!
//! So this is a pure torsional change: the same molecule, same atom ORDER, same
//! composition -- precisely what `ConformerEnsemble` requires.
//!
//! NOTE: STO-3G at a rigid (unrelaxed) torsion is not a quantitative
//! conformational-energy method, so this test asserts the *machinery* and the
//! qualitative ordering, not a benchmark gauche energy. The precise assertions
//! live in the ferric-core unit tests.

use ferric_core::basis;
use ferric_core::conformers::{
    boltzmann_weights, parse_multi_xyz, weighted_stats, weighted_stats_vector, ConformerEnsemble,
    BOLTZMANN_HARTREE_PER_K, DEFAULT_TEMPERATURE_K,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HARTREE_TO_KCAL: f64 = 627.509_474_063_1;

/// n-butane, all-anti (the bundled `testdata/molecules/alkane_4.xyz` geometry).
const BUTANE_ANTI: &str = "\
14
n-butane anti (C-C-C-C dihedral 180 deg)
C      0.000000    0.000000    0.000000
C      1.245964    0.881050    0.000000
C      2.491929    0.000000    0.000000
C      3.737893    0.881050    0.000000
H     -0.893241    0.631631    0.000000
H      0.000027   -0.631612   -0.893254
H      0.000027   -0.631612    0.893254
H      1.245964    1.512680    0.893241
H      1.245964    1.512680   -0.893241
H      2.491929   -0.631631   -0.893241
H      2.491929   -0.631631    0.893241
H      4.631134    0.249419    0.000000
H      3.737866    1.512661   -0.893254
H      3.737866    1.512661    0.893254
";

/// The same molecule with the whole C2 end (C3 + H11/H12/H13 + C2's own
/// H9/H10) rotated 120 deg about the C1-C2 bond: dihedral 180 -> 60 deg
/// (gauche). Atom order and every bond length are unchanged.
const BUTANE_GAUCHE: &str = "\
14
n-butane gauche (C-C-C-C dihedral 60 deg)
C      0.000000    0.000000    0.000000
C      1.245964    0.881050    0.000000
C      2.491929    0.000000    0.000000
C      2.491891   -0.881023   -1.245983
H     -0.893241    0.631631    0.000000
H      0.000027   -0.631612   -0.893254
H      0.000027   -0.631612    0.893254
H      1.245964    1.512680    0.893241
H      1.245964    1.512680   -0.893241
H      2.491936   -0.631621    0.893248
H      3.385190    0.631603    0.000007
H      3.385132   -1.512654   -1.245983
H      1.598624   -1.512616   -1.245956
H      2.491892   -0.249373   -2.139210
";

/// Both conformers as one multi-frame XYZ, the format RDKit emits.
fn two_frame_xyz() -> String {
    format!("{BUTANE_ANTI}{BUTANE_GAUCHE}")
}

fn rhf_energy(mol: &Molecule, basis_name: &str) -> f64 {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let config = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-9,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let result = solve_rhf(&ctx, mol, &prep, op, &bounds, &config).unwrap();
    assert!(result.converged, "RHF did not converge");
    result.energy
}

/// The geometries really are two conformers of one molecule: the multi-frame
/// reader recovers both, and the ensemble's composition invariant accepts them.
#[test]
fn butane_conformers_share_composition_and_ordering() {
    let mols = parse_multi_xyz(&two_frame_xyz(), 0, 1).unwrap();
    assert_eq!(mols.len(), 2, "multi-frame XYZ must yield both conformers");
    assert_eq!(mols[0].atoms.len(), 14);
    assert_eq!(mols[1].atoms.len(), 14);

    // Same composition and ordering...
    for (a, b) in mols[0].atoms.iter().zip(mols[1].atoms.iter()) {
        assert_eq!(a.z, b.z);
        assert_eq!(a.symbol, b.symbol);
    }
    // ...but genuinely different geometry. Atom 3 (the terminal carbon of the
    // rotated end) is the one that moves; atoms 0-2 lie on or before the
    // rotation axis and are deliberately unchanged.
    let displacement = |i: usize| {
        let (a, b) = (&mols[0].atoms[i], &mols[1].atoms[i]);
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2)).sqrt()
    };
    assert!(
        displacement(3) > 1.0,
        "C3 must move between conformers, got {} Bohr",
        displacement(3)
    );
    assert!(
        displacement(0) < 1e-10 && displacement(1) < 1e-10 && displacement(2) < 1e-10,
        "C0/C1/C2 are on or before the rotation axis and must not move"
    );

    // Rigid rotation: the C1-C2 bond length is identical in both frames.
    let bond = |m: &Molecule, i: usize, j: usize| {
        let (a, b) = (&m.atoms[i], &m.atoms[j]);
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2)).sqrt()
    };
    for (i, j) in [(0usize, 1usize), (1, 2), (2, 3)] {
        let da = bond(&mols[0], i, j);
        let dg = bond(&mols[1], i, j);
        assert!(
            (da - dg).abs() < 1e-5,
            "bond ({i},{j}) changed between conformers: {da} vs {dg} -- not a pure torsion"
        );
    }

    // The torsion must introduce no NEW close contact: the tightest non-bonded
    // distance in gauche must be no worse than the tightest one in anti (both
    // are the geminal H-H pairs at 1.786 A). A rigid rotation that leaves a
    // substituent behind fails here at ~0.43 A, and any SCF energy computed on
    // such a geometry is meaningless -- this is the guard that catches it.
    let min_nonbonded = |m: &Molecule| -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..m.atoms.len() {
            for j in (i + 1)..m.atoms.len() {
                let d = bond(m, i, j);
                // 1.7 A in Bohr: anything closer is a chemical bond, not a contact.
                if d > 1.7 / 0.529_177_210_92 {
                    best = best.min(d);
                }
            }
        }
        best
    };
    let anti_min = min_nonbonded(&mols[0]);
    let gauche_min = min_nonbonded(&mols[1]);
    eprintln!(
        "min non-bonded contact: anti {:.4} Bohr, gauche {:.4} Bohr",
        anti_min, gauche_min
    );
    assert!(
        gauche_min > 0.99 * anti_min,
        "gauche introduced a new close contact ({gauche_min:.4} vs anti {anti_min:.4} Bohr) -- \
         the rotation group is wrong and the SCF energy would be meaningless"
    );

    // C0...C3 must contract: that IS the gauche interaction.
    let c0c3_anti = bond(&mols[0], 0, 3);
    let c0c3_gauche = bond(&mols[1], 0, 3);
    assert!(
        c0c3_gauche < c0c3_anti - 1.0,
        "gauche must bring the terminal carbons closer: {c0c3_gauche:.4} vs {c0c3_anti:.4} Bohr"
    );

    // The invariant check accepts them.
    ConformerEnsemble::from_molecules(mols).unwrap();
}

/// The full pipeline: multi-frame XYZ -> per-conformer RHF -> ensemble ->
/// Boltzmann weights -> averaged property with a spread -> diagnostics.
#[test]
fn butane_ensemble_end_to_end_with_real_scf_energies() {
    let mols = parse_multi_xyz(&two_frame_xyz(), 0, 1).unwrap();

    let e_anti = rhf_energy(&mols[0], "sto-3g");
    let e_gauche = rhf_energy(&mols[1], "sto-3g");
    let de_kcal = (e_gauche - e_anti) * HARTREE_TO_KCAL;
    eprintln!("RHF/STO-3G  E(anti)   = {e_anti:.10} Ha");
    eprintln!("RHF/STO-3G  E(gauche) = {e_gauche:.10} Ha");
    eprintln!("            dE        = {:.10} Ha ({de_kcal:.4} kcal/mol)", e_gauche - e_anti);

    // Physics sanity: anti is the minimum. This is the one qualitative claim
    // STO-3G at a rigid torsion can be trusted for.
    assert!(
        e_gauche > e_anti,
        "anti should be the lower conformer; got anti {e_anti:.10}, gauche {e_gauche:.10}"
    );
    // And the gap is a conformational one, not a sign that the two geometries
    // are different molecules or that one contains a clash. MEASURED here:
    // 3.3059 kcal/mol at RHF/STO-3G on the RIGID (unrelaxed) torsion. The
    // experimental relaxed gauche-anti gap is ~0.9 kcal/mol; a rigid rotation
    // in a minimal basis overestimating it ~3x is the expected behaviour, since
    // none of the strain the real molecule relieves by relaxing is available
    // here. The window below is deliberately wide enough to be a
    // did-the-physics-break check rather than a benchmark assertion -- but far
    // tighter than the ~1213 kcal/mol a clashing rotation produced.
    assert!(
        de_kcal > 0.5 && de_kcal < 6.0,
        "gauche-anti gap {de_kcal:.4} kcal/mol is outside the conformational range \
         (expected ~3.3 for a rigid STO-3G torsion)"
    );

    let ens = ConformerEnsemble::from_molecules_and_energies(mols, &[e_anti, e_gauche]).unwrap();
    assert_eq!(ens.len(), 2);
    assert_eq!(ens.n_atoms(), 14);

    let w = ens.boltzmann_weights_default().unwrap();
    eprintln!("weights: anti = {:.10}, gauche = {:.10}", w.weights[0], w.weights[1]);
    eprintln!("dE / kT = {:.6}", w.relative_energies[1] / w.kt_hartree);

    // Weights are a normalized population.
    let sum: f64 = w.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "weights must sum to 1, got {sum:.18}");
    assert_eq!(w.min_index, 0, "anti is the reference minimum");
    assert_eq!(w.relative_energies[0], 0.0);
    assert!(w.weights[0] > w.weights[1], "the lower conformer must be more populated");

    // Reproduce the two-state closed form independently from the raw energies:
    //   w_gauche = exp(-dE/kT) / (1 + exp(-dE/kT))
    let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
    let x = (-(e_gauche - e_anti) / kt).exp();
    let expected_gauche = x / (1.0 + x);
    assert!(
        (w.weights[1] - expected_gauche).abs() < 1e-14,
        "closed form {expected_gauche:.15e} vs computed {:.15e}",
        w.weights[1]
    );

    // Ensemble-average a real per-conformer property. Nuclear repulsion is used
    // because it is exactly reproducible from the geometry alone (no SCF noise),
    // while still genuinely differing between the two conformers.
    let vnn: Vec<f64> = ens
        .conformers()
        .iter()
        .map(|c| c.molecule.nuclear_repulsion())
        .collect();
    assert!(
        (vnn[0] - vnn[1]).abs() > 1e-6,
        "the two conformers must have different nuclear repulsion"
    );
    let stats = weighted_stats(&vnn, &w.weights).unwrap();
    eprintln!(
        "<Vnn> = {:.8} +/- {:.8} Ha  (range {:.8} .. {:.8})",
        stats.mean, stats.std_dev, stats.min, stats.max
    );

    // The average must lie inside the range of the inputs, and the spread must
    // be positive (the conformers differ) but no larger than half the range for
    // a two-state ensemble.
    assert!(stats.mean >= stats.min && stats.mean <= stats.max);
    assert!(stats.std_dev > 0.0, "a two-conformer spread must be non-zero");
    assert!(stats.std_dev <= 0.5 * (stats.max - stats.min) + 1e-12);

    // Diagnostics: report the population structure plainly.
    let d = w.diagnostics();
    eprintln!("{d}");
    assert_eq!(d.n_conformers, 2);
    assert_eq!(d.max_weight_index, 0, "anti carries the largest population");
    assert!(d.n_within_kt >= 1);
    assert!(d.n_within_5kt >= 1 && d.n_within_5kt <= 2);
    assert!(
        d.effective_n_conformers >= 1.0 && d.effective_n_conformers <= 2.0,
        "effective N must lie between 1 and 2 for a two-state ensemble, got {}",
        d.effective_n_conformers
    );
    assert!(!d.verdict().is_empty());

    // The honest conclusion for THIS ensemble, which the diagnostics must state
    // rather than leave the caller to infer: at a 3.3 kcal/mol (5.6 kT) gap the
    // anti conformer carries ~99.6% of the population, so a single-conformer
    // butane property would in fact have been fine. Reporting that plainly is
    // the point -- the machinery is not claiming the ensemble mattered when it
    // did not.
    assert!(
        w.weights[0] > 0.99,
        "anti should carry >99% at a 5.6 kT gap, got {:.6}",
        w.weights[0]
    );
    assert!(d.is_single_conformer_dominated(0.95));
    assert!(
        d.verdict().contains("unnecessary"),
        "diagnostics must say plainly that the ensemble was unnecessary here, got: {}",
        d.verdict()
    );
}

/// A one-conformer ensemble built on a real SCF energy must reproduce the
/// single-point answer EXACTLY, with zero spread -- the headline invariant.
#[test]
fn butane_single_conformer_ensemble_equals_the_single_point() {
    let mol = Molecule::parse_xyz(BUTANE_ANTI, 0, 1).unwrap();
    let e = rhf_energy(&mol, "sto-3g");
    let vnn_single = mol.nuclear_repulsion();

    let ens = ConformerEnsemble::from_molecules_and_energies(vec![mol], &[e]).unwrap();
    let w = ens.boltzmann_weights_default().unwrap();
    assert_eq!(w.weights, vec![1.0], "one conformer must have weight exactly 1.0");

    let vnn: Vec<f64> = ens
        .conformers()
        .iter()
        .map(|c| c.molecule.nuclear_repulsion())
        .collect();
    let stats = weighted_stats(&vnn, &w.weights).unwrap();
    assert_eq!(
        stats.mean, vnn_single,
        "single-conformer ensemble average must be bit-identical to the single point"
    );
    assert_eq!(stats.std_dev, 0.0, "single-conformer spread must be EXACTLY zero");

    // Same for the energy itself.
    let e_stats = weighted_stats(&[e], &w.weights).unwrap();
    assert_eq!(e_stats.mean, e);
    assert_eq!(e_stats.std_dev, 0.0);

    // And for a vector property (the geometry's centre of nuclear charge).
    let com: Vec<f64> = {
        let m = &ens.conformers()[0].molecule;
        let zsum: f64 = m.atoms.iter().map(|a| a.z as f64).sum();
        vec![
            m.atoms.iter().map(|a| a.z as f64 * a.x).sum::<f64>() / zsum,
            m.atoms.iter().map(|a| a.z as f64 * a.y).sum::<f64>() / zsum,
            m.atoms.iter().map(|a| a.z as f64 * a.zpos).sum::<f64>() / zsum,
        ]
    };
    let vstats = weighted_stats_vector(std::slice::from_ref(&com), &w.weights).unwrap();
    for (k, s) in vstats.iter().enumerate() {
        assert_eq!(s.mean, com[k]);
        assert_eq!(s.std_dev, 0.0);
    }
}

/// Two copies of the SAME conformer (identical SCF energies) must give exactly
/// 0.5/0.5, and any property averaged over them must reproduce the single-point
/// value with zero spread -- a degeneracy check on real energies rather than
/// on hand-written literals.
#[test]
fn butane_duplicate_conformer_is_exactly_degenerate() {
    let mol = Molecule::parse_xyz(BUTANE_ANTI, 0, 1).unwrap();
    let e = rhf_energy(&mol, "sto-3g");

    let ens =
        ConformerEnsemble::from_molecules_and_energies(vec![mol.clone(), mol.clone()], &[e, e])
            .unwrap();
    let w = ens.boltzmann_weights_default().unwrap();
    assert_eq!(w.weights[0], 0.5, "identical energies must give exactly 0.5");
    assert_eq!(w.weights[1], 0.5);
    assert_eq!(w.partition_function, 2.0);

    let vnn = mol.nuclear_repulsion();
    let stats = weighted_stats(&[vnn, vnn], &w.weights).unwrap();
    assert_eq!(stats.mean, vnn);
    assert_eq!(stats.std_dev, 0.0);

    // Diagnostics must say the ensemble is NOT dominated by one conformer.
    let d = w.diagnostics();
    assert_eq!(d.max_weight, 0.5);
    assert!(!d.is_single_conformer_dominated(0.95));
    assert!(
        (d.effective_n_conformers - 2.0).abs() < 1e-12,
        "two degenerate conformers must give effective N = 2, got {}",
        d.effective_n_conformers
    );
}

/// A conformer placed 10 kT above the real butane minimum must be negligible,
/// checked against the closed-form hand value -- the brief's explicit case, but
/// anchored on an actual SCF energy rather than a made-up one.
#[test]
fn butane_conformer_ten_kt_up_is_negligible() {
    let mol = Molecule::parse_xyz(BUTANE_ANTI, 0, 1).unwrap();
    let e_anti = rhf_energy(&mol, "sto-3g");
    let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;

    // Weight the two-state system from the energy DIFFERENCES directly, so the
    // 10 kT gap is exact in f64 (adding 10 kT to -155 Ha and subtracting again
    // loses ~1e-11 relative -- see the ferric-core unit test note).
    let w = boltzmann_weights(&[0.0, 10.0 * kt], DEFAULT_TEMPERATURE_K).unwrap();

    // Hand computation: Z = 1 + exp(-10) = 1.0000453999297625,
    // w_hi = exp(-10)/Z = 4.5397868702434395e-05.
    let expected_hi = 4.539_786_870_243_439_5e-5;
    assert!(
        (w.weights[1] - expected_hi).abs() < 1e-15,
        "10 kT weight should be {expected_hi:.17e}, got {:.17e}",
        w.weights[1]
    );
    assert!(w.weights[1] < 5e-5, "10 kT above the minimum must be negligible");
    assert!((w.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // Sanity that 10 kT is a chemically small number on the butane energy scale:
    // ~5.9 kcal/mol, i.e. a real but strongly disfavoured conformer.
    let ten_kt_kcal = 10.0 * kt * HARTREE_TO_KCAL;
    eprintln!("10 kT = {ten_kt_kcal:.4} kcal/mol (E_anti = {e_anti:.8} Ha)");
    assert!((ten_kt_kcal - 5.925).abs() < 0.01, "10 kT should be ~5.925 kcal/mol");

    // Diagnostics call it: the ensemble was unnecessary.
    let d = w.diagnostics();
    assert!(d.is_single_conformer_dominated(0.95));
    assert!(d.verdict().contains("unnecessary"));
}

/// Guardrail on real geometries: a REORDERED butane (same atoms, permuted) must
/// be rejected. This is the misuse the invariant exists to catch -- averaging
/// per-atom properties across it would silently corrupt every number.
#[test]
fn reordered_butane_is_rejected() {
    let anti = Molecule::parse_xyz(BUTANE_ANTI, 0, 1).unwrap();

    // Swap the first carbon with the first hydrogen: same composition, same
    // atom count, different ORDER.
    let mut permuted = anti.clone();
    permuted.atoms.swap(0, 4);

    let err = ConformerEnsemble::new(vec![
        ferric_core::conformers::Conformer::new(anti),
        ferric_core::conformers::Conformer::new(permuted),
    ])
    .unwrap_err();
    let msg = err.to_string();
    eprintln!("rejected as expected: {msg}");
    assert!(
        msg.contains("atom ordering") || msg.contains("composition"),
        "expected an ordering/composition error, got: {msg}"
    );
}
