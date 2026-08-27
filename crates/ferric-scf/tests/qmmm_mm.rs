//! `ferric_scf::qmmm::qmmm_mm_terms` / `full_gradient_with_mm`: assembling
//! `ferric-mm`'s force field into the QM/MM partition layer.
//!
//! Test order: (1) exactness anchor (empty/no-op topology is bit-identical
//! to zero), (2) a hand-derived surviving-term count on capped ethane's full
//! topology (the additive-scheme convention: keep bonded terms with >=1 MM
//! atom, drop all-QM ones), (3) the MM-part gradient vs finite difference.

use ferric_mm::{Angle, Bond, LjParams, MmTopology, Torsion};
use ferric_scf::qmmm::{qmmm_mm_terms, QmSelection, QmmmAtom, QmmmSystem, DEFAULT_LINK_SCALE};
use ndarray::Array2;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;
const ETHANE_CC: f64 = 1.53 * ANG2BOHR;

/// Mirrors `crates/ferric-scf/tests/qmmm.rs::ethane_atoms` exactly (staggered
/// ethane, C0=0, C1=1, H's 2,4,6 on C0 / 3,5,7 on C1).
fn ethane_atoms() -> Vec<QmmmAtom> {
    let cc = ETHANE_CC;
    let ch = 1.09 * ANG2BOHR;
    let theta = 109.5_f64.to_radians();
    let (s, c) = (theta.sin(), theta.cos());
    let mut atoms = vec![
        QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.1),
        QmmmAtom::new("C", 6, 0.0, 0.0, cc, -0.1),
    ];
    for k in 0..3 {
        let phi = 2.0 * std::f64::consts::PI * (k as f64) / 3.0;
        atoms.push(QmmmAtom::new("H", 1, ch * s * phi.cos(), ch * s * phi.sin(), ch * c, 0.033));
        atoms.push(QmmmAtom::new("H", 1, ch * s * phi.cos(), ch * s * phi.sin(), cc - ch * c, 0.033));
    }
    atoms
}

fn ethane_bonds() -> Vec<(usize, usize)> {
    vec![(0, 1), (0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
}

fn capped_ethane() -> QmmmSystem {
    QmmmSystem::new(&ethane_atoms(), QmSelection::Indices(vec![0, 2, 4, 6]), 0, 1)
        .unwrap()
        .with_link_atoms(&ethane_bonds(), DEFAULT_LINK_SCALE)
        .unwrap()
}

fn ethane_coords_full() -> Array2<f64> {
    let atoms = ethane_atoms();
    let mut c = Array2::<f64>::zeros((atoms.len(), 3));
    for (i, a) in atoms.iter().enumerate() {
        c[(i, 0)] = a.x;
        c[(i, 1)] = a.y;
        c[(i, 2)] = a.z_pos;
    }
    c
}

/// The full-ethane bonded topology (all 8 atoms, all 7 bonds, every H-C-C
/// and H-C-H angle, every H-C-C-H torsion), with nonzero LJ/charges so the
/// nonbonded assembly is exercised too. Parameters are made-up-but-generic
/// (not sourced from a real force field — this is a bookkeeping test, not a
/// physics one).
fn full_ethane_topology() -> MmTopology {
    let bonds = vec![
        Bond { i: 0, j: 1, k: 0.35, r0: ETHANE_CC },
        Bond { i: 0, j: 2, k: 0.4, r0: 1.09 * ANG2BOHR },
        Bond { i: 0, j: 4, k: 0.4, r0: 1.09 * ANG2BOHR },
        Bond { i: 0, j: 6, k: 0.4, r0: 1.09 * ANG2BOHR },
        Bond { i: 1, j: 3, k: 0.4, r0: 1.09 * ANG2BOHR },
        Bond { i: 1, j: 5, k: 0.4, r0: 1.09 * ANG2BOHR },
        Bond { i: 1, j: 7, k: 0.4, r0: 1.09 * ANG2BOHR },
    ];
    let theta0 = 109.5_f64.to_radians();
    let mut angles = vec![];
    for h in [2, 4, 6] {
        angles.push(Angle { i: h, j: 0, k: 1, k_theta: 0.06, theta0 });
    }
    for h in [3, 5, 7] {
        angles.push(Angle { i: h, j: 1, k: 0, k_theta: 0.06, theta0 });
    }
    let hc0 = [2, 4, 6];
    for a in 0..3 {
        for b in (a + 1)..3 {
            angles.push(Angle { i: hc0[a], j: 0, k: hc0[b], k_theta: 0.04, theta0 });
        }
    }
    let hc1 = [3, 5, 7];
    for a in 0..3 {
        for b in (a + 1)..3 {
            angles.push(Angle { i: hc1[a], j: 1, k: hc1[b], k_theta: 0.04, theta0 });
        }
    }
    let mut torsions = vec![];
    for &hi in &[2, 4, 6] {
        for &hj in &[3, 5, 7] {
            torsions.push(Torsion { i: hi, j: 0, k: 1, l: hj, periodicity: 3, k_phi: 0.02, phase: 0.0 });
        }
    }

    let charges = vec![-0.1, -0.1, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033];
    // sigma is deliberately small (not a realistic 3.4/2.6 A carbon/hydrogen
    // sigma) for the same reason as ferric-mm's own
    // combined_topology_all_terms_gradient_vs_fd test: this ethane
    // geometry's nonbonded pairs sit at BONDED-range separations
    // (~1.5-2.9 A), so a realistic sigma puts every pair deep in the LJ
    // repulsive wall, where FD truncation error swamps a 1e-8 tolerance at
    // h=1e-5 for reasons that have nothing to do with the assembly logic
    // this test actually checks (LJ correctness itself is covered, at an
    // appropriate separation, by ferric-mm's own nonbonded_hand_computed_
    // and_gradient_vs_fd and the OpenMM cross-validation).
    let lj = vec![
        LjParams { sigma: 0.6 * ANG2BOHR, epsilon: 0.109 / 627.509_474 },
        LjParams { sigma: 0.6 * ANG2BOHR, epsilon: 0.109 / 627.509_474 },
        LjParams { sigma: 0.5 * ANG2BOHR, epsilon: 0.0157 / 627.509_474 },
        LjParams { sigma: 0.5 * ANG2BOHR, epsilon: 0.0157 / 627.509_474 },
        LjParams { sigma: 0.5 * ANG2BOHR, epsilon: 0.0157 / 627.509_474 },
        LjParams { sigma: 0.5 * ANG2BOHR, epsilon: 0.0157 / 627.509_474 },
        LjParams { sigma: 0.5 * ANG2BOHR, epsilon: 0.0157 / 627.509_474 },
        LjParams { sigma: 0.5 * ANG2BOHR, epsilon: 0.0157 / 627.509_474 },
    ];

    MmTopology::new(charges, lj, bonds, angles, torsions).unwrap()
}

/// Exactness anchor: a topology with no bonded terms and zero LJ/charges
/// (every energy term is identically zero for ANY coordinates) gives exactly
/// zero total energy and an exactly zero gradient, regardless of the QM/MM
/// partition. `n_atoms()` must still match the full structure.
#[test]
fn qmmm_mm_terms_with_empty_topology_is_zero() {
    let sys = capped_ethane();
    let n = sys.atoms.len();
    let top = MmTopology::new(
        vec![0.0; n],
        vec![LjParams { sigma: 0.0, epsilon: 0.0 }; n],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();
    let coords = ethane_coords_full();
    let (e, g) = qmmm_mm_terms(&sys, &top, &coords).unwrap();
    assert_eq!(e.total, 0.0);
    assert_eq!(e.bond, 0.0);
    assert_eq!(e.angle, 0.0);
    assert_eq!(e.torsion, 0.0);
    assert_eq!(e.lj, 0.0);
    assert_eq!(e.coulomb, 0.0);
    for i in 0..n {
        for c in 0..3 {
            assert_eq!(g[(i, c)], 0.0, "row {i} col {c}");
        }
    }
}

/// Hand-derived surviving-term counts on capped ethane's full topology.
/// QM = {0,2,4,6} (C0 + 3 H on C0), MM = {1,3,5,7} (C1 + 3 H on C1).
///
/// Bonds (7 total): (0,1) C-C survives (crosses); (0,2)/(0,4)/(0,6) are
/// QM-QM, dropped; (1,3)/(1,5)/(1,7) are MM-MM... wait MM-MM bonds ALSO
/// survive (they involve >=1 MM atom, and in fact both atoms are MM) ->
/// 4 bonds survive: (0,1), (1,3), (1,5), (1,7).
///
/// Angles (12 total: 3 H-C0-C1 + 3 H-C1-C0 + 3 H-C0-H + 3 H-C1-H): the 3
/// H(QM)-C0-C1(MM) survive (cross), the 3 H(MM)-C1-C0(QM) survive (cross),
/// the 3 H(QM)-C0-H(QM) are dropped (all QM), the 3 H(MM)-C1-H(MM) survive
/// (all MM, but MM-only terms are exactly the "MM-MM bonded" case which
/// also survives under "keep any term with >=1 MM atom") -> 9 survive.
///
/// Torsions (9 total, H(C0)-C0-C1-H(C1)): every one has one QM H and one MM
/// H, so all 9 survive (all cross the boundary).
#[test]
fn capped_ethane_surviving_bonded_term_counts_match_hand_derivation() {
    let sys = capped_ethane();
    let top = full_ethane_topology();
    let coords = ethane_coords_full();

    // Cross-check by directly counting from the QM/MM partition + topology
    // bond lists (independent of qmmm_mm_terms's internal filtering, so this
    // is a real check on the FUNCTION's behavior, not a restatement of it):
    // an energy/gradient computed with a manually-filtered topology
    // (containing exactly the bonds/angles/torsions this test asserts
    // survive) must match qmmm_mm_terms's bonded-only result exactly.
    let qm: std::collections::HashSet<usize> = sys.qm_indices.iter().copied().collect();
    let has_mm = |atoms: &[usize]| atoms.iter().any(|a| !qm.contains(a));

    let surviving_bonds: Vec<_> = top.bonds.iter().filter(|b| has_mm(&[b.i, b.j])).cloned().collect();
    let surviving_angles: Vec<_> =
        top.angles.iter().filter(|a| has_mm(&[a.i, a.j, a.k])).cloned().collect();
    let surviving_torsions: Vec<_> =
        top.torsions.iter().filter(|t| has_mm(&[t.i, t.j, t.k, t.l])).cloned().collect();

    assert_eq!(surviving_bonds.len(), 4, "hand count: 4 bonds cross or are MM-MM");
    assert_eq!(surviving_angles.len(), 9, "hand count: 9 of 12 angles survive");
    assert_eq!(surviving_torsions.len(), 9, "hand count: all 9 torsions cross");

    let n = sys.atoms.len();
    let manual_top = MmTopology::new(
        vec![0.0; n],
        vec![LjParams { sigma: 0.0, epsilon: 0.0 }; n],
        surviving_bonds,
        surviving_angles,
        surviving_torsions,
    )
    .unwrap();
    let (e_manual, g_manual) = ferric_mm::gradient(&manual_top, &coords).unwrap();

    let (e_assembled, _g_assembled) = qmmm_mm_terms(&sys, &top, &coords).unwrap();
    assert!((e_assembled.bond - e_manual.bond).abs() < 1e-14);
    assert!((e_assembled.angle - e_manual.angle).abs() < 1e-14);
    assert!((e_assembled.torsion - e_manual.torsion).abs() < 1e-14);
    // g_manual (bonded-terms-only, zero LJ/charges) is a cross-check on the
    // ENERGY split above; the assembled gradient also carries MM-MM and
    // QM-MM nonbonded contributions ferric_mm::gradient's own totals don't
    // isolate, so the full gradient is checked against FD directly in
    // qmmm_mm_terms_gradient_matches_fd below instead of re-deriving a
    // bonded-only gradient slice here.
    let _ = g_manual;
}

/// MM-part gradient vs central finite difference of the assembled MM
/// energy, on capped ethane with the full topology.
#[test]
fn qmmm_mm_terms_gradient_matches_fd() {
    let sys = capped_ethane();
    let top = full_ethane_topology();
    let coords = ethane_coords_full();

    let (e0, g) = qmmm_mm_terms(&sys, &top, &coords).unwrap();
    let h = 1e-5;
    let n = coords.nrows();
    let mut max_err = 0.0_f64;
    for i in 0..n {
        for c in 0..3 {
            let mut plus = coords.clone();
            plus[(i, c)] += h;
            let mut minus = coords.clone();
            minus[(i, c)] -= h;
            let e_plus = qmmm_mm_terms(&sys, &top, &plus).unwrap().0.total;
            let e_minus = qmmm_mm_terms(&sys, &top, &minus).unwrap().0.total;
            let fd = (e_plus - e_minus) / (2.0 * h);
            max_err = max_err.max((fd - g[(i, c)]).abs());
        }
    }
    assert!(max_err < 1e-8, "max analytic-vs-FD err {max_err:.3e} (e0.total={:.10})", e0.total);
}

/// Trivial-limit sanity: an all-QM system (no MM atoms at all) has the MM
/// gradient reduce to whatever bonded terms are entirely... actually an
/// all-QM system means qm_indices covers everything, so every bonded term
/// is "all QM" and is dropped, and there are no MM atoms for nonbonded/LJ
/// terms either -> exactly zero, matching the empty-topology anchor's
/// numeric result even with a NONTRIVIAL topology.
#[test]
fn all_qm_system_has_zero_mm_contribution_even_with_nontrivial_topology() {
    let atoms = ethane_atoms();
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2, 3, 4, 5, 6, 7]), 0, 1).unwrap();
    let top = full_ethane_topology();
    let coords = ethane_coords_full();
    let (e, g) = qmmm_mm_terms(&sys, &top, &coords).unwrap();
    assert_eq!(e.total, 0.0);
    for i in 0..8 {
        for c in 0..3 {
            assert_eq!(g[(i, c)], 0.0);
        }
    }
}
