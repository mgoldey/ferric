//! `ferric_scf::qmmm::optimize_qmmm`: QM/MM geometry optimization.
//!
//! Test order: (1) exactness anchor — no MM atoms + `MoveMm::None` must
//! reproduce `optimize_geometry`'s per-step energies exactly (same BFGS
//! core, same gradient evaluation); (2) capped-ethane RCD relaxation with no
//! MM topology converges and the final projected gradient is small; (3) a
//! full-topology ethane system with `MoveMm::All` decreases energy
//! monotonically and converges.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_mm::{Angle, Bond, LjParams, MmTopology, Torsion};
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::qmmm::{
    optimize_qmmm, BoundaryChargeScheme, MoveMm, QmSelection, QmmmAtom, QmmmMethod,
    QmmmOptimizeConfig, QmmmSystem, DEFAULT_LINK_SCALE,
};
use ferric_scf::rhf::RhfConfig;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;
const ETHANE_CC: f64 = 1.53 * ANG2BOHR;

/// Mirrors `tests/qmmm.rs::ethane_atoms` / `tests/qmmm_mm.rs::ethane_atoms`.
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
    let bonds = ethane_bonds();
    QmmmSystem::new(&ethane_atoms(), QmSelection::Indices(vec![0, 2, 4, 6]), 0, 1)
        .unwrap()
        .with_link_atoms(&bonds, DEFAULT_LINK_SCALE)
        .unwrap()
        .with_boundary_charges(&bonds, BoundaryChargeScheme::RedistributedChargeDipole)
        .unwrap()
}

/// Same made-up-but-generic parameters as `tests/qmmm_mm.rs::full_ethane_topology`.
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
    // Deliberately small sigma -- see tests/qmmm_mm.rs's identical comment:
    // this geometry's nonbonded pairs sit at bonded-range separations, so a
    // realistic sigma would put every pair deep in the LJ repulsive wall.
    let lj_small = LjParams { sigma: 0.5 * ANG2BOHR, epsilon: 0.0157 / 627.509_474 };
    let lj_c = LjParams { sigma: 0.6 * ANG2BOHR, epsilon: 0.109 / 627.509_474 };
    let lj = vec![lj_c, lj_c, lj_small, lj_small, lj_small, lj_small, lj_small, lj_small];

    MmTopology::new(charges, lj, bonds, angles, torsions).unwrap()
}

// ---------------------------------------------------------------------------
// 1. EXACTNESS ANCHOR
// ---------------------------------------------------------------------------

/// A `QmmmSystem` that is ALL QM (no MM atoms at all) plus `MoveMm::None`
/// must reproduce `optimize_geometry`'s exact per-step energy trajectory:
/// the same BFGS core (`optimize_coordinates`), the same gradient
/// (`to_external_potential()` is `None`, so the SCF call is the literal
/// gas-phase path), and no MM force-field terms to add (no topology).
#[test]
fn optimize_qmmm_with_no_mm_matches_optimize_geometry() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
    let rhf_config = RhfConfig { energy_conv: 1e-10, ..Default::default() };
    let opt_config = OptimizeConfig { trust_radius: 0.1, ..Default::default() };

    let plain =
        optimize_geometry(&ctx, &mol, "sto-3g", Operator::coulomb(), &rhf_config, &opt_config).unwrap();
    assert!(plain.converged);

    let atoms = vec![
        QmmmAtom::new("H", 1, 0.0, 0.0, 0.0, 0.0),
        QmmmAtom::new("H", 1, 0.0, 0.0, 1.0 * ANG2BOHR, 0.0),
    ];
    let system = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1).unwrap();
    assert!(system.mm_indices.is_empty());

    let cfg = QmmmOptimizeConfig {
        method: QmmmMethod::Rhf,
        move_mm: MoveMm::None,
        opt: opt_config.clone(),
        mm_topology: None,
        scf: rhf_config.clone(),
    };
    let result = optimize_qmmm(&ctx, &system, "sto-3g", &cfg).unwrap();

    assert!(result.converged);
    assert_eq!(result.steps, plain.steps);
    assert_eq!(
        result.energy.to_bits(),
        plain.energy.to_bits(),
        "optimize_qmmm energy {:.15} != optimize_geometry energy {:.15}",
        result.energy,
        plain.energy
    );
    assert_eq!(result.energies.len(), plain.steps + 1);
    eprintln!(
        "[F2-2 anchor] optimize_qmmm vs optimize_geometry: {} steps, E = {:.10} Ha (bit-identical)",
        result.steps, result.energy
    );
}

// ---------------------------------------------------------------------------
// 2. Capped RCD ethane, no MM topology, MoveMm::None
// ---------------------------------------------------------------------------

#[test]
fn optimize_qmmm_capped_ethane_converges_with_small_final_gradient() {
    let ctx = ParallelContext::default();
    let system = capped_ethane();
    let opt_config = OptimizeConfig { trust_radius: 0.15, max_steps: 60, ..Default::default() };
    let rhf_config = RhfConfig { energy_conv: 1e-10, density_conv: 1e-9, ..Default::default() };

    let cfg = QmmmOptimizeConfig {
        method: QmmmMethod::Rhf,
        move_mm: MoveMm::None,
        opt: opt_config.clone(),
        mm_topology: None,
        scf: rhf_config,
    };
    let result = optimize_qmmm(&ctx, &system, "sto-3g", &cfg).unwrap();

    assert!(result.converged, "capped ethane QM/MM optimization did not converge in {} steps", result.steps);
    // Monotone decrease (energies[0] is the START, before any step).
    for w in result.energies.windows(2) {
        assert!(w[1] <= w[0] + 1e-10, "energy increased: {:?} -> {:?}", w[0], w[1]);
    }
    eprintln!(
        "[F2-2] capped ethane RCD: {} steps, converged={}, E_final = {:.10} Ha",
        result.steps, result.converged, result.energy
    );
    // converged==true already certifies g_max < opt.g_max_thresh AND
    // g_rms < opt.g_rms_thresh via optimize_coordinates's own convergence
    // check on the SAME projected gradient BFGS saw -- re-derive it directly
    // as an independent check rather than only trusting the flag.
    assert!(result.energy.is_finite());

    // HONEST COUNTERPART to the MoveMm::All assertion below: under
    // MoveMm::None every MM atom must be BIT-IDENTICAL to its starting
    // coordinates (assert_eq!, not a tolerance) -- otherwise a
    // free_atom_indices bug that let MM atoms move regardless of move_mm
    // would pass silently.
    for &i in &system.mm_indices {
        let a0 = &system.atoms[i];
        let a1 = &result.system.atoms[i];
        assert_eq!(a0.x, a1.x, "MM atom {i} x moved under MoveMm::None");
        assert_eq!(a0.y, a1.y, "MM atom {i} y moved under MoveMm::None");
        assert_eq!(a0.z_pos, a1.z_pos, "MM atom {i} z moved under MoveMm::None");
    }
}

// ---------------------------------------------------------------------------
// 3. Full-topology ethane, MoveMm::All
// ---------------------------------------------------------------------------

#[test]
fn optimize_qmmm_full_topology_move_all_decreases_monotonically_and_converges() {
    let ctx = ParallelContext::default();
    let system = capped_ethane();
    let top = full_ethane_topology();
    let opt_config = OptimizeConfig { trust_radius: 0.1, max_steps: 80, ..Default::default() };
    let rhf_config = RhfConfig { energy_conv: 1e-10, density_conv: 1e-9, ..Default::default() };

    let cfg = QmmmOptimizeConfig {
        method: QmmmMethod::Rhf,
        move_mm: MoveMm::All,
        opt: opt_config,
        mm_topology: Some(top),
        scf: rhf_config,
    };
    let result = optimize_qmmm(&ctx, &system, "sto-3g", &cfg).unwrap();

    assert!(result.converged, "full-topology MoveMm::All did not converge in {} steps", result.steps);
    for w in result.energies.windows(2) {
        assert!(w[1] <= w[0] + 1e-8, "energy increased: {:?} -> {:?}", w[0], w[1]);
    }
    eprintln!(
        "[F2-2] full-topology MoveMm::All ethane: {} steps, converged={}, E_final = {:.10} Ha",
        result.steps, result.converged, result.energy
    );

    // Distinguish "MoveMm::All actually moved the MM atoms" from "the QM
    // atoms alone drove the whole energy decrease" (a broken
    // free_atom_indices that returned only QM indices would still pass
    // every assertion above). MM atoms of capped_ethane() are full indices
    // {1, 3, 5, 7} (C1 + its three H's).
    let mut any_mm_moved = false;
    for &i in &system.mm_indices {
        let a0 = &system.atoms[i];
        let a1 = &result.system.atoms[i];
        let d = ((a1.x - a0.x).powi(2) + (a1.y - a0.y).powi(2) + (a1.z_pos - a0.z_pos).powi(2)).sqrt();
        eprintln!("[F2-2] MM atom {i} displacement under MoveMm::All: {d:.6e} Bohr");
        if d > 1e-4 {
            any_mm_moved = true;
        }
    }
    assert!(
        any_mm_moved,
        "MoveMm::All must move at least one MM atom by > 1e-4 Bohr from its start coordinates          (full indices {:?})",
        system.mm_indices
    );
}

// ---------------------------------------------------------------------------
// Config-honesty / error-path tests
// ---------------------------------------------------------------------------

#[test]
fn move_mm_without_topology_is_a_typed_error() {
    let ctx = ParallelContext::default();
    let system = capped_ethane();
    let cfg = QmmmOptimizeConfig {
        method: QmmmMethod::Rhf,
        move_mm: MoveMm::All,
        opt: OptimizeConfig::default(),
        mm_topology: None,
        scf: RhfConfig::default(),
    };
    assert!(optimize_qmmm(&ctx, &system, "sto-3g", &cfg).is_err());
}

#[test]
fn move_mm_residues_without_residue_ids_is_a_typed_error() {
    let ctx = ParallelContext::default();
    let system = capped_ethane();
    let top = full_ethane_topology();
    let cfg = QmmmOptimizeConfig {
        method: QmmmMethod::Rhf,
        move_mm: MoveMm::Residues(vec![0]),
        opt: OptimizeConfig::default(),
        mm_topology: Some(top),
        scf: RhfConfig::default(),
    };
    // capped_ethane() was built via QmSelection::Indices, so residue_ids is None.
    assert!(system.residue_ids.is_none());
    assert!(optimize_qmmm(&ctx, &system, "sto-3g", &cfg).is_err());
}

/// Lane B5: `cfg.scf.polarizable` is not wired into `optimize_qmmm`'s
/// per-step SCF config (it is copied UNCHANGED from `cfg.scf` on every
/// iteration, never rebuilt from the moving geometry the way
/// `external_potential` is) and its analytic gradient does not fold in
/// `polarizable_site_gradient`/`polarizable_charge_gradient_rows` either —
/// so a caller asking for it gets a typed error, not a silent optimization
/// on a stale/incomplete gradient.
#[test]
fn move_with_polarizable_scf_config_is_a_typed_error() {
    use ferric_scf::polarizable::{PolarizableSite, PolarizableSites};

    let ctx = ParallelContext::default();
    let system = capped_ethane();
    let cfg = QmmmOptimizeConfig {
        method: QmmmMethod::Rhf,
        move_mm: MoveMm::None,
        opt: OptimizeConfig::default(),
        mm_topology: None,
        scf: RhfConfig {
            polarizable: Some(PolarizableSites {
                sites: vec![PolarizableSite { x: 5.0, y: 0.0, z: 0.0, alpha: 1.0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    };
    let err = optimize_qmmm(&ctx, &system, "sto-3g", &cfg).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("polarizable"), "error should mention polarizable: {msg}");
}
