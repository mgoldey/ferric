//! `solve_krhf_keep` / `solve_kuhf_keep` (ferric_pbc::kscf, kuscf): the SCF
//! that also hands back the hcore output and J/K object it ran on, so the
//! analytic forces and stress differentiate the energy's own build.
//!
//! Anchors: (1) the energy of the `_keep` entry equals the plain entry bit for
//! bit (dense and RS-GDF RHF, staged-Ewald UHF); (2) the force assembled from
//! the retained objects equals the central finite difference of ferric's own
//! k energy (the same bar as `pbc_kgrad.rs`), for the dense RHF Ewald energy;
//! (3) mutation: a force differentiated against an hcore built at other atom
//! positions (what a stale retained object would be) is refused or misses.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kgrad::{kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig};
use ferric_pbc::kpts::{KPointMesh, MeshCentring};
use ferric_pbc::kscf::{solve_krhf, solve_krhf_keep, KJkKind, KRhfConfig, KScfConfig};
use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig};
use ferric_pbc::kuscf::{solve_kuhf, solve_kuhf_keep, KUhfConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::KRsGdfConfig;
use ferric_pbc::rsgdf::RsGdfConfig;

const FD_H: f64 = 1e-4;
const FD_BAR: f64 = 1e-7;
const ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

fn scf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

fn cell_at(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn mesh(cell: &Cell) -> KPointMesh {
    KPointMesh::new(cell, [1, 1, 3], MeshCentring::Gamma).expect("mesh")
}

fn rhf_cfg(cell: &Cell, exx: ExxDiv, jk: KJkKind) -> KRhfConfig {
    let mut c = KRhfConfig::for_cell(cell, exx);
    c.scf = scf_cfg();
    c.hcore = PeriodicHcoreConfig::with_omega(0.8);
    c.jk = jk;
    c.rsgdf = KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            budget_bytes: Some(1 << 31),
            ..Default::default()
        },
        mutation: None,
    };
    c
}

fn aux() -> BasisSet {
    ferric_core::basis::bundled("cc-pvdz-ri").expect("aux")
}

#[test]
fn keep_energy_is_bitwise_the_plain_energy() {
    let cell = cell_at(&ATOMS);
    let m = mesh(&cell);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    for exx in [ExxDiv::None, ExxDiv::Ewald] {
        let c = rhf_cfg(&cell, exx, KJkKind::Dense);
        let a = solve_krhf(&cell, &prep, None, &m, &c).expect("plain");
        let (b, _) = solve_krhf_keep(&cell, &prep, None, &m, &c).expect("keep");
        assert_eq!(a.energy.to_bits(), b.energy.to_bits(), "{exx:?}");
        assert_eq!(a.iterations, b.iterations);
    }
    let auxp = PreparedBasis::new(cell.mol(), &aux()).expect("aux prep");
    let c = rhf_cfg(&cell, ExxDiv::Ewald, KJkKind::RsGdf);
    let a = solve_krhf(&cell, &prep, Some(&auxp), &m, &c).expect("plain gdf");
    let (b, _) = solve_krhf_keep(&cell, &prep, Some(&auxp), &m, &c).expect("keep gdf");
    assert_eq!(a.energy.to_bits(), b.energy.to_bits(), "rsgdf");
}

#[test]
fn keep_uhf_energy_is_bitwise_the_plain_energy_staged_ewald() {
    let pos = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
    let mut mol = hydrogens(&pos);
    mol.multiplicity = 2;
    let cell = Cell::new(mol, cubic(4.5)).expect("cell");
    let m = mesh(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mut c = KUhfConfig::for_cell(&cell, ExxDiv::Ewald);
    c.scf = scf_cfg();
    c.hcore = PeriodicHcoreConfig::with_omega(0.8);
    let a = solve_kuhf(&cell, &prep, None, &m, &c).expect("plain");
    let (b, kept) = solve_kuhf_keep(&cell, &prep, None, &m, &c).expect("keep");
    assert_eq!(a.scf.energy.to_bits(), b.scf.energy.to_bits());
    assert!(b.none_stage.is_some(), "staged run");
    // the retained pieces differentiate the final stage
    let g = kpoint_uhf_gradient(
        &cell,
        &prep,
        &m,
        &c.hcore,
        &kept.hk,
        kept.grad_jk(None, &c.rsgdf).expect("jk"),
        &b.scf,
        ExxDiv::Ewald,
        &KGradConfig::default(),
    )
    .expect("uhf gradient");
    assert!(g.net_force < 1e-8, "net force {:e}", g.net_force);
}

fn energy(pos: &[[f64; 3]]) -> f64 {
    let cell = cell_at(pos);
    let m = mesh(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let c = rhf_cfg(&cell, ExxDiv::Ewald, KJkKind::Dense);
    solve_krhf(&cell, &prep, None, &m, &c).expect("e").energy
}

fn moved(a: usize, x: usize, d: f64) -> Vec<[f64; 3]> {
    let mut p = ATOMS.to_vec();
    p[a][x] += d;
    p
}

#[test]
fn retained_force_matches_fd_and_a_stale_hcore_does_not() {
    let cell = cell_at(&ATOMS);
    let m = mesh(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let c = rhf_cfg(&cell, ExxDiv::Ewald, KJkKind::Dense);
    let (scf, kept) = solve_krhf_keep(&cell, &prep, None, &m, &c).expect("keep");
    let jk = kept.grad_jk(None, &c.rsgdf).expect("jk");
    let g = kpoint_rhf_gradient(
        &cell,
        &prep,
        &m,
        &c.hcore,
        &kept.hk,
        jk,
        &scf,
        ExxDiv::Ewald,
        &KGradConfig::default(),
    )
    .expect("gradient");
    // The stress goes through the same retained objects.
    let st = kpoint_rhf_stress(
        &cell,
        &prep,
        &m,
        &c.hcore,
        &kept.hk,
        kept.stress_jk(None, &c.rsgdf).expect("jk"),
        &scf,
        ExxDiv::Ewald,
        &KStressConfig::default(),
    )
    .expect("stress");
    assert!(st.antisymmetry() < 1e-9, "{:e}", st.antisymmetry());
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 0), (0, 2), (1, 1)] {
        let fd = (energy(&moved(a, x, FD_H)) - energy(&moved(a, x, -FD_H))) / (2.0 * FD_H);
        let d = g.grad[(a, x)] - fd;
        eprintln!(
            "({a},{x}) analytic {:+.10e} FD {fd:+.10e} diff {d:+.2e}",
            g.grad[(a, x)]
        );
        worst = worst.max(d.abs());
    }
    assert!(worst <= FD_BAR, "FD miss {worst:e}");

    // MUTANT: an hcore built at atoms moved by 0.05 Bohr (a stale retained
    // object) is refused or misses by orders of magnitude.
    let cell2 = cell_at(&moved(0, 2, 0.05));
    let prep2 = prep_for(&cell2, &pyscf_sto3g_h());
    let m2 = mesh(&cell2);
    let hk_bad = periodic_hcore_kpts(&cell2, &prep2, &m2, &c.hcore).expect("hk");
    let bad = kpoint_rhf_gradient(
        &cell,
        &prep,
        &m,
        &c.hcore,
        &hk_bad,
        kept.grad_jk(None, &c.rsgdf).unwrap(),
        &scf,
        ExxDiv::Ewald,
        &KGradConfig::default(),
    );
    match bad {
        Err(e) => eprintln!("stale hcore refused: {e}"),
        Ok(b) => {
            let miss = (b.grad[(0, 2)] - g.grad[(0, 2)]).abs();
            assert!(miss > 1e-5, "stale hcore not detected: {miss:e}");
        }
    }
}
