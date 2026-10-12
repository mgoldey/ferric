//! `solve_krks_keep` (ferric_pbc::kdft): the closed-shell k-point KS SCF that
//! also hands back the hcore output and J/K object it ran on, so
//! `kpoint_rks_gradient` differentiates the energy's own build.
//!
//! Anchors: (1) the energy, E_xc and iteration count of the `_keep` entry equal
//! `solve_krks` bit for bit (LDA and PBE0, dense; PBE0, RS-GDF); (2) the force
//! assembled from the retained objects equals the central finite difference of
//! ferric's own k energy on one explicit neighbour cutoff; (3) mutation: a
//! force differentiated against an hcore built at other atom positions (what a
//! stale retained object would be) is refused or misses; (4) the retained
//! bytes are charged against the derivative budget and a budget that covers
//! only them gives a typed ledger error, not an unbounded run.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks, solve_krks_keep, KRksConfig};
use ferric_pbc::kgrad::KGradConfig;
use ferric_pbc::kpts::{KPointMesh, MeshCentring};
use ferric_pbc::kscf::{KJkKind, KScfConfig};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::KRsGdfConfig;
use ferric_pbc::rsgdf::RsGdfConfig;

const FD_H: f64 = 1e-4;
const FD_BAR: f64 = 2e-6;
const CUTOFF: f64 = 10.0;
const ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

fn grid() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(40, 110);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_at(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn mesh(cell: &Cell) -> KPointMesh {
    KPointMesh::new(cell, [1, 1, 3], MeshCentring::Gamma).expect("mesh")
}

fn cfg(cell: &Cell, functional: &str, exx: ExxDiv, jk: KJkKind) -> KRksConfig {
    let mut c = KRksConfig::new(cell, exx, functional);
    c.krhf.scf = KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    };
    c.krhf.hcore = PeriodicHcoreConfig::with_omega(0.8);
    c.krhf.jk = jk;
    c.krhf.rsgdf = KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            budget_bytes: Some(1 << 31),
            ..Default::default()
        },
        mutation: None,
    };
    c.grid = grid();
    c
}

fn aux() -> BasisSet {
    ferric_core::basis::bundled("cc-pvdz-ri").expect("aux")
}

fn grad_cfg(exx: ExxDiv, budget: Option<usize>) -> KRksGradConfig {
    KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: budget,
            ..Default::default()
        },
        grid: KXcGrid::Atom(grid()),
        exxdiv: Some(exx),
        ..Default::default()
    }
}

#[test]
fn keep_is_bitwise_the_plain_solve_krks() {
    let cell = cell_at(&ATOMS);
    let m = mesh(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    for functional in ["LDA", "PBE0"] {
        let c = cfg(&cell, functional, ExxDiv::Ewald, KJkKind::Dense);
        let a = solve_krks(&cell, &prep, None, &m, &c).expect("plain");
        let (b, _) = solve_krks_keep(&cell, &prep, None, &m, &c).expect("keep");
        assert_eq!(a.scf.energy.to_bits(), b.scf.energy.to_bits(), "{functional}");
        assert_eq!(a.e_xc.to_bits(), b.e_xc.to_bits(), "{functional}");
        assert_eq!(a.scf.iterations, b.scf.iterations, "{functional}");
    }
    let auxp = PreparedBasis::new(cell.mol(), &aux()).expect("aux prep");
    let c = cfg(&cell, "PBE0", ExxDiv::Ewald, KJkKind::RsGdf);
    let a = solve_krks(&cell, &prep, Some(&auxp), &m, &c).expect("plain gdf");
    let (b, _) = solve_krks_keep(&cell, &prep, Some(&auxp), &m, &c).expect("keep gdf");
    assert_eq!(a.scf.energy.to_bits(), b.scf.energy.to_bits(), "rsgdf");
}

fn energy(pos: &[[f64; 3]]) -> f64 {
    let cell = cell_at(pos);
    let m = mesh(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let c = cfg(&cell, "PBE0", ExxDiv::Ewald, KJkKind::Dense);
    solve_krks(&cell, &prep, None, &m, &c)
        .expect("e")
        .scf
        .energy
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
    let c = cfg(&cell, "PBE0", ExxDiv::Ewald, KJkKind::Dense);
    let (r, kept) = solve_krks_keep(&cell, &prep, None, &m, &c).expect("keep");
    let force = |hk: &ferric_pbc::hcore::kpoint::PeriodicHcoreK| {
        kpoint_rks_gradient(
            &cell,
            &prep,
            &m,
            &c.krhf.hcore,
            hk,
            kept.grad_jk(None, &c.krhf.rsgdf).unwrap(),
            &r.scf,
            "PBE0",
            &grad_cfg(ExxDiv::Ewald, None),
        )
    };
    let g = force(&kept.hk).expect("gradient");
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 0), (0, 2), (1, 1)] {
        let fd = (energy(&moved(a, x, FD_H)) - energy(&moved(a, x, -FD_H))) / (2.0 * FD_H);
        let d = g.grad[(a, x)] - fd;
        eprintln!(
            "({a},{x}) analytic {:+.10e} FD {fd:+.10e} diff {d:+.2e}",
            g.grad[(a, x)]
        );
        assert!(g.grad[(a, x)].abs() > 1e-3, "trivial component");
        worst = worst.max(d.abs());
    }
    assert!(worst <= FD_BAR, "FD miss {worst:e}");

    // MUTANT: an hcore built at atoms moved by 0.05 Bohr (a stale retained
    // object) is refused or misses by orders of magnitude.
    let cell2 = cell_at(&moved(0, 2, 0.05));
    let prep2 = prep_for(&cell2, &pyscf_sto3g_h());
    let m2 = mesh(&cell2);
    let hk_bad = periodic_hcore_kpts(&cell2, &prep2, &m2, &c.krhf.hcore).expect("hk");
    match force(&hk_bad) {
        Err(e) => eprintln!("stale hcore refused: {e}"),
        Ok(b) => {
            let miss = (b.grad[(0, 2)] - g.grad[(0, 2)]).abs();
            assert!(miss > 1e-5, "stale hcore not detected: {miss:e}");
        }
    }
}

#[test]
fn retained_bytes_are_charged_against_the_derivative_budget() {
    let cell = cell_at(&ATOMS);
    let m = mesh(&cell);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let c = cfg(&cell, "PBE0", ExxDiv::Ewald, KJkKind::Dense);
    let (r, kept) = solve_krks_keep(&cell, &prep, None, &m, &c).expect("keep");
    let resident = kept.resident_bytes();
    // nk = 3, nao = 2: the dense kernels alone are 2 * 9 * 16 * 16 bytes.
    assert!(resident >= 2 * 9 * 16 * 16, "resident {resident}");
    let ample = 1usize << 31;
    assert_eq!(kept.derivative_budget(Some(ample)), ample - resident);
    assert_eq!(kept.derivative_budget(Some(1)), 1);
    let grad = |budget: usize| {
        kpoint_rks_gradient(
            &cell,
            &prep,
            &m,
            &c.krhf.hcore,
            &kept.hk,
            kept.grad_jk(None, &c.krhf.rsgdf).unwrap(),
            &r.scf,
            "PBE0",
            &grad_cfg(ExxDiv::Ewald, Some(budget)),
        )
    };
    assert!(grad(kept.derivative_budget(Some(ample))).is_ok());
    // A budget that only covers the retained objects leaves the derivative
    // ~1 byte (never 0, which would mean "unset"): a typed ledger error.
    let e = grad(kept.derivative_budget(Some(resident)))
        .err()
        .expect("a budget covering only the retained objects must fail")
        .to_string();
    eprintln!("budget-exhausted derivative: {e}");
    assert!(
        e.contains("budget") || e.contains("memory"),
        "not a ledger error: {e}"
    );
}
