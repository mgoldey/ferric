//! Weekly validation of the closed-shell meta-GGA derivatives of issue #492:
//! the Gamma-point force and stress routed through the 1x1x1 mesh
//! (`ferric_pbc::gamma_mgga`) for SCAN / r2SCAN / TPSS, and the k-point TPSS
//! force and stress (`kpoint_rks_gradient` / `kpoint_rks_stress`). The
//! per-commit subsets, hypotheses and mutation ledgers are in
//! `pbc_gamma_mgga_derivs.rs`, `pbc_kxcgrad_mgga.rs` and
//! `pbc_kxcstress_mgga.rs`.
//!
//! Oracles: central FD of ferric's own SCF energy (fresh SCF per
//! displacement or strain; the Gamma energy for the routed derivatives, an
//! independent construction), with an h sweep (the residual must scale as
//! h^2, not stay constant); the k-mesh == supercell identity for TPSS; and
//! the two-orbital H4 triclinic cell (Gamma: SCAN, r2SCAN, TPSS; k-point:
//! TPSS, 1.4e-2 Ha from SCAN there). The Gamma H2 cell has ONE occupied
//! orbital, where SCAN and r2SCAN coincide exactly (alpha = 0 everywhere;
//! asserted), so it alone cannot separate them.

mod common;
#[path = "gamma_mgga_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig};
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kstress::{KStressConfig, KStressJk};
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid};
use ferric_pbc::kxcstress::{kpoint_rks_stress, KRksStress, KRksStressConfig};
use ferric_pbc::lattice::Cell;
use support::*;

// ----------------------------------------------------------- Gamma, routed

/// Routed Gamma force vs FD of the Gamma energy, SCAN / r2SCAN / TPSS, three
/// components, h = 1e-4 and 2e-4, on the ONE-orbital H2 cell. MEASURED
/// (2026-10-22): 2.32e-9 at h = 1e-4 for all three (the h^2 truncation of
/// the bond-direction third derivative, dominated by terms the functionals
/// share), 9.25e-9 at h = 2e-4 (ratio 4.0).
///
/// IDENTITY (measured 5.6e-17, asserted): with one occupied orbital tau =
/// tau_W, so SCAN's alpha and r2SCAN's regularised alpha are both 0 at every
/// point, where both reduce to the same alpha = 0 limit: the SCAN and r2SCAN
/// Gamma forces COINCIDE on this cell. This fixture therefore cannot tell
/// SCAN from r2SCAN (TPSS differs, asserted); the two-orbital H4 cell below
/// can.
#[test]
#[ignore = "validation: 3 functionals x 2 steps x 6 Gamma SCFs (36 SCFs); about 90 s on a quiet box"]
fn gamma_mgga_force_fd_all_functionals_and_h_sweep() {
    let mut forces = Vec::new();
    for f in MGGA {
        let an = force(&gamma(cell_of(&H2), f), None);
        forces.push(an.grad.clone());
        let mut r = [0.0; 2];
        for (k, h) in [H, 2.0 * H].into_iter().enumerate() {
            let fd = fd_force(f, h, &FORCE_COMPS);
            r[k] = worst_force(&an.grad, &fd, &FORCE_COMPS);
            eprintln!("{f} h={h:.0e}: Gamma routed force vs FD {:.3e}", r[k]);
        }
        assert!(r[0] < FD_BAR, "{f}: {:e}", r[0]);
        assert!(r[1] < 4.0 * FD_BAR, "{f} h=2e-4: {:e}", r[1]);
        // h^2: the residual grows ~4x with the step (a constant offset, i.e.
        // a missing term, would not).
        assert!(
            r[1] > 2.0 * r[0],
            "{f}: residual not h^2 ({:e} -> {:e})",
            r[0],
            r[1]
        );
    }
    let scan_r2 = diff(&forces[0], &forces[1]);
    let scan_tpss = diff(&forces[0], &forces[2]);
    eprintln!("one-orbital H2: SCAN vs r2SCAN force {scan_r2:.3e}, SCAN vs TPSS {scan_tpss:.3e}");
    assert!(scan_r2 < 1e-12, "alpha = 0 identity broken: {scan_r2:e}");
    assert!(scan_tpss > MUTANT_BAR, "{scan_tpss:e}");
}

/// The routed Gamma force on the TWO-orbital H4 triclinic cell (alpha != 0,
/// so SCAN, r2SCAN and TPSS all differ; asserted pairwise) against FD of the
/// Gamma energy, two components, h = 1e-4, atom grid (20, 50), cutoff 14.
/// MEASURED (2026-10-22): SCAN 1.6e-10, r2SCAN 5.1e-10, TPSS 1.1e-9; the
/// forces differ pairwise by 1.5e-3 (SCAN / r2SCAN) to 3.3e-3.
#[test]
#[ignore = "validation: H4 triclinic Gamma, 3 functionals x 5 SCFs"]
fn gamma_mgga_force_fd_on_the_two_orbital_triclinic_cell() {
    let tri = |p: &[[f64; 3]]| Cell::new(hydrogens(p), TRI_A).expect("triclinic cell");
    let comps = [(0, 0), (2, 1)];
    let mut forces = Vec::new();
    for f in MGGA {
        let an = force(&gamma(tri(&TRI_ATOMS), f), None);
        let mut worst = 0.0_f64;
        for &(a, x) in &comps {
            let e = |h: f64| {
                let mut p = TRI_ATOMS.to_vec();
                p[a][x] += h;
                energy(tri(&p), f)
            };
            let fd = (e(H) - e(-H)) / (2.0 * H);
            worst = worst.max((an.grad[(a, x)] - fd).abs());
        }
        eprintln!("H4 tri Gamma {f}: routed force vs FD {worst:.3e}");
        assert!(worst < FD_BAR, "{f}: {worst:e}");
        forces.push(an.grad.clone());
    }
    for i in 0..3 {
        for j in i + 1..3 {
            let d = diff(&forces[i], &forces[j]);
            eprintln!("H4 tri Gamma {} vs {} force: {d:.3e}", MGGA[i], MGGA[j]);
            assert!(d > MUTANT_BAR, "{} and {} coincide", MGGA[i], MGGA[j]);
        }
    }
}

/// Routed Gamma dE/deps vs FD of the Gamma energy under strain, SCAN /
/// r2SCAN / TPSS, three components, h = 1e-4 and 2e-4. MEASURED
/// (2026-10-22): 2.95e-8 (SCAN, r2SCAN) / 2.99e-8 (TPSS) at h = 1e-4,
/// 1.23e-7 / 1.24e-7 at h = 2e-4 (ratio 4.2: h^2). SCAN and r2SCAN coincide on
/// this one-orbital cell (the alpha = 0 identity of the force test).
#[test]
#[ignore = "validation: 3 functionals x 2 steps x 6 strained Gamma SCFs (36 SCFs); about 90 s on a quiet box"]
fn gamma_mgga_stress_fd_all_functionals_and_h_sweep() {
    for f in MGGA {
        let s = stress(&gamma(cell_of(&H2), f), None);
        let mut r = [0.0; 2];
        for (k, h) in [H, 2.0 * H].into_iter().enumerate() {
            let fd = fd_stress(f, h, &STRAIN_COMPS);
            r[k] = worst_stress(&s.de_deps, &fd, &STRAIN_COMPS);
            eprintln!("{f} h={h:.0e}: Gamma routed dE/deps vs FD {:.3e}", r[k]);
        }
        assert!(r[0] < STRESS_FD_BAR, "{f}: {:e}", r[0]);
        assert!(r[1] < 4.0 * STRESS_FD_BAR, "{f} h=2e-4: {:e}", r[1]);
    }
}

// ------------------------------------------------------------ k-point TPSS

fn kcfg(cell: &Cell, f: &str, grid: &KXcGrid) -> KRksConfig {
    let mut c = KRksConfig::new(cell, ExxDiv::None, f);
    c.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.krhf.scf.energy_conv = 1e-13;
    c.krhf.scf.grad_conv = 1e-10;
    c.krhf.scf.max_iter = 300;
    if let KXcGrid::Atom(g) = grid {
        c.grid = g.clone();
    }
    c
}

fn kscf(cell: &Cell, n: [usize; 3], f: &str, grid: &KXcGrid) -> KRksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let c = kcfg(cell, f, grid);
    match grid {
        KXcGrid::Atom(_) => solve_krks(cell, &prep, None, &mesh, &c),
        KXcGrid::Uniform(m) => {
            let g = PeriodicGrid::uniform(cell, *m).unwrap();
            solve_krks_on_grid(cell, &prep, None, &mesh, &g, &c)
        }
    }
    .unwrap_or_else(|e| panic!("solve_krks {f}: {e}"))
}

/// `(force, stress)` of a fresh k-point SCF on `cell`.
fn kderivs(cell: &Cell, n: [usize; 3], f: &str, grid: &KXcGrid) -> (KRksGradient, KRksStress) {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let res = kscf(cell, n, f, grid);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let eri =
        KDenseAftEri::build(cell, &prep, &mesh, &hk.s, ExxDiv::None, &Default::default()).unwrap();
    let g = kpoint_rks_gradient(
        cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KGradJk::Dense(&eri),
        &res.scf,
        f,
        &KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: grid.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    let s = kpoint_rks_stress(
        cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KStressJk::Dense(&eri),
        &res.scf,
        f,
        &KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            grid: grid.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    (g, s)
}

fn tri_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), TRI_A).expect("triclinic cell")
}

/// TPSS on the H4 triclinic cell (two occupied orbitals; TPSS is 1.4e-2 Ha
/// from SCAN here), 1x1x2: the force against FD of the SCF energy on the
/// atom grid (cutoff 14 Bohr: no image crosses it under the step) and the
/// stress against FD of the strained SCF energy on a uniform 24^3 grid
/// (smooth under strain). Also asserts the TPSS force is not the SCAN force.
/// MEASURED (2026-10-22): force 3.0e-9, the SCAN force misses the TPSS FD
/// by 2.9e-3; dE/deps 1.6e-8.
#[test]
#[ignore = "validation: H4 triclinic TPSS 1x1x2, 8 displaced + 4 strained SCFs and the SCAN control"]
fn tpss_triclinic_h4_force_and_stress_match_fd() {
    let mut gc = PeriodicGridConfig::with_size(20, 50);
    gc.neighbour_cutoff = Some(14.0);
    let atom = KXcGrid::Atom(gc);
    let n = [1, 1, 2];
    let (g, _) = kderivs(&tri_of(&TRI_ATOMS), n, "TPSS", &atom);
    let (gs, _) = kderivs(&tri_of(&TRI_ATOMS), n, "SCAN", &atom);
    let comps = [(0, 0), (1, 2), (2, 1), (3, 2)];
    let mut worst = 0.0_f64;
    let mut gap = 0.0_f64;
    for (a, x) in comps {
        let e = |h: f64| {
            let mut p = TRI_ATOMS.to_vec();
            p[a][x] += h;
            kscf(&tri_of(&p), n, "TPSS", &atom).scf.energy
        };
        let fd = (e(H) - e(-H)) / (2.0 * H);
        worst = worst.max((g.grad[(a, x)] - fd).abs());
        gap = gap.max((gs.grad[(a, x)] - fd).abs());
    }
    eprintln!("H4 tri TPSS 1x1x2: force vs FD {worst:.3e}; SCAN force vs the TPSS FD {gap:.3e}");
    assert!(worst < FD_BAR, "{worst:e}");
    assert!(gap > MUTANT_BAR, "{gap:e}");

    let uni = KXcGrid::Uniform([24; 3]);
    let cell = tri_of(&TRI_ATOMS);
    let (_, s) = kderivs(&cell, n, "TPSS", &uni);
    let mut ws = 0.0_f64;
    for (i, j) in [(0, 0), (2, 1)] {
        let e = |h: f64| kscf(&strained(&cell, i, j, h), n, "TPSS", &uni).scf.energy;
        let fd = (e(H) - e(-H)) / (2.0 * H);
        ws = ws.max((s.de_deps[i][j] - fd).abs());
    }
    eprintln!("H4 tri TPSS 1x1x2 uniform 24^3: dE/deps vs FD {ws:.3e}");
    assert!(ws < STRESS_FD_BAR, "{ws:e}");
}

fn supercell3(pos: &[[f64; 3]]) -> Cell {
    let mut atoms = Vec::new();
    for m in 0..3 {
        for p in pos {
            atoms.push([p[0], p[1], p[2] + m as f64 * 4.0]);
        }
    }
    let mut lat = cubic(4.0);
    lat[2][2] = 12.0;
    Cell::new(hydrogens(&atoms), lat).unwrap()
}

/// TPSS k-mesh == supercell: the 1x1x3 force on one copy of each atom and
/// the 1x1x3 dE/deps x 3 equal the explicit 3-cell supercell at 1x1x1 (same
/// code, same explicit cutoff). Checks the k sum and the Bloch phases of the
/// TPSS tau terms, not the tau derivative itself. MEASURED (2026-10-22):
/// force 1.2e-14, dE/deps 6.9e-13.
#[test]
#[ignore = "validation: TPSS 1x1x3 and the 3-cell supercell, one SCF each plus force and stress"]
fn tpss_k_mesh_is_the_supercell() {
    let mut gc = PeriodicGridConfig::with_size(20, 50);
    gc.neighbour_cutoff = Some(10.0);
    let grid = KXcGrid::Atom(gc);
    let (kg, ks) = kderivs(&cell_of(&H2), [1, 1, 3], "TPSS", &grid);
    let (sg, ss) = kderivs(&supercell3(&H2), [1, 1, 1], "TPSS", &grid);
    let first = sg.grad.slice(ndarray::s![0..2, ..]).to_owned();
    let df = diff(&kg.grad, &first);
    let mut k3 = ks.de_deps;
    for r in k3.iter_mut() {
        for v in r.iter_mut() {
            *v *= 3.0;
        }
    }
    let ds = mdiff(&k3, &ss.de_deps);
    eprintln!("TPSS 1x1x3 vs supercell: force {df:.2e}, dE/deps {ds:.2e}");
    assert!(df < 1e-9 && ds < 1e-9, "{df:e} {ds:e}");
}
