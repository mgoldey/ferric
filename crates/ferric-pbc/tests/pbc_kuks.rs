//! k-point open-shell KS-DFT (`ferric_pbc::kuks::solve_kuks`): LDA, GGA and
//! global hybrids on a k-mesh, built on the k-point UHF loop
//! (`run_kuhf`) with a spin-polarized Bloch-AO XC builder.
//!
//! What is independent of what:
//! * `closed_shell_limit_is_krks` — `solve_kuks` on a singlet against
//!   `solve_krks` (closed-shell code, different SCF loop and XC kernel
//!   entry: unpolarized `closed_kernel` vs `polarized_kernel`), same grid.
//! * `one_point_mesh_is_gamma_uks` — 1x1x1 mesh against `gamma_uks` (real
//!   lattice-summed AOs, `ferric_scf::uhf`), open shell, same grid.
//! * `k_mesh_is_the_gamma_supercell` — E/cell of an N-point mesh against the
//!   Gamma UKS energy of the explicit supercell / N on the replicated grid,
//!   open shell (giant-determinant multiplicity `(N_a - N_b) N_k + 1`).
//! * `uniform_grid_kuks_matches_pinned_pyscf` — PySCF 2.13 `pbc.dft.KUKS`
//!   with `UniformGrids` (AFTDF, mesh 61^3, exxdiv ewald, `mf.nelec` set to
//!   `(N_a N_k, N_b N_k)`): an external pin of the whole path.
//! * `mutants_are_detected` — the k-UHF mutants applied through the KS path
//!   move the closed-shell hybrid energy off `solve_krks`.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_uks, gamma_uks_with_xc, GammaUksConfig, PeriodicGrid, PeriodicGridConfig, PeriodicXc,
    PeriodicXcConfig,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kuks::{solve_kuks_on_grid, KUksConfig};
use ferric_pbc::kuscf::KUhfMutation;
use ferric_pbc::lattice::Cell;
use ferric_pbc::uhf::GammaUhfIntegrals;

const OMEGA: f64 = 0.8;

// PySCF 2.13 pbc.dft.KUKS, gamma-centred mesh, AFTDF mesh 61^3,
// exxdiv='ewald', UniformGrids(61^3), conv_tol 1e-12, STO-3G, H at
// (0.3, 0.2, 0.1) (+ (0.3, 0.2, 1.5)), a = 4 (scratchpad `ref/kuks.py`,
// 2026-10-09). (system, functional, mesh, E/cell)
const PYSCF: &[(&str, &str, [usize; 3], f64)] = &[
    ("H2t", "LDA", [1, 1, 1], -0.261234434634),
    ("H2t", "LDA", [1, 1, 2], -0.563618534277),
    ("H2t", "PBE", [1, 1, 1], -0.307145811810),
    ("H2t", "PBE", [1, 1, 2], -0.596440525795),
    ("H2t", "PBE0", [1, 1, 1], -0.331746374746),
    ("H2t", "PBE0", [1, 1, 2], -0.615439984635),
    ("Hd", "LDA", [1, 1, 1], -0.667511209738),
    ("Hd", "LDA", [1, 1, 2], -0.570539780438),
    ("Hd", "LDA", [1, 1, 3], -0.581403226992),
    ("Hd", "PBE", [1, 1, 1], -0.677170729041),
    ("Hd", "PBE", [1, 1, 2], -0.586191001233),
    ("Hd", "PBE", [1, 1, 3], -0.596095454697),
    ("Hd", "PBE0", [1, 1, 1], -0.699723106670),
    ("Hd", "PBE0", [1, 1, 2], -0.598023508587),
    ("Hd", "PBE0", [1, 1, 3], -0.605110288382),
];

fn cell_mult(atoms: &[[f64; 3]], a: f64, mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, cubic(a)).expect("cell")
}

fn kuks_cfg(cell: &Cell, functional: &str, exxdiv: ExxDiv) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, exxdiv, functional);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.kuhf.scf.energy_conv = 1e-12;
    cfg.kuhf.scf.grad_conv = 1e-9;
    cfg
}

fn grid_cfg(n_rad: usize, n_ang: usize) -> PeriodicGridConfig {
    PeriodicGridConfig::with_size(n_rad, n_ang)
}

fn kuks_on(
    cell: &Cell,
    bs: &BasisSet,
    mesh: &KPointMesh,
    grid: &PeriodicGrid,
    functional: &str,
) -> ferric_pbc::kuks::KUksResult {
    let prep = prep_for(cell, bs);
    let cfg = kuks_cfg(cell, functional, ExxDiv::Ewald);
    solve_kuks_on_grid(cell, &prep, None, mesh, grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_kuks {functional}: {e}"))
}

// -------------------------------------------------- closed-shell limit

#[test]
fn closed_shell_limit_is_krks() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
    for n in [[1, 1, 3], [2, 1, 1]] {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        for functional in ["LDA", "PBE", "PBE0"] {
            let mut rcfg = KRksConfig::new(&cell, ExxDiv::Ewald, functional);
            rcfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
            rcfg.krhf.scf.energy_conv = 1e-12;
            rcfg.krhf.scf.grad_conv = 1e-9;
            let r = solve_krks_on_grid(&cell, &prep, None, &mesh, &grid, &rcfg).unwrap();
            let u = kuks_on(&cell, &bs, &mesh, &grid, functional);
            eprintln!(
                "H2 {n:?} {functional}: E_u {:.12} E_r {:.12} dE {:.2e}; E_xc dE {:.2e}; <S2> {:.1e}",
                u.scf.energy,
                r.scf.energy,
                u.scf.energy - r.scf.energy,
                u.e_xc - r.e_xc,
                u.scf.s2
            );
            assert!((u.scf.energy - r.scf.energy).abs() < 1e-9, "{functional}");
            assert!((u.e_xc - r.e_xc).abs() < 1e-9, "{functional} E_xc");
            assert!(
                (u.electrons_on_grid - r.electrons_on_grid).abs() < 1e-9,
                "{functional} N_grid"
            );
            assert!(u.scf.s2.abs() < 1e-8);
        }
    }
}

#[test]
fn mutants_are_detected() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let grid = PeriodicGrid::build(&cell, &grid_cfg(40, 110)).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let rcfg = {
        let mut c = KRksConfig::new(&cell, ExxDiv::Ewald, "PBE0");
        c.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
        c.krhf.scf.energy_conv = 1e-12;
        c.krhf.scf.grad_conv = 1e-9;
        c
    };
    let r = solve_krks_on_grid(&cell, &prep, None, &mesh, &grid, &rcfg).unwrap();
    for m in [
        KUhfMutation::KFromTotalDensity,
        KUhfMutation::MadelungHalfPerSpin,
    ] {
        let mut cfg = kuks_cfg(&cell, "PBE0", ExxDiv::Ewald);
        cfg.kuhf.mutation = Some(m);
        let u = solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg).unwrap();
        let d = (u.scf.energy - r.scf.energy).abs();
        eprintln!("{m:?}: |dE| {d:.3e}");
        assert!(d > 1e-3, "{m:?} not detected");
    }
}

// ----------------------------------------------------- 1-point = Gamma

#[test]
fn one_point_mesh_is_gamma_uks() {
    let cases: Vec<(&str, Cell, BasisSet)> = vec![
        (
            "H doublet sp",
            cell_mult(&[[0.3, 0.2, 0.1]], 4.0, 2),
            sp_basis_h(),
        ),
        ("H2 triplet", cell_mult(&H2_ATOMS, 4.0, 3), pyscf_sto3g_h()),
    ];
    for (label, cell, bs) in cases {
        let prep = prep_for(&cell, &bs);
        let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
        let eri = DenseAftEri::build(
            &cell,
            &prep,
            &hc.s,
            ExxDiv::None,
            DEFAULT_DENSE_AFT_PRECISION,
            DEFAULT_DENSE_AFT_MAX_BYTES,
        )
        .unwrap();
        let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
        for functional in ["LDA", "PBE", "PBE0"] {
            let mut gcfg = GammaUksConfig::new(functional);
            gcfg.grid = grid_cfg(50, 110);
            gcfg.scf.density_conv = 1e-11;
            let g = gamma_uks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gcfg).unwrap();
            let grid = PeriodicGrid::build(&cell, &grid_cfg(50, 110)).unwrap();
            let k = kuks_on(&cell, &bs, &mesh, &grid, functional);
            eprintln!(
                "{label} {functional}: E_k {:.12} E_gamma {:.12} dE {:.2e}; E_xc dE {:.2e}; \
                 S2 {:.6} vs {:.6}",
                k.scf.energy,
                g.scf.energy,
                k.scf.energy - g.scf.energy,
                k.e_xc - g.e_xc,
                k.scf.s2,
                g.s2
            );
            assert_eq!(k.n_grid_points, g.grid.as_ref().unwrap().n_grid_points);
            assert!(
                (k.scf.energy - g.scf.energy).abs() < 1e-9,
                "{label} {functional}"
            );
            assert!((k.e_xc - g.e_xc).abs() < 1e-9, "{label} {functional} E_xc");
            assert!((k.scf.s2 - g.s2).abs() < 1e-8, "{label} {functional} S2");
        }
    }
}

// ----------------------------------------------------- k-mesh = supercell

fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> (Cell, Vec<[f64; 3]>) {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    let mut shifts = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                shifts.push([t[0], t[1], t[2]]);
                for p in cell.positions() {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= n[i] as f64;
        }
    }
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    (Cell::new(mol, lat).expect("supercell"), shifts)
}

fn gamma_uks_on_grid(sc: &Cell, bs: &BasisSet, grid: &PeriodicGrid, functional: &str) -> f64 {
    let prep = prep_for(sc, bs);
    let hc = periodic_hcore(sc, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = DenseAftEri::build(
        sc,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut pxc = PeriodicXc::new(sc, bs, functional, grid, &PeriodicXcConfig::default()).unwrap();
    let mut cfg = GammaUksConfig::new(functional);
    cfg.scf.density_conv = 1e-11;
    let r = gamma_uks_with_xc(
        sc,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        &mut pxc,
        &cfg,
    )
    .unwrap();
    assert!(r.scf.converged);
    r.scf.energy
}

fn supercell_anchor(
    name: &str,
    cell: &Cell,
    bs: &BasisSet,
    n: [usize; 3],
    functional: &str,
    na: usize,
    nb: usize,
) {
    let nk = n[0] * n[1] * n[2];
    let prep = prep_for(cell, bs);
    let cell_grid = PeriodicGrid::build(cell, &grid_cfg(40, 110)).unwrap();
    let mult = (na - nb) * nk + 1;
    let (sc, shifts) = supercell(cell, n, mult);
    let sc_grid = cell_grid.replicated(&shifts);
    let e_sc = gamma_uks_on_grid(&sc, bs, &sc_grid, functional) / nk as f64;
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let cfg = kuks_cfg(cell, functional, ExxDiv::Ewald);
    let r = solve_kuks_on_grid(cell, &prep, None, &mesh, &cell_grid, &cfg).unwrap();
    eprintln!(
        "{name} {n:?} {functional}: E_k {:.12} E_super/N {e_sc:.12} dE {:.2e} ({} it)",
        r.scf.energy,
        r.scf.energy - e_sc,
        r.scf.iterations
    );
    assert!((r.scf.energy - e_sc).abs() < 1e-8, "{name} {functional}");
}

#[test]
fn k_mesh_is_the_gamma_supercell() {
    let h2t = cell_mult(&H2_ATOMS, 4.0, 3);
    let bs = pyscf_sto3g_h();
    for functional in ["LDA", "PBE", "PBE0"] {
        supercell_anchor("H2 triplet", &h2t, &bs, [1, 1, 3], functional, 2, 0);
    }
    let hd = cell_mult(&[[0.3, 0.2, 0.1]], 4.0, 2);
    let sp = sp_basis_h();
    for functional in ["PBE", "PBE0"] {
        supercell_anchor("H doublet sp", &hd, &sp, [1, 1, 3], functional, 1, 0);
    }
    // Both spins populated (H2 cell, alpha 2 / beta 1 impossible for H2;
    // use a 3-H cell: doublet, N_a = 2, N_b = 1).
    let h3 = cell_mult(&[[0.3, 0.2, 0.1], [0.3, 0.2, 1.5], [0.3, 1.9, 0.4]], 4.0, 2);
    supercell_anchor("H3 doublet", &h3, &bs, [1, 1, 2], "PBE0", 2, 1);
}

// ------------------------------------------------------------- PySCF pins

#[test]
fn uniform_grid_kuks_matches_pinned_pyscf() {
    let bs = pyscf_sto3g_h();
    for &(sys, functional, n, e_ref) in PYSCF {
        let cell = match sys {
            "H2t" => cell_mult(&H2_ATOMS, 4.0, 3),
            "Hd" => cell_mult(&[[0.3, 0.2, 0.1]], 4.0, 2),
            _ => unreachable!(),
        };
        // 61^3: the 40^3 grid is under-converged for the PBE triplet (1.2e-5).
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let grid = PeriodicGrid::uniform(&cell, [61; 3]).unwrap();
        let r = kuks_on(&cell, &bs, &mesh, &grid, functional);
        eprintln!(
            "{sys} {functional} {n:?}: {:.12} PySCF {e_ref:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-8,
            "{sys} {functional} {n:?}: {} vs {e_ref}",
            r.scf.energy
        );
    }
}

/// The open-shell solver has no tau term, so it must refuse meta-GGA on both
/// entry points even where the closed-shell XC accepts it.
#[test]
fn kuks_refuses_meta_gga() {
    let cell = cell_mult(&[[0.0, 0.0, 0.0]], 4.0, 2);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 2]).unwrap();
    let cfg = kuks_cfg(&cell, "SCAN", ExxDiv::None);
    let grid = PeriodicGrid::build(&cell, &cfg.grid).unwrap();
    assert!(
        solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg).is_err(),
        "solve_kuks_on_grid accepted SCAN"
    );
    assert!(
        ferric_pbc::kuks::solve_kuks(&cell, &prep, None, &mesh, &cfg).is_err(),
        "solve_kuks accepted SCAN"
    );
}
