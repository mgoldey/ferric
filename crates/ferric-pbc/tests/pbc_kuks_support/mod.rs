//! Shared setup of `pbc_kuks.rs` (per commit) and `validation_pbc_kuks.rs`
//! (weekly): every non-test item of the original file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::BasisSet;
pub use ferric_core::mol::Molecule;
pub use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
pub use ferric_pbc::dft::{
    gamma_uks, gamma_uks_with_xc, GammaUksConfig, PeriodicGrid, PeriodicGridConfig, PeriodicXc,
    PeriodicXcConfig,
};
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
pub use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kuks::{solve_kuks_on_grid, KUksConfig};
pub use ferric_pbc::kuscf::KUhfMutation;
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::uhf::GammaUhfIntegrals;

pub const OMEGA: f64 = 0.8;

// PySCF 2.13 pbc.dft.KUKS, gamma-centred mesh, AFTDF mesh 61^3,
// exxdiv='ewald', UniformGrids(61^3), conv_tol 1e-12, STO-3G, H at
// (0.3, 0.2, 0.1) (+ (0.3, 0.2, 1.5)), a = 4 (scratchpad `ref/kuks.py`,
// 2026-10-09). (system, functional, mesh, E/cell)
pub const PYSCF: &[(&str, &str, [usize; 3], f64)] = &[
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

pub fn cell_mult(atoms: &[[f64; 3]], a: f64, mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, cubic(a)).expect("cell")
}

pub fn kuks_cfg(cell: &Cell, functional: &str, exxdiv: ExxDiv) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, exxdiv, functional);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.kuhf.scf.energy_conv = 1e-12;
    cfg.kuhf.scf.grad_conv = 1e-9;
    cfg
}

pub fn grid_cfg(n_rad: usize, n_ang: usize) -> PeriodicGridConfig {
    PeriodicGridConfig::with_size(n_rad, n_ang)
}

pub fn kuks_on(
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

// ----------------------------------------------------- k-mesh = supercell

pub fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> (Cell, Vec<[f64; 3]>) {
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

pub fn gamma_uks_on_grid(sc: &Cell, bs: &BasisSet, grid: &PeriodicGrid, functional: &str) -> f64 {
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

pub fn supercell_anchor(
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

/// Every `(system, functional, mesh, E/cell)` of `pins` against the PySCF
/// KUKS pin (61^3 uniform grid, 1e-8).
pub fn assert_kuks_pins(pins: &[(&str, &str, [usize; 3], f64)]) {
    let bs = pyscf_sto3g_h();
    for &(sys, functional, n, e_ref) in pins {
        let cell = match sys {
            "H2t" => cell_mult(&H2_ATOMS, 4.0, 3),
            "Hd" => cell_mult(&[[0.3, 0.2, 0.1]], 4.0, 2),
            _ => unreachable!(),
        };
        // 61^3: the 40^3 grid is under-converged for the PBE triplet (1.2e-5).
        // Issue #356 re-checked 31^3 (measured 2026-10-22): ferric at 31^3
        // misses these 61^3 pins by 3.8e-8 (H2t PBE 1x1x1; LDA <= 9e-10), over
        // the 1e-8 bar, and a same-grid PySCF 31^3 KUKS LDA re-pin converged to
        // a DIFFERENT SCF state (-0.2078 vs -0.2612 Ha), so no 31^3 pin exists.
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
