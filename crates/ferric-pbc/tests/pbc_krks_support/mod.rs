//! Shared setup of `pbc_krks.rs` (per commit) and `validation_pbc_krks.rs`
//! (weekly): every non-test item of the original file, made `pub`.
#![allow(dead_code, unused_imports)]

pub use crate::common::*;
pub use ferric_core::basis::BasisSet;
pub use ferric_core::parallel::ParallelContext;
pub use ferric_integrals::operator::Operator;
pub use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
pub use ferric_pbc::dft::{
    gamma_rks, GammaRksConfig, PeriodicGrid, PeriodicGridConfig, PeriodicXc, PeriodicXcConfig,
};
pub use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
pub use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
pub use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KPeriodicXc, KRksConfig, KRksResult};
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::uhf::GammaUhfIntegrals;
pub use ferric_scf::screening::SchwarzBounds;

pub const OMEGA: f64 = 0.8;

// PySCF 2.13 pbc.dft.KRKS, gamma-centred mesh, AFTDF mesh 61^3,
// exxdiv='ewald', UniformGrids(61^3), conv_tol 1e-12, STO-3G H2 a = 4
// (scratchpad `ref/krks.py`, 2026-10-09).
pub const PYSCF: [(&str, [usize; 3], f64); 9] = [
    ("LDA", [1, 1, 1], -1.521492168319),
    ("LDA", [1, 1, 2], -1.313021155290),
    ("LDA", [2, 2, 2], -1.054361178633),
    ("PBE", [1, 1, 1], -1.527136159909),
    ("PBE", [1, 1, 2], -1.325652288662),
    ("PBE", [2, 2, 2], -1.080236280438),
    ("PBE0", [1, 1, 1], -1.576992652863),
    ("PBE0", [1, 1, 2], -1.347642203115),
    ("PBE0", [2, 2, 2], -1.086899600638),
];

pub fn krks_cfg(cell: &Cell, functional: &str, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    cfg
}

pub fn grid_cfg(n_rad: usize, n_ang: usize) -> PeriodicGridConfig {
    PeriodicGridConfig::with_size(n_rad, n_ang)
}

pub fn krks_uniform(
    cell: &Cell,
    bs: &BasisSet,
    mesh: &KPointMesh,
    functional: &str,
    n: usize,
) -> KRksResult {
    let prep = prep_for(cell, bs);
    let grid = PeriodicGrid::uniform(cell, [n; 3]).unwrap();
    let cfg = krks_cfg(cell, functional, ExxDiv::Ewald);
    solve_krks_on_grid(cell, &prep, None, mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

// ----------------------------------------------------- k-mesh = supercell

/// Explicit diag(n) supercell and the cell-translation vectors of its copies
/// (atoms of cell m at `R + Σ m_i a_i`, m0 outer).
pub fn supercell(cell: &Cell, n: [usize; 3]) -> (Cell, Vec<[f64; 3]>) {
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
    (
        Cell::new(hydrogens(&atoms), lat).expect("supercell"),
        shifts,
    )
}

/// Gamma RKS energy of `sc` on an explicit grid (`gamma_rks` with the grid
/// supplied), dense AFT J/K.
pub fn gamma_rks_on_grid(sc: &Cell, bs: &BasisSet, grid: &PeriodicGrid, functional: &str) -> f64 {
    use ferric_pbc::ewald::madelung_constant;
    use ferric_scf::rhf::{solve_rhf_injected, PeriodicInjection};
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
    let pxc = PeriodicXc::new_closed_mgga(sc, bs, functional, grid, &PeriodicXcConfig::default())
        .unwrap();
    let vm = madelung_constant(sc).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j: Box::new(eri.j_builder()),
        k: Box::new(eri.k_builder_with_madelung(vm)),
        xc: Some(Box::new(pxc)),
    };
    let mut scf = GammaRksConfig::new(functional).scf;
    scf.density_conv = 1e-11;
    let r = solve_rhf_injected(
        &ParallelContext::default(),
        sc.mol(),
        &prep,
        op,
        &bounds,
        &scf,
        inj,
    )
    .unwrap();
    assert!(r.converged, "supercell {functional} did not converge");
    r.energy
}

pub fn supercell_anchor(name: &str, cell: &Cell, bs: &BasisSet, n: [usize; 3], functional: &str) {
    let nk = (n[0] * n[1] * n[2]) as f64;
    let prep = prep_for(cell, bs);
    let cell_grid = PeriodicGrid::build(cell, &grid_cfg(40, 110)).unwrap();
    let (sc, shifts) = supercell(cell, n);
    let sc_grid = cell_grid.replicated(&shifts);
    let e_sc = gamma_rks_on_grid(&sc, bs, &sc_grid, functional) / nk;
    let mesh = KPointMesh::gamma_centred(cell, n).unwrap();
    let cfg = krks_cfg(cell, functional, ExxDiv::Ewald);
    let r = solve_krks_on_grid(cell, &prep, None, &mesh, &cell_grid, &cfg).unwrap();
    eprintln!(
        "{name} {n:?} {functional}: E_k {:.12} E_super/N {e_sc:.12} dE {:.2e} ({} it)",
        r.scf.energy,
        r.scf.energy - e_sc,
        r.scf.iterations
    );
    assert!((r.scf.energy - e_sc).abs() < 1e-8, "{name} {functional}");
}

// ------------------------------------------------------------- meta-GGA

// PySCF 2.13 pbc.dft.KRKS, same setup as `PYSCF` above, xc = SCAN / R2SCAN
// (scratchpad `ref/kmgga.py`, 2026-10-09).
pub const PYSCF_MGGA: [(&str, [usize; 3], f64); 6] = [
    ("SCAN", [1, 1, 1], -1.557880049858),
    ("SCAN", [1, 1, 2], -1.343585556975),
    ("SCAN", [2, 2, 2], -1.085236512624),
    ("R2SCAN", [1, 1, 1], -1.557880049858),
    ("R2SCAN", [1, 1, 2], -1.343079341892),
    ("R2SCAN", [2, 2, 2], -1.084507483367),
];

// PySCF 2.13.1 pbc.dft.KRKS (AFTDF + UniformGrids 61^3, exxdiv ewald, conv_tol
// 1e-12), STO-3G, the 4-hydrogen triclinic cell (TRI_A / TRI_ATOMS): TWO
// occupied orbitals, so alpha != 0 and SCAN != r2SCAN already at 1x1x1
// (issue #355; the H2 cell above has one occupied orbital => alpha = 0 =>
// identical energies). Generated by `scripts/gen_pyscf_pbc_mgga_refs.py`.
pub const PYSCF_MGGA_TRI: [(&str, [usize; 3], f64); 4] = [
    ("SCAN", [1, 1, 1], -2.248625389935),
    ("R2SCAN", [1, 1, 1], -2.246390226285),
    ("SCAN", [1, 1, 2], -2.337536896331),
    ("R2SCAN", [1, 1, 2], -2.335205637508),
];

// PySCF 2.13.1 pbc.dft.RKS at the single Gamma point (NOT KRKS), same cell,
// basis, UniformGrids 61^3, AFTDF, ewald, conv_tol 1e-12; run_gamma() in
// `scripts/gen_pyscf_pbc_mgga_refs.py` (equals the KRKS 1x1x1 entries above).
pub const PYSCF_MGGA_TRI_GAMMA: [(&str, f64); 2] =
    [("SCAN", -2.248625389935), ("R2SCAN", -2.246390226285)];

// PySCF 2.13.1 pbc.dft.KRKS / pbc.dft.RKS, xc = 'TPSS' (libxc MGGA_X_TPSS +
// MGGA_C_TPSS), same cell, basis and settings as `PYSCF_MGGA_TRI`; generated by
// `scripts/gen_pyscf_pbc_tpss_refs.py`. The Gamma entry is the single-point
// pbc.dft.RKS.
pub const PYSCF_TPSS_TRI: [([usize; 3], f64); 2] =
    [([1, 1, 1], -2.262696973177), ([1, 1, 2], -2.348692030742)];

pub const PYSCF_TPSS_TRI_GAMMA: f64 = -2.262696973177;
