//! Shared fixtures and helpers of `pbc_kxcstress_mgga.rs` (per commit) and
//! `validation_pbc_kxcstress_mgga.rs` (weekly): the H2 / STO-3G k-RKS
//! meta-GGA fixture, the stress drivers, the fixed-density XC FD, the PySCF
//! stress pin case and the RS-GDF SCF-energy FD case.
#![allow(dead_code, unused_imports)]

use crate::common::*;
pub use ferric_core::basis::{BasisSet, Shell};
pub use ferric_pbc::dense_aft::ExxDiv;
pub use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig};
pub use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use ferric_pbc::hcore::PeriodicHcoreConfig;
pub use ferric_pbc::kdense_aft::KDenseAftEri;
pub use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KPeriodicXc, KRksConfig, KRksResult};
pub use ferric_pbc::kgrad::KRsGdfGradSource;
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kscf::{KJkKind, KPointXc, KScfResult};
pub use ferric_pbc::kstress::KStressConfig;
pub use ferric_pbc::kstress::KStressJk;
pub use ferric_pbc::kxcgrad::KXcGrid;
pub use ferric_pbc::kxcstress::{
    kpoint_rks_stress, KRksStress, KRksStressConfig, KXcStressMutation,
};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
pub use ferric_pbc::rsgdf::RsGdfConfig;
pub use ferric_pbc::stress::Mat3;
pub use ndarray::Array2;
pub use num_complex::Complex64;
pub use std::collections::HashMap;

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const ALL9: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];
/// Neighbour cutoff shared by every atom-grid comparison that rebuilds a grid
/// (the default depends on the cell).
pub const CUTOFF: f64 = 10.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Jk {
    Dense,
    Gdf,
}

/// The prototype's ET-sp aux on H (as `pbc_kxcgrad.rs`).
pub fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for a in [4.7, 1.9, 0.75, 0.3] {
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    for a in [1.25, 0.5] {
        shells.push(Shell {
            l: 1,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

pub fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

pub fn atom_grid(nr: usize, na: usize, cutoff: Option<f64>) -> KXcGrid {
    let mut g = PeriodicGridConfig::with_size(nr, na);
    g.neighbour_cutoff = cutoff;
    KXcGrid::Atom(g)
}

pub fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

pub fn krks_cfg(
    cell: &Cell,
    functional: &str,
    jk: Jk,
    grid: &KXcGrid,
    exxdiv: ExxDiv,
) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 200;
    if jk == Jk::Gdf {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = kgdf_cfg();
    }
    if let KXcGrid::Atom(g) = grid {
        cfg.grid = g.clone();
    }
    cfg
}

/// Fresh SCF (energy) on `cell`.
pub fn scf_on(
    cell: &Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: &KXcGrid,
    exxdiv: ExxDiv,
) -> KRksResult {
    scf_bs(cell, mesh_n, functional, jk, grid, exxdiv, &pyscf_sto3g_h())
}

pub fn scf_bs(
    cell: &Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: &KXcGrid,
    exxdiv: ExxDiv,
    bs: &BasisSet,
) -> KRksResult {
    let prep = prep_for(cell, bs);
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let cfg = krks_cfg(cell, functional, jk, grid, exxdiv);
    let aux_bs = et_sp_aux();
    let aux = (jk == Jk::Gdf).then(|| prep_for(cell, &aux_bs));
    match grid {
        KXcGrid::Atom(_) => solve_krks(cell, &prep, aux.as_ref(), &mesh, &cfg),
        KXcGrid::Uniform(n) => {
            let g = PeriodicGrid::uniform(cell, *n).unwrap();
            solve_krks_on_grid(cell, &prep, aux.as_ref(), &mesh, &g, &cfg)
        }
    }
    .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

pub fn energy_on(cell: &Cell, mesh_n: [usize; 3], f: &str, jk: Jk, grid: &KXcGrid) -> f64 {
    scf_on(cell, mesh_n, f, jk, grid, ExxDiv::None).scf.energy
}

pub struct Fx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub aux: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub res: KRksResult,
    pub jk: Jk,
    pub grid: KXcGrid,
    pub functional: String,
    pub exxdiv: ExxDiv,
}

pub fn fixture_cell(cell: Cell, mesh_n: [usize; 3], functional: &str, jk: Jk, grid: KXcGrid) -> Fx {
    fixture_exx(cell, mesh_n, functional, jk, grid, ExxDiv::None)
}

pub fn fixture_exx(
    cell: Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: KXcGrid,
    exxdiv: ExxDiv,
) -> Fx {
    fixture_bs(cell, mesh_n, functional, jk, grid, exxdiv, &pyscf_sto3g_h())
}

pub fn fixture_bs(
    cell: Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: KXcGrid,
    exxdiv: ExxDiv,
    bs: &BasisSet,
) -> Fx {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = scf_bs(&cell, mesh_n, functional, jk, &grid, exxdiv, bs);
    let cfg = krks_cfg(&cell, functional, jk, &grid, exxdiv);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let aux = prep_for(&cell, &et_sp_aux());
    Fx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        res,
        jk,
        grid,
        functional: functional.to_string(),
        exxdiv,
    }
}

pub fn fixture(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: KXcGrid,
) -> Fx {
    fixture_cell(cell_of(pos), mesh_n, functional, jk, grid)
}

pub fn scfg(grid: &KXcGrid, mutation: Option<KXcStressMutation>) -> KRksStressConfig {
    KRksStressConfig {
        stress: KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: grid.clone(),
        mutation,
        ..Default::default()
    }
}

pub fn try_stress_cfg(
    fx: &Fx,
    cfg: &KRksStressConfig,
) -> Result<KRksStress, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    match fx.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &fx.hk.s,
                fx.exxdiv,
                &Default::default(),
            )
            .unwrap();
            kpoint_rks_stress(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KStressJk::Dense(&eri),
                &fx.res.scf,
                &fx.functional,
                cfg,
            )
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(fx.exxdiv);
            kpoint_rks_stress(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KStressJk::RsGdf(KRsGdfGradSource {
                    gdf: &gdf,
                    cfg: &kc,
                    aux: &fx.aux,
                }),
                &fx.res.scf,
                &fx.functional,
                cfg,
            )
        }
    }
}

pub fn stress_cfg(fx: &Fx, cfg: &KRksStressConfig) -> KRksStress {
    try_stress_cfg(fx, cfg).expect("kpoint_rks_stress")
}

pub fn stress(fx: &Fx, mutation: Option<KXcStressMutation>) -> KRksStress {
    stress_cfg(fx, &scfg(&fx.grid, mutation))
}

// ------------------------------------------------------------ small helpers

pub fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

pub fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn mscale(a: &Mat3, f: f64) -> Mat3 {
    let mut o = *a;
    for r in o.iter_mut() {
        for x in r.iter_mut() {
            *x *= f;
        }
    }
    o
}

pub fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

pub fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

// ----------------------------------------------------------- fixed-D XC FD

/// `E_xc[D]` at FIXED Bloch densities `d` on `cell` (a strained cell keeps the
/// fractional k mesh and the AO coefficients; the grid is rebuilt on the cell).
pub fn exc_fixed_d_on(
    cell: &Cell,
    bs: &BasisSet,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
) -> f64 {
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let pg = match grid {
        KXcGrid::Atom(cfg) => PeriodicGrid::build(cell, cfg).unwrap(),
        KXcGrid::Uniform(n) => PeriodicGrid::uniform(cell, *n).unwrap(),
    };
    let mut xc = KPeriodicXc::new(cell, bs, f, &pg, &mesh, &PeriodicXcConfig::default()).unwrap();
    xc.build(d).unwrap().0
}

/// Central FD of `E_xc[D_fixed]` under strain: all nine components.
pub fn fd_xc_all9(
    cell: &Cell,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
    h: f64,
) -> Mat3 {
    fd_xc_all9_bs(cell, &pyscf_sto3g_h(), mesh_n, f, grid, d, h)
}

pub fn fd_xc_all9_bs(
    cell: &Cell,
    bs: &BasisSet,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
    h: f64,
) -> Mat3 {
    fd_xc_comps(cell, bs, mesh_n, f, grid, d, h, &ALL9)
}

/// Central FD of `E_xc[D_fixed]` on the listed strain components only (the
/// others are left 0).
#[allow(clippy::too_many_arguments)]
pub fn fd_xc_comps(
    cell: &Cell,
    bs: &BasisSet,
    mesh_n: [usize; 3],
    f: &str,
    grid: &KXcGrid,
    d: &[Array2<Complex64>],
    h: f64,
    comps: &[(usize, usize)],
) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for &(i, j) in comps {
        let ep = exc_fixed_d_on(&strained(cell, i, j, h), bs, mesh_n, f, grid, d);
        let em = exc_fixed_d_on(&strained(cell, i, j, -h), bs, mesh_n, f, grid, d);
        out[i][j] = (ep - em) / (2.0 * h);
    }
    out
}

pub fn xc_part(s: &KRksStress) -> Mat3 {
    let mut o = s.xc_ao;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] += s.xc_weight[i][j];
        }
    }
    o
}

/// FD step of the atom-grid fixed-D FD (as the GGA anchor's `FD_H_ATOM`).
pub const FD_H_XC: f64 = 2.5e-5;
pub const FD_BAR_XC: f64 = 3e-8;
pub const MUTANT_BAR: f64 = 1e-5;
/// The defects of the XC strain part (the Pulay-only `TauNoVxc` is not in the
/// fixed-density XC FD).
pub const XC_MUTANTS: [KXcStressMutation; 7] = [
    KXcStressMutation::NoTauStrain,
    KXcStressMutation::TauDoubled,
    KXcStressMutation::WrongFactor,
    KXcStressMutation::NoWeightDeriv,
    KXcStressMutation::NoPointMotion,
    KXcStressMutation::NoHessian,
    KXcStressMutation::NoStrainMoment,
];

// ---------------------------------------------------------------- supercell

pub fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3]) -> Cell {
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                for p in pos {
                    atoms.push([
                        p[0] + m0 as f64 * a,
                        p[1] + m1 as f64 * a,
                        p[2] + m2 as f64 * a,
                    ]);
                }
            }
        }
    }
    let lat = [
        [a * n[0] as f64, 0.0, 0.0],
        [0.0, a * n[1] as f64, 0.0],
        [0.0, 0.0, a * n[2] as f64],
    ];
    Cell::new(hydrogens(&atoms), lat).unwrap()
}

/// Both sides, measured (1x1x2, uniform 24^3, h = 1e-4): correct 3.4e-8;
/// `TauTranspose` 6.0e-4, `NoHessian` 6.6e-2, `NoTauStrain` 1.4e-1. The bar is
/// 3x above the floor and 6000x below the smallest defect.
pub const P_SHELL_BAR: f64 = 1e-7;

// ----------------------------------------------------------- PySCF anchor

/// PySCF 2.13.1 `pbc.grad.krks_stress` for SCAN / r2SCAN (FFTDF on
/// `UniformGrids(61^3)`, exxdiv ewald, conv_tol 1e-12, H2/STO-3G in the a = 4
/// cube, 1x1x2 gamma-centred mesh; the settings of the GGA anchor in
/// `pbc_kxcstress.rs`), written by `scripts/gen_pyscf_kks_stress_refs.py` to
/// `testdata/reference/pbc_h2_sto3g_kks_stress_mgga.json`: the full 3x3
/// `sigma = (1/Omega) dE/d eps` (Hartree/Bohr^3). PySCF's off-diagonals are
/// zero to 2.6e-11 (its own noise). Bar 1e-9: the GGA anchor's PySCF
/// analytic-vs-FD agreement is 2.3e-10 and the observed ferric agreement is
/// 7e-11 (printed); the mutant side is asserted against the same reference.
pub const PYSCF_MGGA_BAR: f64 = 1e-9;

pub fn pyscf_mgga_case(f: &str, key: &str) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/pbc_h2_sto3g_kks_stress_mgga.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("reference json")).unwrap();
    assert_eq!(doc["pyscf_version"], "2.13.1");
    let rows = &doc["rows"][key]["sigma"];
    let mut want = [[0.0; 3]; 3];
    for (i, row) in want.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = rows[i][j].as_f64().expect("sigma entry");
        }
    }
    let fx = fixture_exx(
        cell_of(&H2_ATOMS),
        [1, 1, 2],
        f,
        Jk::Dense,
        KXcGrid::Uniform([61, 61, 61]),
        ExxDiv::Ewald,
    );
    let s = stress(&fx, None);
    let d = mdiff(&s.sigma, &want);
    eprintln!(
        "{f} 61^3 1x1x2 vs PySCF: |sigma - sigma_pyscf| {d:.2e} (xx {:.9e} vs {:.9e}, zz {:.9e} vs {:.9e})",
        s.sigma[0][0], want[0][0], s.sigma[2][2], want[2][2]
    );
    assert!(d < PYSCF_MGGA_BAR, "{f}: {d:e}");
    // Both sides: the tau defect misses the same reference.
    let m = stress(&fx, Some(KXcStressMutation::NoTauStrain));
    let dm = mdiff(&m.sigma, &want);
    eprintln!("{f} NoTauStrain vs PySCF: {dm:.2e}");
    assert!(dm > 1e4 * PYSCF_MGGA_BAR, "{f}: mutant {dm:e}");
}

/// FD step and bar of the SCF-energy FD on the uniform grid (the dense
/// `meta_gga_stress_matches_fd_of_the_scf_energy` values).
pub const SCF_FD_H: f64 = 1e-4;
pub const SCF_FD_BAR: f64 = 5e-7;

pub fn rsgdf_scf_fd_case(f: &str, comps: &[(usize, usize)], mutants: bool) {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Gdf, grid.clone());
    let s = stress(&fx, None);
    let mut fds = Vec::new();
    let mut worst = 0.0_f64;
    for &(i, j) in comps {
        let ep = energy_on(
            &strained(&fx.cell, i, j, SCF_FD_H),
            [1, 1, 3],
            f,
            Jk::Gdf,
            &grid,
        );
        let em = energy_on(
            &strained(&fx.cell, i, j, -SCF_FD_H),
            [1, 1, 3],
            f,
            Jk::Gdf,
            &grid,
        );
        let fd = (ep - em) / (2.0 * SCF_FD_H);
        let miss = (s.de_deps[i][j] - fd).abs();
        eprintln!(
            "{f} RS-GDF uniform 24^3 eps({i},{j}): analytic {:.9e} FD {fd:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst = worst.max(miss);
        fds.push(((i, j), fd));
    }
    eprintln!("{f} RS-GDF: |analytic - FD of SCF energy| {worst:.2e}");
    assert!(worst < SCF_FD_BAR, "{f}: {worst:e}");
    if !mutants {
        return;
    }
    for m in [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
    ] {
        let t = stress(&fx, Some(m));
        let mm = fds.iter().fold(0.0_f64, |w, &((i, j), fd)| {
            w.max((t.de_deps[i][j] - fd).abs())
        });
        eprintln!("  {f} RS-GDF {m:?}: FD miss {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{f} RS-GDF {m:?} survives the FD: {mm:e}");
    }
}

/// Measured (H2/STO-3G, 1x1x3, atom grid (20, 50), et-sp aux, omega = 1):
/// route difference max |RS-GDF - dense| 7.08e-5 (PBE) / 7.10e-5 (SCAN),
/// |SCAN - PBE| 3.0e-7; RS-GDF-only mutants NoTauStrain 5.2e-2, TauNoVxc
/// 2.4e-2. Bar 10x the measured gap, 8000x below the smallest mutant.
pub const TRANSFER_BAR: f64 = 3e-6;

/// `max |a - b|` over the listed components.
pub fn mdiff_on(a: &Mat3, b: &Mat3, comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .map(|&(i, j)| (a[i][j] - b[i][j]).abs())
        .fold(0.0, f64::max)
}

/// Fixed-density XC stress against central FD of `E_xc[D]` on the uniform
/// 24^3 grid (no partition, no home atom), over `comps`, for each functional
/// in `fs`; every XC strain defect must miss the same FD. The weekly test runs
/// SCAN and r2SCAN on all nine components; the per-commit canary SCAN on two.
pub fn uniform_fixed_density_xc_case(fs: &[&str], comps: &[(usize, usize)]) {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    for &f in fs {
        let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
        let s = stress(&fx, None);
        let d0 = fx.res.scf.densities.clone();
        let fd = fd_xc_comps(
            &fx.cell,
            &pyscf_sto3g_h(),
            [1, 1, 3],
            f,
            &grid,
            &d0,
            1e-4,
            comps,
        );
        let miss = mdiff_on(&xc_part(&s), &fd, comps);
        eprintln!("{f} uniform 24^3 {comps:?}: |xc - FD| {miss:.2e}");
        assert!(miss < FD_BAR_XC, "{f}: {miss:e}");
        for m in XC_MUTANTS {
            let t = stress(&fx, Some(m));
            let mm = mdiff_on(&xc_part(&t), &fd, comps);
            eprintln!("  {f} uniform {m:?}: |xc - FD| {mm:.2e}");
            // A uniform grid has no partition/home atom: `NoPointMotion` is
            // still a real defect there (points move with the lattice), and
            // `PointSelf` is not in the list (an identity on this grid).
            assert!(mm > MUTANT_BAR, "{f} uniform {m:?} survives: {mm:e}");
        }
    }
}
