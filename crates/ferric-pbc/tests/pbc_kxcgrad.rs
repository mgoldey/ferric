//! Analytic forces of the k-point closed-shell RKS energy for pure LDA/GGA
//! functionals (`ferric_pbc::kxcgrad`; derivation in its module doc).
//!
//! Artifact hypotheses next to physics hypotheses are written at each test.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KRsGdfGradSource};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::KJkKind;
use ferric_pbc::kxcgrad::{
    kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid, KXcMutation,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::PeriodicGrid;
use ndarray::Array2;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Jk {
    Dense,
    Gdf,
}

/// The prototype's ET-sp aux on H (as `pbc_kgrad.rs`).
fn et_sp_aux() -> BasisSet {
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

fn kgdf_cfg() -> KRsGdfConfig {
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

fn atom_grid(nr: usize, na: usize) -> KXcGrid {
    KXcGrid::Atom(PeriodicGridConfig::with_size(nr, na))
}

fn krks_cfg(cell: &Cell, functional: &str, jk: Jk, grid: &KXcGrid) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::None, functional);
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

/// Fresh SCF (energy) at `pos`.
fn energy(pos: &[[f64; 3]], mesh_n: [usize; 3], functional: &str, jk: Jk, grid: &KXcGrid) -> f64 {
    scf(&cell_of(pos), mesh_n, functional, jk, grid).scf.energy
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn scf(cell: &Cell, mesh_n: [usize; 3], functional: &str, jk: Jk, grid: &KXcGrid) -> KRksResult {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(cell, &bs);
    let mesh = KPointMesh::gamma_centred(cell, mesh_n).unwrap();
    let cfg = krks_cfg(cell, functional, jk, grid);
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

/// Everything the force needs, built the way `solve_krks` built it.
struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KRksResult,
    jk: Jk,
    grid: KXcGrid,
    functional: String,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
}

fn fixture(pos: &[[f64; 3]], mesh_n: [usize; 3], functional: &str, jk: Jk, grid: KXcGrid) -> Fx {
    let cell = cell_of(pos);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = scf(&cell, mesh_n, functional, jk, &grid);
    let cfg = krks_cfg(&cell, functional, jk, &grid);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let aux = prep_for(&cell, &et_sp_aux());
    Fx {
        cell,
        prep,
        mesh,
        hk,
        res,
        jk,
        grid,
        functional: functional.to_string(),
        aux,
    }
}

fn gcfg(grid: &KXcGrid, mutation: Option<KXcMutation>) -> KRksGradConfig {
    KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        grid: grid.clone(),
        mutation,
        ..Default::default()
    }
}

fn force(fx: &Fx, mutation: Option<KXcMutation>) -> KRksGradient {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let cfg = gcfg(&fx.grid, mutation);
    match fx.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &fx.hk.s,
                ExxDiv::None,
                &Default::default(),
            )
            .unwrap();
            kpoint_rks_gradient(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KGradJk::Dense(&eri),
                &fx.res.scf,
                &fx.functional,
                &cfg,
            )
            .unwrap()
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(ExxDiv::None);
            kpoint_rks_gradient(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KGradJk::RsGdf(KRsGdfGradSource {
                    gdf: &gdf,
                    cfg: &kc,
                    aux: &fx.aux,
                }),
                &fx.res.scf,
                &fx.functional,
                &cfg,
            )
            .unwrap()
        }
    }
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn fd(
    pos: &[[f64; 3]],
    a: usize,
    x: usize,
    h: f64,
    mesh_n: [usize; 3],
    f: &str,
    jk: Jk,
    grid: &KXcGrid,
) -> f64 {
    (energy(&moved(pos, a, x, h), mesh_n, f, jk, grid)
        - energy(&moved(pos, a, x, -h), mesh_n, f, jk, grid))
        / (2.0 * h)
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

#[test]
#[ignore]
fn probe_force_vs_fd() {
    for (f, grid) in [
        ("LDA", atom_grid(20, 50)),
        ("PBE", atom_grid(20, 50)),
        ("PBE", KXcGrid::Uniform([24, 24, 24])),
    ] {
        let t = std::time::Instant::now();
        let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
        let g = force(&fx, None);
        eprintln!(
            "{f} {:?}: fixture+force {:?}; e_xc {:.8} ne {:.8} fock_mismatch {:.2e}",
            matches!(grid, KXcGrid::Uniform(_)),
            t.elapsed(),
            g.e_xc,
            g.electrons_on_grid,
            g.hf.fock_mismatch
        );
        eprintln!(
            "  parts |ao| {:.3e} |point| {:.3e} |weight| {:.3e} net {:.2e}",
            max_abs(&g.xc_ao),
            max_abs(&g.xc_point),
            max_abs(&g.xc_weight),
            (0..3).map(|c| g.grad.column(c).sum().abs()).fold(0.0, f64::max)
        );
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 0), (0, 2), (1, 1), (1, 2)] {
            let d = fd(&H2, a, x, 1e-4, [1, 1, 3], f, Jk::Dense, &grid);
            eprintln!("  ({a},{x}) analytic {:.10} fd {:.10} diff {:.2e}", g.grad[(a, x)], d, g.grad[(a, x)] - d);
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("  worst {worst:.2e}");
    }
}
