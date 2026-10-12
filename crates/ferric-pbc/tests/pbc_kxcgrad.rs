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
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::{gamma_rks, GammaRksConfig};
use ferric_pbc::grad::{gamma_rks_gradient_with, GammaGradConfig};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointJk, KPointXc, KScfResult,
};
use ferric_pbc::kxcgrad::{kpoint_hartree_gradient, kpoint_null_xc_gradient};
use ferric_pbc::uhf::GammaUhfIntegrals;
use num_complex::Complex64;
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

// ------------------------------------------------------------------ anchors

/// Fixed (tight) precision of the dense kernels for the Gamma-side anchors.
const ANCHOR_PRECISION: f64 = 1e-12;

fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Zero XC: the exactness-anchor builder (E_xc = 0, V_xc = 0).
struct ZeroXc(usize);
impl KPointXc for ZeroXc {
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>), ferric_core::FerricError> {
        Ok((0.0, dm.iter().map(|d| Array2::zeros((self.0, self.0))).collect()))
    }
}

struct Hartree {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    scf: KScfResult,
    jk: Jk,
}

/// Hartree-only k SCF (F = h + J; exchange fraction 0, no XC).
fn hartree(pos: &[[f64; 3]], mesh_n: [usize; 3], jk: Jk) -> Hartree {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let mut scfcfg = ferric_pbc::kscf::KScfConfig::default();
    scfcfg.energy_conv = 1e-13;
    scfcfg.grad_conv = 1e-10;
    scfcfg.max_iter = 200;
    let eri;
    let gdf;
    let jkb: Box<dyn KPointJk + '_> = match jk {
        Jk::Dense => {
            eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &dense_cfg())
                .unwrap();
            Box::new(eri.jk_builder())
        }
        Jk::Gdf => {
            gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kgdf_cfg()).unwrap();
            Box::new(gdf.jk_builder())
        }
    };
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: jkb,
    };
    let mut z = ZeroXc(prep.nbasis());
    let scf = solve_krks_injected(&cell, &mesh, &scfcfg, inj, &mut z, 0.0).unwrap();
    assert!(scf.converged);
    Hartree {
        cell,
        prep,
        aux,
        mesh,
        hk,
        scf,
        jk,
    }
}

fn with_jk<R>(h: &Hartree, f: impl FnOnce(KGradJk<'_>) -> R) -> R {
    match h.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(&h.cell, &h.prep, &h.mesh, &h.hk.s, ExxDiv::None, &dense_cfg())
                .unwrap();
            f(KGradJk::Dense(&eri))
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&h.cell, &h.prep, &h.aux, &h.mesh, &h.hk.s, &kc).unwrap();
            f(KGradJk::RsGdf(KRsGdfGradSource {
                gdf: &gdf,
                cfg: &kc,
                aux: &h.aux,
            }))
        }
    }
}

fn kcfg() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// (a) NULL-XC EXACTNESS ANCHOR. With E_xc = 0, V_xc = 0 and exchange fraction
/// 0 the KS entry is the exchange-free HF machinery plus zero XC terms: it must
/// equal `kpoint_hartree_gradient` (same `assemble`, none of the XC plumbing or
/// KS guard) EXACTLY, part by part, dense and RS-GDF. Artifact hypothesis: an
/// XC leak (a stray add of a zero-initialised array with the wrong shape, a
/// guard that mutates F) would show as a nonzero difference; physics predicts
/// exactly 0.
#[test]
fn null_xc_equals_the_exchange_free_machinery_exactly() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let h = hartree(&H2, [1, 1, 3], jk);
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let (a, b) = with_jk(&h, |j| {
            let a = kpoint_null_xc_gradient(
                &h.cell, &h.prep, &h.mesh, &hcfg, &h.hk, j, &h.scf, &gcfg(&KXcGrid::default(), None),
            )
            .unwrap();
            (a, ())
        });
        let g = with_jk(&h, |j| {
            kpoint_hartree_gradient(&h.cell, &h.prep, &h.mesh, &hcfg, &h.hk, j, &h.scf, &kcfg())
                .unwrap()
        });
        let _ = b;
        let d = (&a.grad - &g.grad).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let dp = (&a.hf.parts.coulomb - &g.parts.coulomb)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        eprintln!("{jk:?}: null-XC vs Hartree machinery: total {d:.1e}, coulomb {dp:.1e}; |F| {:.3e}", max_abs(&g.grad));
        assert_eq!(d, 0.0);
        assert_eq!(dp, 0.0);
        assert!(max_abs(&g.parts.exchange) == 0.0, "no exchange part");
        assert!(max_abs(&g.grad) > 1e-3, "force is not trivially zero");
    }
}

/// Independent anchor of the exchange-fraction-0 plumbing: central FD of the
/// Hartree-only k energy (fresh SCF per point; no grid, so smooth).
#[test]
fn hartree_only_force_matches_fd_of_its_own_energy() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let h = hartree(&H2, [1, 1, 3], jk);
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let g = with_jk(&h, |j| {
            kpoint_hartree_gradient(&h.cell, &h.prep, &h.mesh, &hcfg, &h.hk, j, &h.scf, &kcfg())
                .unwrap()
        });
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 0), (0, 2), (1, 1)] {
            let hh = 1e-4;
            let ep = hartree(&moved(&H2, a, x, hh), [1, 1, 3], jk).scf.energy;
            let em = hartree(&moved(&H2, a, x, -hh), [1, 1, 3], jk).scf.energy;
            let d = (ep - em) / (2.0 * hh);
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("{jk:?}: Hartree-only analytic vs FD worst {worst:.2e}");
        assert!(worst < 1e-7, "{jk:?}: {worst:e}");
    }
}

/// Analytic vs central FD of the k RKS energy: dense AFT and RS-GDF, PBE.
#[test]
fn pbe_force_matches_fd_dense_and_rsgdf() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let grid = atom_grid(20, 50);
        let fx = fixture(&H2, [1, 1, 3], "PBE", jk, grid.clone());
        let g = force(&fx, None);
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 0), (0, 2), (1, 1)] {
            let d = fd(&H2, a, x, 1e-4, [1, 1, 3], "PBE", jk, &grid);
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("{jk:?}: PBE analytic vs FD worst {worst:.2e}");
        assert!(worst < 1e-7, "{jk:?}: {worst:e}");
    }
}
