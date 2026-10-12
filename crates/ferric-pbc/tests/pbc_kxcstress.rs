//! Analytic stress of the k-point closed-shell RKS energy for pure LDA/GGA
//! functionals (`ferric_pbc::kxcstress`; derivation in its module doc).
//!
//! Artifact hypotheses next to physics hypotheses are written at each test.
//!
//! Anchors: null-XC exactness against the Hartree-only stress (per part),
//! FD of ferric's own k RKS energy under cell strain (`Cell::strained`, fresh
//! SCF per point, all nine components) on the uniform grid (dense AFT and
//! RS-GDF) and on the atom grid, 1x1x1 and 1x1x3 supercell identities against
//! the Gamma KS stress per part, the uniform-grid exactly-zero/symmetric
//! structure, Ewald-exxdiv equality, thread and chunk bit-identity, the budget
//! gates, the Fock guard from both sides, refusals and production-wired
//! mutants.
//!
//! The atom-centred grid is not rotation invariant: its stress has a
//! grid-size-dependent antisymmetric part. It is MEASURED and printed here
//! (not symmetrised away); the analytic stress reproduces the FD of the
//! energy component by component, the antisymmetric part included.
//!
//! The energy has hard one-step discontinuities from the neighbour cutoff D
//! (a point's neighbour atom crossing D), so atom-grid FD tests use an
//! explicit shared `neighbour_cutoff`, and the step window is chosen from the
//! measured floors on both sides (see `FD_BAR_*`).

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::dft::{gamma_rks, GammaRksConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kgrad::KRsGdfGradSource;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointJk, KPointXc, KScfResult,
};
use ferric_pbc::kstress::{KStress, KStressConfig, KStressJk};
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::{
    kpoint_hartree_stress, kpoint_null_xc_stress, kpoint_rks_stress, KRksStress, KRksStressConfig,
    KXcStressMutation,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::{gamma_rks_stress, GammaStress, GammaStressConfig, Mat3};
use ferric_pbc::uhf::GammaUhfIntegrals;
use ferric_pbc::PeriodicGrid;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const ALL9: [(usize, usize); 9] = [
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
const CUTOFF: f64 = 10.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Jk {
    Dense,
    Gdf,
}

/// The prototype's ET-sp aux on H (as `pbc_kxcgrad.rs`).
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

fn atom_grid(nr: usize, na: usize, cutoff: Option<f64>) -> KXcGrid {
    let mut g = PeriodicGridConfig::with_size(nr, na);
    g.neighbour_cutoff = cutoff;
    KXcGrid::Atom(g)
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn krks_cfg(cell: &Cell, functional: &str, jk: Jk, grid: &KXcGrid, exxdiv: ExxDiv) -> KRksConfig {
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
fn scf_on(
    cell: &Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: &KXcGrid,
    exxdiv: ExxDiv,
) -> KRksResult {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(cell, &bs);
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

fn energy_on(cell: &Cell, mesh_n: [usize; 3], f: &str, jk: Jk, grid: &KXcGrid) -> f64 {
    scf_on(cell, mesh_n, f, jk, grid, ExxDiv::None).scf.energy
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KRksResult,
    jk: Jk,
    grid: KXcGrid,
    functional: String,
}

fn fixture_cell(cell: Cell, mesh_n: [usize; 3], functional: &str, jk: Jk, grid: KXcGrid) -> Fx {
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = scf_on(&cell, mesh_n, functional, jk, &grid, ExxDiv::None);
    let cfg = krks_cfg(&cell, functional, jk, &grid, ExxDiv::None);
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
    }
}

fn fixture(pos: &[[f64; 3]], mesh_n: [usize; 3], functional: &str, jk: Jk, grid: KXcGrid) -> Fx {
    fixture_cell(cell_of(pos), mesh_n, functional, jk, grid)
}

fn scfg(grid: &KXcGrid, mutation: Option<KXcStressMutation>) -> KRksStressConfig {
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

fn stress_cfg(fx: &Fx, cfg: &KRksStressConfig) -> KRksStress {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
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
            .unwrap()
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(ExxDiv::None);
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
            .unwrap()
        }
    }
}

fn stress(fx: &Fx, mutation: Option<KXcStressMutation>) -> KRksStress {
    stress_cfg(fx, &scfg(&fx.grid, mutation))
}

// ------------------------------------------------------------ small helpers

fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn mmax(a: &Mat3) -> f64 {
    a.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn mscale(a: &Mat3, f: f64) -> Mat3 {
    let mut o = *a;
    for r in o.iter_mut() {
        for x in r.iter_mut() {
            *x *= f;
        }
    }
    o
}

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    e
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
}

/// Central FD of the SCF energy: all nine `dE/dε_ij` at step `h`.
fn fd_all9(cell: &Cell, mesh_n: [usize; 3], f: &str, jk: Jk, grid: &KXcGrid, h: f64) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, j) in ALL9 {
        let ep = energy_on(&strained(cell, i, j, h), mesh_n, f, jk, grid);
        let em = energy_on(&strained(cell, i, j, -h), mesh_n, f, jk, grid);
        out[i][j] = (ep - em) / (2.0 * h);
    }
    out
}

// ------------------------------------------------------ Hartree-only fixture

/// Zero XC: the exactness-anchor builder (E_xc = 0, V_xc = 0).
struct ZeroXc(usize);
impl KPointXc for ZeroXc {
    fn build(
        &mut self,
        dm: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>), ferric_core::FerricError> {
        Ok((
            0.0,
            dm.iter().map(|_| Array2::zeros((self.0, self.0))).collect(),
        ))
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
            eri = KDenseAftEri::build(
                &cell,
                &prep,
                &mesh,
                &hk.s,
                ExxDiv::None,
                &KDenseAftConfig {
                    precision: 1e-12,
                    ..Default::default()
                },
            )
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

fn with_jk<R>(h: &Hartree, f: impl FnOnce(KStressJk<'_>) -> R) -> R {
    match h.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &h.cell,
                &h.prep,
                &h.mesh,
                &h.hk.s,
                ExxDiv::None,
                &KDenseAftConfig {
                    precision: 1e-12,
                    ..Default::default()
                },
            )
            .unwrap();
            f(KStressJk::Dense(&eri))
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&h.cell, &h.prep, &h.aux, &h.mesh, &h.hk.s, &kc).unwrap();
            f(KStressJk::RsGdf(KRsGdfGradSource {
                gdf: &gdf,
                cfg: &kc,
                aux: &h.aux,
            }))
        }
    }
}

fn kcfg() -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn hf_parts(p: &ferric_pbc::kstress::KStressParts) -> Vec<(&'static str, Mat3)> {
    vec![
        ("overlap_pulay", p.overlap_pulay),
        ("overlap_c0", p.overlap_c0),
        ("overlap_madelung", p.overlap_madelung),
        ("kinetic", p.kinetic),
        ("vsr", p.vsr),
        ("ecp", p.ecp),
        ("vlr", p.vlr),
        ("c0_volume", p.c0_volume),
        ("coulomb", p.coulomb),
        ("exchange", p.exchange),
        ("madelung", p.madelung),
        ("nn_sr", p.nn_sr),
        ("nn_lr", p.nn_lr),
        ("nn_background", p.nn_background),
        ("fit_j3_sr", p.fit_j3_sr),
        ("fit_j3_lr", p.fit_j3_lr),
        ("fit_j2_sr", p.fit_j2_sr),
        ("fit_j2_lr", p.fit_j2_lr),
        ("fit_g0_overlap", p.fit_g0_overlap),
        ("fit_j3_g0_volume", p.fit_j3_g0_volume),
        ("fit_j2_g0_volume", p.fit_j2_g0_volume),
    ]
}

/// (a) NULL-XC EXACTNESS ANCHOR. With E_xc = 0, V_xc = 0 and exchange fraction
/// 0 the KS entry is the exchange-free HF machinery plus zero XC terms: it must
/// equal `kpoint_hartree_stress` (same `assemble`, none of the XC plumbing or
/// KS guard) EXACTLY, part by part, dense and RS-GDF. Artifact hypothesis: an
/// XC leak (a stray add of a nonzero-initialised array, a guard that mutates F)
/// shows as a nonzero difference; physics predicts exactly 0. The exchange
/// parts must be exactly 0 (the K walks are skipped), and the stress itself
/// must not be trivially zero.
#[test]
fn null_xc_equals_the_exchange_free_machinery_exactly() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let h = hartree(&H2, [1, 1, 3], jk);
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let a = with_jk(&h, |j| {
            kpoint_null_xc_stress(
                &h.cell,
                &h.prep,
                &h.mesh,
                &hcfg,
                &h.hk,
                j,
                &h.scf,
                &scfg(&KXcGrid::default(), None),
            )
            .unwrap()
        });
        let g: KStress = with_jk(&h, |j| {
            kpoint_hartree_stress(&h.cell, &h.prep, &h.mesh, &hcfg, &h.hk, j, &h.scf, &kcfg())
                .unwrap()
        });
        let mut worst = 0.0_f64;
        for ((n, pa), (_, pg)) in hf_parts(&a.hf.parts).into_iter().zip(hf_parts(&g.parts)) {
            let d = mdiff(&pa, &pg);
            worst = worst.max(d);
            assert_eq!(d, 0.0, "{jk:?} part {n}");
        }
        let d = mdiff(&a.de_deps, &g.de_deps);
        eprintln!(
            "{jk:?}: null-XC vs Hartree machinery: parts {worst:.1e}, total {d:.1e}; |dE/de| {:.3e}",
            mmax(&g.de_deps)
        );
        assert_eq!(d, 0.0);
        assert_eq!(mmax(&g.parts.exchange), 0.0, "no exchange part");
        assert_eq!(mmax(&g.parts.madelung), 0.0);
        assert_eq!(mmax(&a.xc_ao) + mmax(&a.xc_weight), 0.0);
        assert!(mmax(&g.de_deps) > 1e-3, "stress is not trivially zero");
    }
}

/// Independent anchor of the exchange-fraction-0 stress plumbing: central FD of
/// the Hartree-only k energy under strain (fresh SCF per point; no grid).
/// Dense AFT 1x1x3, three components; the energy is smooth (no cutoff mask).
#[test]
fn hartree_only_stress_matches_fd_of_its_own_energy() {
    use ferric_pbc::kscf::KScfConfig;
    let _ = KScfConfig::default;
    for jk in [Jk::Dense, Jk::Gdf] {
        let h = hartree(&H2, [1, 1, 3], jk);
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        let g = with_jk(&h, |j| {
            kpoint_hartree_stress(&h.cell, &h.prep, &h.mesh, &hcfg, &h.hk, j, &h.scf, &kcfg())
                .unwrap()
        });
        let mut worst = 0.0_f64;
        for (i, j) in [(0, 0), (1, 2), (2, 2)] {
            let hh = 1e-4;
            let e = |s: f64| {
                let c = strained(&h.cell, i, j, s);
                hartree_energy_on(&c, jk)
            };
            let d = (e(hh) - e(-hh)) / (2.0 * hh);
            worst = worst.max((g.de_deps[i][j] - d).abs());
        }
        eprintln!("{jk:?}: Hartree-only analytic vs FD worst {worst:.2e}");
        assert!(worst < 5e-8, "{jk:?}: {worst:e}");
    }
}

fn hartree_energy_on(cell: &Cell, jk: Jk) -> f64 {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(cell, [1, 1, 3]).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let mut scfcfg = ferric_pbc::kscf::KScfConfig::default();
    scfcfg.energy_conv = 1e-13;
    scfcfg.grad_conv = 1e-10;
    scfcfg.max_iter = 200;
    let eri;
    let gdf;
    let jkb: Box<dyn KPointJk + '_> = match jk {
        Jk::Dense => {
            eri = KDenseAftEri::build(
                cell,
                &prep,
                &mesh,
                &hk.s,
                ExxDiv::None,
                &KDenseAftConfig {
                    precision: 1e-12,
                    ..Default::default()
                },
            )
            .unwrap();
            Box::new(eri.jk_builder())
        }
        Jk::Gdf => {
            gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kgdf_cfg()).unwrap();
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
    solve_krks_injected(cell, &mesh, &scfcfg, inj, &mut z, 0.0)
        .unwrap()
        .energy
}

// ------------------------------------------------------------ Gamma anchors

fn gamma_side(
    cell: &Cell,
    functional: &str,
    grid: &PeriodicGridConfig,
    bs: &BasisSet,
) -> GammaStress {
    let prep = prep_for(cell, bs);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        1e-12,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut gc = GammaRksConfig::new(functional);
    gc.grid = grid.clone();
    gc.exxdiv = ExxDiv::None;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 200;
    let r = gamma_rks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gc).unwrap();
    gamma_rks_stress(
        cell,
        &prep,
        &hcfg,
        &hc,
        &eri,
        &r.scf,
        &gc,
        &GammaStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Rows of (name, |k part − Gamma part|) for a mesh of `n_cells` cells against
/// the supercell Gamma stress (`dE/dε` per cell = supercell value / n_cells).
fn gamma_rows(k: &KRksStress, g: &GammaStress, n_cells: f64) -> Vec<(&'static str, f64)> {
    let p = &g.parts;
    let kp = &k.hf.parts;
    let s = 1.0 / n_cells;
    vec![
        ("xc_ao", mdiff(&k.xc_ao, &mscale(&p.xc_ao, s))),
        ("xc_weight", mdiff(&k.xc_weight, &mscale(&p.xc_weight, s))),
        ("overlap", mdiff(&kp.overlap(), &mscale(&p.overlap, s))),
        ("kinetic", mdiff(&kp.kinetic, &mscale(&p.kinetic, s))),
        ("vsr", mdiff(&kp.vsr, &mscale(&p.vsr, s))),
        ("vlr", mdiff(&kp.vlr, &mscale(&p.vlr, s))),
        ("c0_volume", mdiff(&kp.c0_volume, &mscale(&p.c0_volume, s))),
        ("coulomb", mdiff(&kp.coulomb, &mscale(&p.eri, s))),
        ("sigma", mdiff(&k.sigma, &g.sigma)),
    ]
}

/// (c) 1x1x1 mesh == the Gamma KS stress, PER PART (the total can hide a
/// defect of the size of a fit error). Independent construction: real
/// lattice-summed AO strain moments vs complex Bloch ones; Gamma's own SCF.
/// Floor: both SCFs converge to ~1e-10 in the density. Artifact hypothesis:
/// the complex path at k = 0 differs from the real path only by a missing
/// `Re`/`2`/`1/N_k`; none of the phase mutants can show here (k = 0).
#[test]
fn one_point_mesh_matches_the_gamma_ks_stress_per_part() {
    for f in ["LDA", "PBE"] {
        let grid = PeriodicGridConfig::with_size(20, 50);
        let gam = gamma_side(&cell_of(&H2), f, &grid, &pyscf_sto3g_h());
        let fx = fixture(&H2, [1, 1, 1], f, Jk::Dense, KXcGrid::Atom(grid));
        let k = stress(&fx, None);
        for (n, d) in gamma_rows(&k, &gam, 1.0) {
            eprintln!("{f} 1x1x1 vs Gamma: {n:<10} {d:.2e}");
            assert!(d < 1e-11, "{f} {n}: {d:e}");
        }
        assert!(mmax(&k.xc_weight) > 1e-3 && mmax(&k.xc_ao) > 1e-3);
        eprintln!(
            "{f} 1x1x1: antisymmetry {:.2e} (Gamma {:.2e})",
            k.antisymmetry(),
            gam.antisymmetry()
        );
    }
}

/// Explicit diag(n) supercell of `pos` (copy-major, copy 0 first).
fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3]) -> Cell {
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

/// (d) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh: the k stress per cell is
/// the Gamma stress of the explicit supercell (sigma equal, dE/dε / N). Both
/// sides use the same explicit `neighbour_cutoff`. Carries the Bloch phases
/// (ConjPhase is an identity on TRIM meshes, not here). Artifact hypothesis:
/// if the cell and supercell atom grids differ (the default cutoff depends on
/// the cell), the xc_weight part differs at the grid-size scale, not at 1e-10.
#[test]
fn k_mesh_stress_is_the_gamma_supercell_stress_per_part() {
    let mut grid = PeriodicGridConfig::with_size(20, 50);
    grid.neighbour_cutoff = Some(CUTOFF);
    let f = "PBE";
    let sc = supercell_pos(&H2, 4.0, [1, 1, 3]);
    let gam = gamma_side(&sc, f, &grid, &pyscf_sto3g_h());
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, KXcGrid::Atom(grid.clone()));
    let k = stress(&fx, None);
    for (n, d) in gamma_rows(&k, &gam, 3.0) {
        eprintln!("{f} 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < 1e-10, "{f} {n}: {d:e}");
    }
    eprintln!(
        "antisymmetry: k {:.2e}, supercell {:.2e}",
        k.antisymmetry(),
        gam.antisymmetry()
    );
}

// ------------------------------------------------------------- uniform grid

/// On a uniform grid the weight response is exactly the volume term
/// `w δ_ab` (diagonal, equal entries `Σ_g w_g ε_g`) and the stress is
/// symmetric to roundoff (the energy is rotation invariant when the points are
/// at fixed fractional coordinates). Artifact hypothesis: a stress that kept
/// the atom-grid partition term on a uniform grid would be non-diagonal /
/// asymmetric; physics predicts exactly diagonal and a symmetric total.
#[test]
fn uniform_grid_weight_is_the_volume_term_and_stress_is_symmetric() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        KXcGrid::Uniform([24, 24, 24]),
    );
    let s = stress(&fx, None);
    let w = &s.xc_weight;
    for a in 0..3 {
        for b in 0..3 {
            if a != b {
                assert_eq!(w[a][b], 0.0);
            }
        }
        assert_eq!(w[a][a], w[0][0]);
    }
    // sum_g w_g eps_g = E_xc ... only up to rho*exc vs the energy's e: equal.
    let e_expected = s.e_xc;
    eprintln!(
        "uniform weight diag {:.10e}, E_xc {:.10e}; antisymmetry {:.2e}",
        w[0][0],
        e_expected,
        s.antisymmetry()
    );
    assert!((w[0][0] - e_expected).abs() < 1e-12);
    assert!(s.antisymmetry() < 1e-9, "{:e}", s.antisymmetry());
    assert!(mmax(&s.xc_ao) > 1e-2);
}

/// (e) FD of ferric's own k RKS energy under cell strain, uniform grid (no
/// neighbour mask: smooth), dense AFT and RS-GDF, PBE, all nine components.
/// Artifact hypothesis: a missing term misses by the term's size; the floor is
/// the central-FD h² truncation measured on the component set.
#[test]
fn uniform_grid_pbe_stress_matches_fd_all_nine_dense() {
    fd_anchor(Jk::Dense, "PBE", FD_BAR);
}

fn fd_anchor(jk: Jk, f: &str, bar: f64) {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let fx = fixture(&H2, [1, 1, 3], f, jk, grid.clone());
    let s = stress(&fx, None);
    let fd = fd_all9(&fx.cell, [1, 1, 3], f, jk, &grid, FD_H);
    let miss = mdiff(&s.de_deps, &fd);
    eprintln!(
        "{jk:?} {f} uniform 24^3: |analytic - FD| {miss:.2e} (h {FD_H}); |dE/de| max {:.3e}",
        mmax(&s.de_deps)
    );
    assert!(miss < bar, "{jk:?} {f}: {miss:e}");
}

/// Floor: measured on the uniform grid, central FD h = 1e-4 (see the test log).
const FD_H: f64 = 1e-4;
const FD_BAR: f64 = 5e-8;
