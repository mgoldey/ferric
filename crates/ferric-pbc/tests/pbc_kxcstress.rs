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
    scf_bs(cell, mesh_n, functional, jk, grid, exxdiv, &pyscf_sto3g_h())
}

fn scf_bs(
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
    exxdiv: ExxDiv,
}

fn fixture_cell(cell: Cell, mesh_n: [usize; 3], functional: &str, jk: Jk, grid: KXcGrid) -> Fx {
    fixture_exx(cell, mesh_n, functional, jk, grid, ExxDiv::None)
}

fn fixture_exx(
    cell: Cell,
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    grid: KXcGrid,
    exxdiv: ExxDiv,
) -> Fx {
    fixture_bs(cell, mesh_n, functional, jk, grid, exxdiv, &pyscf_sto3g_h())
}

fn fixture_bs(
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

fn try_stress_cfg(fx: &Fx, cfg: &KRksStressConfig) -> Result<KRksStress, ferric_core::FerricError> {
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

fn stress_cfg(fx: &Fx, cfg: &KRksStressConfig) -> KRksStress {
    try_stress_cfg(fx, cfg).expect("kpoint_rks_stress")
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
                0.0,
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

/// Uniform-grid FD step and bar. Floor MEASURED: 1.02e-8 over all nine
/// components at h = 1e-4 for PBE and LDA, dense and RS-GDF (the central-FD
/// h^2 truncation: 4.3e-8 at 2e-4, 1.0e-8 at 1e-4, then 1.8e-9 at 5e-5 where
/// roundoff starts to matter; `uniform_grid_fd_miss_scales_as_h_squared`
/// asserts the 2e-4 -> 1e-4 ratio is in [3, 5.5]); the smallest production-wired
/// mutant misses by 1.2e-2 (NoHessian / NoStrainMoment). The bar 5e-8 is 5x
/// the floor and 2e5x below the smallest mutant, and equals the k HF stress
/// FD bar of `pbc_kstress.rs`.
const FD_H: f64 = 1e-4;
const FD_BAR: f64 = 5e-8;

// ------------------------------------------------- FD smoothness under strain

/// PROBE (not a pass/fail anchor): is the energy smooth under CELL STRAIN on
/// the atom grid? For three components, the central FD at h = 1e-5..4e-4
/// against the analytic value, for the default neighbour cutoff and two
/// explicit ones. A smooth energy shows the h^2 truncation trend; the hard
/// neighbour mask shows an isolated jump (a miss that is not monotone in h).
#[test]
#[ignore = "probe: 3 cutoffs x 3 components x 6 steps x 2 SCFs; measured 93 s"]
fn fd_smoothness_probe_under_strain() {
    for cutoff in [None, Some(10.0), Some(14.0)] {
        let grid = atom_grid(20, 50, cutoff);
        let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
        let s = stress(&fx, None);
        for (i, j) in [(2, 2), (0, 2), (1, 0)] {
            let mut row = String::new();
            for h in [1e-5, 2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4] {
                let ep = energy_on(
                    &strained(&fx.cell, i, j, h),
                    [1, 1, 3],
                    "PBE",
                    Jk::Dense,
                    &grid,
                );
                let em = energy_on(
                    &strained(&fx.cell, i, j, -h),
                    [1, 1, 3],
                    "PBE",
                    Jk::Dense,
                    &grid,
                );
                let d = (ep - em) / (2.0 * h);
                row += &format!(" h={h:.1e}:{:.2e}", (d - s.de_deps[i][j]).abs());
            }
            eprintln!("cutoff {cutoff:?} eps({i},{j}) |FD-analytic|:{row}");
        }
    }
}

// ------------------------------------------------------------- atom-grid FD

/// Central-FD step of the atom-grid anchors. From the smoothness probe above:
/// at h = 2.5e-5 the three probed components miss by 3e-12..2e-10 (cutoff 10
/// and 14); h = 1e-5 is noise limited (1.5e-9: energy roundoff / 2h), and from
/// h = 1e-4 up the h^2 truncation (8e-9 .. 1.6e-7) and, at cutoff 10, the hard
/// neighbour-mask jump (5e-8 at h = 2e-4 on eps_02) dominate.
const FD_H_ATOM: f64 = 2.5e-5;
/// Atom-grid FD bar. Floor MEASURED over all nine components (h = 2.5e-5,
/// cutoff 10): 7.0e-9 (`fd_window_probe_all_nine_atom_grid`; 1.5e-9 at 1e-5,
/// 5.7e-9 at 5e-5, and 2.0e-6 at 1e-4 where eps_11 crosses the mask). The
/// floor is NOT monotone in h: it is roundoff plus mask jumps, not h^2
/// truncation. The bar is only ~4x that floor; the both-sides argument rests
/// on the smallest production-wired mutant missing by 1.2e-2 (NoHessian),
/// 4e5x above the bar.
const FD_BAR_ATOM: f64 = 3e-8;

/// (e) FD of the energy on the ATOM-CENTRED grid, PBE (20,50), explicit shared
/// cutoff, all nine components, dense AFT. The analytic stress is the stress of
/// the energy as computed: the FD reproduces it component by component, the
/// antisymmetric part (the lab-fixed Lebedev grid is not rotation invariant)
/// included. Artifact hypothesis: if the stress were symmetrised or the point
/// motion mis-anchored, the FD antisymmetric part would disagree at the
/// antisymmetry's size, not at the floor.
#[test]
fn atom_grid_pbe_stress_matches_fd_all_nine_antisymmetric_part_included() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let fd = fd_all9(&fx.cell, [1, 1, 3], "PBE", Jk::Dense, &grid, FD_H_ATOM);
    let miss = mdiff(&s.de_deps, &fd);
    let mut fd_anti = 0.0_f64;
    let mut an_anti = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            fd_anti = fd_anti.max((fd[i][j] - fd[j][i]).abs());
            an_anti = an_anti.max((s.de_deps[i][j] - s.de_deps[j][i]).abs());
        }
    }
    eprintln!(
        "atom (20,50) PBE: |analytic - FD| {miss:.2e}; dE/de antisymmetry analytic {an_anti:.3e} FD {fd_anti:.3e}; sigma antisymmetry {:.3e}",
        s.antisymmetry()
    );
    assert!(miss < FD_BAR_ATOM, "{miss:e}");
    assert!(an_anti > 1e-3, "the antisymmetric part is real");
    assert!((an_anti - fd_anti).abs() < FD_BAR_ATOM);
}

/// Pressure from isotropic scaling (a homogeneous-scaling identity valid on
/// every grid): `Σ_a dE/dε_aa` equals the FD of E(1 + s) at fixed fractional
/// coordinates; uniform grid, dense AFT.
#[test]
fn trace_matches_isotropic_scaling_fd_on_the_uniform_grid() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let h = 1e-4;
    let e = |t: f64| {
        let eps = [[t, 0.0, 0.0], [0.0, t, 0.0], [0.0, 0.0, t]];
        energy_on(
            &fx.cell.strained(&eps).unwrap(),
            [1, 1, 3],
            "PBE",
            Jk::Dense,
            &grid,
        )
    };
    let d = (e(h) - e(-h)) / (2.0 * h);
    let tr = s.de_deps[0][0] + s.de_deps[1][1] + s.de_deps[2][2];
    let p_fd = -d / (3.0 * s.volume);
    eprintln!(
        "isotropic: trace {tr:.10e} vs FD {d:.10e} (diff {:.2e}); P {:.6e} vs FD {p_fd:.6e}",
        (tr - d).abs(),
        s.pressure()
    );
    assert!((tr - d).abs() < 3.0 * FD_BAR, "{:e}", (tr - d).abs());
}

// -------------------------------------------------- translation and rotation

/// Rigid translation of every atom leaves the atom-grid stress unchanged
/// (the grid is atom-centred, Bloch phases depend on lattice vectors only).
/// Artifact hypothesis: an anchor error (point not moving with its home atom)
/// would make the XC stress depend on absolute positions.
#[test]
fn stress_is_translation_invariant_on_the_atom_grid() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let a = stress(
        &fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone()),
        None,
    );
    let t = [0.37, -0.21, 0.53];
    let moved: Vec<[f64; 3]> = H2
        .iter()
        .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
        .collect();
    let b = stress(&fixture(&moved, [1, 1, 3], "PBE", Jk::Dense, grid), None);
    let d = mdiff(&a.de_deps, &b.de_deps);
    eprintln!("translation: |dE/de - dE/de'| {d:.2e}");
    assert!(d < 2e-9, "{d:e}");
}

/// A rigid rotation of cell + atoms: on the UNIFORM grid (points at fixed
/// fractional coordinates) the energy is rotation invariant and
/// `dE/dε' = R dE/dε Rᵀ` to roundoff; on the ATOM grid (lab-fixed Lebedev
/// offsets) it is violated at the grid-error scale, reported. Artifact
/// hypothesis: a wrong strain-moment index would break the uniform identity;
/// the atom-grid violation is the antisymmetric-part physics, not a bug.
#[test]
fn rotation_covariance_holds_on_the_uniform_grid() {
    rotation_case("uniform 24^3", KXcGrid::Uniform([24, 24, 24]), 1e-8);
}

/// The atom-grid half of the rotation test: the violation is grid sized
/// (reported, bounded by the antisymmetry scale, not asserted to vanish).
#[test]
#[ignore = "atom-grid rotation violation report (2 fixtures); measured 9 s"]
fn rotation_violation_on_the_atom_grid_is_grid_sized() {
    rotation_case("atom (20,50)", atom_grid(20, 50, Some(CUTOFF)), 5e-2);
}

fn rotation_case(name: &str, grid: KXcGrid, bar: f64) {
    let th = 0.5_f64;
    let (c, sn) = (th.cos(), th.sin());
    let rot = [[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]];
    let apply = |v: [f64; 3]| -> [f64; 3] {
        let mut o = [0.0; 3];
        for i in 0..3 {
            o[i] = rot[i][0] * v[0] + rot[i][1] * v[1] + rot[i][2] * v[2];
        }
        o
    };
    let lat0 = cubic(4.0);
    let lat = [apply(lat0[0]), apply(lat0[1]), apply(lat0[2])];
    let pos: Vec<[f64; 3]> = H2.iter().map(|p| apply(*p)).collect();
    let cell_r = Cell::new(hydrogens(&pos), lat).unwrap();
    let rsr = |m: &Mat3| -> Mat3 {
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for a in 0..3 {
                    for b in 0..3 {
                        o[i][j] += rot[i][a] * m[a][b] * rot[j][b];
                    }
                }
            }
        }
        o
    };
    let s0 = stress(
        &fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone()),
        None,
    );
    let s1 = stress(
        &fixture_cell(cell_r, [1, 1, 3], "PBE", Jk::Dense, grid),
        None,
    );
    let d = mdiff(&s1.de_deps, &rsr(&s0.de_deps));
    eprintln!("rotation {name}: |dE/de' - R dE/de R^T| {d:.2e}");
    assert!(d < bar, "{name}: {d:e}");
}

// ---------------------------------------------------------------- exxdiv

/// For a pure functional the SCF builds no `K`, so the exxdiv it ran under
/// cannot matter, and `kpoint_rks_stress` ignores `KRksStressConfig::exxdiv`
/// for it (`vm = 0` and exchange scale 0 go to the HF assembly). This test checks that the SCF
/// energy and the stress computed from a `ewald` SCF (dense AFT and RS-GDF
/// built with the Ewald shift) equal those from a `none` SCF; it is a
/// wiring check (a Madelung leak into the KS SCF or into the stress Fock guard
/// would be v_M-sized, ~0.1 Ha), not an independent physics anchor.
#[test]
fn ewald_and_none_exxdiv_agree_for_a_pure_functional() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    for jk in [Jk::Dense, Jk::Gdf] {
        let a = fixture_exx(
            cell_of(&H2),
            [1, 1, 3],
            "PBE",
            jk,
            grid.clone(),
            ExxDiv::None,
        );
        let b = fixture_exx(
            cell_of(&H2),
            [1, 1, 3],
            "PBE",
            jk,
            grid.clone(),
            ExxDiv::Ewald,
        );
        let (sa, sb) = (stress(&a, None), stress(&b, None));
        let de = (a.res.scf.energy - b.res.scf.energy).abs();
        let ds = mdiff(&sa.de_deps, &sb.de_deps);
        eprintln!("{jk:?} exxdiv none vs ewald: energy {de:.2e}, dE/de {ds:.2e}");
        assert!(de < 1e-11, "{jk:?} {de:e}");
        assert!(ds < 1e-9, "{jk:?} {ds:e}");
    }
}

// ---------------------------------------------------------------- bitwise

/// The result does not depend on the thread count or on how many chunks run
/// concurrently (chunk-order reduction): bit-identical.
#[test]
fn stress_is_bit_identical_across_threads_and_live_chunks() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let run = |threads: usize, cap: Option<usize>| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut cfg = scfg(&fx.grid, None);
                cfg.max_live_chunks = cap;
                stress_cfg(&fx, &cfg)
            })
    };
    let a = run(1, None);
    let b = run(4, None);
    let c = run(4, Some(1));
    assert_eq!(a.de_deps, b.de_deps);
    assert_eq!(a.e_xc, b.e_xc);
    assert_eq!(a.de_deps, c.de_deps);
    assert_eq!(c.live_chunks, 1);
    eprintln!(
        "bit-identical across 1/4 threads and live-chunk cap; live {} / {} / {}",
        a.live_chunks, b.live_chunks, c.live_chunks
    );
}

// ------------------------------------------------------------------- memory

/// The strain pass is pre-flighted on the ledger: a budget that holds the fixed
/// HF arrays but not one AO-moment chunk is a typed error naming the quantity;
/// a one-chunk cap gives the bit-identical stress of the unconstrained run.
#[test]
fn xc_chunks_are_budgeted() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let eri = KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let at = |budget: usize| {
        let mut cfg = scfg(&fx.grid, None);
        cfg.stress.budget_bytes = Some(budget);
        kpoint_rks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            &fx.res.scf,
            "PBE",
            &cfg,
        )
    };
    let full = at(AMPLE).unwrap();
    // Every budget either reproduces the stress bitwise or fails with a typed
    // error; the XC chunk gate and the Fock-guard reservation both trip.
    let (mut saw, mut saw_guard, mut n_ok) = (false, false, 0);
    let mut b = 20_000usize;
    while b < 80_000_000 {
        match at(b) {
            Ok(s) => {
                assert_eq!(s.de_deps, full.de_deps, "budget {b}");
                n_ok += 1;
            }
            Err(e) => {
                let m = e.to_string();
                saw |= m.contains("k-point XC stress AO moment chunk");
                saw_guard |= m.contains("k KS stress Fock-guard matrices");
            }
        }
        b += b / 2;
    }
    eprintln!("budget sweep: {n_ok} bitwise-equal, chunk gate {saw} (the 768-byte guard reservation never trips alone: {saw_guard})");
    assert!(saw, "some budget must fail on the XC chunk gate");
    assert!(n_ok > 0);
    let mut cfg = scfg(&fx.grid, None);
    cfg.max_live_chunks = Some(1);
    let one = stress_cfg(&fx, &cfg);
    assert_eq!(one.live_chunks, 1);
    assert_eq!(one.de_deps, full.de_deps);
    eprintln!(
        "chunk {} points, live {} -> 1, bit-identical",
        one.chunk_points, full.live_chunks
    );
}

// ------------------------------------------------- refusals and Fock guard

#[test]
fn refusals_and_the_fock_guard() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let eri = KDenseAftEri::build(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &fx.hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let cfg = scfg(&fx.grid, None);
    let run = |f: &str, scf: &KScfResult| {
        kpoint_rks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KStressJk::Dense(&eri),
            scf,
            f,
            &cfg,
        )
    };
    for (name, key) in [
        ("HSE06", "range-separated"),
        ("SCAN", "meta-GGA"),
        ("wB97X-V", "VV10"),
    ] {
        let e = run(name, &fx.res.scf)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    // The HF entry still refuses a KS result.
    let hf = ferric_pbc::kstress::kpoint_rhf_stress(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KStressJk::Dense(&eri),
        &fx.res.scf,
        ExxDiv::None,
        &kcfg(),
    );
    assert!(hf.is_err());
    // Guard, both sides.
    let ok = run("PBE", &fx.res.scf).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    let wrong = run("LDA", &fx.res.scf)
        .expect_err("wrong functional")
        .to_string();
    eprintln!("wrong functional: {wrong}");
    assert!(wrong.contains("differs"));
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-6, 0.0);
    assert!(run("PBE", &t).is_err(), "1e-6 tamper must be refused");
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-10, 0.0);
    let tamp = run("PBE", &t).expect("1e-10 tamper is below the bar");
    eprintln!(
        "guard: matching {:.1e}, 1e-10 tamper {:.1e} passes, 1e-6 refused",
        ok.hf.fock_mismatch, tamp.hf.fock_mismatch
    );
    // Unconverged SCF.
    let mut t = fx.res.scf.clone();
    t.converged = false;
    assert!(run("PBE", &t).is_err());
}

// ------------------------------------------------------------------ mutants

const MUTANT_BAR: f64 = 1e-5;

fn miss(fx: &Fx, m: KXcStressMutation, base: &KRksStress) -> f64 {
    mdiff(&stress(fx, Some(m)).de_deps, &base.de_deps)
}

/// Smoke mutants on the non-TRIM 1x1x3 mesh (production-wired seams in
/// `KRksStressConfig::mutation`): each must miss the unmutated (FD-,
/// Gamma- and supercell-anchored) stress by far more than the bar.
#[test]
fn smoke_mutants_are_caught() {
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let base = stress(&fx, None);
    for m in [
        KXcStressMutation::NoPointMotion,
        KXcStressMutation::NoWeightDeriv,
        KXcStressMutation::NoHessian,
        KXcStressMutation::ConjPhase,
    ] {
        let d = miss(&fx, m, &base);
        eprintln!("{m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
}

/// PROBE: the all-nine FD miss against the step on the atom grid (chooses the
/// window of `FD_H_ATOM`).
#[test]
#[ignore = "probe: 4 steps x 18 SCFs; measured 43 s"]
fn fd_window_probe_all_nine_atom_grid() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    for h in [1e-5, 2.5e-5, 5e-5, 1e-4] {
        let fd = fd_all9(&fx.cell, [1, 1, 3], "PBE", Jk::Dense, &grid, h);
        let mut row = String::new();
        for (i, j) in ALL9 {
            row += &format!(" {:.1e}", (fd[i][j] - s.de_deps[i][j]).abs());
        }
        eprintln!(
            "h={h:.1e} worst {:.2e}; per component (00 01 02 10 11 12 20 21 22):{row}",
            mdiff(&fd, &s.de_deps)
        );
    }
}

// ------------------------------------------------------------- PySCF anchor

/// PySCF 2.13.1 `pbc.grad.krks_stress` (FFTDF on `UniformGrids(61^3)`,
/// exxdiv ewald, conv_tol 1e-12, STO-3G H2 in the a = 4 cube, atoms
/// (0.3, 0.2, 0.1) and (0.3, 0.2, 1.5), 1x1x2 gamma-centred mesh): the stress
/// `σ = (1/Ω) dE/dε` (Hartree/Bohr^3), which PySCF itself verifies against the
/// central FD of its own energy to 2.3e-10. Its FFTDF and AFTDF energies agree
/// to 6.5e-13 at this mesh, and ferric's dense-AFT energy on the same uniform
/// grid matches PySCF's (`pbc_krks.rs` pins), so this is the stress of the
/// same energy. Diagonal values; the off-diagonals are symmetric zeros
/// (|.| < 1e-13). The `LDA` row is PySCF's `LDA,VWN` (ferric's LDA is SVWN;
/// PySCF's bare `LDA` is exchange only). The constants block below is the
/// verbatim output of `scripts/gen_pyscf_kks_stress_refs.py`, which asserts
/// the PySCF version cited here.
const PYSCF_STRESS: [(&str, f64, f64); 2] = [
    ("LDA", 6.102756319e-03, -5.161318677e-03),
    ("PBE", 5.824134026e-03, -5.078720384e-03),
];
const PYSCF_STRESS_BAR: f64 = 1e-9;

/// Independent EXTERNAL anchor: ferric's k RKS stress on the 61^3 uniform grid
/// vs PySCF's analytic FFTDF stress, LDA and PBE. Artifact hypothesis: a
/// self-consistent construction error (the supercell, 1x1x1 and FD anchors all
/// share ferric's energy) cannot show here; a missing XC term would miss by
/// its size (6e-3 sigma units is the whole XC+HF stress here).
#[test]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_pbe() {
    pyscf_case(1);
}

/// The LDA row of the PySCF anchor (a second 61^3 SCF).
#[test]
#[ignore = "second 61^3 PySCF-anchor row; measured 12 s"]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_lda() {
    pyscf_case(0);
}

fn pyscf_case(row: usize) {
    let (f, xx, zz) = PYSCF_STRESS[row];
    {
        let fx = fixture_exx(
            cell_of(&H2_ATOMS),
            [1, 1, 2],
            f,
            Jk::Dense,
            KXcGrid::Uniform([61, 61, 61]),
            ExxDiv::Ewald,
        );
        let s = stress(&fx, None);
        let want = [[xx, 0.0, 0.0], [0.0, xx, 0.0], [0.0, 0.0, zz]];
        let d = mdiff(&s.sigma, &want);
        eprintln!(
            "{f} 61^3 1x1x2 vs PySCF: |sigma - sigma_pyscf| {d:.2e} (xx {:.9e} vs {xx:.9e}, zz {:.9e} vs {zz:.9e}); antisymmetry {:.1e}",
            s.sigma[0][0], s.sigma[2][2], s.antisymmetry()
        );
        assert!(d < PYSCF_STRESS_BAR, "{f}: {d:e}");
    }
}

// ----------------------------------------------------------- full mutant sweep

const ALL_MUTANTS: [KXcStressMutation; 12] = [
    KXcStressMutation::NoAo,
    KXcStressMutation::NoPointMotion,
    KXcStressMutation::PointSelf,
    KXcStressMutation::NoWeightDeriv,
    KXcStressMutation::NoGga,
    KXcStressMutation::NoHessian,
    KXcStressMutation::NoStrainMoment,
    KXcStressMutation::NoPhase,
    KXcStressMutation::ConjPhase,
    KXcStressMutation::WrongFactor,
    KXcStressMutation::NoInvNk,
    KXcStressMutation::PulayNoVxc,
];

fn sweep(fx: &Fx, tag: &str) -> Vec<(KXcStressMutation, f64, f64)> {
    let base = stress(fx, None);
    ALL_MUTANTS
        .iter()
        .map(|&m| {
            let s = stress(fx, Some(m));
            let all = mdiff(&s.de_deps, &base.de_deps);
            // Largest OFF-diagonal miss (volume-only defects have none).
            let mut off = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    if i != j {
                        off = off.max((s.de_deps[i][j] - base.de_deps[i][j]).abs());
                    }
                }
            }
            eprintln!("{tag}: {m:?}: miss {all:.2e} (off-diagonal {off:.2e})");
            (m, all, off)
        })
        .collect()
}

fn miss_of(rows: &[(KXcStressMutation, f64, f64)], m: KXcStressMutation) -> (f64, f64) {
    rows.iter()
        .find(|r| r.0 == m)
        .map(|r| (r.1, r.2))
        .expect("mutant swept")
}

/// The full sweep with the blind spots ASSERTED (artifact hypotheses: a mutant
/// that is an identity on a mesh, grid or functional must show exactly 0 there;
/// a live one must miss by more than the bar). Smallest live miss on the
/// anchored configuration is printed.
#[test]
#[ignore = "full mutant sweep: 12 mutants x 5 configurations; measured 260 s"]
fn full_mutant_sweep() {
    // Atom grid, PBE, non-TRIM 1x1x3: every mutant live except none.
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let rows = sweep(&fx, "atom PBE 1x1x3");
    let mut smallest = f64::INFINITY;
    for (m, d, _) in &rows {
        assert!(*d > MUTANT_BAR, "{m:?} survives on atom PBE 1x1x3: {d:e}");
        smallest = smallest.min(*d);
    }
    eprintln!("smallest mutant miss, atom PBE 1x1x3: {smallest:.2e}");

    // Uniform grid: PointSelf is an IDENTITY (the point is its own anchor),
    // NoWeightDeriv (the volume term) is diagonal only.
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, KXcGrid::Uniform([24; 3]));
    let rows = sweep(&fx, "uniform PBE 1x1x3");
    assert_eq!(miss_of(&rows, KXcStressMutation::PointSelf).0, 0.0);
    let (d, off) = miss_of(&rows, KXcStressMutation::NoWeightDeriv);
    assert!(d > MUTANT_BAR && off == 0.0, "volume term: {d:e} / {off:e}");
    for m in ALL_MUTANTS {
        if m != KXcStressMutation::PointSelf {
            assert!(miss_of(&rows, m).0 > MUTANT_BAR, "{m:?} survives (uniform)");
        }
    }

    // LDA: the GGA-only mutants are identities (c = 0).
    let fx = fixture(
        &H2,
        [1, 1, 3],
        "LDA",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let rows = sweep(&fx, "atom LDA 1x1x3");
    for m in [
        KXcStressMutation::NoGga,
        KXcStressMutation::NoHessian,
        KXcStressMutation::NoStrainMoment,
    ] {
        assert_eq!(miss_of(&rows, m).0, 0.0, "{m:?} is an identity for LDA");
    }

    // TRIM-only mesh (1x1x2): ConjPhase is an identity (phases are +-1),
    // NoPhase is live.
    let fx = fixture(
        &H2,
        [1, 1, 2],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let rows = sweep(&fx, "atom PBE 1x1x2 (TRIM)");
    assert_eq!(miss_of(&rows, KXcStressMutation::ConjPhase).0, 0.0);
    assert!(miss_of(&rows, KXcStressMutation::NoPhase).0 > MUTANT_BAR);

    // One-point mesh: phase and 1/N_k mutants are identities.
    let fx = fixture(
        &H2,
        [1, 1, 1],
        "PBE",
        Jk::Dense,
        atom_grid(20, 50, Some(CUTOFF)),
    );
    let rows = sweep(&fx, "atom PBE 1x1x1");
    for m in [
        KXcStressMutation::NoPhase,
        KXcStressMutation::ConjPhase,
        KXcStressMutation::NoInvNk,
    ] {
        assert_eq!(miss_of(&rows, m).0, 0.0, "{m:?} is an identity at k = 0");
    }
}

// ------------------------------------------------- further FD / mesh anchors

/// LDA FD on the uniform grid (no GGA terms: the AO strain is the first-moment
/// term alone), dense AFT, all nine.
#[test]
#[ignore = "second all-nine FD (18 SCFs); measured 16 s"]
fn uniform_grid_lda_stress_matches_fd_all_nine_dense() {
    fd_anchor(Jk::Dense, "LDA", FD_BAR);
}

/// RS-GDF J (unsplit): the same FD anchor with the fitted Coulomb. The fit's
/// own strain is exercised by `pbc_kstress`; here the XC stress sits on the
/// RS-GDF density/Fock.
#[test]
#[ignore = "RS-GDF all-nine FD (18 SCFs with 18 fits); measured 22 s"]
fn uniform_grid_pbe_stress_matches_fd_all_nine_rsgdf() {
    fd_anchor(Jk::Gdf, "PBE", FD_BAR);
}

/// RS-GDF vs dense AFT: the XC parts are J-route independent up to the density
/// difference; the total differs by a fit-sized amount (not 0, not O(1)).
#[test]
#[ignore = "two SCFs + stresses; measured 12 s"]
fn rsgdf_stress_differs_from_dense_by_a_fit_sized_amount() {
    let grid = KXcGrid::Uniform([24; 3]);
    let a = stress(
        &fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone()),
        None,
    );
    let b = stress(&fixture(&H2, [1, 1, 3], "PBE", Jk::Gdf, grid), None);
    let d = mdiff(&a.de_deps, &b.de_deps);
    eprintln!("dense vs RS-GDF dE/de: {d:.2e}");
    assert!(d > 1e-9 && d < 1e-2);
}

/// The 2x2x2 Gamma-centred (TRIM-only) mesh against FD of its own energy: the
/// uniform grid, so smooth; eight k-points, all real phases.
#[test]
#[ignore = "8-point mesh, 3 components, 6 SCFs; measured 24 s"]
fn uniform_grid_pbe_stress_matches_fd_on_the_2x2x2_mesh_three_components() {
    let grid = KXcGrid::Uniform([24; 3]);
    let fx = fixture(&H2, [2, 2, 2], "PBE", Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let mut worst = 0.0_f64;
    for (i, j) in [(0, 0), (1, 2), (2, 2)] {
        let ep = energy_on(
            &strained(&fx.cell, i, j, FD_H),
            [2, 2, 2],
            "PBE",
            Jk::Dense,
            &grid,
        );
        let em = energy_on(
            &strained(&fx.cell, i, j, -FD_H),
            [2, 2, 2],
            "PBE",
            Jk::Dense,
            &grid,
        );
        worst = worst.max(((ep - em) / (2.0 * FD_H) - s.de_deps[i][j]).abs());
    }
    eprintln!("2x2x2 PBE uniform: worst FD miss {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}

/// The uniform-grid FD miss is the central-FD h^2 truncation (not a missing
/// term): halving h divides it by four. (Artifact hypothesis: a stress error
/// would not depend on h.)
#[test]
#[ignore = "3 steps x 18 SCFs; measured 44 s"]
fn uniform_grid_fd_miss_scales_as_h_squared() {
    let grid = KXcGrid::Uniform([24; 3]);
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
    let s = stress(&fx, None);
    let mut prev = f64::NAN;
    for h in [2e-4, 1e-4, 5e-5] {
        let fd = fd_all9(&fx.cell, [1, 1, 3], "PBE", Jk::Dense, &grid, h);
        let m = mdiff(&fd, &s.de_deps);
        eprintln!(
            "h={h:.1e}: worst miss {m:.3e} (ratio to previous {:.2})",
            prev / m
        );
        if h == 1e-4 {
            assert!((3.0..5.5).contains(&(prev / m)), "ratio {}", prev / m);
        }
        prev = m;
    }
}

// ------------------------------------------------- column rotation (Auto/Off)

/// cc-pVDZ H2, a = 5 Bohr, RS-GDF with cc-pvdz-ri, PBE on a 30^3 uniform grid
/// (smooth under strain): the cell where the SR column rotation actually runs.
/// Returns (energy, stress when asked, hcore rotated columns, fit rotated
/// columns).
fn rotated_case(
    cell: &Cell,
    rot: ferric_pbc::sr_rotation::SrColumnRotation,
    want_stress: bool,
) -> (f64, Option<KRksStress>, usize, usize) {
    let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
    let abs = ferric_core::basis::bundled("cc-pvdz-ri").unwrap();
    let prep = prep_for(cell, &bs);
    let aux = prep_for(cell, &abs);
    let mesh = KPointMesh::gamma_centred(cell, [1, 1, 3]).unwrap();
    let mut cfg = KRksConfig::new(cell, ExxDiv::None, "PBE");
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(rot);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg.krhf.jk = KJkKind::RsGdf;
    let mut kc = kgdf_cfg();
    kc.gdf.sr_column_rotation = rot;
    cfg.krhf.rsgdf = kc;
    let grid = PeriodicGrid::uniform(cell, [30; 3]).unwrap();
    let res = solve_krks_on_grid(cell, &prep, Some(&aux), &mesh, &grid, &cfg).unwrap();
    if !want_stress {
        return (res.scf.energy, None, 0, 0);
    }
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kc).unwrap();
    let s = kpoint_rks_stress(
        cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &hk,
        KStressJk::RsGdf(KRsGdfGradSource {
            gdf: &gdf,
            cfg: &kc,
            aux: &aux,
        }),
        &res.scf,
        "PBE",
        &scfg(&KXcGrid::Uniform([30; 3]), None),
    )
    .unwrap();
    let hc = s.hf.hcore_rotated_columns;
    let fc = s.hf.fit_checks.map_or(0, |c| c.rotated_columns);
    (res.scf.energy, Some(s), hc, fc)
}

/// The strain walks follow the walks the energy ran: with the rotation ON
/// (Auto) the stress matches FD of the ROTATED energy; with Off it matches FD
/// of the unrotated one; the two stresses agree to the screening precision
/// (the rotated-vs-unrotated defects are blind at the default screen, as in
/// `pbc_kstress.rs`).
#[test]
#[ignore = "cc-pVDZ RS-GDF, 2 modes x (1 + 6) SCFs; measured 91 s"]
fn rotated_and_unrotated_sr_walks_are_consistent_with_the_energy() {
    use ferric_pbc::sr_rotation::SrColumnRotation::{Auto, Off};
    let cell = Cell::new(hydrogens(&[[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]]), cubic(5.0)).unwrap();
    let mut st = Vec::new();
    for (name, rot) in [("Auto", Auto), ("Off", Off)] {
        let (_, s, hc, fc) = rotated_case(&cell, rot, true);
        let s = s.unwrap();
        eprintln!("{name}: rotated columns hcore {hc} fit {fc}");
        assert_eq!(hc > 0, name == "Auto");
        assert_eq!(fc > 0, name == "Auto");
        let mut worst = 0.0_f64;
        for (i, j) in [(2, 2), (0, 2), (1, 0)] {
            let h = FD_H;
            let ep = rotated_case(&strained(&cell, i, j, h), rot, false).0;
            let em = rotated_case(&strained(&cell, i, j, -h), rot, false).0;
            let d = s.de_deps[i][j] - (ep - em) / (2.0 * h);
            eprintln!("  ({i},{j}) h={h:e} diff {d:.3e}");
            worst = worst.max(d.abs());
        }
        eprintln!("{name}: analytic vs FD of its own energy {worst:.2e}");
        assert!(worst < FD_BAR, "{name}: {worst:e}");
        st.push(s.de_deps);
    }
    let d = mdiff(&st[0], &st[1]);
    eprintln!("Auto vs Off dE/de difference {d:.2e}");
    assert!(d < 1e-6);
}

// ------------------------------------------- antisymmetric part vs grid size

/// The antisymmetric part of the atom-grid stress against the grid size
/// (ROTATION non-invariance of the lab-fixed Lebedev grid; module doc of
/// `kxcstress`). Reported for four grids (not monotone: 1.1e-4, 7.1e-5,
/// 7.8e-5, 7.6e-6); ASSERTED: the default (75,302) value is below the minimum
/// of the coarser grids. Artifact hypothesis: a bug in the point-anchor terms
/// would not drop on the fine grid.
#[test]
#[ignore = "4 SCFs up to (75,302) grids; measured 26 s"]
fn atom_grid_antisymmetry_shrinks_with_the_grid() {
    let mut prev = f64::INFINITY;
    for (nr, na) in [(20, 50), (30, 110), (40, 110), (75, 302)] {
        let fx = fixture(
            &H2,
            [1, 1, 3],
            "PBE",
            Jk::Dense,
            atom_grid(nr, na, Some(CUTOFF)),
        );
        let s = stress(&fx, None);
        eprintln!(
            "({nr},{na}) PBE 1x1x3: sigma antisymmetry {:.3e}, E_xc {:.10}, {} points",
            s.antisymmetry(),
            s.e_xc,
            s.n_grid_points
        );
        if (nr, na) == (75, 302) {
            assert!(s.antisymmetry() < prev);
        }
        prev = prev.min(s.antisymmetry());
    }
}

// ------------------------------------------- p shell on a triclinic cell

/// Two H atoms of the triclinic `TRI_A` lattice.
const TRI2: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];

/// s + Cartesian p on the non-orthogonal `TRI_A` cell, uniform 24^3 grid
/// (smooth under strain), PBE, dense AFT, 1x1x3 mesh: analytic stress vs
/// central FD of the energy for three components including off-diagonal ones.
/// This is the cheap non-ignored cover of `eval_bloch_strain` with n > 1
/// functions per shell (`mu = off + i`), p-shell first AND Hessian moments
/// (m1, m2) and non-orthogonal lattice phases, none of which the cubic s-only
/// fixtures reach. Artifact hypothesis: a wrong function offset or a p-shell
/// Hessian-moment index error misses by the p-term size (>1e-4), not 5e-8.
#[test]
fn p_shell_triclinic_uniform_grid_stress_matches_fd() {
    let bs = sp_basis_h();
    let cell = Cell::new(hydrogens(&TRI2), TRI_A).expect("triclinic cell");
    let grid = KXcGrid::Uniform([24; 3]);
    let fx = fixture_bs(
        cell,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        grid.clone(),
        ExxDiv::None,
        &bs,
    );
    let s = stress(&fx, None);
    let mut worst = 0.0_f64;
    for (i, j) in [(0, 2), (1, 1), (2, 0)] {
        let e = |h: f64| {
            scf_bs(
                &strained(&fx.cell, i, j, h),
                [1, 1, 3],
                "PBE",
                Jk::Dense,
                &grid,
                ExxDiv::None,
                &bs,
            )
            .scf
            .energy
        };
        let d = (e(FD_H) - e(-FD_H)) / (2.0 * FD_H);
        worst = worst.max((d - s.de_deps[i][j]).abs());
    }
    eprintln!(
        "triclinic s+p uniform: worst FD miss {worst:.2e}; |dE/de| max {:.3e}; antisymmetry {:.1e}",
        mmax(&s.de_deps),
        s.antisymmetry()
    );
    assert!(worst < 2.0 * FD_BAR, "{worst:e}");
    // The p-shell moment terms matter here: dropping the Hessian moment misses.
    let nh = mdiff(
        &stress(&fx, Some(KXcStressMutation::NoHessian)).de_deps,
        &s.de_deps,
    );
    eprintln!("NoHessian miss on the p-shell cell: {nh:.2e}");
    assert!(nh > MUTANT_BAR);
}

// ------------------------------------------------- hybrid SCF, pure stress

/// A PBE0 SCF handed to the PBE stress: the rebuilt F has no `K`, so the
/// exact Fock guard refuses (artifact hypothesis: a guard that only compared
/// the XC part would pass this and return a finite wrong stress).
#[test]
fn a_hybrid_scf_is_refused_by_the_pure_stress() {
    let mut fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, KXcGrid::Uniform([24; 3]));
    fx.functional = "PBE".to_string();
    let e = try_stress_cfg(&fx, &scfg(&fx.grid, None))
        .expect_err("must refuse")
        .to_string();
    eprintln!("PBE0 SCF -> PBE stress: {e}");
    assert!(e.contains("differs"), "{e}");
}
