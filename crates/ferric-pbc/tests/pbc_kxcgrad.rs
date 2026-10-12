//! Analytic forces of the k-point closed-shell RKS energy for pure LDA/GGA
//! functionals (`ferric_pbc::kxcgrad`; derivation in its module doc).
//!
//! Artifact hypotheses next to physics hypotheses are written at each test.
//!
//! Anchors: null-XC exactness, Hartree-only FD, FD of the PBE energy (dense,
//! RS-GDF, uniform grid, production grid (40,110), 2x2x2), 1x1x1 and 1x1x3
//! supercell identities against the Gamma KS gradient per part, Ewald-exxdiv
//! equality, SR column rotation Auto/Off, a triclinic s+p+d cell (1x2x3 FD
//! and the 1x1x3 supercell), thread/chunk bit-identity, the budget gates, the
//! Fock guard from both sides, refusals and production-wired mutants.
//!
//! NOT covered: heavy atoms and ECPs (the k-point HF ECP force has its own
//! suites; the XC term does not see the ECP, but no KS+ECP cell is run), the
//! default (75, 302) grid (the energy and force use the same grid builder, but
//! no FD is run at that size), shifted meshes, metals/fractional occupations,
//! and anything open shell. The FD smooth-window caveat: the energy has hard
//! one-step discontinuities from the neighbour cutoff (default 10 Bohr), so FD
//! tests use h = 2.5e-5, or a larger explicit cutoff where noted.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::dft::{gamma_rks, GammaRksConfig};
use ferric_pbc::grad::{gamma_rks_gradient_with, GammaGradConfig};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{KGradConfig, KGradJk, KRsGdfGradSource};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointJk, KPointXc, KScfResult,
};
use ferric_pbc::kxcgrad::{kpoint_hartree_gradient, kpoint_null_xc_gradient};
use ferric_pbc::kxcgrad::{
    kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid, KXcMutation,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ferric_pbc::PeriodicGrid;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
/// FD bar. Both sides: every correct force measured here is within 2.3e-9 of
/// the central FD at h = 1e-4 (h = 2.5e-5 where noted: <= 1.4e-9), while the
/// smallest production-wired mutant misses by 8.4e-3 (NoHessian). 1e-7 sits
/// ~40x above the worst correct residual and ~5 orders below the smallest
/// mutant, and equals the k-point HF force FD bar of `pbc_kgrad.rs`.
const FD_BAR: f64 = 1e-7;
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
            let eri = KDenseAftEri::build(
                &h.cell,
                &h.prep,
                &h.mesh,
                &h.hk.s,
                ExxDiv::None,
                &dense_cfg(),
            )
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
        let a = with_jk(&h, |j| {
            kpoint_null_xc_gradient(
                &h.cell,
                &h.prep,
                &h.mesh,
                &hcfg,
                &h.hk,
                j,
                &h.scf,
                &gcfg(&KXcGrid::default(), None),
            )
            .unwrap()
        });
        let g = with_jk(&h, |j| {
            kpoint_hartree_gradient(&h.cell, &h.prep, &h.mesh, &hcfg, &h.hk, j, &h.scf, &kcfg())
                .unwrap()
        });
        let d = (&a.grad - &g.grad)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        let dp = (&a.hf.parts.coulomb - &g.parts.coulomb)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        eprintln!(
            "{jk:?}: null-XC vs Hartree machinery: total {d:.1e}, coulomb {dp:.1e}; |F| {:.3e}",
            max_abs(&g.grad)
        );
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
        assert!(worst < FD_BAR, "{jk:?}: {worst:e}");
    }
}

/// Analytic vs central FD of the k RKS energy: dense AFT and RS-GDF, PBE.
#[test]
fn pbe_force_matches_fd_dense_and_rsgdf() {
    let mut forces = Vec::new();
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
        assert!(worst < FD_BAR, "{jk:?}: {worst:e}");
        forces.push(g.grad);
    }
    // The RS-GDF route is really wired: its fit differs from the dense AFT
    // force by a fit-sized amount (not 0, not O(1)).
    let d = parts_diff(&forces[0], &forces[1]);
    eprintln!("dense vs RS-GDF force difference {d:.2e}");
    assert!(d > 1e-9 && d < 1e-2);
}

fn parts_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn gamma_side(
    cell: &Cell,
    functional: &str,
    grid: &PeriodicGridConfig,
) -> ferric_pbc::grad::GammaGradient {
    gamma_side_bs(cell, functional, grid, &pyscf_sto3g_h())
}

fn gamma_side_bs(
    cell: &Cell,
    functional: &str,
    grid: &PeriodicGridConfig,
    bs: &BasisSet,
) -> ferric_pbc::grad::GammaGradient {
    let prep = prep_for(cell, bs);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        ANCHOR_PRECISION,
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
    gamma_rks_gradient_with(
        cell,
        &prep,
        &hcfg,
        &hc,
        &eri,
        &r.scf,
        &gc,
        &GammaGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap()
}

/// (c) 1x1x1 mesh == the Gamma KS gradient, PER PART (the total can hide a
/// defect of the size of a fit error). Independent construction: real
/// lattice-summed AOs vs complex Bloch AOs; Gamma's own SCF. Floor: both SCFs
/// converge to ~1e-10 in the density.
#[test]
fn one_point_mesh_matches_the_gamma_ks_gradient_per_part() {
    for f in ["LDA", "PBE"] {
        let grid = PeriodicGridConfig::with_size(20, 50);
        let gam = gamma_side(&cell_of(&H2), f, &grid);
        let fx = fixture(&H2, [1, 1, 1], f, Jk::Dense, KXcGrid::Atom(grid));
        let k = force(&fx, None);
        let p = &gam.parts;
        let kp = &k.hf.parts;
        let rows = [
            ("xc_ao", parts_diff(&k.xc_ao, &p.xc_ao)),
            ("xc_point", parts_diff(&k.xc_point, &p.xc_point)),
            ("xc_weight", parts_diff(&k.xc_weight, &p.xc_weight)),
            ("overlap", parts_diff(&kp.overlap, &p.overlap)),
            ("kinetic", parts_diff(&kp.kinetic, &p.kinetic)),
            ("coulomb", parts_diff(&kp.coulomb, &p.eri)),
            ("total", parts_diff(&k.grad, &gam.grad)),
        ];
        for (n, d) in rows {
            eprintln!("{f} 1x1x1 vs Gamma: {n:<10} {d:.2e}");
            assert!(d < 1e-12, "{f} {n}: {d:e}");
        }
        assert!(max_abs(&k.xc_weight) > 1e-3 && max_abs(&k.xc_point) > 1e-3);
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

/// (d) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh: the k force on atom A is
/// the Gamma force of the explicit supercell on ONE copy of A (not the sum).
/// Both sides use the same explicit `neighbour_cutoff` (the default depends on
/// the cell). Carries the Bloch phases (ConjPhase is an identity on TRIM
/// meshes, not here).
#[test]
fn k_mesh_force_is_the_gamma_supercell_force_per_part() {
    let mut grid = PeriodicGridConfig::with_size(20, 50);
    grid.neighbour_cutoff = Some(10.0);
    let f = "PBE";
    let sc = supercell_pos(&H2, 4.0, [1, 1, 3]);
    let gam = gamma_side(&sc, f, &grid);
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, KXcGrid::Atom(grid.clone()));
    let k = force(&fx, None);
    let first = |a: &Array2<f64>| a.slice(ndarray::s![0..2, ..]).to_owned();
    let rows = [
        ("xc_ao", parts_diff(&k.xc_ao, &first(&gam.parts.xc_ao))),
        (
            "xc_point",
            parts_diff(&k.xc_point, &first(&gam.parts.xc_point)),
        ),
        (
            "xc_weight",
            parts_diff(&k.xc_weight, &first(&gam.parts.xc_weight)),
        ),
        ("total", parts_diff(&k.grad, &first(&gam.grad))),
    ];
    for (n, d) in rows {
        eprintln!("{f} 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < 1e-10, "{f} {n}: {d:e}");
    }
}

// ------------------------------------------------------------- uniform grid

/// On a uniform grid the points are fixed in space and the weights constant:
/// X3 and X4 are absent EXACTLY (not small), and the remaining AO-derivative
/// term alone reproduces FD of the energy. Artifact hypothesis: a force that
/// kept the atom-grid terms on a uniform grid would miss by ~0.17 (their size
/// on the atom grid); physics predicts exactly 0 for both.
#[test]
fn uniform_grid_force_has_no_grid_terms_and_matches_fd() {
    let grid = KXcGrid::Uniform([24, 24, 24]);
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, grid.clone());
    let g = force(&fx, None);
    assert_eq!(max_abs(&g.xc_point), 0.0);
    assert_eq!(max_abs(&g.xc_weight), 0.0);
    assert!(max_abs(&g.xc_ao) > 1e-2);
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 0), (0, 2), (1, 1)] {
        let d = fd(&H2, a, x, 1e-4, [1, 1, 3], "PBE", Jk::Dense, &grid);
        worst = worst.max((g.grad[(a, x)] - d).abs());
    }
    eprintln!("uniform 24^3 PBE: analytic vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}

// ------------------------------------------------------- invariants / bitwise

/// Translation invariance on the atom grid (sees the weight response: without
/// X4 the net force is grid-error sized) and a nonzero-force sanity check.
#[test]
fn net_force_vanishes_on_the_atom_grid() {
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, atom_grid(20, 50));
    let g = force(&fx, None);
    let net = (0..3)
        .map(|c| g.grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    eprintln!("net force {net:.2e}, max |F| {:.3e}", max_abs(&g.grad));
    assert!(net < 1e-10, "{net:e}");
    assert!(max_abs(&g.grad) > 1e-2);
    let nw = force(&fx, Some(KXcMutation::NoWeightDeriv));
    let net_nw = (0..3)
        .map(|c| nw.grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    eprintln!("net force without X4 {net_nw:.2e}");
    // BLIND SPOT, asserted: on this cell X1, X3 and X4 each sum to zero over
    // the atoms individually (measured ~1e-17), so the net force cannot see a
    // dropped XC term; only the per-atom anchors (FD, Gamma, supercell) can.
    assert!(net_nw < 1e-10, "{net_nw:e}");
    let np = force(&fx, Some(KXcMutation::NoPointMotion));
    let net_np = (0..3)
        .map(|c| np.grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    eprintln!("net force without X3 {net_np:.2e}");
    assert!(net_np < 1e-10, "{net_np:e}");
}

/// The result does not depend on the thread count or on how many chunks run
/// concurrently (chunk-order reduction): bit-identical.
#[test]
fn force_is_bit_identical_across_threads_and_live_chunks() {
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, atom_grid(20, 50));
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| force(&fx, None))
    };
    let a = run(1);
    let b = run(4);
    assert_eq!(a.grad, b.grad);
    assert_eq!(a.e_xc, b.e_xc);
    eprintln!(
        "1 vs 4 threads: bit-identical; live chunks {} / {}",
        a.live_chunks, b.live_chunks
    );
}

// ------------------------------------------------------------------- memory

/// The derivative pass is pre-flighted on the ledger: a budget that holds the
/// fixed HF arrays but not one AO/Hessian chunk is a typed error naming the
/// quantity; a budget for ONE live chunk gives the bit-identical force of the
/// unconstrained run.
#[test]
fn xc_chunks_are_budgeted_and_the_result_is_budget_independent() {
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, atom_grid(20, 50));
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
    let at = |budget: usize| {
        let mut cfg = gcfg(&fx.grid, None);
        cfg.grad.budget_bytes = Some(budget);
        kpoint_rks_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KGradJk::Dense(&eri),
            &fx.res.scf,
            "PBE",
            &cfg,
        )
    };
    let full = at(AMPLE).unwrap();
    // Budgets just above the fixed arrays: the XC chunk gate trips first.
    let mut saw_err = false;
    let mut b = 300_000usize;
    while b < 500_000 {
        if let Err(err) = at(b) {
            saw_err |= err
                .to_string()
                .contains("k-point XC force AO/Hessian chunk");
        }
        b += b / 8;
    }
    assert!(saw_err, "some budget must fail on the XC chunk gate");
    // Live-chunk cap: bit-identical result with one chunk at a time.
    let mut cfg = gcfg(&fx.grid, None);
    cfg.max_live_chunks = Some(1);
    let one = kpoint_rks_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KGradJk::Dense(&eri),
        &fx.res.scf,
        "PBE",
        &cfg,
    )
    .unwrap();
    assert_eq!(one.live_chunks, 1);
    eprintln!(
        "live chunks {} -> 1, chunk {} points",
        full.live_chunks, one.chunk_points
    );
    assert_eq!(one.grad, full.grad);
}

// ------------------------------------------------- refusals and Fock guard

#[test]
fn refusals_and_the_fock_guard() {
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, atom_grid(20, 50));
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
    let cfg = gcfg(&fx.grid, None);
    let run = |f: &str, scf: &KScfResult| {
        kpoint_rks_gradient(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &hcfg,
            &fx.hk,
            KGradJk::Dense(&eri),
            scf,
            f,
            &cfg,
        )
    };
    for (name, key) in [
        ("PBE0", "global hybrid"),
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
    // The HF entry still refuses a KS result (and says what covers it).
    let hf = ferric_pbc::kgrad::kpoint_rhf_gradient(
        &fx.cell,
        &fx.prep,
        &fx.mesh,
        &hcfg,
        &fx.hk,
        KGradJk::Dense(&eri),
        &fx.res.scf,
        ExxDiv::None,
        &kcfg(),
    );
    assert!(hf.is_err());
    // Guard, both sides: the matching Fock passes at roundoff; a wrong
    // functional (V_xc of LDA vs the stored PBE Fock) and a 1e-6 tamper refuse;
    // a 1e-10 tamper (below the 1e-8 bar) passes.
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
}

// ------------------------------------------------------------------ mutants

const ALL_MUTANTS: [KXcMutation; 10] = [
    KXcMutation::NoAo,
    KXcMutation::NoPointMotion,
    KXcMutation::NoWeightDeriv,
    KXcMutation::NoGga,
    KXcMutation::NoHessian,
    KXcMutation::NoPhase,
    KXcMutation::ConjPhase,
    KXcMutation::WrongFactor,
    KXcMutation::NoInvNk,
    KXcMutation::PulayNoVxc,
];
const MUTANT_BAR: f64 = 1e-5;

fn miss(fx: &Fx, m: KXcMutation, base: &KRksGradient) -> f64 {
    parts_diff(&force(fx, Some(m)).grad, &base.grad)
}

/// Smoke mutants on the non-TRIM 1x1x3 mesh (production-wired seams in
/// `KRksGradConfig::mutation`): each must miss the unmutated (FD- and
/// supercell-anchored) force by far more than the bar.
#[test]
fn smoke_mutants_are_caught() {
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, atom_grid(20, 50));
    let base = force(&fx, None);
    for m in [
        KXcMutation::NoAo,
        KXcMutation::NoPointMotion,
        KXcMutation::NoWeightDeriv,
        KXcMutation::NoHessian,
        KXcMutation::NoPhase,
        KXcMutation::ConjPhase,
        KXcMutation::WrongFactor,
        KXcMutation::NoInvNk,
    ] {
        let d = miss(&fx, m, &base);
        eprintln!("{m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
}

/// The full sweep, with the blind spots asserted (artifact hypotheses: a
/// mutant that is an IDENTITY on a mesh or functional must show exactly 0).
#[test]
#[ignore = "full mutant sweep, 10 mutants + LDA/TRIM/uniform rows; measured 45 s"]
fn full_mutant_sweep() {
    let fx = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, atom_grid(20, 50));
    let base = force(&fx, None);
    for m in ALL_MUTANTS {
        let d = miss(&fx, m, &base);
        eprintln!("PBE 1x1x3 {m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
    // LDA: the GGA mutants are identities.
    let lda = fixture(&H2, [1, 1, 3], "LDA", Jk::Dense, atom_grid(20, 50));
    let lb = force(&lda, None);
    for m in [KXcMutation::NoGga, KXcMutation::NoHessian] {
        let d = miss(&lda, m, &lb);
        eprintln!("LDA 1x1x3 {m:?}: miss {d:.2e} (identity)");
        assert_eq!(d, 0.0);
    }
    // TRIM-only mesh 1x1x2: D(k) real, so the conjugate phase is an identity.
    let trim = fixture(&H2, [1, 1, 2], "PBE", Jk::Dense, atom_grid(20, 50));
    let tb = force(&trim, None);
    let d = miss(&trim, KXcMutation::ConjPhase, &tb);
    eprintln!("PBE 1x1x2 ConjPhase: miss {d:.2e} (TRIM identity)");
    assert!(d < 1e-12);
    // Uniform grid: the grid-term mutants are identities.
    let uni = fixture(
        &H2,
        [1, 1, 3],
        "PBE",
        Jk::Dense,
        KXcGrid::Uniform([24, 24, 24]),
    );
    let ub = force(&uni, None);
    for m in [KXcMutation::NoPointMotion, KXcMutation::NoWeightDeriv] {
        assert_eq!(miss(&uni, m, &ub), 0.0);
    }
}

// ------------------------------------------------- column rotation (Auto/Off)

/// cc-pVDZ H2, a = 5 Bohr, RS-GDF with cc-pvdz-ri, PBE: the cell where the
/// SR column rotation actually runs. Returns (energy, force when asked,
/// hcore rotated columns, fit rotated columns).
fn rotated_case(
    pos: &[[f64; 3]],
    rot: ferric_pbc::sr_rotation::SrColumnRotation,
    want_force: bool,
    neighbour: Option<f64>,
) -> (f64, Option<KRksGradient>, usize, usize) {
    let cell = Cell::new(hydrogens(pos), cubic(5.0)).unwrap();
    let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
    let abs = ferric_core::basis::bundled("cc-pvdz-ri").unwrap();
    let prep = prep_for(&cell, &bs);
    let aux = prep_for(&cell, &abs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let mut cfg = KRksConfig::new(&cell, ExxDiv::None, "PBE");
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(rot);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg.krhf.jk = KJkKind::RsGdf;
    let mut kc = kgdf_cfg();
    kc.gdf.sr_column_rotation = rot;
    cfg.krhf.rsgdf = kc;
    cfg.grid = PeriodicGridConfig::with_size(20, 50);
    cfg.grid.neighbour_cutoff = neighbour;
    let res = solve_krks(&cell, &prep, Some(&aux), &mesh, &cfg).unwrap();
    if !want_force {
        return (res.scf.energy, None, 0, 0);
    }
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kc).unwrap();
    let g = kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &hk,
        KGradJk::RsGdf(KRsGdfGradSource {
            gdf: &gdf,
            cfg: &kc,
            aux: &aux,
        }),
        &res.scf,
        "PBE",
        &gcfg(&KXcGrid::Atom(cfg.grid.clone()), None),
    )
    .unwrap();
    let hc = g.hf.hcore_rotated_columns;
    let fc = g.hf.fit_checks.map_or(0, |c| c.rotated_columns);
    (res.scf.energy, Some(g), hc, fc)
}

/// The derivative walks follow the walks the energy ran: with the rotation
/// ON (Auto) the force matches FD of the ROTATED energy; with Off it matches
/// FD of the unrotated one; the two forces agree to the screening precision
/// (RotOff-type defects are blind by construction at the default screen, as in
/// `pbc_sr_rotation_kderiv.rs`).
#[test]
fn rotated_and_unrotated_sr_walks_are_consistent_with_the_energy() {
    use ferric_pbc::sr_rotation::SrColumnRotation::{Auto, Off};
    let pos = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
    let mut forces = Vec::new();
    for (name, rot) in [("Auto", Auto), ("Off", Off)] {
        let (_, g, hc, fc) = rotated_case(&pos, rot, true, None);
        let g = g.unwrap();
        eprintln!("{name}: rotated columns hcore {hc} fit {fc}");
        assert_eq!(hc > 0, name == "Auto");
        assert_eq!(fc > 0, name == "Auto");
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 2), (1, 1)] {
            // h = 2.5e-5 (inside the smooth window at the DEFAULT neighbour
            // cutoff of 10 Bohr). A probe (`probe_cc_pvdz_fd_step`) found a
            // one-step energy discontinuity of ~7e-10 Ha between h = 5e-5 and
            // 7.5e-5 on this fixture that disappears at a 14 Bohr neighbour
            // cutoff and on a uniform grid and does not move with the AO
            // threshold (1e-12 .. 1e-18): it is the energy's hard neighbour
            // cutoff, not a derivative term. The cutoff-14 6-component check is
            // `rotated_force_all_components_at_larger_cutoff`.
            {
                let h = 2.5e-5;
                let ep = rotated_case(&moved(&pos, a, x, h), rot, false, None).0;
                let em = rotated_case(&moved(&pos, a, x, -h), rot, false, None).0;
                let d = g.grad[(a, x)] - (ep - em) / (2.0 * h);
                eprintln!("  ({a},{x}) h={h:e} diff {d:.3e}");
                worst = worst.max(d.abs());
            }
        }
        eprintln!("{name}: analytic vs FD of its own energy {worst:.2e}");
        assert!(worst < FD_BAR, "{name}: {worst:e}");
        forces.push(g.grad);
    }
    let d = parts_diff(&forces[0], &forces[1]);
    eprintln!("Auto vs Off force difference {d:.2e}");
    assert!(d < 1e-6);
}

/// Production-class grid (40,110): LDA and PBE, dense and RS-GDF, 1x1x3 and a
/// 2x2x2 mesh, force vs FD at h = 2.5e-5 (the atom-grid step window).
#[test]
#[ignore = "production-class grid FD, 4 fixtures x 7 SCFs; measured 45 s"]
fn production_grid_forces_match_fd() {
    let grid = atom_grid(40, 110);
    for (f, jk, mesh) in [
        ("LDA", Jk::Dense, [1, 1, 3]),
        ("PBE", Jk::Dense, [1, 1, 3]),
        ("PBE", Jk::Gdf, [1, 1, 3]),
        ("PBE", Jk::Dense, [2, 2, 2]),
    ] {
        let fx = fixture(&H2, mesh, f, jk, grid.clone());
        let g = force(&fx, None);
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 0), (0, 2), (1, 1)] {
            let d = fd(&H2, a, x, 2.5e-5, mesh, f, jk, &grid);
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("{f} {jk:?} {mesh:?} (40,110): analytic vs FD worst {worst:.2e}");
        assert!(worst < FD_BAR, "{f} {jk:?} {mesh:?}: {worst:e}");
    }
}

/// cc-pVDZ rotation fixture energy with optional neighbour cutoff / AO
/// threshold / uniform grid (probe of the FD step discontinuity).
fn cc_energy(pos: &[[f64; 3]], neighbour: Option<f64>, ao_thr: f64, uniform: Option<usize>) -> f64 {
    let cell = Cell::new(hydrogens(pos), cubic(5.0)).unwrap();
    let bs = ferric_core::basis::bundled("cc-pvdz").unwrap();
    let abs = ferric_core::basis::bundled("cc-pvdz-ri").unwrap();
    let prep = prep_for(&cell, &bs);
    let aux = prep_for(&cell, &abs);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let mut cfg = KRksConfig::new(&cell, ExxDiv::None, "PBE");
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg.krhf.jk = KJkKind::RsGdf;
    cfg.krhf.rsgdf = kgdf_cfg();
    cfg.grid = PeriodicGridConfig::with_size(20, 50);
    cfg.grid.neighbour_cutoff = neighbour;
    cfg.xc.ao_threshold = ao_thr;
    match uniform {
        None => solve_krks(&cell, &prep, Some(&aux), &mesh, &cfg),
        Some(n) => {
            let g = PeriodicGrid::uniform(&cell, [n; 3]).unwrap();
            solve_krks_on_grid(&cell, &prep, Some(&aux), &mesh, &g, &cfg)
        }
    }
    .unwrap()
    .scf
    .energy
}

/// PROBE (not a regression test): where does the one-step energy
/// discontinuity of the cc-pVDZ fixture (atom 1, y) come from? Scan the odd
/// part (E(h) - E(-h))/(2h) and the even part against h for the default
/// settings, a larger neighbour cutoff, tighter / looser AO thresholds and a
/// uniform grid, and see which one moves or removes the step.
#[test]
#[ignore = "probe, ~3 min"]
fn probe_cc_pvdz_fd_step() {
    let pos = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
    let settings: [(&str, Option<f64>, f64, Option<usize>); 5] = [
        ("default", None, 1e-15, None),
        ("neighbour 14", Some(14.0), 1e-15, None),
        ("ao 1e-12", None, 1e-12, None),
        ("ao 1e-18", None, 1e-18, None),
        ("uniform 32^3", None, 1e-15, Some(32)),
    ];
    for (name, nb, thr, uni) in settings {
        let e0 = cc_energy(&pos, nb, thr, uni);
        let mut prev_odd = 0.0;
        let mut line = String::new();
        for i in 1..=8 {
            let h = 2.5e-5 * i as f64;
            let ep = cc_energy(&moved(&pos, 1, 1, h), nb, thr, uni);
            let em = cc_energy(&moved(&pos, 1, 1, -h), nb, thr, uni);
            let odd = (ep - em) / (2.0 * h);
            let even = (ep + em - 2.0 * e0) / (h * h);
            line += &format!(
                " h={h:.1e}: odd {odd:.10} (d {:+.2e}) even/h2 {even:.4};",
                odd - prev_odd
            );
            prev_odd = odd;
        }
        eprintln!("{name}:{line}");
    }
}

// ------------------------------------------------------------ exxdiv = Ewald

fn ks_force_exx(jk: Jk, exx: ExxDiv) -> (f64, KRksGradient) {
    let cell = cell_of(&H2);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let grid = atom_grid(20, 50);
    let mut cfg = krks_cfg(&cell, "PBE", jk, &grid);
    cfg.krhf.exxdiv = exx;
    let res = solve_krks(&cell, &prep, (jk == Jk::Gdf).then_some(&aux), &mesh, &cfg).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let gc = gcfg(&grid, None);
    let g = match jk {
        Jk::Dense => {
            let eri =
                KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, exx, &Default::default()).unwrap();
            kpoint_rks_gradient(
                &cell,
                &prep,
                &mesh,
                &hcfg,
                &hk,
                KGradJk::Dense(&eri),
                &res.scf,
                "PBE",
                &gc,
            )
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kc)
                .unwrap()
                .with_exxdiv(exx);
            kpoint_rks_gradient(
                &cell,
                &prep,
                &mesh,
                &hcfg,
                &hk,
                KGradJk::RsGdf(KRsGdfGradSource {
                    gdf: &gdf,
                    cfg: &kc,
                    aux: &aux,
                }),
                &res.scf,
                "PBE",
                &gc,
            )
        }
    }
    .unwrap();
    (res.scf.energy, g)
}

/// A pure functional has no exchange, so the exchange-divergence treatment of
/// the SCF must not matter: the PBE energy, density and force from an
/// Ewald-exxdiv SCF equal the exxdiv = none ones (dense and RS-GDF). The
/// force entry takes no exxdiv at all (a = 0 removes K and the Madelung
/// M-term), so a force that still depended on it would show here.
#[test]
fn ewald_exxdiv_scf_gives_the_same_pure_functional_force() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let (e0, g0) = ks_force_exx(jk, ExxDiv::None);
        let (e1, g1) = ks_force_exx(jk, ExxDiv::Ewald);
        let dg = parts_diff(&g0.grad, &g1.grad);
        eprintln!(
            "{jk:?}: Ewald vs none: |dE| {:.2e}, force diff {dg:.2e}, bitwise {}",
            (e0 - e1).abs(),
            g0.grad == g1.grad
        );
        assert!((e0 - e1).abs() < 1e-11);
        assert!(dg < 1e-11, "{dg:e}");
    }
}

// ------------------------------------------- triclinic cell, s + p + d basis

/// H basis with an s contraction, a Cartesian p and a Cartesian d primitive
/// (the d shell exercises the AO Hessian beyond l = 1; single primitives with
/// coefficient 1 are already unit-normalised).
fn spd_basis_h() -> BasisSet {
    let mut bs = sp_basis_h();
    bs.shells.get_mut(&1).unwrap().push(Shell {
        l: 2,
        pure: false,
        exponents: vec![0.9],
        coefficients: vec![1.0],
    });
    bs
}

/// Two H atoms of the triclinic `TRI_A` lattice (as `pbc_kgrad.rs` tri2).
const TRI2: [[f64; 3]; 2] = [[0.13, 0.25, 0.31], [0.02, 0.27, 1.66]];
/// Explicit shared neighbour cutoff: both the cell and its supercell need a
/// cutoff above their covering radii, and a cutoff larger than the default
/// removes the hard-mask energy steps from the FD (see the cc-pVDZ probe).
const TRI_CUTOFF: f64 = 14.0;

fn tri_cell(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), TRI_A).expect("triclinic cell")
}

fn tri_case(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: &str,
    want_force: bool,
) -> (f64, Option<KRksGradient>) {
    let cell = tri_cell(pos);
    let bs = spd_basis_h();
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let mut cfg = KRksConfig::new(&cell, ExxDiv::None, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 300;
    cfg.grid = PeriodicGridConfig::with_size(20, 50);
    cfg.grid.neighbour_cutoff = Some(TRI_CUTOFF);
    let res = solve_krks(&cell, &prep, None, &mesh, &cfg).unwrap();
    if !want_force {
        return (res.scf.energy, None);
    }
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &cfg.krhf.hcore).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let g = kpoint_rks_gradient(
        &cell,
        &prep,
        &mesh,
        &cfg.krhf.hcore,
        &hk,
        KGradJk::Dense(&eri),
        &res.scf,
        functional,
        &gcfg(&KXcGrid::Atom(cfg.grid.clone()), None),
    )
    .unwrap();
    (res.scf.energy, Some(g))
}

/// Triclinic (non-orthogonal) cell with s, p and d functions on a mesh with
/// two non-TRIM directions: force vs central FD of the energy (h = 1e-4 is
/// inside the smooth window at the larger neighbour cutoff). Covers the
/// non-orthogonal lattice phases, the p and d AO derivatives and Hessians and
/// a nonzero off-diagonal force, none of which the cubic STO-3G fixtures do.
#[test]
#[ignore = "triclinic s+p+d 1x2x3 FD, 13 SCFs; measured ~24 min wall when run concurrently with the supercell test (each <= 24 min)"]
fn triclinic_spd_force_matches_fd() {
    let mesh = [1, 2, 3];
    let (_, g) = tri_case(&TRI2, mesh, "PBE", true);
    let g = g.unwrap();
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] {
        let h = 1e-4;
        let ep = tri_case(&moved(&TRI2, a, x, h), mesh, "PBE", false).0;
        let em = tri_case(&moved(&TRI2, a, x, -h), mesh, "PBE", false).0;
        let d = g.grad[(a, x)] - (ep - em) / (2.0 * h);
        eprintln!("tri ({a},{x}) analytic {:.9} diff {d:.2e}", g.grad[(a, x)]);
        worst = worst.max(d.abs());
    }
    eprintln!("triclinic s+p+d 1x2x3: analytic vs FD worst {worst:.2e}");
    assert!(max_abs(&g.grad) > 1e-3);
    assert!(worst < FD_BAR, "{worst:e}");
}

/// The same cell on the non-TRIM 1x1x3 mesh against the Gamma force of the
/// explicit supercell (copy 0), per XC part, with the shared cutoff.
#[test]
#[ignore = "triclinic s+p+d 1x1x3 vs the 6-atom Gamma supercell; measured ~24 min wall when run concurrently with the FD test (each <= 24 min)"]
fn triclinic_spd_k_mesh_matches_the_gamma_supercell() {
    let mut grid = PeriodicGridConfig::with_size(20, 50);
    grid.neighbour_cutoff = Some(TRI_CUTOFF);
    let bs = spd_basis_h();
    // Supercell along a3.
    let a3 = TRI_A[2];
    let mut atoms = Vec::new();
    for m in 0..3 {
        for p in TRI2 {
            atoms.push([
                p[0] + m as f64 * a3[0],
                p[1] + m as f64 * a3[1],
                p[2] + m as f64 * a3[2],
            ]);
        }
    }
    let lat = [TRI_A[0], TRI_A[1], [3.0 * a3[0], 3.0 * a3[1], 3.0 * a3[2]]];
    let sc = Cell::new(hydrogens(&atoms), lat).unwrap();
    let gam = gamma_side_bs(&sc, "PBE", &grid, &bs);
    let (_, k) = tri_case(&TRI2, [1, 1, 3], "PBE", true);
    let k = k.unwrap();
    let first = |a: &Array2<f64>| a.slice(ndarray::s![0..2, ..]).to_owned();
    for (n, d) in [
        ("xc_ao", parts_diff(&k.xc_ao, &first(&gam.parts.xc_ao))),
        (
            "xc_point",
            parts_diff(&k.xc_point, &first(&gam.parts.xc_point)),
        ),
        (
            "xc_weight",
            parts_diff(&k.xc_weight, &first(&gam.parts.xc_weight)),
        ),
        ("total", parts_diff(&k.grad, &first(&gam.grad))),
    ] {
        eprintln!("triclinic s+p+d 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < 1e-8, "{n}: {d:e}");
    }
}

/// All six force components of the rotated cc-pVDZ fixture at h = 1e-4, made
/// possible by the 14 Bohr neighbour cutoff (smooth energy, see the probe).
#[test]
#[ignore = "cc-pVDZ rotated RS-GDF, 13 SCFs; measured <= 61 s (run with two other ignored tests)"]
fn rotated_force_all_components_at_larger_cutoff() {
    use ferric_pbc::sr_rotation::SrColumnRotation::Auto;
    let pos = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
    let nb = Some(14.0);
    let (_, g, hc, fc) = rotated_case(&pos, Auto, true, nb);
    let g = g.unwrap();
    assert!(hc > 0 && fc > 0);
    let mut worst = 0.0_f64;
    for a in 0..2 {
        for x in 0..3 {
            let h = 1e-4;
            let ep = rotated_case(&moved(&pos, a, x, h), Auto, false, nb).0;
            let em = rotated_case(&moved(&pos, a, x, -h), Auto, false, nb).0;
            let d = g.grad[(a, x)] - (ep - em) / (2.0 * h);
            eprintln!("rotated cc-pVDZ ({a},{x}) diff {d:.2e}");
            worst = worst.max(d.abs());
        }
    }
    eprintln!("rotated cc-pVDZ, cutoff 14, all components: worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}
