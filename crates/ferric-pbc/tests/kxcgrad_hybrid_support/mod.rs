//! Shared fixtures and helpers of `pbc_kxcgrad_hybrid.rs` (per commit) and
//! `validation_pbc_kxcgrad_hybrid.rs` (weekly): the H2 k-RKS hybrid
//! fixtures, the force drivers, the zero-XC entry and the mutant bodies.
#![allow(dead_code, unused_imports)]

use crate::common::*;
pub use ferric_core::basis::{BasisSet, Shell};
pub use ferric_pbc::dense_aft::ExxDiv;
pub use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
pub use ferric_pbc::dft::PeriodicGridConfig;
pub use ferric_pbc::dft::{gamma_rks, GammaRksConfig};
pub use ferric_pbc::grad::{
    gamma_rks_gradient_rsgdf, gamma_rks_gradient_with, GammaGradConfig, RsGdfGradSource,
};
pub use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use ferric_pbc::hcore::periodic_hcore;
pub use ferric_pbc::hcore::PeriodicHcoreConfig;
pub use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
pub use ferric_pbc::kdft::{solve_krks, KRksConfig, KRksResult};
pub use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource,
};
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointJk, KPointXc, KScfResult,
};
pub use ferric_pbc::kxcgrad::{kpoint_null_xc_gradient, kpoint_rks_gradient};
pub use ferric_pbc::kxcgrad::{KRksGradConfig, KRksGradient, KXcGrid, KXcMutation};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
pub use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
pub use ferric_pbc::uhf::GammaUhfIntegrals;
pub use ndarray::Array2;
pub use num_complex::Complex64;
pub use std::collections::HashMap;

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
/// Explicit XC neighbour cutoff shared by every energy and force here: the
/// default depends on the cell and creates one-step energy discontinuities.
pub const CUTOFF: f64 = 10.0;
/// FD bar (see the FD tests for the measured floors on both sides).
pub const FD_BAR: f64 = 1e-7;
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
pub const ANCHOR_PRECISION: f64 = 1e-12;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Jk {
    Dense,
    Gdf,
    /// RS-GDF with the range split (lambda = 1).
    GdfSplit,
}

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

pub fn kgdf_cfg_of(jk: Jk) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            range_split: (jk == Jk::GdfSplit).then(|| ferric_pbc::rsgdf::RangeSplit::new(1.0)),
            ..Default::default()
        },
        mutation: None,
    }
}

thread_local! {
    /// (radial, angular) of the XC grid on THIS test thread (default (20, 50);
    /// only the ignored production-class test changes it).
    pub static GRID: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((20, 50)) };
}

pub fn grid_cfg() -> PeriodicGridConfig {
    let (nr, na) = GRID.with(|g| g.get());
    let mut g = PeriodicGridConfig::with_size(nr, na);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

pub fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

pub fn krks_cfg(cell: &Cell, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 200;
    if jk != Jk::Dense {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = kgdf_cfg_of(jk);
    }
    cfg.grid = grid_cfg();
    cfg
}

pub fn scf(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    exxdiv: ExxDiv,
) -> KRksResult {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let cfg = krks_cfg(&cell, functional, jk, exxdiv);
    let aux = (jk != Jk::Dense).then(|| prep_for(&cell, &et_sp_aux()));
    solve_krks(&cell, &prep, aux.as_ref(), &mesh, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

pub struct Fx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub aux: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub res: KRksResult,
    pub jk: Jk,
    pub exxdiv: ExxDiv,
    pub functional: String,
}

pub fn fixture(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    exxdiv: ExxDiv,
) -> Fx {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = scf(pos, mesh_n, functional, jk, exxdiv);
    let cfg = krks_cfg(&cell, functional, jk, exxdiv);
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
        exxdiv,
        functional: functional.to_string(),
    }
}

pub fn gcfg(
    exxdiv: Option<ExxDiv>,
    xm: Option<KXcMutation>,
    gm: Option<KGradMutation>,
) -> KRksGradConfig {
    KRksGradConfig {
        grad: KGradConfig {
            budget_bytes: Some(AMPLE),
            mutation: gm,
            ..Default::default()
        },
        grid: KXcGrid::Atom(grid_cfg()),
        exxdiv,
        mutation: xm,
        ..Default::default()
    }
}

/// The force of `fx` with the J/K object rebuilt the way the SCF built it.
pub fn force_with(
    fx: &Fx,
    functional: &str,
    scf: &KScfResult,
    cfg: &KRksGradConfig,
) -> Result<KRksGradient, ferric_core::FerricError> {
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
            kpoint_rks_gradient(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KGradJk::Dense(&eri),
                scf,
                functional,
                cfg,
            )
        }
        Jk::Gdf | Jk::GdfSplit => {
            let kc = kgdf_cfg_of(fx.jk);
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(fx.exxdiv);
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
                scf,
                functional,
                cfg,
            )
        }
    }
}

pub fn force(fx: &Fx) -> KRksGradient {
    force_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &gcfg(Some(fx.exxdiv), None, None),
    )
    .unwrap()
}

pub fn force_mut(fx: &Fx, xm: Option<KXcMutation>, gm: Option<KGradMutation>) -> KRksGradient {
    force_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &gcfg(Some(fx.exxdiv), xm, gm),
    )
    .unwrap()
}

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

pub fn fd(
    pos: &[[f64; 3]],
    a: usize,
    x: usize,
    h: f64,
    mesh_n: [usize; 3],
    f: &str,
    jk: Jk,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |p: &[[f64; 3]]| scf(p, mesh_n, f, jk, exxdiv).scf.energy;
    (e(&moved(pos, a, x, h)) - e(&moved(pos, a, x, -h))) / (2.0 * h)
}

pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn parts_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

// ------------------------------------------------ zero-XC fixed-density SCF

pub struct ZeroXc(usize);
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

pub struct Zx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub aux: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub scf: KScfResult,
    pub jk: Jk,
    pub exxdiv: ExxDiv,
}

pub fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Hartree + `a`/2 exact-exchange k SCF with a zero XC (a = 1 is k-point HF).
pub fn zero_xc(pos: &[[f64; 3]], mesh_n: [usize; 3], jk: Jk, exxdiv: ExxDiv, a: f64) -> Zx {
    let cell = cell_of(pos);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 200;
    let eri;
    let gdf;
    let jkb: Box<dyn KPointJk + '_> = match jk {
        Jk::Dense => {
            eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, exxdiv, &dense_cfg()).unwrap();
            Box::new(eri.jk_builder())
        }
        Jk::Gdf | Jk::GdfSplit => {
            gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kgdf_cfg_of(jk))
                .unwrap()
                .with_exxdiv(exxdiv);
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
    let scf = solve_krks_injected(&cell, &mesh, &sc, inj, &mut z, a).unwrap();
    assert!(scf.converged);
    Zx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        scf,
        jk,
        exxdiv,
    }
}

pub fn with_jk<R>(z: &Zx, f: impl FnOnce(KGradJk<'_>) -> R) -> R {
    match z.jk {
        Jk::Dense => {
            let eri =
                KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg())
                    .unwrap();
            f(KGradJk::Dense(&eri))
        }
        Jk::Gdf | Jk::GdfSplit => {
            let kc = kgdf_cfg_of(z.jk);
            let gdf = KRsGdf::build(&z.cell, &z.prep, &z.aux, &z.mesh, &z.hk.s, &kc)
                .unwrap()
                .with_exxdiv(z.exxdiv);
            f(KGradJk::RsGdf(KRsGdfGradSource {
                gdf: &gdf,
                cfg: &kc,
                aux: &z.aux,
            }))
        }
    }
}

/// J(k), K(k) (K with the build's Madelung shift) of the SCF density.
pub fn jk_mats(z: &Zx) -> (Vec<Array2<Complex64>>, Vec<Array2<Complex64>>) {
    let n = z.prep.nbasis();
    let nk = z.mesh.nk();
    let mut j: Vec<Array2<Complex64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k = j.clone();
    match z.jk {
        Jk::Dense => {
            let eri =
                KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg())
                    .unwrap();
            eri.jk_builder()
                .build(&z.scf.densities, &mut j, &mut k)
                .unwrap();
        }
        Jk::Gdf | Jk::GdfSplit => {
            let gdf = KRsGdf::build(
                &z.cell,
                &z.prep,
                &z.aux,
                &z.mesh,
                &z.hk.s,
                &kgdf_cfg_of(z.jk),
            )
            .unwrap()
            .with_exxdiv(z.exxdiv);
            gdf.jk_builder()
                .build(&z.scf.densities, &mut j, &mut k)
                .unwrap();
        }
    }
    (j, k)
}

pub fn kcfg() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

// ------------------------------------------------------------------- Gamma

pub fn gamma_pbe0(
    cell: &Cell,
    grid: &PeriodicGridConfig,
    f: &str,
) -> ferric_pbc::grad::GammaGradient {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        ANCHOR_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut gc = GammaRksConfig::new(f);
    gc.grid = grid.clone();
    gc.exxdiv = ExxDiv::Ewald;
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

// ------------------------------------------------------------------ mutants

pub const MUTANT_BAR: f64 = 1e-5;

pub fn miss_g(fx: &Fx, gm: KGradMutation, base: &KRksGradient) -> f64 {
    parts_diff(&force_mut(fx, None, Some(gm)).grad, &base.grad)
}

// ------------------------------------------------------------------ ignored

/// FD step of the production-grid test. At (40, 110) the energy has a hard
/// one-step discontinuity of ~9e-11 Ha along x that a step of 5e-5 or more
/// crosses (the residual of the lateral components then scales as 1/h: 2.2e-7
/// at 2e-4, 4.4e-7 at 1e-4, 8.8e-7 at 5e-5, PBE0; plain PBE the same, 2.9e-7
/// .. 1.2e-6, so it is not a hybrid effect); 2.5e-5 stays inside the smooth
/// window (residual 1e-10), which the SCF noise floor eps/2h ~ 2.4e-10 bounds.
pub const H_STEP: f64 = 2.5e-5;

// -------------------------------------------- RS-GDF Gamma anchor (1x1x1)

/// The Gamma RS-GDF hybrid gradient: separate fit-gradient code
/// (`gamma_rks_gradient_rsgdf`), the same aux basis and RS-GDF config.
pub fn gamma_rsgdf_hybrid(cell: &Cell, f: &str, exxdiv: ExxDiv) -> ferric_pbc::grad::GammaGradient {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let auxp = prep_for(cell, &et_sp_aux());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let gdf =
        RsGdf::build_for_gradient(cell, &prep, &auxp, &hc.s, &kgdf_cfg_of(Jk::Gdf).gdf).unwrap();
    let mut gc = GammaRksConfig::new(f);
    gc.grid = grid_cfg();
    gc.exxdiv = exxdiv;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 200;
    let r = gamma_rks(cell, &prep, &hc, GammaUhfIntegrals::RsGdf(&gdf), &gc).unwrap();
    gamma_rks_gradient_rsgdf(
        cell,
        &prep,
        &hcfg,
        &hc,
        &RsGdfGradSource {
            gdf: &gdf,
            aux: &auxp,
            aux_jac: None,
        },
        &r.scf,
        &gc,
        &GammaGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap()
}

pub fn fit_rows(k: &KRksGradient, g: &ferric_pbc::grad::GammaGradient) -> Vec<(&'static str, f64)> {
    let (kp, gp) = (&k.hf.parts, &g.parts);
    vec![
        ("xc_ao", parts_diff(&k.xc_ao, &gp.xc_ao)),
        ("overlap", parts_diff(&kp.overlap, &gp.overlap)),
        ("fit_orb_sr", parts_diff(&kp.fit_orb_sr, &gp.fit_orb_sr)),
        ("fit_orb_lr", parts_diff(&kp.fit_orb_lr, &gp.fit_orb_lr)),
        ("fit_aux_sr", parts_diff(&kp.fit_aux_sr, &gp.fit_aux_sr)),
        ("fit_aux_lr", parts_diff(&kp.fit_aux_lr, &gp.fit_aux_lr)),
        (
            "fit_metric_sr",
            parts_diff(&kp.fit_metric_sr, &gp.fit_metric_sr),
        ),
        (
            "fit_metric_lr",
            parts_diff(&kp.fit_metric_lr, &gp.fit_metric_lr),
        ),
        ("fit_g0", parts_diff(&kp.fit_g0, &gp.fit_g0)),
        ("total", parts_diff(&k.grad, &gp_total(g))),
    ]
}

pub fn gp_total(g: &ferric_pbc::grad::GammaGradient) -> Array2<f64> {
    g.grad.clone()
}
