//! Analytic forces of the k-point closed-shell RKS energy for GLOBAL HYBRID
//! functionals (`ferric_pbc::kxcgrad`, module doc "Global hybrids").
//!
//! Anchors (each with its artifact hypothesis next to the physics hypothesis):
//! a = 1 with a zero XC remainder == the k-point HF force; the force at a fixed
//! density is affine in a; 1x1x1 and the non-TRIM 1x1x3 supercell identities
//! against the Gamma PBE0 gradient per part; central FD of the PBE0 / B3LYP
//! energy (fresh SCF per displacement), dense AFT and RS-GDF, exxdiv none and
//! ewald; Ewald == None force equality; thread bit-identity; the Fock guard
//! from every side; typed refusals; production-wired mutants.
//!
//! NOT covered: range-separated hybrids (refused), meta-GGA, VV10/double
//! hybrids (refused), range-split RS-GDF with a hybrid, heavy atoms / ECP with a
//! hybrid, shifted meshes, open shell, stress.

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
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks, KRksConfig, KRksResult};
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{
    solve_krks_injected, KJkKind, KPointInjection, KPointJk, KPointXc, KScfResult,
};
use ferric_pbc::kxcgrad::{kpoint_null_xc_gradient, kpoint_rks_gradient};
use ferric_pbc::kxcgrad::{KRksGradConfig, KRksGradient, KXcGrid, KXcMutation};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
/// Explicit XC neighbour cutoff shared by every energy and force here: the
/// default depends on the cell and creates one-step energy discontinuities.
const CUTOFF: f64 = 10.0;
/// FD bar (see the FD tests for the measured floors on both sides).
const FD_BAR: f64 = 1e-7;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const ANCHOR_PRECISION: f64 = 1e-12;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Jk {
    Dense,
    Gdf,
}

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

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(4.0)).expect("cell")
}

fn krks_cfg(cell: &Cell, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, exxdiv, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-13;
    cfg.krhf.scf.grad_conv = 1e-10;
    cfg.krhf.scf.max_iter = 200;
    if jk == Jk::Gdf {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = kgdf_cfg();
    }
    cfg.grid = grid_cfg();
    cfg
}

fn scf(
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
    let aux = (jk == Jk::Gdf).then(|| prep_for(&cell, &et_sp_aux()));
    solve_krks(&cell, &prep, aux.as_ref(), &mesh, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KRksResult,
    jk: Jk,
    exxdiv: ExxDiv,
    functional: String,
}

fn fixture(
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

fn gcfg(exxdiv: Option<ExxDiv>, xm: Option<KXcMutation>, gm: Option<KGradMutation>) -> KRksGradConfig {
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
fn force_with(
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
        Jk::Gdf => {
            let kc = kgdf_cfg();
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

fn force(fx: &Fx) -> KRksGradient {
    force_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &gcfg(Some(fx.exxdiv), None, None),
    )
    .unwrap()
}

fn force_mut(fx: &Fx, xm: Option<KXcMutation>, gm: Option<KGradMutation>) -> KRksGradient {
    force_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &gcfg(Some(fx.exxdiv), xm, gm),
    )
    .unwrap()
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
    exxdiv: ExxDiv,
) -> f64 {
    let e = |p: &[[f64; 3]]| scf(p, mesh_n, f, jk, exxdiv).scf.energy;
    (e(&moved(pos, a, x, h)) - e(&moved(pos, a, x, -h))) / (2.0 * h)
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn parts_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

// ------------------------------------------------ zero-XC fixed-density SCF

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

struct Zx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    scf: KScfResult,
    jk: Jk,
    exxdiv: ExxDiv,
}

fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Hartree + `a`/2 exact-exchange k SCF with a zero XC (a = 1 is k-point HF).
fn zero_xc(pos: &[[f64; 3]], mesh_n: [usize; 3], jk: Jk, exxdiv: ExxDiv, a: f64) -> Zx {
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
        Jk::Gdf => {
            gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kgdf_cfg())
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

fn with_jk<R>(z: &Zx, f: impl FnOnce(KGradJk<'_>) -> R) -> R {
    match z.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg())
                .unwrap();
            f(KGradJk::Dense(&eri))
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
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
fn jk_mats(z: &Zx) -> (Vec<Array2<Complex64>>, Vec<Array2<Complex64>>) {
    let n = z.prep.nbasis();
    let nk = z.mesh.nk();
    let mut j: Vec<Array2<Complex64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k = j.clone();
    match z.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg())
                .unwrap();
            eri.jk_builder().build(&z.scf.densities, &mut j, &mut k).unwrap();
        }
        Jk::Gdf => {
            let gdf = KRsGdf::build(&z.cell, &z.prep, &z.aux, &z.mesh, &z.hk.s, &kgdf_cfg())
                .unwrap()
                .with_exxdiv(z.exxdiv);
            gdf.jk_builder().build(&z.scf.densities, &mut j, &mut k).unwrap();
        }
    }
    (j, k)
}

fn kcfg() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// (a) EXACTNESS ANCHOR. a = 1 with a zero XC remainder IS the k-point HF
/// force: same `assemble`, but the KS entry rebuilds F itself (h + J - K/2 with
/// the Madelung shift, exact guard) and takes its exxdiv from the config.
/// Artifact hypothesis: a leaked XC term, a dropped Madelung, a K scaled by
/// anything but 1 would show as a nonzero difference; physics predicts 0 to
/// roundoff (F is hermitised on one side only: ~1e-16).
#[test]
fn a_equal_one_with_zero_xc_is_the_hf_force_per_part() {
    for jk in [Jk::Dense, Jk::Gdf] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let z = zero_xc(&H2, [1, 1, 3], jk, exxdiv, 1.0);
            let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
            let ks = with_jk(&z, |j| {
                kpoint_null_xc_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &z.scf,
                    1.0,
                    &gcfg(Some(exxdiv), None, None),
                )
                .unwrap()
            });
            let hf = with_jk(&z, |j| {
                kpoint_rhf_gradient(
                    &z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, j, &z.scf, exxdiv, &kcfg(),
                )
                .unwrap()
            });
            let (p, q) = (&ks.hf.parts, &hf.parts);
            let rows = [
                ("overlap", parts_diff(&p.overlap, &q.overlap)),
                ("kinetic", parts_diff(&p.kinetic, &q.kinetic)),
                ("vsr", parts_diff(&p.vsr_basis, &q.vsr_basis)),
                ("coulomb", parts_diff(&p.coulomb, &q.coulomb)),
                ("exchange", parts_diff(&p.exchange, &q.exchange)),
                ("fit_orb_sr", parts_diff(&p.fit_orb_sr, &q.fit_orb_sr)),
                ("fit_aux_lr", parts_diff(&p.fit_aux_lr, &q.fit_aux_lr)),
                ("fit_metric_sr", parts_diff(&p.fit_metric_sr, &q.fit_metric_sr)),
                ("fit_g0", parts_diff(&p.fit_g0, &q.fit_g0)),
                ("total", parts_diff(&ks.grad, &hf.grad)),
            ];
            for (n, d) in rows {
                eprintln!("{jk:?} {exxdiv:?}: a=1 null-XC vs HF force, {n:<13} {d:.1e}");
                assert!(d < 1e-13, "{jk:?} {exxdiv:?} {n}: {d:e}");
            }
            // The anchor is not vacuous.
            assert!(max_abs(&hf.grad) > 1e-3);
            match jk {
                Jk::Dense => assert!(max_abs(&hf.parts.exchange) > 1e-4),
                Jk::Gdf => assert!(max_abs(&hf.parts.fit_orb_sr) > 1e-4),
            }
        }
    }
}

/// (b) LINEARITY. At a FIXED density the energy's two-electron part is
/// J-terms + a (K-terms) and F = h + J - (a/2) K: every term of the force (the
/// K walks, the RS-GDF fit's exchange pieces, the M-term, the energy-weighted
/// Pulay term through F) is affine in a. Artifact hypothesis: a scale applied
/// twice (a^2), to J as well, or only to part of the exchange pieces breaks
/// affinity (and, for the fit, the K-only ones, at a different size).
/// Density: the PBE0 SCF's; the stored Fock is replaced by the rebuilt
/// F(a) so the guard passes (the formula, not self-consistency, is under test).
#[test]
fn force_is_affine_in_the_exchange_fraction_at_fixed_density() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let z = zero_xc(&H2, [1, 1, 3], jk, ExxDiv::Ewald, 0.25);
        let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
        // J and K of the fixed density.
        let (jm, km) = jk_mats(&z);
        let at = |a: f64| -> KRksGradient {
            let mut s = z.scf.clone();
            for q in 0..z.mesh.nk() {
                s.fock[q] = &(&z.hk.h[q] + &jm[q]) - &km[q].mapv(|x| x * (0.5 * a));
            }
            with_jk(&z, |j| {
                kpoint_null_xc_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &s,
                    a,
                    &gcfg(Some(ExxDiv::Ewald), None, None),
                )
                .unwrap()
            })
        };
        let (g0, g1, gh, gq) = (at(0.0), at(1.0), at(0.5), at(0.25));
        let aff = |x0: &Array2<f64>, x1: &Array2<f64>, xa: &Array2<f64>, a: f64| {
            parts_diff(xa, &(&x0.mapv(|v| v * (1.0 - a)) + &x1.mapv(|v| v * a)))
        };
        let rows = [
            ("total a=.5", aff(&g0.grad, &g1.grad, &gh.grad, 0.5)),
            ("total a=.25", aff(&g0.grad, &g1.grad, &gq.grad, 0.25)),
            (
                "overlap a=.5",
                aff(&g0.hf.parts.overlap, &g1.hf.parts.overlap, &gh.hf.parts.overlap, 0.5),
            ),
            (
                "exchange a=.5",
                aff(&g0.hf.parts.exchange, &g1.hf.parts.exchange, &gh.hf.parts.exchange, 0.5),
            ),
            (
                "fit_orb_sr a=.5",
                aff(&g0.hf.parts.fit_orb_sr, &g1.hf.parts.fit_orb_sr, &gh.hf.parts.fit_orb_sr, 0.5),
            ),
            (
                "fit_metric_lr a=.25",
                aff(
                    &g0.hf.parts.fit_metric_lr,
                    &g1.hf.parts.fit_metric_lr,
                    &gq.hf.parts.fit_metric_lr,
                    0.25,
                ),
            ),
        ];
        for (n, d) in rows {
            eprintln!("{jk:?}: affinity residual {n:<20} {d:.1e}");
            assert!(d < 1e-12, "{jk:?} {n}: {d:e}");
        }
        // Not vacuous: the a-dependent part is large against the residual, and
        // J-only (a = 0) is far from HF.
        let span = parts_diff(&g0.grad, &g1.grad);
        eprintln!("{jk:?}: |g(1) - g(0)| = {span:.3e}");
        assert!(span > 1e-3);
        assert_eq!(max_abs(&g0.hf.parts.exchange), 0.0);
    }
}

// ------------------------------------------------------------------- Gamma

fn gamma_pbe0(cell: &Cell, grid: &PeriodicGridConfig, f: &str) -> ferric_pbc::grad::GammaGradient {
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

/// (c) 1x1x1 == the Gamma hybrid KS gradient, per part (Ewald exxdiv, so the
/// scaled Madelung M-term is inside `overlap`). Independent construction: real
/// lattice sums vs complex Bloch AOs and a separate SCF. Floor: the two SCFs
/// converge to ~1e-10 in the density.
#[test]
fn one_point_mesh_matches_the_gamma_hybrid_gradient_per_part() {
    for f in ["PBE0", "B3LYP"] {
        let grid = grid_cfg();
        let gam = gamma_pbe0(&cell_of(&H2), &grid, f);
        let fx = fixture(&H2, [1, 1, 1], f, Jk::Dense, ExxDiv::Ewald);
        let k = force(&fx);
        let (p, kp) = (&gam.parts, &k.hf.parts);
        let rows = [
            ("xc_ao", parts_diff(&k.xc_ao, &p.xc_ao)),
            ("xc_point", parts_diff(&k.xc_point, &p.xc_point)),
            ("xc_weight", parts_diff(&k.xc_weight, &p.xc_weight)),
            ("overlap", parts_diff(&kp.overlap, &p.overlap)),
            ("kinetic", parts_diff(&kp.kinetic, &p.kinetic)),
            ("eri=J+K", parts_diff(&(&kp.coulomb + &kp.exchange), &p.eri)),
            ("total", parts_diff(&k.grad, &gam.grad)),
        ];
        for (n, d) in rows {
            eprintln!("{f} 1x1x1 vs Gamma: {n:<10} {d:.2e}");
            assert!(d < 1e-10, "{f} {n}: {d:e}");
        }
        assert!(max_abs(&kp.exchange) > 1e-4, "K walks are exercised");
        assert!(max_abs(&k.xc_weight) > 1e-3);
    }
}

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

/// (d) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh (carries the Bloch phases,
/// and with Ewald the MESH Madelung constant in the scaled M-term: the
/// primitive-cell constant would differ). The k force on atom A is the Gamma
/// force of the explicit supercell on ONE copy of A. Both sides use the same
/// explicit `neighbour_cutoff`.
#[test]
fn k_mesh_hybrid_force_is_the_gamma_supercell_force_per_part() {
    let grid = grid_cfg();
    let f = "PBE0";
    let gam = gamma_pbe0(&supercell_pos(&H2, 4.0, [1, 1, 3]), &grid, f);
    let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, ExxDiv::Ewald);
    let k = force(&fx);
    let first = |a: &Array2<f64>| a.slice(ndarray::s![0..2, ..]).to_owned();
    let kp = &k.hf.parts;
    let rows = [
        ("xc_ao", parts_diff(&k.xc_ao, &first(&gam.parts.xc_ao))),
        ("overlap", parts_diff(&kp.overlap, &first(&gam.parts.overlap))),
        ("eri=J+K", parts_diff(&(&kp.coulomb + &kp.exchange), &first(&gam.parts.eri))),
        ("total", parts_diff(&k.grad, &first(&gam.grad))),
    ];
    for (n, d) in rows {
        eprintln!("{f} 1x1x3 vs supercell: {n:<10} {d:.2e}");
        assert!(d < 1e-9, "{f} {n}: {d:e}");
    }
}

// ----------------------------------------------------------------------- FD

/// (e) Central FD of the hybrid energy (fresh SCF per displacement), h = 1e-4
/// with the shared explicit cutoff (no neighbour discontinuity inside +-h).
/// Bar: see `FD_BAR`. Both exxdiv modes (f): each is checked against its OWN
/// energy's FD, and the two analytic forces must agree (derived equal: the
/// shift is a constant on the occupied space, the density does not change, and
/// its force contribution cancels between K and the M-term).
#[test]
fn pbe0_force_matches_fd_dense_and_rsgdf_both_exxdiv() {
    let mut by_mode: HashMap<(Jk, bool), Array2<f64>> = HashMap::new();
    for jk in [Jk::Dense, Jk::Gdf] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let fx = fixture(&H2, [1, 1, 3], "PBE0", jk, exxdiv);
            let g = force(&fx);
            let mut worst = 0.0_f64;
            for (a, x) in [(0, 2), (1, 1)] {
                let d = fd(&H2, a, x, 1e-4, [1, 1, 3], "PBE0", jk, exxdiv);
                worst = worst.max((g.grad[(a, x)] - d).abs());
            }
            eprintln!("{jk:?} {exxdiv:?}: PBE0 analytic vs FD worst {worst:.2e}");
            assert!(worst < FD_BAR, "{jk:?} {exxdiv:?}: {worst:e}");
            by_mode.insert((jk, exxdiv == ExxDiv::Ewald), g.grad);
        }
        let dd = parts_diff(&by_mode[&(jk, false)], &by_mode[&(jk, true)]);
        eprintln!("{jk:?}: Ewald vs None force difference {dd:.2e}");
        assert!(dd < 1e-9, "{jk:?}: {dd:e}");
    }
    // The RS-GDF fit is really wired: it differs from dense AFT by a fit-sized
    // amount (not 0, not O(1)).
    let d = parts_diff(&by_mode[&(Jk::Dense, false)], &by_mode[&(Jk::Gdf, false)]);
    eprintln!("dense vs RS-GDF PBE0 force difference {d:.2e}");
    assert!(d > 1e-9 && d < 1e-2);
}

/// A second global hybrid with a different a and a different semilocal
/// remainder (B3LYP, a = 0.2): dense AFT, Ewald.
#[test]
fn b3lyp_force_matches_fd() {
    let fx = fixture(&H2, [1, 1, 3], "B3LYP", Jk::Dense, ExxDiv::Ewald);
    assert!((fx.res.exact_exchange_fraction - 0.2).abs() < 1e-12);
    let g = force(&fx);
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 2), (1, 1)] {
        let d = fd(&H2, a, x, 1e-4, [1, 1, 3], "B3LYP", Jk::Dense, ExxDiv::Ewald);
        worst = worst.max((g.grad[(a, x)] - d).abs());
    }
    eprintln!("B3LYP analytic vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}

// -------------------------------------------------------------- bitwise etc.

#[test]
fn hybrid_force_is_bit_identical_across_threads() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let fx = fixture(&H2, [1, 1, 3], "PBE0", jk, ExxDiv::Ewald);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| force(&fx))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.grad, b.grad, "{jk:?}");
        assert_eq!(a.hf.parts.exchange, b.hf.parts.exchange, "{jk:?}");
    }
}

// ---------------------------------------------------- refusals / Fock guard

#[test]
fn refusals_and_the_hybrid_fock_guard() {
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = gcfg(Some(ExxDiv::Ewald), None, None);
    let run = |f: &str, s: &KScfResult, cfg: &KRksGradConfig| force_with(&fx, f, s, cfg);
    // Typed refusals.
    for (name, key) in [
        ("HSE06", "range-separated"),
        ("wB97X-V", "VV10"),
        ("SCAN", "meta-GGA"),
    ] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    // A hybrid needs the exxdiv: a typed error, not a silent default.
    let e = run("PBE0", &fx.res.scf, &gcfg(None, None, None))
        .err()
        .expect("no exxdiv")
        .to_string();
    eprintln!("no exxdiv: {e}");
    assert!(e.contains("exxdiv"));
    // Matching inputs pass at roundoff.
    let ok = run("PBE0", &fx.res.scf, &ok_cfg).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    // Guard, every side: wrong exxdiv; wrong functional with another a and
    // another remainder (B3LYP on a PBE0 SCF); a pure functional's force asked
    // of a hybrid SCF and vice versa; a tamper.
    let pbe = fixture(&H2, [1, 1, 3], "PBE", Jk::Dense, ExxDiv::Ewald);
    let cases: Vec<(&str, Result<KRksGradient, ferric_core::FerricError>)> = vec![
        ("exxdiv None on an Ewald SCF", run("PBE0", &fx.res.scf, &gcfg(Some(ExxDiv::None), None, None))),
        ("B3LYP on a PBE0 SCF", run("B3LYP", &fx.res.scf, &ok_cfg)),
        ("PBE force on a PBE0 SCF", run("PBE", &fx.res.scf, &ok_cfg)),
        ("PBE0 force on a PBE SCF", run("PBE0", &pbe.res.scf, &ok_cfg)),
    ];
    for (n, r) in cases {
        let e = r.err().unwrap_or_else(|| panic!("{n} accepted")).to_string();
        eprintln!("{n}: {e}");
        assert!(e.contains("differs"), "{n}: {e}");
    }
    // Wrong a with the RIGHT remainder, via the zero-XC entry (a = 0.5 asked of
    // an a = 1 SCF).
    let z = zero_xc(&H2, [1, 1, 3], Jk::Dense, ExxDiv::Ewald, 1.0);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let r = with_jk(&z, |j| {
        kpoint_null_xc_gradient(
            &z.cell, &z.prep, &z.mesh, &hcfg, &z.hk, j, &z.scf, 0.5, &ok_cfg,
        )
    });
    let e = r.err().expect("wrong a accepted").to_string();
    eprintln!("wrong a: {e}");
    assert!(e.contains("differs"));
    let mut t = fx.res.scf.clone();
    t.fock[1][(0, 1)] += Complex64::new(1e-6, 0.0);
    assert!(run("PBE0", &t, &ok_cfg).is_err(), "1e-6 tamper must be refused");
    // A pure functional ignores the exxdiv entirely.
    let a = force_with(&pbe, "PBE", &pbe.res.scf, &gcfg(None, None, None)).unwrap();
    let b = force_with(&pbe, "PBE", &pbe.res.scf, &gcfg(Some(ExxDiv::None), None, None)).unwrap();
    assert_eq!(a.grad, b.grad);
    assert_eq!(max_abs(&a.hf.parts.exchange), 0.0);
}

/// Memory: every budget either runs and gives the bit-identical force of the
/// unconstrained run, or fails with a typed error (never a panic or a wrong
/// number). The Fock-guard matrices (D, J, K, F, V_xc) are charged to the
/// ledger after the XC pass (see `kxcgrad` "Memory"); on this tiny cell the XC
/// chunk gate (>= 10 kB) always trips before that 1 kB charge could, so the
/// charge itself is covered by the sizing arithmetic, not by a tripping budget
/// (stated, not claimed).
#[test]
fn budgets_either_reproduce_the_force_or_fail_typed() {
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let full = force(&fx);
    let (mut ok, mut err) = (0, 0);
    let mut b = 20_000usize;
    while b <= AMPLE {
        let mut cfg = gcfg(Some(ExxDiv::Ewald), None, None);
        cfg.grad.budget_bytes = Some(b);
        match force_with(&fx, &fx.functional, &fx.res.scf, &cfg) {
            Ok(g) => {
                ok += 1;
                assert_eq!(g.grad, full.grad, "budget {b}");
            }
            Err(e) => {
                err += 1;
                assert!(e.to_string().contains("budget") || e.to_string().contains("memory"), "{b}: {e}");
            }
        }
        b *= 3;
    }
    eprintln!("budget sweep: {ok} runs identical, {err} typed refusals");
    assert!(ok > 0 && err > 0);
}

// ------------------------------------------------------------------ mutants

const MUTANT_BAR: f64 = 1e-5;

fn miss_g(fx: &Fx, gm: KGradMutation, base: &KRksGradient) -> f64 {
    parts_diff(&force_mut(fx, None, Some(gm)).grad, &base.grad)
}

/// Production-wired defects of the exchange scale (`KGradConfig::mutation`,
/// `KRksGradConfig::mutation`), each against the FD-anchored force. Artifact
/// hypothesis: a mutant that is an IDENTITY on a route/exxdiv must show exactly
/// 0 there (asserted), otherwise it must miss by far more than the bar.
#[test]
fn smoke_mutants_are_caught() {
    use KGradMutation as G;
    // Dense AFT + Ewald: every dense-reachable defect.
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = force(&fx);
    for m in [
        G::ExchDenseUnscaled,
        G::ExchMadelungUnscaled,
        G::ExchJScaled,
        G::ExchScaledTwice,
        G::ExchSpinFactor,
    ] {
        let d = miss_g(&fx, m, &base);
        eprintln!("dense ewald {m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
    // The fit-only defect is an IDENTITY on the dense route.
    assert_eq!(miss_g(&fx, G::ExchFitUnscaled, &base), 0.0);
    // The remainder at full strength (pure PBE instead of the PBE0 remainder).
    let d = parts_diff(
        &force_mut(&fx, Some(KXcMutation::FullStrengthXc), None).grad,
        &base.grad,
    );
    eprintln!("FullStrengthXc: miss {d:.2e}");
    assert!(d > MUTANT_BAR);
    // RS-GDF + Ewald.
    let fg = fixture(&H2, [1, 1, 3], "PBE0", Jk::Gdf, ExxDiv::Ewald);
    let gb = force(&fg);
    for m in [
        G::ExchFitUnscaled,
        G::ExchMadelungUnscaled,
        G::ExchScaledTwice,
        G::ExchSpinFactor,
    ] {
        let d = miss_g(&fg, m, &gb);
        eprintln!("gdf ewald {m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "gdf {m:?} survives: {d:e}");
    }
    // The dense-only defects are identities on RS-GDF (no dense K walk exists).
    assert_eq!(miss_g(&fg, G::ExchDenseUnscaled, &gb), 0.0);
    // exxdiv None: the Madelung defect is an identity (v_M = 0): BLIND by
    // construction, asserted so it cannot be mistaken for coverage.
    let fn0 = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::None);
    let b0 = force(&fn0);
    assert_eq!(miss_g(&fn0, G::ExchMadelungUnscaled, &b0), 0.0);
}
