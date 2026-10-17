//! RS-GDF J/K for the k-point ROHF / ROKS stress
//! (`kpoint_rohf_stress` / `kpoint_roks_stress` with `KStressJk::RsGdf`).
//!
//! ARTIFACT-vs-PHYSICS HYPOTHESIS (written before the implementation change):
//! the RS-GDF strain pieces of `kstress::assemble` are generic over `KSpins`
//! (the fit pass pairs `D_alpha`, `D_beta` for `Unrestricted` and
//! `RestrictedOpen` alike, and `W_RO` enters only as the overlap weight), so
//! removing the refusal should be CORRECT if the fit's strain is the exact
//! derivative of the fitted energy. If something RO-specific were missing
//! (e.g. a fit term using the unrestricted `Sum_s D_s F_s D_s`), the closed-shell
//! limit would still pass (it is blind to the W form) but the FD anchor on the
//! open-shell H3 doublet would miss by >> the bar.
//!
//! The same file anchors the FORCES (`kpoint_rohf_gradient` / `kpoint_roks_gradient`
//! with `KGradJk::RsGdf`): FD of atom positions, closed-shell limit vs the RHF
//! RS-GDF force, and the RS-GDF force mutants.
//!
//! ANCHORS: (1) closed-shell ROHF == k-RHF RS-GDF stress per part; (2) FD of
//! the strained RS-GDF ROHF / ROKS energy; (3) mutants of the RS-GDF strain and
//! of `W_RO` are visible against the unmutated stress.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGridConfig;
use ferric_pbc::hcore::kpoint::periodic_hcore_kpts;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kroks::{solve_krohf, solve_kroks, KRohfConfig, KRoksConfig, KRoksResult};
use ferric_pbc::kscf::{solve_krhf, KJkKind, KRhfConfig};
use ferric_pbc::kstress::{kpoint_rhf_stress, KStressConfig, KStressJk, KStressMutation};
use ferric_pbc::kuxcgrad::{
    kpoint_rohf_gradient, kpoint_roks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient,
};
use ferric_pbc::kuxcstress::{
    kpoint_rohf_stress, kpoint_roks_stress, KUStressMutation, KUksStress, KUksStressConfig,
};
use ferric_pbc::kxcgrad::KRksGradConfig;
use ferric_pbc::kxcgrad::KXcGrid;
use ferric_pbc::kxcstress::KRksStressConfig;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::stress::Mat3;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
const CUTOFF: f64 = 14.0;
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const FD_H: f64 = 2.5e-5;
const FD_BAR: f64 = 3e-8;
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

fn mesh_of(cell: &Cell, n: [usize; 3]) -> KPointMesh {
    KPointMesh::gamma_centred(cell, n).unwrap()
}

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(CUTOFF);
    g
}

fn cell_of(pos: &[[f64; 3]]) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = if pos.len() % 2 == 1 { 2 } else { 1 };
    Cell::new(mol, cubic(4.0)).expect("cell")
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

fn rohf_cfg(cell: &Cell, exxdiv: ExxDiv) -> KRohfConfig {
    let mut c = KRohfConfig::for_cell(cell, exxdiv);
    c.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.scf.energy_conv = 1e-13;
    c.scf.grad_conv = 1e-10;
    c.scf.max_iter = 600;
    c.jk = KJkKind::RsGdf;
    c.rsgdf = kgdf_cfg();
    c
}

fn scf_on(cell: &Cell, n: [usize; 3], f: Option<&str>, exxdiv: ExxDiv) -> KRoksResult {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let rc = rohf_cfg(cell, exxdiv);
    let r = match f {
        None => solve_krohf(cell, &prep, Some(&aux), &mesh, &rc),
        Some(f) => {
            let mut c = KRoksConfig::new(cell, exxdiv, f);
            c.rohf = rc;
            if f.contains("PBE0") {
                c.rohf.level_shift = 0.05;
            }
            c.grid = grid_cfg();
            solve_kroks(cell, &prep, Some(&aux), &mesh, &c)
        }
    }
    .unwrap_or_else(|e| panic!("RS-GDF k-ROHF/ROKS {f:?}: {e}"));
    assert!(r.scf.converged);
    r
}

fn stress_of(
    cell: &Cell,
    n: [usize; 3],
    f: Option<&str>,
    exxdiv: ExxDiv,
    res: &KRoksResult,
    hm: Option<KStressMutation>,
    sm: Option<KUStressMutation>,
) -> KUksStress {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let kc = kgdf_cfg();
    let gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kc)
        .unwrap()
        .with_exxdiv(exxdiv);
    let cfg = KUksStressConfig {
        base: KRksStressConfig {
            stress: KStressConfig {
                budget_bytes: Some(AMPLE),
                mutation: hm,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(exxdiv),
            ..Default::default()
        },
        spin_mutation: sm,
    };
    let jk = KStressJk::RsGdf(KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    });
    match f {
        None => kpoint_rohf_stress(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, exxdiv, &cfg),
        Some(f) => kpoint_roks_stress(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, f, &cfg),
    }
    .expect("RS-GDF RO stress")
}

fn mdiff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    let mut e = [[0.0; 3]; 3];
    e[i][j] = h;
    cell.strained(&e).expect("strained cell")
}

fn fd_case(f: Option<&str>, exxdiv: ExxDiv, n: [usize; 3], comps: &[(usize, usize)]) {
    let cell = cell_of(&H3);
    let res = scf_on(&cell, n, f, exxdiv);
    let s = stress_of(&cell, n, f, exxdiv, &res, None, None);
    let mut worst = 0.0_f64;
    for &(i, j) in comps {
        let ep = scf_on(&strained(&cell, i, j, FD_H), n, f, exxdiv)
            .scf
            .energy;
        let em = scf_on(&strained(&cell, i, j, -FD_H), n, f, exxdiv)
            .scf
            .energy;
        let fd = (ep - em) / (2.0 * FD_H);
        let miss = (s.de_deps[i][j] - fd).abs();
        eprintln!(
            "RS-GDF {f:?} {exxdiv:?} eps({i},{j}): analytic {:.9e} FD {fd:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst = worst.max(miss);
    }
    assert!(worst < FD_BAR, "{f:?} {exxdiv:?}: {worst:e}");
}

#[test]
fn rsgdf_rohf_stress_matches_fd_1x1x2() {
    fd_case(None, ExxDiv::Ewald, [1, 1, 2], &[(2, 2)]);
}

#[test]
#[ignore = "RS-GDF ROHF 1x1x3 FD, 3 components x 2 SCFs; measured 31 s"]
fn rsgdf_rohf_stress_matches_fd_1x1x3() {
    fd_case(None, ExxDiv::Ewald, [1, 1, 3], &[(2, 2), (0, 2), (0, 0)]);
}

#[test]
#[ignore = "RS-GDF ROKS PBE0 1x1x3 FD; measured 32 s"]
fn rsgdf_roks_pbe0_stress_matches_fd_1x1x3() {
    fd_case(Some("PBE0"), ExxDiv::Ewald, [1, 1, 3], &[(2, 2), (0, 2)]);
}

#[test]
#[ignore = "RS-GDF ROKS PBE 1x1x3 FD; measured 35 s"]
fn rsgdf_roks_pbe_stress_matches_fd_1x1x3() {
    fd_case(
        Some("PBE"),
        ExxDiv::None,
        [1, 1, 3],
        &[(2, 2), (0, 2), (1, 0)],
    );
}

#[test]
#[ignore = "all-nine FD, ROHF Ewald + ROKS PBE0 (36 SCFs); measured 165 s"]
fn rsgdf_ro_stress_matches_fd_all_nine() {
    for (f, exxdiv) in [(None, ExxDiv::Ewald), (Some("PBE0"), ExxDiv::Ewald)] {
        let n = [1, 1, 3];
        let cell = cell_of(&H3);
        let res = scf_on(&cell, n, f, exxdiv);
        let s = stress_of(&cell, n, f, exxdiv, &res, None, None);
        let mut worst = 0.0_f64;
        for &(i, j) in &ALL9 {
            let ep = scf_on(&strained(&cell, i, j, FD_H), n, f, exxdiv)
                .scf
                .energy;
            let em = scf_on(&strained(&cell, i, j, -FD_H), n, f, exxdiv)
                .scf
                .energy;
            worst = worst.max((s.de_deps[i][j] - (ep - em) / (2.0 * FD_H)).abs());
        }
        eprintln!("RS-GDF {f:?}: all-nine FD miss {worst:.2e}");
        assert!(worst < FD_BAR, "{f:?}: {worst:e}");
    }
}

/// Closed-shell ROHF through RS-GDF equals the k-RHF RS-GDF stress.
#[test]
fn rsgdf_closed_shell_rohf_equals_rhf() {
    let n = [1, 1, 2];
    let cell = cell_of(&H2);
    let exxdiv = ExxDiv::Ewald;
    let res = scf_on(&cell, n, None, exxdiv);
    assert_eq!(res.scf.nalpha, res.scf.nbeta);
    let ro = stress_of(&cell, n, None, exxdiv, &res, None, None);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = mesh_of(&cell, n);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let mut rc = KRhfConfig::for_cell(&cell, exxdiv);
    rc.hcore = hcfg;
    rc.scf.energy_conv = 1e-14;
    rc.scf.grad_conv = 1e-12;
    rc.scf.max_iter = 300;
    rc.jk = KJkKind::RsGdf;
    rc.rsgdf = kgdf_cfg();
    let r = solve_krhf(&cell, &prep, Some(&aux), &mesh, &rc).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let kc = kgdf_cfg();
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kc)
        .unwrap()
        .with_exxdiv(exxdiv);
    let rhf = kpoint_rhf_stress(
        &cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KStressJk::RsGdf(KRsGdfGradSource {
            gdf: &gdf,
            cfg: &kc,
            aux: &aux,
        }),
        &r,
        exxdiv,
        &KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let d = mdiff(&ro.de_deps, &rhf.de_deps);
    eprintln!("RS-GDF closed-shell ROHF vs RHF: {d:.2e}");
    assert!(d < 1e-11, "{d:e}");
    let drop = stress_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        None,
        Some(KUStressMutation::RoWDropCross),
    );
    let dm = mdiff(&drop.de_deps, &rhf.de_deps);
    eprintln!("closed-shell RoWDropCross visible: {dm:.2e}");
    assert!(dm > 1e-5);
}

/// Mutants of the RS-GDF strain and of W_RO on the open-shell 1x1x3 state.
#[test]
#[ignore = "RS-GDF mutant sweep, 1 ROHF SCF + 11 stresses; measured 175 s"]
fn rsgdf_ro_mutants_are_caught() {
    let n = [1, 1, 3];
    let cell = cell_of(&H3);
    let exxdiv = ExxDiv::Ewald;
    let res = scf_on(&cell, n, None, exxdiv);
    let good = stress_of(&cell, n, None, exxdiv, &res, None, None);
    for m in [
        KStressMutation::NoMetric,
        KStressMutation::NoAuxFt,
        KStressMutation::NoAuxFt3,
        KStressMutation::NoJ2G0Vol,
        KStressMutation::NoJ3G0Vol,
        KStressMutation::NoG0,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
        KStressMutation::FitCrossSpin,
    ] {
        let s = stress_of(&cell, n, None, exxdiv, &res, Some(m), None);
        let d = mdiff(&s.de_deps, &good.de_deps);
        eprintln!("mutant {m:?}: {d:.2e}");
        assert!(d > 1e-5, "{m:?} not visible: {d:e}");
    }
    let s = stress_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        None,
        Some(KUStressMutation::RoWDropCross),
    );
    let d = mdiff(&s.de_deps, &good.de_deps);
    eprintln!("RoWDropCross: {d:.2e}");
    assert!(d > 1e-5, "{d:e}");
}

// ------------------------------------------------------------------ forces

fn force_of(
    cell: &Cell,
    n: [usize; 3],
    f: Option<&str>,
    exxdiv: ExxDiv,
    res: &KRoksResult,
    gm: Option<KGradMutation>,
    sm: Option<KUSpinMutation>,
) -> KUksGradient {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(cell, &prep, &mesh, &hcfg).unwrap();
    let kc = kgdf_cfg();
    let gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kc)
        .unwrap()
        .with_exxdiv(exxdiv);
    let cfg = KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                mutation: gm,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv: Some(exxdiv),
            ..Default::default()
        },
        spin_mutation: sm,
    };
    let jk = KGradJk::RsGdf(KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    });
    match f {
        None => kpoint_rohf_gradient(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, exxdiv, &cfg),
        Some(f) => kpoint_roks_gradient(cell, &prep, &mesh, &hcfg, &hk, jk, &res.scf, f, &cfg),
    }
    .expect("RS-GDF RO force")
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

const FD_H_F: f64 = 5e-5;
const FD_BAR_F: f64 = 1e-8;

fn force_fd_case(f: Option<&str>, exxdiv: ExxDiv, comps: &[(usize, usize)]) {
    let n = [1, 1, 3];
    let cell = cell_of(&H3);
    let res = scf_on(&cell, n, f, exxdiv);
    let g = force_of(&cell, n, f, exxdiv, &res, None, None);
    let e = |p: &[[f64; 3]]| scf_on(&cell_of(p), n, f, exxdiv).scf.energy;
    let mut worst = 0.0_f64;
    for &(a, x) in comps {
        let fd = (e(&moved(&H3, a, x, FD_H_F)) - e(&moved(&H3, a, x, -FD_H_F))) / (2.0 * FD_H_F);
        let miss = (g.grad[(a, x)] - fd).abs();
        eprintln!(
            "RS-GDF force {f:?} {exxdiv:?} ({a},{x}): analytic {:.9e} FD {fd:.9e} miss {miss:.2e}",
            g.grad[(a, x)]
        );
        worst = worst.max(miss);
    }
    assert!(worst < FD_BAR_F, "{f:?} {exxdiv:?}: {worst:e}");
}

#[test]
fn rsgdf_rohf_force_matches_fd() {
    force_fd_case(None, ExxDiv::Ewald, &[(0, 2)]);
}

#[test]
#[ignore = "RS-GDF ROHF / ROKS PBE / ROKS PBE0 force FD, 1x1x3, 2 components each; measured 87 s"]
fn rsgdf_ro_force_matches_fd() {
    force_fd_case(None, ExxDiv::Ewald, &[(0, 2), (2, 1)]);
    force_fd_case(Some("PBE"), ExxDiv::None, &[(0, 2), (2, 1)]);
    force_fd_case(Some("PBE0"), ExxDiv::Ewald, &[(0, 2), (2, 1)]);
}

#[test]
fn rsgdf_closed_shell_rohf_force_equals_rhf() {
    let n = [1, 1, 2];
    let cell = cell_of(&H2);
    let exxdiv = ExxDiv::Ewald;
    let res = scf_on(&cell, n, None, exxdiv);
    let ro = force_of(&cell, n, None, exxdiv, &res, None, None);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = mesh_of(&cell, n);
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let mut rc = KRhfConfig::for_cell(&cell, exxdiv);
    rc.hcore = hcfg;
    rc.scf.energy_conv = 1e-14;
    rc.scf.grad_conv = 1e-12;
    rc.scf.max_iter = 300;
    rc.jk = KJkKind::RsGdf;
    rc.rsgdf = kgdf_cfg();
    let r = solve_krhf(&cell, &prep, Some(&aux), &mesh, &rc).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let kc = kgdf_cfg();
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kc)
        .unwrap()
        .with_exxdiv(exxdiv);
    let rhf = kpoint_rhf_gradient(
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
        &r,
        exxdiv,
        &KGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let d = (&ro.grad - &rhf.grad)
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    eprintln!("RS-GDF closed-shell ROHF force vs RHF: {d:.2e}");
    assert!(d < 1e-11, "{d:e}");
    let drop = force_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        None,
        Some(KUSpinMutation::RoWDropCross),
    );
    let dm = (&drop.grad - &rhf.grad)
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    eprintln!("closed-shell force RoWDropCross visible: {dm:.2e}");
    assert!(dm > 1e-5);
}

#[test]
#[ignore = "RS-GDF force mutant sweep, 1 ROHF SCF + 5 forces; measured 76 s"]
fn rsgdf_ro_force_mutants_are_caught() {
    let n = [1, 1, 3];
    let cell = cell_of(&H3);
    let exxdiv = ExxDiv::Ewald;
    let res = scf_on(&cell, n, None, exxdiv);
    let good = force_of(&cell, n, None, exxdiv, &res, None, None);
    let dist = |g: &KUksGradient| {
        (&g.grad - &good.grad)
            .iter()
            .fold(0.0_f64, |m, x| m.max(x.abs()))
    };
    for m in [
        KGradMutation::FitAuxPhase,
        KGradMutation::FitNoMetric,
        KGradMutation::FitNoG0,
    ] {
        let d = dist(&force_of(&cell, n, None, exxdiv, &res, Some(m), None));
        eprintln!("force mutant {m:?}: {d:.2e}");
        assert!(d > 1e-5, "{m:?}: {d:e}");
    }
    let d = dist(&force_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        None,
        Some(KUSpinMutation::RoWDropCross),
    ));
    eprintln!("force RoWDropCross: {d:.2e}");
    assert!(d > 1e-5);
}
