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
use ferric_pbc::kroks::{
    solve_krohf, solve_krohf_keep, solve_kroks, KRohfConfig, KRoksConfig, KRoksResult,
};
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

thread_local! {
    /// `true`: the orbital basis is cc-pVDZ H (s contraction sharing primitives
    /// with an s singlet, plus a p shell), so the default SR column rotation of
    /// the RS-GDF / hcore builds is ACTIVE (it does nothing on STO-3G). Set per
    /// test thread.
    static POLARIZED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// `Some(lambda)`: every RS-GDF build and config of this test thread (SCF,
    /// force, stress) runs the range split at `lambda`. Set per test thread.
    static SPLIT: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
}

fn orbital_basis() -> BasisSet {
    if POLARIZED.with(|p| p.get()) {
        ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz")
    } else {
        pyscf_sto3g_h()
    }
}
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
            range_split: SPLIT
                .with(|s| s.get())
                .map(ferric_pbc::rsgdf::RangeSplit::new),
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
    let prep = prep_for(cell, &orbital_basis());
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
    let prep = prep_for(cell, &orbital_basis());
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
    let prep = prep_for(&cell, &orbital_basis());
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
    let prep = prep_for(cell, &orbital_basis());
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
    let prep = prep_for(&cell, &orbital_basis());
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

// ------------------------------------------------- default ROKS (non-ignored)

/// ROKS PBE RS-GDF force on a 1x1x3 mesh (where the phase slips ConjD / WrongQ
/// are not exact identities, unlike 1x1x2), in the default set so CI covers a
/// Kohn-Sham RS-GDF open-shell path.
#[test]
fn rsgdf_roks_pbe_force_matches_fd_1x1x3() {
    let n = [1, 1, 3];
    let cell = cell_of(&H3);
    let res = scf_on(&cell, n, Some("PBE"), ExxDiv::None);
    let g = force_of(&cell, n, Some("PBE"), ExxDiv::None, &res, None, None);
    let e = |p: &[[f64; 3]]| scf_on(&cell_of(p), n, Some("PBE"), ExxDiv::None).scf.energy;
    let fd = (e(&moved(&H3, 2, 1, FD_H_F)) - e(&moved(&H3, 2, 1, -FD_H_F))) / (2.0 * FD_H_F);
    let miss = (g.grad[(2, 1)] - fd).abs();
    eprintln!("ROKS PBE RS-GDF 1x1x3 force (2,1): miss {miss:.2e}");
    assert!(miss < FD_BAR_F, "{miss:e}");
}

// --------------------------------------------- rotated (p-shell) fixture

/// Step and bars of the rotating-basis test. cc-pVDZ energies carry ~1e-12 of
/// SCF/screening noise, so the FD miss is noise-dominated and FALLS with h
/// (force probe, (0,2)/(2,1): 3.3e-8/6.2e-8 at h = 2.5e-5, 1.4e-8/4.2e-8 at
/// 5e-5, 3.1e-9/7.9e-9 at 1e-4, 6.9e-9/6.0e-9 at 2e-4 where h^2 truncation
/// starts); stress (2,2) at h = 1.25e-5 .. 2e-4: 2.3e-7, 1.4e-7, 8.4e-8, 3.9e-7, 1.6e-6 (noise
/// below, h^2 truncation above; (0,2) <= 4.5e-9 from 5e-5). So force h = 1e-4 with
/// bar 5e-8 and stress h = 5e-5 with bar 3e-7, still >= 3 orders below any mutant.
const ROT_H: f64 = 1e-4;
const ROT_HS: f64 = 5e-5;
const ROT_BAR_F: f64 = 5e-8;
const ROT_BAR_S: f64 = 3e-7;

/// cc-pVDZ H (s + p): the default SR column rotation runs the ROTATED
/// walks (asserted via the kept hcore's rotated-column count).
#[test]
#[ignore = "rotating-basis (cc-pVDZ) RS-GDF ROHF: force FD + 2 stress components on 1x1x2; measured 190 s"]
fn rsgdf_rohf_with_a_rotating_basis_force_and_stress_match_fd() {
    POLARIZED.with(|p| p.set(true));
    let n = [1, 1, 2];
    let exxdiv = ExxDiv::Ewald;
    let cell = cell_of(&H3);
    {
        let prep = prep_for(&cell, &orbital_basis());
        let aux = prep_for(&cell, &et_sp_aux());
        let mesh = mesh_of(&cell, n);
        let (_, kept) =
            solve_krohf_keep(&cell, &prep, Some(&aux), &mesh, &rohf_cfg(&cell, exxdiv)).unwrap();
        assert!(kept.hk.sr_rotated_columns > 0, "rotation inactive");
    }
    let res = scf_on(&cell, n, None, exxdiv);
    let g = force_of(&cell, n, None, exxdiv, &res, None, None);
    let s = stress_of(&cell, n, None, exxdiv, &res, None, None);
    let e = |p: &[[f64; 3]]| scf_on(&cell_of(p), n, None, exxdiv).scf.energy;
    let mut worst_f = 0.0_f64;
    for (a, x) in [(0, 2), (2, 1)] {
        let fd = (e(&moved(&H3, a, x, ROT_H)) - e(&moved(&H3, a, x, -ROT_H))) / (2.0 * ROT_H);
        worst_f = worst_f.max((g.grad[(a, x)] - fd).abs());
    }
    let mut worst_s = 0.0_f64;
    for (i, j) in [(2, 2), (0, 2)] {
        let ep = scf_on(&strained(&cell, i, j, ROT_HS), n, None, exxdiv)
            .scf
            .energy;
        let em = scf_on(&strained(&cell, i, j, -ROT_HS), n, None, exxdiv)
            .scf
            .energy;
        worst_s = worst_s.max((s.de_deps[i][j] - (ep - em) / (2.0 * ROT_HS)).abs());
    }
    eprintln!("rotating-basis (cc-pVDZ) RS-GDF ROHF: force FD miss {worst_f:.2e}, stress FD miss {worst_s:.2e}");
    assert!(worst_f < ROT_BAR_F, "force {worst_f:e}");
    assert!(worst_s < ROT_BAR_S, "stress {worst_s:e}");
}

// ------------------------------------------- range-split RS-GDF (lambda = 1)
//
// The RO force and stress run the same fit pass (`kpoint_fit_gradient`,
// `kstress::assemble`) as the RHF/UKS ones, which differentiate a range-split
// build along its partition. PHYSICS hypothesis: the RO split force/stress is
// the derivative of the split energy (FD at the unsplit floor) and its
// closed-shell limit is the RHF split result to round-off. ARTIFACT
// hypotheses: (1) "split passes because the unsplit code ran" -- guarded by
// `assert_split_ran` (moved primitives on both sides) and by the split-only
// mutants, which delete only split terms and must miss FD; (2) "an RO-specific
// split term is missing" -- the closed-shell limit is blind to `W_RO`, so the
// open-shell H3 FD carries it.

/// lambda = 1 at omega = 1 moves the aux primitives s(0.75), s(0.3), p(0.5)
/// and the STO-3G orbital primitive 0.1689 (as `pbc_kstress.rs`).
const SPLIT_LAMBDA: f64 = 1.0;

/// Panics unless this thread's RS-GDF build runs the range split and moves
/// orbital and aux primitives.
fn assert_split_ran(cell: &Cell, n: [usize; 3]) {
    let prep = prep_for(cell, &orbital_basis());
    let aux = prep_for(cell, &et_sp_aux());
    let mesh = mesh_of(cell, n);
    let hk =
        periodic_hcore_kpts(cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let kc = kgdf_cfg();
    assert!(kc.gdf.range_split.is_some(), "SPLIT not set on this thread");
    let gdf = KRsGdf::build(cell, &prep, &aux, &mesh, &hk.s, &kc).unwrap();
    let moved = |name: &str| {
        gdf.stats()
            .split_counters
            .iter()
            .find(|(c, _)| *c == name)
            .map_or(0, |&(_, v)| v)
    };
    let (o, a) = (
        moved("rsgdf split orbital prims smooth"),
        moved("rsgdf split aux prims smooth"),
    );
    eprintln!("split build: moved orbital {o}, aux {a}");
    assert!(o > 0 && a > 0, "split path did not move primitives");
}

/// The split-only stress terms (deleting one changes only the split build).
const SPLIT_STRESS_MUTANTS: [KStressMutation; 6] = [
    KStressMutation::SplitNoSrKernelStrain,
    KStressMutation::SplitNoSmoothPair,
    KStressMutation::SplitFullG0,
    KStressMutation::SplitNoSmoothOverlap,
    KStressMutation::SplitFlipSrSlope,
    KStressMutation::SplitNoSmoothVolume,
];

/// The split-only force terms.
const SPLIT_FORCE_MUTANTS: [KGradMutation; 3] = [
    KGradMutation::SplitGammaKernel,
    KGradMutation::SplitNoSmoothOverlap,
    KGradMutation::SplitFullG0,
];

/// Range-split RS-GDF ROHF (H3 doublet, 1x1x2, Ewald): the force and stress
/// against FD of the split SCF energy, and every split-only mutant missing
/// the same FD. Measured (release): stress FD miss 2.4e-9 (2,2) / 3.0e-10
/// (0,2) (bar 3e-8), force (0,2) 2.1e-10 (bar 1e-8). Mutation ledger (miss
/// against the same FD): stress SplitNoSrKernelStrain 5.6e-3,
/// SplitNoSmoothPair 6.5e-3, SplitFullG0 1.7e-3, SplitNoSmoothOverlap 1.6e-2,
/// SplitFlipSrSlope 1.1e-2, SplitNoSmoothVolume 3.6e-3; force SplitGammaKernel
/// 4.6e-5 (smallest; 1x1x2 is near its blind 1x1x1 limit), SplitNoSmoothOverlap
/// 5.3e-4, SplitFullG0 8.2e-3. Closed-shell ROHF vs RHF on the split build:
/// stress 3.3e-13, force 2.3e-13 (bar 1e-11); split mutant 1.0e-2 / 1.5e-3.
/// ROKS PBE0 1x1x3 (ignored): stress 1.3e-9, force 3.7e-10.
#[test]
fn rsgdf_range_split_rohf_force_and_stress_match_fd() {
    SPLIT.with(|s| s.set(Some(SPLIT_LAMBDA)));
    let n = [1, 1, 2];
    let exxdiv = ExxDiv::Ewald;
    let cell = cell_of(&H3);
    assert_split_ran(&cell, n);
    let res = scf_on(&cell, n, None, exxdiv);
    let s = stress_of(&cell, n, None, exxdiv, &res, None, None);
    let g = force_of(&cell, n, None, exxdiv, &res, None, None);
    let e = |c: &Cell| scf_on(c, n, None, exxdiv).scf.energy;
    let mut fd_s = Vec::new();
    let mut worst_s = 0.0_f64;
    for (i, j) in [(2, 2), (0, 2)] {
        let fd =
            (e(&strained(&cell, i, j, FD_H)) - e(&strained(&cell, i, j, -FD_H))) / (2.0 * FD_H);
        let miss = (s.de_deps[i][j] - fd).abs();
        eprintln!(
            "split ROHF stress ({i},{j}): analytic {:.9e} FD {fd:.9e} miss {miss:.2e}",
            s.de_deps[i][j]
        );
        worst_s = worst_s.max(miss);
        fd_s.push(((i, j), fd));
    }
    let (a, x) = (0, 2);
    let fd_f = (e(&cell_of(&moved(&H3, a, x, FD_H_F))) - e(&cell_of(&moved(&H3, a, x, -FD_H_F))))
        / (2.0 * FD_H_F);
    let miss_f = (g.grad[(a, x)] - fd_f).abs();
    eprintln!(
        "split ROHF force ({a},{x}): analytic {:.9e} FD {fd_f:.9e} miss {miss_f:.2e}",
        g.grad[(a, x)]
    );
    assert!(worst_s < FD_BAR, "split stress {worst_s:e}");
    assert!(miss_f < FD_BAR_F, "split force {miss_f:e}");
    for m in SPLIT_STRESS_MUTANTS {
        let t = stress_of(&cell, n, None, exxdiv, &res, Some(m), None);
        let mm = fd_s.iter().fold(0.0_f64, |w, &((i, j), fd)| {
            w.max((t.de_deps[i][j] - fd).abs())
        });
        eprintln!("  split stress mutant {m:?}: FD miss {mm:.2e}");
        assert!(mm > 1e-5, "{m:?} survives the split FD: {mm:e}");
    }
    for m in SPLIT_FORCE_MUTANTS {
        let t = force_of(&cell, n, None, exxdiv, &res, Some(m), None);
        let mm = (t.grad[(a, x)] - fd_f).abs();
        eprintln!("  split force mutant {m:?}: FD miss {mm:.2e}");
        assert!(mm > 1e-5, "{m:?} survives the split FD: {mm:e}");
    }
}

/// Closed-shell ROHF on the range-split build equals the k-RHF split force
/// and stress (the RHF split paths are anchored in `pbc_kstress.rs` /
/// `pbc_kxcgrad_hybrid.rs`).
#[test]
fn rsgdf_range_split_closed_shell_rohf_equals_rhf() {
    SPLIT.with(|s| s.set(Some(SPLIT_LAMBDA)));
    let n = [1, 1, 2];
    let cell = cell_of(&H2);
    let exxdiv = ExxDiv::Ewald;
    assert_split_ran(&cell, n);
    let res = scf_on(&cell, n, None, exxdiv);
    assert_eq!(res.scf.nalpha, res.scf.nbeta);
    let ro_s = stress_of(&cell, n, None, exxdiv, &res, None, None);
    let ro_g = force_of(&cell, n, None, exxdiv, &res, None, None);
    let prep = prep_for(&cell, &orbital_basis());
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
    let src = || KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kc,
        aux: &aux,
    };
    let rhf_s = kpoint_rhf_stress(
        &cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KStressJk::RsGdf(src()),
        &r,
        exxdiv,
        &KStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let rhf_g = kpoint_rhf_gradient(
        &cell,
        &prep,
        &mesh,
        &hcfg,
        &hk,
        KGradJk::RsGdf(src()),
        &r,
        exxdiv,
        &KGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let ds = mdiff(&ro_s.de_deps, &rhf_s.de_deps);
    let dg = (&ro_g.grad - &rhf_g.grad)
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    eprintln!("split closed-shell ROHF vs RHF: stress {ds:.2e}, force {dg:.2e}");
    assert!(ds < 1e-11, "stress {ds:e}");
    assert!(dg < 1e-11, "force {dg:e}");
    // Both sides: a split-only defect on the RO side misses the RHF reference.
    let m = stress_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        Some(KStressMutation::SplitNoSmoothOverlap),
        None,
    );
    let dm = mdiff(&m.de_deps, &rhf_s.de_deps);
    let mg = force_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        Some(KGradMutation::SplitNoSmoothOverlap),
        None,
    );
    let dmg = (&mg.grad - &rhf_g.grad)
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    eprintln!("closed-shell split mutant visible: stress {dm:.2e}, force {dmg:.2e}");
    assert!(dm > 1e-5 && dmg > 1e-5, "{dm:e} {dmg:e}");
}

/// `SplitGammaKernel` (the moved blocks' weights with the Gamma kernel) is
/// blind at 1x1x1 and only weakly visible at 1x1x2 (4.6e-5 above): on a 1x1x3
/// mesh (non-TRIM q) the split ROHF force against FD of the split energy, and
/// the mutant against the same FD.
#[test]
fn rsgdf_range_split_rohf_force_1x1x3_sees_the_gamma_kernel_mutant() {
    SPLIT.with(|s| s.set(Some(SPLIT_LAMBDA)));
    let n = [1, 1, 3];
    let exxdiv = ExxDiv::Ewald;
    let cell = cell_of(&H3);
    assert_split_ran(&cell, n);
    let res = scf_on(&cell, n, None, exxdiv);
    let g = force_of(&cell, n, None, exxdiv, &res, None, None);
    let (a, x) = (0, 2);
    let e = |p: &[[f64; 3]]| scf_on(&cell_of(p), n, None, exxdiv).scf.energy;
    let fd = (e(&moved(&H3, a, x, FD_H_F)) - e(&moved(&H3, a, x, -FD_H_F))) / (2.0 * FD_H_F);
    let miss = (g.grad[(a, x)] - fd).abs();
    let m = force_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        Some(KGradMutation::SplitGammaKernel),
        None,
    );
    let mm = (m.grad[(a, x)] - fd).abs();
    eprintln!("split ROHF 1x1x3 force ({a},{x}): FD miss {miss:.2e}; SplitGammaKernel {mm:.2e}");
    assert!(miss < FD_BAR_F, "{miss:e}");
    assert!(mm > 1e-5, "SplitGammaKernel survives at 1x1x3: {mm:e}");
}

/// Range-split ROKS PBE0 (Ewald) on the 1x1x3 mesh: force and stress against
/// FD of the split SCF energy (the Kohn-Sham + exchange-scale path).
#[test]
#[ignore = "split RS-GDF ROKS PBE0 1x1x3: 1 SCF + 6 FD SCFs on a 20x50 atom grid"]
fn rsgdf_range_split_roks_pbe0_force_and_stress_match_fd() {
    SPLIT.with(|s| s.set(Some(SPLIT_LAMBDA)));
    let n = [1, 1, 3];
    let f = Some("PBE0");
    let exxdiv = ExxDiv::Ewald;
    let cell = cell_of(&H3);
    assert_split_ran(&cell, n);
    let res = scf_on(&cell, n, f, exxdiv);
    let s = stress_of(&cell, n, f, exxdiv, &res, None, None);
    let g = force_of(&cell, n, f, exxdiv, &res, None, None);
    let e = |c: &Cell| scf_on(c, n, f, exxdiv).scf.energy;
    let mut worst_s = 0.0_f64;
    for (i, j) in [(2, 2), (0, 2)] {
        let fd =
            (e(&strained(&cell, i, j, FD_H)) - e(&strained(&cell, i, j, -FD_H))) / (2.0 * FD_H);
        worst_s = worst_s.max((s.de_deps[i][j] - fd).abs());
    }
    let fd_f = (e(&cell_of(&moved(&H3, 2, 1, FD_H_F))) - e(&cell_of(&moved(&H3, 2, 1, -FD_H_F))))
        / (2.0 * FD_H_F);
    let miss_f = (g.grad[(2, 1)] - fd_f).abs();
    eprintln!("split ROKS PBE0 1x1x3: stress FD miss {worst_s:.2e}, force FD miss {miss_f:.2e}");
    assert!(worst_s < FD_BAR, "stress {worst_s:e}");
    assert!(miss_f < FD_BAR_F, "force {miss_f:e}");
}
