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
//!
//! WEEKLY HALF: `rsgdf_range_split_rohf_force_and_stress_match_fd` (CI >=
//! 120 s: range-split RS-GDF ROHF force + two strains vs FD with every
//! split-only mutant, 1x1x2) runs in `validation_pbc_kuxcstress_ro_rsgdf.rs`;
//! the helpers are `kuxcstress_ro_rsgdf_support/mod.rs`. Per-commit canaries:
//! `rsgdf_range_split_rohf_stress_zz_matches_fd_canary` (the zz strain vs FD
//! and one split stress mutant), `rsgdf_range_split_rohf_force_1x1x3_sees_the_gamma_kernel_mutant`
//! (split force vs FD) and `rsgdf_range_split_closed_shell_rohf_equals_rhf`.

mod common;
#[path = "kuxcstress_ro_rsgdf_support/mod.rs"]
mod support;

use common::*;
use support::*;

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

/// Per-commit canary of the weekly
/// `rsgdf_range_split_rohf_force_and_stress_match_fd`
/// (`validation_pbc_kuxcstress_ro_rsgdf.rs`): the range-split RS-GDF ROHF
/// stress of the H3 doublet (1x1x2, Ewald) against FD of the split SCF energy
/// on the zz strain only, and the largest split-only stress defect
/// (`SplitNoSmoothOverlap`, 1.6e-2 in the weekly ledger) against the same
/// FD. The split force FD is per commit in
/// `rsgdf_range_split_rohf_force_1x1x3_sees_the_gamma_kernel_mutant`.
#[test]
fn rsgdf_range_split_rohf_stress_zz_matches_fd_canary() {
    SPLIT.with(|s| s.set(Some(SPLIT_LAMBDA)));
    let n = [1, 1, 2];
    let exxdiv = ExxDiv::Ewald;
    let cell = cell_of(&H3);
    assert_split_ran(&cell, n);
    let res = scf_on(&cell, n, None, exxdiv);
    let s = stress_of(&cell, n, None, exxdiv, &res, None, None);
    let e = |c: &Cell| scf_on(c, n, None, exxdiv).scf.energy;
    let (i, j) = (2, 2);
    let fd = (e(&strained(&cell, i, j, FD_H)) - e(&strained(&cell, i, j, -FD_H))) / (2.0 * FD_H);
    let miss = (s.de_deps[i][j] - fd).abs();
    let t = stress_of(
        &cell,
        n,
        None,
        exxdiv,
        &res,
        Some(KStressMutation::SplitNoSmoothOverlap),
        None,
    );
    let mm = (t.de_deps[i][j] - fd).abs();
    eprintln!("split ROHF stress (2,2): FD miss {miss:.2e}; SplitNoSmoothOverlap {mm:.2e}");
    assert!(miss < FD_BAR, "split stress {miss:e}");
    assert!(
        mm > 1e-5,
        "SplitNoSmoothOverlap survives the split FD: {mm:e}"
    );
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
