//! Weekly validation of the OPEN-SHELL (UKS, ROKS) range-separated k-point
//! STRESS on the dense-AFT and the RS-GDF J/K (derivation and hypotheses:
//! `pbc_kuks_rsh_stress.rs`). Every test is `#[ignore = "validation: ..."]`
//! (`ferric-cli/tests/ci_validation_tier_guard.rs`).
//!
//! MEASURED: see the PR body of the change that added this file and the
//! `site/src/reference/validation.md` row "Periodic OPEN-shell (UKS, ROKS)
//! range-separated hybrid STRESS"; each test prints its rows.

mod common;
#[path = "kuks_rsh_force_support/mod.rs"]
mod dsup;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod rs;
#[path = "kuks_rsh_stress_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::kstress::KStressMutation as M;
use ferric_pbc::kuxcstress::KUStressMutation;
use ferric_pbc::rsh::RshParams;
use support::*;

/// Bar on `|analytic − FD(h)|` at `h = 1e-4` (h²-dominated; #553's stress
/// bar at this step).
const FD_BAR: f64 = 1e-6;
/// Bar on the Richardson residue `|(4 FD(h/2) − FD(h))/3 − analytic|`
/// (#553's).
const RICH_BAR: f64 = 2e-8;
const H: f64 = 1e-4;

/// `(FD(h), FD(h/2), Richardson)` misses of one strain component.
fn sweep(fd: &dyn Fn(f64) -> f64, analytic: f64) -> (f64, f64, f64) {
    let (f1, f2) = (fd(H), fd(H / 2.0));
    let rich = (4.0 * f2 - f1) / 3.0;
    (
        (f1 - analytic).abs(),
        (f2 - analytic).abs(),
        (rich - analytic).abs(),
    )
}

/// All nine strain components of the polarized zero-XC energy, CAM-type
/// separation on `TRI_A`, UKS and ROKS, both J/K, both exxdiv: FD at h and
/// h/2 and the Richardson residue.
fn zx_all_nine(route: Route, spin: Spin, exxdiv: ExxDiv) {
    let cell = h3d();
    let rsh = cam_type();
    let z = zx(cell.clone(), (route, spin), Some(rsh), 1.0, exxdiv);
    assert!(z.scf.asym() > 0.1, "polarized");
    let s = zx_stress(&z);
    let (mut w1, mut w2, mut wr) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (i, j) in ALL9 {
        let (m1, m2, mr) = sweep(
            &|h| zx_fd(&cell, (i, j, h), (route, spin), rsh, exxdiv),
            s.de_deps[i][j],
        );
        eprintln!(
            "{route:?} {spin:?} {exxdiv:?} ({i},{j}) {:+.6e}: h {m1:.2e} h/2 {m2:.2e} R {mr:.2e}",
            s.de_deps[i][j]
        );
        w1 = w1.max(m1);
        w2 = w2.max(m2);
        wr = wr.max(mr);
    }
    eprintln!("{route:?} {spin:?} {exxdiv:?}: worst h {w1:.2e} h/2 {w2:.2e} Richardson {wr:.2e}");
    assert!(w1 < FD_BAR, "{w1:e}");
    assert!(wr < RICH_BAR, "{wr:e}");
}

#[test]
#[ignore = "validation: 9 components x 4 fresh dense SCFs, UKS and ROKS, ewald"]
fn zx_fd_all_nine_dense_ewald() {
    zx_all_nine(Route::Dense, Spin::Uks, ExxDiv::Ewald);
    zx_all_nine(Route::Dense, Spin::Roks, ExxDiv::Ewald);
}

#[test]
#[ignore = "validation: 9 components x 4 fresh RS-GDF builds + SCFs, UKS and ROKS, ewald"]
fn zx_fd_all_nine_rsgdf_ewald() {
    zx_all_nine(Route::RsGdf, Spin::Uks, ExxDiv::Ewald);
    zx_all_nine(Route::RsGdf, Spin::Roks, ExxDiv::Ewald);
}

#[test]
#[ignore = "validation: 9 components x 4 fresh SCFs per J/K, UKS, exxdiv none"]
fn zx_fd_all_nine_exxdiv_none() {
    zx_all_nine(Route::Dense, Spin::Uks, ExxDiv::None);
    zx_all_nine(Route::RsGdf, Spin::Uks, ExxDiv::None);
}

/// HSE-type (`ω = 0.11`, `c_lr = 0`) in a 6.5 Bohr cube: the attenuated
/// piece is live there (on `TRI_A` HSE's erf piece is nearly empty), UKS and
/// ROKS, both J/K: three components at h, h/2, Richardson.
#[test]
#[ignore = "validation: HSE-type cube, 3 components x 4 SCFs x 2 spins x 2 J/K"]
fn zx_fd_hse_type_cube() {
    let cell = cell_in(common::cubic(6.5), 2, &H3D);
    let rsh = hse_type();
    for route in [Route::Dense, Route::RsGdf] {
        for spin in [Spin::Uks, Spin::Roks] {
            let z = zx(cell.clone(), (route, spin), Some(rsh), 1.0, ExxDiv::Ewald);
            let s = zx_stress(&z);
            let live = match route {
                Route::Dense => mdiff(
                    &zx_stress_m(&z, Some(M::RshNoKernelSlope), None)
                        .unwrap()
                        .de_deps,
                    &s.de_deps,
                ),
                Route::RsGdf => mmax(&s.hf.parts.fit_attenuated),
            };
            eprintln!("{route:?} {spin:?} HSE-type: attenuated piece {live:.2e}");
            for (i, j) in [(0, 0), (1, 1), (2, 2)] {
                let (m1, m2, mr) = sweep(
                    &|h| zx_fd(&cell, (i, j, h), (route, spin), rsh, ExxDiv::Ewald),
                    s.de_deps[i][j],
                );
                eprintln!(
                    "{route:?} {spin:?} HSE-type ({i},{j}): h {m1:.2e} h/2 {m2:.2e} R {mr:.2e}"
                );
                assert!(m1 < FD_BAR && mr < RICH_BAR, "{m1:e} {mr:e}");
                assert!(
                    live > 100.0 * mr,
                    "attenuated piece resolved: {live:e} vs {mr:e}"
                );
            }
        }
    }
}

/// Equal fractions == global hybrid on the remaining legs: dense ROKS
/// (ewald), and every route/spin under exxdiv none, plus the spin defects
/// against this anchor (the global side is correct, so a spin defect on the
/// RSH side shows).
#[test]
#[ignore = "validation: equal-fraction anchor, 6 legs x 2 SCFs + spin mutants"]
fn equal_fractions_remaining_legs_and_spin_mutants() {
    let c = 0.4;
    let eq = RshParams {
        omega: 0.33,
        c_sr: c,
        c_lr: c,
    };
    let legs = [
        (Route::Dense, Spin::Roks, ExxDiv::Ewald),
        (Route::Dense, Spin::Uks, ExxDiv::None),
        (Route::Dense, Spin::Roks, ExxDiv::None),
        (Route::RsGdf, Spin::Uks, ExxDiv::None),
        (Route::RsGdf, Spin::Roks, ExxDiv::None),
        (Route::Dense, Spin::Uks, ExxDiv::Ewald),
    ];
    for (route, spin, exxdiv) in legs {
        let zr = zx(h3d(), (route, spin), Some(eq), 1.0, exxdiv);
        let zg = zx(h3d(), (route, spin), None, c, exxdiv);
        let sr = zx_stress(&zr);
        let sg = zx_stress(&zg);
        let d = mdiff(&sr.de_deps, &sg.de_deps);
        eprintln!("{route:?} {spin:?} {exxdiv:?}: equal fractions vs global: {d:.2e}");
        assert!(d < ANCHOR_TOL, "{d:e}");
        if (route, spin, exxdiv) == (Route::Dense, Spin::Uks, ExxDiv::Ewald) {
            for (n, m, sm) in [
                ("ExchDropBeta", Some(M::ExchDropBeta), None),
                (
                    "ExchClosedShellFactor",
                    Some(M::ExchClosedShellFactor),
                    None,
                ),
                ("MTermTotalDensity", Some(M::MTermTotalDensity), None),
                ("PulayKSwap", None, Some(KUStressMutation::PulayKSwap)),
            ] {
                let miss = mdiff(&zx_stress_m(&zr, m, sm).unwrap().de_deps, &sg.de_deps);
                eprintln!("  equal-fraction anchor vs {n}: miss {miss:.2e}");
                assert!(miss > 1e-3, "{n}: {miss:e}");
            }
        }
    }
}

/// The full mutation ledger on the polarized zero-XC CAM-type cell (ewald):
/// each defect's miss against the production stress (which is FD-validated
/// to the Richardson bar above), UKS and ROKS, both J/K. ROKS adds its `W_RO`
/// cross-term defect.
#[test]
#[ignore = "validation: full mutation ledger (about 40 stresses)"]
fn mutation_ledger() {
    let rsh = cam_type();
    let mut smallest = f64::INFINITY;
    for route in [Route::Dense, Route::RsGdf] {
        for spin in [Spin::Uks, Spin::Roks] {
            let z = zx(h3d(), (route, spin), Some(rsh), 1.0, ExxDiv::Ewald);
            let s = zx_stress(&z);
            let mut rows = ledger_mutants();
            if spin == Spin::Roks {
                rows.push((
                    "RoWDropCross",
                    None,
                    Some(KUStressMutation::RoWDropCross),
                    None,
                ));
            }
            for (n, m, sm, only) in rows {
                if only.is_some_and(|r| r != route) {
                    continue;
                }
                let miss = match zx_stress_m(&z, m, sm) {
                    Ok(x) => mdiff(&x.de_deps, &s.de_deps),
                    Err(e) => {
                        eprintln!("{route:?} {spin:?} {n}: refused ({e})");
                        continue;
                    }
                };
                eprintln!("{route:?} {spin:?} {n}: miss {miss:.2e}");
                assert!(miss > 1e-5, "{route:?} {spin:?} {n}: {miss:e}");
                smallest = smallest.min(miss);
            }
        }
    }
    eprintln!("smallest ledger miss {smallest:.2e}");
}

/// KS: FD of the polarized HSE06 / CAM-B3LYP energy (the grid of the energy,
/// SSF atom grid (20, 50), explicit neighbour cutoff), UKS and ROKS, both
/// J/K: three components at h, h/2 and the Richardson residue.
fn ks_fd_case(route: Route, spin: Spin) {
    let cell = h3d();
    for f in FUNCTIONALS {
        let k = ks(cell.clone(), (route, spin), f, ExxDiv::Ewald);
        assert!(k.scf.asym() > 0.1, "polarized");
        let s = ks_stress(&k);
        for (i, j) in [(0, 1), (1, 1), (2, 2)] {
            let (m1, m2, mr) = sweep(
                &|h| ks_fd(&cell, (i, j, h), (route, spin), f, ExxDiv::Ewald),
                s.de_deps[i][j],
            );
            eprintln!(
                "{route:?} {spin:?} {f} ({i},{j}) {:+.6e}: h {m1:.2e} h/2 {m2:.2e} R {mr:.2e}",
                s.de_deps[i][j]
            );
            assert!(m1 < FD_BAR, "{f} ({i},{j}) h: {m1:e}");
            assert!(mr < RICH_BAR, "{f} ({i},{j}) Richardson: {mr:e}");
        }
    }
}

#[test]
#[ignore = "validation: KS FD, 2 functionals x 3 components x 4 SCFs, dense UKS"]
fn ks_fd_uks_dense() {
    ks_fd_case(Route::Dense, Spin::Uks);
}

#[test]
#[ignore = "validation: KS FD, 2 functionals x 3 components x 4 SCFs, RS-GDF UKS"]
fn ks_fd_uks_rsgdf() {
    ks_fd_case(Route::RsGdf, Spin::Uks);
}

#[test]
#[ignore = "validation: KS FD, 2 functionals x 3 components x 4 SCFs, dense ROKS"]
fn ks_fd_roks_dense() {
    ks_fd_case(Route::Dense, Spin::Roks);
}

#[test]
#[ignore = "validation: KS FD, 2 functionals x 3 components x 4 SCFs, RS-GDF ROKS"]
fn ks_fd_roks_rsgdf() {
    ks_fd_case(Route::RsGdf, Spin::Roks);
}

/// KS SINGLET: the UKS and ROKS HSE06 / CAM-B3LYP stress on the H2 singlet
/// equals the RKS one (independent SCFs; the closed-shell stress is #503 /
/// #514's), both J/K.
#[test]
#[ignore = "validation: KS singlet anchor, 2 functionals x 2 J/K x 3 SCFs"]
fn ks_singlet_open_is_closed() {
    for route in [Route::Dense, Route::RsGdf] {
        for f in FUNCTIONALS {
            let r = ks_stress(&ks(h2s(), (route, Spin::Rks), f, ExxDiv::Ewald));
            for spin in [Spin::Uks, Spin::Roks] {
                let k = ks(h2s(), (route, spin), f, ExxDiv::Ewald);
                let u = ks_stress(&k);
                let rows = [
                    ("hf", mdiff(&u.hf.de_deps, &r.hf.de_deps)),
                    ("xc_ao", mdiff(&u.xc_ao, &r.xc_ao)),
                    ("xc_weight", mdiff(&u.xc_weight, &r.xc_weight)),
                    ("total", mdiff(&u.de_deps, &r.de_deps)),
                ];
                for (n, d) in rows {
                    eprintln!("{route:?} {f} {spin:?} singlet vs RKS {n}: {d:.2e}");
                    assert!(d < ANCHOR_TOL, "{route:?} {f} {spin:?} {n}: {d:e}");
                }
            }
        }
    }
}

/// Gamma == 1x1x1: `gamma_uks_rsh_stress` (Gamma UKS SCF, dense J and
/// exchange) equals the 1x1x1 k-point UKS stress on the H3 doublet (the
/// dense UKS force file's fixtures).
#[test]
#[ignore = "validation: Gamma vs 1x1x1, 2 functionals x 2 KS SCFs"]
fn gamma_is_the_one_point_mesh() {
    use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUksStressConfig};
    for f in dsup::FUNCTIONALS {
        let g = dsup::gamma_fixture(f, ExxDiv::Ewald);
        let gs = ferric_pbc::gamma_rsh::gamma_uks_rsh_stress(
            &g.cell,
            &g.prep,
            &g.hcfg,
            &g.scf,
            &g.gc,
            &g.ex,
            &ferric_pbc::kstress::KStressConfig {
                budget_bytes: Some(dsup::AMPLE),
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("gamma {f}: {e}"));
        let fx = dsup::fixture(dsup::Kind::Tri(2), &dsup::H3D, [1, 1, 1], f, ExxDiv::Ewald);
        let mut base = base_cfg(ExxDiv::Ewald, None);
        base.grid = ferric_pbc::kxcgrad::KXcGrid::Atom(dsup::grid_cfg());
        base.stress.budget_bytes = Some(dsup::AMPLE);
        let ks = kpoint_uks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &ferric_pbc::hcore::PeriodicHcoreConfig::with_omega(dsup::OMEGA),
            &fx.hk,
            ferric_pbc::kstress::KStressJk::Dense(&fx.eri),
            &fx.res.scf,
            f,
            &KUksStressConfig {
                base,
                spin_mutation: None,
            },
        )
        .unwrap();
        let d = mdiff(&gs.de_deps, &ks.de_deps);
        eprintln!(
            "{f}: Gamma vs 1x1x1 stress {d:.2e} (|dE/de| {:.2e}, Fock guard {:.1e})",
            mmax(&ks.de_deps),
            gs.hf.fock_mismatch
        );
        assert!(d < 1e-9, "{f}: {d:e}");
    }
}

/// The 1x1x2 mesh stress of the H3 doublet equals half the stress of the
/// explicit 2-cell supercell (multiplicity 3, one k-point): dE/dε per
/// primitive cell. HSE06 and CAM-B3LYP, dense AFT (the dense UKS force file's
/// fixtures).
#[test]
#[ignore = "validation: mesh vs supercell, a 6-atom KS SCF per functional"]
fn mesh_is_the_supercell() {
    use ferric_pbc::kuxcstress::{kpoint_uks_stress, KUksStressConfig};
    let st = |fx: &dsup::Fx| {
        let mut base = base_cfg(ExxDiv::Ewald, None);
        base.grid = ferric_pbc::kxcgrad::KXcGrid::Atom(dsup::grid_cfg());
        kpoint_uks_stress(
            &fx.cell,
            &fx.prep,
            &fx.mesh,
            &ferric_pbc::hcore::PeriodicHcoreConfig::with_omega(dsup::OMEGA),
            &fx.hk,
            ferric_pbc::kstress::KStressJk::Dense(&fx.eri),
            &fx.res.scf,
            &fx.f,
            &KUksStressConfig {
                base,
                spin_mutation: None,
            },
        )
        .unwrap()
    };
    for f in dsup::FUNCTIONALS {
        let k = st(&dsup::fixture(
            dsup::Kind::Tri(2),
            &dsup::H3D,
            [1, 1, 2],
            f,
            ExxDiv::Ewald,
        ));
        let sc = st(&dsup::fixture_cell(
            dsup::supercell(&dsup::H3D, [1, 1, 2], 3),
            [1, 1, 1],
            f,
            ExxDiv::Ewald,
        ));
        let mut d = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                d = d.max((k.de_deps[i][j] - 0.5 * sc.de_deps[i][j]).abs());
            }
        }
        eprintln!(
            "{f}: 1x1x2 vs supercell/2: {d:.2e} (|dE/de| {:.2e})",
            mmax(&k.de_deps)
        );
        assert!(d < 1e-9, "{f}: {d:e}");
    }
}
