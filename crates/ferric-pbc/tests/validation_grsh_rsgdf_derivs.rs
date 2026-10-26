//! Validation tier (weekly, `--ignored`) of the Gamma-point range-separated
//! hybrid forces and stress on RS-GDF J/K (`gamma_rsh::gamma_*_rsh_gradient`,
//! `gamma_rks_rsh_stress` on a `GammaRshExchange::build_rsgdf` exchange;
//! mgoldey/ferric#492). The per-commit glue anchors and the two-component FD
//! force check are in
//! `pbc_grsh_rsgdf_derivs.rs`.
//!
//! Here: central FD of ferric's own Gamma RS-GDF energy (fresh fits + SCF per
//! point) for the open-shell forces and the closed-shell stress, with an h
//! sweep (h², ~4x per halving); exxdiv none on the force anchor; and the
//! RS-GDF − dense-AFT Gamma RSH force gap against the plain-Coulomb (PBE0)
//! fit gap of the same aux on the 1x1x1 mesh (the attenuated fit adds no
//! error beyond the full-Coulomb fit's order).

mod common;
#[path = "grsh_rsgdf_support/mod.rs"]
mod gs;
#[path = "krsh_rsgdf_force_support/mod.rs"]
mod rs;

use ferric_pbc::dense_aft::ExxDiv;
use gs::*;

/// Gamma == 1x1x1 force under exxdiv none (the per-commit file runs ewald).
#[test]
#[ignore = "validation: Gamma RS-GDF RSH force == 1x1x1 under exxdiv none"]
fn gamma_rsgdf_rsh_force_is_the_one_point_mesh_force_exxdiv_none() {
    for sp in [Sp::Rks, Sp::Uks, Sp::Roks] {
        let x = ExxDiv::None;
        let g = gamma_run(cell_at(sp.atoms()), sp, CAM, x, Jk::RsGdf);
        let gf = gamma_force(&g).unwrap();
        let kf = k1_force(&k1_run(cell_at(sp.atoms()), sp, CAM, x), CAM, x);
        let d = diff(&gf.grad, &kf.grad);
        eprintln!("{sp:?} none: Gamma vs 1x1x1 {d:.2e}");
        assert!(d < 1e-9, "{sp:?}: {d:e}");
    }
}

/// FD of the open-shell (UKS, ROKS) Gamma RS-GDF CAM-B3LYP energy, ewald,
/// three non-zero components, h = 1e-4 and 5e-5. Physics: the h² floor, ~4x per
/// halving; artifact: a spin-factor slip in the glue or the k-point
/// derivative misses by >= 3e-2 (the #555 ledger).
#[test]
#[ignore = "validation: FD of the open-shell Gamma RS-GDF RSH force"]
fn gamma_rsgdf_rsh_open_shell_force_matches_fd() {
    for sp in [Sp::Uks, Sp::Roks] {
        let g = gamma_run(cell_at(&H3D), sp, CAM, ExxDiv::Ewald, Jk::RsGdf);
        let gf = gamma_force(&g).unwrap();
        for (a, x) in [(0, 2), (0, 0), (2, 1)] {
            let f1 = fd_force(sp, CAM, ExxDiv::Ewald, (a, x, 1e-4));
            let f2 = fd_force(sp, CAM, ExxDiv::Ewald, (a, x, 5e-5));
            let (r1, r2) = ((gf.grad[(a, x)] - f1).abs(), (gf.grad[(a, x)] - f2).abs());
            let rich = (gf.grad[(a, x)] - (4.0 * f2 - f1) / 3.0).abs();
            eprintln!(
                "{sp:?} ({a},{x}): analytic {:+.10e} |diff| h=1e-4 {r1:.2e} h=5e-5 {r2:.2e} \
                 Richardson {rich:.2e}",
                gf.grad[(a, x)]
            );
            assert!(gf.grad[(a, x)].abs() > 1e-3, "a non-zero component");
            assert!(r2 < 2e-8, "{sp:?} ({a},{x}): {r2:e}");
            assert!(rich < 2e-9, "{sp:?} ({a},{x}) Richardson: {rich:e}");
        }
    }
}

/// FD of the closed-shell Gamma RS-GDF CAM-B3LYP energy under strain,
/// ewald and none, three components (diagonal, off-diagonal and its
/// transpose), h = 4e-4 and 2e-4, plus their Richardson combination. Physics:
/// the h² truncation (4.5x per halving: 7.0e-8 / 1.57e-8 on ε_zz, ewald;
/// Richardson ≤ 5.0e-9 on every component; bars 5e-8 at h = 2e-4 and 2e-8
/// Richardson, still ~1e5 below the 1.3e-3 attenuated-strain mutant);
/// artifact: the k-point stress ledger (attenuated strain dropped, Madelung
/// strain) misses by >= 1e-1.
///
/// Why not smaller h: below h = 2e-4 the strained-cell KS energy carries a
/// ~7e-9 FD noise floor from the atom-centred grid (ε_zz at h = 1e-4 / 5e-5 /
/// 2.5e-5: 4.7e-10 / 6.8e-9 / 3.0e-10, non-monotone), and the Gamma
/// DENSE-AFT route on the same cell shows the same floor at the same h
/// (9.2e-10 / 6.8e-9 / 2.8e-10; measured 2026-10-26), so it is the grid, not
/// the RS-GDF route.
#[test]
#[ignore = "validation: FD of the Gamma RS-GDF RSH stress"]
fn gamma_rsgdf_rsh_stress_matches_fd() {
    for x in [ExxDiv::Ewald, ExxDiv::None] {
        let g = gamma_run(cell_at(&H2), Sp::Rks, CAM, x, Jk::RsGdf);
        let st = gamma_stress(&g).unwrap();
        for (i, j) in [(2, 2), (0, 1), (1, 0)] {
            let f1 = fd_strain(Sp::Rks, CAM, x, (i, j, 4e-4));
            let f2 = fd_strain(Sp::Rks, CAM, x, (i, j, 2e-4));
            let a = st.de_deps[i][j];
            let (r1, r2) = ((a - f1).abs(), (a - f2).abs());
            let rich = (a - (4.0 * f2 - f1) / 3.0).abs();
            eprintln!(
                "{x:?} eps_{i}{j}: analytic {a:+.10e} |diff| h=4e-4 {r1:.2e} h=2e-4 {r2:.2e} \
                 Richardson {rich:.2e}"
            );
            assert!(r2 < 5e-8, "{x:?} ({i},{j}): {r2:e}");
            assert!(a.abs() > 1e-3, "a non-zero component");
            assert!(rich < 2e-8, "{x:?} ({i},{j}) Richardson: {rich:e}");
        }
    }
}

/// FD of the open-shell UKS Gamma RS-GDF CAM-B3LYP energy under strain
/// (`gamma_uks_rsh_stress` on the RS-GDF exchange), ewald, the closed-shell
/// test's components and steps. This cell's stress is larger (|dE/dε_zz| ≈
/// 1.3), so the h² truncation at h = 2e-4 is larger too (ε_zz: 4.0e-7 /
/// 1.0e-7 at h = 4e-4 / 2e-4, ratio 4.0, Richardson 2.5e-11): bar 2e-7 at
/// h = 2e-4, 2e-8 on Richardson.
#[test]
#[ignore = "validation: FD of the Gamma RS-GDF UKS RSH stress"]
fn gamma_rsgdf_rsh_uks_stress_matches_fd() {
    let x = ExxDiv::Ewald;
    let g = gamma_run(cell_at(&H3D), Sp::Uks, CAM, x, Jk::RsGdf);
    let st = gamma_stress(&g).unwrap();
    for (i, j) in [(2, 2), (0, 1), (1, 0)] {
        let f1 = fd_strain(Sp::Uks, CAM, x, (i, j, 4e-4));
        let f2 = fd_strain(Sp::Uks, CAM, x, (i, j, 2e-4));
        let a = st.de_deps[i][j];
        let (r1, r2) = ((a - f1).abs(), (a - f2).abs());
        let rich = (a - (4.0 * f2 - f1) / 3.0).abs();
        eprintln!(
            "UKS eps_{i}{j}: analytic {a:+.10e} |diff| h=4e-4 {r1:.2e} h=2e-4 {r2:.2e} \
             Richardson {rich:.2e}"
        );
        assert!(a.abs() > 1e-3, "a non-zero component");
        assert!(r2 < 2e-7, "({i},{j}): {r2:e}");
        assert!(rich < 2e-8, "({i},{j}) Richardson: {rich:e}");
    }
}

/// RS-GDF vs the DENSE-AFT oracle: the Gamma RS-GDF RSH force minus the
/// Gamma dense-AFT RSH force (both routed) is a fit error of the same order
/// as the plain-Coulomb (PBE0) RS-GDF − dense force gap of the same aux on
/// the 1x1x1 mesh. Artifact: a wrong or missing attenuated fit derivative
/// (#514 ledger, >= 2e-2) would dwarf the ~1e-5 fit gap.
#[test]
#[ignore = "validation: Gamma RS-GDF RSH force vs the dense-AFT oracle"]
fn gamma_rsgdf_rsh_force_is_dense_to_the_fit_error() {
    let x = ExxDiv::Ewald;
    for sp in [Sp::Rks, Sp::Uks, Sp::Roks] {
        let cell = || cell_at(sp.atoms());
        let rs = gamma_force(&gamma_run(cell(), sp, CAM, x, Jk::RsGdf)).unwrap();
        let de = gamma_force(&gamma_run(cell(), sp, CAM, x, Jk::Dense)).unwrap();
        let gap = diff(&rs.grad, &de.grad);
        let p_rs = k1_force(&k1_run_on(cell(), sp, "PBE0", x, true), "PBE0", x);
        let p_de = k1_force(&k1_run_on(cell(), sp, "PBE0", x, false), "PBE0", x);
        let scale = diff(&p_rs.grad, &p_de.grad);
        eprintln!(
            "{sp:?}: Gamma RS-GDF − dense CAM-B3LYP force {gap:.2e}, PBE0 1x1x1 gap {scale:.2e} \
             (ratio {:.2})",
            gap / scale
        );
        assert!(scale > 1e-8, "the fit gap is resolved");
        assert!(gap < 1.5 * scale, "{sp:?}: {gap:e} vs {scale:e}");
    }
}
