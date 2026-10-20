//! Weekly validation of the range-separated k-point RKS STRESS (dense AFT):
//! central FD of ferric's own energy (fresh SCF per strained cell) for all
//! nine components, HSE06 and CAM-B3LYP, exxdiv ewald and none; the FD step
//! sweep (h^2 behaviour); and the anchor rows the per-commit file
//! (`pbc_krsh_stress.rs`, which carries the hypotheses and the mutation
//! ledger) runs for one functional only. PySCF has no hybrid stress, so there
//! is no external pin.
//!
//! MEASURED 2026-10-19 (H2/STO-3G, a = 4 Bohr, 1x1x3, h = 2.5e-5, shared
//! grid cutoff 10): all nine components, both functionals, both exxdiv:
//! worst |analytic - FD| 5.43e-9 / 5.56e-9 (HSE06 ewald / none) and 4.54e-9 /
//! 4.68e-9 (CAM-B3LYP), always the (0, 0) component (see the sweep); every
//! other component <= 1.31e-9. HSE06 1x1x3 vs 3-cell supercell sigma
//! 3.5e-15.

mod common;
#[path = "krsh_stress_support/mod.rs"]
mod stress_support;
#[path = "krsh_force_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::stress::Mat3;
use stress_support::*;
use support::*;

/// (1) All nine components of `dE/de` against central FD (h = FD_H, the
/// shared explicit grid cutoff), dense AFT, 1x1x3, both functionals, both
/// exxdiv. Physics: the FD floor; artifact: a wrong kernel slope, J factor
/// or Madelung strain misses by >= the ledger's smallest entry.
#[test]
#[ignore = "validation: 2 functionals x 2 exxdiv x 9 components x 2 fresh k SCFs (72 SCFs)"]
fn rsh_stress_matches_fd_all_nine_both_exxdiv() {
    for f in ["HSE06", CAM] {
        for exxdiv in [ExxDiv::Ewald, ExxDiv::None] {
            let fx = fixture(&H2, [1, 1, 3], f, exxdiv);
            let s = stress(&fx);
            let mut worst = 0.0_f64;
            for c in ALL9 {
                let d = fd_strain(&fx.cell, c, FD_H, [1, 1, 3], f, exxdiv);
                let miss = (s.de_deps[c.0][c.1] - d).abs();
                eprintln!(
                    "{f} {exxdiv:?} eps{c:?}: analytic {:+.10e} FD {d:+.10e} diff {:+.2e}",
                    s.de_deps[c.0][c.1],
                    d - s.de_deps[c.0][c.1]
                );
                worst = worst.max(miss);
            }
            eprintln!("{f} {exxdiv:?}: worst {worst:.2e}");
            assert!(worst < STRESS_FD_BAR, "{f} {exxdiv:?}: {worst:e}");
            assert!(mmax(&s.hf.parts.exchange) > 1e-4, "K walks exercised");
        }
    }
}

/// (2) FD step sweep on CAM-B3LYP (ewald), the mesh-axis diagonal (2, 2): the
/// residual must fall ~4x per halving of h while the h^2 truncation
/// dominates. Artifact hypothesis: a stress defect gives an h-independent
/// residual (ratio ~1). MEASURED 2026-10-19 (residual = FD - analytic):
/// (2,2) -1.274e-8 / -1.056e-8 / -2.397e-9 / -7.23e-10 / -6.6e-11 at h =
/// 2e-4 / 1e-4 / 5e-5 / 2.5e-5 / 1.25e-5 (ratios 4.4, 3.3 below 1e-4). The
/// diagonal (0, 0) is NOT smooth: -2.093e-5 at 2e-4, -1.063e-8 at 1e-4
/// (close to, but a separate printout from, the (2,2) value), -2.9e-10 at
/// 5e-5, +4.54e-9 at 2.5e-5, +4.0e-10 at 1.25e-5; the GLOBAL hybrid PBE0 on
/// the same fixture shows the same pattern (-3.17e-5, -1.79e-8, -1.67e-9,
/// +5.18e-9, +2.2e-10). The step is the atom-grid's hard neighbour_cutoff
/// (mgoldey/ferric#495: an atom crossing the cutoff under strain makes
/// E(strain) discontinuous), not the range separation; the +4.5e-9 at
/// 2.5e-5 is FD-floor noise (it is not h^2: it changes sign and grows from
/// 5e-5 to 2.5e-5). This is why FD_H is 2.5e-5 and the bar (3e-8) sits ~6x
/// above the worst all-nine miss (5.6e-9, HSE06 none (0, 0)).
#[test]
#[ignore = "validation: FD step sweep h = 1e-4 .. 2.5e-5 on eps(2, 2) (3 steps x 2 SCFs)"]
fn cam_b3lyp_stress_fd_residual_follows_h_squared() {
    let fx = fixture(&H2, [1, 1, 3], CAM, ExxDiv::Ewald);
    let s = stress(&fx);
    let c = (2, 2);
    let r: Vec<f64> = [1e-4, 5e-5, 2.5e-5]
        .iter()
        .map(|&h| fd_strain(&fx.cell, c, h, [1, 1, 3], CAM, ExxDiv::Ewald) - s.de_deps[c.0][c.1])
        .collect();
    eprintln!(
        "CAM-B3LYP eps{c:?} residual h=1e-4 {:+.2e}, 5e-5 {:+.2e}, 2.5e-5 {:+.2e}",
        r[0], r[1], r[2]
    );
    let ratio = r[0] / r[1];
    assert!(ratio > 3.0 && ratio < 6.0, "not h^2: ratio {ratio}");
    assert!(r[2].abs() < STRESS_FD_BAR, "{:e}", r[2]);
}

/// (3) The HSE06 rows of the per-commit anchors: 1x1x3 mesh == 3-cell
/// supercell (sigma), and Gamma (via the routed `gamma_rks_rsh_stress`) is in
/// the per-commit file for CAM-B3LYP ewald and HSE06 none.
#[test]
#[ignore = "validation: HSE06 supercell row (a 6-atom SCF)"]
fn hse06_k_mesh_stress_is_the_supercell_stress() {
    let k = stress(&fixture(&H2, [1, 1, 3], "HSE06", ExxDiv::Ewald));
    let sc = stress(&fixture_cell(
        supercell(&H2, [1, 1, 3]),
        [1, 1, 1],
        "HSE06",
        ExxDiv::Ewald,
    ));
    let d: f64 = mdiff(&k.sigma, &sc.sigma);
    let mut third: Mat3 = sc.hf.parts.madelung;
    third.iter_mut().flatten().for_each(|x| *x /= 3.0);
    let dm = mdiff(&k.hf.parts.madelung, &third);
    eprintln!("HSE06: 1x1x3 vs supercell sigma {d:.2e}, Madelung part {dm:.2e}");
    assert!(d < 1e-10 && dm < 1e-9, "{d:e} {dm:e}");
}
