//! Weekly validation of the range-separated k-point RKS force (dense AFT):
//! central FD of ferric's own energy (fresh SCF per displacement) for HSE06
//! and CAM-B3LYP under exxdiv ewald and none, and the measured FD floor. The
//! per-commit subset (one FD row, the anchors, the mutant ledger and the
//! refusals) is `pbc_krsh_force.rs`, whose header carries the hypotheses and
//! the ledger.
//!
//! MEASURED (H2/STO-3G, a = 4 Bohr, 1x1x3, h = 1e-4, release, 2026-10-18):
//! analytic - FD at (0,2) is +2.82e-9 (HSE06) / +2.89e-9 (CAM-B3LYP) in both
//! exxdiv modes, every other component <= 2.6e-11; the (0,2) residual falls
//! 1.16e-8 -> 2.89e-9 -> 7.14e-10 for h = 2e-4 -> 1e-4 -> 5e-5 (ratio 4.0,
//! 4.0: the h^2 truncation of the bond-direction third derivative, the same
//! 2.3e-9 the global hybrids show); ewald vs none force 2.4e-13 / 7.6e-13.

mod common;
#[path = "krsh_force_support/mod.rs"]
mod support;

use ferric_pbc::dense_aft::ExxDiv;
use ndarray::Array2;
use support::*;

// --------------------------------------------------------------------- FD

/// (1) Central FD of the RSH energy (fresh SCF per displacement, h = 1e-4,
/// shared explicit grid cutoff), dense AFT, 1x1x3 (non-TRIM, carries the
/// Bloch phases), both exxdiv modes, HSE06 and CAM-B3LYP. The two exxdiv
/// forces must agree (the mixed-Madelung shift cancels between the Pulay and
/// M terms, kxcgrad module doc). Physics: |analytic - FD| at the h^2 floor
/// (~2e-9); artifact: a wrong kernel/Madelung would miss by >= 1e-5 (ledger).
#[test]
#[ignore = "validation: HSE06 + CAM-B3LYP x exxdiv ewald/none central FD, 16 fresh k SCFs; measured ~70 s release"]
fn rsh_force_matches_fd_both_exxdiv() {
    for f in ["HSE06", CAM] {
        let mut by: Vec<Array2<f64>> = Vec::new();
        for exxdiv in [ExxDiv::Ewald, ExxDiv::None] {
            let fx = fixture(&H2, [1, 1, 3], f, exxdiv);
            let g = force(&fx);
            let mut worst = 0.0_f64;
            for (a, x) in [(0, 2), (1, 1)] {
                let d = fd(&H2, a, x, 1e-4, [1, 1, 3], f, exxdiv);
                eprintln!(
                    "{f} {exxdiv:?} ({a},{x}): analytic {:+.10e} FD {d:+.10e} diff {:+.2e}",
                    g.grad[(a, x)],
                    d - g.grad[(a, x)]
                );
                worst = worst.max((g.grad[(a, x)] - d).abs());
            }
            assert!(worst < FD_BAR, "{f} {exxdiv:?}: {worst:e}");
            assert!(max_abs(&g.hf.parts.exchange) > 1e-4, "K walks exercised");
            by.push(g.grad);
        }
        let dd = diff(&by[0], &by[1]);
        eprintln!("{f}: ewald vs none force {dd:.2e}");
        assert!(dd < 1e-9, "{f}: {dd:e}");
    }
}

/// FD floor of the RSH force, MEASURED on CAM-B3LYP (the case whose erf piece
/// is live), component (0, 2): the residual must shrink ~4x per halving of h
/// (h^2 truncation) down to the SCF noise. Artifact hypothesis: a force
/// defect gives an h-independent residual (ratio ~1).
#[test]
#[ignore = "validation: FD floor sweep h = 2e-4, 1e-4, 5e-5 (6 fresh k SCFs)"]
fn cam_b3lyp_fd_residual_follows_h_squared() {
    let fx = fixture(&H2, [1, 1, 3], CAM, ExxDiv::Ewald);
    let g = force(&fx).grad[(0, 2)];
    let r = |h: f64| fd(&H2, 0, 2, h, [1, 1, 3], CAM, ExxDiv::Ewald) - g;
    let (r2, r1, r05) = (r(2e-4), r(1e-4), r(5e-5));
    eprintln!("CAM-B3LYP (0,2) residual h=2e-4 {r2:+.2e}, 1e-4 {r1:+.2e}, 5e-5 {r05:+.2e}");
    assert!(r2 / r1 > 3.5 && r2 / r1 < 6.0, "not h^2: ratio {}", r2 / r1);
    assert!(r05.abs() < FD_BAR);
}

/// The erf piece is live on the CAM-B3LYP fixture and nearly dead for HSE06
/// at this cell (measured max weight printed); every `Rsh*` mutant moves the
/// FD-validated force by far more than the FD bar (ledger in the doc of
/// `pbc_krsh_force.rs`; per-commit canary `rsh_mutants_are_caught_canary`).
#[test]
#[ignore = "validation: every Rsh* force mutant on CAM-B3LYP and HSE06, H2 1x1x3 dense ewald; CI ~162 s"]
fn rsh_mutants_are_caught() {
    use ferric_pbc::kgrad::KGradMutation;
    // Each functional's production force is FD-validated (FD_BAR) above, so a
    // miss against it is a miss against FD to within 2.9e-9.
    rsh_mutant_sweep(
        &[(CAM, 1e-5), ("HSE06", 10.0 * FD_BAR)],
        &[
            KGradMutation::RshNoKernelFactor,
            KGradMutation::RshKernelOnJ,
            KGradMutation::RshSwapFractions,
            KGradMutation::RshPlainMadelung,
            KGradMutation::RshOmegaScaled,
            KGradMutation::RshNoErfPiece,
            KGradMutation::RshNoMadelungTerm,
        ],
    );
}
