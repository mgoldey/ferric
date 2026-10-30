//! Weekly half of `pbc_kuxcstress_ro_rsgdf.rs` (k-point ROHF / ROKS force and
//! stress on RS-GDF): `rsgdf_range_split_rohf_force_and_stress_match_fd`,
//! whose per-commit CI time was >= 120 s (runs 40957025137 / 40953725119,
//! mean 158 s), moved here so the per-commit shards stay short. Every test is
//! `#[ignore = "validation: ..."]`. Hypotheses are in the doc of
//! `pbc_kuxcstress_ro_rsgdf.rs`; the helpers are
//! `kuxcstress_ro_rsgdf_support/mod.rs`. Per-commit canaries that remain:
//! `rsgdf_range_split_rohf_stress_zz_matches_fd_canary`,
//! `rsgdf_range_split_rohf_force_1x1x3_sees_the_gamma_kernel_mutant`,
//! `rsgdf_range_split_closed_shell_rohf_equals_rhf`.

mod common;
#[path = "kuxcstress_ro_rsgdf_support/mod.rs"]
mod support;

use support::*;

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
#[ignore = "validation: range-split RS-GDF k-ROHF force + stress vs FD with every split-only mutant, H3 1x1x2; CI ~158 s"]
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
