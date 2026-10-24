//! DLPNO-MP2 pair screening on Boys-localized occupieds (issue #475): the
//! Fock-coupled amplitude solve in `ferric_mp2::dlpno_localized`.
//!
//! # Hypotheses (written BEFORE any pair-retention measurement)
//!
//! * PHYSICS: localized occupieds make distant pair energies small, so the
//!   diagonal-f estimate drops them and pair retention falls below 1 at a
//!   useful `t_cut_pairs` on C6/C8, with a one-sided (less negative) error that
//!   grows monotonically with the threshold.
//! * ARTIFACT: a wrong coupled solve, transpose bookkeeping or pair weighting
//!   would miss canonical RI-MP2 in the untruncated anchor; a wrong pair
//!   estimator would not change the anchor at all (every pair is kept at
//!   `t_cut_pairs = 0`, so the error reaches the anchor floor by construction
//!   and is NOT a discriminator). The real discriminators are the anchor (solve),
//!   the independent-estimate test (screen), and, in the validation file,
//!   retention falling with chain length.
//!
//! # Exactness anchor
//!
//! `t_cut_pairs = 0` retains all pairs; the energy is invariant to the occupied
//! rotation, so it must equal `ri_mp2` to the solve tolerance. The bar is
//! [`ANCHOR_BAR`]; the measured error is printed by the anchor tests (see the
//! comment on the constant). The U = identity case is an independent
//! construction: it must equal the canonical `dlpno_mp2_from_b_ov` path.
//!
//! # Mutation ledger
//!
//! See [`MUTATIONS`].

#[path = "common/dlpno_loc.rs"]
mod dlpno_loc;
use dlpno_loc::load;

use ferric_mp2::dlpno_localized::{
    dlpno_mp2_localized_from_b_ov, localized_occupied_fock, LocalizedMp2Config,
};
use ndarray::Array2;

const WATER: &str = "../../testdata/molecules/water.xyz";

/// Measured anchor error (water/cc-pVDZ, all electron and frozen core) is
/// 3.2e-12 Ha; the bar is >= 10x that.
const ANCHOR_BAR: f64 = 1e-10;

/// Mutation ledger (water/cc-pVDZ, all electron; each mutant was applied to
/// `dlpno_localized.rs`, the suite run, and the listed tests failed). Measured
/// anchor error 3.2e-12 Ha (2.8e-15 at conv 1e-13); bar 1e-10 Ha.
#[allow(dead_code)]
const MUTATIONS: &[(&str, &str)] = &[
    (
        "drop the (k,j) coupling sum",
        "anchors fail, error +3.55e-3 Ha",
    ),
    (
        "drop the (i,k) coupling sum",
        "anchors fail, error +3.76e-3 Ha",
    ),
    (
        "transposed partner not transposed",
        "anchors fail, error +1.11e-3 Ha",
    ),
    ("coupling scaled by 0.5", "anchors fail, error +6.4e-4 Ha"),
    (
        "denominator uses f_ii twice",
        "anchors + identity-U fail, +5.3e-3 / 1.1e-2 Ha",
    ),
    (
        "mirror weight fac = 1",
        "anchors + identity-U fail, +6.2e-2 Ha",
    ),
    (
        "same-spin sign flipped",
        "anchors + identity-U fail, -2.0e-1 Ha",
    ),
    (
        "pair estimate uses f[0,0] for every i",
        "pair_screen_matches_an_independent_diagonal_f_estimate fails (all others pass)",
    ),
    (
        "f[j,kk] for f[kk,j] (EQUIVALENT: f symmetric)",
        "survives, as it must",
    ),
    (
        "add-back subtracted instead of added",
        "add_back_restores_the_discarded_estimate_and_reduces_the_error fails",
    ),
    (
        "add-back folds in the OS part only",
        "add_back_restores_the_discarded_estimate_and_reduces_the_error fails",
    ),
    (
        "add-back sums RETAINED pairs instead of discarded",
        "both add_back tests fail (identity anchor included)",
    ),
    (
        "add-back mirror weight fac = 1",
        "add_back_restores_the_discarded_estimate_and_reduces_the_error fails",
    ),
    (
        "add-back same-spin exchange sign flipped",
        "add_back_restores_the_discarded_estimate_and_reduces_the_error fails",
    ),
];

fn exact_cfg() -> LocalizedMp2Config {
    LocalizedMp2Config {
        t_cut_pairs: 0.0,
        ..LocalizedMp2Config::default()
    }
}

#[test]
fn rotation_is_orthogonal_and_f_is_genuinely_nondiagonal() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let n = fx.nocc;
    let utu = fx.u.t().dot(&fx.u);
    let dev = (&utu - &Array2::<f64>::eye(n))
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(dev < 1e-9, "U not orthogonal: {dev:e}");
    let mut off = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                off = off.max(fx.f_loc[(i, j)].abs());
            }
        }
    }
    eprintln!("water: max |f_ij| (i!=j) = {off:.3e}");
    assert!(
        off > 1e-2,
        "premise: Fock must be non-diagonal (got {off:e})"
    );
}

fn anchor(frozen: bool) -> f64 {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", frozen);
    let (got, diag) =
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &exact_cfg())
            .unwrap();
    assert_eq!(diag.pair_retention, 1.0);
    let d = got.e_total - fx.e_ri_mp2;
    eprintln!(
        "ANCHOR water frozen={frozen}: loc {:.14} vs ri_mp2 {:.14}, diff {d:+.3e}, {} iters",
        got.e_total, fx.e_ri_mp2, diag.iterations
    );
    d.abs()
}

#[test]
fn anchor_untruncated_localized_equals_canonical_all_electron() {
    assert!(anchor(false) < ANCHOR_BAR);
}

#[test]
fn anchor_untruncated_localized_equals_canonical_frozen_core() {
    assert!(anchor(true) < ANCHOR_BAR);
}

/// The 3e-12 anchor residual must be the Jacobi tolerance, not a systematic
/// offset: tightening `conv_thresh` by 1e3 must shrink it by >= 10x.
#[test]
fn anchor_residual_is_the_solve_tolerance() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let tight = LocalizedMp2Config {
        t_cut_pairs: 0.0,
        conv_thresh: 1e-13,
        max_iter: 400,
        ..LocalizedMp2Config::default()
    };
    let (got, diag) =
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &tight)
            .unwrap();
    let d = (got.e_total - fx.e_ri_mp2).abs();
    eprintln!("tight anchor: {d:.3e} after {} iters", diag.iterations);
    assert!(d < 3e-13, "residual {d:e} is not a convergence effect");
}

/// Independent construction: U = identity must reproduce the canonical
/// DIAGONAL-eps path (`dlpno_mp2_from_b_ov` with every screen off), in which
/// the coupling term is exactly zero.
#[test]
fn identity_rotation_matches_the_canonical_diagonal_path() {
    use ferric_mp2::dlpno_mp2::{dlpno_mp2_from_b_ov, DlpnoConfig};
    use ferric_mp2::pair_domains::complete_pair_domains;
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let f = localized_occupied_fock(&fx.eps_occ, &Array2::eye(fx.nocc));
    let (got, _) =
        dlpno_mp2_localized_from_b_ov(&fx.b_ov, &f, &fx.eps_vir, &fx.centers, &exact_cfg())
            .unwrap();
    let mut eps = fx.eps_occ.clone();
    eps.extend_from_slice(&fx.eps_vir);
    let d = complete_pair_domains(&fx.centers).unwrap();
    let (want, _) = dlpno_mp2_from_b_ov(
        &fx.b_ov,
        &eps,
        fx.nocc,
        fx.nvir,
        0,
        fx.nocc,
        &d,
        &DlpnoConfig::exact(),
    )
    .unwrap();
    let dd = (got.e_total - want.e_total).abs();
    eprintln!("identity-U vs canonical dlpno path: {dd:.3e}");
    assert!(dd < ANCHOR_BAR);
    assert!((got.e_os - want.e_os).abs() < ANCHOR_BAR);
    assert!((got.e_ss - want.e_ss).abs() < ANCHOR_BAR);
}

#[test]
fn screen_is_live_one_sided_and_vanishes_with_the_threshold() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let mut prev_err = f64::INFINITY;
    let mut saw_screen = false;
    for tc in [1e-3, 1e-4, 1e-6, 0.0] {
        let cfg = LocalizedMp2Config {
            t_cut_pairs: tc,
            ..LocalizedMp2Config::default()
        };
        let (got, diag) =
            dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &cfg)
                .unwrap();
        let err = got.e_total - fx.e_ri_mp2;
        eprintln!(
            "water tc={tc:e}: retention {:.3} err {err:+.3e}",
            diag.pair_retention
        );
        assert!(
            err >= -ANCHOR_BAR,
            "screening must not over-correlate: {err:e}"
        );
        assert!(
            err <= prev_err + ANCHOR_BAR,
            "error must shrink with the threshold"
        );
        prev_err = err;
        if diag.pair_retention < 1.0 {
            saw_screen = true;
            assert!(
                err > 10.0 * ANCHOR_BAR,
                "a live screen must exceed the anchor bar"
            );
        }
    }
    assert!(
        saw_screen,
        "premise: some threshold must actually drop pairs"
    );
    assert!(prev_err.abs() < ANCHOR_BAR);
}

/// The screen must use the diagonal-f estimate of the LOCALIZED pair energies.
/// Recomputed here independently (plain loops from `b_loc`), the retained
/// count and the discarded estimated energy must match the library's.
#[test]
fn ample_memory_budget_runs_unchanged() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let base =
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &exact_cfg())
            .unwrap()
            .0;
    let cfg = LocalizedMp2Config {
        memory_budget_bytes: Some(1 << 34),
        ..exact_cfg()
    };
    let got = dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &cfg)
        .unwrap()
        .0;
    assert_eq!(got.e_total.to_bits(), base.e_total.to_bits());
}

#[test]
fn pair_screen_matches_an_independent_diagonal_f_estimate() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let (n, nv) = (fx.nocc, fx.nvir);
    let tc = 1e-3;
    let (mut kept, mut disc) = (0usize, 0.0);
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0;
            for a in 0..nv {
                for b in 0..nv {
                    let (mut kab, mut kba) = (0.0, 0.0);
                    for p in 0..fx.b_loc.nrows() {
                        kab += fx.b_loc[(p, i * nv + a)] * fx.b_loc[(p, j * nv + b)];
                        kba += fx.b_loc[(p, i * nv + b)] * fx.b_loc[(p, j * nv + a)];
                    }
                    let d = fx.f_loc[(i, i)] + fx.f_loc[(j, j)] - fx.eps_vir[a] - fx.eps_vir[b];
                    acc += (2.0 * kab - kba) * kab / d;
                }
            }
            let e = if i == j { acc } else { 2.0 * acc };
            if i == j || e.abs() >= tc {
                kept += 1;
            } else {
                disc += e;
            }
        }
    }
    let cfg = LocalizedMp2Config {
        t_cut_pairs: tc,
        ..LocalizedMp2Config::default()
    };
    let (_, diag) =
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &cfg)
            .unwrap();
    assert_eq!(diag.n_pairs, kept);
    assert!(
        kept < n * (n + 1) / 2,
        "premise: the screen must drop pairs"
    );
    assert!((diag.discarded_estimated_energy - disc).abs() < 1e-12);
}

#[test]
fn unconverged_solve_and_memory_overrun_are_hard_errors() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let few = LocalizedMp2Config {
        t_cut_pairs: 0.0,
        max_iter: 1,
        ..LocalizedMp2Config::default()
    };
    assert!(
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &few)
            .is_err()
    );
    let tiny = LocalizedMp2Config {
        t_cut_pairs: 0.0,
        memory_budget_bytes: Some(1024),
        ..LocalizedMp2Config::default()
    };
    assert!(
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &tiny)
            .is_err()
    );
}

/// Add-back exactness anchor: with nothing screened there is nothing to add
/// back, so the result must be BIT-identical with the knob on or off.
#[test]
fn add_back_is_the_identity_when_no_pair_is_screened() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let off =
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &exact_cfg())
            .unwrap();
    let cfg = LocalizedMp2Config {
        add_back_truncation: true,
        ..exact_cfg()
    };
    let on = dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &cfg)
        .unwrap();
    assert_eq!(on.1.pair_retention, 1.0);
    assert_eq!(on.1.truncation_correction, 0.0);
    assert_eq!(on.0.e_total.to_bits(), off.0.e_total.to_bits());
    assert_eq!(on.0.e_os.to_bits(), off.0.e_os.to_bits());
    assert_eq!(on.0.e_ss.to_bits(), off.0.e_ss.to_bits());
}

/// With a live screen, add-back must (1) add exactly the discarded estimate,
/// split consistently into OS/SS, and (2) move the energy TOWARD canonical
/// RI-MP2 without overshooting past the measured bound below.
#[test]
fn add_back_restores_the_discarded_estimate_and_reduces_the_error() {
    let fx = load(WATER, "cc-pvdz", "cc-pvdz-ri", false);
    let mut live = 0;
    for tc in [1e-3, 1e-4] {
        let run = |add_back_truncation: bool| {
            let cfg = LocalizedMp2Config {
                t_cut_pairs: tc,
                add_back_truncation,
                ..LocalizedMp2Config::default()
            };
            dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &cfg)
                .unwrap()
        };
        let (off, doff) = run(false);
        let (on, don) = run(true);
        let (e_off, e_on) = (off.e_total - fx.e_ri_mp2, on.e_total - fx.e_ri_mp2);
        eprintln!(
            "water tc={tc:e}: retention {:.3}, err off {e_off:+.3e}, on {e_on:+.3e}, \
             correction {:+.3e}",
            don.pair_retention, don.truncation_correction
        );
        assert_eq!(doff.truncation_correction, 0.0);
        if don.pair_retention == 1.0 {
            continue;
        }
        live += 1;
        assert!(
            (don.truncation_correction - don.discarded_estimated_energy).abs() < 1e-12,
            "the correction must be the screen's own discarded estimate"
        );
        assert!(
            (on.e_total - off.e_total - don.truncation_correction).abs() < 1e-12,
            "the correction must be added exactly once"
        );
        assert!((on.e_total - on.e_os - on.e_ss).abs() < 1e-14);
        // Measured: +1.908e-3 -> +4.795e-6 Ha at tc = 1e-3 (~400x). The bar asks
        // for 10x; a sign flip doubles the error and an OS-only add-back
        // leaves most of it.
        assert!(
            e_on.abs() < 0.1 * e_off.abs(),
            "add-back must reduce the screening error >= 10x: {e_off:+.3e} -> {e_on:+.3e}"
        );
    }
    assert!(live > 0, "premise: some threshold must actually drop pairs");
}
