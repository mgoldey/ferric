//! VALIDATION tier — DLPNO-MP2 pair screening on Boys-localized occupieds
//! (issue #475, follow-up to #269).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 \
//!     --test validation_dlpno_localized -- --ignored --nocapture
//! ```
//!
//! # What is measured
//!
//! `ferric_mp2::dlpno_localized::dlpno_mp2_localized_from_b_ov` on n-butane,
//! n-hexane, n-octane / cc-pVDZ (cc-pvdz-ri), all electrons and one frozen 1s
//! per carbon: occupieds Boys-localized (core and valence separately), the
//! Fock-coupled amplitude solve over the retained pairs, pair retention and
//! recovered correlation energy against ferric's own canonical RI-MP2 at a
//! sweep of `t_cut_pairs`. Python prototype of the same formulation:
//! `scripts/validation/proto_dlpno_loc_occ.py`.
//!
//! # Hypotheses (written BEFORE the sweep)
//!
//! * PHYSICS: localized pair energies decay with distance, so retention at
//!   `t_cut_pairs = 1e-4` is below 1 on C6 and C8, and falls with chain length;
//!   the error is one-sided (E_loc >= E_canonical) and shrinks monotonically
//!   with the threshold.
//! * ARTIFACT: a solve that ignores coupling or mis-handles transposed pairs
//!   would fail the `t_cut_pairs = 0` anchor (asserted first, per system). The
//!   error reaching the anchor floor as the threshold goes to 0 is true by
//!   construction (every pair kept) and does NOT discriminate. The real
//!   discriminators are the anchor, the independent-estimate test in
//!   `dlpno_localized_occupied.rs`, and retention falling with chain length.
//!
//! # Limits
//!
//! * Pair screening only; no PNO/PAO virtual truncation (see the module docs).
//! * C4-C8 are below the locality onset (~30 Bohr); retention here says
//!   nothing about larger chains, and any "retains everything" outcome would be
//!   provisional.
//! * No ORCA comparison, no ranking, no timing or speedup claim.
//!
//! # Mutation ledger
//!
//! See `MUTATIONS` in `dlpno_localized_occupied.rs`; the same mutants fail the
//! anchor here too.

#[path = "common/dlpno_loc.rs"]
mod dlpno_loc;
use dlpno_loc::load;

use ferric_mp2::dlpno_localized::{dlpno_mp2_localized_from_b_ov, LocalizedMp2Config};

const SWEEP: [f64; 4] = [1e-3, 1e-4, 1e-5, 1e-6];

fn sweep(name: &str, frozen: bool) -> Vec<(f64, f64, f64)> {
    let path = format!("../../testdata/molecules/{name}.xyz");
    let fx = load(&path, "cc-pvdz", "cc-pvdz-ri", frozen);
    let run = |tc: f64| {
        let cfg = LocalizedMp2Config {
            t_cut_pairs: tc,
            ..LocalizedMp2Config::default()
        };
        dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &cfg).unwrap()
    };
    // ANCHOR first.
    let (a, ad) = run(0.0);
    let anchor = (a.e_total - fx.e_ri_mp2).abs();
    eprintln!(
        "{name} fc={frozen}: nocc={} nvir={} anchor {anchor:.3e} ({} iters)",
        fx.nocc, fx.nvir, ad.iterations
    );
    assert!(
        anchor < ANCHOR_BAR,
        "{name}: anchor {anchor:e} >= {ANCHOR_BAR:e}"
    );
    let mut rows = Vec::new();
    let mut prev = f64::INFINITY;
    for tc in SWEEP {
        let (got, d) = run(tc);
        let err = got.e_total - fx.e_ri_mp2;
        eprintln!(
            "  tc={tc:e}: retention {:.3} ({} pairs), recovered {:.4}%, err {err:+.3e} Ha, {} iters",
            d.pair_retention,
            d.n_pairs,
            100.0 * got.e_total / fx.e_ri_mp2,
            d.iterations
        );
        assert!(err >= -ANCHOR_BAR, "over-correlation {err:e}");
        if d.pair_retention < 1.0 {
            assert!(
                err > 10.0 * ANCHOR_BAR,
                "live screen must exceed the anchor bar"
            );
        }
        assert!(err <= prev + ANCHOR_BAR, "error not monotone in tc");
        prev = err;
        rows.push((tc, d.pair_retention, err));
    }
    rows
}

/// Anchor floor measured 3.2e-12 (water) and 1.9e-12 / 3.9e-12 / 5.9e-12 (C4 / C6 / C8);
/// bar ~17x the largest.
const ANCHOR_BAR: f64 = 1e-10;

#[test]
#[ignore = "validation: localized-occupied DLPNO-MP2 pair retention, C4/C6/C8 cc-pVDZ (minutes)"]
fn localized_pair_retention_on_alkanes() {
    for frozen in [false, true] {
        let mut ret_1e4 = Vec::new();
        for name in ["alkane_4", "alkane_6", "alkane_8"] {
            let rows = sweep(name, frozen);
            ret_1e4.push(rows[1].1);
        }
        eprintln!("retention at 1e-4 (C4,C6,C8) fc={frozen}: {ret_1e4:?}");
        // PHYSICS hypothesis, measured 2026-10-18 (all electron 0.667 / 0.511 /
        // 0.410; frozen core 0.901 / 0.716 / 0.585): the screen is live on C6
        // and C8 and retention falls with chain length. The C4 frozen-core
        // value is also < 1. Provisional below the locality onset.
        assert!(
            ret_1e4[1] < 1.0 && ret_1e4[2] < 1.0,
            "screen not live: {ret_1e4:?}"
        );
        assert!(
            ret_1e4[0] > ret_1e4[1] && ret_1e4[1] > ret_1e4[2],
            "retention must fall with chain length: {ret_1e4:?}"
        );
    }
}

/// Add-back truncation and the amplitude `conv_thresh`, measured 2026-10-23
/// (cc-pVDZ / cc-pvdz-ri, all electron, err vs canonical RI-MP2 in kcal/mol):
///
/// ```text
///          tc     retention  err add-back off   on       gain
///   C4   1e-3     0.268        +26.585         +3.408    7.80x
///        1e-4     0.667         +0.366         +0.076    4.79x
///        1e-5     0.725         +0.116         +0.004   26.40x
///   C6   1e-3     0.188        +44.762         +5.877    7.62x
///        1e-4     0.511         +1.052         +0.248    4.24x
///        1e-5     0.600         +0.311         +0.036    8.58x
///   C8   1e-3     0.144        +62.960         +8.353    7.54x
///        1e-4     0.410         +1.759         +0.425    4.14x
///        1e-5     0.494         +0.550         +0.082    6.71x
/// ```
///
/// No overshoot anywhere (the corrected error stays positive). Bars: gain
/// >= 2x (half the smallest measured) and corrected error >= -ANCHOR_BAR.
///
/// `conv_thresh` at `t_cut_pairs = 0` (Jacobi iterations; |dE| vs 1e-10):
/// 1e-10 -> 23 iters; 1e-9 -> 20 iters, 2.1e-11 / 4.0e-11 / 5.9e-11 Ha
/// (C4/C6/C8); 1e-8 -> 16 iters, up to 1.1e-9 Ha; 1e-7 -> 13 iters, up to
/// 1.2e-8 Ha. Plain Jacobi converges linearly, so one decade costs only ~3
/// iterations: loosening the default to 1e-9 saves 13% of the iterations and
/// brings the anchor residual to within 2x of `ANCHOR_BAR`. The default stays
/// 1e-10. (The LNO lane's 1.63x came from loosening a DIIS solve from 1e-12,
/// a different solver and a tighter starting point; it does not transfer.)
/// Wall times on the shared box were too noisy to rank (C8 at 1e-10 took
/// 31.5 s in one run and the 20-iteration 1e-9 run 19.4 s).
#[test]
#[ignore = "validation: add-back truncation and amplitude conv_thresh sweep, C4/C6/C8 cc-pVDZ (minutes)"]
fn add_back_and_conv_thresh_on_alkanes() {
    for name in ["alkane_4", "alkane_6", "alkane_8"] {
        let path = format!("../../testdata/molecules/{name}.xyz");
        let fx = load(&path, "cc-pvdz", "cc-pvdz-ri", false);
        let run = |tc: f64, add_back_truncation: bool, conv_thresh: f64| {
            let cfg = LocalizedMp2Config {
                t_cut_pairs: tc,
                add_back_truncation,
                conv_thresh,
                max_iter: 400,
                ..LocalizedMp2Config::default()
            };
            dlpno_mp2_localized_from_b_ov(&fx.b_loc, &fx.f_loc, &fx.eps_vir, &fx.centers, &cfg)
                .unwrap()
        };
        for tc in [1e-3, 1e-4, 1e-5] {
            let (off, d) = run(tc, false, 1e-10);
            let (on, _) = run(tc, true, 1e-10);
            let (eo, en) = (off.e_total - fx.e_ri_mp2, on.e_total - fx.e_ri_mp2);
            let gain = eo.abs() / en.abs().max(1e-15);
            eprintln!(
                "ADDBACK {name} tc={tc:e}: retention {:.3}, err off {:+.4} kcal/mol, on {:+.4}, gain {gain:.2}x",
                d.pair_retention,
                eo * 627.5095,
                en * 627.5095
            );
            assert!(d.pair_retention < 1.0, "premise: the screen must be live");
            assert!(
                gain >= 2.0,
                "{name} tc={tc:e}: add-back gain {gain:.2}x < 2x"
            );
            assert!(
                en >= -ANCHOR_BAR,
                "{name} tc={tc:e}: add-back overshoots ({en:+.3e})"
            );
        }
        let (base, b) = run(0.0, false, 1e-10);
        let (loose, l) = run(0.0, false, 1e-9);
        let de = (loose.e_total - base.e_total).abs();
        eprintln!(
            "CONV {name}: 1e-10 {} iters, 1e-9 {} iters, |dE| {de:.3e} Ha",
            b.iterations, l.iterations
        );
        assert!(de < 1e-9, "1e-9 moved the energy by {de:e}");
        assert!(
            (b.iterations as f64) < 1.3 * l.iterations as f64,
            "Jacobi iteration saving from 1e-10 -> 1e-9 exceeds 1.3x ({} vs {}); \
             revisit the conv_thresh default",
            b.iterations,
            l.iterations
        );
    }
}
