#!/usr/bin/env python3
"""Test whether a single empirical multiplicative rescale factor k can bring
RPA@PBE's molecular C6 accuracy to parity with (or better than) TS's, using
ONLY the existing consolidated dataset (scripts/dosd_combined/consolidated_c6.csv
plus the same fresh-TS-numbers loading path consolidate.py already uses for
aug-cc-pVDZ). No new ferric calculations are run here.

Motivation (docs/rpa-vs-ts-statistical-verdict.md, docs/dosd-c6-rpa-vs-ts.md):
RPA@PBE underbinds EVERY molecule in the 24-molecule combined dataset by a
roughly uniform relative amount (~-15% aTZ, ~-19% aDZ) -- a systematic,
single-signed bias is exactly the profile a single multiplicative correction
can fix, in principle. This script fits that correction and cross-validates
it honestly (leave-one-out), rather than reporting the optimistic in-sample
number.

Methodology notes on choosing k
--------------------------------
Three candidate point estimators for a single multiplicative factor k such
that rpa_rescaled = k * rpa_pbe:

  1. mean-of-ratios:      k = mean(ref_i / rpa_i)
  2. ratio-of-means:      k = mean(ref_i) / mean(rpa_i)
  3. least-squares slope through the origin: k = sum(rpa_i*ref_i)/sum(rpa_i^2)

These differ because they optimize different loss functions:
  - (1) minimizes the average of relative errors in a symmetric sense and is
    what you get if you directly average the per-molecule "correction factor
    needed" -- but it can be pulled around by a single molecule with an
    unusually large ref/rpa ratio (H2's ratio is the largest in this set).
  - (2) minimizes something closer to a weighted absolute-error criterion
    where large-C6 molecules (e.g. C6H6, CCl4) dominate -- not what we want
    if the target metric is *relative* (percent) error, since it is exactly
    the ratio-of-sums, which downweights small molecules.
  - (3) is the classic least-squares (minimize sum of squared *absolute*
    residuals through the origin) -- also dominated by the largest-C6
    molecules (squared absolute error), the wrong loss for a %-error target.

Since the actual accuracy metric used throughout the existing docs (and in
consolidate.py) is MEAN/MEDIAN ABSOLUTE RELATIVE ERROR (MARE), the estimator
that is most consistent with that evaluation metric is the MEDIAN of the
per-molecule ratios ref_i/rpa_i: it directly targets a robust central
tendency of "how much would I need to multiply rpa_i by to hit ref_i exactly,"
is insensitive to the one or two largest-ratio outliers (H2 in this set), and
its optimality criterion (minimizing sum of |log(k) - log(ratio_i)| under a
log-symmetric read, or equivalently being the L1-optimal multiplicative
correction) lines up with MARE much better than an L2/sum-based estimator
does. We report all three for transparency but use the MEDIAN-of-ratios as
the primary k, and note where the others would have changed the conclusion.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FRESH_TS = json.loads((HERE / "fresh_ts_c6.json").read_text())


def load_basis(basis_key):
    """Reproduce consolidate.py's row-building logic for an arbitrary basis
    (augccpvdz or augccpvtz), so we get the identical 24-molecule set at DZ
    that consolidate.py already produces at TZ (same molecules pass the
    "fresh TS number exists" filter at both bases -- verified: o2 is null at
    both bases in fresh_ts_c6.json, so the N=24 selection is basis-invariant)."""
    rows = []

    fresh_dosd = FRESH_TS["dosd"][basis_key]
    with open(ROOT / "scripts/dosd/results.csv") as f:
        for r in csv.DictReader(f):
            if r["basis"] != basis_key:
                continue
            mol = r["molecule"]
            ts_fresh = fresh_dosd.get(mol)
            if ts_fresh is None:
                continue
            rows.append({
                "molecule": mol,
                "round": "dosd",
                "class": "first-row",
                "ref_c6": float(r["dosd"]),
                "rpa_pbe_c6": float(r["rpa_pbe"]),
                "ts_c6": ts_fresh,
            })

    fresh_dosd2 = FRESH_TS["dosd2"][basis_key]
    with open(ROOT / "scripts/dosd2/results.csv") as f:
        for r in csv.DictReader(f):
            if r["basis"] != basis_key:
                continue
            mol = r["molecule"]
            if r["ref"] == "":
                continue
            ts_fresh = fresh_dosd2.get(mol)
            if ts_fresh is None:
                continue
            rows.append({
                "molecule": mol,
                "round": "dosd2",
                "class": r["class"],
                "ref_c6": float(r["ref"]),
                "rpa_pbe_c6": float(r["rpa_pbe"]),
                "ts_c6": ts_fresh,
            })

    return rows


def fit_k(ref, rpa, method="median_ratio"):
    ratios = ref / rpa
    if method == "median_ratio":
        return float(np.median(ratios))
    if method == "mean_ratio":
        return float(np.mean(ratios))
    if method == "ratio_of_means":
        return float(np.mean(ref) / np.mean(rpa))
    if method == "least_squares":
        return float(np.sum(rpa * ref) / np.sum(rpa * rpa))
    raise ValueError(method)


def abs_rel_err_pct(pred, ref):
    return 100.0 * np.abs(pred - ref) / ref


def loo_cv(ref, rpa, method="median_ratio"):
    """Leave-one-out CV: fit k on all-but-one, evaluate held-out error.
    Returns (held_out_abs_rel_err_pct array, per-fold k array)."""
    n = len(ref)
    held_out_err = np.zeros(n)
    ks = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        k = fit_k(ref[mask], rpa[mask], method=method)
        ks[i] = k
        pred_i = k * rpa[i]
        held_out_err[i] = abs_rel_err_pct(pred_i, ref[i])
    return held_out_err, ks


def analyze_basis(basis_key, basis_label):
    rows = load_basis(basis_key)
    n = len(rows)
    mols = [r["molecule"] for r in rows]
    ref = np.array([r["ref_c6"] for r in rows])
    rpa = np.array([r["rpa_pbe_c6"] for r in rows])
    ts = np.array([r["ts_c6"] for r in rows])

    ts_abs_err = abs_rel_err_pct(ts, ref)

    print(f"\n{'='*78}\n{basis_label}  (N={n})\n{'='*78}")

    # --- candidate k estimators ---
    k_candidates = {m: fit_k(ref, rpa, m)
                     for m in ("median_ratio", "mean_ratio", "ratio_of_means",
                               "least_squares")}
    print("Candidate k estimators (in-sample, full N):")
    for m, k in k_candidates.items():
        rescaled = k * rpa
        err = abs_rel_err_pct(rescaled, ref)
        print(f"  {m:16s} k={k:.4f}  in-sample MAE={err.mean():.2f}%  "
              f"median={np.median(err):.2f}%")

    k_primary_method = "median_ratio"
    k_primary = k_candidates[k_primary_method]

    # --- in-sample (optimistic) rescaled error, primary k ---
    rescaled_in_sample = k_primary * rpa
    in_sample_err = abs_rel_err_pct(rescaled_in_sample, ref)

    # --- honest LOO-CV ---
    loo_err, loo_ks = loo_cv(ref, rpa, method=k_primary_method)

    # --- raw (unscaled) RPA@PBE error for reference ---
    raw_err = abs_rel_err_pct(rpa, ref)

    print(f"\nPrimary k (median-of-ratios) = {k_primary:.4f}")
    print(f"LOO-CV per-fold k range: [{loo_ks.min():.4f}, {loo_ks.max():.4f}], "
          f"std={loo_ks.std():.4f}  (stability check: tight range == not "
          f"overfit to any single molecule)")

    print(f"\n{'Metric':<28s}{'raw RPA@PBE':>14s}{'rescaled (in-sample)':>22s}"
          f"{'rescaled (LOO-CV)':>20s}{'TS (fresh)':>14s}")
    print(f"{'Mean |rel err| %':<28s}{raw_err.mean():>14.2f}"
          f"{in_sample_err.mean():>22.2f}{loo_err.mean():>20.2f}"
          f"{ts_abs_err.mean():>14.2f}")
    print(f"{'Median |rel err| %':<28s}{np.median(raw_err):>14.2f}"
          f"{np.median(in_sample_err):>22.2f}{np.median(loo_err):>20.2f}"
          f"{np.median(ts_abs_err):>14.2f}")
    print(f"{'Worst case %':<28s}{raw_err.max():>14.2f}"
          f"{in_sample_err.max():>22.2f}{loo_err.max():>20.2f}"
          f"{ts_abs_err.max():>14.2f}")

    # --- paired stats: LOO-CV rescaled RPA vs TS (the fair, apples-to-apples
    # comparison -- both are "what a user actually gets", not the in-sample
    # number) ---
    wilcoxon = stats.wilcoxon(loo_err, ts_abs_err, alternative="two-sided")
    ttest = stats.ttest_rel(loo_err, ts_abs_err)
    rpa_wins = int((loo_err < ts_abs_err).sum())
    ts_wins = int((ts_abs_err < loo_err).sum())
    ties = n - rpa_wins - ts_wins

    print(f"\nPaired test, rescaled-RPA(LOO-CV) vs TS |rel err|:")
    print(f"  Wilcoxon: W={wilcoxon.statistic:.1f}, p={wilcoxon.pvalue:.4g}")
    print(f"  Paired t: t={ttest.statistic:.3f}, p={ttest.pvalue:.4g}")
    print(f"  Win rate: rescaled-RPA closer on {rpa_wins}/{n}, "
          f"TS closer on {ts_wins}/{n}, ties {ties}")

    # --- also compare in-sample (optimistic) rescaled RPA vs TS, to show the
    # gap between the honest and optimistic numbers ---
    wilcoxon_opt = stats.wilcoxon(in_sample_err, ts_abs_err, alternative="two-sided")
    rpa_wins_opt = int((in_sample_err < ts_abs_err).sum())
    ts_wins_opt = int((ts_abs_err < in_sample_err).sum())
    print(f"\n(For comparison, IN-SAMPLE/optimistic rescaled-RPA vs TS: "
          f"Wilcoxon p={wilcoxon_opt.pvalue:.4g}, win rate "
          f"{rpa_wins_opt}/{n} vs {ts_wins_opt}/{n} -- "
          f"do not use this number as the headline result.)")

    # Overfitting sanity check: how much does LOO-CV degrade vs in-sample?
    degradation = loo_err.mean() - in_sample_err.mean()
    print(f"\nOverfitting check: LOO-CV mean MAE - in-sample mean MAE = "
          f"{degradation:+.3f} pp (small ==> one free parameter on N={n} "
          f"generalizes fine; large ==> in-sample number was misleading)")

    return {
        "basis": basis_label,
        "n": n,
        "molecules": mols,
        "k_candidates": k_candidates,
        "k_primary_method": k_primary_method,
        "k_primary": k_primary,
        "loo_k_min": float(loo_ks.min()),
        "loo_k_max": float(loo_ks.max()),
        "loo_k_std": float(loo_ks.std()),
        "raw_mean_abs_err_pct": float(raw_err.mean()),
        "raw_median_abs_err_pct": float(np.median(raw_err)),
        "raw_worst_pct": float(raw_err.max()),
        "in_sample_mean_abs_err_pct": float(in_sample_err.mean()),
        "in_sample_median_abs_err_pct": float(np.median(in_sample_err)),
        "in_sample_worst_pct": float(in_sample_err.max()),
        "loo_cv_mean_abs_err_pct": float(loo_err.mean()),
        "loo_cv_median_abs_err_pct": float(np.median(loo_err)),
        "loo_cv_worst_pct": float(loo_err.max()),
        "loo_cv_worst_molecule": mols[int(np.argmax(loo_err))],
        "ts_mean_abs_err_pct": float(ts_abs_err.mean()),
        "ts_median_abs_err_pct": float(np.median(ts_abs_err)),
        "ts_worst_pct": float(ts_abs_err.max()),
        "wilcoxon_loo_vs_ts_statistic": float(wilcoxon.statistic),
        "wilcoxon_loo_vs_ts_pvalue": float(wilcoxon.pvalue),
        "ttest_loo_vs_ts_statistic": float(ttest.statistic),
        "ttest_loo_vs_ts_pvalue": float(ttest.pvalue),
        "win_rate_rescaled_rpa": rpa_wins,
        "win_rate_ts": ts_wins,
        "ties": ties,
        "wilcoxon_in_sample_vs_ts_pvalue": float(wilcoxon_opt.pvalue),
        "win_rate_in_sample_rescaled_rpa": rpa_wins_opt,
        "win_rate_in_sample_ts": ts_wins_opt,
        "overfitting_degradation_pp": float(degradation),
    }


def main():
    results = {}
    results["aug-cc-pVTZ"] = analyze_basis("augccpvtz", "aug-cc-pVTZ")
    results["aug-cc-pVDZ"] = analyze_basis("augccpvdz", "aug-cc-pVDZ")

    # --- cross-basis transfer check: does the aTZ-fitted k work at aDZ (and
    # vice versa)? This is the key "is this overfit-by-a-different-name"
    # check -- a k fitted on one basis's systematic offset should NOT be
    # expected to transfer to a different basis if the offset is genuinely
    # basis-dependent (per the existing docs' finding that RPA@PBE's error
    # shrinks from -18.6% (aDZ) to -15.0% (aTZ)).
    print(f"\n{'='*78}\nCross-basis transfer check\n{'='*78}")
    k_tz = results["aug-cc-pVTZ"]["k_primary"]
    k_dz = results["aug-cc-pVDZ"]["k_primary"]
    print(f"k(aTZ) = {k_tz:.4f}   k(aDZ) = {k_dz:.4f}   "
          f"difference = {100*(k_dz-k_tz)/k_tz:+.1f}% relative")

    rows_dz = load_basis("augccpvdz")
    ref_dz = np.array([r["ref_c6"] for r in rows_dz])
    rpa_dz = np.array([r["rpa_pbe_c6"] for r in rows_dz])
    ts_dz = np.array([r["ts_c6"] for r in rows_dz])
    # apply the aTZ-fitted k to the aDZ data (wrong-basis transfer)
    mistransfer_err = abs_rel_err_pct(k_tz * rpa_dz, ref_dz)
    correct_err = abs_rel_err_pct(k_dz * rpa_dz, ref_dz)
    ts_dz_err = abs_rel_err_pct(ts_dz, ref_dz)
    print(f"Applying aTZ's k={k_tz:.4f} to aDZ data (WRONG basis transfer): "
          f"mean |err| = {mistransfer_err.mean():.2f}%  "
          f"(vs {correct_err.mean():.2f}% using aDZ's own k, "
          f"vs {ts_dz_err.mean():.2f}% for TS, "
          f"vs {abs_rel_err_pct(rpa_dz, ref_dz).mean():.2f}% unscaled RPA@PBE)")

    results["cross_basis_transfer"] = {
        "k_atz": k_tz,
        "k_adz": k_dz,
        "relative_difference_pct": float(100*(k_dz-k_tz)/k_tz),
        "mistransfer_mean_abs_err_pct": float(mistransfer_err.mean()),
        "correct_transfer_mean_abs_err_pct": float(correct_err.mean()),
    }

    out = HERE / "rescale_summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
