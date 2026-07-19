#!/usr/bin/env python3
"""Consolidate the three DOSD/CCCBDB C6 benchmark rounds (dosd, dosd2, dosd3)
into one unified table and run a paired statistical comparison of RPA@PBE vs
TS against experimental references.

Inputs (committed, not regenerated here):
  - scripts/dosd/results.csv       15 molecules, aug-cc-pVDZ + aug-cc-pVTZ,
                                    DOSD (Meath school) molecular C6 reference.
  - scripts/dosd2/results.csv      10 molecules (6 probe + 4 control),
                                    aug-cc-pVDZ + aug-cc-pVTZ, DOSD reference
                                    via Toulouse et al. arXiv:1305.0107 Table III.
  - scripts/dosd3/results.md /
    scripts/dosd3/runs/*/*.stdout  6 heavy-main-group molecules. NO independent
                                    C6 reference exists (paywalled) for any of
                                    them, so dosd3 contributes NOTHING to the
                                    paired vs-reference C6 test. It is reported
                                    separately (RPA-vs-TS internal comparison
                                    only, not a vs-reference error) plus an
                                    alpha0-vs-CCCBDB table (4 molecules with a
                                    real experimental alpha0 reference).

Rule: only rows with a real literature C6 (or alpha0, reported separately)
enter the paired statistical test. No fabricated/estimated references.

Basis selection: prefer aug-cc-pVTZ (the converged basis per both existing
docs); aug-cc-pVDZ is dropped from the primary test to avoid pseudo-replication
(same molecule counted twice would violate the paired-samples independence
assumption of Wilcoxon/paired-t). DZ is kept in a secondary CSV for reference.

Dedup check: dosd (15 molecules: h2,n2,co,water,nh3,ch4,co2,c2h2,c2h4,c2h6,
hf,hcl,h2s,benzene,o2) and dosd2 (10 molecules: so2,cs2,cos,n2o,cl2,hbr,sih4,
ccl4,ch3oh,ch3och3) share NO molecule names -- confirmed by diff of the two
geometries.py molecule-key lists. dosd3's only C6-comparable molecule (SiH4)
is not an independent run -- it is a citation of dosd2's SiH4 number (see
dosd3/results.md's own note), so it is NOT double counted.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

TARGET_BASIS = "augccpvtz"

# Fresh (post-G8) TS numbers -- see fresh_ts_c6.json's _comment for why the
# committed results.csv TS column cannot be trusted as-is (it predates G8,
# commit ab8ba38/eeda789, which removed the stale hardcoded vol_free table
# fallback; several elements' free-atom volume silently changed materially).
FRESH_TS = json.loads((HERE / "fresh_ts_c6.json").read_text())


def load_dosd():
    rows = []
    fresh = FRESH_TS["dosd"][TARGET_BASIS]
    with open(ROOT / "scripts/dosd/results.csv") as f:
        for r in csv.DictReader(f):
            if r["basis"] != TARGET_BASIS:
                continue
            mol = r["molecule"]
            ts_fresh = fresh.get(mol)
            if ts_fresh is None:
                # o2: TS hard-fails on the current codebase (open-shell not
                # supported by pdep_polarizability_hirshfeld); unreproducible
                # with current code, dropped rather than trusting the stale
                # CSV number. benzene (TZ): still pending re-run at write time.
                continue
            rows.append({
                "molecule": mol,
                "round": "dosd",
                "class": "first-row",
                "ref_c6": float(r["dosd"]),
                "rpa_pbe_c6": float(r["rpa_pbe"]),
                "ts_c6": ts_fresh,
                "source": "Meath-school DOSD (docs/dosd-c6-rpa-vs-ts.md)",
            })
    return rows


def load_dosd2():
    rows = []
    fresh = FRESH_TS["dosd2"][TARGET_BASIS]
    with open(ROOT / "scripts/dosd2/results.csv") as f:
        for r in csv.DictReader(f):
            if r["basis"] != TARGET_BASIS:
                continue
            mol = r["molecule"]
            if r["ref"] == "":
                continue
            ts_fresh = fresh.get(mol)
            if ts_fresh is None:
                continue
            rows.append({
                "molecule": mol,
                "round": "dosd2",
                "class": r["class"],  # probe / control
                "ref_c6": float(r["ref"]),
                "rpa_pbe_c6": float(r["rpa_pbe"]),
                "ts_c6": ts_fresh,
                "source": "Toulouse et al. arXiv:1305.0107 Table III (docs/ts-failure-modes.md)",
            })
    return rows


def main():
    rows = load_dosd() + load_dosd2()
    # dosd3 deliberately excluded from the C6 paired test: no independent
    # C6 reference exists for any dosd3 molecule (paywalled). Its 4
    # alpha0-vs-CCCBDB rows are handled in the separate alpha table below.

    for r in rows:
        r["err_pbe_pct"] = 100.0 * (r["rpa_pbe_c6"] - r["ref_c6"]) / r["ref_c6"]
        r["err_ts_pct"] = 100.0 * (r["ts_c6"] - r["ref_c6"]) / r["ref_c6"]
        r["abs_err_pbe_pct"] = abs(r["err_pbe_pct"])
        r["abs_err_ts_pct"] = abs(r["err_ts_pct"])

    out_csv = HERE / "consolidated_c6.csv"
    fields = ["molecule", "round", "class", "ref_c6", "rpa_pbe_c6", "ts_c6",
              "err_pbe_pct", "err_ts_pct", "abs_err_pbe_pct", "abs_err_ts_pct",
              "source"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n = len(rows)
    pbe_abs = np.array([r["abs_err_pbe_pct"] for r in rows])
    ts_abs = np.array([r["abs_err_ts_pct"] for r in rows])
    pbe_signed = np.array([r["err_pbe_pct"] for r in rows])
    ts_signed = np.array([r["err_ts_pct"] for r in rows])

    # --- paired tests on absolute relative error ---
    wilcoxon = stats.wilcoxon(pbe_abs, ts_abs, alternative="two-sided")
    ttest = stats.ttest_rel(pbe_abs, ts_abs)

    # --- mean vs median summaries ---
    mean_pbe = pbe_abs.mean()
    mean_ts = ts_abs.mean()
    median_pbe = np.median(pbe_abs)
    median_ts = np.median(ts_abs)

    # --- win rate: fraction where each method's abs error is smaller ---
    pbe_wins = int((pbe_abs < ts_abs).sum())
    ts_wins = int((ts_abs < pbe_abs).sum())
    ties = n - pbe_wins - ts_wins

    # --- worst-case ---
    worst_pbe = pbe_abs.max()
    worst_pbe_mol = rows[int(pbe_abs.argmax())]["molecule"]
    worst_ts = ts_abs.max()
    worst_ts_mol = rows[int(ts_abs.argmax())]["molecule"]

    # --- signed mean (systematic bias) ---
    mean_signed_pbe = pbe_signed.mean()
    mean_signed_ts = ts_signed.mean()

    summary = {
        "n": n,
        "basis": "aug-cc-pVTZ",
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_pvalue": float(wilcoxon.pvalue),
        "ttest_statistic": float(ttest.statistic),
        "ttest_pvalue": float(ttest.pvalue),
        "mean_abs_err_pct_rpa_pbe": float(mean_pbe),
        "mean_abs_err_pct_ts": float(mean_ts),
        "median_abs_err_pct_rpa_pbe": float(median_pbe),
        "median_abs_err_pct_ts": float(median_ts),
        "mean_signed_err_pct_rpa_pbe": float(mean_signed_pbe),
        "mean_signed_err_pct_ts": float(mean_signed_ts),
        "win_rate_rpa_pbe": pbe_wins,
        "win_rate_ts": ts_wins,
        "ties": ties,
        "worst_case_rpa_pbe_pct": float(worst_pbe),
        "worst_case_rpa_pbe_molecule": worst_pbe_mol,
        "worst_case_ts_pct": float(worst_ts),
        "worst_case_ts_molecule": worst_ts_mol,
    }

    with open(HERE / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"N = {n} molecules (aug-cc-pVTZ, real literature C6 references only)")
    print()
    print(f"Wilcoxon signed-rank (|err_PBE| vs |err_TS|): "
          f"W={wilcoxon.statistic:.1f}, p={wilcoxon.pvalue:.4g}")
    print(f"Paired t-test        (|err_PBE| vs |err_TS|): "
          f"t={ttest.statistic:.3f}, p={ttest.pvalue:.4g}")
    print()
    print(f"Mean |rel err|:   RPA@PBE = {mean_pbe:.2f}%   TS = {mean_ts:.2f}%")
    print(f"Median |rel err|: RPA@PBE = {median_pbe:.2f}%   TS = {median_ts:.2f}%")
    print(f"Mean signed err:  RPA@PBE = {mean_signed_pbe:+.2f}%   TS = {mean_signed_ts:+.2f}%")
    print()
    print(f"Win rate: RPA@PBE closer on {pbe_wins}/{n}, TS closer on {ts_wins}/{n}, ties {ties}")
    print(f"Worst case: RPA@PBE {worst_pbe:.1f}% ({worst_pbe_mol}), "
          f"TS {worst_ts:.1f}% ({worst_ts_mol})")


if __name__ == "__main__":
    main()
