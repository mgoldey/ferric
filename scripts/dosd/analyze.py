#!/usr/bin/env python3
"""Compare swept C6 and static polarizability against DOSD/CRC references;
emit CSV, plots, and MARE tables.

Two stories:
  1. Molecular C6 vs DOSD (RPA@PBE vs RPA@HF vs TS).
  2. Static isotropic alpha(0) vs CRC/Olney-Meath — the upstream driver. A
     better alpha is *why* RPA@PBE gives a better C6 (C6 ∝ ∫ alpha(iω)² dω).

The molecular C6 is read from results.json (parsed from the CLI's printed
`molecular C6 = X a.u.`, the global-origin c6_molecular_iso). The static alpha
is read from each NPZ's `alpha_tensor` (iso = trace/3).
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = json.loads((HERE / "results.json").read_text())
REF = json.loads((ROOT / "testdata" / "reference" / "dosd_c6.json").read_text())
DOSD = REF["molecular_c6_aa"]
ALPHA_REF = REF["molecular_alpha0"]

DOSD_KEY = {"h2": "H2", "n2": "N2", "co": "CO", "water": "H2O", "nh3": "NH3",
            "ch4": "CH4", "co2": "CO2", "c2h2": "C2H2", "c2h4": "C2H4",
            "c2h6": "C2H6", "hf": "HF", "hcl": "HCl", "h2s": "H2S",
            "benzene": "C6H6", "o2": "O2"}
MOLS = list(DOSD_KEY)
METHODS = ["rpa_pbe", "rpa_hf", "ts"]
BASES = ["augccpvdz", "augccpvtz"]
APPROX = {"o2"}  # flagged-approximate reference


def c6_ref(mol):
    return DOSD[DOSD_KEY[mol]]["c6"]


def alpha_ref(mol):
    return ALPHA_REF[DOSD_KEY[mol]]["alpha0"]


def c6_get(mol, method, basis):
    e = RESULTS.get(f"{basis}/{mol}/{method}")
    return e["c6"] if e and e.get("ok") else None


def alpha_iso(mol, method, basis):
    """Static isotropic alpha from the NPZ alpha_tensor (trace/3)."""
    f = HERE / "runs" / basis / f"{mol}_{method}.npz"
    if not f.exists():
        return None
    try:
        z = np.load(f)
        if "alpha_tensor" in z:
            return float(np.trace(z["alpha_tensor"]) / 3.0)
    except Exception:
        return None
    return None


def mare(errs):
    a = [abs(e) for e in errs if e is not None]
    return float(np.mean(a)) if a else None


def mse(errs):
    a = [e for e in errs if e is not None]
    return float(np.mean(a)) if a else None


def main():
    # ---------- C6 CSV ----------
    rows = ["molecule,basis,dosd,rpa_pbe,rpa_hf,ts,err_pbe_%,err_hf_%,err_ts_%,approx"]
    for basis in BASES:
        for mol in MOLS:
            r = c6_ref(mol)
            vals = {m: c6_get(mol, m, basis) for m in METHODS}
            errs = {m: (100.0 * (vals[m] - r) / r if vals[m] and r else None)
                    for m in METHODS}

            def f(x):
                return "" if x is None else f"{x:.2f}"
            rows.append(",".join([mol, basis, f(r), f(vals["rpa_pbe"]),
                                  f(vals["rpa_hf"]), f(vals["ts"]),
                                  f(errs["rpa_pbe"]), f(errs["rpa_hf"]),
                                  f(errs["ts"]), "yes" if mol in APPROX else ""]))
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # ---------- alpha CSV ----------
    arows = ["molecule,basis,alpha_ref,alpha_rpa_pbe,alpha_rpa_hf,"
             "err_pbe_%,err_hf_%"]
    for basis in BASES:
        for mol in MOLS:
            ar = alpha_ref(mol)
            ap = alpha_iso(mol, "rpa_pbe", basis)
            ah = alpha_iso(mol, "rpa_hf", basis)
            ep = 100.0 * (ap - ar) / ar if ap and ar else None
            eh = 100.0 * (ah - ar) / ar if ah and ar else None

            def f(x):
                return "" if x is None else f"{x:.2f}"
            arows.append(",".join([mol, basis, f(ar), f(ap), f(ah),
                                   f(ep), f(eh)]))
    (HERE / "alpha.csv").write_text("\n".join(arows) + "\n")

    # ---------- MARE tables ----------
    print("=== Molecular C6 vs DOSD ===")
    print(f"{'basis':<12}{'method':<10}{'MARE %':>8}{'MSE %':>8}{'n':>4}")
    for basis in BASES:
        for m in METHODS:
            errs = [100.0 * (c6_get(mol, m, basis) - c6_ref(mol)) / c6_ref(mol)
                    if c6_get(mol, m, basis) else None for mol in MOLS]
            ma, ms = mare(errs), mse(errs)
            n = sum(1 for e in errs if e is not None)
            if ma is not None:
                print(f"{basis:<12}{m:<10}{ma:>8.1f}{ms:>8.1f}{n:>4}")

    print("\n=== Static alpha(0) vs CRC/Olney-Meath ===")
    print(f"{'basis':<12}{'method':<10}{'MARE %':>8}{'MSE %':>8}{'n':>4}")
    for basis in BASES:
        for m in ["rpa_pbe", "rpa_hf"]:
            errs = [100.0 * (alpha_iso(mol, m, basis) - alpha_ref(mol)) / alpha_ref(mol)
                    if alpha_iso(mol, m, basis) else None for mol in MOLS]
            ma, ms = mare(errs), mse(errs)
            n = sum(1 for e in errs if e is not None)
            if ma is not None:
                print(f"{basis:<12}{m:<10}{ma:>8.1f}{ms:>8.1f}{n:>4}")
    print()

    # ---------- Plots ----------
    plotdir = HERE / "plots"
    plotdir.mkdir(exist_ok=True)
    colors = {"rpa_pbe": "#1f77b4", "rpa_hf": "#d62728", "ts": "#2ca02c"}
    labels = {"rpa_pbe": "RPA@PBE", "rpa_hf": "RPA@HF", "ts": "TS (static)"}

    for basis in BASES:
        # --- C6 scatter (log-log) ---
        fig, ax = plt.subplots(figsize=(6, 6))
        lo, hi = 5, 3000
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x")
        for m in METHODS:
            xs = [c6_ref(mol) for mol in MOLS if c6_get(mol, m, basis)]
            ys = [c6_get(mol, m, basis) for mol in MOLS if c6_get(mol, m, basis)]
            if xs:
                ax.scatter(xs, ys, c=colors[m], label=labels[m], s=40, alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("DOSD reference C6 (a.u.)")
        ax.set_ylabel("Computed C6 (a.u.)")
        ax.set_title(f"Molecular C6 vs DOSD — {basis}")
        ax.legend(); ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout(); fig.savefig(plotdir / f"scatter_c6_{basis}.png", dpi=130)
        plt.close(fig)

        # --- alpha scatter ---
        fig, ax = plt.subplots(figsize=(6, 6))
        lo, hi = 3, 90
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y = x")
        for m in ["rpa_pbe", "rpa_hf"]:
            xs = [alpha_ref(mol) for mol in MOLS if alpha_iso(mol, m, basis)]
            ys = [alpha_iso(mol, m, basis) for mol in MOLS if alpha_iso(mol, m, basis)]
            if xs:
                ax.scatter(xs, ys, c=colors[m], label=labels[m], s=40, alpha=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Reference static α₀ (a.u.)")
        ax.set_ylabel("Computed static α₀ (a.u.)")
        ax.set_title(f"Static polarizability vs CRC/DOSD — {basis}")
        ax.legend(); ax.grid(True, alpha=0.2)
        fig.tight_layout(); fig.savefig(plotdir / f"scatter_alpha_{basis}.png", dpi=130)
        plt.close(fig)

        # --- C6 signed-error bars ---
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(MOLS)); w = 0.26
        for i, m in enumerate(METHODS):
            es = [100.0 * (c6_get(mol, m, basis) - c6_ref(mol)) / c6_ref(mol)
                  if c6_get(mol, m, basis) else np.nan for mol in MOLS]
            ax.bar(x + (i - 1) * w, es, w, color=colors[m], label=labels[m])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([DOSD_KEY[m] for m in MOLS],
                                             rotation=45, ha="right")
        ax.set_ylabel("Signed C6 error vs DOSD (%)")
        ax.set_title(f"Per-molecule C6 error — {basis}")
        ax.legend(); ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout(); fig.savefig(plotdir / f"signed_error_c6_{basis}.png", dpi=130)
        plt.close(fig)

        # --- alpha signed-error bars ---
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, m in enumerate(["rpa_pbe", "rpa_hf"]):
            es = [100.0 * (alpha_iso(mol, m, basis) - alpha_ref(mol)) / alpha_ref(mol)
                  if alpha_iso(mol, m, basis) else np.nan for mol in MOLS]
            ax.bar(x + (i - 0.5) * w, es, w, color=colors[m], label=labels[m])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([DOSD_KEY[m] for m in MOLS],
                                             rotation=45, ha="right")
        ax.set_ylabel("Signed α₀ error vs ref (%)")
        ax.set_title(f"Per-molecule static-α error — {basis}")
        ax.legend(); ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout(); fig.savefig(plotdir / f"signed_error_alpha_{basis}.png", dpi=130)
        plt.close(fig)

    print(f"wrote results.csv, alpha.csv, and plots to {plotdir}")


if __name__ == "__main__":
    main()
