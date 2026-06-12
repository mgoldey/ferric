#!/usr/bin/env python3
"""Generate LaTeX tables of A24 CP interaction energies for the paper.

Reads matrix.csv (produced by collect.py) and writes:
  - tab_mae_by_cat.tex  : MAE-by-category summary
  - tab_persys_a24.tex  : per-system absolute binding energies (aDZ + aTZ)

Methods reported: MP2, LRC[kappa] (B), LRC[E] (T), dRPA.
Columns are absolute CP interaction energies (kcal/mol), not errors.
MAE footer rows show |method - ref|.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "tex_tables"
OUT.mkdir(exist_ok=True)

rows = list(csv.DictReader((ROOT / "matrix.csv").open()))

METHODS = ["MP2", "B", "T", "dRPA"]
METHOD_LABELS = {
    "MP2":  r"\textrm{MP2}",
    "B":    r"\lrckshort{}",
    "T":    r"\lrceshort{}",
    "dRPA": r"\textrm{dRPA}",
}

LABELS = {
    1:  r"H$_2$O$\cdot$NH$_3$",
    2:  r"H$_2$O$\cdot$H$_2$O",
    3:  r"HCN$\cdot$HCN",
    4:  r"HF$\cdot$HF",
    5:  r"NH$_3\cdot$NH$_3$",
    6:  r"CH$_4\cdot$HF",
    7:  r"NH$_3\cdot$CH$_4$",
    8:  r"CH$_4\cdot$H$_2$O",
    9:  r"(HCHO)$_2$",
    10: r"C$_2$H$_4\cdot$H$_2$O",
    11: r"C$_2$H$_4\cdot$HCHO",
    12: r"C$_2$H$_2\cdot$C$_2$H$_2$ (T)",
    13: r"C$_2$H$_4\cdot$NH$_3$",
    14: r"C$_2$H$_4\cdot$C$_2$H$_4$ (stack)",
    15: r"CH$_4\cdot$C$_2$H$_4$",
    16: r"CH$_4\cdot$BH$_3$",
    17: r"CH$_4\cdot$C$_2$H$_6$",
    18: r"CH$_4\cdot$C$_2$H$_6$ ($C_3$)",
    19: r"CH$_4\cdot$CH$_4$",
    20: r"CH$_4\cdot$Ar",
    21: r"C$_2$H$_4\cdot$Ar",
    22: r"C$_2$H$_4\cdot$C$_2$H$_2$ ($\pi$)",
    23: r"C$_2$H$_4\cdot$C$_2$H$_4$ ($D_{2h}$)",
    24: r"C$_2$H$_2\cdot$C$_2$H$_2$ ($D_{2h}$)",
}

GROUPS = [
    (r"\textit{Hydrogen bonds}",                 [1, 2, 3, 4, 5]),
    (r"\textit{Mixed/dipole}",                   [6, 7, 8, 9, 10, 11, 12, 13]),
    (r"\textit{$\pi$-stacked}",                  [14]),
    (r"\textit{Dispersion-bound}",               [15, 16, 17, 18, 19, 20]),
    (r"\textit{$\pi$-contacts (repulsive wall)}", [22, 23, 24]),
]

MAE_GROUPS = [
    ("H-bond (1--5)",        [1, 2, 3, 4, 5]),
    (r"$\pi$/saddle (14,22--24)", [14, 22, 23, 24]),
    ("Mixed/dipole (6--13)", [6, 7, 8, 9, 10, 11, 12, 13]),
    (r"Dispersion (15--21)", [15, 16, 17, 18, 19, 20, 21]),
    (r"All A24",             list(range(1, 25))),
]


def fval(r, col):
    v = r.get(col, "")
    return float(v) if v.strip() else None


by_id = {r["system"]: r for r in rows}


def best_method(vals_dict, ref):
    """Return method key with smallest |val - ref|, among non-None vals."""
    candidates = {m: abs(v - ref) for m, v in vals_dict.items() if v is not None}
    return min(candidates, key=candidates.get) if candidates else None


# ──────────────────────────────────────────────────────────────────────────────
# Table 1: MAE by category
# ──────────────────────────────────────────────────────────────────────────────
def make_mae_table():
    """MAE and MSE (computed - ref; binding negative => MSE>0 = underbinding)
    by interaction-type subset, at aDZ, aTZ and CBS."""
    ncol = len(METHODS)
    col_spec = "@{}llr" + "r" * ncol + "@{}"
    hdr_methods = " & ".join(METHOD_LABELS[m] for m in METHODS)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{%",
        r"  Mean absolute and mean signed errors (MAE/MSE, kcal\,mol$^{-1}$, CP)",
        r"  by interaction-type subset of A24 at aug-cc-pVDZ, aug-cc-pVTZ, and",
        r"  two-point CBS extrapolation (correlation $27/8/19$ on aDZ/aTZ;",
        r"  HF at aTZ). Errors are $\Delta E_\mathrm{method} - \Delta E_\mathrm{ref}$;",
        r"  binding energies are negative, so $\mathrm{MSE} > 0$ is underbinding.",
        r"  Reference: CCSD(T)/CBS.~\cite{Rezac2013}",
        r"  Bold = lowest MAE per row.",
        r"}",
        r"\label{tab:mae_by_cat}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Basis / subset & stat & $N$ & {hdr_methods} \\",
        r"\midrule",
    ]

    for b, blabel in (("adz", "aDZ"), ("atz", "aTZ"), ("cbs", "CBS")):
        lines.append(rf"\multicolumn{{{ncol+3}}}{{l}}{{\textit{{{blabel}}}}} \\")
        for name, ids in MAE_GROUPS:
            errs = {m: [] for m in METHODS}
            for i in ids:
                r = by_id.get(f"a24-{i:02d}")
                if not r:
                    continue
                ref = fval(r, "ref")
                for m in METHODS:
                    v = fval(r, f"{m}_{b}")
                    if v is not None and ref is not None:
                        errs[m].append(v - ref)
            n = len(errs["MP2"])
            if not n:
                continue
            mae = {m: sum(abs(e) for e in errs[m]) / len(errs[m]) for m in METHODS if errs[m]}
            mse = {m: sum(errs[m]) / len(errs[m]) for m in METHODS if errs[m]}
            best = min(mae, key=mae.get)
            mae_cells = [rf"$\mathbf{{{mae[m]:.3f}}}$" if m == best else f"${mae[m]:.3f}$"
                         if m in mae else "---" for m in METHODS]
            mse_cells = [f"${mse[m]:+.3f}$" if m in mse else "---" for m in METHODS]
            lines.append(rf"\quad {name} & MAE & {n} & " + " & ".join(mae_cells) + r" \\")
            lines.append(rf"\quad & MSE & & " + " & ".join(mse_cells) + r" \\")
        lines.append(r"\addlinespace[3pt]")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Table 2: Per-system absolute binding energies
# ──────────────────────────────────────────────────────────────────────────────
def make_persys_table():
    # 9 data columns: ref | MP2 B T dRPA (adz) | MP2 B T dRPA (atz)
    ncols_data = 1 + 4 + 4   # ref + 4 methods × 2 bases
    col_spec = "@{}llr" + "r" * 4 + "r" * 4 + "@{}"
    hdr_m = " & ".join(METHOD_LABELS[m] for m in METHODS)

    lines = [
        r"\begin{longtable}{" + col_spec + "}",
        r"\caption{%",
        r"  CP interaction energies (kcal\,mol$^{-1}$) for the A24 dataset",
        r"  at aug-cc-pVDZ (aDZ) and aug-cc-pVTZ (aTZ).",
        r"  Reference $\Delta E_\mathrm{ref}$: CCSD(T)/CBS from",
        r"  Rez\'{a}\v{c} and Hobza.~\cite{Rezac2013}",
        r"  dRPA = RHF + $\Delta$dRPA[Coulomb] + 2$E_\mathrm{OS}$[Coulomb].",
        r"  Bold = method closest to reference per row (aDZ and aTZ independently).",
        r"  Dashes: calculation not yet completed.",
        r"}\label{tab:persys_a24}\\",
        r"\toprule",
        r" & System & $\Delta E_\mathrm{ref}$ & \multicolumn{4}{c}{aDZ} & \multicolumn{4}{c}{aTZ} \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        rf" & & & {hdr_m} & {hdr_m} \\",
        r"\midrule\endfirsthead",
        r"\midrule",
        r" & System & $\Delta E_\mathrm{ref}$ & \multicolumn{4}{c}{aDZ} & \multicolumn{4}{c}{aTZ} \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        rf" & & & {hdr_m} & {hdr_m} \\",
        r"\midrule\endhead",
        r"\bottomrule\endfoot",
    ]

    for group_label, ids in GROUPS:
        lines.append(r"\addlinespace[2pt]")
        lines.append(rf"\multicolumn{{11}}{{l}}{{{group_label}}} \\")
        for i in ids:
            r = by_id.get(f"a24-{i:02d}")
            if not r:
                continue
            ref = fval(r, "ref")
            if ref is None:
                continue

            row_cells = [str(i), LABELS.get(i, "?"), f"${ref:.3f}$"]

            for b in ["adz", "atz"]:
                vals = {m: fval(r, f"{m}_{b}") for m in METHODS}
                bm = best_method(vals, ref)
                for m in METHODS:
                    v = vals[m]
                    if v is None:
                        row_cells.append("---")
                    elif m == bm:
                        row_cells.append(rf"$\mathbf{{{v:.3f}}}$")
                    else:
                        row_cells.append(f"${v:.3f}$")

            lines.append("  " + " & ".join(row_cells) + r" \\")

    # MAE footer
    lines.append(r"\midrule")
    lines.append(r"\addlinespace[2pt]")
    for name, ids in MAE_GROUPS:
        row_cells = [r"\multicolumn{2}{l}{MAE: " + name + "}", "---"]
        for b in ["adz", "atz"]:
            errs = {m: [] for m in METHODS}
            for i in ids:
                r = by_id.get(f"a24-{i:02d}")
                if not r:
                    continue
                ref = fval(r, "ref")
                for m in METHODS:
                    v = fval(r, f"{m}_{b}")
                    if v is not None and ref is not None:
                        errs[m].append(abs(v - ref))
            n = len(errs["MP2"])
            if not n:
                row_cells.extend(["---"] * 4)
                continue
            mae = {m: (sum(errs[m]) / len(errs[m]) if errs[m] else None)
                   for m in METHODS}
            bm = min((m for m in METHODS if mae[m] is not None),
                     key=lambda m: mae[m])
            for m in METHODS:
                if mae[m] is None:
                    row_cells.append("---")
                elif m == bm:
                    row_cells.append(rf"$\mathbf{{{mae[m]:.3f}}}$")
                else:
                    row_cells.append(f"${mae[m]:.3f}$")
        lines.append("  " + " & ".join(row_cells) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


if __name__ == "__main__":
    mae_tex = make_mae_table()
    persys_tex = make_persys_table()

    (OUT / "tab_mae_by_cat.tex").write_text(mae_tex)
    (OUT / "tab_persys_a24.tex").write_text(persys_tex)
    print(f"Wrote {OUT}/tab_mae_by_cat.tex")
    print(f"Wrote {OUT}/tab_persys_a24.tex")
