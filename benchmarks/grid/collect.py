#!/usr/bin/env python3
"""Collect grid outputs into the paper matrix.

Columns per system: CP interaction energies (kcal/mol) for
  MP2   = RHF + E(MP2, Coulomb)                      [dlr042]
  A     = RHF + E_corr naive (diagnostic)            [dlr042]
  B     = LRC[kappa]-MP2:dRPA total                  [dlr042]
  T     = LRC[E]-MP2:dRPA total                      [cr02]
  dRPA  = RHF + dRPA(Coulomb) + 2*E_OS(Coulomb)      [cr02 + scs]
at aDZ, aTZ, and CBS (corr two-point 27/8/19 on aDZ/aTZ; HF@aTZ).
Refs: A24 BIND (CCSD(T)/CBS), S22 BIND_S22B. Writes matrix.csv + matrix.md.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
K = 627.509474
FRAGS = ("dimer", "mA_cp", "mB_cp")
METHODS = ("MP2", "A", "B", "T", "dRPA")
RHF_TOL = 5e-7  # cross-method RHF consistency (same SCF settings)


def grab(text, label):
    m = re.search(re.escape(label) + r"\s*=\s*(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def parse_out(key):
    p = OUT / f"{key}.out"
    if not p.exists():
        return None
    t = p.read_text()
    if key.endswith("_scs"):
        return dict(rhf=grab(t, "RHF energy"), e_os=grab(t, "E_OS"))
    if key.endswith("_dlr042"):
        tot, b = grab(t, "Total energy"), grab(t, "E_corr Δ-form (B)")
        if tot is None or b is None:
            return None
        rhf = tot - b
        return dict(rhf=rhf, mp2=grab(t, "E(MP2, Coulomb)"),
                    a=grab(t, "E_corr naive (A)"), b=b, tot_b=tot)
    # --- aTZ comparison gather (2026-07-26) -------------------------------
    # These four methods print `Total      =` (padded), NOT `Total energy =`,
    # and had NO branch here at all: parse_out returned None even for a
    # perfectly good output file, so all 288 gather jobs would have completed
    # and produced nothing readable. The runner's own success test is
    # `grep -q Total`, which passes for both spellings -- so the failure would
    # have been silent on both sides. Verified against real CLI output.
    if key.endswith("_mp2v"):
        # MP2-V: RHF + attenuated MP2 correlation + VV10 nonlocal.
        rhf, tot = grab(t, "RHF energy"), grab(t, "Total")
        if rhf is None or tot is None:
            return None
        return dict(rhf=rhf, corr=grab(t, "attMP2corr"), e_os=grab(t, "E_OS"),
                    e_ss=grab(t, "E_SS"), e_nl=grab(t, "VV10 E_nl"), tot=tot)
    if key.endswith("_atterfc"):
        rhf, tot = grab(t, "RHF energy"), grab(t, "Total")
        if rhf is None or tot is None:
            return None
        return dict(rhf=rhf, corr=grab(t, "MP2 corr"), e_os=grab(t, "E_OS"),
                    e_ss=grab(t, "E_SS"), tot=tot)
    if key.endswith("_scs2terfc"):
        rhf, tot = grab(t, "RHF energy"), grab(t, "Total")
        if rhf is None or tot is None:
            return None
        return dict(rhf=rhf, corr=grab(t, "SCS corr"), e_os=grab(t, "E_OS"),
                    e_ss=grab(t, "E_SS"), tot=tot)
    if key.endswith("_mp2"):
        # Must come AFTER _scs2terfc/_atterfc/_mp2v: plain "_mp2" is a suffix
        # of none of them, but keep the ordering explicit so a future tag
        # ending in "_mp2" cannot be swallowed by this branch.
        rhf, tot = grab(t, "RHF energy"), grab(t, "Total")
        if rhf is None or tot is None:
            return None
        return dict(rhf=rhf, corr=grab(t, "MP2 corr"), tot=tot)
    if key.endswith("_cr02"):
        tot, tc = grab(t, "Total energy"), grab(t, "E_corr coupled (T)")
        if tot is None or tc is None:
            return None
        return dict(rhf=tot - tc, t=tc, tot_t=tot,
                    drpa_c=grab(t, "E(ΔdRPA, Coulomb)"))
    return None


def frag_energies(sysname, frag, basis):
    """Totals per method for one fragment, or None if incomplete."""
    scs = parse_out(f"{sysname}_{frag}_{basis}_scs")
    dlr = parse_out(f"{sysname}_{frag}_{basis}_dlr042")
    cr = parse_out(f"{sysname}_{frag}_{basis}_cr02")
    if not (scs and dlr and cr) or None in (scs["rhf"], scs["e_os"], dlr["mp2"]):
        return None
    rhfs = (scs["rhf"], dlr["rhf"], cr["rhf"])
    if max(rhfs) - min(rhfs) > RHF_TOL:
        print(f"WARN: RHF mismatch {sysname} {frag} {basis}: spread "
              f"{(max(rhfs)-min(rhfs)):.2e}", file=sys.stderr)
    rhf = dlr["rhf"]
    return dict(RHF=rhf,
                MP2=rhf + dlr["mp2"],
                A=rhf + dlr["a"],
                B=dlr["tot_b"],
                T=cr["tot_t"],
                dRPA=rhf + cr["drpa_c"] + 2 * scs["e_os"])


def eint(frags_e, m):
    return (frags_e["dimer"][m] - frags_e["mA_cp"][m] - frags_e["mB_cp"][m]) * K


def cbs(adz, atz, m):
    """corr 2-pt (27*aTZ-8*aDZ)/19 per fragment; HF@aTZ."""
    e = 0.0
    for frag, sign in (("dimer", 1), ("mA_cp", -1), ("mB_cp", -1)):
        ca = adz[frag][m] - adz[frag]["RHF"]
        ct = atz[frag][m] - atz[frag]["RHF"]
        e += sign * (atz[frag]["RHF"] + (27 * ct - 8 * ca) / 19)
    return e * K


def main():
    refs = json.loads((ROOT / "refs.json").read_text())
    rows = []
    for dbse in ("a24", "s22"):
        for idx in sorted(int(i) for i in refs[dbse]):
            sysname = f"{dbse}-{idx:02d}"
            row = dict(system=sysname, ref=refs[dbse][str(idx)])
            per_basis = {}
            for basis in ("adz", "atz"):
                fe = {f: frag_energies(sysname, f, basis) for f in FRAGS}
                if all(fe.values()):
                    per_basis[basis] = fe
                    for m in METHODS:
                        row[f"{m}_{basis}"] = eint(fe, m)
            if "adz" in per_basis and "atz" in per_basis:
                for m in METHODS:
                    row[f"{m}_cbs"] = cbs(per_basis["adz"], per_basis["atz"], m)
            if "adz" in per_basis or "atz" in per_basis:
                rows.append(row)

    cols = ["system", "ref"] + [f"{m}_{b}" for b in ("adz", "atz", "cbs")
                                for m in METHODS]
    with open(ROOT / "matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r.get(c), float) else
                            r.get(c, "")) for c in cols})

    # MAE summary
    lines = ["# SR-MP2+LR-RPA production matrix", "",
             "CP interaction energies (kcal/mol); ref = CCSD(T)/CBS "
             "(A24: BIND; S22: BIND_S22B).",
             "Errors = computed - ref; binding is negative, so MSE > 0 "
             "means underbinding.", "",
             "| set | basis | N | stat | " + " | ".join(METHODS) + " |",
             "|---|---|---|---|" + "---|" * len(METHODS)]
    for subset, pred in (("A24", lambda r: r["system"].startswith("a24")),
                         ("S22", lambda r: r["system"].startswith("s22")),
                         ("all", lambda r: True)):
        for b in ("adz", "atz", "cbs"):
            sel = [r for r in rows if pred(r) and f"MP2_{b}" in r]
            if not sel:
                continue
            maes = [sum(abs(r[f"{m}_{b}"] - r["ref"]) for r in sel) / len(sel)
                    for m in METHODS]
            mses = [sum(r[f"{m}_{b}"] - r["ref"] for r in sel) / len(sel)
                    for m in METHODS]
            lines.append(f"| {subset} | {b} | {len(sel)} | MAE | "
                         + " | ".join(f"{x:.3f}" for x in maes) + " |")
            lines.append(f"| {subset} | {b} | {len(sel)} | MSE | "
                         + " | ".join(f"{x:+.3f}" for x in mses) + " |")
    lines += ["", "## Per-system", "",
              "| system | ref | " + " | ".join(
                  f"{m}({b})" for b in ("adz", "atz", "cbs") for m in METHODS) + " |",
              "|---|---|" + "---|" * (3 * len(METHODS))]
    for r in rows:
        cells = [f"{r[f'{m}_{b}']:.3f}" if f"{m}_{b}" in r else "—"
                 for b in ("adz", "atz", "cbs") for m in METHODS]
        lines.append(f"| {r['system']} | {r['ref']:.3f} | " + " | ".join(cells) + " |")
    (ROOT / "matrix.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} systems collected -> matrix.csv, matrix.md")
    done = sum(1 for r in rows if f"MP2_cbs" in r)
    print(f"  complete through CBS: {done}")


if __name__ == "__main__":
    main()
