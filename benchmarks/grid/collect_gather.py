#!/usr/bin/env python3
"""Collect the A24/aTZ comparison gather into CP interaction energies.

Separate from collect.py on purpose. That script builds the *paper matrix*
(MP2/A/B/T/dRPA) and requires the `_scs`+`_dlr042`+`_cr02` trio for every
fragment -- `frag_energies` returns None unless all three are present, so the
gather methods can never reach its table no matter how `parse_out` behaves.

The gather is a different question: four independent published methods, each
self-contained in one output file, compared against A24 CCSD(T)/CBS.

  mp2        plain RI-MP2 (baseline)
  atterfc    attenuated MP2, erfc, omega = 0.420 A^-1   (dissertation optimal)
  scs2terfc  SCS-MP2(2terfc), r0 0.75/1.05, c_OS 1.27, c_SS 4.05 (JCTC 2015)
  mp2v       MP2-V, r0 1.00, b 11.0, C 0.0089, terfc     (JCTC 11, 4159 (2015))

Counterpoise throughout: E_int = E(dimer) - E(mA_cp) - E(mB_cp), all three in
the dimer basis. Usage:  python3 collect_gather.py [--basis atz]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect as C  # reuse parse_out / grab / K / A24 refs

ROOT = Path(__file__).resolve().parent
K = C.K
FRAGS = ("dimer", "mA_cp", "mB_cp")
GATHER_METHODS = ("mp2", "atterfc", "scs2terfc", "mp2v")


def load_bind():
    """A24 CCSD(T)/CBS references (kcal/mol) from A24.py's literals."""
    txt = (ROOT / "A24.py").read_text()
    out = {}
    for m in re.finditer(
        r"BIND\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\s*\)\]\s*=\s*(-?\d+\.\d+)", txt
    ):
        out[int(m.group(1))] = float(m.group(2))
    if not out:
        raise SystemExit("could not parse BIND from A24.py")
    return out


def interaction(sysname, method, basis):
    """CP interaction energy in kcal/mol, or None if any fragment is missing.

    Returns None rather than a partial number: a CP energy formed from two of
    three fragments is not a smaller-error version of the right answer, it is
    a different quantity entirely.
    """
    tot = {}
    for frag in FRAGS:
        r = C.parse_out(f"{sysname}_{frag}_{basis}_{method}")
        if r is None or r.get("tot") is None:
            return None
        tot[frag] = r["tot"]
    return (tot["dimer"] - tot["mA_cp"] - tot["mB_cp"]) * K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="atz")
    a = ap.parse_args()

    bind = load_bind()
    rows, errs = [], {m: [] for m in GATHER_METHODS}

    for idx in sorted(bind):
        name = f"a24-{idx:02d}" if idx >= 10 else f"a24-{idx:02d}"
        row = {"sys": idx, "ref": bind[idx]}
        for m in GATHER_METHODS:
            ie = interaction(name, m, a.basis)
            row[m] = ie
            if ie is not None:
                errs[m].append(ie - bind[idx])
        rows.append(row)

    hdr = f"{'sys':>4} {'ref':>9} " + " ".join(f"{m:>10}" for m in GATHER_METHODS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = " ".join(
            f"{r[m]:>10.4f}" if r[m] is not None else f"{'--':>10}"
            for m in GATHER_METHODS
        )
        print(f"{r['sys']:>4} {r['ref']:>9.4f} {cells}")

    print()
    print(f"{'MAE':>4} {'':>9} ", end="")
    for m in GATHER_METHODS:
        if errs[m]:
            mae = sum(abs(e) for e in errs[m]) / len(errs[m])
            print(f"{mae:>10.4f}", end="")
        else:
            print(f"{'--':>10}", end="")
    print()
    print(f"{'n':>4} {'':>9} ", end="")
    for m in GATHER_METHODS:
        print(f"{len(errs[m]):>10}", end="")
    print()
    print()
    print("MAE is over COMPLETE systems only; n shows how many contributed.")
    print("Compare MAEs only at equal n -- a method scored on a different")
    print("subset is not comparable, and partial gathers make that easy to miss.")


if __name__ == "__main__":
    main()
