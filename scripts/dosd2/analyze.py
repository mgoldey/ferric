#!/usr/bin/env python3
"""TS-failure-mode analysis: RPA@PBE vs TS on probes (anisotropic/multiply-bonded)
vs controls (isotropic/saturated), both bases."""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results.json").read_text()) if (HERE / "results.json").exists() else {}
REF = json.loads((HERE / "refs.json").read_text())["molecular_c6_aa"]

KM = {"so2": "SO2", "cs2": "CS2", "cos": "COS", "n2o": "N2O", "cl2": "Cl2",
      "hbr": "HBr", "sih4": "SiH4", "ccl4": "CCl4", "ch3oh": "CH3OH",
      "ch3och3": "CH3OCH3"}
MOLS = list(KM)
BASES = ["augccpvdz", "augccpvtz"]


def c6(m, meth, b):
    e = RES.get(f"{b}/{m}/{meth}")
    return e["c6"] if e and e.get("ok") else None


def err(m, meth, b):
    v, r = c6(m, meth, b), REF[KM[m]]["c6"]
    return 100.0 * (v - r) / r if (v is not None and r) else None


def main():
    rows = ["molecule,class,basis,ref,rpa_pbe,rpa_hf,ts,"
            "err_pbe_%,err_hf_%,err_ts_%"]
    for b in BASES:
        for m in MOLS:
            r = REF[KM[m]]["c6"]
            cls = REF[KM[m]]["class"]
            cp, ch, ct = (c6(m, "rpa_pbe", b), c6(m, "rpa_hf", b),
                          c6(m, "ts", b))
            ep, eh, et = (err(m, "rpa_pbe", b), err(m, "rpa_hf", b),
                          err(m, "ts", b))

            def f(x):
                return "" if x is None else f"{x:.1f}"
            rows.append(",".join([m, cls, b, f(r), f(cp), f(ch), f(ct),
                                  f(ep), f(eh), f(et)]))
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    print(f"{'mol':9}{'class':8}{'ref':>8}  | "
          f"{'PBE_DZ':>8}{'TS_DZ':>9} | {'PBE_TZ':>8}{'TS_TZ':>9}")
    print("-" * 72)
    for m in MOLS:
        r = REF[KM[m]]["c6"]
        cls = REF[KM[m]]["class"]

        def pe(meth, b):
            e = err(m, meth, b)
            return f"{e:+.0f}%" if e is not None else "  -"
        print(f"{KM[m]:9}{cls:8}{r:8.1f}  | "
              f"{pe('rpa_pbe','augccpvdz'):>8}{pe('ts','augccpvdz'):>9} | "
              f"{pe('rpa_pbe','augccpvtz'):>8}{pe('ts','augccpvtz'):>9}")
    print()

    # MARE by class x method x basis
    print(f"{'class':9}{'method':9}{'basis':12}{'MARE %':>8}{'MSE %':>8}{'n':>4}")
    for cls in ["probe", "control"]:
        for meth in ["rpa_pbe", "rpa_hf", "ts"]:
            for b in BASES:
                es = [err(m, meth, b) for m in MOLS if REF[KM[m]]["class"] == cls]
                es = [e for e in es if e is not None]
                if es:
                    print(f"{cls:9}{meth:9}{b:12}"
                          f"{statistics.mean(abs(e) for e in es):>8.1f}"
                          f"{statistics.mean(es):>8.1f}{len(es):>4}")


if __name__ == "__main__":
    main()
