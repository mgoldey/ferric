"""S66 equilibrium benchmark of MP2-V and SCS-MP2(2terfc) (mgoldey/ferric#278).

Computes S66 (scale 1.00) interaction energies with ferric's Python bindings
for three methods, each both counterpoise-corrected (CP) and uncorrected:

  mp2        plain RI-MP2 (control; must overbind pi-stacks)
  scs2terfc  SCS-MP2(2terfc), published defaults  (run_scs_mp2_2terfc)
  mp2v       MP2-V(terfc, aTZ), published defaults (run_mp2_v)

at aug-cc-pVTZ / aug-cc-pvtz-rifit, frozen core = 1 per first-row heavy atom.
The reference is the S66 CCSD(T)/CBS CP interaction energy
(testdata/reference/c9_refs/s66x8_ccsdt_cbs.json, BEGDB).

The published fit is aTZ, frozen core, NO counterpoise; the CP column is the
convention-mismatch/negative-control column and must never be ranked alone.

Fragment split: the first `a_size` atoms are fragment A, with `a_size` parsed
from S66_FRAGA in benchmarks/harness/examples/c9_driver.rs and VERIFIED against
the independent monomer nuclear repulsion energies in benchmarks/grid/S66.py
(Psi4 database; never a connectivity heuristic). Monomers match to <1e-8 for
all 66. The S66x8 1.00 dimer nuclear repulsion differs from the original S66 by
1e-4..1e-2 relative, so the intermolecular geometry (and the reference table)
is the BEGDB S66x8 one, not the original S66 one.

Usage (from the repo root, through scripts/ferric-limited):
    python scripts/validation/s66_terfc_benchmark.py selftest
    python scripts/validation/s66_terfc_benchmark.py check-split
    python scripts/validation/s66_terfc_benchmark.py run --dimers 1,2,24 --out OUT.jsonl
    python scripts/validation/s66_terfc_benchmark.py report --out OUT.jsonl --json RESULTS.json

`run` appends one JSON line per (dimer, calc, method) and skips lines already
present, so it is resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S66_DIR = ROOT / "testdata/molecules/c9_systems/s66x8"
REF_JSON = ROOT / "testdata/reference/c9_refs/s66x8_ccsdt_cbs.json"
C9_DRIVER = ROOT / "benchmarks/harness/examples/c9_driver.rs"
S66_PY = ROOT / "benchmarks/grid/S66.py"
TERF_DIR = ROOT / "terf-tables"

HARTREE_TO_KCAL = 627.5094740631  # CODATA 2014 as used by Psi4/BEGDB tables
BASIS = "aug-cc-pvtz"
AUX = "aug-cc-pvtz-rifit"
METHODS = ("mp2", "scs2terfc", "mp2v")
CALCS = ("AB", "A_in_AB", "B_in_AB", "A", "B")
# Z of first-row heavy atoms (frozen core = 1 each). S66 has none beyond Ne.
Z = {"H": 1, "C": 6, "N": 7, "O": 8}

# Published statistics are quoted ONLY as the issue states them (issue #278):
# SCS-MP2(2terfc, aTZ) S66 RMSD 0.228 kcal/mol (JCTC 11, 4159 (2015), Table 2).
# The MP2-V S66 RMSD is "Table 1/§3 of the same paper" and is NOT quoted here.
PUBLISHED = {"scs2terfc": {"rmsd_kcal": 0.228, "set": "full S66, aTZ, no CP, fitted"}}


def s66_fraga() -> list[int]:
    text = C9_DRIVER.read_text()
    m = re.search(r"const S66_FRAGA: \[usize; 67\] = \[(.*?)\];", text, re.S)
    nums = [
        int(x) for x in re.findall(r"\b(\d+)\b", re.sub(r"//[^\n]*", "", m.group(1)))
    ]
    assert len(nums) == 67, len(nums)
    return nums


def s66_nre() -> dict[int, tuple[float, float, float]]:
    """(dimer, monoA, monoB) nuclear repulsion from the Psi4 S66 database."""
    out: dict[int, list] = {}
    pat = re.compile(
        r'DATA\["NUCLEAR REPULSION ENERGY"\]\["S66-(\d+)-(dimer|monoA-unCP|monoB-unCP)"\] = ([\d.]+)'
    )
    for m in pat.finditer(S66_PY.read_text()):
        out.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(3))
    return {k: (v["dimer"], v["monoA-unCP"], v["monoB-unCP"]) for k, v in out.items()}


def s66_files() -> dict[int, Path]:
    d = {}
    for p in sorted(S66_DIR.glob("s66_*_1.00.xyz")):
        d[int(p.name.split("_")[1])] = p
    assert len(d) == 66, len(d)
    return d


def read_atoms(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for ln in lines[2 : 2 + n]:
        s, x, y, z = ln.split()[:4]
        atoms.append((s, float(x), float(y), float(z)))
    assert len(atoms) == n
    return atoms


def xyz_string(atoms, real: list[bool]) -> str:
    rows = [
        f"{('' if r else '@') + s} {x:.9f} {y:.9f} {z:.9f}"
        for (s, x, y, z), r in zip(atoms, real)
    ]
    return f"{len(rows)}\n\n" + "\n".join(rows) + "\n"


def build_calc(atoms, a_size: int, calc: str):
    n = len(atoms)
    ia = set(range(a_size))
    if calc == "AB":
        sel, real = list(range(n)), [True] * n
    elif calc == "A":
        sel = sorted(ia)
        real = [True] * len(sel)
    elif calc == "B":
        sel = [i for i in range(n) if i not in ia]
        real = [True] * len(sel)
    elif calc == "A_in_AB":
        sel, real = list(range(n)), [i in ia for i in range(n)]
    elif calc == "B_in_AB":
        sel, real = list(range(n)), [i not in ia for i in range(n)]
    else:
        raise ValueError(calc)
    sub = [atoms[i] for i in sel]
    nfc = sum(1 for (s, *_), r in zip(sub, real) if r and Z[s] > 2)
    return xyz_string(sub, real), nfc


def check_split(dimers: list[int]) -> bool:
    import ferric

    fraga, nre, files = s66_fraga(), s66_nre(), s66_files()
    ok = True
    for k in dimers:
        atoms = read_atoms(files[k])
        a = fraga[k]
        e = {}
        for c in ("AB", "A", "B"):
            xyz, _ = build_calc(atoms, a, c)
            e[c] = ferric.Molecule.from_xyz_string(xyz).nuclear_repulsion()
        ref = nre[k]
        rel = [abs(e[c] - r) / r for c, r in zip(("AB", "A", "B"), ref)]
        # Both monomers closed-shell neutral?
        ne = [
            ferric.Molecule.from_xyz_string(build_calc(atoms, a, c)[0]).nelec() % 2
            for c in ("A", "B")
        ]
        # Monomers (intramolecular geometry + atom assignment) must match to ~1e-8. The dimer NRE differs
        # 1e-4..1e-2 because the BEGDB S66x8 1.00 intermolecular geometry is not the original S66 one
        # (informational; the S66x8 reference energies belong to the x8 geometry).
        good = max(rel[1:]) < 2e-8 and ne == [0, 0]
        ok &= good
        print(
            f"S66-{k:02d} a_size={a:2d} NRE rel dev dimer/A/B = {rel[0]:.1e} {rel[1]:.1e} {rel[2]:.1e}  "
            f"even-electron={ne == [0, 0]} {'OK' if good else 'BAD'}"
        )
    return ok


def git_info() -> dict:
    def g(*a):
        return subprocess.run(
            ["git", *a], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "head": g("rev-parse", "HEAD"),
        "script_blob": g("rev-parse", "HEAD:scripts/validation/s66_terfc_benchmark.py"),
        "dirty_script": bool(
            g(
                "status",
                "--porcelain",
                "--",
                "scripts/validation/s66_terfc_benchmark.py",
            )
        ),
    }


def compute(method: str, xyz: str, nfc: int, basis: str, aux: str, mem_gb: float):
    import ferric

    mol = ferric.Molecule.from_xyz_string(xyz)
    bs, ab = ferric.BasisSet.bundled(basis), ferric.BasisSet.bundled(aux)
    t0 = time.perf_counter()
    extra: dict = {}
    if method == "mp2":
        r = ferric.run_rimp2(mol, bs, ab, frozen_core=nfc, memory_budget_gb=mem_gb)
        e, rhf = r.total_energy, r.rhf_energy
    elif method == "scs2terfc":
        r = ferric.run_scs_mp2_2terfc(
            mol, bs, ab, frozen_core=nfc, memory_budget_gb=mem_gb
        )
        e, rhf = r.total_energy, r.rhf_energy
        extra = {"e_os": r.e_os, "e_ss": r.e_ss, "scs_corr": r.scs_corr}
    elif method == "mp2v":
        r = ferric.run_mp2_v(mol, bs, ab, frozen_core=nfc, memory_budget_gb=mem_gb)
        e, rhf = r.total_energy, r.rhf_energy
        extra = {
            "att_mp2_corr": r.att_mp2_corr,
            "vv10_e_nl": r.vv10_e_nl,
            "n_nlc_points": r.n_nlc_points,
        }
    else:
        raise ValueError(method)
    return e, rhf, time.perf_counter() - t0, extra


def load_done(path: Path, basis: str | None = None) -> dict:
    done = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if basis is not None and r.get("basis") != basis:
                    continue
                done[(r["dimer"], r["calc"], r["method"])] = r
    return done


def cmd_run(a) -> None:
    os.environ.setdefault("FERRIC_TERF_TABLE_DIR", str(TERF_DIR))
    assert (Path(os.environ["FERRIC_TERF_TABLE_DIR"]) / "16_4_2.bin").is_file(), (
        "terfc tables missing"
    )
    import ferric

    files, fraga = s66_files(), s66_fraga()
    dimers = [int(x) for x in a.dimers.split(",")]
    methods = a.methods.split(",")
    calcs = a.calcs.split(",")
    out = Path(a.out)
    done = load_done(out, a.basis)
    gi = git_info()
    for k in dimers:
        atoms = read_atoms(files[k])
        for calc in calcs:
            xyz, nfc = build_calc(atoms, fraga[k], calc)
            for m in methods:
                if (k, calc, m) in done:
                    continue
                e, rhf, dt, extra = compute(m, xyz, nfc, a.basis, a.aux, a.mem_gb)
                rec = {
                    "dimer": k,
                    "name": files[k].name,
                    "calc": calc,
                    "method": m,
                    "energy": e,
                    "rhf_energy": rhf,
                    "wall_s": dt,
                    "frozen_core": nfc,
                    "natoms": xyz.count("\n") - 2,
                    "basis": a.basis,
                    "aux": a.aux,
                    "ferric_build": getattr(ferric, "__build__", None),
                    "script_head": gi["head"],
                    **extra,
                }
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"S66-{k:02d} {calc:8s} {m:9s} E={e:.10f} {dt:7.1f}s", flush=True)


def stats(errs: list[float]) -> dict:
    n = len(errs)
    return {
        "n": n,
        "mean_signed": sum(errs) / n,
        "mae": sum(abs(x) for x in errs) / n,
        "rmsd": math.sqrt(sum(x * x for x in errs) / n),
        "max_abs": max(abs(x) for x in errs),
    }


def cmd_report(a) -> None:
    done = load_done(Path(a.out), a.basis)
    ref = json.loads(REF_JSON.read_text())
    files = s66_files()
    rows, per = {}, {m: {"cp": [], "uncp": []} for m in METHODS}
    for k in sorted({d for d, _, _ in done}):
        row = {
            "dimer": k,
            "name": files[k].name,
            "ref": ref[files[k].stem],
            "wall_s": {},
        }
        for m in METHODS:
            g = lambda c: done.get((k, c, m), {}).get("energy")  # noqa: E731
            es = {c: g(c) for c in CALCS}
            row["wall_s"][m] = sum(
                done[(k, c, m)]["wall_s"] for c in CALCS if (k, c, m) in done
            )
            if es["AB"] is not None and es["A"] is not None and es["B"] is not None:
                row[f"{m}_uncp"] = (es["AB"] - es["A"] - es["B"]) * HARTREE_TO_KCAL
                per[m]["uncp"].append(row[f"{m}_uncp"] - row["ref"])
            if (
                es["AB"] is not None
                and es["A_in_AB"] is not None
                and es["B_in_AB"] is not None
            ):
                row[f"{m}_cp"] = (
                    es["AB"] - es["A_in_AB"] - es["B_in_AB"]
                ) * HARTREE_TO_KCAL
                per[m]["cp"].append(row[f"{m}_cp"] - row["ref"])
        rows[k] = row
    summ = {m: {c: stats(v) for c, v in per[m].items() if v} for m in METHODS}
    res = {
        "dimers_done": sorted(rows),
        "n_dimers": len(rows),
        "full_set": len(rows) == 66,
        "basis": a.basis,
        "aux": a.aux,
        "units": "kcal/mol",
        "published": PUBLISHED,
        "summary": summ,
        "rows": list(rows.values()),
        "provenance": {
            "git": git_info(),
            "python": platform.python_version(),
            "host": platform.node(),
            "hartree_to_kcal": HARTREE_TO_KCAL,
            "ferric_build": next((r["ferric_build"] for r in done.values()), None),
            "reference": "S66x8 CCSD(T)/CBS CP, BEGDB, testdata/reference/c9_refs/s66x8_ccsdt_cbs.json",
            "settings": "RI aux aug-cc-pvtz-rifit; frozen core 1/heavy atom; published default params; SCF default k_builder; "
            "counterpoise: uncorrected and full CP both reported; VV10 NLC grid = library default (n_nlc_points per record)",
            "counterpoise": ["uncp", "cp"],
        },
    }
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(
        f"{len(rows)} dimers; full set: {res['full_set']} (subset stats are NOT comparable to the published full-set RMSD)"
    )
    for m in METHODS:
        for c, s in summ[m].items():
            print(
                f"{m:10s} {c:5s} n={s['n']:2d} MSE={s['mean_signed']:+.3f} MAE={s['mae']:.3f} "
                f"RMSD={s['rmsd']:.3f} max={s['max_abs']:.3f} kcal/mol"
            )
    for r in res["rows"]:
        print(
            f"S66-{r['dimer']:02d} ref={r['ref']:+.2f} "
            + " ".join(
                f"{m}:uncp={r.get(m + '_uncp', float('nan')):+.2f}/cp={r.get(m + '_cp', float('nan')):+.2f}"
                for m in METHODS
            )
            + "  t="
            + "/".join(f"{r['wall_s'][m]:.0f}" for m in METHODS)
            + "s"
        )


def cmd_selftest(a) -> None:
    """Exactness anchors against committed PySCF/numpy references (repo tests' bars)."""
    os.environ.setdefault("FERRIC_TERF_TABLE_DIR", str(TERF_DIR))
    import ferric

    val = ROOT / "testdata/reference/validation"
    fails = 0
    # Python run_* use the library-default SCF (density_conv 1e-6); the Rust validation tests use 1e-9, hence
    # the Rust bars (5e-11) are not reachable here. Measured diffs 5.7e-9 (SCS) and 1.4e-8 (MP2 corr) Ha = 4e-6 kcal/mol.
    TOL = 5e-8

    def chk(name, got, want, tol):
        nonlocal fails
        bad = abs(got - want) > tol
        fails += bad
        print(
            f"{'FAIL' if bad else 'ok  '} {name}: got {got:.12f} want {want:.12f} diff {got - want:+.2e} (tol {tol:.0e})"
        )

    ref = json.loads((val / "scs_mp2_2terfc/h2o_aug-cc-pvdz.json").read_text())
    h2o = (ROOT / "testdata/molecules/validation/h2o.xyz").read_text()
    e = [x for x in ref["energies"] if x["frozen_core"] == 1][0]["scs_2terfc"][
        "total_energy"
    ]
    mol = ferric.Molecule.from_xyz_string(h2o)
    r = ferric.run_scs_mp2_2terfc(
        mol,
        ferric.BasisSet.bundled("aug-cc-pvdz"),
        ferric.BasisSet.bundled("aug-cc-pvdz-rifit"),
        frozen_core=1,
    )
    chk("SCS-MP2(2terfc) H2O/aDZ fc1 total", r.total_energy, e, TOL)

    ref = json.loads((val / "mp2_v/water_dimer_aug-cc-pvdz.json").read_text())
    wd = ferric.Molecule.from_xyz_string(
        (ROOT / "testdata/molecules/s22/water_dimer.xyz").read_text()
    )
    bs, ab = (
        ferric.BasisSet.bundled("aug-cc-pvdz"),
        ferric.BasisSet.bundled("aug-cc-pvdz-rifit"),
    )
    r = ferric.run_rimp2(wd, bs, ab, frozen_core=2)
    chk("RHF water dimer/aDZ", r.rhf_energy, ref["rhf_energy"], 1e-9)
    chk(
        "RI-MP2 corr water dimer/aDZ fc2",
        r.mp2_corr,
        ref["mp2_erfc_control"]["coulomb_e_corr"],
        TOL,
    )
    r = ferric.run_mp2_v(wd, bs, ab, frozen_core=2)
    chk(
        "MP2-V E_nl (b=11, r0=1.0 damped) water dimer/aDZ",
        r.vv10_e_nl,
        ref["vv10"]["b=11.0"]["damped"]["e_nl_numpy"],
        1e-9,
    )
    print("SELFTEST", "FAILED" if fails else "PASSED")
    sys.exit(1 if fails else 0)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("check-split")
    c.add_argument("--dimers", default=",".join(str(i) for i in range(1, 67)))
    for name in ("run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--basis", default=BASIS)
        s.add_argument("--aux", default=AUX)
        if name == "run":
            s.add_argument("--dimers", required=True)
            s.add_argument("--methods", default=",".join(METHODS))
            s.add_argument("--calcs", default=",".join(CALCS))
            s.add_argument("--mem-gb", type=float, default=6.0)
        else:
            s.add_argument("--json", required=True)
    a = p.parse_args()
    if a.cmd == "selftest":
        cmd_selftest(a)
    elif a.cmd == "check-split":
        sys.exit(0 if check_split([int(x) for x in a.dimers.split(",")]) else 1)
    elif a.cmd == "run":
        cmd_run(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
