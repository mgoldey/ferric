#!/usr/bin/env python3
"""Plan an r0 sweep against what is ALREADY on disk, and emit it as JSON.

The problem this solves: `r0_sweep` runs N points on one SCF, so a TOML is
written with a fixed point list. If some of those points already exist, the
runner's per-JOB idempotence check cannot help — it only skips a job whose
output file is complete, not a job that needs 1 of 3 points. Re-running a
generator by hand trims once and then goes stale the moment new data lands.

So the plan is DATA, not a side effect: this reads the existing outputs, writes
`{system, fragment, r0_have, r0_need}` to a JSON file, and (optionally)
materializes TOMLs containing only the missing points. Re-run it any time; it
recomputes from disk, so it converges rather than double-counting.

Usage:
  plan_sweep.py --form B --basis aqz --r0 0.7,0.8,0.9 --tag r0Bmin --plan-only
  plan_sweep.py --form B --basis aqz --r0 0.7,0.8,0.9 --tag r0Bmin --write

`--plan-only` writes just the JSON (safe to run repeatedly, touches no TOMLs).
`--write` additionally creates/trims/removes TOMLs to match the plan.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mae_spline as M  # noqa: E402

FRAGS = ("dimer", "mA_cp", "mB_cp")
# Core orbitals of REAL atoms. Ghosts (@X) carry no electrons and must be
# EXCLUDED -- counting them mismatched 14 of 21 existing TOMLs, all CP monomers.
CORE = {"H": 0, "HE": 0, "LI": 1, "BE": 1, "B": 1, "C": 1, "N": 1, "O": 1,
        "F": 1, "NE": 1, "NA": 5, "MG": 5, "AL": 5, "SI": 5, "P": 5, "S": 5,
        "CL": 5, "AR": 5}


def frozen_core(xyz):
    n = 0
    for line in Path(xyz).read_text().splitlines()[2:]:
        if not line.strip():
            continue
        tok = line.split()[0]
        if tok.startswith("@"):
            continue
        n += CORE.get(tok.lstrip("@").upper(), 0)
    return n


def n_atoms(xyz):
    return int(Path(xyz).read_text().splitlines()[0])


def build_plan(basis, form, r0_want, systems):
    """{(sys, frag): {'have': [...], 'need': [...]}} from outputs on disk."""
    data = M.collect(basis, form)
    present = {}
    for r0, per in data.items():
        r0r = round(r0, 4)
        if r0r not in r0_want:
            continue
        for idx, frags in per.items():
            for f in frags:
                present.setdefault((idx, f), set()).add(r0r)

    rows = []
    for s in systems:
        for frag in FRAGS:
            xyz = ROOT / "geoms" / f"a24-{s:02d}_{frag}.xyz"
            if not xyz.exists():
                rows.append({"system": s, "fragment": frag, "status": "no_geometry",
                             "r0_have": [], "r0_need": []})
                continue
            have = sorted(present.get((s, frag), set()))
            need = [r for r in r0_want if r not in have]
            rows.append({
                "system": s,
                "fragment": frag,
                "xyz": str(xyz.relative_to(ROOT.parent.parent)),
                "n_atoms": n_atoms(xyz),
                "frozen_core": frozen_core(xyz),
                "r0_have": have,
                "r0_need": need,
                "status": "complete" if not need else
                          ("partial" if have else "missing"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=("B", "T"), default="B")
    ap.add_argument("--basis", default="aqz")
    ap.add_argument("--r0", required=True, help="comma-separated, e.g. 0.7,0.8,0.9")
    ap.add_argument("--tag", required=True, help="filename tag, e.g. r0Bmin")
    ap.add_argument("--systems", default="", help="comma A24 indices (default 1-24)")
    ap.add_argument("--template", default=None, help="TOML to clone (default: an existing sweep)")
    ap.add_argument("--out", default=None, help="plan JSON path")
    ap.add_argument("--write", action="store_true", help="also create/trim/remove TOMLs")
    a = ap.parse_args()

    r0_want = [round(float(x), 4) for x in a.r0.split(",")]
    systems = ([int(x) for x in a.systems.split(",") if x.strip()]
               if a.systems else list(range(1, 25)))
    rows = build_plan(a.basis, a.form, r0_want, systems)

    need_pts = sum(len(r["r0_need"]) for r in rows)
    have_pts = sum(len(r["r0_have"]) for r in rows)
    doc = {
        "basis": a.basis, "formulation": a.form, "tag": a.tag,
        "r0_requested": r0_want,
        "units": "r0 in Angstrom",
        "summary": {
            "jobs_total": len(rows),
            "jobs_complete": sum(1 for r in rows if r["status"] == "complete"),
            "jobs_partial": sum(1 for r in rows if r["status"] == "partial"),
            "jobs_missing": sum(1 for r in rows if r["status"] == "missing"),
            "r0_points_already_on_disk": have_pts,
            "r0_points_to_run": need_pts,
        },
        "note": ("Re-run this to replan; it recomputes from outputs on disk, so "
                 "it is idempotent and converges as data lands. A CP triple needs "
                 "all three fragments before it yields an interaction energy."),
        "jobs": rows,
    }
    out = Path(a.out) if a.out else ROOT / "scans" / f"plan_{a.basis}_{a.form}_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    s = doc["summary"]
    print(f"wrote {out}")
    print(f"  jobs: {s['jobs_total']} total / {s['jobs_complete']} complete / "
          f"{s['jobs_partial']} partial / {s['jobs_missing']} missing")
    print(f"  r0 points: {s['r0_points_already_on_disk']} on disk, "
          f"{s['r0_points_to_run']} to run")

    if not a.write:
        return 0

    tpl_path = Path(a.template) if a.template else ROOT / "toml" / "a24-12_dimer_aqz_r0fine_B.toml"
    tpl = tpl_path.read_text()
    created = trimmed = removed = 0
    for r in rows:
        if r["status"] == "no_geometry":
            continue
        stem = f"a24-{r['system']:02d}_{r['fragment']}"
        p = ROOT / "toml" / f"{stem}_aqz_{a.tag}_{a.form}.toml"
        if not r["r0_need"]:
            if p.exists():
                p.unlink()
                removed += 1
            continue
        sweep = "[" + ", ".join(f"{x:.4f}" for x in r["r0_need"]) + "]"
        t = re.sub(r'xyz = "[^"]+"', f'xyz = "benchmarks/grid/geoms/{stem}.xyz"', tpl)
        t = re.sub(r"r0_sweep = \[[^\]]*\]", f"r0_sweep = {sweep}", t)
        t = re.sub(r"frozen_core = \d+", f"frozen_core = {r['frozen_core']}", t)
        existed = p.exists()
        old = p.read_text() if existed else None
        if old != t:
            p.write_text(t)
            trimmed += 1 if existed else 0
            created += 0 if existed else 1
    print(f"  TOMLs: {created} created, {trimmed} updated, {removed} removed "
          f"(already complete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
