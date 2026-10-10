"""RS-GDF omega sweep driver (issue #227). One ferric CLI run per (cell, omega, repeat), under
scripts/ferric-limited, OPENBLAS_NUM_THREADS=1, RAYON_NUM_THREADS=6. Appends one JSON line per run
(wall, load before/after, energy, stage timings, counters) to the output file. Run from the repo root:

  python reference/pbc/bench/omega_sweep.py --out sweep.jsonl --cells diamond_prim dryice \
      --omegas 0.4 0.7 1.0 --reps 3 [--split] [--auto]

--omegas accepts numbers, 'default' (no gdf_omega key: today's 1.0 build) and 'auto'
(`gdf_omega = "auto"`, the per-cell cost-model choice).
Cells are <name>.xyz/.lattice in this directory; basis cc-pvdz, aux cc-pvdz-ri.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def toml_text(cell, omega, split, basis="cc-pvdz", aux="cc-pvdz-ri"):
    rows = [
        r.split()
        for r in open(os.path.join(HERE, cell + ".lattice")).read().split("\n")
        if r.strip()
    ]
    lat = ", ".join("[" + ", ".join(repr(float(v)) for v in r) + "]" for r in rows)
    lines = [
        "[molecule]",
        f'xyz = "reference/pbc/bench/{cell}.xyz"',
        "",
        "[basis]",
        f'name = "{basis}"',
        "",
        "[method]",
        'kind = "rhf"',
        "",
        "[cell]",
        'unit = "bohr"',
        f"lattice = [{lat}]",
        'exxdiv = "ewald"',
        'jk = "rsgdf"',
        f'auxbasis = "{aux}"',
    ]
    if split:
        lines.append("range_split = true")
    if omega == "auto":
        lines.append('gdf_omega = "auto"')
    elif omega != "default":
        lines.append(f"gdf_omega = {float(omega)}")
    lines += ["", "[memory]", "budget_gb = 10", ""]
    return "\n".join(lines)


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def cpu_jiffies():
    """(busy, total) jiffies over all CPUs from /proc/stat."""
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v) - idle, sum(v)


def wait_quiet(max_busy=0.04, window=2.0, timeout=900):
    """Block until the box's CPU busy fraction over a `window` s sample is below `max_busy`."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        b0, t0 = cpu_jiffies()
        time.sleep(window)
        b1, t1 = cpu_jiffies()
        if (b1 - b0) / max(t1 - t0, 1) < max_busy:
            return True
    return False


def run_one(cell, omega, split, threads, binary):
    with tempfile.TemporaryDirectory() as td:
        toml = os.path.join(td, "run.toml")
        open(toml, "w").write(toml_text(cell, omega, split))
        quiet = wait_quiet()
        l0 = load1()
        b0, tj0 = cpu_jiffies()
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        p = subprocess.run(
            [
                os.path.join(ROOT, "scripts/ferric-limited"),
                "--max=10G",
                "--high=8G",
                "--",
                "env",
                "OPENBLAS_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
                f"RAYON_NUM_THREADS={threads}",
                binary,
                toml,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        wall = time.time() - t0
        l1 = load1()
        b1, tj1 = cpu_jiffies()
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    out = p.stdout
    rec = dict(
        cell=cell,
        omega=omega,
        split=split,
        threads=threads,
        exit=p.returncode,
        wall=wall,
        load_start=l0,
        load_end=l1,
        quiet_at_start=quiet,
    )
    hz = os.sysconf("SC_CLK_TCK")
    mine = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    # CPU seconds burned on the whole box during the run that this run's process tree did not use
    rec["other_cpu_s"] = (b1 - b0) / hz - mine
    rec["cpu_s"] = mine
    m = re.search(r"energy\s+=\s+(-?[\d.]+) Hartree/cell", out)
    rec["energy"] = float(m.group(1)) if m else None
    m = re.search(r"gdf_omega\s+=\s+([\d.e+-]+) Bohr", out)
    rec["gdf_omega_used"] = float(m.group(1)) if m else None
    m = re.search(r"nbasis\s+=\s+(\d+)", out)
    rec["nbasis"] = int(m.group(1)) if m else None
    m = re.search(r"iterations\s+=\s+(\d+)", out)
    rec["iters"] = int(m.group(1)) if m else None
    st = {}
    for ln in out.split("\n"):
        m = re.match(r"\s{4}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", ln)
        if m:
            st[m.group(1)] = float(m.group(2))
        m = re.match(r"\s{4}(\S.*?)\s{2,}(\d+)\s*$", ln)
        if m:
            st["#" + m.group(1)] = int(m.group(2))
    rec["stages"] = st
    if p.returncode:
        rec["stderr"] = p.stderr[-500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--omegas", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--bin", default=os.path.join(ROOT, "target/release/ferric"))
    a = ap.parse_args()
    for cell in a.cells:
        for rep in range(a.reps):
            for w in a.omegas:
                rec = run_one(cell, w, a.split, a.threads, a.bin)
                rec["rep"] = rep
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    cell,
                    w,
                    rep,
                    f"{rec['wall']:.2f}s load {rec['load_start']:.2f}->{rec['load_end']:.2f}",
                    rec["energy"],
                    flush=True,
                )


if __name__ == "__main__":
    main()
