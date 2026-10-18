"""Interleaved ferric/PySCF timing driver for issue #228 (F P F P ... in one session).

  python scatter_driver.py --out bench_results/timing.jsonl --cells diamond_prim dryice \
      --threads 1 --reps 5 --precision 1e-11

One record per run (JSON line): code, cell, threads, rep, wall (harness wall around the whole
subprocess), load1 before/after, other_cpu_s (box CPU not used by the run's own process tree, from
/proc/stat), a validity flag (gate below), energy, iterations, stage seconds.

GATE (per run): 1-minute loadavg < 1.5 immediately before AND after the run, for N=1.
For N>1 the run's own threads push the 1-minute loadavg above 1.5, so the gate there is the
foreign-CPU fraction other_cpu_s / (wall * ncpu_logical) < 0.02 (before: the same measure over a
2 s window). A run that fails either side is recorded with valid=false and re-run (<= 3 attempts).
Fresh process for every run; PySCF deletes its cderi file (private temp dir) before exit.
Ferric input: toml_ri/<cell>_rhf.toml (ferric defaults, SR column rotation ON) unless --toml-dir
overrides; run from the repo root with the release binary in target/release/ferric.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from omega_sweep import cpu_jiffies, load1, wait_quiet  # noqa: E402

PY = os.environ.get("PYSCF_PY", "/home/matt/qc/ferric/.venv/bin/python")
NCPU = os.cpu_count() or 12


def parse_ferric(out):
    rec = {}
    for key, pat, cast in (
        ("energy", r"energy\s+=\s+(-?[\d.]+) Hartree/cell", float),
        ("iterations", r"iterations\s+=\s+(\d+)", int),
        ("omega", r"gdf_omega\s+=\s+([\d.e+-]+) Bohr", float),
        ("nbasis", r"nbasis\s+=\s+(\d+)", int),
    ):
        m = re.search(pat, out)
        rec[key] = cast(m.group(1)) if m else None
    m = re.search(r"naux (\d+), kept (\d+), dropped (\d+)", out)
    rec["aux"] = dict(naux=int(m.group(1)), kept=int(m.group(2)), dropped=int(m.group(3))) if m else None
    st = {}
    for ln in out.split("\n"):
        m = re.match(r"\s{4}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", ln)
        if m:
            st[m.group(1)] = float(m.group(2))
        m = re.match(r"\s{4}(total)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
        if m:
            st["total"] = float(m.group(2))
        m = re.match(r"\s{4}(other \(SCF linear algebra, bookkeeping\))\s+([\d.]+)\s*$", ln)
        if m:
            st[m.group(1)] = float(m.group(2))
    rec["stages"] = st
    scf = sum(v for k, v in st.items() if k.startswith("scf ") or k.startswith("other ("))
    tot = st.get("total")
    rec["t_scf_s"] = scf
    rec["t_setup_s"] = None if tot is None else tot - scf
    if rec["iterations"]:
        rec["s_per_iter"] = scf / rec["iterations"]
    return rec


def run_cmd(cmd, env, threads):
    full = dict(os.environ)
    full.update(env)
    ok_before = False
    if threads == 1:
        t_end = time.time() + 900
        while time.time() < t_end:
            if load1() < 1.5:
                ok_before = True
                break
            time.sleep(10)
    else:
        ok_before = wait_quiet(max_busy=0.02, window=2.0, timeout=900)
    l0 = load1()
    b0, t0j = cpu_jiffies()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    w0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=full)
    wall = time.time() - w0
    l1 = load1()
    b1, t1j = cpu_jiffies()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    hz = os.sysconf("SC_CLK_TCK")
    mine = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    other = (b1 - b0) / hz - mine
    if threads == 1:
        valid = ok_before and l0 < 1.5 and l1 < 1.5
    else:
        valid = ok_before and (other / (wall * NCPU) < 0.02)
    return p, dict(wall=wall, load_start=l0, load_end=l1, other_cpu_s=other, cpu_s=mine, valid=bool(valid))


def run_ferric(cell, threads, binary, toml_dir, tag=""):
    toml = os.path.join(toml_dir, f"{cell}_rhf{tag}.toml")
    env = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS=str(threads))
    p, g = run_cmd([binary, toml], env, threads)
    rec = dict(code="ferric", cell=cell, threads=threads, toml=os.path.relpath(toml, ROOT), exit=p.returncode, **g)
    rec.update(parse_ferric(p.stdout))
    if p.returncode:
        rec["stderr"] = p.stderr[-500:]
    return rec


def run_pyscf(cell, threads, precision, method="rhf"):
    env = dict(OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    cmd = [PY, os.path.join(HERE, "pyscf_bench.py"), cell, "--threads", str(threads), "--precision", str(precision), "--method", method]
    p, g = run_cmd(cmd, env, threads)
    rec = dict(code="pyscf", cell=cell, threads=threads, exit=p.returncode, precision=precision, **g)
    try:
        inner = json.loads(p.stdout.strip().split("\n")[-1])
        rec["energy"] = inner["energy"]
        rec["iterations"] = inner["cycles"]
        rec["aux"] = inner["aux_metric"]
        rec["omega"] = inner["omega"]
        rec["t_setup_s"] = inner["t_df_build_s"] + inner["t_hcore_s"]
        rec["t_df_build_s"] = inner["t_df_build_s"]
        rec["t_hcore_s"] = inner["t_hcore_s"]
        rec["t_scf_s"] = inner["t_scf_excl_hcore_s"]
        rec["s_per_iter"] = inner["s_per_iter_median"]
        rec["t_inner_total_s"] = inner["t_total_s"]
        rec["peak_rss_bytes"] = inner["peak_rss_bytes"]
        rec["converged"] = inner["converged"]
    except Exception as e:  # noqa: BLE001
        rec["parse_error"] = repr(e)
        rec["stderr"] = p.stderr[-500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--precision", type=float, default=1e-11)
    ap.add_argument("--bin", default=os.path.join(ROOT, "target/release/ferric"))
    ap.add_argument("--toml-dir", default=os.path.join(HERE, "toml_ri"))
    ap.add_argument("--ferric-tag", default="", help="e.g. _unrot to time <cell>_rhf_unrot.toml")
    ap.add_argument("--only", choices=["ferric", "pyscf", "both"], default="both")
    a = ap.parse_args()
    for cell in a.cells:
        for rep in range(a.reps):
            for code in ("ferric", "pyscf"):
                if a.only not in ("both", code):
                    continue
                for attempt in range(3):
                    rec = (
                        run_ferric(cell, a.threads, a.bin, a.toml_dir, a.ferric_tag)
                        if code == "ferric"
                        else run_pyscf(cell, a.threads, a.precision)
                    )
                    rec.update(rep=rep, attempt=attempt, session=os.path.basename(a.out))
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(
                        code, cell, f"N={a.threads}", rep, attempt, f"wall {rec['wall']:.1f}s",
                        f"load {rec['load_start']:.2f}->{rec['load_end']:.2f}",
                        "valid" if rec["valid"] else "INVALID", rec.get("energy"), flush=True,
                    )
                    if rec["valid"] and rec["exit"] == 0:
                        break


if __name__ == "__main__":
    main()
