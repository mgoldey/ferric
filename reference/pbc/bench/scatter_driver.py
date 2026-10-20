"""Interleaved ferric/PySCF timing driver for issue #228 (F P F P ... in one session).

  python scatter_driver.py --out bench_results/timing.jsonl --cells diamond_prim dryice \
      --threads 1 --reps 5 --precision 1e-11

One record per run (JSON line): code, cell, threads, rep, wall (harness wall around the whole
subprocess), load1 before/after, other_cpu_s (box CPU not used by the run's own process tree, from
/proc/stat), a validity flag (gate below), energy, iterations, stage seconds.

GATE (per run): 1-minute loadavg < 1.5 immediately before AND after the run, for N=1.
For N>1 the run's own threads push the 1-minute loadavg above 1.5, so the gate there is the
foreign-CPU fraction other_cpu_s / (wall * ncpu_logical) < 0.12 (the ambient box baseline measured here is 4-6%) (before: the same measure over a
2 s window). A run that fails either side is recorded with valid=false and re-run (<= 3 attempts).
Fresh process for every run; PySCF deletes its cderi file (private temp dir) before exit.
Ferric input: toml_ri/<cell>_<method><tag>.toml (ferric defaults, SR column rotation ON) unless
--toml-dir overrides; run from the repo root with the release binary in target/release/ferric.
A cell named <base>_k<abc> (e.g. diamond_prim_k222) runs PySCF on <base> with --kmesh a b c.

Addendum (issue #228 second session): every run holds BOTH heavy slots (/tmp/ferric-compile.lock and
/tmp/ferric-compile.2.lock, exclusive flock, fds not inherited by the child = `flock -o`) from
BEFORE the quiet-wait until the run exits, so no other ferric agent compiles or tests while the gate
is evaluated or the run is timed.  The code order alternates per rep (F P on even reps, P F on odd).
Peak RSS per run is the child's own ru_maxrss from wait4; ferric --version and `converged` are
recorded per ferric run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from omega_sweep import cpu_jiffies, load1, wait_quiet  # noqa: E402

PY = os.environ.get("PYSCF_PY", sys.executable)
NCPU = os.cpu_count() or 12
# The machine-wide heavy-slot lock files of the cargo shim (a convention, not a temp file).
# N=1 end-of-run gate: "load" = 1-min loadavg < 1.5 after the run (first session, pre-registered);
# "foreign" = foreign-CPU fraction < 0.12 (the N>1 rule).  The start gate (loadavg < 1.5) is kept in both.
N1_GATE = os.environ.get("N1_GATE", "load")
LOCKS = (
    "/tmp/ferric-compile.lock",  # nosec B108
    "/tmp/ferric-compile.2.lock",  # nosec B108
)


def is_scf_stage(name):
    """ferric stage rows counted as SCF (Gamma: "scf J/K ...", k-mesh: "k SCF")."""
    return name.startswith(("scf ", "k SCF", "other ("))


def parse_ferric(out):
    rec = {}
    for key, pat, cast in (
        ("energy", r"energy\s+=\s+(-?[\d.]+) Hartree/cell", float),
        ("iterations", r"iterations\s+=\s+(\d+)", int),
        ("omega", r"gdf_omega\s+=\s+([\d.e+-]+) Bohr", float),
        ("nbasis", r"nbasis\s+=\s+(\d+)", int),
        ("converged", r"converged\s+=\s+(true|false)", lambda v: v == "true"),
    ):
        m = re.search(pat, out)
        rec[key] = cast(m.group(1)) if m else None
    m = re.search(r"naux (\d+), kept (\d+), dropped (\d+)", out)
    rec["aux"] = (
        dict(naux=int(m.group(1)), kept=int(m.group(2)), dropped=int(m.group(3)))
        if m
        else None
    )
    m = re.search(r"naux (\d+), dropped (\d+) \(max over q", out)
    if m and rec["aux"] is None:
        rec["aux"] = dict(naux=int(m.group(1)), dropped_max_over_q=int(m.group(2)))
    st = {}
    for ln in out.split("\n"):
        m = re.match(r"\s{4}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", ln)
        if m:
            st[m.group(1)] = float(m.group(2))
        m = re.match(r"\s{4}(total)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
        if m:
            st["total"] = float(m.group(2))
        m = re.match(
            r"\s{4}(other \(SCF linear algebra, bookkeeping\))\s+([\d.]+)\s*$", ln
        )
        if m:
            st[m.group(1)] = float(m.group(2))
    rec["stages"] = st
    scf = sum(v for k, v in st.items() if is_scf_stage(k))
    tot = st.get("total")
    rec["t_scf_s"] = scf
    rec["t_setup_s"] = None if tot is None else tot - scf
    if rec["iterations"]:
        rec["s_per_iter"] = scf / rec["iterations"]
    return rec


def acquire_locks():
    fds = []
    t0 = time.time()
    for path in LOCKS:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        fds.append(fd)
    return fds, time.time() - t0


def release_locks(fds):
    for fd in fds:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def run_child(cmd, env):
    """Run cmd; return (returncode, stdout, stderr, own peak RSS bytes from wait4)."""
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, cwd=ROOT, env=env, text=True)
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        fo.seek(0)
        fe.seek(0)
        return p.returncode, fo.read(), fe.read(), ru.ru_maxrss * 1024


def run_cmd(cmd, env, threads):
    fds, lock_wait = acquire_locks()
    try:
        return _run_cmd_locked(cmd, env, threads, lock_wait)
    finally:
        release_locks(fds)


def _run_cmd_locked(cmd, env, threads, lock_wait):
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
        ok_before = wait_quiet(max_busy=0.12, window=2.0, timeout=900)
    l0 = load1()
    b0, t0j = cpu_jiffies()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    w0 = time.time()
    rc, out, err, rss = run_child(cmd, full)
    wall = time.time() - w0
    l1 = load1()
    b1, t1j = cpu_jiffies()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    hz = os.sysconf("SC_CLK_TCK")
    mine = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    other = (b1 - b0) / hz - mine
    if threads == 1 and N1_GATE == "foreign":
        valid = ok_before and l0 < 1.5 and (other / (wall * NCPU) < 0.12)
    elif threads == 1:
        valid = ok_before and l0 < 1.5 and l1 < 1.5
    else:
        valid = ok_before and (other / (wall * NCPU) < 0.12)
    p = subprocess.CompletedProcess(cmd, rc, out, err)
    return p, dict(
        wall=wall,
        n1_gate=N1_GATE if threads == 1 else None,
        lock_wait_s=lock_wait,
        foreign_cpu_frac=other / (wall * NCPU),
        child_peak_rss_bytes=rss,
        load_start=l0,
        load_end=l1,
        other_cpu_s=other,
        cpu_s=mine,
        valid=bool(valid),
    )


def ferric_version(binary):
    p = subprocess.run([binary, "--version"], capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


def run_ferric(cell, threads, binary, toml_dir, tag="", method="rhf"):
    toml = os.path.join(toml_dir, f"{cell}_{method}{tag}.toml")
    version = ferric_version(binary)
    env = dict(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS=str(threads)
    )
    p, g = run_cmd([binary, toml], env, threads)
    rec = dict(
        code="ferric",
        cell=cell,
        method=method,
        threads=threads,
        ferric_version=version,
        toml=os.path.relpath(toml, ROOT),
        exit=p.returncode,
        **g,
    )
    rec.update(parse_ferric(p.stdout))
    if p.returncode:
        rec["stderr"] = p.stderr[-500:]
    return rec


def split_kmesh(cell):
    m = re.match(r"^(.*)_k(\d)(\d)(\d)$", cell)
    return (m.group(1), [m.group(2), m.group(3), m.group(4)]) if m else (cell, None)


def run_pyscf(cell, threads, precision, method="rhf", grid=None):
    env = dict(OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    base, kmesh = split_kmesh(cell)
    cmd = [
        PY,
        os.path.join(HERE, "pyscf_bench.py"),
        base,
        "--threads",
        str(threads),
        "--precision",
        str(precision),
        "--method",
        method,
    ] + (["--kmesh"] + kmesh if kmesh else [])
    if grid:
        cmd += ["--grid", str(grid[0]), str(grid[1])]
    p, g = run_cmd(cmd, env, threads)
    rec = dict(
        code="pyscf",
        cell=cell,
        method=method,
        threads=threads,
        exit=p.returncode,
        precision=precision,
        **g,
    )
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
        rec["hcore_calls"] = inner.get("hcore_calls")
        rec["pyscf_version"] = inner.get("pyscf_version")
        rec["conv_tol"] = inner.get("conv_tol")
        rec["conv_tol_grad"] = inner.get("conv_tol_grad")
        rec["grid"] = inner.get("grid")
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
    ap.add_argument(
        "--ferric-tag", default="", help="e.g. _unrot to time <cell>_rhf_unrot.toml"
    )
    ap.add_argument("--only", choices=["ferric", "pyscf", "both"], default="both")
    ap.add_argument("--method", choices=["rhf", "pbe"], default="rhf")
    ap.add_argument("--pyscf-grid", type=int, nargs=2, default=None)
    ap.add_argument(
        "--fixed-order",
        action="store_true",
        help="F P every rep (the first session's order); default alternates F P / P F",
    )
    a = ap.parse_args()
    for cell in a.cells:
        for rep in range(a.reps):
            order = ("ferric", "pyscf")
            if rep % 2 and not a.fixed_order:
                order = ("pyscf", "ferric")
            for code in order:
                if a.only not in ("both", code):
                    continue
                for attempt in range(3):
                    rec = (
                        run_ferric(
                            cell, a.threads, a.bin, a.toml_dir, a.ferric_tag, a.method
                        )
                        if code == "ferric"
                        else run_pyscf(
                            cell, a.threads, a.precision, a.method, a.pyscf_grid
                        )
                    )
                    rec.update(
                        rep=rep, attempt=attempt, session=os.path.basename(a.out)
                    )
                    # A run without a parsed energy cannot pass the energy gate.
                    if rec.get("energy") is None or "parse_error" in rec:
                        rec["valid"] = False
                    if rec.get("converged") is False:
                        rec["valid"] = False
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(
                        code,
                        cell,
                        f"N={a.threads}",
                        rep,
                        attempt,
                        f"wall {rec['wall']:.1f}s",
                        f"load {rec['load_start']:.2f}->{rec['load_end']:.2f}",
                        "valid" if rec["valid"] else "INVALID",
                        rec.get("energy"),
                        flush=True,
                    )
                    if rec["valid"] and rec["exit"] == 0:
                        break


if __name__ == "__main__":
    main()
