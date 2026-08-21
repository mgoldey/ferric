#!/usr/bin/env python3
"""Pyrazine-dimer (S22-12) counterpoise ω-sweep for SR-MP2+LR-RPA at aug-cc-pVDZ.

Mirrors benchmarks/omega_diag/derisk_sweep.py / derisk_cp_arm.py /
derisk_atz_cp_par.py conventions exactly:
  - same [molecule]/[basis]/[scf]/[method]/[mp2]/[rpa]/[quadrature] TOML shape
  - same fc_count() convention (ghosts + H excluded, heavy real atoms counted)
  - same output/toml directories (benchmarks/omega_diag/derisk/{out,toml})
  - same idempotent skip: output already contains "Total energy" (rs-mp2-rpa)
    or "RHF energy" (scs-mp2 RHF-reference job) => skip, never overwrite
  - same env: OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 per child
  - same oom_score_adj=1000 preexec pattern as derisk_atz_cp_par.py

CP ARM ONLY: fragments are dimer, cpA (real A + ghost B), cpB (ghost A + real
B). No plain (non-CP) monomers here — those aren't needed for the CP binding
energy and pyrazine at aDZ is cheap enough to just run all fragments per ω.

ω grid: [0.20, 0.30, 0.42, 0.55, 0.673, 0.80] — note 0.20 is NEW vs the old
5-point grid; it probes the T-formulation (coupled-rings) operating point.

Formulations: delta-lr (B, default) and coupled-rings (T).

Concurrency: memory-gated, at most 2 concurrent jobs; before each dispatch,
require /proc/meminfo MemAvailable >= 6 GB, else sleep 60s and retry (the box
is shared with other sessions' jobs). Per-job timeout 7200s.
"""
from pathlib import Path
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = str(Path(__file__).resolve().parents[2])
os.chdir(ROOT)
OUT = "benchmarks/omega_diag/derisk"
GEO = "benchmarks/grid/geoms"
BIN = "target/release/ferric-cli"
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

SID, LABEL, BTAG = "12", "pyrazine_D", "adz"
BASIS, AUX = "aug-cc-pvdz", "aug-cc-pvdz-rifit"
OMEGAS = [0.20, 0.30, 0.42, 0.55, 0.673, 0.80]
FORMS = [("delta-lr", "B"), ("coupled-rings", "T")]

MAX_CONCURRENCY = int(os.environ.get("PYR_CP_JOBS", "2"))
MIN_AVAIL_GB = float(os.environ.get("PYR_CP_MIN_AVAIL_GB", "6"))
MEM_WAIT_S = int(os.environ.get("PYR_CP_MEM_WAIT_S", "60"))
TIMEOUT = int(os.environ.get("PYR_CP_TIMEOUT", "7200"))


def fc_count(xyz):
    """Frozen core = # of REAL (non-ghost, non-H) atoms. Ghosts (@) have zero
    electrons -> contribute zero frozen core. Copied verbatim from
    derisk_cp_arm.py / derisk_atz_cp_par.py."""
    n = 0
    for ln in open(xyz).read().splitlines()[2:]:
        if not ln.strip():
            continue
        sym = ln.split()[0]
        if sym.startswith('@'):
            continue
        if sym.upper().startswith('H'):
            continue
        n += 1
    return n


def absxyz(p):
    return p if p.startswith("/") else f"{ROOT}/{p}"


def out(key):
    return f"{OUT}/out/{key}.out"


def rsmp2_toml(xyz, omega, form, fc):
    return f"""[molecule]
xyz = "{xyz}"
[basis]
name = "{BASIS}"
[scf]
df_j_aux = "def2-universal-jkfit"
df_k_aux = "def2-universal-jkfit"
max_iter = 400
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "{AUX}"
omega = {omega}
formulation = "{form}"
frozen_core = {fc}
[rpa]
trunc_thresh = 0.0
[quadrature]
n_points = 12
"""


def scf_toml(xyz, fc):
    return f"""[molecule]
xyz = "{xyz}"
[basis]
name="{BASIS}"
[scf]
df_j_aux="def2-universal-jkfit"
df_k_aux="def2-universal-jkfit"
max_iter=400
[method]
kind="scs-mp2"
[mp2]
auxbasis="{AUX}"
frozen_core={fc}
"""


def needs_run(key, marker):
    op = out(key)
    return not (os.path.exists(op) and marker in open(op).read())


def enumerate_jobs():
    """Build the full (key, toml, marker) work-list, then drop already-complete."""
    jobs = []
    frags = {
        "dimer": f"{GEO}/s22-{SID}_dimer.xyz",
        "cpA": f"{GEO}/s22-{SID}_mA_cp.xyz",
        "cpB": f"{GEO}/s22-{SID}_mB_cp.xyz",
    }
    for fr, xyz in frags.items():
        fc = fc_count(xyz)
        for omega in OMEGAS:
            for form, ftag in FORMS:
                key = f"{LABEL}_{SID}_{BTAG}_w{omega}_{ftag}_{fr}"
                if needs_run(key, "Total energy"):
                    jobs.append((key, rsmp2_toml(absxyz(xyz), omega, form, fc),
                                 "Total energy"))
        key = f"{LABEL}_{SID}_{BTAG}_RHF_{fr}"
        if needs_run(key, "RHF energy"):
            jobs.append((key, scf_toml(absxyz(xyz), fc), "RHF energy"))
    return jobs


def _mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemAvailable"):
                    return int(l.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _wait_for_memory(key):
    """Block until MemAvailable >= MIN_AVAIL_GB, sleeping MEM_WAIT_S between
    checks. The box is shared with other sessions' jobs (GW100 sweeps, cargo
    test); never launch a job into a box that can't hold it."""
    while _mem_available_gb() < MIN_AVAIL_GB:
        print(f"[preflight] {key}: only {_mem_available_gb():.1f}GB free "
              f"(<{MIN_AVAIL_GB}); waiting {MEM_WAIT_S}s", flush=True)
        time.sleep(MEM_WAIT_S)


def _raise_oom_score():
    try:
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
            f.write("1000")
    except Exception:
        pass


def run_one(job):
    key, toml, marker = job
    op = out(key)
    # double-check skip (another worker/script may have just finished it)
    if os.path.exists(op) and marker in open(op).read():
        return key, "skip", 0.0
    _wait_for_memory(key)
    open(f"{OUT}/toml/{key}.toml", 'w').write(toml)
    t0 = time.monotonic()
    try:
        with open(op, 'w') as f, open(op + ".err", 'w') as e:
            subprocess.run([BIN, f"{OUT}/toml/{key}.toml"], stdout=f, stderr=e,
                           env=ENV, timeout=TIMEOUT, preexec_fn=_raise_oom_score)
    except subprocess.TimeoutExpired:
        return key, "TIMEOUT", time.monotonic() - t0
    dt = time.monotonic() - t0
    ok = os.path.exists(op) and marker in open(op).read()
    return key, ("ok" if ok else "FAIL"), dt


def main():
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    jobs = enumerate_jobs()
    total_possible = 3 * (len(OMEGAS) * len(FORMS) + 1)  # 3 frags * (6*2 + RHF)
    print(f"[pyrazine-cp] {len(jobs)} jobs to run (of {total_possible} total; "
          f"{total_possible - len(jobs)} already complete), "
          f"concurrency={MAX_CONCURRENCY}, min_avail_gb={MIN_AVAIL_GB}, "
          f"timeout={TIMEOUT}s", flush=True)
    if not jobs:
        print("[pyrazine-cp] nothing to do — all jobs complete.", flush=True)
        return

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            key, status, dt = fut.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {status:8s} {dt:7.1f}s  {key}",
                  flush=True)
    print("PYRAZINE CP SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
