#!/usr/bin/env python3
"""Finish the benzene aug-cc-pVTZ CP omega-crossing — CONCURRENT, post-T12.

Why this supersedes the serial finish_benzene_atz.py: the T12 paneled full-rank
eigensolve (main commit 56a7459, run_lanczos_full_rank) dropped the benzene
aTZ-dimer peak transient from ~17 GB to ~6 GB (measured 2026-07-07, live RSS
trace: SCF ~2.2 GB, RPA eigensolve peak ~6.2 GB, steady tail ~4.7 GB), bit-
identical full-rank (Δ=0 @ 10 digits per the commit). The old "memory-bound to
serial, one 17 GB job at a time" constraint ([[atz-benzene-rpa-memory-bound]])
is GONE. At ~6-7 GB/job a 23 GB box holds 2-3 concurrent jobs.

Concurrency at the JOB level (each job RAYON=1), NOT threads within a job:
- Bit-reproducible: RAYON=1 means no cross-thread summation reordering anywhere
  (the RI-K try_reduce non-determinism is fixed in a8ec76f/8038338, but RAYON=1
  sidesteps it entirely — belt and suspenders for the reported crossing numbers).
- Better scaling than in-job threading: the monomer probe showed RAYON=4 buys
  only ~1.4x on one job (SCF prelude + serial phases dominate), whereas 3 jobs
  RAYON=1 is a clean ~3x throughput.
- Full-rank preserved (trunc_thresh=0); no PDEP truncation ([[no-pdep-trunc-in-method-grid]]).

Reuses ALL of finish_benzene_atz.py's proven machinery (enumerate_jobs, toml
builders, skip-markers, output dir) by import — same idempotent, additive,
nothing-recomputed behavior. Only the scheduler changes: a memory-aware pool
that admits a new job only when MemAvailable >= per-job headroom, so it self-
throttles from CONC down to 1 under pressure and never OOMs.

Safety: OPENBLAS/RAYON=1 per child, child oom_score_adj=1000 (kernel sacrifices
a ferric child, never the session), timestamped log + heartbeat, mem-gated
admission, yields to the ACONF scan.

Launch (detached):
  setsid nohup python3 benchmarks/omega_diag/finish_benzene_atz_par.py \
      >> benchmarks/omega_diag/finish_benzene.log 2>&1 & disown
Tunables (env): BZ_CONC (3), BZ_PER_JOB_GB (7 — admission headroom per job),
  BZ_HEARTBEAT_S (120), BZ_YIELD_TO (default "run_aconf_cli"), BZ_TIMEOUT (21600).
"""
import os
import subprocess
import threading
import time

import finish_benzene_atz as base

ROOT = base.ROOT
os.chdir(ROOT)
OUT, BIN, ENV = base.OUT, base.BIN, base.ENV

CONC = int(os.environ.get("BZ_CONC", "3"))
PER_JOB_GB = float(os.environ.get("BZ_PER_JOB_GB", "7"))
HEARTBEAT_S = int(os.environ.get("BZ_HEARTBEAT_S", "120"))
TIMEOUT = int(os.environ.get("BZ_TIMEOUT", "21600"))
WAIT_S = int(os.environ.get("BZ_WAIT_S", "30"))

_lock = threading.Lock()
_state = {"running": set(), "done": 0, "total": 0}


def log(m):
    base.log(m)


def admit(key):
    """Block until the box can hold ANOTHER job: no yielder contending AND
    MemAvailable >= PER_JOB_GB (headroom for one more ~6-7 GB job on top of
    whatever is already running)."""
    while True:
        y = base.yielders_running()
        if y:
            log(f"[gate] {key}: yielding to '{y}'; waiting {WAIT_S}s")
            time.sleep(WAIT_S)
            continue
        avail = base.mem_available_gb()
        if avail >= PER_JOB_GB:
            return
        with _lock:
            nr = len(_state["running"])
        log(f"[gate] {key}: {avail:.1f}GB avail (<{PER_JOB_GB}); "
            f"{nr} running; waiting {WAIT_S}s")
        time.sleep(WAIT_S)


def heartbeat():
    while True:
        time.sleep(HEARTBEAT_S)
        with _lock:
            r = sorted(_state["running"])
            d, t = _state["done"], _state["total"]
        mi = base.meminfo()
        log(f"[hb] {d}/{t} done, avail={mi.get('MemAvailable', 0):.1f}GB, "
            f"running={r or '(none)'}")


def run_one(job):
    key, toml, marker = job
    op = base.out(key)
    if os.path.exists(op) and marker in open(op).read():
        return "skip", 0.0
    admit(key)
    open(f"{OUT}/toml/{key}.toml", "w").write(toml)

    def _oom():
        try:
            with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
                f.write("1000")
        except Exception:
            pass

    with _lock:
        _state["running"].add(key)
    t0 = time.monotonic()
    status = "FAIL"
    try:
        with open(op, "w") as f, open(op + ".err", "w") as e:
            subprocess.run([BIN, f"{OUT}/toml/{key}.toml"], stdout=f, stderr=e,
                           env=ENV, timeout=TIMEOUT, preexec_fn=_oom)
        ok = os.path.exists(op) and marker in open(op).read()
        status = "ok" if ok else "FAIL"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    finally:
        with _lock:
            _state["running"].discard(key)
    return status, time.monotonic() - t0


def main():
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    jobs = base.enumerate_jobs()
    _state["total"] = len(jobs)
    if not jobs:
        log("nothing to do — benzene aTZ tail already complete.")
    else:
        log(f"benzene aTZ tail: {len(jobs)} jobs, CONC={CONC}, "
            f"per-job-gate={PER_JOB_GB}GB, yield-to={base.YIELD_TO or '(none)'}")
        threading.Thread(target=heartbeat, daemon=True).start()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=CONC) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                status, dt = fut.result()
                key = futs[fut][0]
                with _lock:
                    _state["done"] += 1
                    d, t = _state["done"], _state["total"]
                log(f"[{d}/{t}] {status:8s} {dt:8.1f}s  {key}")
    analysis = "benchmarks/omega_diag/derisk_atz_cp.py"
    if os.path.exists(analysis):
        log("running analysis stage (derisk_atz_cp.py)")
        subprocess.run(["python3", analysis], env=ENV)
    log("done.")


if __name__ == "__main__":
    main()
