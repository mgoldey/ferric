#!/usr/bin/env python3
"""Parallel executor for the remaining aTZ CP ω-sweep jobs (benzene tail).

The serial derisk_atz_cp.py runs one ferric job at a time = 1 core on a 12-core
box. A single ferric job is BLAS-serial (threading is a no-op for one job), so
throughput = run MANY jobs at once, each pinned to 1 thread. This driver does
exactly that with a MEMORY-aware concurrency cap.

Binding constraint here is RAM, not cores: a benzene aTZ rs-mp2-rpa job (532 bf
with ghosts, full-rank dRPA) peaks at ~5-8 GB RSS. With ~13 GB free and other
jobs (GW100 sweeps) sharing the box, default concurrency is 3. Override with
ATZ_CP_JOBS.

ADDITIVE / idempotent: identical toml generation + skip-marker logic to
derisk_atz_cp.py; same output paths. Skips any output already carrying its
completion marker, never overwrites. Safe to run alongside the serial script
(they skip each other's completed outputs); but prefer running THIS one alone.

After running, invoke derisk_atz_cp.py for its analysis/report stage (it will
find every job already complete and just write DERISK_ATZ_CP.md).

Each child: OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1.
"""
import os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa"
os.chdir(ROOT)
OUT = "benchmarks/omega_diag/derisk"
GEO = "benchmarks/grid/geoms"
BIN = "target/release/ferric-cli"
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

ANCHORS = [("01", "ammonia_HB"), ("02", "water_HB"), ("08", "methane_D"),
           ("09", "ethene_D"), ("11", "benzene_PD")]
OMEGAS = [0.30, 0.42, 0.55, 0.673, 0.80]
FORMS = [("delta-lr", "B"), ("coupled-rings", "T")]
BASIS, AUX, BT = "aug-cc-pvtz", "aug-cc-pvtz-rifit", "atz"

# MEASURED: a benzene aTZ rs-mp2-rpa job (532 bf w/ ghosts, full-rank dRPA,
# Davidson max_vecs=3·naux) PEAKS at ~17 GB RSS. On a 23 GB box that means
# exactly ONE benzene job fits — two would OOM into a 1 GB swap and thrash
# (far slower than serial, likely OOM-killed). So benzene parallelism is
# memory-forbidden, not a scheduler bug: the correct concurrency is 1.
# Small anchors (≤~250 bf) are cheap and already complete, so in practice this
# driver only ever runs the benzene tail ⇒ default JOBS=1.
# Override only if per-job memory is reduced or the box has more RAM.
JOBS = int(os.environ.get("ATZ_CP_JOBS", "1"))
PER_JOB_GB = float(os.environ.get("ATZ_CP_PER_JOB_GB", "17"))
TIMEOUT = int(os.environ.get("ATZ_CP_TIMEOUT", "21600"))  # 6h/job ceiling


def _mem_safe_jobs(requested, per_job_gb=PER_JOB_GB):
    """Clamp concurrency to what RAM allows: available_GB / per_job_gb, ≥1.
    Prevents the OOM/swap-thrash that a naive high JOBS would cause."""
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
        avail_gb = mi.get("MemAvailable", 0) / (1024 * 1024)
    except Exception:
        return max(1, requested)
    cap = max(1, int(avail_gb // per_job_gb))
    return max(1, min(requested, cap))


def fc_count(xyz):
    n = 0
    for ln in open(xyz).read().splitlines()[2:]:
        if not ln.strip():
            continue
        s = ln.split()[0]
        if s.startswith('@') or s.upper().startswith('H'):
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
    for sid, label in ANCHORS:
        frags = {"dimer": f"{GEO}/s22-{sid}_dimer.xyz",
                 "cpA": f"{GEO}/s22-{sid}_mA_cp.xyz",
                 "cpB": f"{GEO}/s22-{sid}_mB_cp.xyz"}
        for fr, xyz in frags.items():
            fc = fc_count(xyz)
            # rs-mp2-rpa jobs
            for omega in OMEGAS:
                for form, ftag in FORMS:
                    key = f"{label}_{sid}_{BT}_w{omega}_{ftag}_{fr}"
                    if needs_run(key, "Total energy"):
                        jobs.append((key, rsmp2_toml(absxyz(xyz), omega, form, fc),
                                     "Total energy"))
            # per-fragment RHF (omega-independent)
            frtag = {"dimer": "dimer", "cpA": "cpA", "cpB": "cpB"}[fr]
            key = f"{label}_{sid}_{BT}_RHF_{frtag}"
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


# A benzene aTZ job peaks ~17 GB. Other sessions' jobs on this shared box (GW100
# sweeps) spike unpredictably (seen 2.5 GB → 20 GB). The startup-time mem clamp is
# not enough: we must RE-CHECK right before each launch and WAIT for headroom,
# else a co-tenant spike + our 17 GB = OOM that previously killed Claude Code.
PREFLIGHT_GB = float(os.environ.get("ATZ_CP_PREFLIGHT_GB", "20"))
PREFLIGHT_WAIT_S = int(os.environ.get("ATZ_CP_PREFLIGHT_WAIT_S", "60"))
PREFLIGHT_MAX_WAIT_S = int(os.environ.get("ATZ_CP_PREFLIGHT_MAX_WAIT_S", "10800"))


def _wait_for_memory(key):
    """Block until MemAvailable ≥ PREFLIGHT_GB, so a heavy job never starts into a
    box that can't hold it. Gives up (and lets the job try anyway) only after
    PREFLIGHT_MAX_WAIT_S so the sweep can't hang forever."""
    waited = 0
    while _mem_available_gb() < PREFLIGHT_GB:
        if waited >= PREFLIGHT_MAX_WAIT_S:
            print(f"[preflight] {key}: waited {waited}s, still <{PREFLIGHT_GB}GB "
                  f"free — launching anyway (cap reached)", flush=True)
            return
        print(f"[preflight] {key}: only {_mem_available_gb():.1f}GB free "
              f"(<{PREFLIGHT_GB}); waiting {PREFLIGHT_WAIT_S}s", flush=True)
        time.sleep(PREFLIGHT_WAIT_S)
        waited += PREFLIGHT_WAIT_S


def _is_heavy(key):
    return "benzene" in key


def run_one(job):
    key, toml, marker = job
    op = out(key)
    # double-check skip (another worker/script may have just finished it)
    if os.path.exists(op) and marker in open(op).read():
        return key, "skip", 0.0
    # heavy jobs: gate on real free memory right before launch
    if _is_heavy(key):
        _wait_for_memory(key)
    open(f"{OUT}/toml/{key}.toml", 'w').write(toml)
    t0 = time.monotonic()
    # Make the kernel OOM-killer prefer THIS ferric child over everything else
    # (esp. Claude Code): bump its oom_score_adj to the max after spawn.
    def _raise_oom_score():
        try:
            with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
                f.write("1000")
        except Exception:
            pass
    try:
        with open(op, 'w') as f, open(op + ".err", 'w') as e:
            subprocess.run([BIN, f"{OUT}/toml/{key}.toml"], stdout=f, stderr=e,
                           env=ENV, timeout=TIMEOUT, preexec_fn=_raise_oom_score)
    except subprocess.TimeoutExpired:
        return key, "TIMEOUT", time.monotonic() - t0
    dt = time.monotonic() - t0
    ok = os.path.exists(op) and marker in open(op).read()
    return key, ("ok" if ok else "FAIL"), dt


def _run_pool(jobs, concurrency, tag):
    if not jobs:
        return
    print(f"[par:{tag}] {len(jobs)} jobs, concurrency={concurrency}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            key, status, dt = fut.result()
            done += 1
            print(f"[{tag} {done}/{len(jobs)}] {status:8s} {dt:7.1f}s  {key}",
                  flush=True)


def main():
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    jobs = enumerate_jobs()
    if not jobs:
        print("[par] nothing to do — all aTZ CP jobs complete.", flush=True)
        return

    # Two memory tiers run in two passes:
    #   light  = the 4 small anchors (≤~250 bf, <1 GB each) — parallelize wide.
    #   heavy  = benzene aTZ (~17 GB each) — memory forbids >1 at a time.
    # Running light first clears the cheap RHF/analysis-input jobs fast, then the
    # box is free for the serial benzene tail.
    light = [j for j in jobs if "benzene" not in j[0]]
    heavy = [j for j in jobs if "benzene" in j[0]]

    # heavy ordering: dimers before monomers so binding can be read as it lands.
    order = {"dimer": 0, "cpA": 1, "cpB": 2}
    heavy.sort(key=lambda j: order.get(j[0].rsplit("_", 1)[-1], 9))

    # light jobs are ≤~250 bf, <1.5 GB each → wide; heavy uses PER_JOB_GB(17).
    light_conc = _mem_safe_jobs(int(os.environ.get("ATZ_CP_LIGHT_JOBS", "6")),
                                per_job_gb=1.5)
    heavy_conc = _mem_safe_jobs(JOBS)
    print(f"[par] {len(light)} light + {len(heavy)} heavy jobs; "
          f"light_conc={light_conc}, heavy_conc={heavy_conc} "
          f"(PER_JOB_GB={PER_JOB_GB}), 1 thread/job, {TIMEOUT}s/job", flush=True)

    _run_pool(light, light_conc, "light")
    # re-clamp heavy after light frees its memory
    _run_pool(heavy, _mem_safe_jobs(JOBS), "heavy")
    print("[par] done.", flush=True)


if __name__ == "__main__":
    main()
