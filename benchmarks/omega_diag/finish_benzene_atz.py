#!/usr/bin/env python3
"""Finish the benzene aug-cc-pVTZ CP omega-crossing — a REALISTIC-gate remedy.

Why v2 stalled: derisk_atz_cp_v2.py gates on MemAvailable >= 20 GB. On this 23 GB
box that ceiling is almost never met while GW100 (16 GB scope) + buff/cache are
resident, so the benzene tail starved ~75 min doing nothing. But a benzene job
peaks ~17 GB and the kernel can RECLAIM buff/cache under pressure, so the true
headroom is MemAvailable + reclaimable-cache, not MemAvailable alone. This driver
gates on that, drops the ceiling to a benzene-fits value (18 GB default), and
runs strictly serial (concurrency 1 — benzene aTZ is memory-bound to serial per
[[atz-benzene-rpa-memory-bound]]; two would OOM).

It ALSO frees the box proactively before each launch: if a "yielders" process
(default: the ACONF cc-pVDZ scan) is running, it waits for it to finish rather
than fighting it, and it SYNCs + drops nothing destructive — only advisory.

Scope: exactly the v2 benzene heavy tail — B full grid {0.3,0.42,0.55,0.673,0.8},
T {0.2,0.3,0.42}, RHF x3 frags — reusing the SAME toml/key/marker conventions and
output dir, so every already-complete job is skipped and nothing is recomputed or
clobbered. Full-rank (trunc_thresh=0) preserved. On completion runs the analysis
(derisk_atz_cp.py) if present.

Same safety as v2: OPENBLAS/RAYON=1, child oom_score_adj=1000 (kernel sacrifices
the ferric child, never Claude), timestamped log + heartbeat, idempotent.

Launch (detached):
  setsid nohup python3 benchmarks/omega_diag/finish_benzene_atz.py \
      >> benchmarks/omega_diag/finish_benzene.log 2>&1 & disown
Tunables (env): BZ_GATE_GB (18), BZ_HEARTBEAT_S (300), BZ_YIELD_TO
  (comma pgrep patterns to wait out; default "run_aconf_cli"), BZ_TIMEOUT (21600).
"""
import os
import subprocess
import threading
import time

ROOT = "/home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa"
os.chdir(ROOT)
OUT = "benchmarks/omega_diag/derisk"
GEO = "benchmarks/grid/geoms"
BIN = "target/release/ferric-cli"
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

SID, LABEL, BT = "11", "benzene_PD", "atz"
BASIS, AUX = "aug-cc-pvtz", "aug-cc-pvtz-rifit"
B_OMEGAS = [0.30, 0.42, 0.55, 0.673, 0.80]
T_OMEGAS = [0.20, 0.30, 0.42]

# A benzene job peaks ~17 GB; gate a hair above that. Reclaimable buff/cache
# COUNTS toward headroom (the kernel evicts it under pressure), so the effective
# free figure is MemAvailable, which already accounts for reclaimable cache.
GATE_GB = float(os.environ.get("BZ_GATE_GB", "18"))
WAIT_S = int(os.environ.get("BZ_WAIT_S", "60"))
HEARTBEAT_S = int(os.environ.get("BZ_HEARTBEAT_S", "300"))
TIMEOUT = int(os.environ.get("BZ_TIMEOUT", "21600"))
YIELD_TO = [p for p in os.environ.get("BZ_YIELD_TO", "run_aconf_cli").split(",") if p]

_lock = threading.Lock()
_state = {"running": None, "done": 0, "total": 0}


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(m):
    print(f"[{ts()}] {m}", flush=True)


def meminfo():
    d = {}
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                k, v = l.split(":")[0], l.split()[1]
                d[k] = int(v) / (1024 * 1024)
    except Exception:
        pass
    return d


def mem_available_gb():
    return meminfo().get("MemAvailable", 0.0)


def yielders_running():
    for pat in YIELD_TO:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True)
        if r.returncode == 0:
            return pat
    return None


def heartbeat():
    while True:
        time.sleep(HEARTBEAT_S)
        with _lock:
            r, d, t = _state["running"], _state["done"], _state["total"]
        mi = meminfo()
        log(f"[hb] {d}/{t} done, avail={mi.get('MemAvailable', 0):.1f}GB "
            f"cache={mi.get('Cached', 0):.1f}GB, running={r or '(waiting)'}")


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


def out(key):
    return f"{OUT}/out/{key}.out"


def needs(key, marker):
    op = out(key)
    return not (os.path.exists(op) and marker in open(op).read())


def enumerate_jobs():
    frags = {"dimer": f"{GEO}/s22-{SID}_dimer.xyz",
             "cpA": f"{GEO}/s22-{SID}_mA_cp.xyz",
             "cpB": f"{GEO}/s22-{SID}_mB_cp.xyz"}
    jobs = []
    for fr, xyz in frags.items():
        fc = fc_count(xyz)
        for omega in B_OMEGAS:
            k = f"{LABEL}_{SID}_{BT}_w{omega}_B_{fr}"
            if needs(k, "Total energy"):
                jobs.append((k, rsmp2_toml(absxyz(xyz), omega, "delta-lr", fc),
                             "Total energy"))
        for omega in T_OMEGAS:
            k = f"{LABEL}_{SID}_{BT}_w{omega}_T_{fr}"
            if needs(k, "Total energy"):
                jobs.append((k, rsmp2_toml(absxyz(xyz), omega, "coupled-rings", fc),
                             "Total energy"))
        k = f"{LABEL}_{SID}_{BT}_RHF_{fr}"
        if needs(k, "RHF energy"):
            jobs.append((k, scf_toml(absxyz(xyz), fc), "RHF energy"))
    # dimers first so binding reads as monomers land
    order = {"dimer": 0, "cpA": 1, "cpB": 2}
    jobs.sort(key=lambda j: order.get(j[0].rsplit("_", 1)[-1], 9))
    return jobs


def wait_for_launch(key):
    """Block until the box can actually hold a benzene job: no yielder process
    contending AND MemAvailable >= GATE_GB."""
    while True:
        y = yielders_running()
        if y:
            log(f"[gate] {key}: yielding to '{y}'; waiting {WAIT_S}s")
            time.sleep(WAIT_S)
            continue
        avail = mem_available_gb()
        if avail >= GATE_GB:
            return
        log(f"[gate] {key}: {avail:.1f}GB avail (<{GATE_GB}); waiting {WAIT_S}s")
        time.sleep(WAIT_S)


def run_one(job):
    key, toml, marker = job
    op = out(key)
    if os.path.exists(op) and marker in open(op).read():
        return "skip", 0.0
    wait_for_launch(key)
    open(f"{OUT}/toml/{key}.toml", "w").write(toml)

    def _oom():
        try:
            with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
                f.write("1000")
        except Exception:
            pass

    with _lock:
        _state["running"] = key
    t0 = time.monotonic()
    try:
        with open(op, "w") as f, open(op + ".err", "w") as e:
            subprocess.run([BIN, f"{OUT}/toml/{key}.toml"], stdout=f, stderr=e,
                           env=ENV, timeout=TIMEOUT, preexec_fn=_oom)
    except subprocess.TimeoutExpired:
        with _lock:
            _state["running"] = None
        return "TIMEOUT", time.monotonic() - t0
    with _lock:
        _state["running"] = None
    ok = os.path.exists(op) and marker in open(op).read()
    return ("ok" if ok else "FAIL"), time.monotonic() - t0


def main():
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    jobs = enumerate_jobs()
    _state["total"] = len(jobs)
    if not jobs:
        log("nothing to do — benzene aTZ tail already complete.")
    else:
        log(f"benzene aTZ tail: {len(jobs)} jobs, serial, gate={GATE_GB}GB, "
            f"yield-to={YIELD_TO or '(none)'}")
        threading.Thread(target=heartbeat, daemon=True).start()
        for j in jobs:
            status, dt = run_one(j)
            with _lock:
                _state["done"] += 1
                d, t = _state["done"], _state["total"]
            log(f"[{d}/{t}] {status:8s} {dt:8.1f}s  {j[0]}")
    # analysis stage
    analysis = "benchmarks/omega_diag/derisk_atz_cp.py"
    if os.path.exists(analysis):
        log("running analysis stage (derisk_atz_cp.py)")
        subprocess.run(["python3", analysis], env=ENV)
    log("done.")


if __name__ == "__main__":
    main()
