#!/usr/bin/env python3
"""terfc/terf r0-sweep for SR-MP2 + LR-RPA — SMALL test cases, memory-safe.

Purpose: drive the *tempered* attenuator arm (Dutoi/Goldey terfc, exact via 2D
interpolation tables) of rs-mp2-rpa across the range-separation parameter **r0**
(Bohr), for both formulations B (delta-lr) and T (coupled-rings), and compare its
crossing behaviour against the erf/erfc arm on the SAME machinery. terf + terfc = 1,
so terf is the exact long-range complement — same exact limits as erf/erfc:
  r0 -> inf  (omega -> 0):   terfc -> Coulomb, terf -> 0     => plain MP2
  r0 -> 0    (omega -> inf): terfc -> 0,       terf -> Coulomb => MP2 + dRPA[Coulomb]
The curvature constraint is r0 * omega = 1/sqrt(2), i.e. omega = 1/(r0*sqrt(2)),
so r0 is the SINGLE knob and omega is derived — never set both.

SCOPE: small, fast systems only (water, water dimer, ethene dimer) — this is a
method/shape probe, NOT the benzene aTZ production grid. Runs ONE job at a time
(CONC=1) by default because the box just OOM'd under 3 concurrent benzene jobs +
a subagent build. Small cc-pVDZ jobs are <1.5 GB each so serial is plenty fast
and never contends.

DEPENDENCY (read before launching): the terf arm must exist in the binary. The
CLI [mp2] section must accept `attenuator = "terf"` and `r0 = <Bohr>`. As of the
driver's writing this is being wired in (task #20). Preflight below hard-checks
the binary understands the stanza and ABORTS with a clear message if not, rather
than silently running the erf default. Run the preflight before the sweep.

Conventions reused from finish_benzene_atz.py: OPENBLAS/RAYON=1, child
oom_score_adj=1000, timestamped log + heartbeat, idempotent skip-markers,
additive output dir. r0 in the TOML is BOHR (matches OperatorKind::Terfc.distance);
run_terfc_rimp2's Python API takes Angstrom, but the CLI [mp2] r0 is Bohr — keep
that straight.

Launch (detached, serial):
  setsid nohup python3 benchmarks/omega_diag/terfc_sweep.py \
      >> benchmarks/omega_diag/terfc_sweep.log 2>&1 & disown
Preflight only (no sweep):
  TERFC_PREFLIGHT_ONLY=1 python3 benchmarks/omega_diag/terfc_sweep.py
Tunables (env): TERFC_CONC (1), TERFC_R0_LIST (comma Bohr), TERFC_SYSTEMS
  (comma keys from SYSTEMS), TERFC_HEARTBEAT_S (120), TERFC_TIMEOUT (7200),
  TERFC_PER_JOB_GB (2.0).
"""
import math
import os
import subprocess
import threading
import time

ROOT = "/home/matt/qc/ferric/.claude/worktrees/sr-mp2-lr-rpa"
os.chdir(ROOT)

OUT = "benchmarks/omega_diag/terfc_out"
BIN = "target/release/ferric-cli"

# terf integrals need the interpolation tables. The .bin tables are uncommitted
# and live only at the main-checkout terf-tables/ (the worktree copy is
# generators-only). Auto-resolve FERRIC_TERF_TABLE_DIR if the caller didn't set
# it, so `terf` runs don't silently fail engine creation.
_TERF_DIR = os.environ.get("FERRIC_TERF_TABLE_DIR", "")
if not _TERF_DIR or not os.path.exists(os.path.join(_TERF_DIR, "16_4_2.bin")):
    for cand in ("/home/matt/qc/ferric/terf-tables",
                 f"{ROOT}/terf-tables"):
        if os.path.exists(os.path.join(cand, "16_4_2.bin")):
            _TERF_DIR = cand
            break
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           FERRIC_TERF_TABLE_DIR=_TERF_DIR)

# Small test systems: (key, xyz, basis, aux, frozen_core). cc-pVDZ keeps each
# job <1.5 GB. Geometries live in testdata/molecules; dimers reuse S22 CP geoms.
GEO = f"{ROOT}/benchmarks/grid/geoms"
SYSTEMS = {
    "water":       (f"{ROOT}/testdata/molecules/water.xyz",   "cc-pvdz", "cc-pvdz-ri", 1),
    "water_dimer": (f"{GEO}/s22-02_dimer.xyz",                "cc-pvdz", "cc-pvdz-ri", 2),
    "ethene_dimer":(f"{GEO}/s22-09_dimer.xyz",                "cc-pvdz", "cc-pvdz-ri", 4),
}

# r0 sweep in BOHR. Chosen to bracket the limits and pass through the erf arm's
# operating point (erf omega=0.42 Ang^-1 = 0.2223 Bohr^-1 -> matching terf r0 via
# r0 = 1/(omega*sqrt2) = 1/(0.2223*1.41421) = 3.18 Bohr). Extremes probe limits.
R0_DEFAULT = [0.30, 1.00, 2.00, 3.18, 5.00, 12.0]
R0_LIST = [float(x) for x in os.environ.get("TERFC_R0_LIST", "").split(",") if x.strip()] or R0_DEFAULT
SYS_KEYS = [s for s in os.environ.get("TERFC_SYSTEMS", "").split(",") if s.strip()] or list(SYSTEMS)

FORMS = [("B", "delta-lr"), ("T", "coupled-rings")]

CONC = int(os.environ.get("TERFC_CONC", "1"))
PER_JOB_GB = float(os.environ.get("TERFC_PER_JOB_GB", "2.0"))
HEARTBEAT_S = int(os.environ.get("TERFC_HEARTBEAT_S", "120"))
TIMEOUT = int(os.environ.get("TERFC_TIMEOUT", "7200"))
WAIT_S = int(os.environ.get("TERFC_WAIT_S", "20"))
PREFLIGHT_ONLY = os.environ.get("TERFC_PREFLIGHT_ONLY", "") == "1"

_lock = threading.Lock()
_state = {"running": set(), "done": 0, "total": 0}


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(m):
    print(f"[{ts()}] {m}", flush=True)


def meminfo():
    d = {}
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                d[l.split(":")[0]] = int(l.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return d


def mem_available_gb():
    return meminfo().get("MemAvailable", 0.0)


def omega_of(r0):
    return 1.0 / (r0 * math.sqrt(2.0))


def terf_toml(xyz, r0, form, basis, aux, fc):
    """rs-mp2-rpa with the terf attenuator, swept on r0 (Bohr). omega is DERIVED
    and written for provenance/logs; the CLI must ignore/validate it when
    attenuator='terf' (r0 is authoritative)."""
    return f"""[molecule]
xyz = "{xyz}"
[basis]
name = "{basis}"
[scf]
df_j_aux = "def2-universal-jkfit"
df_k_aux = "def2-universal-jkfit"
max_iter = 400
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "{aux}"
attenuator = "terf"
r0 = {r0}
formulation = "{form}"
frozen_core = {fc}
[rpa]
trunc_thresh = 0.0
[quadrature]
n_points = 12
"""


def erf_toml(xyz, omega, form, basis, aux, fc):
    """Matched erf/erfc arm at the SAME derived omega, for shape comparison."""
    return f"""[molecule]
xyz = "{xyz}"
[basis]
name = "{basis}"
[scf]
df_j_aux = "def2-universal-jkfit"
df_k_aux = "def2-universal-jkfit"
max_iter = 400
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "{aux}"
attenuator = "erf"
omega = {omega}
formulation = "{form}"
frozen_core = {fc}
[rpa]
trunc_thresh = 0.0
[quadrature]
n_points = 12
"""


def out(key):
    return f"{OUT}/out/{key}.out"


def preflight():
    """Verify the binary accepts attenuator='terf' + r0 on a trivial water job.
    ABORT the sweep if the stanza is rejected OR silently ignored — we must not
    record 'terf' numbers that are secretly erf."""
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    if not os.path.exists(BIN):
        log(f"PREFLIGHT FAIL: binary {BIN} missing — build ferric-cli first.")
        return False
    xyz, basis, aux, fc = SYSTEMS["water"]
    tp = f"{OUT}/toml/_preflight_terf.toml"
    open(tp, "w").write(terf_toml(xyz, 3.18, "delta-lr", basis, aux, fc))
    r = subprocess.run([BIN, tp], capture_output=True, text=True, env=ENV,
                       timeout=600)
    combined = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "Total energy" not in combined:
        log("PREFLIGHT FAIL: terf stanza rejected or job errored. The terf arm "
            "is not wired into the CLI yet (task #20). Output tail:")
        for ln in combined.strip().splitlines()[-15:]:
            log(f"    | {ln}")
        return False
    # Guard against silent erf-fallback: unknown-key tolerant parsers would run
    # the erf default. If the binary echoes the attenuator, confirm it says terf.
    low = combined.lower()
    if "attenuator" in low and "terf" not in low.split("attenuator", 1)[1][:40]:
        log("PREFLIGHT FAIL: binary ran but attenuator is not 'terf' — likely "
            "silent erf-fallback (unknown-key tolerance). Refusing to sweep.")
        return False
    log("PREFLIGHT OK: terf stanza accepted, water/cc-pVDZ terf-B ran to "
        "'Total energy'.")
    return True


def heartbeat():
    while True:
        time.sleep(HEARTBEAT_S)
        with _lock:
            r = sorted(_state["running"]); d, t = _state["done"], _state["total"]
        log(f"[hb] {d}/{t} done, avail={mem_available_gb():.1f}GB, "
            f"running={r or '(none)'}")


def admit(key):
    while mem_available_gb() < PER_JOB_GB:
        log(f"[gate] {key}: {mem_available_gb():.1f}GB avail (<{PER_JOB_GB}); wait")
        time.sleep(WAIT_S)


def run_one(job):
    key, toml_text, marker = job
    op = out(key)
    if os.path.exists(op) and marker in open(op).read():
        return "skip", 0.0
    admit(key)
    open(f"{OUT}/toml/{key}.toml", "w").write(toml_text)

    def _oom():
        try:
            open(f"/proc/{os.getpid()}/oom_score_adj", "w").write("1000")
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


def enumerate_jobs():
    jobs = []
    for sk in SYS_KEYS:
        xyz, basis, aux, fc = SYSTEMS[sk]
        for r0 in R0_LIST:
            w = omega_of(r0)
            for tag, form in FORMS:
                # terf arm (swept on r0)
                k = f"{sk}_terf_r0{r0}_{tag}"
                jobs.append((k, terf_toml(xyz, r0, form, basis, aux, fc),
                             "Total energy"))
                # matched erf/erfc arm at the same derived omega (comparison)
                ke = f"{sk}_erf_w{w:.4f}_{tag}"
                jobs.append((ke, erf_toml(xyz, w, form, basis, aux, fc),
                             "Total energy"))
    # de-dup erf jobs that repeat across r0 within a system (same omega only if
    # r0 repeats — it won't here, but keep idempotent by key)
    seen, uniq = set(), []
    for j in jobs:
        if j[0] in seen:
            continue
        seen.add(j[0]); uniq.append(j)
    return uniq


def main():
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    log(f"terfc sweep: systems={SYS_KEYS}, r0(Bohr)={R0_LIST}, forms={[f[0] for f in FORMS]}")
    if not preflight():
        log("ABORT: preflight failed. Wire the terf arm (task #20), rebuild, retry.")
        return
    if PREFLIGHT_ONLY:
        log("PREFLIGHT_ONLY set — stopping after preflight.")
        return
    jobs = enumerate_jobs()
    todo = [j for j in jobs if not (os.path.exists(out(j[0]))
                                    and j[2] in open(out(j[0])).read())]
    _state["total"] = len(todo)
    log(f"{len(jobs)} total job keys, {len(todo)} to run, CONC={CONC}")
    if not todo:
        log("nothing to do — terfc sweep already complete.")
    else:
        threading.Thread(target=heartbeat, daemon=True).start()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=CONC) as ex:
            futs = {ex.submit(run_one, j): j for j in todo}
            for fut in as_completed(futs):
                status, dt = fut.result()
                key = futs[fut][0]
                with _lock:
                    _state["done"] += 1
                    d, t = _state["done"], _state["total"]
                log(f"[{d}/{t}] {status:8s} {dt:7.1f}s  {key}")
    log("done.")


if __name__ == "__main__":
    main()
