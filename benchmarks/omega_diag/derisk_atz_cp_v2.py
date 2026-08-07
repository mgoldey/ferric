#!/usr/bin/env python3
"""RE-SCOPED aTZ CP benzene tail (v2) — supersedes derisk_atz_cp_par.py's plan.

Scope change vs v1 (evidence-driven, see DERISK_ATZ_CP.md interim + aDZ report):
  * B  (delta-lr):      full grid  ω ∈ {0.30, 0.42, 0.55, 0.673, 0.80} — benzene only
                        (light anchors already complete at these ω).
  * T  (coupled-rings): ω ∈ {0.20, 0.30, 0.42} ONLY.
      - {0.20, 0.30}: T's aDZ-CP MAE minimum sits AT the old grid edge (0.30,
        still falling) — its optimum is likely ~0.2, which v1 never sampled.
      - 0.42: one mid point so the paper can show the B-vs-T head-to-head on the
        discriminating π-stack at B's operating ω (documents T's over-coupling).
      - Dropping T @ {0.55, 0.673, 0.80} saves 9 × (~17 GB × hours) benzene jobs
        we already know T loses (aDZ CP: T MAE rises monotonically past 0.30).
  * T @ 0.20 also needs the 4 LIGHT anchors (new grid point ⇒ new light jobs),
    so the ω=0.20 MAE is an n=5 row, not benzene-only.
  * benzene RHF ×3 frags (scs-mp2 toml, "RHF energy" marker) still needed.

Total: 15 (B bz) + 9 (T bz) + 3 (RHF bz) = 27 heavy + 12 light  (vs v1's 33 all-heavy plan).

Operational fixes vs v1 (v1 died silently on 2026-06-26 and its log was
indistinguishable from a healthy preflight wait for TEN DAYS):
  * every log line timestamped;
  * a heartbeat thread prints progress + MemAvailable + running keys every 5 min,
    so `tail` alone distinguishes DEAD from WAITING;
  * light tier also memory-gated (the pyrazine aDZ sweep shares the box).

Same conventions as v1: additive/idempotent (skip on completion marker, never
overwrite), identical toml template & key naming, OPENBLAS/RAYON=1 per child,
child oom_score_adj=1000 so the OOM killer never takes Claude/other sessions.

After completion run derisk_atz_cp.py for the analysis stage (extend its OMEGAS
to include 0.20/0.42-T rows if it filters them).
"""
import os, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/home/matt/qc/ferric"
os.chdir(ROOT)
OUT = "benchmarks/omega_diag/derisk"
GEO = "benchmarks/grid/geoms"
BIN = "target/release/ferric-cli"
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

LIGHT_ANCHORS = [("01", "ammonia_HB"), ("02", "water_HB"),
                 ("08", "methane_D"), ("09", "ethene_D")]
BZ = ("11", "benzene_PD")
B_OMEGAS = [0.30, 0.42, 0.55, 0.673, 0.80]     # benzene, delta-lr
T_OMEGAS = [0.20, 0.30, 0.42]                  # benzene, coupled-rings
T_LIGHT_OMEGAS = [0.20]                        # new grid point for the anchors
BASIS, AUX, BT = "aug-cc-pvtz", "aug-cc-pvtz-rifit", "atz"

TIMEOUT = int(os.environ.get("ATZ_CP_TIMEOUT", "21600"))
PREFLIGHT_HEAVY_GB = float(os.environ.get("ATZ_CP_PREFLIGHT_GB", "20"))
PREFLIGHT_LIGHT_GB = float(os.environ.get("ATZ_CP_LIGHT_PREFLIGHT_GB", "4"))
PREFLIGHT_WAIT_S = int(os.environ.get("ATZ_CP_PREFLIGHT_WAIT_S", "60"))
HEARTBEAT_S = int(os.environ.get("ATZ_CP_HEARTBEAT_S", "300"))
LIGHT_CONC = int(os.environ.get("ATZ_CP_LIGHT_JOBS", "3"))

_lock = threading.Lock()
_running = set()
_done = [0]
_total = [0]


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def _mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemAvailable"):
                    return int(l.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _heartbeat():
    while True:
        time.sleep(HEARTBEAT_S)
        with _lock:
            run = sorted(_running)
            d, t = _done[0], _total[0]
        log(f"[hb] {d}/{t} done, mem={_mem_available_gb():.1f}GB, "
            f"running={run if run else '(waiting)'}")


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


def frags_for(sid):
    return {"dimer": f"{GEO}/s22-{sid}_dimer.xyz",
            "cpA": f"{GEO}/s22-{sid}_mA_cp.xyz",
            "cpB": f"{GEO}/s22-{sid}_mB_cp.xyz"}


def enumerate_jobs():
    light, heavy = [], []
    # NEW light: T @ 0.20 on the 4 small anchors
    for sid, label in LIGHT_ANCHORS:
        for fr, xyz in frags_for(sid).items():
            fc = fc_count(xyz)
            for omega in T_LIGHT_OMEGAS:
                key = f"{label}_{sid}_{BT}_w{omega}_T_{fr}"
                if needs_run(key, "Total energy"):
                    light.append((key, rsmp2_toml(absxyz(xyz), omega,
                                                  "coupled-rings", fc),
                                  "Total energy"))
    # HEAVY: benzene tail (re-scoped)
    sid, label = BZ
    for fr, xyz in frags_for(sid).items():
        fc = fc_count(xyz)
        for omega in B_OMEGAS:
            key = f"{label}_{sid}_{BT}_w{omega}_B_{fr}"
            if needs_run(key, "Total energy"):
                heavy.append((key, rsmp2_toml(absxyz(xyz), omega,
                                              "delta-lr", fc), "Total energy"))
        for omega in T_OMEGAS:
            key = f"{label}_{sid}_{BT}_w{omega}_T_{fr}"
            if needs_run(key, "Total energy"):
                heavy.append((key, rsmp2_toml(absxyz(xyz), omega,
                                              "coupled-rings", fc),
                              "Total energy"))
        key = f"{label}_{sid}_{BT}_RHF_{fr}"
        if needs_run(key, "RHF energy"):
            heavy.append((key, scf_toml(absxyz(xyz), fc), "RHF energy"))
    # heavy ordering: dimers first so binding is readable as monomers land
    order = {"dimer": 0, "cpA": 1, "cpB": 2}
    heavy.sort(key=lambda j: order.get(j[0].rsplit("_", 1)[-1], 9))
    return light, heavy


def _wait_for_memory(key, need_gb):
    while _mem_available_gb() < need_gb:
        log(f"[preflight] {key}: {_mem_available_gb():.1f}GB free "
            f"(<{need_gb}); waiting {PREFLIGHT_WAIT_S}s")
        time.sleep(PREFLIGHT_WAIT_S)


def run_one(job, need_gb):
    key, toml, marker = job
    op = out(key)
    if os.path.exists(op) and marker in open(op).read():
        return key, "skip", 0.0
    _wait_for_memory(key, need_gb)
    with _lock:
        _running.add(key)
    open(f"{OUT}/toml/{key}.toml", 'w').write(toml)
    t0 = time.monotonic()

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
    finally:
        with _lock:
            _running.discard(key)
    dt = time.monotonic() - t0
    ok = os.path.exists(op) and marker in open(op).read()
    return key, ("ok" if ok else "FAIL"), dt


def _run_pool(jobs, concurrency, tag, need_gb):
    if not jobs:
        return
    log(f"[par:{tag}] {len(jobs)} jobs, concurrency={concurrency}, "
        f"preflight={need_gb}GB")
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(run_one, j, need_gb): j[0] for j in jobs}
        for fut in as_completed(futs):
            key, status, dt = fut.result()
            with _lock:
                _done[0] += 1
                d, t = _done[0], _total[0]
            log(f"[{tag} {d}/{t}] {status:8s} {dt:7.1f}s  {key}")


def main():
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    light, heavy = enumerate_jobs()
    _total[0] = len(light) + len(heavy)
    if _total[0] == 0:
        log("[par] nothing to do — all v2 aTZ CP jobs complete.")
        return
    log(f"[par] v2 scope: {len(light)} light (T@0.2 anchors) + "
        f"{len(heavy)} heavy (benzene: B×{len(B_OMEGAS)}ω, T×{len(T_OMEGAS)}ω, "
        f"RHF) jobs")
    threading.Thread(target=_heartbeat, daemon=True).start()
    _run_pool(light, LIGHT_CONC, "light", PREFLIGHT_LIGHT_GB)
    _run_pool(heavy, 1, "heavy", PREFLIGHT_HEAVY_GB)
    log("[par] done.")


if __name__ == "__main__":
    main()
