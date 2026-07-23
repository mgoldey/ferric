#!/usr/bin/env python3
"""Standalone ethylene terf/terfc r0-sweep launcher.

Self-contained: does NOT depend on benchmarks/omega_diag/terfc_sweep.py (that
file lives on a git branch and can drift/revert independently of this script,
same lesson as scripts/run_benzene_atz_sweep.py). Runnable from any shell,
cron, or outside Claude Code entirely -- plain python3 + systemd-run (via
scripts/ferric-limited), no other deps.

Drives the *tempered* attenuator arm (Dutoi/Goldey terf/terfc, exact via 2D
interpolation tables) of rs-mp2-rpa across the range-separation parameter r0
(Å at the CLI/TOML boundary -- FIXED 2026-07-21, [mp2] r0 used to be Bohr,
now Å, matching r0_bonded/r0_nonbonded's always-Å convention; see
crates/ferric-cli/src/config.rs's r0 doc) on the ethylene monomer and
ethylene dimer (S22 D2d), for both
formulations B (delta-lr) and T (coupled-rings), plus a matched erf/erfc arm
at the derived omega for shape comparison. terf + terfc = 1 exactly, so terf
is the long-range complement of terfc -- same exact limits as erf/erfc:
  r0 -> inf  (omega -> 0):    terfc -> Coulomb, terf -> 0      => plain MP2
  r0 -> 0    (omega -> inf):  terfc -> 0,       terf -> Coulomb => MP2 + dRPA[Coulomb]
The curvature constraint is r0 * omega = 1/sqrt(2), i.e. omega = 1/(r0*sqrt(2)) --
r0 is the single knob, omega is DERIVED, never set both independently.

Small systems (ethylene monomer 6 atoms, dimer 12 atoms) at cc-pVDZ are cheap
(<1.5 GB/job) -- no ferric-limited memory concern like the benzene aTZ grid,
but every job is still wrapped in it for consistency/safety, and every job
TOML still sets [memory] budget_gb explicitly for the same reason as the
benzene sweep (forces the streamed/blocked 3-index path instead of an
auto-detected budget that can materialize the whole AO tensor in-core --
irrelevant at this tiny basis size, but cheap and harmless to set).

PREFLIGHT runs first and ABORTS the sweep if the binary rejects the terf
stanza or silently falls back to erf -- we must never record "terf" numbers
that are secretly erf. Also verifies the terf-tables directory (needed for
the 2D interpolation) is discoverable.

WHY LD_LIBRARY_PATH is set: some ferric-cli builds on this shared checkout
link against MPI (libmpi.so.40 in ~/.local/lib, not on the default loader
path -- observed 2026-07-20 when another session's mpi_rimp2 build replaced
the binary mid-sweep). Prepending it here means this script runs regardless
of which binary variant is currently built.

Usage:
  python3 scripts/run_ethylene_terfc_sweep.py
  nohup python3 scripts/run_ethylene_terfc_sweep.py \
      >> benchmarks/omega_diag/terfc_out/ethylene_sweep_standalone.log 2>&1 &
      disown

Env tunables:
  ET_R0_LIST         (0.1588,0.5292,1.0584,1.6828,2.6459,6.3501) -- comma Å
                     values (2026-07-21: converted from the old Bohr-valued
                     default 0.30,1.00,2.00,3.18,5.00,12.0 by dividing each
                     by 1.8897259886 -- same physical r0 grid, correct unit)
  ET_SYSTEMS         (monomer,dimer) -- comma keys from SYSTEMS
  ET_FERRIC_MAX_GB   (6)    -- ferric-limited hard cap per job
  ET_FERRIC_HIGH_GB  (5)    -- ferric-limited soft throttle per job
  ET_MEMORY_BUDGET_GB (2.0) -- [memory] budget_gb baked into every job TOML
  ET_RAYON_THREADS   (8)    -- rayon worker threads (OPENBLAS stays pinned
                               at 1 regardless -- see openblas-rayon-dgetrf-crash)
  ET_GATE_GB         (4)    -- pre-launch admission gate (MemAvailable must
                               clear this before starting the next job)
  ET_WAIT_S          (20)   -- seconds to sleep between gate re-checks
  ET_TIMEOUT         (7200) -- per-job subprocess timeout (seconds)
  ET_TABLE_DIR       (auto) -- FERRIC_TERF_TABLE_DIR override
"""
import math
import os
import subprocess
import time

ROOT = "/home/matt/qc/ferric"
os.chdir(ROOT)

OUT = "benchmarks/omega_diag/terfc_out"
BIN = "target/release/ferric-cli"
FERRIC_LIMITED = "scripts/ferric-limited"

RAYON_NUM_THREADS = os.environ.get("ET_RAYON_THREADS", "8")

_TERF_DIR = os.environ.get("ET_TABLE_DIR", "")
if not _TERF_DIR or not os.path.exists(os.path.join(_TERF_DIR, "16_4_2.bin")):
    for cand in ("/home/matt/qc/ferric/terf-tables", f"{ROOT}/terf-tables"):
        if os.path.exists(os.path.join(cand, "16_4_2.bin")):
            _TERF_DIR = cand
            break

_LD_LIBRARY_PATH = os.pathsep.join(
    p for p in ["/home/matt/.local/lib", os.environ.get("LD_LIBRARY_PATH", "")] if p)

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS=RAYON_NUM_THREADS,
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           LD_LIBRARY_PATH=_LD_LIBRARY_PATH, FERRIC_TERF_TABLE_DIR=_TERF_DIR)

# (xyz, basis, aux, frozen_core)
SYSTEMS = {
    "monomer": (f"{ROOT}/testdata/molecules/thiel_set/ethylene.xyz", "cc-pvdz", "cc-pvdz-ri", 2),
    "dimer":   (f"{ROOT}/testdata/molecules/s22/ethylene_dimer.xyz", "cc-pvdz", "cc-pvdz-ri", 4),
}

# Å values (2026-07-21: converted from the old Bohr-valued
# [0.30, 1.00, 2.00, 3.18, 5.00, 12.0] by /1.8897259886 -- [mp2] r0 is now Å
# at the CLI boundary, see crates/ferric-cli/src/config.rs's r0 doc).
R0_DEFAULT = [0.1588, 0.5292, 1.0584, 1.6828, 2.6459, 6.3501]
R0_LIST = [float(x) for x in os.environ.get("ET_R0_LIST", "").split(",") if x.strip()] or R0_DEFAULT
SYS_KEYS = [s for s in os.environ.get("ET_SYSTEMS", "").split(",") if s.strip()] or list(SYSTEMS)
FORMS = [("B", "delta-lr"), ("T", "coupled-rings")]

FERRIC_MAX_GB = os.environ.get("ET_FERRIC_MAX_GB", "6")
FERRIC_HIGH_GB = os.environ.get("ET_FERRIC_HIGH_GB", "5")
MEMORY_BUDGET_GB = os.environ.get("ET_MEMORY_BUDGET_GB", "2.0")
GATE_GB = float(os.environ.get("ET_GATE_GB", "4"))
WAIT_S = int(os.environ.get("ET_WAIT_S", "20"))
TIMEOUT = int(os.environ.get("ET_TIMEOUT", "7200"))


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(m):
    print(f"[{ts()}] {m}", flush=True)


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


BOHR_INV_PER_ANG_INV = 1.0 / 1.8897259886  # crates/ferric-mp2/src/attenuated.rs:142


def omega_of(r0_ang):
    """Angstrom^-1, matched to the given r0 (now Å -- 2026-07-21, [mp2] r0
    is Å at the CLI boundary, crates/ferric-cli/src/config.rs's r0 doc) for
    erf_toml()'s omega= field (the CLI parses [mp2] omega as Angstrom^-1 --
    main.rs:597, omega_ang_inv * BOHR_INV_PER_ANG_INV). The terf r0<->omega
    relation itself (main.rs:691, w_derived = 1/(r0*sqrt(2))) is computed in
    Bohr internally, so r0 must convert to Bohr FIRST, then the resulting
    omega (Bohr^-1) converts to Angstrom^-1 by dividing by
    BOHR_INV_PER_ANG_INV.

    FIXED 2026-07-21 (caught by a fable r0-sensitivity cross-check on the
    parallel bug in benchmarks/grid/run_grid.py): this function used to take
    r0 in Bohr and return the bare Bohr^-1 value, fed straight into
    erf_toml()'s omega= -- so every erf comparison arm this script ever ran
    was at the wrong absolute scale (effective error factor
    1/BOHR_INV_PER_ANG_INV = 1.8897259886). A second, independent unit fix
    landed the same day: [mp2] r0 itself changed from Bohr to Å at the CLI
    boundary, so this function's INPUT also changed meaning -- it now takes
    r0 in Å (matching R0_LIST/R0_DEFAULT below) and converts to Bohr before
    applying the Bohr-native r0<->omega formula. The already-completed
    benchmarks/omega_diag/terfc_out/out/*erf*.out AND *terf*.out files from
    before these fixes are STALE -- re-run to regenerate, don't trust them
    as-is."""
    r0_bohr = r0_ang * 1.8897259886
    omega_bohr = 1.0 / (r0_bohr * math.sqrt(2.0))
    return omega_bohr / BOHR_INV_PER_ANG_INV


def terf_toml(xyz, r0, form, basis, aux, fc):
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
n_quad = 12
[memory]
budget_gb = {MEMORY_BUDGET_GB}
"""


def erf_toml(xyz, omega, form, basis, aux, fc):
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
n_quad = 12
[memory]
budget_gb = {MEMORY_BUDGET_GB}
"""


def out_path(key):
    return f"{OUT}/out/{key}.out"


def preflight():
    """Verify the binary accepts attenuator='terf' + r0 on the ethylene monomer.
    ABORT if the stanza is rejected or silently ignored -- never record 'terf'
    numbers that are secretly erf."""
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    if not os.path.exists(BIN):
        log(f"PREFLIGHT FAIL: binary {BIN} missing -- build ferric-cli first.")
        return False
    if not _TERF_DIR:
        log("PREFLIGHT FAIL: terf-tables directory not found (looked in "
            "/home/matt/qc/ferric/terf-tables). Set ET_TABLE_DIR explicitly.")
        return False
    xyz, basis, aux, fc = SYSTEMS["monomer"]
    tp = f"{OUT}/toml/_preflight_terf_ethylene.toml"
    open(tp, "w").write(terf_toml(xyz, 1.6828, "delta-lr", basis, aux, fc))  # 3.18 Bohr in Å
    cmd = [FERRIC_LIMITED, f"--max={FERRIC_MAX_GB}G", f"--high={FERRIC_HIGH_GB}G",
           "--", BIN, tp]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=600)
    combined = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "Total energy" not in combined:
        log("PREFLIGHT FAIL: terf stanza rejected or job errored. Output tail:")
        for ln in combined.strip().splitlines()[-15:]:
            log(f"    | {ln}")
        return False
    low = combined.lower()
    if "attenuator" in low and "terf" not in low.split("attenuator", 1)[1][:40]:
        log("PREFLIGHT FAIL: binary ran but attenuator is not 'terf' -- likely "
            "silent erf-fallback. Refusing to sweep.")
        return False
    log("PREFLIGHT OK: terf stanza accepted, ethylene monomer/cc-pVDZ terf-B "
        "ran to 'Total energy'.")
    return True


def enumerate_jobs():
    jobs = []
    for sk in SYS_KEYS:
        xyz, basis, aux, fc = SYSTEMS[sk]
        for r0 in R0_LIST:
            w = omega_of(r0)
            for tag, form in FORMS:
                k = f"ethylene_{sk}_terf_r0{r0}_{tag}"
                if not (os.path.exists(out_path(k)) and "Total energy" in open(out_path(k)).read()):
                    jobs.append((k, terf_toml(xyz, r0, form, basis, aux, fc), "Total energy"))
                ke = f"ethylene_{sk}_erf_w{w:.4f}_{tag}"
                if not (os.path.exists(out_path(ke)) and "Total energy" in open(out_path(ke)).read()):
                    jobs.append((ke, erf_toml(xyz, w, form, basis, aux, fc), "Total energy"))
    seen, uniq = set(), []
    for j in jobs:
        if j[0] in seen:
            continue
        seen.add(j[0])
        uniq.append(j)
    return uniq


def wait_for_gate(key):
    while True:
        avail = mem_available_gb()
        if avail >= GATE_GB:
            return
        log(f"[gate] {key}: {avail:.1f}GB avail (<{GATE_GB}); waiting {WAIT_S}s")
        time.sleep(WAIT_S)


def run_one(job):
    key, toml, marker = job
    op = out_path(key)
    if os.path.exists(op) and marker in open(op).read():
        return "skip", 0.0
    wait_for_gate(key)
    toml_path = f"{OUT}/toml/{key}.toml"
    open(toml_path, "w").write(toml)

    cmd = [FERRIC_LIMITED, f"--max={FERRIC_MAX_GB}G", f"--high={FERRIC_HIGH_GB}G",
           "--", BIN, toml_path]
    t0 = time.monotonic()
    try:
        with open(op, "w") as f, open(op + ".err", "w") as e:
            subprocess.run(cmd, stdout=f, stderr=e, env=ENV, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.monotonic() - t0
    ok = os.path.exists(op) and marker in open(op).read()
    return ("ok" if ok else "FAIL"), time.monotonic() - t0


def main():
    os.makedirs(f"{OUT}/toml", exist_ok=True)
    os.makedirs(f"{OUT}/out", exist_ok=True)
    log(f"ethylene terfc sweep: systems={SYS_KEYS}, r0(A)={R0_LIST}, "
        f"forms={[f[0] for f in FORMS]}")
    if not preflight():
        log("ABORT: preflight failed.")
        return
    jobs = enumerate_jobs()
    total = len(jobs)
    if not jobs:
        log("nothing to do -- ethylene terfc sweep already complete.")
        return
    log(f"{total} jobs to run, serial, ferric-limited max={FERRIC_MAX_GB}G "
        f"high={FERRIC_HIGH_GB}G, budget_gb={MEMORY_BUDGET_GB}, gate={GATE_GB}GB")
    done = 0
    for j in jobs:
        status, dt = run_one(j)
        done += 1
        log(f"[{done}/{total}] {status:8s} {dt:7.1f}s  {j[0]}")
    log("done.")


if __name__ == "__main__":
    main()
