#!/usr/bin/env python3
"""Adaptive r0 scan of terf/terfc tempered-attenuator RS-MP2-RPA against the
FULL A24 set (all 24 systems), at aug-cc-pVQZ -- an A24-WIDE MAE-vs-r0 curve,
not a per-system bisection (contrast scripts/bisect_a24_aqz_terfc_r0.py, which
bisects each system individually toward its own CCSD(T) reference; Matt:
"not per system, on a24" -- this script evaluates the aggregate error metric
across all 24 systems at each r0, matching how scripts/run_a24_omega_check.py
reports an omega-vs-MAE table for the erf/erfc arm).

WHY 0-2 Angstrom, not the CLI's own default anchor (r0=3.18 Bohr=1.68 Angstrom,
matched to omega=0.42 A^-1): crates/ferric-mp2/src/rimp2.rs:1178's terfc
monotonicity regression test establishes E(2.0 Angstrom)/E(Coulomb) > 0.95 --
by r0=2.0 Angstrom terfc has already recovered >95% of the full-Coulomb
(long-range) limit, so r0 values beyond ~2 Angstrom are past the physically
discriminating region for probing r0-sensitivity. An earlier version of this
work (benchmarks/grid/run_grid.py's terf042/terfcr0 methods, since removed)
picked r0 by matching the erf omega=0.42/0.2 anchors directly, which for the
T/coupled-rings arm landed at r0=3.54 Angstrom -- already deep in the
saturated regime, not a useful comparison point. This script instead treats
r0 as the independent variable and scans it directly across 0-2 Angstrom,
the region rimp2.rs's own regression test identifies as where terfc actually
transitions between its two limits.

THREE-PHASE ADAPTIVE PROCEDURE (Matt, 2026-07-21: "this needs more
granularity than just a few. get the general shape then do every 0.05 near
the minimum and bisect down to 0.01"):
  Phase 1 (COARSE): evaluate MAE(r0) vs CCSD(T)/CBS across all 24 A24 systems
    at a coarse grid over [0, 2] Angstrom (default 0.1 Angstrom spacing --
    R0_COARSE_STEP), for both formulations (B=delta-lr, T=coupled-rings).
    This establishes the general SHAPE of the MAE-vs-r0 curve (monotonic?
    single minimum? flat plateau?).
  Phase 2 (REFINE): find the coarse-grid r0 with lowest MAE, then evaluate
    every 0.05 Angstrom (R0_REFINE_STEP) in a window around it (default
    +/- 2 coarse-steps, i.e. re-covers the immediate neighborhood at 2x
    density) to locate the refined minimum more precisely.
  Phase 3 (BISECT): golden-section/parabolic refinement between the two
    REFINE-grid points that bracket the refined minimum, iterating until the
    bracket width is <= 0.01 Angstrom (R0_BISECT_TOL) -- NOT a
    root-bisection toward zero-error (unlike bisect_a24_aqz_terfc_r0.py,
    there is no single target value here, MAE has no natural "zero"), this
    minimizes the MAE(r0) curve via a bracketed 1D line-search (golden-section
    search, robust to a non-parabolic/asymmetric minimum).

Reuses benchmarks/grid/geoms/a24-{01..24}_{dimer,mA_cp,mB_cp}.xyz (all 24
systems, already staged by benchmarks/grid/run_grid.py's build_jobs()) and
benchmarks/grid/refs.json's "a24" CCSD(T)/CBS references -- does NOT
regenerate geometries a third way. Output keys land in
benchmarks/grid/out/a24-{idx:02d}_{tag}_aqz_terfr0_{r0}_{form}.out, separate
from run_grid.py's own {method} keys so this script's runs never collide
with or get swept up by a bare run_grid.py re-run.

r0 unit note (2026-07-21): [mp2] r0 is Å directly at the CLI/TOML boundary
(crates/ferric-cli/src/config.rs's r0 doc, fixed the same day from an
earlier Bohr convention -- this script writes r0_ang straight into the TOML
with NO Å->Bohr conversion; the CLI itself converts to Bohr internally
before it ever reaches Operator::terf/terfc, which stay hard-Bohr, per the
usual coordinate-system convention).

PREFLIGHT verifies the terf stanza is accepted (not silently falling back to
erf) before spending any scan compute, same safeguard as every other
terf/terfc script this session.

Usage:
  python3 scripts/scan_a24_aqz_terfc_r0_mae.py              # both B and T
  python3 scripts/scan_a24_aqz_terfc_r0_mae.py B             # B only
  python3 scripts/scan_a24_aqz_terfc_r0_mae.py --phase coarse B

Env tunables:
  SCAN_R0_MIN/MAX        (0.2 / 2.0)   -- Angstrom, phase-1 scan range. MIN
                                          floors at 0.2 (KNOWN-SAFE from an
                                          earlier smoke test): r0=0.0 exactly
                                          drives omega=1/(r0*sqrt(2)) -> inf
                                          and crashes ferric-cli ("internal
                                          error in eri2: status -3", see
                                          benchmarks/grid/out/a24-02_dimer_aqz_
                                          terfr0_0p0000_B.out.err). r0=0.1 is
                                          UNTESTED -- do not lower R0_MIN below
                                          0.2 without a real (non-scan) smoke
                                          probe first.
  SCAN_R0_COARSE_STEP    (0.1)         -- Angstrom, phase-1 spacing
  SCAN_R0_REFINE_STEP    (0.05)        -- Angstrom, phase-2 spacing
  SCAN_R0_REFINE_HALFWIDTH (2)         -- phase-2 window, in COARSE steps
  SCAN_R0_BISECT_TOL     (0.01)        -- Angstrom, phase-3 convergence
  SCAN_FERRIC_MAX_GB/HIGH_GB (16 / 14) -- ferric-limited caps per job
  SCAN_MEMORY_BUDGET_GB  (4.0)         -- [memory] budget_gb per job TOML
  SCAN_GATE_GB           (6)           -- pre-launch admission gate
  SCAN_WAIT_S            (20)          -- gate re-check interval
  SCAN_TIMEOUT           (1800)        -- per-job subprocess timeout (s)
  SCAN_RAYON_THREADS     (12)          -- solo 12-core-per-job (Matt,
                                          2026-07-21: "use 12 cores per job")
  SCAN_SYSTEMS           (all 24)      -- comma A24 indices to restrict to
  SCAN_TABLE_DIR         (auto)        -- FERRIC_TERF_TABLE_DIR override
"""
import math
import os
import re
import subprocess
import sys
import time

ROOT = "/home/matt/qc/ferric"
os.chdir(ROOT)

GEOM_DIR = "benchmarks/grid/geoms"
OUT = "benchmarks/grid/out"
REFS_JSON = "benchmarks/grid/refs.json"
BIN = "target/release/ferric-cli"
FERRIC_LIMITED = "scripts/ferric-limited"
K = 627.509474

RAYON_NUM_THREADS = os.environ.get("SCAN_RAYON_THREADS", "12")
_LD_LIBRARY_PATH = os.pathsep.join(
    p for p in ["/home/matt/.local/lib", os.environ.get("LD_LIBRARY_PATH", "")] if p)

_TERF_DIR = os.environ.get("SCAN_TABLE_DIR", "")
if not _TERF_DIR or not os.path.exists(os.path.join(_TERF_DIR, "16_4_2.bin")):
    for cand in ("/home/matt/qc/ferric/terf-tables", f"{ROOT}/terf-tables"):
        if os.path.exists(os.path.join(cand, "16_4_2.bin")):
            _TERF_DIR = cand
            break

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS=RAYON_NUM_THREADS,
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LD_LIBRARY_PATH=_LD_LIBRARY_PATH,
           FERRIC_TERF_TABLE_DIR=_TERF_DIR)

BASIS, AUX = "aug-cc-pvqz", "aug-cc-pvqz-rifit"

FERRIC_MAX_GB = os.environ.get("SCAN_FERRIC_MAX_GB", "16")
FERRIC_HIGH_GB = os.environ.get("SCAN_FERRIC_HIGH_GB", "14")
MEMORY_BUDGET_GB = os.environ.get("SCAN_MEMORY_BUDGET_GB", "4.0")
GATE_GB = float(os.environ.get("SCAN_GATE_GB", "6"))
WAIT_S = int(os.environ.get("SCAN_WAIT_S", "20"))
TIMEOUT = int(os.environ.get("SCAN_TIMEOUT", "1800"))

R0_MIN = float(os.environ.get("SCAN_R0_MIN", "0.2"))
R0_MAX = float(os.environ.get("SCAN_R0_MAX", "2.0"))
R0_COARSE_STEP = float(os.environ.get("SCAN_R0_COARSE_STEP", "0.1"))
R0_REFINE_STEP = float(os.environ.get("SCAN_R0_REFINE_STEP", "0.05"))
R0_REFINE_HALFWIDTH = int(os.environ.get("SCAN_R0_REFINE_HALFWIDTH", "2"))
R0_BISECT_TOL = float(os.environ.get("SCAN_R0_BISECT_TOL", "0.01"))

if R0_MIN <= 0.0:
    raise SystemExit(
        f"SCAN_R0_MIN={R0_MIN} is <= 0: r0=0 drives omega=1/(r0*sqrt(2)) to "
        f"infinity and crashes ferric-cli (confirmed: 'internal error in "
        f"eri2: status -3'). Use a value > 0 (0.2 A is the known-safe floor; "
        f"lower values are untested -- probe with a single real run before "
        f"trusting them, not a scan).")

FORMS = {"B": "delta-lr", "T": "coupled-rings"}


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


def load_refs():
    import json
    if not os.path.exists(REFS_JSON):
        log(f"FATAL: {REFS_JSON} missing -- run benchmarks/grid/run_grid.py "
            "once (even --dry-run triggers build_jobs()) to stage it.")
        sys.exit(1)
    refs = json.load(open(REFS_JSON))["a24"]
    return {int(k): v for k, v in refs.items()}


def fc_count(xyz):
    n = 0
    for ln in open(xyz).read().splitlines()[2:]:
        if not ln.strip():
            continue
        s = ln.split()[0]
        if s.startswith("@") or s.upper().startswith("H"):
            continue
        n += 1
    return n


def geom_path(idx, tag):
    return f"{GEOM_DIR}/a24-{idx:02d}_{tag}.xyz"


def terf_toml(xyz, r0_ang, form, fc):
    # [mp2] r0 is Å at the CLI boundary (2026-07-21: fixed from Bohr, see
    # crates/ferric-cli/src/config.rs's r0 doc) -- pass r0_ang straight
    # through, no conversion needed here anymore.
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
attenuator = "terf"
r0 = {r0_ang}
formulation = "{form}"
frozen_core = {fc}
[rpa]
trunc_thresh = 1e-3
n_quad = 12
[memory]
budget_gb = {MEMORY_BUDGET_GB}
"""


TOT_RE = re.compile(r'Total energy\s*=\s*(-?[0-9.]+)')


def r0_key(r0_ang):
    return f"{r0_ang:.4f}".replace(".", "p")


def out_path(key):
    return f"{OUT}/{key}.out"


def grab_total(key):
    p = out_path(key)
    if not os.path.exists(p):
        return None
    m = TOT_RE.search(open(p).read())
    return float(m.group(1)) if m else None


def wait_for_gate(key):
    while True:
        avail = mem_available_gb()
        if avail >= GATE_GB:
            return
        log(f"[gate] {key}: {avail:.1f}GB avail (<{GATE_GB}); waiting {WAIT_S}s")
        time.sleep(WAIT_S)


def run_one(key, toml):
    op = out_path(key)
    if os.path.exists(op) and "Total energy" in open(op).read():
        return "skip", 0.0
    wait_for_gate(key)
    toml_path = f"{OUT.replace('/out', '/toml')}/{key}.toml"
    os.makedirs(os.path.dirname(toml_path), exist_ok=True)
    open(toml_path, "w").write(toml)
    cmd = [FERRIC_LIMITED, f"--max={FERRIC_MAX_GB}G", f"--high={FERRIC_HIGH_GB}G",
           "--", BIN, toml_path]
    t0 = time.monotonic()
    try:
        with open(op, "w") as f, open(op + ".err", "w") as e:
            subprocess.run(cmd, stdout=f, stderr=e, env=ENV, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.monotonic() - t0
    ok = os.path.exists(op) and "Total energy" in open(op).read()
    return ("ok" if ok else "FAIL"), time.monotonic() - t0


def preflight():
    """Verify the binary accepts attenuator='terf' + r0 (Å) -- not a
    silent erf fallback -- before spending any scan compute. Uses the
    smallest A24 system (idx 19, CH4-CH4)."""
    xyz = geom_path(19, "mA_cp")
    if not os.path.exists(xyz):
        log(f"PREFLIGHT FAIL: geometry {xyz} missing -- run "
            "benchmarks/grid/run_grid.py once first to stage A24 geometries.")
        return False
    if not _TERF_DIR:
        log("PREFLIGHT FAIL: terf-tables directory not found. Set SCAN_TABLE_DIR.")
        return False
    fc = fc_count(xyz)
    tp = f"{OUT.replace('/out', '/toml')}/_preflight_terf_a24-19.toml"
    os.makedirs(os.path.dirname(tp), exist_ok=True)
    open(tp, "w").write(terf_toml(xyz, 1.0, "delta-lr", fc))  # 1.0 A, mid-scan-range probe
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
            "silent erf-fallback. Refusing to scan.")
        return False
    log("PREFLIGHT OK: terf stanza accepted at aQZ, CH4-CH4 fragment ran to 'Total energy'.")
    return True


def evaluate_system(idx, r0_ang, form, tag_suffix):
    frags = {"dimer": geom_path(idx, "dimer"),
             "mA_cp": geom_path(idx, "mA_cp"),
             "mB_cp": geom_path(idx, "mB_cp")}
    results = {}
    for fr, xyz in frags.items():
        if not os.path.exists(xyz):
            log(f"  missing geometry {xyz}")
            return None
        fc = fc_count(xyz)
        key = f"a24-{idx:02d}_{fr}_aqz_terfr0_{r0_key(r0_ang)}_{tag_suffix}"
        status, dt = run_one(key, terf_toml(xyz, r0_ang, form, fc))
        log(f"    [{status:8s}] {dt:6.1f}s  {key}  (r0={r0_ang:.4f} A)")
        if status not in ("ok", "skip"):
            return None
        results[fr] = grab_total(key)
    if None in results.values():
        return None
    return (results["dimer"] - results["mA_cp"] - results["mB_cp"]) * K


def evaluate_r0_mae(r0_ang, form, tag_suffix, idxs, refs):
    """Evaluate every A24 system in idxs at this r0, return (mae, per_system)
    or None if ANY system failed (a partial-data MAE would silently mix
    scales/coverage across scan points -- refuse rather than mislead).

    KNOWN LIMITATION (fable review, 2026-07-21, deliberately left as-is):
    this is all-or-nothing by design, with no per-system blacklist or
    partial-coverage fallback. One slow/timing-out/crashing system at a
    given r0 nukes that ENTIRE scan point (golden_section_min then aborts
    the whole bisection phase -- see its "a probe point FAILED" branch).
    A partial-coverage fallback (e.g. drop the failing system and average
    over whatever succeeded) was considered and rejected here: silently
    changing which systems contribute to MAE from one r0 point to the next
    would make the curve incomparable across points, and deciding on a
    safe fallback policy (blacklist threshold? per-system retry? widen the
    timeout?) needs to be validated against real failures, which "no runs"
    precludes doing blind. If a specific system reliably fails at aQZ
    (check benchmarks/grid/out/*.err for the failing key), fix or exclude
    that system explicitly via SCAN_SYSTEMS rather than adding an implicit
    fallback here."""
    errs = {}
    for idx in idxs:
        e = evaluate_system(idx, r0_ang, form, tag_suffix)
        if e is None:
            log(f"  A24-{idx}: FAILED at r0={r0_ang:.4f} -- this r0 point is incomplete.")
            return None
        errs[idx] = e - refs[idx]
    mae = sum(abs(v) for v in errs.values()) / len(errs)
    return mae, errs


def golden_section_min(f, a, b, tol, cache):
    """Minimize f (unimodal-assumed) on [a,b] via golden-section search until
    the bracket width <= tol. f(x) must return a plain float MAE (or None on
    failure) -- NOT a (mae, errs) tuple; the caller is responsible for
    unwrapping evaluate_r0_mae's tuple before passing a callable here (see
    run_formulation's `f`, which does exactly that). cache: dict r0->float
    MAE so re-evaluated endpoints (shared between iterations) reuse prior
    job runs via run_one's own on-disk skip logic anyway, but this avoids
    redundant evaluate_r0_mae bookkeeping too.

    Returns (xm, mae_at_xm) where mae_at_xm is a plain float, or None if any
    probe point failed. FIXED 2026-07-21 (fable review): this used to return
    cf(...) directly as the second element while the caller then indexed
    into it with r[0] (assuming a tuple) -- cf always returns whatever f()
    returns, which is a float here, so that indexing crashed with
    TypeError on every real phase-3 run. Also fixed: the cache seeded from
    `results` (mae, errs) tuples must be unwrapped to plain floats, or a
    cache hit on a pre-seeded key returns a tuple while a fresh miss
    returns a float -- an inconsistent cache that breaks the fc_ < fd_
    comparison the moment any golden-section probe happens to land on an
    already-evaluated coarse/refine r0."""
    invphi = (math.sqrt(5) - 1) / 2  # 1/phi
    def cf(x):
        xr = round(x, 4)
        if xr not in cache:
            cache[xr] = f(xr)
        return cache[xr]
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc_, fd_ = cf(c), cf(d)
    while abs(b - a) > tol:
        if fc_ is None or fd_ is None:
            log("  golden-section: a probe point FAILED -- aborting bisection phase.")
            return None
        if fc_ < fd_:
            b, d, fd_ = d, c, fc_
            c = b - invphi * (b - a)
            fc_ = cf(c)
        else:
            a, c, fc_ = c, d, fd_
            d = a + invphi * (b - a)
            fd_ = cf(d)
    xm = (a + b) / 2
    mae_xm = cf(round(xm, 4))
    if mae_xm is None:
        log("  golden-section: final midpoint probe FAILED.")
        return None
    return xm, mae_xm


def run_formulation(tag, idxs, refs, phases):
    form = FORMS[tag]
    log(f"=== {tag} ({form}) A24-wide aQZ terf r0-scan: "
        f"{len(idxs)} systems, r0 in [{R0_MIN}, {R0_MAX}] Angstrom ===")

    results = {}  # r0_ang -> (mae, per_system_errs)

    def eval_and_record(r0_ang):
        r = evaluate_r0_mae(round(r0_ang, 4), form, tag, idxs, refs)
        if r is not None:
            results[round(r0_ang, 4)] = r
            log(f"  r0={r0_ang:.4f} A -> MAE={r[0]:.4f} kcal/mol "
                f"(vs {len(idxs)} A24 refs)")
        return r

    coarse_points = []
    if "coarse" in phases:
        log(f"--- Phase 1 (COARSE, step={R0_COARSE_STEP} A) ---")
        n = round((R0_MAX - R0_MIN) / R0_COARSE_STEP)
        for i in range(n + 1):
            r0 = round(R0_MIN + i * R0_COARSE_STEP, 4)
            r = eval_and_record(r0)
            if r is not None:
                coarse_points.append((r0, r[0]))
        if not coarse_points:
            log(f"{tag}: coarse phase produced NO usable points -- aborting.")
            return
        best_coarse = min(coarse_points, key=lambda p: p[1])
        log(f"{tag}: coarse minimum at r0={best_coarse[0]:.4f} A, "
            f"MAE={best_coarse[1]:.4f} kcal/mol")
        log(f"{tag}: coarse shape: " +
            ", ".join(f"{r0:.2f}A={mae:.3f}" for r0, mae in coarse_points))

    refine_points = []
    if "refine" in phases:
        # locate the coarse minimum to center the refine window on (works
        # even if phase 1 was skipped this run, by reusing cached .out files
        # via a fresh coarse pass at zero marginal job cost -- run_one skips
        # anything already on disk).
        if not coarse_points:
            n = round((R0_MAX - R0_MIN) / R0_COARSE_STEP)
            for i in range(n + 1):
                r0 = round(R0_MIN + i * R0_COARSE_STEP, 4)
                r = eval_and_record(r0)
                if r is not None:
                    coarse_points.append((r0, r[0]))
        if not coarse_points:
            log(f"{tag}: no coarse data available to center refine phase -- aborting.")
            return
        best_coarse = min(coarse_points, key=lambda p: p[1])
        center = best_coarse[0]
        lo = max(R0_MIN, center - R0_REFINE_HALFWIDTH * R0_COARSE_STEP)
        hi = min(R0_MAX, center + R0_REFINE_HALFWIDTH * R0_COARSE_STEP)
        log(f"--- Phase 2 (REFINE, step={R0_REFINE_STEP} A, window "
            f"[{lo:.4f}, {hi:.4f}] around coarse min {center:.4f}) ---")
        n = round((hi - lo) / R0_REFINE_STEP)
        for i in range(n + 1):
            r0 = round(lo + i * R0_REFINE_STEP, 4)
            r = eval_and_record(r0)
            if r is not None:
                refine_points.append((r0, r[0]))
        if not refine_points:
            log(f"{tag}: refine phase produced NO usable points -- skipping bisect phase.")
        else:
            best_refine = min(refine_points, key=lambda p: p[1])
            log(f"{tag}: refined minimum at r0={best_refine[0]:.4f} A, "
                f"MAE={best_refine[1]:.4f} kcal/mol")

    if "bisect" in phases:
        pts = refine_points or coarse_points
        if len(pts) < 3:
            log(f"{tag}: not enough points ({len(pts)}) to bracket a bisection "
                "-- skipping bisect phase.")
        else:
            pts_sorted = sorted(pts, key=lambda p: p[0])
            best_i = min(range(len(pts_sorted)), key=lambda i: pts_sorted[i][1])
            # BOUNDARY-MINIMUM CHECK (fable review, 2026-07-21): the previous
            # `a == b` check here could never fire (lo_i/hi_i are always
            # clamped to distinct valid indices whenever len(pts_sorted)>=3),
            # so a minimum sitting at the very first or last scanned point
            # was silently golden-sectioned inside that edge interval and
            # reported as a converged interior minimum -- with NO warning
            # that the true optimum may lie outside [R0_MIN, R0_MAX]
            # entirely (plausible per the erf/erfc data: C2H4/CH4 were still
            # moving at r0 well beyond 2 A). Check this FIRST, explicitly,
            # before attempting to bracket.
            if best_i == 0 or best_i == len(pts_sorted) - 1:
                edge_r0 = pts_sorted[best_i][0]
                edge = "lower" if best_i == 0 else "upper"
                log(f"{tag}: WARNING -- best point (r0={edge_r0:.4f} A) is at "
                    f"the {edge} edge of the scanned range "
                    f"[{pts_sorted[0][0]:.4f}, {pts_sorted[-1][0]:.4f}] A -- "
                    "the true minimum may lie OUTSIDE this range. Skipping "
                    "bisect phase (bracketing inside the edge interval would "
                    "silently understate this). Widen SCAN_R0_MIN/MAX (or "
                    "the refine window) and re-scan before trusting any "
                    "reported minimum for this formulation.")
            else:
                lo_i, hi_i = best_i - 1, best_i + 1
                a, b = pts_sorted[lo_i][0], pts_sorted[hi_i][0]
                log(f"--- Phase 3 (BISECT, golden-section, tol={R0_BISECT_TOL} A, "
                    f"bracket [{a:.4f}, {b:.4f}]) ---")
                # cache holds plain float MAEs (golden_section_min's cf()
                # returns whatever f() returns, a float) -- seed it from
                # `results`' (mae, errs) tuples by unwrapping to mae only.
                # FIXED 2026-07-21 (fable review): this used to store
                # (mae, None) tuples, so a cache HIT returned a tuple while
                # a cache MISS (via f()) returned a float -- an inconsistent
                # cache that breaks fc_ < fd_ the moment a golden-section
                # probe lands on an already-evaluated coarse/refine r0.
                cache = {r0: mae for r0, (mae, _errs) in results.items()}

                def f(r0):
                    r = eval_and_record(r0)
                    return r[0] if r is not None else None

                out = golden_section_min(f, a, b, R0_BISECT_TOL, cache)
                if out is not None:
                    xm, mae_xm = out
                    log(f"{tag}: BISECTED minimum r0={xm:.4f} A, "
                        f"MAE={mae_xm:.4f} kcal/mol (tol {R0_BISECT_TOL} A)")

    log(f"{tag}: {len(results)} total (r0, MAE) points evaluated this run.")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT.replace("/out", "/toml"), exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    phase_arg = None
    if "--phase" in sys.argv:
        phase_arg = sys.argv[sys.argv.index("--phase") + 1]
    phases = ["coarse", "refine", "bisect"] if not phase_arg else [phase_arg]
    tags = [a for a in args if a in ("B", "T")] or ["B", "T"]

    refs = load_refs()
    idx_env = [int(x) for x in os.environ.get("SCAN_SYSTEMS", "").split(",") if x.strip()]
    idxs = idx_env or sorted(refs)

    log(f"A24-wide aQZ terf r0-scan: systems={idxs} ({len(idxs)}), "
        f"forms={tags}, phases={phases}")
    if not preflight():
        log("ABORT: preflight failed.")
        return
    for tag in tags:
        run_formulation(tag, idxs, refs, phases)
    log("done.")


if __name__ == "__main__":
    main()
