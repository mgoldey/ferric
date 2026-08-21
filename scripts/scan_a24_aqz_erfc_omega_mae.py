#!/usr/bin/env python3
"""Adaptive omega scan of erf/erfc RS-MP2-RPA against the FULL A24 set (all
24 systems), at aug-cc-pVQZ -- an A24-WIDE MAE-vs-omega curve, not a
per-system bisection. Direct companion to scripts/scan_a24_aqz_terfc_r0_mae.py
(same 3-phase adaptive design, same A24-wide MAE aggregation), so B and T get
directly comparable MAE curves across both attenuator families (erf/erfc vs
the tempered terf/terfc).

[mp2] omega is Angstrom^-1 directly at the TOML boundary (crates/ferric-mp2/
src/attenuated.rs:6 -- "Bohr^-1 internally, Angstrom^-1 at the user-facing
boundary"; main.rs converts via BOHR_INV_PER_ANG_INV before it reaches
RsMp2RpaConfig). No manual unit conversion is needed here -- this field was
ALWAYS Angstrom^-1 at the TOML boundary, unlike [mp2] r0 (the terf r0
companion script's parameter), which used to be Bohr and was fixed to Å at
the same CLI boundary on 2026-07-21 (crates/ferric-cli/src/config.rs's r0
doc). Both fields are now Å/Å^-1-at-the-boundary, Bohr-internally -- this
is worth stating explicitly since a unit-conversion slip in the terf r0
companion script is exactly what triggered writing this docstring section
(scripts/scan_a24_aqz_terfc_r0_mae.py's history, and the earlier
benchmarks/grid/run_grid.py bug it was named after).

THREE-PHASE ADAPTIVE PROCEDURE (mirrors scripts/scan_a24_aqz_terfc_r0_mae.py):
  Phase 1 (COARSE): evaluate MAE(omega) vs CCSD(T)/CBS across all 24 A24
    systems at a coarse grid over [OMEGA_MIN, OMEGA_MAX] Angstrom^-1 (default
    0.05 spacing over [0.05, 1.0]), for both formulations.
  Phase 2 (REFINE): locate the coarse minimum, then evaluate every 0.025 (by
    default) in a window around it.
  Phase 3 (BISECT): golden-section-search the MAE(omega) curve between the
    two REFINE-grid points bracketing the refined minimum, to a default
    0.005 Angstrom^-1 tolerance.

Default ranges/steps differ from the r0 scan's 0-2 Angstrom / 0.1 / 0.05 /
0.01 because omega and r0 are reciprocal-ish quantities on different natural
scales (existing production omega values span ~0.1-0.8 Angstrom^-1, per
scripts/run_a24_omega_check.py's OMEGA_DEFAULT and
scripts/bisect_a24_aqz_crossing.py's seed brackets) -- 0.05-1.0 at 0.05
spacing covers that range at comparable relative resolution to the r0 scan's
0.1/2.0 = 5% coarse steps.

Reuses benchmarks/grid/geoms/a24-{01..24}_{dimer,mA_cp,mB_cp}.xyz and
benchmarks/grid/refs.json, same as the r0 scan -- does not regenerate
geometries a third way. Output keys land in
benchmarks/grid/out/a24-{idx:02d}_{tag}_aqz_erfw_{omega}_{form}.out, distinct
from both run_grid.py's own {method} keys and the terf scan's {terfr0} keys.

PREFLIGHT verifies a representative erf job actually completes before
spending scan compute (erf has no table-file failure mode like terf, but a
preflight keeps the same safety convention as every other script this
session, and catches basis/env misconfiguration early).

Usage:
  python3 scripts/scan_a24_aqz_erfc_omega_mae.py              # both B and T
  python3 scripts/scan_a24_aqz_erfc_omega_mae.py B             # B only
  python3 scripts/scan_a24_aqz_erfc_omega_mae.py --phase coarse B

Env tunables:
  SCAN_OMEGA_MIN/MAX       (0.05 / 1.0)  -- Angstrom^-1, phase-1 scan range
  SCAN_OMEGA_COARSE_STEP   (0.05)        -- Angstrom^-1, phase-1 spacing
  SCAN_OMEGA_REFINE_STEP   (0.025)       -- Angstrom^-1, phase-2 spacing
  SCAN_OMEGA_REFINE_HALFWIDTH (2)        -- phase-2 window, in COARSE steps
  SCAN_OMEGA_BISECT_TOL    (0.005)       -- Angstrom^-1, phase-3 convergence
  SCAN_FERRIC_MAX_GB/HIGH_GB (16 / 14)   -- ferric-limited caps per job
  SCAN_MEMORY_BUDGET_GB    (4.0)         -- [memory] budget_gb per job TOML
  SCAN_GATE_GB             (6)           -- pre-launch admission gate
  SCAN_WAIT_S              (20)          -- gate re-check interval
  SCAN_TIMEOUT             (1800)        -- per-job subprocess timeout (s)
  SCAN_RAYON_THREADS       (12)          -- solo 12-core-per-job
  SCAN_SYSTEMS             (all 24)      -- comma A24 indices to restrict to
"""
from pathlib import Path
import math
import os
import re
import subprocess
import sys
import time

ROOT = str(Path(__file__).resolve().parents[1])
os.chdir(ROOT)

GEOM_DIR = "benchmarks/grid/geoms"
OUT = "benchmarks/grid/out"
REFS_JSON = "benchmarks/grid/refs.json"
BIN = "target/release/ferric-cli"
FERRIC_LIMITED = "scripts/ferric-limited"
K = 627.509474

RAYON_NUM_THREADS = os.environ.get("SCAN_RAYON_THREADS", "12")
_LD_LIBRARY_PATH = os.pathsep.join(
    p for p in [os.path.expanduser("~/.local/lib"), os.environ.get("LD_LIBRARY_PATH", "")] if p)

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS=RAYON_NUM_THREADS,
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LD_LIBRARY_PATH=_LD_LIBRARY_PATH)

BASIS, AUX = "aug-cc-pvqz", "aug-cc-pvqz-rifit"

FERRIC_MAX_GB = os.environ.get("SCAN_FERRIC_MAX_GB", "16")
FERRIC_HIGH_GB = os.environ.get("SCAN_FERRIC_HIGH_GB", "14")
MEMORY_BUDGET_GB = os.environ.get("SCAN_MEMORY_BUDGET_GB", "4.0")
GATE_GB = float(os.environ.get("SCAN_GATE_GB", "6"))
WAIT_S = int(os.environ.get("SCAN_WAIT_S", "20"))
TIMEOUT = int(os.environ.get("SCAN_TIMEOUT", "1800"))

OMEGA_MIN = float(os.environ.get("SCAN_OMEGA_MIN", "0.05"))
OMEGA_MAX = float(os.environ.get("SCAN_OMEGA_MAX", "1.0"))
OMEGA_COARSE_STEP = float(os.environ.get("SCAN_OMEGA_COARSE_STEP", "0.05"))
OMEGA_REFINE_STEP = float(os.environ.get("SCAN_OMEGA_REFINE_STEP", "0.025"))
OMEGA_REFINE_HALFWIDTH = int(os.environ.get("SCAN_OMEGA_REFINE_HALFWIDTH", "2"))
OMEGA_BISECT_TOL = float(os.environ.get("SCAN_OMEGA_BISECT_TOL", "0.005"))

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


def erf_toml(xyz, omega, form, fc):
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
trunc_thresh = 1e-3
n_quad = 12
[memory]
budget_gb = {MEMORY_BUDGET_GB}
"""


TOT_RE = re.compile(r'Total energy\s*=\s*(-?[0-9.]+)')


def omega_key(omega):
    return f"{omega:.4f}".replace(".", "p")


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
    """Verify a representative erf job completes at aQZ (CH4-CH4, the
    smallest A24 system) before spending scan compute."""
    xyz = geom_path(19, "mA_cp")
    if not os.path.exists(xyz):
        log(f"PREFLIGHT FAIL: geometry {xyz} missing -- run "
            "benchmarks/grid/run_grid.py once first to stage A24 geometries.")
        return False
    fc = fc_count(xyz)
    tp = f"{OUT.replace('/out', '/toml')}/_preflight_erf_a24-19.toml"
    os.makedirs(os.path.dirname(tp), exist_ok=True)
    open(tp, "w").write(erf_toml(xyz, 0.42, "delta-lr", fc))
    cmd = [FERRIC_LIMITED, f"--max={FERRIC_MAX_GB}G", f"--high={FERRIC_HIGH_GB}G",
           "--", BIN, tp]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=600)
    combined = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "Total energy" not in combined:
        log("PREFLIGHT FAIL: erf job rejected or errored. Output tail:")
        for ln in combined.strip().splitlines()[-15:]:
            log(f"    | {ln}")
        return False
    log("PREFLIGHT OK: erf job accepted at aQZ, CH4-CH4 fragment ran to 'Total energy'.")
    return True


def evaluate_system(idx, omega, form, tag_suffix):
    frags = {"dimer": geom_path(idx, "dimer"),
             "mA_cp": geom_path(idx, "mA_cp"),
             "mB_cp": geom_path(idx, "mB_cp")}
    results = {}
    for fr, xyz in frags.items():
        if not os.path.exists(xyz):
            log(f"  missing geometry {xyz}")
            return None
        fc = fc_count(xyz)
        key = f"a24-{idx:02d}_{fr}_aqz_erfw_{omega_key(omega)}_{tag_suffix}"
        status, dt = run_one(key, erf_toml(xyz, omega, form, fc))
        log(f"    [{status:8s}] {dt:6.1f}s  {key}  (omega={omega:.4f} A^-1)")
        if status not in ("ok", "skip"):
            return None
        results[fr] = grab_total(key)
    if None in results.values():
        return None
    return (results["dimer"] - results["mA_cp"] - results["mB_cp"]) * K


def evaluate_omega_mae(omega, form, tag_suffix, idxs, refs):
    """KNOWN LIMITATION (fable review, 2026-07-21, deliberately left as-is,
    mirrors scan_a24_aqz_terfc_r0_mae.py's evaluate_r0_mae): all-or-nothing,
    no per-system blacklist or partial-coverage fallback -- one failing
    system nukes the entire omega point rather than averaging over whatever
    succeeded, to keep the MAE curve comparable point-to-point. See the
    terfc script's docstring for the full rationale; not fixed here under
    "no runs" since validating a fallback policy needs real failures to
    test against."""
    errs = {}
    for idx in idxs:
        e = evaluate_system(idx, omega, form, tag_suffix)
        if e is None:
            log(f"  A24-{idx}: FAILED at omega={omega:.4f} -- this omega point is incomplete.")
            return None
        errs[idx] = e - refs[idx]
    mae = sum(abs(v) for v in errs.values()) / len(errs)
    return mae, errs


def golden_section_min(f, a, b, tol, cache):
    """Minimize f (unimodal-assumed) on [a,b] via golden-section search until
    the bracket width <= tol. f(x) must return a plain float MAE (or None on
    failure) -- NOT a (mae, errs) tuple; the caller is responsible for
    unwrapping evaluate_omega_mae's tuple before passing a callable here
    (see run_formulation's `f`). cache: dict omega->float MAE so re-evaluated
    endpoints (shared between iterations) reuse prior job runs via
    run_one's own on-disk skip logic anyway, but this avoids redundant
    evaluate_omega_mae bookkeeping too.

    Returns (xm, mae_at_xm) where mae_at_xm is a plain float, or None if any
    probe point failed. FIXED 2026-07-21 (fable review, same bug as the
    terfc companion script scan_a24_aqz_terfc_r0_mae.py): this used to
    return cf(...) directly as the second element while the caller then
    indexed into it with r[0] (assuming a tuple) -- cf always returns
    whatever f() returns, which is a float here, so that indexing crashed
    with TypeError on every real phase-3 run. Also fixed: the cache seeded
    from `results` (mae, errs) tuples must be unwrapped to plain floats, or
    a cache hit returns a tuple while a fresh miss returns a float."""
    invphi = (math.sqrt(5) - 1) / 2
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
    log(f"=== {tag} ({form}) A24-wide aQZ erf omega-scan: "
        f"{len(idxs)} systems, omega in [{OMEGA_MIN}, {OMEGA_MAX}] A^-1 ===")

    results = {}

    def eval_and_record(omega):
        r = evaluate_omega_mae(round(omega, 4), form, tag, idxs, refs)
        if r is not None:
            results[round(omega, 4)] = r
            log(f"  omega={omega:.4f} A^-1 -> MAE={r[0]:.4f} kcal/mol "
                f"(vs {len(idxs)} A24 refs)")
        return r

    coarse_points = []
    if "coarse" in phases:
        log(f"--- Phase 1 (COARSE, step={OMEGA_COARSE_STEP} A^-1) ---")
        n = round((OMEGA_MAX - OMEGA_MIN) / OMEGA_COARSE_STEP)
        for i in range(n + 1):
            w = round(OMEGA_MIN + i * OMEGA_COARSE_STEP, 4)
            r = eval_and_record(w)
            if r is not None:
                coarse_points.append((w, r[0]))
        if not coarse_points:
            log(f"{tag}: coarse phase produced NO usable points -- aborting.")
            return
        best_coarse = min(coarse_points, key=lambda p: p[1])
        log(f"{tag}: coarse minimum at omega={best_coarse[0]:.4f} A^-1, "
            f"MAE={best_coarse[1]:.4f} kcal/mol")
        log(f"{tag}: coarse shape: " +
            ", ".join(f"{w:.3f}={mae:.3f}" for w, mae in coarse_points))

    refine_points = []
    if "refine" in phases:
        if not coarse_points:
            n = round((OMEGA_MAX - OMEGA_MIN) / OMEGA_COARSE_STEP)
            for i in range(n + 1):
                w = round(OMEGA_MIN + i * OMEGA_COARSE_STEP, 4)
                r = eval_and_record(w)
                if r is not None:
                    coarse_points.append((w, r[0]))
        if not coarse_points:
            log(f"{tag}: no coarse data available to center refine phase -- aborting.")
            return
        best_coarse = min(coarse_points, key=lambda p: p[1])
        center = best_coarse[0]
        lo = max(OMEGA_MIN, center - OMEGA_REFINE_HALFWIDTH * OMEGA_COARSE_STEP)
        hi = min(OMEGA_MAX, center + OMEGA_REFINE_HALFWIDTH * OMEGA_COARSE_STEP)
        log(f"--- Phase 2 (REFINE, step={OMEGA_REFINE_STEP} A^-1, window "
            f"[{lo:.4f}, {hi:.4f}] around coarse min {center:.4f}) ---")
        n = round((hi - lo) / OMEGA_REFINE_STEP)
        for i in range(n + 1):
            w = round(lo + i * OMEGA_REFINE_STEP, 4)
            r = eval_and_record(w)
            if r is not None:
                refine_points.append((w, r[0]))
        if not refine_points:
            log(f"{tag}: refine phase produced NO usable points -- skipping bisect phase.")
        else:
            best_refine = min(refine_points, key=lambda p: p[1])
            log(f"{tag}: refined minimum at omega={best_refine[0]:.4f} A^-1, "
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
            # that the true optimum may lie outside [OMEGA_MIN, OMEGA_MAX].
            # Check this FIRST, explicitly, before attempting to bracket.
            if best_i == 0 or best_i == len(pts_sorted) - 1:
                edge_w = pts_sorted[best_i][0]
                edge = "lower" if best_i == 0 else "upper"
                log(f"{tag}: WARNING -- best point (omega={edge_w:.4f} A^-1) is "
                    f"at the {edge} edge of the scanned range "
                    f"[{pts_sorted[0][0]:.4f}, {pts_sorted[-1][0]:.4f}] A^-1 -- "
                    "the true minimum may lie OUTSIDE this range. Skipping "
                    "bisect phase (bracketing inside the edge interval would "
                    "silently understate this). Widen SCAN_OMEGA_MIN/MAX (or "
                    "the refine window) and re-scan before trusting any "
                    "reported minimum for this formulation.")
            else:
                lo_i, hi_i = best_i - 1, best_i + 1
                a, b = pts_sorted[lo_i][0], pts_sorted[hi_i][0]
                log(f"--- Phase 3 (BISECT, golden-section, tol={OMEGA_BISECT_TOL} A^-1, "
                    f"bracket [{a:.4f}, {b:.4f}]) ---")
                # cache holds plain float MAEs (golden_section_min's cf()
                # returns whatever f() returns, a float) -- seed it from
                # `results`' (mae, errs) tuples by unwrapping to mae only.
                # FIXED 2026-07-21 (fable review): this used to store
                # (mae, None) tuples, an inconsistent cache vs a fresh f()
                # miss (a plain float) -- breaks fc_ < fd_ the moment a
                # golden-section probe lands on an already-evaluated point.
                cache = {w: mae for w, (mae, _errs) in results.items()}

                def f(w):
                    r = eval_and_record(w)
                    return r[0] if r is not None else None

                out = golden_section_min(f, a, b, OMEGA_BISECT_TOL, cache)
                if out is not None:
                    xm, mae_xm = out
                    log(f"{tag}: BISECTED minimum omega={xm:.4f} A^-1, "
                        f"MAE={mae_xm:.4f} kcal/mol (tol {OMEGA_BISECT_TOL} A^-1)")

    log(f"{tag}: {len(results)} total (omega, MAE) points evaluated this run.")


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

    log(f"A24-wide aQZ erf omega-scan: systems={idxs} ({len(idxs)}), "
        f"forms={tags}, phases={phases}")
    if not preflight():
        log("ABORT: preflight failed.")
        return
    for tag in tags:
        run_formulation(tag, idxs, refs, phases)
    log("done.")


if __name__ == "__main__":
    main()
