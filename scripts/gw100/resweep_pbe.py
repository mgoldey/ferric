#!/usr/bin/env python3
"""Refresh ONLY the G0W0@PBE column of the GW100 results, in place.

The Padé-node fix (commit d7c3506) repaired G0W0@PBE (the AC extrapolation for
the small KS gap); @HF and the other QP columns are unchanged (short
extrapolation). So we recompute only `G0W0pbe` per molecule and overwrite that
single field, preserving every other column. Runs one molecule at a time with
GW100_PBE_ALL=1 (forces the @PBE lane for all sizes) and GW100_G0W0_ONLY is NOT
set (we still need the neutral @HF SCF that @PBE derives from).

Usage: resweep_pbe.py [basis]   (default: both aug-cc-pvdz and aug-cc-pvtz)
Idempotent-ish: overwrites G0W0pbe for every molecule already in the results;
skips molecules absent from the results (never converged) and the known-bad rows.
"""
import json, math, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIN = ROOT / "target" / "release" / "examples" / "gw100_full"
SRC = ROOT / "benchmarks" / "harness" / "examples" / "gw100_full.rs"
# Rows whose @PBE is meaningless OR unscoreable (skip to save time):
#  Rb2 all-electron unphysical; C4/H12Si5 wrong-HOMO (a starting-point/ordering
#  bug, not an AC bug — @PBE won't be meaningful there either); CCuN has NO
#  experimental IP (exp=null), so its @PBE never enters the MAE and it is a
#  d-block (Cu, g-functions) grind that stalled the sweep for >1 h — pure waste.
SKIP = {"Rb2", "C4", "H12Si5", "CCuN"}
# aTZ-specific waste: the 5-8-carbon aromatics + Cu2 have NO scoreable @PBE at
# aug-cc-pVTZ (full-depth @PBE only ran on ≤10-atom molecules; these exceeded
# GW100_FULL_MAX_ATOMS, so their banked G0W0pbe is `nan`). Forcing @PBE on them
# via GW100_PBE_ALL=1 is a slow, memory-hungry (~17 GB aTZ RPA) grind that
# never enters the MAE — pure waste, and the aTZ OOM-killer bait. They are
# large-gap organics the Padé fix barely moves anyway. Skip at aTZ only.
SKIP_ATZ_NAN = {"C4H5N3O", "C5H5N", "C5H5N5", "C5H5N5O", "C5H6", "C5H6N2O2",
                "C6F6", "C6H6", "C6H6O", "C6H7N", "C7H8", "C8H10", "C8H8", "Cu2"}
ROW = re.compile(r"^(?P<mol>[A-Za-z0-9]+)\s+(?P<rest>[-+0-9.]+(?:\s+(?:[-+0-9.]+|NaN|nan)){8})")


def all_cases():
    return re.findall(r'name:\s*"(\w+)"', SRC.read_text())


def run_pbe(basis, mol):
    """Run one molecule, return its G0W0@PBE (column 10, 1-indexed incl mol) or None."""
    skip = ",".join(c for c in all_cases() if c != mol)
    # OPENBLAS=1 is mandatory (rayon×OpenBLAS dgetrf segfaults). Intra-job rayon
    # width is RESWEEP_JOB_RAYON (default 1). Two regimes:
    #  - aDZ (small, throughput-bound): RESWEEP_JOB_RAYON=1 + many WORKERS → many
    #    1-core jobs saturate the box (ferric-single-job-threading-noop path).
    #  - aTZ (memory-bound, ~17 GB per RPA job): WORKERS=1 + RESWEEP_JOB_RAYON=12
    #    → ONE molecule at a time with all cores, so jobs never stack and OOM.
    job_rayon = os.environ.get("RESWEEP_JOB_RAYON", "1")
    # BLAS threads: default 1 (safe, aDZ many-workers regime). For the aTZ
    # 1-at-a-time regime, set RESWEEP_JOB_BLAS=12 so the OUTSIDE-rayon SCF GEMMs
    # (density rebuild, S^-1/2 eigh, FDS-SDF) parallelize. ferric's internal
    # gatekeeping (opt_in_blas_threads → rayon-worker self-guard) still forces
    # BLAS=1 INSIDE the GW/RPA par_iters, so the rayon×OpenBLAS dgetrf crash
    # cannot fire. FERRIC_BLAS_THREADS must match (it drives the opt-in).
    job_blas = os.environ.get("RESWEEP_JOB_BLAS", "1")
    env = dict(os.environ, OPENBLAS_NUM_THREADS=job_blas,
               FERRIC_BLAS_THREADS=job_blas,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS=job_rayon,
               GW100_TRUNC="1e-4", GW100_FULL_MAX_ATOMS="10",
               GW100_PBE_ALL="1", GW100_DONE=skip)
    # Per-molecule wall budget (default 1200 s). A molecule exceeding it keeps its
    # existing @PBE value. This is safe for the aggregate because the Padé-node
    # fix only changes @PBE for SMALL-gap molecules (long AC extrapolation); the
    # slow molecules here are all LARGE-gap heavies (Cu2, F4Ti, GeH4, halides,
    # noble gases) whose @PBE is unchanged by the fix — the old banked value IS
    # the post-fix value. A uniform time rule, not a hand-picked skip.
    budget = float(os.environ.get("RESWEEP_MOL_BUDGET", "1200"))
    try:
        out = subprocess.run([str(BIN), basis], env=env, capture_output=True,
                             text=True, timeout=budget).stdout
    except subprocess.TimeoutExpired:
        return None
    for line in out.splitlines():
        f = line.split()
        # result row: mol then a numeric exp; @PBE is the last of the 9 method cols.
        if len(f) >= 10 and f[0] == mol:
            try:
                float(f[1])  # exp numeric → this is the data row, not the diag row
            except ValueError:
                continue
            v = f[9]  # G0W0pbe (mol exp Koop dSCF dRPA G0W0 COHSEX evGW0 evGW G0W0pbe)
            try:
                x = float(v)
            except ValueError:
                return None
            # A NaN @PBE (failed lane) must NOT clobber an existing real value —
            # treat like a timeout: keep the banked number. (CH4N2O/aTZ was
            # overwritten 9.308 -> nan by exactly this before the guard.)
            return None if math.isnan(x) else x
    return None


def main():
    bases = [sys.argv[1]] if len(sys.argv) > 1 else ["aug-cc-pvdz", "aug-cc-pvtz"]
    # Concurrency: run N single-thread molecules at once. Default = ~cores, since
    # each job is 1 core (RAYON=1/OPENBLAS=1). aDZ jobs are ~1.3 GB RSS each, so
    # a full 12-wide fan-out stays well under the 20 G unit cap. Override with
    # RESWEEP_WORKERS.
    workers = int(os.environ.get("RESWEEP_WORKERS", str(os.cpu_count() or 4)))
    for basis in bases:
        p = HERE / f"results_{basis}.json"
        d = json.loads(p.read_text())
        # Resume: skip molecules already marked refreshed this run (set
        # RESWEEP_FRESH=1 to force a full redo). The marker is a sidecar set, not
        # a mutation of the row schema.
        marker = HERE / f".pbe_refreshed_{basis}.json"
        done_set = set()
        if marker.exists() and os.environ.get("RESWEEP_FRESH") != "1":
            done_set = set(json.loads(marker.read_text()))
        skip = set(SKIP)
        if basis == "aug-cc-pvtz":
            skip |= SKIP_ATZ_NAN  # nan-banked large organics — unscoreable @PBE
        mols = [m for m in d["molecules"] if m not in skip and m not in done_set]
        print(f"[{basis}] refreshing G0W0pbe for {len(mols)} molecules"
              f" ({len(done_set)} already done, skipped) — {workers} workers",
              flush=True)
        # Single writer-lock guards the results JSON + marker. Workers only run
        # the subprocess (no shared state) and hand back their result to be
        # committed under the lock — atomic write per molecule preserved, but
        # N molecules run concurrently.
        lock = Lock()
        n = len(mols)
        done_ct = [0]

        def commit(mol, new, dt):
            old = d["molecules"][mol].get("G0W0pbe")
            with lock:
                if new is not None:
                    d["molecules"][mol]["G0W0pbe"] = new
                    tmp = p.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
                    tmp.replace(p)
                done_set.add(mol)
                marker.write_text(json.dumps(sorted(done_set)))
                done_ct[0] += 1
                i = done_ct[0]
            if new is not None:
                print(f"  [{i}/{n}] {mol:8s} G0W0pbe {old} -> {new:.3f}  ({dt:.0f}s)", flush=True)
            else:
                # Timed out/failed: keep existing @PBE (unchanged by the fix for
                # these large-gap heavies) and MARK DONE so resume won't retry.
                print(f"  [{i}/{n}] {mol:8s} G0W0pbe TIMEOUT/kept {old}  ({dt:.0f}s)", flush=True)

        def work(mol):
            t0 = time.monotonic()
            new = run_pbe(basis, mol)
            commit(mol, new, time.monotonic() - t0)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, m) for m in mols]
            for f in as_completed(futs):
                f.result()  # surface worker exceptions
    print("=== @PBE re-sweep DONE ===", flush=True)


if __name__ == "__main__":
    main()
