#!/usr/bin/env python3
"""Idempotent GW100 sweep runner: runs the gw100_full driver per basis, parses
the per-molecule IP table + MAE row, stores into results.json. Resumable — skips
a (basis) whose results are already present and valid unless --force.

Starts with the validated 10-molecule subset (the driver's `cases()`); expand by
adding molecules to gw100_full.rs's cases() and re-running (idempotent).

Each run is the user's job to launch memory-scoped/gated; this script only
orchestrates parsing + storage. Run ONE basis at a time to avoid box contention.

Usage:
  run_sweep.py aug-cc-pvdz      # run + store one basis
  run_sweep.py aug-cc-pvtz
  run_sweep.py --show           # print the aggregated cross-basis table
  run_sweep.py <basis> --force  # recompute even if present
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIN = ROOT / "target" / "release" / "examples" / "gw100_full"
RESULTS = HERE / "results.json"

METHODS = ["Koop", "dSCF", "dRPA", "G0W0", "COHSEX", "evGW0", "evGW", "G0W0pbe"]
# per-molecule data row: "H2O  12.62  13.889  10.989  12.549  12.890  14.672  12.835  12.801"
ROW = re.compile(
    r"^(?P<mol>[A-Za-z0-9]+)\s+(?P<exp>[-+0-9.]+)\s+" + r"\s+".join(
        rf"(?P<{m}>[-+0-9.]+)" for m in METHODS
    )
)
MAE = re.compile(r"^MAE\s+(?P<vals>.+)$")

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")


def _basis_path(basis):
    """Per-basis result file. Concurrent sweeps in different bases must NOT share
    one file — load()/mutate/save() over a shared file is a read-modify-write race
    (last writer clobbers the other basis). One file per basis = no contention."""
    return HERE / f"results_{basis}.json"


def load_basis(basis):
    """Load one basis's results. Falls back to the legacy combined results.json
    so older committed results stay visible until recomputed."""
    p = _basis_path(basis)
    if p.exists():
        return json.loads(p.read_text())
    legacy = RESULTS
    if legacy.exists():
        return json.loads(legacy.read_text()).get(basis, {})
    return {}


def save_basis(basis, slot):
    p = _basis_path(basis)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(slot, indent=2, sort_keys=True))
    tmp.replace(p)


def load():
    """Combined view across all per-basis files + legacy results.json (read-only,
    for --show)."""
    res = {}
    if RESULTS.exists():
        res.update(json.loads(RESULTS.read_text()))
    for p in HERE.glob("results_*.json"):
        basis = p.stem[len("results_"):]
        res[basis] = json.loads(p.read_text())
    return res


def parse_output(txt):
    """Return {mol: {exp, Koop, ..., evGW}} + {'MAE': {method: val}}."""
    mols, mae = {}, {}
    for line in txt.splitlines():
        m = ROW.match(line.strip())
        if m:
            d = m.groupdict()
            mol = d.pop("mol")
            mols[mol] = {k: float(v) for k, v in d.items()}
        mm = MAE.match(line.strip())
        if mm:
            vals = mm.group("vals").split()
            for name, v in zip(METHODS, vals):
                try:
                    mae[name] = float(v)
                except ValueError:
                    pass
    return mols, mae


FAILED_RE = re.compile(r"^(\w+)\s+FAILED\s*$")

# Case order as compiled into gw100_full.rs (the driver runs cases() in this
# order). Used to identify which molecule was in flight when a stall-watchdog
# kills the driver: the first case not yet done or failed.
_CASES_SRC = ROOT / "crates" / "ferric-gw" / "examples" / "gw100_full.rs"


def _case_order():
    if not _CASES_SRC.exists():
        return []
    txt = _CASES_SRC.read_text()
    # rows look like: Case { name: "Na6", ... }
    return re.findall(r'Case\s*\{\s*name:\s*"(\w+)"', txt)


def _next_undone_case(mols, failed):
    done = set(mols) | set(failed)
    for name in _case_order():
        if name not in done:
            return name
    return None


def run_basis(basis, force=False):
    """Stream the driver, persisting each molecule row to results_<basis>.json as
    it lands. Per-basis file → concurrent sweeps in different bases don't clobber.
    Resumable: a restart loses at most the in-flight molecule; rerun continues via
    GW100_DONE. `--force` clears this basis and recomputes all."""
    slot = {} if force else load_basis(basis)
    mols = slot.setdefault("molecules", {})
    failed = set(slot.get("failed", []))
    if not BIN.exists():
        sys.exit(f"binary missing: {BIN} (build gw100_full first)")

    # Per-molecule wall-clock budget: one pathologically slow molecule (e.g. a
    # floppy alkali cluster on cross-family aux) burned 4+ CPU-hours and starved
    # the sweep. A stall-watchdog kills the driver if no new row lands within
    # MOL_BUDGET seconds. CRUCIALLY we then RE-LAUNCH the driver (skipping
    # done+failed via GW100_DONE) so the sweep CONTINUES past the bad molecule —
    # without this loop a single stall ended the whole run (the aTZ-stopped-at-28
    # bug). Loop until every case is accounted for (done or failed).
    mol_budget = float(os.environ.get("GW100_MOL_BUDGET", "1800"))  # 30 min default
    import threading
    all_names = set(_case_order())

    while True:
        remaining = all_names - set(mols) - failed
        if not remaining:
            break  # every case done or failed
        done = sorted(set(mols) | failed)
        print(f"[run] gw100_full {basis} ({len(done)} skipped, {len(remaining)} to go, "
              f"{mol_budget:.0f}s/mol budget) ...", flush=True)
        env = dict(ENV, GW100_DONE=",".join(done))
        proc = subprocess.Popen([str(BIN), basis], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

        last_progress = [__import__("time").monotonic()]
        stalled = [False]

        def watchdog(p=proc, lp=last_progress, st=stalled):
            import time
            while p.poll() is None:
                time.sleep(15)
                if time.monotonic() - lp[0] > mol_budget:
                    st[0] = True
                    p.kill()
                    return
        wd = threading.Thread(target=watchdog, daemon=True)
        wd.start()

        for line in proc.stdout:
            line = line.rstrip("\n")
            m = ROW.match(line.strip())
            if m:
                d = m.groupdict(); name = d.pop("mol")
                mols[name] = {k: float(v) for k, v in d.items()}
                slot["molecules"] = mols
                save_basis(basis, slot)         # persist EACH molecule immediately
                last_progress[0] = __import__("time").monotonic()
                print(f"  [+] {name} ({len(mols)} done)", flush=True)
                continue
            fm = FAILED_RE.match(line.strip())
            if fm:
                failed.add(fm.group(1)); slot["failed"] = sorted(failed)
                save_basis(basis, slot)
                last_progress[0] = __import__("time").monotonic()
                print(f"  [x] {fm.group(1)} FAILED", flush=True)
                continue
        proc.wait()

        if stalled[0]:
            # The molecule in flight when we killed the driver is the next un-done
            # case. Mark it FAILED so the re-launch skips it and CONTINUES.
            nxt = _next_undone_case(mols, failed)
            if nxt:
                failed.add(nxt); slot["failed"] = sorted(failed)
                save_basis(basis, slot)
                print(f"  [!] {nxt} exceeded {mol_budget:.0f}s/mol budget — FAILED, resuming past it", flush=True)
            else:
                break  # stalled but nothing left to attribute it to — stop
        elif proc.returncode not in (0, None):
            # Driver died for a non-stall reason (panic/OOM). Mark the in-flight
            # molecule failed and resume, but guard against an infinite loop.
            nxt = _next_undone_case(mols, failed)
            if nxt:
                failed.add(nxt); slot["failed"] = sorted(failed)
                save_basis(basis, slot)
                print(f"  [!] {nxt} — driver exited {proc.returncode}, FAILED, resuming past it", flush=True)
            else:
                break
        # clean exit with nothing stalled → loop re-checks `remaining` (should be empty)

    # Recompute MAE from the persisted molecule set (independent of run completion).
    _recompute_mae(slot)
    save_basis(basis, slot)
    print(f"[done] {basis}: {len(mols)} converged, {len(failed)} FAILED {sorted(failed)}")
    print(f"       evGW MAE = {slot.get('mae', {}).get('evGW', '?')} eV")


def _recompute_mae(slot):
    """MAE vs experiment from the stored molecules (resilient to interruption)."""
    mols = slot.get("molecules", {})
    mae = {}
    for meth in METHODS:
        errs = [abs(d[meth] - d["exp"]) for d in mols.values()
                if d.get(meth) is not None and d.get("exp") is not None
                and abs(d[meth]) < 1e6 and d[meth] == d[meth]]  # finite, not NaN
        if errs:
            mae[meth] = round(sum(errs) / len(errs), 4)
    slot["mae"] = mae
    slot["n_converged"] = len(mols)
    slot["n_attempted"] = len(mols) + len(slot.get("failed", []))


def show():
    res = load()
    if not res:
        print("no results yet")
        return
    print(f"# GW100 sweep — MAE vs experiment (eV) by basis")
    print(f"{'basis':14} " + " ".join(f"{m:>7}" for m in METHODS) + f"  {'#mol':>5}")
    print("-" * 80)
    for basis in sorted(res):
        slot = res[basis]
        # Compute MAE live from stored molecules so --show works MID-RUN (the
        # stored "mae"/"n_converged" are only written at run completion).
        _recompute_mae(slot)
        mae = slot.get("mae", {})
        n = slot.get("n_converged", len(slot.get("molecules", {})))
        nf = len(slot.get("failed", []))
        print(f"{basis:14} " + " ".join(f"{mae.get(m, float('nan')):>7.3f}" for m in METHODS)
              + f"  {n:>5}" + (f"  ({nf} fail)" if nf else ""))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if "--show" in flags or not args:
        show()
    else:
        run_basis(args[0], force="--force" in flags)
