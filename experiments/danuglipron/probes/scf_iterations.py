#!/usr/bin/env python
"""Report SCF ITERATIONS and exit reason, not just wall time.

Tier 4's cost anomaly survived elimination of size, basis, composition,
charge, and AO-cache batching (RESULTS.md M10). What remains is SCF
convergence behaviour, and wall time alone cannot distinguish:

  * few iterations, each expensive  -> the grid/basis is the cost
  * many iterations, each cheap     -> convergence is the cost

Those have opposite fixes, so the iteration count is the discriminator.

Requires the `iterations` / `exit_reason` getters on PyDftResult
(`cargo build --release -p ferric-python`).

RUN ON AN UNCONTENDED BOX, under `scripts/ferric-limited`, and PIN the memory
budget: ferric's auto-detect reads *live* MemAvailable, so an unrelated job
shrinks the budget and can flip the AO-cache path mid-comparison. A run on
2026-09-02 resolved 7.26 GB instead of the expected ~9.6 GB for exactly that
reason.

Usage:
    FERRIC_MEM_BUDGET_GB=9 scripts/ferric-limited -- uv run --no-sync python \
        experiments/danuglipron/probes/scf_iterations.py <xyz> <charge> [basis]
"""
from __future__ import annotations

import os
import sys
import time


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    import ferric

    path, charge = sys.argv[1], int(sys.argv[2])
    basis = sys.argv[3] if len(sys.argv) > 3 else "sto-3g"

    mol = ferric.Molecule.from_xyz(path, charge, 1)
    bs = ferric.BasisSet.bundled(basis)

    t0 = time.time()
    res = ferric.run_dft(mol, bs, functional="PBE")
    dt = time.time() - t0

    # `iterations`/`exit_reason` are recent; fail loudly rather than silently
    # reporting a bare timing that cannot answer the question being asked.
    for attr in ("iterations", "exit_reason"):
        if not hasattr(res, attr):
            print(f"ERROR: PyDftResult has no `{attr}` — rebuild with "
                  f"`cargo build --release -p ferric-python`", file=sys.stderr)
            return 1

    n = res.iterations
    print(f"\n{os.path.basename(path)}  natoms={mol.natoms()}  "
          f"nelec={mol.nelec()}  charge={charge}  basis={basis}")
    print(f"  wall       = {dt:8.1f} s")
    print(f"  iterations = {n:8d}   (final ladder rung)")
    print(f"  exit       = {res.exit_reason}")
    print(f"  converged  = {res.converged}")
    if n:
        print(f"  ~ {dt / n:.1f} s per iteration (upper bound: includes "
              f"one-time grid/AO setup and any earlier ladder rungs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
