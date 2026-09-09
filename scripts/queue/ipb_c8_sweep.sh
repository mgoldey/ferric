#!/usr/bin/env bash
# Kept-work-at-matched-error sweep for one system, WITHOUT re-running the
# anchors.
#
# Split out of ipb_distance_proto.py's --mode sweep for two reasons:
#   * the full seven-threshold x four-bound sweep exceeds this session's
#     five-minute per-script budget at C8;
#   * `run()` always runs A0 first, and A0 at C8 (903 pairs x 197 probes x 3
#     bounds) costs more than the sweep it precedes while adding nothing over
#     the C4/def2-QZVP anchor runs, which already cover l up to 4.
#
# Anchors are NOT skipped for the study -- they are in
# scripts/queue/out/ipb_distance_results.md 2 and 2.0, run on water/cc-pVDZ,
# alkane_4/def2-SVP and alkane_4/def2-QZVP.  This runner exists only so the
# kept-work axis can reach a larger system inside the time budget.
set -euo pipefail
cd "$(dirname "$0")/../.."
OMP_NUM_THREADS=1 python3 -u -c "
import sys, math; sys.path.insert(0,'scripts')
import numpy as np
import ipb_distance_proto as M

system, basis = '${1:-alkane_8}', '${2:-sto-3g}'
thresholds = [float(t) for t in '${3:-1e-4,1e-5,1e-6}'.split(',')]
grid = tuple(int(x) for x in '${4:-25,50}'.split(','))

mol, D, coords, weights, bfe, bipb, info = M.prepare_system(system, basis, grid)
print(f'SYSTEM {system}/{basis} grid {grid} nbf={mol.nao} nsh={mol.nbas} '
      f'batches={math.ceil(len(weights)/M.BATCH_POINTS)} '
      f'diameter={M.molecular_diameter(mol):.2f} Bohr', flush=True)

ref = M.build_k(mol, coords, weights, D, M.ScreenNone(), info)
K_an = M.analytic_k(mol, D)
e_grid = float(np.max(np.abs(ref.K - K_an)))
print(f'  E_grid = {e_grid:.3e}  max|K| = {np.max(np.abs(K_an)):.3e}', flush=True)

table = {}
print(f\"  {'bound':<22}{'thresh':>9}{'kept %':>10}{'kept-wt %':>11}{'K error':>12}\", flush=True)
for nm, cls in M.SCREENS.items():
    table[nm] = []
    for t in thresholds:
        r = M.build_k(mol, coords, weights, D, cls(t, bfe, bipb), info)
        err = float(np.max(np.abs(r.K - ref.K)))
        kp = 100.0*r.kept/r.total
        kw = 100.0*r.kept_weighted/r.total_weighted
        table[nm].append((t, kp, kw, err))
        print(f'  {nm:<22}{t:9.0e}{kp:10.3f}{kw:11.3f}{err:12.3e}', flush=True)

M.matched_error_report(table, [1e-4, 1e-5, 1e-6])
"
