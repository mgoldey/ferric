#!/usr/bin/env bash
# Kept-work-at-matched-error sweep for alkane_8 (19.9 Bohr), STO-3G.
#
# Split out of ipb_distance_proto.py's --mode sweep because the full seven
# thresholds x four bounds on C8 exceeds this session's five-minute per-script
# budget; four thresholds still bracket the 1e-5..1e-7 K-error band that the
# matched-error interpolation needs.
set -euo pipefail
cd "$(dirname "$0")/../.."
OMP_NUM_THREADS=1 python3 -u -c "
import sys; sys.path.insert(0,'scripts')
import ipb_distance_proto as M
a = M.Anchors()
tb = M.run('${1:-alkane_8}', '${2:-sto-3g}', (25, 50), a,
           [float(t) for t in '${3:-1e-3,1e-4,1e-5,1e-6}'.split(',')],
           do_sweep=True)
M.matched_error_report(tb, [1e-5, 1e-6])
"
