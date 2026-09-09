#!/usr/bin/env bash
# d(kept-work)/d(log threshold) across the size axis -- the ONE number that
# decides whether the ceiling argument of ipb_distance_results.md 5 is a general
# mechanism or an alkane_4 observation.
#
# A g-times tighter bound is exactly a threshold raised by g, so its maximum
# possible kept-work benefit is log10(g) times the LOCAL SLOPE of kept work
# versus log threshold.  If that slope stays flat as systems grow, no bound
# improvement can pay at scale; if it steepens, the lane reopens.
#
# No K build and no SCF-quality density is needed for a slope: the screen's
# decisions depend on D only through F = D X, so a converged density is used but
# no exchange matrix is ever assembled.  That makes this cheap enough to reach
# C12/C16 where the kept-work sweep could not.
set -euo pipefail
cd "$(dirname "$0")/../.."
OMP_NUM_THREADS=1 python3 -u -c "
import sys, math; sys.path.insert(0,'scripts')
import numpy as np
import ipb_distance_proto as M

systems = '${1:-alkane_4,alkane_8,alkane_12}'.split(',')
basis = '${2:-sto-3g}'
grid = tuple(int(x) for x in '${3:-25,50}'.split(','))
thresholds = [float(t) for t in '${4:-1e-4,1e-5,1e-6,1e-7}'.split(',')]

print(f'basis={basis} grid={grid}  kept-work %% (weighted), NO K build')
hdr = f\"{'system':<12}{'diam':>7}\" + ''.join(f'{t:>11.0e}' for t in thresholds)
print(hdr + f\"{'slope/dec':>11}\")
for name in systems:
    mol, D, coords, weights, bfe, bipb, info = M.prepare_system(name, basis, grid)
    kept = []
    for t in thresholds:
        scr = M.ScreenFerric(t, bfe, bipb)
        k = w = tk = tw = 0
        for b0 in range(0, coords.shape[0], M.BATCH_POINTS):
            pts = coords[b0:b0+M.BATCH_POINTS]
            nb = pts.shape[0]
            X = (mol.eval_gto('GTOval', pts) * np.sqrt(weights[b0:b0+nb])[:,None]).T
            fmax = M.shell_max(D @ X, info)
            centre, radius = M.batch_sphere(pts)
            for s1 in range(mol.nbas):
                for s2 in range(s1+1):
                    u = info.ncart[s1]*info.ncart[s2]*nb
                    tk += 1; tw += u
                    if scr.keep(s1, s2, centre, radius, fmax):
                        k += 1; w += u
        kept.append(100.0*w/tw)
    # local slope over the LAST decade -- the production end of the range, and
    # the tail rather than a global fit (repo rule: fit the tail).
    slope = kept[-1] - kept[-2]
    print(f'{name:<12}{M.molecular_diameter(mol):7.2f}'
          + ''.join(f'{v:11.3f}' for v in kept) + f'{slope:11.3f}', flush=True)
"
