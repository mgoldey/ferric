"""Local stability of the PBE0 exxdiv=none ROKS ground state (-1.465458280448) under ferric's update rules: start
from the CONVERGED ground-state MOs rotated by exp(t K) and run (i) plain Roothaan (diis_size 1), (ii) ferric DIIS
(size 8), (iii) diis_size 12.  If the minimum is an attracting fixed point of the map, small t must converge.
Part of FINDINGS "ROKS PBE0 CI non-convergence (Python diagnosis) — 2026-09-25"."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import roks_replica as R  # noqa: E402
from run_roks_trap import E_REF, ND, NO, rotate  # noqa: E402

S, h, enn, vm, jk, I, gr = R.tri_setup()
g = R.ferric_roks(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, level_shift=0.25, max_iter=800, witness=False)
assert g["converged"] and abs(g["e"] - E_REF) < 1e-8, (g["exit"], g["e"])
Cs = g["C"]
a = R.analyse(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, Cs)
print(f"ground state E {a['e']:.12f} gmax {a['gmax']:.1e} labels {a['labels']} gapA {a['gap_a']:+.5f} "
      f"gapB {a['gap_b']:+.5f} Roothaan eps[:6] {np.round(a['eps'][:6], 5)}", flush=True)
for name, kw in (("roothaan(no DIIS)", dict(diis_size=1)), ("DIIS 8 (ferric)", {}), ("DIIS 12", dict(diis_size=12))):
    for t in (1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
        row = []
        for seed in range(3):
            r = R.ferric_roks(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, C0=rotate(Cs, S, t, seed), record=True, **kw)
            ok = r["exit"] == "Converged" and abs(r["e"] - E_REF) < 1e-8
            errs = [x[2] for x in r["hist"]]
            row.append(f"{'OK' if ok else '--'} it {r['it']:3d} err0 {errs[0]:.1e} err10 {errs[min(9, len(errs)-1)]:.1e} "
                       f"errmax {max(errs):.1e}")
        print(f"  {name:18s} t {t:.0e}: " + " | ".join(row), flush=True)
