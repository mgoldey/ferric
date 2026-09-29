"""Sampled estimate of the NEW screen's truncation error: sum over triples dropped at p (bound in [p*1e-3, p)) of
the exact block max-abs, per shell pair (a, b), Horvitz-Thompson scaled from a uniform subsample. Triples below
p*1e-3 are covered by the a-priori bound sum (printed)."""

import math
import sys

import numpy as np

from screen import home_shells, shifted, triple_value
from tripl import enumerate_triples

d = enumerate_triples()
sh = home_shells()
rng = np.random.default_rng(11)
PS = [float(x) for x in sys.argv[1:]] or [1e-8, 1e-10, 1e-12]
bnd = d["new"]
for p in PS:
    band = np.where((bnd < p) & (bnd >= 1e-3 * p))[0]
    tail = (bnd < 1e-3 * p)
    pick = rng.choice(band, size=min(300, len(band)), replace=False)
    w = len(band) / len(pick)
    est = np.zeros((5, 5))
    ratio = []
    for i in pick:
        a, b = int(d["a"][i]), int(d["b"][i])
        v = triple_value(sh[a], shifted(sh[b], d["L"][d["il"][i]]), d["S"][d["iu"][i]])
        est[a, b] += w * v
        ratio.append(math.log(max(v, 1e-300) / bnd[i]))
    tail_b = max(float(np.sum(bnd[tail & (d["a"] == a) & (d["b"] == b)])) for a in range(5) for b in range(5))
    print(f"p={p:.0e} band n={len(band)} est max_ab sum|exact| = {est.max():.2e} ({est.max() / p:.2f} p); "
          f"tail a-priori <= {tail_b:.1e}; max ln(exact/bound) {max(ratio):.2f}", flush=True)
