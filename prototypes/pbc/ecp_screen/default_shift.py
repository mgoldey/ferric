"""Default-precision (1e-14) change old -> new screen: exact sum over triples the NEW screen keeps and the OLD one
dropped (the antipodal triples the old reference missed), per shell pair; plus the a-priori bound on the old-only
triples the new screen drops."""

import math

import numpy as np

from screen import home_shells, shifted, triple_value
from tripl import enumerate_triples

d = enumerate_triples()
sh = home_shells()
p = 1e-14
kn, ko = d["new"] >= p, d["lnold"] >= math.log(p)
add = np.where(kn & ~ko)[0]
est = np.zeros((5, 5))
big = 0.0
for i in add:
    a, b = int(d["a"][i]), int(d["b"][i])
    v = triple_value(sh[a], shifted(sh[b], d["L"][d["il"][i]]), d["S"][d["iu"][i]])
    est[a, b] += v
    big = max(big, v)
drop = ko & ~kn
db = max(float(np.sum(d["new"][drop & (d["a"] == a) & (d["b"] == b)])) for a in range(5) for b in range(5))
print(f"new-only triples {len(add)}: max per-pair sum|exact| {est.max():.2e}, largest single {big:.2e}")
print(f"old-only triples {int(drop.sum())}: per-pair a-priori bound sum <= {db:.2e}")
