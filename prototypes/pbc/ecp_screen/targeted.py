"""Exact values for (i) triples the OLD screen drops at p = 1e-12 but the new bound calls >= 1e-12 (the plateau
suspects), (ii) a log-stratified random sample over all triples (rigour check exact <= new bound)."""

import math

import numpy as np

from screen import home_shells, shifted, triple_value
from tripl import enumerate_triples

d = enumerate_triples()
sh = home_shells()
n = len(d["a"])
print("triples", n, flush=True)
lnnew = np.log(d["new"])
sus = np.where((d["lnold"] < math.log(1e-12)) & (lnnew >= math.log(1e-12)))[0]
print("suspects (old drops @1e-12, new >= 1e-12):", len(sus), flush=True)
rng = np.random.default_rng(3)
strat = []
for lo in range(-55, -2, 4):
    idx = np.where((lnnew >= lo) & (lnnew < lo + 4))[0]
    if len(idx):
        strat += list(rng.choice(idx, size=min(25, len(idx)), replace=False))
if len(sus) > 400:
    sus = sus[np.argsort(-lnnew[sus])][:400]
rows = []
for tag, idxs in (("sus", sus), ("strat", strat)):
    for i in idxs:
        a, b = int(d["a"][i]), int(d["b"][i])
        lv, rc = d["L"][d["il"][i]], d["S"][d["iu"][i]]
        v = triple_value(sh[a], shifted(sh[b], lv), rc)
        ra = np.asarray(sh[a]["center"])
        rb = np.asarray(sh[b]["center"]) + lv
        rows.append((tag == "sus", a, b, math.log(max(v, 1e-300)), lnnew[i], d["lnold"][i],
                     np.linalg.norm(ra - rc), np.linalg.norm(rb - rc), np.linalg.norm(ra - rb)))
R = np.array(rows)
np.save("targeted.npy", R)
s = R[R[:, 0] == 1]
t = R[R[:, 0] == 0]
print("RIGOUR: max ln(exact/new) over", len(R), "=", float(np.max(R[:, 3] - R[:, 4])))
print("stratified: max ln(exact/old) =", float(np.max(t[:, 3] - t[:, 5])),
      " median ln(new/exact) =", float(np.median(t[:, 4] - t[:, 3])))
if len(s):
    print("suspects: largest exact values")
    print(" a b lnExact lnNew lnOld RA RB dAB")
    for r in s[np.argsort(-s[:, 3])][:15]:
        print(" ".join(f"{x:7.2f}" for x in r[1:]))
