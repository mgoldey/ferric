"""Screen work at p = 1e-14: L images walked, (a, b, L) pairs passing the pair pre-test, and inner site checks,
old vs new (replicas of EcpPlan::kept)."""

import math

import numpy as np

from screen import POS, NewScreen, OldScreen, home_shells, mu, translations

p = 1e-14
sh = home_shells()
cent = [np.asarray(s["center"]) for s in sh]
o = OldScreen(p)
Lo = translations(o.r_pair)
Mo = translations(o.r_ecp)
so = [
    POS[1] + m
    for m in Mo
    if np.min(np.linalg.norm(POS - (POS[1] + m), axis=1)) <= o.r_ecp
]
pairs_o = sum(
    mu(o.amin[a], o.amin[b]) * np.sum((cent[a] - cent[b] - lv) ** 2) <= o.x_max
    for lv in Lo
    for a in range(5)
    for b in range(5)
)
n = NewScreen(p)
lnP = np.log(n.P)
lnw = math.log(sum(w * math.sqrt(math.pi / z) for w, z in n.terms))
zlo = min(z for _, z in n.terms)
xmax = 2 * lnP.max() + lnw - math.log(p)
r_ecp = math.sqrt(xmax / mu(min(n.ap), zlo))
Ln = translations(2 * r_ecp)
Mn = translations(r_ecp)
sn = [
    POS[1] + m
    for m in Mn
    if np.min(np.linalg.norm(POS - (POS[1] + m), axis=1)) <= r_ecp
]
pairs_n = 0
for lv in Ln:
    for a in range(5):
        for b in range(5):
            x = lnP[a] + lnP[b] + lnw - math.log(p)
            reach = math.sqrt(x / mu(n.ap[a], zlo)) + math.sqrt(x / mu(n.ap[b], zlo))
            pairs_n += np.linalg.norm(cent[a] - cent[b] - lv) <= reach
print(
    f"old: r_ecp {o.r_ecp:.2f} r_pair {o.r_pair:.2f} nL {len(Lo)} sites {len(so)} pairs {pairs_o} "
    f"site checks {pairs_o * len(so)}"
)
print(
    f"new: r_ecp {r_ecp:.2f} r_pair {2 * r_ecp:.2f} nL {len(Ln)} sites {len(sn)} pairs {pairs_n} "
    f"site checks {pairs_n * len(sn)}"
)
