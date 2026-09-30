"""Stratified sample of triples: exact magnitude vs the old screen's implied bound
B_old = D_u e^{3} e^{lp_a + lp_b} e^{-E} (keep iff B_old >= p). Violators: actual > B_old."""

import math

import numpy as np

from screen import POS, OldScreen, home_shells, mu, shifted, translations, triple_value

rng = np.random.default_rng(7)
s = OldScreen(1e-14)
sh = home_shells()
dsum = sum(abs(c) for c in s.e["coefficients"])
rows = []
L = translations(2 * s.r_ecp)
M = translations(s.r_ecp)
sites = [
    POS[1] + m
    for m in M
    if np.min(np.linalg.norm(POS - (POS[1] + m), axis=1)) <= s.r_ecp
]
for lv in L:
    for a in range(5):
        ra = np.asarray(sh[a]["center"])
        for b in range(5):
            rb = np.asarray(sh[b]["center"]) + lv
            for rc in sites:
                ta = mu(s.amin[a], s.zmin) * np.sum((ra - rc) ** 2)
                tb = mu(s.amin[b], s.zmin) * np.sum((rb - rc) ** 2)
                if max(ta, tb) > 60:
                    continue
                tab = mu(s.amin[a], s.amin[b]) * np.sum((ra - rb) ** 2)
                e = max(ta, tb, tab)
                lb = math.log(dsum) + 3 + s.log_pref[a] + s.log_pref[b] - e
                which = int(np.argmax([ta, tb, tab]))
                rows.append((a, b, lv, rc, lb, which))
print("candidates", len(rows), flush=True)
lbs = np.array([r[4] for r in rows])
wh = np.array([r[5] for r in rows])
pick = []
for w in range(3):
    for lo, hi in [(-80, -40), (-40, -30), (-30, -20), (-20, -10)]:
        idx = np.where((wh == w) & (lbs >= lo) & (lbs < hi))[0]
        if len(idx):
            pick += list(rng.choice(idx, size=min(60, len(idx)), replace=False))
print("sampled", len(pick), flush=True)
out = []
for i in pick:
    a, b, lv, rc, lb, w = rows[i]
    ra = np.asarray(sh[a]["center"])
    rb = np.asarray(sh[b]["center"]) + lv
    v = triple_value(sh[a], shifted(sh[b], lv), rc)
    vs = triple_value(sh[a], shifted(sh[b], lv), rc, "semi")
    vl = triple_value(sh[a], shifted(sh[b], lv), rc, "local")
    RA, RB = np.linalg.norm(ra - rc), np.linalg.norm(rb - rc)
    out.append(
        (
            a,
            b,
            w,
            lb,
            math.log(max(v, 1e-300)),
            math.log(max(vs, 1e-300)),
            math.log(max(vl, 1e-300)),
            RA,
            RB,
            np.linalg.norm(ra - rb),
        )
    )
out = np.array(out)
np.save("sample.npy", out)
viol = out[out[:, 4] > out[:, 3]]
print("violators", len(viol), "of", len(out))
print(" a b which lnBold lnV lnVsemi lnVloc RA RB dAB")
for r in viol[np.argsort(viol[:, 3] - viol[:, 4])][:40]:
    print(" ".join(f"{x:7.2f}" for x in r))
