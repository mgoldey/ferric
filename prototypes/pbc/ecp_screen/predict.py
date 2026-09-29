"""Kept-triple counts and the a-priori Gamma truncation error sum_{dropped} bound, old vs new screen."""

import math

import numpy as np

from screen import POS, NewScreen, OldScreen, home_shells, mu, translations

sh = home_shells()
cent = [np.asarray(s["center"]) for s in sh]
new0 = NewScreen(1e-16)
# enumeration radius for the new bound at 1e-16: K >= mu(a', z') R^2 with the largest prefactor
pmax = max(new0.P) ** 2 * sum(w * math.sqrt(math.pi / z) for w, z in new0.terms)
amin_p, zmin_p = min(new0.ap), min(z for _, z in new0.terms)
r_ecp = math.sqrt(math.log(pmax / 1e-16) / mu(amin_p, zmin_p))
print(f"new enumeration: r_ecp {r_ecp:.2f} r_pair {2 * r_ecp:.2f}")
M = translations(r_ecp)
sites = [POS[1] + m for m in M if np.min(np.linalg.norm(POS - (POS[1] + m), axis=1)) <= r_ecp]
L = translations(2 * r_ecp)
print("nL", len(L), "nsites", len(sites), flush=True)
S = np.array(sites)
recs = []  # (a, b, lnB_new, lnB_old)
olds = OldScreen(1.0)
dsum = sum(abs(c) for c in olds.e["coefficients"])
for lv in L:
    for a in range(5):
        RA = np.linalg.norm(S - cent[a], axis=1)
        for b in range(5):
            rb = cent[b] + lv
            RB = np.linalg.norm(S - rb, axis=1)
            ok = np.abs(RA - RB) <= np.linalg.norm(cent[a] - rb)  # always true; keeps shapes
            al, be = new0.ap[a], new0.ap[b]
            tot = np.zeros(len(S))
            for w, z in new0.terms:
                p = al + be + z
                K = (al * be * (RA - RB) ** 2 + al * z * RA**2 + be * z * RB**2) / p
                tot += w * math.sqrt(math.pi / p) * np.exp(-K)
            bn = new0.P[a] * new0.P[b] * tot
            dab2 = np.sum((cent[a] - rb) ** 2)
            e = np.maximum.reduce([
                np.full(len(S), mu(olds.amin[a], olds.amin[b]) * dab2),
                mu(olds.amin[a], olds.zmin) * RA**2,
                mu(olds.amin[b], olds.zmin) * RB**2,
            ])
            lbo = math.log(dsum) + 3 + max(olds.log_pref[a] + olds.log_pref[b], 0.0) - e
            sel = ok & ((bn > 1e-22) | (lbo > math.log(1e-22)))
            for x, y in zip(bn[sel], lbo[sel]):
                recs.append((a, b, x, y))
R = np.array(recs)
np.save("bounds.npy", R)
print("records", len(R))
print(" p      kept_old  kept_new  sum_dropped_newbound(max over shell pair)")
for p in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
    ko = int(np.sum(R[:, 3] >= math.log(p)))
    kn = int(np.sum(R[:, 2] >= p))
    err = 0.0
    for a in range(5):
        for b in range(5):
            m = (R[:, 0] == a) & (R[:, 1] == b) & (R[:, 2] < p)
            err = max(err, float(np.sum(R[m, 2])))
    print(f"{p:.0e} {ko:9d} {kn:9d} {err:.2e}")
