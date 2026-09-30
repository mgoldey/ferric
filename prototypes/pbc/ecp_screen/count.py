"""Count candidate triples of the old screen at several precisions and time one exact triple."""

import time

import numpy as np

from screen import POS, OldScreen, home_shells, shifted, translations, triple_value

for p in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
    s = OldScreen(p)
    L = translations(s.r_pair)
    M = translations(s.r_ecp)
    sites = [
        POS[1] + m
        for m in M
        if np.min(np.linalg.norm(POS - (POS[1] + m), axis=1)) <= s.r_ecp
    ]
    sh = home_shells()
    n = 0
    for lv in L:
        for a in range(5):
            for b in range(5):
                rb = np.asarray(sh[b]["center"]) + lv
                for rc in sites:
                    n += s.keep(a, b, np.asarray(sh[a]["center"]), rb, rc)
    print(
        f"p={p:.0e} x={s.x:.2f} x_max={s.x_max:.2f} r_ecp={s.r_ecp:.2f} r_pair={s.r_pair:.2f} "
        f"nL={len(L)} nsites={len(sites)} kept={n}",
        flush=True,
    )

sh = home_shells()
t = time.time()
for k in range(10):
    triple_value(
        sh[4], shifted(sh[4], np.array([6.0, 0, 0])), POS[1] + np.array([3.0, 0, 0])
    )
print("per triple (p-p) s:", (time.time() - t) / 10)
