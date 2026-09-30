"""Enumerate every candidate triple (shell a, shell b at +L, ECP image) once with both bounds; cached to triples.npz."""

import math
from pathlib import Path

import numpy as np

from screen import POS, NewScreen, OldScreen, home_shells, mu, translations


def enumerate_triples():
    f = Path(__file__).with_name("triples.npz")
    if f.exists():
        d = np.load(f)
        return {k: d[k] for k in d.files}
    sh = home_shells()
    cent = [np.asarray(s["center"]) for s in sh]
    new0 = NewScreen(1e-16)
    olds = OldScreen(1.0)
    dsum = sum(abs(c) for c in olds.e["coefficients"])
    pmax = max(new0.P) ** 2 * sum(w * math.sqrt(math.pi / z) for w, z in new0.terms)
    r_ecp = math.sqrt(
        math.log(pmax / 1e-16) / mu(min(new0.ap), min(z for _, z in new0.terms))
    )
    M = translations(r_ecp)
    S = np.array(
        [
            POS[1] + m
            for m in M
            if np.min(np.linalg.norm(POS - (POS[1] + m), axis=1)) <= r_ecp
        ]
    )
    L = translations(2 * r_ecp)
    out = {k: [] for k in ("a", "b", "il", "iu", "new", "lnold")}
    for il, lv in enumerate(L):
        for a in range(5):
            RA = np.linalg.norm(S - cent[a], axis=1)
            for b in range(5):
                rb = cent[b] + lv
                RB = np.linalg.norm(S - rb, axis=1)
                al, be = new0.ap[a], new0.ap[b]
                tot = np.zeros(len(S))
                for w, z in new0.terms:
                    p = al + be + z
                    tot += (
                        w
                        * math.sqrt(math.pi / p)
                        * np.exp(
                            -(
                                al * be * (RA - RB) ** 2
                                + al * z * RA**2
                                + be * z * RB**2
                            )
                            / p
                        )
                    )
                bn = new0.P[a] * new0.P[b] * tot
                dab2 = np.sum((cent[a] - rb) ** 2)
                e = np.maximum.reduce(
                    [
                        np.full(len(S), mu(olds.amin[a], olds.amin[b]) * dab2),
                        mu(olds.amin[a], olds.zmin) * RA**2,
                        mu(olds.amin[b], olds.zmin) * RB**2,
                    ]
                )
                lbo = (
                    math.log(dsum)
                    + 3
                    + max(olds.log_pref[a] + olds.log_pref[b], 0.0)
                    - e
                )
                sel = np.where((bn > 1e-24) | (lbo > math.log(1e-24)))[0]
                out["a"] += [a] * len(sel)
                out["b"] += [b] * len(sel)
                out["il"] += [il] * len(sel)
                out["iu"] += list(sel)
                out["new"] += list(bn[sel])
                out["lnold"] += list(lbo[sel])
    d = {k: np.array(v) for k, v in out.items()}
    d["L"], d["S"] = L, S
    np.savez(f, **d)
    return d
