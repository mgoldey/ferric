"""Candidate (d): extrapolation / fitting choice.  Series from FINDINGS (Iteration 12 + n = 5, 6 addenda),
H2/STO-3G a = 6 z, per-cell MP2.  Theory for a Gamma-centred mesh: odd-degree head terms cancel, so the
expansion is E_inf + c3 n^-3 + c5 n^-5 (+ exponentially small band sampling); an n^-1 term exists only for
UNshifted denominators (Madelung), never for shifted.  Artifact hypothesis: if E1's flatness were a fit artefact,
E1's c3 would change sign/magnitude between fit forms; if E_inf were ill-determined at the 7e-5/n^3 level, the E_inf
spread across fits would exceed 7e-5/27 = 2.6e-6."""

import itertools
import numpy as np

n = np.array([3, 4, 5, 6.0])
S = np.array([-1.3275179e-2, -1.3322562e-2, -1.333924912e-2, -1.334680512359e-2])
E1 = np.array([-1.3356869412e-2, -1.3357033893e-2, -1.335689898e-2, -1.335701919192e-2])
U = np.array([-1.5183439e-2, -1.4710788e-2, -1.442895829e-2, -1.424335457943e-2])


def fit(y, pw, idx):
    A = np.array([[1.0] + [x**-p for p in pw] for x in n[idx]])
    c, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
    return c


for name, y in (("S (shifted)", S), ("E1 (cubic head)", E1)):
    for pw in ((3,), (3, 5), (3, 4), (1,), (1, 3)):
        for idx in ([0, 1, 2, 3], [1, 2, 3], [2, 3]):
            if len(idx) < len(pw) + 1:
                continue
            c = fit(y, pw, idx)
            r = y[idx] - np.array([[1.0] + [x**-p for p in pw] for x in n[idx]]) @ c
            print(
                f"{name:16s} powers {str(pw):7s} n={list(n[idx].astype(int))}: E_inf {c[0]:.9e} coefs {np.array2string(c[1:], precision=4)} max|res| {abs(r).max():.1e}"
            )
print("E1 max - min over n=3..6:", E1.max() - E1.min())
