"""Issue #230: the head-corrected k-MP2 energy E3d has no n^-3 remainder.

E3d (fastk.account) = S + cubic linear head + Phi-tensor quadratic head on the direct term + first-order Fock head.
If its N_k^-1 coefficient is complete, E3d - E_inf decays as n^-5 or faster and the step ratio
    rho(n) = (E(n) - E(n+1)) / (E(n+1) - E(n+2))
sits well above the pure n^-3 value. Without the correction (S, or E1 = S + cubic head only) an n^-3 term survives
and rho falls back to the n^-3 value. Neither statistic needs an E_inf.

| fixture (H2/STO-3G, z) | pure n^-3 | pure n^-5 | E3d | E1 (mutant) | S (mutant) | bar |
|---|---|---|---|---|---|---|
| a = 10, n = 2, 3, 4 | 4.11 | 8.65 | 9.07 | 3.68 | 4.03 | rho in [7.5, 11.5] |
| a = 6,  n = 5, 6, 7 | 1.97 | 2.77 | 3.21 | 1.51 | 1.96 | rho >= 2.6, free c3 <= 1.5e-5 |

The free c3 is the n^-3 coefficient of an exact E_inf + c3 n^-3 + c5 n^-5 fit through n = 5, 6, 7; measured
+9.1e-6, 0.4% of the S coefficient 2.27e-3 (FINDINGS.md, "Addendum 2026-11-19: resolution").

The recorded rows below were measured 2026-11-19 by fastk.py (z, prec 1e-10). `test_recorded_series_*` re-reads them
(no SCF; it checks the bars are reachable and that both mutants fail them). The live tests recompute the series:
    PBC_HEAD=1  a = 10, n = 2, 3, 4   (serial, about 40 min on a quiet box)
    PBC_SLOW=1  a = 6,  n = 5, 6, 7   (serial, several hours)
Run from prototypes/pbc/kmp2_head with OPENBLAS_NUM_THREADS=1.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAST_BAND = (7.5, 11.5)
SLOW_RHO_MIN = 2.6
SLOW_C3_MAX = 1.5e-5

# (a, n) -> (S, E1, E3d), per-cell MP2 in Hartree
RECORDED = {
    (10.0, 2): (-1.3089458512929619e-2, -1.3153951930846623e-2, -1.3168499556315875e-2),
    (10.0, 3): (-1.3143374618302244e-2, -1.3162484702442833e-2, -1.3166816027326100e-2),
    (10.0, 4): (-1.3156738807462690e-2, -1.3164800984148177e-2, -1.3166630448778633e-2),
    (6.0, 5): (-1.3339249122388480e-2, -1.3356898982431729e-2, -1.3357400948646294e-2),
    (6.0, 6): (-1.3346805123587258e-2, -1.3357019191924810e-2, -1.3357310802312155e-2),
    (6.0, 7): (-1.3350666551321933e-2, -1.3357098741631306e-2, -1.3357282761493916e-2),
}
KEYS = ("S", "E1", "E3d")


def step_ratio(e):
    e0, e1, e2 = e
    return (e0 - e1) / (e1 - e2)


def free_n3_term(ns, e):
    """c3 of the exact fit E_inf + c3 n^-3 + c5 n^-5 through three meshes."""
    a = np.array([[1.0, n**-3.0, n**-5.0] for n in ns])
    return np.linalg.solve(a, np.asarray(e, float))[1]


def recorded(a, ns, key):
    return [RECORDED[(a, n)][KEYS.index(key)] for n in ns]


def live(a, ns):
    import fastk as FK

    out = {k: [] for k in KEYS}
    for n in ns:
        res, cell = FK.run("z", (n, n, n), a=a, nproc=1, verbose=False)
        acc = FK.account(res, cell, verbose=False)
        for k in KEYS:
            out[k].append(acc[k])
    return out


def check_fast(e3d):
    rho = step_ratio(e3d)
    assert FAST_BAND[0] <= rho <= FAST_BAND[1], (
        f"a = 10 E3d step ratio {rho:.3f} outside {FAST_BAND}"
    )


def check_slow(e3d):
    rho = step_ratio(e3d)
    c3 = free_n3_term((5, 6, 7), e3d)
    assert rho >= SLOW_RHO_MIN, f"a = 6 E3d step ratio {rho:.3f} < {SLOW_RHO_MIN}"
    assert abs(c3) <= SLOW_C3_MAX, f"a = 6 E3d free n^-3 term {c3:+.3e}"


def test_recorded_series_pass_and_uncorrected_mutants_fail():
    check_fast(recorded(10.0, (2, 3, 4), "E3d"))
    check_slow(recorded(6.0, (5, 6, 7), "E3d"))
    for key in ("S", "E1"):
        with pytest.raises(AssertionError):
            check_fast(recorded(10.0, (2, 3, 4), key))
        with pytest.raises(AssertionError):
            check_slow(recorded(6.0, (5, 6, 7), key))


@pytest.mark.skipif(
    not os.environ.get("PBC_HEAD"),
    reason="set PBC_HEAD=1 (a = 10, n = 2..4, ~40 min serial)",
)
def test_live_a10_head_corrected_energy_converges_faster_than_n3():
    out = live(10.0, (2, 3, 4))
    check_fast(out["E3d"])
    for k in KEYS:
        np.testing.assert_allclose(
            out[k], recorded(10.0, (2, 3, 4), k), rtol=0, atol=1e-11
        )


@pytest.mark.skipif(
    not os.environ.get("PBC_SLOW"),
    reason="set PBC_SLOW=1 (a = 6, n = 5..7, hours serial)",
)
def test_live_a6_head_corrected_energy_has_no_free_n3_term():
    out = live(6.0, (5, 6, 7))
    check_slow(out["E3d"])
    for k in KEYS:
        np.testing.assert_allclose(
            out[k], recorded(6.0, (5, 6, 7), k), rtol=0, atol=1e-11
        )
