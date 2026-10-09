"""Tests for the ferric-owned ECP quadrature prototype (FINDINGS "Iteration 25").

Run (PySCF venv; PySCF is used ONLY as an oracle here, never by ecpq.py):
  cd reference/pbc/ecp_quadrature && OPENBLAS_NUM_THREADS=1 python -m pytest -q test_ecpq.py      (~4-5 min)
Bars are set from the measured values (FINDINGS tables) with headroom, and each rejects the known-bad alternative
(libecpint / PySCF ipnuc / a deliberately broken construction -- the negative controls at the end)."""

import os
import re

import mpmath as mp
import numpy as np
import pytest

import ecpq as E
import fixtures as F
import oracle as O
from measure import AU_CS, AU_L, HI_CS, moved, moved_e, rich, val

HERE = os.path.dirname(os.path.abspath(__file__))


def hi_triples():
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    return [(sa, sb, F.ecp_i(c)) for sa in bra for sb in ket for c in HI_CS]


def au_triples(sel=(0, 5, 6, 9, 11, 12, 14)):
    bra = F.auh_shells(F.AU0, F.H_AU)
    ket = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    return [(bra[i], ket[j], F.ecp_au(c)) for i in sel for j in sel for c in AU_CS]


# ------------------------------------------------------------------ building blocks


def test_real_harmonics_orthonormal():
    worst = 0.0
    for l in range(10):
        for l2 in range(l + 1):
            for i, a in enumerate(E.ylm_coeffs(l)):
                for j, b in enumerate(E.ylm_coeffs(l2)):
                    v = sum(
                        ca
                        * cb
                        * E.sphere_monomial(k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2])
                        for k1, ca in a.items()
                        for k2, cb in b.items()
                    )
                    worst = max(worst, abs(v - float(l == l2 and i == j)))
    assert worst < 1e-13, worst


def test_cart2sph_equals_ferric_ecp_rs_tables():
    """Parse C2S0..C2S4 out of crates/ferric-integrals/src/ecp.rs: the prototype's spherical output IS ferric's."""
    src = open(os.path.join(HERE, "../../../crates/ferric-integrals/src/ecp.rs")).read()
    for l in range(5):
        m = re.search(rf"static C2S{l}: \[f64; \d+\] = \[(.*?)\];", src, re.S)
        vals = np.array(
            [
                float(x.replace("_", ""))
                for x in m.group(1).replace("\n", " ").split(",")
                if x.strip()
            ]
        )
        tab = vals.reshape(E.ncart(l), 2 * l + 1)
        assert np.abs(tab - E.c2s(l)).max() < 1e-14, l


def test_bessel_matches_mpmath():
    mp.mp.dps = 40
    zs = np.concatenate(
        [
            [0, 1e-300, 1e-30, 1e-9, 1e-8, 1e-6],
            np.geomspace(1e-4, 1e5, 150),
            [15.999, 16.0, 16.001, 47.99, 48.0, 48.01],
        ]
    )
    worst = 0.0
    for nm in (0, 1, 3, 6, 9, 12):
        k = E.bessel_ktil(nm, zs)
        for i, z in enumerate(zs):
            for n in range(nm + 1):
                r = (
                    float(
                        mp.exp(-mp.mpf(z))
                        * mp.sqrt(mp.pi / (2 * mp.mpf(z)))
                        * mp.besseli(n + 0.5, mp.mpf(z))
                    )
                    if z > 0
                    else float(n == 0)
                )
                worst = max(worst, abs(k[n, i] - r) / abs(r) if r else abs(k[n, i]))
    assert worst < 5e-15, worst


def test_radial_tensor_form_equals_explicit_projections():
    """Production (radial tensor R + one angular contraction) vs forming F^A_lm(r), F^B_lm(r) at every node."""
    worst = 0.0
    for sa, sb, e in au_triples((0, 6, 9, 11, 14))[::2]:
        _, semi, _ = E.split_ecp(e)
        v1 = E.type2_cart(sa, sb, e["center"], semi, screen=False)
        v2 = E.type2_cart_direct(sa, sb, e["center"], semi)
        worst = max(worst, np.abs(v1 - v2).max() / max(np.abs(v2).max(), 1e-3))
    assert worst < 1e-12, worst


def test_screening_changes_nothing_measurable():
    worst = 0.0
    for sa, sb, e in au_triples((0, 9, 11, 14)):
        worst = max(
            worst,
            np.abs(
                E.triple_cart(sa, sb, e) - E.triple_cart(sa, sb, e, screen=False)
            ).max(),
        )
    assert worst < 1e-20, worst


# ------------------------------------------------------------------ values vs the oracles


def test_hi_values_match_pyscf():
    worst = max(
        np.abs(val(sa, sb, e) - O.pyscf_value(sa, sb, e)).max()
        for sa, sb, e in hi_triples()
    )
    assert worst < 1e-13, (
        worst
    )  # measured 2.8e-14 on elements up to 1.5; libecpint is 2e-7..7e-7 off


def test_auh_values_match_pyscf_f_shells():
    worst_big = worst_all = 0.0
    for sa, sb, e in au_triples():
        v, p = val(sa, sb, e), O.pyscf_value(sa, sb, e)
        d = np.abs(v - p)
        worst_all = max(worst_all, d.max())
        big = np.abs(p) > 1e-6
        if big.any():
            worst_big = max(worst_big, d[big].max())
    assert worst_big < 5e-13, worst_big  # measured 9.1e-14
    assert worst_all < 2e-12, (
        worst_all
    )  # PySCF's own floor on ~1e-12 elements (see the Lebedev test below)


def test_tiny_element_where_pyscf_is_off_matches_lebedev():
    """AuH: Au s(tight) - Au d_L around an ECP at the H end, elements ~2e-12. PySCF is 5e-13 off; the independent
    Lebedev quadrature (2030 x 240) agrees with ours to 1e-17 (3074 x 300: 1.6e-19)."""
    aub = F.auh_shells(F.AU0, F.H_AU)
    auk = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    e = F.ecp_au(AU_CS[2])
    v = val(aub[0], auk[9], e)
    q = O.lebedev_value(aub[0], auk[9], e, 1202, 200)
    p = O.pyscf_value(aub[0], auk[9], e)
    assert np.abs(v - q).max() < 1e-15
    assert np.abs(p - q).max() > 1e-13  # the oracle disagreement is real, not ours


def test_mid_element_matches_lebedev():
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    e = F.ecp_i(HI_CS[3])
    assert (
        np.abs(
            val(bra[4], ket[4], e) - O.lebedev_value(bra[4], ket[4], e, 1202, 160)
        ).max()
        < 1e-14
    )


# ------------------------------------------------------------------ smoothness, derivatives, on-centre


@pytest.mark.parametrize("case", ["mid", "dproj", "aufF"])
def test_value_is_smooth_in_geometry(case):
    """41 points over +-5e-4 Bohr, cubic-fit residual. libecpint: 6.9e-7 on the MID element (FINDINGS 2026-09-27)."""
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    aub = F.auh_shells(F.AU0, F.H_AU)
    auk = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    f = {
        "mid": lambda h: val(bra[4], ket[4], moved_e(F.ecp_i(HI_CS[3]), 0, h))[0, 2],
        "dproj": lambda h: val(
            bra[4], ket[4], moved_e(F.only_channels(F.ecp_i(HI_CS[3]), [2]), 0, h)
        )[0, 2],
        "aufF": lambda h: val(aub[11], auk[11], moved_e(F.ecp_au(AU_CS[1]), 1, h))[
            2, 3
        ],
    }[case]
    hs = np.arange(-20, 21) * 2.5e-5
    v = np.array([f(h) for h in hs])
    r = v - np.polyval(np.polyfit(hs, v, 3), hs)
    assert np.abs(r).max() < 1e-13, np.abs(r).max()


def _deriv_check(sa, sb, e, h=1e-3):
    """max relative |analytic - Richardson FD| over slot-blocks with max|dV| > 1e-6 (smaller blocks: absolute, < 1e-11,
    else returned as a failure). Below 1e-6 the FD is roundoff-limited (value roundoff / h ~ 1e-17 on 1e-10 blocks)."""
    D = E.ecp_block_deriv_spherical([sa], [sb], [e])
    worst = 0.0
    for x in range(3):
        for an, fd in (
            (D["bra"][x], rich(lambda d: val(moved(sa, x, d), sb, e), h)),
            (D["ket"][x], rich(lambda d: val(sa, moved(sb, x, d), e), h)),
            (D["centre"][0][x], rich(lambda d: val(sa, sb, moved_e(e, x, d)), h)),
        ):
            sc = np.abs(an).max()
            d = np.abs(an - fd).max()
            if sc > 1e-6:
                worst = max(worst, d / sc)
            elif d > 1e-11:
                return 1.0
    return worst


def test_derivatives_match_fd_of_own_values():
    ts = hi_triples()[::3] + au_triples((6, 9, 11, 14))[::5]
    worst = max(_deriv_check(*t) for t in ts)
    assert worst < 1e-9, (
        worst
    )  # measured: FINDINGS table (FD floor ~1e-12 abs from value roundoff)


def test_quadrature_target_element():
    """dB_z <H1s|U_I|H1s_L>, L = (0,0,7): target -6.658296236e-3 (independent quadrature, FINDINGS Iteration 22).
    libecpint 1.8e-8 off, PySCF ECPscalar_ipnuc 1.1e-7 off; measured here 4.4e-13."""
    sa = F.shell(F.H_STO3G, F.H0)
    sb = F.shell(F.H_STO3G, F.H0 + np.array([0.0, 0.0, 7.0]))
    D = E.ecp_block_deriv_spherical([sa], [sb], [F.ecp_i(F.I0)])
    assert abs(D["ket"][2][0, 0] - (-6.658296236e-3)) < 5e-12


def test_on_centre_shell_exact_and_differentiable_through_the_centre():
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    aub = F.auh_shells(F.AU0, F.H_AU)
    auk = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    for sa, sb, e in (
        (bra[3], ket[0], F.ecp_i(F.I0)),
        (bra[1], ket[4], F.ecp_i(F.I0)),
        (aub[9], auk[11], F.ecp_au(F.AU0)),
        (aub[11], aub[14], F.ecp_au(F.AU0)),
    ):
        assert np.abs(val(sa, sb, e) - O.pyscf_value(sa, sb, e)).max() < 1e-14
        assert (
            _deriv_check(sa, sb, e) < 1e-9
        )  # FD moves the shell THROUGH the centre (libecpint cannot)


def test_translation_invariance_and_block_bookkeeping():
    """bra + ket + sum_g centre = 0 per element; masked multi-centre block == sum of its enabled triples."""
    bra = F.hi_shells(F.H0, F.I0)[:3]
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)[2:]
    ecps = [F.ecp_i(c) for c in HI_CS[:3]]
    mask = [int(k % 3 != 0) for k in range(len(bra) * len(ket) * len(ecps))]
    D = E.ecp_block_deriv_spherical(bra, ket, ecps, mask, [0, 1, 0], 2)
    assert np.abs(D["bra"] + D["ket"] + D["centre"].sum(axis=0)).max() < 1e-15
    V = E.ecp_block_spherical(bra, ket, ecps, mask)
    ref = np.zeros_like(V)
    ro = np.cumsum([0] + [2 * s["l"] + 1 for s in bra])
    co = np.cumsum([0] + [2 * s["l"] + 1 for s in ket])
    for a, sa in enumerate(bra):
        for b, sb in enumerate(ket):
            for u, e in enumerate(ecps):
                if mask[(a * len(ket) + b) * len(ecps) + u]:
                    ref[ro[a] : ro[a + 1], co[b] : co[b + 1]] += val(sa, sb, e)
    assert np.abs(V - ref).max() < 1e-15


# ------------------------------------------------------------------ negative controls (the tests CAN fail)


def _hi_worst():
    return max(
        np.abs(val(sa, sb, e) - O.pyscf_value(sa, sb, e)).max()
        for sa, sb, e in hi_triples()[::4]
    )


def test_negative_control_truncated_bessel_expansion(monkeypatch):
    """Dropping the lam > l terms of the plane-wave expansion (an 'on-centre' approximation) must fail the oracle."""
    orig = E._proj_tensor_cached.__wrapped__

    def trunc(lA, l, ax, ay, az):
        T = orig(lA, l, ax, ay, az).copy()
        T[..., l + 1 :] = 0.0
        return T

    monkeypatch.setattr(E, "_proj_tensor_cached", trunc)
    assert _hi_worst() > 1e-3


def test_negative_control_wrong_radial_power(monkeypatch):
    """U = d r^(n-1) e^(-zeta r^2) instead of r^(n-2) (the alternative FINDINGS Iteration 22 ruled out) must fail."""
    orig = E.split_ecp

    def shifted(ecp):
        L, semi, loc = orig(ecp)
        return (
            L,
            {l: [(n + 1, z, d) for n, z, d in ts] for l, ts in semi.items()},
            [(n + 1, z, d) for n, z, d in loc],
        )

    monkeypatch.setattr(E, "split_ecp", shifted)
    assert _hi_worst() > 1e-3


def test_negative_control_underresolved_radial_grid(monkeypatch):
    """16 Gauss-Legendre nodes per window must fail the 1e-13 value bar (so the node count is load-bearing)."""
    monkeypatch.setattr(E, "RAD_N", 16)
    monkeypatch.setattr(E, "RAD_NH", 8)
    assert _hi_worst() > 1e-10
