"""Measurements for FINDINGS "Iteration 25". Run sections by name:  python measure.py values smooth deriv target oncentre
(OPENBLAS_NUM_THREADS=1; PySCF venv — PySCF is the oracle only)."""

import sys
import time

import numpy as np

import ecpq as E
import fixtures as F
import oracle as O

HI_CS = [
    F.I0,
    F.I0 + F.LV + np.array([0.3, 0.2, -0.25]),
    F.H0 + np.array([-0.5, 0.4, 0.6]),
    F.I0 + np.array([1.5, -1.2, 0.8]),
]
AU_L = np.array([0.7, -1.1, 2.3])
AU_CS = [
    F.AU0,
    F.AU0 + AU_L + np.array([0.3, 0.2, -0.25]),
    F.H_AU + np.array([-0.5, 0.4, 0.6]),
]


def val(sa, sb, e, **kw):
    return E.to_sph(E.triple_cart(sa, sb, e, **kw), sa["l"], sb["l"])


def moved(s, x, d):
    t = dict(s)
    c = np.array(s["center"], float)
    c[x] += d
    t["center"] = c
    return t


def moved_e(e, x, d):
    t = dict(e)
    c = np.array(e["center"], float)
    c[x] += d
    t["center"] = c
    return t


def rich(f, h):
    fd = lambda hh: (f(hh) - f(-hh)) / (2 * hh)
    return (4 * fd(h / 2) - fd(h)) / 3


def fixture_triples():
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    hi = [("HI", sa, sb, F.ecp_i(c)) for sa in bra for sb in ket for c in HI_CS]
    bra = F.auh_shells(F.AU0, F.H_AU)
    ket = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    au = [("AuH", sa, sb, F.ecp_au(c)) for sa in bra for sb in ket for c in AU_CS]
    return hi, au


def sec_values():
    hi, au = fixture_triples()
    for name, ts in (("HI/LANL2DZ", hi), ("AuH/def2-SVP", au)):
        worst_abs = worst_rel = 0.0
        worst_big = 0.0  # |diff| over elements with |ref| > 1e-6
        n = 0
        for _, sa, sb, e in ts:
            v = val(sa, sb, e)
            p = O.pyscf_value(sa, sb, e)
            d = np.abs(v - p)
            worst_abs = max(worst_abs, d.max())
            big = np.abs(p) > 1e-6
            if big.any():
                worst_big = max(worst_big, d[big].max())
                worst_rel = max(worst_rel, (d[big] / np.abs(p[big])).max())
            n += v.size
        print(
            f"{name}: {len(ts)} triples, {n} elements: max|ours - PySCF| {worst_abs:.2e}; "
            f"over |V| > 1e-6: abs {worst_big:.2e}, rel {worst_rel:.2e}",
            flush=True,
        )


def sec_lebedev():
    """Selected triples vs the independent Lebedev quadrature (3 levels) and PySCF."""
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    aub = F.auh_shells(F.AU0, F.H_AU)
    auk = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    cases = [
        ("HI MID p(1)-p(1), U far", bra[4], ket[4], F.ecp_i(HI_CS[3])),
        ("HI H s - I p(1)_L, U@I0", bra[0], ket[4], F.ecp_i(F.I0)),
        ("HI I p(2) on-centre - I s(2)_L", bra[3], ket[1], F.ecp_i(F.I0)),
        ("AuH Au f on-centre - Au f_L", aub[11], auk[11], F.ecp_au(F.AU0)),
        ("AuH Au f_L - Au d, U off both", auk[11], aub[9], F.ecp_au(AU_CS[1])),
        ("AuH Au s(tight) - Au d_L, U@H", aub[0], auk[9], F.ecp_au(AU_CS[2])),
    ]
    for name, sa, sb, e in cases:
        v = val(sa, sb, e)
        p = O.pyscf_value(sa, sb, e)
        q = O.lebedev_value(sa, sb, e, 2030, 240)
        sc = np.abs(q).max()
        print(
            f"{name}: scale {sc:.2e}  |ours - Leb(2030,240)| {np.abs(v - q).max():.2e}  "
            f"|PySCF - Leb| {np.abs(p - q).max():.2e}  |ours - PySCF| {np.abs(v - p).max():.2e}",
            flush=True,
        )


def sec_smooth():
    """41 points over +-5e-4 Bohr; cubic-fit residual (FINDINGS 2026-09-27 scan: libecpint 6.9e-7, PySCF 1.7e-16)."""
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    aub = F.auh_shells(F.AU0, F.H_AU)
    auk = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    hs = np.arange(-20, 21) * 2.5e-5
    cases = [
        (
            "HI MID (d-proj case): centre x, I p(1) - I p(1)_L, U@I0+(1.5,-1.2,0.8), elem (0,2)",
            lambda h: val(bra[4], ket[4], moved_e(F.ecp_i(HI_CS[3]), 0, h))[0, 2],
        ),
        (
            "HI d-projector only, same element",
            lambda h: val(
                bra[4], ket[4], moved_e(F.only_channels(F.ecp_i(HI_CS[3]), [2]), 0, h)
            )[0, 2],
        ),
        (
            "HI bra ON centre moving through it: bra I p(2)@I0+h x, U@I0, ket H s_L, elem (0,0)",
            lambda h: val(moved(bra[3], 0, h), ket[0], F.ecp_i(F.I0))[0, 0],
        ),
        (
            "AuH f-f: centre y, Au f - Au f_L, U@Au+L+(.3,.2,-.25), elem (2,3)",
            lambda h: val(aub[11], auk[11], moved_e(F.ecp_au(AU_CS[1]), 1, h))[2, 3],
        ),
        (
            "AuH Au d ON centre moving through it: ket, elem (1,1) with H p",
            lambda h: val(aub[14], moved(aub[9], 2, h), F.ecp_au(F.AU0))[1, 1],
        ),
    ]
    for name, f in cases:
        v = np.array([f(h) for h in hs])
        c = np.polyfit(hs, v, 3)
        r = v - np.polyval(c, hs)
        print(
            f"{name}: |V| {abs(v).max():.3e}, cubic-fit residual max {abs(r).max():.2e}",
            flush=True,
        )


def sec_deriv():
    """Analytic (raised/lowered shells) vs Richardson FD of OWN values, every slot of every triple of the fixtures.
    Relative to each block's max |dV|; bands by min shell-centre distance."""
    hi, au = fixture_triples()
    au = [t for i, t in enumerate(au) if i % 3 == 0]  # every 3rd AuH triple (time)
    for name, ts, h in (("HI", hi, 1e-3), ("AuH", au, 1e-3)):
        rows = []
        for _, sa, sb, e in ts:
            D = E.ecp_block_deriv_spherical([sa], [sb], [e])
            dmin = min(
                np.linalg.norm(sa["center"] - e["center"]),
                np.linalg.norm(sb["center"] - e["center"]),
            )
            for x in range(3):
                fds = {
                    "bra": rich(lambda d: val(moved(sa, x, d), sb, e), h),
                    "ket": rich(lambda d: val(sa, moved(sb, x, d), e), h),
                    "centre": rich(lambda d: val(sa, sb, moved_e(e, x, d)), h),
                }
                an = {
                    "bra": D["bra"][x],
                    "ket": D["ket"][x],
                    "centre": D["centre"][0][x],
                }
                for k in fds:
                    rows.append(
                        (dmin, np.abs(an[k] - fds[k]).max(), np.abs(an[k]).max())
                    )
        for lo, hi_, band in (
            (0.0, 1e-9, "ON centre"),
            (1e-9, 1.0, "< 1 Bohr"),
            (1.0, 1e9, ">= 1 Bohr"),
        ):
            rs = [r for r in rows if lo <= r[0] < hi_]
            if rs:
                wa = max(r[1] for r in rs)
                big = [r for r in rs if r[2] > 1e-6]
                wr = max(r[1] / r[2] for r in big) if big else 0.0
                print(
                    f"{name} {band}: {len(rs)} slot-blocks, max|an - FD| {wa:.2e} (all); over {len(big)} blocks with "
                    f"max|dV| > 1e-6: max rel {wr:.2e}",
                    flush=True,
                )


def sec_target():
    sa = F.shell(F.H_STO3G, F.H0)
    sb = F.shell(F.H_STO3G, F.H0 + np.array([0.0, 0.0, 7.0]))
    e = F.ecp_i(F.I0)
    D = E.ecp_block_deriv_spherical([sa], [sb], [e])
    an = D["ket"][2][0, 0]
    fd = rich(lambda d: val(sa, moved(sb, 2, d), e), 1e-3)[0, 0]
    T = -6.658296236e-3
    print(
        f"dB_z <H1s|U_I|H1s_L>, L=(0,0,7): analytic {an:.12e}  own FD {fd:.12e}  target {T:.9e}  "
        f"|an - target| {abs(an - T):.1e}  |an - FD| {abs(an - fd):.1e}  (libecpint 1.8e-8, PySCF ipnuc 1.1e-7)"
    )


def sec_oncentre():
    """Shell ON its ECP centre: value vs PySCF, continuity as the shell leaves the centre, derivative vs FD across it."""
    aub = F.auh_shells(F.AU0, F.H_AU)
    bra = F.hi_shells(F.H0, F.I0)
    auk = F.auh_shells(F.AU0 + AU_L, F.H_AU + AU_L)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    # (on-centre shell, off-centre partner) -- an on-centre/on-centre pair of different parity is zero by symmetry
    for name, sa, sb, e in (
        ("HI I p(2)@I0 - H s_L, U@I0", bra[3], ket[0], F.ecp_i(F.I0)),
        ("HI I s(2)@I0 - I p(1)_L, U@I0", bra[1], ket[4], F.ecp_i(F.I0)),
        ("AuH Au d@Au - Au f_L, U@Au", aub[9], auk[11], F.ecp_au(F.AU0)),
        ("AuH Au f@Au - H p, U@Au", aub[11], aub[14], F.ecp_au(F.AU0)),
    ):
        v0 = val(sa, sb, e)
        p = O.pyscf_value(sa, sb, e)
        D = E.ecp_block_deriv_spherical([sa], [sb], [e])
        worst = 0.0
        for x in range(3):
            fd = rich(lambda d: val(moved(sa, x, d), sb, e), 1e-3)
            worst = max(
                worst, np.abs(D["bra"][x] - fd).max() / max(np.abs(fd).max(), 1e-30)
            )
        cont = [
            np.abs(val(moved(sa, 0, d), sb, e) - v0).max() for d in (1e-4, 1e-8, 1e-12)
        ]
        print(
            f"{name}: |ours - PySCF| {np.abs(v0 - p).max():.2e} (scale {np.abs(p).max():.2e}); "
            f"bra-derivative vs FD THROUGH the centre rel {worst:.2e}; |V(d) - V(0)| at d=1e-4/1e-8/1e-12: "
            + " ".join(f"{c:.1e}" for c in cont),
            flush=True,
        )


if __name__ == "__main__":
    t0 = time.time()
    for s in sys.argv[1:]:
        print(f"== {s}", flush=True)
        globals()["sec_" + s]()
    print(f"({time.time() - t0:.0f} s)")
