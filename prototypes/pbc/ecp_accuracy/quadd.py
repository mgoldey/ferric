import numpy as np, ecp_clean as E, quad as Q


def rich(f, h=1e-3):
    fd = lambda hh: (f(hh) - f(-hh)) / (2 * hh)
    return (4 * fd(h / 2) - fd(h)) / 3


cases = [
    (
        "MID worst: centre x0, bra I p(1)@I0, ket I p(1)@I0+L, U3, elem (0,2)",
        E.I_SH[3],
        E.I0,
        E.I_SH[3],
        E.I0 + E.LV,
        E.I0 + np.array([1.5, -1.2, 0.8]),
        "centre",
        0,
        (0, 2),
    ),
    (
        "CLEAN worst: ket x0, bra I p(1)@I0, ket H s@H0+L, U1, elem (0,0)",
        E.I_SH[3],
        E.I0,
        E.H_STO3G,
        E.H0 + E.LV,
        E.I0 + E.LV + np.array([0.3, 0.2, -0.25]),
        "ket",
        0,
        (0, 0),
    ),
    (
        "CLEAN #2: bra x2, bra H s@H0, ket I p(1)@I0+L, U0=I0, elem (0,0)",
        E.H_STO3G,
        E.H0,
        E.I_SH[3],
        E.I0 + E.LV,
        E.I0,
        "bra",
        2,
        (0, 0),
    ),
]
for name, sa, pa, sb, pb, pc, slot, x, idx in cases:
    e = np.eye(3)[x]
    mv = lambda d: {
        "bra": (pa + d * e, pb, pc),
        "ket": (pa, pb + d * e, pc),
        "centre": (pa, pb, pc + d * e),
    }[slot]
    k = {"bra": 0, "ket": 1, "centre": 2}[slot]
    an = E.lib_deriv(sa, pa, sb, pb, pc)[k][x][idx]
    libfd = (
        E.rich(lambda d: E.lib_value(sa, *mv(d)[:1], sb, *mv(d)[1:])) if False else None
    )
    q = rich(
        lambda d: Q.quad_value(sa, mv(d)[0], sb, mv(d)[1], mv(d)[2], 590, 120)[idx]
    )
    py = E.rich(lambda d: E.pyscf_value(sa, mv(d)[0], sb, mv(d)[1], mv(d)[2]))[idx]
    lf = E.rich(lambda d: E.lib_value(sa, mv(d)[0], sb, mv(d)[1], mv(d)[2]))[idx]
    v_l = E.lib_value(sa, pa, sb, pb, pc)[idx]
    v_q = Q.quad_value(sa, pa, sb, pb, pc, 590, 120)[idx]
    v_p = E.pyscf_value(sa, pa, sb, pb, pc)[idx]
    print(name)
    print(
        f"  derivative: libecpint analytic {an:+.12e}  libecpint FD {lf:+.12e}  quadrature FD {q:+.12e}  PySCF FD {py:+.12e}"
    )
    print(
        f"    |an - quad| {abs(an - q):.2e}   |libFD - quad| {abs(lf - q):.2e}   |PySCF FD - quad| {abs(py - q):.2e}"
    )
    print(
        f"  value: libecpint {v_l:+.15e} quadrature {v_q:+.15e} PySCF {v_p:+.15e}  |lib-quad| {abs(v_l - v_q):.2e} |PySCF-quad| {abs(v_p - v_q):.2e}",
        flush=True,
    )
