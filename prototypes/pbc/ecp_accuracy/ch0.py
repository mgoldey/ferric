import numpy as np, ecp_clean as E, quad as Q

FULL = list(E.ECP_I)


def rich(f, h=1e-3):
    fd = lambda hh: (f(hh) - f(-hh)) / (2 * hh)
    return (4 * fd(h / 2) - fd(h)) / 3


cases = [
    (0, "centre", 0, E.I_SH[2], E.I0, E.I_SH[0], E.I0 + E.LV, 3),
    (0, "bra", 0, E.I_SH[2], E.I0, E.I_SH[1], E.I0 + E.LV, 3),
    (1, "centre", 0, E.H_STO3G, E.H0, E.H_STO3G, E.H0 + E.LV, 2),
]
cs = [
    E.I0,
    E.I0 + E.LV + np.array([0.3, 0.2, -0.25]),
    E.H0 + np.array([-0.5, 0.4, 0.6]),
    E.I0 + np.array([1.5, -1.2, 0.8]),
]
for ch, slot, x, sa, pa, sb, pb, iu in cases:
    E.ECP_I[:] = [
        (ll, ts) if ll == ch else (ll, [(n, z, 0.0) for n, z, d in ts])
        for ll, ts in FULL
    ]
    pc = cs[iu]
    e = np.eye(3)[x]
    k = {"bra": 0, "ket": 1, "centre": 2}[slot]
    mv = lambda d: {
        "bra": (pa + d * e, pb, pc),
        "ket": (pa, pb + d * e, pc),
        "centre": (pa, pb, pc + d * e),
    }[slot]
    an = E.lib_deriv(sa, pa, sb, pb, pc)[k][x][0, 0]
    lf = rich(lambda d: E.lib_value(sa, mv(d)[0], sb, mv(d)[1], mv(d)[2])[0, 0])
    q = rich(
        lambda d: Q.quad_value(sa, mv(d)[0], sb, mv(d)[1], mv(d)[2], 590, 120)[0, 0]
    )
    vl = E.lib_value(sa, pa, sb, pb, pc)[0, 0]
    vq = Q.quad_value(sa, pa, sb, pb, pc, 590, 120)[0, 0]
    print(
        f"channel {ch} {slot} x{x} U{iu}: analytic {an:+.10e} libFD {lf:+.10e} quadFD {q:+.10e} | |an-quad| {abs(an - q):.2e} |libFD-quad| {abs(lf - q):.2e} | value lib-quad {vl - vq:+.2e}",
        flush=True,
    )
