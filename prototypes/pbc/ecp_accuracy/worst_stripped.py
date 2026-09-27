import numpy as np, ecp_clean as E

FULL = list(E.ECP_I)


def scan(ecp, h=1e-3, top=6):
    E.ECP_I[:] = ecp
    bra = [(E.H_STO3G, E.H0, "H s")] + [
        (s, E.I0, n) for s, n in zip(E.I_SH, E.NAMES[1:])
    ]
    ket = [(s, p + E.LV, n) for s, p, n in bra]
    cs = [
        E.I0,
        E.I0 + E.LV + np.array([0.3, 0.2, -0.25]),
        E.H0 + np.array([-0.5, 0.4, 0.6]),
        E.I0 + np.array([1.5, -1.2, 0.8]),
    ]
    rows = []
    for sa, pa, na in bra:
        for sb, pb, nb in ket:
            for iu, pc in enumerate(cs):
                d = min(np.linalg.norm(pa - pc), np.linalg.norm(pb - pc))
                if d < 0.5:
                    continue
                an = E.lib_deriv(sa, pa, sb, pb, pc)
                for x in range(3):
                    e = np.eye(3)[x]
                    for k, f in enumerate(
                        (
                            lambda t: E.lib_value(sa, pa + t * e, sb, pb, pc),
                            lambda t: E.lib_value(sa, pa, sb, pb + t * e, pc),
                            lambda t: E.lib_value(sa, pa, sb, pb, pc + t * e),
                        )
                    ):
                        fd = E.rich(f, h)
                        df = np.abs(an[k][x] - fd)
                        i = np.unravel_index(df.argmax(), df.shape)
                        rows.append(
                            (
                                df[i],
                                ["bra", "ket", "centre"][k],
                                x,
                                na,
                                nb,
                                iu,
                                i,
                                an[k][x][i],
                                fd[i],
                                d,
                            )
                        )
    rows.sort(key=lambda r: -r[0])
    for r in rows[:top]:
        print(
            "  %.2e %s x%d bra %s ket %s U%d elem %s an %+.10e fd %+.10e dmin %.2f" % r
        )
    return rows


for l in (3, 0, 1):
    print("channel", l, "only:")
    scan(
        [
            (ll, ts) if ll == l else (ll, [(n, z, 0.0) for n, z, d in ts])
            for ll, ts in FULL
        ],
        top=3,
    )
