import numpy as np, ecp_clean as E

FULL = list(E.ECP_I)


def bands(ecp, h):
    E.ECP_I[:] = ecp
    bra = [(E.H_STO3G, E.H0)] + [(s, E.I0) for s in E.I_SH]
    ket = [(s, p + E.LV) for s, p in bra]
    cs = [
        E.I0,
        E.I0 + E.LV + np.array([0.3, 0.2, -0.25]),
        E.H0 + np.array([-0.5, 0.4, 0.6]),
        E.I0 + np.array([1.5, -1.2, 0.8]),
    ]
    res = {}
    for sa, pa in bra:
        for sb, pb in ket:
            for pc in cs:
                d = min(np.linalg.norm(pa - pc), np.linalg.norm(pb - pc))
                if d < 0.5:
                    continue
                band = "CLEAN" if d >= 1 else "MID"
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
                        w, s = res.get(band, (0, 0))
                        res[band] = (
                            max(w, np.abs(an[k][x] - fd).max()),
                            max(s, np.abs(an[k][x]).max()),
                        )
    return res


for name, ecp in (
    ("full ECP", FULL),
    ("no d projector (local+s+p)", [c for c in FULL if c[0] != 2]),
):
    for h in (1e-4, 1e-3):
        r = bands(ecp, h)
        print(
            name,
            f"h {h:.0e}:",
            " ".join(
                f"{b}: worst {w:.2e} scale {s:.3f} rel {w / s:.1e}"
                for b, (w, s) in r.items()
            ),
            flush=True,
        )
