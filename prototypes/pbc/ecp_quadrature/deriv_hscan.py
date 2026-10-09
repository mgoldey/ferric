"""h-scan of |analytic - Richardson FD| on the worst derivative slot-blocks of the AuH fixture.
Truncation error falls as h^4; value noise (non-smooth values) grows as 1/h. Prints both regimes per block."""

import numpy as np

import ecpq as E
from measure import fixture_triples, moved, moved_e, rich, val


def slots(sa, sb, e, x):
    return {
        "bra": lambda d: val(moved(sa, x, d), sb, e),
        "ket": lambda d: val(sa, moved(sb, x, d), e),
        "centre": lambda d: val(sa, sb, moved_e(e, x, d)),
    }


def main():
    _, au = fixture_triples()
    rows = []
    for _, sa, sb, e in au[::3]:
        D = E.ecp_block_deriv_spherical([sa], [sb], [e])
        dmin = min(
            np.linalg.norm(sa["center"] - e["center"]),
            np.linalg.norm(sb["center"] - e["center"]),
        )
        for x in range(3):
            an = {"bra": D["bra"][x], "ket": D["ket"][x], "centre": D["centre"][0][x]}
            for k, f in slots(sa, sb, e, x).items():
                sc = np.abs(an[k]).max()
                if sc < 1e-10:
                    continue
                err = np.abs(an[k] - rich(f, 1e-3)).max()
                rows.append((err / sc, err, sc, dmin, k, x, sa, sb, e))
    rows.sort(key=lambda r: -r[0])
    for rel, err, sc, dmin, k, x, sa, sb, e in rows[:6]:
        an = {"bra": 0, "ket": 1, "centre": 2}[k]
        D = E.ecp_block_deriv_spherical([sa], [sb], [e])
        a = [D["bra"], D["ket"], D["centre"][0]][an][x]
        f = slots(sa, sb, e, x)[k]
        hs = (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4)
        es = [np.abs(a - rich(f, h)).max() / sc for h in hs]
        print(
            f"l {sa['l']},{sb['l']} {k} x{x} dmin {dmin:.2f} scale {sc:.2e}: rel err vs h "
            + " ".join(f"{h:.0e}:{v:.1e}" for h, v in zip(hs, es)),
            flush=True,
        )


if __name__ == "__main__":
    main()
