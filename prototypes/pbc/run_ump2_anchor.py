"""Stage-8 exactness anchors for Gamma UMP2 / URPA (pbc_ump2.py).

(a) closed shell through the U path == R path (pbc_mp2 / pbc_rpa), both denominator conventions:
    H2/STO-3G a=4 pure-AFT and the one-s tri 4H anchor cell (dense ERI and trivial-aux B).
(b) open shell, trivial-aux RS-GDF B == dense pure-AFT ERI: tri one-s TRIPLET (na 3, nb 1), same C and own B-SCF.
(c) URPA O(Pi^2) term == direct UMP2 (same- and opposite-spin Coulomb terms).
Mutations are printed next to each anchor (they must be LARGE where the anchor can see them)."""

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_ump2 as U  # noqa: E402
from pbc_gamma import Cell, build_integrals, rhf  # noqa: E402
from pbc_gdf import build_gdf, jk_from_B  # noqa: E402
from pbc_mp2 import denominators, gamma_mp2  # noqa: E402
from pbc_rpa import drpa_quad, gamma_drpa  # noqa: E402
from pbc_mp2 import bia_from_B  # noqa: E402
from pbc_uhf import uhf  # noqa: E402
from test_prototype import ANCHOR_ALPHA, H2_A, H2_ATOMS, TRI_A, TRI_ATOMS, _tri_anchor  # noqa: E402
from pyscf import gto  # noqa: E402

PENTA_ATOMS = TRI_ATOMS + [("H", (1.3, 3.1, 4.0))]


def penta_anchor():
    """Trivial-aux limit with ALL THREE UMP2 blocks nonzero: 5 one-s H in the tri cell, doublet (na 3, nb 2)
    -> alpha 3 occ / 2 vir, beta 2 occ / 3 vir.  Aux = all 15 x 8 periodic pair products (exact span)."""
    cell = Cell(TRI_A, PENTA_ATOMS, {"H": [[0, [ANCHOR_ALPHA, 1.0]]]})
    n = len(PENTA_ATOMS)
    cen = [
        0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a)
        for i in range(n)
        for j in range(i, n)
        for h in np.ndindex(2, 2, 2)
    ]
    aux = gto.M(
        atom=[("X", c) for c in cen],
        basis={"X": [[0, [2 * ANCHOR_ALPHA, 1.0]]]},
        unit="B",
        cart=True,
        verbose=0,
    )
    return cell, aux


CONVS = ("shifted", "unshifted")


def mutate(attr, value, fn):
    old = getattr(U, attr)
    setattr(U, attr, value)
    try:
        return fn()
    finally:
        setattr(U, attr, old)


t0 = time.time()
h2 = build_integrals(
    Cell(H2_A, H2_ATOMS, "sto-3g"), None, exxdiv="ewald", verbose=False
)
cell, aux = _tri_anchor()
tri = build_integrals(cell, None, exxdiv="ewald", verbose=False)
Btri = build_gdf(cell, None, auxmol=aux)["B"]
pcell, paux = penta_anchor()
penta = build_integrals(pcell, None, exxdiv="ewald", verbose=False)
Bpenta = build_gdf(pcell, None, auxmol=paux, prec=1e-15)[
    "B"
]  # 1e-13 leaves max|dI| 3.5e-10 (SR truncation) -> dUMP2 1e-11
print(f"integrals built ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------------------ (a)
print("(a) closed shell, U path - R path")
for name, I, nel, B in (("H2 a=4", h2, 2, None), ("tri one-s", tri, 4, Btri)):
    n = nel // 2
    _, eps, _, C = rhf(
        I["S"], I["h"], I["I"], I["enn"], nel, conv=1e-12, return_mo=True
    )
    u = uhf(I["S"], I["h"], I["I"], I["enn"], n, n, conv=1e-12)
    for c in CONVS:
        e = np.concatenate(denominators(eps, n, I["madelung"], c))
        den = U.u_denominators(eps, eps, n, n, I["madelung"], c)
        rm = gamma_mp2(C, e, n, eri=I["I"])[0]
        rr = gamma_drpa(C, e, n, eri=I["I"], method="plasmon")
        um = U.gamma_ump2(C, C, den, n, n, eri=I["I"])[0]
        ur = U.gamma_urpa(C, C, den, n, n, eri=I["I"], method="plasmon")
        uq = U.gamma_urpa(C, C, den, n, n, eri=I["I"], method="quad")
        denu = U.u_denominators(u["eps_a"], u["eps_b"], n, n, I["madelung"], c)
        um2 = U.gamma_ump2(u["Ca"], u["Cb"], denu, n, n, eri=I["I"])[0]
        ur2 = U.gamma_urpa(u["Ca"], u["Cb"], denu, n, n, eri=I["I"], method="plasmon")
        line = (
            f"  {name:10s} {c:9s} MP2 {rm:+.12e}  U-R {um - rm:+.1e} (UHF orbitals {um2 - rm:+.1e});  "
            f"dRPA {rr:+.12e}  U-R plasmon {ur - rr:+.1e} quad {uq - rr:+.1e} (UHF orbitals {ur2 - rr:+.1e})"
        )
        if B is not None:
            ubq = U.gamma_urpa(C, C, den, n, n, B=B)
            rbq = drpa_quad(bia_from_B(B, C[:, :n], C[:, n:]), e[:n], e[n:])
            line += f"; B-quad U-R {ubq - rbq:+.1e}"
        print(line)
        mf4 = mutate(
            "SPIN_FACTOR",
            4.0,
            lambda: U.gamma_urpa(C, C, den, n, n, eri=I["I"], method="quad"),
        )
        _, eaa, ebb, eab = U.gamma_ump2(C, C, den, n, n, eri=I["I"])
        print(
            f"      MUTANTS: URPA per-spin factor 4 {mf4 - rr:+.1e};  UMP2 same-spin 1/2 dropped "
            f"{eaa + ebb - rm:+.1e};  UMP2 ab x1/2 {-0.5 * eab:+.1e}"
        )

# ------------------------------------------------------------------------------ (b) + (c)
for label, tri, Btri, na, nb in (
    ("tri one-s TRIPLET (na 3, nb 1): only ab nonzero", tri, Btri, 3, 1),
    ("penta one-s DOUBLET (na 3, nb 2): aa, bb, ab nonzero", penta, Bpenta, 3, 2),
):
    print(f"(b) {label}: trivial-aux B - dense ERI")
    for c in CONVS:
        vm = tri["madelung"]
        ud = uhf(
            tri["S"], tri["h"], tri["I"], tri["enn"], na, nb, conv=1e-12
        )  # none-first (Iteration 6 trap)
        ub = uhf(
            tri["S"], tri["h"], None, tri["enn"], na, nb, conv=1e-12, jk=jk_from_B(Btri)
        )
        den = U.u_denominators(ud["eps_a"], ud["eps_b"], na, nb, vm, c)
        denb = U.u_denominators(ub["eps_a"], ub["eps_b"], na, nb, vm, c)
        Ca, Cb = ud["Ca"], ud["Cb"]
        em, eaa, ebb, eab = U.gamma_ump2(Ca, Cb, den, na, nb, eri=tri["I"])
        er = U.gamma_urpa(Ca, Cb, den, na, nb, eri=tri["I"], method="plasmon")
        bm = U.gamma_ump2(Ca, Cb, den, na, nb, B=Btri)[0]
        bm_own = U.gamma_ump2(ub["Ca"], ub["Cb"], denb, na, nb, B=Btri)[0]
        bq = U.gamma_urpa(Ca, Cb, den, na, nb, B=Btri)
        bq_own = U.gamma_urpa(ub["Ca"], ub["Cb"], denb, na, nb, B=Btri)
        eq = U.gamma_urpa(Ca, Cb, den, na, nb, eri=tri["I"], method="quad")
        print(
            f"  {c:9s} <S2> {ud['s2']:.6f}  UMP2 {em:+.12e} (aa {eaa:+.3e} bb {ebb:+.3e} ab {eab:+.3e})  "
            f"B - dense {bm - em:+.1e} (own B-SCF {bm_own - em:+.1e})"
        )
        print(
            f"            URPA plasmon {er:+.12e}  B-quad - plasmon {bq - er:+.1e} (own B-SCF {bq_own - er:+.1e})"
            f"  dense-quad - plasmon {eq - er:+.1e}"
        )
        # mutations visible only in the open shell
        swap = lambda B, Ca_, Cb_, na_, nb_, frozen=0: (
            U._bia(B, Ca_, na_, frozen),
            U._bia(B, Ca_, nb_, frozen),
        )  # noqa: E731
        mm = mutate("u_bia", swap, lambda: U.gamma_ump2(Ca, Cb, den, na, nb, B=Btri)[0])
        mr = mutate("u_bia", swap, lambda: U.gamma_urpa(Ca, Cb, den, na, nb, B=Btri))
        mf4 = mutate(
            "SPIN_FACTOR", 4.0, lambda: U.gamma_urpa(Ca, Cb, den, na, nb, B=Btri)
        )
        print(
            f"            MUTANTS: alpha C for beta B: UMP2 {mm - em:+.1e} URPA {mr - er:+.1e};  per-spin factor 4 "
            f"URPA {mf4 - er:+.1e};  same-spin 1/2 dropped {eaa + ebb:+.1e};  exchange dropped (aa,bb) "
            f"{U.direct_ump2(U.u_ovov(Ca, Cb, na, nb, eri=tri['I']), *den) - em:+.1e}"
        )
        # (c)
        Ba, Bb = U.u_bia(Btri, Ca, Cb, na, nb)
        so = U.urpa_second_order_quad(Ba, Bb, *den)
        dm = U.direct_ump2(U.u_ovov(Ca, Cb, na, nb, eri=tri["I"]), *den)
        so4 = mutate("SPIN_FACTOR", 4.0, lambda: U.urpa_second_order_quad(Ba, Bb, *den))
        print(
            f"(c) {c:9s} tr Pi^2 term {so:+.12e}  - direct UMP2 {so - dm:+.1e}  (URPA - direct UMP2 {er - dm:+.1e});"
            f"  MUTANT factor 4: {so4 - dm:+.1e}"
        )
print(f"done ({time.time() - t0:.0f}s)")
