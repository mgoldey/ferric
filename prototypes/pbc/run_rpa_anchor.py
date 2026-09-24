"""Stage 7 exactness anchors on the periodic cells (H2 a=4, triclinic 4H; one s primitive al=0.5):
 (a) dRPA from the RS-GDF B in the trivial-aux limit (aux = every periodic pair product) via the
     FREQUENCY INTEGRAL  ==  dRPA from the dense pure-AFT (ia|jb) via the PLASMON formula (and
     ring-CCD Riccati) -- independent constructions of both the ERI and the energy;
 (b) O(V^2): -(1/2pi) int tr Pi^2/2 from B == direct MP2 from the dense (ia|jb);
plus mutations that must break them.  Both denominator conventions."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
import pbc_gdf
from pbc_gamma import build_integrals, rhf
from pbc_gdf import build_gdf, jk_from_B
from pbc_mp2 import denominators, ovov_from_eri, bia_from_B
from pbc_rpa import (
    gamma_drpa,
    drpa_quad,
    drpa_plasmon,
    drpa_riccati,
    direct_mp2,
    drpa_second_order_quad,
)
from run_mp2_anchor import anchor, CELLS

for name in CELLS:
    cell, aux = anchor(name)
    nel = 2 * (len(cell.atoms) // 2)
    nocc = nel // 2
    t = time.time()
    ref = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    g = build_gdf(cell, None, auxmol=aux)
    B = g["B"]
    vm = ref["madelung"]
    e0, eps0, _, C0 = rhf(
        ref["S"], ref["h"], ref["I"], ref["enn"], nel, conv=1e-12, return_mo=True
    )
    e1, eps1, _, C1 = rhf(
        ref["S"],
        ref["h"],
        None,
        ref["enn"],
        nel,
        conv=1e-12,
        jk=jk_from_B(B),
        return_mo=True,
    )
    print(
        f"{name}: build {time.time() - t:.1f}s naux {g['info']['naux_kept']}/{g['info']['naux']}  v_M {vm:.10f}",
        flush=True,
    )
    Co, Cv = C0[:, :nocc], C0[:, nocc:]
    for conv in ("shifted", "unshifted"):
        e = np.concatenate(denominators(eps0, nocc, vm, conv))
        eo, ev = e[:nocc], e[nocc:]
        e_1 = np.concatenate(denominators(eps1, nocc, vm, conv))
        ov_ex = ovov_from_eri(ref["I"], Co, Cv)
        pl = drpa_plasmon(ov_ex, eo, ev)
        ric = drpa_riccati(ov_ex, eo, ev) - pl
        q_same, nq = drpa_quad(bia_from_B(B, Co, Cv), eo, ev, return_n=True)
        q_own = gamma_drpa(C1, e_1, nocc, B=B)
        dm = direct_mp2(ov_ex, eo, ev)
        e2 = drpa_second_order_quad(bia_from_B(B, Co, Cv), eo, ev, n=256)
        print(
            f"  {conv:9s}: dRPA plasmon(dense AFT) {pl:.13e}  riccati {ric:+.1e}  "
            f"B-quad(n={nq}) same C {q_same - pl:+.1e}  own B-SCF {q_own - pl:+.1e}  |  "
            f"dMP2 {dm:.13e}  B trPi^2 term - dMP2 {e2 - dm:+.1e}  (dRPA-dMP2 {pl - dm:+.3e})",
            flush=True,
        )
    # mutations (shifted, exact orbitals)
    e = np.concatenate(denominators(eps0, nocc, vm, "shifted"))
    eo, ev = e[:nocc], e[nocc:]
    ov_ex = ovov_from_eri(ref["I"], Co, Cv)
    pl = drpa_plasmon(ov_ex, eo, ev)
    Bia = bia_from_B(B, Co, Cv)
    muts = {
        "Pi factor 4 -> 2 (spin factor lost)": drpa_quad(Bia / np.sqrt(2), eo, ev),
        "1/pi instead of 1/2pi": 2 * drpa_quad(Bia, eo, ev),
        "exchange in K (A = D + 2K - K_x)": drpa_plasmon(
            ov_ex - 0.5 * ov_ex.transpose(0, 3, 2, 1), eo, ev
        ),
        "unshifted eps fed while claiming shifted": drpa_quad(
            Bia, *denominators(eps0, nocc, vm, "unshifted")
        ),
        "Cv for both ov indices": drpa_quad(
            np.einsum("Pmn,mi,na->Pia", B, Cv, Cv), eo, ev
        )
        if nocc == Cv.shape[1]
        else None,
    }
    for k, v in muts.items():
        if v is not None:
            print(f"  MUTANT {k}: dE {v - pl:+.2e}")
    c2, a7 = anchor(name, classes=range(7))
    B7 = build_gdf(c2, None, auxmol=a7)["B"]
    print(
        f"  aux 7/8 classes: dE {drpa_quad(bia_from_B(B7, Co, Cv), eo, ev) - pl:+.2e}"
    )
    orig = pbc_gdf._subtract_g0
    for label, mut in (
        (
            "G=0 removed from J2 only",
            lambda J2, J3, S, q, c0: (J2 - c0 * np.outer(q, q), J3),
        ),
        (
            "G=0 removed from J3 only",
            lambda J2, J3, S, q, c0: (
                J2,
                J3 - c0 * S.reshape(-1)[:, None] * q[None, :],
            ),
        ),
        ("G=0 kept in both", lambda J2, J3, S, q, c0: (J2, J3)),
    ):
        pbc_gdf._subtract_g0 = mut
        Bm = build_gdf(cell, None, auxmol=aux)["B"]
        pbc_gdf._subtract_g0 = orig
        print(
            f"  MUTANT {label}: dE_dRPA {drpa_quad(bia_from_B(Bm, Co, Cv), eo, ev) - pl:+.2e}",
            flush=True,
        )
