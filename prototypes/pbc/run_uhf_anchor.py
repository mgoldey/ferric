"""Stage-4 exactness anchors (Gamma UHF), run BEFORE any oracle/sweep.

(a) closed shell through UHF == pbc_gamma.rhf, both exxdiv (H2/STO-3G a=4 pure-AFT; tri 4H s+p pure-AFT)
(b) open shell, dense pure-AFT ERI vs RS-GDF in the trivial-aux limit (tri 4H one-s anchor, triplet)
Mutations: K from the total density; Madelung factor 1/2 per spin (RHF's D/2 applied twice).
"""

import sys
import time


import pbc_uhf
from pbc_gamma import Cell, build_integrals, rhf
from pbc_gdf import build_gdf, jk_from_B
from test_prototype import H2_A, H2_ATOMS, SP_BASIS, TRI_A, TRI_ATOMS, _tri_anchor

t0 = time.time()
print("(a) closed-shell UHF vs RHF")
systems = {
    "H2/STO-3G a=4": (Cell(H2_A, H2_ATOMS, "sto-3g"), 2),
    "tri 4H s+p": (Cell(TRI_A, TRI_ATOMS, SP_BASIS), 4),
}
ints = {}
for name, (cell, nel) in systems.items():
    ints[name] = I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    for ex in ("none", "ewald"):
        vm = I["madelung"] if ex == "ewald" else 0.0
        er, epsr, _ = rhf(I["S"], I["h"], I["I"], I["enn"], nel, conv=1e-12, kshift=vm)
        u = pbc_uhf.uhf(
            I["S"], I["h"], I["I"], I["enn"], nel // 2, nel // 2, conv=1e-12, kshift=vm
        )
        print(
            f"  {name:16s} {ex:5s} E_rhf {er:.12f}  dE {u['e'] - er:+.1e}  d eps_a {abs(u['eps_a'] - epsr).max():.1e}"
            f"  d eps_b {abs(u['eps_b'] - epsr).max():.1e}  <S2> {u['s2']:.1e}  it {u['it']}"
        )
        # mutations
        for mname, attr, fn in [
            (
                "K[D_total]",
                "_jk_spin",
                lambda jk, Da, Db: (lambda J, K: (J, K, K))(*jk(Da + Db)),
            ),
            ("v_M/2 per spin", "_madelung_term", lambda S, Ds, v: 0.5 * v * S @ Ds @ S),
        ]:
            if ex == "none" and attr == "_madelung_term":
                continue
            orig = getattr(pbc_uhf, attr)
            setattr(pbc_uhf, attr, fn)
            try:
                m = pbc_uhf.uhf(
                    I["S"],
                    I["h"],
                    I["I"],
                    I["enn"],
                    nel // 2,
                    nel // 2,
                    conv=1e-12,
                    kshift=vm,
                )
                print(f"      MUTANT {mname:15s} dE {m['e'] - er:+.3e}")
            except RuntimeError as exc:
                print(f"      MUTANT {mname:15s} {exc}")
            finally:
                setattr(pbc_uhf, attr, orig)
print(f"  [{time.time() - t0:.0f}s]")

print("(b) open shell: dense AFT vs trivial-aux RS-GDF")
for label, (cell, aux), na, nb in [
    ("tri one-s triplet", _tri_anchor(), 3, 1),
    ("tri one-s singlet", _tri_anchor(), 2, 2),
]:
    ref = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    B = build_gdf(cell, None, auxmol=aux)["B"]
    for ex in ("none", "ewald"):
        vm = ref["madelung"] if ex == "ewald" else 0.0
        ud = pbc_uhf.uhf(
            ref["S"], ref["h"], ref["I"], ref["enn"], na, nb, conv=1e-12, kshift=vm
        )
        ub = pbc_uhf.uhf(
            ref["S"],
            ref["h"],
            None,
            ref["enn"],
            na,
            nb,
            conv=1e-12,
            kshift=vm,
            jk=jk_from_B(B),
        )
        print(
            f"  {label} {ex:5s} E {ud['e']:.12f}  dE(B-dense) {ub['e'] - ud['e']:+.1e}  <S2> {ud['s2']:.10f}"
            f"  d eps_a {abs(ub['eps_a'] - ud['eps_a']).max():.1e}"
        )
    # 7/8 aux classes: sensitivity of (b)
    cell7, aux7 = _tri_anchor(classes=range(7))
    B7 = build_gdf(cell7, None, auxmol=aux7)["B"]
    u7 = pbc_uhf.uhf(
        ref["S"], ref["h"], None, ref["enn"], na, nb, conv=1e-12, jk=jk_from_B(B7)
    )
    ud = pbc_uhf.uhf(ref["S"], ref["h"], ref["I"], ref["enn"], na, nb, conv=1e-12)
    print(f"      aux 7/8 classes: dE {u7['e'] - ud['e']:+.2e}")
    # K[D_total] mutant through B
    orig = pbc_uhf._jk_spin
    pbc_uhf._jk_spin = lambda jk, Da, Db: (lambda J, K: (J, K, K))(*jk(Da + Db))
    try:
        um = pbc_uhf.uhf(
            ref["S"], ref["h"], None, ref["enn"], na, nb, conv=1e-12, jk=jk_from_B(B)
        )
        print(
            f"      MUTANT K[D_total] via B: dE vs correct dense {um['e'] - ud['e']:+.3e}"
        )
    except RuntimeError as exc:
        print("      MUTANT K[D_total]", exc)
    finally:
        pbc_uhf._jk_spin = orig
print(f"done [{time.time() - t0:.0f}s]")
sys.stdout.flush()
