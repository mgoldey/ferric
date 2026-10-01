"""Stage 6 aux-fitting error of Gamma MP2: RS-GDF B (ferric bundled aux, spherical) vs the exact
dense pure-AFT (ia|jb), same cell/basis.  Two numbers per aux:
  'same C'  : fitted (ia|jb) with the EXACT SCF orbitals/eps  (pure MP2 fitting error)
  'own SCF' : SCF also run on the fitted B (the full pipeline a Rust port would run)
Also the G=0 one-sided mutant (metric only) with a REAL aux set: is ov-MP2 blind to it here too?"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto
import pbc_gdf
from pbc_gamma import Cell, build_integrals, rhf
from pbc_gdf import build_gdf, ferric_basis, even_tempered, jk_from_B
from pbc_mp2 import gamma_mp2, denominators

SP = {
    "H": gto.parse("""
H S
  3.42525091  0.15432897
  0.62391373  0.53532814
  0.16885540  0.44463454
H P
  0.8         1.0
""")
}
SYS = {
    "H2/STO-3G a=4": (
        np.eye(3) * 4.0,
        [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))],
        "sto-3g",
    ),
    "tri4H s+p": (
        np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]),
        [
            ("H", (0.1, 0.2, 0.3)),
            ("H", (0.1, 0.2, 1.7)),
            ("H", (2.4, 2.5, 2.2)),
            ("H", (3.6, 2.9, 2.6)),
        ],
        SP,
    ),
}
AUX = {
    "cc-pvdz-ri": lambda: ferric_basis("cc-pvdz-ri", ["H"]),
    "def2-universal-jkfit": lambda: ferric_basis("def2-universal-jkfit", ["H"]),
    "def2-svp-rifit": lambda: ferric_basis("def2-svp-rifit", ["H"]),
    "ET l<=2 b=2.2": lambda: even_tempered(
        ["H"], 2, 0.1, 2.2, int(np.ceil(np.log(40 / 0.1) / np.log(2.2))) + 1
    ),
}
if __name__ == "__main__":
    for name in sys.argv[1:] or SYS:
        a, atoms, basis = SYS[name]
        nel = len(atoms)
        nocc = nel // 2
        cell = Cell(a, atoms, basis)
        ints = build_integrals(cell, None, exxdiv="ewald", verbose=False)
        vm = ints["madelung"]
        e0, eps0, _, C0 = rhf(
            ints["S"],
            ints["h"],
            ints["I"],
            ints["enn"],
            nel,
            conv=1e-12,
            return_mo=True,
        )
        exact = {
            c: gamma_mp2(
                C0, np.concatenate(denominators(eps0, nocc, vm, c)), nocc, eri=ints["I"]
            )[0]
            for c in ("shifted", "unshifted")
        }
        print(
            f"{name}: exact MP2 shifted {exact['shifted']:.10e} unshifted {exact['unshifted']:.10e}",
            flush=True,
        )
        for an, fn in AUX.items():
            t = time.time()
            g = build_gdf(cell, fn())
            B = g["B"]
            e1, eps1, _, C1 = rhf(
                ints["S"],
                ints["h"],
                None,
                ints["enn"],
                nel,
                conv=1e-12,
                jk=jk_from_B(B),
                return_mo=True,
            )
            row = []
            for c in ("shifted", "unshifted"):
                same = (
                    gamma_mp2(
                        C0, np.concatenate(denominators(eps0, nocc, vm, c)), nocc, B=B
                    )[0]
                    - exact[c]
                )
                own = (
                    gamma_mp2(
                        C1, np.concatenate(denominators(eps1, nocc, vm, c)), nocc, B=B
                    )[0]
                    - exact[c]
                )
                row.append(
                    f"{c}: same-C {same:+.3e} own-SCF {own:+.3e} (rel {same / exact[c]:+.1e})"
                )
            print(
                f"  {an:22s} naux {g['info']['naux_kept']}/{g['info']['naux']}  dE_HF {e1 - e0:+.3e}  "
                + "  ".join(row)
                + f"  ({time.time() - t:.0f}s)",
                flush=True,
            )
            if an == "cc-pvdz-ri":
                orig = pbc_gdf._subtract_g0
                pbc_gdf._subtract_g0 = lambda J2, J3, S, q, c0: (
                    J2 - c0 * np.outer(q, q),
                    J3,
                )
                Bm = build_gdf(cell, fn())["B"]
                pbc_gdf._subtract_g0 = orig
                dm = (
                    gamma_mp2(
                        C0,
                        np.concatenate(denominators(eps0, nocc, vm, "shifted")),
                        nocc,
                        B=Bm,
                    )[0]
                    - exact["shifted"]
                )
                print(
                    f"  {'':22s} MUTANT G=0 from J2 only (cc-pvdz-ri): dMP2 shifted {dm:+.3e}",
                    flush=True,
                )
