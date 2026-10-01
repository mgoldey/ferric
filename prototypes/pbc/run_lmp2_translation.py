"""Stage 8b translation-equivalence check (Gamma LMP2).

Molecules related by a primitive lattice translation inside the supercell must get equivalent
LMOs (T phi_i == +-phi_pi(i), pi a bijection), equivalent VV-HV virtuals, equal spreads and equal
pair energies e_ij == e_pi(i)pi(j).  At Gamma the translation is an exact AO permutation, so the
deviation is measured directly (translation_map: 1 - max |<phi_j|S|T phi_i>|).

Localisation variants:
  berghold          periodic Resta/Berghold functional (the method)
  berghold seed=s   same, from a random orthogonal start (maximum must not depend on the start)
  boys-molecular    MUTANT: L=0 molecular <mu|r|nu> of the supercell's AOs (not periodic)
Geometries: straight supercell, and 'wrapped' = the last atom moved by -a_sc along z so the last
molecule straddles the supercell boundary (physics-neutral at Gamma: all Gamma matrices are
identical; only a non-periodic operator can see it).
Usage: python3 run_lmp2_translation.py [NxMxK ...]
"""

import sys
import time

import numpy as np

import pbc_lmp2 as L
from pbc_supercell import Supercell, build_supercell
from run_lmp2_anchor import A0, AUX, H2_ATOMS


def run(n, wrapped=False, eps=1e-4):
    nat = 2 * int(np.prod(n))
    wrap = (lambda i, r: (0, 0, -1) if i == nat - 1 else (0, 0, 0)) if wrapped else None
    sc = Supercell(A0, H2_ATOMS, "6-31g", n, wrap=wrap)
    d = build_supercell(sc, AUX, w=0.5)
    scf = L.gamma_scf(d, 2 * sc.R)
    rows = []
    for loc, seed in (
        ("berghold", None),
        ("berghold", 1),
        ("berghold", 2),
        ("boys-molecular", None),
    ):
        p = L.prepare(sc, d, scf, loc=loc, seed=seed)
        r0 = L.solve(p, 0.0)
        r1 = L.solve(p, eps)
        te0 = L.translation_equivalence(sc, d, p, r0["epair"])
        te1 = L.translation_equivalence(sc, d, p, r1["epair"])
        rows.append((loc, seed, p, r0, r1))
        print(
            f"  {str(n):10s} {'wrapped' if wrapped else 'straight':8s} {loc:15s} seed={seed!s:4s} f={p['loc']['f']:.8f} "
            f"|grad|={p['loc']['grad']:.1e}  occ_dev {te0['occ_dev']:.1e} vir_dev {te0['vir_dev']:.1e} "
            f"spread_dev {te0['spread_dev']:.1e} bij {te0['bijective']}  pair_dev(eps=0) {te0['pair_dev']:.1e} "
            f"pair_dev(eps={eps:g}) {te1['pair_dev']:.1e}  E(eps={eps:g}) {r1['e']:.12e} pairs {r1['pairs_kept']}"
            f"  partners min/max {r1['partners'].min()}/{r1['partners'].max()}",
            flush=True,
        )
    e_b = [r[4]["e"] for r in rows[:3]]
    print(
        f"  -> E(eps={eps:g}) spread over Berghold starts {max(e_b) - min(e_b):.1e}; molecular-Boys minus Berghold "
        f"{rows[3][4]['e'] - rows[0][4]['e']:+.2e}; E_HF {scf['e']:.12f}"
    )
    return rows


if __name__ == "__main__":
    sizes = [tuple(int(x) for x in s.split("x")) for s in sys.argv[1:]] or [
        (1, 1, 6),
        (2, 2, 2),
        (3, 3, 3),
    ]
    for n in sizes:
        t0 = time.time()
        for wrapped in (False, True):
            run(n, wrapped)
        print(f"  [{time.time() - t0:.0f} s]", flush=True)
