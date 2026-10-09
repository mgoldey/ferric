"""Stage 6 exactness anchor: MP2 from the dense pure-AFT (ia|jb) vs MP2 from the RS-GDF B in the
trivial-aux limit (aux = every periodic pair product).  Two cells: H2 (nocc=nvir=1, blind to the
exchange term) and triclinic 4H (nocc=nvir=2).  Plus mutations that must break it."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto
from pbc_gamma import Cell, build_integrals, rhf
import pbc_gdf
from pbc_gdf import build_gdf
from pbc_mp2 import gamma_mp2, ovov_from_B, ovov_from_eri, mp2_energy

AL = 0.5
CELLS = {
    "H2": (np.eye(3) * 4.0, [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]),
    "tri4H": (
        np.array([[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]),
        [
            ("H", (0.1, 0.2, 0.3)),
            ("H", (0.1, 0.2, 1.7)),
            ("H", (2.4, 2.5, 2.2)),
            ("H", (3.6, 2.9, 2.6)),
        ],
    ),
}


def anchor(name, classes=range(8)):
    a, atoms = CELLS[name]
    cell = Cell(a, atoms, {"H": [[0, [AL, 1.0]]]})
    n = len(atoms)
    cen = [
        0.5 * (cell.R[i] + cell.R[j] + np.array(h) @ cell.a)
        for i in range(n)
        for j in range(i, n)
        for k, h in enumerate(np.ndindex(2, 2, 2))
        if k in classes
    ]
    aux = gto.M(
        atom=[("X", c) for c in cen],
        basis={"X": [[0, [2 * AL, 1.0]]]},
        unit="B",
        cart=True,
        verbose=0,
    )
    return cell, aux


if __name__ == "__main__":
    for name in CELLS:
        cell, aux = anchor(name)
        nel = 2 * (len(cell.atoms) // 2)
        nocc = nel // 2
        t = time.time()
        ref = build_integrals(cell, None, exxdiv="ewald", verbose=False)
        g = build_gdf(cell, None, auxmol=aux)
        B = g["B"]
        print(
            f"{name}: build {time.time() - t:.1f}s naux={g['info']['naux']} kept={g['info']['naux_kept']} "
            f"metric min {g['info']['metric_min']:.2e}",
            flush=True,
        )
        for ex, ks in (("none", 0.0), ("ewald", ref["madelung"])):
            e0, eps0, _, C0 = rhf(
                ref["S"],
                ref["h"],
                ref["I"],
                ref["enn"],
                nel,
                conv=1e-12,
                kshift=ks,
                return_mo=True,
            )
            from pbc_gdf import jk_from_B

            e1, eps1, _, C1 = rhf(
                ref["S"],
                ref["h"],
                None,
                ref["enn"],
                nel,
                conv=1e-12,
                kshift=ks,
                jk=jk_from_B(B),
                return_mo=True,
            )
            m_exact = gamma_mp2(C0, eps0, nocc, eri=ref["I"])[0]
            m_fit_sameC = gamma_mp2(C0, eps0, nocc, B=B)[0]
            m_fit = gamma_mp2(C1, eps1, nocc, B=B)[0]
            print(
                f"  exxdiv={ex}: E_HF {e0:.12f} dE_HF {e1 - e0:.1e}  E_MP2 exact {m_exact:.12e}  "
                f"B(same C) {m_fit_sameC - m_exact:.1e}  B(own SCF) {m_fit - m_exact:.1e}",
                flush=True,
            )
        # mutations (ewald, exact SCF orbitals)
        e0, eps0, _, C = rhf(
            ref["S"],
            ref["h"],
            ref["I"],
            ref["enn"],
            nel,
            conv=1e-12,
            kshift=ref["madelung"],
            return_mo=True,
        )
        Co, Cv = C[:, :nocc], C[:, nocc:]
        eo, ev = eps0[:nocc], eps0[nocc:]
        m_exact = mp2_energy(ovov_from_eri(ref["I"], Co, Cv), eo, ev)[0]
        ov = ovov_from_B(B, Co, Cv)
        muts = {
            "exchange dropped (2J-J -> J)": np.einsum(
                "iajb,iajb->",
                ov
                / (
                    eo[:, None, None, None]
                    - ev[None, :, None, None]
                    + eo[None, None, :, None]
                    - ev[None, None, None, :]
                ),
                ov,
            ),
            "exchange transpose (0,1,2,3)->(2,3,0,1) [real: identical]": mp2_energy(
                ov, eo, ev
            )[0]
            if False
            else None,
            "Cv used for both ov indices": mp2_energy(
                np.einsum(
                    "Pia,Pjb->iajb", *(2 * [np.einsum("Pmn,mi,na->Pia", B, Cv, Cv)])
                ),
                eo,
                ev,
            )[0]
            if nocc == Cv.shape[1]
            else None,
        }
        for k, v in muts.items():
            if v is not None:
                print(f"  MUTANT {k}: dE {v - m_exact:+.2e}")
        # aux incomplete (7/8 classes) and G=0 mutants on the B side
        for label, kw in (("aux 7/8 classes", dict(classes=range(7))),):
            c2, a2 = anchor(name, **kw)
            B2 = build_gdf(c2, None, auxmol=a2)["B"]
            print(
                f"  {label}: dE_MP2 {mp2_energy(ovov_from_B(B2, Co, Cv), eo, ev)[0] - m_exact:+.2e}"
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
                f"  MUTANT {label}: dE_MP2 {mp2_energy(ovov_from_B(Bm, Co, Cv), eo, ev)[0] - m_exact:+.2e}  "
                f"max|dI| {abs(np.einsum('Pmn,Pls->mnls', Bm, Bm) - ref['I']).max():.1e}",
                flush=True,
            )
