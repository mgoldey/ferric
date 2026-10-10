"""Decoupled-omega terfc integral check against the independent k-space construction.

Compares ferric's AO (P|terfc|mu nu) and (P|terfc|Q) for water/cc-pVDZ +
cc-pVDZ-RI at r0 = 1 A and r0*omega in {1/sqrt2, 4, 8, 16} with the Fourier
generator of gen_terfc_integrals.py (PySCF ft_ao + Lebedev/Gauss-Legendre
k-space quadrature; shares no MD recursion, table or series with ferric).
The k-space quadrature itself is validated at EACH omega: the erf kernel
(same w) must reproduce PySCF's analytic range-separated integrals, and the
terf tensor must be stable between the last two grid rungs.

  1. cargo test -p ferric-integrals --release --test terfc_omega_dump \
         -- --ignored   (FERRIC_DUMP_DIR=<dir>, FERRIC_TERF_TABLE_DIR set)
  2. ~/qc/ferric/.venv/bin/python scripts/validation/terfc_omega_fourier_check.py <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_properties as gp  # noqa: E402
import gen_terfc_integrals as g  # noqa: E402

LADDER = ((60, 1202), (80, 2030), (120, 3470))


def main(dump: str) -> int:
    import numpy as np
    from pyscf import gto

    symbols, coords = common.read_xyz(Path(__file__).resolve().parents[2] / "testdata/molecules/water.xyz")
    mol = common.build_pyscf_mol(Path(__file__).resolve().parents[2] / "testdata/molecules/water.xyz", "cc-pvdz")
    auxb, _ = common.pyscf_basis("cc-pvdz-ri", symbols)
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=auxb,
        cart=False,
        verbose=0,
    )
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    perm = np.asarray(gp.ferric_ao_permutation(mol, "cc-pvdz", symbols))
    aperm = np.asarray(gp.ferric_ao_permutation(auxmol, "cc-pvdz-ri", symbols))
    lobs = g.ferric_l_list("cc-pvdz", symbols)
    laux = g.ferric_l_list("cc-pvdz-ri", symbols)
    c3, c2 = g.pyscf_3c2c(mol, auxmol)
    c3f = c3[np.ix_(aperm, perm, perm)]
    c2f = c2[np.ix_(aperm, aperm)]
    r0 = g.r0_bohr(1.0)
    print(f"nao={nao} naux={naux} r0={r0:.6f} Bohr")
    worst_all = 0.0
    for tag, r0w in [("0.7071", 0.5**0.5), ("4", 4.0), ("8", 8.0), ("16", 16.0)]:
        w = r0w / r0
        f3 = np.fromfile(f"{dump}/e3_{tag}.bin", dtype="<f8").reshape(naux, nao, nao)
        f2 = np.fromfile(f"{dump}/e2_{tag}.bin", dtype="<f8").reshape(naux, naux)
        e3, e2 = g.pyscf_3c2c(mol, auxmol, omega=w)
        kernels = [g.g_erf(w), g.g_terf(r0, w)]
        prev = None
        for nrad, nang in LADDER:
            gv, wt, kk = g.kgrid(nrad, nang, g.KMAX_FACTOR * w)
            i3, i2 = g.kspace_tensors(mol, auxmol, gv, wt, kk, kernels)
            anchor3 = g.maxabs(i3[0] - e3)
            anchor2 = g.maxabs(i2[0] - e2)
            t3 = c3 - i3[1]
            t2 = c2 - i2[1]
            t3f = t3[np.ix_(aperm, perm, perm)]
            t2f = t2[np.ix_(aperm, aperm)]
            d3 = np.abs(f3 - t3f)
            d2 = np.abs(f2 - t2f)
            selfchg = None if prev is None else g.maxabs(i3[1] - prev)
            prev = i3[1]
            print(
                f"r0w={tag:>6} grid=({nrad},{nang}) erf-anchor 3c {anchor3:.1e} 2c "
                f"{anchor2:.1e} | self-change(3c terf) "
                f"{'-' if selfchg is None else f'{selfchg:.1e}'} | ferric-vs-kspace 3c max "
                f"{d3.max():.2e} (rel to max|I| {d3.max() / np.abs(t3f).max():.1e}) 2c max "
                f"{d2.max():.2e}"
            )
        # worst class at the production (last) rung
        worst = {}
        for p in range(naux):
            for m in range(nao):
                for n in range(m + 1):
                    lm, ln = sorted((lobs[m], lobs[n]), reverse=True)
                    k = g.class_label((laux[p], lm, ln))
                    worst[k] = max(worst.get(k, 0.0), d3[p, m, n])
        top = sorted(worst.items(), key=lambda kv: -kv[1])[:4]
        print("   worst 3c classes:", ", ".join(f"{k}:{v:.1e}" for k, v in top))
        worst_all = max(worst_all, d3.max(), d2.max())
        print(
            f"   min eig of ferric V_terfc(2c) = {np.linalg.eigvalsh(0.5 * (f2 + f2.T)).min():.3e}"
            f"   (kspace {np.linalg.eigvalsh(0.5 * (t2f + t2f.T)).min():.3e})"
        )
    print(f"OVERALL worst |ferric - kspace| = {worst_all:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
