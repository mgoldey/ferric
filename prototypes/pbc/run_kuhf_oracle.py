"""Stage 9 oracle: pbc_kuhf k-point UHF (dense pure-AFT kernels, default gcut) vs PySCF 2.13 pbc.scf.KUHF + AFTDF
(mesh 61^3, cell.precision 1e-12, conv 1e-12), exxdiv None and 'ewald'; E per cell and <S^2> (giant determinant,
PySCF convention).  PySCF is run from its default guess AND from our converged densities, so a different-basin
outcome is distinguishable from a Fock/energy disagreement.

Predictions (before the run): E agrees to <= 1e-12 (Iteration 9 k-RHF: 2e-14 on H2); <S^2> equals Sz(Sz+1) of
the giant determinant for these nb = 0 systems (H atom 1x1x2: Sz = 1 -> 2; 1x1x3: 3.75; H2 triplet 1x1x2: Sz = 2 -> 6);
ewald - none = -v_M(n) (Na + Nb)/2 per cell exactly.  nb = 0 everywhere, so these pins cannot see a K[D_total]
mutant (the supercell anchor on tri does).
Usage: python3 run_kuhf_oracle.py [h1x2|h1x3|h2t]"""

import sys
import time

import numpy as np
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

sys.path.insert(0, ".")
import pbc_kpts as PK  # noqa: E402
import pbc_kuhf as KU  # noqa: E402
from pbc_gamma import Cell  # noqa: E402
from test_prototype import H2_A, H2_ATOMS  # noqa: E402

H_ATOM = [("H", (0.3, 0.2, 0.1))]
SYS = {
    "h1x2": (H_ATOM, (1, 1, 2), 1, 0),
    "h1x3": (H_ATOM, (1, 1, 3), 1, 0),
    "h2t": (H2_ATOMS, (1, 1, 2), 2, 0),
}

for name in sys.argv[1:] or list(SYS):
    atoms, n, na, nb = SYS[name]
    t = time.time()
    kb = PK.build_k(Cell(H2_A, atoms, "sto-3g"), n, exxdiv="ewald", verbose=False)
    ours = {"none": KU.kuhf(kb, na, nb, conv=1e-12, kshift=0.0)}
    ours["ewald"] = KU.kuhf(
        kb, na, nb, conv=1e-12, guess=(ours["none"]["Da"], ours["none"]["Db"])
    )
    vm = kb["madelung"]
    print(
        f"{name} {n}: ours ({time.time() - t:.0f}s) v_M {vm:.12f}; ewald-none {ours['ewald']['e'] - ours['none']['e']:+.3e} "
        f"vs -v_M N/2 {-vm * (na + nb) / 2:+.3e}",
        flush=True,
    )
    pc = pgto.Cell(
        a=H2_A, atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0, spin=na - nb
    )
    pc.precision = 1e-12
    pc.max_memory = 1200
    pc.build()
    kp = pc.make_kpts(list(n))
    for ex in (None, "ewald"):
        key = "none" if ex is None else "ewald"
        u = ours[key]
        print(
            f"   ours {key:5s} E/cell {u['e']:.12f} <S2> {u['s2']:.12f} nocc_a/k {u['nocc_a_k']} gap_a {u['gap_a']:.4f}"
        )
        for gname in ("pyscf-default", "our-density"):
            t = time.time()
            mf = pscf.KUHF(pc, kp, exxdiv=ex)
            mf.with_df = pdf.AFTDF(pc, kp)
            mf.with_df.mesh = [61] * 3
            mf.conv_tol = 1e-12
            mf.nelec = (
                na * kb["Nk"],
                nb * kb["Nk"],
            )  # PySCF cell.spin is Na - Nb over the WHOLE mesh
            dm0 = None if gname == "pyscf-default" else np.array([u["Da"], u["Db"]])
            e = mf.kernel(dm0=dm0)
            s2 = mf.spin_square()[0]
            de = max(
                abs(np.asarray(u["eps_a"][k]) - mf.mo_energy[0][k]).max()
                for k in range(kb["Nk"])
            )
            print(
                f"   PySCF exxdiv={str(ex):5s} guess={gname:13s} E {e:.12f} (d {e - u['e']:+.1e}) <S2> {s2:.12f} "
                f"(d {s2 - u['s2']:+.1e}) max|d eps_a| {de:.1e} conv {mf.converged} ({time.time() - t:.0f}s)",
                flush=True,
            )
