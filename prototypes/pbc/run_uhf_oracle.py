"""Stage-4 oracle: our Gamma UHF (pure-AFT dense ERI) vs PySCF 2.13 pbc.scf.UHF on AFTDF (mesh 61^3),
exxdiv None and 'ewald'; energies and <S^2>.  PySCF is run twice: its own default guess, and from our
converged densities (so a different-basin outcome is distinguishable from a Fock/energy disagreement)."""

import sys
import time

import numpy as np
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

sys.path.insert(0, ".")
from pbc_gamma import Cell, build_integrals  # noqa: E402
from pbc_uhf import uhf  # noqa: E402
from test_prototype import H2_A, H2_ATOMS, SP_BASIS, TRI_A, TRI_ATOMS  # noqa: E402

SYS = {
    "H atom a=4 (doublet)": (H2_A, [("H", (0.3, 0.2, 0.1))], "sto-3g", 1, 0),
    "H2 a=4 (triplet)": (H2_A, H2_ATOMS, "sto-3g", 2, 0),
    "tri 4H s+p (triplet)": (TRI_A, TRI_ATOMS, SP_BASIS, 3, 1),
}
which = sys.argv[1:] or list(SYS)
for name in which:
    a, atoms, basis, na, nb = SYS[name]
    t = time.time()
    ints = build_integrals(Cell(a, atoms, basis), None, exxdiv="ewald", verbose=False)
    vm = ints["madelung"]
    ours = {}
    for ex in ("none", "ewald"):
        ours[ex] = uhf(
            ints["S"],
            ints["h"],
            ints["I"],
            ints["enn"],
            na,
            nb,
            conv=1e-12,
            kshift=vm if ex == "ewald" else 0.0,
        )
    print(f"{name}: ours ({time.time() - t:.0f}s) v_M={vm:.10f}", flush=True)
    for ex in ("none", "ewald"):
        u = ours[ex]
        print(
            f"   ours {ex:5s} E {u['e']:.12f}  <S2> {u['s2']:.12f}  eps_a[:{na + 1}] {np.round(u['eps_a'][: na + 1], 8)}"
            f"  eps_b[:{nb + 1}] {np.round(u['eps_b'][: nb + 1], 8)}  it {u['it']}"
        )
    print(
        f"   ours ewald - none = {ours['ewald']['e'] - ours['none']['e']:+.12f}  vs -v_M(Na+Nb)/2 = {-vm * (na + nb) / 2:+.12f}"
    )
    pc = pgto.Cell(
        a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=na - nb
    )
    pc.precision = 1e-12
    pc.build()
    for ex in (None, "ewald"):
        key = "none" if ex is None else "ewald"
        for gname in ("pyscf-default", "our-density"):
            t = time.time()
            mf = pscf.UHF(pc, exxdiv=ex)
            mf.with_df = pdf.AFTDF(pc)
            mf.with_df.mesh = [61] * 3
            mf.conv_tol = 1e-12
            dm0 = (
                None
                if gname == "pyscf-default"
                else np.array([ours[key]["Da"], ours[key]["Db"]])
            )
            e = mf.kernel(dm0=dm0)
            s2 = mf.spin_square()[0]
            print(
                f"   PySCF exxdiv={str(ex):5s} guess={gname:13s} E {e:.12f} (d {e - ours[key]['e']:+.1e})  "
                f"<S2> {s2:.12f} (d {s2 - ours[key]['s2']:+.1e})  eps_a0 d {mf.mo_energy[0][0] - ours[key]['eps_a'][0]:+.1e}"
                f"  conv {mf.converged} ({time.time() - t:.0f}s)",
                flush=True,
            )
