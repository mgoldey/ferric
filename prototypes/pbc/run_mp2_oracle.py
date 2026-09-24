"""Stage 6 oracle: our Gamma MP2 (pure-AFT dense ERI) vs PySCF pbc.mp.RMP2 on AFTDF, exxdiv None and
'ewald', plus the MP2 that PySCF pbc.cc.RCCSD builds from ITS eris (ao2mo rebuilds the Fock with
exxdiv=None, then _adjust_occ(eps, nocc, -madelung) unconditionally): mycc.init_amps(eris)[0].
(ccsd(mbpt2=True) is NOT that: it calls RMP2(mf) with mf.mo_energy.)"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto
from pyscf.pbc import gto as pgto, scf as pscf, df as pdf, mp as pmp, cc as pcc
from pbc_gamma import Cell, build_integrals, rhf
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
which = sys.argv[1:] or list(SYS)
for name in which:
    a, atoms, basis = SYS[name]
    nel = len(atoms)
    nocc = nel // 2
    t = time.time()
    cell = Cell(a, atoms, basis)
    ints = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    vm = ints["madelung"]
    e_n, eps_n, _, C = rhf(
        ints["S"], ints["h"], ints["I"], ints["enn"], nel, conv=1e-12, return_mo=True
    )
    e_e, eps_e, _, Ce = rhf(
        ints["S"],
        ints["h"],
        ints["I"],
        ints["enn"],
        nel,
        conv=1e-12,
        kshift=vm,
        return_mo=True,
    )
    ours = {}
    for conv in ("unshifted", "shifted"):
        eo, ev = denominators(eps_n, nocc, vm, conv)
        ours[conv] = gamma_mp2(C, np.concatenate([eo, ev]), nocc, eri=ints["I"])[0]
    ours["ewald-HF eps"] = gamma_mp2(Ce, eps_e, nocc, eri=ints["I"])[0]
    print(
        f"{name}: ours ({time.time() - t:.0f}s) v_M={vm:.10f}  E_HF none {e_n:.12f} ewald {e_e:.12f}\n"
        f"   MP2 unshifted {ours['unshifted']:.12e}  shifted {ours['shifted']:.12e}  "
        f"(ewald-SCF eps & C directly: {ours['ewald-HF eps']:.12e})",
        flush=True,
    )
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    pc.build()
    for ex in (None, "ewald"):
        t = time.time()
        mf = pscf.RHF(pc, exxdiv=ex)
        mf.with_df = pdf.AFTDF(pc)
        mf.with_df.mesh = [61] * 3
        mf.conv_tol = 1e-12
        ehf = mf.kernel()
        emp = pmp.RMP2(mf).kernel()[0]
        mycc = pcc.RCCSD(mf)
        mycc.conv_tol = 1e-12
        ecc_mp2 = mycc.init_amps(mycc.ao2mo())[0]
        ref = ours["unshifted" if ex is None else "shifted"]
        print(
            f"   PySCF exxdiv={ex}: E_HF {ehf:.12f}  RMP2 {emp:.12e} (d vs ours-{'unshifted' if ex is None else 'shifted'} "
            f"{emp - ref:+.1e})  RCCSD-eris MP2 {ecc_mp2:.12e} (d vs ours-shifted {ecc_mp2 - ours['shifted']:+.1e})  "
            f"({time.time() - t:.0f}s)",
            flush=True,
        )
