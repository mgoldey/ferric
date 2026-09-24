"""Stage 7 independent oracle: Gamma dRPA of H2 in cubic boxes a -> molecular dRPA (exact ERI, plasmon;
same geometry, cart).  ERI by the same w-scaled Ewald split as run_mp2_box_limit.py.

Hypotheses written BEFORE the sweep (predictions from MOLECULAR quantities only):
  physics, shifted  : residual = c3/a^3 + O(a^-5); for H2/STO-3G (nov = 1) the frozen-orbital lattice
                      model gives c3 = a^3 [dE/dD k(sigma^2+|d|^2) - dE/dK k|d|^2]  (pbc_rpa docstring).
  physics, unshifted: residual - shifted residual = E_mol(D - v_M) - E_mol(D) (all e_ia lowered by v_M),
                      -> c1/a with c1 = -(v_M a) dE_mol/dD_uniform (from a molecular finite difference).
  artifact (a) broken transform / missing images   -> plateau at a != 0 (no convergence);
  artifact (b) Madelung sign flipped               -> 'shifted' converges as 1/a with c1 of the OPPOSITE sign
                                                      (D + v_M instead of D - v_M);
  artifact (c) spin factor / 1/2pi normalisation   -> residual -> a nonzero FRACTION of E_mol (plateau).
Exponents (3 vs 1 vs 0) and coefficients are predicted, so the outcomes are distinguishable.
Usage: run_rpa_box_limit.py BASIS a1 a2 ..."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto, scf
from pbc_gamma import Cell, build_integrals, rhf
from pbc_mp2 import denominators, gamma_mp2
from pbc_rpa import (
    gamma_drpa,
    drpa_plasmon,
    direct_mp2,
    drpa_moment_prediction_h2_minimal,
)

ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
VM_A = 2.8372974794806
basis = sys.argv[1]
edges = [float(x) for x in sys.argv[2:]] or [6, 8, 10, 12, 14, 16, 20, 24, 32, 40]

mol = gto.M(atom=ATOMS, basis=basis, unit="B", cart=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-13
mf.kernel()
nocc = 1
Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
ov = mol.ao2mo((Co, Cv, Co, Cv), compact=False).reshape(
    nocc, Cv.shape[1], nocc, Cv.shape[1]
)
eo, ev = mf.mo_energy[:nocc], mf.mo_energy[nocc:]
E_mol = drpa_plasmon(ov, eo, ev)
from pyscf import mp

e_mp2_mol = mp.MP2(mf).kernel()[0]
h = 1e-5
dEdD = (drpa_plasmon(ov, eo - h, ev) - drpa_plasmon(ov, eo + h, ev)) / (
    2 * h
)  # d/d(uniform e_ia shift)
c1 = -VM_A * dEdD
print(
    f"{basis}: E_dRPA(mol) {E_mol:.12e}  direct-MP2(mol) {direct_mp2(ov, eo, ev):.12e}  MP2(mol) {e_mp2_mol:.12e}"
)
print(f"  unshifted prediction: c1 = -(v_M a) dE/dD = {c1:.8f}  (dE/dD {dEdD:.8f})")
if mol.nao == 2:
    p, info = drpa_moment_prediction_h2_minimal(mol, mf.mo_coeff, mf.mo_energy, 1.0)
    print(
        f"  shifted prediction (moment model): c3 = {p:.8f}  (sigma2 {info['sigma2']:.6f} |d|^2 {info['d_ia'] @ info['d_ia']:.6f} "
        f"K {info['K']:.6f} D {info['D']:.6f} dE/dD {info['dEdD']:.6f} dE/dK {info['dEdK']:.6f})"
    )
print(
    " a      w     v_M           dRPA_shifted          dRPA_unshifted        unsh-sh  vs shift-fn    dMP2_shifted(same run)  wall",
    flush=True,
)
for a in edges:
    t = time.time()
    w = min(1.0, 8.0 / a)
    cell = Cell(np.eye(3) * a, ATOMS, basis)
    ints = build_integrals(
        cell, w, rcut_bra=18.0, rcut_2e=18.0 + 6.0 / w, exxdiv="ewald", verbose=False
    )
    vm = ints["madelung"]
    e_n, eps_n, _, C = rhf(
        ints["S"], ints["h"], ints["I"], ints["enn"], 2, conv=1e-13, return_mo=True
    )
    out = {}
    for conv in ("shifted", "unshifted"):
        e = np.concatenate(denominators(eps_n, nocc, vm, conv))
        out[conv] = gamma_drpa(C, e, nocc, eri=ints["I"], method="plasmon") - E_mol
    eq = (
        gamma_drpa(
            C,
            np.concatenate(denominators(eps_n, nocc, vm, "shifted")),
            nocc,
            eri=ints["I"],
        )
        - E_mol
    )  # quad route
    shift_fn = drpa_plasmon(ov, eo + vm, ev) - E_mol
    mp2s = (
        gamma_mp2(
            C,
            np.concatenate(denominators(eps_n, nocc, vm, "shifted")),
            nocc,
            eri=ints["I"],
        )[0]
        - e_mp2_mol
    )
    print(
        f"{a:5.1f} {w:5.3f} {vm:.10f}  {out['shifted']:+.12e}  {out['unshifted']:+.12e}  "
        f"{(out['unshifted'] - out['shifted']) / shift_fn:.6f}  quad-plasmon {eq - out['shifted']:+.1e}  {mp2s:+.6e}  {time.time() - t:.0f}s",
        flush=True,
    )
