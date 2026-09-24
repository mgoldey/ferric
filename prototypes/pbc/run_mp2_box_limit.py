"""Stage 6 independent oracle: Gamma MP2 of H2 in cubic boxes a -> molecular MP2 (cart, same geometry).

Ewald-split ERI (w = min(1, 8/a), bra images 18 Bohr, ket 18 + 6/w) -- checked against pure-AFT at
a=8 (STO-3G) to 1e-14 and against wider cutoffs / w=0.5 (6-31G) to 4e-16 in E_MP2.

Hypotheses written BEFORE the sweep (see FINDINGS.md Iteration 3):
  physics  : 'shifted' -> residual O(a^-3), coefficient = moment prediction (STO-3G);
             'unshifted' -> residual = sum N/(D + 2 v_M) - sum N/D + O(a^-3), i.e. c1/a with
             c1 = -2 (v_M a) sum N/D^2 (molecular N, D; v_M a = 2.8373 simple cubic).
  artifact : broken transform / missing images -> plateau at a != 0 constant (no convergence);
             Madelung shift with the wrong sign -> 'shifted' converges as 1/a with ~2x c1.
Usage: run_mp2_box_limit.py BASIS a1 a2 ..."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto, scf, mp
from pbc_gamma import Cell, build_integrals, rhf
from pbc_mp2 import gamma_mp2, denominators, dipole_prediction_h2_minimal

ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
VM_A = 2.8372974794806  # simple-cubic v_M * a
basis = sys.argv[1]
edges = [float(x) for x in sys.argv[2:]] or [6, 8, 10, 12, 14, 16, 20, 24, 32, 40]

mol = gto.M(atom=ATOMS, basis=basis, unit="B", cart=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol = 1e-13
e_hf_mol = mf.kernel()
pt = mp.MP2(mf)
e_mp2_mol = pt.kernel()[0]
nocc = 1
Co, Cv = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
ov = mol.ao2mo((Co, Cv, Co, Cv), compact=False).reshape(
    nocc, Cv.shape[1], nocc, Cv.shape[1]
)
eo, ev = mf.mo_energy[:nocc], mf.mo_energy[nocc:]
D = (
    eo[:, None, None, None]
    - ev[None, :, None, None]
    + eo[None, None, :, None]
    - ev[None, None, None, :]
)
N = ov * (2 * ov - ov.transpose(0, 3, 2, 1))
print(
    f"{basis}: E_HF(mol) {e_hf_mol:.12f}  E_MP2(mol) {e_mp2_mol:.12e}  (check sum N/D {np.sum(N / D):.12e})"
)
c1 = -2 * VM_A * np.sum(N / D**2)
c2 = 4 * VM_A**2 * np.sum(N / D**3)
print(f"  unshifted prediction: c1 = {c1:.8f}  c2 = {c2:.8f}")
if mol.nao == 2:
    for a in edges:
        pred, info = dipole_prediction_h2_minimal(mol, mf.mo_coeff, mf.mo_energy, a)
    print(
        f"  shifted prediction (moment model): c3 = {pred * a**3:.8f}  (sigma2 {info['sigma2']:.6f} |d_ia|^2 {info['d_ia'] @ info['d_ia']:.6f} (ia|ia) {info['iaia']:.6f} D {info['D']:.6f})"
    )
print(
    " a      w     v_M            dMP2_shifted          dMP2_unshifted        dE_HF_ewald          d(eps_occ)_ewald    d(eps_vir0)_ewald  wall",
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
        o, v = denominators(eps_n, nocc, vm, conv)
        out[conv] = (
            gamma_mp2(C, np.concatenate([o, v]), nocc, eri=ints["I"])[0] - e_mp2_mol
        )
    dhf = e_n - vm - e_hf_mol  # E_HF(ewald) = E_HF(none) - v_M at Gamma (Iteration 1)
    print(
        f"{a:5.1f} {w:5.3f} {vm:.10f}  {out['shifted']:+.12e}  {out['unshifted']:+.12e}  {dhf:+.12e}  "
        f"{eps_n[0] - vm - eo[0]:+.6e}  {eps_n[1] - ev[0]:+.6e}  {time.time() - t:.0f}s",
        flush=True,
    )
