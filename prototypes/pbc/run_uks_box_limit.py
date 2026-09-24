"""Iteration 10 box limit (independent of PySCF pbc): Gamma UKS of an H atom (doublet) in a cubic box a vs
the MOLECULAR UKS on the identical atom-centred grid (75x302, SSF; one atom -> molecular weight 1).
Usage: python3 run_uks_box_limit.py BASIS a1 a2 ...    (BASIS sto-3g | 6-31g; a=0 prints predictions only)

PREDICTIONS (written before the sweep):
  physics, PBE0 ewald: dE = c3/a^3 + ..., c3 = hyb x the UHF coefficient evaluated with the UKS orbital:
      c3 = -(2pi/3) hyb Omega_a  (|d| = 0; semilocal XC is local, so it adds no a^-3).
      STO-3G: one AO, orbital fixed -> Omega_a = 1.948573 (Iteration 6) -> c3 = 0.25 x -4.081081 = -1.020270.
      6-31G: Omega_a from the PBE0 orbital (differs from HF); cross-checked by the relaxed r2-kernel FD.
      Next term: the density is spherical and the kernel beyond r^2 is l>=4 cubic harmonics -> NO a^-5.
      Orbital relaxation is second order in k -> a^-6, and for 6-31G only (STO-3G cannot relax).
      Unlike UHF (Hartree == self-exchange for one electron), the hybrid leaves (1-hyb) of the harmonic
      self-interaction uncancelled, so 6-31G/PBE0 should show a visible a^-6 tail where UHF showed none.
  physics, LDA/PBE ewald: c3 = 0 and no relaxation (J + en harmonic potentials cancel for a neutral atom-centred
      density) -> dE is grid/image tails only (control floor).
  identity, exxdiv none: dE_none - dE_ewald = hyb v_M N/2 exactly (1/a).
  artifacts: Madelung on full K -> extra -(1-hyb) v_M N/2 -> 1/a (coefficient -(1-hyb) 2.8373/2);
      UHF coefficient (hyb dropped from the prediction or K) -> c3 4x; hyb/2 per spin -> O(1) plateau."""

import sys
import time

import numpy as np
from pyscf import dft, gto
from pyscf.dft import radi

sys.path.insert(0, ".")
import pbc_dft as pd  # noqa: E402
import pbc_uks as U  # noqa: E402
from pbc_gamma import Cell, build_integrals  # noqa: E402

basis = sys.argv[1]
edges = [float(x) for x in sys.argv[2:]]
atoms = [("H", (0.3, 0.2, 0.1))]
NR, NA = 75, 302
mol = gto.M(atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=1)
cm, wm = pd.molecular_grid_reference(mol, NR, NA, scheme="ssf")
mg = U.MolGrid(mol, cm, wm)
Sm = mol.intor("int1e_ovlp_cart")
hm = mol.intor("int1e_kin_cart") + mol.intor("int1e_nuc_cart")
jkm = pd.dense_jk(mol.intor("int2e_cart"))
XCS = ("LDA,VWN", "PBE", "PBE0")
ref = {}
for xc in XCS:
    u = U.uks(Sm, hm, jkm, 0.0, 1, 0, mg, xc, conv=1e-13)
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.atom_grid = (NR, NA)
    mf.grids.prune = None
    mf.grids.radi_method = radi.treutler
    mf.grids.becke_scheme = dft.gen_grid.stratmann
    mf.grids.cutoff = 1e-100
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1e-13
    mf.verbose = 0
    e_py = mf.kernel()
    c3, p = U.uks_c3_closed_form(mol, u["Ca"], u["Cb"], 1, 0, u["hyb"])
    c3_r2 = U.uks_r2_kernel_c3(mol, 1, 0, mg, xc, guess=(u["Da"], u["Db"]))
    c6 = U.uks_r2_kernel_c3(
        mol, 1, 0, mg, xc, h=1e-3, guess=(u["Da"], u["Db"]), second=True
    )[1]
    ref[xc] = (u, c3, c6)
    print(
        f"H/{basis} {xc:8s}: E_mol ours {u['e']:.12f}  PySCF mol UKS same grid {e_py - u['e']:+.1e}  "
        f"Omega_a {p['omega_a']:.6f}  PREDICTED c3 {c3:.6f}  r2-kernel FD {c3_r2:.6f}  relaxation c6 {c6:+.4f}",
        flush=True,
    )
print(f"  PREDICTED none-ewald = hyb x {2.8372974794806 / 2:.6f}/a")
for edge in edges:
    t = time.time()
    w = min(1.0, 8.0 / edge)
    cell = Cell(np.eye(3) * edge, atoms, basis)
    ints = build_integrals(
        cell, w, rcut_bra=18.0, rcut_2e=18.0 + 6.0 / w, exxdiv="ewald", verbose=False
    )
    vm, S, h, enn = ints["madelung"], ints["S"], ints["h"], ints["enn"]
    jk = pd.dense_jk(ints["I"])
    g = pd.PeriodicGrid(cell, NR, NA, D=0.9 * edge, scheme="ssf")
    for xc in XCS:
        u0, c3, c6 = ref[xc]
        r = {
            ex: U.uks(
                S,
                h,
                jk,
                enn,
                1,
                0,
                g,
                xc,
                kshift=vm if ex == "ewald" else 0.0,
                conv=1e-13,
                guess=(u0["Da"], u0["Db"]),
            )
            for ex in ("none", "ewald")
        }
        U._MUTANT = "madelung_full_k"
        try:
            em = U.uks(
                S,
                h,
                jk,
                enn,
                1,
                0,
                g,
                xc,
                kshift=vm,
                conv=1e-13,
                guess=(u0["Da"], u0["Db"]),
            )["e"]
        finally:
            U._MUTANT = None
        de, dn = r["ewald"]["e"] - u0["e"], r["none"]["e"] - u0["e"]
        print(
            f"  a={edge:5.1f} {xc:8s} dE_ewald {de:+.10e}  dE*a^3 {de * edge**3:+.6f} (pred c3 {c3:+.6f}, c3+c6/a^3 {c3 + c6 / edge**3:+.6f})  "
            f"(none-ewald)-hyb vM/2 {dn - de - u0['hyb'] * vm / 2:+.1e}  MUTANT full-K dE {em - u0['e']:+.3e}  "
            f"({time.time() - t:.0f}s)",
            flush=True,
        )
