"""Iteration 19 PySCF oracle for the Gamma stress tensor (pbc_stress.py).  Usage: python3 run_stress_oracle.py {pieces|rks|uks}

PySCF 2.13 HAS an all-electron-capable Gamma stress for semilocal KS: pyscf.pbc.grad.rks_stress / uks_stress (FFTDF +
UniformGrids only, no hybrids, no exxdiv).  `mf.Gradients()` refuses ("pbc-RKS must be computed with MultiGridNumInt2"),
but `pyscf.pbc.grad.rks.Gradients(mf)` built directly runs, and its all-electron branch (with_nuc: point charges x
coulG on the mesh) matches central FD of PySCF's own energy (checked below).  So:
  pieces  (1) Ewald stress vs FD of PySCF cell.ewald() on strained cells (independent Ewald code, its own eta/cutoffs)
          (2) d v_M/d eps vs FD of pyscf.pbc.tools.madelung on strained cells
          (3) sum D dT/d eps and sum M dS/d eps vs rks_stress.get_kin / get_ovlp (FD of PySCF's pbc_intor lattice sums)
  rks/uks (4) END TO END: PySCF RKS/UKS (FFTDF mesh n, UniformGrids mesh n) stress vs ours (pure AFT at the default
          converged G sphere + our uniform grid n, i.e. construction B with fractional points and weights Omega/n^3),
          each from its own SCF; plus FD of PySCF's energy as a check that PySCF's stress is the derivative of ITS energy.
PREDICTIONS: (1)-(3) at the FD floor of PySCF's own differencing (disp 1e-5 => ~1e-9 .. 1e-10);  (4) E agrees to
~1e-10 (FFTDF-vs-AFT and mesh-vs-sphere differences), sigma to ~1e-8 or better.  An error in any piece shared by both
exchange-free functionals (T, S/Pulay, V_ne, J, Ewald, XC AO + grid-volume terms) shows as >= 1e-4 there."""

import sys
import time

import numpy as np
from pyscf.pbc import dft as pdft
from pyscf.pbc import gto as pgto
from pyscf.pbc import tools as ptools
from pyscf.pbc.dft import gen_grid as pgg
from pyscf.pbc.grad import rks as prks
from pyscf.pbc.grad import rks_stress, uks_stress
from pyscf.pbc.grad import uks as puks

import pbc_gamma as PG
import pbc_stress as ST
from run_grad_oracle import TRI_MOVED
from run_stress_anchor import H2_ATOMS, H3_ATOMS
from test_prototype import SP_BASIS, TRI_A


def pcell(a, atoms, basis, mesh=None, spin=0):
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=spin)
    pc.precision = 1e-12
    if mesh:
        pc.mesh = [mesh] * 3
    return pc.build()


def fd_pyscf(pc, f, disp=1e-5):
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            c1, c2 = rks_stress._finite_diff_cells(pc, i, j, disp)
            out[i, j] = (f(c1) - f(c2)) / (2 * disp)
    return out


def pieces():
    for name, a, atoms, basis in (("H2 cubic", np.eye(3) * 4.0, H2_ATOMS, "sto-3g"), ("tri s+p", TRI_A, TRI_MOVED, SP_BASIS)):
        cell = ST.FixedCell(a, atoms, basis)
        pc = pcell(a, atoms, basis)
        ours = ST.ewald_stress(cell, cell.Z, cell.R, 1.0)[0]
        theirs = fd_pyscf(pc, lambda c: c.ewald())
        print(f"{name}: (1) Ewald stress ours vs FD(PySCF cell.ewald) max diff {abs(ours - theirs).max():.2e} "
              f"(|stress| {abs(ours).max():.4f}); E_nn ours - PySCF {PG.ewald_nn(cell, 1.0) - pc.ewald():.1e}")
        ours = ST.madelung_stress(cell)
        theirs = fd_pyscf(pc, lambda c: ptools.madelung(c, np.zeros((1, 3))))
        print(f"{name}: (2) dv_M/deps ours vs FD(tools.madelung) max diff {abs(ours - theirs).max():.2e} "
              f"(|dv_M| {abs(ours).max():.4f}); v_M ours - PySCF {PG.madelung(cell) - ptools.madelung(pc, np.zeros((1, 3))):.1e}")
        rng = np.random.default_rng(7)
        D = rng.standard_normal((cell.mol.nao,) * 2)
        D = D + D.T
        dT = ST.latsum_virial(cell, "int1e_ipkin_cart", 22.0)
        dS = ST.latsum_virial(cell, "int1e_ipovlp_cart", 22.0)
        kin = np.array([[np.sum(t * D) for t in rks_stress.get_kin(pc)[3 * i : 3 * i + 3]] for i in range(3)])
        ovl = np.array([[np.sum(s * D) for s in rks_stress.get_ovlp(pc)[3 * i : 3 * i + 3]] for i in range(3)])
        ot, os_ = np.einsum("ijmn,mn->ij", dT, D), np.einsum("ijmn,mn->ij", dS, D)
        print(f"{name}: (3) sum D dT ours vs FD(pbc_intor int1e_kin) {abs(ot - kin).max():.2e} (|.| {abs(ot).max():.3f}); "
              f"sum D dS {abs(os_ - ovl).max():.2e} (|.| {abs(os_).max():.3f})", flush=True)


def end_to_end(kind):
    if kind == "rks":
        systems = [("H2 a=4 RKS", np.eye(3) * 4.0, H2_ATOMS, 1, 1, 45)]
    else:
        systems = [("H3 a=4.5 UKS doublet", np.eye(3) * 4.5, H3_ATOMS, 2, 1, 51)]
    for label, a, atoms, na, nb, n in systems:
        for xc in ("LDA,VWN", "PBE"):
            t = time.time()
            pc = pcell(a, atoms, "sto-3g", mesh=n, spin=na - nb)
            mf = (pdft.RKS if kind == "rks" else pdft.UKS)(pc, xc=xc)
            mf.grids = pgg.UniformGrids(pc)
            mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
            ep = mf.kernel()
            g = (prks.Gradients if kind == "rks" else puks.Gradients)(mf)
            sp = (rks_stress if kind == "rks" else uks_stress).kernel(g) * pc.vol  # dE/d eps
            tp = time.time() - t
            cell = ST.FixedCell(a, atoms, "sto-3g")
            pmax = 2 * max(cell.mol.bas_exp(i).max() for i in range(cell.mol.nbas))
            spec = dict(na=na, nb=nb, xc=xc, restricted=kind == "rks", exxdiv="none",
                        gcut=2 * np.sqrt(pmax * np.log(1e14)), grid=("uniform", n))
            ints, _, r = ST.solve(cell, spec)
            so, _ = ST.analytic(cell, spec, ints, r)
            print(f"\n{label} {xc} mesh {n} (PySCF {tp:.0f}s, ours {time.time() - t - tp:.0f}s): E ours {r['e']:.12f} "
                  f"PySCF {ep:.12f} diff {r['e'] - ep:+.1e}")
            print("  PySCF dE/deps:\n" + np.array2string(sp, precision=10))
            print(f"  max|ours - PySCF| = {abs(so - sp).max():.2e}   |PySCF - PySCF^T| = {abs(sp - sp.T).max():.1e}   "
                  f"|ours - ours^T| = {abs(so - so.T).max():.1e}")
            print("  ours - PySCF:\n" + np.array2string(so - sp, precision=2), flush=True)
            # PySCF's stress vs FD of PySCF's own energy (2 components, disp 1e-4), started from its density
            dm = mf.make_rdm1()
            for i, j in ((0, 0), (1, 2)):
                es = []
                for c in rks_stress._finite_diff_cells(pc, i, j, 1e-4):
                    c.mesh = [n] * 3
                    c.build(False, False)
                    m = (pdft.RKS if kind == "rks" else pdft.UKS)(c, xc=xc)
                    m.grids = pgg.UniformGrids(c)
                    m.conv_tol = 1e-13
                    es.append(m.kernel(dm0=dm))
                print(f"  PySCF stress - FD(PySCF E) ({i},{j}): {sp[i, j] - (es[0] - es[1]) / 2e-4:+.1e}", flush=True)


if __name__ == "__main__":
    w = sys.argv[1]
    pieces() if w == "pieces" else end_to_end(w)
