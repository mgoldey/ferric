"""Iteration 16 PySCF oracle for the Gamma RHF forces (pbc_grad.py).

PySCF 2.13 pbc.grad.{rhf,krhf} raise NotImplementedError for ALL-ELECTRON cells (get_hcore needs cell.pseudo),
and get_jk_e1 exists only on FFTDF.  So the oracle is assembled from the pieces PySCF CAN do:
  (1) grad_nuc (Ewald)                       vs pbc_grad.ewald_grad
  (2) cell.pbc_intor int1e_ipovlp / ipkin    vs pbc_grad._latsum_ip  (matrix level)
  (3) FFTDF get_jk_e1 contracted with OUR D   vs our LR (pure-AFT) 2e gradient   (FFTDF mesh error expected)
  (4) TOTAL force: central FD of PySCF's own AFTDF Gamma RHF energy (independent of every line of pbc_grad)
Usage: python3 run_grad_oracle.py [h2|tri|both]
"""

import sys
import time

import numpy as np
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.grad import krhf as pkgrad

import pbc_grad as PGd
from pbc_gamma import Cell
from test_prototype import SP_BASIS, TRI_A

H2 = (np.eye(3) * 4.0, [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))], "sto-3g", 2, 61, [(0, 0), (0, 2)])
TRI_MOVED = [("H", (0.13, 0.25, 0.31)), ("H", (0.02, 0.27, 1.66)), ("H", (2.47, 2.41, 2.25)), ("H", (3.52, 2.98, 2.71))]
TRI = (TRI_A, TRI_MOVED, SP_BASIS, 4, 45, [(2, 1)])


def pcell(a, atoms, basis, mesh):
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    pc.mesh = [mesh] * 3
    return pc.build()


def pyscf_e(a, atoms, basis, mesh, exxdiv, dm0=None):
    pc = pcell(a, atoms, basis, mesh)
    mf = pscf.RHF(pc, exxdiv=exxdiv)
    mf.with_df = pdf.AFTDF(pc)
    mf.with_df.mesh = [mesh] * 3
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-8
    e = mf.kernel(dm0)
    return e, mf.make_rdm1()


def run(sysdef, name):
    a, atoms, basis, nelec, mesh, comps = sysdef
    cell = Cell(a, atoms, basis)
    pc = pcell(a, atoms, basis, mesh)
    t = time.time()
    r = PGd.gamma_rhf_grad(cell, nelec, exxdiv="none")
    print(f"[{name}] ours pure-AFT E={r['e']:.12f} ({time.time() - t:.0f}s) comm {r['comm']:.1e}  sum F {abs(r['grad'].sum(0)).max():.1e}")
    print(np.array2string(r["grad"], precision=10))
    # (1) Ewald
    gn = pkgrad.grad_nuc(pc, None)
    print(f"  (1) grad_nuc  max|ours-PySCF| = {abs(PGd.ewald_grad(cell) - gn).max():.2e}")
    # (2) lattice-summed derivative overlap / kinetic
    for o in ("int1e_ipovlp", "int1e_ipkin"):
        ref = np.asarray(pc.pbc_intor(o, comp=3, hermi=0)).real
        print(f"  (2) {o:13s} max|ours-PySCF| = {abs(PGd._latsum_ip(cell, o + '_cart', 22.0) - ref).max():.2e}")
    # (3) FFTDF J/K derivative contracted with OUR density (exxdiv=None)
    fft = pdf.FFTDF(pc)
    D = r["D"]
    vj, vk = fft.get_jk_e1(D[None], np.zeros((1, 3)), exxdiv=None)
    vhf = np.asarray(vj).reshape(3, *D.shape) - 0.5 * np.asarray(vk).reshape(3, *D.shape)
    per_m = 2 * np.einsum("xmn,mn->mx", vhf.real, D)
    g2 = PGd._fold(per_m, PGd.ao_atom(cell.mol), cell.mol.natm)
    print(f"  (3) FFTDF(mesh {mesh}) jk_e1 vs our Ilr: max diff {abs(g2 - r['parts']['Ilr']).max():.2e} "
          f"(|Ilr| {abs(r['parts']['Ilr']).max():.2e})")
    # (4) total: FD of PySCF AFTDF energy, both exxdiv
    for ex in ("none", "ewald"):
        rr = r if ex == "none" else PGd.gamma_rhf_grad(cell, nelec, exxdiv="ewald")
        for A, x in comps:
            t = time.time()
            h = 1e-4
            ep, dm = pyscf_e(a, PGd.displaced(cell, A, x, h).atoms, basis, mesh, None if ex == "none" else ex)
            em, _ = pyscf_e(a, PGd.displaced(cell, A, x, -h).atoms, basis, mesh, None if ex == "none" else ex, dm)
            fd = (ep - em) / (2 * h)
            print(f"  (4) exxdiv={ex:5s} dE/dR[{A},{x}]  PySCF-AFTDF FD {fd:.10f}  ours {rr['grad'][A, x]:.10f}  "
                  f"diff {rr['grad'][A, x] - fd:.2e} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("h2", "both"):
        run(H2, "H2/STO-3G a=4")
    if which in ("tri", "both"):
        run(TRI, "triclinic 4H s+p (moved)")
