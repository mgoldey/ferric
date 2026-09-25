"""Iteration 17 PySCF oracle for Gamma UHF / UKS forces (pbc_grad_open.py).  Usage: python3 run_grad_open_oracle.py [h3|tri] [xc ...]

PySCF 2.13 pbc.grad.{uhf,kuhf,uks,kuks,rks,krks} cannot be used whole: all-electron cells raise NotImplementedError
(get_hcore needs cell.pseudo), grid_response raises NotImplementedError, and get_jk_e1 exists only on FFTDF.  So the
oracle is assembled from what PySCF CAN do, all on the SAME uniform grid (construction B: fixed points => the
analytic force has no grid response and FD of a uniform-grid energy is the AO term alone):
  (1) FFTDF get_jk_e1 per spin (J from D_a + D_b, K per spin) contracted with OUR D_s  vs our LR 2e force (UHF Gam)
  (2) pbc.grad.kuks.get_vxc (Gamma only) on UniformGrids(mesh) with OUR D_s        vs our XC AO-derivative term
      (independent AO lattice sums, libxc packing and GGA contraction)
  (3) TOTAL: central FD of PySCF's own pbc UHF / UKS energy (AFTDF mesh M, UniformGrids mesh M, exxdiv None)
      vs our analytic force (pure-AFT default gcut, our uniform grid mesh M)
PREDICTIONS: (1) FFTDF mesh error only (Iteration 16: 9e-16 at mesh 61 s-only, 9e-14 with p); (2) ~1e-12 (same
points, same libxc; only AO-sum truncation differs); (3) ~1e-9 (FD floor) up to the J/K AFT-vs-mesh difference
(Iteration 16 RHF: 9e-11 .. 5e-10).  Artifact: a per-spin error in (1)/(2) shows as O(1e-2) on an open shell."""

import sys
import time

import numpy as np
from pyscf.pbc import df as pdf
from pyscf.pbc import dft as pdft
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.dft import gen_grid as pgg
from pyscf.pbc.grad import kuks as pkuks

import pbc_grad as PGd
import pbc_grad_open as O
from pbc_gamma import Cell
from run_grad_oracle import TRI_MOVED
from test_prototype import SP_BASIS, TRI_A

H3 = (np.eye(3) * 4.5, [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (1.6, 0.9, 0.7))], "sto-3g",
      (2, 1), 45, [(0, 0), (2, 1)])
TRI = (TRI_A, TRI_MOVED, SP_BASIS, (3, 1), 45, [(2, 1)])


def pcell(a, atoms, basis, mesh, spin):
    pc = pgto.Cell(a=a, atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=spin)
    pc.precision = 1e-12
    pc.mesh = [mesh] * 3
    return pc.build()


def pyscf_e(a, atoms, basis, mesh, spin, xc, dm0=None):
    pc = pcell(a, atoms, basis, mesh, spin)
    if xc == "HF":
        mf = pscf.UHF(pc, exxdiv=None)
    else:
        mf = pdft.UKS(pc, xc=xc, exxdiv=None)
        mf.grids = pgg.UniformGrids(pc)
        mf.grids.mesh = [mesh] * 3
        mf.small_rho_cutoff = 0.0
    mf.with_df = pdf.AFTDF(pc)
    mf.with_df.mesh = [mesh] * 3
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-8
    e = mf.kernel(dm0)
    return e, mf.make_rdm1()


def run(sysdef, name, xcs=("HF", "PBE", "PBE0")):
    a, atoms, basis, (na, nb), mesh, comps = sysdef
    cell = Cell(a, atoms, basis)
    pc = pcell(a, atoms, basis, mesh, na - nb)
    aoat, natm = PGd.ao_atom(cell.mol), cell.mol.natm
    t = time.time()
    ints = O.integrals(cell, None, "none")
    ints["madelung"] = 0.0
    ug = O.uniform_grad_grid(cell, mesh)
    print(f"[{name}] ints + uniform {mesh}^3 grid {time.time() - t:.0f}s", flush=True)
    for xc in xcs:
        t = time.time()
        r = O.run_case(cell, na, nb, xc, ints, grid=ug)
        Da, Db = r["scf"]["Da"], r["scf"]["Db"]
        print(f"  {xc:5s} ours E {r['e']:.12f} ({time.time() - t:.0f}s) sumF {abs(r['grad'].sum(0)).max():.1e}", flush=True)
        if xc == "HF":  # (1) 2e LR force per spin
            fft = pdf.FFTDF(pc)
            fft.mesh = [61] * 3
            k0 = np.zeros((1, 3))
            vja, vka = fft.get_jk_e1(Da[None], k0, exxdiv=None)
            vjb, vkb = fft.get_jk_e1(Db[None], k0, exxdiv=None)
            sh = (3,) + Da.shape
            vj = (np.asarray(vja).reshape(sh) + np.asarray(vjb).reshape(sh)).real
            vka, vkb = np.asarray(vka).reshape(sh).real, np.asarray(vkb).reshape(sh).real
            per_m = 2 * (np.einsum("xmn,mn->mx", vj, Da + Db) - np.einsum("xmn,mn->mx", vka, Da)
                         - np.einsum("xmn,mn->mx", vkb, Db))
            g2 = PGd._fold(per_m, aoat, natm)
            print(f"   (1) FFTDF(mesh 61) jk_e1 per spin vs our Ilr: {abs(g2 - r['parts']['Ilr']).max():.2e} "
                  f"(|Ilr| {abs(r['parts']['Ilr']).max():.2e})", flush=True)
        else:  # (2) XC AO term
            ni = pdft.numint.KNumInt()
            grids = pgg.UniformGrids(pc)
            grids.mesh = [mesh] * 3
            dms = np.stack([Da[None], Db[None]])
            v = pkuks.get_vxc(ni, pc, grids, xc, dms, np.zeros((1, 3))).real  # (3, 2, 1, nao, nao)
            per_m = 2 * (np.einsum("xmn,mn->mx", v[:, 0, 0], Da) + np.einsum("xmn,mn->mx", v[:, 1, 0], Db))
            gx = PGd._fold(per_m, aoat, natm)
            print(f"   (2) kuks.get_vxc (UniformGrids {mesh}) vs our xc_ao: {abs(gx - r['parts']['xc_ao']).max():.2e} "
                  f"(|xc_ao| {abs(r['parts']['xc_ao']).max():.2e})", flush=True)
        # (3) total vs FD of PySCF's energy
        h = 1e-4
        # seeded with OUR (D_a, D_b): on the tri triplet PBE PySCF's default guess lands 4.2e-4 Ha higher (another state)
        e0, dm0 = pyscf_e(a, atoms, basis, mesh, na - nb, xc, np.stack([Da, Db]))
        print(f"   (3) PySCF E {e0:.12f}  ours - PySCF {r['e'] - e0:+.2e}", flush=True)
        for A, x in comps:
            t = time.time()
            ep, _ = pyscf_e(a, PGd.displaced(cell, A, x, h).atoms, basis, mesh, na - nb, xc, dm0)
            em, _ = pyscf_e(a, PGd.displaced(cell, A, x, -h).atoms, basis, mesh, na - nb, xc, dm0)
            fd = (ep - em) / (2 * h)
            print(f"   (3) dE/dR[{A},{x}] PySCF FD {fd:.10f} ours {r['grad'][A, x]:.10f} diff {r['grad'][A, x] - fd:.2e} "
                  f"({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "h3"
    run(H3 if which == "h3" else TRI, which, *( [tuple(sys.argv[2:])] if len(sys.argv) > 2 else []))
