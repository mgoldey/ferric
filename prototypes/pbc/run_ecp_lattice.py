"""Periodic ECP matrix elements: lattice-sum convergence, image counts, and the PySCF pbc oracle.

Predictions (written before the run):
  P1 (physics) V_ECP is short-range: the per-ECP-image contribution max|V[M]| decays like a Gaussian in the
     distance of the image M from the home cell, exp(-a z/(a+z) d^2) with a = most diffuse basis exponent,
     z = most diffuse ECP exponent, so the truncation error falls FASTER than any power of the cutoff and
     the ecp_ranges() radius (prec 1e-14) is conservative.
  A1 (artifact) if the ECP image set were wrong (e.g. only M = 0, or images selected by the wrong atom),
     the error would NOT fall with the cutoff but plateau at the size of the missing images, and the
     PySCF comparison (independent Int3cBuilder lattice code, same libcint kernel) would show a k-dependent
     residual of the same size.  A phase error (e^{-ik.L}) is invisible at Gamma, visible at k != 0.
  P2 E_nn(Ewald, Z_eff) == PySCF cell.energy_nuc() (which uses atom_charges = Z_eff) to 1e-12; with the bare
     Z the cell is charged and the difference is O(Z^2) Hartree.

Usage: python3 run_ecp_lattice.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_ecp as PE  # noqa: E402
from pbc_gamma import _ewald  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402
from pyscf.pbc.gto import ecp as pecp  # noqa: E402

SYSTEMS = {
    # heavy ECP: def2-SVP + def2-ecp on I (28 core e, s/p/d projectors), tight + diffuse basis, triclinic
    "HI/def2-SVP tri": dict(
        a=np.array([[6.2, 0.0, 0.0], [0.8, 5.9, 0.0], [0.4, 0.6, 6.6]]),
        atoms=[("H", (0.2, 0.3, 0.4)), ("I", (0.2, 0.3, 3.44))],
        basis={"H": "sto-3g", "I": "def2-svp"}, ecp={"I": "def2-svp"}),
    # the SCF test system: LANL2DZ I (46 core e), soft valence basis
    "HI/lanl2dz cub": dict(
        a=np.diag([6.0, 6.0, 7.0]), atoms=[("H", (0.3, 0.2, 0.4)), ("I", (0.3, 0.2, 3.44))],
        basis={"H": "sto-3g", "I": "lanl2dz"}, ecp={"I": "lanl2dz"}),
}


def pyscf_cell(s):
    pc = pgto.Cell(a=s["a"], atom=s["atoms"], basis=s["basis"], ecp=s["ecp"], unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    return pc.build()


def main():
    for name, s in SYSTEMS.items():
        cell = PE.EcpCell(s["a"], s["atoms"], s["basis"], s["ecp"])
        r_e, r_o, amin, zmin = PE.ecp_ranges(cell)
        t = time.time()
        img = PE.ecp_images(cell)
        print(f"\n== {name}: nao {cell.mol.nao}, Z {cell.Zfull} -> Z_eff {cell.Z}; a_min {amin:.4f} z_min {zmin:.4f}; "
              f"r_ecp {r_e:.2f} r_orb {r_o:.2f}: nM {len(img['Lecp'])} nL {len(img['Lorb'])} ({time.time() - t:.1f}s)")
        pc = pyscf_cell(s)
        print(f"  E_nn Ewald(Z_eff) {_ewald(cell, cell.Z, cell.R, 1.0):.12f}  PySCF {pc.energy_nuc():.12f}  "
              f"d {_ewald(cell, cell.Z, cell.R, 1.0) - pc.energy_nuc():.1e};  Ewald(bare Z) - that = "
              f"{_ewald(cell, cell.Zfull, cell.R, 1.0) - pc.energy_nuc():.3f}")
        # per-image decay
        dM = np.linalg.norm(img["Lecp"], axis=1)
        mx = np.abs(img["V"]).max(axis=(1, 2, 3))
        for r in np.unique(np.round(dM, 1))[:12]:
            sel = np.abs(dM - r) < 0.051
            print(f"    |M| {r:5.1f}: n {sel.sum():3d} max|V_M| {mx[sel].max():.2e}")
        # truncation errors vs cutoff (Gamma and max over a 3x3x3 mesh + a generic k)
        ints, kpts = PK_mesh(cell)
        ref = PE.ecp_k(img, kpts)
        print("  ECP-image cutoff (orbital images full) | orbital-image cutoff (ECP images full)")
        for R in (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0):
            e1 = abs(PE.ecp_k(img, kpts, rcut_ecp=R) - ref).max()
            n1 = int((dM <= R + 1e-9).sum())
            e2 = abs(PE.ecp_k(img, kpts, rcut_orb=R) - ref).max()
            n2 = int((np.linalg.norm(img["Lorb"], axis=1) <= R + 1e-9).sum())
            print(f"    R {R:5.1f}: nM {n1:4d} err {e1:.2e} | nL {n2:4d} err {e2:.2e}")
        # PySCF oracle.  PySCF 2.13 pbc.gto.ecp.ecp_int is only trustworthy on small MP meshes: it returns
        # garbage (|V| up to 1e9) for the 3x3x3 mesh at nao 29 and for any non-mesh k list (Gamma + generic k),
        # while <= 9-k meshes agree with ours to 3e-8 (measured 2026-09-24).  Compare on 2x2x2, report the rest.
        k8 = PK_mesh(cell, (2, 2, 2))[1][:8]
        vp = np.asarray(pecp.ecp_int(pc, k8))
        r8 = PE.ecp_k(img, k8)
        print(f"  vs PySCF pbc ecp_int (2x2x2 mesh): max|dV| {abs(r8 - vp).max():.2e} (Gamma {abs(r8[0] - vp[0]).max():.2e}), "
              f"max|V| {abs(vp).max():.2e}")
        v27 = np.asarray(pecp.ecp_int(pc, kpts[:27]))
        vg = np.asarray(pecp.ecp_int(pc, kpts[[0, 27]]))
        print(f"  PySCF ecp_int 3x3x3 mesh: max|dV| {abs(v27 - ref[:27]).max():.2e}; [Gamma, generic k]: max|V| "
              f"{abs(vg).max():.2e} (ours {abs(ref).max():.2e})")
        PE._MUTANT = "ecp_molecular"
        print(f"  mutant ecp_molecular (M = 0, L = 0): max|dV| vs full {abs(PE.ecp_k(img, kpts) - ref).max():.2e}")
        PE._MUTANT = None
        bad = np.einsum("kL,mLn->kmn", np.exp(-1j * kpts @ img["Lorb"].T), img["V"].sum(0))
        print(f"  mutant phase e^(-ik.L): Gamma max|dV| {abs(bad[0] - ref[0]).max():.1e}, all k {abs(bad - ref).max():.2e}")

def PK_mesh(cell, n=(3, 3, 3)):
    import pbc_kpts as PK

    ints, k = PK.mp_mesh(cell, n)
    kg = np.array([[0.13, -0.27, 0.31]]) @ cell.b
    return ints, np.vstack([k, kg])


if __name__ == "__main__":
    main()
