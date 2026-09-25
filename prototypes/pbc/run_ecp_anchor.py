"""Periodic-ECP exactness anchors and PySCF oracle (Iteration 14).

System: HI, H STO-3G, I LANL2DZ basis + LANL2DZ ECP (46 core electrons, local + s/p/d semi-local
projectors), 8 valence electrons, nao 9 (cart).  The basis is soft (max exponent 3.43 on H), so the
pure-AFT route of pbc_gamma/pbc_kpts is affordable; the ECP adds only V_ECP and Z_eff.

Predictions (stated before running):
  box  (a) cubic box a -> molecular RHF (PySCF, same basis + ECP):
         physics:  E(a) - E_mol = c3/a^3 + c5/a^5 in the tail, c3 = -(4pi/3) Omega_I - (2pi/3)|p|^2 with
                   Omega_I the occupied gauge-invariant spread and p the TOTAL dipole built with Z_eff
                   (Hartree G=0 cell of a neutral dipolar cell + exchange q^2 head; exxdiv='ewald').
         artifact: bare Z in V_ne => charged cell for the electrons, O(10-100 Ha) error, no convergence;
                   bare Z in E_nn only => a rigid O(Z^2 / a) offset (charged-cell Madelung/background term),
                   converging like 1/a, not a^-3.  Missing ECP images: INVISIBLE in the box limit (the
                   neighbouring ECP centres leave the orbital range as exp(-0.1 a^2)) -- the box anchor is
                   blind to it by construction.
  kmesh (c) k-mesh E/cell == Gamma supercell E/N (same K sphere both sides => exact at any gcut):
         physics:  <= 1e-12 with the ECP (V_ECP(k) is a Bloch sum; the supercell's own ECP lattice sum
                   contains the same (m, M, L) triples, relabelled).
         artifact: 'ecp_molecular' (M = 0, L = 0: the molecular ECP routine on the cell basis) differs between
                   the sides (the supercell's home block holds the intra-supercell cross terms) => O(1e-2) Ha visible; bare Z applied consistently to
                   BOTH sides => INVISIBLE (the anchor tests the Bloch construction, not the charge).
  oracle PySCF KRHF + AFTDF (cell.ecp; hcore = AFTDF get_nuc(Z_eff) + pbc.gto.ecp.ecp_int): agreement at
         the level of PySCF's own ECP lattice-sum residual (run_ecp_lattice.py measures it separately).

Usage: python3 run_ecp_anchor.py box a1 a2 ... | kmesh | oracle [mesh] | mutbox a
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_ecp as PE  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import rhf  # noqa: E402

BASIS = {"H": "sto-3g", "I": "lanl2dz"}
ECP = {"I": "lanl2dz"}
MOL = [("H", (0.0, 0.0, 0.0)), ("I", (0.0, 0.0, 3.04))]  # r_e(HI) = 1.609 A = 3.04 Bohr
CELL_A = np.diag([6.0, 6.0, 7.0])
CELL_ATOMS = [("H", (0.3, 0.2, 0.4)), ("I", (0.3, 0.2, 3.44))]
NELEC = 8


def box_cell(a):
    shift = np.array([0.37, 0.21, 0.5 * a - 1.52])  # centre the bond; position is physics-neutral
    return PE.EcpCell(np.eye(3) * a, [(s, np.asarray(r) + shift) for s, r in MOL], BASIS, ECP)


def box(alist, prec=1e-10):
    ref = PE.molecular_reference(MOL, BASIS, ECP)
    print(f"molecular RHF E {ref['e']:.12f}; Omega_I {ref['omega_I']:.8f}; p(Z_eff) {ref['p']} "
          f"p(bare Z) {ref['p_full']}; predicted c3 = {ref['c3']:.6f} "
          f"(exchange {-(4 * np.pi / 3) * ref['omega_I']:.6f}, dipole {-(2 * np.pi / 3) * ref['p'] @ ref['p']:.6f})",
          flush=True)
    for a in alist:
        cell = box_cell(a)
        t = time.time()
        g = PE.gamma_ecp(cell, "ewald", gcut=PK.aft_gcut(cell, prec))
        vm = g["madelung"]
        e, eps, it = rhf(g["S"], g["h"], g["I"], g["enn"], NELEC, conv=1e-11, kshift=vm)
        d = e - ref["e"]
        print(f"a {a:5.1f}: nG {g['nG']} E {e:.12f} E-E_mol {d:+.10e} a^3*dE {d * a**3:+.6f} "
              f"v_M {vm:.10f} it {it} ({time.time() - t:.0f}s)", flush=True)


def mutbox(a, prec=1e-10):
    ref = PE.molecular_reference(MOL, BASIS, ECP)
    cell = box_cell(a)
    out = {}
    for m in (None, "ecp_molecular", "z_full_enn", "z_full_vne"):
        PE._MUTANT = m
        try:
            g = PE.gamma_ecp(cell, "ewald", gcut=PK.aft_gcut(cell, prec))
            try:
                e = rhf(g["S"], g["h"], g["I"], g["enn"], NELEC, conv=1e-10, kshift=g["madelung"], maxiter=300)[0]
            except RuntimeError:
                e = float("nan")
        finally:
            PE._MUTANT = None
        out[m] = e
        print(f"box a={a} mutant {m}: E-E_mol {e - ref['e']:+.6e}", flush=True)
    return out


def kmesh(n=(1, 1, 3), prec=1e-4):
    cell = PE.EcpCell(CELL_A, CELL_ATOMS, BASIS, ECP)
    gcut = PK.aft_gcut(cell, prec)
    Nk = int(np.prod(n))
    sc = PE.supercell_ecp(cell, n)
    for m in (None, "ecp_molecular", "z_full_enn", "z_full_vne"):
        PE._MUTANT = m
        try:
            t = time.time()
            kb = PE.build_k_ecp(cell, n, gcut=gcut)
            g = PE.gamma_ecp(sc, gcut=gcut)
            for ex in ("none", "ewald"):
                vk = kb["madelung"] if ex == "ewald" else 0.0
                vs = g["madelung"] if ex == "ewald" else 0.0
                try:
                    ek, epk, _ = PK.krhf(kb, NELEC, conv=1e-12, kshift=vk, maxiter=300)
                    es, eps_s, _ = rhf(g["S"], g["h"], g["I"], g["enn"], NELEC * Nk, conv=1e-12, kshift=vs, maxiter=300)
                    de = ek - es / Nk
                    dep = abs(np.sort(np.concatenate(epk)) - np.sort(eps_s)).max()
                except RuntimeError as err:
                    ek, de, dep = float("nan"), float("nan"), float("nan")
                    print("   ", err)
                print(f"(c) HI {n} mutant {m} {ex}: E_k/cell {ek:.12f} dE(k - sc/N) {de:.1e} d(all eps) {dep:.1e} "
                      f"v_M k {kb['madelung']:.10f} sc {g['madelung']:.10f} ({time.time() - t:.0f}s)", flush=True)
        finally:
            PE._MUTANT = None


def oracle(n=(1, 1, 2), mesh=(61, 61, 71)):
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import AFTDF
    from pyscf.pbc.gto import ecp as pecp

    cell = PE.EcpCell(CELL_A, CELL_ATOMS, BASIS, ECP)
    kb = PE.build_k_ecp(cell, n, verbose=True)
    pc = pgto.Cell(a=CELL_A, atom=CELL_ATOMS, basis=BASIS, ecp=ECP, unit="B", cart=True, verbose=0)
    pc.precision = 1e-12
    pc.build()
    kpts = pc.make_kpts(n)
    assert abs(kpts - kb["kpts"]).max() < 1e-12
    vp = np.asarray(pecp.ecp_int(pc, kpts))
    print(f"  V_ECP(k) ours vs PySCF ecp_int: {abs(kb['Vecp'] - vp).max():.2e}; E_nn d {kb['enn'] - pc.energy_nuc():.1e}")
    for ex in ("none", "ewald"):
        e, eps, _ = PK.krhf(kb, NELEC, conv=1e-12, kshift=kb["madelung"] if ex == "ewald" else 0.0, return_mo=False)
        mf = pscf.KRHF(pc, kpts, exxdiv=(None if ex == "none" else "ewald"))
        mf.with_df = AFTDF(pc, kpts)
        mf.with_df.mesh = list(mesh)
        mf.conv_tol = 1e-11
        # our V_ECP in PySCF's hcore isolates the SCF/ERI comparison from PySCF's ECP lattice residual
        h_p = mf.get_hcore()
        dh = abs(h_p - kb["h"]).max()
        dm0 = np.array([2 * c for c in _dm_guess(kb, ex)])
        ep = mf.kernel(dm0=dm0)
        mf2 = pscf.KRHF(pc, kpts, exxdiv=(None if ex == "none" else "ewald"))
        mf2.with_df = mf.with_df
        mf2.conv_tol = 1e-11
        mf2.get_hcore = lambda *a, **k: kb["h"]
        ep2 = mf2.kernel(dm0=mf.make_rdm1())
        print(f"(oracle) HI {n} {ex}: E ours {e:.12f} PySCF {ep:.12f} dE {e - ep:.1e}; with OUR hcore in PySCF "
              f"{ep2:.12f} dE {e - ep2:.1e}; max|dh| {dh:.1e}; d eps {abs(np.concatenate(eps) - np.concatenate(mf2.mo_energy)).max():.1e}",
              flush=True)


def _dm_guess(kb, ex):
    e, eps, _, C, occ = PK.krhf(kb, NELEC, conv=1e-12, kshift=kb["madelung"] if ex == "ewald" else 0.0, return_mo=True)
    return [C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T for k in range(kb["Nk"])]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "box":
        box([float(x) for x in sys.argv[2:]])
    elif cmd == "mutbox":
        mutbox(float(sys.argv[2]))
    elif cmd == "kmesh":
        kmesh()
    elif cmd == "oracle":
        oracle(mesh=tuple(int(x) for x in sys.argv[2:5]) if len(sys.argv) > 2 else (61, 61, 71))
