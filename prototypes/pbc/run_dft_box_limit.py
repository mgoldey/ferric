"""Iteration 8 independent oracle (not PySCF pbc): huge box -> molecular KS with the SAME grid settings.

Part A (grid construction only, no Coulomb): LiH/STO-3G (heteronuclear -> Becke size adjustment active) in a
cubic box a.  E_xc^PBE of the MOLECULAR PBE density matrix on the periodic A2 grid vs on PySCF's molecular grid
(TA-M4 + Lebedev, original Becke, Becke size adjust, no pruning = ferric's molecular grid).  D = 0.9 a (must
exceed the covering radius of the atom lattice, else points with no atom within D get zero weight).
  physics: images leave the partition only through s(mu) tails and AO tails -> |dExc| falls fast with a.
  artifact: a wrong image/home bookkeeping or size-adjust mismatch -> a PLATEAU at the grid-error level (~1e-5).
  control: adjust=False on the periodic side must break the largest-a agreement.
  Becke tails are POLYNOMIAL in 1/a (s(mu) near mu=-1 is ~eps^8 after 3 iterations, mu ~ -1 + 2r/a), so Becke
  agreement improves only algebraically; SSF has compact support |nu|<0.64 -> points within ~0.18a of the
  molecule must agree to machine precision (the sharp anchor).  CORRECTION after the first run: with the Becke
  size adjustment clipped at |a|=1/2 (Li-H), nu -> -1 + 4r/a, so the SSF exact region shrinks to r < ~0.09a
  (measured: at a=60 an H point 8.7 Bohr out saw a Li image 51 Bohr away, nu=-0.445); compare within 0.08a.

Part B (full KS): H2/STO-3G, pure-AFT exact J/K (exxdiv=ewald), 75x302 grid, vs pyscf.dft.RKS molecular on the
identical grid.  Predictions written before running:
  LDA/PBE: no exact exchange, H2 neutral + centrosymmetric (dipole 0) -> NO a^-3 term; residual a^-5 or faster.
  PBE0: exchange Makov-Payne term scaled by the exact-exchange fraction: c3 = -hyb (4 pi/3) sigma^2(PBE0 orbital).
  artifact: Madelung on the full K instead of hyb*K -> c3 4x too big; grid broken -> plateau.
Usage: python3 run_dft_box_limit.py {A|B} [edges...]"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from pyscf import gto
from pyscf.dft import libxc
from pbc_gamma import Cell, build_integrals
import pbc_dft as pd

part = sys.argv[1] if len(sys.argv) > 1 else "A"
edges = [float(x) for x in sys.argv[2:]]
NR, NA = 75, 302

if part == "A":
    from scipy.spatial import cKDTree
    from pyscf.dft import numint

    atoms = [("Li", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 3.1))]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
    Dm = pd.molecular_rks(mol, "PBE", NR, NA).make_rdm1()
    cen = mol.atom_coords().mean(0)
    for scheme in ("becke", "ssf"):
        cm, wm = pd.molecular_grid_reference(mol, NR, NA, scheme=scheme)

        class G:  # molecular grid in PeriodicGrid's shape
            ao, weights = numint.eval_ao(mol, cm, deriv=1), wm

        e_mol = pd.eval_vxc(G, Dm, "PBE")[0]
        tree = cKDTree(cm)
        print(
            f"LiH {scheme}: molecular npts {len(wm)}  E_xc^PBE {e_mol:.14f}", flush=True
        )
        for a in edges or (12.0, 20.0, 30.0, 60.0):
            for adjust in (True, False) if a == (edges or [60.0])[-1] else (True,):
                t = time.time()
                cell = Cell(np.eye(3) * a, atoms, "sto-3g")
                g = pd.PeriodicGrid(
                    cell, NR, NA, D=0.9 * a, adjust=adjust, scheme=scheme
                )
                e = pd.eval_vxc(g, Dm, "PBE")[0]
                d, i = tree.query(g.coords)
                near = (d < 1e-9) & (np.linalg.norm(g.coords - cen, axis=1) < 0.08 * a)
                dw = np.abs(g.weights[near] - wm[i[near]]).max()
                print(
                    f"  a={a:5.1f} adjust={adjust!s:5s} npts {g.size:6d} <nb/pt> {g.n_nb.mean():5.1f} <img/chunk> {g.n_img.mean():5.1f}"
                    f"  E_xc - mol = {e - e_mol:+.3e}  max|dw| (|r-c|<0.08a, {near.sum()} pts) {dw:.1e}  ({time.time() - t:.0f}s)",
                    flush=True,
                )
else:
    atoms = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="B", cart=True, verbose=0)
    ref = {}
    for xc in ("LDA,VWN", "PBE", "PBE0"):
        mf = pd.molecular_rks(mol, xc, NR, NA)
        c = mf.mo_coeff[:, :1]
        ref[xc] = (mf.e_tot, pd.ov_moment_sigma2(mol, c), mf.make_rdm1())
        hyb = libxc.hybrid_coeff(xc)
        print(
            f"{xc}: E_mol {mf.e_tot:.12f} sigma2 {ref[xc][1]:.8f} predicted c3 {-hyb * 4 * np.pi / 3 * ref[xc][1]:+.6f}",
            flush=True,
        )
    for a in edges or (12.0, 16.0, 20.0):
        t = time.time()
        cell = Cell(np.eye(3) * a, atoms, "sto-3g")
        I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
        g = pd.PeriodicGrid(cell, NR, NA, D=min(0.9 * a, 18.0))
        row = []
        for xc in ("LDA,VWN", "PBE", "PBE0"):
            e = pd.rks(
                I["S"],
                I["h"],
                pd.dense_jk(I["I"]),
                I["enn"],
                2,
                g,
                xc,
                kshift=I["madelung"],
                conv=1e-12,
                D0=ref[xc][2],
            )[0]
            row.append(e - ref[xc][0])
        print(
            f"  a={a:5.1f} npts {g.size}  dE(LDA) {row[0]:+.6e}  dE(PBE) {row[1]:+.6e}  dE(PBE0) {row[2]:+.6e}"
            f"  dE(PBE0)*a^3 {row[2] * a**3:+.5f}  ({time.time() - t:.0f}s)",
            flush=True,
        )
