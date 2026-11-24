"""PySCF periodic Becke grids on a FIXED density (issue #228, dry-ice PBE grid ladder).

Loads a ferric density matrix (and ferric's lattice overlap, written by
`cargo run --example grid_ladder_diag -- --mode scf --save-dm PATH`), checks
that PySCF's lattice overlap in ferric's own shell order equals ferric's, and
prints `N = Σ w ρ` and `E_xc[D]` on PySCF's `pbc.dft.gen_grid.BeckeGrids` for
every requested `(n_radial, n_angular, variant)`:

* `stock`: PySCF's periodic Becke partition (original Becke, no atomic-size
  adjustment, images to the basis rcut);
* `bragg`: the same grid with the Becke size adjustment on Bragg radii,
  `a_ij = (1/χ − χ)/4`, `χ = R_i/R_j`, clipped to ±½ — the partition ferric's
  `PartitionScheme::Becke` uses.

  OPENBLAS_NUM_THREADS=1 python grid_diag_pyscf.py dryice --dm d.bin \
      --grid 150 590 stock --grid 150 590 bragg
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pyscf import dft, lib  # noqa: E402
from pyscf.dft import radi  # noqa: E402
from pyscf.pbc import gto as pgto  # noqa: E402
from pyscf.pbc import dft as pdft  # noqa: E402
from pyscf.pbc.dft import gen_grid as pgen  # noqa: E402
from pyscf.pbc.gto import eval_gto as pbc_eval_gto  # noqa: E402

from pyscf_bench import ferric_basis_pyscf, read_cell  # noqa: E402

BASIS = {"dryice": ("cc-pvdz",), "diamond_prim": ("cc-pvdz",)}


def load_bin(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    return np.frombuffer(raw[8:], dtype="<f8").reshape(n, n).copy()


def becke_grids(cell, atom_grid, adjust):
    """`pbc.dft.gen_grid.get_becke_grids` with an optional Bragg size adjustment."""
    rcut = pbc_eval_gto._estimate_rcut(cell).max()
    Ls = pbc_eval_gto.get_lattice_Ls(cell, rcut=rcut)
    atm_coords = Ls.reshape(-1, 1, 3) + cell.atom_coords()
    tab = pgen.gen_atomic_grids(cell, atom_grid, radi.treutler, 3, None)
    coords_all, weights_all, supatm_idx = [], [], []
    b = cell.reciprocal_vectors(norm_to=1)
    tol = 1e-15
    k = 0
    for iL in range(len(Ls)):
        for ia in range(cell.natm):
            coords, vol = tab[cell.atom_symbol(ia)]
            coords = coords + atm_coords[iL, ia]
            c = b.dot(coords.T)
            mask = np.ones(c.shape[1], dtype=bool)
            for d in range(3):
                mask &= (c[d] > -0.5 - tol) & (c[d] < 0.5 + tol)
            vol = vol[mask]
            if vol.size > 8:
                c = c[:, mask]
                for d in range(3):
                    vol[abs(c[d] + 0.5) < tol] *= 0.5
                    vol[abs(c[d] - 0.5) < tol] *= 0.5
                coords_all.append(coords[mask])
                weights_all.append(vol)
                supatm_idx.append(k)
            k += 1
    offs = np.append(0, np.cumsum([w.size for w in weights_all]))
    coords_all = np.vstack(coords_all)
    weights_all = np.hstack(weights_all)
    sup = np.asarray(atm_coords.reshape(-1, 3)[supatm_idx], order="C")
    n_sup = len(sup)
    if adjust:
        z = np.tile(cell.atom_charges(), len(Ls))[supatm_idx]
        rad = radi.BRAGG_RADII[z]
        rr = rad.reshape(-1, 1) * (1.0 / rad)  # rr[i, j] = R_i / R_j = χ_ij
        table = np.ascontiguousarray(np.clip(0.25 * (rr.T - rr), -0.5, 0.5))
        p_radii = table.ctypes.data_as(ctypes.c_void_p)
    else:
        table = None
        p_radii = lib.c_null_ptr()
    fn = dft.gen_grid.libdft.VXCgen_grid
    blk = min(weights_all.size, max(2000, int(2e9 / 8 / n_sup)))
    displs = lib.misc._blocksize_partition(offs, blk)
    for n0, n1 in zip(displs[:-1], displs[1:]):
        p0, p1 = offs[n0], offs[n1]
        pb = np.empty((n_sup, p1 - p0))
        cc = np.asarray(coords_all[p0:p1], order="F")
        fn(
            pb.ctypes.data_as(ctypes.c_void_p),
            cc.ctypes.data_as(ctypes.c_void_p),
            sup.ctypes.data_as(ctypes.c_void_p),
            p_radii,
            ctypes.c_int(n_sup),
            ctypes.c_int(p1 - p0),
        )
        weights_all[p0:p1] /= pb.sum(axis=0)
        for ia in range(n0, n1):
            i0, i1 = offs[ia], offs[ia + 1]
            weights_all[i0:i1] *= pb[ia, i0 - p0 : i1 - p0]
    del table
    return coords_all, weights_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--dm", required=True)
    ap.add_argument("--grid", nargs=3, action="append", required=True)
    ap.add_argument("--xc", default="PBE")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    lib.num_threads(a.threads)
    atoms, lat = read_cell(a.cell)
    syms = [s for s, _ in atoms]
    cell = pgto.Cell()
    cell.a = lat
    cell.unit = "B"
    cell.atom = atoms
    cell.basis = ferric_basis_pyscf(BASIS[a.cell][0], syms, "segmented")
    cell.cart = False
    cell.precision = 1e-12
    cell.verbose = 0
    cell.build()
    dm = load_bin(a.dm)
    s_f = load_bin(a.dm + ".S")
    s_p = cell.pbc_intor("int1e_ovlp")
    print(
        f"nao {cell.nao} |S_pyscf - S_ferric|max {abs(s_p - s_f).max():.3e} "
        f"tr(D S_pyscf) {np.einsum('ij,ji', dm, s_p):.12f} rcut "
        f"{pbc_eval_gto._estimate_rcut(cell).max():.3f}"
    )
    mf = pdft.RKS(cell)
    ni = mf._numint
    for nr, na, var in a.grid:
        t = time.perf_counter()
        g = pgen.BeckeGrids(cell)
        g.atom_grid = (int(nr), int(na))
        g.prune = None
        if var == "stock":
            g.build(with_non0tab=True)
        elif var == "bragg":
            g.coords, g.weights = becke_grids(cell, g.atom_grid, adjust=True)
        elif var == "stockcopy":
            g.coords, g.weights = becke_grids(cell, g.atom_grid, adjust=False)
        else:
            sys.exit(f"unknown variant {var}")
        # numint rebuilds a grid whose non0tab is None, which would replace
        # the custom weights with the stock ones.
        g.non0tab = g.make_mask(cell, g.coords)
        n, exc, _ = ni.nr_rks(cell, g, a.xc, dm, kpt=np.zeros(3))
        print(
            f"PYSCF {nr}x{na} {var} npts={g.weights.size} wsum={g.weights.sum():.12f} "
            f"N={n:.12f} Exc={exc:.12f} wall={time.perf_counter() - t:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
