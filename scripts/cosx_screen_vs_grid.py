#!/usr/bin/env python3
"""COMPOSITION CHECK: does the screening benefit survive grid coarsening?

THE QUESTION
------------
The retracted budget multiplies a screening factor (~2x, measured at grid
(50,110) on alkane_4..20) by a grid factor (right-sizing to a COARSER grid).
Those two were measured at DIFFERENT grids, so multiplying them assumes the
screening fraction is grid-INDEPENDENT.  That assumption has never been
tested, and there is a concrete reason to doubt it:

  A coarser Becke-Lebedev grid removes points preferentially from the
  DIFFUSE outer radial shells -- exactly the region where a grid point is
  far from most shell pairs and therefore where screening does its work.
  If the surviving points are disproportionately the near-nuclear ones,
  where many pairs are significant, the surviving-pair FRACTION goes UP as
  the grid gets coarser and the two factors ANTI-COMPOSE.

ARTIFACT / OUTCOME HYPOTHESIS (pre-registered, before running)
--------------------------------------------------------------
  COMPOSES:      fraction is flat (within a few %) across the grid series.
                 Then screening_gain x grid_gain is legitimate.
  ANTI-COMPOSES: fraction RISES as the grid coarsens.  Then the product
                 OVERSTATES the benefit and the two factors partly cancel.
  (A FALLING fraction on coarsening would be a bonus, but there is no
  mechanism for it -- if seen, suspect a bug before believing it.)

SCREEN
------
The COSX per-point screen keeps a shell pair (mu,nu) at grid point g when its
estimated contribution exceeds a threshold.  The standard estimate is the
product of the AO magnitudes at that point times the Coulomb factor; here we
use the directly measurable proxy actually used by the Rust Stage 2 sweep:

    keep (mu,nu) at g   iff   max_g |chi_mu(r_g)| * |chi_nu(r_g)| > thresh

evaluated per grid point, then the fraction is over (points x pairs).  This
is the SAME quantity the Rust sweep screened on, so the numbers are
comparable; the point here is the GRID DEPENDENCE, not the absolute level.

THRESHOLD is fixed at the Stage 2 production value 1e-7 BEFORE measuring and
is not tuned.
"""

import numpy as np
from pyscf import gto, dft

THRESH = 1e-7          # fixed before measuring; NOT tuned (Stage 2 value)
GRIDS = [(25, 50), (35, 86), (50, 110), (75, 194)]


def load_xyz(path):
    lines = open(path).read().strip().splitlines()
    return "\n".join(lines[2:])


def frac_kept(mol, nrad, nang, thresh=THRESH):
    """Fraction of (grid point, shell pair) combinations surviving the screen.

    Works shell-wise (not AO-wise): the screen in a real implementation is a
    SHELL-pair screen, because integrals are computed per shell quartet/pair.
    """
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = (nrad, nang)
    g.prune = None
    g.build()
    coords, npts = g.coords, len(g.weights)

    # Per-shell max |chi| at each grid point, chunked to bound memory.
    nsh = mol.nbas
    ao_slices = mol.aoslice_by_atom()  # unused; keep shell slices below
    shell_lo = [mol.ao_loc_nr()[i] for i in range(nsh)]
    shell_hi = [mol.ao_loc_nr()[i + 1] for i in range(nsh)]

    kept = 0
    total = 0
    chunk = 4000
    for lo in range(0, npts, chunk):
        hi = min(lo + chunk, npts)
        ao = dft.numint.eval_ao(mol, coords[lo:hi])        # (nc, nbf)
        # shell magnitude: max over the AOs in that shell
        smag = np.empty((hi - lo, nsh))
        for s in range(nsh):
            smag[:, s] = np.abs(ao[:, shell_lo[s]:shell_hi[s]]).max(axis=1)
        # pair estimate = outer product per point; count upper triangle
        # without materializing (nc, nsh, nsh) for large nsh.
        for c in range(hi - lo):
            v = smag[c]
            est = np.outer(v, v)
            iu = np.triu_indices(nsh)
            kept += int((est[iu] > thresh).sum())
            total += len(iu[0])
        del ao, smag
    return kept / total, npts, nsh


def main():
    import sys
    base = "/home/matt/qc/ferric/.claude/worktrees/cosx/testdata/molecules"
    systems = sys.argv[1:] or ["alkane_4", "alkane_8", "alkane_12"]
    basis = "cc-pvdz"

    print("COMPOSITION CHECK: screening pair fraction vs GRID COARSENESS")
    print(f"threshold {THRESH:.0e} (fixed before measuring), basis {basis}")
    print()
    print(f"{'system':>10} {'nsh':>5} " +
          "".join(f"{f'{a}x{b}':>12}" for a, b in GRIDS))
    print("-" * (16 + 12 * len(GRIDS)))

    for name in systems:
        mol = gto.M(atom=load_xyz(f"{base}/{name}.xyz"), basis=basis,
                    unit="Angstrom", verbose=0)
        fr = []
        for (nr, na) in GRIDS:
            f, npts, nsh = frac_kept(mol, nr, na)
            fr.append(f)
        print(f"{name:>10} {mol.nbas:5d} " +
              "".join(f"{f:11.4f}" for f in fr))
        # verdict per system
        coarse, fine = fr[0], fr[-1]
        rel = (coarse - fine) / fine * 100
        if abs(rel) < 5:
            v = "COMPOSES (flat within 5%)"
        elif rel > 0:
            v = f"ANTI-COMPOSES (coarse grid keeps {rel:+.1f}% MORE pairs)"
        else:
            v = f"favourable ({rel:+.1f}%) -- suspect a bug, verify"
        print(f"{'':>10} -> {v}")


if __name__ == "__main__":
    main()
