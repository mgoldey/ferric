"""Q1: how fast does the lattice-summed S(k) go singular as diffuse functions are added / the cell shrinks?

PREDICTIONS (written before the run):
  PHYSICS: S(k) of a diffuse function is its FT sampled at k+G, so lam_min(S(k)) falls like exp(-|k_ZB|^2/(2 amin))
  and is smallest at the zone-boundary corner (R), not at Gamma; aug (H s 0.0297, Li p 0.0058) is orders of
  magnitude worse than cc-pVDZ at the same a; shrinking a makes |k_ZB| = pi/a larger, i.e. conditioning WORSENS
  as the cell is compressed.  After diagonal normalisation the smallness of a lone Bloch function disappears,
  so lam_min(normalised) >> lam_min(raw) whenever the near-null vector is ~one AO.
  ARTIFACT: a truncated real-space sum would put a floor ~exp(-amin rcut^2/2) under lam_min (or make it
  negative) and would NOT reproduce the single-Gaussian Poisson closed form at R.  Anchor (a) distinguishes.
Usage: python3 run_lindep_spectrum.py [anchor|h2|lih|oracle]
"""

import sys

import numpy as np

sys.path.insert(0, ".")
import pbc_lindep as LD  # noqa: E402

H2_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
what = sys.argv[1] if len(sys.argv) > 1 else "all"


def lih(a):
    A = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    return A, [("Li", (0.0, 0.0, 0.0)), ("H", (0.5 * a, 0.0, 0.0))]


def sweep(label, cells, bases, n=(4, 4, 4)):
    for basis in bases:
        for a, (A, atoms) in cells:
            mol, Ls, S_L = LD.lattice_overlap(A, atoms, basis)
            k = LD.gamma_mesh(A, n)
            rep = LD.spectrum_report(LD.s_of_k(Ls, S_L, k))
            i = int(np.argmin(rep[:, 0]))
            g = rep[0]
            print(f"{label} {basis:12s} a={a:5.2f} nao={mol.nao:3d} nL={len(Ls):6d}: min_k lam(S)={rep[i, 0]:.2e} "
                  f"at k#{i} (frac {np.round(np.linalg.solve((2*np.pi*np.linalg.inv(A).T).T, k[i]), 2)}), "
                  f"Gamma lam={g[0]:.2e}; min_k lam(norm S)={np.nanmin(rep[:, 1]):.2e}; min diag S(k)={rep[:, 2].min():.2e}; "
                  f"dropped/k at 1e-6 {int(rep[:, 3].min())}..{int(rep[:, 3].max())}, at 1e-8 {int(rep[:, 4].min())}..{int(rep[:, 4].max())}, "
                  f"normalised 1e-6 {int(np.nan_to_num(rep[:, 5]).min())}..{int(np.nan_to_num(rep[:, 5]).max())}", flush=True)


if what in ("anchor", "all"):
    # (a) single unit s Gaussian per cubic cell: real-space lattice sum vs Poisson closed form, at Gamma, X, M, R
    for alpha, a in ((0.0297, 4.0), (0.0297, 3.0), (0.00864, 7.0)):
        basis = {"H": [[0, [alpha, 1.0]]]}
        A = np.eye(3) * a
        mol, Ls, S_L = LD.lattice_overlap(A, [("H", (0.0, 0.0, 0.0))], basis)
        fr = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
        k = fr @ (2 * np.pi * np.linalg.inv(A).T)
        num = LD.s_of_k(Ls, S_L, k)[:, 0, 0].real
        ref = LD.single_s_poisson(alpha, A, k)
        tot = np.abs(S_L).sum()
        print(f"anchor alpha={alpha} a={a}: nL={len(Ls)} sum|S_L|={tot:.1f}")
        for f, x, y in zip(fr, num, ref):
            print(f"   k={f}: lattice {x:.6e}  Poisson {y:.6e}  abs err {abs(x - y):.1e}  rel {abs(x - y) / y:.1e}")

if what in ("h2", "all"):
    sweep("H2 cubic", [(a, (np.eye(3) * a, H2_ATOMS)) for a in (3.0, 4.0, 5.0, 6.0, 8.0)], ["cc-pvdz", "aug-cc-pvdz"])

if what in ("lih", "all"):
    sweep("LiH fcc", [(a, lih(a)) for a in (6.5, 7.0, 7.72, 8.5, 10.0)], ["cc-pvdz", "aug-cc-pvdz"])

if what in ("oracle", "all"):
    # PySCF pbc S(k) as the independent oracle for the lattice sum (spherical, same k)
    from pyscf.pbc import gto as pgto

    for basis in ("aug-cc-pvdz",):
        A = np.eye(3) * 4.0
        c = pgto.M(a=A, atom=H2_ATOMS, basis=basis, unit="B", verbose=0, precision=1e-16)
        k = LD.gamma_mesh(A, (2, 2, 2))
        Sp = np.array(c.pbc_intor("int1e_ovlp", kpts=k))
        mol, Ls, S_L = LD.lattice_overlap(A, H2_ATOMS, basis)
        So = LD.s_of_k(Ls, S_L, k)
        print(f"oracle H2 a=4 {basis}: max|S_ours - S_pyscf| = {abs(So - Sp).max():.1e}; lam_min per k ours "
              f"{[f'{x:.3e}' for x in LD.spectrum_report(So)[:, 0]]} pyscf {[f'{np.linalg.eigvalsh(s)[0]:.3e}' for s in Sp]}")
