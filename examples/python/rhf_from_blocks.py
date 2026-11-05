"""Closed-shell RHF written in numpy on ferric's integral blocks.

The PySCF pattern (``mf.get_hcore()``, ``mf.get_jk(dm)``, DIIS on the
commutator FDS - SDF) on ``ferric.AoIntegrals``: ``overlap()``, ``hcore()``
and ``jk(D)``. ``jk`` takes the TOTAL density and returns (J, K), so the
closed-shell Fock matrix is ``hcore + J - 0.5 K``. Bohr and Hartree.

Run as a script: converges water/cc-pVDZ and asserts the energy equals
``ferric.run_rhf`` to 1e-8 Ha (and PySCF to the same bar when it is installed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ferric

ROOT = Path(__file__).resolve().parents[2]
WATER = ROOT / "testdata" / "molecules" / "water.xyz"


def orthogonalizer(s: np.ndarray, lindep: float = 1e-8) -> np.ndarray:
    """Canonical orthogonalizer X with X^T S X = 1, dropping S eigenvalues < lindep."""
    w, v = np.linalg.eigh(s)
    keep = w > lindep
    return v[:, keep] / np.sqrt(w[keep])


def rhf(
    ints: ferric.AoIntegrals,
    e_conv: float = 1e-11,
    d_conv: float = 1e-9,
    max_iter: int = 100,
    diis_size: int = 8,
) -> dict:
    """Closed-shell RHF with Pulay DIIS from a core-Hamiltonian guess.

    Returns ``e_tot`` (Hartree), ``mo_energy``, ``mo_coeff`` (nao, nmo),
    ``mo_occ`` (2/0) and the total density ``dm`` (nao, nao).
    """
    s, h = ints.overlap(), ints.hcore()
    x = orthogonalizer(s)
    nocc = ints.nelec // 2

    def diagonalize(f):
        e, c = np.linalg.eigh(x.T @ f @ x)
        return e, x @ c

    e_mo, c = diagonalize(h)
    dm = 2.0 * c[:, :nocc] @ c[:, :nocc].T
    e_old, focks, errors = 0.0, [], []
    for _ in range(max_iter):
        j, k = ints.jk(dm)
        f = h + j - 0.5 * k
        e_tot = 0.5 * np.sum(dm * (h + f)) + ints.nuclear_repulsion()
        err = x.T @ (f @ dm @ s - s @ dm @ f) @ x
        focks, errors = (focks + [f])[-diis_size:], (errors + [err])[-diis_size:]
        n = len(focks)
        b = -np.ones((n + 1, n + 1))
        b[n, n] = 0.0
        b[:n, :n] = [[np.sum(ei * ej) for ej in errors] for ei in errors]
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        coef = np.linalg.solve(b, rhs)[:n]
        e_mo, c = diagonalize(sum(ci * fi for ci, fi in zip(coef, focks)))
        dm_new = 2.0 * c[:, :nocc] @ c[:, :nocc].T
        d_change = np.abs(dm_new - dm).max()
        dm = dm_new
        if abs(e_tot - e_old) < e_conv and d_change < d_conv:
            break
        e_old = e_tot
    else:
        raise RuntimeError(f"RHF did not converge in {max_iter} iterations")
    # Canonical orbitals and energy of the final density.
    j, k = ints.jk(dm)
    f = h + j - 0.5 * k
    e_mo, c = diagonalize(f)
    occ = np.zeros(c.shape[1])
    occ[:nocc] = 2.0
    return {
        "e_tot": 0.5 * np.sum(dm * (h + f)) + ints.nuclear_repulsion(),
        "mo_energy": e_mo,
        "mo_coeff": c,
        "mo_occ": occ,
        "dm": dm,
    }


def pyscf_rhf_energy(xyz: Path, basis: str) -> float | None:
    """PySCF's RHF energy for the same molecule, or None without PySCF."""
    try:
        from pyscf import gto, scf
    except ImportError:
        return None
    atoms = xyz.read_text().splitlines()[2:]
    mol = gto.M(atom="\n".join(atoms), basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    return mf.kernel()


def main() -> None:
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    mf = rhf(ferric.AoIntegrals(mol, bs))
    ref = ferric.run_rhf(mol, bs, energy_conv=1e-11, density_conv=1e-9).energy
    print(f"numpy RHF   {mf['e_tot']:.10f}")
    print(f"run_rhf     {ref:.10f}   diff {mf['e_tot'] - ref:.1e}")
    assert abs(mf["e_tot"] - ref) < 1e-8
    e_pyscf = pyscf_rhf_energy(WATER, "cc-pvdz")
    if e_pyscf is not None:
        print(f"PySCF       {e_pyscf:.10f}   diff {mf['e_tot'] - e_pyscf:.1e}")
        assert abs(mf["e_tot"] - e_pyscf) < 1e-8


if __name__ == "__main__":
    main()
