"""Static dipole polarizability by CPHF / CPKS written on ferric's blocks.

Coupled-perturbed Hartree-Fock (Gerratt, Mills, JCP 49, 1719 (1968); Pople,
Krishnan, Schlegel, Binkley, IJQC S13, 225 (1979)), in the form of PySCF's
``prop.polarizability.rhf``. A uniform field F adds h1 = F . r to the
one-electron Hamiltonian (the electron charge is -1). For each direction j
the orbital response U_j (nvir, nocc) solves

    (e_a - e_i) U + C_v^T F1(D1(U)) C_o = -C_v^T r_j C_o,

``AoIntegrals.cphf(...).solve``, with D1 = 2 (C_v U C_o^T + h.c.) the
first-order density. The dipole is mu_i = -tr[D r_i] + nuclear, so

    alpha_ij = d mu_i / d F_j = -tr[D1(U_j) r_i].

With ``xc`` the same code is CPKS: ``cphf`` adds the XC kernel and the
functional's exact-exchange fraction to F1. Bohr, Hartree, atomic units of
polarizability.

The check is a finite field on ferric's own SCF (``run_rhf`` /
``run_dft(external_field=)``): alpha_nn = -d2E/dF_n^2 by central differences,
Richardson-extrapolated in the step, along six directions that fix the
symmetric tensor.

Run as a script: water/cc-pVDZ RHF, asserting the tensor equals the finite
field to 1e-5 a.u. and PySCF's CPHF equations solved densely to 1e-8 a.u.
(PySCF's own Krylov solve, ``scf.cphf.solve``, stops up to 2.5e-7 a.u. short
of that solution; see scripts/gen_pyscf_block_ports_set2_refs.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rhf_from_blocks import ROOT, WATER  # noqa: E402

PYSCF_REF = ROOT / "testdata" / "reference" / "block_ports_set2_pyscf.json"

EXACT_JK = {"df_j_aux": "exact", "df_k_aux": "exact"}
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}
# Unit vectors whose second derivatives -d2E/dF^2 = n^T alpha n fix alpha.
DIRECTIONS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def polarizability(
    ints: ferric.AoIntegrals,
    mo_coeff: np.ndarray,
    mo_energy: np.ndarray,
    nocc: int,
    xc: ferric.XcFunctional | None = None,
) -> np.ndarray:
    """The static polarizability tensor (3, 3) of a closed-shell SCF."""
    r = ints.dipole()
    co, cv = mo_coeff[:, :nocc], mo_coeff[:, nocc:]
    response = ints.cphf(mo_coeff, list(mo_energy), nocc, xc=xc)
    d1 = [
        response.density(response.solve(-cv.T @ r[j] @ co, tol=1e-10)) for j in range(3)
    ]
    return np.array([[-np.sum(d1[j] * r[i]) for j in range(3)] for i in range(3)])


def _direction(i: int, j: int) -> np.ndarray:
    n = np.zeros(3)
    n[i] += 1.0
    n[j] += 1.0
    return n / np.linalg.norm(n)


def finite_field(
    mol: ferric.Molecule, bs: ferric.BasisSet, functional: str | None, h: float = 0.005
) -> np.ndarray:
    """alpha from -d2E/dF^2 along DIRECTIONS, steps h and 2h, Richardson."""

    def energy(field):
        kw = {"external_field": tuple(field), **EXACT_JK, **TIGHT}
        if functional is None:
            return ferric.run_rhf(mol, bs, **kw).energy
        return ferric.run_dft(mol, bs, functional=functional, **kw).total_energy

    e0 = energy(np.zeros(3))
    curv = {}
    for i, j in DIRECTIONS:
        n = _direction(i, j)
        c = [
            (energy(k * h * n) + energy(-k * h * n) - 2.0 * e0) / (k * h) ** 2
            for k in (1, 2)
        ]
        curv[i, j] = -(4.0 * c[0] - c[1]) / 3.0
    alpha = np.diag([curv[k, k] for k in range(3)])
    for i, j in DIRECTIONS[3:]:
        # n^T alpha n with n = (e_i + e_j)/sqrt(2) is (a_ii + a_jj)/2 + a_ij.
        alpha[i, j] = alpha[j, i] = curv[i, j] - 0.5 * (curv[i, i] + curv[j, j])
    return alpha


def main() -> None:
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    ints = ferric.AoIntegrals(mol, bs)
    scf = ferric.run_rhf(mol, bs, **EXACT_JK, **TIGHT)
    alpha = polarizability(
        ints, scf.mo_coefficients(), scf.orbital_energies(), ints.nelec // 2
    )
    ff = finite_field(mol, bs, None)
    print("CPHF alpha (a.u.)\n", np.array2string(alpha, precision=6))
    print(f"finite field max diff {np.abs(alpha - ff).max():.1e}")
    assert np.abs(alpha - ff).max() < 1e-5
    if PYSCF_REF.exists():
        ref = json.loads(PYSCF_REF.read_text())["molecules"]["water"]["polarizability"]
        diff = np.abs(alpha - np.asarray(ref["alpha_dense"])).max()
        print(f"PySCF max diff        {diff:.1e}")
        assert diff < 1e-8


if __name__ == "__main__":
    main()
