"""Three PySCF-style methods written in numpy on the `AoIntegrals` blocks.

The methods live in examples/python/ (RHF, RI-MP2, CIS); each is checked
against an independent construction, ferric's own Rust driver:

* `rhf_from_blocks.rhf` (DIIS on hcore() + jk()) equals `run_rhf`;
* `rimp2_from_blocks.rimp2_corr` (df_mo B^P_ia) equals `run_rimp2`;
* `cis_from_blocks.cis_singlets` on df_mo integrals equals
  `run_tddft(functional=None, method="tda")`, which builds and solves its A
  matrix in Rust (`ferric_tddft`). Both run on the orbitals of `run_rhf` at
  its default thresholds, the SCF `run_tddft` runs internally.

HOW THESE FAIL IF BROKEN: K without the 0.5 (or J/K swapped) moves the RHF
energy by Hartrees; dropping the MP2 exchange term or its factor 2 moves
E_corr by >1e-2; a CIS A matrix with (ia|jb) instead of 2(ia|jb) (the triplet
or a mis-spin-adapted singlet) or with (ib|ja) for (ij|ab) moves the roots by
>1e-2; a transposed B tensor raises or mismatches.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples" / "python"
WATER = ROOT / "testdata" / "molecules" / "water.xyz"
NAMES = ["rhf_from_blocks", "rimp2_from_blocks", "cis_from_blocks"]


def _load(name: str):
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


rhf_mod, mp2_mod, cis_mod = (_load(n) for n in NAMES)


def _system(basis: str):
    mol = ferric.Molecule.from_xyz(str(WATER))
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


@pytest.fixture(scope="module")
def water_dz():
    mol, bs, ints = _system("cc-pvdz")
    return mol, bs, ints, rhf_mod.rhf(ints), ferric.BasisSet.bundled("cc-pvdz-ri")


@pytest.mark.parametrize("basis", ["sto-3g", "cc-pvdz"])
def test_rhf_port_matches_run_rhf(basis):
    mol, bs, ints = _system(basis)
    mf = rhf_mod.rhf(ints)
    ref = ferric.run_rhf(mol, bs, energy_conv=1e-11, density_conv=1e-9)
    assert abs(mf["e_tot"] - ref.energy) < 1e-8
    np.testing.assert_allclose(mf["mo_energy"], ref.orbital_energies(), atol=1e-7)
    assert np.trace(mf["dm"] @ ints.overlap()) == pytest.approx(ints.nelec, abs=1e-10)


def test_rimp2_port_matches_run_rimp2(water_dz):
    mol, bs, ints, mf, aux = water_dz
    e_corr = mp2_mod.rimp2_corr(ints, aux, mf)
    ref = ferric.run_rimp2(mol, bs, aux, frozen_core=0).mp2_corr
    assert abs(e_corr - ref) < 1e-8


def test_cis_port_matches_run_tddft_tda(water_dz):
    mol, bs, ints, _, aux = water_dz
    mf = cis_mod.orbitals_of(ferric.run_rhf(mol, bs), ints.nelec // 2)
    w = cis_mod.cis_singlets(ints, mf, 4, aux)
    ref = ferric.run_tddft(mol, bs, aux, functional=None, n_roots=4, method="tda")
    np.testing.assert_allclose(
        w, np.asarray(ref.excitation_energies), rtol=0, atol=1e-8
    )


def test_exact_and_fitted_cis_differ_only_by_the_fitting_error(water_dz):
    # Same A-matrix code on two integral sources: they agree to the RI error
    # (~1e-4 Ha at cc-pVDZ-RI) and are not identical, so the `aux=None` branch
    # really uses ao2mo.
    _, _, ints, mf, aux = water_dz
    diff = np.abs(
        cis_mod.cis_singlets(ints, mf, 4) - cis_mod.cis_singlets(ints, mf, 4, aux)
    )
    assert 1e-8 < diff.max() < 1e-3


@pytest.mark.parametrize("name", NAMES)
def test_example_script_runs(name):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / f"{name}.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
