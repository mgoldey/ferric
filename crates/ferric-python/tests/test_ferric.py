import ferric
import json
import os

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testdata")


def test_molecule_from_xyz():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    assert mol.natoms() == 3
    assert mol.nelec() == 10
    vnn = mol.nuclear_repulsion()
    assert abs(vnn - 9.189193229309746) < 1e-6


def test_basis_bundled():
    bs = ferric.BasisSet.bundled("sto-3g")
    assert bs is not None


def test_rhf_water_sto3g():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    result = ferric.run_rhf(mol, bs)
    assert result.converged
    ref_path = os.path.join(TESTDATA, "reference", "h2o_sto-3g_rhf.json")
    with open(ref_path) as f:
        ref = json.load(f)
    assert abs(result.energy - ref["energy"]) < 5e-8, (
        f"got {result.energy:.10f}, ref {ref['energy']:.10f}"
    )


def test_rhf_result_arrays():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    result = ferric.run_rhf(mol, bs)
    D = result.density()
    assert D.shape == (7, 7)
    eps = result.orbital_energies()
    assert len(eps) == 7
