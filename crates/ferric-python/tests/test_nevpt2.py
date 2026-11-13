"""`ferric.run_nevpt2`: SC-NEVPT2 through the binding.

H2O / 6-31G CAS(4,4) above 3 core orbitals (`testdata/molecules/water.xyz`).
The construction is anchored in `crates/ferric-ci/tests/nevpt2.rs` (numpy
prototype to rounding, PySCF `mrpt.NEVPT` per class, exact zeros for empty
external spaces, the MP2 limit); these checks cover the binding: the classes
are PySCF's to the bar the Rust test derives, the arrays are consistent, and
bad input is refused.

Class bar (as in the Rust test): |dE_class/dkappa| times the orbital distance
between two converged CASSCF references, (|g_ferric| + |g_pyscf|) / lambda_min,
plus twice the class spread under rotations inside degenerate orbital groups,
plus 1e-10 Ha for rounding. All inputs come from
`testdata/reference/nevpt2_refs.json` (`scripts/gen_nevpt2_refs.py`).
"""

import json
from pathlib import Path

import numpy as np
import pytest

import ferric

ROOT = Path(__file__).resolve().parents[3]
NAMES = ("Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir")


def _ref():
    data = json.loads((ROOT / "testdata/reference/nevpt2_refs.json").read_text())
    return next(s for s in data["systems"] if s["system"] == "h2o")


@pytest.fixture(scope="module")
def ref():
    return _ref()


@pytest.fixture(scope="module")
def mol():
    return ferric.Molecule.from_xyz(str(ROOT / "testdata/molecules/water.xyz"))


@pytest.fixture(scope="module")
def res(ref, mol):
    return ferric.run_nevpt2(
        mol,
        ferric.BasisSet.bundled("6-31g"),
        4,
        (2, 2),
        casscf_grad_tol=ref["ferric_casscf"]["casscf_grad_tol"],
    )


def test_classes_match_pyscf(ref, res):
    p = ref["pyscf"]
    dk = (res.reference_grad_norm + p["grad_norm"]) / p["lambda_min"]
    for name in NAMES:
        bar = p["dE_dkappa"][name] * dk + 2.0 * p["degenerate_spread"][name] + 1e-10
        assert abs(res.energies[name] - p["classes"][name]) < bar, name
    assert abs(res.e_corr - sum(res.energies.values())) < 1e-14
    assert abs(res.e_total - (res.e_reference + res.e_corr)) < 1e-12
    assert res.e_corr < 0.0


def test_arrays_and_certificate(res):
    assert (res.n_core, res.n_active) == (3, 4)
    nmo = res.mo_coeff.shape[1]
    assert res.n_virtual == nmo - 7
    assert res.mo_energy.shape == (nmo,)
    eps = res.mo_energy
    assert np.all(np.diff(eps[:3]) >= 0.0) and np.all(np.diff(eps[7:]) >= 0.0)
    assert set(res.norms) == set(NAMES)
    assert all(v >= 0.0 for v in res.norms.values())
    assert res.ci_residual_norm < 1e-8
    cert = res.certificate
    assert cert["capability"] == "nevpt2"
    assert cert["grade"] == "proven"
    assert cert["build"]["commit"] == ferric.build_info()["commit"]


def test_casci_reference(mol, res):
    """On RHF orbitals the reference is CAS-CI, above the CASSCF energy."""
    r = ferric.run_nevpt2(
        mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2), reference="casci"
    )
    assert r.reference == "casci"
    assert r.certificate["grade"] == "ungraded"
    assert r.e_reference > res.e_reference
    assert r.e_corr < 0.0


def test_bad_input_is_refused(mol):
    bs = ferric.BasisSet.bundled("6-31g")
    with pytest.raises(Exception, match="reference must be"):
        ferric.run_nevpt2(mol, bs, 4, (2, 2), reference="rhf")
    with pytest.raises(Exception, match="open-shell"):
        ferric.run_nevpt2(mol, bs, 4, (3, 1))
    with pytest.raises(Exception, match="converge"):
        ferric.run_nevpt2(mol, bs, 4, (2, 2), casscf_max_iter=2)
