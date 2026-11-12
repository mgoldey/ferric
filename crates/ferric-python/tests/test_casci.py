"""`ferric.run_casci`: CAS-CI energy, CI vector, spin-summed RDMs, certificate.

H2O / 6-31G CAS(4,4) above 3 inactive orbitals (36 determinants), the
system of `crates/ferric-ci/tests/casci_rdms.rs`. The PySCF comparison of the
RDMs lives in that Rust test; these checks need no PySCF.

Anchor: `e_core + sum(h*dm1) + 0.5*sum(eri*dm2)` from the returned arrays
equals `e_total`. That sum is `<c|H|c>` for the normalized CI vector, so it
differs from the Davidson eigenvalue by at most the residual norm
(`conv_thresh`) plus rounding (~1e-13 here).

Mutation (2026-11-11): returning `eri_active` in physicist order
(`permuted_axes([0, 2, 1, 3])`) fails the anchor; the other tests pass.
"""

import numpy as np
import pytest

import ferric

_H2O = (
    "3\n\nO 0.000000 0.000000 0.117790\n"
    "H 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n"
)
CONV = 1e-10


@pytest.fixture(scope="module")
def res():
    mol = ferric.Molecule.from_xyz_string(_H2O)
    return ferric.run_casci(
        mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2), conv_thresh=CONV
    )


def test_energy_from_rdms_equals_the_eigenvalue(res):
    e = (
        res.e_core
        + np.einsum("pq,pq", res.h_active, res.dm1)
        + 0.5 * np.einsum("pqrs,pqrs", res.eri_active, res.dm2)
    )
    assert res.converged
    assert abs(e - res.e_total) < CONV + 1e-12
    assert abs(res.e_total - res.e_core - res.e_active) < 1e-12


def test_shapes_traces_and_window(res):
    assert res.active_start == 3
    assert res.n_determinants == 36
    c = res.ci_vector
    assert c.shape == (6, 6)
    assert abs(np.sum(c * c) - 1.0) < 1e-12
    assert res.dm1.shape == (4, 4) and res.dm2.shape == (4, 4, 4, 4)
    assert res.mo_coeff_active.shape[1] == 4
    assert abs(np.trace(res.dm1) - 4.0) < 1e-11
    assert abs(np.einsum("pprr", res.dm2) - 12.0) < 1e-11
    occ = np.linalg.eigvalsh(res.dm1)
    assert np.all(occ > -1e-12) and np.all(occ < 2.0 + 1e-12)
    assert res.e_total < res.rhf_energy


def test_certificate_is_ungraded_library_capability(res):
    c = res.certificate
    assert c["capability"] == "casci"
    assert c["grade"] == "ungraded"
    assert c["build"]["commit"] == ferric.build_info()["commit"]
    assert c["warnings"] == []


def test_rdms_are_opt_in():
    mol = ferric.Molecule.from_xyz_string(_H2O)
    r = ferric.run_casci(
        mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 2), compute_rdms=False
    )
    assert r.dm1 is None and r.dm2 is None


def test_bad_electron_count_is_an_error():
    mol = ferric.Molecule.from_xyz_string(_H2O)
    with pytest.raises(Exception, match="inactive"):
        ferric.run_casci(mol, ferric.BasisSet.bundled("6-31g"), 4, (2, 1))
