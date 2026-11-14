"""CASSCF / SC-NEVPT2 on the density-fitted route through the Python bindings.

The RI route's numerical anchors (RI == dense fed the RI-reconstructed AO
tensor; PySCF ``density_fit()`` CASSCF/NEVPT2) are Rust tests in
``crates/ferric-ci/tests/casscf_ri.rs``. These tests cover what the binding
adds:

* The kwargs reach the driver: ``casscf_integrals="ri"`` with
  ``casscf_auxbasis`` gives a different converged energy from the exact route,
  by less than 1e-3 Ha (water/6-31G with def2-universal-jkfit is ~1e-5 Ha;
  the bound only catches a route that was not taken, |dE| < 1e-7, or a wrong
  Hamiltonian, |dE| of tenths of Ha).
* The result names its route (``integrals``, ``auxbasis``), and the
  certificate records the ``ri`` approximation and is capped below Proven.
* Strict parsing: ``auxbasis`` without ``ri``, ``ri`` without ``auxbasis``.
* FCIDUMP: ``fcidump.write`` always transforms the EXACT AO integrals, so a
  file from RI orbitals holds the exact Hamiltonian at those orbitals. The
  CAS-CI energy of that file is variational in the orbitals: it is at or
  above the exact CASSCF energy (bar -1e-9 for the two CI solves) and above
  it by a second-order term only, far below the RI energy shift (bar: 10% of
  |E_RI - E_exact|).
* Molden natural orbitals of an RI result: the occupations sum to the
  electron count (they read only ``mo_coeff`` and ``dm1``).
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric
from ferric import fcidump, molden

H2O = (
    "3\n\nO 0.000000 0.000000 0.117790\n"
    "H 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n"
)
AUX = "def2-universal-jkfit"
NACT, NEL = 4, (2, 2)


@pytest.fixture(scope="module")
def runs():
    mol, bs = ferric.Molecule.from_xyz_string(H2O), ferric.BasisSet.bundled("6-31g")
    exact = ferric.run_casscf(mol, bs, NACT, NEL, canonicalize=True)
    ri = ferric.run_casscf(
        mol,
        bs,
        NACT,
        NEL,
        canonicalize=True,
        casscf_integrals="ri",
        casscf_auxbasis=AUX,
    )
    assert exact.converged and ri.converged
    return mol, bs, exact, ri


def test_ri_route_is_taken_and_named(runs):
    _, _, exact, ri = runs
    d = ri.e_total - exact.e_total
    print(f"RI minus exact CASSCF: {d:.3e}")
    assert 1e-7 < abs(d) < 1e-3
    assert (exact.integrals, exact.auxbasis) != ("ri", AUX)
    assert (ri.integrals, ri.auxbasis) == ("ri", AUX)
    cert = ri.certificate
    assert cert["grade"] == "proven_narrow", cert
    assert {"name": "ri", "detail": f"aux {AUX}"} in cert["approximations"], cert
    assert exact.certificate["grade"] == "proven"


def test_ri_parsing_is_strict(runs):
    mol, bs, _, _ = runs
    with pytest.raises(RuntimeError, match="needs auxbasis"):
        ferric.run_casscf(mol, bs, NACT, NEL, casscf_integrals="ri")
    with pytest.raises(RuntimeError, match="read by integrals"):
        ferric.run_casscf(mol, bs, NACT, NEL, casscf_auxbasis=AUX)
    with pytest.raises(RuntimeError, match="not a bundled basis"):
        ferric.run_casscf(
            mol, bs, NACT, NEL, casscf_integrals="ri", casscf_auxbasis="no-such-aux"
        )


def test_ri_nevpt2_runs_on_the_ri_reference(runs):
    mol, bs, _, ri = runs
    r = ferric.run_nevpt2(
        mol, bs, NACT, NEL, casscf_integrals="ri", casscf_auxbasis=AUX
    )
    assert r.e_reference == pytest.approx(ri.e_total, abs=1e-8)
    assert r.e_corr < 0.0
    with pytest.raises(RuntimeError, match="CAS-CI reference"):
        ferric.run_nevpt2(
            mol,
            bs,
            NACT,
            NEL,
            reference="casci",
            casscf_integrals="ri",
            casscf_auxbasis=AUX,
        )


def test_fcidump_of_an_ri_result_holds_the_exact_hamiltonian(runs, tmp_path):
    pytest.importorskip("pyscf")
    from pyscf import fci
    from pyscf.tools import fcidump as pfcidump

    mol, bs, exact, ri = runs
    path = tmp_path / "ri.fcidump"
    fcidump.write(path, mol, bs, ri.mo_coeff, ri.n_core, NACT, sum(NEL), 0)
    d = pfcidump.read(str(path))
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    e, _ = solver.kernel(d["H1"], d["H2"], NACT, NEL, ecore=d["ECORE"])
    above = e - exact.e_total
    shift = abs(ri.e_total - exact.e_total)
    print(f"file CAS-CI minus exact CASSCF {above:.3e}; RI shift {shift:.3e}")
    assert -1e-9 < above < 0.1 * shift


def test_molden_natural_orbitals_of_an_ri_result(runs):
    mol, bs, _, ri = runs
    occ = molden.orbitals(mol, bs, ri)[0]["occupations"]
    assert occ.sum() == pytest.approx(2 * ri.n_core + sum(NEL), abs=1e-10)
    assert np.all(occ[: ri.n_core] == 2.0)
