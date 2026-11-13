"""`ferric.qcschema.run`: QCSchema AtomicInput -> AtomicResult.

Anchors:

* Energy and gradient through ``run`` equal the direct ferric call with the same
  settings bit for bit (same code path, no arithmetic in between).
* Units: the geometry is Bohr and the gradient Hartree/Bohr. The nuclear
  repulsion in the output equals the independent sum Z_i Z_j / r_ij over the
  Bohr input (a factor 1.8897 geometry mix-up changes it by ~89%), and a
  central finite difference of ``run`` energies at +-h Bohr reproduces the
  returned gradient. Bar: FD_BAR = 2e-6 Ha/Bohr at h = 1.89e-3 Bohr, the bar
  derived in test_python_gradients.py (h^2 |E'''| / 6 + delta_E / h, about
  6e-7 + 5e-9 for these systems; ~3x margin) with energies converged to
  1e-10 Ha.
* QCElemental round trip (skipped without it): an AtomicInput it builds is
  accepted and the output parses as an AtomicResult, gradient shape (natom, 3).
* Refusals return ``success=False`` with a stable ``error_type`` and never raise.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")
try:
    import ferric.qcschema as qcs
except ModuleNotFoundError as err:  # the extension without the python package layer
    pytest.skip(f"ferric.qcschema is not importable: {err}", allow_module_level=True)

TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-9}
FD_BAR = 2e-6
H = 1.89e-3
# Bohr, deliberately not symmetric: every gradient component is nonzero.
WATER_BOHR = [0.02, 0.04, 0.22, 0.05, 1.43, -0.88, -0.04, -1.42, -0.85]
SYMBOLS = ["O", "H", "H"]
Z = {"O": 8, "H": 1}


def _inp(
    method="hf", basis="sto-3g", driver="energy", keywords=None, geometry=None, **mol
):
    molecule = {
        "symbols": SYMBOLS,
        "geometry": list(WATER_BOHR if geometry is None else geometry),
    }
    molecule.update(mol)
    return {
        "schema_name": "qcschema_input",
        "schema_version": 1,
        "molecule": molecule,
        "driver": driver,
        "model": {"method": method, "basis": basis},
        "keywords": dict(keywords or {}),
    }


def _mol(geometry=WATER_BOHR, units="bohr", **kw):
    return ferric.Molecule.from_arrays(
        SYMBOLS, np.asarray(geometry).reshape(-1, 3), units=units, **kw
    )


def _bs(name):
    return ferric.BasisSet.bundled(name)


def test_hf_energy_and_gradient_equal_the_direct_call_bitwise():
    e = qcs.run(_inp(driver="energy"))
    assert e["success"] is True
    assert e["schema_name"] == "qcschema_output"
    direct = ferric.run_rhf(_mol(), _bs("sto-3g"))
    assert e["return_result"] == direct.energy
    assert e["properties"]["return_energy"] == direct.energy
    assert e["properties"]["scf_total_energy"] == direct.energy

    g = qcs.run(_inp(driver="gradient"))
    dg = ferric.run_rhf(_mol(), _bs("sto-3g"), with_gradient=True)
    assert g["return_result"] == dg.gradient().tolist()
    assert g["properties"]["return_gradient"] == dg.gradient().tolist()
    assert g["properties"]["return_energy"] == dg.energy
    assert np.shape(g["return_result"]) == (3, 3)


def test_uhf_dft_and_mp2_equal_the_direct_call_bitwise():
    u = qcs.run(_inp(driver="gradient", molecular_charge=1, molecular_multiplicity=2))
    du = ferric.run_uhf(
        _mol(charge=1, multiplicity=2), _bs("sto-3g"), with_gradient=True
    )
    assert u["return_result"] == du.gradient().tolist()
    assert (
        u["properties"]["calcinfo_nalpha"] == 5
        and u["properties"]["calcinfo_nbeta"] == 4
    )

    d = qcs.run(_inp(method="PBE", basis="6-31g", driver="gradient"))
    dd = ferric.run_dft(_mol(), _bs("6-31g"), functional="pbe", with_gradient=True)
    assert d["return_result"] == dd.gradient().tolist()
    assert d["properties"]["return_energy"] == dd.total_energy
    assert d["properties"]["scf_xc_energy"] == dd.e_xc

    m = qcs.run(_inp(method="mp2", basis="cc-pvdz", driver="gradient"))
    dm = ferric.run_rimp2(_mol(), _bs("cc-pvdz"), _bs("cc-pvdz-ri"), with_gradient=True)
    assert m["return_result"] == dm.gradient().tolist()
    p = m["properties"]
    assert p["return_energy"] == dm.total_energy == p["mp2_total_energy"]
    assert p["scf_total_energy"] == dm.rhf_energy
    assert p["mp2_correlation_energy"] == dm.mp2_corr
    assert "calcinfo_nbasis" not in p


def test_units_bohr_in_and_hartree_per_bohr_out():
    out = qcs.run(_inp(driver="gradient", keywords={**TIGHT, "ri_j_precision": "f64"}))
    xyz = np.asarray(WATER_BOHR).reshape(3, 3)
    z = [Z[s] for s in SYMBOLS]
    vnn = sum(
        z[i] * z[j] / np.linalg.norm(xyz[i] - xyz[j])
        for i, j in itertools.combinations(range(3), 2)
    )
    assert out["properties"]["nuclear_repulsion_energy"] == pytest.approx(
        vnn, rel=1e-12
    )
    # The Angstrom reading of the same numbers is a different molecule.
    ang = ferric.run_rhf(_mol(units="angstrom"), _bs("sto-3g"), **TIGHT).energy
    # measured difference 0.445 Ha
    assert abs(ang - out["properties"]["return_energy"]) > 0.1

    grad = np.asarray(out["return_result"])
    fd = np.zeros_like(grad)
    for a, c in itertools.product(range(3), range(3)):
        e = []
        for sign in (+1, -1):
            geo = xyz.copy()
            geo[a, c] += sign * H
            r = qcs.run(
                _inp(geometry=geo.ravel(), keywords={**TIGHT, "ri_j_precision": "f64"})
            )
            e.append(r["return_result"])
        fd[a, c] = (e[0] - e[1]) / (2 * H)
    assert np.abs(fd - grad).max() < FD_BAR


def test_geometry_may_be_nested_and_multiplicity_defaults_from_electrons():
    flat = qcs.run(_inp())
    nested = qcs.run(_inp(geometry=np.asarray(WATER_BOHR).reshape(3, 3).tolist()))
    assert nested["return_result"] == flat["return_result"]
    radical = qcs.run(_inp(molecular_charge=1))
    assert radical["success"] is True
    assert radical["properties"]["calcinfo_nalpha"] == 5


def test_properties_and_provenance():
    out = qcs.run(_inp())
    p = out["properties"]
    assert p["calcinfo_natom"] == 3
    assert p["calcinfo_nbasis"] == p["calcinfo_nmo"] == 7
    assert p["calcinfo_nalpha"] == p["calcinfo_nbeta"] == 5
    assert p["scf_iterations"] >= 1
    assert out["provenance"]["creator"] == "ferric"
    assert out["provenance"]["version"] == ferric.__version__
    fx = out["extras"]["ferric"]
    assert set(fx["build"]) >= {"commit", "dirty"}
    assert fx["certificate"]["capability"] == "rhf"


def _failed(out, error_type, needle=""):
    assert out["success"] is False
    assert out["error"]["error_type"] == error_type, out["error"]
    assert needle in out["error"]["error_message"]
    assert "input_data" in out


def test_refusals_have_stable_error_types():
    _failed(qcs.run(_inp(keywords={"scf_type": "df"})), "input_error", "scf_type")
    # A keyword valid for hf is not valid for mp2.
    _failed(
        qcs.run(_inp(method="mp2", basis="cc-pvdz", keywords={"diis_size": 4})),
        "input_error",
        "diis_size",
    )
    _failed(qcs.run(_inp(driver="hessian")), "unsupported_error", "hessian")
    _failed(qcs.run(_inp(driver="optimization")), "input_error")
    _failed(
        qcs.run(_inp(method="pbe", molecular_charge=1, molecular_multiplicity=2)),
        "unsupported_error",
        "open-shell",
    )
    _failed(qcs.run(_inp(basis="no-such-basis")), "input_error", "no-such-basis")
    _failed(qcs.run(_inp(method="not-a-functional")), "input_error")
    _failed(qcs.run(_inp(method="mp2", basis="sto-3g")), "input_error", "auxbasis")
    wfn = _inp()
    wfn["protocols"] = {"wavefunction": "all"}
    _failed(qcs.run(wfn), "unsupported_error", "wavefunction")
    _failed(qcs.run(_inp(geometry=WATER_BOHR[:-1])), "input_error", "geometry")
    _failed(qcs.run(_inp(molecular_multiplicity=2)), "input_error")
    _failed(qcs.run("not a dict"), "input_error")
    bad = _inp()
    bad["schema_version"] = 2
    _failed(qcs.run(bad), "input_error", "schema_version")


def test_unconverged_scf_is_a_convergence_error():
    out = qcs.run(_inp(keywords={"max_iter": 1}))
    _failed(out, "convergence_error")


def test_qcelemental_round_trip():
    qcel = pytest.importorskip("qcelemental", reason="QCElemental is optional")
    from qcelemental.models import AtomicInput, AtomicResult

    mol = qcel.models.Molecule.from_data(
        "O 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467"
    )
    ai = AtomicInput(
        molecule=mol, driver="gradient", model={"method": "hf", "basis": "sto-3g"}
    )
    out = qcs.run(ai.dict())
    assert out["success"] is True, out
    res = AtomicResult(**out)
    assert np.asarray(res.return_result).shape == (3, 3)
    direct = ferric.run_rhf(
        ferric.Molecule.from_arrays(
            list(mol.symbols), np.asarray(mol.geometry).reshape(-1, 3), units="bohr"
        ),
        _bs("sto-3g"),
        with_gradient=True,
    )
    assert np.asarray(res.return_result).tolist() == direct.gradient().tolist()
    assert res.properties.calcinfo_natom == 3

    failed = qcs.run(
        AtomicInput(
            molecule=mol, driver="hessian", model={"method": "hf", "basis": "sto-3g"}
        ).dict()
    )
    from qcelemental.models import FailedOperation

    assert FailedOperation(**failed).error.error_type == "unsupported_error"
