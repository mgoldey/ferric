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
* ccsd / ccsd(t) / hessian routes equal the direct driver calls bit for bit
  (correlation energies, total = reference + correlation, the Cartesian
  Hessian), and the hessian driver's rows are the central differences of
  ``run`` gradients at +-h/2 Bohr, bar FD_BAR = 2e-6 (truncation
  (h/2)^2 |E^(4)| / 6 + delta_g / (h/2)). Measured max deviation 2.11e-6 at
  h = 1.89e-3 and 5.28e-7 at h/2: a factor 4.0, i.e. pure h^2 truncation, so
  the analytic Hessian is the limit and the bar has ~4x margin.
* Certificate under ``extras["ferric"]["certificate"]`` equals the direct
  result's certificate; the build is ``provenance["build"]``.
* Refusals return ``success=False`` with a stable ``error_type`` and never raise.
"""

from __future__ import annotations

import itertools
import json

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

    # Open-shell Kohn-Sham: UKS (run_uks) at multiplicity 2.
    o = qcs.run(
        _inp(
            method="PBE",
            basis="6-31g",
            driver="gradient",
            molecular_charge=1,
            molecular_multiplicity=2,
        )
    )
    do = ferric.run_uks(
        _mol(charge=1, multiplicity=2), _bs("6-31g"), "pbe", with_gradient=True
    )
    assert o["success"] is True, o
    assert o["return_result"] == do.gradient().tolist()
    assert o["properties"]["return_energy"] == do.energy
    assert o["properties"]["scf_total_energy"] == do.energy
    assert o["properties"]["calcinfo_nbasis"] == 13
    assert o["extras"]["ferric"]["certificate"] == json.loads(
        json.dumps(do.certificate)
    )
    oe = qcs.run(
        _inp(
            method="PBE",
            basis="6-31g",
            molecular_charge=1,
            molecular_multiplicity=2,
        )
    )
    assert (
        oe["return_result"]
        == ferric.run_uks(_mol(charge=1, multiplicity=2), _bs("6-31g"), "pbe").energy
    )

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
    assert set(out["provenance"]["build"]) >= {"commit", "dirty"}
    assert out["provenance"]["build"] == dict(ferric.build_info())
    fx = out["extras"]["ferric"]
    assert "certificate" in fx, "the certificate is dropped from extras"
    assert fx["certificate"]["capability"] == "rhf"
    direct = ferric.run_rhf(_mol(), _bs("sto-3g"))
    assert fx["certificate"] == json.loads(json.dumps(direct.certificate))


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
    _failed(qcs.run(_inp(driver="properties")), "unsupported_error", "properties")
    # No analytic Hessian: mp2, CC, open-shell Kohn-Sham.
    _failed(
        qcs.run(_inp(method="mp2", basis="cc-pvdz", driver="hessian")),
        "unsupported_error",
        "analytic Hessian",
    )
    _failed(
        qcs.run(_inp(method="ccsd", basis="cc-pvdz", driver="hessian")),
        "unsupported_error",
        "analytic Hessian",
    )
    _failed(
        qcs.run(
            _inp(method="wb97x-v", basis="6-31g", driver="hessian", molecular_charge=1)
        ),
        "unsupported_error",
        "analytic Hessian",
    )
    # CC: no gradient, closed shell only.
    _failed(
        qcs.run(_inp(method="ccsd", basis="cc-pvdz", driver="gradient")),
        "unsupported_error",
        "gradient",
    )
    _failed(
        qcs.run(_inp(method="ccsd(t)", basis="cc-pvdz", molecular_charge=1)),
        "unsupported_error",
        "closed-shell",
    )
    # A keyword valid for ccsd is not valid for hf, nor for the hessian driver.
    _failed(qcs.run(_inp(keywords={"ladder": "ri"})), "input_error", "ladder")
    _failed(
        qcs.run(_inp(driver="hessian", keywords={"grid_radial": 50, "max_iter": 9})),
        "input_error",
        "grid_radial",
    )
    _failed(qcs.run(_inp(driver="optimization")), "input_error")
    _failed(
        qcs.run(
            _inp(
                method="pbe",
                molecular_charge=1,
                molecular_multiplicity=2,
                keywords={"dispersion": "d3bj"},
            )
        ),
        "unsupported_error",
        "dispersion",
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
            molecule=mol, driver="properties", model={"method": "hf", "basis": "sto-3g"}
        ).dict()
    )
    from qcelemental.models import FailedOperation

    assert FailedOperation(**failed).error.error_type == "unsupported_error"


def test_cc_routes_equal_the_direct_calls_bitwise():
    kw = {"auxbasis": "cc-pvdz-ri", "frozen_core": 1}
    mol, bs, aux = _mol(), _bs("6-31g"), _bs("cc-pvdz-ri")
    ref = ferric.run_rhf(mol, bs)
    for method, driver in (("ccsd", ferric.run_ccsd), ("ccsd(t)", ferric.run_ccsd_t)):
        out = qcs.run(_inp(method=method, basis="6-31g", keywords=kw))
        assert out["success"] is True, out
        direct = driver(mol, bs, aux, frozen_core=1)
        p = out["properties"]
        assert p["ccsd_correlation_energy"] == direct.correlation_energy
        assert p["scf_total_energy"] == ref.energy
        assert p["ccsd_total_energy"] == ref.energy + direct.correlation_energy
        if method == "ccsd":
            assert out["return_result"] == p["ccsd_total_energy"]
            assert "ccsd_prt_pr_total_energy" not in p
        else:
            e_corr = direct.correlation_energy + direct.t_correction
            assert p["ccsd_prt_pr_correlation_energy"] == e_corr
            assert out["return_result"] == ref.energy + e_corr
            assert direct.t_correction != 0.0
        assert out["return_result"] == p["return_energy"]
        cert = out["extras"]["ferric"]["certificate"]
        assert cert == json.loads(json.dumps(direct.certificate))
        assert cert["capability"] == method
        assert out["provenance"]["routine"] == "ferric.qcschema.run:cc"


def test_hessian_routes_equal_the_direct_calls_bitwise():
    closed = qcs.run(_inp(driver="hessian"))
    assert closed["success"] is True, closed
    hess = np.asarray(closed["return_result"])
    assert hess.shape == (9, 9)
    direct = ferric.run_frequencies(
        _mol(), "sto-3g", reference="rhf", hessian="analytic"
    )
    assert direct.hessian_source == "analytic"
    assert hess.tolist() == np.asarray(direct.cartesian_hessian).tolist()
    assert closed["properties"]["return_hessian"] == closed["return_result"]
    assert closed["properties"]["return_energy"] == direct.energy
    cert = closed["extras"]["ferric"]["certificate"]
    assert cert == json.loads(json.dumps(direct.certificate))
    assert cert["capability"].startswith("frequencies")

    # Open shell: UHF at multiplicity 2.
    open_ = qcs.run(_inp(driver="hessian", molecular_charge=1))
    assert open_["success"] is True, open_
    du = ferric.run_frequencies(
        _mol(charge=1, multiplicity=2),
        "sto-3g",
        reference="uhf",
        multiplicity=2,
        hessian="analytic",
    )
    assert open_["return_result"] == np.asarray(du.cartesian_hessian).tolist()

    # Closed-shell Kohn-Sham: the moving-grid analytic Hessian.
    ks = qcs.run(_inp(method="pbe", basis="6-31g", driver="hessian"))
    assert ks["success"] is True, ks
    dk = ferric.run_frequencies(
        _mol(), "6-31g", xc="pbe", reference="rhf", hessian="analytic-moving-grid"
    )
    assert dk.hessian_source == "analytic-moving-grid"
    assert ks["return_result"] == np.asarray(dk.cartesian_hessian).tolist()

    # Open-shell Kohn-Sham: the analytic UKS Hessian on the moving grid.
    uks = qcs.run(
        _inp(method="pbe", basis="6-31g", driver="hessian", molecular_charge=1)
    )
    assert uks["success"] is True, uks
    du = ferric.run_frequencies(
        _mol(charge=1, multiplicity=2),
        "6-31g",
        xc="pbe",
        reference="uhf",
        multiplicity=2,
        hessian="analytic-moving-grid",
    )
    assert du.hessian_source == "analytic-moving-grid"
    assert uks["return_result"] == np.asarray(du.cartesian_hessian).tolist()


def test_hessian_is_the_finite_difference_of_the_gradient_in_the_input_frame():
    kw = {**TIGHT, "ri_j_precision": "f64"}
    h = H / 2
    hess = np.asarray(qcs.run(_inp(driver="hessian", keywords=TIGHT))["return_result"])
    xyz = np.asarray(WATER_BOHR)
    fd = np.zeros((9, 9))
    for k in range(9):
        g = []
        for sign in (+1, -1):
            geo = xyz.copy()
            geo[k] += sign * h
            out = qcs.run(_inp(driver="gradient", geometry=geo, keywords=kw))
            g.append(np.asarray(out["return_result"]).ravel())
        fd[k] = (g[0] - g[1]) / (2 * h)
    # Row k is d(gradient)/d(coordinate k): a frame or ordering slip fails here.
    assert np.abs(fd - hess).max() < FD_BAR
    assert np.abs(hess - hess.T).max() < FD_BAR


def test_qcelemental_validates_the_new_routes():
    pytest.importorskip("qcelemental", reason="QCElemental is optional")
    from qcelemental.models import AtomicResult

    for out in (
        qcs.run(
            _inp(method="ccsd", basis="6-31g", keywords={"auxbasis": "cc-pvdz-ri"})
        ),
        qcs.run(_inp(driver="hessian")),
        qcs.run(_inp(driver="gradient", molecular_charge=1)),
    ):
        assert out["success"] is True, out
        res = AtomicResult(**out)
        assert res.success is True
        assert res.provenance.creator == "ferric"
        assert res.extras["ferric"]["certificate"]["capability"]
    hess = AtomicResult(**qcs.run(_inp(driver="hessian")))
    assert np.asarray(hess.return_result).shape in ((9, 9), (81,))


def test_molecule_qcschema_round_trip_and_ghost():
    import ferric.qcschema as qcs

    mol = ferric.Molecule.from_arrays(
        ["H", "H", "He"],
        [[0, 0, 0], [0, 0, 1.4], [0, 0, 5.0]],
        units="bohr",
        ghost=[False, False, True],
    )
    d = qcs.molecule_to_qcschema(mol, charge=0, multiplicity=1)
    assert d["schema_name"] == "qcschema_molecule" and d["real"] == [True, True, False]
    assert len(d["geometry"]) == 9 and d["molecular_multiplicity"] == 1
    back = qcs.molecule_from_qcschema(d)
    assert back.symbols() == mol.symbols()
    assert back.coords_bohr() == mol.coords_bohr()
    assert back.is_ghost() == mol.is_ghost()


def test_molecule_from_qcschema_charge_and_errors():
    import ferric.qcschema as qcs

    d = {
        "symbols": ["O", "H"],
        "geometry": [[0, 0, 0], [0, 0, 1.8]],
        "molecular_charge": -1,
        "molecular_multiplicity": 1,
    }
    assert qcs.molecule_from_qcschema(d).nelec() == 10
    with pytest.raises(ValueError, match="geometry"):
        qcs.molecule_from_qcschema({"symbols": ["H"], "geometry": [0, 0]})
    with pytest.raises(ValueError):
        qcs.molecule_from_qcschema("nope")
