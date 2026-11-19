"""`ferric.mlip`: MLIP training-data export (extended XYZ and ASE database).

Anchors:

* Unit conversion against constants written out here, not read from
  ``ferric.units``: 1 Hartree = 27.211386245988 eV (CODATA 2018) and
  1 Bohr = 0.52917721092 Angstrom (CODATA 2010), the values the Rust crates
  use. A gradient of 1 Hartree/Bohr is therefore a force of
  -27.211386245988 / 0.52917721092 = -51.42208...  eV/Angstrom (the
  well-known 51.42 eV/Angstrom per atomic unit of force).
* Round trip through ``ase.io.read`` (extxyz) and ``ase.db`` returns the same
  geometry, energy, forces and level-of-theory metadata.
* The input is a synthetic QCSchema ``AtomicResult`` (the shape
  ``ferric.qcschema.run`` returns), so the conversion tests need no SCF; one
  test runs the real driver and is the only one that needs the extension to
  compute (the package import itself needs it, hence ``importorskip``).
* Frames build without ASE; only the writers need it.
"""

from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

import ferric.mlip as mlip  # noqa: E402

HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.52917721092
FORCE_PER_AU = HARTREE_EV / BOHR_ANGSTROM  # eV/Angstrom per Hartree/Bohr

GEOMETRY = [0.0, 0.0, 0.2217, 0.0, 1.4309, -0.8867, 0.0, -1.4309, -0.8867]
GRADIENT = [[0.0, 0.0, 0.01], [0.0, -0.02, -0.005], [0.0, 0.02, -0.005]]
ENERGY = -75.98


def _result(**over):
    out = {
        "schema_name": "qcschema_output",
        "schema_version": 1,
        "success": True,
        "driver": "gradient",
        "molecule": {
            "symbols": ["O", "H", "H"],
            "geometry": list(GEOMETRY),
            "molecular_charge": 0,
            "molecular_multiplicity": 1,
        },
        "model": {"method": "hf", "basis": "6-31g"},
        "keywords": {},
        "return_result": [list(r) for r in GRADIENT],
        "properties": {"return_energy": ENERGY},
        "provenance": {
            "creator": "ferric",
            "version": "9.9.9",
            "build": {"commit": "deadbeef" * 5, "dirty": True},
        },
    }
    out.update(over)
    return out


def test_conversion_against_known_constants():
    (f,) = mlip.frames(_result())
    assert f.energy == pytest.approx(ENERGY * 27.211386245988, rel=1e-14)
    # 1 Ha/Bohr = 51.4220867... eV/Angstrom (CODATA 2010/2018 mix)
    assert FORCE_PER_AU == pytest.approx(51.42208, abs=1e-4)
    np.testing.assert_allclose(f.forces, -np.array(GRADIENT) * FORCE_PER_AU, rtol=1e-14)
    np.testing.assert_allclose(
        f.positions, np.array(GEOMETRY).reshape(3, 3) * BOHR_ANGSTROM, rtol=1e-14
    )
    # one Bohr in -> 0.52917721092 Angstrom out
    one = _result()
    one["molecule"]["geometry"] = [1.0, 0, 0, 0, 0, 0, 0, 0, 2.0]
    assert mlip.frames(one)[0].positions[0, 0] == pytest.approx(
        0.52917721092, rel=1e-14
    )


def test_force_is_minus_gradient():
    (f,) = mlip.frames(_result(return_result=[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert f.forces[0, 0] == pytest.approx(-27.211386245988 / 0.52917721092, rel=1e-14)


def test_metadata_states_units_and_level_of_theory():
    (f,) = mlip.frames(_result())
    assert f.info["energy_unit"] == "eV"
    assert f.info["forces_unit"] == "eV/Angstrom"
    assert f.info["positions_unit"] == "Angstrom"
    assert f.info["qc_method"] == "hf" and f.info["qc_basis"] == "6-31g"
    assert f.info["qc_charge"] == 0 and f.info["qc_multiplicity"] == 1
    assert "qc_functional" not in f.info and "qc_dispersion" not in f.info
    assert f.info["ferric_version"] == "9.9.9"
    assert f.info["ferric_git_sha"] == "deadbeef" * 5
    assert f.info["ferric_dirty"] is True


def test_functional_dispersion_charge_multiplicity():
    r = _result(
        model={"method": "PBE0", "basis": "def2-svp"}, keywords={"dispersion": "d3bj"}
    )
    r["molecule"].update(molecular_charge=1, molecular_multiplicity=2)
    (f,) = mlip.frames(r)
    assert f.info["qc_method"] == "pbe0" and f.info["qc_functional"] == "pbe0"
    assert f.info["qc_dispersion"] == "d3bj"
    assert (f.info["qc_charge"], f.info["qc_multiplicity"]) == (1, 2)


def test_build_stamp_falls_back_to_running_ferric(monkeypatch):
    r = _result()
    del r["provenance"]
    monkeypatch.setattr(
        ferric, "__build__", {"git_sha": "abc123", "dirty": False}, raising=False
    )
    info = mlip.frames(r)[0].info
    assert info["ferric_version"] == ferric.__version__
    assert info["ferric_git_sha"] == "abc123" and info["ferric_dirty"] is False
    monkeypatch.setattr(ferric, "__build__", None, raising=False)
    info = mlip.frames(r)[0].info
    assert info["ferric_git_sha"] == "unknown" and "ferric_dirty" not in info


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ({"success": False, "error": {"error_message": "boom"}}, "boom"),
        ({"driver": "energy"}, "no forces"),
    ],
)
def test_unusable_results_refuse(bad, fragment):
    with pytest.raises(ValueError, match=fragment):
        mlip.frames(_result(**bad))


def test_ghost_atoms_refuse():
    r = _result()
    r["molecule"]["real"] = [True, True, False]
    with pytest.raises(ValueError, match="ghost"):
        mlip.frames(r)


def test_bad_input_and_empty():
    with pytest.raises(ValueError, match="unsupported input"):
        mlip.frames([42])
    with pytest.raises(ValueError, match="no results"):
        mlip.frames([])


def test_frames_accepts_iterables_and_frames_without_ase(monkeypatch):
    monkeypatch.setitem(sys.modules, "ase", None)
    one = mlip.frames(_result())[0]
    got = mlip.frames(iter([_result(), one]))
    assert len(got) == 2 and got[1] is one


def test_writers_need_ase(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ase", None)  # makes `import ase` raise
    for write, name in ((mlip.write_extxyz, "a.xyz"), (mlip.write_asedb, "a.db")):
        with pytest.raises(ImportError, match="pip install ase"):
            write(tmp_path / name, _result())
        assert not (tmp_path / name).exists()
    with pytest.raises(ImportError, match="pip install ase"):
        mlip.to_atoms(_result())


def _two_frames():
    second = _result(
        return_result=[[0.0, 0.0, -0.01], [0.0, 0.02, 0.005], [0.0, -0.02, 0.005]]
    )
    second["properties"] = {"return_energy": ENERGY - 0.001}
    second["molecule"] = copy.deepcopy(second["molecule"])
    second["molecule"]["geometry"][2] += 0.1
    return [_result(), second]


def test_extxyz_round_trip(tmp_path):
    pytest.importorskip("ase", reason="ASE is an optional dependency")
    from ase.io import read

    path = tmp_path / "train.xyz"
    assert mlip.write_extxyz(path, _two_frames()) == 2
    back = read(path, index=":", format="extxyz")
    assert len(back) == 2
    for atoms, frame in zip(back, mlip.frames(_two_frames()), strict=True):
        assert atoms.get_chemical_symbols() == ["O", "H", "H"]
        np.testing.assert_allclose(atoms.positions, frame.positions, atol=1e-8)
        assert atoms.get_potential_energy() == pytest.approx(frame.energy, abs=1e-8)
        np.testing.assert_allclose(atoms.get_forces(), frame.forces, atol=1e-8)
        assert atoms.info["qc_method"] == "hf" and atoms.info["qc_basis"] == "6-31g"
        assert atoms.info["qc_charge"] == 0 and atoms.info["qc_multiplicity"] == 1
        assert atoms.info["energy_unit"] == "eV"
        assert atoms.info["forces_unit"] == "eV/Angstrom"
        assert atoms.info["positions_unit"] == "Angstrom"
        assert atoms.info["ferric_version"] == "9.9.9"
        assert atoms.info["ferric_git_sha"] == "deadbeef" * 5
        assert bool(atoms.info["ferric_dirty"]) is True


def test_asedb_round_trip(tmp_path):
    pytest.importorskip("ase", reason="ASE is an optional dependency")
    from ase.db import connect

    path = tmp_path / "train.db"
    assert mlip.write_asedb(path, _two_frames()) == 2
    db = connect(str(path))
    rows = list(db.select())
    assert len(rows) == 2
    expected = mlip.frames(_two_frames())
    for row, frame in zip(rows, expected, strict=True):
        atoms = row.toatoms()
        np.testing.assert_allclose(atoms.positions, frame.positions, atol=1e-8)
        assert row.energy == pytest.approx(frame.energy, abs=1e-8)
        np.testing.assert_allclose(row.forces, frame.forces, atol=1e-8)
        assert row.qc_method == "hf" and row.qc_basis == "6-31g"
        assert row.qc_charge == 0 and row.qc_multiplicity == 1
        assert row.energy_unit == "eV" and row.forces_unit == "eV/Angstrom"
        assert row.positions_unit == "Angstrom"
        assert row.ferric_version == "9.9.9" and row.ferric_dirty
    # the metadata is queryable
    assert db.count("qc_method=hf") == 2 and db.count("qc_method=pbe") == 0


def test_ase_atoms_with_ferric_calculator_composes():
    pytest.importorskip("ase", reason="ASE is an optional dependency")
    from ase import Atoms
    from ferric.ase import FerricCalculator

    atoms = Atoms(
        "OH2", positions=[(0, 0, 0.117), (0, 0.757, -0.467), (0, -0.757, -0.467)]
    )
    calc = FerricCalculator(
        method="dft", functional="pbe", basis="sto-3g", options={"dispersion": "d3bj"}
    )
    atoms.calc = calc
    # What calculate() leaves behind, without running SCF: ASE serves these
    # from the cache because calc.atoms equals atoms.
    forces = np.array([[0.0, 0.0, 0.3], [0.0, -0.1, -0.15], [0.0, 0.1, -0.15]])
    calc.atoms = atoms.copy()
    calc.results = {"energy": -2000.0, "free_energy": -2000.0, "forces": forces}
    (f,) = mlip.frames(atoms)
    assert f.energy == -2000.0
    np.testing.assert_array_equal(f.forces, forces)
    assert f.info["qc_method"] == "dft" and f.info["qc_functional"] == "pbe"
    assert f.info["qc_dispersion"] == "d3bj" and f.info["qc_basis"] == "sto-3g"
    assert (f.info["qc_charge"], f.info["qc_multiplicity"]) == (0, 1)
    assert f.info["ferric_version"] == ferric.__version__
    assert f.info["forces_unit"] == "eV/Angstrom"

    atoms.calc = None
    with pytest.raises(ValueError):
        mlip.frames(atoms)


def test_real_qcschema_gradient_converts_exactly():
    """End to end on the real driver: forces are exactly -gradient in eV/Angstrom."""
    import ferric.qcschema as qcs

    out = qcs.run(
        {
            "schema_name": "qcschema_input",
            "schema_version": 1,
            "molecule": {"symbols": ["H", "H"], "geometry": [0, 0, 0, 0, 0, 1.5]},
            "driver": "gradient",
            "model": {"method": "hf", "basis": "sto-3g"},
            "keywords": {},
        }
    )
    assert out["success"], out
    (f,) = mlip.frames(out)
    grad = np.asarray(out["return_result"])
    np.testing.assert_allclose(
        f.forces, -grad * (27.211386245988 / 0.52917721092), rtol=1e-13
    )
    assert f.energy == pytest.approx(
        out["properties"]["return_energy"] * 27.211386245988, rel=1e-14
    )
    assert f.info["ferric_version"] == ferric.__version__
