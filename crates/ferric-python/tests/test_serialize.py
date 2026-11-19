"""`ferric.serialize`: dict/JSON views of Molecule and results, run-log reader."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

import ferric.serialize as fs  # noqa: E402

XYZ = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"


@pytest.fixture(scope="module")
def mol():
    return ferric.Molecule.from_xyz_string(XYZ)


@pytest.fixture(scope="module")
def basis():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def rhf(mol, basis):
    return ferric.run_rhf(mol, basis)


@pytest.fixture(scope="module")
def dft(mol, basis):
    return ferric.run_dft(mol, basis, functional="pbe")


def test_molecule_round_trip_exact(mol):
    d = fs.to_dict(mol)
    assert d["type"] == "Molecule" and d["charge"] == 0
    back = fs.molecule_from_dict(json.loads(fs.to_json(mol)))
    assert back.symbols() == mol.symbols()
    assert back.coords_bohr() == mol.coords_bohr()
    assert back.is_ghost() == mol.is_ghost()


def test_molecule_charge_ghost_round_trip():
    m = ferric.Molecule.from_arrays(
        ["N", "H", "H", "H", "H", "He"],
        [[0, 0, 0], [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [0, 0, 4]],
        charge=1,
        ghost=[False] * 5 + [True],
    )
    d = fs.to_dict(m)
    assert d["charge"] == 1 and d["multiplicity"] == 1 and d["ghost"][-1] is True
    back = fs.molecule_from_dict(d)
    assert back.nelec() == m.nelec() and back.is_ghost() == m.is_ghost()
    with pytest.raises(ValueError):
        fs.molecule_from_dict({"symbols": ["H"]})


def test_rhf_dict_and_json(rhf):
    d = fs.to_dict(rhf)
    assert d["type"] == type(rhf).__name__
    assert d["energy"] == rhf.energy
    assert d["iterations"] == rhf.iterations and d["converged"] is True
    assert d["certificate"] == json.loads(json.dumps(rhf.certificate))
    assert d["ferric_build"] == ferric.__build__ or d["ferric_build"] is None
    assert "density" not in d
    back = json.loads(fs.to_json(rhf, indent=1))
    assert back["energy"] == rhf.energy


def test_dft_energy_normalised_original_kept(dft):
    d = fs.to_dict(dft)
    assert d["energy"] == dft.total_energy
    assert d["total_energy"] == dft.total_energy
    assert d["e_xc"] == dft.e_xc


def test_arrays(rhf):
    d = fs.to_dict(rhf, arrays=True)
    assert np.array(d["orbital_energies"]).shape == (7,)
    assert np.array(d["mo_coefficients"]).shape == (7, 7)
    assert np.array_equal(d["density"], np.asarray(rhf.density()))
    json.dumps(d)


def test_non_finite_becomes_null():
    class Fake:
        energy = float("nan")
        other = float("inf")
        n = 3

    d = json.loads(fs.to_json(Fake()))
    assert d["energy"] is None and d["other"] is None and d["n"] == 3
    assert math.isnan(fs.to_dict(Fake())["energy"])


def test_pandas_frame(rhf, dft):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([fs.to_dict(rhf), fs.to_dict(dft)])
    assert list(df["energy"]) == [rhf.energy, dft.total_energy]
    assert list(df["type"]) == [type(rhf).__name__, type(dft).__name__]


def test_read_run_log(tmp_path):
    p = tmp_path / "x.ferric.jsonl"
    recs = [
        {"record": "run_start", "seq": 0, "t": 0.0, "schema": 1, "config": {}},
        {"record": "scf_iter", "seq": 1, "t": 0.1, "energy": -1.0},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + '\n\n{"record": "sc')
    assert fs.read_run_log(p) == recs  # truncated last line dropped
    p.write_text("{bad\n" + json.dumps(recs[0]) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        fs.read_run_log(p)


def test_multiplicity_round_trip_and_old_dict_fallback():
    m = ferric.Molecule.from_arrays(
        ["O", "O"], [[0, 0, 0], [0, 0, 1.2]], multiplicity=3
    )
    d = fs.to_dict(m)
    assert d["multiplicity"] == 3
    assert fs.molecule_from_dict(d).multiplicity == 3
    assert fs.molecule_from_dict(d, multiplicity=1).multiplicity == 1
    old = {k: v for k, v in d.items() if k != "multiplicity"}
    assert fs.molecule_from_dict(old).multiplicity == 1  # parity fallback
