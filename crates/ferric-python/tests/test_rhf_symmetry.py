"""run_rhf(symmetry=True): point-group-blocked RHF through the Python binding (#294).

Reference: PySCF symmetry-constrained RHF (scripts/validation/gen_scf_symmetry.py,
testdata/reference/validation/scf_symmetry/*.json). The energy belongs to the
SYMMETRIZED geometry and the result type carries no input-frame matrices.
"""

import json
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "testdata" / "reference" / "validation" / "scf_symmetry"
XYZ = {"h2o": "water.xyz", "nh3": "nh3.xyz", "c2h4": "c2h4.xyz"}
CASES = [
    ("h2o", "sto-3g"),
    ("h2o", "6-31g"),
    ("nh3", "sto-3g"),
    ("nh3", "6-31g"),
    ("c2h4", "sto-3g"),
    ("c2h4", "6-31g"),
]
TIGHT = dict(density_conv=1e-10, energy_conv=1e-12, max_iter=200)


def _mol(name):
    return ferric.Molecule.from_xyz(str(ROOT / "testdata" / "molecules" / XYZ[name]))


def _ref(name, basis):
    return json.loads((REF / f"{name}_{basis}.json").read_text())


@pytest.mark.parametrize("name,basis", CASES)
def test_matches_pyscf_symmetry_constrained_scf(name, basis):
    ref = _ref(name, basis)["ground"]
    r = ferric.run_rhf(
        _mol(name), ferric.BasisSet.bundled(basis), symmetry=True, **TIGHT
    )
    assert isinstance(r, ferric.SymmetricRhfResult)
    assert r.converged
    assert abs(r.energy - ref["energy"]) < 1e-8
    assert r.point_group == ref["group"]
    # Doubly occupied orbitals per irrep. D2h axis labels differ between the
    # codes, so compare as a multiset there (see gen_scf_symmetry.py).
    got = sorted(n // 2 for n in r.irrep_occupations().values())
    want = sorted(ref["docc_per_irrep"].values())
    if r.point_group == "D2h":
        assert got == want
    else:
        # Cs: ferric spells A-double-prime as A'' where PySCF writes A".
        mine = {k.replace("''", '"'): v // 2 for k, v in r.irrep_occupations().items()}
        assert mine == ref["docc_per_irrep"]
    assert sum(r.irrep_occupations().values()) == _mol(name).nelec()
    assert r.max_symmetrization_shift < 1e-4
    assert len(r.orbital_irreps) == len(r.orbital_energies())


def test_irrep_nelec_reaches_the_excited_state_aufbau_cannot():
    ref = _ref("h2o", "sto-3g")["excited"]
    r = ferric.run_rhf(
        _mol("h2o"),
        ferric.BasisSet.bundled("sto-3g"),
        symmetry=True,
        irrep_nelec=ref["irrep_nelec"],
        **TIGHT,
    )
    assert abs(r.energy - ref["energy"]) < 1e-8
    assert r.irrep_occupations() == ref["irrep_nelec"]


def test_symmetry_false_is_bit_identical_to_the_default():
    """Exactness anchor: the new kwargs change nothing when off."""
    mol, bs = _mol("h2o"), ferric.BasisSet.bundled("6-31g")
    a = ferric.run_rhf(mol, bs)
    b = ferric.run_rhf(mol, bs, symmetry=False)
    assert type(a) is type(b) is ferric.RhfResult
    assert a.energy == b.energy
    assert a.iterations == b.iterations


def test_frame_dependent_outputs_raise_instead_of_returning_rotated_data():
    r = ferric.run_rhf(_mol("nh3"), ferric.BasisSet.bundled("sto-3g"), symmetry=True)
    for accessor in (r.density, r.mo_coefficients, r.gradient):
        with pytest.raises(
            NotImplementedError, match="symmetry frame|symmetrized frame"
        ):
            accessor()
    # The documented frame is available, and consumers of RhfResult refuse it.
    rot, origin = r.frame_transform()
    assert len(rot) == 3 and len(origin) == 3
    assert len(r.symmetrized_coords()) == len(r.symbols()) == 4
    # The result slot of the RhfResult consumers rejects it by type. The
    # control: the same call with the plain RhfResult succeeds, so the
    # TypeError below comes from the `result` argument and nothing else.
    mol, bs = _mol("nh3"), ferric.BasisSet.bundled("sto-3g")
    plain = ferric.run_rhf(mol, bs)
    assert len(ferric.esp_at_atoms(mol, bs, plain)) == 4
    for consumer in (ferric.hirshfeld_charges, ferric.esp_at_atoms):
        with pytest.raises(
            TypeError, match=r"result.*expected an RhfResult or DftResult"
        ):
            consumer(mol, bs, r)


def _rot(axis, angle):
    import numpy as np

    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _write_xyz(tmp_path, symbols, coords, name="m.xyz"):
    path = tmp_path / name
    lines = [str(len(symbols)), "rotated"] + [
        f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(symbols, coords)
    ]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize("name", ["h2o", "nh3", "c2h4"])
def test_frame_transform_maps_input_coords_onto_symmetrized_coords(name, tmp_path):
    """r_sym = R @ (r_in - origin), all Angstrom; the transpose is wrong by Bohrs."""
    import numpy as np

    bs = ferric.BasisSet.bundled("sto-3g")
    mol = _mol(name)
    # Also an arbitrarily rotated and translated copy of the same molecule.
    rot = _rot([0.3, -0.5, 0.8], 0.9)
    coords = np.array(mol.coords())
    moved = coords @ rot.T + np.array([1.3, -0.7, 2.1])
    rotated = ferric.Molecule.from_xyz(_write_xyz(tmp_path, mol.symbols(), moved))
    e0 = ferric.run_rhf(mol, bs, symmetry=True, **TIGHT).energy
    for m, xyz in ((mol, coords), (rotated, moved)):
        r = ferric.run_rhf(m, bs, symmetry=True, **TIGHT)
        R, origin = (np.array(x) for x in r.frame_transform())
        sym = np.array(r.symmetrized_coords())
        assert np.abs((xyz - origin) @ R.T - sym).max() < 1e-5
        assert np.abs((xyz - origin) @ R - sym).max() > 0.05 or name == "h2o"
        assert abs(r.energy - e0) < 1e-8
        assert r.symbols() == mol.symbols()


def test_orbital_irrep_labels_follow_the_orbital_columns():
    r = ferric.run_rhf(
        _mol("h2o"), ferric.BasisSet.bundled("6-31g"), symmetry=True, **TIGHT
    )
    labels, eps = r.orbital_irreps, r.orbital_energies()
    # 1a1 2a1 1b2 3a1 1b1 are occupied, in energy order (ferric's B1 is the
    # out-of-plane pair, the HOMO).
    assert labels[:5] == ["A1", "A1", "B2", "A1", "B1"]
    assert all(a <= b for a, b in zip(eps[:5], eps[1:5]))
    occ = r.irrep_occupations()
    for lab in set(labels[:5]):
        assert occ[lab] == 2 * labels[:5].count(lab)


def test_cosx_is_refused_with_symmetry():
    mol, bs = _mol("h2o"), ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError, match=r"symmetry=True.*COSX"):
        ferric.run_rhf(mol, bs, symmetry=True, k_builder="cosx")


def test_refusals_are_valueerrors_naming_symmetry_true():
    mol, bs = _mol("h2o"), ferric.BasisSet.bundled("sto-3g")
    with pytest.raises(ValueError, match="requires symmetry=True"):
        ferric.run_rhf(mol, bs, irrep_nelec={"A1": 4})
    # Bad irrep requests and unsupported combinations are ValueErrors that
    # name symmetry=True, never silently ignored and never a bare RuntimeError.
    for kwargs in (
        {"irrep_nelec": {"Z9": 2}},
        {"irrep_nelec": {"A1": 3}},
        {"solvent": "water"},
        {"point_charges": [(0.1, 5.0, 0.0, 0.0)]},
        {"external_field": (0.0, 0.0, 0.01)},
        {"stability_descent": True},
        {"soscf": True},
        {"k_builder": "cosx"},
        {"level_shift": 0.0, "mom_after_iter": 3},
    ):
        with pytest.raises(ValueError, match=r"symmetry=True") as exc:
            ferric.run_rhf(mol, bs, symmetry=True, **kwargs)
        assert not isinstance(exc.value, RuntimeError), kwargs
