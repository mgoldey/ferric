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
    with pytest.raises(TypeError):
        ferric.hirshfeld_charges(r, _mol("nh3"), ferric.BasisSet.bundled("sto-3g"))


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
        {"level_shift": 0.0, "mom_after_iter": 3},
    ):
        with pytest.raises(ValueError, match=r"symmetry=True") as exc:
            ferric.run_rhf(mol, bs, symmetry=True, **kwargs)
        assert not isinstance(exc.value, RuntimeError), kwargs
