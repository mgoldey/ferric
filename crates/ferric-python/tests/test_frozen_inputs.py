"""Input and block classes with no mutating method are frozen.

`BasisSet`, `QmmmSystem`, `MmTopology`, `CdftConstraint`, `AoIntegrals`,
`Cphf` and `Ucphf` are `#[pyclass(frozen)]` (an item 13 follow-up): none has
a setter or a `&mut self` method, so a borrow takes no flag and the objects
can be shared across threads like the result classes. From Python this
pins what is visible: a getter still refuses assignment, new attributes are
refused, and the objects still work from several threads at once.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
OH = "2\n\nO 0 0 0\nH 0 0 0.97\n"


@pytest.fixture(scope="module")
def objects():
    bs = ferric.BasisSet.bundled("sto-3g")
    water = ferric.Molecule.from_xyz_string(WATER)
    ao = ferric.AoIntegrals(water, bs)
    rhf = ferric.run_rhf(water, bs)
    nocc = ao.nelec // 2
    cphf = ao.cphf(rhf.mo_coefficients(), rhf.orbital_energies(), nocc)
    oh = ferric.Molecule.from_xyz_string(OH, multiplicity=2)
    uhf = ferric.run_uhf(oh, bs)
    ao_oh = ferric.AoIntegrals(oh, bs)
    ucphf = ao_oh.ucphf(
        uhf.mo_coefficients_alpha(),
        uhf.mo_coefficients_beta(),
        uhf.orbital_energies_alpha(),
        uhf.orbital_energies_beta(),
        5,
        4,
    )
    qmmm = ferric.QmmmSystem(
        ["O", "H", "H", "X"],
        [(0, 0, 0.1173), (0, 0.7572, -0.4692), (0, -0.7572, -0.4692), (0, 0, 4.0)],
        [0.0, 0.0, 0.0, -0.5],
        qm_indices=[0, 1, 2],
    )
    topo = ferric.MmTopology.from_amber_units(
        charges=[-0.834, 0.417, 0.417],
        sigmas_angstrom=[3.15061, 1.0, 1.0],
        epsilons_kcal=[0.1521, 0.0, 0.0],
        bonds=[(0, 1, 450.0, 0.9572), (0, 2, 450.0, 0.9572)],
        angles=[(1, 0, 2, 55.0, 104.52)],
        torsions=[],
    )
    return {
        "BasisSet": bs,
        "AoIntegrals": ao,
        "Cphf": cphf,
        "Ucphf": ucphf,
        "QmmmSystem": qmmm,
        "MmTopology": topo,
        "CdftConstraint": ferric.CdftConstraint([0], 8.5),
    }


NAMES = [
    "BasisSet",
    "AoIntegrals",
    "Cphf",
    "Ucphf",
    "QmmmSystem",
    "MmTopology",
    "CdftConstraint",
]


@pytest.mark.parametrize("name", NAMES)
def test_new_attribute_refused(objects, name):
    obj = objects[name]
    assert type(obj).__name__ == name
    with pytest.raises(AttributeError):
        obj.note = "x"


@pytest.mark.parametrize("name", NAMES)
def test_method_cannot_be_replaced(objects, name):
    obj = objects[name]
    method = next(m for m in dir(type(obj)) if not m.startswith("_"))
    with pytest.raises(AttributeError):
        setattr(obj, method, None)
    assert getattr(obj, method) is not None


@pytest.mark.parametrize(
    ("attr", "value"), [("atoms", [1]), ("target", 7.0), ("kind", "spin")]
)
def test_cdft_constraint_getters_refuse_assignment(objects, attr, value):
    c = objects["CdftConstraint"]
    before = getattr(c, attr)
    with pytest.raises(AttributeError, match="not writable"):
        setattr(c, attr, value)
    assert getattr(c, attr) == before


def test_shared_across_threads(objects):
    """Several threads use the same `AoIntegrals`, `Cphf` and `BasisSet` at
    once and get the serial answers."""
    ao, cphf, bs = objects["AoIntegrals"], objects["Cphf"], objects["BasisSet"]
    u = np.full((ao.nao - ao.nelec // 2, ao.nelec // 2), 0.01)
    want = (ao.hcore(), cphf.matvec(u), bs.elements())
    got: list[tuple[object, ...]] = []
    errors: list[BaseException] = []

    def work():
        try:
            for _ in range(5):
                got.append((ao.hcore(), cphf.matvec(u), bs.elements()))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert len(got) == 20
    for h, mv, el in got:
        assert np.array_equal(h, want[0])
        assert np.array_equal(mv, want[1])
        assert el == want[2]
