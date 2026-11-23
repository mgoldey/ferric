"""Input and block classes with no mutating method are frozen.

`BasisSet`, `QmmmSystem`, `MmTopology`, `CdftConstraint`, `AoIntegrals`,
`Cphf` and `Ucphf` are `#[pyclass(frozen)]` (an item 13 follow-up): none has
a setter or a `&mut self` method, so a borrow takes no flag and the objects
can be shared across threads like the result classes. From Python this
pins what is visible: a getter still refuses assignment, new attributes are
refused, and the objects still work from several threads at once (the RI
`Cphf` exercises its fitted response's `Mutex`).

These cases also pass without `frozen`: a pyclass with no setter already
refuses assignment, and shared borrows did not conflict. The real guarantee
is at compile time: `frozen` forbids `&mut self` methods, `PyRefMut` and
setters, so adding one to these classes no longer compiles. (Thread safety
itself comes from pyo3 0.23 requiring every non-`unsendable` pyclass to be
`Sync`, frozen or not, so these classes must not hold non-`Sync` fields.)
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
    jkfit = ferric.BasisSet.bundled("def2-universal-jkfit")
    cphf_ri, cphf_ri_fresh = (
        ao.cphf(rhf.mo_coefficients(), rhf.orbital_energies(), nocc, aux=jkfit)
        for _ in range(2)
    )
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
        "Cphf_ri": cphf_ri,
        "Cphf_ri_fresh": cphf_ri_fresh,
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
    """Several threads use the same `AoIntegrals`, `Cphf` (exact and RI) and
    `BasisSet` at once and get the serial answers. The RI handle the threads
    share has not been used before, so its first calls run concurrently."""
    ao, bs = objects["AoIntegrals"], objects["BasisSet"]
    cphf, cphf_ri = objects["Cphf"], objects["Cphf_ri_fresh"]
    u = np.full((ao.nao - ao.nelec // 2, ao.nelec // 2), 0.01)
    want = (ao.hcore(), cphf.matvec(u), objects["Cphf_ri"].matvec(u), bs.elements())
    got: list[tuple[object, ...]] = []
    errors: list[BaseException] = []

    def work():
        try:
            for _ in range(5):
                got.append(
                    (ao.hcore(), cphf.matvec(u), cphf_ri.matvec(u), bs.elements())
                )
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert len(got) == 20
    for h, mv, mv_ri, el in got:
        assert np.array_equal(h, want[0])
        assert np.array_equal(mv, want[1])
        assert np.array_equal(mv_ri, want[2])
        assert el == want[3]
    assert not np.array_equal(want[1], want[2])  # the RI path is a different one
