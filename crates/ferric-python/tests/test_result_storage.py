"""Result classes are frozen and share their solve (roadmap item 13).

`RhfResult` and `DftResult` keep their solve behind an `Arc`, so a function
that takes one as `result=` and releases the GIL (`hirshfeld_charges`) shares
it instead of copying the density; the classes are `#[pyclass(frozen)]`, so a
reader never takes a borrow flag. Neither is observable from Python as such
(the getters return fresh copies, as before), so these cases pin what is: the
attributes still refuse assignment with the same `AttributeError`, the array
getters still return independent writable copies, results can be read from
several threads while a GIL-releasing consumer runs, and
`FrequencyResult.ir_intensities` (whose first-call cache was the one `&mut
self` method of a result class) caches the same value under concurrent first
calls. The copy that was removed is measured, not tested: see the item 13
entry of ARCHITECTURE.md.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
H2 = "2\n\nH 0 0 0\nH 0 0 0.74\n"


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz_string(WATER)


@pytest.fixture(scope="module")
def basis():
    return ferric.BasisSet.bundled("6-31g")


@pytest.fixture(scope="module")
def results(water, basis):
    return {
        "rhf": ferric.run_rhf(water, basis),
        "dft": ferric.run_dft(water, basis, functional="pbe"),
        "uhf": ferric.run_uhf(
            ferric.Molecule.from_xyz_string(
                "2\n\nO 0 0 0\nH 0 0 0.97\n", multiplicity=2
            ),
            basis,
        ),
    }


@pytest.mark.parametrize(
    ("kind", "attr"),
    [
        ("rhf", "energy"),
        ("rhf", "converged"),
        ("dft", "total_energy"),
        ("uhf", "iterations"),
    ],
)
def test_getter_refuses_assignment(results, kind, attr):
    r = results[kind]
    before = getattr(r, attr)
    with pytest.raises(AttributeError, match="not writable"):
        setattr(r, attr, before)
    assert getattr(r, attr) == before


@pytest.mark.parametrize("kind", ["rhf", "dft", "uhf"])
def test_new_attribute_refused(results, kind):
    with pytest.raises(AttributeError):
        results[kind].note = "x"  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["rhf", "dft"])
def test_array_getters_return_independent_copies(results, kind):
    r = results[kind]
    d = r.density()
    eps = r.orbital_energies()
    assert d.flags.writeable and eps.flags.writeable
    d_before, eps_before = d.copy(), eps.copy()
    d[:] = 0.0
    eps[:] = 0.0
    assert np.array_equal(r.density(), d_before)
    assert np.array_equal(r.orbital_energies(), eps_before)


@pytest.mark.parametrize("kind", ["rhf", "dft"])
def test_reference_read_during_gil_released_consumer(water, basis, results, kind):
    """A reader thread reads the result while `hirshfeld_charges` (which
    releases the GIL around its free-atom SCFs) holds it; both see the
    same values as a serial run."""
    r = results[kind]
    serial = ferric.hirshfeld_charges(water, basis, r)
    density = r.density()
    out: dict[str, object] = {}
    errors: list[BaseException] = []
    stop = threading.Event()

    def consumer():
        try:
            out["q"] = ferric.hirshfeld_charges(water, basis, r)
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    def reader():
        n = 0
        while not stop.is_set() or n == 0:
            assert np.array_equal(r.density(), density)
            n += 1
        out["reads"] = n

    threads = [threading.Thread(target=consumer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert out["q"] == serial
    assert out["reads"] >= 1


def test_ir_intensities_cache_under_concurrent_first_calls():
    freq = ferric.run_frequencies(ferric.Molecule.from_xyz_string(H2), "sto-3g")
    got: list[list[float]] = []
    errors: list[BaseException] = []

    def call():
        try:
            got.append(freq.ir_intensities())
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(got) == 3 and got[0] == got[1] == got[2]
    assert freq.ir_intensities() == got[0]
