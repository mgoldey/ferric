"""Density properties run with the GIL released (an item 13 follow-up).

ESP (`esp_at_atoms`, `esp_at_points`, `esp_on_surface`), CHELPG, RESP,
Löwdin, Mulliken and `write_cube` prepare the basis and compute with the GIL
released, sharing the result's solve as `hirshfeld_charges` does. The values
are unchanged (compared bitwise against the previous build when the change
was made, see ARCHITECTURE.md). These cases pin that a reader thread can use
the same result while one of them runs with the serial answer, that the
reader makes progress *during* the slower calls (which it cannot while the
GIL is held), and that the validation errors and their order are unchanged.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

WATER = "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
POINTS = np.array([[0.0, 0.0, 3.0], [1.0, 2.0, -1.0], [-2.5, 0.3, 0.7]])


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
    }


def _cube(path):
    def run(mol, bs, r):
        ferric.write_cube(str(path), mol, bs, r, spacing=0.4)
        return Path(path).read_bytes()

    return run


PROPERTIES = {
    "esp_at_atoms": ferric.esp_at_atoms,
    "esp_at_points": lambda m, b, r: ferric.esp_at_points(m, b, r, POINTS),
    "esp_on_surface": lambda m, b, r: ferric.esp_on_surface(m, b, r)[1],
    "chelpg_charges": ferric.chelpg_charges,
    "resp_charges": ferric.resp_charges,
    "lowdin_charges": ferric.lowdin_charges,
    "mulliken_charges": ferric.mulliken_charges,
}


@pytest.mark.parametrize("kind", ["rhf", "dft"])
@pytest.mark.parametrize("prop", [*PROPERTIES, "write_cube"])
def test_reference_read_while_property_runs(
    tmp_path, water, basis, results, kind, prop
):
    """A reader thread reads the result while the property runs on it; the
    property's answer equals a serial run's. (Whether the GIL is actually
    released is `test_gil_is_released_during_the_call`'s job.)"""
    r = results[kind]
    if prop == "write_cube":
        fn = _cube(tmp_path / "serial.cube")
        serial = fn(water, basis, r)
        fn = _cube(tmp_path / "threaded.cube")
    else:
        fn = PROPERTIES[prop]
        serial = fn(water, basis, r)
    density = r.density()
    out: dict[str, object] = {}
    errors: list[BaseException] = []
    stop = threading.Event()

    def consumer():
        try:
            out["value"] = fn(water, basis, r)
        except BaseException as exc:  # noqa: BLE001 - reported below
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
    assert out["reads"] >= 1
    assert out["value"] == serial


def test_errors_are_unchanged(water, basis, results, tmp_path):
    """Validation still raises the same exceptions (raised before or inside
    the GIL release)."""
    r = results["rhf"]
    with pytest.raises(ValueError, match="points must have shape"):
        ferric.esp_at_points(water, basis, r, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="requires mo_index"):
        ferric.write_cube(str(tmp_path / "x.cube"), water, basis, r, what="mo")
    with pytest.raises(IndexError, match="out of range"):
        ferric.write_cube(
            str(tmp_path / "x.cube"), water, basis, r, what="mo", mo_index=999
        )
    with pytest.raises(ValueError, match="basis functions"):
        ferric.write_cube(
            str(tmp_path / "x.cube"),
            water,
            ferric.BasisSet.bundled("sto-3g"),
            r,
            spacing=0.6,
        )
    with pytest.raises(RuntimeError):
        ferric.write_cube(
            str(tmp_path / "no-such-dir" / "x.cube"), water, basis, r, spacing=0.6
        )


# Calls of roughly 0.3 s or more on water/cc-pVDZ (measured 0.3-1.9 s on a
# 4-core machine), long enough that a reader frozen by a held GIL stands out.
SLOW = {
    "esp_at_points": lambda m, b, r, _: ferric.esp_at_points(
        m, b, r, np.random.default_rng(0).normal(size=(20000, 3)) * 3.0
    ),
    "chelpg_charges": lambda m, b, r, _: ferric.chelpg_charges(m, b, r),
    "resp_charges": lambda m, b, r, _: ferric.resp_charges(m, b, r),
    "write_cube": lambda m, b, r, path: ferric.write_cube(
        str(path), m, b, r, spacing=0.15
    ),
}

# How the test tells a held GIL from a released one. Counting the reader's
# iterations between "call started" and "call returned" does not separate
# them: a thread switch right after the call returns lets the reader run a
# full switch interval (5 ms) before the consumer can mark the end, so the
# previous build, which holds the GIL, still scored ~15-75k reads "during"
# the call next to ~0.4-1.3M with it released. What does separate them is the
# longest pause between two reader iterations, measured over the reader's
# whole life (it starts before the call and stops after it): with the GIL
# held the reader is frozen for the whole call (pause / duration measured
# 0.94-1.0 on the previous build), with it released it pauses only when the
# OS deschedules it while the call's worker threads fill the cores (0.07-0.13
# s, a ratio of at most 0.13). The bar is half the call's duration.
MAX_PAUSE_FRACTION = 0.5


@pytest.fixture(scope="module")
def slow_reference(water):
    bs = ferric.BasisSet.bundled("cc-pvdz")
    return bs, ferric.run_rhf(water, bs)


@pytest.mark.parametrize("prop", list(SLOW))
def test_gil_is_released_during_the_call(tmp_path, water, slow_reference, prop):
    bs, r = slow_reference
    fn = SLOW[prop]
    stop = threading.Event()
    errors: list[BaseException] = []
    out = {"pause": 0.0, "reads": 0, "duration": 0.0}

    def consumer():
        try:
            t0 = time.perf_counter()
            fn(water, bs, r, tmp_path / "slow.cube")
            out["duration"] = time.perf_counter() - t0
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            stop.set()

    def reader():
        # One loop from before the call to after it: a reader frozen by a
        # held GIL may resume only after the call, so the pause must be
        # measured across the start of the call too.
        last = time.perf_counter()
        while not stop.is_set():
            r.density()
            now = time.perf_counter()
            out["pause"] = max(out["pause"], now - last)
            out["reads"] += 1
            last = now

    threads = [threading.Thread(target=reader), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    # The call must be long enough for a frozen reader to stand out.
    assert out["duration"] >= 0.2, out
    assert out["pause"] < MAX_PAUSE_FRACTION * out["duration"], out


def test_basis_error_beats_points_error(results):
    """`esp_at_points` prepares the basis before checking `points`, as it did
    before the GIL was released: an element the basis lacks (Rb in 6-31G) is reported
    (`BasisError`, a `RuntimeError` here) even when `points` is also bad."""
    rubidium = ferric.Molecule.from_xyz_string("1\n\nRb 0 0 0\n", multiplicity=2)
    with pytest.raises(ferric.BasisError) as info:
        ferric.esp_at_points(
            rubidium,
            ferric.BasisSet.bundled("6-31g"),
            results["rhf"],
            np.zeros((2, 2)),
        )
    assert isinstance(info.value, RuntimeError)
    assert "points must have shape" not in str(info.value)
