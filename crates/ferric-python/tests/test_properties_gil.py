"""Density properties run with the GIL released (an item 13 follow-up).

ESP (`esp_at_atoms`, `esp_at_points`, `esp_on_surface`), CHELPG, RESP,
Löwdin, Mulliken and `write_cube` prepare the basis and compute with the GIL
released, sharing the result's density through its `Arc`, as
`hirshfeld_charges` does. The values are unchanged (compared bitwise against
the previous build when the change was made, see ARCHITECTURE.md); these
cases pin that a reader thread can use the same result while one of them
runs, and that the answers equal the serial ones.
"""

from __future__ import annotations

import threading
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
    property's answer equals a serial run's."""
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
