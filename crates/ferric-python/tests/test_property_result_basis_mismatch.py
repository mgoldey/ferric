"""Property functions refuse a result from another molecule or basis set.

Each function below takes `(mol, basis_set, result)` and indexes the result's
density or MO coefficients by the number of basis functions `basis_set`
places on `mol`. Given a result computed for a different molecule or basis,
they used to hit an ndarray shape panic, which reaches Python as
`pyo3_runtime.PanicException` (a `BaseException`, so `except Exception` and
`except ValueError` both miss it). They now raise `ValueError` naming both
sizes before any array is indexed.

The siblings that already refused a mismatch with `RuntimeError` (their Rust
kernels check the density shape) keep raising `RuntimeError`: ARCHITECTURE.md
allows an exception class only to become a narrower subclass, never to be
swapped, and `ValueError` is not a subclass of `RuntimeError`.

Run:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest \\
        crates/ferric-python/tests/test_property_result_basis_mismatch.py -q
"""

from pathlib import Path

import numpy as np
import pytest

import ferric

ROOT = Path(__file__).resolve().parents[3]
WATER = ROOT / "testdata" / "molecules" / "water.xyz"
HI = ROOT / "testdata" / "molecules" / "validation" / "ecp" / "hi.xyz"

# Water/STO-3G has 7 basis functions, water/6-31G 13 and HI/def2-SVP 31.
NBF_WATER_STO3G = 7
TARGETS = {
    "other_basis": ("water", "6-31g", 13),
    "other_molecule": ("hi", "def2-svp", 31),
}
AUX = {"6-31g": "cc-pvdz-ri", "def2-svp": "def2-svp-rifit"}


@pytest.fixture(scope="module")
def systems():
    water = ferric.Molecule.from_xyz(str(WATER), 0, 1)
    hi = ferric.Molecule.from_xyz(str(HI))
    sto = ferric.BasisSet.bundled("sto-3g")
    rhf = ferric.run_rhf(water, sto, energy_conv=1e-10, density_conv=1e-8)
    assert rhf.converged
    return {"water": water, "hi": hi, "sto": sto, "rhf": rhf}


PTS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _calls(tmp_path):
    """name -> f(mol, basis_set, aux_basis_set, result) for every function
    that used to panic on a mismatched result."""
    return {
        "orbital_moments": lambda m, b, a, r: ferric.orbital_moments(m, b, r),
        "density_second_moment": lambda m, b, a, r: ferric.density_second_moment(
            m, b, r
        ),
        "eval_density_on_points": lambda m, b, a, r: ferric.eval_density_on_points(
            m, b, r, PTS
        ),
        "eval_mo_on_points": lambda m, b, a, r: ferric.eval_mo_on_points(
            m, b, r, [0], PTS
        ),
        "write_cube": lambda m, b, a, r: ferric.write_cube(
            str(tmp_path / "d.cube"), m, b, r, spacing=1.0
        ),
        "write_cube_mo": lambda m, b, a, r: ferric.write_cube(
            str(tmp_path / "mo.cube"), m, b, r, what="mo", mo_index=0, spacing=1.0
        ),
        "hirshfeld_polarizability": lambda m, b, a, r: ferric.hirshfeld_polarizability(
            m, b, a, r
        ),
        "response_modes": lambda m, b, a, r: ferric.response_modes(
            m, b, a, r, n_modes=1
        ),
    }


FIXED = list(_calls(Path(".")).keys())


@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("name", FIXED)
def test_mismatched_result_raises_value_error(systems, tmp_path, name, target):
    mol_key, basis_name, nbf = TARGETS[target]
    mol = systems[mol_key]
    bs = ferric.BasisSet.bundled(basis_name)
    aux = ferric.BasisSet.bundled(AUX[basis_name])
    call = _calls(tmp_path)[name]
    prefix = "write_cube" if name.startswith("write_cube") else name
    with pytest.raises(ValueError) as info:
        call(mol, bs, aux, systems["rhf"])
    msg = str(info.value)
    assert msg.startswith(f"{prefix}: "), msg
    assert (
        f"result has {NBF_WATER_STO3G} basis functions but basis_set places "
        f"{nbf} on mol" in msg
    ), msg


@pytest.mark.parametrize("name", FIXED)
def test_matching_result_still_runs(systems, tmp_path, name):
    call = _calls(tmp_path)[name]
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    call(systems["water"], systems["sto"], aux, systems["rhf"])


def test_matching_result_values(systems):
    """Spot values on matching inputs: the check adds no arithmetic."""
    water, sto, rhf = systems["water"], systems["sto"], systems["rhf"]
    m = np.asarray(ferric.density_second_moment(water, sto, rhf))
    # Trace = <r^2> of the 10-electron density; symmetric tensor.
    assert np.allclose(m, m.T)
    assert np.trace(m) > 0.0
    rho = ferric.eval_density_on_points(water, sto, rhf, PTS)
    assert len(rho) == 2 and all(v >= 0.0 for v in rho)
    centers, spreads = ferric.orbital_moments(water, sto, rhf)
    assert len(centers) == len(spreads) == NBF_WATER_STO3G


ALREADY_RUNTIME_ERROR = {
    "esp_at_atoms": lambda m, b, r: ferric.esp_at_atoms(m, b, r),
    "esp_at_points": lambda m, b, r: ferric.esp_at_points(m, b, r, PTS),
    "esp_on_surface": lambda m, b, r: ferric.esp_on_surface(m, b, r),
    "hirshfeld_charges": lambda m, b, r: ferric.hirshfeld_charges(m, b, r),
    "lowdin_charges": lambda m, b, r: ferric.lowdin_charges(m, b, r),
    "mulliken_charges": lambda m, b, r: ferric.mulliken_charges(m, b, r),
    "chelpg_charges": lambda m, b, r: ferric.chelpg_charges(m, b, r),
    "resp_charges": lambda m, b, r: ferric.resp_charges(m, b, r),
}


@pytest.mark.parametrize("name", list(ALREADY_RUNTIME_ERROR))
def test_siblings_keep_their_runtime_error(systems, name):
    bs = ferric.BasisSet.bundled("6-31g")
    with pytest.raises(RuntimeError, match=r"7, 7"):
        ALREADY_RUNTIME_ERROR[name](systems["water"], bs, systems["rhf"])
