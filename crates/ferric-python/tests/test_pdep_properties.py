"""run_pdep_rpa(properties=True / export_npz=): the CLI's [rpa] property bundle.

The Python path and the CLI's `[rpa] export_npz` call one library function
(`ferric_rpa::property_bundle`), so the anchor is exactness, not tolerance:
for the same settings every array in the CLI's NPZ is BITWISE equal to the
Python NPZ (`np.array_equal`, same dtype and shape), including the
certificate, and the eigenpotential cubes are byte-identical.

"Same settings" means: the same SCF convergence thresholds, RI-JK on
def2-universal-jkfit (the CLI's pdep-rpa reference default; Python's default is
exact J/K, so the Python side passes df_j_aux/df_k_aux), and the CLI's default
DF-guess pre-stage turned off (`[scf] df_guess = false`; Python has no
pre-stage, and a different starting density moves the converged SCF by ~5e-14,
which every downstream array inherits). With those, the two SCFs are the same
computation, so nothing weaker than bitwise equality is the right bar. The
CLI runs in a subprocess through `ferric._cli_main` (the console script's entry
point, same compiled extension).

Artifact hypothesis: a wrong partition, a dropped frequency weight, a different
density or a transposed (N, 3) array would each change at least one NPZ array
by far more than one ulp, so bitwise equality cannot pass on any of them. Each
test below was mutation-checked (see the commit message).

Also: the result attributes equal the NPZ the same call wrote (getter layout),
unknown strings raise ValueError before the SCF, property switches without the
bundle requested raise ValueError, and a failed requested property is listed
in `property_gaps`, raised as RuntimeWarning, and fails an `export_npz=` call
after the file is written unless `allow_partial_npz=True`.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
MOLS = ROOT / "testdata" / "molecules"
AUX = "cc-pvdz-ri"
JK_AUX = "def2-universal-jkfit"

SCF_PY = dict(
    scf_max_iter=200,
    scf_energy_conv=1e-10,
    scf_density_conv=1e-9,
    df_j_aux=JK_AUX,
    df_k_aux=JK_AUX,
)

# (id, molecule, basis, [rpa] TOML extras, run_pdep_rpa kwargs). Water/TS also
# exercises the ESP surface shell; NH3 a PDEP source on a non-default partition
# with a heteroatom beyond water; the third a model source with most switches off.
CASES = [
    (
        "water-ts-surface",
        "water",
        "sto-3g",
        "compute_esp_surface = true\nesp_surface_n_angular = 26",
        dict(compute_esp_surface=True, esp_surface_n_angular=26),
    ),
    (
        "nh3-pdep-becke",
        "nh3",
        "sto-3g",
        'c6_source = "pdep"\nc6_partition = "becke"',
        dict(c6_source="pdep", c6_partition="becke"),
    ),
    (
        "water-mbd-hirshfeld-subset",
        "water",
        "sto-3g",
        'c6_source = "mbd"\nc6_partition = "hirshfeld"\ncompute_esp = false\n'
        "compute_electric_field = false\ncompute_dipole = false\n"
        "compute_chelpg_charges = false\ncompute_resp_charges = false",
        dict(
            c6_source="mbd",
            c6_partition="hirshfeld",
            compute_esp=False,
            compute_electric_field=False,
            compute_dipole=False,
            compute_chelpg_charges=False,
            compute_resp_charges=False,
        ),
    ),
]

TOML = """\
[molecule]
xyz = "{xyz}"

[basis]
name = "{basis}"

[method]
kind = "pdep-rpa"
task = "energy"

[scf]
max_iter = 200
energy_conv = 1e-10
density_conv = 1e-9
df_guess = false

[rpa]
auxbasis = "{aux}"
export_npz = "{npz}"
export_eigpot_prefix = "{cube}"
export_eigpot_count = 2
cube_spacing = 0.7
{extra}
"""


def _cli(tmp, mol, basis, extra):
    npz = tmp / "cli.npz"
    toml = tmp / "cli.toml"
    toml.write_text(
        TOML.format(
            # POSIX separators: a Windows backslash is an escape inside a
            # TOML basic string.
            xyz=(MOLS / f"{mol}.xyz").as_posix(),
            basis=basis,
            aux=AUX,
            npz=npz.as_posix(),
            cube=(tmp / "cli").as_posix(),
            extra=extra,
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, ferric; sys.argv = ['ferric', sys.argv[1], '--no-json']; ferric._cli_main()",
            str(toml),
        ],
        cwd=tmp,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"CLI exited {proc.returncode}\n{proc.stderr[-3000:]}"
    return npz


def _python(tmp, mol, basis, kwargs):
    return ferric.run_pdep_rpa(
        ferric.Molecule.from_xyz(str(MOLS / f"{mol}.xyz")),
        ferric.BasisSet.bundled(basis),
        ferric.BasisSet.bundled(AUX),
        export_npz=str(tmp / "py.npz"),
        export_eigpot_prefix=str(tmp / "py"),
        export_eigpot_count=2,
        cube_spacing=0.7,
        **SCF_PY,
        **kwargs,
    )


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def run(request, tmp_path_factory):
    _, mol, basis, extra, kwargs = request.param
    tmp = tmp_path_factory.mktemp(request.param[0])
    cli_npz = _cli(tmp, mol, basis, extra)
    result = _python(tmp, mol, basis, kwargs)
    with np.load(cli_npz) as c, np.load(tmp / "py.npz") as p:
        cli = {k: c[k] for k in c.files}
        py = {k: p[k] for k in p.files}
    return tmp, result, cli, py


def test_python_npz_is_bitwise_the_cli_npz(run):
    _, _, cli, py = run
    assert sorted(cli) == sorted(py)
    assert len(cli) >= 15  # not a vacuous comparison
    for k in sorted(cli):
        assert cli[k].dtype == py[k].dtype and cli[k].shape == py[k].shape, k
        assert np.array_equal(cli[k], py[k]), f"{k} differs from the CLI's"


def test_eigenpotential_cubes_are_byte_identical(run):
    tmp, result, _, _ = run
    for i in range(2):
        cli = (tmp / f"cli_eigpot_{i:03d}.cube").read_bytes()
        py = (tmp / f"py_eigpot_{i:03d}.cube").read_bytes()
        assert len(cli) > 1000 and cli == py
    assert str(tmp / "py.npz") in result.written_files
    assert str(tmp / "py_eigpot_001.cube") in result.written_files


# NPZ key -> result attribute holding the same array.
_ATTR = {
    "esp_atoms": "esp_atoms",
    "esp_surface": "esp_surface",
    "esp_points": "esp_points",
    "electric_field": "electric_field",
    "alpha_tensor": "alpha_tensor",
    "alpha_atomic": "alpha_atomic",
    "alpha_ct": "alpha_ct",
    "alpha_atomic_dynamic": "alpha_atomic_dynamic",
    "alpha_ct_dynamic": "alpha_ct_dynamic",
    "c6_iso": "c6_iso",
    "c6_aniso": "c6_aniso",
    "c6_freqs": "c6_freqs",
    "c6_weights": "c6_weights",
    "dipole": "dipole",
    "hirshfeld_charges": "hirshfeld_charges",
    "lowdin_charges": "lowdin_charges",
    "mulliken_charges": "mulliken_charges",
    "chelpg_charges": "chelpg_charges",
    "resp_charges": "resp_charges",
    "orbital_centers": "orbital_centers",
    "orbital_spreads": "orbital_spreads",
    "density_second_moment": "density_second_moment",
    "density_matrix": "density_matrix",
    "mo_coeffs": "mo_coeffs",
    "orbital_energies": "orbital_energies",
    "pdep_eigenvectors": "eigenpotentials",
}


def test_result_attributes_are_the_arrays_it_wrote(run):
    _, result, _, py = run
    for key, attr in _ATTR.items():
        got = getattr(result, attr)
        if key not in py:
            assert got is None, f"{attr} is set but {key} is not in the NPZ"
        else:
            assert got is not None, attr
            assert got.shape == py[key].shape and np.array_equal(got, py[key]), attr
    if "c6_molecular_iso" in py:
        assert result.c6_molecular_iso == float(py["c6_molecular_iso"].item())
        assert result.c6_partition == py["c6_partition"].tobytes().decode()
        assert result.c6_source == py["c6_source"].tobytes().decode()
    assert result.property_gaps == [] and result.property_warnings == []


def test_alpha_ct_is_the_exact_remainder_in_python(run):
    _, result, _, _ = run
    if result.alpha_ct is None:
        pytest.skip("polarizabilities not computed in this case")
    rem = result.alpha_tensor - result.alpha_atomic.sum(axis=0)
    assert np.allclose(result.alpha_ct, rem, rtol=0, atol=1e-13)
    assert np.abs(result.alpha_ct).max() > 1e-3  # not a vacuous zero


def test_dynamic_alpha_tends_to_static_per_atom_alpha(run):
    """The omega -> 0 side of dynamic(0) == static (the exact omega = 0 equality
    is asserted at library level in crates/ferric-rpa/tests/property_bundle.rs):
    alpha^A(i omega) is non-increasing in omega and the smallest node is within a
    few percent of the static value."""
    _, result, _, _ = run
    if result.c6_source != "pdep":
        pytest.skip("PDEP source only")
    dyn_iso = np.trace(result.alpha_atomic_dynamic, axis1=2, axis2=3) / 3
    stat_iso = np.trace(result.alpha_atomic, axis1=1, axis2=2) / 3
    order = np.argsort(result.c6_freqs)
    d = dyn_iso[:, order]
    assert np.all(np.diff(d, axis=1) <= 1e-12)
    assert np.all(d[:, 0] <= stat_iso + 1e-12)
    assert np.allclose(d[:, 0], stat_iso, rtol=5e-2)


def test_no_property_request_leaves_the_result_as_before(tmp_path):
    mol = ferric.Molecule.from_xyz(str(MOLS / "water.xyz"))
    r = ferric.run_pdep_rpa(
        mol, ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled(AUX)
    )
    assert r.alpha_tensor is None and r.esp_atoms is None and r.c6_iso is None
    assert r.property_gaps == [] and r.written_files == []
    assert r.density_matrix is None and r.mo_coeffs is None


def _water():
    return (
        ferric.Molecule.from_xyz(str(MOLS / "water.xyz")),
        ferric.BasisSet.bundled("sto-3g"),
        ferric.BasisSet.bundled(AUX),
    )


@pytest.mark.parametrize(
    "kwargs",
    [dict(c6_source="tsx"), dict(c6_partition="mulliken"), dict(c6_source="")],
)
def test_unknown_c6_strings_raise_valueerror_before_the_scf(kwargs):
    with pytest.raises(ValueError):
        ferric.run_pdep_rpa(*_water(), properties=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(compute_c6=False),
        dict(c6_source="ts"),
        dict(esp_surface_n_angular=26),
        dict(compute_dipole=True),
    ],
)
def test_property_switch_without_the_bundle_raises_valueerror(kwargs):
    with pytest.raises(ValueError, match="properties=True"):
        ferric.run_pdep_rpa(*_water(), **kwargs)


def test_cube_options_without_a_prefix_raise_valueerror():
    with pytest.raises(ValueError, match="export_eigpot_prefix"):
        ferric.run_pdep_rpa(*_water(), cube_spacing=0.5)


def test_a_failed_property_is_a_gap_a_warning_and_fails_the_export(tmp_path):
    """A Lebedev order that does not exist makes the ESP surface fail: the
    bundle is still written without it, the call raises unless
    allow_partial_npz=True, and the failure is a RuntimeWarning + property_gaps."""
    kw = dict(
        compute_esp_surface=True,
        esp_surface_n_angular=7,
        compute_c6=False,
        compute_chelpg_charges=False,
        compute_resp_charges=False,
        **SCF_PY,
    )
    path = tmp_path / "partial.npz"
    with pytest.warns(RuntimeWarning, match="esp_on_surface"):
        with pytest.raises(RuntimeError, match="esp_surface"):
            ferric.run_pdep_rpa(*_water(), export_npz=str(path), **kw)
    with np.load(path) as z:  # written before the error
        assert "esp_atoms" in z.files and "esp_surface" not in z.files
    with pytest.warns(RuntimeWarning, match="esp_on_surface"):
        r = ferric.run_pdep_rpa(
            *_water(), export_npz=str(path), allow_partial_npz=True, **kw
        )
    assert len(r.property_gaps) == 1 and "esp_surface" in r.property_gaps[0]
    assert r.esp_surface is None and r.esp_atoms is not None
    assert any("esp_on_surface" in w for w in r.property_warnings)
