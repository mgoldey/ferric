"""VALIDATION tier -- VALIDATION.md row "NPZ export" (design §5.3, row 157).

The gap this row closes: the NPZ feature bundle had no FOREIGN reader, and its
call sites were unverified -- a key could go missing (several arms warn and
drop an array instead of failing), a (natoms, 3) array could be written
transposed, or a property could be computed from the wrong density, and
nothing downstream would notice until a model trained on it.

What this test does, end to end:

1. Runs the real CLI (`ferric._cli_main`, the same entry point as the
   `ferric` console script and the same compiled extension as the bindings)
   on water / STO-3G, `method.kind = "pdep-rpa"` with `[rpa] export_npz`, in
   a temp dir. `compute_c6 = false` keeps it light; every other property
   takes its export default (on).
2. Reads the bundle with `numpy.load` -- the foreign reader.
3. STRUCTURE (exact): the key set is exactly EXPECTED_KEYS (nothing requested
   is absent: the "no None fields" requirement), every array is finite, has
   the documented shape and dtype (float64; atomic_numbers int64), and is
   C-contiguous as loaded.
4. VALUES, against the Python bindings on an SCF run with the SAME settings
   (RI-JK def2-universal-jkfit, which is what the CLI's pdep-rpa HF reference
   uses by default, and the same convergence thresholds):
   * coords / atomic_numbers: EXACT (both parse the same xyz; a transposed or
     reordered write fails here because coords is not symmetric);
   * density, orbital energies, ESP at nuclei, Loewdin/Mulliken/CHELPG/RESP
     charges, orbital centroids and spreads, density second moment: at the
     SCF-reproducibility bar (two separate SCF runs; not bit-identical by
     construction -- see TOLERANCES);
   * electric_field: against a central difference of the binding's
     `esp_at_points` at R_A +- h (the own-nucleus Z/r term and the
     spherically symmetric cusp part are even in h and cancel exactly), which
     checks the field's SIGN and AXIS order, not just its magnitude;
   * dipole: against sum_A Z_A R_A - 2 sum_{i occ} <i|r|i> from the binding's
     orbital centroids (an exact identity for a closed shell);
   * alpha_tensor: symmetric and positive definite; alpha_tensor and the
     per-atom alpha_atomic are unchanged by a rigid translation of the
     molecule (origin independence -- what the atom-centred dipole operator
     guarantees). Sum_A alpha_atomic is NOT the molecular alpha: the
     charge-transfer part is excluded by construction (72% apart on
     water/STO-3G), so it is not asserted.

Artifact hypothesis: if the NPZ were written from a different density (e.g.
the core guess) every value comparison would miss by >1e-2; if an (N, 3)
array were transposed the shape check fails for N != 3 and the coords/field
checks fail for water (N = 3, non-symmetric); if a requested array were
silently dropped the key-set check fails. None of these can pass by accident
at the bars below.

TOLERANCES are set from measured maxima (recorded next to the constants
below), with 10-100x headroom.

Run (weekly tier; excluded from the default `-m "not validation"`):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest -m validation \\
        crates/ferric-python/tests/test_validation_npz_export.py -q
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ferric

pytestmark = pytest.mark.validation

ROOT = Path(__file__).resolve().parents[3]
WATER = ROOT / "testdata" / "molecules" / "water.xyz"
BASIS = "sto-3g"
AUX = "cc-pvdz-ri"
SCF_JK_AUX = "def2-universal-jkfit"  # ferric-cli config::DEFAULT_SCF_JK_AUX
MAX_ITER = 200
ENERGY_CONV = 1e-9
DENSITY_CONV = 1e-8

# Bars from the measured maxima (2026-09-24, CLI vs Python bindings on the
# same machine): density / orbital energies 5.3e-14, properties <= 1.6e-13,
# field vs central-FD ESP 2.3e-8 (FD-limited), dipole identity 8.2e-16.
TOL_DENSITY = 1e-12
TOL_EPS = 1e-12
TOL_PROP = 5e-12
TOL_FIELD_FD = 2e-7
FD_H = 1e-4  # Bohr
TOL_DIPOLE = 1e-13
TOL_ALPHA_TRANSLATION_REL = 1e-11  # measured 3.4e-13 (atomic), 1.9e-13 (tensor)

EXPECTED_KEYS = {
    "mo_coeffs",
    "orbital_energies",
    "pdep_eigenvectors",
    "orbital_centers",
    "orbital_spreads",
    "density_second_moment",
    "coords",
    "atomic_numbers",
    "density_matrix",
    "dipole",
    "esp_atoms",
    "electric_field",
    "alpha_tensor",
    "alpha_atomic",
    "hirshfeld_charges",
    "lowdin_charges",
    "mulliken_charges",
    "chelpg_charges",
    "resp_charges",
}

TOML = """\
[molecule]
xyz = "{xyz}"

[basis]
name = "{basis}"

[method]
kind = "pdep-rpa"
task = "energy"

[scf]
max_iter = {max_iter}
energy_conv = {energy_conv}
density_conv = {density_conv}

[rpa]
auxbasis = "{aux}"
export_npz = "{npz}"
compute_c6 = false
"""


def _run_export(tmp, xyz, tag):
    npz = tmp / f"{tag}.npz"
    toml = tmp / f"{tag}.toml"
    toml.write_text(
        TOML.format(
            xyz=xyz,
            basis=BASIS,
            max_iter=MAX_ITER,
            energy_conv=ENERGY_CONV,
            density_conv=DENSITY_CONV,
            aux=AUX,
            npz=npz,
        )
    )
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, ferric; sys.argv = ['ferric', sys.argv[1], '--no-json']; ferric._cli_main()",
            str(toml),
        ],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"CLI exited {proc.returncode} (an incomplete NPZ exits non-zero)\n"
        f"--- stdout\n{proc.stdout[-4000:]}\n--- stderr\n{proc.stderr[-4000:]}"
    )
    assert npz.is_file(), f"CLI exited 0 but wrote no {npz}"
    with np.load(npz, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return _run_export(tmp_path_factory.mktemp("npz_export"), WATER, "water_sto3g")


# A rigid translation (Angstrom). Every exported per-atom and molecular
# property below is origin-independent, so the translated export must agree.
SHIFT_ANGSTROM = np.array([3.1, -2.4, 5.7])


@pytest.fixture(scope="module")
def bundle_shifted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("npz_export_shifted")
    lines = WATER.read_text().splitlines()
    n = int(lines[0].split()[0])
    out = [lines[0], lines[1]]
    for line in lines[2 : 2 + n]:
        sym, x, y, zc = line.split()[:4]
        r = np.array([float(x), float(y), float(zc)]) + SHIFT_ANGSTROM
        out.append(f"{sym} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}")
    xyz = tmp / "water_shifted.xyz"
    xyz.write_text("\n".join(out) + "\n")
    return _run_export(tmp, xyz, "water_sto3g_shifted")


@pytest.fixture(scope="module")
def python_side():
    mol = ferric.Molecule.from_xyz(str(WATER), 0, 1)
    bs = ferric.BasisSet.bundled(BASIS)
    rhf = ferric.run_rhf(
        mol,
        bs,
        max_iter=MAX_ITER,
        energy_conv=ENERGY_CONV,
        density_conv=DENSITY_CONV,
        df_j_aux=SCF_JK_AUX,
        df_k_aux=SCF_JK_AUX,
    )
    assert rhf.converged
    return mol, bs, rhf


def test_key_set_is_complete(bundle):
    keys = set(bundle)
    missing = EXPECTED_KEYS - keys
    extra = keys - EXPECTED_KEYS
    assert not missing, (
        f"requested NPZ arrays absent from the bundle: {sorted(missing)}"
    )
    assert not extra, (
        f"unexpected NPZ arrays {sorted(extra)} (compute_c6=false; a new export key needs "
        "its shape/dtype/value check added here)"
    )


def test_shapes_dtypes_layout(bundle, python_side):
    mol, bs, rhf = python_side
    nat = len(mol.symbols())
    nbf = rhf.density().shape[0]
    naux = ferric.compute_metric_2c(mol, bs, ferric.BasisSet.bundled(AUX)).shape[0]
    expected = {
        "coords": (nat, 3),
        "atomic_numbers": (nat,),
        "density_matrix": (nbf, nbf),
        "mo_coeffs": (nbf, nbf),
        "orbital_energies": (nbf,),
        "orbital_centers": (nbf, 3),
        "orbital_spreads": (nbf,),
        "density_second_moment": (3, 3),
        "dipole": (3,),
        "esp_atoms": (nat,),
        "electric_field": (nat, 3),
        "alpha_tensor": (3, 3),
        "alpha_atomic": (nat, 3, 3),
        "hirshfeld_charges": (nat,),
        "lowdin_charges": (nat,),
        "mulliken_charges": (nat,),
        "chelpg_charges": (nat,),
        "resp_charges": (nat,),
    }
    for key, arr in bundle.items():
        want_dtype = np.int64 if key == "atomic_numbers" else np.float64
        assert arr.dtype == want_dtype, f"{key}: dtype {arr.dtype}, want {want_dtype}"
        assert arr.flags.c_contiguous, f"{key}: not C-contiguous as loaded"
        assert np.all(np.isfinite(arr)), f"{key}: non-finite entries"
        if key in expected:
            assert arr.shape == expected[key], (
                f"{key}: shape {arr.shape}, want {expected[key]}"
            )
    pdep = bundle["pdep_eigenvectors"]
    assert pdep.ndim == 2 and naux in pdep.shape, (
        f"pdep_eigenvectors shape {pdep.shape} has no aux axis of length {naux}"
    )


def test_geometry_is_exact(bundle, python_side):
    mol, _, _ = python_side
    np.testing.assert_array_equal(bundle["coords"], np.asarray(mol.coords_bohr()))
    np.testing.assert_array_equal(
        bundle["atomic_numbers"], np.asarray(mol.atomic_numbers())
    )


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_scf_quantities_match_bindings(bundle, python_side):
    mol, bs, rhf = python_side
    checks = [
        ("density_matrix", bundle["density_matrix"], rhf.density(), TOL_DENSITY),
        (
            "orbital_energies",
            bundle["orbital_energies"],
            rhf.orbital_energies(),
            TOL_EPS,
        ),
        ("esp_atoms", bundle["esp_atoms"], ferric.esp_at_atoms(mol, bs, rhf), TOL_PROP),
        (
            "lowdin_charges",
            bundle["lowdin_charges"],
            ferric.lowdin_charges(mol, bs, rhf),
            TOL_PROP,
        ),
        (
            "mulliken_charges",
            bundle["mulliken_charges"],
            ferric.mulliken_charges(mol, bs, rhf),
            TOL_PROP,
        ),
        (
            "chelpg_charges",
            bundle["chelpg_charges"],
            ferric.chelpg_charges(mol, bs, rhf),
            TOL_PROP,
        ),
        (
            "resp_charges",
            bundle["resp_charges"],
            ferric.resp_charges(mol, bs, rhf),
            TOL_PROP,
        ),
        (
            "density_second_moment",
            bundle["density_second_moment"],
            ferric.density_second_moment(mol, bs, rhf),
            TOL_PROP,
        ),
    ]
    centers, spreads = ferric.orbital_moments(mol, bs, rhf)
    checks += [
        ("orbital_centers", bundle["orbital_centers"], centers, TOL_PROP),
        ("orbital_spreads", bundle["orbital_spreads"], spreads, TOL_PROP),
    ]
    failures = []
    for name, got, want, tol in checks:
        d = _maxdiff(got, want)
        print(f"{name:24s} max|d| {d:.2e} (tol {tol:.0e})")
        if not d < tol:
            failures.append(f"{name}: {d:.2e} >= {tol:.0e}")
    assert not failures, "NPZ values disagree with the bindings:\n  " + "\n  ".join(
        failures
    )


def test_electric_field_matches_esp_gradient(bundle, python_side):
    mol, bs, rhf = python_side
    xyz = np.asarray(mol.coords_bohr())
    pts = []
    for a in range(len(xyz)):
        for d in range(3):
            for s in (+1.0, -1.0):
                p = xyz[a].copy()
                p[d] += s * FD_H
                pts.append(p)
    v = np.asarray(ferric.esp_at_points(mol, bs, rhf, np.asarray(pts))).reshape(
        len(xyz), 3, 2
    )
    field_fd = -(v[:, :, 0] - v[:, :, 1]) / (2.0 * FD_H)
    d = _maxdiff(bundle["electric_field"], field_fd)
    print(f"electric_field vs -grad(ESP) FD max|d| {d:.2e} (tol {TOL_FIELD_FD:.0e})")
    assert d < TOL_FIELD_FD
    # Resolving power: the FD field is far from its own transpose/negation.
    assert _maxdiff(bundle["electric_field"], -field_fd) > 1e3 * TOL_FIELD_FD


def test_dipole_matches_orbital_centroids(bundle, python_side):
    mol, bs, rhf = python_side
    z = np.asarray(mol.atomic_numbers(), dtype=float)
    xyz = np.asarray(mol.coords_bohr())
    nocc = mol.nelec() // 2
    centers, _ = ferric.orbital_moments(mol, bs, rhf)
    mu = z @ xyz - 2.0 * np.asarray(centers)[:nocc].sum(axis=0)
    d = _maxdiff(bundle["dipole"], mu)
    print(f"dipole vs centroid identity max|d| {d:.2e} (tol {TOL_DIPOLE:.0e})")
    assert d < TOL_DIPOLE


def test_alpha_is_physical_and_origin_independent(bundle, bundle_shifted):
    """alpha_tensor is symmetric positive definite, and both alpha_tensor and
    the per-atom alpha_atomic are unchanged by a rigid translation.

    The per-atom partition deliberately uses the ATOM-CENTRED dipole operator
    (r - R_A) with no renormalisation to the molecular total, because a
    lab-frame operator leaks R_A x (field-induced charge transfer) into each
    atom and breaks origin independence. So sum_A alpha_atomic is NOT the
    molecular alpha (the charge-transfer part is excluded; measured 72%
    apart on water/STO-3G) and is not asserted. Origin independence is what
    the construction guarantees, and a lab-frame regression breaks it by
    O(|shift| x delta q)."""
    a = bundle["alpha_tensor"]
    assert _maxdiff(a, a.T) < 1e-12, "alpha_tensor is not symmetric"
    assert np.all(np.linalg.eigvalsh(a) > 0.0), "alpha_tensor is not positive definite"
    scale = float(np.max(np.abs(bundle["alpha_atomic"])))
    d_atomic = _maxdiff(bundle["alpha_atomic"], bundle_shifted["alpha_atomic"]) / scale
    d_mol = _maxdiff(a, bundle_shifted["alpha_tensor"]) / float(np.max(np.abs(a)))
    print(
        f"translation: alpha_atomic rel {d_atomic:.2e}, alpha_tensor rel {d_mol:.2e} "
        f"(tol {TOL_ALPHA_TRANSLATION_REL:.0e})"
    )
    assert d_atomic < TOL_ALPHA_TRANSLATION_REL
    assert d_mol < TOL_ALPHA_TRANSLATION_REL
