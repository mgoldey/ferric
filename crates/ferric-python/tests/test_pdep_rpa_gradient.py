"""`run_pdep_rpa(..., with_gradient=True)`: the analytic RHF + RI-dRPA gradient.

The route calls `ferric_rpa::gradient_analytic::rpa_analytic_gradient_on` on the
reference RHF of the same run, the function behind the CLI's pdep-rpa
`task = "optimize"` (default `rpa_gradient`). Registry method: `rpa-gradient`.

Pinned here:
- the gradient is the derivative of `total_energy`: a 5-point central
  difference of `run_pdep_rpa` energies on strained H2O / STO-3G;
- it equals the gradient the CLI's optimize path computes at the starting
  geometry, bit for bit: `crates/ferric-rpa/tests/rpa_gradient_python_route.rs`
  pins the binding's library call against `total_rpa_gradient_by` (the CLI's
  call) and, with `FERRIC_RPA_GRAD_DUMP=<file>`, dumps the bits that
  `test_gradient_equals_the_library_dump_bit_for_bit` compares (skipped when
  the variable is unset); the CLI's printed Max Grad / RMS Grad is a smoke
  check;
- every setting the gradient does not differentiate raises ValueError before
  any SCF;
- the result carries `gradient()` only on request, shape (natoms, 3).

FD bar (derived). Five-point stencil, step h = 2e-3 Bohr: truncation
h^4 |E^(5)| / 30 ~ 5e-13 |E^(5)| with |E^(5)| of order 1..10 Ha/Bohr^5 on
this PES, so <= 5e-12. Noise: every energy carries an error eps_E of order
the SCF residual (density_conv 1e-9, E_RPA first order in the density) plus
the eigensolver tolerance (none for the full-rank solve); the stencil amplifies it by
(8 + 8 + 1 + 1) / (12 h) = 0.75 / h = 375, so eps_E = 1e-9 gives 4e-7. The
analytic side adds the Z-vector residual (|r| < 1e-8). The estimate is
conservative: measured |analytic - FD| is 8e-9..1.3e-8 on the three
coordinates (the full-rank Lanczos energy has no eigensolver tolerance), so
the bar is 2e-7, 15x the measured maximum.

Mutations, each applied alone to `crates/ferric-python/src/rpa.rs`, rebuilt,
run, and restored (the failing line is pytest's `--tb=short` "E" line):

| mutation | failing test | failing line |
|---|---|---|
| `pdep_rpa_total_gradient` sets `cfg.quadrature.n_points = 16` (ignores `n_quad=4`) | `test_gradient_matches_central_difference[0-2]`, `[1-1]`, `[2-2]` | `assert abs(got - fd) < FD_BAR`: 1.7e-3, 1.4e-3, 1.4e-3 vs 2e-7 |
| same | `test_gradient_equals_the_library_dump_bit_for_bit` | `assert np.array_equal(g, ref)` (max diff 1.7e-3) |
| same | `test_cli_optimize_step_smoke` | `assert abs(np.abs(g).max() - max_grad) <= 5e-3 * max_grad` (2.1e-3 vs 8.6e-4) |
| `RingOrder::Full` replaced by `RingOrder::SecondOrder` | the same five tests | the same assertions: FD residual 1.7e-2, 1.2e-2, 1.3e-2; dump diff 1.7e-2; smoke 1.6e-2 |
| `refuse_pdep_rpa_gradient`: `if t != 0.0` made `if false && t != 0.0` | `test_refusals[trunc_thresh]` | `with pytest.raises(ValueError, ...)`: the call reaches the library's own `check_analytic_scope` and raises `RuntimeError`, which `pytest.raises(ValueError)` does not catch |
| same | `test_refusal_precedes_the_scf` | `with pytest.raises(ValueError, ...)`: the SCF runs and raises `RuntimeError: SCF did not converge after 1 iterations` |

No failure was a panic, TypeError or import error. The other refusal clauses
are covered by their own `test_refusals` case and were not mutated one by one.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
XYZ = "3\nstrained water\nO 0 0 0\nH 0 0.9 0.1\nH 0 -0.3 0.85\n"
BASIS = "sto-3g"
AUX = "cc-pvdz-ri"
N_QUAD = 4
BOHR = 0.529177210903


def _bs(name):
    return ferric.BasisSet.bundled(name)


def _run(mol, **kw):
    return ferric.run_pdep_rpa(
        mol,
        _bs(BASIS),
        _bs(AUX),
        n_quad=N_QUAD,
        **kw,
    )


@pytest.fixture(scope="module")
def mol():
    return ferric.Molecule.from_xyz_string(XYZ)


@pytest.fixture(scope="module")
def graded(mol):
    return _run(mol, with_gradient=True)


def _shifted(mol, atom, axis, delta_bohr):
    coords = np.array(mol.coords(), dtype=float)
    coords[atom, axis] += delta_bohr * BOHR
    syms = mol.symbols()
    body = "\n".join(
        f"{s} {x:.15f} {y:.15f} {z:.15f}" for s, (x, y, z) in zip(syms, coords)
    )
    return ferric.Molecule.from_xyz_string(f"{len(syms)}\nshifted\n{body}\n")


def test_gradient_is_none_unless_requested(mol):
    assert _run(mol).gradient() is None


def test_gradient_shape_and_translation_invariance(graded):
    g = graded.gradient()
    assert g.shape == (3, 3)
    assert np.all(np.isfinite(g))
    # Translational invariance: the net force vanishes.
    assert np.abs(g.sum(axis=0)).max() < 1e-7


FD_BAR = 2e-7
H = 2e-3


@pytest.mark.parametrize("atom,axis", [(0, 2), (1, 1), (2, 2)])
def test_gradient_matches_central_difference(mol, graded, atom, axis):
    e = {k: _run(_shifted(mol, atom, axis, k * H)).total_energy for k in (-2, -1, 1, 2)}
    assert len(set(e.values())) == 4, e  # four distinct displaced energies
    fd = (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * H)
    got = graded.gradient()[atom, axis]
    print(
        f"[fd] atom {atom} axis {axis}: analytic {got:+.10e} fd {fd:+.10e} "
        f"|diff| {abs(got - fd):.3e} bar {FD_BAR:.1e}; 4 energy evaluations"
    )
    assert abs(got - fd) < FD_BAR, (got, fd, got - fd)


def test_default_run_is_unchanged_by_the_option(mol, graded):
    """with_gradient only adds the gradient: with the SCF density threshold
    and trunc_thresh pinned the energies are identical."""
    plain = _run(mol, trunc_thresh=0.0, scf_density_conv=1e-9)
    assert plain.total_energy == graded.total_energy


def test_gradient_equals_the_library_dump_bit_for_bit(mol):
    import json
    import struct

    dump = os.environ.get("FERRIC_RPA_GRAD_DUMP")
    if not dump:
        pytest.skip("FERRIC_RPA_GRAD_DUMP not set (see the module docstring)")
    bits = json.loads(pathlib.Path(dump).read_text(encoding="utf-8"))["bits"]
    ref = np.array([struct.unpack(">d", bytes.fromhex(b))[0] for b in bits]).reshape(
        3, 3
    )
    # blas_threads=1 is the setting under which results are bit-identical
    # across thread counts (set FERRIC_BLAS_THREADS=1 for the Rust dump too);
    # at other settings serial BLAS sections move the last bit (~1e-14 rel).
    ferric.set_blas_threads(1)
    try:
        g = _run(mol, with_gradient=True).gradient()
    finally:
        ferric.set_blas_threads(None)  # the process default for later tests
    assert np.array_equal(g, ref), np.abs(g - ref).max()


def _cli():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


@pytest.mark.validation
def test_cli_optimize_step_smoke(mol, graded, tmp_path):
    cli = _cli()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    geom = tmp_path / "mol.xyz"
    geom.write_text(XYZ)
    toml = tmp_path / "opt.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{geom}"\n\n[basis]\nname = "{BASIS}"\n\n'
        f'[method]\nkind = "pdep-rpa"\ntask = "optimize"\n\n'
        f'[rpa]\nauxbasis = "{AUX}"\nn_quad = {N_QUAD}\n'
        f'quadrature = "gauss-legendre"\n\n'
        f"[optimize]\nmax_steps = 1\n"
    )
    out = subprocess.run(
        [str(cli), str(toml)], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    row = re.search(
        r"^\s*0 \|\s*(-?[\d.]+) \|[^|]*\|\s*([\d.]+e[+-]\d+) \|\s*([\d.]+e[+-]\d+)",
        out.stdout,
        re.M,
    )
    assert row, out.stdout
    g = graded.gradient()
    max_grad, rms = float(row.group(2)), float(row.group(3))
    # "{:8.2e}" keeps 3 significant digits: relative half-ulp 5e-3.
    assert abs(np.abs(g).max() - max_grad) <= 5e-3 * max_grad
    assert abs(np.sqrt((g**2).mean()) - rms) <= 5e-3 * rms


REFUSALS = [
    ("trunc_thresh", {"trunc_thresh": 1e-4}, "trunc_thresh"),
    ("chi0_sparsity", {"chi0_sparsity": "boys"}, "chi0_sparsity"),
    ("cosx", {"k_builder": "cosx"}, "COSX"),
    ("df_j_aux", {"df_j_aux": "def2-universal-jkfit"}, "df_j_aux"),
    ("df_k_aux", {"df_k_aux": "def2-universal-jkfit"}, "df_k_aux"),
    ("point_charges", {"point_charges": [(0.1, 5.0, 5.0, 5.0)]}, "point_charges"),
    ("external_field", {"external_field": (0.0, 0.0, 1e-3)}, "external_field"),
    ("solvent", {"solvent": "water"}, "solvent"),
    ("properties", {"properties": True}, "properties"),
    ("export_eigpot", {"export_eigpot_prefix": "x"}, "export_eigpot"),
    ("run_diagnostics", {"run_diagnostics": True}, "run_diagnostics"),
]


@pytest.mark.parametrize("kw", [r[1] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_refusals(mol, kw):
    needle = next(r[2] for r in REFUSALS if r[1] is kw)
    with pytest.raises(ValueError, match="with_gradient=True") as ei:
        _run(mol, with_gradient=True, **kw)
    assert needle in str(ei.value)


def test_open_shell_is_refused():
    radical = ferric.Molecule.from_xyz_string(
        "2\nOH\nO 0 0 0\nH 0 0 0.97\n", charge=0, multiplicity=2
    )
    with pytest.raises(ValueError, match="closed-shell only"):
        _run(radical, with_gradient=True)


def test_refusal_precedes_the_scf(mol):
    """A refused call returns at once: a 1-iteration SCF budget would raise a
    convergence error if the SCF ran first."""
    with pytest.raises(ValueError, match="with_gradient=True"):
        _run(mol, with_gradient=True, trunc_thresh=1e-4, scf_max_iter=1)


def test_trunc_zero_is_accepted(mol):
    assert _run(mol, with_gradient=True, trunc_thresh=0.0).gradient() is not None
