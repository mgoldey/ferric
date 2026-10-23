"""`run_frequencies(hessian="analytic-moving-grid")` (issue #490).

The opt-in analytic closed-shell Kohn-Sham Hessian on the production (moving)
grid, `ferric_scf::hessian::rks_moving_grid_hessian`. The Hessian itself is
validated in the library (`crates/ferric-scf/tests/ks_moving_grid_hessian.rs`)
and its CLI wiring in `crates/ferric-cli/tests/ks_moving_grid_frequencies.rs`
(CLI == library bit for bit). These tests pin the Python surface:

* `test_python_equals_the_cli_bit_for_bit`: the binding's Hessian and
  wavenumbers equal the CLI's (which equal the library's) bit for bit. It needs
  the `ferric-cli` binary of the same tree (`target/release/ferric-cli`, or
  `FERRIC_CLI`) and skips without it.
* `test_moving_grid_matches_fd` (validation tier): against the binding's own
  finite-difference Hessian, Richardson-extrapolated.
* `test_auto_is_still_finite_differences_for_ks`: the default is unchanged.
* the refusals, each named.

Mutation ledger (each measured to fail): the moving-grid route sent to the
fixed-grid Hessian fails `test_moving_grid_matches_fd`; `Auto` sent to the
moving-grid Hessian for KS fails `test_auto_is_still_finite_differences_for_ks`;
the binding dropping `hessian=` (always "auto") fails
`test_moving_grid_reports_its_source`.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
WATER = "testdata/molecules/water.xyz"
H2 = "testdata/molecules/h2.xyz"
MOVING = "analytic-moving-grid"


def _mol(path):
    return ferric.Molecule.from_xyz(str(ROOT / path), 0, 1)


def _max_abs_diff(a, b):
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b))


def test_moving_grid_reports_its_source():
    r = ferric.run_frequencies(_mol(WATER), "sto-3g", xc="LDA", hessian=MOVING)
    assert r.hessian_source == MOVING
    assert r.n_gradient_evaluations == 0
    h = r.cartesian_hessian
    assert len(h) == 9 and all(len(row) == 9 for row in h)
    # The analytic Hessian is symmetric to rounding (its response term is
    # assembled unsymmetrized; `asymmetry` reports it).
    assert max(abs(h[i][j] - h[j][i]) for i in range(9) for j in range(9)) < 1e-12
    assert len(r.frequencies) == 3


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    path = pathlib.Path(env) if env else ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def test_python_equals_the_cli_bit_for_bit(tmp_path):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    # The SCF settings of `run_frequencies` (RhfConfig::default() + xc). The CLI
    # states the default RI-J/RI-K basis that `solve_rhf` resolves an unset
    # one to for a Kohn-Sham SCF.
    toml = tmp_path / "freq.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{ROOT / WATER}"\n\n[basis]\nname = "sto-3g"\n\n'
        '[method]\nkind = "ksdft"\ntask = "frequencies"\n\n'
        "[scf]\nmax_iter = 200\nenergy_conv = 1e-3\ndensity_conv = 1e-6\n"
        "df_guess = false\n\n"
        '[dft]\nfunctional = "PBE"\n\n'
        f'[frequencies]\nhessian = "{MOVING}"\n'
    )
    log = tmp_path / "run.jsonl"
    env = dict(os.environ)
    # Same rayon width as this process (rayon's default is the CPUs this
    # process may run on), so both run the same reduction order.
    env.setdefault("RAYON_NUM_THREADS", str(len(os.sched_getaffinity(0))))
    out = subprocess.run(
        [str(cli), str(toml), "--json", str(log)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    recs = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    rec = [r for r in recs if r.get("record") == "frequencies"]
    assert len(rec) == 1
    rec = rec[0]
    py = ferric.run_frequencies(_mol(WATER), "sto-3g", xc="PBE", hessian=MOVING)
    assert rec["hessian_source"] == py.hessian_source == MOVING
    assert rec["energy"] == py.energy
    assert rec["frequencies_cm1"] == py.frequencies
    assert rec["cartesian_hessian"] == [x for row in py.cartesian_hessian for x in row]


# Measured (water / STO-3G, the default SCF of run_frequencies, density_conv
# 1e-6, h = 5e-3 and 2.5e-3, 2026-10-23): max |d nu| 2.66e-3 (LDA) and
# 2.68e-3 (PBE) cm^-1 against the Richardson FD (1.8e-2 against plain FD at
# h/2): the loose default SCF limits it, the CLI validation test reaches
# 2.1e-7 at density_conv 1e-11. The bar is 11x that; the fixed-grid mutant
# misses by 4.63 (LDA) and 2.34 (PBE) cm^-1, 78 bars or more.
FD_TOL_CM1 = 0.03


@pytest.mark.validation
@pytest.mark.parametrize("xc", ["LDA", "PBE"])
def test_moving_grid_matches_fd(xc):
    mol = _mol(WATER)
    moving = ferric.run_frequencies(mol, "sto-3g", xc=xc, hessian=MOVING)
    h = 5e-3
    coarse = ferric.run_frequencies(mol, "sto-3g", xc=xc, hessian="fd", delta=h)
    fine = ferric.run_frequencies(mol, "sto-3g", xc=xc, hessian="fd", delta=0.5 * h)
    rich = [(4.0 * f - c) / 3.0 for f, c in zip(fine.frequencies, coarse.frequencies)]
    miss = _max_abs_diff(moving.frequencies, rich)
    print(
        f"{xc}: moving {moving.frequencies} FD(Richardson) {rich}: max |d nu| "
        f"{miss:.3e} cm^-1 (plain FD at h/2: "
        f"{_max_abs_diff(moving.frequencies, fine.frequencies):.3e})"
    )
    assert miss < FD_TOL_CM1


def test_auto_is_still_finite_differences_for_ks():
    mol = _mol(H2)
    auto = ferric.run_frequencies(mol, "sto-3g", xc="PBE")
    fd = ferric.run_frequencies(mol, "sto-3g", xc="PBE", hessian="fd")
    assert auto.hessian_source == fd.hessian_source == "finite-difference"
    assert auto.n_gradient_evaluations == 12
    assert auto.energy == fd.energy
    assert auto.frequencies == fd.frequencies
    assert auto.cartesian_hessian == fd.cartesian_hessian


def _hi(tmp_path):
    p = tmp_path / "hi.xyz"
    p.write_text("2\nHI\nH 0 0 0\nI 0 0 1.75\n")
    return ferric.Molecule.from_xyz(str(p), 0, 1)


@pytest.mark.parametrize(
    "mol, basis, kwargs, fragment",
    [
        (WATER, "sto-3g", {}, "Hartree-Fock has no XC grid"),
        (H2, "sto-3g", {"xc": "LDA", "reference": "uhf"}, "open-shell (Uhf"),
        (H2, "sto-3g", {"xc": "LDA", "reference": "rohf"}, "open-shell (Rohf"),
        (WATER, "sto-3g", {"xc": "SCAN"}, "a meta-GGA ('SCAN')"),
        (WATER, "sto-3g", {"xc": "wB97X-V"}, "VV10 nonlocal correlation"),
        (WATER, "sto-3g", {"xc": "HSE06"}, "a range-separated hybrid ('HSE06')"),
        ("HI", "def2-svp", {"xc": "LDA"}, "an ECP or an external potential"),
        (
            WATER,
            "sto-3g",
            {"xc": "LDA", "point_charges": [(0.5, 0.0, 0.0, 6.0)]},
            "an ECP or an external potential",
        ),
        (
            WATER,
            "sto-3g",
            {"xc": "LDA", "external_field": (0.0, 0.0, 1e-3)},
            "an ECP or an external potential",
        ),
        (
            WATER,
            "sto-3g",
            {"xc": "B3LYP", "dispersion": "d3bj"},
            'hessian = "analytic-moving-grid" is not available',
        ),
    ],
)
def test_refusals_are_named(tmp_path, mol, basis, kwargs, fragment):
    m = _hi(tmp_path) if mol == "HI" else _mol(mol)
    with pytest.raises(Exception) as exc:
        ferric.run_frequencies(m, basis, hessian=MOVING, **kwargs)
    assert fragment in str(exc.value)


def test_unknown_method_lists_the_new_value():
    with pytest.raises(ValueError, match="analytic-moving-grid"):
        ferric.run_frequencies(_mol(H2), "sto-3g", xc="LDA", hessian="moving")


def test_saddle_refuses_the_moving_grid_hessian():
    with pytest.raises(ValueError, match="not available in run_saddle"):
        ferric.run_saddle(_mol(H2), "sto-3g", xc="LDA", hessian=MOVING)
