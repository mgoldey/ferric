"""The CLI's integral-direct local MP2 equals Python's on the same input.

`[local] scheme = "amplitude-threshold"`, `integral_direct = true` (CLI
`kind = "rimp2"`) and `run_rimp2(local="amplitude-threshold",
integral_direct=True)` both build their library configs with
`ferric_cli::LocalModel::rimp2_configs`, whose map defaults come from
`ferric_mp2::lmp2_direct::DirectConfig::production`. With no map key given,
the two entry points must resolve the same model (every map, the Fock screen,
the pair gate, lambda-Boys, frozen core) and land on the same energy.

The CLI reference SCF is pinned to the Python route (`[scf] jk = "exact"`,
`df_guess = false`): the CLI's default RI-JK reference and its DF-guess
pre-stage move E_corr by 1e-5 and 4e-8 Ha on n-butane, which is a documented
difference of the SCF defaults, not of the local model.

Artifact hypothesis: a default that differs between the entry points (a map,
the Fock screen, the gate, lambda-Boys) changes the mask, which moves E_corr
by >= 1e-7 Ha on n-butane / 6-31G at eps = 1e-4 and changes the reported map
fields. Real agreement: identical model fields and |dE_corr| at the
round-off floor (the same code on the same orbitals).

Mutation ledger (measured to fail, see the commit): Python's
`run_rimp2_local` building its DirectConfig with `virt_radius_bohr =
Some(12.0)` instead of the shared builder.

Needs the `ferric-cli` binary of the same tree: `FERRIC_CLI` (the test FAILS
if that path is missing), else `target/release/ferric-cli` (skips if absent).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
XYZ = "testdata/molecules/alkane_4.xyz"

# Model fields both entry points report (certificate fields).
MODEL_KEYS = (
    "eps",
    "integral_direct",
    "gate_cal",
    "fock_screen",
    "frozen_core",
    "lambda_boys",
    "maps",
    "operator",
    "precision",
)

# Same code on the same orbitals: measured dE_corr = 0.0 exactly (bit for
# bit, 2026-11-18, same rayon width). The bar leaves room for a reduction-order
# change only; a default changed on one side moves the mask by far more.
TOL_E = 1e-10


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        # Resolve now: the CLI runs with cwd=ROOT, so a relative path given
        # from elsewhere would point at the wrong file.
        path = pathlib.Path(env).resolve()
        assert path.is_file(), f"FERRIC_CLI={env} does not exist"
        return path
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _cli_result(tmp_path, basis):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    toml = tmp_path / "input.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{XYZ}"\n[basis]\nname = "{basis}"\n'
        '[method]\nkind = "rimp2"\ntask = "energy"\n'
        '[scf]\njk = "exact"\ndf_guess = false\n'
        '[mp2]\nauxbasis = "cc-pvdz-ri"\n'
        '[local]\nscheme = "amplitude-threshold"\neps = 1e-4\n'
        "integral_direct = true\nreference = true\n"
    )
    log = tmp_path / "run.jsonl"
    env = dict(os.environ)
    # Same rayon width as this process, so both run the same reduction order.
    if "RAYON_NUM_THREADS" not in env:
        # sched_getaffinity is Linux-only (absent on macOS and Windows).
        affinity = getattr(os, "sched_getaffinity", None)
        width = len(affinity(0)) if affinity else (os.cpu_count() or 1)
        env["RAYON_NUM_THREADS"] = str(width)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
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
    hits = [r for r in recs if r.get("record") == "result" and r.get("kind") == "rimp2"]
    assert len(hits) == 1, [r.get("record") for r in recs]
    return hits[0]["components"]


@pytest.mark.parametrize("basis", ["6-31g"])
def test_cli_and_python_integral_direct_lmp2_agree(tmp_path, basis):
    cli = _cli_result(tmp_path, basis)
    mol = ferric.Molecule.from_xyz(str(ROOT / XYZ))
    py = ferric.run_rimp2(
        mol,
        ferric.BasisSet.bundled(basis),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
        local="amplitude-threshold",
        eps=1e-4,
        integral_direct=True,
        compute_reference=True,
    )
    loc = py.local
    for k in MODEL_KEYS:
        assert cli["local"][k] == loc[k], (k, cli["local"][k], loc[k])
    # the defaults, as resolved (a change here is a model change)
    assert loc["maps"]["virt_radius"] == 20.0 and loc["maps"]["ao_tail"] == 1e-4
    assert loc["fock_screen"] == 1.0 and loc["gate_cal"] is None
    assert loc["lambda_boys"] == 10.0
    assert abs(cli["e_corr_canonical_ri"] - loc["e_corr_canonical_ri"]) <= TOL_E
    assert abs(cli["e_corr"] - py.mp2_corr) <= TOL_E, (cli["e_corr"], py.mp2_corr)
