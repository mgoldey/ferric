"""`[scf] mom_reference` equals `run_uhf(mom_reference=)` bit for bit.

System: the water cation doublet, UHF/STO-3G, from the example
`examples/water-cation-uhf-initial-mom.toml`. The CLI has no starting-density
input, so the "initial" reference here holds the occupied block of the
configured guess; a core hole (`dm0=`) is a Python-only start and is covered
by the Delta-SCF tests.

Bar: bitwise (`==` on the printed 17-digit energies). Both surfaces call the
same `solve_uhf`. The two references differ at the last digits on this system
(about 6e-14 Ha), so ignoring the key in the CLI changes the energy.

They need the `ferric-cli` binary of the same tree: `FERRIC_CLI`, else
`target/release/ferric-cli` (skips if absent).

Mutation ledger (measured to fail): the CLI mapping hard-coding
`MomReference::Previous` fails the bitwise test.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples" / "water-cation-uhf-initial-mom.toml"


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _run_cli(tmp_path, scf_lines, kind="uhf", mult=2):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    log = tmp_path / "run.jsonl"
    toml = tmp_path / "in.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{ROOT}/testdata/molecules/water.xyz"\n'
        f"charge = {mult - 1}\nmultiplicity = {mult}\n\n"
        f'[basis]\nname = "sto-3g"\n\n[method]\nkind = "{kind}"\n\n'
        "[scf]\ndensity_conv = 1e-9\nenergy_conv = 1e-10\nmax_iter = 100\n"
        f'{scf_lines}\n[output]\njson = "{log}"\n'
    )
    p = subprocess.run([str(cli), str(toml)], capture_output=True, text=True)
    return p, log


def _cli_energy(log):
    events = [json.loads(line) for line in log.read_text().splitlines()]
    return [e["energy"] for e in events if "energy" in e][-1]


def _py_energy(reference):
    mol = ferric.Molecule.from_xyz(
        str(ROOT / "testdata/molecules/water.xyz"), charge=1, multiplicity=2
    )
    r = ferric.run_uhf(
        mol,
        ferric.BasisSet.bundled("sto-3g"),
        max_iter=100,
        energy_conv=1e-10,
        density_conv=1e-9,
        mom_reference=reference,
    )
    return r.energy


@pytest.mark.parametrize("reference", ["previous", "initial"])
def test_cli_energy_is_bitwise_the_python_energy(tmp_path, reference):
    p, log = _run_cli(tmp_path, f'mom_reference = "{reference}"\n')
    assert p.returncode == 0, p.stderr[-400:]
    assert _cli_energy(log) == _py_energy(reference)


def test_the_example_parses_and_runs(tmp_path):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    p = subprocess.run([str(cli), str(EXAMPLE)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-400:]


@pytest.mark.parametrize(
    "scf_lines, kind, mult, needle",
    [
        ('mom_reference = "start"\n', "uhf", 2, "unknown value"),
        ('mom_reference = "initial"\n', "rhf", 1, "UHF/UKS only"),
        ('mom_reference = "initial"\nmom_after_iter = 3\n', "uhf", 2, "mom_after_iter"),
        (
            'mom_reference = "initial"\nstability_descent = true\n',
            "uhf",
            2,
            "scf_stability_descent",
        ),
    ],
)
def test_cli_refuses_what_the_library_refuses(tmp_path, scf_lines, kind, mult, needle):
    p, _ = _run_cli(tmp_path, scf_lines, kind=kind, mult=mult)
    assert p.returncode != 0
    assert needle in p.stderr, p.stderr[-600:]
