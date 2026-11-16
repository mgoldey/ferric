"""`run_frequencies(method="att-rimp2" | "pdep-rpa")` equals the CLI's
`task = "frequencies"` for the same kinds, bit for bit.

Both call the same library gradient (`total_att_rimp2_gradient_on`,
`total_rpa_gradient_by`) through `harmonic_frequencies_from_gradient`; the
CLI side is pinned against an energy stencil in
`crates/ferric-cli/tests/correlated_frequencies_cli.rs`. These tests pin the
binding: its kwargs reach the same configuration the TOML keys do.

They need the `ferric-cli` binary of the same tree: `FERRIC_CLI` (the test
FAILS if that path is missing), else `target/release/ferric-cli` (skips if
absent).

Mutation ledger (measured to fail, each on an assertion, then restored):
- `run_frequencies_att_rimp2` ignoring `omega` (0.420 in its place) fails
  `test_att_rimp2_omega_reaches_the_gradient`;
- `run_frequencies_att_rimp2` passing the `pdep-rpa` label / the wrong kind
  to `frequencies_certificate` fails the certificate capability assertion in
  `test_att_rimp2_equals_the_cli_bit_for_bit`;
- `run_frequencies_att_rimp2` ignoring `scf_jk` (always four-centre) fails
  `test_att_rimp2_rijk_reference_equals_the_cli`.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

ROOT = pathlib.Path(__file__).resolve().parents[3]
H2 = "2\nh2\nH 0 0 0\nH 0 0 0.78\n"
AUX = "cc-pvdz-ri"


def _cli_binary():
    env = os.environ.get("FERRIC_CLI")
    if env:
        assert pathlib.Path(env).is_file(), f"FERRIC_CLI={env} does not exist"
        return pathlib.Path(env)
    path = ROOT / "target" / "release" / "ferric-cli"
    return path if path.is_file() else None


def _cli_frequencies(tmp_path, kind, section, scf_extra):
    cli = _cli_binary()
    if cli is None:
        pytest.skip("no ferric-cli binary (build it, or set FERRIC_CLI)")
    geom = tmp_path / "h2.xyz"
    geom.write_text(H2)
    toml = tmp_path / "freq.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{geom}"\n\n[basis]\nname = "cc-pvdz"\n\n'
        f'[method]\nkind = "{kind}"\ntask = "frequencies"\n\n{section}\n'
        f"[frequencies]\ndelta = 5e-3\n\n[scf]\n{scf_extra}"
    )
    log = tmp_path / "run.jsonl"
    env = dict(os.environ)
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
    recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    rec = [r for r in recs if r.get("record") == "frequencies"]
    assert len(rec) == 1
    return rec[0]


@pytest.fixture(scope="module")
def h2():
    return ferric.Molecule.from_xyz_string(H2)


def _same(rec, py):
    assert rec["energy"] == py.energy
    assert rec["frequencies_cm1"] == py.frequencies
    assert rec["cartesian_hessian"] == [x for row in py.cartesian_hessian for x in row]
    assert rec["n_gradient_evaluations"] == py.n_gradient_evaluations


def test_att_rimp2_equals_the_cli_bit_for_bit(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path, "att-rimp2", f'[mp2]\nauxbasis = "{AUX}"\n', 'jk = "exact"\n'
    )
    py = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:att-rimp2"
    assert rec["certificate"]["capability"] == py.certificate["capability"]
    assert len(py.frequencies) == 1


def test_att_rimp2_omega_reaches_the_gradient(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path,
        "att-rimp2",
        f'[mp2]\nauxbasis = "{AUX}"\nomega = 0.8\n',
        'jk = "exact"\n',
    )
    py = ferric.run_frequencies(
        h2,
        "cc-pvdz",
        method="att-rimp2",
        auxbasis=AUX,
        scf_jk="exact",
        omega=0.8,
        delta=5e-3,
    )
    _same(rec, py)
    default = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    assert abs(default.frequencies[0] - py.frequencies[0]) > 1.0  # cm^-1


def test_att_rimp2_rijk_reference_equals_the_cli(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path, "att-rimp2", f'[mp2]\nauxbasis = "{AUX}"\n', 'jk = "rijk"\n'
    )
    py = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="rijk", delta=5e-3
    )
    _same(rec, py)
    exact = ferric.run_frequencies(
        h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, scf_jk="exact", delta=5e-3
    )
    assert exact.frequencies[0] != py.frequencies[0]


def test_pdep_rpa_equals_the_cli_bit_for_bit(tmp_path, h2):
    rec = _cli_frequencies(
        tmp_path, "pdep-rpa", f'[rpa]\nauxbasis = "{AUX}"\nn_quad = 8\n', ""
    )
    py = ferric.run_frequencies(
        h2, "cc-pvdz", method="pdep-rpa", auxbasis=AUX, n_quad=8, delta=5e-3
    )
    _same(rec, py)
    assert py.certificate["capability"] == "frequencies:pdep-rpa"


def test_att_rimp2_refusals(h2):
    with pytest.raises(ValueError, match="auxbasis"):
        ferric.run_frequencies(h2, "cc-pvdz", method="att-rimp2")
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, xc="pbe"
        )
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, rpa_gradient="fd"
        )
    with pytest.raises(ValueError, match="omega"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="att-rimp2", auxbasis=AUX, omega=-1.0
        )
    # att-rimp2-only kwargs are refused by the other methods.
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(h2, "cc-pvdz", omega=0.4)
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(
            h2, "cc-pvdz", method="pdep-rpa", auxbasis=AUX, scf_jk="exact"
        )
