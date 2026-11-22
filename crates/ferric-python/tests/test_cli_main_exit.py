"""`ferric._cli_main` (the `ferric` console script) exits like the native CLI.

It runs `ferric_cli::try_run`; an error is written to stderr in the CLI's
exact text and raised as `SystemExit(status)`, so the process exits with the
CLI's status (2 for a usage error, 1 otherwise) and prints no traceback.
`--help`, `--version` and a completed run exit 0. Each case runs in a
subprocess, as the console script does.
"""

import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("ferric")

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_WATER = _ROOT / "testdata" / "molecules" / "water.xyz"
_SYNOPSIS = (
    "usage: ferric [--verbose|-v] [--json <path>|--no-json] [--gpu <preset>] <input.toml>"
)


def _cli(*args, cwd=None):
    code = (
        "import sys, ferric; sys.argv = ['ferric'] + sys.argv[1:]; "
        "sys.exit(ferric._cli_main())"
    )
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=600,
    )


def test_help_and_version_exit_zero():
    h = _cli("--help")
    assert h.returncode == 0, h.stderr
    assert h.stderr.startswith(_SYNOPSIS + "\n")
    assert h.stdout == ""
    v = _cli("--version")
    assert v.returncode == 0, v.stderr
    assert v.stdout.startswith("ferric ")
    assert v.stderr == ""


def test_no_arguments_is_a_usage_error():
    r = _cli()
    assert r.returncode == 2
    assert r.stderr.startswith(_SYNOPSIS + "\n")
    assert r.stderr.endswith("libint version and limits) and exit.\n")
    assert "Traceback" not in r.stderr
    assert r.stdout == ""


def test_bad_gpu_preset_is_a_usage_error():
    r = _cli("--gpu", "fast", "in.toml")
    assert r.returncode == 2
    assert r.stderr == (
        'error: --gpu: invalid GPU preset "fast" '
        "(expected off, auto, on, mixed or auto-mixed)\n"
    )


def test_missing_input_exits_one_without_traceback(tmp_path):
    r = _cli(str(tmp_path / "nope.toml"))
    assert r.returncode == 1
    assert r.stderr.startswith(f'error: cannot read config file "{tmp_path / "nope.toml"}": ')
    assert r.stderr.count("\n") == 1
    assert r.stdout == ""


def test_small_run_exits_zero(tmp_path):
    toml = tmp_path / "in.toml"
    toml.write_text(
        f'[molecule]\nxyz = "{_WATER}"\n[basis]\nname = "sto-3g"\n[method]\nkind = "rhf"\n'
    )
    r = _cli(str(toml), "--no-json", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr
    assert "RHF" in r.stdout or "Total" in r.stdout or "energy" in r.stdout.lower()
