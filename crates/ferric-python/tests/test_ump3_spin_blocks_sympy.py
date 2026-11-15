"""scripts/ump3_spin_blocks_sympy.py must hold, and must fail under each mutation.

The script proves that the spin-blocked unrestricted MP3 of
``ferric-mp2/src/u_mp3.rs`` and ``u_mp3_residual.rs`` equals the spin-orbital
MP3 (exact sympy polynomial identities per term class) and checks it against
brute-force Rayleigh-Schroedinger theory. The unmutated run must exit 0. Each
mutation (dropped alpha-beta exchange ring, halved same-spin particle ladder)
must exit nonzero on its own term class.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sympy", reason="sympy is needed for the symbolic identities")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ump3_spin_blocks_sympy.py"


def _run(mutation):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", UMP3_MUTATION=mutation)
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_identities_hold():
    r = _run("")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all identities hold" in r.stdout


@pytest.mark.parametrize(
    ("mutation", "failing"),
    [
        ("drop_ab_exchange", "S2 E3 ph rings + exchange rings"),
        ("ladder_half", "S2 E3 pp ladders"),
    ],
)
def test_each_mutation_turns_the_script_red(mutation, failing):
    r = _run(mutation)
    assert r.returncode != 0, r.stdout
    lines = r.stdout.splitlines()
    assert any(ln.startswith("FAIL") and failing in ln for ln in lines), r.stdout
    assert any(ln.startswith("FAIL N E3") for ln in lines), r.stdout
