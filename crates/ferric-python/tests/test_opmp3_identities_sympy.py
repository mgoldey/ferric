"""scripts/opmp3_identities_sympy.py must hold, and must fail under each mutation.

The script proves the OP-MP3 identities of ``ferric-mp2/src/opmp3.rs``: the
Eq. 18 same-spin / opposite-spin split of W from the spin-orbital Eq. 11a,
the matrix Pade series and scalar limit, its size consistency (and the
scalar Pade's failure of it), and Tr W^(n) = E^(n), with a brute-force
Rayleigh-Schroedinger cross-check. The unmutated run must exit 0; each
mutation must exit nonzero with its own identity red.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sympy", reason="sympy is needed for the symbolic identities")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "opmp3_identities_sympy.py"


def _run(mutation):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OPMP3_MUTATION=mutation)
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
    assert "FAIL" not in r.stdout


@pytest.mark.parametrize(
    ("mutation", "failing"),
    [
        ("os_eighth", "a3 k-in-sigma-bar part == Eq. 18"),
        ("pade_swapped", "b1 P(A, tB)"),
        ("scalar_in_block", "c1 Tr P(A1+A2, B1+B2)"),
        ("w_unsymmetrized", "d2 Tr W^(2) == RS E2"),
    ],
)
def test_each_mutation_turns_the_script_red(mutation, failing):
    r = _run(mutation)
    assert r.returncode != 0, r.stdout
    lines = r.stdout.splitlines()
    assert any(ln.startswith("FAIL") and failing in ln for ln in lines), r.stdout
