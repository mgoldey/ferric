"""scripts/local_opmp3_sympy.py must hold, and must fail under each mutation.

The script proves what local (amplitude-threshold) OP-MP3 relies on: the
trace Pade and the W matrices are invariant under the occupied and virtual
localization, the Wigner 2n+1 identity E3 = <G c2, J> = <G c1, R(c1)> holds
when c1 and c2 share one mask (and fails on two), and the non-Brillouin
singles energy is a reference-wide term the doubles mask does not touch,
with a canonical-vs-local numerical cross-check.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sympy", reason="sympy is needed for the symbolic identities")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "local_opmp3_sympy.py"


def _run(mutation):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", LOCAL_OPMP3_MUTATION=mutation)
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT)],
        capture_output=True,
        check=False,
        text=True,
        env=env,
        timeout=600,
    )


def test_identities_hold():
    r = _run("")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all identities hold" in r.stdout
    assert "FAIL" not in r.stdout


@pytest.mark.parametrize(
    ("mutation", "failing"),
    [
        ("pade_half_rotated", "a1 Tr P(U^T A U"),
        ("w_virtual_once", "a2 W[c U V"),
        ("two_masks", "N2 one mask"),
        ("semicanonical", "N1 vacuous mask"),
        ("nbs_local_diag", "N3 E_NBS"),
    ],
)
def test_each_mutation_turns_the_script_red(mutation, failing):
    r = _run(mutation)
    assert r.returncode != 0, r.stdout
    lines = r.stdout.splitlines()
    assert any(ln.startswith("FAIL") and failing in ln for ln in lines), r.stdout
