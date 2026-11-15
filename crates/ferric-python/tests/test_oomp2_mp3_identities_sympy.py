"""scripts/oomp2_mp3_identities_sympy.py must hold, and must fail under each mutation.

The script proves, with sympy.physics.secondquant plus exact numeric oracles,
the MP3 ring/ladder prefactors of ``ferric-mp2/src/mp3.rs`` and the full OO-MP2
orbital gradient of ``ferric-mp2/src/oo_rimp2.rs``. The unmutated run must exit
0. Each mutation (ring coefficient 1 -> 1/4, flipped kappa generator, dropped
two-particle correlation density) must exit nonzero: a gate that has never been
seen to fail is an assumption.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "sympy", reason="sympy is needed to run the second-quantization identities"
)

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "oomp2_mp3_identities_sympy.py"
)


def _run(mutation):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OOMP2_MP3_MUTATION=mutation)
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
        ("ring_quarter", "A2 ring: the mp3.rs coefficient equals the derived one"),
        ("kappa_sign", "B2 generalized Fock"),
        ("drop_two_particle", "B2 generalized Fock"),
    ],
)
def test_each_mutation_turns_the_script_red(mutation, failing):
    r = _run(mutation)
    assert r.returncode != 0, r.stdout
    assert any(
        line.startswith("FAIL") and failing in line for line in r.stdout.splitlines()
    ), r.stdout
