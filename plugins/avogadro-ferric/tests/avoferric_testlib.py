"""Drive the plugin the way Avogadro 2.1.0 does: a subprocess per invocation,
input JSON on stdin, JSON (or the energy text protocol) on stdout.

The package is not installed for these tests: ``src/`` goes on PYTHONPATH, and
the entry point is run as ``python -m avogadro_ferric <identifier> ...``, which
calls the same ``main()`` the ``avogadro-ferric`` console script does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled ferric extension")

PLUGIN_DIR = Path(__file__).resolve().parents[1]
SRC = PLUGIN_DIR / "src"
sys.path.insert(0, str(SRC))

# Water, Angstrom.
WATER = [(8, 0.0, 0.0, 0.117), (1, 0.0, 0.757, -0.469), (1, 0.0, -0.757, -0.469)]
# OH radical (doublet), Angstrom.
OH = [(8, 0.0, 0.0, 0.0), (1, 0.0, 0.0, 0.97)]
# Hydronium, Angstrom (pyramidal).
H3O = [
    (8, 0.0, 0.0, 0.0),
    (1, 0.0, 0.94, 0.30),
    (1, 0.814, -0.47, 0.30),
    (1, -0.814, -0.47, 0.30),
]


def make_cjson(atoms, charge=0, mult=1, bonds=None):
    cj = {
        "chemicalJson": 1,
        "atoms": {
            "elements": {"number": [a[0] for a in atoms]},
            "coords": {"3d": [c for a in atoms for c in a[1:]]},
        },
        "properties": {"totalCharge": charge, "totalSpinMultiplicity": mult},
    }
    if bonds:
        cj["bonds"] = {
            "connections": {"index": [i for b in bonds for i in b]},
            "order": [1] * len(bonds),
        }
    return cj


def menu_input(atoms, charge=0, mult=1, options=None, bonds=None):
    """The JSON Avogadro writes for a package menu command (interfacescript.cpp)."""
    return {
        "cjson": make_cjson(atoms, charge, mult, bonds),
        "charge": charge,
        "spin": mult,
        "selectedAtoms": [],
        "options": options or {},
    }


@pytest.fixture
def plugin_env(tmp_path):
    """Environment for a plugin subprocess with its own config file."""

    def make(method="HF", basis="sto-3g", charge_model="lowdin"):
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {"method": method, "basis": basis, "charge_model": charge_model}
            ),
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        env["AVOGADRO_FERRIC_CONFIG"] = str(cfg)
        # Never read or write the user's ~/.cache from a test.
        env["AVOGADRO_FERRIC_CACHE"] = str(tmp_path / "cache")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        return env

    return make


def run_plugin(env, args, payload, timeout=600):
    """One-shot invocation; returns the parsed JSON reply."""
    proc = subprocess.run(
        [sys.executable, "-m", "avogadro_ferric", *args, "--lang", "en_US"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    assert out, f"no stdout; stderr:\n{proc.stderr}"
    # Avogadro skips progress envelopes and parses the rest; the reply must be
    # exactly one JSON document.
    return json.loads(out), proc.stderr
