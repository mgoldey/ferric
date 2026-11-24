"""A libint2 without an integral class must never abort `run_frequencies`.

conda-forge's win-64 libint2 2.13.1 has no 4-centre second derivatives; the
analytic Hessian used to build the engine anyway and abort the interpreter on
libint2's "null build function ptr" assert. ferric now checks the per-class
capability table first. The test stands in for that library by overriding the
table (`ferric._set_libint_max_am_for_test`, process-wide), in a subprocess so
that neither the override nor a native abort can reach this pytest process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import json, sys, ferric
    linked = ferric._set_libint_max_am_for_test(None)
    if linked[0][2] < 1:
        print(json.dumps({"skip": "linked libint2 lacks 4-centre deriv-2"}))
        sys.exit(0)
    mol = ferric.Molecule.from_xyz_string(
        "3\\n\\nO 0 0 0.117\\nH 0 0.757 -0.467\\nH 0 -0.757 -0.467\\n"
    )
    table = [row[:] for row in linked]
    table[0][2] = -1  # no 4-centre second derivatives
    ferric._set_libint_max_am_for_test(table)
    auto = ferric.run_frequencies(mol, "sto-3g")
    try:
        ferric.run_frequencies(mol, "sto-3g", hessian="analytic")
        analytic = None
    except Exception as e:  # noqa: BLE001 - the type is part of what is reported
        analytic = f"{type(e).__name__}: {e}"
    ferric._set_libint_max_am_for_test(None)
    full = ferric.run_frequencies(mol, "sto-3g")
    print(json.dumps({
        "auto": auto.hessian_source,
        "analytic": analytic,
        "full": full.hessian_source,
        "max_diff": max(abs(a - b) for a, b in zip(auto.frequencies, full.frequencies)),
    }))
    """
)


def test_missing_class_falls_back_or_raises_never_aborts():
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\nstderr:\n{proc.stderr[-4000:]}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skip" in out:
        import pytest

        pytest.skip(out["skip"])
    assert out["auto"] == "finite-difference"
    assert 'hessian = "auto" falls back to finite differences' in proc.stderr
    assert out["analytic"] is not None
    assert "libint2 integral class unavailable" in out["analytic"]
    assert "second-derivative 4-centre ERI" in out["analytic"]
    assert out["full"] == "analytic"
    assert out["max_diff"] < 1.0
