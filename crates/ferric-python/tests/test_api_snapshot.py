"""Snapshot of the public `ferric` Python API: names, signatures, defaults.

`test_pyi_*` check that names and parameter order agree with the stub, but not
default VALUES, result-class attributes, or methods. Downstream users (e.g. the
smeltery app) rely on stable kwargs, defaults, attribute names and feature
presence, so this test compares the live module against the committed
`data/api_snapshot.json`.

Policy:
  * Removals, renames, and changed defaults/kinds are BREAKING. They need
    deliberate review: do not just regenerate to make the test green.
  * Additions (new functions, classes, members, or trailing keyword
    parameters) are fine, but the snapshot must be regenerated so the new
    surface is pinned from then on.

Regenerate with either of:

    FERRIC_UPDATE_API_SNAPSHOT=1 pytest crates/ferric-python/tests/test_api_snapshot.py
    python crates/ferric-python/tests/api_snapshot.py
"""

from __future__ import annotations

import difflib
import json
import os

import pytest
from api_snapshot import SNAPSHOT, generate, optional_missing, render

HOW_TO = (
    "To regenerate after a DELIBERATE, reviewed API change:\n"
    "  FERRIC_UPDATE_API_SNAPSHOT=1 pytest crates/ferric-python/tests/test_api_snapshot.py\n"
    "  (or: python crates/ferric-python/tests/api_snapshot.py)\n"
    "Additions are fine; removals/renames/default changes are breaking and\n"
    "need review (see the module docstring)."
)


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        for k in sorted(node):
            yield from _flatten(node[k], f"{prefix}{k}.")
    else:
        yield prefix.rstrip("."), json.dumps(node)


def test_public_api_matches_snapshot():
    live = render(generate())
    if os.environ.get("FERRIC_UPDATE_API_SNAPSHOT"):
        assert not optional_missing(), (
            f"regenerate with the optional deps installed: {optional_missing()}"
        )
        SNAPSHOT.write_text(live)
        pytest.skip(f"snapshot rewritten: {SNAPSHOT}")
    assert SNAPSHOT.exists(), f"missing {SNAPSHOT}\n{HOW_TO}"
    committed = SNAPSHOT.read_text()
    if live == committed:
        return
    committed_data = json.loads(committed)
    # A submodule whose optional dependency is missing here (ferric.ase without
    # ASE) is not compared: drop its committed section, deterministically.
    skipped = optional_missing()
    for name in skipped:
        committed_data.get("submodules", {}).pop(name, None)
    if json.loads(live) == committed_data:
        return
    old, new = dict(_flatten(committed_data)), dict(_flatten(json.loads(live)))
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    lines = []
    if removed or changed:
        lines.append("BREAKING (removed or changed):")
        lines += [f"  - {k} = {old[k]}" for k in removed]
        lines += [f"  ~ {k}: {old[k]} -> {new[k]}" for k in changed]
    if added:
        lines.append("Additions (regenerate the snapshot):")
        lines += [f"  + {k} = {new[k]}" for k in added]
    udiff = "".join(
        list(
            difflib.unified_diff(
                committed.splitlines(True),
                live.splitlines(True),
                "committed",
                "live",
                n=1,
            )
        )[:200]
    )
    pytest.fail(
        "public API drifted from snapshot\n"
        + "\n".join(lines)
        + f"\n\n{udiff}\n{HOW_TO}"
    )
