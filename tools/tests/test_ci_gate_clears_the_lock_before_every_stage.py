"""Every cargo stage in the gate must clear a squatting sccache first.

`scripts/ci-gate.sh` serializes cargo behind `flock /tmp/ferric-cargo.lock`
with a multi-hour wait. The sccache daemon cargo auto-spawns INHERITS that
lock fd and keeps it after the stage that spawned it exits, so the NEXT stage
queues behind a daemon that will never release it.

`ferric_clear_stale_sccache_lock` handles that, but it used to run ONCE at
gate startup -- before any stage had spawned a daemon, so it could not
possibly see the squatter that blocks a LATER stage.

MEASURED 2026-09-21: a `cargo doc` stage sat on a lock whose only holders were
an idle sccache (zero CPU-tick delta, zero rustc) and the gate's own queued
flock, with a 14400s wait ahead of it.

This test is DERIVED from the script rather than hand-listing the known call
sites: a stage added later without the clear fails here instead of silently
reintroducing a multi-hour stall.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "ci-gate.sh"
CLEAR = "ferric_clear_stale_sccache_lock"


def _lines():
    return GATE.read_text().splitlines()


def test_the_gate_script_exists_and_takes_the_lock():
    """Guard the guard: if these move, the test below passes vacuously."""
    assert GATE.is_file(), f"{GATE} is missing -- this test selects on it"
    body = GATE.read_text()
    assert CLEAR in body, f"{CLEAR} is not called at all in {GATE.name}"
    assert re.search(r"flock\s+-w", body), (
        "no `flock -w` in the gate -- the serialization this test is about "
        "has moved, and the check below would select nothing"
    )


def test_every_flock_acquisition_is_preceded_by_a_stale_lock_clear():
    """Derived, not hand-listed -- a NEW stage must also clear."""
    lines = _lines()
    flock_lines = [
        i
        for i, ln in enumerate(lines)
        if re.search(r"flock\s+-w", ln) and not ln.lstrip().startswith("#")
    ]
    assert flock_lines, "found no flock acquisitions to check"

    # The clear must appear within a short window above each acquisition:
    # close enough that it belongs to THIS stage, not to gate startup.
    WINDOW = 12
    unguarded = []
    for i in flock_lines:
        window = lines[max(0, i - WINDOW) : i]
        if not any(CLEAR in w and not w.lstrip().startswith("#") for w in window):
            unguarded.append(f"  line {i + 1}: {lines[i].strip()[:78]}")

    assert not unguarded, (
        "these cargo-lock acquisitions can queue behind a squatting sccache "
        f"daemon for the full flock wait -- call {CLEAR} immediately before "
        "each one:\n" + "\n".join(unguarded)
    )
