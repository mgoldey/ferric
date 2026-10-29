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

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "scripts" / "ci-gate.sh"
LIB = REPO / "scripts" / "cargo-lock-lib.sh"
CLEAR = "ferric_clear_stale_sccache_lock"


def _lines():
    return GATE.read_text().splitlines()


# Any `flock ... -w` acquisition, whatever option order (`flock -o -w`, #591).
ACQUIRE = r"\bflock\b[^\n]*\s-w\b"
FUNC_START = re.compile(r"^\w+\(\) \{$")


def _enclosing_function(lines, i):
    """(start, end) line indices of the function containing line i, or None."""
    for start in range(i, -1, -1):
        if FUNC_START.match(lines[start]):
            break
    else:
        return None
    for end in range(i, len(lines)):
        if lines[end] == "}":
            return start, end
    return None


def test_the_gate_script_exists_and_takes_the_lock():
    """Guard the guard: if these move, the test below passes vacuously."""
    assert GATE.is_file(), f"{GATE} is missing -- this test selects on it"
    body = GATE.read_text()
    assert CLEAR in body, f"{CLEAR} is not called at all in {GATE.name}"
    assert re.search(ACQUIRE, body), (
        "no `flock -w` in the gate -- the serialization this test is about "
        "has moved, and the check below would select nothing"
    )


def test_every_flock_acquisition_is_preceded_by_a_stale_lock_clear():
    """Derived, not hand-listed -- a NEW stage must also clear."""
    lines = _lines()
    flock_lines = [
        i
        for i, ln in enumerate(lines)
        if re.search(ACQUIRE, ln) and not ln.lstrip().startswith("#")
    ]
    assert flock_lines, "found no flock acquisitions to check"

    # The clear must come before each acquisition in the same function, or
    # within a short window above a top-level one: close enough that it
    # belongs to THIS stage, not to gate startup.
    WINDOW = 12
    unguarded = []
    for i in flock_lines:
        func = _enclosing_function(lines, i)
        window = lines[func[0] : i] if func else lines[max(0, i - WINDOW) : i]
        if not any(CLEAR in w and not w.lstrip().startswith("#") for w in window):
            unguarded.append(f"  line {i + 1}: {lines[i].strip()[:78]}")

    assert not unguarded, (
        "these cargo-lock acquisitions can queue behind a squatting sccache "
        f"daemon for the full flock wait -- call {CLEAR} immediately before "
        "each one:\n" + "\n".join(unguarded)
    )


def test_every_flock_acquisition_closes_the_lock_fd_in_the_command():
    """Without -o, the sccache cargo spawns inherits the lock and squats it.

    The clearer above cannot help once other gates are queued, because their
    waiting flocks count as real consumers (#591: five pushes stuck 20-33 min).

    Every acquisition carries -o, whatever its shape. In the fd form
    `flock -o -w ... "$fd"` (#593's run_locked, where a subshell holds the lock)
    there is no command for -o to act on, so that shape must ALSO close the fd
    in the command it runs (`{fd}>&-`). The behaviour itself is pinned in
    test_ci_gate_timeout_starts_after_the_lock.py.
    """
    lines = _lines()
    missing = []
    leaks_fd = []
    for i, ln in enumerate(lines):
        if not re.search(ACQUIRE, ln) or ln.lstrip().startswith("#"):
            continue
        if not re.search(r"\bflock\b[^\n]*\s-o\b", ln):
            missing.append(ln.strip())
        fd = re.search(r'"\$(\w+)"\s*$', ln)
        func = _enclosing_function(lines, i)
        # On a COMMAND's redirection: `exec {fd}>&-` (the holder dropping
        # the lock itself) does not keep the fd from the command.
        after = "\n".join(
            ln
            for ln in lines[i + 1 : func[1] if func else i + 8]
            if not ln.lstrip().startswith("exec ")
        )
        if fd and not re.search(r"\{" + fd.group(1) + r"\}>&-", after):
            leaks_fd.append(ln.strip())
    assert not missing, f"flock acquisitions without -o: {missing}"
    assert not leaks_fd, f"fd-form acquisitions whose command keeps the fd: {leaks_fd}"


# ---- the clearer's discriminator, driven for real --------------------------
# A process whose comm is `sccache`: a copy of `sleep` under that name.

needs_flock = pytest.mark.skipif(
    not (shutil.which("flock") and shutil.which("sleep")),
    reason="needs util-linux flock and coreutils sleep",
)


def _fake_sccache(tmp_path):
    fake = tmp_path / "sccache"
    shutil.copy(shutil.which("sleep"), fake)
    return fake


def _alive(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _held(lock):
    return subprocess.run(["flock", "-n", str(lock), "true"]).returncode != 0


def _wait_until(cond, secs=5):
    deadline = time.monotonic() + secs
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _clear(lock):
    return subprocess.run(
        ["bash", "-c", f'source {LIB}; ferric_clear_stale_sccache_lock "{lock}"'],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout


def _stop(*procs):
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGKILL)  # by PID, never by pattern
        p.wait()


@needs_flock
def test_a_squatting_sccache_is_cleared_even_with_gates_queued(tmp_path):
    """sccache alone HOLDS the lock; a queued waiter must not protect it.

    The old check counted every process with the file open, so a waiting
    `flock` read as a real consumer (#591: five queued pushes, 20-33 min).
    """
    lock = tmp_path / "cargo.lock"
    fake = _fake_sccache(tmp_path)
    squatter = subprocess.Popen(
        ["bash", "-c", f'exec 3>>"{lock}"; flock 3; exec "{fake}" 60']
    )
    waiter = None
    try:
        assert _wait_until(
            lambda: (
                _held(lock)
                and Path(f"/proc/{squatter.pid}/comm").read_text() == "sccache\n"
            )
        ), "the fake sccache never took the lock"
        waiter = subprocess.Popen(["flock", str(lock), "true"])
        time.sleep(0.2)
        out = _clear(lock)
        assert f"releasing stale sccache lock holder (pid {squatter.pid}" in out, out
        assert _wait_until(lambda: not _alive(squatter.pid))
        assert waiter.wait(timeout=10) == 0, "the queued waiter never got the lock"
    finally:
        _stop(squatter, *([waiter] if waiter else []))


@needs_flock
def test_an_sccache_that_does_not_hold_the_lock_is_left_alone(tmp_path):
    """A gate's run_locked subshell (comm bash) holds; sccache only has the
    file open. Killing it would take the cache server out from under a build.

    The old check saw bash + sccache, no cargo/rustc/flock, and killed it.
    """
    lock = tmp_path / "cargo.lock"
    fake = _fake_sccache(tmp_path)
    holder = subprocess.Popen(
        ["bash", "-c", f'exec 3>>"{lock}"; flock 3; sleep 60 3>&-; :']
    )
    server = None
    try:
        assert _wait_until(lambda: _held(lock)), "the holder never took the lock"
        server = subprocess.Popen(["bash", "-c", f'exec 3>>"{lock}"; exec "{fake}" 60'])
        assert _wait_until(
            lambda: Path(f"/proc/{server.pid}/comm").read_text() == "sccache\n"
        )
        out = _clear(lock)
        assert "releasing" not in out, out
        time.sleep(0.3)
        assert _alive(server.pid), "the clearer killed an sccache serving a build"
        assert os.path.exists(f"/proc/{holder.pid}")
    finally:
        _stop(holder, *([server] if server else []))
