"""The gate's step timeout must bound the BUILD, not the queue for the lock.

`scripts/ci-gate.sh` used to run
`timeout 1800 flock -w 14400 /tmp/ferric-cargo.lock -c CMD`, so the 1800 s
clock also covered the time spent WAITING for the lock. On 2026-10-28, with
five gates queued, one step was killed while still waiting, never ran, and
was reported as "TIMEOUT under heavy load". The rest had to be stopped.

These tests extract the gate's real `run_locked` / `report_step` / `run_step`
functions and run them with second-scale timeouts against a private lock file
held by a background `flock`. No cargo or ferric is involved.

Outcomes the gate must tell apart:
  (a) the lock wait expired, so the step never ran;
  (b) the step got the lock and then timed out;
  (c) the step ran and failed.

And one invariant (review of #593): an INT/TERM/HUP to the gate's process
group must never leave the build ALIVE with the lock FREE. `timeout` puts the
build in its own process group, so killing only the lock holder would let the
next queued gate build alongside an orphan.
"""

import os
import signal

import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "scripts" / "ci-gate.sh"
FUNCS = ("run_locked", "lock_holder", "report_step", "run_step")

pytestmark = pytest.mark.skipif(
    not (shutil.which("flock") and shutil.which("timeout")),
    reason="needs util-linux flock and coreutils timeout",
)

# The pre-fix structure, for checking that the tests can see the bug.
OLD_RUN_LOCKED = """run_locked() {
    ferric_clear_stale_sccache_lock "$LOCK_FILE"
    timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECS}" \\
        flock -o -w "$LOCK_WAIT_SECS" "$LOCK_FILE" -c "$*"
}
"""

# 59b8dc361's run_locked: right ordering, but the lock holder (flock) stays in
# the gate's process group while `timeout` moves the build into its own, so a
# signal to the gate's group frees the lock and orphans the build.
UNGUARDED_RUN_LOCKED = """run_locked() {
    ferric_clear_stale_sccache_lock "$LOCK_FILE"
    flock -o -w "$LOCK_WAIT_SECS" -E "$LOCK_WAIT_EXPIRED_RC" "$LOCK_FILE" \\
        timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECS}" bash -c "$*"
}
"""


def _function(src, name):
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", src, re.MULTILINE | re.DOTALL)
    assert m, f"{name}() not found in {GATE.name} -- this test drives it"
    return m.group(0)


def _gate_functions(run_locked=None):
    src = GATE.read_text()
    rc = re.search(r"^LOCK_WAIT_EXPIRED_RC=(\d+)$", src, re.MULTILINE)
    assert rc, "LOCK_WAIT_EXPIRED_RC is not defined in the gate"
    parts = [f"LOCK_WAIT_EXPIRED_RC={rc.group(1)}"]
    for name in FUNCS:
        body = _function(src, name)
        if name == "run_locked" and run_locked is not None:
            body = run_locked
        parts.append(body)
    return "\n".join(parts)


def _harness(tmp_path, timeout_secs, lock_wait_secs, run_locked=None, clear=":"):
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -uo pipefail\n"
        f"ferric_clear_stale_sccache_lock() {{ {clear}; }}\n"
        "exec {GATE_OUT_FD}>&1\n"
        f"LOCK_FILE={tmp_path / 'cargo.lock'}\n"
        f"TIMEOUT_SECS={timeout_secs}\n"
        f"LOCK_WAIT_SECS={lock_wait_secs}\n"
        "FAILED=0 STEP_TIMED_OUT=0 STEP_LOCK_WAIT_EXPIRED=0 STEP_CHECK_FAILED=0\n"
        + _gate_functions(run_locked)
        + '\nrun_step "probe" "$1"; rc=$?\n'
        'echo "FLAGS rc=$rc FAILED=$FAILED STEP_TIMED_OUT=$STEP_TIMED_OUT '
        "STEP_LOCK_WAIT_EXPIRED=$STEP_LOCK_WAIT_EXPIRED "
        'STEP_CHECK_FAILED=$STEP_CHECK_FAILED"\n'
        "exit $rc\n"
    )
    return harness


def _run(tmp_path, cmd, timeout_secs, lock_wait_secs, run_locked=None):
    """Run one gate step; return (exit status, output, flags dict, wall start)."""
    harness = _harness(tmp_path, timeout_secs, lock_wait_secs, run_locked)
    t0 = time.time()
    p = subprocess.run(
        ["bash", str(harness), cmd],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    flags_line = [ln for ln in p.stdout.splitlines() if ln.startswith("FLAGS ")]
    assert flags_line, p.stdout + p.stderr
    flags = dict(kv.split("=") for kv in flags_line[-1].split()[1:])
    return p.returncode, p.stdout, flags, t0


def _alive(pid):
    """True for a live process; False if gone or a zombie (zombie-safe)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _kill_if_alive(pid):
    if _alive(pid):
        os.kill(pid, signal.SIGKILL)  # by PID, never by pattern


class _HeldLock:
    """Hold tmp_path/cargo.lock with a background flock for `secs` seconds.

    The holder stamps its own release time (just before it exits and drops
    the lock) into `released`, so tests do not depend on polling latency.
    """

    def __init__(self, tmp_path, secs):
        self.lock = tmp_path / "cargo.lock"
        self.stamp = tmp_path / "released"
        self.secs = secs

    def __enter__(self):
        self.proc = subprocess.Popen(
            [
                "flock",
                str(self.lock),
                "bash",
                "-c",
                f"sleep {self.secs}; date +%s.%N > {self.stamp}",
            ]
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            free = subprocess.run(["flock", "-n", str(self.lock), "true"]).returncode
            if free != 0:
                return self
            time.sleep(0.02)
        raise RuntimeError("background flock never took the lock")

    @property
    def released(self):
        return float(self.stamp.read_text())

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.kill()  # by PID, never by pattern
        self.proc.wait()


def _lock_is_free(tmp_path):
    lock = tmp_path / "cargo.lock"
    return subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0


def test_timeout_clock_starts_only_once_the_lock_is_held(tmp_path):
    """Lock held 3 s, step timeout 2 s, command takes 1 s: must PASS.

    The old ordering spends the 2 s timeout queueing and kills the step.
    """
    started, ended = tmp_path / "started", tmp_path / "ended"
    cmd = f"date +%s.%N > {started}; sleep 1; date +%s.%N > {ended}"
    with _HeldLock(tmp_path, 3) as held:
        rc, out, flags, t0 = _run(tmp_path, cmd, 2, 30)
    assert rc == 0, out
    assert "-- probe: PASS --" in out
    assert flags["FAILED"] == "0"
    # The command only started after the holder stamped its release.
    assert float(started.read_text()) >= held.released
    # ...and the step outlived its 2 s timeout counted from when it was
    # called, which only works if the clock started at the lock.
    assert float(ended.read_text()) - t0 > 2


def test_a_lock_wait_expiry_is_reported_as_not_run(tmp_path):
    """(a): the step never ran, and is not called a timeout or a failure."""
    marker = tmp_path / "ran"
    with _HeldLock(tmp_path, 4) as held:
        # A second waiter: `fuser` would list it too; the report must not.
        waiter = subprocess.Popen(["flock", str(held.lock), "true"])
        try:
            rc, out, flags, _ = _run(tmp_path, f"touch {marker}", 30, 1)
        finally:
            waiter.kill()
            waiter.wait()
    assert rc == 1, out
    assert not marker.exists(), "the command ran without the lock"
    assert "NOT RUN -- never got the cargo lock in 1s" in out
    # Names the HOLDER (the background flock), not a queued waiter.
    assert f"Held now by: pid {held.proc.pid} (flock " in out, out
    assert f"pid {waiter.pid}" not in out
    assert "TIMEOUT" not in out and "FAIL (exit" not in out
    assert flags["STEP_LOCK_WAIT_EXPIRED"] == "1"
    assert flags["STEP_TIMED_OUT"] == "0" and flags["STEP_CHECK_FAILED"] == "0"
    assert flags["FAILED"] == "1"


def test_a_build_timeout_is_reported_as_timeout_and_kills_the_build(tmp_path):
    """(b): the build is killed (including its children) and the lock freed."""
    pidfile = tmp_path / "child.pid"
    t0 = time.monotonic()
    rc, out, flags, _ = _run(tmp_path, f"sleep 30 & echo $! > {pidfile}; wait", 1, 30)
    elapsed = time.monotonic() - t0
    assert rc == 1, out
    assert "-- probe: TIMEOUT after 1s holding the lock (exit 124) --" in out
    assert flags["STEP_TIMED_OUT"] == "1"
    assert flags["STEP_LOCK_WAIT_EXPIRED"] == "0" and flags["STEP_CHECK_FAILED"] == "0"
    assert elapsed < 10, elapsed
    child = int(pidfile.read_text())
    try:
        assert not _alive(child), "TERM did not reach the build's child process"
    finally:
        _kill_if_alive(child)
    assert _lock_is_free(tmp_path), "lock still held after the step returned"


def test_a_build_failure_is_reported_as_fail(tmp_path):
    """(c): an ordinary nonzero exit is a FAIL with the command's status."""
    rc, out, flags, _ = _run(tmp_path, "exit 3", 30, 30)
    assert rc == 1, out
    assert "-- probe: FAIL (exit 3) --" in out
    assert flags["STEP_CHECK_FAILED"] == "1"
    assert flags["STEP_TIMED_OUT"] == "0" and flags["STEP_LOCK_WAIT_EXPIRED"] == "0"
    assert _lock_is_free(tmp_path)


def test_the_old_ordering_fails_the_queueing_scenario(tmp_path):
    """Guard the guard: the first test must be able to see the bug.

    With `timeout` wrapping `flock -w`, the same 3 s hold / 2 s timeout /
    1 s command is killed while still queued and reported as a TIMEOUT.
    """
    with _HeldLock(tmp_path, 3):
        rc, out, flags, _ = _run(tmp_path, "sleep 1", 2, 30, run_locked=OLD_RUN_LOCKED)
    assert rc != 0, out
    assert flags["STEP_TIMED_OUT"] == "1", out


def test_the_command_does_not_inherit_the_lock_fd(tmp_path):
    """What -o guaranteed: a daemon the build spawns (sccache) cannot squat."""
    fds = tmp_path / "fds"
    rc, out, _, _ = _run(tmp_path, f"ls -l /proc/$$/fd > {fds}", 30, 30)
    assert rc == 0, out
    assert "cargo.lock" not in fds.read_text(), fds.read_text()


def test_lock_notes_are_not_captured_by_a_step_redirect(tmp_path):
    """Clippy sends run_locked's stdout+stderr to its JSON file; the stale-
    sccache note must still reach the gate's own output."""
    harness = _harness(tmp_path, 30, 30, clear="echo NOTE-stale-sccache")
    script = harness.read_text().replace(
        'run_step "probe" "$1"; rc=$?',
        f'run_locked "$1" > {tmp_path / "captured"} 2>&1; rc=$?',
    )
    harness.write_text(script)
    p = subprocess.run(
        ["bash", str(harness), "echo payload"], capture_output=True, text=True
    )
    assert "NOTE-stale-sccache" in p.stdout
    captured = (tmp_path / "captured").read_text()
    assert "payload" in captured and "NOTE" not in captured


def _interrupt_mid_step(tmp_path, sig, run_locked=None):
    """Signal the harness's whole process group while the step is building.

    Returns a list of violations: moments where the build was alive and the
    lock free. Leaves no process behind.
    """
    pidfile = tmp_path / "child.pid"
    harness = _harness(tmp_path, 60, 30, run_locked)
    cmd = f"sleep 60 & echo $! > {pidfile}; wait"
    p = subprocess.Popen(["bash", str(harness), cmd], start_new_session=True)
    child = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (
            pidfile.exists() and pidfile.read_text().strip()
        ):
            time.sleep(0.02)
        child = int(pidfile.read_text())
        os.killpg(p.pid, sig)  # what Ctrl-C / a hangup / a tool does
        p.wait(timeout=60)
        violations = []
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            alive, free = _alive(child), _lock_is_free(tmp_path)
            if alive and free:
                violations.append("build alive with the lock free")
                break
            if not alive:
                break
            time.sleep(0.05)
        return violations, p.returncode, child
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        if child is not None:
            _kill_if_alive(child)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_a_signal_never_leaves_the_build_running_without_the_lock(tmp_path, sig):
    violations, harness_rc, child = _interrupt_mid_step(tmp_path, sig)
    assert not violations, violations
    # The forwarded TERM actually stopped the build, and the gate stopped too
    # instead of carrying on to its next step.
    assert not _alive(child), "the build survived the interrupt"
    assert harness_rc != 0


def test_the_unguarded_holder_fails_the_signal_invariant(tmp_path):
    """Guard the guard: 59b8dc361's run_locked must fail the test above."""
    violations, _, _ = _interrupt_mid_step(
        tmp_path, signal.SIGINT, run_locked=UNGUARDED_RUN_LOCKED
    )
    assert violations == ["build alive with the lock free"]
