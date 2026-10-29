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
"""

import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "scripts" / "ci-gate.sh"
FUNCS = ("run_locked", "report_step", "run_step")

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


def _run(tmp_path, cmd, timeout_secs, lock_wait_secs, run_locked=None):
    """Run one gate step; return (exit status, output, flags dict, seconds)."""
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -uo pipefail\n"
        "ferric_clear_stale_sccache_lock() { :; }\n"
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
    t0 = time.monotonic()
    p = subprocess.run(
        ["bash", str(harness), cmd],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    elapsed = time.monotonic() - t0
    flags_line = [ln for ln in p.stdout.splitlines() if ln.startswith("FLAGS ")]
    assert flags_line, p.stdout + p.stderr
    flags = dict(kv.split("=") for kv in flags_line[-1].split()[1:])
    return p.returncode, p.stdout, flags, elapsed


class _HeldLock:
    """Hold tmp_path/cargo.lock with a background flock for `secs` seconds."""

    def __init__(self, tmp_path, secs):
        self.lock = tmp_path / "cargo.lock"
        self.secs = secs

    def __enter__(self):
        self.proc = subprocess.Popen(["flock", str(self.lock), "sleep", str(self.secs)])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            free = subprocess.run(["flock", "-n", str(self.lock), "true"]).returncode
            if free != 0:
                self.acquired = time.time()
                return self
            time.sleep(0.02)
        raise RuntimeError("background flock never took the lock")

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
    stamp = tmp_path / "started"
    with _HeldLock(tmp_path, 3) as held:
        rc, out, flags, elapsed = _run(
            tmp_path, f"date +%s.%N > {stamp}; sleep 1", 2, 30
        )
        released = held.acquired + held.secs
    assert rc == 0, out
    assert "-- probe: PASS --" in out
    assert flags["FAILED"] == "0"
    # The command only started after the holder let go of the lock.
    started = float(stamp.read_text())
    assert started >= released - 0.25, (started, released)
    assert elapsed >= 3.5, elapsed  # 3 s queued + 1 s of work


def test_a_lock_wait_expiry_is_reported_as_not_run(tmp_path):
    """(a): the step never ran, and is not called a timeout or a failure."""
    marker = tmp_path / "ran"
    with _HeldLock(tmp_path, 4):
        rc, out, flags, _ = _run(tmp_path, f"touch {marker}", 30, 1)
    assert rc == 1, out
    assert not marker.exists(), "the command ran without the lock"
    assert "NOT RUN -- never got the cargo lock in 1s" in out
    assert "TIMEOUT" not in out and "FAIL (exit" not in out
    assert flags["STEP_LOCK_WAIT_EXPIRED"] == "1"
    assert flags["STEP_TIMED_OUT"] == "0" and flags["STEP_CHECK_FAILED"] == "0"
    assert flags["FAILED"] == "1"


def test_a_build_timeout_is_reported_as_timeout_and_kills_the_build(tmp_path):
    """(b): the build is killed (including its children) and the lock freed."""
    pidfile = tmp_path / "child.pid"
    rc, out, flags, elapsed = _run(
        tmp_path, f"sleep 30 & echo $! > {pidfile}; wait", 1, 30
    )
    assert rc == 1, out
    assert "-- probe: TIMEOUT after 1s holding the lock (exit 124) --" in out
    assert flags["STEP_TIMED_OUT"] == "1"
    assert flags["STEP_LOCK_WAIT_EXPIRED"] == "0" and flags["STEP_CHECK_FAILED"] == "0"
    assert elapsed < 10, elapsed
    child = int(pidfile.read_text())
    try:
        alive = Path(f"/proc/{child}").exists()
        assert not alive, "TERM did not reach the build's child process"
    finally:
        if Path(f"/proc/{child}").exists():
            subprocess.run(["kill", "-9", str(child)])
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
