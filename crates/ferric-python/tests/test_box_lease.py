"""The box lease: a timing job owns the machine and cargo waits for it.

These run the real `scripts/box-lease` and `scripts/cargo-shim`, copied beside a
stub `cargo`, with every lock in a private directory (FERRIC_LOCK_DIR), so the
locks live agents are using are never touched. The stub and the jobs log to one
events file and hold until a release file appears: ordering is read from the
log, not from timing. The only timed part is "has NOT started after HOLD_OFF",
which is many shim polls long.

`test_mutant_*` break the scripts on purpose and check the scenario sees it.
"""

import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

from test_ci_gate_timeout_starts_after_the_lock import LIB, _alive, _gate_functions

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
POLL = "0.05"
HOLD_OFF = 0.6  # 12 polls
LIMIT = 20

pytestmark = pytest.mark.skipif(
    not (shutil.which("flock") and shutil.which("timeout")),
    reason="needs util-linux flock and coreutils timeout",
)

STUB_CARGO = """#!/usr/bin/env bash
echo "start ${STUB_NAME:-cargo} $*" >> "$EVENTS"
eval "${STUB_BODY:-}"
if [ -n "${STUB_UNTIL:-}" ]; then
  SECONDS=0  # bounded, so a killed test run leaves nothing behind for long
  until [ -e "$STUB_UNTIL" ] || [ $SECONDS -gt 120 ]; do sleep 0.02; done
fi
echo "end ${STUB_NAME:-cargo}" >> "$EVENTS"
exit "${STUB_RC:-0}"
"""


class Box:
    """A private shim directory, stub cargo and lock directory."""

    def __init__(self, tmp_path, lease_edit=None, shim_edit=None, flock_edit=None):
        self.tmp = tmp_path
        self.locks = tmp_path / "locks"
        self.events = tmp_path / "events"
        self.shim_dir = tmp_path / "ferric-cargo-shim"
        real = self.real = tmp_path / "real"
        for d in (self.locks, self.shim_dir, real):
            d.mkdir()
        self.events.touch()
        self._copy("box-lease", "box-lease", lease_edit)
        self._copy("cargo-shim", "cargo", shim_edit)
        self._copy("cargo-lock-lib.sh", "cargo-lock-lib.sh", None)
        (self.shim_dir / "under-lease").mkdir()
        self._copy("under-lease/flock", "under-lease/flock", flock_edit)
        self._executable(real / "cargo", STUB_CARGO)
        self._executable(real / "sccache", "#!/bin/sh\n")
        self.lease = self.locks / "ferric-box.lease"
        self.info = self.locks / "ferric-box.lease.info"
        self.cargo_lock = self.locks / "ferric-cargo.lock"
        self.slot0 = self.locks / "ferric-compile.lock"
        self.slot1 = self.locks / "ferric-compile.2.lock"
        self.direct = self.locks / "ferric-box.direct"
        dropped = ("FERRIC_HEAVY_LOCK_HELD", "FERRIC_BOX_LEASE_PID", "FERRIC_LOCK_DIR")
        self.env = {k: v for k, v in os.environ.items() if k not in dropped}
        self.env.update(
            PATH=f"{self.shim_dir}:{real}:{os.environ['PATH']}",
            EVENTS=str(self.events),
            FERRIC_LOCK_DIR=str(self.locks),
            FERRIC_SCCACHE_BASEDIRS_SET="1",
            FERRIC_SHIM_POLL_SECS=POLL,
            FERRIC_BOX_LEASE_POLL_SECS=POLL,
        )
        self.procs = []

    def _copy(self, source, name, edit):
        text = (SCRIPTS / source).read_text()
        if edit:
            old, new = edit
            assert old in text, f"{source} changed; update the mutant: {old!r}"
            text = text.replace(old, new)
        self._executable(self.shim_dir / name, text)

    @staticmethod
    def _executable(path, text):
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def release(self, name):
        return self.tmp / f"release-{name}"

    def open(self, name):
        self.release(name).touch()

    def popen(self, argv, **env):
        p = subprocess.Popen(
            [str(a) for a in argv],
            env={**self.env, **env},
            cwd=self.tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.procs.append(p)
        return p

    def compile(self, name, sub="build", hold=True):
        env = {"STUB_NAME": name}
        if hold:
            env["STUB_UNTIL"] = str(self.release(name))
        return self.popen(["cargo", sub], **env)

    def job_argv(self, name, body=""):
        release = self.release(name)
        return [
            "bash",
            "-c",
            f'echo "start {name}" >> "$EVENTS"; {body}\n'
            f"SECONDS=0; until [ -e {release} ] || [ $SECONDS -gt 120 ]; do sleep 0.02; done\n"
            f'echo "end {name}" >> "$EVENTS"',
        ]

    def lease_job(self, name, body="", args=()):
        return self.popen(
            [
                self.shim_dir / "box-lease",
                "run",
                "--label",
                name,
                *args,
                "--",
                *self.job_argv(name, body),
            ]
        )

    def popen_lease(self, label, argv):
        return self.popen(
            [self.shim_dir / "box-lease", "run", "--label", label, "--", *argv]
        )

    def lines(self):
        return self.events.read_text().splitlines()

    def happened(self, event):
        return any(ln == event or ln.startswith(event + " ") for ln in self.lines())

    def wait_for(self, event, limit=LIMIT):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if self.happened(event):
                return
            time.sleep(0.01)
        raise AssertionError(f"never saw {event!r}; events: {self.lines()}")

    def wait_until(self, check, what, limit=LIMIT):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if check():
                return
            time.sleep(0.01)
        raise AssertionError(f"never: {what}; events: {self.lines()}")

    def busy(self, lock):
        return subprocess.run(["flock", "-s", "-n", lock, "true"]).returncode != 0

    def wait_lease_requested(self):
        self.wait_until(lambda: self.busy(self.lease), "lease requested")

    def settle_on(self, lock, limit=3):
        """Give a polling lease time to reach `lock`. Only a hold-and-wait lease
        stays on it, so this is not a requirement, just time for a mutant to show."""
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline and not self.busy(lock):
            time.sleep(0.01)

    def index(self, event):
        return next(
            i
            for i, ln in enumerate(self.lines())
            if ln == event or ln.startswith(event + " ")
        )

    def finish(self, proc, limit=LIMIT):
        out, _ = proc.communicate(timeout=limit)
        return proc.returncode, out

    def done(self, proc, limit):
        """The exit status, or None if it is still stuck after `limit` seconds."""
        try:
            return self.finish(proc, limit)[0]
        except subprocess.TimeoutExpired:
            return None

    def status(self):
        return subprocess.run(
            [self.shim_dir / "box-lease", "status"],
            env=self.env,
            capture_output=True,
            text=True,
        )

    def cleanup(self):
        """Kill whatever is left, jobs live in their own sessions: by exact pid,
        found by the private events path in its environment."""
        needle = f"EVENTS={self.events}".encode()
        for proc_dir in Path("/proc").glob("[0-9]*"):
            try:
                if needle in (proc_dir / "environ").read_bytes().split(b"\0"):
                    os.kill(int(proc_dir.name), signal.SIGKILL)
            except (OSError, ValueError):
                pass
        for p in self.procs:
            p.communicate()


@pytest.fixture
def make_box(tmp_path):
    boxes = []

    def make(**edits):
        sub = tmp_path / str(len(boxes))
        sub.mkdir()
        boxes.append(Box(sub, **edits))
        return boxes[-1]

    yield make
    for b in boxes:
        b.cleanup()


# ---- scenarios, each also run against a broken script below --------------


def compile_waits_for_held_lease(box):
    """Return (started while the lease was held, compile output)."""
    job = box.lease_job("L")
    box.wait_for("start L")
    c = box.compile("C", hold=False)
    time.sleep(HOLD_OFF)
    early = box.happened("start C")
    box.open("L")
    assert box.finish(job)[0] == 0
    rc, out = box.finish(c)
    assert rc == 0, out
    return early, out


def pending_lease_vs_later_compile(box):
    """A in slot 0, lease requested, C arrives with slot 1 free. Return events."""
    a = box.compile("A")
    box.wait_for("start A")
    job = box.lease_job("L")
    box.wait_lease_requested()
    c = box.compile("C", hold=False)
    time.sleep(HOLD_OFF)
    box.open("A")
    box.wait_for("start L")
    time.sleep(HOLD_OFF)
    box.open("L")
    for p in (a, job, c):
        assert box.finish(p)[0] == 0
    return box.lines()


def lease_vs_inflight_compiles(box):
    """Both slots busy. Return whether the job started before they drained."""
    a = box.compile("A")  # in order: A has slot 0, the first one the lease takes
    box.wait_for("start A")
    b = box.compile("B")
    box.wait_for("start B")
    job = box.lease_job("L")
    time.sleep(HOLD_OFF)
    early = box.happened("start L")
    box.open("A")
    time.sleep(HOLD_OFF)
    early_one_left = box.happened("start L")
    box.open("B")
    box.wait_for("start L")
    box.open("L")
    for p in (a, b, job):
        assert box.finish(p)[0] == 0
    return early, early_one_left


def cargo_under_own_lease(box):
    """A bench that calls cargo under its lease. Return (rc, output)."""
    job = box.lease_job("L", body="cargo build")
    box.open("L")
    return box.finish(job)


def gate_script(box, command, timeout_secs, lock_wait_secs=60):
    """Write a script running the gate's REAL run_step on its first argument."""
    harness = box.tmp / "gate.sh"
    harness.write_text(
        "set -uo pipefail\n"
        f"source {LIB}\n"
        "ferric_clear_stale_sccache_lock() { :; }\n"
        "exec {GATE_OUT_FD}>&1\n"
        f"LOCK_FILE={box.cargo_lock}\n"
        f"TIMEOUT_SECS={timeout_secs}\nLOCK_WAIT_SECS={lock_wait_secs}\n"
        "FAILED=0 STEP_TIMED_OUT=0 STEP_LOCK_WAIT_EXPIRED=0 STEP_CHECK_FAILED=0\n"
        "RUN_LOCKED_TIMED_OUT=0\n" + _gate_functions() + '\nrun_step "probe" "$1"\n'
    )
    return harness


def gate_step(box, command, timeout_secs, lock_wait_secs=60):
    harness = gate_script(box, command, timeout_secs, lock_wait_secs)
    return box.popen(["bash", harness, command], STUB_NAME="G")


def gate_step_calls_cargo_with_a_lease_pending(box):
    """The gate holds the cargo lock, a lease queues on it, THEN the step's cargo
    starts. Waiting for that lease would deadlock. Return (gate output, events)."""
    go = box.tmp / "go"
    gate = gate_step(box, f"until [ -e {go} ]; do sleep 0.02; done; cargo build", 4)
    box.wait_until(lambda: box.busy(box.cargo_lock), "gate holds the cargo lock")
    job = box.lease_job("L")
    box.wait_lease_requested()
    time.sleep(0.2)  # let the lease reach its blocking flock on the cargo lock
    go.touch()
    _, out = box.finish(gate)
    box.open("L")
    box.finish(job)
    return out, box.lines()


def gate_step_during_an_active_lease(box):
    """Lease held past the step's 3 s timeout. Return (gate output, events)."""
    job = box.lease_job("L")
    box.wait_for("start L")
    gate = gate_step(box, "cargo build", 3)
    time.sleep(4.5)
    box.open("L")
    box.finish(job)
    _, out = box.finish(gate)
    return out, box.lines()


# ---- the required behaviours ------------------------------------------------


def test_a_compile_waits_for_a_held_lease_and_proceeds_on_release(make_box):
    early, out = compile_waits_for_held_lease(make_box())
    assert not early, "the compile ran under someone else's lease"
    lines = [ln for ln in out.splitlines() if "waiting for box lease" in ln]
    assert len(lines) == 1, out  # says so once, not once per poll
    assert "held by L since 20" in lines[0], out


def test_a_killed_holder_releases(make_box):
    box = make_box()
    job = box.lease_job("L", body=f"echo $$ > {box.tmp / 'job.pid'}")
    box.wait_for("start L")
    os.kill(job.pid, signal.SIGKILL)
    job.wait()
    # The orphaned job runs on (documented) but the lease is gone, and the info
    # file it could not clean up is not believed.
    assert box.info.exists()
    assert not box.busy(box.lease)
    c = box.compile("C", hold=False)
    assert box.finish(c)[0] == 0
    status = box.status()
    assert "lease: free" in status.stdout, status.stdout
    box.open("L")


def test_a_holder_killed_while_pending_does_not_keep_the_lease(make_box):
    box = make_box()
    a = box.compile("A")
    box.wait_for("start A")
    job = box.lease_job("L")
    box.wait_until(lambda: box.info.exists(), "lease pending")
    assert box.busy(box.lease)
    os.kill(job.pid, signal.SIGKILL)
    job.wait()
    # Its flock is still queued on slot 0; that must not hold the lease.
    box.wait_until(lambda: not box.busy(box.lease), "lease freed", limit=5)
    box.open("A")
    assert box.finish(a)[0] == 0


def test_a_pending_lease_goes_ahead_of_later_compiles(make_box):
    box = make_box()
    events = pending_lease_vs_later_compile(box)
    assert events.index("end L") < next(
        i for i, ln in enumerate(events) if ln.startswith("start C")
    ), events


def test_a_lease_waits_for_every_inflight_compile(make_box):
    assert lease_vs_inflight_compiles(make_box()) == (False, False)


def test_cargo_under_the_lease_does_not_deadlock(make_box):
    box = make_box()
    rc, out = cargo_under_own_lease(box)
    assert rc == 0, out
    assert "waiting for" not in out, out
    assert box.index("start cargo") < box.index("end L")


def test_nested_lease_commands_do_not_deadlock(make_box):
    box = make_box()
    lease = box.shim_dir / "box-lease"
    body = (
        f"{lease} wait && {lease} run --label inner -- bash -c "
        '\'echo "start inner" >> "$EVENTS"\' && echo "after inner" >> "$EVENTS"'
    )
    job = box.lease_job("L", body=body)
    box.wait_for("after inner")
    box.open("L")
    assert box.finish(job)[0] == 0


def test_the_nested_check_is_by_lease_not_by_a_stale_env_pid(make_box):
    """A leftover FERRIC_BOX_LEASE_PID must not let `run` skip the lease."""
    box = make_box()
    p = box.popen(
        [box.shim_dir / "box-lease", "run", "--label", "M", "--", "bash", "-c", "true"],
        FERRIC_BOX_LEASE_PID=str(os.getpid()),
    )
    assert box.finish(p)[0] == 0
    a = box.lease_job("A")
    box.wait_for("start A")
    b = box.popen(
        [box.shim_dir / "box-lease", "run", "--label", "B", "--wait-secs", "0"]
        + ["--", "true"],
        FERRIC_BOX_LEASE_PID=str(os.getpid()),
    )
    assert box.finish(b)[0] == 75
    box.open("A")
    box.finish(a)


def test_the_gate_step_still_runs_with_a_lease_pending(make_box):
    box = make_box()
    out, events = gate_step_calls_cargo_with_a_lease_pending(box)
    assert "-- probe: PASS --" in out, out
    assert events.index("end G") < events.index("start L"), events


def test_a_gate_step_clock_does_not_burn_during_a_lease(make_box):
    """The 3 s step timeout starts only after the lease freed the cargo lock."""
    out, events = gate_step_during_an_active_lease(make_box())
    assert "-- probe: PASS --" in out, out
    assert events.index("end L") < events.index("start G build"), events


def test_a_gate_queued_past_its_lock_wait_reports_not_run(make_box):
    box = make_box()
    job = box.lease_job("L")
    box.wait_for("start L")
    out = box.finish(gate_step(box, "cargo build", 30, lock_wait_secs=1))[1]
    assert "NOT RUN -- never got the cargo lock in 1s" in out, out
    box.open("L")
    box.finish(job)


# ---- the rest of the contract -----------------------------------------------


def test_the_shim_lock_spans_the_test_run_and_the_exit_status_passes_through(make_box):
    box = make_box()
    c = box.compile("T", sub="test")
    box.wait_for("start T test")
    assert box.busy(box.slot0), "the test run is outside the compile lock"
    box.open("T")
    assert box.finish(c)[0] == 0
    assert not box.busy(box.slot0)
    p = box.popen(["cargo", "build"], STUB_RC="7")
    assert box.finish(p)[0] == 7


def test_cargo_commands_that_do_not_compile_ignore_the_lease(make_box):
    box = make_box()
    job = box.lease_job("L")
    box.wait_for("start L")
    p = box.popen(["cargo", "metadata"], STUB_NAME="M")
    assert box.finish(p)[0] == 0
    box.open("L")
    box.finish(job)


def test_a_flock_wrapped_compile_is_held_out_and_does_not_retake_its_slot(make_box):
    box = make_box()
    job = box.lease_job("L")
    box.wait_for("start L")
    p = box.popen(["flock", "-o", box.slot0, "cargo", "build"], STUB_NAME="W")
    time.sleep(HOLD_OFF)
    assert not box.happened("start W")
    box.open("L")
    box.finish(job)
    assert box.finish(p)[0] == 0
    assert box.index("end L") < box.index("start W")


def test_the_job_inherits_no_lock_and_a_daemon_it_starts_cannot_squat(make_box):
    box = make_box()
    fds, pidfile = box.tmp / "fds", box.tmp / "daemon.pid"
    body = (
        f"ls -l /proc/$$/fd > {fds}\n"
        f"setsid sleep 30 </dev/null >/dev/null 2>&1 & echo $! > {pidfile}"
    )
    job = box.lease_job("L", body=body)
    box.open("L")
    assert box.finish(job)[0] == 0
    daemon = int(pidfile.read_text())
    try:
        assert "ferric-" not in fds.read_text(), fds.read_text()
        assert not box.busy(box.lease)
        assert not box.busy(box.cargo_lock)
        assert not box.busy(box.slot0)
        assert not box.busy(box.slot1)
    finally:
        os.kill(daemon, signal.SIGKILL)


def test_max_hold_warns_and_never_kills(make_box):
    box = make_box()
    job = box.lease_job("L", args=("--max-hold", "1"))
    box.wait_for("start L")
    time.sleep(1.5)
    assert job.poll() is None, "max-hold killed the job"
    status = box.status()
    assert "HELD by L" in status.stdout and "max-hold" in status.stdout, status.stdout
    box.open("L")
    rc, out = box.finish(job)
    assert rc == 0
    assert out.count("WARNING: L has held the box") == 1, out


def test_status_names_the_holder_and_the_locks(make_box):
    box = make_box()

    def status():
        return box.status().stdout

    assert status().count(": free") == 5
    job = box.lease_job("L")
    box.wait_for("start L")
    out = status()
    assert "lease: HELD by L (pid " in out and "active since" in out, out
    assert out.count(": held") == 4, out
    box.open("L")
    box.finish(job)
    assert status().count(": free") == 5
    assert not box.info.exists()


def test_wait_blocks_for_the_lease_then_returns(make_box):
    box = make_box()
    lease = box.shim_dir / "box-lease"
    assert box.finish(box.popen([lease, "wait"]))[0] == 0
    job = box.lease_job("L")
    box.wait_for("start L")
    assert box.finish(box.popen([lease, "wait", "--wait-secs", "0"]))[0] == 75
    w = box.popen([lease, "wait"])
    time.sleep(HOLD_OFF)
    assert w.poll() is None
    box.open("L")
    box.finish(job)
    rc, out = box.finish(w)
    assert rc == 0 and "held by L (pid" in out, out


def test_a_lease_under_a_held_compile_lock_refuses_rather_than_deadlocks(make_box):
    box = make_box()
    p = box.popen(
        ["flock", "-o", box.slot0, box.shim_dir / "box-lease", "run"]
        + ["--label", "L", "--", "true"]
    )
    rc, out = box.finish(p)
    assert rc == 2 and "ancestor holds" in out, out


def test_run_returns_the_command_status_and_runs_nothing_on_expiry(make_box):
    box = make_box()
    lease = box.shim_dir / "box-lease"
    run = [lease, "run", "--label", "X"]
    assert box.finish(box.popen([*run, "--", "bash", "-c", "exit 9"]))[0] == 9
    a = box.compile("A")
    box.wait_for("start A")
    marker = box.tmp / "ran"
    p = box.popen([*run, "--wait-secs", "1", "--", "touch", marker])
    rc, out = box.finish(p)
    assert rc == 75 and "NOT run" in out, out
    assert not marker.exists()
    assert not box.busy(box.lease)
    box.open("A")
    box.finish(a)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_a_signal_stops_the_job_and_only_then_frees_the_lease(make_box, sig):
    box = make_box()
    job = box.lease_job("L")
    box.wait_for("start L")
    # Only the wrapper. The job ignores nothing, so TERM must reach it from there.
    os.kill(job.pid, sig)
    rc, _ = box.finish(job)
    assert rc != 0
    assert not box.happened("end L")
    assert not box.busy(box.lease)


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--", "true"],
        ["run", "--label", "L"],
        ["run", "--label", "L", "--bogus", "--", "true"],
        ["run", "--label"],
        ["nonsense"],
    ],
)
def test_bad_usage_is_an_error(make_box, argv):
    box = make_box()
    p = box.popen([box.shim_dir / "box-lease", *argv])
    assert box.finish(p)[0] == 2


def test_the_installer_replaces_by_rename_and_keeps_the_previous(tmp_path):
    dest = tmp_path / "ferric-cargo-shim"
    dest.mkdir()
    (dest / "cargo").write_text("old\n")
    old_inode = (dest / "cargo").stat().st_ino
    env = {**os.environ, "FERRIC_SHIM_DIR": str(dest)}
    subprocess.run([SCRIPTS / "install-cargo-shim.sh"], env=env, check=True)
    assert (dest / "cargo").read_text() == (SCRIPTS / "cargo-shim").read_text()
    assert (dest / "cargo").stat().st_ino != old_inode
    assert (dest / "cargo.prev").read_text() == "old\n"
    assert os.access(dest / "box-lease", os.X_OK)
    assert (dest / "under-lease" / "flock").exists()
    # Same content again must not overwrite the rollback with itself.
    subprocess.run([SCRIPTS / "install-cargo-shim.sh"], env=env, check=True)
    assert (dest / "cargo.prev").read_text() == "old\n"
    leftovers = [p.name for p in dest.rglob("*.new.*")]
    assert not leftovers, leftovers


# ---- mutants: each scenario must SEE the defect it exists for ---------------

ALL_LOCKS = "if flock -n 201 && flock -n 202 && flock -n 203 && flock -n 204; then"
NO_WAIT = (
    'if lease_busy && ! ancestor_holds "$CARGO_LOCK" "$DIRECT"; then',
    "if false; then",
)
NO_ANCESTOR_CHECK = (
    'if lease_busy && ! ancestor_holds "$CARGO_LOCK" "$DIRECT"; then',
    "if lease_busy; then",
)
NO_DRAIN = (ALL_LOCKS, "if flock -n 201 && flock -n 204; then")
ONE_SLOT = (ALL_LOCKS, "if flock -n 201 && flock -n 202 && flock -n 204; then")
NO_CARGO_LOCK = (ALL_LOCKS, "if flock -n 202 && flock -n 203 && flock -n 204; then")
NO_EXEC_LOCK = (ALL_LOCKS, "if flock -n 201 && flock -n 202 && flock -n 203; then")
# Keep what it got while waiting for the rest: the lock cycles come back.
HOLD_AND_WAIT = ("        flock -u 201; flock -u 202; flock -u 203; flock -u 204\n", "")
# Nothing marks the lease as pending, so it cannot hold compiles back.
LEASE_NOT_HELD = ("    take_lease\n", "")
NO_REENTRY = (
    "export FERRIC_BOX_LEASE_PID=$$ FERRIC_HEAVY_LOCK_HELD=$$ PATH=$UNDER_LEASE:$PATH",
    "export PATH=$UNDER_LEASE:$PATH",
)
NO_FLOCK_WRAPPER = (
    "export FERRIC_BOX_LEASE_PID=$$ FERRIC_HEAVY_LOCK_HELD=$$ PATH=$UNDER_LEASE:$PATH",
    "export FERRIC_BOX_LEASE_PID=$$ FERRIC_HEAVY_LOCK_HELD=$$",
)
LEAKS_FDS = (
    '"$@" <&0 200>&- 201>&- 202>&- 203>&- 204>&- &',
    '"$@" <&0 &',
)
OLD_SIGNAL = (
    """trap 'kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null' INT""",
    """trap 'kill -TERM "$child" 2>/dev/null' INT""",
)
NO_GROUP_WAIT = (
    'while kill -0 "$child" 2>/dev/null || kill -0 -- "-$child" 2>/dev/null; do',
    'while kill -0 "$child" 2>/dev/null; do',
)
KEEPS_IGNORED_SIGNALS = ("env --default-signal=INT,QUIT setsid", "setsid")
NO_STALE_CLEAR = (
    'ferric_clear_stale_sccache_lock "$f" >&2',
    ":",
)
NO_NOTICE = ("box-lease: NOT drained or held out", "box-lease: note")
OLD_SUBCOMMAND_PARSE = (
    """    --manifest-path|--config|--color|--target-dir|--lockfile-path|--explain|-C|-Z) skip=1;;
""",
    "",
)
HELD_ALWAYS = (
    """case "${FERRIC_HEAVY_LOCK_HELD:-}" in ""|*[!0-9]*|1) return 1;; esac
  [ -d "/proc/$FERRIC_HEAVY_LOCK_HELD" ]""",
    """[ -n "${FERRIC_HEAVY_LOCK_HELD:-}" ]""",
)
NO_ANCESTOR_SLOT_CHECK = (
    'elif ancestor_holds "$SLOT0" "$SLOT1"; then',
    "elif false; then",
)
CMDLINE_TEXT_CHECK = (
    'elif ancestor_holds "$SLOT0" "$SLOT1"; then',
    "elif tr '\\0' ' ' < \"/proc/$PPID/cmdline\" | grep -q ferric-compile; then",
)
READS_75_AS_BUSY = (
    'if [ $rc -ne 75 ] || [ -e "$ran" ]; then',
    "if [ $rc -ne 75 ]; then",
)
SHIM_FINDS_ITSELF = ('  [ "$d" -ef "$SELF_DIR" ] && continue\n', "")


def test_mutant_no_lease_wait_does_not_say_why_it_waits(make_box):
    """An ACTIVE lease holds compiles out through the slots regardless, so the
    check's job there is the message; ordering (next test) is its real job."""
    early, out = compile_waits_for_held_lease(make_box(shim_edit=NO_WAIT))
    assert not early
    assert "waiting for box lease" not in out, out


def test_mutant_no_lease_wait_starves_the_lease(make_box):
    events = pending_lease_vs_later_compile(make_box(shim_edit=NO_WAIT))
    start_c = next(i for i, ln in enumerate(events) if ln.startswith("start C"))
    assert start_c < events.index("start L"), events


def test_mutant_lease_not_held_while_pending_starves_too(make_box):
    box = make_box(lease_edit=LEASE_NOT_HELD)
    events = _ordering_without_the_lease_file(box)
    start_c = next(i for i, ln in enumerate(events) if ln.startswith("start C"))
    assert start_c < events.index("start L"), events


def _ordering_without_the_lease_file(box):
    a = box.compile("A")
    box.wait_for("start A")
    job = box.lease_job("L")
    time.sleep(0.2)
    c = box.compile("C", hold=False)
    box.wait_for("start C")
    box.open("A")
    box.wait_for("start L")
    box.open("L")
    for p in (a, job, c):
        assert box.finish(p)[0] == 0
    return box.lines()


@pytest.mark.parametrize(
    ("edit", "expected"), [(NO_DRAIN, (True, True)), (ONE_SLOT, (False, True))]
)
def test_mutant_without_the_drain_runs_over_compiles(make_box, edit, expected):
    assert lease_vs_inflight_compiles(make_box(lease_edit=edit)) == expected


def test_mutant_no_reentry_deadlocks_under_its_own_lease(make_box):
    # Two routes pass cargo under a lease: the env and the ancestor check.
    box = make_box(lease_edit=NO_REENTRY, shim_edit=NO_ANCESTOR_SLOT_CHECK)
    job = box.lease_job("L", body="cargo build")
    time.sleep(HOLD_OFF)
    assert box.happened("start L") and not box.happened("start cargo")
    assert job.poll() is None


def test_mutant_no_ancestor_check_deadlocks_the_gate(make_box):
    box = make_box(shim_edit=NO_ANCESTOR_CHECK)
    out, _ = gate_step_calls_cargo_with_a_lease_pending(box)
    assert "-- probe: TIMEOUT" in out, out


def test_mutant_lease_without_the_cargo_lock_burns_the_gate_clock(make_box):
    out, _ = gate_step_during_an_active_lease(make_box(lease_edit=NO_CARGO_LOCK))
    assert "-- probe: TIMEOUT" in out, out


def test_mutant_leaked_fds_let_a_daemon_squat(make_box):
    box = make_box(lease_edit=LEAKS_FDS)
    pidfile = box.tmp / "daemon.pid"
    body = f"setsid sleep 30 </dev/null >/dev/null 2>&1 & echo $! > {pidfile}"
    job = box.lease_job("L", body=body)
    box.open("L")
    box.finish(job)
    daemon = int(pidfile.read_text())
    try:
        assert box.busy(box.lease), "the daemon should have kept the lease"
    finally:
        os.kill(daemon, signal.SIGKILL)


# ---- review of 98acff2d9: each scenario returns True when the behaviour holds,
# and is run against the real scripts and against the mutant that removes it. --


def flock_on_a_slot_under_a_lease(box):
    """Item 1: the standing wrapper must run, not wait on its own lease."""
    marker = box.tmp / "ran"
    wrapper = ["flock", "-o", box.slot0, "touch", marker]
    return box.done(box.popen_lease("L", wrapper), 5) == 0 and marker.exists()


def flock_c_on_the_cargo_lock_under_a_lease(box):
    marker = box.tmp / "ran"
    wrapper = ["flock", "-o", "-w", "1", box.cargo_lock, "-c", f"touch {marker}"]
    return box.done(box.popen_lease("L", wrapper), 5) == 0 and marker.exists()


def the_gate_under_a_lease(box):
    """Item 1: `box-lease run -- git push`: fd-form flock on its own lease's lock."""
    gate_script(box, "cargo build", timeout_secs=5, lock_wait_secs=1)
    p = box.popen_lease("L", ["bash", box.tmp / "gate.sh", "cargo build"])
    rc, out = box.finish(p, LIMIT)
    return rc == 0 and "-- probe: PASS --" in out


def term_with_a_grandchild(box):
    """Item 4: wrapper TERMed while CMD (a script) runs a slow-dying process.
    Fail if that process is ever alive with the lease free."""
    pidfile = box.tmp / "worker.pid"
    worker = box.tmp / "worker.sh"
    worker.write_text(
        f"#!/bin/bash\ntrap 'sleep 1; exit' TERM; echo $$ > {pidfile}; sleep 30 & wait\n"
    )
    worker.chmod(0o755)
    script = box.tmp / "script.sh"
    script.write_text(f"#!/bin/bash\n{worker}\n")
    script.chmod(0o755)
    job = box.popen_lease("L", [script])
    box.wait_until(lambda: pidfile.exists() and pidfile.read_text(), "worker up")
    worker_pid = int(pidfile.read_text())
    os.kill(job.pid, signal.SIGTERM)
    # It must be STOPPED (the worker takes 1 s, it would otherwise live 30), and
    # never run on with the lease free.
    deadline = time.monotonic() + 20
    while _alive(worker_pid):
        assert box.busy(box.lease), "worker alive with the lease free"
        assert time.monotonic() < deadline, "worker survived the signal"
        time.sleep(0.02)
    assert box.done(job, 20) is not None, "wrapper did not exit"
    assert not box.busy(box.lease), "lease still held after the wrapper exited"
    return True


def straggler_holds_the_lease(box):
    """Item 4: CMD exits leaving a background process in its group."""
    go = box.tmp / "go"
    body = f"until [ -e {go} ]; do sleep 0.02; done"
    cmd = f"""bash -c '{body}' & echo "start L" >> "$EVENTS"; exit 0"""
    job = box.popen_lease("L", ["bash", "-c", cmd])
    box.wait_for("start L")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if job.poll() is not None or not box.busy(box.lease):
            go.touch()
            return False
        time.sleep(0.02)
    go.touch()
    return box.done(job, 10) == 0


def wrapper_slot_then_cargo_lock(box):
    """Item 5: slot 0 then the cargo lock, the opposite of the gate's order."""
    go, step = box.tmp / "go", box.tmp / "step"
    inner = (
        f"until [ -e {go} ]; do sleep 0.02; done; "
        f"flock -o -w 3 {box.cargo_lock} touch {step}"
    )
    w = box.popen(["flock", "-o", box.slot0, "bash", "-c", inner])
    box.wait_until(lambda: box.busy(box.slot0), "wrapper holds slot 0")
    job = box.popen_lease("L", ["true"])
    box.wait_lease_requested()
    box.settle_on(box.cargo_lock)
    go.touch()
    box.done(w, 10)
    return step.exists() and box.done(job, 10) == 0


def cargo_in_slot1_whose_child_takes_slot0(box):
    """Item 5: slot 1 then slot 0, the opposite of the lease's order."""
    go, step = box.tmp / "go", box.tmp / "step"
    a = box.compile("A")
    box.wait_for("start A")
    body = (
        f"until [ -e {go} ]; do sleep 0.02; done; flock -w 3 {box.slot0} touch {step}"
    )
    b = box.popen(["cargo", "build"], STUB_NAME="B", STUB_BODY=body)
    box.wait_for("start B")
    job = box.popen_lease("L", ["true"])
    box.wait_lease_requested()
    box.open("A")
    box.done(a, 10)
    box.settle_on(box.slot0)
    go.touch()
    box.done(b, 10)
    return step.exists() and box.done(job, 10) == 0


def a_squatter_is_cleared(box):
    """Item 6: an sccache alone on a slot must not stall the lease."""
    squatter = box.tmp / "sccache"
    shutil.copy(shutil.which("sleep"), squatter)
    d = box.popen(
        ["bash", "-c", f'exec 9>>"{box.slot0}"; flock 9; exec {squatter} 60'],
    )
    box.wait_until(lambda: box.busy(box.slot0), "squatter up")
    time.sleep(0.2)
    job = box.popen(
        [box.shim_dir / "box-lease", "run", "--label", "L", "--wait-secs", "10"]
        + ["--", "true"]
    )
    ok = box.done(job, 30) == 0
    if d.poll() is None:
        d.kill()
    return ok


def the_notice_lists_what_is_not_drained(box):
    """Item 3."""
    _, out = box.finish(box.popen_lease("L", ["true"]))
    return "NOT drained or held out: direct binaries" in out


def a_lease_waits_for_exec(box):
    """Item 3: a direct binary under `box-lease exec` is drained."""
    d = box.popen([box.shim_dir / "box-lease", "exec", "--", *box.job_argv("D")])
    box.wait_for("start D")
    job = box.lease_job("L")
    time.sleep(HOLD_OFF)
    early = box.happened("start L")
    box.open("D")
    box.wait_for("start L")
    box.open("L")
    box.finish(d)
    box.finish(job)
    return not early


CHECKS = {
    "flock_slot": (flock_on_a_slot_under_a_lease, {"lease_edit": NO_FLOCK_WRAPPER}),
    "flock_c": (
        flock_c_on_the_cargo_lock_under_a_lease,
        {"lease_edit": NO_FLOCK_WRAPPER},
    ),
    "gate_under": (the_gate_under_a_lease, {"lease_edit": NO_FLOCK_WRAPPER}),
    "term": (term_with_a_grandchild, {"lease_edit": OLD_SIGNAL}),
    "term_hold": (term_with_a_grandchild, {"lease_edit": NO_GROUP_WAIT}),
    "straggler": (straggler_holds_the_lease, {"lease_edit": NO_GROUP_WAIT}),
    "inv_cargo": (wrapper_slot_then_cargo_lock, {"lease_edit": HOLD_AND_WAIT}),
    "inv_slot": (cargo_in_slot1_whose_child_takes_slot0, {"lease_edit": HOLD_AND_WAIT}),
    "squatter": (a_squatter_is_cleared, {"lease_edit": NO_STALE_CLEAR}),
    "notice": (the_notice_lists_what_is_not_drained, {"lease_edit": NO_NOTICE}),
    "exec": (a_lease_waits_for_exec, {"lease_edit": NO_EXEC_LOCK}),
}


@pytest.mark.parametrize("name", CHECKS)
def test_holds(make_box, name):
    assert CHECKS[name][0](make_box())


@pytest.mark.parametrize("name", CHECKS)
def test_mutant_is_seen(make_box, name):
    check, edit = CHECKS[name]
    try:
        assert not check(make_box(**edit))
    except (subprocess.TimeoutExpired, AssertionError) as e:
        # A mutant that wedges or misses a step has been seen too, but not an
        # assertion from the line above.
        if "assert not" in str(e):
            raise


def test_job_gets_int_and_quit_back(make_box):
    """Item 13: `&` makes a job ignore them; it must die on ^C like any other."""

    def ignored(box):
        # grep itself: a bash here would report the QUIT it ignores on its own.
        rc, out = box.finish(
            box.popen_lease("L", ["grep", "SigIgn", "/proc/self/status"])
        )
        assert rc == 0, out
        line = next(ln for ln in out.splitlines() if ln.startswith("SigIgn"))
        return int(line.split()[1], 16) & 0b110

    assert ignored(make_box()) == 0
    assert ignored(make_box(lease_edit=KEEPS_IGNORED_SIGNALS)) == 0b110


HEAVY_FORMS = [
    ["b"],
    ["t"],
    ["--manifest-path", "x", "build"],
    ["--manifest-path", "metadata", "build"],
    ["-C", "d", "test"],
    ["--config", "a=b", "build"],
    ["llvm-cov"],
    ["install", "x"],
]


@pytest.mark.parametrize("args", HEAVY_FORMS)
def test_every_spelling_of_a_heavy_command_waits(make_box, args):
    box = make_box()
    job = box.lease_job("L")
    box.wait_for("start L")
    c = box.popen(["cargo", *args], STUB_NAME="C")
    time.sleep(0.3)
    early = box.happened("start C")
    box.open("L")
    box.finish(job)
    box.finish(c)
    assert not early, args


def test_mutant_the_old_parse_lets_a_value_spell_a_trivial_command(make_box):
    box = make_box(shim_edit=OLD_SUBCOMMAND_PARSE)
    job = box.lease_job("L")
    box.wait_for("start L")
    c = box.popen(["cargo", "--manifest-path", "metadata", "build"], STUB_NAME="C")
    box.wait_for("start C")
    box.open("L")
    box.finish(job)
    box.finish(c)


@pytest.mark.parametrize("args", [["--version"], ["-V"], ["--help"], ["metadata"]])
def test_trivial_commands_pass_a_held_lease(make_box, args):
    box = make_box()
    job = box.lease_job("L")
    box.wait_for("start L")
    assert box.done(box.popen(["cargo", *args]), 5) == 0
    box.open("L")
    box.finish(job)


def _dead_pid():
    p = subprocess.Popen(["true"])
    p.wait()
    return str(p.pid)


def _held_env_is_held_out(box, value):
    job = box.lease_job("L")
    box.wait_for("start L")
    c = box.popen(["cargo", "build"], STUB_NAME="C", FERRIC_HEAVY_LOCK_HELD=value)
    time.sleep(HOLD_OFF)
    early = box.happened("start C")
    box.open("L")
    box.finish(job)
    box.finish(c)
    return not early


@pytest.mark.parametrize("value", ["dead", "1", "yes"])
def test_a_held_env_outliving_its_holder_does_not_pass(make_box, value):
    """Item 8."""
    assert _held_env_is_held_out(make_box(), _dead_pid() if value == "dead" else value)


def test_mutant_an_unconditional_held_env_passes(make_box):
    assert not _held_env_is_held_out(make_box(shim_edit=HELD_ALWAYS), _dead_pid())


def _env_i_cargo_under_lease(box):
    """Item 9: FERRIC_LOCK_DIR and PATH stay only because the test needs them."""
    env_i = (
        f"env -i PATH={box.env['PATH']} FERRIC_LOCK_DIR={box.locks} "
        f"EVENTS={box.events} FERRIC_SCCACHE_BASEDIRS_SET=1 cargo build"
    )
    return box.done(box.popen_lease("L", ["bash", "-c", env_i]), 3)


def test_env_i_cargo_under_its_own_lease_runs(make_box):
    assert _env_i_cargo_under_lease(make_box()) == 0


def test_mutant_env_i_cargo_waits_on_itself(make_box):
    assert _env_i_cargo_under_lease(make_box(shim_edit=NO_ANCESTOR_SLOT_CHECK)) is None


def _flock_mentioning_the_name(box):
    """Item 10: a flock on some other file whose command mentions ferric-compile."""
    a, b = box.compile("A"), None
    box.wait_for("start A")
    b = box.compile("B")
    box.wait_for("start B")
    other = box.tmp / "other.lock"
    p = box.popen(["flock", other, "cargo", "build", "ferric-compile"], STUB_NAME="C")
    time.sleep(HOLD_OFF)
    early = box.happened("start C")
    box.open("A")
    box.open("B")
    for x in (a, b, p):
        box.finish(x)
    return not early


def test_only_the_lock_itself_counts_not_text_in_a_command_line(make_box):
    assert _flock_mentioning_the_name(make_box())


def test_mutant_matching_the_command_line_skips_the_slot(make_box):
    assert not _flock_mentioning_the_name(make_box(shim_edit=CMDLINE_TEXT_CHECK))


def test_a_flock_wrapper_found_through_fdinfo_is_not_queued(make_box):
    """Item 10's other half: fd-held, no flock in the chain's text at all."""
    box = make_box()
    a = box.compile("A")
    box.wait_for("start A")
    box.open("A")
    box.finish(a)
    script = f'exec 9>>"{box.slot0}"; flock 9; cargo build'
    p = box.popen(["bash", "-c", script], STUB_NAME="F")
    assert box.done(p, 5) == 0


def _runs_of_a_cargo_exiting_75(box):
    box.done(box.popen(["cargo", "build"], STUB_RC="75", STUB_NAME="R"), 1)
    return sum(ln.startswith("start R") for ln in box.lines())


def test_a_real_cargo_exit_75_is_final(make_box):
    """Item 11."""
    box = make_box()
    p = box.popen(["cargo", "build"], STUB_RC="75", STUB_NAME="R")
    assert box.done(p, 5) == 75
    assert sum(ln.startswith("start R") for ln in box.lines()) == 1


def test_mutant_reading_75_as_busy_reruns_it(make_box):
    assert _runs_of_a_cargo_exiting_75(make_box(shim_edit=READS_75_AS_BUSY)) > 1


def test_a_pending_lease_gives_up_naming_the_holder(make_box):
    """Item 6: the exit is 75, nothing ran, and it says who it waited for."""
    box = make_box()
    a = box.compile("A")
    box.wait_for("start A")
    marker = box.tmp / "ran"
    p = box.popen(
        [box.shim_dir / "box-lease", "run", "--label", "L", "--wait-secs", "1"]
        + ["--", "touch", marker]
    )
    box.wait_until(lambda: box.info.exists(), "pending")
    out = box.status().stdout
    assert "draining since" in out and "waiting on whatever" in out, out
    assert f"{box.slot0}: held\n  pid " in out, out
    rc, out = box.finish(p)
    assert rc == 75 and "NOT run" in out and f"{box.slot0}\n  pid " in out, out
    assert not marker.exists()
    box.open("A")
    box.finish(a)


def test_the_default_wait_is_bounded(make_box):
    assert "DEFAULT_RUN_WAIT=3600\n" in (SCRIPTS / "box-lease").read_text()
    box = make_box(lease_edit=("DEFAULT_RUN_WAIT=3600", "DEFAULT_RUN_WAIT=1"))
    a = box.compile("A")
    box.wait_for("start A")
    rc, _ = box.finish(box.popen_lease("L", ["true"]))
    assert rc == 75
    box.open("A")
    box.finish(a)


def test_status_says_unknown_when_a_lock_cannot_open(tmp_path, make_box):
    """Item 15: never "free" when it could not tell."""
    box = make_box()
    box.env["FERRIC_LOCK_DIR"] = str(tmp_path / "no" / "such")
    s = box.status()
    assert s.returncode == 1 and "UNKNOWN" in s.stdout, s.stdout
    assert "free" not in s.stdout, s.stdout


def test_an_uncontended_lease_prints_no_error(make_box):
    """Item 15: the false "No such file" from reading an info file not yet there."""
    box = make_box()
    p = box.popen([box.shim_dir / "box-lease", "run", "--label", "L", "--", "true"])
    rc, out = box.finish(p)
    assert rc == 0 and "No such file" not in out, out


def _shim_under_another_name(box):
    other = box.tmp / "elsewhere"
    other.mkdir()
    shutil.copy(box.shim_dir / "cargo", other / "cargo")
    env = {"PATH": f"{other}:{box.real}:{os.environ['PATH']}"}
    # Under `timeout`: the mutant exec-loops forever, and must not outlive a killed run.
    return box.done(box.popen(["timeout", "3", "cargo", "--version"], **env), 10)


def test_a_misnamed_shim_dir_finds_the_real_cargo(make_box):
    """Item 15."""
    box = make_box()
    assert _shim_under_another_name(box) == 0
    assert box.happened("start cargo")


def test_mutant_a_misnamed_shim_dir_execs_itself(make_box):
    assert _shim_under_another_name(make_box(shim_edit=SHIM_FINDS_ITSELF)) == 124


def _ensure(tmp_path, repo_scripts_ok=True, on_path=True, mutate=False):
    """Item 2: run the REAL lib's check with a private shim directory."""
    dest = tmp_path / "shim-dir"
    lib = LIB.read_text()
    if mutate:
        old = 'if [[ ! -x "$dir/cargo" ]]; then'
        assert old in lib
        lib = lib.replace(old, "if false; then")
    copy = tmp_path / "lib.sh"
    copy.write_text(lib)
    repo = REPO if repo_scripts_ok else tmp_path / "no-repo"
    path = f"{dest}:{os.environ['PATH']}" if on_path else os.environ["PATH"]
    p = subprocess.run(
        ["bash", "-c", f'source {copy}; ferric_ensure_cargo_shim "{repo}"; echo rc=$?'],
        env={**os.environ, "FERRIC_SHIM_DIR": str(dest), "PATH": path},
        capture_output=True,
        text=True,
    )
    assert "rc=0" in p.stdout, p.stdout + p.stderr  # never fails the gate
    return dest, p.stderr


def test_a_missing_shim_is_reinstalled_and_said(tmp_path):
    dest, err = _ensure(tmp_path)
    assert (dest / "cargo").exists() and "missing" in err, err


def test_mutant_a_missing_shim_stays_missing(tmp_path):
    dest, _ = _ensure(tmp_path, mutate=True)
    assert not (dest / "cargo").exists()


def test_a_failed_reinstall_warns_and_does_not_fail_the_gate(tmp_path):
    _, err = _ensure(tmp_path, repo_scripts_ok=False)
    assert "could not reinstall" in err, err


def test_a_shim_not_on_path_warns(tmp_path):
    _, err = _ensure(tmp_path, on_path=False)
    assert "not first on PATH" in err, err


def test_a_shim_behind_the_real_cargo_warns(tmp_path):
    dest = tmp_path / "shim-dir"
    other = tmp_path / "real-bin"
    other.mkdir()
    (other / "cargo").write_text("#!/bin/sh\n")
    (other / "cargo").chmod(0o755)
    subprocess.run(
        [SCRIPTS / "install-cargo-shim.sh"],
        env={**os.environ, "FERRIC_SHIM_DIR": str(dest)},
        check=True,
        capture_output=True,
    )
    lib = LIB.read_text()
    env = {
        **os.environ,
        "FERRIC_SHIM_DIR": str(dest),
        "PATH": f"{other}:{dest}:{os.environ['PATH']}",
    }
    cmd = f'source {LIB}; ferric_ensure_cargo_shim "{REPO}"'
    p = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    assert "not first on PATH" in p.stderr, p.stderr
    # The old on-PATH-anywhere check stays silent here, so this test can tell them apart.
    old = 'if [[ "$(command -v cargo)" != "$dir/cargo" ]]; then'
    assert old in lib
    copy = tmp_path / "mutant.sh"
    copy.write_text(lib.replace(old, 'if [[ ":$PATH:" != *":$dir:"* ]]; then'))
    cmd = f'source {copy}; ferric_ensure_cargo_shim "{REPO}"'
    p = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    assert "not first on PATH" not in p.stderr, p.stderr


def test_a_shim_differing_from_this_tree_warns(tmp_path):
    dest, _ = _ensure(tmp_path)
    (dest / "cargo").write_text("#!/bin/sh\n# old\n")
    env = {
        **os.environ,
        "FERRIC_SHIM_DIR": str(dest),
        "PATH": f"{dest}:{os.environ['PATH']}",
    }
    cmd = f'source {LIB}; ferric_ensure_cargo_shim "{REPO}"'
    p = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    assert "differs from this tree" in p.stderr, p.stderr


def test_the_default_shim_dir_is_stable_and_survives_an_unset_home(tmp_path):
    home = tmp_path / "home"
    (home / ".local/share/ferric-cargo-shim").mkdir(parents=True)
    shim = home / ".local/share/ferric-cargo-shim/cargo"
    shim.write_text("#!/bin/sh\n")
    shim.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "FERRIC_SHIM_DIR"}
    cmd = f'source {LIB}; ferric_ensure_cargo_shim "{REPO}"; echo rc=$?'
    p = subprocess.run(
        ["bash", "-c", cmd],
        env={**env, "HOME": str(home)},
        capture_output=True,
        text=True,
    )
    assert "rc=0" in p.stdout and f"{shim} is not first" in p.stderr, (
        p.stdout + p.stderr
    )
    env.pop("HOME", None)
    p = subprocess.run(["bash", "-uc", cmd], env=env, capture_output=True, text=True)
    assert "rc=0" in p.stdout, p.stdout + p.stderr


def test_the_gate_and_the_hook_both_check_for_the_shim():
    assert (
        'ferric_ensure_cargo_shim "$REPO_ROOT"' in (SCRIPTS / "ci-gate.sh").read_text()
    )
    assert (
        'ferric_ensure_cargo_shim "$TOPLEVEL"'
        in (SCRIPTS / "install-hooks.sh").read_text()
    )
