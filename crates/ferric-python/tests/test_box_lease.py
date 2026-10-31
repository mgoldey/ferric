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

from test_ci_gate_timeout_starts_after_the_lock import LIB, _gate_functions

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
if [ -n "${STUB_UNTIL:-}" ]; then
  until [ -e "$STUB_UNTIL" ]; do sleep 0.02; done
fi
echo "end ${STUB_NAME:-cargo}" >> "$EVENTS"
exit "${STUB_RC:-0}"
"""


class Box:
    """A private shim directory, stub cargo and lock directory."""

    def __init__(self, tmp_path, lease_edit=None, shim_edit=None):
        self.tmp = tmp_path
        self.locks = tmp_path / "locks"
        self.events = tmp_path / "events"
        self.shim_dir = tmp_path / "ferric-cargo-shim"
        real = tmp_path / "real"
        for d in (self.locks, self.shim_dir, real):
            d.mkdir()
        self.events.touch()
        self._copy("box-lease", "box-lease", lease_edit)
        self._copy("cargo-shim", "cargo", shim_edit)
        self._executable(real / "cargo", STUB_CARGO)
        self._executable(real / "sccache", "#!/bin/sh\n")
        self.lease = self.locks / "ferric-box.lease"
        self.info = self.locks / "ferric-box.lease.info"
        self.cargo_lock = self.locks / "ferric-cargo.lock"
        self.slot0 = self.locks / "ferric-compile.lock"
        self.slot1 = self.locks / "ferric-compile.2.lock"
        self.env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("FERRIC_HEAVY_LOCK_HELD", "FERRIC_BOX_LEASE_PID", "STUB_NAME")
        }
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
            f"until [ -e {release} ]; do sleep 0.02; done\n"
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

    def index(self, event):
        return next(
            i
            for i, ln in enumerate(self.lines())
            if ln == event or ln.startswith(event + " ")
        )

    def finish(self, proc, limit=LIMIT):
        out, _ = proc.communicate(timeout=limit)
        return proc.returncode, out

    def cleanup(self):
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)  # its own session, by pid
                except ProcessLookupError:
                    pass
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


def gate_step(box, command, timeout_secs, lock_wait_secs=60):
    """Run the gate's REAL run_step on `command`, in its own bash."""
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
    """Lease held past the step's 1 s timeout. Return (gate output, events)."""
    job = box.lease_job("L")
    box.wait_for("start L")
    gate = gate_step(box, "cargo build", 1)
    time.sleep(2.5)
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
    status = subprocess.run(
        [box.shim_dir / "box-lease", "status"],
        env=box.env,
        capture_output=True,
        text=True,
    )
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
    """The 1 s step timeout starts only after the lease freed the cargo lock."""
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
    status = subprocess.run(
        [box.shim_dir / "box-lease", "status"],
        env=box.env,
        capture_output=True,
        text=True,
    )
    assert "HELD by L" in status.stdout and "max-hold" in status.stdout, status.stdout
    box.open("L")
    rc, out = box.finish(job)
    assert rc == 0
    assert out.count("WARNING: L has held the box") == 1, out


def test_status_names_the_holder_and_the_locks(make_box):
    box = make_box()

    def status():
        return subprocess.run(
            [box.shim_dir / "box-lease", "status"],
            env=box.env,
            capture_output=True,
            text=True,
        ).stdout

    assert status().count(": free") == 4
    job = box.lease_job("L")
    box.wait_for("start L")
    out = status()
    assert "lease: HELD by L (pid " in out and "active since" in out, out
    assert out.count(": held") == 3, out
    box.open("L")
    box.finish(job)
    assert status().count(": free") == 4
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
    assert sorted(p.name for p in dest.iterdir()) == [
        "box-lease",
        "cargo",
        "cargo.prev",
    ]


# ---- mutants: each scenario must SEE the defect it exists for ---------------

NO_WAIT = (
    "if lease_busy && ! ancestor_holds_heavy_lock; then",
    "if false; then",
)
NO_ANCESTOR_CHECK = (
    "if lease_busy && ! ancestor_holds_heavy_lock; then",
    "if lease_busy; then",
)
NO_DRAIN = ('    take 202 "compile slot 0"\n    take 203 "compile slot 1"\n', "")
ONE_SLOT = ('    take 203 "compile slot 1"\n', "")
NO_CARGO_LOCK = ('    take 201 "the gate\'s cargo lock"\n', "")
# The order is what prevents the gate deadlock: slots first, then the cargo lock.
SLOTS_BEFORE_CARGO_LOCK = (
    '    take 201 "the gate\'s cargo lock"\n'
    '    take 202 "compile slot 0"\n'
    '    take 203 "compile slot 1"\n',
    '    take 202 "compile slot 0"\n'
    '    take 203 "compile slot 1"\n'
    '    take 201 "the gate\'s cargo lock"\n',
)
# Nothing marks the lease as pending, so it cannot hold compiles back.
LEASE_NOT_HELD = (
    '    take 200 "the box lease, held by $(holder_text)"\n',
    "",
)
NO_REENTRY = ("export FERRIC_BOX_LEASE_PID=$$ FERRIC_HEAVY_LOCK_HELD=1", "true")
LEAKS_FDS = ('"$@" <&0 200>&- 201>&- 202>&- 203>&- &', '"$@" <&0 &')


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
    box = make_box(lease_edit=NO_REENTRY)
    job = box.lease_job("L", body="cargo build")
    time.sleep(HOLD_OFF)
    assert box.happened("start L") and not box.happened("start cargo")
    assert job.poll() is None


def test_mutant_no_ancestor_check_deadlocks_the_gate(make_box):
    box = make_box(shim_edit=NO_ANCESTOR_CHECK)
    out, _ = gate_step_calls_cargo_with_a_lease_pending(box)
    assert "-- probe: TIMEOUT" in out, out


def test_mutant_slots_before_cargo_lock_deadlocks_the_gate(make_box):
    box = make_box(lease_edit=SLOTS_BEFORE_CARGO_LOCK)
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
