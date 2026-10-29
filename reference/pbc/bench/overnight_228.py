"""Unattended, resumable quiet-window run of the remaining issue-#228 plan (ferric vs PySCF GDF).

  # one-time: build ferric-cli + ferric-python at origin/main (see OVERNIGHT.md), then
  python overnight_228.py --run-dir DIR [--hours 10] [--tiers anchor,core,pbe-gated]
  python overnight_228.py --run-dir DIR --dry-run      # parse + print every command, run nothing
  python overnight_228.py --run-dir DIR --smoke        # one toy cell once, gate NOT enforced
  python analyze_228.py DIR/results.jsonl               # medians, spread, ratio + bootstrap CI

What it adds on top of scatter_driver.py (whose run_ferric / run_pyscf / locks / pinning / sha
checks / stage parsing it calls unchanged, through the `ctl` hook):
- FIXED BINARY: on the first launch the ferric CLI and the python extension are COPIED into
  DIR/bin (never run from target/: a rebuild under a running job SIGBUSes it); `ferric --version`
  and the extension's `__build__` are recorded in DIR/manifest.json and must name the same clean
  commit, which must contain #508 (fc0ccc041).  A resumed launch re-hashes DIR/bin/ferric and
  refuses a mismatch.  All rows carry the same `bin_sha256`.
- QUIET-WINDOW GATE (class Gate), read before AND after every sample, logged on every row
  (`gate_before` / `gate_after`).  A sample counts (`gated` = true, and `valid` only then) iff both
  ends pass; otherwise it is discarded and re-run (<= --max-attempts per key).  If the box is not
  quiet the driver sleeps and polls until --hours runs out, then exits 3 (resume later).
  The busy / foreign bars are the box's IDLE FLOOR (measured at startup, recorded in the manifest
  and on every row) + the pre-registered 0.02 margin; a floor > --max-floor refuses to start.
- RESUMABLE: DIR/results.jsonl is append-only, one row per sample, keyed by (plan id, code,
  variant, threads, rep).  A key with a gated, converged row is done; a killed run picks up at the
  first key that is not.  A second launch on the same DIR while one runs exits (flock on DIR/lock).
  `run_id` = `session` = DIR's name, so reps pair across resumed launches of one DIR (same binary);
  `invocation` tells the launches apart.
- MEMORY: every process runs under scripts/ferric-limited (MemoryMax = MemoryHigh = --mem, so the
  cgroup kills a runaway instead of throttling it, which would silently distort a timing).
- ORDER: rep-major (rep 0 of every cell, then rep 1, ...) so each cell's samples are spread over
  the night; inside a rep the [ferric variants..., PySCF] sequence is back to back and rotated by
  the rep index (scatter_driver.rotated_order).
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import pyscf_bench as pb  # noqa: E402
import scatter_driver as sd  # noqa: E402
import summarize as sm  # noqa: E402

PR508 = "fc0ccc041"  # perf(pbc): RS-GDF LR chunk (#508); every pre-#508 dryice_112 number is stale
TOML_RI = os.path.join(HERE, "toml_ri")
MAINLINE = os.path.expanduser("~/qc/ferric-mainline/target/release")
PYSCF_PREC_FACTOR = (
    1.3  # ASSUMED slowdown of PySCF at precision 1e-13 vs the 1e-11 timings
)
GATE_OVERHEAD_S = 15.0  # two 5 s gate windows + process start, per sample (estimate)
OK_EXIT = 0
DEADLINE_EXIT = 3
EXHAUSTED_EXIT = 4


class Deadline(Exception):
    pass


# --------------------------------------------------------------------------------- the plan ----
@dataclasses.dataclass(frozen=True)
class Cell:
    """One plan row.  Ferric input = <toml_dir>/<cell>_<method><tag>.toml (scatter_driver's rule);
    PySCF runs pyscf_bench.py on <cell> (a `_kabc` suffix becomes --kmesh a b c).

    est: {threads: (ferric s, PySCF s)} per sample, ESTIMATES from README.md session tables
    (N=6 measured; N=1 measured where a table has it, else 5 x N=6); PySCF scaled by
    PYSCF_PREC_FACTOR for precision 1e-13."""

    id: str
    cell: str
    method: str
    tags: dict
    reps: dict
    tier: str
    est: dict
    kind: str = "timed"
    pyscf_extra: tuple = ()
    toml_dir: str = TOML_RI

    def threads(self):
        return sorted(self.tags, reverse=True)


W = ("_w0.3", "_w0.5", "_w0.7")
W4 = W + ("_w1.0",)
LADDER = ("", "_150x302", "_150x434", "_150x590", "_100x590", "_200x590", "_250x590")
R6, R1 = 5, 3  # pre-registered valid reps (README "Session 4 pre-registration")

# fmt: off
PLAN = [
    # Untimed energy anchors for the dryice PBE ladder gate (summarize.ladder_gate); N=6, once.
    Cell("dryice_pbe_ladder", "dryice", "pbe", {6: LADDER}, {6: 1}, "anchor",
         {6: (60.0, 90.0)}, kind="anchor"),
    # Timed rows.  omega sweep (ferric tags) where the session-4 pre-registration put it.
    Cell("diamond_prim_rhf", "diamond_prim", "rhf", {6: ("",) + W, 1: ("",)}, {6: R6, 1: R1},
         "core", {6: (13.0, 11.7), 1: (39.3, 47.0)}),
    Cell("dryice_rhf", "dryice", "rhf", {6: ("",) + W, 1: ("", "_w0.5")}, {6: R6, 1: R1},
         "core", {6: (15.4, 35.4), 1: (72.0, 138.4)}),
    Cell("diamond_prim_k222_rhf", "diamond_prim_k222", "rhf", {6: ("",), 1: ("",)},
         {6: R6, 1: R1}, "core", {6: (13.0, 23.6), 1: (54.4, 89.4)}),
    Cell("diamond_prim_k333_rhf", "diamond_prim_k333", "rhf", {6: ("",)}, {6: R6}, "core",
         {6: (150.0, 270.0)}),  # k333 never timed: ~(27/8)^2 x k222, a GUESS
    Cell("diamond_conv_rhf", "diamond_conv", "rhf", {6: ("",)}, {6: R6}, "core",
         {6: (33.5, 43.0)}),
    Cell("diamond_prim222_rhf", "diamond_prim222", "rhf", {6: ("",)}, {6: R6}, "core",
         {6: (92.7, 171.4)}),
    Cell("dryice_112_rhf", "dryice_112", "rhf", {6: W4}, {6: R6}, "core",
         {6: (157.0, 142.9)}),  # ferric = 314 s / 2 (#508 "halved"); omega variants unknown
    Cell("diamond_prim_pbe", "diamond_prim", "pbe", {6: W4, 1: ("",)}, {6: R6, 1: R1}, "core",
         {6: (19.0, 21.9), 1: (63.8, 102.1)}),
    # Only if the ladder gate on the anchors above passes.
    Cell("dryice_pbe", "dryice", "pbe", {6: ("",), 1: ("",)}, {6: R6, 1: R1}, "pbe-gated",
         {6: (30.0, 50.0), 1: (150.0, 250.0)}),  # never timed: a GUESS
    # Opt-in (--tiers ...,n1-large): the N=1 large cells.
    Cell("diamond_conv_rhf_n1", "diamond_conv", "rhf", {1: ("",)}, {1: R1}, "n1-large",
         {1: (167.0, 215.0)}),
    Cell("diamond_prim222_rhf_n1", "diamond_prim222", "rhf", {1: ("",)}, {1: R1}, "n1-large",
         {1: (464.0, 857.0)}),
    Cell("dryice_112_rhf_n1", "dryice_112", "rhf", {1: ("_w1.0",)}, {1: R1}, "n1-large",
         {1: (785.0, 715.0)}),
]
SMOKE = Cell("smoke_h2", "smoke_h2", "rhf", {1: ("",)}, {1: 1}, "smoke", {1: (2.0, 5.0)},
             pyscf_extra=("--basis", "sto-3g"))
# fmt: on
DEFAULT_TIERS = "anchor,core,pbe-gated"


def grid_of(cell, tag):
    """PySCF --grid for a PBE ferric tag (`_RxA`); None = pyscf_bench's 75x302 default."""
    m = re.fullmatch(r"_(\d+)x(\d+)", tag)
    return (int(m.group(1)), int(m.group(2))) if (cell.method == "pbe" and m) else None


def sequence(cell, threads, rep):
    """[(code, tag)] for one rep: every ferric tag then PySCF, rotated by rep (anchors: one
    PySCF run per grid tag, since the grid is the variable)."""
    tags = cell.tags[threads]
    if cell.kind == "anchor":
        return [("ferric", t) for t in tags] + [("pyscf", t) for t in tags]
    return sd.rotated_order([("ferric", t) for t in tags] + [("pyscf", "")], rep)


def key_of(cell, code, tag, threads, rep):
    return (cell.id, code, tag, threads, rep)


def row_key(r):
    return (r["plan_id"], r["code"], r["variant"], r["threads"], r["rep"])


def pyscf_argv(cell, threads, tag, precision):
    """pyscf_bench.py argv (what scatter_driver.run_pyscf builds; used for the dry run)."""
    base, kmesh = sd.split_kmesh(cell.cell)
    argv = [
        base,
        "--threads",
        str(threads),
        "--precision",
        str(precision),
        "--method",
        cell.method,
    ] + (["--kmesh"] + kmesh if kmesh else [])
    g = grid_of(cell, tag)
    if g:
        argv += ["--grid", str(g[0]), str(g[1])]
    return argv + ["--match-ferric-conv"] + list(cell.pyscf_extra)


# ------------------------------------------------------------------------------------ gate ----
# Alive at all = a compile is running.  cargo / ninja / sccache only count when they burn CPU: a
# cargo shim BLOCKED on the heavy-slot lock this driver holds is idle and must not stall the gate.
DENY = ("rustc", "cc1plus", "cc1", "collect2", "ld", "ld.lld", "rust-lld")
DENY_BUSY, DENY_BUSY_CPU = ("cargo", "ninja", "sccache"), 0.01
PYLIKE = ("pytest", "uv", "uvx")  # get the python cap too (comm of a script = its name)


def physical_cores():
    """Distinct (package, core) pairs: 6 on the i7-6800K (12 logical).  The SMT siblings of the
    pinned cores are outside the taskset, so foreign work is measured against physical cores."""
    seen = set()
    for d in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology"):
        try:
            pkg = open(os.path.join(d, "physical_package_id")).read().strip()
            core = open(os.path.join(d, "core_id")).read().strip()
        except OSError:
            continue
        seen.add((pkg, core))
    return len(seen) or (os.cpu_count() or 1)


PHYS = physical_cores()
LOGICAL = os.cpu_count() or PHYS


def to_phys(frac_of_logical):
    """A fraction of all logical CPUs, re-expressed as a fraction of the physical cores."""
    return frac_of_logical * LOGICAL / PHYS


def read_loadavg():
    la = open("/proc/loadavg").read().split()
    return float(la[0]), float(la[1])


def read_stat():
    """Aggregate /proc/stat cpu line: dict(total, busy, iowait, steal) in jiffies."""
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    steal = v[7] if len(v) > 7 else 0
    return dict(
        total=sum(v[:8]), busy=sum(v[:8]) - v[3] - v[4], iowait=v[4], steal=steal
    )


def proc_times(own=()):
    """{pid: (comm, utime+stime jiffies)} from /proc/<pid>/stat (never pgrep)."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in own:
            continue
        try:
            s = open(f"/proc/{d}/stat").read()
        except OSError:
            continue
        lp, rp = s.find("("), s.rfind(")")
        f = s[rp + 2 :].split()
        out[int(d)] = (s[lp + 1 : rp], int(f[11]) + int(f[12]))
    return out


def stat_fracs(s0, s1):
    tot = max(s1["total"] - s0["total"], 1)
    return {k: (s1[k] - s0[k]) / tot for k in ("busy", "iowait", "steal")}


def own_load(history, now):
    """Our own processes' remaining share of the 1-min loadavg: each finished sample of
    effective parallelism p = cpu_s / wall adds p (1 - e^{-wall/60}) e^{-(now - end)/60}
    (the kernel's exponential average with a 60 s constant).  A model, so raw load1 is logged."""
    return sum(
        p * (1 - math.exp(-w / 60)) * math.exp(-(now - e) / 60) for e, p, w in history
    )


MARGIN = (
    0.02  # pre-registered quiet bar, now read as a margin ABOVE the measured idle floor
)
MAX_FLOOR = 0.10  # an idle floor above this means the box is not idle: refuse to start
FLOOR_SAMPLES = 6
UNREACHABLE_AFTER_MIN = 30.0


class FloorTooHigh(SystemExit):
    pass


def resolve_bar(th, which, floor):
    """The bar for `which` in ("busy", "foreign"), as a fraction of the physical cores.

    An explicit --max-busy / --max-foreign is ABSOLUTE and wins.  Otherwise the bar is the idle
    floor measured at startup (Gate.measure_floor) plus the pre-registered margin.  Without
    either there is no bar at all, and asking for one is an error (never a silent 0.02)."""
    absolute = getattr(th, f"max_{which}", None)
    if absolute is not None:
        return absolute
    if floor is None:
        raise RuntimeError(
            f"no {which} bar: pass --max-{which} or measure the idle floor first"
        )
    return floor[f"{which}_floor"] + getattr(th, f"{which}_margin", MARGIN)


def judge(reading, th, phase="before", busy_bar=None):
    """Reasons a reading fails the thresholds (empty = quiet).  `busy_bar` defaults to
    th.max_busy (absolute); Gate passes its floor + margin bar.

    The loadavg is judged only BEFORE a sample.  Right after one, load1 still holds the sample's
    own ~N (1 - e^{-wall/60}): judging it there would discard every long N=6 sample and keep
    the fast ones (a selection bias in the timed statistic).  The after phase judges FOREIGN
    activity only: busy CPU in the post-run window (we are idle then), processes, iowait/steal,
    plus the run-window foreign CPU (Gate.gate)."""
    why = []
    raw = getattr(th, "load_mode", "raw") == "raw"
    load = reading["load1"] if raw else reading["load1_adj"]
    if phase != "after" and load >= th.max_load:
        why.append(
            f"load1 {reading['load1']:.2f} (own-adjusted {reading['load1_adj']:.2f})"
        )
    bar = th.max_busy if busy_bar is None else busy_bar
    if bar is None:
        raise RuntimeError(
            "no busy bar: pass --max-busy or measure the idle floor first"
        )
    if reading["busy_phys"] >= bar:
        why.append(
            f"busy {reading['busy_phys']:.3f} of the physical cores (bar {bar:.3f})"
        )
    if th.check_steal_iowait:
        if reading["iowait_frac"] >= th.max_iowait:
            why.append(f"iowait {reading['iowait_frac']:.3f}")
        if reading["steal_frac"] >= th.max_steal:
            why.append(f"steal {reading['steal_frac']:.3f}")
    for p in reading["heavy"]:
        why.append(f"proc {p['comm']}[{p['pid']}] {p['cpu']:.0%}")
    return why


class Gate:
    """Quiet-window gate.  `gate(phase, **ctx)` is scatter_driver's ctl hook.

    A reading = `window` s of /proc/stat (box busy / iowait / steal; our own process is idle
    then, so busy = foreign; busy is judged as a fraction of the PHYSICAL cores) and of
    /proc/<pid>/stat (per-process CPU), plus the 1-min loadavg (raw, or minus own_load).  Quiet
    iff load < max_load (before/recheck only, see judge), busy < busy bar, iowait/steal under
    their caps, and no heavy process: rustc/cc1plus/ld/... alive at all, cargo/ninja/sccache
    above 1% of a core, python*/pytest/uv above max_python_cpu of one core, or any other process
    above max_proc_cpu.
    before:  poll until quiet (sleep `poll` s between readings), or raise Deadline.  Called by
             scatter_driver.run_cmd WITHOUT the heavy-slot locks.
    recheck: one reading right after the locks are taken; never waits.
    after:   one reading (no loadavg), plus the run window: foreign CPU (scatter_driver's
             accounting, re-expressed per physical core) < foreign bar, steal under its cap.
             Never waits.

    BARS (resolve_bar): the busy and foreign bars are the box's IDLE FLOOR, measured once per
    launch by measure_floor() while nothing of ours runs, plus the pre-registered 0.02 margin.
    An absolute 0.02 was unreachable on the desktop box (idle floor 0.035-0.045: night 1 waited
    2 h 13 min and took no sample).  Explicit --max-busy / --max-foreign stay absolute."""

    def __init__(self, th, deadline, enforce=True):
        self.th, self.deadline, self.enforce = th, deadline, enforce
        self.history = []  # (end time, effective parallelism, wall) of our recent samples
        self._stat_before = None
        self._waited, self._polls = 0.0, 0
        self.floor = None  # measure_floor(); recorded in the manifest and on every row

    @property
    def busy_bar(self):
        return resolve_bar(self.th, "busy", self.floor)

    @property
    def foreign_bar(self):
        return resolve_bar(self.th, "foreign", self.floor)

    def bars(self):
        """What every row records: the floor this launch measured and the bars it judged by."""
        f = self.floor or {}
        return dict(
            busy_floor=f.get("busy_floor"),
            foreign_floor=f.get("foreign_floor"),
            floor_measured_at=f.get("measured_at"),
            busy_bar=round(self.busy_bar, 4),
            foreign_bar=round(self.foreign_bar, 4),
            busy_bar_mode="absolute"
            if self.th.max_busy is not None
            else "floor+margin",
            foreign_bar_mode="absolute"
            if self.th.max_foreign is not None
            else "floor+margin",
        )

    def measure_floor(self, k=None, max_floor=None, refuse=True):
        """The box's idle floor: k back-to-back readings while nothing of ours runs.

        busy_floor = MEDIAN busy_phys (what one before/after window is judged on);
        foreign_floor = MEAN busy_phys over the same readings (the run-window foreign CPU is an
        average over the whole run, and with our process idle, busy == foreign).
        Readings with a heavy process (a compile, a python sweep) are excluded: they are not the
        idle desktop.  Refuses (FloorTooHigh) if fewer than half the readings are clean or a
        floor exceeds max_floor (default 0.10): the box is not idle, and a bar built on it would
        quietly accept a loaded box."""
        k = k or getattr(self.th, "floor_samples", FLOOR_SAMPLES)
        max_floor = (
            max_floor
            if max_floor is not None
            else getattr(self.th, "max_floor", MAX_FLOOR)
        )
        reads = [self._read_raw("floor") for _ in range(k)]
        clean = [r for r in reads if not r["heavy"]]
        # No clean reading at all (refused anyway): report the numbers over every reading.
        busy = sorted(r["busy_phys"] for r in (clean or reads))
        problems = []
        if len(clean) * 2 < k:
            heavy = sorted({p["comm"] for r in reads for p in r["heavy"]})
            problems.append(
                f"only {len(clean)}/{k} floor readings had no heavy process ({heavy})"
            )
        floor = dict(
            measured_at=time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())
            ),
            n=k,
            n_clean=len(clean),
            window_s=self.th.window,
            busy_floor=round(_median(busy), 4),
            foreign_floor=round(sum(busy) / len(busy), 4),
            readings=[
                dict(t=r["t"], busy_phys=r["busy_phys"], heavy=r["heavy"])
                for r in reads
            ],
            max_floor=max_floor,
        )
        for which in ("busy", "foreign"):
            v = floor[f"{which}_floor"]
            if v > max_floor:
                problems.append(
                    f"{which} idle floor {v:.3f} of the physical cores > {max_floor:.2f}"
                )
        floor["problems"] = problems
        if problems and refuse:
            raise FloorTooHigh(
                "refused: the box is not idle enough to measure a quiet-window floor:\n  "
                + "\n  ".join(problems)
                + "\n  readings (busy_phys): "
                + ", ".join(f"{r['busy_phys']:.3f}" for r in reads)
                + "\n  stop the load, or raise --max-floor knowingly (rows record the floor)"
            )
        if not problems:
            self.floor = floor
        return floor

    def read(self, phase="before"):
        r = self._read_raw(phase)
        r["busy_bar"] = round(self.busy_bar, 4)
        r["reasons"] = judge(r, self.th, phase, busy_bar=self.busy_bar)
        r["ok"] = not r["reasons"]
        return r

    def _read_raw(self, phase):
        """One window's numbers, not judged."""
        own = {os.getpid()}
        s0, p0, t0 = read_stat(), proc_times(own), time.time()
        time.sleep(self.th.window)
        s1, p1, t1 = read_stat(), proc_times(own), time.time()
        hz, dt = os.sysconf("SC_CLK_TCK"), t1 - t0
        heavy = []
        for pid, (comm, j1) in p1.items():
            cpu = (j1 - p0[pid][1]) / hz / dt if pid in p0 else 0.0
            cap = (
                self.th.max_python_cpu
                if comm.startswith("python") or comm in PYLIKE
                else self.th.max_proc_cpu
            )
            if comm in DENY or (comm in DENY_BUSY and cpu > DENY_BUSY_CPU) or cpu > cap:
                heavy.append(dict(pid=pid, comm=comm, cpu=round(cpu, 3)))
        l1, l5 = read_loadavg()
        tail = own_load(self.history, t1)
        fr = stat_fracs(s0, s1)
        r = dict(
            phase=phase,
            t=round(t1, 1),
            window_s=round(dt, 2),
            load1=l1,
            load5=l5,
            own_load=round(tail, 3),
            load1_adj=round(l1 - tail, 3),
            busy_frac=round(fr["busy"], 4),
            busy_phys=round(to_phys(fr["busy"]), 4),
            iowait_frac=round(fr["iowait"], 4),
            steal_frac=round(fr["steal"], 4),
            heavy=heavy,
        )
        return r

    def gate(self, phase, **ctx):
        if phase == "recheck":
            # Locks just taken (scatter_driver.run_cmd): one reading, no waiting.  A failure
            # makes run_cmd release the locks and call gate("before") again.
            r = self.read("recheck")
            r.update(enforced=self.enforce, waited_s=self._waited, polls=self._polls)
            self._stat_before = read_stat()
            return r
        if phase == "before":
            t_start, n = time.time(), 0
            idle_fail_since, idle_busy, warned = None, [], False
            while True:
                r = self.read()
                n += 1
                if r["ok"] or not self.enforce:
                    break
                # Failing on busy alone, at idle-desktop level (<= max_floor): the bar itself may
                # be below what this box can do.  Say so once per wait; never wait silently.
                idle_level = r["busy_phys"] <= getattr(self.th, "max_floor", MAX_FLOOR)
                if idle_level and all(w.startswith("busy") for w in r["reasons"]):
                    if idle_fail_since is None:
                        idle_fail_since = time.time()
                    idle_busy.append(r["busy_phys"])
                else:
                    idle_fail_since, idle_busy = None, []
                after = 60 * getattr(
                    self.th, "unreachable_after", UNREACHABLE_AFTER_MIN
                )
                if (
                    not warned
                    and idle_fail_since is not None
                    and time.time() - idle_fail_since >= after
                ):
                    warned = True
                    b = sorted(idle_busy)
                    print(
                        f"[gate] WARNING: the busy bar {self.busy_bar:.3f} may be UNREACHABLE: "
                        f"{(time.time() - idle_fail_since) / 60:.0f} min of failures from "
                        f"idle-level busy readings alone (n={len(b)}, min {b[0]:.3f} / median "
                        f"{_median(b):.3f} / max {b[-1]:.3f}; floor {self.bars()['busy_floor']}, "
                        f"mode {self.bars()['busy_bar_mode']}). Still waiting until the deadline.",
                        flush=True,
                    )
                if time.time() + self.th.poll > self.deadline:
                    raise Deadline(
                        f"box not quiet by the deadline; last: {r['reasons']}"
                    )
                if n == 1 or n % 20 == 0:
                    print(f"[gate] waiting: {'; '.join(r['reasons'])}", flush=True)
                # Only our own previous sample's load tail left: re-check soon, not in `poll`.
                own_tail_only = r.get("own_load", 0) > 0 and all(
                    w.startswith("load1") for w in r["reasons"]
                )
                time.sleep(min(self.th.poll, 10.0) if own_tail_only else self.th.poll)
            self._waited, self._polls = round(time.time() - t_start, 1), n
            r.update(
                waited_s=self._waited,
                polls=n,
                enforced=self.enforce,
                unreachable_warned=warned,
            )
            self._stat_before = read_stat()
            return r
        wall, cpu = ctx["wall"], ctx["cpu_s"]
        run = stat_fracs(self._stat_before, read_stat()) if self._stat_before else None
        self.history = (self.history + [(time.time(), cpu / max(wall, 1e-9), wall)])[
            -8:
        ]
        r = self.read("after")
        foreign = to_phys(ctx["foreign"])  # scatter_driver divides by ALL logical CPUs
        r.update(
            enforced=self.enforce,
            run_foreign_frac=round(ctx["foreign"], 4),
            run_foreign_phys=round(foreign, 4),
            run_iowait_frac=run and round(run["iowait"], 4),
            run_steal_frac=run and round(run["steal"], 4),
        )
        r["foreign_bar"] = round(self.foreign_bar, 4)
        if foreign >= self.foreign_bar:
            r["reasons"].append(
                f"foreign CPU over the run {foreign:.3f} of the cores "
                f"(bar {self.foreign_bar:.3f})"
            )
        # Run-window iowait is NOT judged: PySCF writes its own cderi file, so it is partly
        # ours (logged only; the post-run window's iowait above is foreign).  Steal is never ours.
        if run and self.th.check_steal_iowait and run["steal"] >= self.th.max_steal:
            r["reasons"].append(f"steal over the run {run['steal']:.3f}")
        r["ok"] = not r["reasons"]
        return r


def _median(xs):
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


class Ctl:
    """scatter_driver's ctl: argv prefix (memory cap) + gate hook."""

    def __init__(self, prefix, gate):
        self.prefix, self._gate = prefix, gate

    def gate(self, phase, **ctx):
        return self._gate.gate(phase, **ctx)


# --------------------------------------------------------------------------- fixed binary ----
def ferric_version(binary):
    p = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, timeout=60
    )
    txt = (p.stdout + p.stderr).strip()
    m, d = (
        re.search(r"commit:\s*([0-9a-f]{7,40})", txt),
        re.search(r"dirty:\s*(\w+)", txt),
    )
    return dict(
        text=txt,
        commit=m and m.group(1),
        dirty=d and d.group(1) == "true",
        ok=p.returncode == 0,
    )


def so_build(so, python):
    """__build__ of a ferric extension FILE, loaded by path in an isolated interpreter (a
    PYTHONPATH shim can lose to the installed package; this cannot)."""
    code = (
        "import importlib.machinery as m, importlib.util as u, json, sys\n"
        "l = m.ExtensionFileLoader('ferric', sys.argv[1])\n"
        "mod = u.module_from_spec(u.spec_from_loader('ferric', l)); l.exec_module(mod)\n"
        "print(json.dumps(dict(build=getattr(mod, '__build__', None),"
        " version=getattr(mod, '__version__', None))))"
    )
    p = subprocess.run(
        [python, "-I", "-c", code, so], capture_output=True, text=True, timeout=120
    )
    if p.returncode:
        return dict(error=p.stderr.strip()[-400:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def contains(commit, ancestor=PR508):
    if not commit:
        return None
    p = subprocess.run(
        ["git", "-C", ROOT, "merge-base", "--is-ancestor", ancestor, commit],
        capture_output=True,
    )
    return {0: True, 1: False}.get(p.returncode)  # 128 = unknown commit -> None


def provenance(bin_path, so_path, python):
    v = ferric_version(bin_path)
    b = (
        so_build(so_path, python)
        if so_path and os.path.exists(so_path)
        else dict(error="no .so")
    )
    so_sha = ((b.get("build") or {}).get("git_sha")) if "build" in b else None
    problems = []
    if not v["ok"] or not v["commit"]:
        problems.append(f"`ferric --version` gave no commit: {v['text'][:200]!r}")
    if v["dirty"]:
        problems.append("ferric binary is dirty")
    if "error" in b:
        problems.append(f"python extension: {b['error']}")
    elif not isinstance(b.get("build"), dict):
        problems.append("python extension has no __build__ (built without build info)")
    elif b["build"].get("dirty") is not False:
        problems.append(f"python extension dirty flag is {b['build'].get('dirty')!r}")
    if "error" not in b and not re.fullmatch(r"[0-9a-f]{7,40}", str(so_sha)):
        problems.append(f"python extension commit unknown: {so_sha!r}")
    elif v["commit"] and so_sha and not so_sha.startswith(v["commit"][:12]):
        problems.append(
            f"binary commit {v['commit'][:12]} != extension commit {so_sha[:12]}"
        )
    has = contains(v["commit"])
    if has is not True:
        problems.append(f"cannot confirm the binary contains #508 ({PR508}): {has}")
    return dict(ferric_version=v, so_build=b, contains_pr508=has), problems


def python_env(python):
    """Interpreter + library versions the PySCF side runs on (recorded; checked on resume)."""
    code = (
        "import json, sys, numpy, scipy, pyscf\n"
        "print(json.dumps(dict(executable=sys.executable, version=sys.version.split()[0],"
        " pyscf=pyscf.__version__, numpy=numpy.__version__, scipy=scipy.__version__)))"
    )
    p = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, timeout=120
    )
    if p.returncode:
        raise SystemExit(
            f"refused: {python} cannot import pyscf/numpy/scipy: {p.stderr[-300:]}"
        )
    return dict(path=python, **json.loads(p.stdout.strip().splitlines()[-1]))


def prepare_bin(a, run_dir, python, smoke):
    """Copy the binary + extension into run_dir/bin once; later launches verify the hash."""
    bdir = os.path.join(run_dir, "bin")
    man_path = os.path.join(run_dir, "manifest.json")
    binary, so = os.path.join(bdir, "ferric"), os.path.join(bdir, "libferric.so")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        sha = sd.sha256_file(binary)
        if sha != man["bin_sha256"]:
            raise SystemExit(
                f"refused: {binary} sha256 {sha} != manifest {man['bin_sha256']}"
            )
        so_sha = sd.sha256_file(so) if os.path.exists(so) else None
        if so_sha != man.get("so_sha256"):
            raise SystemExit(
                f"refused: {so} sha256 {so_sha} != manifest {man.get('so_sha256')}"
            )
        env = python_env(python)
        if env != man.get("python_env"):
            raise SystemExit(
                f"refused: python environment changed since this run dir started:\n  "
                f"now {env}\n  manifest {man.get('python_env')}"
            )
        for k in ("precision", "mem"):
            if man["settings"].get(k) != getattr(a, k):
                raise SystemExit(
                    f"refused: --{k} {getattr(a, k)} != this run dir's "
                    f"{man['settings'].get(k)} (start a new --run-dir)"
                )
        return binary, man
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(a.ferric_bin, binary)  # copy, not link: a private inode, st_nlink == 1
    if os.path.exists(a.ferric_so):
        shutil.copy2(a.ferric_so, so)
    prov, problems = provenance(binary, so, python)
    if problems and not (smoke or a.allow_unverified):
        shutil.rmtree(bdir)
        raise SystemExit(
            "refused (pass --allow-unverified to override):\n  " + "\n  ".join(problems)
        )
    sd.check_private_binary(binary)
    man = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        ferric_src=a.ferric_bin,
        so_src=a.ferric_so,
        bin_sha256=sd.sha256_file(binary),
        so_sha256=sd.sha256_file(so) if os.path.exists(so) else None,
        provenance_problems=problems,
        settings=dict(precision=a.precision, mem=a.mem),
        python=python,
        python_env=python_env(python),
        **prov,
    )
    with open(man_path, "w") as f:
        json.dump(man, f, indent=1)
    return binary, man


def record_floor(run_dir, floor, bars):
    """Append this launch's floor to manifest.json (atomically); resume checks ignore it."""
    path = os.path.join(run_dir, "manifest.json")
    man = json.load(open(path))
    man.setdefault("idle_floors", []).append(dict(floor, bars=bars))
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------- driver ----
def session_id(run_dir):
    """Unique per run dir: basename + hash of the absolute path (two `night1`s never pair)."""
    h = hashlib.sha256(os.path.abspath(run_dir).encode()).hexdigest()[:8]
    return f"{os.path.basename(run_dir)}-{h}"


def load_rows(path):
    return sm.read_jsonl(path) if os.path.exists(path) else []


def append_row(path, rec):
    """Append one JSON line.  If a killed run left a partial last line, it is MOVED to
    <path>.truncated (kept, never parsed) and cut from the file first, so every row starts on a
    fresh line and the file never carries a bad line in the middle."""
    with open(path, "a+b") as f:
        f.seek(0)
        data = f.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            with open(path + ".truncated", "ab") as g:
                g.write(data[cut:] + b"\n")
            f.truncate(cut)
            print(f"warning: moved a partial last line of {path} to {path}.truncated")
        f.seek(0, os.SEEK_END)
        f.write((json.dumps(rec) + "\n").encode())


def is_done(r):
    """A key is done once it has a gated row (anchors: any converged row with an energy)."""
    good = (
        r.get("exit") == 0
        and r.get("energy") is not None
        and r.get("converged") is not False
    )
    return good and (r.get("kind") == "anchor" or bool(r.get("gated")))


def ladder_status(rows, precision):
    anchors = [r for r in rows if r.get("plan_id") == "dryice_pbe_ladder"]
    res = sm.section_ladder(
        [sm.normalise(dict(r)) for r in anchors], "dryice", precision
    )
    return (res or {}).get("status", "undecidable")


def run_sample(cell, code, tag, threads, rep, attempt, ctx):
    a = ctx["args"]
    smoke = ctx["smoke"]
    # Anchors are energy rows: their gate readings are logged but never waited on.
    ctx["gate"].enforce = cell.kind == "timed" and not smoke
    cpus = sd.default_cpus(threads)
    if code == "ferric":
        rec = sd.run_ferric(
            cell.cell,
            threads,
            ctx["binary"],
            cell.toml_dir,
            tag,
            cell.method,
            cpus,
            smoke,
            ctx["sha"],
            ctl=ctx["ctl"],
        )
    else:
        rec = sd.run_pyscf(
            cell.cell,
            threads,
            a.precision,
            cell.method,
            grid_of(cell, tag),
            cpus,
            smoke,
            True,
            cell.pyscf_extra,
            ctl=ctx["ctl"],
        )
        rec["bin_sha256"] = ctx["sha"]
    gb, ga = rec.get("gate_before") or {}, rec.get("gate_after") or {}
    gated = bool(gb.get("ok") and ga.get("ok") and gb.get("enforced")) and not smoke
    converged = rec.get("energy") is not None and rec.get("converged") is not False
    rec.update(
        plan_id=cell.id,
        tier=cell.tier,
        kind=cell.kind,
        variant=tag,
        rep=rep,
        attempt=attempt,
        session=ctx["session"],
        run_id=ctx["session"],
        invocation=ctx["invocation"],
        gated=gated,
        gate_reasons=(gb.get("reasons") or []) + (ga.get("reasons") or []),
        idle_floor=ctx[
            "gate"
        ].bars(),  # this launch's measured floor + the bars judged by
    )
    # summarize.py reads `valid`: only gated, converged, exit-0 samples may enter a statistic.
    rec["valid"] = bool(rec.get("valid") and gated and converged and rec["exit"] == 0)
    if cell.kind == "anchor":  # an energy row, never a timing
        rec["valid"] = bool(converged and rec["exit"] == 0)
    if smoke:
        rec["note"] = "smoke run: gate not enforced, timing INVALID"
    return rec


def walk(cells, ctx):
    """Yield (cell, code, tag, threads, rep) in run order: rep-major, back to back inside a rep."""
    for rep in range(max(max(c.reps.values()) for c in cells)):
        for c in cells:
            for n in c.threads():
                if rep < c.reps[n]:
                    for code, tag in sequence(c, n, rep):
                        yield c, code, tag, n, rep


def execute(cells, ctx, out):
    """Run every key not yet done; return the keys that used all attempts without a done row."""
    rows = load_rows(out)
    done = {row_key(r) for r in rows if is_done(r)}
    tries = {}
    for r in rows:
        tries[row_key(r)] = tries.get(row_key(r), 0) + 1
    for c, code, tag, n, rep in walk(cells, ctx):
        k = key_of(c, code, tag, n, rep)
        while k not in done and tries.get(k, 0) < ctx["args"].max_attempts:
            if time.time() > ctx["gate"].deadline:
                raise Deadline("--hours elapsed")
            rec = run_sample(c, code, tag, n, rep, tries.get(k, 0), ctx)
            tries[k] = tries.get(k, 0) + 1
            append_row(out, rec)
            print(
                f"{c.id} {code}{tag} N={n} rep {rep} try {rec['attempt']} "
                f"wall {rec['wall']:.1f}s E {rec.get('energy')} "
                + ("GATED" if rec["gated"] else f"discarded: {rec['gate_reasons']}"),
                flush=True,
            )
            if is_done(rec):
                done.add(k)
            if ctx["smoke"]:
                break
    if ctx["smoke"]:
        return []
    return [
        key_of(c, code, tag, n, rep)
        for c, code, tag, n, rep in walk(cells, ctx)
        if key_of(c, code, tag, n, rep) not in done
    ]


def decay_wait(threads, wall, max_load=1.5, ambient=0.3):
    """Seconds until a sample's own loadavg tail (0.92 N busy threads for `wall` s) drops below
    max_load - ambient: the raw-load gate's wait before the next sample (an ESTIMATE)."""
    tail = 0.92 * threads * (1 - math.exp(-wall / 60))
    return max(0.0, 60 * math.log(tail / (max_load - ambient))) if tail > 0 else 0.0


def estimate(cells, overhead=GATE_OVERHEAD_S, raw_load=True):
    """[(cell id, threads, samples, seconds)]: ESTIMATED wall per plan row."""
    out = []
    for c in cells:
        for n in c.threads():
            f, p = c.est[n]
            nf = len(c.tags[n])
            npy = nf if c.kind == "anchor" else 1
            samples = (nf + npy) * c.reps[n]
            pp = p * PYSCF_PREC_FACTOR
            s = c.reps[n] * (nf * f + npy * pp) + samples * overhead
            if raw_load and c.kind == "timed":  # anchors never wait
                s += c.reps[n] * (nf * decay_wait(n, f) + npy * decay_wait(n, pp))
            out.append((c.id, n, samples, s))
    return out


def select(tiers):
    want = set(tiers.split(","))
    bad = want - {c.tier for c in PLAN}
    if bad:
        raise SystemExit(
            f"unknown tiers {sorted(bad)}; known {sorted({c.tier for c in PLAN})}"
        )
    return [c for c in PLAN if c.tier in want]


def check_inputs(cells, precision):
    """Every ferric TOML parses and names an existing xyz; every PySCF argv parses and its cell
    files exist.  Returns (problems, [command strings])."""
    problems, cmds = [], []
    parser = pb.build_parser()
    for c in cells:
        for n in c.threads():
            for tag in c.tags[n]:
                toml = os.path.join(c.toml_dir, f"{c.cell}_{c.method}{tag}.toml")
                try:
                    cfg = tomllib.load(open(toml, "rb"))
                    xyz = os.path.join(ROOT, cfg["molecule"]["xyz"])
                    if not os.path.exists(xyz):
                        problems.append(f"{toml}: missing {xyz}")
                    kind = cfg["method"]["kind"]
                    if (c.method == "pbe") != (kind == "ksdft") or "cell" not in cfg:
                        problems.append(
                            f"{toml}: kind {kind} / [cell] vs method {c.method}"
                        )
                    km = sd.split_kmesh(c.cell)[1]
                    if km and cfg["cell"].get("kmesh") != [int(x) for x in km]:
                        problems.append(
                            f"{toml}: kmesh {cfg['cell'].get('kmesh')} != {km}"
                        )
                except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
                    problems.append(f"{toml}: {e!r}")
                cmds.append(
                    (
                        c.id,
                        n,
                        "ferric",
                        ["<run_dir>/bin/ferric", os.path.relpath(toml, ROOT)],
                    )
                )
            ptags = c.tags[n] if c.kind == "anchor" else ("",)
            for tag in ptags:
                argv = pyscf_argv(c, n, tag, precision)
                try:
                    parser.parse_args(argv)
                except SystemExit:
                    problems.append(f"pyscf_bench.py rejects {argv}")
                base = os.path.join(HERE, argv[0])
                for ext in (".xyz", ".lattice"):
                    if not os.path.exists(base + ext):
                        problems.append(f"missing {base}{ext}")
                cmds.append((c.id, n, "pyscf", [sd.PY, "pyscf_bench.py"] + argv))
    return problems, cmds


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--tiers",
        default=DEFAULT_TIERS,
        help=f"default {DEFAULT_TIERS}; add n1-large for the N=1 large cells",
    )
    ap.add_argument(
        "--hours", type=float, default=10.0, help="overall deadline from launch"
    )
    ap.add_argument(
        "--precision", type=float, default=1e-13, help="PySCF cell.precision"
    )
    ap.add_argument(
        "--max-attempts", type=int, default=3, help="per (cell, code, var, N, rep)"
    )
    ap.add_argument(
        "--mem", default="14G", help="cgroup MemoryMax = MemoryHigh per process"
    )
    ap.add_argument("--ferric-bin", default=os.path.join(MAINLINE, "ferric"))
    ap.add_argument("--ferric-so", default=os.path.join(MAINLINE, "libferric.so"))
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="accept a dirty / mismatched / pre-#508 binary (rows are not quotable)",
    )
    g = ap.add_argument_group("gate")
    g.add_argument(
        "--max-load", type=float, default=1.5, help="1-min loadavg (see --load-mode)"
    )
    g.add_argument(
        "--load-mode",
        choices=["raw", "adjusted"],
        default="raw",
        help="raw (default, the pre-registered gate): /proc/loadavg as read, so an N=6 sample "
        "waits for its own tail to decay; adjusted: minus own_load (a model; both are logged)",
    )
    g.add_argument(
        "--max-busy",
        type=float,
        default=None,
        help="ABSOLUTE busy bar per window (physical cores); default: idle floor + --busy-margin",
    )
    g.add_argument(
        "--max-foreign",
        type=float,
        default=None,
        help="ABSOLUTE foreign-CPU bar over the run; default: idle floor + --foreign-margin",
    )
    g.add_argument(
        "--busy-margin",
        type=float,
        default=MARGIN,
        help="the pre-registered 2%% quiet bar, ABOVE the measured idle floor",
    )
    g.add_argument("--foreign-margin", type=float, default=MARGIN, help="likewise")
    g.add_argument(
        "--floor-samples",
        type=int,
        default=FLOOR_SAMPLES,
        help="gate readings taken at startup to measure the idle floor",
    )
    g.add_argument(
        "--max-floor",
        type=float,
        default=MAX_FLOOR,
        help="refuse to start if the idle floor is above this (the box is not idle)",
    )
    g.add_argument(
        "--unreachable-after",
        type=float,
        default=UNREACHABLE_AFTER_MIN,
        help="minutes of idle-level busy-only failures before warning the bar may be unreachable",
    )
    g.add_argument("--max-python-cpu", type=float, default=0.10, help="one core = 1.0")
    g.add_argument("--max-proc-cpu", type=float, default=0.25, help="one core = 1.0")
    g.add_argument("--max-iowait", type=float, default=0.02)
    g.add_argument("--max-steal", type=float, default=0.01)
    g.add_argument("--no-steal-iowait", dest="check_steal_iowait", action="store_false")
    g.add_argument("--window", type=float, default=5.0, help="seconds per gate reading")
    g.add_argument(
        "--poll", type=float, default=60.0, help="seconds between failed readings"
    )
    m = ap.add_mutually_exclusive_group()
    m.add_argument(
        "--dry-run", action="store_true", help="validate + print, run nothing"
    )
    m.add_argument(
        "--smoke",
        action="store_true",
        help="toy cell once, gate not enforced, rows smoke=true (timing INVALID)",
    )
    return ap


def dry_run(a, cells):
    problems, cmds = check_inputs(cells, a.precision)
    pre = ["scripts/ferric-limited", f"--max={a.mem}", f"--high={a.mem}", "--"]
    print(
        f"# plan: tiers {a.tiers}; PySCF precision {a.precision}; --match-ferric-conv"
    )
    for cid, n, code, argv in cmds:
        env = (
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=%d" % n
            if code == "ferric"
            else f"OPENBLAS_NUM_THREADS={n} OMP_NUM_THREADS={n}"
        )
        print(
            f"{cid:24s} N={n} {env} "
            + shlex.join(pre + ["taskset", "-c", sd.default_cpus(n)] + argv)
        )
    print("\n# binary sources (NOT copied in a dry run)")
    for p in (a.ferric_bin, a.ferric_so):
        print(f"  {p}: {'exists' if os.path.exists(p) else 'MISSING'}")
    if os.path.exists(a.ferric_bin):
        prov, bad = provenance(a.ferric_bin, a.ferric_so, sys.executable)
        print(
            "  ferric --version:", prov["ferric_version"]["text"].replace("\n", " | ")
        )
        print("  extension:", prov["so_build"])
        for b in bad:
            print("  WOULD REFUSE:", b)
    print(f"\n# idle floor now ({a.floor_samples} readings; not enforced)")
    g = Gate(a, time.time())
    fl = g.measure_floor(refuse=False)
    print(f"  busy floor {fl['busy_floor']}  foreign floor {fl['foreign_floor']}")
    for p in fl["problems"]:
        print("  WOULD REFUSE:", p)
    if not fl["problems"]:
        print("  bars:", g.bars())
        print("  one gate reading:", g.read())
    print(
        "\n# ESTIMATED wall per plan row (README session medians; PySCF x%.1f for 1e-13; "
        "+%.0f s gate overhead per sample%s; waits for FOREIGN load and retries NOT "
        "included)"
        % (
            PYSCF_PREC_FACTOR,
            GATE_OVERHEAD_S,
            " + own-load decay (raw gate)" if a.load_mode == "raw" else "",
        )
    )
    tot = 0.0
    for cid, n, samples, s in estimate(cells, raw_load=a.load_mode == "raw"):
        tot += s
        print(f"  {cid:24s} N={n} samples {samples:3d}  ~{s / 60:6.1f} min")
    print(f"  TOTAL ~{tot / 3600:.1f} h (estimate)")
    print("\n# input problems:", "none" if not problems else "")
    for p in problems:
        print("  ", p)
    return 1 if problems else 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    cells = [SMOKE] if a.smoke else select(a.tiers)
    if a.dry_run:
        return dry_run(a, cells)
    sd.check_no_lindep_override()
    run_dir = os.path.abspath(a.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    lock = os.open(
        os.path.join(run_dir, "lock"), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644
    )
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another overnight_228.py holds {run_dir}/lock") from None
    binary, man = prepare_bin(a, run_dir, sd.PY, a.smoke)
    deadline = time.time() + a.hours * 3600
    gate = Gate(a, deadline, enforce=not a.smoke)
    # Idle floor, measured now while nothing of ours runs (we hold no heavy-slot lock).
    fl = gate.measure_floor(refuse=not a.smoke)
    if fl["problems"]:  # smoke only: proceed, the gate is not enforced anyway
        print(f"[overnight] smoke: ignoring {fl['problems']}", flush=True)
        gate.floor = fl
    record_floor(run_dir, fl, gate.bars())
    print(f"[overnight] idle floor + bars: {gate.bars()}", flush=True)
    prefix = [
        os.path.join(ROOT, "scripts", "ferric-limited"),
        f"--max={a.mem}",
        f"--high={a.mem}",
        "--",
    ]
    ctx = dict(
        args=a,
        smoke=a.smoke,
        binary=binary,
        sha=man["bin_sha256"],
        gate=gate,
        ctl=Ctl(prefix, gate),
        session=session_id(run_dir),
        invocation=sd.new_run_id() + "-" + uuid.uuid4().hex[:4],
    )
    out = os.path.join(run_dir, "smoke.jsonl" if a.smoke else "results.jsonl")
    print(
        f"[overnight] {run_dir} bin {man['bin_sha256'][:12]} "
        f"commit {man['ferric_version']['commit']} -> {out}",
        flush=True,
    )
    try:
        timed = [c for c in cells if c.tier != "pbe-gated"]
        exhausted = execute(timed, ctx, out)
        gated_pbe = [c for c in cells if c.tier == "pbe-gated"]
        if gated_pbe:
            status = ladder_status(load_rows(out), a.precision)
            print(f"[overnight] dryice PBE ladder gate: {status}", flush=True)
            if status == "pass":
                exhausted += execute(gated_pbe, ctx, out)
    except Deadline as e:
        print(
            f"[overnight] stopped at the deadline: {e}. Re-run the same command to resume."
        )
        return DEADLINE_EXIT
    if exhausted:
        print(
            f"[overnight] INCOMPLETE: {len(exhausted)} key(s) used all "
            f"{a.max_attempts} attempts without a gated sample:"
        )
        for k in exhausted:
            print("   ", k)
        return EXHAUSTED_EXIT
    print(f"[overnight] plan complete. python {HERE}/analyze_228.py {out}")
    return OK_EXIT


if __name__ == "__main__":
    sys.exit(main())
