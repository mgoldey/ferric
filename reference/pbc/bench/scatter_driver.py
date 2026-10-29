"""Interleaved ferric/PySCF timing driver for issue #228 (F P F P ... in one session).

  cp target/release/ferric "$SCRATCH/ferric-<sha>"   # private copy, never the target/ file
  python scatter_driver.py --out bench_results/timing.jsonl --cells diamond_prim dryice \
      --threads 1 --reps 5 --precision 1e-11 --bin "$SCRATCH/ferric-<sha>" [--cpus 0] \
      [--ferric-tags '' _w0.3 _w0.5] [--pyscf-match-ferric-conv]

One record per run (JSON line): code, cell, threads, rep, wall (harness wall around the whole
subprocess), load1 before/after, other_cpu_s (box CPU not used by the run's own process tree, from
/proc/stat), a validity flag (gate below), energy, iterations, stage seconds.

GATE (per run): 1-minute loadavg < 1.5 immediately before AND after the run, for N=1.
For N>1 the run's own threads push the 1-minute loadavg above 1.5, so the gate there is the
foreign-CPU fraction other_cpu_s / (wall * ncpu_logical) < 0.12 (the ambient box baseline measured here is 4-6%) (before: the same measure over a
2 s window). A run that fails either side is recorded with valid=false and re-run (<= 3 attempts).
Fresh process for every run; PySCF deletes its cderi file (private temp dir) before exit.
Ferric input: toml_ri/<cell>_<method><tag>.toml (ferric defaults, SR column rotation ON) unless
--toml-dir overrides; the binary is --bin (a private copy, see the fourth-session addendum below).
A cell named <base>_k<abc> (e.g. diamond_prim_k222) runs PySCF on <base> with --kmesh a b c.

Addendum (issue #228 second session): every run holds BOTH heavy slots (/tmp/ferric-compile.lock and
/tmp/ferric-compile.2.lock, exclusive flock, fds not inherited by the child = `flock -o`) from
BEFORE the quiet-wait until the run exits, so no other ferric agent compiles or tests while the gate
is evaluated or the run is timed.  The code order alternates per rep (F P on even reps, P F on odd).
Peak RSS per run is the child's own ru_maxrss from wait4; ferric --version and `converged` are
recorded per ferric run.

Addendum (issue #228 fourth session, harness hardening):
  --bin is REQUIRED and must be a private copy of the binary: any path (as given or resolved) with a
  `target` component, or a file with more than one hard link, is refused, because a rebuild under a
  running job SIGBUSes it.  Its sha256 is taken once at start and recorded in every row
  (`bin_sha256`); before every ferric run (outside the timed window) it is re-hashed and the driver
  EXITS if it changed.  FERRIC_LINDEP_THRESH in the environment is refused (the PySCF stopping-rule
  hook hard-codes ferric's 1e-6).
  --cpus pins BOTH codes with `taskset -c` (default N=1 -> "1" (core 0 takes most IRQs), N<=6 ->
  "0-(N-1)"; the box is an i7-6800K, physical cores 0-5, SMT siblings 6-11).  Recorded as `cpus`.
  Every invocation gets a `run_id` (start time + random suffix) on its rows; reps are paired on
  (session, run_id, rep).  An existing --out is refused unless --append (a top-up run).
  Every row carries scaling_cur_freq (kHz, all CPUs) and the coretemp hwmon temperatures (milli-C)
  at the start and the end of the timed window (`freq_*_khz`, `temp_*_mc`), and `box_state` =
  "quiet" iff start loadavg < --quiet-load (1.5) AND foreign-CPU fraction < --quiet-foreign (0.02),
  else "ambient".  `valid` is the unchanged gate above; `box_state` is a label next to it.
  --ferric-tags T1 T2 ... runs several ferric inputs (<cell>_<method><T>.toml) per rep, interleaved
  with PySCF; the sequence [ferric T1, ..., ferric Tk, pyscf] is rotated by the rep index (rep r
  starts at element r mod (k+1)); with one tag this is the old F P / P F alternation.
  --smoke: no locks, no quiet wait, one attempt, rows labelled smoke=true and valid=false.

Addendum (overnight_228.py): run_ferric / run_pyscf / run_cmd take an optional `ctl` object with
  `prefix` (argv put in front of `taskset`, e.g. scripts/ferric-limited) and `gate(phase, **ctx)`.
  With a ctl, the built-in quiet wait is REPLACED by: gate("before") (may wait for hours) WITHOUT
  the heavy-slot locks, then acquire the locks, then gate("recheck") (one short reading, decides
  ok_before); if the recheck fails the locks are released and the wait starts again, so a busy box
  never makes this driver sit on the locks.  gate("after", wall=, cpu_s=, foreign=, threads=) runs
  right after the child exits.  The recheck and after readings are stored on the row as
  `gate_before` / `gate_after`.  Without a ctl nothing changes.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import re
import resource
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from omega_sweep import cpu_jiffies, load1, wait_quiet  # noqa: E402

PY = os.environ.get("PYSCF_PY", sys.executable)
NCPU = os.cpu_count() or 12
# The machine-wide heavy-slot lock files of the cargo shim (a convention, not a temp file).
# N=1 end-of-run gate: "load" = 1-min loadavg < 1.5 after the run (first session, pre-registered);
# "foreign" = foreign-CPU fraction < 0.12 (the N>1 rule).  The start gate (loadavg < 1.5) is kept in both.
N1_GATE = os.environ.get("N1_GATE", "load")
LOCKS = (
    "/tmp/ferric-compile.lock",  # nosec B108
    "/tmp/ferric-compile.2.lock",  # nosec B108
)


def is_scf_stage(name):
    """ferric stage rows counted as SCF (Gamma: "scf J/K ...", k-mesh: "k SCF")."""
    return name.startswith(("scf ", "k SCF", "other ("))


def parse_ferric(out):
    rec = {}
    for key, pat, cast in (
        ("energy", r"energy\s+=\s+(-?[\d.]+) Hartree/cell", float),
        ("iterations", r"iterations\s+=\s+(\d+)", int),
        ("omega", r"gdf_omega\s+=\s+([\d.e+-]+) Bohr", float),
        ("nbasis", r"nbasis\s+=\s+(\d+)", int),
        ("converged", r"converged\s+=\s+(true|false)", lambda v: v == "true"),
    ):
        m = re.search(pat, out)
        rec[key] = cast(m.group(1)) if m else None
    m = re.search(r"naux (\d+), kept (\d+), dropped (\d+)", out)
    rec["aux"] = (
        dict(naux=int(m.group(1)), kept=int(m.group(2)), dropped=int(m.group(3)))
        if m
        else None
    )
    m = re.search(r"naux (\d+), dropped (\d+) \(max over q", out)
    if m and rec["aux"] is None:
        rec["aux"] = dict(naux=int(m.group(1)), dropped_max_over_q=int(m.group(2)))
    st = {}
    for ln in out.split("\n"):
        m = re.match(r"\s{4}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", ln)
        if m:
            st[m.group(1)] = float(m.group(2))
        m = re.match(r"\s{4}(total)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
        if m:
            st["total"] = float(m.group(2))
        m = re.match(
            r"\s{4}(other \(SCF linear algebra, bookkeeping\))\s+([\d.]+)\s*$", ln
        )
        if m:
            st[m.group(1)] = float(m.group(2))
    rec["stages"] = st
    scf = sum(v for k, v in st.items() if is_scf_stage(k))
    tot = st.get("total")
    rec["t_scf_s"] = scf
    rec["t_setup_s"] = None if tot is None else tot - scf
    if rec["iterations"]:
        rec["s_per_iter"] = scf / rec["iterations"]
    return rec


def check_private_binary(path):
    """Refuse a binary under a `target/` directory (as given or after resolving symlinks)."""
    given = os.path.abspath(path)
    real = os.path.realpath(path)
    for p in (given, real):
        if "target" in p.split(os.sep):
            raise SystemExit(
                f"refused: --bin {path} resolves under a target/ directory ({p}); copy the "
                "binary to a private path first (a rebuild under a running job SIGBUSes it)"
            )
    if not (os.path.isfile(real) and os.access(real, os.X_OK)):
        raise SystemExit(f"refused: --bin {path} is not an executable file")
    if os.stat(real).st_nlink > 1:
        raise SystemExit(
            f"refused: --bin {path} has {os.stat(real).st_nlink} hard links (cargo hardlinks its "
            "outputs, so a link can be rewritten by a rebuild); `cp` it to an unlinked file"
        )
    return real


def check_binary_unchanged(binary, expected_sha):
    """sha256 of the binary now; SystemExit if it differs from the session's."""
    digest = sha256_file(binary)
    if digest != expected_sha:
        raise SystemExit(
            f"refused: {binary} changed during the session (sha256 {digest} != session "
            f"{expected_sha}); its rows would mix two binaries"
        )
    return digest


def check_no_lindep_override(environ=None):
    """pyscf_bench --match-ferric-conv hard-codes ferric's lindep 1e-6; an override breaks it."""
    env = os.environ if environ is None else environ
    if "FERRIC_LINDEP_THRESH" in env:
        raise SystemExit(
            "refused: FERRIC_LINDEP_THRESH is set; ferric's orthogonaliser cut would differ from "
            "the 1e-6 the PySCF stopping-rule hook reproduces.  Unset it."
        )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def default_cpus(threads):
    """taskset list for N threads on the 6-core box: physical cores only.

    N=1 runs on core 1, not core 0 (core 0 takes most device interrupts)."""
    if threads == 1:
        return "1"
    if 1 < threads <= 6:
        return f"0-{threads - 1}"
    raise SystemExit(f"--threads {threads} > 6 physical cores: pass --cpus explicitly")


def cpu_list_count(cpus):
    n = 0
    for part in cpus.split(","):
        lo, _, hi = part.partition("-")
        n += (int(hi) - int(lo) + 1) if hi else 1
    return n


def snapshot_freq_khz(pattern="/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"):
    """{cpu index: kHz} for every CPU exposing scaling_cur_freq."""
    out = {}
    for p in glob.glob(pattern):
        m = re.search(r"/cpu(\d+)/", p)
        try:
            out[int(m.group(1))] = int(open(p).read().split()[0])
        except (OSError, ValueError, AttributeError, IndexError):
            continue
    return {str(k): out[k] for k in sorted(out)}


def snapshot_temp_mc(root="/sys/class/hwmon"):
    """{"<hwmon name>/<label or tempN>": milli-C} for the coretemp sensors (all hwmon if none)."""
    out, fallback = {}, {}
    for d in sorted(glob.glob(os.path.join(root, "hwmon*"))):
        try:
            name = open(os.path.join(d, "name")).read().strip()
        except OSError:
            continue
        for t in sorted(glob.glob(os.path.join(d, "temp*_input"))):
            stem = os.path.basename(t)[: -len("_input")]
            try:
                label = open(os.path.join(d, stem + "_label")).read().strip()
            except OSError:
                label = stem
            try:
                v = int(open(t).read().split()[0])
            except (OSError, ValueError, IndexError):
                continue
            (out if name == "coretemp" else fallback)[f"{name}/{label}"] = v
    return out or fallback


def rotated_order(seq, rep, fixed=False):
    """seq rotated so rep r starts at element r mod len(seq) (fixed: always seq)."""
    if fixed or not seq:
        return list(seq)
    k = rep % len(seq)
    return list(seq[k:]) + list(seq[:k])


def box_state(load_start, foreign_frac, quiet_load=1.5, quiet_foreign=0.02):
    return (
        "quiet"
        if load_start < quiet_load and foreign_frac < quiet_foreign
        else "ambient"
    )


QUIET = dict(load=1.5, foreign=0.02)


def new_run_id():
    """One id per driver invocation: start time + random suffix (pairs never cross runs)."""
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def check_out_path(out, append):
    if os.path.exists(out) and not append:
        raise SystemExit(
            f"refused: --out {out} exists; pass --append to add a top-up run to it (its rows get "
            "a new run_id, so reps are never paired across invocations)"
        )


def acquire_locks():
    fds = []
    t0 = time.time()
    for path in LOCKS:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        fds.append(fd)
    return fds, time.time() - t0


def release_locks(fds):
    for fd in fds:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def run_child(cmd, env):
    """Run cmd; return (returncode, stdout, stderr, own peak RSS bytes from wait4)."""
    with tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, cwd=ROOT, env=env, text=True)
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        fo.seek(0)
        fe.seek(0)
        return p.returncode, fo.read(), fe.read(), ru.ru_maxrss * 1024


def run_cmd(cmd, env, threads, cpus, smoke=False, ctl=None):
    cmd = list(getattr(ctl, "prefix", ())) + ["taskset", "-c", cpus] + list(cmd)
    if smoke:
        gb = ctl.gate("before", threads=threads) if ctl is not None else None
        return _run_cmd_locked(cmd, env, threads, None, True, ctl, gb)
    while True:
        if ctl is not None:
            ctl.gate("before", threads=threads)  # the long wait: NO locks held
        fds, lock_wait = acquire_locks()
        try:
            gb = ctl.gate("recheck", threads=threads) if ctl is not None else None
            if gb is not None and not gb["ok"] and gb.get("enforced", True):
                continue  # `finally` releases the locks; wait again without them
            return _run_cmd_locked(cmd, env, threads, lock_wait, False, ctl, gb)
        finally:
            release_locks(fds)


def _run_cmd_locked(
    cmd, env, threads, lock_wait, smoke=False, ctl=None, gate_before=None
):
    full = dict(os.environ)
    full.update(env)
    ok_before = False
    if gate_before is not None:
        ok_before = bool(gate_before["ok"])
    elif smoke:
        pass
    elif threads == 1:
        t_end = time.time() + 900
        while time.time() < t_end:
            if load1() < 1.5:
                ok_before = True
                break
            time.sleep(10)
    else:
        ok_before = wait_quiet(max_busy=0.12, window=2.0, timeout=900)
    l0 = load1()
    freq0, temp0 = snapshot_freq_khz(), snapshot_temp_mc()
    b0, t0j = cpu_jiffies()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    w0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(w0))
    rc, out, err, rss = run_child(cmd, full)
    wall = time.time() - w0
    b1, t1j = cpu_jiffies()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    freq1, temp1 = snapshot_freq_khz(), snapshot_temp_mc()
    l1 = load1()
    hz = os.sysconf("SC_CLK_TCK")
    mine = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    other = (b1 - b0) / hz - mine
    if threads == 1 and N1_GATE == "foreign":
        valid = ok_before and l0 < 1.5 and (other / (wall * NCPU) < 0.12)
    elif threads == 1:
        valid = ok_before and l0 < 1.5 and l1 < 1.5
    else:
        valid = ok_before and (other / (wall * NCPU) < 0.12)
    if smoke:
        valid = False
    foreign = other / (wall * NCPU)
    gate_after = (
        ctl.gate("after", wall=wall, cpu_s=mine, foreign=foreign, threads=threads)
        if ctl is not None
        else None
    )
    p = subprocess.CompletedProcess(cmd, rc, out, err)
    return p, dict(
        gate_before=gate_before,
        gate_after=gate_after,
        wall=wall,
        smoke=smoke,
        box_state=box_state(l0, foreign, QUIET["load"], QUIET["foreign"]),
        freq_start_khz=freq0,
        freq_end_khz=freq1,
        temp_start_mc=temp0,
        temp_end_mc=temp1,
        started_at=started,
        n1_gate=N1_GATE if threads == 1 else None,
        lock_wait_s=lock_wait,
        foreign_cpu_frac=foreign,
        child_peak_rss_bytes=rss,
        load_start=l0,
        load_end=l1,
        other_cpu_s=other,
        cpu_s=mine,
        valid=bool(valid),
    )


def ferric_version(binary):
    p = subprocess.run([binary, "--version"], capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


def run_ferric(
    cell,
    threads,
    binary,
    toml_dir,
    tag="",
    method="rhf",
    cpus="1",
    smoke=False,
    expected_sha=None,
    ctl=None,
):
    toml = os.path.join(toml_dir, f"{cell}_{method}{tag}.toml")
    version = ferric_version(binary)
    digest = (
        check_binary_unchanged(binary, expected_sha)
        if expected_sha
        else sha256_file(binary)
    )
    env = dict(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS=str(threads)
    )
    p, g = run_cmd([binary, toml], env, threads, cpus, smoke, ctl)
    rec = dict(
        code="ferric",
        cell=cell,
        method=method,
        variant=tag,
        threads=threads,
        cpus=cpus,
        bin=binary,
        bin_sha256=digest,
        ferric_version=version,
        toml=os.path.relpath(toml, ROOT),
        exit=p.returncode,
        **g,
    )
    rec.update(parse_ferric(p.stdout))
    if p.returncode:
        rec["stderr"] = p.stderr[-500:]
    return rec


def split_kmesh(cell):
    m = re.match(r"^(.*)_k(\d)(\d)(\d)$", cell)
    return (m.group(1), [m.group(2), m.group(3), m.group(4)]) if m else (cell, None)


def run_pyscf(
    cell,
    threads,
    precision,
    method="rhf",
    grid=None,
    cpus="1",
    smoke=False,
    match_ferric_conv=False,
    extra=(),
    ctl=None,
):
    env = dict(OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    base, kmesh = split_kmesh(cell)
    cmd = [
        PY,
        os.path.join(HERE, "pyscf_bench.py"),
        base,
        "--threads",
        str(threads),
        "--precision",
        str(precision),
        "--method",
        method,
    ] + (["--kmesh"] + kmesh if kmesh else [])
    if grid:
        cmd += ["--grid", str(grid[0]), str(grid[1])]
    if match_ferric_conv:
        cmd += ["--match-ferric-conv"]
    cmd += list(extra)
    p, g = run_cmd(cmd, env, threads, cpus, smoke, ctl)
    extra_txt = " ".join(extra)
    rec = dict(
        code="pyscf",
        cell=cell,
        method=method,
        variant="",
        threads=threads,
        cpus=cpus,
        exit=p.returncode,
        precision=precision,
        pyscf_args=extra_txt or None,
        **g,
    )
    try:
        inner = json.loads(p.stdout.strip().split("\n")[-1])
        rec["energy"] = inner["energy"]
        rec["iterations"] = inner["cycles"]
        rec["aux"] = inner["aux_metric"]
        rec["omega"] = inner["omega"]
        rec["t_setup_s"] = inner["t_df_build_s"] + inner["t_hcore_s"]
        rec["t_df_build_s"] = inner["t_df_build_s"]
        rec["t_hcore_s"] = inner["t_hcore_s"]
        rec["t_scf_s"] = inner["t_scf_excl_hcore_s"]
        rec["s_per_iter"] = inner["s_per_iter_median"]
        rec["t_inner_total_s"] = inner["t_total_s"]
        rec["peak_rss_bytes"] = inner["peak_rss_bytes"]
        rec["converged"] = inner["converged"]
        rec["hcore_calls"] = inner.get("hcore_calls")
        rec["pyscf_version"] = inner.get("pyscf_version")
        rec["conv_tol"] = inner.get("conv_tol")
        rec["conv_tol_grad"] = inner.get("conv_tol_grad")
        rec["conv_rule"] = inner.get("conv_rule", "pyscf")
        rec["max_cycle"] = inner.get("max_cycle")
        rec["ferric_conv_final"] = inner.get("ferric_conv_final")
        rec["grid"] = inner.get("grid")
        rec["n_grid_points"] = inner.get("n_grid_points")
    except Exception as e:  # noqa: BLE001
        rec["parse_error"] = repr(e)
        rec["stderr"] = p.stderr[-500:]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--precision", type=float, default=1e-11)
    ap.add_argument(
        "--bin",
        required=True,
        help="PRIVATE copy of the ferric binary (any path under target/ is refused)",
    )
    ap.add_argument(
        "--cpus",
        default=None,
        help="taskset -c list for both codes (default: N=1 -> 1, N<=6 -> 0-(N-1))",
    )
    ap.add_argument("--toml-dir", default=os.path.join(HERE, "toml_ri"))
    ap.add_argument(
        "--ferric-tag", default="", help="e.g. _unrot to time <cell>_rhf_unrot.toml"
    )
    ap.add_argument(
        "--ferric-tags",
        nargs="+",
        default=None,
        help="several ferric inputs per rep (overrides --ferric-tag); use '' for the base file",
    )
    ap.add_argument("--only", choices=["ferric", "pyscf", "both"], default="both")
    ap.add_argument("--method", choices=["rhf", "pbe"], default="rhf")
    ap.add_argument("--pyscf-grid", type=int, nargs=2, default=None)
    ap.add_argument(
        "--pyscf-match-ferric-conv",
        action="store_true",
        help="pass --match-ferric-conv to pyscf_bench.py (ferric's SCF stopping rule)",
    )
    ap.add_argument(
        "--pyscf-args",
        default="",
        help="extra pyscf_bench.py arguments, one shell-quoted string (e.g. '--basis sto-3g')",
    )
    ap.add_argument(
        "--fixed-order",
        action="store_true",
        help="F P every rep (the first session's order); default rotates the sequence per rep",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="allow an existing --out (a top-up run; rows get a new run_id)",
    )
    ap.add_argument("--quiet-load", type=float, default=QUIET["load"])
    ap.add_argument("--quiet-foreign", type=float, default=QUIET["foreign"])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="no locks / quiet wait, one attempt, rows smoke=true valid=false (harness check only)",
    )
    a = ap.parse_args(argv)
    check_out_path(a.out, a.append)
    check_no_lindep_override()
    binary = check_private_binary(a.bin)
    cpus = a.cpus or default_cpus(a.threads)
    if cpu_list_count(cpus) < a.threads:
        raise SystemExit(f"--cpus {cpus} has fewer CPUs than --threads {a.threads}")
    QUIET.update(load=a.quiet_load, foreign=a.quiet_foreign)
    session_sha = sha256_file(binary)
    run_id = new_run_id()
    tags = a.ferric_tags if a.ferric_tags is not None else [a.ferric_tag]
    seq = [("ferric", t) for t in tags] + [("pyscf", None)]
    seq = [x for x in seq if a.only in ("both", x[0])]
    for cell in a.cells:
        for rep in range(a.reps):
            for code, tag in rotated_order(seq, rep, a.fixed_order):
                for attempt in range(1 if a.smoke else 3):
                    if code == "ferric":
                        rec = run_ferric(
                            cell,
                            a.threads,
                            binary,
                            a.toml_dir,
                            tag,
                            a.method,
                            cpus,
                            a.smoke,
                            session_sha,
                        )
                    else:
                        rec = run_pyscf(
                            cell,
                            a.threads,
                            a.precision,
                            a.method,
                            a.pyscf_grid,
                            cpus,
                            a.smoke,
                            a.pyscf_match_ferric_conv,
                            shlex.split(a.pyscf_args),
                        )
                        rec["bin_sha256"] = session_sha
                    rec.update(
                        rep=rep,
                        attempt=attempt,
                        session=os.path.basename(a.out),
                        run_id=run_id,
                    )
                    # A run without a parsed energy cannot pass the energy gate.
                    if rec.get("energy") is None or "parse_error" in rec:
                        rec["valid"] = False
                    if rec.get("converged") is False:
                        rec["valid"] = False
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(
                        code + (tag or ""),
                        cell,
                        f"N={a.threads}",
                        f"cpus={cpus}",
                        rep,
                        attempt,
                        f"wall {rec['wall']:.1f}s",
                        f"load {rec['load_start']:.2f}->{rec['load_end']:.2f}",
                        rec["box_state"],
                        "valid" if rec["valid"] else "INVALID",
                        rec.get("energy"),
                        flush=True,
                    )
                    if rec["valid"] and rec["exit"] == 0:
                        break


if __name__ == "__main__":
    main()
