"""GFN2-xTB via the `xtb` command-line binary.

## Why a subprocess and not the `ferric-xtb` Rust binding

`crates/ferric-xtb` binds libxtb through its C ABI and is the right long-term
home, but it is (a) feature-gated OFF by default and (b) not exposed to Python
at all. More importantly, **libxtb is not thread-safe** — it carries
process-global state, and 3 of 8 tests fail under a parallel harness on values
that are exact when serialized (see the `xtb-gfortran-o3-miscompile` note).
Parallelism therefore has to be across PROCESSES, which is exactly what a
subprocess-per-conformer gives for free. Nothing here needs the FFI.

## The environment trap, encoded once

meson installs libxtb into the **multiarch** subdir
`~/.local/lib/x86_64-linux-gnu`, NOT `~/.local/lib`, so the `~/.local/lib`
convention used for libint2 elsewhere in ferric is not sufficient; `xtb` fails
with `libxtb.so.6: cannot open shared object file`. `XTBPATH` must also point at
the GFN parameter files. `_xtb_env()` below sets both.

## The correctness caveat that matters most

gfortran 13.3 **miscompiles** xtb 6.7.1's GFN1/GFN2 analytic gradient at `-O3`:
forces come out ~20x wrong while **energies stay byte-identical**. So a
single-point energy cannot detect a bad build — only a geometry optimization
can. `verify_xtb_build()` performs that discriminating check (a deliberately
distorted water must relax onto the GFN2 reference), and `relax()` refuses to
run until it has passed. This is not paranoia: the installed `~/.local` copy's
hash matches none of the known-good build artifacts, so the build provenance is
genuinely unknown and has to be established empirically.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

HARTREE_TO_KCAL_MOL = 627.5094740631

# Cached outcome of verify_xtb_build(), so a batch pays the check once.
_BUILD_VERIFIED: bool | None = None
_BUILD_ERROR: str | None = None


def _xtb_env() -> dict[str, str]:
    """Environment `xtb` needs on this box. See module docstring."""
    prefix = Path(os.environ.get("XTB_PREFIX", Path.home() / ".local"))
    env = dict(os.environ)
    libdirs = [str(prefix / "lib" / "x86_64-linux-gnu"), str(prefix / "lib")]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([d for d in libdirs if d] + ([existing] if existing else []))
    env.setdefault("XTBPATH", str(prefix / "share" / "xtb"))
    # libxtb is not thread-safe and we parallelize across processes; a threaded
    # BLAS underneath would also fight the process-level parallelism.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def xtb_available() -> bool:
    return shutil.which("xtb") is not None


@dataclass
class XtbRun:
    """One xtb invocation's outcome.

    `energy` is `None` on any failure — never 0.0, for the same reason
    `tools.tox` never fabricates a zero: a 0 Hartree energy would sort as
    infinitely unstable (or stable, depending on the comparison) and silently
    corrupt a ranking.
    """
    energy: float | None                 # Hartree
    coords_angstrom: list[tuple[float, float, float]] | None
    symbols: list[str]
    converged: bool
    error: str | None = None
    stdout_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.energy is not None


@dataclass
class AnnealRun:
    """Outcome of an MD/annealing run.

    Separate from `XtbRun` on purpose: a trajectory has no single energy, so
    reusing `XtbRun` would leave `ok` False (it requires `energy is not None`)
    while `error` was None -- a contradictory state that the first version of
    `anneal` actually produced.
    """
    n_frames: int
    ok: bool
    error: str | None = None
    stdout_tail: str = ""


def _write_xyz(path: Path, symbols, coords_angstrom, comment="") -> None:
    lines = [str(len(symbols)), comment]
    for s, (x, y, z) in zip(symbols, coords_angstrom):
        lines.append(f"{s:<3s} {x:14.8f} {y:14.8f} {z:14.8f}")
    path.write_text("\n".join(lines) + "\n")


def _read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    symbols, coords = [], []
    for line in lines[2:2 + n]:
        p = line.split()
        symbols.append(p[0])
        coords.append((float(p[1]), float(p[2]), float(p[3])))
    return symbols, coords


_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")


def _run_xtb(
    symbols,
    coords_angstrom,
    args: list[str],
    charge: int = 0,
    uhf: int = 0,
    timeout: float = 1800.0,
    point_charges: list[tuple[float, float, float, float]] | None = None,
) -> tuple[XtbRun, Path | None, str]:
    """Core invocation. Returns (result, workdir-relative optimized xyz, workdir).

    The work directory is a fresh temp dir per call: xtb writes a pile of
    fixed-name files (`xtbopt.xyz`, `charges`, `wbo`, ...) into the CWD, so two
    concurrent runs in one directory would overwrite each other's output. This
    is the process-isolation that makes parallelism safe.
    """
    if not xtb_available():
        return (
            XtbRun(None, None, list(symbols), False,
                   error="the `xtb` binary is not on PATH"),
            None, "",
        )

    workdir = tempfile.mkdtemp(prefix="ferric-xtb-")
    wd = Path(workdir)
    inp = wd / "mol.xyz"
    _write_xyz(inp, symbols, coords_angstrom)

    cmd = ["xtb", inp.name, "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf)]
    if point_charges:
        # xtb reads external point charges from a file named `pcharge` in the
        # working directory (one "q x y z" row per charge, coordinates in BOHR),
        # activated by `--gfn2 ... ` with the file present. Written explicitly
        # here rather than via `--input`, which is xtb's DETAILED-INPUT flag and
        # takes a completely different (namelist) format -- passing a pcharge
        # file to `--input` makes xtb ignore the charges silently, which would
        # turn every "in field" number into a vacuum number without any error.
        rows = [str(len(point_charges))]
        for q, x, y, z in point_charges:
            rows.append(f"{q:18.10f} {x:18.10f} {y:18.10f} {z:18.10f}")
        (wd / "pcharge").write_text("\n".join(rows) + "\n")
    cmd += args

    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=_xtb_env(), capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            XtbRun(None, None, list(symbols), False,
                   error=f"xtb timed out after {timeout:.0f}s"),
            None, workdir,
        )

    out = proc.stdout
    tail = "\n".join(out.splitlines()[-25:])
    m = _ENERGY_RE.search(out)
    if proc.returncode != 0 or m is None:
        return (
            XtbRun(None, None, list(symbols), False,
                   error=(f"xtb exited {proc.returncode} without a parseable "
                          f"TOTAL ENERGY; stderr: {proc.stderr.strip()[:300]}"),
                   stdout_tail=tail),
            None, workdir,
        )

    energy = float(m.group(1))
    converged = "GEOMETRY OPTIMIZATION CONVERGED" in out or "--opt" not in args
    opt_xyz = wd / "xtbopt.xyz"
    return (
        XtbRun(energy, None, list(symbols), converged, stdout_tail=tail),
        opt_xyz if opt_xyz.exists() else None,
        workdir,
    )


def singlepoint(
    symbols,
    coords_angstrom,
    charge: int = 0,
    uhf: int = 0,
    point_charges=None,
    timeout: float = 1800.0,
) -> XtbRun:
    """GFN2-xTB single-point energy. Safe on any build (energies are unaffected
    by the `-O3` gradient miscompile)."""
    run, _, workdir = _run_xtb(
        symbols, coords_angstrom, [], charge, uhf, timeout, point_charges
    )
    run.coords_angstrom = [tuple(c) for c in coords_angstrom]
    shutil.rmtree(workdir, ignore_errors=True)
    return run


def relax(
    symbols,
    coords_angstrom,
    charge: int = 0,
    uhf: int = 0,
    point_charges=None,
    timeout: float = 3600.0,
    skip_build_check: bool = False,
) -> XtbRun:
    """GFN2-xTB geometry optimization.

    Refuses to run unless `verify_xtb_build()` has passed, because a `-O3`
    miscompiled build optimizes *uphill* while reporting correct energies —
    a silent wrong answer, not a crash. Pass `skip_build_check=True` only if
    the build was verified out-of-band in this same process.
    """
    if not skip_build_check:
        ok, err = verify_xtb_build()
        if not ok:
            return XtbRun(
                None, None, list(symbols), False,
                error=f"refusing to optimize: xtb build check failed -- {err}",
            )

    run, opt_xyz, workdir = _run_xtb(
        symbols, coords_angstrom, ["--opt"], charge, uhf, timeout, point_charges
    )
    if run.ok and opt_xyz is not None:
        try:
            _, coords = _read_xyz(opt_xyz)
            run.coords_angstrom = coords
        except Exception as e:  # noqa: BLE001
            run.error = f"could not read xtbopt.xyz: {e}"
    shutil.rmtree(workdir, ignore_errors=True)
    return run


def anneal(
    symbols,
    coords_angstrom,
    charge: int = 0,
    uhf: int = 0,
    point_charges=None,
    temperature_k: float = 500.0,
    picoseconds: float = 5.0,
    dump_every_fs: float = 250.0,
    timestep_fs: float = 1.0,
    timeout: float = 7200.0,
    skip_build_check: bool = False,
) -> "tuple[list[list[tuple[float, float, float]]], XtbRun]":
    """GFN2 molecular dynamics, returning every dumped frame as a candidate pose.

    ## What this is for

    Conformer *generation* (ETKDG) samples free-solution torsional space. For a
    ligand with many rotatable bonds the bound conformer is one receptor-selected
    point in that space, and unbiased generation misses it -- measured on
    danuglipron (9 rotatable bonds): the best of 20 generated conformers was
    2.23 A from the bound pose against a 2.0 A success bar, and geometry
    OPTIMIZATION only improved that by 0.04-0.16 A because relaxation settles
    bonds and angles, not torsions.

    MD at elevated temperature crosses torsional barriers, so it *can* reach a
    different basin. Run with `point_charges` (the pocket field), the receptor
    biases which torsions are populated -- which is the whole point. Verified
    2026-08-29 that xtb applies the `pcharge` field during MD, not just to
    single points: water at frame 0 gives -5.046870 Ha in a test field vs
    -5.070374 Ha in vacuum, a 14.7 kcal/mol shift.

    ## Returns

    `(frames, run)` -- every dumped geometry, plus an `AnnealRun` record. Frames are
    RAW MD snapshots: hot, unrelaxed, and NOT energy-ordered. The caller is
    expected to relax and score them; scoring a raw frame directly compares
    a thermally excited geometry against a relaxed one.

    An empty frame list with `run.ok == False` means the MD did not run;
    `run.error` says why.

    ## Cost

    5 ps at 1 fs on a 70-atom molecule is 5000 GFN2 gradient calls -- tens of
    minutes. This is a *pose search*, priced accordingly; it is not a screening
    step.
    """
    if not skip_build_check:
        ok, err = verify_xtb_build()
        if not ok:
            return [], AnnealRun(
                0, False, f"refusing to run MD: xtb build check failed -- {err}")

    if not xtb_available():
        return [], AnnealRun(0, False, "the `xtb` binary is not on PATH")

    workdir = tempfile.mkdtemp(prefix="ferric-xtb-md-")
    wd = Path(workdir)
    _write_xyz(wd / "mol.xyz", symbols, coords_angstrom)
    (wd / "md.inp").write_text(
        "$md\n"
        f"   temp={temperature_k}\n"
        f"   time={picoseconds}\n"
        f"   step={timestep_fs}\n"
        f"   dump={dump_every_fs}\n"
        "   shake=2\n"     # constrain X-H, so a 1 fs step is stable
        "   hmass=4\n"     # hydrogen mass repartitioning, same reason
        "$end\n"
    )
    if point_charges:
        rows = [str(len(point_charges))]
        for q, x, y, z in point_charges:
            rows.append(f"{q:18.10f} {x:18.10f} {y:18.10f} {z:18.10f}")
        (wd / "pcharge").write_text("\n".join(rows) + "\n")

    cmd = ["xtb", "mol.xyz", "--gfn", "2", "--chrg", str(charge), "--uhf",
           str(uhf), "--md", "--input", "md.inp"]
    try:
        proc = subprocess.run(cmd, cwd=workdir, env=_xtb_env(),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        shutil.rmtree(workdir, ignore_errors=True)
        return [], AnnealRun(0, False, f"xtb MD timed out after {timeout:.0f}s")

    trj = wd / "xtb.trj"
    if proc.returncode != 0 or not trj.exists():
        tail = "\n".join(proc.stdout.splitlines()[-20:])
        shutil.rmtree(workdir, ignore_errors=True)
        return [], AnnealRun(
            0, False,
            f"xtb MD exited {proc.returncode} with no trajectory; "
            f"stderr: {proc.stderr.strip()[:300]}",
            stdout_tail=tail,
        )

    frames = _read_trajectory(trj, len(symbols))
    shutil.rmtree(workdir, ignore_errors=True)
    if not frames:
        return [], AnnealRun(0, False, "xtb wrote a trajectory but no frame parsed")
    return frames, AnnealRun(len(frames), True, None)


def _read_trajectory(path: Path, natoms: int):
    """Parse a multi-frame xyz trajectory into a list of coordinate arrays."""
    lines = path.read_text().splitlines()
    frames, i = [], 0
    while i + 1 + natoms <= len(lines):
        try:
            n = int(lines[i].split()[0])
        except (ValueError, IndexError):
            break
        if n != natoms:
            break
        coords = []
        for row in lines[i + 2:i + 2 + n]:
            p = row.split()
            if len(p) < 4:
                break
            coords.append((float(p[1]), float(p[2]), float(p[3])))
        if len(coords) == n:
            frames.append(coords)
        i += 2 + n
    return frames


# ── the build-provenance check ──

# A deliberately distorted water: O-H stretched to 1.20 A and HOH compressed to
# 90 deg. A correct GFN2 build relaxes this to O-H 0.9589 A / HOH 107.16 deg.
_DISTORTED_WATER_SYMBOLS = ["O", "H", "H"]
_DISTORTED_WATER = [
    (0.0, 0.0, 0.0),
    (1.20, 0.0, 0.0),
    (0.0, 1.20, 0.0),
]
_GFN2_WATER_OH = 0.9589
_GFN2_WATER_HOH = 107.16


def verify_xtb_build(force: bool = False) -> tuple[bool, str | None]:
    """Empirically establish that this xtb build's GRADIENTS are usable.

    Optimizes a distorted water and checks it lands on the GFN2 reference
    geometry. This is the discriminating test: under the gfortran `-O3`
    miscompile, energies are byte-identical to a good build while forces are
    ~20x wrong, so a single-point comparison would pass on a broken library.

    Result is cached for the process (`force=True` re-runs it).
    """
    global _BUILD_VERIFIED, _BUILD_ERROR
    if _BUILD_VERIFIED is not None and not force:
        return _BUILD_VERIFIED, _BUILD_ERROR

    if not xtb_available():
        _BUILD_VERIFIED, _BUILD_ERROR = False, "the `xtb` binary is not on PATH"
        return _BUILD_VERIFIED, _BUILD_ERROR

    run, opt_xyz, workdir = _run_xtb(
        _DISTORTED_WATER_SYMBOLS, _DISTORTED_WATER, ["--opt"], timeout=300.0
    )
    try:
        if not run.ok or opt_xyz is None:
            _BUILD_VERIFIED = False
            _BUILD_ERROR = f"water optimization did not run: {run.error}"
            return _BUILD_VERIFIED, _BUILD_ERROR
        _, coords = _read_xyz(opt_xyz)
        o, h1, h2 = coords[0], coords[1], coords[2]
        r1, r2 = math.dist(o, h1), math.dist(o, h2)
        v1 = [h1[i] - o[i] for i in range(3)]
        v2 = [h2[i] - o[i] for i in range(3)]
        cos = sum(a * b for a, b in zip(v1, v2)) / (r1 * r2)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))

        bad = []
        if abs(r1 - _GFN2_WATER_OH) > 0.02 or abs(r2 - _GFN2_WATER_OH) > 0.02:
            bad.append(f"O-H {r1:.4f}/{r2:.4f} A vs GFN2 {_GFN2_WATER_OH} A")
        if abs(angle - _GFN2_WATER_HOH) > 2.0:
            bad.append(f"HOH {angle:.2f} deg vs GFN2 {_GFN2_WATER_HOH} deg")

        if bad:
            _BUILD_VERIFIED = False
            _BUILD_ERROR = (
                "xtb gradients look WRONG: " + "; ".join(bad) + ". This is the "
                "signature of the gfortran -O3 miscompile of xtb 6.7.1 "
                "(energies stay correct, forces go ~20x wrong). Rebuild libxtb "
                "with `-Doptimization=2`; do NOT trust any optimized geometry "
                "or gradient from this build."
            )
        else:
            _BUILD_VERIFIED, _BUILD_ERROR = True, None
        return _BUILD_VERIFIED, _BUILD_ERROR
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
