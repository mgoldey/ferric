"""S66 equilibrium benchmark of MP2-V and SCS-MP2(2terfc) (mgoldey/ferric#278).

Computes S66 (scale 1.00) interaction energies for three methods, each both
counterpoise-corrected (CP) and uncorrected:

  mp2        plain RI-MP2 (control; must overbind pi-stacks)
  scs2terfc  SCS-MP2(2terfc), published aTZ parameters
  mp2v       MP2-V(terfc, aTZ), published parameters (post-HF VV10)

at aug-cc-pVTZ / aug-cc-pvtz-rifit, frozen core = 1 per first-row heavy atom.
All three come from ONE SCF per system via `ferric.run_mp2_v_and_scs_2terfc`
(the shared-SCF entry point; bit-identical to the per-method bindings, pinned by
crates/ferric-mp2/tests/terfc_pair_shared_scf.rs and the binding's pytest).
The reference is the S66x8 "1.00" CCSD(T)/CBS CP interaction energy
(testdata/reference/c9_refs/s66x8_ccsdt_cbs.json, BEGDB).

The published fits are aTZ, frozen core, NO counterpoise. The no-CP statistics
are therefore the FIT-REPRODUCTION check and the CP statistics the PHYSICS
check (CP removes BSSE the fit absorbed). Neither column is ranked alone.

Fragment split: the first `a_size` atoms are fragment A, with `a_size` parsed
from S66_FRAGA in benchmarks/harness/examples/c9_driver.rs and VERIFIED against
the independent monomer nuclear repulsion energies in benchmarks/grid/S66.py
(Psi4 database; never a connectivity heuristic). The S66x8 1.00 dimer nuclear
repulsion differs from the original S66 by 1e-4..1e-2 relative, so the
intermolecular geometry (and the reference table) is the BEGDB S66x8 one.

One JOB = one (dimer, fragment, ghosted) system: AB, A, B, A-in-AB basis
(ghosted) and B-in-AB basis (ghosted). Each job runs in a child process, so a
crash (SIGBUS, cgroup OOM kill) loses one job, not the scheduler.

Subcommands (run from the repo root):
    dry-run      parse all 66 geometries + fragment splits, no ferric, no SCF
    estimate     cost model from the #474 pilot timings (ESTIMATES, not measurements)
    freeze-so    copy a built libferric.so to a private package dir for a long run
    run          unattended, resumable, load-gated, cheapest-first production run
    report       RMSD/MAD/MSE vs CCSD(T)/CBS, with and without CP, explicit n
    selftest / check-split / unit   (from #474) anchors and split checks

Launch (one line; see `run --help`):
    scripts/ferric-limited --max=12G --high=10G -- python \
        scripts/validation/s66_terfc_benchmark.py run --so-dir ~/s66-ferric-so \
        --out scripts/queue/out/s66_all66_atz.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S66_DIR = ROOT / "testdata/molecules/c9_systems/s66x8"
REF_JSON = ROOT / "testdata/reference/c9_refs/s66x8_ccsdt_cbs.json"
C9_DRIVER = ROOT / "benchmarks/harness/examples/c9_driver.rs"
S66_PY = ROOT / "benchmarks/grid/S66.py"
TERF_DIR = ROOT / "terf-tables"
BUNDLED = ROOT / "crates/ferric-core/src/basis/bundled"
PILOT_JSONL = ROOT / "testdata/reference/validation/s66_terfc/s66_atz.jsonl"

HARTREE_TO_KCAL = 627.5094740631  # CODATA 2014 as used by Psi4/BEGDB tables
ANGSTROM_TO_BOHR = 1.0 / 0.52917721092  # ferric_core::units
BASIS = "aug-cc-pvtz"
AUX = "aug-cc-pvtz-rifit"
SMOKE_BASIS, SMOKE_AUX = "cc-pvdz", "cc-pvdz-ri"
METHODS = ("mp2", "scs2terfc", "mp2v")
CALCS = ("AB", "A_in_AB", "B_in_AB", "A", "B")
# calc name -> (fragment, ghosted): the JSONL key is (dimer, fragment, ghosted).
CALC_KEY = {
    "AB": ("AB", False),
    "A": ("A", False),
    "B": ("B", False),
    "A_in_AB": ("A", True),
    "B_in_AB": ("B", True),
}
KEY_CALC = {v: k for k, v in CALC_KEY.items()}
SCHEMA = "s66_terfc_pair/1"
# Z of first-row heavy atoms (frozen core = 1 each). S66 has none beyond Ne.
Z = {"H": 1, "C": 6, "N": 7, "O": 8}

# Published S66 statistics, aug-cc-pVTZ, NO counterpoise, frozen core, full set
# (Goldey/Belzunces/Head-Gordon, JCTC 11, 4159 (2015)). Quoted from the values
# transcribed in crates/ferric-mp2/src/att_vv10.rs (Table 1 / Table 2) and
# crates/ferric-mp2/src/scs.rs; MP2-V's 0.199 is the self-consistent VV10
# variant, 0.202 the post-HF one (the variant ferric implements).
PUBLISHED = {
    "scs2terfc": {"rmsd": 0.228, "source": "JCTC 11, 4159 (2015) Table 2"},
    "mp2v": {
        "rmsd": 0.202,
        "rmsd_self_consistent": 0.199,
        "mse_self_consistent": -0.019,
        "mue_self_consistent": 0.167,
        "source": "JCTC 11, 4159 (2015) Table 1 (r0 1.00 A, b 11.0) and text (post-HF 0.202)",
    },
    "mp2": {"rmsd": 1.533, "source": "JCTC 11, 4159 (2015) Table 2, MP2/aTZ"},
}
GEOMETRY_CAVEAT = (
    "published RMSDs (0.228 / 0.202 / 1.533) are on the ORIGINAL S66 geometries; this run "
    "uses the S66x8 1.00 geometries, whose nuclear repulsion differs by up to 1e-2 relative, "
    "against the S66x8 1.00 CCSD(T)/CBS reference - NOT like-for-like"
)

# ── geometry, fragments, basis sizes (pure Python; no ferric) ──────────────


def s66_fraga() -> list[int]:
    text = C9_DRIVER.read_text()
    m = re.search(r"const S66_FRAGA: \[usize; 67\] = \[(.*?)\];", text, re.S)
    nums = [
        int(x) for x in re.findall(r"\b(\d+)\b", re.sub(r"//[^\n]*", "", m.group(1)))
    ]
    assert len(nums) == 67, len(nums)
    return nums


def s66_nre() -> dict[int, tuple[float, float, float]]:
    """(dimer, monoA, monoB) nuclear repulsion from the Psi4 S66 database."""
    out: dict[int, dict] = {}
    pat = re.compile(
        r'DATA\["NUCLEAR REPULSION ENERGY"\]\["S66-(\d+)-(dimer|monoA-unCP|monoB-unCP)"\] = ([\d.]+)'
    )
    for m in pat.finditer(S66_PY.read_text()):
        out.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(3))
    return {k: (v["dimer"], v["monoA-unCP"], v["monoB-unCP"]) for k, v in out.items()}


def s66_files() -> dict[int, Path]:
    d = {}
    for p in sorted(S66_DIR.glob("s66_*_1.00.xyz")):
        d[int(p.name.split("_")[1])] = p
    assert len(d) == 66, len(d)
    return d


def read_atoms(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for ln in lines[2 : 2 + n]:
        s, x, y, z = ln.split()[:4]
        atoms.append((s, float(x), float(y), float(z)))
    assert len(atoms) == n
    return atoms


def xyz_string(atoms, real: list[bool]) -> str:
    rows = [
        f"{('' if r else '@') + s} {x:.9f} {y:.9f} {z:.9f}"
        for (s, x, y, z), r in zip(atoms, real)
    ]
    return f"{len(rows)}\n\n" + "\n".join(rows) + "\n"


def calc_atoms(atoms, a_size: int, calc: str):
    """(selected atoms, real-flags) for one calculation."""
    n = len(atoms)
    ia = set(range(a_size))
    if calc == "AB":
        sel, real = list(range(n)), [True] * n
    elif calc == "A":
        sel = sorted(ia)
        real = [True] * len(sel)
    elif calc == "B":
        sel = [i for i in range(n) if i not in ia]
        real = [True] * len(sel)
    elif calc == "A_in_AB":
        sel, real = list(range(n)), [i in ia for i in range(n)]
    elif calc == "B_in_AB":
        sel, real = list(range(n)), [i not in ia for i in range(n)]
    else:
        raise ValueError(calc)
    return [atoms[i] for i in sel], real


def build_calc(atoms, a_size: int, calc: str):
    sub, real = calc_atoms(atoms, a_size, calc)
    nfc = sum(1 for (s, *_), r in zip(sub, real) if r and Z[s] > 2)
    return xyz_string(sub, real), nfc


def nre_py(atoms) -> float:
    """Nuclear repulsion (Hartree) of real atoms, coordinates in Angstrom."""
    e = 0.0
    for i, (si, *ri) in enumerate(atoms):
        for sj, *rj in atoms[:i]:
            r = math.dist(ri, rj) * ANGSTROM_TO_BOHR
            e += Z[si] * Z[sj] / r
    return e


_NBF_CACHE: dict[str, dict[str, int]] = {}
_SYM = {"H": "1", "C": "6", "N": "7", "O": "8"}


def nbf_per_element(basis: str) -> dict[str, int]:
    """Spherical basis-function count per element, read from ferric's bundled JSON."""
    if basis not in _NBF_CACHE:
        data = json.loads((BUNDLED / f"{basis}.json").read_text())["elements"]
        out = {}
        for sym, z in _SYM.items():
            n = 0
            for sh in data[z]["electron_shells"]:
                am, rows = sh["angular_momentum"], sh["coefficients"]
                ls = am if len(am) > 1 else am * len(rows)
                n += sum(2 * ll + 1 for ll in ls)
            out[sym] = n
        _NBF_CACHE[basis] = out
    return _NBF_CACHE[basis]


def job_nbf(atoms, a_size: int, calc: str, basis: str) -> int:
    """Basis functions of one job; ghost atoms carry their basis (that is CP)."""
    per = nbf_per_element(basis)
    sub, _ = calc_atoms(atoms, a_size, calc)
    return sum(per[s] for s, *_ in sub)


def all_jobs(dimers: list[int], basis: str) -> list[dict]:
    files, fraga = s66_files(), s66_fraga()
    jobs = []
    for k in dimers:
        atoms = read_atoms(files[k])
        for calc in CALCS:
            frag, ghosted = CALC_KEY[calc]
            jobs.append(
                {
                    "dimer": k,
                    "calc": calc,
                    "fragment": frag,
                    "ghosted": ghosted,
                    "nbf": job_nbf(atoms, fraga[k], calc, basis),
                }
            )
    return jobs


# ── #474 checks (kept) ─────────────────────────────────────────────────────


def check_split(dimers: list[int]) -> bool:
    import ferric

    fraga, nre, files = s66_fraga(), s66_nre(), s66_files()
    ok = True
    for k in dimers:
        atoms = read_atoms(files[k])
        a = fraga[k]
        e = {}
        for c in ("AB", "A", "B"):
            xyz, _ = build_calc(atoms, a, c)
            e[c] = ferric.Molecule.from_xyz_string(xyz).nuclear_repulsion()
        ref = nre[k]
        rel = [abs(e[c] - r) / r for c, r in zip(("AB", "A", "B"), ref)]
        ne = [
            ferric.Molecule.from_xyz_string(build_calc(atoms, a, c)[0]).nelec() % 2
            for c in ("A", "B")
        ]
        good = max(rel[1:]) < 2e-8 and ne == [0, 0]
        ok &= good
        print(
            f"S66-{k:02d} a_size={a:2d} NRE rel dev dimer/A/B = {rel[0]:.1e} {rel[1]:.1e} {rel[2]:.1e}  "
            f"even-electron={ne == [0, 0]} {'OK' if good else 'BAD'}"
        )
    return ok


def git_info() -> dict:
    def g(*a):
        return subprocess.run(
            ["git", *a], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {
        "head": g("rev-parse", "HEAD"),
        "dirty_script": bool(
            g(
                "status",
                "--porcelain",
                "--",
                "scripts/validation/s66_terfc_benchmark.py",
            )
        ),
    }


# ── dry-run ────────────────────────────────────────────────────────────────


def cmd_dry_run(a) -> None:
    """Parse all geometries and fragment splits; verify monomer NREs; no ferric."""
    fraga, nre, files = s66_fraga(), s66_nre(), s66_files()
    ref = json.loads(REF_JSON.read_text())
    dimers = parse_dimers(a.dimers)
    bad = 0
    for k in dimers:
        atoms = read_atoms(files[k])
        a_size = fraga[k]
        assert all(s in Z for s, *_ in atoms), f"S66-{k}: element outside H/C/N/O"
        mono = [calc_atoms(atoms, a_size, c)[0] for c in ("A", "B")]
        rel = [
            abs(nre_py(m) - r) / r for m, r in zip(mono, nre[k][1:])
        ]  # monomers only
        even = [sum(Z[s] for s, *_ in m) % 2 == 0 for m in mono]
        nfc = {c: build_calc(atoms, a_size, c)[1] for c in CALCS}
        fc_ok = (
            nfc["AB"] == nfc["A"] + nfc["B"]
            and nfc["A_in_AB"] == nfc["A"]
            and nfc["B_in_AB"] == nfc["B"]
        )
        has_ref = files[k].stem in ref
        nbf = {c: job_nbf(atoms, a_size, c, a.basis) for c in CALCS}
        good = (
            max(rel) < 2e-8
            and all(even)
            and fc_ok
            and has_ref
            and 0 < a_size < len(atoms)
        )
        bad += not good
        print(
            f"S66-{k:02d} {files[k].stem[7:-5]:22s} natoms={len(atoms):2d} a_size={a_size:2d} "
            f"monoNRE rel={max(rel):.1e} even={all(even)} fc(AB/A/B)={nfc['AB']}/{nfc['A']}/{nfc['B']} "
            f"nbf AB={nbf['AB']} A={nbf['A']} B={nbf['B']} ref={ref.get(files[k].stem, float('nan')):+.2f} "
            f"{'OK' if good else 'BAD'}"
        )
    print(
        f"DRY-RUN {len(dimers)} dimers x {len(CALCS)} jobs = {len(dimers) * len(CALCS)} jobs; "
        f"{'all OK' if not bad else f'{bad} BAD'} (no ferric import, no SCF)"
    )
    sys.exit(1 if bad else 0)


# ── JSONL (crash-tolerant) ─────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file. A malformed LAST line with no trailing newline is a
    torn append (crash mid-write): logged and skipped. A malformed line anywhere
    else is corruption: ValueError naming the line."""
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:  # no newline after it: a torn final append
                print(
                    f"WARNING: {path}:{i + 1}: skipping truncated last line ({len(ln)} chars)",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            raise ValueError(f"{path}:{i + 1}: malformed JSONL line: {e}") from e
    return out


def append_jsonl(path: Path, rec: dict) -> None:
    """Append one record on a fresh line. A torn (unterminated) tail from an
    earlier crash is moved to `<path>.torn` and cut, so it can never end up in
    the middle of the file where `read_jsonl` would reject it."""
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as f:
            data = f.read()
            if not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1
                with path.with_suffix(path.suffix + ".torn").open("ab") as t:
                    t.write(data[cut:] + b"\n")
                f.truncate(cut)
                print(f"WARNING: moved a torn tail of {path} to .torn", file=sys.stderr)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ── cost model / estimate ──────────────────────────────────────────────────


def fit_power(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit of log y = log c + p log x. Returns (c, p)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    p = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    return math.exp(my - p * mx), p


def pilot_cost_model(path: Path, basis: str) -> dict:
    """Per-method power law t = c * nbf^p from the #474 pilot (one SCF per method)."""
    files, fraga = s66_files(), s66_fraga()
    pts: dict[str, list[tuple[int, float]]] = {m: [] for m in METHODS}
    for r in read_jsonl(path):
        if r.get("basis") != basis or "method" not in r:
            continue
        nbf = job_nbf(
            read_atoms(files[r["dimer"]]), fraga[r["dimer"]], r["calc"], basis
        )
        pts[r["method"]].append((nbf, r["wall_s"]))
    model = {}
    for m, v in pts.items():
        c, p = fit_power([x for x, _ in v], [y for _, y in v])
        model[m] = {
            "c": c,
            "p": p,
            "n": len(v),
            "nbf_range": (min(x for x, _ in v), max(x for x, _ in v)),
        }
    return model


def shared_cost_model(path: Path, basis: str) -> dict | None:
    """Power law for the shared job (whole child wall) from production records, if any."""
    if not path.exists():
        return None
    xs, ys, scf = [], [], []
    for r in read_jsonl(path):
        if r.get("schema") == SCHEMA and r.get("basis") == basis and not r.get("smoke"):
            xs.append(r["nbf"])
            ys.append(r["wall_s"])
            scf.append(r["timings"]["scf_s"] / r["wall_s"])
    if len(xs) < 3 or len(set(xs)) < 2:
        return None
    c, p = fit_power(xs, ys)
    return {
        "c": c,
        "p": p,
        "n": len(xs),
        "nbf_range": (min(xs), max(xs)),
        "scf_fraction_mean": sum(scf) / len(scf),
    }


def cmd_estimate(a) -> None:
    model = pilot_cost_model(Path(a.pilot), a.basis)
    shared = shared_cost_model(Path(a.out), a.basis) if a.out else None
    jobs = all_jobs(parse_dimers(a.dimers), a.basis)
    print(
        "ESTIMATES ONLY - extrapolated from timings, never measured for these dimers."
    )
    print(
        f"Pilot: {a.pilot} (#474: 5 dimers, loaded box, load 13-22, RAYON=3; per-method SCF)."
    )
    for m, v in model.items():
        print(
            f"  {m:9s} t = {v['c']:.3e} * nbf^{v['p']:.2f} s   (n={v['n']}, fitted nbf "
            f"{v['nbf_range'][0]}-{v['nbf_range'][1]}; beyond that range is extrapolation)"
        )
    print(
        "Separate = sum of the three per-method fits (three SCFs per job).\n"
        "Shared = one SCF: saves 2 SCFs per job. The pilot cannot separate SCF from\n"
        "correlation time, so the shared cost is a RANGE: upper = separate (SCF free),\n"
        "lower = separate - 2*min(t_mp2, t_scs, t_mp2v) (the SCF cannot exceed the\n"
        "cheapest method's whole record). The ~3x pre-build guess needs the SCF to\n"
        "dominate ALL THREE methods; in the pilot's own range t_mp2 is ~0.5x the\n"
        "others, so the bound there is ~1.7-2x. Larger bounds at large nbf come from\n"
        "extrapolating fits with different exponents and are NOT evidence of a win."
    )
    if shared:
        print(
            f"Production shared-job fit: t = {shared['c']:.3e} * nbf^{shared['p']:.2f} s "
            f"(n={shared['n']}, nbf {shared['nbf_range'][0]}-{shared['nbf_range'][1]}, "
            f"SCF fraction mean {shared['scf_fraction_mean']:.2f}) - supersedes the range."
        )

    def t(m, nbf):
        return model[m]["c"] * nbf ** model[m]["p"]

    per_dimer: dict[int, list[float]] = {}
    for j in jobs:
        sep = sum(t(m, j["nbf"]) for m in METHODS)
        lo = sep - 2 * min(t(m, j["nbf"]) for m in METHODS)
        sh = shared["c"] * j["nbf"] ** shared["p"] if shared else float("nan")
        acc = per_dimer.setdefault(j["dimer"], [0.0, 0.0, 0.0, 0])
        acc[0] += sep
        acc[1] += lo
        acc[2] += sh
        acc[3] = max(acc[3], j["nbf"])
    order = sorted(per_dimer, key=lambda k: per_dimer[k][0])
    print(
        f"{'dimer':>6} {'nbf(AB)':>7} {'separate_h':>10} {'shared_lo_h':>11} {'shared_hi_h':>11}"
        + (f" {'shared_fit_h':>12}" if shared else "")
    )
    tot = [0.0, 0.0, 0.0]
    for k in order:
        sep, lo, sh, nbf = per_dimer[k]
        tot[0] += sep
        tot[1] += lo
        tot[2] += sh
        print(
            f"S66-{k:02d} {nbf:7d} {sep / 3600:10.2f} {lo / 3600:11.2f} {sep / 3600:11.2f}"
            + (f" {sh / 3600:12.2f}" if shared else "")
        )
    print(
        f"TOTAL ({len(order)} dimers, {len(jobs)} jobs): separate {tot[0] / 3600:.0f} h; "
        f"shared {tot[1] / 3600:.0f}-{tot[0] / 3600:.0f} h (speedup bound <= "
        f"{tot[0] / tot[1]:.2f}x)"
        + (f"; production-fit shared {tot[2] / 3600:.0f} h" if shared else "")
        + ". Loaded-box pilot timings: on a quiet box with RAYON=6 expect less."
    )


# ── load gate / single instance ────────────────────────────────────────────

BUSY_COMMS = ("cargo", "rustc")


# Box-wide heavy-step slot shared with scripts/validation/run_slot.sh.
SLOT_LOCK = Path(
    os.environ.get(
        "FERRIC_SLOT_LOCK", str(Path.home() / ".cache/ferric/validation-slot.lock")
    )
)
LIBRARY_DENSITY_CONV = (
    1e-6  # run_mp2_v_and_scs_2terfc default (returned as density_conv)
)


def busy_reason(max_load: float | None) -> str | None:
    """None if the box is quiet; else why not. Reads /proc only (no pgrep -f).
    `max_load=None` judges foreign processes only (no load average)."""
    if max_load is not None:
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
        if load1 >= max_load:
            return f"load1 {load1:.2f} >= {max_load}"
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            comm = (d / "comm").read_text().strip()
        except OSError:
            continue
        if comm in BUSY_COMMS:
            return f"{comm} running (pid {d.name})"
    return None


def cpu_busy_s() -> float:
    """Box-wide busy CPU-seconds so far (user+nice+system+irq+softirq+steal)."""
    f = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
    busy = sum(int(f[i]) for i in (0, 1, 2, 5, 6, 7))
    return busy / os.sysconf("SC_CLK_TCK")


def own_cpu_s() -> float:
    import resource

    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def foreign_cores(busy0: float, own0: float, wall: float) -> float:
    """Average cores used by OTHER processes over a window: box busy CPU minus
    this process's own CPU. Never raw load: the job's own load would fail long
    jobs and bias the survivors fast (post-sample-load-gate memory)."""
    return max(cpu_busy_s() - busy0 - (own_cpu_s() - own0), 0.0) / max(wall, 1e-9)


def wait_until_quiet(max_load: float, poll_s: float, heartbeat: Path) -> float:
    waited = 0.0
    while True:
        why = busy_reason(max_load)
        if why is None:
            return waited
        heartbeat.write_text(
            json.dumps({"t": time.time(), "state": "waiting", "why": why})
        )
        if waited == 0.0 or int(waited) % 1800 < poll_s:
            print(f"[{time.strftime('%F %T')}] busy: {why}; waiting", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def single_instance(lock_path: Path):
    fh = lock_path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another run holds {lock_path}; refusing to start a second instance")
    fh.write(str(os.getpid()))
    fh.flush()
    return fh  # keep open for the process lifetime


# ── fixed .so ──────────────────────────────────────────────────────────────


def cmd_freeze_so(a) -> None:
    """Copy a built libferric.so into DEST/ferric/ (a private, never-rebuilt package)."""
    src = Path(a.so).resolve()
    pkg = Path(a.dest).expanduser() / "ferric"
    pkg.mkdir(parents=True, exist_ok=True)
    so = pkg / "ferric.cpython-311-x86_64-linux-gnu.so"
    shutil.copy2(src, so)  # a COPY: a rebuild of src can never fault this mapping
    (pkg / "__init__.py").write_text(
        "from .ferric import *\n\n__doc__ = ferric.__doc__\n"
        'if hasattr(ferric, "__all__"):\n    __all__ = ferric.__all__\n'
    )
    print(f"froze {src} -> {so} sha256 {sha256(so)[:16]}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def import_frozen_ferric(so_dir: str | None):
    if so_dir:
        d = str(Path(so_dir).expanduser().resolve())
        sys.path.insert(0, d)
    import ferric

    if so_dir and not Path(ferric.__file__).resolve().is_relative_to(Path(d)):
        sys.exit(f"ferric imported from {ferric.__file__}, not the frozen {d}")
    return ferric


# ── one job (child process) ────────────────────────────────────────────────


def cmd_job(a) -> None:
    os.environ.setdefault("FERRIC_TERF_TABLE_DIR", str(TERF_DIR))
    ferric = import_frozen_ferric(a.so_dir)
    files, fraga = s66_files(), s66_fraga()
    atoms = read_atoms(files[a.dimer])
    calc = KEY_CALC[(a.fragment, a.ghosted == 1)]
    xyz, nfc = build_calc(atoms, fraga[a.dimer], calc)
    mol = ferric.Molecule.from_xyz_string(xyz)
    busy0, own0 = cpu_busy_s(), own_cpu_s()
    t0 = time.perf_counter()
    r = ferric.run_mp2_v_and_scs_2terfc(
        mol,
        ferric.BasisSet.bundled(a.basis),
        ferric.BasisSet.bundled(a.aux),
        frozen_core=nfc,
        include_coulomb_mp2=True,
        density_conv=a.density_conv,
        memory_budget_gb=a.mem_gb,
    )
    wall = time.perf_counter() - t0
    foreign = foreign_cores(busy0, own0, wall)
    so = Path(ferric.__file__).with_name("ferric.cpython-311-x86_64-linux-gnu.so")
    rec = {
        "schema": SCHEMA,
        "dimer": a.dimer,
        "name": files[a.dimer].name,
        "fragment": a.fragment,
        "ghosted": a.ghosted == 1,
        "calc": calc,
        "energies": {
            "mp2": r["mp2_total"],
            "scs2terfc": r["scs_total"],
            "mp2v": r["mp2v_total"],
        },
        "components": {
            k: r[k]
            for k in (
                "mp2_corr",
                "scs_corr",
                "scs_e_os",
                "scs_e_ss",
                "mp2v_att_mp2_corr",
                "mp2v_vv10_e_nl",
                "mp2v_e_os",
                "mp2v_e_ss",
                "n_nlc_points",
            )
        },
        "rhf_energy": r["rhf_energy"],
        "scf_iterations": r["scf_iterations"],
        "density_conv": r["density_conv"],
        "timings": r["timings"],
        "wall_s": wall,
        "foreign_cpu_cores": foreign,
        "frozen_core": nfc,
        "natoms": len(atoms)
        if a.fragment == "AB" or a.ghosted
        else xyz.count("\n") - 2,
        "nbf": job_nbf(atoms, fraga[a.dimer], calc, a.basis),
        "basis": a.basis,
        "aux": a.aux,
        "smoke": bool(a.smoke),
        "ferric_build": getattr(ferric, "__build__", None),
        "ferric_file": ferric.__file__,
        "so_sha256": sha256(so) if so.exists() else None,
        "threads": {
            k: os.environ.get(k) for k in ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "host": platform.node(),
    }
    print("RESULT " + json.dumps(rec), flush=True)


# ── scheduler ──────────────────────────────────────────────────────────────


def record_key(r: dict) -> tuple:
    return (r["dimer"], r["fragment"], bool(r["ghosted"]))


def load_pair_records(path: Path, basis: str, aux: str, smoke: bool = False) -> dict:
    done = {}
    for r in read_jsonl(path):
        if r.get("schema") != SCHEMA or r.get("basis") != basis or r.get("aux") != aux:
            continue
        if bool(r.get("smoke")) != smoke:
            continue
        done[record_key(r)] = r
    return done


def resume_mismatches(done: dict, density_conv: float, so_sha: str | None) -> list[str]:
    """Settings of existing records that differ from this run's. A resume that
    mixed SCF thresholds or binaries would put two methods into one statistic."""
    bad = []
    for r in done.values():
        if r.get("density_conv") != density_conv:
            bad.append(
                f"{record_key(r)} density_conv {r.get('density_conv')} != {density_conv}"
            )
        if so_sha is not None and r.get("so_sha256") != so_sha:
            bad.append(
                f"{record_key(r)} so_sha256 {str(r.get('so_sha256'))[:12]} != {so_sha[:12]}"
            )
    return bad


def judge_contended(rec: dict, a) -> tuple[bool, list[str]]:
    """After-job verdict on FOREIGN activity only: other processes' CPU over the
    job's compute window (measured in the child), and compile jobs still running.
    A skipped pre-job gate always counts as contended."""
    why = []
    if a.no_load_gate:
        why.append("pre-job load gate skipped (--no-load-gate)")
    fc = rec.get("foreign_cpu_cores")
    if fc is None or fc >= a.max_load:
        why.append(f"foreign CPU {fc} cores >= {a.max_load}")
    proc = busy_reason(None)
    if proc:
        why.append(proc)
    return bool(why), why


def cmd_run(a) -> None:
    if a.smoke:
        a.basis, a.aux = SMOKE_BASIS, SMOKE_AUX
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _lock = single_instance(out.with_suffix(out.suffix + ".lock"))  # noqa: F841
    heartbeat = out.with_suffix(out.suffix + ".heartbeat")
    failures = out.with_suffix(out.suffix + ".failures")
    if not a.so_dir and not a.smoke:
        sys.exit(
            "--so-dir is required for a production run (see freeze-so): never load the shared target .so"
        )
    dimers = parse_dimers(a.dimers)
    if a.smoke and a.dimers == "all":  # smoke = the smallest dimer only
        nbf = {
            j["dimer"]: j["nbf"] for j in all_jobs(dimers, a.basis) if j["calc"] == "AB"
        }
        dimers = [min(nbf, key=lambda k: (nbf[k], k))]
    jobs = all_jobs(dimers, a.basis)
    # Cheapest dimer first (by its five jobs' summed nbf^4), all five jobs of a
    # dimer together, so complete dimers - the unit the statistics need - accrue early.
    dcost: dict[int, float] = {}
    for j in jobs:
        dcost[j["dimer"]] = dcost.get(j["dimer"], 0.0) + j["nbf"] ** 4
    jobs.sort(key=lambda j: (dcost[j["dimer"]], j["dimer"], -j["nbf"]))
    env = dict(os.environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env["RAYON_NUM_THREADS"] = str(a.threads)
    env.setdefault("FERRIC_TERF_TABLE_DIR", str(TERF_DIR))
    so_sha = None
    if a.so_dir:
        so = (
            Path(a.so_dir).expanduser()
            / "ferric/ferric.cpython-311-x86_64-linux-gnu.so"
        )
        so_sha = sha256(so) if so.exists() else None
    want_dc = a.density_conv if a.density_conv is not None else LIBRARY_DENSITY_CONV
    bad = resume_mismatches(
        load_pair_records(out, a.basis, a.aux, a.smoke), want_dc, so_sha
    )
    if bad:
        sys.exit(
            f"refusing to resume {out}: {len(bad)} existing records were made with other "
            "settings (use a new --out, or the original --density-conv / --so-dir):\n  "
            + "\n  ".join(bad[:10])
        )
    SLOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    tries: dict[tuple, int] = {}
    for _pass in range(a.max_tries):
        done = load_pair_records(out, a.basis, a.aux, a.smoke)
        todo = [
            j for j in jobs if (j["dimer"], j["fragment"], j["ghosted"]) not in done
        ]
        print(f"pass {_pass + 1}: {len(done)} done, {len(todo)} to do", flush=True)
        if not todo:
            break
        for j in todo:
            key = (j["dimer"], j["fragment"], j["ghosted"])
            if tries.get(key, 0) >= a.max_tries:
                continue
            waited = (
                0.0
                if a.no_load_gate
                else wait_until_quiet(a.max_load, a.poll_s, heartbeat)
            )
            load1 = float(Path("/proc/loadavg").read_text().split()[0])
            heartbeat.write_text(
                json.dumps({"t": time.time(), "state": "running", "job": key})
            )
            tries[key] = tries.get(key, 0) + 1
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "job",
                "--dimer",
                str(j["dimer"]),
                "--fragment",
                j["fragment"],
                "--ghosted",
                str(int(j["ghosted"])),
                "--basis",
                a.basis,
                "--aux",
                a.aux,
                "--mem-gb",
                str(a.mem_gb),
            ]
            if a.so_dir:
                cmd += ["--so-dir", a.so_dir]
            if a.density_conv is not None:
                cmd += ["--density-conv", str(a.density_conv)]
            if a.smoke:
                cmd += ["--smoke"]
            # The whole job holds the box-wide heavy slot (run_slot.sh's lock);
            # -o so a daemon the child spawns cannot inherit and keep it.
            cmd = ["flock", "-o", str(SLOT_LOCK), *cmd]
            t0 = time.time()
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            res = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            if p.returncode == 0 and res:
                rec = json.loads(res[-1][len("RESULT ") :])
                rec["load1_at_start"] = load1
                rec["waited_s"] = waited
                rec["contended"], rec["contended_why"] = judge_contended(rec, a)
                rec["script_head"] = git_info()["head"]
                if a.smoke:
                    rec["label"] = "SMOKE: not a result"
                append_jsonl(out, rec)
                print(
                    f"[{time.strftime('%F %T')}] S66-{j['dimer']:02d} {j['calc']:8s} nbf={j['nbf']:4d} "
                    f"wall={rec['wall_s']:.1f}s scf={rec['timings']['scf_s']:.1f}s load1={load1:.1f}",
                    flush=True,
                )
            else:
                fail = {
                    "t": time.time(),
                    "job": list(key),
                    "returncode": p.returncode,
                    "elapsed_s": time.time() - t0,
                    "stderr_tail": p.stderr[-2000:],
                }
                append_jsonl(failures, fail)
                print(f"FAILED {key} rc={p.returncode} (see {failures})", flush=True)
    heartbeat.write_text(json.dumps({"t": time.time(), "state": "finished"}))


# ── report ─────────────────────────────────────────────────────────────────


def stats(errs: list[float]) -> dict:
    n = len(errs)
    return {
        "n": n,
        "mse": sum(errs) / n,
        "mad": sum(abs(x) for x in errs) / n,
        "rmsd": math.sqrt(sum(x * x for x in errs) / n),
        "max_abs": max(abs(x) for x in errs),
    }


def cmd_report(a) -> None:
    done = load_pair_records(Path(a.out), a.basis, a.aux)
    ref = json.loads(REF_JSON.read_text())
    files = s66_files()
    builds = sorted(
        {json.dumps(r.get("ferric_build"), sort_keys=True) for r in done.values()}
    )
    rows, per = {}, {m: {"cp": [], "nocp": []} for m in METHODS}
    for k in sorted({d for d, _, _ in done}):
        e = {c: done.get((k, *CALC_KEY[c])) for c in CALCS}
        row = {
            "dimer": k,
            "name": files[k].name,
            "ref": ref[files[k].stem],
            "calcs_done": sum(v is not None for v in e.values()),
        }
        for m in METHODS:
            E = {c: (v["energies"][m] if v else None) for c, v in e.items()}
            if None not in (E["AB"], E["A"], E["B"]):
                row[f"{m}_nocp"] = (E["AB"] - E["A"] - E["B"]) * HARTREE_TO_KCAL
                per[m]["nocp"].append(row[f"{m}_nocp"] - row["ref"])
            if None not in (E["AB"], E["A_in_AB"], E["B_in_AB"]):
                row[f"{m}_cp"] = (
                    E["AB"] - E["A_in_AB"] - E["B_in_AB"]
                ) * HARTREE_TO_KCAL
                per[m]["cp"].append(row[f"{m}_cp"] - row["ref"])
        rows[k] = row
    n_complete = sum(r["calcs_done"] == 5 for r in rows.values())
    summ = {m: {c: stats(v) for c, v in per[m].items() if v} for m in METHODS}
    res = {
        "n_dimers_with_any_record": len(rows),
        "n_dimers_complete": n_complete,
        "full_set": n_complete == 66,
        "basis": a.basis,
        "aux": a.aux,
        "units": "kcal/mol",
        "published": PUBLISHED,
        "published_caveat": GEOMETRY_CAVEAT,
        "summary": summ,
        "rows": list(rows.values()),
        "provenance": {
            "git": git_info(),
            "python": platform.python_version(),
            "host": platform.node(),
            "ferric_builds": builds,
            "hartree_to_kcal": HARTREE_TO_KCAL,
            "reference": "S66x8 1.00 CCSD(T)/CBS CP, BEGDB, testdata/reference/c9_refs/s66x8_ccsdt_cbs.json",
            "settings": "shared SCF (run_mp2_v_and_scs_2terfc); RI aux per record; frozen core 1/heavy atom; "
            "published default params; ghost atoms carry basis in CP",
            "contended_records": sum(bool(r.get("contended")) for r in done.values()),
        },
    }
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(
        f"{n_complete}/66 dimers complete ({len(rows)} with any record); "
        + (
            "FULL SET"
            if res["full_set"]
            else "PARTIAL COVERAGE: statistics are over the n shown, NOT comparable to published full-set values"
        )
    )
    if len(builds) > 1:
        print(
            f"WARNING: {len(builds)} different ferric builds in these records: {builds}"
        )
    print(
        "no-CP = fit-reproduction check (published fits are no-CP); CP = physics check."
    )
    print(f"CAVEAT: {GEOMETRY_CAVEAT}.")
    for m in METHODS:
        pub = PUBLISHED[m]["rmsd"]
        for c in ("nocp", "cp"):
            s = summ[m].get(c)
            if s:
                print(
                    f"{m:10s} {c:5s} n={s['n']:2d} MSE={s['mse']:+.3f} MAD={s['mad']:.3f} "
                    f"RMSD={s['rmsd']:.3f} max={s['max_abs']:.3f}"
                    + (
                        f"   (published no-CP full-set RMSD {pub}, original S66 geometries; see CAVEAT)"
                        if c == "nocp"
                        else ""
                    )
                )


# ── selftest / unit (from #474) ────────────────────────────────────────────


def cmd_unit(a) -> None:
    """No-ferric checks: key/filters, cost-model fit, calc<->key round trip."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.jsonl"
        base = {
            "schema": SCHEMA,
            "dimer": 1,
            "fragment": "A",
            "ghosted": True,
            "basis": "b",
            "aux": "x",
        }
        f.write_text(
            json.dumps(base)
            + "\n"
            + json.dumps({**base, "ghosted": False, "aux": "y"})
            + "\n"
            + json.dumps({**base, "fragment": "B", "smoke": True})
            + "\n"
            + json.dumps(
                {"dimer": 1, "calc": "AB", "method": "mp2", "basis": "b", "aux": "x"}
            )
            + "\n"
        )
        assert set(load_pair_records(f, "b", "x")) == {(1, "A", True)}
        assert set(load_pair_records(f, "b", "y")) == {(1, "A", False)}
        assert set(load_pair_records(f, "b", "x", smoke=True)) == {(1, "B", True)}
        assert load_pair_records(f, "other", "x") == {}
    for c, k in CALC_KEY.items():
        assert KEY_CALC[k] == c
    c, p = fit_power([10, 20, 40], [3 * 10**4.5, 3 * 20**4.5, 3 * 40**4.5])
    assert abs(p - 4.5) < 1e-9 and abs(c - 3) < 1e-6
    assert nbf_per_element("aug-cc-pvtz") == {"H": 23, "C": 46, "N": 46, "O": 46}
    assert nbf_per_element("cc-pvdz") == {"H": 5, "C": 14, "N": 14, "O": 14}
    unit_jsonl_and_resume()
    print("UNIT OK")


def unit_jsonl_and_resume() -> None:
    """Torn-append tolerance, mid-file corruption error, fresh-line appends,
    resume refusal on settings mismatch, and the foreign-only contention verdict."""
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "r.jsonl"
        f.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "tr')  # crash mid-append
        assert read_jsonl(f) == [{"a": 1}, {"a": 2}]
        append_jsonl(f, {"a": 4})  # must start on a fresh line, torn tail moved out
        assert read_jsonl(f) == [{"a": 1}, {"a": 2}, {"a": 4}]
        assert f.read_text().endswith('{"a": 4}\n')
        assert '{"a": 3, "tr' in f.with_suffix(".jsonl.torn").read_text()
        g = Path(d) / "bad.jsonl"
        g.write_text('{"a": 1}\n{oops\n{"a": 2}\n')
        try:
            read_jsonl(g)
            raise AssertionError("mid-file corruption must be an error")
        except ValueError as e:
            assert ":2:" in str(e)
        h = Path(d) / "badlast.jsonl"
        h.write_text('{"a": 1}\n{oops\n')  # newline-terminated: not a torn append
        try:
            read_jsonl(h)
            raise AssertionError("a complete malformed line must be an error")
        except ValueError:
            pass
        assert read_jsonl(Path(d) / "missing.jsonl") == []
    rec = {
        "dimer": 1,
        "fragment": "A",
        "ghosted": False,
        "density_conv": 1e-6,
        "so_sha256": "ab" * 32,
    }
    done = {record_key(rec): rec}
    assert resume_mismatches(done, 1e-6, "ab" * 32) == []
    assert len(resume_mismatches(done, 1e-9, "ab" * 32)) == 1
    assert len(resume_mismatches(done, 1e-6, "cd" * 32)) == 1
    assert resume_mismatches(done, 1e-6, None) == []  # smoke: no frozen .so to compare
    a = SimpleNamespace(no_load_gate=False, max_load=1.5)
    # A long job's OWN load is never counted: only the measured foreign cores.
    if busy_reason(None) is None:
        assert judge_contended({"foreign_cpu_cores": 0.2}, a) == (False, [])
    assert judge_contended({"foreign_cpu_cores": 2.0}, a)[0]
    assert judge_contended({}, a)[0]
    assert judge_contended(
        {"foreign_cpu_cores": 0.0}, SimpleNamespace(no_load_gate=True, max_load=1.5)
    )[0]


def cmd_selftest(a) -> None:
    """Exactness anchors against committed PySCF/numpy references (repo tests' bars)."""
    os.environ.setdefault("FERRIC_TERF_TABLE_DIR", str(TERF_DIR))
    ferric = import_frozen_ferric(getattr(a, "so_dir", None))

    val = ROOT / "testdata/reference/validation"
    fails = 0
    # Python run_* use the library-default SCF (density_conv 1e-6); the Rust validation tests use 1e-9.
    TOL = 5e-8

    def chk(name, got, want, tol):
        nonlocal fails
        bad = abs(got - want) > tol
        fails += bad
        print(
            f"{'FAIL' if bad else 'ok  '} {name}: got {got:.12f} want {want:.12f} diff {got - want:+.2e} (tol {tol:.0e})"
        )

    ref = json.loads((val / "scs_mp2_2terfc/h2o_aug-cc-pvdz.json").read_text())
    h2o = (ROOT / "testdata/molecules/validation/h2o.xyz").read_text()
    e = [x for x in ref["energies"] if x["frozen_core"] == 1][0]["scs_2terfc"][
        "total_energy"
    ]
    mol = ferric.Molecule.from_xyz_string(h2o)
    adz, adzri = (
        ferric.BasisSet.bundled("aug-cc-pvdz"),
        ferric.BasisSet.bundled("aug-cc-pvdz-rifit"),
    )
    r = ferric.run_mp2_v_and_scs_2terfc(mol, adz, adzri, frozen_core=1)
    chk(
        "SCS-MP2(2terfc) H2O/aDZ fc1 total (shared entry point)", r["scs_total"], e, TOL
    )

    ref = json.loads((val / "mp2_v/water_dimer_aug-cc-pvdz.json").read_text())
    wd = ferric.Molecule.from_xyz_string(
        (ROOT / "testdata/molecules/s22/water_dimer.xyz").read_text()
    )
    r = ferric.run_mp2_v_and_scs_2terfc(wd, adz, adzri, frozen_core=2)
    chk("RHF water dimer/aDZ", r["rhf_energy"], ref["rhf_energy"], 1e-9)
    chk(
        "RI-MP2 corr water dimer/aDZ fc2",
        r["mp2_corr"],
        ref["mp2_erfc_control"]["coulomb_e_corr"],
        TOL,
    )
    chk(
        "MP2-V E_nl (b=11, r0=1.0 damped) water dimer/aDZ",
        r["mp2v_vv10_e_nl"],
        ref["vv10"]["b=11.0"]["damped"]["e_nl_numpy"],
        1e-9,
    )
    print("SELFTEST", "FAILED" if fails else "PASSED")
    sys.exit(1 if fails else 0)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_dimers(s: str) -> list[int]:
    if s == "all":
        return list(range(1, 67))
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    assert all(1 <= k <= 66 for k in out), out
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--so-dir")
    sub.add_parser("unit")
    c = sub.add_parser("check-split")
    c.add_argument("--dimers", default="all")
    d = sub.add_parser("dry-run")
    d.add_argument("--dimers", default="all")
    d.add_argument("--basis", default=BASIS)
    e = sub.add_parser("estimate")
    e.add_argument("--dimers", default="all")
    e.add_argument("--basis", default=BASIS)
    e.add_argument("--pilot", default=str(PILOT_JSONL))
    e.add_argument(
        "--out", help="production JSONL; refits the shared cost model from its timings"
    )
    f = sub.add_parser("freeze-so")
    f.add_argument(
        "--so",
        required=True,
        help="built libferric.so (e.g. a worktree's target/release)",
    )
    f.add_argument("--dest", required=True)
    j = sub.add_parser("job", help="(internal) one system in a child process")
    j.add_argument("--dimer", type=int, required=True)
    j.add_argument("--fragment", choices=("AB", "A", "B"), required=True)
    j.add_argument("--ghosted", type=int, choices=(0, 1), required=True)
    j.add_argument("--basis", required=True)
    j.add_argument("--aux", required=True)
    j.add_argument("--mem-gb", type=float, required=True)
    j.add_argument("--so-dir")
    j.add_argument("--density-conv", type=float)
    j.add_argument("--smoke", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument(
        "--so-dir", help="frozen package dir from freeze-so (required unless --smoke)"
    )
    r.add_argument("--dimers", default="all")
    r.add_argument("--basis", default=BASIS)
    r.add_argument("--aux", default=AUX)
    r.add_argument("--mem-gb", type=float, default=6.0)
    r.add_argument(
        "--threads",
        type=int,
        default=6,
        help="RAYON_NUM_THREADS (box: 6 physical cores)",
    )
    r.add_argument(
        "--density-conv",
        type=float,
        help="SCF RMS density gate (default: library 1e-6)",
    )
    r.add_argument(
        "--max-load",
        type=float,
        default=1.5,
        help="pre-job gate on load1, and post-job limit on FOREIGN CPU cores",
    )
    r.add_argument("--poll-s", type=float, default=120.0)
    r.add_argument("--max-tries", type=int, default=2)
    r.add_argument(
        "--no-load-gate",
        action="store_true",
        help="do not wait for a quiet box before each job (smoke/debug); every record "
        "is then marked contended. The slot lock is still taken.",
    )
    r.add_argument(
        "--smoke",
        action="store_true",
        help=f"{SMOKE_BASIS}/{SMOKE_AUX}, records labelled 'SMOKE: not a result'",
    )
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    rp.add_argument("--json")
    rp.add_argument("--basis", default=BASIS)
    rp.add_argument("--aux", default=AUX)
    a = p.parse_args()
    {
        "unit": cmd_unit,
        "selftest": cmd_selftest,
        "dry-run": cmd_dry_run,
        "estimate": cmd_estimate,
        "freeze-so": cmd_freeze_so,
        "job": cmd_job,
        "run": cmd_run,
        "report": cmd_report,
    }.get(a.cmd, lambda a: sys.exit(0 if check_split(parse_dimers(a.dimers)) else 1))(a)


if __name__ == "__main__":
    main()
