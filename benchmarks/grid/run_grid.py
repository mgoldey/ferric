#!/usr/bin/env python3
"""Idempotent, memory-safe production grid for the SR-MP2+LR-RPA paper.

Matrix: A24 (24 systems) + S22 (22 systems)
      x {dimer, mA_cp, mB_cp}            (CP via @-ghost atoms)
      x {aug-cc-pVDZ, aug-cc-pVTZ}
      x {scs       : scs-mp2                          -> E_OS (dRPA column)
         dlr042    : rs-mp2-rpa delta-lr      w=0.42  -> MP2 / naive A / LRC[kappa] B
         cr02      : rs-mp2-rpa coupled-rings w=0.20  -> LRC[E] T / dRPA[Coulomb]}

Idempotency: the job key IS the filename; TOML content is a pure function of
the key. A job is skipped iff out/<key>.out exists with the completion marker
AND toml/<key>.toml matches the regenerated content (settings drift => rerun).
Outputs are written to .part and renamed on success (atomic).

Memory safety (the previous grid OOM'd the box into a reboot):
 1. per-child hard cap: resource.setrlimit(RLIMIT_AS) in preexec_fn —
    a runaway child gets malloc failure -> abort, never kernel OOM
 2. admission control: sum of running job ESTIMATES <= MEM_BUDGET, and live
    MemAvailable must leave FLOOR+1 GB headroom after admitting
 3. watchdog: if MemAvailable < FLOOR, SIGKILL the process group of the
    largest running job and requeue it (max 2 attempts, then failed marker)
 4. jobs whose estimate exceeds JOB_CAP are excluded up front (logged);
    with FERRIC_ERI3_BUDGET_GB streaming, even AT-stacked/aTZ fits
 5. all jobs run at 1 thread; throughput comes from running many concurrently
    (SCF + MP2 are BLAS-serial at OPENBLAS=1, so per-job threading is a no-op —
    see the PARALLELISM MODEL note on the constants below)

S22 rs-mp2-rpa jobs use [rpa] trunc_thresh=1e-4 (PDEP truncation), gated on a
benzene-fragment validation: full-rank vs truncated total energy must agree
to TRUNC_TOL before any other S22 rs job is admitted.
"""
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = str((ROOT / "../../target/release/ferric-cli").resolve())
BASIS_DIR = ROOT / "../../crates/ferric-core/src/basis/bundled"

# terf/terfc (tempered attenuator) 2D interpolation tables, needed by the
# terf042/terfcr0 methods (2026-07-21 addition). Same discovery convention as
# scripts/run_ethylene_terfc_sweep.py / scripts/bisect_a24_aqz_terfc_r0.py --
# without this, ferric-cli errors "terf 2-center engine not available".
_TERF_DIR = os.environ.get("GRID_TERF_TABLE_DIR", "")
if not _TERF_DIR or not os.path.exists(os.path.join(_TERF_DIR, "16_4_2.bin")):
    for _cand in (str(ROOT / "../../terf-tables"), "/home/matt/qc/ferric/terf-tables"):
        if os.path.exists(os.path.join(_cand, "16_4_2.bin")):
            _TERF_DIR = _cand
            break

GB = 1e9
NCORES = 12               # box has 12 cores
MEM_BUDGET = 18 * GB      # sum of running estimates (concurrent jobs). Lowered
                          # 21->18 on 2026-06-14: at 21 GB, three nbf=736 aTZ
                          # π-stack jobs (est 7.0 each = 21.0, exactly at budget)
                          # packed, and their TRANSIENT dRPA-stage memory peaks
                          # coincided → MemAvailable <FLOOR → watchdog killed +
                          # requeued one (no data lost, but churn). 18 GB packs
                          # 2 such jobs and leaves ~5-6 GB headroom for the
                          # peaks. The watchdog + RLIMIT_AS remain the hard net.
FLOOR = 2.5 * GB          # MemAvailable floor -> watchdog kills largest job
JOB_CAP = 17 * GB         # above this: infeasible even run alone (box: ~19G avail)
RLIMIT_MAX = 18 * GB      # absolute per-child address-space ceiling
MAX_WORKERS = NCORES      # don't oversubscribe cores
SOLO_GB = 10 * GB         # est >= this -> run alone with all BLAS cores (can't
                          # pack >1, so use BLAS threading instead of idling ~11
                          # cores). Captures the nbf=1127 S22 aTZ dimers (~11 GB).
PACK_BLAS = 1             # packed jobs MUST stay single-BLAS-thread (see below)
# PARALLELISM MODEL (calibrated 2026-06-14 on s22-17_mB_cp_atz_dlr042, nbf=506).
# Two regimes:
#  * SMALL jobs pack ~3-up, each at OPENBLAS=1, RAYON=1. Concurrency is the lever
#    — their serial SCF/MP2 phases (and disk-spill I/O stalls) overlap to fill
#    cores. Per-job rayon threading is a no-op (1 thread 173.6s vs 6 threads
#    175.1s: the only rayon-parallel stage, the dRPA quad par_iter, is a small
#    fraction). Memory caps packing at ~3 on this 23 GB box (a 4th breaches
#    FLOOR), so packed-regime utilization is ~3 cores; that is accepted.
#  * BIG jobs (est >= SOLO_GB) can't pack, so they run SOLO at OPENBLAS=NCORES,
#    RAYON=1. The MP2 G_i = B_i^T·B wide GEMMs (BLAS3) use all 12 cores. Safe
#    because a solo job is the ONLY process — no cross-process OpenBLAS contention.
#
# CRASH-SAFETY (learned the hard way 2026-06-14): BLAS threading is safe ONLY for
# the SOLO big jobs, NOT for packed jobs. We briefly tried PACK_BLAS=3 (an
# isolated dlr042 test looked fine) and it crashed CONCURRENTLY-RUN cr02 jobs
# with "stack overflow, aborting" (rc=-6) intermittently — 7 S66 cr02 jobs died.
# Root cause: it is NOT enough that OPENBLAS>1 pairs with RAYON=1. When MULTIPLE
# processes each run multithreaded OpenBLAS, their dgetrf worker threads (the
# coupled-rings path LU-factorizes the dielectric at 3 operators x ~16 ω, the
# heaviest LU load) overflow OpenBLAS's own thread stacks under contention. This
# is a STACK overflow, not OOM, so RLIMIT_AS / the watchdog cannot catch it.
# RULE: OPENBLAS>1 is allowed ONLY when the job runs ALONE (solo big jobs).
# Packed/concurrent jobs are always OPENBLAS=1. Enforced by assert in launch().
# FERRIC_ERI3_BUDGET_GB caps the resident raw 3-index tensor (aux-blocked
# recompute in the MP2/RPA transforms; disk-spill in the DF-JK SCF), so the
# in-core AO term saturates at the budget:
#   est = BASE + AO_MULT*min(naux*nbf^2*8, budget) + 3*naux*nia*8 + nvir*nia*8
# (nia = nocc*nvir from REAL atoms only; ghosts carry basis, not electrons).
# AO_MULT recalibrated 2026-06-14 against LIVE peak RSS: an nbf=462 aDZ job and
# an nbf=506 aTZ job both peaked at 5.8 GB resident (not the 8.0/7.6 GB the old
# 3.5 multiplier predicted). 2.9 tracks that with a ~1 GB margin, so three
# nbf~500 jobs (est ~6.8 GB, real ~5.8 GB) pack into the 21 GB budget on the
# 23 GB box. The big nbf=1127 dimers stay solo (est ~11 GB).
AO_MULT = 2.9
BUDGET = 2.0 * GB
BASE = 0.5 * GB
TIMEOUT = 6 * 3600
POLL = 3.0
TRUNC = 1e-3      # 2026-07-21: raised from 1e-4 -- Matt decided PDEP truncation
                  # now applies everywhere, including this validation grid, not
                  # just production runs (supersedes the full-rank-only rule in
                  # memory [[no-pdep-trunc-in-method-grid]]). Re-run
                  # validate_trunc() and re-check headline numbers at this
                  # tolerance before trusting existing 1e-4-era conclusions.
TRUNC_TOL = 2e-5
K = 627.509474

BASES = {"adz": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
         "atz": ("aug-cc-pvtz", "aug-cc-pvtz-rifit"),
         "aqz": ("aug-cc-pvqz", "aug-cc-pvqz-rifit")}
METHODS = ("scs", "dlr042", "cr02")
# terf/terfc (tempered Dutoi/Goldey attenuator, r0 in Angstrom at the
# user-facing config boundary -- Bohr internally, see config.rs's r0 doc)
# is NOT a fixed-matrix method here: 2026-07-21, Matt corrected an earlier
# single-point terf042/terfcr0 matrix entry (omega-matched r0=1.68/3.54 Å --
# the latter already >95% Coulomb-saturated per rimp2.rs:1178's terfc
# monotonicity test, i.e. past the physically interesting region) to an
# ADAPTIVE r0 scan over the actual discriminating range (~0-2 Å): coarse
# shape first, then 0.05 Å refinement near the minimum, then bisection to
# 0.01 Å. This needs per-system adaptive control flow that this fixed
# job-matrix scheduler doesn't have -- see
# scripts/bisect_a24_aqz_terfc_r0.py (the adaptive companion script) instead.
Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ar": 18}

for d in ("geoms", "toml", "out"):
    (ROOT / d).mkdir(exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "grid.log", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- databases
PSI4_RAW = "https://raw.githubusercontent.com/psi4/psi4/master/psi4/share/psi4/databases"


def ensure_db(name):
    """psi4 database files are fetched, not committed (upstream LGPL)."""
    p = ROOT / name
    if not p.exists():
        subprocess.run(["curl", "-sf", "-o", str(p), f"{PSI4_RAW}/{name}"], check=True)
    return p


def parse_db(path, bind_re):
    src = open(path).read()
    geos, refs = {}, {}
    pat = r"GEOS\['%s-%s-dimer' % \(dbse, '(\d+)'\)\] = qcdb\.Molecule\(\"\"\"(.*?)\"\"\"\)"
    for m in re.finditer(pat, src, re.S):
        idx, body = int(m.group(1)), m.group(2)
        frags = []
        for part in body.split("--"):
            atoms = []
            for line in part.splitlines():
                t = line.split()
                if len(t) == 4 and t[0].capitalize() in Z:
                    atoms.append((t[0].capitalize(),
                                  float(t[1]), float(t[2]), float(t[3])))
            if atoms:
                frags.append(atoms)
        assert len(frags) == 2, f"{path} sys {idx}: {len(frags)} fragments"
        geos[idx] = frags
    for m in re.finditer(bind_re, src):
        refs[int(m.group(1))] = float(m.group(2))
    return geos, refs


def basis_counts(name):
    d = json.load(open(BASIS_DIR / f"{name}.json"))
    out = {}
    for z, ed in d["elements"].items():
        n = 0
        for sh in ed["electron_shells"]:
            for i, l in enumerate(sh["angular_momentum"]):
                ncon = len(sh["coefficients"]) if len(sh["angular_momentum"]) == 1 else 1
                n += (2 * l + 1) * ncon
        out[int(z)] = n
    return out


NBF = {b: basis_counts(obs) for b, (obs, _) in BASES.items()}
NAUX = {b: basis_counts(aux) for b, (_, aux) in BASES.items()}


AQZ_SAFETY_MULT = 1.5  # AO_MULT (2.9) was calibrated at aDZ/aTZ scale only
                        # (nbf<=1127, 2026-06-14). aQZ jobs are new territory
                        # (2026-07-21): a benzene-dimer aQZ probe (nbf=1512,
                        # much bigger than any A24 system) peaked at ~16.95 GB
                        # against a formula that predicted far less -- the gap
                        # was several budget-blind allocations (per-worker
                        # freq-quad scratch, retained intermediates) since
                        # fixed in ferric-rpa, but unvalidated against a fresh
                        # real aQZ measurement. Pad aQZ estimates 1.5x until a
                        # live A24-aQZ peak-RSS measurement recalibrates this
                        # the way AO_MULT itself was calibrated for aTZ.


def estimate(atoms, basis):
    """All atoms (incl. ghosts) carry basis functions; only real atoms
    carry electrons."""
    nbf = sum(NBF[basis][Z[s.lstrip('@')]] for s, *_ in atoms)
    naux = sum(NAUX[basis][Z[s.lstrip('@')]] for s, *_ in atoms)
    nocc = sum(Z[s] for s, *_ in atoms if not s.startswith('@')) // 2
    nvir = nbf - nocc
    nia = nocc * nvir
    est = (BASE + AO_MULT * min(naux * nbf * nbf * 8, BUDGET)
           + 3 * naux * nia * 8 + nvir * nia * 8)
    if basis == "aqz":
        est *= AQZ_SAFETY_MULT
    return est, nbf, naux


# ---------------------------------------------------------------- job setup
def write_if_changed(path, content):
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
        return True
    return False


def xyz_text(atoms, comment):
    lines = [str(len(atoms)), comment]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, x, y, z in atoms]
    return "\n".join(lines) + "\n"


# Per-system SCF overrides (keyed by sysname, applied to all fragments of the
# system for CP consistency). a24-21 (C2H4*Ar dimer): DIIS-8 never finds the
# aufbau state (err_max ~0.9 plateau 33 Ha high); diis_size=16 converges it
# to the correct C2H4+Ar limit. Converged energies are DIIS-size independent.
SCF_EXTRA = {"a24-21": "diis_size = 16\n"}


def toml_text(xyz, basis, method, trunc, scf_extra=""):
    obs, aux = BASES[basis]
    t = f'[molecule]\nxyz = "{xyz}"\n\n[basis]\nname = "{obs}"\n\n'
    # max_iter 400: the A24 argon dimers (20, 21) need 100-400 DIIS
    # iterations; converged energies are iteration-count independent.
    t += ('[scf]\ndf_j_aux = "def2-universal-jkfit"\n'
          'df_k_aux = "def2-universal-jkfit"\nmax_iter = 400\n'
          + scf_extra + '\n')
    if method == "scs":
        t += f'[method]\nkind = "scs-mp2"\n\n[mp2]\nauxbasis = "{aux}"\n'
    else:
        form, omega = (("delta-lr", "0.42") if method == "dlr042"
                       else ("coupled-rings", "0.2"))
        t += f'[method]\nkind = "rs-mp2-rpa"\n\n'
        t += f'[mp2]\nauxbasis = "{aux}"\nomega = {omega}\nformulation = "{form}"\n'
        # Quadrature optimization: use 12 points instead of 20 (~40% speedup, <1% accuracy cost)
        # n_quad lives under [rpa], not a separate [quadrature] section --
        # [quadrature] is not a valid TOML key (deny_unknown_fields hard-errors
        # it; fixed 2026-07-21, this stanza was previously silently never
        # exercised because every job hit this parse error immediately).
        t += f'\n[rpa]\ntrunc_thresh = {TRUNC if trunc else 0.0}\nn_quad = 12\n'
    return t


MARKER = {"scs": "Total      =", "dlr042": "Total energy", "cr02": "Total energy"}


def build_jobs():
    a24_geos, a24_refs = parse_db(
        ensure_db("A24.py"), r"BIND\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\s*\)\]\s*=\s*(-?[\d.]+)")
    s22_geos, s22_refs = parse_db(
        ensure_db("S22.py"), r"BIND_S22B\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\s*\)\]\s*=\s*(-?[\d.]+)")
    # S66 (Rezac 2011): same GEOS layout, but BIND uses a QUOTED index
    # (dbse, '1') and the plain BIND[...] name (CCSD(T)/CBS refs).
    s66_geos, s66_refs = parse_db(
        ensure_db("S66.py"), r"BIND\['%s-%s'\s*%\s*\(dbse,\s*'(\d+)'\s*\)\]\s*=\s*(-?[\d.]+)")
    json.dump({"a24": a24_refs, "s22": s22_refs, "s66": s66_refs},
              open(ROOT / "refs.json", "w"), indent=1)

    ghost = lambda atoms: [("@" + s, x, y, z) for s, x, y, z in atoms]
    jobs, excluded = [], []
    for dbse, geos in (("a24", a24_geos), ("s22", s22_geos), ("s66", s66_geos)):
        for idx, (fA, fB) in sorted(geos.items()):
            frags = {"dimer": fA + fB, "mA_cp": fA + ghost(fB), "mB_cp": ghost(fA) + fB}
            for tag, atoms in frags.items():
                sysname = f"{dbse}-{idx:02d}"
                xyz_path = ROOT / "geoms" / f"{sysname}_{tag}.xyz"
                geom_drift = write_if_changed(xyz_path, xyz_text(atoms, f"{sysname} {tag}"))
                if geom_drift:
                    for stale in (ROOT / "out").glob(f"{sysname}_{tag}_*"):
                        stale.unlink()
                for basis in BASES:
                    est, nbf, naux = estimate(atoms, basis)
                    for method in METHODS:
                        key = f"{sysname}_{tag}_{basis}_{method}"
                        trunc = (dbse in ("s22", "s66") and method != "scs")
                        if est > JOB_CAP:
                            excluded.append((key, est / GB))
                            continue
                        tt = toml_text(xyz_path, basis, method, trunc,
                                       SCF_EXTRA.get(sysname, ""))
                        if write_if_changed(ROOT / "toml" / f"{key}.toml", tt):
                            # settings drift invalidates any prior result
                            (ROOT / "out" / f"{key}.out").unlink(missing_ok=True)
                            (ROOT / "out" / f"{key}.failed").unlink(missing_ok=True)
                        jobs.append(dict(key=key, est=est, nbf=nbf, naux=naux,
                                         method=method, basis=basis, attempts=0))
    return jobs, excluded


def is_done(job):
    out = ROOT / "out" / f"{job['key']}.out"
    return out.exists() and MARKER[job["method"]] in out.read_text()


# ------------------------------------------------------------- proc control
def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) * 1024
    return 0


def make_preexec(limit_bytes):
    def fn():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        # generous default pthread stacks (does not affect rayon's fixed
        # 2 MB worker stacks; those are safe with OPENBLAS_NUM_THREADS=1).
        stack = 64 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_STACK, (stack, stack))
    return fn


def launch(job, blas_threads=1, rayon_threads=1):
    # CRASH-SAFETY INVARIANT: multithreaded OpenBLAS is ONLY safe when rayon has
    # a single worker. Multiple concurrent rayon workers each calling parallel
    # OpenBLAS LU (dgetrf_parallel) overflow rayon's fixed 2 MB worker stacks ->
    # SIGSEGV (gdb-verified on a24-04; the dRPA quad stage LU-factorizes inside
    # par_iter). RLIMIT_STACK does NOT cover rayon's hardcoded worker stacks, so
    # the memory machinery cannot catch this — it's a stack overflow, not OOM.
    # With rayon_threads=1 there is exactly one worker, so the par_iter runs
    # sequentially and OpenBLAS can safely thread the GEMMs. Verified crash-free
    # 2026-06-14 (OPENBLAS=12 RAYON=1 on s22-17_mB_cp_atz, exit 0, valid energy).
    assert blas_threads == 1 or rayon_threads == 1, \
        "OPENBLAS>1 requires RAYON=1 (stack-overflow crash otherwise)"
    key = job["key"]
    limit = int(min(job["est"] * 2 + 3 * GB, RLIMIT_MAX))
    env = dict(os.environ,
               OPENBLAS_NUM_THREADS=str(blas_threads), OMP_NUM_THREADS=str(blas_threads),
               RAYON_NUM_THREADS=str(rayon_threads), FERRIC_ERI3_BUDGET_GB=str(BUDGET / GB),
               FERRIC_TERF_TABLE_DIR=_TERF_DIR)
    part = open(ROOT / "out" / f"{key}.out.part", "w")
    err = open(ROOT / "out" / f"{key}.err", "w")
    proc = subprocess.Popen([BIN, str(ROOT / "toml" / f"{key}.toml")],
                            stdout=part, stderr=err, env=env,
                            preexec_fn=make_preexec(limit), cwd=ROOT)
    log(f"start {key} est={job['est']/GB:.1f}G nbf={job['nbf']} "
        f"blas={blas_threads} rayon={rayon_threads} pid={proc.pid}")
    return dict(job=job, proc=proc, t0=time.time(), part=part, err=err)


def kill_group(rec):
    try:
        os.killpg(rec["proc"].pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    rec["proc"].wait()


def finish(rec):
    """Returns 'ok' | 'requeue' | 'failed'."""
    job, rc = rec["job"], rec["proc"].returncode
    key = job["key"]
    rec["part"].close()
    rec["err"].close()
    part = ROOT / "out" / f"{key}.out.part"
    if rc == 0 and MARKER[job["method"]] in part.read_text():
        part.rename(ROOT / "out" / f"{key}.out")
        log(f"done  {key} ({time.time()-rec['t0']:.0f}s)")
        return "ok"
    job["attempts"] += 1
    part.unlink(missing_ok=True)
    if job["attempts"] <= 2:
        job["est"] = min(job["est"] * 1.5, JOB_CAP)  # killed-by-rlimit? give headroom
        log(f"requeue {key} rc={rc} attempt={job['attempts']} est->{job['est']/GB:.1f}G")
        return "requeue"
    (ROOT / "out" / f"{key}.failed").write_text(f"rc={rc}\n")
    log(f"FAILED {key} rc={rc} after {job['attempts']} attempts")
    return "failed"


# --------------------------------------------------------- trunc validation
def run_sync(toml_path, out_path, est):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               RAYON_NUM_THREADS=str(NCORES),
               FERRIC_ERI3_BUDGET_GB=str(BUDGET / GB),
               FERRIC_TERF_TABLE_DIR=_TERF_DIR)
    with open(out_path, "w") as f:
        subprocess.run([BIN, str(toml_path)], stdout=f, stderr=subprocess.STDOUT,
                       env=env,
                       preexec_fn=make_preexec(int(min(est * 2 + 2 * GB, RLIMIT_MAX))),
                       timeout=TIMEOUT, cwd=ROOT)


def total_energy(path):
    m = re.search(r"Total energy\s*=\s*(-?\d+\.\d+)", open(path).read())
    return float(m.group(1)) if m else None


def validate_trunc():
    """Benzene fragment (s22-11 mA_cp, aDZ, coupled-rings): full-rank vs trunc."""
    vfile = ROOT / "trunc_validated.json"
    if vfile.exists() and json.loads(vfile.read_text()).get("pass"):
        return True
    xyz = ROOT / "geoms" / "s22-11_mA_cp.xyz"
    est, _, _ = estimate(
        [(t.split()[0], 0, 0, 0) for t in xyz.read_text().splitlines()[2:] if t.split()], "adz")
    fr_toml = ROOT / "toml" / "val_benzene_fullrank.toml"
    write_if_changed(fr_toml, toml_text(xyz, "adz", "cr02", trunc=False))
    fr_out = ROOT / "out" / "val_benzene_fullrank.out"
    # an externally launched full-rank run may already be in flight — wait
    while subprocess.run(["pgrep", "-f", "val_benzene_fullrank.toml"],
                         capture_output=True).returncode == 0:
        log("trunc validation: waiting for in-flight full-rank benzene run")
        time.sleep(30)
    if not (fr_out.exists() and "Total energy" in fr_out.read_text()):
        log("trunc validation: running full-rank benzene fragment (aDZ, coupled-rings)")
        run_sync(fr_toml, fr_out, est)
    tr_key = "s22-11_mA_cp_adz_cr02"
    tr_out = ROOT / "out" / f"{tr_key}.out"
    if not (tr_out.exists() and "Total energy" in tr_out.read_text()):
        log("trunc validation: running truncated benzene fragment")
        run_sync(ROOT / "toml" / f"{tr_key}.toml", ROOT / "out" / f"{tr_key}.out.part", est)
        (ROOT / "out" / f"{tr_key}.out.part").rename(tr_out)
    e_fr, e_tr = total_energy(fr_out), total_energy(tr_out)
    ok = e_fr is not None and e_tr is not None and abs(e_fr - e_tr) <= TRUNC_TOL
    json.dump({"fullrank": e_fr, "trunc": e_tr,
               "diff": None if None in (e_fr, e_tr) else e_tr - e_fr,
               "tol": TRUNC_TOL, "pass": ok}, open(vfile, "w"), indent=1)
    log(f"trunc validation: fullrank={e_fr} trunc={e_tr} pass={ok}")
    return ok


# ------------------------------------------------------------------ runner
def main():
    jobs, excluded = build_jobs()
    # Optional scope filter (2026-07-21): GRID_KEY_PREFIXES restricts the run
    # to jobs whose key starts with one of the given comma-separated prefixes
    # -- e.g. "a24" + "aqz" together via GRID_KEY_SUBSTR, used to launch just
    # the new A24-aqz jobs without also picking up S22/S66-aqz (much bigger
    # dimers, closer to JOB_CAP, not requested/validated yet at this basis).
    prefixes = [p for p in os.environ.get("GRID_KEY_PREFIXES", "").split(",") if p.strip()]
    substrs = [s for s in os.environ.get("GRID_KEY_SUBSTR", "").split(",") if s.strip()]
    if prefixes:
        jobs = [j for j in jobs if j["key"].startswith(tuple(prefixes))]
    if substrs:
        jobs = [j for j in jobs if all(s in j["key"] for s in substrs)]
    for key, g in excluded:
        log(f"excluded (est {g:.0f}G > cap): {key}")
    todo = [j for j in jobs if not is_done(j)
            and not (ROOT / "out" / f"{j['key']}.failed").exists()]
    log(f"jobs: {len(jobs)} total, {len(jobs)-len(todo)} done, "
        f"{len(todo)} to run, {len(excluded)} excluded")
    if "--dry-run" in sys.argv:
        for j in sorted(todo, key=lambda j: j["est"]):
            print(f"{j['key']:42s} est={j['est']/GB:5.1f}G nbf={j['nbf']:4d}")
        return
    # S22 rs jobs are gated on the benzene trunc validation, but DON'T block
    # the whole grid on it: defer those jobs and let A24 + scs flow. Rerun
    # the script after validation passes (idempotent) to pick them up.
    vfile = ROOT / "trunc_validated.json"
    validated = vfile.exists() and json.loads(vfile.read_text()).get("pass")
    if not validated:
        trunc_gated = lambda j: (j["key"].startswith(("s22", "s66"))
                                 and j["method"] != "scs")
        n_defer = sum(1 for j in todo if trunc_gated(j))
        log(f"trunc not yet validated: deferring {n_defer} S22/S66 rs jobs "
            f"(run validate_trunc + restart to admit them)")
        todo = [j for j in todo if not trunc_gated(j)]

    # priority: A24 aDZ, A24 aTZ (closes the directional story first), then
    # S22 aDZ/aTZ (the paper's headline π-stacked subset), then S66 last.
    # smallest-first within each class.
    # The paper's headline is the accuracy win on the large S22 π-stacks
    # (systems 11-15: benzene-PD, pyrazine, uracil-stack, indole-benzene-stack,
    # adenine-thymine-stack). They are the BIGGEST S22 aTZ jobs, so a plain
    # smallest-est-first ordering computes them LAST — exactly backwards for
    # closing the directional story. Float them ahead of the other S22 aTZ jobs.
    pistack = tuple(f"s22-{i:02d}" for i in range(11, 16))
    def prio(j):
        a24 = j["key"].startswith("a24")
        s66 = j["key"].startswith("s66")
        is_pistack = j["key"].startswith(pistack)
        if j["key"].startswith("s22-11") and j["basis"] == "adz":
            return 0  # benzene dimer: headline π-stack, surface early
        if a24:
            return 1 if j["basis"] == "adz" else 2
        if s66:
            return 7 if j["basis"] == "adz" else 8  # corroboration, last
        if is_pistack:
            return 3 if j["basis"] == "adz" else 4  # headline π-stacks first
        return 5 if j["basis"] == "adz" else 6      # other S22

    # Within the π-stack aTZ tier, finish ONE system before starting the next so
    # a complete CBS number lands first, instead of all five crawling forward in
    # lockstep by fragment size (which yields zero complete systems for hours).
    # Order: pyrazine (12, smallest) → benzene-PD (11) → the rest, ascending.
    pistack_order = {12: 0, 11: 1, 13: 2, 20: 3, 14: 4, 15: 5}
    def sys_rank(j):
        if j["key"].startswith(pistack):
            return pistack_order.get(int(j["key"].split("-")[1][:2]), 9)
        return 0
    pending = deque(sorted(todo, key=lambda j: (prio(j), sys_rank(j), j["est"])))
    running, n_done, n_fail = [], 0, 0
    while pending or running:
        # reap
        for rec in running[:]:
            if rec["proc"].poll() is not None:
                running.remove(rec)
                res = finish(rec)
                if res == "ok":
                    n_done += 1
                elif res == "requeue":
                    pending.appendleft(rec["job"])
                else:
                    n_fail += 1
            elif time.time() - rec["t0"] > TIMEOUT:
                log(f"TIMEOUT {rec['job']['key']}")
                kill_group(rec)
        # watchdog: live memory floor
        if running and mem_available() < FLOOR:
            victim = max(running, key=lambda r: r["job"]["est"])
            log(f"WATCHDOG: MemAvailable<{FLOOR/GB:.1f}G — killing {victim['job']['key']}")
            kill_group(victim)
        # admit. Two regimes (see PARALLELISM MODEL note):
        #  - small jobs (est < SOLO_GB): pack many at 1 BLAS thread each, so
        #    their serial SCF/MP2 phases overlap to fill cores via CONCURRENCY.
        #  - big jobs (est >= SOLO_GB): too large to pack >1, so they'd otherwise
        #    run alone leaving ~11 cores idle. Run them solo with all BLAS cores
        #    (OPENBLAS=NCORES, RAYON=1 — crash-safe, the single rayon worker
        #    can't trigger the dgetrf_parallel stack overflow). The big MP2
        #    G_i = B_i^T·B wide GEMMs then use the idle cores.
        # A solo-BLAS job and packed jobs never coexist: a big job waits for the
        # box to drain, and once it's running nothing else is admitted (it owns
        # all the cores). Both bounded by MEM_BUDGET + live MemAvailable headroom.
        used = sum(r["job"]["est"] for r in running)
        big_running = any(r["job"]["est"] >= SOLO_GB for r in running)
        # ANTI-STARVATION: walk pending in priority order. The first job that
        # does NOT fit right now becomes a reservation barrier — we only let
        # LOWER-priority jobs backfill if they still leave room for the barrier
        # job to be admitted as memory frees. Without this, a stream of small
        # low-priority jobs (e.g. S66) keeps backfilling the memory gap that a
        # big high-priority job (π-stack aTZ) needs, starving it to ~1-at-a-time.
        reserved = 0.0
        for job in list(pending):
            # aqz jobs always run solo at all 12 cores (Matt, 2026-07-21:
            # "use 12 cores per job") regardless of SOLO_GB -- overrides the
            # packed-concurrency default calibrated for aDZ/aTZ at this job
            # size (~10GB, near but under SOLO_GB). Uses rayon_threads=NCORES
            # with blas_threads=1 (parallelizes the RPA quad loop over
            # frequencies), not OPENBLAS threading -- the crash-safe solo axis
            # per the CRASH-SAFETY note below, same as any other solo job.
            big = job["est"] >= SOLO_GB or job["basis"] == "aqz"
            if big_running:
                break  # a solo-BLAS job owns the whole box; admit nothing else
            fits_mem = (mem_available() - reserved - job["est"] >= FLOOR + 1 * GB)
            if big:
                if running:
                    continue  # wait for the box to drain, then run solo-rayon
                if mem_available() - job["est"] >= FLOOR + 1 * GB:
                    pending.remove(job)
                    # Big solo jobs: OPENBLAS=1, RAYON=12
                    # RPA quad loop parallelizes over 12 frequencies with rayon
                    running.append(launch(job, blas_threads=1, rayon_threads=NCORES))
                    big_running = True
                    break
                # can't run the solo job yet; reserve for it so nothing backfills
                reserved += job["est"]
                continue
            if (len(running) < MAX_WORKERS
                    and used + job["est"] <= MEM_BUDGET
                    and fits_mem):
                # packed: OPENBLAS=1 (PACK_BLAS), concurrency fills cores. Memory
                # is the real cap (~3 on this box). BLAS threading here crashes
                # concurrent cr02 jobs — see CRASH-SAFETY note above.
                pending.remove(job)
                running.append(launch(job, blas_threads=PACK_BLAS, rayon_threads=1))
                used += job["est"]
            else:
                # This higher-priority job can't be admitted now; reserve its
                # footprint so subsequent lower-priority jobs don't take the
                # memory it's waiting for. (Only reserve when worker/mem-bound,
                # not when MAX_WORKERS is just full of equal-priority work.)
                if used + job["est"] > MEM_BUDGET or not fits_mem:
                    reserved += job["est"]
        json.dump(dict(pending=len(pending), running=[r["job"]["key"] for r in running],
                       done=n_done, failed=n_fail, t=time.time()),
                  open(ROOT / "status.json", "w"), indent=1)
        time.sleep(POLL)
    log(f"grid complete: {n_done} ok, {n_fail} failed this run")


if __name__ == "__main__":
    main()
