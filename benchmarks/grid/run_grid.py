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

GB = 1e9
NCORES = 12               # box has 12 cores
MEM_BUDGET = 21 * GB      # sum of running estimates (concurrent jobs); with the
                          # AO_MULT-calibrated estimate (~6.8 GB for an nbf~500
                          # aTZ job) this admits 3 concurrent. The live
                          # MemAvailable guard + watchdog + RLIMIT_AS are the
                          # real safety net, not this sum.
FLOOR = 2.5 * GB          # MemAvailable floor -> watchdog kills largest job
JOB_CAP = 17 * GB         # above this: infeasible even run alone (box: ~19G avail)
RLIMIT_MAX = 18 * GB      # absolute per-child address-space ceiling
MAX_WORKERS = NCORES      # don't oversubscribe cores
SOLO_GB = 10 * GB         # est >= this -> run alone with all BLAS cores (can't
                          # pack >1, so use BLAS threading instead of idling ~11
                          # cores). Captures the nbf=1127 S22 aTZ dimers (~11 GB).
PACK_BLAS = 3             # BLAS threads per packed job: 3 jobs x 3 = 9 < NCORES,
                          # leaving cores for the dRPA rayon burst + OS.
# PARALLELISM MODEL (calibrated 2026-06-14 on s22-17_mB_cp_atz_dlr042, nbf=506).
# The lever is BLAS threads, NOT rayon threads. Rayon-threading a single job is a
# no-op (1 rayon thread 173.6s vs 6 threads 175.1s — the only rayon-parallel
# stage, the dRPA quad par_iter over ~16 ω, is a small fraction of wall time).
# But the SCF (DF-JK) and MP2 G_i = B_i^T·B stages are big BLAS3 GEMMs, and
# OpenBLAS threads DO speed them: OPENBLAS=3 RAYON=1 ran the calibration job in
# 110s vs 174s even while contending with 3 other jobs (~1.6x). So:
#  * SMALL jobs pack ~3-up, each at OPENBLAS=PACK_BLAS, RAYON=1. Concurrency
#    overlaps their disk-spill I/O stalls; BLAS threads fill cores during the
#    GEMM-bound compute phases. (Memory caps packing at 3 on this 23 GB box —
#    a 4th would breach FLOOR — so BLAS threading, not more packing, is the win.)
#  * BIG jobs (est >= SOLO_GB) can't pack, so they run solo at OPENBLAS=NCORES,
#    RAYON=1. Same GEMMs, all 12 cores. Verified crash-safe (OPENBLAS=12 RAYON=1,
#    exit 0, valid energy) 2026-06-14.
# CRASH-SAFETY: OPENBLAS>1 is ONLY ever paired with RAYON=1 (enforced by assert
# in launch()). Multiple concurrent rayon workers each calling parallel OpenBLAS
# LU overflow rayon's fixed 2 MB worker stacks -> dgetrf_parallel SIGSEGV
# (gdb-verified, a24-04). One rayon worker can't race itself, so it's safe.
# This is a STACK overflow, not OOM — the memory machinery cannot catch it,
# which is exactly why the invariant is enforced structurally.
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
TRUNC = 1e-4
TRUNC_TOL = 2e-5
K = 627.509474

BASES = {"adz": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
         "atz": ("aug-cc-pvtz", "aug-cc-pvtz-rifit")}
METHODS = ("scs", "dlr042", "cr02")
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
        if trunc:
            t += f'\n[rpa]\ntrunc_thresh = {TRUNC}\n'
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
               RAYON_NUM_THREADS=str(rayon_threads), FERRIC_ERI3_BUDGET_GB=str(BUDGET / GB))
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
               RAYON_NUM_THREADS=str(THREADS),
               FERRIC_ERI3_BUDGET_GB=str(BUDGET / GB))
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
    def prio(j):
        a24 = j["key"].startswith("a24")
        s66 = j["key"].startswith("s66")
        if j["key"].startswith("s22-11") and j["basis"] == "adz":
            return 0  # benzene dimer: headline π-stack, surface early
        if a24:
            return 1 if j["basis"] == "adz" else 2
        if s66:
            return 6 if j["basis"] == "adz" else 7  # corroboration, last
        return 3 if j["basis"] == "adz" else 4      # S22
    pending = deque(sorted(todo, key=lambda j: (prio(j), j["est"])))
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
        for job in list(pending):
            big = job["est"] >= SOLO_GB
            if big_running:
                break  # a solo-BLAS job owns the whole box; admit nothing else
            if big:
                if running:
                    continue  # wait for the box to drain, then run solo-BLAS
                if mem_available() - job["est"] >= FLOOR + 1 * GB:
                    pending.remove(job)
                    running.append(launch(job, blas_threads=NCORES, rayon_threads=1))
                    big_running = True
                    break
            elif (len(running) < max(1, NCORES // PACK_BLAS)
                    and used + job["est"] <= MEM_BUDGET
                    and mem_available() - job["est"] >= FLOOR + 1 * GB):
                # packed: BLAS-threaded (crash-safe at rayon=1), concurrency
                # overlaps I/O. Worker cap keeps total BLAS threads <= NCORES.
                pending.remove(job)
                running.append(launch(job, blas_threads=PACK_BLAS, rayon_threads=1))
                used += job["est"]
        json.dump(dict(pending=len(pending), running=[r["job"]["key"] for r in running],
                       done=n_done, failed=n_fail, t=time.time()),
                  open(ROOT / "status.json", "w"), indent=1)
        time.sleep(POLL)
    log(f"grid complete: {n_done} ok, {n_fail} failed this run")


if __name__ == "__main__":
    main()
