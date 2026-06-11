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
 4. jobs whose estimate exceeds JOB_CAP are excluded up front (logged) —
    this is what keeps e.g. AT-stacked/aTZ (est ~50 GB) off a 23 GB box
 5. jobs with est > EXCLUSIVE_GB run alone with all cores

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
MEM_BUDGET = 14 * GB      # sum of running estimates
FLOOR = 2.5 * GB          # MemAvailable floor -> watchdog kills largest job
JOB_CAP = 12 * GB         # estimates above this are infeasible on this box
EXCLUSIVE_GB = 6 * GB     # bigger than this -> run alone with all cores
MAX_WORKERS = 4
RAYON = 3                 # 4 workers x 3 threads = 12 cores
RAYON_EXCLUSIVE = 12
CAL, BASE = 2.0, 0.5 * GB  # est = BASE + CAL * naux*nbf^2*8 (empirical)
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
    """All atoms (incl. ghosts) carry basis functions."""
    nbf = sum(NBF[basis][Z[s.lstrip('@')]] for s, *_ in atoms)
    naux = sum(NAUX[basis][Z[s.lstrip('@')]] for s, *_ in atoms)
    return BASE + CAL * naux * nbf * nbf * 8, nbf, naux


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


def toml_text(xyz, basis, method, trunc):
    obs, aux = BASES[basis]
    t = f'[molecule]\nxyz = "{xyz}"\n\n[basis]\nname = "{obs}"\n\n'
    t += '[scf]\ndf_j_aux = "def2-universal-jkfit"\ndf_k_aux = "def2-universal-jkfit"\n\n'
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
    json.dump({"a24": a24_refs, "s22": s22_refs}, open(ROOT / "refs.json", "w"), indent=1)

    ghost = lambda atoms: [("@" + s, x, y, z) for s, x, y, z in atoms]
    jobs, excluded = [], []
    for dbse, geos in (("a24", a24_geos), ("s22", s22_geos)):
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
                        trunc = (dbse == "s22" and method != "scs")
                        if est > JOB_CAP:
                            excluded.append((key, est / GB))
                            continue
                        tt = toml_text(xyz_path, basis, method, trunc)
                        if write_if_changed(ROOT / "toml" / f"{key}.toml", tt):
                            # settings drift invalidates any prior result
                            (ROOT / "out" / f"{key}.out").unlink(missing_ok=True)
                            (ROOT / "out" / f"{key}.failed").unlink(missing_ok=True)
                        jobs.append(dict(key=key, est=est, nbf=nbf, naux=naux,
                                         method=method, basis=basis, attempts=0,
                                         exclusive=est > EXCLUSIVE_GB))
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
    return fn


def launch(job, rayon):
    key = job["key"]
    limit = int(job["est"] * 2 + 2 * GB)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               RAYON_NUM_THREADS=str(rayon))
    part = open(ROOT / "out" / f"{key}.out.part", "w")
    err = open(ROOT / "out" / f"{key}.err", "w")
    proc = subprocess.Popen([BIN, str(ROOT / "toml" / f"{key}.toml")],
                            stdout=part, stderr=err, env=env,
                            preexec_fn=make_preexec(limit), cwd=ROOT)
    log(f"start {key} est={job['est']/GB:.1f}G nbf={job['nbf']} rayon={rayon} pid={proc.pid}")
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
        job["exclusive"] = job["est"] > EXCLUSIVE_GB
        log(f"requeue {key} rc={rc} attempt={job['attempts']} est->{job['est']/GB:.1f}G")
        return "requeue"
    (ROOT / "out" / f"{key}.failed").write_text(f"rc={rc}\n")
    log(f"FAILED {key} rc={rc} after {job['attempts']} attempts")
    return "failed"


# --------------------------------------------------------- trunc validation
def run_sync(toml_path, out_path, est):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               RAYON_NUM_THREADS=str(RAYON_EXCLUSIVE))
    with open(out_path, "w") as f:
        subprocess.run([BIN, str(toml_path)], stdout=f, stderr=subprocess.STDOUT,
                       env=env, preexec_fn=make_preexec(int(est * 2 + 2 * GB)),
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
            print(f"{j['key']:42s} est={j['est']/GB:5.1f}G nbf={j['nbf']:4d}"
                  f"{'  EXCLUSIVE' if j['exclusive'] else ''}")
        return
    if any(j["key"].startswith("s22") and j["method"] != "scs" for j in todo):
        if not validate_trunc():
            log("ABORT: trunc validation failed — not admitting S22 rs jobs")
            todo = [j for j in todo if not (j["key"].startswith("s22")
                                            and j["method"] != "scs")]
    todo = [j for j in todo if not is_done(j)]  # validation may have done one

    # smallest first within basis; aDZ before aTZ so the matrix fills usefully
    pending = deque(sorted(todo, key=lambda j: (j["basis"] != "adz", j["est"])))
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
        # admit
        used = sum(r["job"]["est"] for r in running)
        for job in list(pending):
            if job["exclusive"]:
                if running:
                    continue
                pending.remove(job)
                running.append(launch(job, RAYON_EXCLUSIVE))
                break
            if (len(running) < MAX_WORKERS
                    and not any(r["job"]["exclusive"] for r in running)
                    and used + job["est"] <= MEM_BUDGET
                    and mem_available() - job["est"] >= FLOOR + 1 * GB):
                pending.remove(job)
                running.append(launch(job, RAYON))
                used += job["est"]
        json.dump(dict(pending=len(pending), running=[r["job"]["key"] for r in running],
                       done=n_done, failed=n_fail, t=time.time()),
                  open(ROOT / "status.json", "w"), indent=1)
        time.sleep(POLL)
    log(f"grid complete: {n_done} ok, {n_fail} failed this run")


if __name__ == "__main__":
    main()
