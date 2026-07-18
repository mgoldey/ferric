#!/usr/bin/env python3
"""ACONF ω-scan for BOTH rs-mp2-rpa formulations via the ferric CLI binary.

Why this exists (vs run_aconf.py): the installed Python .so is stale (no
run_rs_mp2_rpa), and the June-10 README numbers came from a pre-fix build that
is no longer reproducible (65 µHa total-energy offset on B_T@0.42 vs the
current binary, quadrature- and JK-independent). So the scan is RE-RUN in full
with the current target/release/ferric-cli, both formulations
(B = delta-lr, T = coupled-rings), on the SAME ω grid as the S22 stack sweeps
{0.2, 0.3, 0.42, 0.55, 0.673, 0.8} for direct comparison. The RI-MP2 baseline
is derived from each B output (Total − E_corr(B) + E(MP2,Coulomb)), so no
separate MP2 runs are needed.

Conventions: cc-pVDZ / cc-pVDZ-RI, frozen_core = 0 (all-electron — matches the
prior README table's Python-default convention), full-rank dRPA
(trunc_thresh = 0), n_points = 12 (quadrature-converged: 8..24 identical to
1e-10 on B_T), OPENBLAS/RAYON = 1 per job.

Idempotent (skips outputs already carrying "Total energy"); timestamped log;
memory-gated. Writes ACONF_RSSCAN.md + aconf_rsscan.json.
"""
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(ROOT, "target", "release", "ferric-cli")
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
KCAL = 627.509474

REACTIONS = [
    ("B_T", "B_G", 0.598), ("P_TT", "P_TG", 0.614), ("P_TT", "P_GG", 0.961),
    ("P_TT", "P_GX", 2.813), ("H_ttt", "H_gtt", 0.595), ("H_ttt", "H_tgt", 0.604),
    ("H_ttt", "H_tgg", 0.934), ("H_ttt", "H_gtg", 1.178), ("H_ttt", "H_g+t+g-", 1.302),
    ("H_ttt", "H_ggg", 1.250), ("H_ttt", "H_g+x-t+", 2.632), ("H_ttt", "H_t+g+x-", 2.740),
    ("H_ttt", "H_g+x-g-", 3.283), ("H_ttt", "H_x+g-g-", 3.083), ("H_ttt", "H_x+g-x+", 4.925),
]
NAMES = sorted({n for r in REACTIONS for n in r[:2]})
OMEGAS = [0.2, 0.3, 0.42, 0.55, 0.673, 0.8]
FORMS = [("delta-lr", "B"), ("coupled-rings", "T")]
CONC = int(os.environ.get("ACONF_JOBS", "3"))
MEM_GATE_GB = float(os.environ.get("ACONF_MEM_GB", "4"))

TOTAL_PAT = r'Total energy\s*=\s*(-?[0-9.]+)'
MP2_PAT = r'E\(MP2, Coulomb\)\s*=\s*(-?[0-9.]+)'
BCORR_PAT = r'E_corr Δ-form \(B\)\s*=\s*(-?[0-9.]+)'
BCORR_PAT_ASCII = r'E_corr .-form \(B\)\s*=\s*(-?[0-9.]+)'


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def mem_gb():
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemAvailable"):
                    return int(l.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def key(name, omega, ftag):
    safe = name.replace("+", "p").replace("-", "m")
    return f"aconf_{safe}_w{omega}_{ftag}"


def toml_for(name, omega, form):
    return f"""[molecule]
xyz = "{HERE}/xyz/{name}.xyz"
[basis]
name = "cc-pvdz"
[scf]
df_j_aux = "def2-universal-jkfit"
df_k_aux = "def2-universal-jkfit"
max_iter = 400
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "cc-pvdz-ri"
omega = {omega}
formulation = "{form}"
frozen_core = 0
[rpa]
trunc_thresh = 0.0
[quadrature]
n_points = 12
"""


def run_one(job):
    name, omega, form, ftag = job
    k = key(name, omega, ftag)
    op = f"{HERE}/out/{k}.out"
    if os.path.exists(op) and "Total energy" in open(op).read():
        return k, "skip", 0.0
    while mem_gb() < MEM_GATE_GB:
        log(f"[gate] {k}: {mem_gb():.1f}GB free; waiting 60s")
        time.sleep(60)
    open(f"{HERE}/toml/{k}.toml", "w").write(toml_for(name, omega, form))

    def _oom():
        try:
            with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
                f.write("1000")
        except Exception:
            pass

    t0 = time.monotonic()
    try:
        with open(op, "w") as f, open(op + ".err", "w") as e:
            subprocess.run([BIN, f"{HERE}/toml/{k}.toml"], stdout=f, stderr=e,
                           env=ENV, timeout=7200, preexec_fn=_oom)
    except subprocess.TimeoutExpired:
        return k, "TIMEOUT", time.monotonic() - t0
    ok = os.path.exists(op) and "Total energy" in open(op).read()
    return k, ("ok" if ok else "FAIL"), time.monotonic() - t0


def grab(k, pat):
    p = f"{HERE}/out/{k}.out"
    if not os.path.exists(p):
        return None
    m = re.search(pat, open(p).read())
    return float(m.group(1)) if m else None


def stats(errs):
    mae = sum(abs(e) for e in errs) / len(errs)
    md = sum(errs) / len(errs)
    rmsd = (sum(e * e for e in errs) / len(errs)) ** 0.5
    mx = max(errs, key=abs)
    return mae, md, rmsd, mx


def analyze():
    rows, data = [], {}
    # derived RI-MP2 baseline from the B outputs at the first omega
    for method_label, ftag, omega_list in (
            [("RI-MP2 (derived)", "B", [OMEGAS[0]])] +
            [(f"{ftag}", ftag, OMEGAS) for _, ftag in FORMS]):
        for omega in omega_list:
            e = {}
            for n in NAMES:
                k = key(n, omega, ftag)
                tot = grab(k, TOTAL_PAT)
                if method_label.startswith("RI-MP2"):
                    mp2 = grab(k, MP2_PAT)
                    bcorr = grab(k, BCORR_PAT) or grab(k, BCORR_PAT_ASCII)
                    e[n] = (tot - bcorr + mp2) if None not in (tot, bcorr, mp2) else None
                else:
                    e[n] = tot
            if any(v is None for v in e.values()):
                rows.append((method_label, omega, None))
                continue
            errs = [((e[p] - e[r]) * KCAL - ref) for r, p, ref in REACTIONS]
            rows.append((method_label, omega, stats(errs)))
            data[f"{method_label}|{omega}"] = {
                "energies": e,
                "errors": dict(zip([f"{r}->{p}" for r, p, _ in REACTIONS], errs)),
            }
    L = ["# ACONF ω-scan, both formulations — cc-pVDZ (CLI re-run)\n",
         "15 reactions vs W1h-val CCSD(T)/CBS; kcal/mol; frozen_core=0,",
         "full-rank dRPA, current binary (see run_aconf_cli.py header for why",
         "this supersedes the June-10 Python-API table).\n",
         "| method | ω (Å⁻¹) | MAE | MD | RMSD | MAX |", "|---|---|---|---|---|---|"]
    for label, omega, s in rows:
        if s is None:
            L.append(f"| {label} | {omega} | — | — | — | — |")
        else:
            mae, md, rmsd, mx = s
            om = "—" if label.startswith("RI-MP2") else omega
            L.append(f"| {label} | {om} | {mae:.3f} | {md:+.3f} | {rmsd:.3f} | {mx:+.3f} |")
    open(f"{HERE}/ACONF_RSSCAN.md", "w").write("\n".join(L) + "\n")
    json.dump(data, open(f"{HERE}/aconf_rsscan.json", "w"), indent=1)
    print("\n".join(L))


def main():
    os.makedirs(f"{HERE}/toml", exist_ok=True)
    os.makedirs(f"{HERE}/out", exist_ok=True)
    jobs = [(n, omega, form, ftag) for omega in OMEGAS
            for form, ftag in FORMS for n in NAMES]
    todo = [j for j in jobs if not (
        os.path.exists(f"{HERE}/out/{key(j[0], j[1], j[3])}.out")
        and "Total energy" in open(f"{HERE}/out/{key(j[0], j[1], j[3])}.out").read())]
    log(f"{len(jobs)} jobs total, {len(todo)} to run, conc={CONC}")
    done = 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = {ex.submit(run_one, j): j for j in todo}
        for fut in as_completed(futs):
            k, status, dt = fut.result()
            done += 1
            log(f"[{done}/{len(todo)}] {status:8s} {dt:6.1f}s  {k}")
    analyze()
    log("done.")


if __name__ == "__main__":
    main()
