#!/usr/bin/env python3
"""ISOLATED ω-sweep diagnostic for the pyrazine dimer (S22-12).

Question: pyrazine's parallel-displaced π-stack overbinds by −2.58 kcal/mol at
MP2/CBS, and LRC[E] at the A24-tuned ω=0.2 Å⁻¹ removes only ~23%. Is that an
ω-tuning shortfall (the scale law says the LRC[E] correction grows with ω) or a
ceiling of range-separated screening? Sweep ω and see whether the residual
closes.

ISOLATION: separate dir (benchmarks/omega_diag/), separate toml/out, never
touches benchmarks/grid/. Runs ONE job at a time at modest resources so it
coexists with the production grid without tripping its memory watchdog. Reuses
the grid's already-computed ω-independent pieces? No — fully self-contained;
the grid geometries are read-only shared inputs.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRID = ROOT / "../grid"
BIN = str((ROOT / "../../target/release/ferric-cli").resolve())
GEOMS = GRID / "geoms"           # read-only: reuse the exact CP geometries
K = 627.509474

# ω values in Å⁻¹ (CLI takes Å⁻¹). 0.2 = A24 LRC[E] optimum (baseline);
# sweep up to test whether a larger ω covers pyrazine's big overbinding.
OMEGAS = [0.2, 0.3, 0.4, 0.5, 0.7]
BASES = {"adz": ("aug-cc-pvdz", "aug-cc-pvdz-rifit"),
         "atz": ("aug-cc-pvtz", "aug-cc-pvtz-rifit")}
FRAGS = ("dimer", "mA_cp", "mB_cp")
REF = -4.25   # S22B reference (kcal/mol)


def toml_text(xyz, basis, omega):
    obs, aux = BASES[basis]
    return (f'[molecule]\nxyz = "{xyz}"\n\n[basis]\nname = "{obs}"\n\n'
            f'[scf]\ndf_j_aux = "def2-universal-jkfit"\n'
            f'df_k_aux = "def2-universal-jkfit"\nmax_iter = 400\n\n'
            f'[method]\nkind = "rs-mp2-rpa"\n\n'
            f'[mp2]\nauxbasis = "{aux}"\nomega = {omega}\n'
            f'formulation = "coupled-rings"\n')


def grab(text, label):
    m = re.search(re.escape(label) + r"\s*=\s*(-?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def grid_running_count():
    """How many jobs the production grid is currently running (its status.json).
    Returns a large number if unreadable, so we conservatively wait."""
    try:
        d = json.loads((GRID / "status.json").read_text())
        return len(d.get("running", []))
    except Exception:
        return 99


# The production grid OWNS the box; this diagnostic only fills genuine gaps.
# A single MemAvailable reading is unreliable (it flickers as the 2-3 grid jobs
# pass through their dRPA memory peaks — the first attempt passed the gate on a
# transient high reading and then tripped the grid watchdog). So gate on the
# grid's actual occupancy: launch a sweep job ONLY when the grid is running
# ≤1 job AND memory is comfortably clear. The grid packs 2 big jobs at its
# MEM_BUDGET=18G, so "≤1 grid job" means there's real room for one ~8G sweep
# job without contending for the second grid slot's memory.
NEED_GB = 11.0          # require this much headroom (sweep aTZ peaks ~7-8G)
MAX_GRID_JOBS = 1       # only run when the grid is at ≤1 job


def wait_for_slot():
    while True:
        g = grid_running_count()
        m = mem_available_gb()
        if g <= MAX_GRID_JOBS and m >= NEED_GB:
            return
        print(f"[{time.strftime('%H:%M:%S')}] waiting (grid={g} jobs, "
              f"avail {m:.1f}G; need grid≤{MAX_GRID_JOBS} & ≥{NEED_GB}G) — grid has priority",
              flush=True)
        time.sleep(60)


def run(key, xyz, basis, omega):
    """Run one fragment/basis/ω; return (E_MP2[Coulomb]+rhf, E_T_total, rhf)."""
    out = ROOT / "out" / f"{key}.out"
    if not (out.exists() and "Total energy" in out.read_text()):
        tt = toml_text(xyz, basis, omega)
        tp = ROOT / "toml" / f"{key}.toml"
        tp.write_text(tt)
        wait_for_slot()   # defer to the grid; never starve it
        # modest resources: 1 job, OPENBLAS=1, RAYON=3 — coexists with the grid.
        env = dict(__import__("os").environ, OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", RAYON_NUM_THREADS="3",
                   FERRIC_ERI3_BUDGET_GB="2.0")
        with open(out, "w") as f:
            subprocess.run([BIN, str(tp)], stdout=f, stderr=subprocess.STDOUT,
                           env=env, timeout=4 * 3600, cwd=ROOT)
    t = out.read_text()
    tot = grab(t, "Total energy")            # = rhf + e_corr_T
    tc = grab(t, "E_corr coupled (T)")       # the coupled-rings correlation
    mp2c = grab(t, "E(MP2, Coulomb)")        # E_MP2[Coulomb] correlation
    if None in (tot, tc, mp2c):
        raise RuntimeError(f"{key}: parse failed (tot={tot} tc={tc} mp2c={mp2c})")
    rhf = tot - tc                           # no "RHF energy" line is printed
    return dict(rhf=rhf, mp2_tot=rhf + mp2c, t_tot=tot)


def cbs_corr(adz, atz, key):
    """corr two-point (27·aTZ − 8·aDZ)/19 per fragment; HF@aTZ. Returns the
    fragment total for a given correlated method ('mp2_tot' or 't_tot')."""
    ca = adz[key] - adz["rhf"]
    ct = atz[key] - atz["rhf"]
    return atz["rhf"] + (27 * ct - 8 * ca) / 19


def main():
    results = []
    for omega in OMEGAS:
        fe = {}
        for frag in FRAGS:
            xyz = GEOMS / f"s22-12_{frag}.xyz"
            per = {}
            for basis in BASES:
                key = f"pyr_{frag}_{basis}_w{int(round(omega*100)):03d}"
                print(f"[{time.strftime('%H:%M:%S')}] run {key}", flush=True)
                per[basis] = run(key, xyz, basis, omega)
            fe[frag] = per
        # interaction energies (kcal/mol): dimer − mA − mB, per method, at CBS
        def eint(method):
            e = 0.0
            for frag, sign in (("dimer", 1), ("mA_cp", -1), ("mB_cp", -1)):
                e += sign * cbs_corr(fe[frag]["adz"], fe[frag]["atz"], method)
            return e * K
        mp2 = eint("mp2_tot")
        t = eint("t_tot")
        row = dict(omega=omega, mp2=mp2, t=t,
                   mp2_err=mp2 - REF, t_err=t - REF,
                   reduction_pct=100 * (abs(mp2 - REF) - abs(t - REF)) / abs(mp2 - REF))
        results.append(row)
        print(f"  ω={omega}: MP2={mp2:.3f} (err {mp2-REF:+.3f})  "
              f"T={t:.3f} (err {t-REF:+.3f})  reduction {row['reduction_pct']:.0f}%",
              flush=True)
        json.dump(results, open(ROOT / "sweep_results.json", "w"), indent=1)
    print(f"\nref = {REF} kcal/mol")
    print("ω      MP2_err   T_err   reduction")
    for r in results:
        print(f"{r['omega']:.2f}   {r['mp2_err']:+.3f}   {r['t_err']:+.3f}   {r['reduction_pct']:.0f}%")


if __name__ == "__main__":
    main()
