"""ROKS PBE0 exxdiv=none on the tri 4H s+p triplet: why ferric's staged-ewald first stage does not converge on CI
(FINDINGS "ROKS PBE0 CI non-convergence (Python diagnosis) — 2026-09-25").  Uses roks_replica.ferric_roks, a numpy
replica of ferric's injected ROHF/ROKS loop.

Hypotheses, stated BEFORE the runs:
  (a) -1.4348827058 (CI) is a genuine ROKS stationary state (aufbau-violating occupation) that a different DIIS
      trajectory converges to.  Predicts: some start converges to E ~ -1.43488 with orbital gradient -> 0 and a
      non-aufbau label pattern; a MOM-held survey of occupation patterns contains a state at -1.43488.
  (b) it is a non-stationary DIIS wander / limit cycle.  Predicts: no stationary state at -1.43488; the failing
      trajectories keep err_max ~ 1e-2..1e-1 for all 200 iterations with the energy drifting over a band, and WHICH
      energy is printed at iteration 200 depends on bit-level perturbations of the start (so -1.43488 is a snapshot).
  (c) something else (e.g. a stationary state the F6 gradient guard / witness keeps rejecting, or an open-space swap
      flip-flop the disabled continuity lock no longer stops).  Predicts: dE, dP small with err_max > 1e-3 repeatedly
      (guard reset loop), or witness restarts in the log, or dp_rms pinned O(0.1) with E constant.

Modes:
  perturb   baseline ferric replica from core (+ random antisymmetric rotation exp(t K), t in THETAS, seeds)
  states    MOM-held survey of occupation patterns (closed 1 of the lowest 6, open 2 of the rest)
  fixes     candidate fixes over the same start set; success = |E - E_REF| < 1e-8 and converged
Usage: python3 run_roks_trap.py [perturb|states|fixes] [nseeds]"""

import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

import roks_replica as R  # noqa: E402

E_REF = -1.465458280448  # prototype PBE0 none (run_roks_ssf_pins.py)
E_CI = -1.4348827058
ND, NO = 1, 2
THETAS = (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1)
_G = {}


def _init():
    _G["ints"] = R.tri_setup()


def core_mos(S, h):
    s, V = np.linalg.eigh(S)
    X = V @ np.diag(s ** -0.5) @ V.T
    return X @ np.linalg.eigh(X @ h @ X)[1]


def rotate(C, S, theta, seed):
    """C exp(theta K), K antisymmetric N(0,1) in the (orthonormal) MO basis: keeps C^T S C = 1."""
    if theta == 0:
        return C
    rng = np.random.default_rng(seed)
    n = C.shape[1]
    K = rng.standard_normal((n, n))
    K = (K - K.T) / 2
    return C @ expm(theta * K)


def starts(nseeds):
    out = [("core", 0.0, 0)]
    for t in THETAS[1:]:
        out += [(f"core+rot{t:g}", t, s) for s in range(nseeds)]
    return out


def tail_stats(hist, n=50):
    t = hist[-n:]
    e = np.array([x[1] for x in t])
    er = np.array([x[2] for x in t])
    dp = np.array([x[3] for x in t])
    return dict(emin=e.min(), emax=e.max(), errmin=er.min(), errmed=np.median(er), dpmed=np.median(dp))


def one(args):
    label, theta, seed, kw, c0kind = args
    S, h, enn, vm, jk, I, gr = _G["ints"]
    C = core_mos(S, h) if c0kind == "core" else _G[c0kind]
    C = rotate(C, S, theta, seed)
    t = time.time()
    r = R.ferric_roks(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, C0=C, **kw)
    a = R.analyse(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, r["C"])
    ts = tail_stats(r["hist"])
    return dict(label=label, theta=theta, seed=seed, exit=r["exit"], e=r["e"], it=r["it"], gmax=a["gmax"],
                labels=a["labels"], gap_a=a["gap_a"], gap_b=a["gap_b"], wit=len(r["witness_log"]),
                restarts=sum(1 for w in r["witness_log"] if w[0] == "restart"), secs=time.time() - t, **ts,
                ok=(r["exit"] == "Converged" and abs(r["e"] - E_REF) < 1e-8))


def fmt(d):
    return (f"{d['label']:18s} s{d['seed']} {d['exit']:12s} E {d['e']:.10f} it {d['it']:3d} gmax {d['gmax']:.1e} "
            f"{d['labels'][:6]} gapA {d['gap_a']:+.4f} gapB {d['gap_b']:+.4f} restarts {d['restarts']} | tail50 E "
            f"[{d['emin']:.5f},{d['emax']:.5f}] err min/med {d['errmin']:.1e}/{d['errmed']:.1e} dp med "
            f"{d['dpmed']:.1e} {'OK' if d['ok'] else '--'}")


def run_set(pool, kw, nseeds, c0kind="core", tag=""):
    jobs = [(lab, t, s, kw, c0kind) for lab, t, s in starts(nseeds)]
    res = pool.map(one, jobs)
    for d in res:
        print(f"  {tag} {fmt(d)}", flush=True)
    ok = sum(d["ok"] for d in res)
    print(f"  {tag} SUCCESS {ok}/{len(res)}", flush=True)
    return res


def mode_states():
    """MOM-held (mom_after_iter=1, guard off by construction) SCFs from every occupation pattern of the converged
    ground-state Roothaan orbitals: closed = 1 of the lowest 6, open = 2 of the lowest 6 minus the closed one."""
    _init()
    S, h, enn, vm, jk, I, gr = _G["ints"]
    g = R.ferric_roks(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, level_shift=0.25, max_iter=800, witness=False)
    assert g["converged"] and abs(g["e"] - E_REF) < 1e-8, g["e"]
    C0 = g["C"]
    print(f"ground state (ls 0.25) E {g['e']:.12f}; Roothaan eps[:8] {np.round(g['eps'][:8], 4)}", flush=True)
    seen = []
    for cl in range(6):
        for op in combinations([i for i in range(6) if i != cl], 2):
            order = [cl, *op] + [i for i in range(C0.shape[1]) if i not in (cl, *op)]
            C = C0[:, order]
            for ls in (0.0, 0.3):
                r = R.ferric_roks(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, C0=C, mom_after_iter=1, level_shift=ls,
                                  max_iter=300)
                a = R.analyse(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, r["C"])
                print(f"  occ closed {cl} open {op} ls {ls}: {r['exit']:8s} E {r['e']:.10f} it {r['it']:3d} "
                      f"gmax {a['gmax']:.1e} labels {a['labels'][:8]} gapA {a['gap_a']:+.4f} gapB {a['gap_b']:+.4f}"
                      f"  E-E_CI {r['e'] - E_CI:+.2e}", flush=True)
                if r["exit"] == "Converged" and a["gmax"] < 1e-6:
                    seen.append(round(r["e"], 7))
                    break
    print("distinct stationary energies:", sorted(set(seen)), flush=True)


FIXES = [
    ("F0 baseline (ferric today)", {}),
    ("F0' prototype DIIS error [F,Dt] (NOT ferric)", dict(err_kind="fds")),
    ("F1 level_shift 0.25 (existing knob)", dict(level_shift=0.25)),
    ("F1 level_shift 0.5 (existing knob)", dict(level_shift=0.5)),
    ("F1 level_shift 1.0 (existing knob)", dict(level_shift=1.0)),
    ("F2 mom_after_iter 20 (existing knob)", dict(mom_after_iter=20)),
    ("F3 damp 0.5 first 20 it (NEW code)", dict(damp=0.5, damp_iters=20)),
    ("F4 diis_size 4 (existing knob)", dict(diis_size=4)),
    ("F5 diis_size 12 (existing knob)", dict(diis_size=12)),
    ("F5b diis_size 16 (existing knob)", dict(diis_size=16)),
    ("F5c diis_size 20 (existing knob)", dict(diis_size=20)),
    ("F5d diis_size 16, max_iter 400 (existing knobs)", dict(diis_size=16, max_iter=400)),
    ("F6 max_iter 400 (existing knob)", dict(max_iter=400)),
    ("F9 level_shift 0.25 for it<=30 then off (NEW: shift window)", dict(level_shift=0.25, shift_until=30)),
    ("F9 level_shift 0.25 for it<=30 then off + DIIS reset (NEW)", dict(level_shift=0.25, shift_until=30,
                                                                      reset_diis_after_shift=True)),
    ("F9 level_shift 0.5 for it<=30 then off (NEW: shift window)", dict(level_shift=0.5, shift_until=30)),
    ("F10 level_shift 0.25, max_iter 600 (existing knobs)", dict(level_shift=0.25, max_iter=600)),
    ("F11 level_shift 0.25, ramp err/(err+1e-1) (NEW constant)", dict(level_shift=0.25, shift_damp_err=1e-1)),
    ("F12 level_shift 0.25 latched off at err<1e-2 (NEW)", dict(level_shift=0.25, shift_off_below=1e-2)),
    ("F12 level_shift 0.25 latched off at err<3e-3 (NEW)", dict(level_shift=0.25, shift_off_below=3e-3)),
    ("F12 level_shift 0.5 latched off at err<3e-3 (NEW)", dict(level_shift=0.5, shift_off_below=3e-3)),
    ("F13 level_shift 0.25 latched off at err<3e-3 + diis_size 12 (NEW)", dict(level_shift=0.25,
                                                                            shift_off_below=3e-3, diis_size=12)),
    ("F14 level_shift 0.05, max_iter 600 (existing knobs)", dict(level_shift=0.05, max_iter=600)),
    ("F14 level_shift 0.1, max_iter 600 (existing knobs)", dict(level_shift=0.1, max_iter=600)),
    ("F15 level_shift 0.1, diis_size 12, max_iter 600 (existing knobs)", dict(level_shift=0.1, diis_size=12,
                                                                             max_iter=600)),
    ("F15 level_shift 0.25, diis_size 12, max_iter 600 (existing knobs)", dict(level_shift=0.25, diis_size=12,
                                                                              max_iter=600)),
    ("F16 CONSTANT level_shift 0.25, no ramp (NEW: ramp off)", dict(level_shift=0.25, shift_damp_err=0.0)),
    ("F16 CONSTANT level_shift 0.5, no ramp (NEW: ramp off)", dict(level_shift=0.5, shift_damp_err=0.0)),
    ("F16 CONSTANT level_shift 0.1, no ramp (NEW: ramp off)", dict(level_shift=0.1, shift_damp_err=0.0)),
    ("F17 level_shift 0.02, max_iter 600 (existing knobs)", dict(level_shift=0.02, max_iter=600)),
    ("F17 level_shift 0.03, max_iter 600 (existing knobs)", dict(level_shift=0.03, max_iter=600)),
    ("F17 level_shift 0.05, diis_size 12, max_iter 600 (existing knobs)", dict(level_shift=0.05, diis_size=12,
                                                                              max_iter=600)),
    ("F8 continuity lock ON (pre-2026-09-25 injected behaviour)", dict(lock=True)),
]


def mode_fixes(nseeds, which):
    with Pool(2, initializer=_init) as pool:
        for name, kw in FIXES:
            if which and not any(name.startswith(w) for w in which):
                continue
            print(f"== {name}  kw={kw}", flush=True)
            run_set(pool, kw, nseeds, tag=name.split()[0])


def mode_semilocal_start(nseeds):
    """F7: initial_mos = the converged ROKS LDA / PBE (none) MOs (existing knob: GammaRoksConfig.initial_mos; in
    run_staged it would be a new pre-stage).  Perturbed the same way."""
    _init()
    S, h, enn, vm, jk, I, gr = _G["ints"]
    for xc in ("LDA,VWN", "PBE"):
        r = R.ferric_roks(S, h, jk, enn, ND, NO, gr, xc, 0.0)
        print(f"{xc} none: {r['exit']} E {r['e']:.12f} it {r['it']}", flush=True)
        _G[xc] = r["C"]
    np.save(os.environ["ROKS_TRAP_CACHE"] + ".sl.npy", np.stack([_G["LDA,VWN"], _G["PBE"]]))

    def init2():
        _init()
        z = np.load(os.environ["ROKS_TRAP_CACHE"] + ".sl.npy")
        _G["LDA,VWN"], _G["PBE"] = z[0], z[1]

    with Pool(2, initializer=init2) as pool:
        for xc in ("LDA,VWN", "PBE"):
            for tag, kw in (("F7", {}), ("F7+ls0.5", dict(level_shift=0.5))):
                print(f"== {tag} start from converged ROKS {xc} none MOs kw={kw}", flush=True)
                jobs = [(lab.replace("core", xc), t, s, kw, xc) for lab, t, s in starts(nseeds)]
                res = pool.map(one, jobs)
                for d in res:
                    print(f"  {tag} {fmt(d)}", flush=True)
                print(f"  {tag} {xc} SUCCESS {sum(d['ok'] for d in res)}/{len(res)}", flush=True)


def mode_validate(ls):
    """The recommended knob on the other functionals and on the ewald stage: LDA,VWN / PBE none from core (+rot) must
    still reach their pins; PBE0 staged (none -> ewald from the none MOs) must give the ewald pin."""
    _init()
    S, h, enn, vm, jk, I, gr = _G["ints"]
    pins = {"LDA,VWN": -1.694206925329, "PBE": -1.731150286924}
    for xc, ref in pins.items():
        for t, seed in ((0.0, 0), (1e-9, 0), (1e-3, 1), (1e-1, 2)):
            for lsx in (0.0, ls):
                r = R.ferric_roks(S, h, jk, enn, ND, NO, gr, xc, 0.0, C0=rotate(core_mos(S, h), S, t, seed),
                                  level_shift=lsx, max_iter=600)
                print(f"  {xc:8s} none rot {t:g} s{seed} ls {lsx}: {r['exit']} E {r['e']:.12f} it {r['it']} "
                      f"dE {r['e'] - ref:+.1e}", flush=True)
    r = R.ferric_roks(S, h, jk, enn, ND, NO, gr, "PBE0", 0.0, level_shift=ls, max_iter=600)
    r2 = R.ferric_roks(S, h, jk, enn, ND, NO, gr, "PBE0", vm, C0=r["C"], level_shift=ls, max_iter=600)
    print(f"  PBE0 staged ls {ls}: none {r['exit']} {r['e']:.12f} it {r['it']} | ewald {r2['exit']} {r2['e']:.12f} "
          f"it {r2['it']} dE vs pin {r2['e'] - (-1.776676719941):+.1e}; ewald-none {r2['e'] - r['e']:.12f} vs "
          f"-a vM N/2 {-0.25 * vm * 2:.12f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "perturb"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 and mode != "validate" else 3
    if mode == "perturb":
        with Pool(2, initializer=_init) as pool:
            print("== baseline ferric replica, PBE0 exxdiv=none, core guess (+ rotations)", flush=True)
            run_set(pool, {}, nseeds, tag="F0")
    elif mode == "states":
        mode_states()
    elif mode == "fixes":
        mode_fixes(nseeds, sys.argv[3:])
    elif mode == "validate":
        mode_validate(float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
    elif mode == "semilocal":
        mode_semilocal_start(nseeds)
