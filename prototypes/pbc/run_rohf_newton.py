"""Iteration 24 driver: second-order (TRAH) ROHF / ROKS on the injected periodic path (pbc_rohf_newton.py).
Usage: OPENBLAS_NUM_THREADS=1 python3 run_rohf_newton.py {derivs|anchor|target|trapped|occ|mirror|diis|pbe0|all}

Systems: H3 doublet (cubic a = 4.5, STO-3G, (nd, no) = (1, 1), run_grad_ro_anchor.H3); tri 4H s+p triplet
(test_prototype TRI_A/TRI_ATOMS/SP_BASIS, (nd, no) = (1, 2), 16 AOs, 41 rotation parameters).  Dense pure-AFT J/K
(pbc_gamma.build_integrals, w = None).  exxdiv none = kshift 0, ewald = kshift v_M.  The tri integrals + SSF (75, 302)
D = 10 grid come from roks_replica.tri_setup() (its cache; ROKS_TRAP_CACHE).

PREDICTIONS (written before any TRAH run; only a syntax smoke of the module had been executed):
 P0 derivatives: |FD(E) - g.x| ~ 1e-10 (h = 1e-4 central, E ~ 1); the true gradient is EXACTLY 2 x ferric's packed
    gradient_blocks in each of the three blocks separately (so rohf_ah.rs's "no single scalar exists" is false for the
    gradient); x^T H y from a 4-point energy FD agrees with the analytic Hessian to ~1e-7 (h^2 truncation); H symmetric
    to ~1e-14 for ROHF, ~1e-8 for ROKS (FD f_xc).  2 x ferric_hvp != H by O(|off-diagonal f|) away from a
    Roothaan-semicanonical basis (the diag-Fock approximation), NOT by a constant factor.
 P1 anchor: where the DIIS replica converges (H3 ROHF none/ewald, H3 ROKS PBE0 none, tri ROKS PBE0 none with the
    0.05 shift / 600 it), TRAH from the core guess reaches the same E to <= 1e-10 and the same state (closed/open
    subspace overlaps = (nd, no) to 1e-8), with a quadratic tail (rho -> 1, max|g| squaring) in <= ~20 macro
    iterations; lowest exact-Hessian eigenvalue > 0 at every anchor minimum.
 P2 target: tri ROHF none from the core guess converges to PySCF -0.581222768976 (construction gap vs PySCF AFTDF
    61^3 <= 1e-9, as the UHF was 1e-12), aufbau labels DSSV..., lowest Hessian eigenvalue > 0.  Ewald staged from the
    none MOs: -1.826096526949 in a few macro iterations; ewald = none - a v_M N/2 exactly.
 ARTIFACT HYPOTHESES (what a broken or trapped run would look like):
  A1 a non-aufbau stationary state (the ROHF analogue of the PBE0 hole state -1.44347): converged, E ABOVE PySCF,
     labels not DSSV...; negative occupation-aware gap.
  A2 a saddle: g -> 0 but lowest Hessian eigenvalue < 0 (AH cannot leave a saddle once g = 0 exactly).
  A3 ewald from the core guess lands in the prototype's trapped -1.810787564982 (+1.53e-2): check whether that is a
     stationary point and a minimum of the ROHF energy (then it is a genuine local minimum, not a solver artifact).
  A4 a wrong analytic Hessian: rho stays far from 1 near convergence and the tail is linear, and P0 fails.
  A5 E BELOW PySCF: PySCF trapped or construction mismatch -> re-run PySCF from our density before claiming anything.
 P3 mirror of the molecular AH branch (stale-Fock Roothaan basis + diag-Fock Hessian + 0.2 clip, armed only at
    err_max < ah_trigger): on tri ROHF the DIIS phase never reaches err < 1e-2 reliably (replica err 1e-2..1e-1), so
    with ah_trigger 1e-2 it mostly stays DIIS; armed from it 4 (trigger 1.0) it inherits the repelling Roothaan
    diagonalization each iteration and may not converge.  Lifting the refusal alone is therefore NOT predicted to fix
    the open item; the exact-gradient-at-current-C TRAH is."""

import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import pbc_dft as pd  # noqa: E402
import pbc_rohf_newton as N  # noqa: E402
import roks_replica as RR  # noqa: E402

PYSCF_TRI_ROHF = {"none": -0.581222768976, "ewald": -1.826096526949}
TRI_EWALD_TRAPPED = -1.810787564982
TRI_PBE0_NONE = -1.465458280448
H3_ATOMS = [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5)), ("H", (1.6, 0.9, 0.7))]
_CACHE = {}


def tri():
    if "tri" not in _CACHE:
        S, h, enn, vm, jk, I, grid = RR.tri_setup()
        _CACHE["tri"] = dict(S=S, h=h, enn=enn, vm=vm, jk=jk, grid=grid, nd=1, no=2)
    return _CACHE["tri"]


def h3():
    if "h3" not in _CACHE:
        from pbc_gamma import Cell, build_integrals

        cell = Cell(np.eye(3) * 4.5, H3_ATOMS, "sto-3g")
        I = build_integrals(cell, None, exxdiv="ewald", verbose=False)
        gr = pd.PeriodicGrid(cell, 50, 194, D=8.0, scheme="ssf")
        _CACHE["h3"] = dict(
            S=I["S"],
            h=I["h"],
            enn=I["enn"],
            vm=I["madelung"],
            jk=pd.dense_jk(I["I"]),
            grid=RR._Grid(gr.weights, gr.ao),
            nd=1,
            no=1,
        )
    return _CACHE["h3"]


def inj(d, xc, ex):
    return N.Injected(
        d["S"],
        d["h"],
        d["enn"],
        d["jk"],
        xc=xc,
        grid=d["grid"],
        kshift=d["vm"] if ex == "ewald" else 0.0,
    )


def core(S, h):
    s, V = np.linalg.eigh(S)
    X = V @ np.diag(s**-0.5) @ V.T
    return X @ np.linalg.eigh(X @ h @ X)[1]


def overlaps(C, Cref, S, nd, no):
    wc = np.sum((C[:, :nd].T @ S @ Cref[:, :nd]) ** 2)
    wo = np.sum((C[:, nd : nd + no].T @ S @ Cref[:, nd : nd + no]) ** 2)
    return wc, wo


def diis_ref(d, xc, ex, ls=0.0, max_iter=600, C0=None):
    ks = d["vm"] if ex == "ewald" else 0.0
    grid = None if xc == "HF" else d["grid"]
    return RR.ferric_roks(
        d["S"],
        d["h"],
        d["jk"],
        d["enn"],
        d["nd"],
        d["no"],
        grid,
        xc,
        kshift=ks,
        C0=C0,
        max_iter=max_iter,
        level_shift=ls,
    )


def fmt_state(r):
    s = (
        f"E {r['e']:.12f} max|g| {r['gmax']:.1e} labels {r['labels'][:8]}.. aufbau {r['aufbau']} "
        f"gaps ({r['gap_a']:+.4f}, {r['gap_b']:+.4f})"
    )
    if "hess_min" in r:
        s += (
            f" | H: min eig {r['hess_min']:+.5f} (next {r['hess_eigs'][1]:+.5f}) asym {r['asym']:.1e}; "
            f"Roothaan-canonical basis (|dE| vs state {r['canon_state_same']:.1e}): exact min {r['canon_min']:+.5f}, "
            f"2*ferric min {r['ferric_min']:+.5f}, "
            f"max|2Hf-H| {r['ferric_vs_exact']:.2e}, "
            f"asym(Hf) {r['ferric_asym']:.1e}"
        )
    return s


# ------------------------------------------------------------------------------------------------ modes
def derivs():
    rng = np.random.default_rng(7)
    for name, d, xc, ex in (
        ("H3 ROHF", h3(), "HF", "ewald"),
        ("H3 ROKS PBE0", h3(), "PBE0", "ewald"),
        ("tri ROHF", tri(), "HF", "ewald"),
        ("tri ROKS PBE0", tri(), "PBE0", "none"),
    ):
        sysm = inj(d, xc, ex)
        nd, no = d["nd"], d["no"]
        C0 = core(d["S"], d["h"])
        P, blocks = N.pairs(C0.shape[1], nd, no)
        C = C0 @ N.cayley(
            N.unpack(0.05 * rng.standard_normal(len(P)), P, C0.shape[1])
        )  # non-stationary point
        pt = N.Point(sysm, C, nd, no)

        def E(k):
            return sysm.fock(*N.dens(pt.rotated(k, use_cayley=False), nd, no))[0]

        h = 1e-4
        x, y = rng.standard_normal(len(P)), rng.standard_normal(len(P))
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        g_fd = (E(h * x) - E(-h * x)) / (2 * h)
        # per-block ratio true/packed via FD along unit vectors of each block
        ratios = []
        off = 0
        for nb in blocks:
            if nb == 0:
                ratios.append(np.nan)
                continue
            e_i = np.zeros(len(P))
            e_i[off] = 1.0
            gi = (E(h * e_i) - E(-h * e_i)) / (2 * h)
            gp = pt.g_packed()[off]
            ratios.append(gi / gp if abs(gp) > 1e-8 else np.nan)
            off += nb
        hfd = (E(h * (x + y)) - E(h * (x - y)) - E(h * (y - x)) + E(-h * (x + y))) / (
            4 * h * h
        )
        Hx = pt.hvp(x)
        Hy = pt.hvp(y)
        Hfx = pt.ferric_hvp(x)
        # ferric approximation vs exact at a Roothaan-semicanonical point is not tested here (non-stationary C)
        print(
            f"{name:14s} nparam {len(P)} blocks {blocks}: |g.x - FD| {abs(pt.g @ x - g_fd):.1e} (g.x {pt.g @ x:+.6f}) "
            f"true/packed per block (vc, vo, oc) = ({ratios[0]:.10f}, {ratios[1]:.10f}, {ratios[2]:.10f}); "
            f"|y.Hx - FD| {abs(y @ Hx - hfd):.1e} (y.Hx {y @ Hx:+.6f}); |x.Hy - y.Hx| {abs(x @ Hy - y @ Hx):.1e}; "
            f"|2 Hf x - H x| {np.abs(2 * Hfx - Hx).max():.2e} (|Hx| {np.abs(Hx).max():.2e})",
            flush=True,
        )


def run_trah(name, d, xc, ex, C0, ref_e=None, ref_C=None, trace=False, gtol=1e-9):
    sysm = inj(d, xc, ex)
    t = time.time()
    r = N.trah(sysm, C0, d["nd"], d["no"], gtol=gtol, trace=trace)
    st = N.state_report(inj(d, xc, ex), r["C"], d["nd"], d["no"])
    line = (
        f"  TRAH {name:22s} {ex:5s}: conv {r['converged']} macro {r['macro']} rejects {r['rejects']} "
        f"jk {r['counts']['jk']} fock {r['counts']['fock']} hvp {r['counts']['hvp']} ({time.time() - t:.1f}s)\n"
        f"      {fmt_state(st)}"
    )
    if ref_e is not None:
        line += f"\n      E - ref {r['e'] - ref_e:+.2e}"
    if ref_C is not None:
        wc, wo = overlaps(r["C"], ref_C, d["S"], d["nd"], d["no"])
        line += f"  overlaps (closed, open) = ({wc:.10f}, {wo:.10f})"
    print(line, flush=True)
    tail = [(hh["macro"], hh["e"], hh["gmax"]) for hh in r["hist"][-6:]]
    print(
        "      tail " + "; ".join(f"{m}: {e:.12f} {g:.1e}" for m, e, g in tail),
        flush=True,
    )
    return r, st


def anchor():
    for name, d, xc, ex, ls in (
        ("H3 ROHF", h3(), "HF", "none", 0.0),
        ("H3 ROHF", h3(), "HF", "ewald", 0.0),
        ("H3 ROKS PBE0", h3(), "PBE0", "none", 0.0),
        ("tri ROKS PBE0", tri(), "PBE0", "none", 0.05),
    ):
        t = time.time()
        dr = diis_ref(d, xc, ex, ls=ls)
        print(
            f"  DIIS {name:22s} {ex:5s} ls {ls}: conv {dr['converged']} it {dr['it']} E {dr['e']:.12f} "
            f"({time.time() - t:.1f}s)",
            flush=True,
        )
        run_trah(name, d, xc, ex, core(d["S"], d["h"]), ref_e=dr["e"], ref_C=dr["C"])


def target():
    d = tri()
    C0 = core(d["S"], d["h"])
    r, st = run_trah(
        "tri ROHF (core)", d, "HF", "none", C0, ref_e=PYSCF_TRI_ROHF["none"], trace=True
    )
    r2, _ = run_trah(
        "tri ROHF (staged)", d, "HF", "ewald", r["C"], ref_e=PYSCF_TRI_ROHF["ewald"]
    )
    n = 2 * d["nd"] + d["no"]
    print(
        f"      ewald - none {r2['e'] - r['e']:+.12f} vs -v_M N/2 {-d['vm'] * n / 2:+.12f}",
        flush=True,
    )
    run_trah(
        "tri ROHF (core, direct)", d, "HF", "ewald", C0, ref_e=PYSCF_TRI_ROHF["ewald"]
    )
    # robustness: rotated core starts
    from scipy.linalg import expm

    print("  rotated core starts (tri ROHF none):", flush=True)
    for t_ in (1e-6, 1e-2, 1e-1):
        for seed in range(3):
            K = np.random.default_rng(seed).standard_normal((16, 16))
            Cs = C0 @ expm(t_ * (K - K.T) / 2)
            sysm = inj(d, "HF", "none")
            rr = N.trah(sysm, Cs, d["nd"], d["no"], gtol=1e-9)
            print(
                f"    t {t_:.0e} seed {seed}: conv {rr['converged']} macro {rr['macro']} rejects {rr['rejects']} "
                f"jk {rr['counts']['jk']} E - PySCF {rr['e'] - PYSCF_TRI_ROHF['none']:+.2e}",
                flush=True,
            )
    # what is the ewald trapped state?
    dr = RR.ferric_roks(
        d["S"],
        d["h"],
        d["jk"],
        d["enn"],
        1,
        2,
        None,
        "HF",
        kshift=d["vm"],
        max_iter=600,
    )
    print(
        f"  DIIS tri ROHF ewald from core (replica): conv {dr['converged']} it {dr['it']} E {dr['e']:.12f}",
        flush=True,
    )
    if dr["converged"]:
        print(
            "      " + fmt_state(N.state_report(inj(d, "HF", "ewald"), dr["C"], 1, 2)),
            flush=True,
        )


def mirror():
    d = tri()
    for trig in (1e-2, 1.0):
        sysm = inj(d, "HF", "none")
        r = N.ferric_ah_mirror(
            sysm, core(d["S"], d["h"]), 1, 2, ah_trigger=trig, max_iter=300
        )
        h = r["hist"]
        tail = h[-50:]
        print(
            f"  mirror ah_trigger {trig}: conv {r['converged']} it {r['it']} AH steps {r['n_ah']} E {r['e']:.12f} "
            f"jk {r['counts']['jk']} | tail-50 E [{min(x[1] for x in tail):.6f}, {max(x[1] for x in tail):.6f}] "
            f"max|g| [{min(x[2] for x in tail):.1e}, {max(x[2] for x in tail):.1e}]",
            flush=True,
        )
    # the same mirror on H3 ROHF (where it is expected to work)
    d = h3()
    r = N.ferric_ah_mirror(
        inj(d, "HF", "none"), core(d["S"], d["h"]), 1, 1, ah_trigger=1e-2, max_iter=300
    )
    print(
        f"  mirror H3 ROHF none ah_trigger 1e-2: conv {r['converged']} it {r['it']} AH steps {r['n_ah']} "
        f"E {r['e']:.12f}",
        flush=True,
    )


def diis():
    d = tri()
    for ls in (0.0, 0.05, 0.1):
        t = time.time()
        dr = diis_ref(d, "HF", "none", ls=ls)
        tail = dr["hist"][-50:]
        print(
            f"  DIIS replica tri ROHF none ls {ls}: conv {dr['converged']} it {dr['it']} E {dr['e']:.10f} "
            f"tail-50 E [{min(x[1] for x in tail):.6f}, {max(x[1] for x in tail):.6f}] err "
            f"[{min(x[2] for x in tail):.1e}, {max(x[2] for x in tail):.1e}] ({time.time() - t:.0f}s)",
            flush=True,
        )


pyscf_fill = N.pyscf_fill


def diis_pyscf_occ(d, xc, ex, C0, max_iter=300, gtol=1e-9):
    """ferric's DIIS replica (F_eff, err = S C (g - g^T) C^T S, history 8, no shift) with ONLY the occupation rule
    swapped for PySCF's (pyscf_fill).  NOT ferric: a candidate first-order fix."""
    sysm = inj(d, xc, ex)
    S, nd, no = d["S"], d["nd"], d["no"]
    s, V = np.linalg.eigh(S)
    X = V @ np.diag(s**-0.5) @ V.T
    C = np.array(C0)
    diis = RR.FerricDiis(8)
    for it in range(1, max_iter + 1):
        pt = N.Point(sysm, C, nd, no)
        if np.abs(pt.g_packed()).max() < gtol:
            return dict(converged=True, it=it, e=pt.e, C=C)
        F = RR.roothaan(pt.Fa, pt.Fb, pt.Da, pt.Db, S)
        ga = RR.mo_gradient(C, pt.Fa, pt.Fb, nd, no)
        Fn, _ = diis.step(F, S @ C @ (ga - ga.T) @ C.T @ S)
        eps, Cn = np.linalg.eigh(X @ Fn @ X)
        Cn = X @ Cn
        C = pyscf_fill(Cn, eps, pt.Fa, nd, no)
    return dict(converged=False, it=max_iter, e=pt.e, C=C)


def occ():
    from scipy.linalg import expm

    d = tri()
    S, nd, no = d["S"], 1, 2
    C0 = core(S, d["h"])
    for ex in ("none", "ewald"):
        sysm = inj(d, "HF", ex)
        r = N.trah(sysm, C0, nd, no, gtol=1e-11)
        pt = N.Point(sysm, r["C"], nd, no)
        F = RR.roothaan(pt.Fa, pt.Fb, pt.Da, pt.Db, S)
        s, V = np.linalg.eigh(S)
        X = V @ np.diag(s**-0.5) @ V.T
        eps, Cr = np.linalg.eigh(X @ F @ X)
        Cr = X @ Cr
        wcl = np.sum((r["C"][:, :nd].T @ S @ Cr) ** 2, axis=0)
        wop = np.sum((r["C"][:, nd : nd + no].T @ S @ Cr) ** 2, axis=0)
        ea = np.einsum("mi,mn,ni->i", Cr, pt.Fa, Cr)
        eb = np.einsum("mi,mn,ni->i", Cr, pt.Fb, Cr)
        print(
            f"  TRAH minimum {ex}: E {r['e']:.12f}; Roothaan eigenpairs (lowest 6): label eps / mo_ea / mo_eb",
            flush=True,
        )
        for i in range(6):
            lab = "D" if wcl[i] > 0.5 else ("S" if wop[i] > 0.5 else "V")
            print(
                f"      {i} {lab} {eps[i]:+.6f} / {ea[i]:+.6f} / {eb[i]:+.6f}",
                flush=True,
            )
        Cp = pyscf_fill(Cr, eps, pt.Fa, nd, no)
        wc, wo = overlaps(Cp, r["C"], S, nd, no)
        print(
            f"      PySCF fill of these eigenvectors reproduces the state: overlaps ({wc:.12f}, {wo:.12f})",
            flush=True,
        )
    print(
        "  ferric DIIS replica with PySCF's occupation rule (NOT ferric), tri ROHF, core + rotated starts:",
        flush=True,
    )
    for ex in ("none", "ewald"):
        starts = [("core", C0)]
        for t_ in (1e-6, 1e-2, 1e-1):
            for seed in range(3):
                K = np.random.default_rng(seed).standard_normal((16, 16))
                starts.append((f"t {t_:.0e} s{seed}", C0 @ expm(t_ * (K - K.T) / 2)))
        res = []
        for lab, Cs in starts:
            rr = diis_pyscf_occ(d, "HF", ex, Cs)
            res.append((lab, rr["converged"], rr["it"], rr["e"] - PYSCF_TRI_ROHF[ex]))
        print(
            f"    {ex}: "
            + "; ".join(f"{a}: {'C' if b else 'X'} {c} {e:+.1e}" for a, b, c, e in res),
            flush=True,
        )


def trapped():
    """The second minimum: TRAH from the rotated starts that land at +1.53e-2 (none), plus its ewald image."""
    from scipy.linalg import expm

    d = tri()
    C0 = core(d["S"], d["h"])
    K = np.random.default_rng(0).standard_normal((16, 16))
    for ex in ("none", "ewald"):
        r = N.trah(inj(d, "HF", ex), C0 @ expm(1e-1 * (K - K.T) / 2), 1, 2, gtol=1e-10)
        st = N.state_report(inj(d, "HF", ex), r["C"], 1, 2)
        print(
            f"  TRAH t 1e-1 seed 0 {ex}: conv {r['converged']} macro {r['macro']} E - PySCF "
            f"{r['e'] - PYSCF_TRI_ROHF[ex]:+.12f}\n      {fmt_state(st)}",
            flush=True,
        )
        # ferric's F6 witness at this state: best unrelaxed swaps (spin-Fock Koopmans), then TRAH from each
        pt = N.Point(inj(d, "HF", ex), r["C"], 1, 2)
        for cand in RR.swap_candidates(r["C"], pt.Fa, pt.Fb, 1, 2):
            Cs = RR.swap_cols(r["C"], cand[1], cand[2])
            es = N.Point(inj(d, "HF", ex), Cs, 1, 2).e
            rs = N.trah(inj(d, "HF", ex), Cs, 1, 2, gtol=1e-10)
            print(
                f"      witness {cand[0]} ({cand[1]}<->{cand[2]}, Koopmans {cand[3]:+.4f}): unrelaxed E - state "
                f"{es - r['e']:+.4e} -> TRAH from it: conv {rs['converged']} macro {rs['macro']} E - PySCF "
                f"{rs['e'] - PYSCF_TRI_ROHF[ex]:+.2e}",
                flush=True,
            )


def pbe0():
    """tri ROKS PBE0 none (the Roothaan-repelling case): TRAH and PySCF-fill DIIS from the core + 9 rotated starts."""
    from scipy.linalg import expm

    d = tri()
    C0 = core(d["S"], d["h"])
    starts = [("core", C0)]
    for t_ in (1e-6, 1e-2, 1e-1):
        for seed in range(3):
            K = np.random.default_rng(seed).standard_normal((16, 16))
            starts.append((f"t {t_:.0e} s{seed}", C0 @ expm(t_ * (K - K.T) / 2)))
    for lab, Cs in starts:
        sysm = inj(d, "PBE0", "none")
        r = N.trah(sysm, Cs, 1, 2, gtol=1e-9)
        rd = diis_pyscf_occ(d, "PBE0", "none", Cs, max_iter=200)
        print(
            f"  {lab:10s} TRAH conv {r['converged']} macro {r['macro']} rejects {r['rejects']} jk {r['counts']['jk']} "
            f"E - pin {r['e'] - TRI_PBE0_NONE:+.1e} | PySCF-fill DIIS conv {rd['converged']} it {rd['it']} "
            f"E - pin {rd['e'] - TRI_PBE0_NONE:+.1e}",
            flush=True,
        )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for m in (
        ("derivs", "anchor", "target", "trapped", "occ", "mirror", "diis")
        if which == "all"
        else (which,)
    ):
        print(f"== {m}", flush=True)
        globals()[m]()
