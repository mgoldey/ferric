"""A numpy replica of ferric's INJECTED ROHF/ROKS loop (crates/ferric-scf/src/rohf.rs::solve_rohf_impl with
PeriodicInjection, as called by crates/ferric-pbc/src/rohf.rs::run_staged), for the tri 4H s+p triplet ROKS PBE0
CI non-convergence diagnosis (FINDINGS "ROKS PBE0 CI non-convergence (Python diagnosis) — 2026-09-25").

What is replicated (read from the Rust, 2026-09-25):
  * symmetric orthogonalizer S^{-1/2} (not the prototype's canonical X);
  * core guess: C = S^{-1/2} eigh(S^{-1/2} h S^{-1/2}), occupations by index (nd closed, no open);
  * per spin F_s = h + J[Da+Db] - a K_s + V_s (pbc_uks._fock_energy, kshift = the stage's v_M);
  * Guest-Saunders/PySCF Roothaan effective Fock (same projector formula as pbc_uks.roks);
  * DIIS: Pulay on (F_eff, err) with err = S C (g - g^T) C^T S, g the ROHF MO gradient blocks (vc: fa+fb, vo: fa,
    oc: fb); history 8 (RhfConfig.diis_size), first call returns F; Gram block normalised; oldest-first shrink on a
    relative pivot < 1e-13 (solve_diis_coeffs / solve_linear);
  * level shift (optional): shift_eff = ls * err_max / (err_max + 1e-3) on S Cv Cv^T S, iter > 1;
  * convergence: dp_rms < dc, dp_max < 10 dc, |dE| < ec (dc 1e-10, ec 1e-3, GammaRksConfig), AND the F6 gradient
    guard err_max <= 1e-3 (else DIIS reset); the continuity lock is OFF (injected path);
  * the F6 swap witness at convergence (best beta closed->open and alpha open->virtual swap by spin-Fock Koopmans,
    unrelaxed energy; restart from it if lower by > 1e-5; <= 3 restarts) -- `witness=True`;
  * MOM (optional, mom_after_iter > 0): crate::mom::mom_reorder by overlap with the previous closed/open blocks.

Extra knobs that do NOT exist in ferric (candidate fixes, labelled as such by the driver): `damp` (static density
damping for the first `damp_iters` iterations), `C0` (arbitrary starting MOs).

Units Bohr / Hartree."""

from __future__ import annotations

import os
import tempfile
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbc_dft as pd  # noqa: E402
import pbc_uks as U  # noqa: E402

CACHE = os.environ.get(
    "ROKS_TRAP_CACHE", os.path.join(tempfile.gettempdir(), "roks_trap_tri_cache.npz")
)


class _Grid:
    def __init__(self, weights, ao):
        self.weights, self.ao = weights, ao

    @property
    def size(self):
        return len(self.weights)


def tri_setup():
    """(S, h, enn, v_M, jk, I, grid) for the Rust test's cell: TRI_A / TRI_ATOMS, s+p H, triplet (3, 1),
    PeriodicGrid(75, 302, D=10, ssf), dense pure-AFT I.  Cached (npz) after the first build."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        I = z["I"]
        return (
            z["S"],
            z["h"],
            float(z["enn"]),
            float(z["vm"]),
            pd.dense_jk(I),
            I,
            _Grid(z["w"], z["ao"]),
        )
    from pbc_gamma import Cell, build_integrals
    from test_prototype import SP_BASIS, TRI_A, TRI_ATOMS

    t = time.time()
    cell = Cell(TRI_A, TRI_ATOMS, SP_BASIS)
    ints = build_integrals(cell, None, exxdiv="ewald", verbose=False)
    gr = pd.PeriodicGrid(cell, 75, 302, D=10.0, scheme="ssf")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(
        CACHE,
        S=ints["S"],
        h=ints["h"],
        enn=ints["enn"],
        vm=ints["madelung"],
        I=ints["I"],
        w=gr.weights,
        ao=gr.ao,
    )
    print(
        f"[setup] built integrals + grid ({gr.size} pts) in {time.time() - t:.0f}s -> {CACHE}",
        flush=True,
    )
    return (
        ints["S"],
        ints["h"],
        ints["enn"],
        ints["madelung"],
        pd.dense_jk(ints["I"]),
        ints["I"],
        _Grid(gr.weights, gr.ao),
    )


def roothaan(Fa, Fb, Da, Db, S):
    n = S.shape[0]
    fc = 0.5 * (Fa + Fb)
    pc, po, pv = Db @ S, (Da - Db) @ S, np.eye(n) - Da @ S
    F = (
        0.5 * (pc.T @ fc @ pc + po.T @ fc @ po + pv.T @ fc @ pv)
        + po.T @ Fb @ pc
        + po.T @ Fa @ pv
        + pv.T @ fc @ pc
    )
    return F + F.T


def mo_gradient(C, Fa, Fb, nd, no):
    fa, fb = C.T @ Fa @ C, C.T @ Fb @ C
    n = C.shape[1]
    na = nd + no
    g = np.zeros((n, n))
    g[na:, :nd] = fa[na:, :nd] + fb[na:, :nd]
    g[na:, nd:na] = fa[na:, nd:na]
    g[nd:na, :nd] = fb[nd:na, :nd]
    return g


def dens(C, nd, no):
    Db = C[:, :nd] @ C[:, :nd].T
    return Db + C[:, nd : nd + no] @ C[:, nd : nd + no].T, Db


class FerricDiis:
    def __init__(self, size=8):
        self.size, self.F, self.E = size, [], []

    def reset(self):
        self.F, self.E = [], []

    def step(self, F, err):
        self.F.append(F.copy())
        self.E.append(err.copy())
        if len(self.F) > self.size:
            self.F.pop(0)
            self.E.pop(0)
        if len(self.F) < 2:
            return F.copy(), None
        for m in range(len(self.F), 1, -1):
            Fs, Es = self.F[-m:], self.E[-m:]
            G = np.array([[np.sum(a * b) for b in Es] for a in Es])
            gs = np.abs(G).max()
            if not (gs > 0 and np.isfinite(gs)):
                continue
            A = np.zeros((m + 1, m + 1))
            A[:m, :m] = G / gs
            A[:m, m] = A[m, :m] = 1.0
            rhs = np.zeros(m + 1)
            rhs[m] = 1.0
            c = _solve_linear(A, rhs)
            if c is not None:
                return sum(ci * f for ci, f in zip(c[:m], Fs)), c[:m]
        return self.F[-1].copy(), None


def _solve_linear(A, b, rtol=1e-13):
    A = A.copy()
    b = b.copy()
    s = np.abs(A).max()
    A /= s
    b /= s
    n = len(b)
    for col in range(n):
        p = col + int(np.argmax(np.abs(A[col:, col])))
        if abs(A[p, col]) < rtol:
            return None
        if p != col:
            A[[col, p]] = A[[p, col]]
            b[[col, p]] = b[[p, col]]
        for r in range(col + 1, n):
            f = A[r, col] / A[col, col]
            A[r, col:] -= f * A[col, col:]
            b[r] -= f * b[col]
    x = np.zeros(n)
    for r in range(n - 1, -1, -1):
        x[r] = (b[r] - A[r, r + 1 :] @ x[r + 1 :]) / A[r, r]
    return x


def mom_reorder(C_new, S, ref_closed, ref_open, nd, no):
    """crate::mom::mom_reorder semantics: closed = top-nd overlap with the previous closed block, open = top-no of the
    rest with the previous open block, rest virtual in eigenvalue order."""
    SC = S @ C_new
    n = C_new.shape[1]
    wc = np.sum((ref_closed.T @ SC) ** 2, axis=0)
    closed = sorted(np.argsort(-wc, kind="stable")[:nd])
    rest = [p for p in range(n) if p not in closed]
    wo = np.sum((ref_open.T @ SC) ** 2, axis=0)
    open_ = sorted(sorted(rest, key=lambda p: -wo[p])[:no])
    virt = [p for p in rest if p not in open_]
    return C_new[:, closed + open_ + virt]


def energy_fock(S, h, jk, enn, Da, Db, grid, xc, hyb, kshift):
    return U._fock_energy(S, h, jk, enn, Da, Db, grid, xc, hyb, kshift)


def ferric_roks(
    S,
    h,
    jk,
    enn,
    nd,
    no,
    grid,
    xc,
    kshift=0.0,
    C0=None,
    max_iter=200,
    diis_size=8,
    level_shift=0.0,
    mom_after_iter=0,
    guard=True,
    witness=True,
    density_conv=1e-10,
    energy_conv=1e-3,
    damp=0.0,
    damp_iters=0,
    diis_start=1,
    trace=False,
    record=False,
    err_kind="mo",
    lock=False,
    shift_until=None,
    shift_damp_err=1e-3,
    reset_diis_after_shift=False,
    shift_off_below=None,
):
    """err_kind: "mo" (ferric) or "fds" (prototype pbc_uks.roks, NOT ferric).  Returns dict(converged, e, it, C, Da, Db, err_max, hist=[(it, E, err_max, dp_rms)], witness_log, exit)."""
    hyb = U.hybrid_fraction(xc)
    if str(xc).upper() == "HF":
        grid = None
    s, V = np.linalg.eigh(S)
    X = V @ np.diag(s**-0.5) @ V.T
    if C0 is None:
        C = X @ np.linalg.eigh(X @ h @ X)[1]
    else:
        C = np.array(C0)
    na = nd + no
    Da, Db = dens(C, nd, no)
    diis = FerricDiis(diis_size)
    guard_on = guard and mom_after_iter == 0
    prev_e = 0.0
    dp_rms = dp_max = np.inf
    prev_Dt = None
    hist, wlog = [], []
    probe = None  # (held_result, pending list, current cand)
    restarts = 0
    probe_iters = 0
    mom_ref = None
    locked, streak = False, 0
    wopen = []
    shift_off = False
    it = 0
    last = None
    while True:
        it += 1
        if it > max_iter + probe_iters:
            break
        Dt = Da + Db
        if prev_Dt is not None:
            d = Dt - prev_Dt
            dp_rms, dp_max = np.sqrt(np.mean(d * d)), np.abs(d).max()
        prev_Dt = Dt.copy()
        e, Fa, Fb, _ = energy_fock(S, h, jk, enn, Da, Db, grid, xc, hyb, kshift)
        if probe is not None:
            held, pending, cand = probe
            probe = None
            if e < held["e"] - 1e-5:
                restarts += 1
                wlog.append(("restart", held["e"], cand, e))
                if restarts > 3:
                    held = dict(held, converged=False, exit="NotCertified", it=it)
                    held["hist"], held["witness_log"] = hist, wlog
                    return held
                diis.reset()
                prev_e, dp_rms, dp_max = 0.0, np.inf, np.inf
                # (lock stays off on the injected path)
            else:
                wlog.append(("probe-rejected", held["e"], cand, e))
                if pending:
                    cand, Cp = pending.pop()
                    probe = (held, pending, cand)
                    C = Cp
                    Da, Db = dens(C, nd, no)
                    probe_iters += 1
                    prev_Dt = None
                    continue
                held["hist"], held["witness_log"] = hist, wlog
                return held
        F = roothaan(Fa, Fb, Da, Db, S)
        g = mo_gradient(C, Fa, Fb, nd, no)
        ga = g - g.T
        err = S @ C @ ga @ C.T @ S
        err_max = np.abs(err).max()
        if (
            err_kind == "fds"
        ):  # NOT ferric: the prototype's (pbc_uks.roks) DIIS error [F_eff, D_t] in the X basis
            Dt_ = Da + Db
            err_diis = X @ (F @ Dt_ @ S - S @ Dt_ @ F) @ X
        else:
            err_diis = err
        de = abs(e - prev_e)
        conv = dp_rms < density_conv and dp_max < 10 * density_conv and de < energy_conv
        hist.append((it, e, err_max, dp_rms))
        if trace:
            print(
                f"    it {it:3d} E {e:.10f} dE {de:.2e} err {err_max:.2e} dp {dp_rms:.2e}",
                flush=True,
            )
        if it > 1 and conv and guard_on and err_max > 1e-3:
            diis.reset()
            conv = False
        if it > 1 and conv:
            eps, Cf = np.linalg.eigh(X @ F @ X)
            Cf = X @ Cf
            if (
                locked or mom_after_iter > 0
            ):  # ferric: guard.relabel (lock) / the MOM-held occupation
                Cf = continuity(Cf, C, S, nd, no)
            res = dict(
                converged=True,
                exit="Converged",
                e=e,
                it=it,
                C=Cf,
                eps=eps,
                Fa=Fa,
                Fb=Fb,
                err_max=err_max,
            )
            res["Da"], res["Db"] = dens(Cf, nd, no)
            if not (guard_on and witness):
                res["hist"], res["witness_log"] = hist, wlog
                return res
            cands = swap_candidates(Cf, Fa, Fb, nd, no)
            pending = [(c, swap_cols(Cf, c[1], c[2])) for c in cands]
            pending.reverse()
            if not pending:
                res["hist"], res["witness_log"] = hist, wlog
                return res
            cand, Cp = pending.pop()
            probe = (res, pending, cand)
            probe_iters += 1
            C = Cp
            Da, Db = dens(C, nd, no)
            prev_Dt = None
            continue
        prev_e = e
        if it >= diis_start:
            Fn, _ = diis.step(F, err_diis)
        else:
            Fn = F
        if shift_until is not None and it == shift_until + 1 and reset_diis_after_shift:
            diis.reset()  # NOT ferric: restart the DIIS history when the shift is dropped
        if shift_off_below is not None and err_max < shift_off_below and not shift_off:
            shift_off = True  # NOT ferric: latch the shift off once the gradient first drops below the bound
        if (
            level_shift > 0
            and it > 1
            and (shift_until is None or it <= shift_until)
            and not shift_off
        ):
            se = level_shift * err_max / (err_max + shift_damp_err)
            if se > 1e-10:
                Cv = C[:, na:]
                Fn = Fn + se * S @ Cv @ Cv.T @ S
        eps, Cn = np.linalg.eigh(X @ Fn @ X)
        Cn = X @ Cn
        if record:
            wopen.append(float(np.sum((C[:, nd:na].T @ S @ Cn[:, nd:na]) ** 2)))
        if (
            lock and guard_on
        ):  # the F6 continuity lock (molecular path only in ferric since 2026-09-25)
            if not locked:
                w = np.sum((C[:, nd:na].T @ S @ Cn[:, nd:na]) ** 2)
                streak = streak + 1 if w < no - 0.5 else 0
                if streak >= 3:
                    locked = True
                    diis.reset()
            if locked:
                Cn = continuity(Cn, C, S, nd, no)
        if mom_after_iter > 0 and it > mom_after_iter and mom_ref is not None:
            Cn = mom_reorder(Cn, S, mom_ref[0], mom_ref[1], nd, no)
        C = Cn
        if mom_after_iter > 0 and it >= mom_after_iter:
            mom_ref = (C[:, :nd].copy(), C[:, nd:na].copy())
        Dan, Dbn = dens(C, nd, no)
        if damp > 0 and it <= damp_iters:
            Dan, Dbn = (1 - damp) * Dan + damp * Da, (1 - damp) * Dbn + damp * Db
            # re-idempotize is NOT done: F is built from the mixed density (standard static damping); C stays the
            # undamped eigenvectors for the gradient/occupations.
        Da, Db = Dan, Dbn
        last = (e, err_max)
    return dict(
        wopen=wopen,
        converged=False,
        exit="MaxIter",
        e=prev_e,
        it=max_iter,
        C=C,
        Da=Da,
        Db=Db,
        err_max=last[1] if last else np.nan,
        hist=hist,
        witness_log=wlog,
    )


def continuity(Cn, Cref, S, nd, no):
    """rohf_occupation::select_by_continuity: closed = top-nd weight on the previous closed block, open = top-no of
    the rest on the previous open block, virtuals in eigenvalue order."""
    na = nd + no
    SC = S @ Cn
    n = Cn.shape[1]
    wc = np.sum((Cref[:, :nd].T @ SC) ** 2, axis=0)
    closed = sorted(sorted(range(n), key=lambda p: -wc[p])[:nd])
    rest = [p for p in range(n) if p not in closed]
    wo = np.sum((Cref[:, nd:na].T @ SC) ** 2, axis=0)
    op = sorted(sorted(rest, key=lambda p: -wo[p])[:no])
    return Cn[:, closed + op + [p for p in rest if p not in op]]


def swap_candidates(C, Fa, Fb, nd, no):
    na = nd + no
    n = C.shape[1]
    out = []
    fbd = np.einsum("mi,mn,ni->i", C, Fb, C)
    fad = np.einsum("mi,mn,ni->i", C, Fa, C)
    if nd > 0:
        best = min(
            ((fbd[t] - fbd[i], i, t) for i in range(nd) for t in range(nd, na)),
            key=lambda x: x[0],
        )
        out.append(("beta_closed_to_open", best[1], best[2], best[0]))
    if na < n:
        best = min(
            ((fad[a] - fad[t], t, a) for t in range(nd, na) for a in range(na, n)),
            key=lambda x: x[0],
        )
        out.append(("alpha_open_to_virtual", best[1], best[2], best[0]))
    return out


def swap_cols(C, i, j):
    C = C.copy()
    C[:, [i, j]] = C[:, [j, i]]
    return C


def analyse(S, h, jk, enn, nd, no, grid, xc, kshift, C):
    """Stationarity + occupation analysis of the ROKS determinant C (first nd closed, next no open).
    Returns dict(e, gnorm (max |MO gradient|), gfro, spin-Fock diagonals per class, occupation-aware per-spin gaps,
    sorted Roothaan eigenvalue positions of the occupied orbitals)."""
    hyb = U.hybrid_fraction(xc)
    Da, Db = dens(C, nd, no)
    e, Fa, Fb, _ = energy_fock(S, h, jk, enn, Da, Db, grid, xc, hyb, kshift)
    g = mo_gradient(C, Fa, Fb, nd, no)
    F = roothaan(Fa, Fb, Da, Db, S)
    s, V = np.linalg.eigh(S)
    X = V @ np.diag(s**-0.5) @ V.T
    eps, Cr = np.linalg.eigh(X @ F @ X)
    Cr = X @ Cr
    # which Roothaan eigenvectors are closed/open/virtual in the current determinant
    SC = S @ C
    wcl = np.sum((C[:, :nd].T @ S @ Cr) ** 2, axis=0)
    wop = np.sum((C[:, nd : nd + no].T @ S @ Cr) ** 2, axis=0)
    lab = np.where(wcl > 0.5, "D", np.where(wop > 0.5, "S", "V"))
    ga, gb = U.occ_gap(Fa, Da, S), U.occ_gap(Fb, Db, S)
    del SC
    return dict(
        e=e,
        gmax=np.abs(g).max(),
        gfro=np.linalg.norm(g),
        eps=eps,
        labels="".join(lab),
        gap_a=ga,
        gap_b=gb,
        aufbau=("".join(lab) == "D" * nd + "S" * no + "V" * (len(eps) - nd - no)),
    )
