#!/usr/bin/env python3
"""eps-linked (magnitude-based) refinements of the integral-direct LMP2
locality maps — Python prototype (PROTOTYPE FIRST; no Rust until the
verdict). Vehicle: PySCF + the machinery of scripts/amplitude_lmp2_proto.py
(Boys occ + VV-HV virtuals, same-kernel RI A=(ia|P), V=(P|Q)).

Two candidate refinements, both measured as size-vs-error FRONTIERS against
the already-validated distance rules of ferric_mp2::lmp2_direct:

  C1  eps-linked i->aux fit domains: replace the distance rule
      (aux_radius_bohr) with magnitude selection on the unwhitened strips,
      s_i(P) = max_a |(P|ia)| (threshold and Sec-26-style L1-tail variants,
      plus a metric-normalized variant s_i(P)/sqrt(V_PP)). Pair fit is the
      unchanged domain-local same-kernel fit J_ij = A_i,D^T V_DD^-1 A_j,D
      with D_ij = D_i u D_j. Error metric: FULL (eps=0) MP2 energy from the
      domain-fitted J minus the same energy from the untruncated global fit
      — the map error in isolation, exactly the quantity the wiki Sec-27
      flatness table tracks.
  C2  bound-based pair virtual candidates: Schwarz-type screen from fitted
      diagonals q_ia = sqrt((ia|ia)_fit): a in C_ij iff
      q_ia*qmax_j >= kappa*eps or q_ja*qmax_i >= kappa*eps (both Eq-8
      orientations), vs the distance ball on dipole centroids
      (virt_radius_bohr). Error metric: masked-CG energy with the candidate
      restriction ANDed into the Eq-8 mask, minus the eps-only masked
      energy (candidate error in isolation, on the SAME global-fit J).
      Includes a strip-local q variant (V_DD-restricted fit diagonal — the
      quantity the Rust direct path can actually compute) and a
      conservativeness count (Eq-8-retained elements escaping C_ij).

EXACTNESS ANCHORS (written BEFORE any sweep; hard exit on failure):
  A1  aux distance rule at r=1e9 reproduces the global fit:
      max|J_dom - J_glob| <= 1e-10.
  A2  aux magnitude rule at tau=0 (trivial limit keeps every aux
      function): same bar.
  A3  virtual candidate rule at kappa=0 (and rv=1e9) leaves the Eq-8 mask
      untouched: the masked energy is BIT-IDENTICAL (dE == 0.0) and the
      candidate mask is all-true.
  A4  the eps=0 pseudo-canonical closed-form solve agrees with the masked
      CG solver (independent algebra) to <= 1e-9 on E.
MUTATION ARMS (each must FAIL its anchor; run before sweeps):
  M1  drop the largest-score aux function from every D_i at tau=0 -> A2
      must fail (max|dJ| > 1e-10).
  M2  zero the largest q_ia before screening at kappa=1 -> the
      conservativeness count must become nonzero (a retained Eq-8 element
      escapes the candidate set).

ARTIFACT HYPOTHESES (X = real, Y = broken construction; X != Y):
  C1  X: magnitude frontier sits at-or-inside the distance frontier —
      at matched |dE| the mean pair-domain is smaller (win), or the two
      frontiers coincide with magnitude picks being distance balls in
      disguise (honest NO-GO: locality is all there is to the magnitudes);
      either way error falls MONOTONELY as tau falls and selected-aux
      distances (locmax/loc95) stay bounded.
      Y: error not controlled by tau (erratic / non-monotone), exactness
      requiring the FULL aux set (domain tracking naux — the ao-laplace
      Sec-5 fingerprint), or V_DD Cholesky failures on magnitude-selected
      (scattered, ill-conditioned) domains.
  C2  X: at kappa <= 1 the candidate error is far below the eps-tier
      truncation error (the screen is near-conservative by construction),
      and at matched mean |C_ij| the Schwarz rule's error is <= the
      distance ball's; conservativeness violations are rare and trace to
      the fitted diagonal underestimating the true diagonal.
      Y: violations at kappa=1 that do NOT trace to fit-diagonal error
      (wrong orientation / wrong max — construction bug), or the
      strip-local q disagreeing wildly with the global q (the
      implementable rule is then NOT the measured rule).
STOP CONDITIONS: too-clean coincidence across systems AND operators at
every size is a fingerprint of arithmetic, not chemistry — audit before
writing up. Negative verdicts get the same evidence bar (the distance
rules are validated; "not beaten" is a real deliverable).

Usage (PySCF may use several BLAS threads — not a ferric/rayon job; long
runs go under scripts/ferric-limited --max=4G --high=3600M --):
  python3 scripts/queue/proto_eps_linked_maps.py \
      --xyz testdata/molecules/water.xyz [--omega 1.0] \
      [--phase anchors|aux|virt|all] [--out scripts/queue/out/x.txt]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amplitude_lmp2_proto as base  # noqa: E402
from pyscf import gto, scf, lo  # noqa: E402
from pyscf.data import elements  # noqa: E402

log = base.log


# ---------------------------------------------------------------- setup

def setup(xyz, basis, omega):
    t0 = time.time()
    atom = base.load_xyz(xyz)
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=1500)
    ncore = elements.chemcore(mol)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, "SCF not converged"
    nocc_tot = np.count_nonzero(mf.mo_occ > 0)
    C_occ_all = mf.mo_coeff[:, :nocc_tot]
    C_act = lo.Boys(mol, mf.mo_coeff[:, ncore:nocc_tot]).kernel()
    C_vloc, n_l, n_h = base.build_vvhv(mol, mf, C_occ_all)
    base.check_construction(mol, mf, C_act, C_vloc)
    F = mf.get_fock()
    Foo = C_act.T @ F @ C_act
    Fvv = C_vloc.T @ F @ C_vloc
    A, V, aux_xyz, auxmol = base.build_ri(mol, C_act, C_vloc, omega)
    Jg = base.ri_j_global(A, V)
    occ_cen = base.boys_centroids(mol, C_act)
    virt_cen = base.boys_centroids(mol, C_vloc)
    no, nv, naux = A.shape
    log(f"== {xyz} basis={basis} op={'coulomb' if omega is None else f'erfc{omega:g}'} "
        f"no={no} nv={nv} naux={naux} nao={mol.nao} ({time.time()-t0:.1f}s)")
    return dict(mol=mol, mf=mf, A=A, V=V, Jg=Jg, Foo=Foo, Fvv=Fvv,
                aux_xyz=aux_xyz, occ_cen=occ_cen, virt_cen=virt_cen,
                no=no, nv=nv, naux=naux)


# ------------------------------------------------- energies (two algebras)

def solve_full_mp2_closed(J, Foo, Fvv):
    """eps=0 (unmasked) MP2 energy via pseudo-canonical rotation — closed
    form, no CG. Independent algebra from base.solve_masked_mp2 (A4)."""
    wo, Uo = np.linalg.eigh(Foo)
    wv, Uv = np.linalg.eigh(Fvv)
    Jc = np.einsum("iajb,ik->kajb", J, Uo, optimize=True)
    Jc = np.einsum("kajb,ac->kcjb", Jc, Uv, optimize=True)
    Jc = np.einsum("kcjb,jl->kclb", Jc, Uo, optimize=True)
    Jc = np.einsum("kclb,bd->kcld", Jc, Uv, optimize=True)
    D = (wv[None, :, None, None] + wv[None, None, None, :]
         - wo[:, None, None, None] - wo[None, None, :, None])
    assert D.min() > 0, "non-positive denominator"
    tc = -Jc / D
    t = np.einsum("kcld,ik->icld", tc, Uo, optimize=True)
    t = np.einsum("icld,ac->iald", t, Uv, optimize=True)
    t = np.einsum("iald,jl->iajd", t, Uo, optimize=True)
    t = np.einsum("iajd,bd->iajb", t, Uv, optimize=True)
    return base.mp2_energy(t, J)


def masked_energy(J, Foo, Fvv, mask):
    t, niter, relres = base.solve_masked_mp2(J, Foo, Fvv, mask)
    if relres > 1e-9:
        log(f"  WARNING: CG relres {relres:.2e}")
    return base.mp2_energy(t, J), niter


# ------------------------------------------------------- C1: aux domains

def fit_domains(A, V, doms):
    """Domain-local same-kernel fit with per-i aux sets `doms` (pair domain
    D_ij = D_i u D_j) — the lmp2_direct stage-5 formulation. Returns
    (J, mean pair-domain size, max pair-domain size, n_solvefail)."""
    no, nv, naux = A.shape
    J = np.zeros((no, nv, no, nv))
    sizes = []
    nfail = 0
    for i in range(no):
        for j in range(i, no):
            d = np.union1d(doms[i], doms[j])
            if len(d) == 0:
                raise RuntimeError(f"empty aux domain for pair ({i},{j})")
            sizes.append(len(d))
            Vdd = V[np.ix_(d, d)]
            try:
                c = np.linalg.solve(Vdd, A[i, :, d])  # (dom, nv)
            except np.linalg.LinAlgError:
                nfail += 1
                continue
            blk = c.T @ A[j, :, d]
            J[i, :, j, :] = blk
            J[j, :, i, :] = blk.T
    return J, float(np.mean(sizes)), int(np.max(sizes)), nfail


def aux_doms_dist(ctx, r):
    dist = np.linalg.norm(ctx["aux_xyz"][None, :, :]
                          - ctx["occ_cen"][:, None, :], axis=2)
    return [np.nonzero(dist[i] <= r)[0] for i in range(ctx["no"])]


def aux_score(ctx, metric_norm):
    """s_i(P) = max_a |(P|ia)| (optionally / sqrt(V_PP)) — the unwhitened
    strip magnitude the Rust direct path holds after stage 3."""
    s = np.abs(ctx["A"]).max(axis=1)  # (no, naux)
    if metric_norm:
        s = s / np.sqrt(np.maximum(np.diag(ctx["V"]), 1e-300))
    return s


def aux_doms_mag(ctx, tau, metric_norm=False, mutate=False):
    s = aux_score(ctx, metric_norm)
    doms = []
    for i in range(ctx["no"]):
        d = np.nonzero(s[i] >= tau)[0]
        if mutate and len(d) > 1:
            d = np.delete(d, int(np.argmax(s[i][d])))
        doms.append(d)
    return doms


def aux_doms_tail(ctx, budget, metric_norm=False):
    """Sec-26-style L1 tail: per i drop the smallest-score aux functions
    while the dropped-score sum stays <= budget."""
    s = aux_score(ctx, metric_norm)
    doms = []
    for i in range(ctx["no"]):
        order = np.argsort(s[i])  # ascending
        cum = np.cumsum(s[i][order])
        ndrop = int(np.searchsorted(cum, budget, side="right"))
        doms.append(np.sort(order[ndrop:]))
    return doms


def dom_locality(ctx, doms):
    """Max/mean-p95 distance of selected aux functions from their
    occupied's centroid — the distance-ball-in-disguise witness."""
    dist = np.linalg.norm(ctx["aux_xyz"][None, :, :]
                          - ctx["occ_cen"][:, None, :], axis=2)
    dmax, d95 = 0.0, []
    for i, d in enumerate(doms):
        if len(d):
            dmax = max(dmax, dist[i][d].max())
            d95.append(np.percentile(dist[i][d], 95))
    return dmax, float(np.mean(d95))


def run_aux(ctx, e_ref, tag, outfh):
    log("  -- C1 aux-domain frontier (dE = E_full(J_dom) - E_full(J_glob)) --")
    sweeps = ([("dist", r, lambda r=r: aux_doms_dist(ctx, r))
               for r in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0)]
              + [("mag", t, lambda t=t: aux_doms_mag(ctx, t))
                 for t in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)]
              + [("magm", t, lambda t=t: aux_doms_mag(ctx, t, metric_norm=True))
                 for t in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)]
              + [("tail", b, lambda b=b: aux_doms_tail(ctx, b))
                 for b in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4)])
    for rule, knob, mk in sweeps:
        t0 = time.time()
        doms = mk()
        if any(len(d) == 0 for d in doms):
            log(f"    {rule:>4s} {knob:g} EMPTY DOMAIN — skipped")
            continue
        Jd, dmean, dmax_sz, nfail = fit_domains(ctx["A"], ctx["V"], doms)
        e = solve_full_mp2_closed(Jd, ctx["Foo"], ctx["Fvv"])
        de = e - e_ref
        djmax = np.abs(Jd - ctx["Jg"]).max()
        locmax, loc95 = dom_locality(ctx, doms)
        row = (f"{tag} aux {rule:>4s} knob={knob:<9g} pairdom={dmean:7.1f}/"
               f"{dmax_sz:4d} of {ctx['naux']} dE={de:+.3e} maxdJ={djmax:.2e} "
               f"locmax={locmax:5.1f} loc95={loc95:5.1f} solvefail={nfail} "
               f"({time.time()-t0:.1f}s)")
        log("    " + row)
        outfh.write(row + "\n")
        outfh.flush()


# --------------------------------------------- C2: virtual candidates

def q_fit_global(ctx):
    """q_ia = sqrt(max((ia|ia)_globalfit, 0))."""
    no, nv, naux = ctx["A"].shape
    Af = ctx["A"].reshape(no * nv, naux)
    w, u = np.linalg.eigh(ctx["V"])
    if w.min() <= 1e-10 * w.max():
        raise RuntimeError("RI metric near-singular")
    C = Af @ ((u / w) @ u.T)
    q2 = np.einsum("xp,xp->x", Af, C)
    return np.sqrt(np.maximum(q2, 0.0)).reshape(no, nv)


def q_fit_local(ctx, r):
    """Strip-local fit diagonal: V_DD restricted to the distance domain
    D_i(r) — the quantity the Rust direct path can actually compute."""
    doms = aux_doms_dist(ctx, r)
    no, nv = ctx["no"], ctx["nv"]
    q = np.zeros((no, nv))
    for i in range(no):
        d = doms[i]
        c = np.linalg.solve(ctx["V"][np.ix_(d, d)], ctx["A"][i, :, d])
        q2 = np.einsum("pa,pa->a", ctx["A"][i, :, d], c)
        q[i] = np.sqrt(np.maximum(q2, 0.0))
    return q


def cand_schwarz(q, threshold):
    """cand[i,a,j] = a in C_ij iff q_ia*qmax_j >= th or q_ja*qmax_i >= th
    (symmetric in i,j — both Eq-8 orientations)."""
    qmax = q.max(axis=1)  # (no,)
    c1 = q[:, :, None] * qmax[None, None, :] >= threshold  # [i,a,j]
    return c1 | c1.transpose(2, 1, 0)


def cand_dist(ctx, rv):
    d = np.linalg.norm(ctx["virt_cen"][None, :, :]
                       - ctx["occ_cen"][:, None, :], axis=2)  # (no, nv)
    inr = d <= rv  # inr[i, a]
    # cand[i,a,j] = inr[i,a] | inr[j,a]
    return inr[:, :, None] | inr.T[None, :, :]


def allow_from_cand(cand):
    """allow[i,a,j,b] = cand[i,a,j] & cand[i,b,j] (a AND b in C_ij)."""
    return cand[:, :, :, None] & cand.transpose(0, 2, 1)[:, None, :, :]


def cand_sizes(cand):
    """Mean/max |C_ij| over unique pairs i<=j."""
    no = cand.shape[0]
    sizes = [int(cand[i, :, j].sum()) for i in range(no) for j in range(i, no)]
    return float(np.mean(sizes)), int(np.max(sizes))


def run_virt(ctx, tag, outfh, eps_list, qloc_radius=10.0):
    log("  -- C2 virtual-candidate frontier (dE vs eps-only mask, same Jg) --")
    J = ctx["Jg"]
    K = J.transpose(0, 3, 2, 1)
    qg = q_fit_global(ctx)
    ql = q_fit_local(ctx, qloc_radius)
    qdev = np.abs(qg - ql).max()
    log(f"    q_fit: global vs strip-local(r={qloc_radius:g}) max dev "
        f"{qdev:.2e} (qmax {qg.max():.3f})")
    for eps in eps_list:
        mask8 = (np.abs(J) > eps) | (np.abs(K) > eps)
        e8, it8 = masked_energy(J, ctx["Foo"], ctx["Fvv"], mask8)
        row = (f"{tag} virt eps={eps:g} eps-only E={e8:.10f} "
               f"kept={int(mask8.sum())} cg={it8}")
        log("    " + row)
        outfh.write(row + "\n")
        sweeps = ([("rv", r, lambda r=r: cand_dist(ctx, r))
                   for r in (4.0, 6.0, 8.0, 10.0, 12.0)]
                  + [("kap", k, lambda k=k: cand_schwarz(qg, k * eps))
                     for k in (0.1, 0.3, 1.0, 3.0, 10.0)]
                  + [("kapl", k, lambda k=k: cand_schwarz(ql, k * eps))
                     for k in (0.3, 1.0, 3.0)])
        for rule, knob, mk in sweeps:
            t0 = time.time()
            cand = mk()
            allow = allow_from_cand(cand)
            escape = mask8 & ~allow
            n_esc = int(escape.sum())
            worst_esc = float(np.abs(J[escape]).max()) if n_esc else 0.0
            e, it = masked_energy(J, ctx["Foo"], ctx["Fvv"], mask8 & allow)
            de = e - e8
            cmean, cmax = cand_sizes(cand)
            row = (f"{tag} virt eps={eps:g} {rule:>4s} knob={knob:<6g} "
                   f"|C|={cmean:6.1f}/{cmax:4d} of {ctx['nv']} dE={de:+.3e} "
                   f"escapes={n_esc} worst|J|esc={worst_esc:.2e} cg={it} "
                   f"({time.time()-t0:.1f}s)")
            log("    " + row)
            outfh.write(row + "\n")
            outfh.flush()


# ----------------------------------------------------- anchors + mutations

def run_anchors(ctx, tag, outfh):
    log("  -- anchors (must pass BEFORE any sweep) --")
    # A4: two-algebra energy cross-check on the global fit
    e_closed = solve_full_mp2_closed(ctx["Jg"], ctx["Foo"], ctx["Fvv"])
    e_cg, _ = masked_energy(ctx["Jg"], ctx["Foo"], ctx["Fvv"],
                            np.ones_like(ctx["Jg"], dtype=bool))
    d4 = abs(e_closed - e_cg)
    log(f"    A4 closed-form vs masked-CG (eps=0): |dE|={d4:.3e} "
        f"E={e_closed:.10f}")
    if d4 > 1e-9:
        raise SystemExit("A4 FAILED: two-algebra eps=0 energies disagree")
    # A1: distance rule, trivial radius
    Jd, _, _, nf = fit_domains(ctx["A"], ctx["V"], aux_doms_dist(ctx, 1e9))
    d1 = np.abs(Jd - ctx["Jg"]).max()
    log(f"    A1 dist(1e9): max|dJ|={d1:.3e} solvefail={nf}")
    if d1 > 1e-10 or nf:
        raise SystemExit("A1 FAILED")
    # A2: magnitude rule, trivial threshold
    Jd, _, _, nf = fit_domains(ctx["A"], ctx["V"], aux_doms_mag(ctx, 0.0))
    d2 = np.abs(Jd - ctx["Jg"]).max()
    log(f"    A2 mag(0): max|dJ|={d2:.3e} solvefail={nf}")
    if d2 > 1e-10 or nf:
        raise SystemExit("A2 FAILED")
    # M1: gutted magnitude rule must fail A2
    Jd, _, _, _ = fit_domains(ctx["A"], ctx["V"],
                              aux_doms_mag(ctx, 0.0, mutate=True))
    dm = np.abs(Jd - ctx["Jg"]).max()
    log(f"    M1 mag(0, drop-top): max|dJ|={dm:.3e} "
        f"{'MUTATION-OK' if dm > 1e-10 else 'MUTATION-BROKEN'}")
    if dm <= 1e-10:
        raise SystemExit("M1 BROKEN: gutted aux rule still passes A2")
    # A3: trivial virtual candidates leave the mask untouched
    eps = 1e-3
    J = ctx["Jg"]
    mask8 = (np.abs(J) > eps) | (np.abs(J.transpose(0, 3, 2, 1)) > eps)
    for name, cand in (("kap0", cand_schwarz(q_fit_global(ctx), 0.0)),
                       ("rv1e9", cand_dist(ctx, 1e9))):
        if not cand.all():
            raise SystemExit(f"A3 FAILED: {name} candidate mask not all-true")
    log("    A3 kap0/rv1e9: candidate masks all-true (mask untouched) OK")
    # M2: corrupt q must break conservativeness at kappa=1
    qm = q_fit_global(ctx).copy()
    qm[np.unravel_index(np.argmax(qm), qm.shape)] = 0.0
    allow = allow_from_cand(cand_schwarz(qm, eps))
    n_esc = int((mask8 & ~allow).sum())
    log(f"    M2 zeroed-top-q kappa=1: escapes={n_esc} "
        f"{'MUTATION-OK' if n_esc > 0 else 'MUTATION-BROKEN'}")
    if n_esc == 0:
        raise SystemExit("M2 BROKEN: corrupted q still conservative")
    outfh.write(f"{tag} anchors PASSED (A1 {d1:.1e} A2 {d2:.1e} A4 {d4:.1e}) "
                f"mutations FAILED-as-required (M1 {dm:.1e} M2 esc={n_esc})\n")
    outfh.flush()
    return e_closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--phase", default="all",
                    choices=["anchors", "aux", "virt", "all"])
    ap.add_argument("--eps", default="1e-3,1e-4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps_list = [float(x) for x in a.eps.split(",") if x]
    tag = (f"{os.path.basename(a.xyz).replace('.xyz','')} {a.basis} "
           f"{'coul' if a.omega is None else f'erfc{a.omega:g}'}")
    outpath = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "out", "eps_linked_maps.txt")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    ctx = setup(a.xyz, a.basis, a.omega)
    with open(outpath, "a") as outfh:
        e_ref = run_anchors(ctx, tag, outfh)
        if a.phase in ("aux", "all"):
            run_aux(ctx, e_ref, tag, outfh)
        if a.phase in ("virt", "all"):
            run_virt(ctx, tag, outfh, eps_list)
    log("  done.")


if __name__ == "__main__":
    main()
