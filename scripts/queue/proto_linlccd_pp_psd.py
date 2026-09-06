#!/usr/bin/env python3
"""pp-ladder PSD measurement for an integral-direct LinLCCD port
(PROTOTYPE FIRST; no Rust pp work until this script has a written verdict).
Vehicle: PySCF + scripts/amplitude_lmp2_proto.py machinery (Boys occupieds +
VV-HV virtuals, same-kernel RI).

THE QUESTION (WIKI-APPEND-drpa-direct.md, LinLCCD deferral): the ragged-CG
license (`solve_ragged_with`: matvec must be SPD on the pattern) rests on
notebook 13's "both ladder blocks are RI Grams of ONE consistent whitened B,
hence PSD". The integral-direct path would naturally build the pp ladder
per pair from a DOMAIN-LOCAL same-kernel fit
    m_fit[(a,c),(b,d)] = sum_{P,Q in D_ij} (ac|P) [V_DD^-1]_PQ (Q|bd),
one V_DD per pair domain — NOT a single consistent Gram. Does that break
PSD in practice, and does any indefiniteness grow with truncation?

STRUCTURE EXPLOITED (stated so the measurement is honest about scope):
  - The pp operator is BLOCK-DIAGONAL over occupied pairs (i,j) in T-space:
    (pp T)_ij depends only on T_ij. Its global spectrum is the union of the
    per-pair block spectra, so per-pair lambda_min IS the pp PSD question.
  - Eigenvalue interlacing: restricting a symmetric block to the Eq-8
    pattern subspace can only RAISE lambda_min, so full-block lambda_min is
    the conservative (eps-independent) bound; pattern-restricted lambda_min
    is what CG actually sees at finite eps. Both are measured.
  - Both fitted and licensed per-pair blocks are SYMMETRIC by construction
    ((ac|P) symmetric in a,c; V_DD^-1 symmetric); measured, not assumed
    (an asymmetric block would be its own CG killer, reported separately).
  - The CG operator is A = F + hh + pp with F PD (gapped; diagonal floor =
    min denominator) and hh kept LICENSED (global-metric Gram — the
    PSD-safe occ-occ strip extension of the deferral note). By Weyl,
    lambda_min(A) >= lambda_min(F) + lambda_min(pp_fit) on the pattern, so
    the load-bearing comparison is |most negative pp eig| vs the Fock
    denominator floor.

EXACTNESS ANCHORS (written BEFORE any sweep; hard exit on failure):
  A1  the LICENSED global pp supermatrix (exchange pairing
      M[(a,b),(c,d)] = sum_P Btld[P,(a,c)] Btld[P,(b,d)], one whitened B)
      is PSD to the numerical floor: lambda_min >= -1e-10 * max|diag|.
      This measures notebook 13's own claim rather than assuming it —
      in the exchange pairing it is sum_P Y_P (x) Y_P with Y_P symmetric,
      which is NOT a manifest Gram, so it gets measured, not cited.
  A2  fitted pp at the trivial domain (D_ij = full aux) equals licensed:
      max|m_fit - m_lic| <= 1e-9 on every sampled pair (V^-1 vs
      V^-1/2.V^-1/2 reassociation floor).
  A3  full-solver anchor (water, eps=0, licensed pp): masked-CG LinLCCD
      energy == direct dense solve of P A P t = -P J on the pattern
      subspace (independent algebra) to <= 1e-9.
  A4  eps=0 trivial-radius fitted CG energy == licensed CG energy <= 1e-9
      (construction end-to-end in the trivial limit).
MUTATION ARMS (each must FAIL loudly; run before sweeps):
  M1  drop the largest-|Avv| aux function from every pair domain at trivial
      radius -> A2 must fail (max|dm| > 1e-6).
  M2  eig-detector mutation: subtract 2x the largest whitened-K rank-1 term
      from one fitted pair block -> its lambda_min must go clearly negative
      (< -1e-6 * max|diag|). Verifies the pipeline can SEE indefiniteness.

ARTIFACT HYPOTHESES (X = real physics/structure, Y = broken construction;
X != Y so the measurement can distinguish):
  X-safe:   fitted per-pair lambda_min stays >= -delta with delta ORDERS
            below the Fock denominator floor at production radii (8-12
            Bohr), roughly flat or shrinking as radius grows, identical
            behavior across systems/operators; CG converges with
            E_fit - E_lic sub-dominant to the eps-truncation error.
  X-broken: lambda_min goes negative at a scale COMPARABLE to the Fock
            floor, monotonically worse as the domain shrinks and as the
            system grows; CG shows p^T A p < 0 events or stalls.
  Y (artifact): lambda_min erratic/non-monotone in radius with failures
            already at the trivial limit, or asymmetric blocks, or licensed
            arm failing A1 the same way (a shared-construction bug, not a
            fit property). A1/A2 + M1/M2 are the Y-detectors.
STOP CONDITION: too-clean coincidence (e.g. lambda_min exactly zero across
all radii/systems) is arithmetic, not chemistry — audit before writing up.

Usage (PySCF may use several BLAS threads; long runs under ferric-limited):
  scripts/ferric-limited --max=4G --high=3600M -- \
    uv run --no-sync python scripts/queue/proto_linlccd_pp_psd.py \
      --xyz testdata/molecules/water.xyz [--omega 1.0] \
      [--phase anchors|mutations|sweep|all] [--eps 1e-3,1e-4] \
      [--radii 1e6,12,10,8,6,4] [--cg] [--out scripts/queue/out/x.txt]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amplitude_lmp2_proto as base  # noqa: E402
from pyscf import gto, scf, lo, df  # noqa: E402
from pyscf.data import elements  # noqa: E402

log = base.log


# ---------------------------------------------------------------- setup

def build_ri_all(mol, C_act, C_vloc, omega):
    """Same-kernel RI ingredients incl. the vv and oo 3-index blocks:
    Aov[i,a,P]=(ia|P), Aoo[i,k,P]=(ik|P), Avv[a,c,P]=(ac|P), V=(P|Q)."""
    auxmol = df.addons.make_auxmol(mol, df.addons.make_auxbasis(mol, mp2fit=True))
    if omega is not None:
        with mol.with_range_coulomb(-omega):
            ints3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        with auxmol.with_range_coulomb(-omega):
            V = auxmol.intor("int2c2e")
    else:
        ints3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        V = auxmol.intor("int2c2e")
    Aov = np.einsum("mnp,mi,na->iap", ints3c, C_act, C_vloc, optimize=True)
    Aoo = np.einsum("mnp,mi,nk->ikp", ints3c, C_act, C_act, optimize=True)
    half = np.einsum("mnp,ma->anp", ints3c, C_vloc, optimize=True)
    Avv = np.einsum("anp,nc->acp", half, C_vloc, optimize=True)
    coords = auxmol.atom_coords()
    aux_atom = np.array([lbl[0] for lbl in auxmol.ao_labels(fmt=None)])
    return Aov, Aoo, Avv, V, coords[aux_atom]


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
    Aov, Aoo, Avv, V, aux_xyz = build_ri_all(mol, C_act, C_vloc, omega)
    no, nv, naux = Aov.shape
    # whitening (one consistent global factor) + V^-1 for the fits
    w, u = np.linalg.eigh(V)
    if w.min() <= 1e-10 * w.max():
        raise RuntimeError(f"RI metric near-singular: {w.min():.2e}")
    Vmsqrt = (u / np.sqrt(w)) @ u.T
    Jg = np.einsum("iap,pq,jbq->iajb",
                   Aov, (u / w) @ u.T, Aov, optimize=True)
    Bvv = Avv.reshape(nv * nv, naux) @ Vmsqrt        # whitened, rows (a,c)
    Boo = Aoo.reshape(no * no, naux) @ Vmsqrt
    oo4 = (Boo @ Boo.T).reshape(no, no, no, no)      # (ik|jl), licensed hh
    occ_cen = base.boys_centroids(mol, C_act)
    log(f"== {os.path.basename(xyz)} basis={basis} "
        f"op={'coulomb' if omega is None else f'erfc{omega:g}'} "
        f"no={no} nv={nv} naux={naux} nao={mol.nao} ({time.time()-t0:.1f}s)")
    return dict(mol=mol, Jg=Jg, Foo=Foo, Fvv=Fvv, Avv=Avv, Bvv=Bvv, oo4=oo4,
                V=V, aux_xyz=aux_xyz, occ_cen=occ_cen, no=no, nv=nv,
                naux=naux)


# ------------------------------------------------- pattern + pair domains

def eq8_mask(J, eps):
    if eps == 0.0:
        return np.ones(J.shape, dtype=bool)
    K = J.transpose(0, 3, 2, 1)
    return (np.abs(J) > eps) | (np.abs(K) > eps)


def pattern_pairs(mask):
    """(i, j, da, db, pat) per surviving pair — base.build_ragged layout."""
    return base.build_ragged(mask)


def aux_domains(S, radius):
    """in_r[i, P] under the distance rule (same as ri_j_domain)."""
    d = np.linalg.norm(S["aux_xyz"][None, :, :] - S["occ_cen"][:, None, :],
                       axis=2)
    return d <= radius


# ------------------------------------------------- per-pair pp blocks

def pp_block_licensed(S, da, db):
    """m_lic[(a,c),(b,d)] over a,c in da; b,d in db — full aux rows, ONE
    global whitening (exactly the current Rust Full-tier gather)."""
    nv = S["nv"]
    ra = (da[:, None] * nv + da[None, :]).ravel()    # (a,c) flat ids
    rb = (db[:, None] * nv + db[None, :]).ravel()
    return S["Bvv"][ra] @ S["Bvv"][rb].T


def pp_block_fitted(S, da, db, dom, mutate_m1=False):
    """Domain-local same-kernel fit: m_fit = Aa V_DD^-1 Ab^T with
    Aa = (ac|P in D_ij). One V_DD per pair domain — the direct path's
    natural formulation. mutate_m1 drops the largest-|Avv| aux function
    from the domain (M1)."""
    nv, naux = S["nv"], S["naux"]
    if mutate_m1:
        anorm = np.abs(S["Avv"].reshape(nv * nv, naux)[:, dom]).sum(axis=0)
        dom = np.delete(dom, int(np.argmax(anorm)))
    Aa = S["Avv"][np.ix_(da, da)][:, :, dom].reshape(len(da) ** 2, len(dom))
    Ab = S["Avv"][np.ix_(db, db)][:, :, dom].reshape(len(db) ** 2, len(dom))
    Vdd = S["V"][np.ix_(dom, dom)]
    return Aa @ np.linalg.solve(Vdd, Ab.T)


def as_operator(m, na, nb):
    """(a,c),(b,d) pairing -> operator O[(a,b),(c,d)] on vec(T[c,d])."""
    return (m.reshape(na, na, nb, nb).transpose(0, 2, 1, 3)
            .reshape(na * nb, na * nb))


def block_lambda_min(m, na, nb, pat=None, skip_full=False):
    """(lambda_min_full, lambda_min_patterned, asym) of one pp block.
    skip_full=True computes only the pattern-restricted eig (what CG sees;
    full-block is the conservative bound via interlacing) — used for large
    systems where the full-block eigh is sampled, not exhaustive."""
    O = as_operator(m, na, nb)
    asym = np.abs(O - O.T).max()
    Os = 0.5 * (O + O.T)
    lmin_full = np.nan if skip_full else float(np.linalg.eigvalsh(Os)[0])
    if pat is None:
        lmin_pat = lmin_full
    else:
        keep = pat.ravel()
        if not keep.any():
            lmin_pat = 0.0
        elif keep.all() and not skip_full:
            lmin_pat = lmin_full
        else:
            lmin_pat = float(np.linalg.eigvalsh(Os[np.ix_(keep, keep)])[0])
    return lmin_full, lmin_pat, asym


# ------------------------------------------------- dense masked solver

def linlccd_matvec(S, t, mask, pp_apply):
    """A(t) = F(t) + hh(t) + pp(t), pattern-projected (dense layout)."""
    Foo, Fvv, oo4 = S["Foo"], S["Fvv"], S["oo4"]
    r = np.einsum("ac,icjb->iajb", Fvv, t, optimize=True)
    r += np.einsum("iajc,cb->iajb", t, Fvv, optimize=True)
    r -= np.einsum("ik,kajb->iajb", Foo, t, optimize=True)
    r -= np.einsum("iakb,kj->iajb", t, Foo, optimize=True)
    r += np.einsum("ikjl,kalb->iajb", oo4, t, optimize=True)
    r += pp_apply(t)
    np.multiply(r, mask, out=r)
    return r


def make_pp_apply_global(S):
    """Licensed pp via one global operator GEMM (eps=0-capable)."""
    nv, no = S["nv"], S["no"]
    M = S["Bvv"] @ S["Bvv"].T                        # (ac),(bd)
    Mop = as_operator(M, nv, nv)                     # (ab),(cd)

    def apply(t):
        tt = t.transpose(1, 3, 0, 2).reshape(nv * nv, no * no)
        out = Mop @ tt
        return out.reshape(nv, nv, no, no).transpose(2, 0, 3, 1)
    return apply


def make_pp_apply_blocks(S, pairs, blocks):
    """pp from per-pair operator blocks (fitted or licensed-restricted):
    outside each pair's (da x db) sub-block pp contributes nothing —
    matching the ragged Rust path, where amplitudes only exist there."""
    no, nv = S["no"], S["nv"]

    def apply(t):
        r = np.zeros_like(t)
        for (i, j, da, db, _), O in zip(pairs, blocks):
            tb = t[i, :, j, :][np.ix_(da, db)]              # (|da|,|db|)
            out = (O @ tb.ravel()).reshape(len(da), len(db))
            r[i, :, j, :][np.ix_(da, db)] += out
        return r
    return apply


def solve_cg(S, mask, pp_apply, rtol=1e-11, maxiter=600):
    """Masked preconditioned CG on P A P t = -P J. Returns
    (t, iters, relres, converged, n_indefinite) where n_indefinite counts
    p^T A p <= 0 events (the in-solver indefiniteness witness)."""
    J, Foo, Fvv = S["Jg"], S["Foo"], S["Fvv"]
    fo, fv = np.diag(Foo), np.diag(Fvv)
    D = (fv[None, :, None, None] + fv[None, None, None, :]
         - fo[:, None, None, None] - fo[None, None, :, None])
    assert D.min() > 0, "non-positive denominator: not a gapped system?"
    r = np.where(mask, -J, 0.0)
    bnorm = np.linalg.norm(r)
    t = np.zeros_like(J)
    if bnorm == 0.0:
        return t, 0, 0.0, True, 0
    z = r / D
    p = z.copy()
    rz = np.vdot(r, z)
    n_indef = 0
    relres = 1.0
    for it in range(1, maxiter + 1):
        Ap = linlccd_matvec(S, p, mask, pp_apply)
        pAp = np.vdot(p, Ap)
        if pAp <= 0.0:
            n_indef += 1
            if n_indef >= 3:
                return t, it, relres, False, n_indef
        alpha = rz / pAp
        t += alpha * p
        r -= alpha * Ap
        relres = np.linalg.norm(r) / bnorm
        if relres < rtol:
            return t, it, relres, True, n_indef
        z = r / D
        rz_new = np.vdot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return t, maxiter, relres, False, n_indef


def linlccd_energy(t, J):
    return 2.0 * np.vdot(t, J) - np.vdot(t.transpose(0, 3, 2, 1), J)


def direct_solve_energy(S, mask, pp_apply):
    """Independent-algebra reference (A3): materialize P A P on the pattern
    subspace column by column, np.linalg.solve. Small systems only."""
    idx = np.nonzero(mask.ravel())[0]
    n = len(idx)
    A = np.zeros((n, n))
    e = np.zeros(mask.size)
    for k, col in enumerate(idx):
        e[:] = 0.0
        e[col] = 1.0
        A[:, k] = linlccd_matvec(
            S, e.reshape(mask.shape), mask, pp_apply).ravel()[idx]
    rhs = -S["Jg"].ravel()[idx]
    tvec = np.linalg.solve(A, rhs)
    t = np.zeros(mask.size)
    t[idx] = tvec
    t = t.reshape(mask.shape)
    return linlccd_energy(t, S["Jg"]), float(np.linalg.eigvalsh(
        0.5 * (A + A.T))[0])


# ------------------------------------------------- phases

def phase_anchors(S):
    no, nv = S["no"], S["nv"]
    ok = True
    # A1: licensed global pp supermatrix PSD? MEASURED, not assumed.
    # REFRAMED 2026-09-06 after the erfc1/alkane_4 run: the LICENSED block
    # itself came back slightly indefinite (lambda_min = -6.85e-4, rel
    # -5.8e-3) with the construction anchors A2/A4 at machine precision —
    # i.e. notebook 13's absolute "RI Gram hence PSD" is FALSE for the
    # attenuated operator (the exact-integral pointwise-positivity
    # argument does not survive the RI projection). The OPERATIVE CG
    # license is therefore bounded pp indefiniteness << the Fock
    # denominator floor (Weyl), for licensed AND fitted alike — that is
    # the quantity gated here. An indefiniteness above 5% of the floor
    # STOPS; a small one is recorded as a FINDING and the sweep proceeds
    # to measure whether FITTING makes it worse with truncation.
    M = S["Bvv"] @ S["Bvv"].T
    Op = as_operator(M, nv, nv)
    asym = np.abs(Op - Op.T).max()
    ev = np.linalg.eigvalsh(0.5 * (Op + Op.T))
    scale = np.abs(np.diag(Op)).max()
    fo, fv = np.diag(S["Foo"]), np.diag(S["Fvv"])
    dmin = float((fv[None, :, None, None] + fv[None, None, None, :]
                  - fo[:, None, None, None] - fo[None, None, :, None]).min())
    log(f"  A1 licensed global pp: lambda_min={ev[0]:+.3e} "
        f"lambda_max={ev[-1]:.3e} maxdiag={scale:.3e} asym={asym:.1e} "
        f"rel_lmin={ev[0]/scale:+.3e} fock_floor={dmin:.3f}")
    if ev[0] < -0.05 * dmin:
        log("  A1 FAILED: licensed pp indefiniteness is at the scale of "
            "the Fock floor — the CG license itself is at issue; STOP")
        ok = False
    elif ev[0] < -1e-10 * scale:
        log(f"  A1 FINDING: licensed pp is slightly INDEFINITE "
            f"(lambda_min={ev[0]:+.3e}, {abs(ev[0])/dmin:.2e} of the Fock "
            f"floor) — the notebook-13 absolute-PSD claim is false for "
            f"this operator; operative license = Fock-floor margin; "
            f"sweep proceeds to compare fitted vs licensed")
    else:
        log("  A1 PASSED (PSD at the numerical floor)")
    # A2: fitted at trivial domain == licensed (domain-independent of the
    # pair at the trivial limit: da = all virtuals, dom = all aux)
    dom = np.arange(S["naux"])
    da = np.arange(nv)
    m_lic = pp_block_licensed(S, da, da)
    m_fit = pp_block_fitted(S, da, da, dom)
    dmax = float(np.abs(m_fit - m_lic).max())
    log(f"  A2 fitted(trivial domain) vs licensed: max|dm|={dmax:.3e} "
        f"{'PASSED' if dmax <= 1e-9 else 'FAILED'}")
    ok &= dmax <= 1e-9
    # A3: CG vs direct solve, licensed pp, eps=0. The direct solve
    # materializes the (no.nv)^2-dim pattern operator — small systems only.
    if (no * nv) ** 2 <= 4096:
        mask = eq8_mask(S["Jg"], 0.0)
        pp = make_pp_apply_global(S)
        t, it, rr, conv, nind = solve_cg(S, mask, pp)
        e_cg = linlccd_energy(t, S["Jg"])
        e_dir, lmin_a = direct_solve_energy(S, mask, pp)
        de = abs(e_cg - e_dir)
        log(f"  A3 CG vs direct solve (eps=0, licensed): E_cg={e_cg:.10f} "
            f"E_direct={e_dir:.10f} |dE|={de:.3e} cg={it} "
            f"lambda_min(P A P)={lmin_a:+.3e} "
            f"{'PASSED' if de <= 1e-9 else 'FAILED'}")
        ok &= de <= 1e-9
    else:
        log("  A3 skipped (system too large for direct solve; anchored on "
            "the small system)")
    # A4: eps=0 trivial-radius fitted CG == licensed CG. At eps=0 every
    # pair has da = db = all virtuals and the trivial domain, so ONE block
    # is shared by every pair (memory: nv^4, guarded).
    if nv ** 4 * 8 <= 1 << 30:
        mask = eq8_mask(S["Jg"], 0.0)
        pairs = pattern_pairs(mask)
        dom = np.arange(S["naux"])
        da0 = np.arange(nv)
        shared = as_operator(pp_block_fitted(S, da0, da0, dom), nv, nv)
        blocks = [shared for _ in pairs]
        ppf = make_pp_apply_blocks(S, pairs, blocks)
        tf, itf, rrf, convf, nindf = solve_cg(S, mask, ppf)
        e_fit = linlccd_energy(tf, S["Jg"])
        ppl = make_pp_apply_global(S)
        tl, itl, _, _, _ = solve_cg(S, mask, ppl)
        e_lic = linlccd_energy(tl, S["Jg"])
        de = abs(e_fit - e_lic)
        log(f"  A4 trivial-radius fitted vs licensed CG (eps=0): "
            f"E_fit={e_fit:.10f} E_lic={e_lic:.10f} |dE|={de:.3e} "
            f"{'PASSED' if de <= 1e-9 else 'FAILED'}")
        ok &= de <= 1e-9
    else:
        log("  A4 skipped at this size (anchored on the small system)")
    if not ok:
        raise SystemExit("exactness anchor FAILED; no sweep run")
    log("  ALL ANCHORS PASSED")


def phase_mutations(S):
    no, nv, naux = S["no"], S["nv"], S["naux"]
    dom = np.arange(naux)
    da = np.arange(nv)
    # M1: gutted domain must break A2
    m_lic = pp_block_licensed(S, da, da)
    m_fit = pp_block_fitted(S, da, da, dom, mutate_m1=True)
    dmax = float(np.abs(m_fit - m_lic).max())
    v1 = "MUTATION-OK (A2 fails as required)" if dmax > 1e-6 else \
         "MUTATION-BROKEN: gutted domain still matches!"
    log(f"  M1 {v1}  max|dm|={dmax:.3e}")
    # M2: injected indefiniteness must be SEEN by the eig detector
    m = pp_block_fitted(S, da, da, dom)
    Btld = S["Bvv"]
    k = int(np.argmax(np.abs(Btld).max(axis=0)))
    m2 = m - 2.0 * np.outer(Btld[:, k], Btld[:, k])
    lmin, _, _ = block_lambda_min(m2, nv, nv)
    scale = np.abs(np.diag(as_operator(m, nv, nv))).max()
    v2 = "MUTATION-OK (indefiniteness detected)" if lmin < -1e-6 * scale \
        else "MUTATION-BROKEN: injected negative mode not seen!"
    log(f"  M2 {v2}  lambda_min={lmin:+.3e} (scale {scale:.3e})")
    if "BROKEN" in v1 or "BROKEN" in v2:
        raise SystemExit("mutation arm BROKEN; measurement not trusted")


def phase_sweep(S, eps_list, radii, run_cg, out=None, tag="",
                sample_full=0):
    """The measurement: per-pair fitted/licensed lambda_min vs radius/eps,
    Fock floor, CG behavior + energies."""
    no, nv = S["no"], S["nv"]
    fo, fv = np.diag(S["Foo"]), np.diag(S["Fvv"])
    Dmin = float((fv[None, :, None, None] + fv[None, None, None, :]
                  - fo[:, None, None, None] - fo[None, None, :, None]).min())
    log(f"  Fock denominator floor (min D) = {Dmin:.4f} Ha")
    e_lic0 = None
    if run_cg and (no * nv) ** 2 <= 2_000_000:
        mask0 = eq8_mask(S["Jg"], 0.0)
        t0_, *_ = solve_cg(S, mask0, make_pp_apply_global(S))
        e_lic0 = linlccd_energy(t0_, S["Jg"])
        log(f"  E(eps=0, licensed) = {e_lic0:.10f}")
    rows = []
    for eps in eps_list:
        mask = eq8_mask(S["Jg"], eps)
        pairs = pattern_pairs(mask)
        keep = mask.mean()
        # licensed CG at this eps (radius-independent baseline)
        e_lic = np.nan
        lic_it = 0
        if run_cg:
            plc = {}
            pl = []
            for (_, _, da, db, _) in pairs:
                key = (da.tobytes(), db.tobytes())
                if key not in plc:
                    plc[key] = as_operator(pp_block_licensed(S, da, db),
                                           len(da), len(db))
                pl.append(plc[key])
            tl, lic_it, rrl, convl, nindl = solve_cg(
                S, mask, make_pp_apply_blocks(S, pairs, pl))
            e_lic = linlccd_energy(tl, S["Jg"])
            del pl, plc
        # stratified pair sample for the full-block eighs on large systems
        # (0 = exhaustive); pattern-restricted eigs stay exhaustive always
        if sample_full and sample_full < len(pairs):
            sel = set(np.linspace(0, len(pairs) - 1, sample_full,
                                  dtype=int).tolist())
        else:
            sel = set(range(len(pairs)))
        # licensed pattern-restricted lambda_min per pair — radius-free,
        # computed once per eps (dedup by (da, db, pat))
        lic_cache = {}
        lmin_lic_pat = []
        for px, (i, j, da, db, pat) in enumerate(pairs):
            key = (da.tobytes(), db.tobytes(), pat.tobytes())
            if key not in lic_cache:
                ml = pp_block_licensed(S, da, db)
                _, lpl, _ = block_lambda_min(ml, len(da), len(db), pat,
                                             skip_full=px not in sel)
                lic_cache[key] = lpl
            lmin_lic_pat.append(lic_cache[key])
        del lic_cache
        for radius in radii:
            t0 = time.time()
            in_r = aux_domains(S, radius)
            # dedup identical (domain, da, db, pat) work — the Rust V_DD
            # grouping analogue; at eps=0 this collapses to unique domains
            fit_cache = {}
            lmins_full, lmins_pat, asyms = [], [], []
            dsizes = []
            blocks = []
            for px, (i, j, da, db, pat) in enumerate(pairs):
                dom = np.nonzero(in_r[i] | in_r[j])[0]
                if len(dom) == 0:
                    raise RuntimeError(f"empty aux domain pair ({i},{j})")
                dsizes.append(len(dom))
                key = (dom.tobytes(), da.tobytes(), db.tobytes(),
                       pat.tobytes())
                if key not in fit_cache:
                    m = pp_block_fitted(S, da, db, dom)
                    lf, lp, asym = block_lambda_min(
                        m, len(da), len(db), pat, skip_full=px not in sel)
                    O = as_operator(m, len(da), len(db)) if run_cg else None
                    fit_cache[key] = (lf, lp, asym, O)
                lf, lp, asym, O = fit_cache[key]
                lmins_full.append(lf)
                lmins_pat.append(lp)
                asyms.append(asym)
                if run_cg:
                    blocks.append(O)
            del fit_cache
            lmins_full = np.array(lmins_full)
            lmins_pat = np.array(lmins_pat)
            n_full = int(np.isfinite(lmins_full).sum())
            nneg = int((lmins_pat < -1e-10).sum())
            row = (f"{tag} eps={eps:g} r={radius:g} pairs={len(pairs)} "
                   f"dom(mean/max)={np.mean(dsizes):.0f}/{max(dsizes)} "
                   f"keep={keep:.4f} "
                   f"lmin_fit(full[{n_full}]/pat)="
                   f"{np.nanmin(lmins_full):+.3e}/"
                   f"{lmins_pat.min():+.3e} "
                   f"lmin_lic(pat)={min(lmin_lic_pat):+.3e} "
                   f"nneg={nneg}/{len(pairs)} "
                   f"asym_max={max(asyms):.1e} Dmin={Dmin:.3f}")
            if run_cg:
                tf, itf, rrf, convf, nindf = solve_cg(
                    S, mask, make_pp_apply_blocks(S, pairs, blocks))
                e_fit = linlccd_energy(tf, S["Jg"])
                row += (f" | cg_fit={itf}{'' if convf else '!DIV'}"
                        f" pAp_neg={nindf} E_fit={e_fit:.10f}"
                        f" dE(fit-lic)={e_fit-e_lic:+.3e}")
                if e_lic0 is not None:
                    row += f" dE_eps(lic-lic0)={e_lic-e_lic0:+.3e}"
                row += f" cg_lic={lic_it}"
            row += f" ({time.time()-t0:.1f}s)"
            log("  " + row)
            rows.append(row)
            if out:
                with open(out, "a") as f:
                    f.write(row + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--phase", default="all",
                    choices=["anchors", "mutations", "sweep", "all"])
    ap.add_argument("--eps", default="1e-3,1e-4")
    ap.add_argument("--radii", default="1e6,12,10,8,6,4")
    ap.add_argument("--cg", action="store_true",
                    help="also run the masked CG + energies per row "
                         "(skipped for large systems by default)")
    ap.add_argument("--sample-full", type=int, default=0,
                    help="full-block eighs only on N stratified pairs "
                         "(0 = exhaustive); pattern-restricted eigs are "
                         "always exhaustive")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = setup(a.xyz, a.basis, a.omega)
    eps_list = [float(x) for x in a.eps.split(",") if x]
    radii = [float(x) for x in a.radii.split(",") if x]
    tag = (f"{os.path.basename(a.xyz).removesuffix('.xyz')} "
           f"{'coul' if a.omega is None else f'erfc{a.omega:g}'}")
    if a.phase in ("anchors", "all"):
        phase_anchors(S)
    if a.phase in ("mutations", "all"):
        phase_mutations(S)
    if a.phase in ("sweep", "all"):
        phase_sweep(S, eps_list, radii, a.cg, a.out, tag,
                    sample_full=a.sample_full)


if __name__ == "__main__":
    main()
