"""EE-EOM-CCSD singlets, closed shell: numpy prototype of ferric-cc/src/eom_ccsd.rs.

Three constructions of the same right-eigenvalue problem, checked against one
another and against PySCF's `eomee_ccsd_singlet` per root:

1. `jacobian_matvec`: the CC Jacobian `J R = dOmega/dt . R` of PySCF's RCCSD
   residual, `Omega(t) = (update_amps(t) - t) * D`, by a five-point central
   difference. The CCSD residual is a polynomial of degree 4 in (t1, t2), and
   the five-point stencil is exact for degree <= 4, so the step size h only
   sets the rounding error (no truncation error). At converged amplitudes the
   Jacobian equals H-bar - E_CC on the singles+doubles space (the reference
   row and column decouple because Omega = 0), so its eigenvalues are the
   EOM-CCSD excitation energies. This shares no code with PySCF's EOM module.
2. `eom_matvec`: the H-bar product the Rust module implements, written out
   from the spin-adapted intermediates (the unblocked form of PySCF
   `eom_rccsd._IMDS.make_ee` + `eeccsd_matvec_singlet`), one einsum per term.
3. A dense matrix from (2) on unit vectors in the packed singlet space,
   diagonalized directly, and a non-Hermitian block Davidson on (2) with
   Olsen's correction: the solver the Rust module uses.

Exactness anchor: with t1 = t2 = 0 the singles-singles block is the CIS
(singlet TDA-HF) matrix delta_ij delta_ab (e_a - e_i) + 2 (ia|jb) - (ij|ab).

The singlet space: r1[i,a] and r2[i,j,a,b] with r2[i,j,a,b] = r2[j,i,b,a];
the packed vector holds r1 and the lower triangle of the symmetric (ia, jb)
matrix. The doubles block of the residual is symmetric by construction.

Run: python scripts/prototypes/eom_ccsd_proto.py   (seconds; exact integrals)
"""

from __future__ import annotations

import numpy as np
from pyscf import cc, gto, scf
from pyscf.cc import ccsd, eom_rccsd
from pyscf.lib import einsum

MAX_MEMORY_MB = 2000


# ----------------------------------------------------------------- packing


def pack(r1, r2):
    no, nv = r1.shape
    m = r2.transpose(0, 2, 1, 3).reshape(no * nv, no * nv)
    il = np.tril_indices(no * nv)
    return np.concatenate([r1.ravel(), m[il]])


def unpack(vec, no, nv):
    nov = no * nv
    r1 = vec[:nov].reshape(no, nv).copy()
    m = np.zeros((nov, nov))
    il = np.tril_indices(nov)
    m[il] = vec[nov:]
    m = m + m.T - np.diag(np.diag(m))
    r2 = m.reshape(no, nv, no, nv).transpose(0, 2, 1, 3).copy()
    return r1, r2


# ------------------------------------------------- construction 1: Jacobian


def residual(mycc, eris, t1, t2):
    """Omega(t) = (update_amps(t) - t) * D, PySCF's RCCSD residual."""
    no, nv = t1.shape
    e = eris.mo_energy
    eia = e[:no, None] - e[None, no:]
    dijab = eia[:, None, :, None] + eia[None, :, None, :]
    n1, n2 = ccsd.update_amps(mycc, t1, t2, eris)
    return (n1 - t1) * eia, (n2 - t2) * dijab


def jacobian_matvec(mycc, eris, t1, t2, r1, r2, h=1.0):
    def f(s):
        return residual(mycc, eris, t1 + s * r1, t2 + s * r2)

    p1, p2 = f(h)
    m1, m2 = f(-h)
    pp1, pp2 = f(2 * h)
    mm1, mm2 = f(-2 * h)
    j1 = (8 * (p1 - m1) - (pp1 - mm1)) / (12 * h)
    j2 = (8 * (p2 - m2) - (pp2 - mm2)) / (12 * h)
    # Omega_ia = (e_a - e_i) t_ia + ...: eia = e_i - e_a and update_amps
    # returns rest / eia, so the Fock diagonal enters J with a + sign.
    return j1, j2


# ------------------------------------------------ construction 2: H-bar


def blocks(eris, no):
    """Chemist MO blocks over the active space, PySCF eris layout, unpacked."""
    nv = eris.fock.shape[0] - no
    ovvv = np.asarray(eris.get_ovvv())
    from pyscf import ao2mo

    vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nv)
    return dict(
        oooo=np.asarray(eris.oooo),
        ovoo=np.asarray(eris.ovoo),
        ovov=np.asarray(eris.ovov),
        oovv=np.asarray(eris.oovv),
        ovvo=np.asarray(eris.ovvo),
        ovvv=ovvv,
        vvvv=vvvv,
    )


def make_tau(t2, t1, r1, fac=1.0):
    tau = np.einsum("ia,jb->ijab", t1, r1)
    tau = tau + tau.transpose(1, 0, 3, 2)
    return t2 + tau * (fac * 0.5)


def intermediates(t1, t2, g, eo, ev):
    """Spin-adapted EE intermediates (canonical HF: foo, fvv diagonal, fov 0)."""
    ovoo, ovov, oovv, ovvo, ovvv, oooo = (
        g["ovoo"],
        g["ovov"],
        g["oovv"],
        g["ovvo"],
        g["ovvv"],
        g["oooo"],
    )
    tau = make_tau(t2, t1, t1)
    theta = 2 * t2 - t2.transpose(0, 1, 3, 2)
    L = 2 * ovov - ovov.transpose(0, 3, 2, 1)  # 2(me|nf) - (mf|ne)

    fvv = 2 * einsum("mf,mfae->ae", t1, ovvv) - einsum("mf,meaf->ae", t1, ovvv)
    w_vo_oo = einsum("mebf,ijef->mbij", ovvv, tau)
    w_vvo = einsum("jf,mebf->mbej", t1, ovvv)  # woVvO
    w_vVo = -einsum("jf,mfbe->mbej", t1, ovvv)  # woVVo
    w_ooov = einsum("if,mfne->mnie", t1, ovov) + ovoo.transpose(2, 0, 3, 1)
    tmp = einsum("njbf,mfne->mbej", t2, ovov)
    w_vvo -= 0.5 * tmp
    w_vVo += tmp
    x = einsum("menf,jf->menj", ovov, t1)
    w_vvo -= einsum("nb,menj->mbej", t1, x)
    x = einsum("mfne,jf->menj", ovov, t1)
    w_vVo += einsum("nb,menj->mbej", t1, x)
    w_vvo += 0.5 * einsum("njfb,menf->mbej", theta, L)
    fov = einsum("nf,menf->me", t1, L)
    tilab = 0.5 * np.einsum("ia,jb->ijab", t1, t1) + t2
    foo = einsum("mief,menf->ni", tilab, L)
    fvv -= einsum("mnaf,menf->ae", tilab, L)
    w_vvo -= einsum("nb,menj->mbej", t1, ovoo)
    w_vVo += einsum("nb,nemj->mbej", t1, ovoo)
    w_vvo += ovvo.transpose(0, 2, 1, 3)
    w_vVo -= oovv.transpose(0, 2, 3, 1)

    foo += np.diag(eo) + 0.5 * einsum("me,ie->mi", fov, t1)
    fvv += np.diag(ev) - 0.5 * einsum("me,ma->ae", fov, t1)

    w_oooo = einsum("je,nemi->mnij", t1, ovoo)
    w_oooo = w_oooo + w_oooo.transpose(1, 0, 3, 2)
    w_oooo += oooo.transpose(0, 2, 1, 3)
    tmp = einsum("meni,jneb->mbji", ovoo, t2)
    w_vo_oo -= 0.5 * tmp.transpose(0, 1, 3, 2)
    w_vo_oo -= tmp
    x = 2 * ovoo - ovoo.transpose(2, 1, 0, 3)
    w_vo_oo += 0.5 * einsum("nemi,njeb->mbij", x, theta)
    foo += einsum("ne,nemi->mi", t1, x)
    w_oooo += einsum("ijef,menf->mnij", tau, ovov)
    w_vo_oo -= einsum("nb,mnij->mbij", t1, w_oooo)
    t_oovv = einsum("njbf,nemf->ejmb", t2, ovov)
    t_ovvo = einsum("nifb,menf->eimb", theta, L)
    t_ovvo *= -0.5
    t_ovvo += 0.5 * t_oovv
    w_vo_oo -= einsum("ie,ejmb->mbij", t1, t_ovvo)
    w_vo_oo -= einsum("ie,ejmb->mbji", t1, t_oovv)
    w_vo_oo += ovoo.transpose(3, 1, 2, 0)
    t_ovvo -= ovvo.transpose(1, 3, 0, 2)
    t_oovv -= oovv.transpose(3, 1, 0, 2)
    w_vo_oo += einsum("mebj,ie->mbij", ovvo, t1)
    w_vo_oo += einsum("mjbe,ie->mbji", oovv, t1)
    w_vo_oo += einsum("me,ijeb->mbij", fov, t2)

    ebmf = ovvv.transpose(1, 2, 0, 3)  # [e,b,m,f] = (me|bf)
    w = einsum("ebmf,miaf->eiab", ebmf, t2)
    w_vovv = -0.5 * w.transpose(0, 1, 3, 2) - w
    w_vovv += ebmf.transpose(1, 2, 3, 0)
    tmp = -0.5 * ebmf + ebmf.transpose(1, 0, 2, 3)
    w_vovv += einsum("efmb,mifa->eiba", tmp, theta)
    w_vovv += einsum("meni,mnab->eiab", ovoo, tau)
    w_vovv -= einsum("me,miab->eiab", fov, t2)
    w_vovv += einsum("ma,eimb->eiab", t1, t_ovvo)
    w_vovv += einsum("ma,eimb->eiba", t1, t_oovv)
    return dict(
        Foo=foo,
        Fvv=fvv,
        Fov=fov,
        woOoO=w_oooo,
        woVoO=w_vo_oo,
        woVvO=w_vvo,
        woVVo=w_vVo,
        woOoV=w_ooov,
        wvOvV=w_vovv,
    )


def eom_matvec(r1, r2, t1, t2, g, im, mutate=None):
    """sigma = (H-bar - E_CC) R on the singlet space."""
    ovov, ovvv, vvvv = g["ovov"], g["ovvv"], g["vvvv"]
    Foo, Fvv, Fov = im["Foo"], im["Fvv"], im["Fov"]
    hr1 = einsum("ae,ie->ia", Fvv, r1) - einsum("mi,ma->ia", Foo, r1)
    hr1 += 2 * einsum("me,imae->ia", Fov, r2) - einsum("me,imea->ia", Fov, r2)

    tau2 = make_tau(r2, r1, t1, fac=2.0)
    hr2 = einsum("ijef,aebf->ijab", tau2, vvvv)
    hr2 += einsum("mnij,mnab->ijab", im["woOoO"], r2)
    hr2 *= 0.5
    hr2 += einsum("be,ijae->ijab", Fvv, r2) - einsum("mj,imab->ijab", Foo, r2)

    theta = 2 * r2 - r2.transpose(0, 1, 3, 2)
    hr1 += einsum("mfae,mife->ia", ovvv, theta)
    tmp = einsum("meaf,ijef->maij", ovvv, tau2)
    hr2 -= einsum("ma,mbij->ijab", t1, tmp)
    tmp = 2 * einsum("meaf,me->af", ovvv, r1) - einsum("mfae,me->af", ovvv, r1)
    # Doubles <- singles, the (ovvv r1) t2 term: R1 and T2 joined through one
    # integral. The Rust mutation test flips this sign.
    sgn = -1.0 if mutate == "ds_ovvv_t2" else 1.0
    hr2 += sgn * einsum("af,ijfb->ijab", tmp, t2)
    hr2 -= einsum("mbij,ma->ijab", im["woVoO"], r1)
    hr2 += einsum("ejab,ie->ijab", im["wvOvV"], r1)

    tmp = einsum("mbej,imea->jiab", im["woVVo"], r2)
    hr2 += tmp
    hr2 += 0.5 * tmp.transpose(0, 1, 3, 2)

    wvo = 0.5 * im["woVVo"] + im["woVvO"]
    hr1 += 2 * einsum("maei,me->ia", wvo, r1)
    hr2 += einsum("mbej,imae->ijab", wvo, theta)

    hr1 -= einsum("mnie,mnae->ia", im["woOoV"], theta)
    tmp = 2 * einsum("nmie,me->ni", im["woOoV"], r1) - einsum(
        "mnie,me->ni", im["woOoV"], r1
    )
    hr2 -= einsum("ni,njab->ijab", tmp, t2)

    tmp = 2 * einsum("mfne,mf->en", ovov, r1) - einsum("menf,mf->en", ovov, r1)
    tmp = einsum("en,nb->eb", tmp, t1)
    tmp += einsum("menf,mnbf->eb", ovov, theta)
    hr2 -= einsum("eb,ijea->jiab", tmp, t2)

    tmp = einsum("nemf,imef->ni", ovov, theta)
    hr1 -= einsum("na,ni->ia", t1, tmp)
    hr2 -= einsum("mj,miab->ijba", tmp, t2)

    tmp = einsum("menf,ijef->mnij", ovov, tau2)
    tau = 0.5 * make_tau(t2, t1, t1)
    hr2 += einsum("mnij,mnab->ijab", tmp, tau)

    hr2 = hr2 + hr2.transpose(1, 0, 3, 2)
    return hr1, hr2


def cis_matrix(g, eo, ev):
    no, nv = eo.size, ev.size
    a = 2 * g["ovov"] - g["oovv"].transpose(0, 2, 1, 3)  # [i,a,j,b]
    a = a.reshape(no * nv, no * nv)
    a += np.diag((ev[None, :] - eo[:, None]).ravel())
    return a


# --------------------------------------------- construction 3: the solver


def diagonal(eo, ev, no, nv):
    d1 = ev[None, :] - eo[:, None]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    return pack(d1, d2)


def olsen(r, x, theta, diag):
    den = theta - diag
    den = np.where(np.abs(den) < 1e-8, np.copysign(1e-8, den), den)
    p = 1.0 / den
    pr, px = p * r, p * x
    eps = x @ pr / (x @ px)
    return pr - eps * px


def orth_against(v, t):
    kept = []
    for c in t.T:
        c = c / np.linalg.norm(c)
        for _ in range(2):
            if v.shape[1]:
                c = c - v @ (v.T @ c)
            for k in kept:
                c = c - k * (k @ c)
        n = np.linalg.norm(c)
        if n > 1e-10:
            kept.append(c / n)
    return np.array(kept).T if kept else np.zeros((v.shape[0], 0))


def davidson_nonsym(apply, diag, nroots, tol=1e-7, max_iter=100):
    dim = diag.size
    order = np.argsort(diag, kind="stable")
    ng = min(max(2 * nroots, nroots + 8), dim)
    v = np.zeros((dim, ng))
    rng = np.random.default_rng(7)
    for k in range(ng):
        v[order[k], k] = 1.0
    v += 0.1 / np.sqrt(dim) * rng.uniform(-1, 1, v.shape)
    v = orth_against(np.zeros((dim, 0)), v)
    av = np.column_stack([apply(c) for c in v.T])
    for it in range(1, max_iter + 1):
        g = v.T @ av
        w, y = np.linalg.eig(g)
        idx = np.argsort(w.real)
        idx = [k for k in idx if w[k].real > 1e-10][:nroots]
        if any(abs(w[k].imag) > 1e-10 for k in idx):
            raise RuntimeError("complex Ritz value among the targets")
        theta = w[idx].real
        yt = y[:, idx].real
        yt /= np.linalg.norm(v @ yt, axis=0)
        x, ax = v @ yt, av @ yt
        corr, res = [], []
        for k in range(len(idx)):
            r = ax[:, k] - theta[k] * x[:, k]
            res.append(np.linalg.norm(r))
            if res[-1] > tol:
                corr.append(olsen(r, x[:, k], theta[k], diag))
        if not corr:
            return theta, x, res, it
        if v.shape[1] + len(corr) > max(ng + 10 * nroots, 50):
            v = orth_against(np.zeros((dim, 0)), np.column_stack([x, *corr]))
            av = np.column_stack([apply(c) for c in v.T])
            continue
        new = orth_against(v, np.column_stack(corr))
        if new.shape[1] == 0:
            raise RuntimeError("stalled")
        v = np.hstack([v, new])
        av = np.hstack([av, np.column_stack([apply(c) for c in new.T])])
    raise RuntimeError("not converged")


# ------------------------------------------------------------------ driver


def run(atom, basis, nroots, label):
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=MAX_MEMORY_MB)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-10
    mycc.verbose = 0
    mycc.max_memory = MAX_MEMORY_MB
    # Canonical-diagonal Fock, as ferric uses (the SCF leaves ~1e-7 off-diagonal).
    eris = mycc.ao2mo()
    eris.fock = np.diag(eris.mo_energy)
    mycc.kernel(eris=eris)
    no = mycc.nocc
    nv = mycc.nmo - no
    e = eris.mo_energy
    eo, ev = e[:no], e[no:]
    g = blocks(eris, no)
    t1, t2 = mycc.t1, mycc.t2
    im = intermediates(t1, t2, g, eo, ev)
    rng = np.random.default_rng(3)
    print(f"== {label}: no={no} nv={nv} E_CCSD_corr={mycc.e_corr:.10f}")

    # (2) vs PySCF's own matvec, and (2) vs (1), on random singlet vectors.
    eom = eom_rccsd.EOMEESinglet(mycc)
    imds = eom.make_imds(eris)
    worst_pyscf = worst_jac = 0.0
    for _ in range(3):
        r1 = rng.standard_normal((no, nv))
        r2 = rng.standard_normal((no, no, nv, nv))
        r2 = r2 + r2.transpose(1, 0, 3, 2)
        h1, h2 = eom_matvec(r1, r2, t1, t2, g, im)
        p1, p2 = eom.vector_to_amplitudes(
            eom.matvec(eom.amplitudes_to_vector(r1, r2), imds)
        )
        j1, j2 = jacobian_matvec(mycc, eris, t1, t2, r1, r2)
        scale = max(np.abs(h1).max(), np.abs(h2).max())
        worst_pyscf = max(worst_pyscf, np.abs(h1 - p1).max(), np.abs(h2 - p2).max())
        worst_jac = max(worst_jac, np.abs(h1 - j1).max(), np.abs(h2 - j2).max())
    print(
        f"  matvec vs PySCF eom: {worst_pyscf:.2e}   vs CC Jacobian: {worst_jac:.2e}"
        f"   (|sigma| ~ {scale:.1e})"
    )

    # Mutation in the prototype: the flagged D<-S sign must move the Jacobian check.
    h1, h2 = eom_matvec(r1, r2, t1, t2, g, im, mutate="ds_ovvv_t2")
    j1, j2 = jacobian_matvec(mycc, eris, t1, t2, r1, r2)
    print(f"  mutated D<-S sign vs Jacobian: {np.abs(h2 - j2).max():.2e}")

    # T = 0: singles block == CIS.
    z1, z2 = np.zeros_like(t1), np.zeros_like(t2)
    im0 = intermediates(z1, z2, g, eo, ev)
    nov = no * nv
    s = np.zeros((nov, nov))
    for k in range(nov):
        u = np.zeros(nov)
        u[k] = 1
        s[:, k] = eom_matvec(u.reshape(no, nv), z2, z1, z2, g, im0)[0].ravel()
    print(
        f"  T=0 singles block vs CIS matrix: {np.abs(s - cis_matrix(g, eo, ev)).max():.2e}"
    )

    # PySCF roots.
    eom = eom_rccsd.EOMEESinglet(mycc)
    eom.conv_tol = 1e-10
    eom.max_memory = MAX_MEMORY_MB
    e_py, _ = eom.kernel(nroots=nroots, eris=eris)
    e_py = np.atleast_1d(e_py)

    def apply(vec):
        r1, r2 = unpack(vec, no, nv)
        return pack(*eom_matvec(r1, r2, t1, t2, g, im))

    diag = diagonal(eo, ev, no, nv)
    dim = diag.size
    if dim <= 400:
        dense = np.column_stack([apply(u) for u in np.eye(dim)])
        w = np.linalg.eigvals(dense)
        w = np.sort(w[w.real > 1e-10].real)[:nroots]
        print(f"  dense dim={dim}: lowest {w}")
    theta, x, res, it = davidson_nonsym(apply, diag, nroots)
    print(f"  Davidson ({it} it, max res {max(res):.1e}): {theta}")
    print(f"  PySCF eomee_ccsd_singlet:             {e_py}")
    print(f"  max |Davidson - PySCF| = {np.abs(theta - e_py).max():.2e}")
    if dim <= 400:
        print(f"  max |dense - Davidson| = {np.abs(theta - w).max():.2e}")


if __name__ == "__main__":
    run("H 0 0 0; H 0 0 0.7414", "6-31g", 3, "H2/6-31G")
    run(
        "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161",
        "6-31g",
        5,
        "H2O/6-31G",
    )
