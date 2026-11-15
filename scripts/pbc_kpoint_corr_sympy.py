#!/usr/bin/env python3
"""Checks of ferric-pbc's k-point Hartree-Fock functional, its exchange-
divergence (Madelung) term, and the Gamma and k-point MP2 and dRPA energies.
Theory: crates/ferric-pbc/theory.md, sections "8. k-point Hartree-Fock and
Kohn-Sham" and "9. Periodic MP2 and dRPA".

The independent route is the SUPERCELL: an N-point Gamma-centred mesh on a
cell is term by term the Gamma calculation of the N-fold supercell. The model
is one-dimensional (cell length a, s Gaussians, kernel v(K) = 4 pi/K^2 with
K = 0 dropped); the identities checked are the k-point bookkeeping (Bloch
phases, momentum transfer q = k' - k, the 1/N_k normalisations, the Madelung
constant per k), which do not depend on the dimension or on the kernel. The
k side transcribes the code's formulas (primitive-cell pair transforms at
K = G + q); the supercell side builds the supercell AO basis, its own pair
transforms on the supercell reciprocal lattice, real supercell orbitals from a
real eigensolve, and the molecular closed-shell formulas.

Claims (each prints "(claim X)" and exits nonzero on a mismatch):

  mesh          sympy: sum_{L=0}^{N-1} e^{2 pi i m L/N} = N delta_{m0}; the
                Gamma-centred mesh {G + q} is the supercell reciprocal
                lattice; Monkhorst-Pack numerators 2m + 1 - N (odd N: the
                Gamma-centred set modulo G; even N: shifted by half a step).
  krhf-supercell   J(k), K(k) (with v_M S(k) D(k) S(k), no 1/N_k) and
                E/cell = (1/N_k) sum_k 1/2 Re tr[(J - K/2)(k) D(k)] equal the
                supercell's Gamma J, K and two-electron energy per cell.
  kuhf-supercell   the same per spin: K_s(k) carries v_M S D_s S with
                coefficient 1 (no per-spin 1/2).
  madelung-shift   at fixed C, exxdiv = ewald moves every occupied level of
                every k by -v_M (virtuals by 0) and E/cell by -v_M N/2.
  madelung-force   with D_s S D_s = D_s, the W term and the explicit S term
                of the Madelung energy cancel: the overlap-coupled matrix
                M = -W - v_M sum_s D_s S D_s is the same for ewald and none.
  ov-neutral    C_o^H S(k) C_v = 0, so every occupied-virtual pair density has
                no K = 0 component: (ia|jb) needs no G = 0 correction.
  kmp2-supercell   E_MP2/cell = (1/N_k^3) sum_{ki,kj,ka} sum conj(V)[2V - V']
                / (e_i + e_j - e_a - e_b), kb = ki + kj - ka, equals the
                supercell's real closed-shell MP2 per cell.
  kdrpa-supercell  E_dRPA/cell = (1/N_k) sum_q (1/2 pi) int du [ln det(1 + Pi)
                - tr Pi], Pi(q) = (4/N_k) sum B B^H e/(u^2 + e^2), equals the
                supercell's plasmon dRPA per cell.
  realify       for complex B = X + iY, B_R = [[X, -Y], [Y, X]] with every
                excitation energy duplicated gives a real Pi_R whose spectrum is
                that of Pi with every eigenvalue twice, so E_q = 1/2 E(Pi_R).
  drpa-second-order   sympy: int_0^oo e1 e2 /((u^2+e1^2)(u^2+e2^2)) du =
                pi/(2(e1+e2)); hence the O(Pi^2) term of the k-point dRPA is
                the direct (2|V|^2) part of the k-point MP2, checked
                numerically.

Rust: crates/ferric-pbc/src/kdense_aft.rs (J/K kernels), kscf.rs
(solve_krhf energy and Fock), kuscf.rs (per-spin Madelung), mp2.rs
occupied_shift, grad.rs (M matrix), kcorr.rs (kpoint_mp2, kpoint_drpa,
drpa_q_realified).

Mutations (each turns the script red):
  - divide the Madelung term of K(k) by N_k (marker
    "[FAIL] (claim krhf-supercell)");
  - exchange kernel at the bare G instead of G + q, as the code's
    KMutation::KernelAtG (marker "[FAIL] (claim krhf-supercell)");
  - kb = ki - kj + ka, as the code's KCorrMutation::KbIndex (marker
    "[FAIL] (claim kmp2-supercell)");
  - Pi(q) prefactor 4/N_k -> 4 (marker "[FAIL] (claim kdrpa-supercell)").

Run: python3 scripts/pbc_kpoint_corr_sympy.py   (a few seconds)
"""

import cmath
import math
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

FAILURES = []


def check(claim, ok, detail=""):
    tag = "ok" if ok else "FAIL"
    print(f"[{tag}] (claim {claim}){(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(claim)


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------


def claim_mesh():
    ok = True
    for n in (2, 3, 4):
        for m in range(n):
            s = sp.nsimplify(
                sp.simplify(sum(sp.exp(2 * sp.pi * sp.I * m * L / n) for L in range(n)))
            )
            ok &= sp.simplify(s - (n if m == 0 else 0)) == 0
    # {G + q}, G = 2 pi g/a, q = 2 pi m/(N a): the supercell lattice 2 pi n/(N a).
    for n in (2, 3, 5):
        sc = {g * n + m for g in range(-4, 5) for m in range(n)}
        ok &= sc == set(range(-4 * n, 5 * n))
    # Monkhorst-Pack: f = (m + 1/2)/N - 1/2 = (2m + 1 - N)/(2N).
    for n in (1, 2, 3, 4, 5):
        mp = [sp.Rational(2 * m + 1 - n, 2 * n) for m in range(n)]
        gc = [sp.Rational(m, n) for m in range(n)]
        mod1 = lambda x: x - sp.floor(x)  # noqa: E731
        if n % 2:
            ok &= sorted(mod1(x) for x in mp) == sorted(gc)
        else:
            ok &= sorted(mod1(x - sp.Rational(1, 2 * n)) for x in mp) == sorted(gc)
    check("mesh", ok, "Bloch sums, supercell reciprocal lattice, MP numerators")


# ---------------------------------------------------------------------------
# 1-D model
# ---------------------------------------------------------------------------

A = 2.5  # cell length
XS = np.array([0.2, 0.9, 1.7])  # AO centres in the cell
ALS = np.array([0.9, 1.3, 0.7])  # AO exponents
NAO = 3
NK = 3  # Gamma-centred mesh 0, 2pi/3a, 4pi/3a (== -2pi/3a)
KCUT = 14.0
LMAX = 14
VM = 0.37  # a fixed "Madelung constant"; the identities hold for any value


def pft(al, xa, be, xb, kv):
    p = al + be
    pc = (al * xa + be * xb) / p
    return (
        math.sqrt(math.pi / p)
        * math.exp(-al * be / p * (xa - xb) ** 2)
        * np.exp(-kv * kv / (4 * p))
        * np.exp(-1j * kv * pc)
    )


def kpt(m):
    return 2 * math.pi * m / (NK * A)


def kgrid(q, gcut=KCUT):
    """K = G + q with |K| <= gcut, K != 0 (primitive reciprocal lattice G)."""
    g = np.arange(-40, 41) * 2 * math.pi / A + q
    return g[(np.abs(g) <= gcut) & (np.abs(g) > 1e-12)]


def bloch_pair(kp, kv):
    """P^{k'}_{mn}(K) = sum_L e^{i k' L a} int phi_m phi_n(. - L a) e^{-iKx}."""
    out = np.zeros((NAO, NAO, kv.size), complex)
    for L in range(-LMAX, LMAX + 1):
        ph = cmath.exp(1j * kp * L * A)
        for m in range(NAO):
            for n in range(NAO):
                out[m, n] += ph * pft(ALS[m], XS[m], ALS[n], XS[n] + L * A, kv)
    return out


def overlap_k(k):
    return bloch_pair(k, np.array([0.0]))[:, :, 0]


def kmesh_jk(dms, madelung_scale=1.0, kernel_at_g=False):
    """The kdense_aft.rs kernels contracted with D(k')."""
    ks = [kpt(m) for m in range(NK)]
    s = [overlap_k(k) for k in ks]
    jm = [np.zeros((NAO, NAO), complex) for _ in range(NK)]
    km = [np.zeros((NAO, NAO), complex) for _ in range(NK)]
    g0 = kgrid(0.0)
    pg = [bloch_pair(k, g0) for k in ks]
    vg = 4 * math.pi / g0**2
    for k in range(NK):
        for kp in range(NK):
            # J: (1/Omega) sum_G v P^k_mn(G) sum_ls conj P^{k'}_ls(G) D_ls(k')
            rho = np.einsum("lsg,ls->g", np.conj(pg[kp]), dms[kp])
            jm[k] += np.einsum("mng,g->mn", pg[k], vg * rho) / A / NK
            # K: (1/Omega) sum_{K in G+q} v P^{k'}_ml(K) conj P^{k'}_ns(K) D_ls(k')
            q = ks[kp] - ks[k]
            kv = kgrid(q)
            pk = bloch_pair(ks[kp], kv)
            if kernel_at_g:
                gb = kv - q
                safe = np.where(np.abs(gb) > 1e-12, gb, 1.0)
                v = np.where(np.abs(gb) > 1e-12, 4 * math.pi / safe**2, 0.0)
            else:
                v = 4 * math.pi / kv**2
            km[k] += np.einsum("mlg,nsg,ls,g->mn", pk, np.conj(pk), dms[kp], v) / A / NK
        mad = VM * (s[k] @ dms[k] @ s[k])
        km[k] += mad * madelung_scale
    return jm, km, s


# ---------------------------------------------------------------------------
# supercell
# ---------------------------------------------------------------------------

NSC = NAO * NK
ASC = NK * A


def sc_basis():
    return [(ALS[m], XS[m] + L * A) for L in range(NK) for m in range(NAO)]


def sc_pair_ft(gv):
    basis = sc_basis()
    out = np.zeros((NSC, NSC, gv.size), complex)
    for M in range(-LMAX, LMAX + 1):
        for i, (ai, xi) in enumerate(basis):
            for j, (aj, xj) in enumerate(basis):
                out[i, j] += pft(ai, xi, aj, xj + M * ASC, gv)
    return out


def sc_eri():
    g = np.arange(-120, 121) * 2 * math.pi / ASC
    g = g[(np.abs(g) <= KCUT) & (np.abs(g) > 1e-12)]
    p = sc_pair_ft(g)
    v = 4 * math.pi / g**2
    eri = np.einsum("abg,cdg,g->abcd", np.conj(p), p, v).real / ASC
    s = sc_pair_ft(np.array([0.0]))[:, :, 0].real
    return eri, s


def sc_index(m, L):
    return L * NAO + m


def to_supercell(dms):
    """D_sc[(m L1),(n L2)] = (1/N) sum_k e^{ik(L1 - L2)a} D(k)_mn."""
    d = np.zeros((NSC, NSC), complex)
    for k in range(NK):
        kk = kpt(k)
        for L1 in range(NK):
            for L2 in range(NK):
                ph = cmath.exp(1j * kk * (L1 - L2) * A)
                for m in range(NAO):
                    for n in range(NAO):
                        d[sc_index(m, L1), sc_index(n, L2)] += ph * dms[k][m, n] / NK
    return d


def from_supercell(x, k):
    """X(k)_mn = sum_L e^{ikLa} X_sc[(m 0),(n L)]."""
    kk = kpt(k)
    out = np.zeros((NAO, NAO), complex)
    for L in range(NK):
        for m in range(NAO):
            for n in range(NAO):
                out[m, n] += (
                    cmath.exp(1j * kk * L * A) * x[sc_index(m, 0), sc_index(n, L)]
                )
    return out


def random_dms(s, nocc, seed, scale=2.0):
    """D(k) = scale C C^H, C S(k)-orthonormal; D(-k) = conj D(k)."""
    rng = np.random.default_rng(seed)
    dms = [None] * NK
    for k in range(NK):
        mk = (-k) % NK
        if dms[mk] is not None:
            dms[k] = np.conj(dms[mk])
            continue
        c = rng.standard_normal((NAO, nocc)) + (
            1j * rng.standard_normal((NAO, nocc)) if k else 0
        )
        ov = c.conj().T @ s[k] @ c
        w, u = np.linalg.eigh(ov)
        c = c @ u @ np.diag(w**-0.5) @ u.conj().T
        dms[k] = scale * c @ c.conj().T
    return dms


def tr_re(x, d):
    return float(np.real(np.einsum("mn,nm->", x, d)))


def claim_krhf(eri, ssc):
    ks = [kpt(m) for m in range(NK)]
    s = [overlap_k(k) for k in ks]
    dms = random_dms(s, 1, seed=3)
    jm, km, _ = kmesh_jk(dms)
    e_k = sum(0.5 * tr_re(jm[k] - 0.5 * km[k], dms[k]) for k in range(NK)) / NK
    dsc = to_supercell(dms)
    ok_real = float(np.max(np.abs(dsc.imag))) < 1e-13
    dsc = dsc.real
    jsc = np.einsum("abcd,cd->ab", eri, dsc)
    ksc = np.einsum("acbd,cd->ab", eri, dsc) + VM * ssc @ dsc @ ssc
    e_sc = 0.5 * float(np.sum((jsc - 0.5 * ksc) * dsc)) / NK
    dj = max(float(np.max(np.abs(from_supercell(jsc, k) - jm[k]))) for k in range(NK))
    dk = max(float(np.max(np.abs(from_supercell(ksc, k) - km[k]))) for k in range(NK))
    ds = max(float(np.max(np.abs(from_supercell(ssc, k) - s[k]))) for k in range(NK))
    check(
        "krhf-supercell",
        ok_real
        and abs(e_k - e_sc) < 1e-11
        and dj < 1e-11
        and dk < 1e-11
        and ds < 1e-12,
        f"E2/cell k {e_k:.12f} supercell {e_sc:.12f} (d {e_k - e_sc:.1e}); "
        f"|dJ| {dj:.1e} |dK| {dk:.1e} |dS| {ds:.1e}",
    )
    return s


def claim_kuhf(eri, ssc, s):
    da = random_dms(s, 2, seed=11, scale=1.0)
    db = random_dms(s, 1, seed=12, scale=1.0)
    ja, ka, _ = kmesh_jk(da)
    jb, kb, _ = kmesh_jk(db)
    j = [ja[k] + jb[k] for k in range(NK)]
    e_k = (
        sum(
            0.5 * tr_re(j[k] - ka[k], da[k]) + 0.5 * tr_re(j[k] - kb[k], db[k])
            for k in range(NK)
        )
        / NK
    )
    dsa, dsb = to_supercell(da).real, to_supercell(db).real
    jsc = np.einsum("abcd,cd->ab", eri, dsa + dsb)
    ksa = np.einsum("acbd,cd->ab", eri, dsa) + VM * ssc @ dsa @ ssc
    ksb = np.einsum("acbd,cd->ab", eri, dsb) + VM * ssc @ dsb @ ssc
    e_sc = (
        0.5 * (float(np.sum((jsc - ksa) * dsa)) + float(np.sum((jsc - ksb) * dsb))) / NK
    )
    check(
        "kuhf-supercell",
        abs(e_k - e_sc) < 1e-11,
        f"E2/cell k {e_k:.12f} sc {e_sc:.12f}",
    )


def claim_madelung_shift(s):
    # At each k: C S(k)-orthonormal, D = 2 C_o C_o^H; dF = -1/2 v_M S D S.
    rng = np.random.default_rng(5)
    ok, de = True, 0.0
    nocc = 1
    for k in range(NK):
        c = rng.standard_normal((NAO, NAO)) + 1j * rng.standard_normal((NAO, NAO))
        w, u = np.linalg.eigh(c.conj().T @ s[k] @ c)
        c = c @ u @ np.diag(w**-0.5) @ u.conj().T
        d = 2 * c[:, :nocc] @ c[:, :nocc].conj().T
        df = -0.5 * VM * s[k] @ d @ s[k]
        mo = c.conj().T @ df @ c
        want = np.diag([-VM] * nocc + [0.0] * (NAO - nocc))
        ok &= float(np.max(np.abs(mo - want))) < 1e-12
        # energy: -1/4 v_M tr(D S D S) per k, averaged over k
        de += -0.25 * VM * tr_re(d @ s[k] @ d, s[k]) / NK
    ok &= abs(de - (-VM * 2 * nocc / 2)) < 1e-12
    check(
        "madelung-shift",
        ok,
        f"MO-basis shift diag(-v_M occ, 0 virt); dE/cell {de:.12f}",
    )


def claim_madelung_force():
    # Gamma, real, open shell: D_s = C_s C_s^T, F_s(ewald) = F_s(none) - v_M S D_s S.
    rng = np.random.default_rng(9)
    n = 5
    x = rng.standard_normal((n, n))
    sm = x @ x.T + n * np.eye(n)
    f0 = rng.standard_normal((n, n))
    f0 = f0 + f0.T
    w, u = np.linalg.eigh(sm)
    xh = u @ np.diag(w**-0.5) @ u.T
    c = xh @ np.linalg.qr(rng.standard_normal((n, n)))[0]
    ds = [c[:, :3] @ c[:, :3].T, c[:, :1] @ c[:, :1].T]

    def mmat(vm, per_spin=True):
        wsum = sum(d @ (f0 - vm * sm @ d @ sm) @ d for d in ds)
        if per_spin:
            mad = vm * sum(d @ sm @ d for d in ds)
        else:
            dt = ds[0] + ds[1]
            mad = 0.5 * vm * dt @ sm @ dt
        return -wsum - mad

    dm = float(np.max(np.abs(mmat(VM) - mmat(0.0))))
    dm_total = float(
        np.max(np.abs(mmat(VM, per_spin=False) - mmat(0.0, per_spin=False)))
    )
    check(
        "madelung-force",
        dm < 1e-12 and dm_total > 1e-3,
        f"|M(ewald) - M(none)| per spin {dm:.1e}; total-density form {dm_total:.1e} (must differ)",
    )


# ---------------------------------------------------------------------------
# canonical orbitals of a lattice-periodic Fock
# ---------------------------------------------------------------------------


def hopping():
    """f(L) for L = -1, 0, 1, f(-L) = f(L)^T; diagonal levels well separated."""
    lev = np.array([-1.6, 0.4, 1.3])
    rng = np.random.default_rng(21)
    f = {}
    f[0] = np.diag(lev) + 0.05 * (lambda r: r + r.T)(rng.standard_normal((NAO, NAO)))
    f[1] = 0.08 * rng.standard_normal((NAO, NAO))
    f[-1] = f[1].T
    return f


def fock_k(f, k):
    return sum(cmath.exp(1j * kpt(k) * L * A) * f[L] for L in f)


def fock_sc(f):
    out = np.zeros((NSC, NSC))
    for L1 in range(NK):
        for L2 in range(NK):
            for L, blk in f.items():
                if (L2 - L1 - L) % NK == 0:
                    for m in range(NAO):
                        for n in range(NAO):
                            out[sc_index(m, L1), sc_index(n, L2)] += blk[m, n]
    return out


def bloch_pair_kk(k1, k2, kv):
    """Pair FT of chi*_{m k1} chi_{n k2} at K in G + (k2 - k1): P^{k2}(K)."""
    return bloch_pair(kpt(k2), kv)


def claim_corr(eri, ssc, s):
    f = hopping()
    nocc = 1
    eps, cs = [], []
    for k in range(NK):
        fk = fock_k(f, k)
        fk = 0.5 * (fk + fk.conj().T)
        sk = 0.5 * (s[k] + s[k].conj().T)
        e, c = eigh(fk, sk)
        eps.append(e)
        cs.append(c)
    ok_gap = max(e[nocc - 1] for e in eps) < min(e[nocc] for e in eps)
    # ov-neutral: C_o^H S(k) C_v = 0 (K = 0 component of every ov pair at q = 0)
    neutral = max(
        float(np.max(np.abs(cs[k][:, :nocc].conj().T @ s[k] @ cs[k][:, nocc:])))
        for k in range(NK)
    )
    check(
        "ov-neutral", neutral < 1e-12 and ok_gap, f"max |C_o^H S C_v| = {neutral:.1e}"
    )

    # MO pair transforms Pov_ia(k1 -> k2)(K) and the class's K list.
    def mo_pair(k1, k2, oi, ov, kv):
        p = bloch_pair_kk(k1, k2, kv)
        return np.einsum("mi,mnK,na->iaK", np.conj(cs[k1][:, oi]), p, cs[k2][:, ov])

    occ, vir = slice(0, nocc), slice(nocc, NAO)
    # ---- k-point MP2 (kcorr.rs kpoint_mp2) ----
    e_mp2, e_dir = 0.0, 0.0
    for ki in range(NK):
        for kj in range(NK):
            for ka in range(NK):
                kb = (ki + kj - ka) % NK
                q = kpt(ka) - kpt(ki)
                kv = kgrid(q)
                v = 4 * math.pi / kv**2
                pia = mo_pair(ki, ka, occ, vir, kv)
                pbj = mo_pair(kb, kj, vir, occ, kv)
                vv = np.einsum("iaK,bjK,K->iajb", pia, np.conj(pbj), v) / A
                # exchange partner V[ki,kj,kb]_{i b j a}
                q2 = kpt(kb) - kpt(ki)
                kv2 = kgrid(q2)
                v2 = 4 * math.pi / kv2**2
                pib = mo_pair(ki, kb, occ, vir, kv2)
                paj = mo_pair(ka, kj, vir, occ, kv2)
                vx = np.einsum("ibK,ajK,K->ibja", pib, np.conj(paj), v2) / A
                ei, ej = eps[ki][occ], eps[kj][occ]
                ea, eb = eps[ka][vir], eps[kb][vir]
                den = (
                    ei[:, None, None, None]
                    + ej[None, None, :, None]
                    - ea[None, :, None, None]
                )
                den = den - eb[None, None, None, :]
                vxt = np.transpose(vx, (0, 3, 2, 1))  # (i, a, j, b)
                e_mp2 += float(np.real(np.sum(np.conj(vv) * (2 * vv - vxt) / den)))
                e_dir += float(np.real(np.sum(2 * np.abs(vv) ** 2 / den)))
    e_mp2 /= NK**3
    e_dir /= NK**3
    # ---- supercell MP2: real orbitals of the real supercell Fock ----
    esc, csc = eigh(fock_sc(f), ssc)
    nocc_sc = nocc * NK
    co, cv = csc[:, :nocc_sc], csc[:, nocc_sc:]
    ovov = np.einsum("ai,bj,ck,dl,abcd->ijkl", co, cv, co, cv, eri)  # (ia|jb)
    eo, ev = esc[:nocc_sc], esc[nocc_sc:]
    den = eo[:, None, None, None] - ev[None, :, None, None] + eo[None, None, :, None]
    den = den - ev[None, None, None, :]
    e_sc = (
        float(np.sum(ovov * (2 * ovov - np.transpose(ovov, (0, 3, 2, 1))) / den)) / NK
    )
    check(
        "kmp2-supercell",
        abs(e_mp2 - e_sc) < 1e-12,
        f"E_MP2/cell k {e_mp2:.14f} supercell {e_sc:.14f} (d {e_mp2 - e_sc:.1e})",
    )
    # ---- supercell plasmon dRPA ----
    no, nv = nocc_sc, NSC - nocc_sc
    kmat = ovov.reshape(no * nv, no * nv)
    dlt = (ev[None, :] - eo[:, None]).reshape(-1)
    sq = np.sqrt(dlt)
    mm = np.diag(dlt**2) + 4 * sq[:, None] * kmat * sq[None, :]
    e_pl = 0.5 * (
        float(np.sum(np.sqrt(np.linalg.eigvalsh(mm)))) - dlt.sum() - 2 * np.trace(kmat)
    )
    e_pl /= NK
    # ---- k-point dRPA by frequency quadrature (kcorr.rs kpoint_drpa) ----
    xg, wg = np.polynomial.legendre.leggauss(160)
    u0 = 0.5
    us = u0 * (1 + xg) / (1 - xg)
    wu = wg * 2 * u0 / (1 - xg) ** 2
    e_rpa, e_2nd = 0.0, 0.0
    bq_all = []
    for iq in range(NK):
        q = kpt(iq)
        kv = kgrid(q)
        cols, es = [], []
        for kp in range(NK):
            k0 = (kp - iq) % NK
            p = mo_pair(k0, kp, occ, vir, kv)  # (i, a, K)
            for i in range(nocc):
                for a in range(NAO - nocc):
                    cols.append(np.sqrt(4 * math.pi / kv**2 / A) * p[i, a])
                    es.append(eps[kp][nocc + a] - eps[k0][i])
        bq = np.array(cols).T  # (K, c)
        es = np.array(es)
        bq_all.append((bq, es))
        for u, wq in zip(us, wu):
            pi_q = (4.0 / NK) * (bq * (es / (u * u + es * es))) @ bq.conj().T
            sign, logdet = np.linalg.slogdet(np.eye(pi_q.shape[0]) + pi_q)
            tr = float(np.real(np.trace(pi_q)))
            e_rpa += wq * (float(np.real(logdet)) - tr) / (2 * math.pi)
            e_2nd += wq * (-0.5 * float(np.real(np.trace(pi_q @ pi_q)))) / (2 * math.pi)
    e_rpa /= NK
    e_2nd /= NK
    check(
        "kdrpa-supercell",
        abs(e_rpa - e_pl) < 1e-10,
        f"E_dRPA/cell k {e_rpa:.14f} supercell plasmon {e_pl:.14f} (d {e_rpa - e_pl:.1e})",
    )
    # ---- second order: sympy integral, then the numbers ----
    u, e1, e2 = sp.symbols("u e1 e2", positive=True)
    integ = sp.integrate(e1 * e2 / ((u**2 + e1**2) * (u**2 + e2**2)), (u, 0, sp.oo))
    ok_int = sp.simplify(integ - sp.pi / (2 * (e1 + e2))) == 0
    check(
        "drpa-second-order",
        ok_int and abs(e_2nd - e_dir) < 1e-10,
        f"integral {ok_int}; O(Pi^2) {e_2nd:.14f} vs direct MP2 {e_dir:.14f}",
    )
    return bq_all


def claim_realify(bq_all):
    ok = True
    worst = 0.0
    for bq, es in bq_all:
        x, y = bq.real, bq.imag
        br = np.block([[x, -y], [y, x]])
        er = np.concatenate([es, es])
        for u in (0.1, 0.7, 3.0):
            f = es / (u * u + es * es)
            fr = er / (u * u + er * er)
            pc = (bq * f) @ bq.conj().T
            pr = (br * fr) @ br.T
            lc = np.sort(np.linalg.eigvalsh(pc))
            lr = np.sort(np.linalg.eigvalsh(pr))
            want = np.sort(np.concatenate([lc, lc]))
            worst = max(worst, float(np.max(np.abs(lr - want))))
            sc = float(np.sum(np.log1p(lc) - lc))
            sr = float(np.sum(np.log1p(lr) - lr))
            ok &= abs(0.5 * sr - sc) < 1e-12
    check(
        "realify",
        ok and worst < 1e-12,
        f"spectrum of Pi_R = spectrum of Pi, each twice ({worst:.1e})",
    )


def main():
    claim_mesh()
    eri, ssc = sc_eri()
    s = claim_krhf(eri, ssc)
    claim_kuhf(eri, ssc, s)
    claim_madelung_shift(s)
    claim_madelung_force()
    bq_all = claim_corr(eri, ssc, s)
    claim_realify(bq_all)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
