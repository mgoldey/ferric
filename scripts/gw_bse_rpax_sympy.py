#!/usr/bin/env python3
"""Proofs: ferric-gw's BSE-TDA matrix and statically screened TDHF/RPAx polarizability.

Derivation: crates/ferric-gw/theory.md, sections 11 (BSE-TDA) and 12
(TDHF/RPAx polarizability). The code transcribed here (index ia = i * nvir + a,
chemist integrals, real orbitals, auxiliary index first, `M[xi,p,q] =
sum_P U[P,xi] B[P,p,q]` from the eigenvectors U of the static dielectric):

  * bse.rs run_bse_tda: bare(p,q,r,s) = sum_P B[P,p,q] B[P,r,s],
    screened(p,q,r,s) = bare + sum_xi w_xi M[xi,p,q] M[xi,r,s], w_xi = 1/lambda_xi - 1,
        A[ia,jb] = (e_a - e_i)^QP delta + 2 bare(i,a,j,b) - screened(a,b,i,j)
  * bse.rs tda_oscillator_strengths: <0|r|n> = sqrt2 sum_ia X_ia r_ia,
    f_n = (2/3) Omega_n |<0|r|n>|^2
  * bse.rs rpax_static_polarizability_with_screening:
        (A+B)[ia,jb] = delta Delta + 4 bare(i,a,j,b) - screened(a,b,i,j) - screened(i,b,a,j)
        (A-B)[ia,jb] = delta Delta + screened(i,b,a,j) - screened(a,b,i,j)
        alpha_xy = 4 mu_x^T [(A-B)(A+B)] ^-1 (A-B) mu_y        (omega = 0)
    and run_bse_c6[_ks], the same at imaginary frequency u:
        alpha_xy(iu) = 4 mu_x^T [(A-B)(A+B) + u^2]^-1 (A-B) mu_y

Checked:
  S1  (symbolic) the spin-orbital blocks Aaa, Aab, Baa, Bab with a Hartree
      term h = (ia|jb) between every spin pair and the screened exchange wx =
      W(ab|ij), wy = W(ib|ja) between equal spins only, rotated by
      U = [[1,1],[1,-1]]/sqrt2, give the singlet A = d + 2h - wx (TDA),
      A+B = d + 4h - wx - wy, A-B = d - wx + wy and block-diagonal triplets.
  S2  (symbolic) the singlet spin-orbital vector (x, x)/sqrt2 has unit norm per
      unit |x|^2 and dipole sqrt2 x r.
  S3  (symbolic) the 2x2 response equation [[A-w, B],[B, A+w]] (X;Y) = (m; m)
      at w = iu gives (X+Y) = 2m (A-B)/((A-B)(A+B) + u^2), so
      alpha = 4 m^2 (A-B)/((A-B)(A+B) + u^2) per singlet unit (the code's form).
  B   Numerical, random closed-shell model (2 occ, 3 vir, 7 aux functions):
      B1  the code's singlet BSE-TDA matrix equals the singlet block of the
          spin-orbital matrix built from the textbook definition with W
          obtained by an independent matrix inverse W = B^T eps^-1 B (no PDEP
          eigendecomposition); the triplet block is d - W(ab|ij) and the
          singlet-triplet blocks vanish; spectrum(spin-orbital) = singlet U
          triplet.
      B2  CIS limit (W -> v, HF diagonal energies) against a brute-force
          Jordan-Wigner Fock-space CIS (2 occ, 2 vir, 8 spin orbitals):
          <Phi_i^a|H - E0|Phi_j^b> equals the spin-orbital matrix elementwise
          and the Brillouin condition holds.
      B3  the singlet transition moment <Phi0|r|Psi_n>, with
          Psi_n = sum_ia X_ia (a+_a a_i + a+_abar a_ibar)|Phi0>/sqrt2 over
          Jordan-Wigner determinants, equals sqrt2 sum_ia X_ia r_ia.
      B4  the sum rule sum_n f_n = (4/3) sum_x r_x^T A r_x (all roots), which
          fixes the sqrt2 and the Omega factor of the oscillator strength.
  R   Numerical, the same model:
      R1  the code's A+B and A-B equal the singlet blocks of the spin-orbital
          A and B (screened W from the matrix inverse); the triplet blocks are
          d - W(ab|ij) -+ W(ib|ja) and off-diagonal blocks vanish.
      R2  alpha(0) and alpha(iu), u = 0.7, by solving the full 2N x 2N
          spin-orbital response equation [[A-iu, B],[B, A+iu]] (X;Y) = (mu;mu)
          with alpha = mu . (X+Y), equal the code's
          4 mu^T [(A-B)(A+B) + u^2]^-1 (A-B) mu; the reversed order
          mu^T (A-B) [...]^-1 mu differs (negative control).
      R3  W -> v at an HF reference (kernel = TDHF): the code's static alpha
          equals the finite-field derivative d<mu>/dF of a self-consistent
          RHF in the same orthonormal MO basis (h = h0 - F r, F = h + J - K/2,
          D = 2 C_o C_o^T, SCF to 1e-15), i.e. the coupled-perturbed HF
          polarizability.

Mutations (the runner substitutes the code line once; each exits 1):
  apb's Hartree prefactor 4 -> 2                      -> R1 fails
  amb = ... + screened(i,b,a,j) - screened(a,b,i,j) with the two terms swapped -> R1 fails
  the BSE-TDA Hartree prefactor 2 -> 1                -> B1 fails
  the transition-dipole sqrt2 dropped                 -> B3 fails

Run: python3 scripts/gw_bse_rpax_sympy.py   (needs numpy, sympy)
"""

import itertools
import sys

import numpy as np
import sympy as sp

FAILS = []
SQRT2 = np.sqrt(2.0)


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{name}] {detail}")
    if not ok:
        FAILS.append(name)


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


# ============================================================ symbolic
def symbolic():
    d, h, wx, wy, m, x, r = sp.symbols("d h wx wy m x r", real=True)
    u = sp.Symbol("u", positive=True)
    # S1 spin-orbital 2x2 blocks (alpha-alpha pair, beta-beta pair).
    a_so = sp.Matrix([[d + h - wx, h], [h, d + h - wx]])
    b_so = sp.Matrix([[h - wy, h], [h, h - wy]])
    rot = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    ar = (rot.T * a_so * rot).applyfunc(sp.simplify)
    br = (rot.T * b_so * rot).applyfunc(sp.simplify)
    ok = (
        sp.simplify(ar[0, 0] - (d + 2 * h - wx)) == 0
        and sp.simplify(ar[1, 1] - (d - wx)) == 0
        and sp.simplify(ar[0, 1]) == 0
        and sp.simplify(br[0, 0] - (2 * h - wy)) == 0
        and sp.simplify(br[1, 1] + wy) == 0
        and sp.simplify(br[0, 1]) == 0
    )
    check("S1", ok, "singlet A = d+2h-wx, B = 2h-wy; triplet A = d-wx, B = -wy")
    apb = sp.simplify(ar[0, 0] + br[0, 0])
    amb = sp.simplify(ar[0, 0] - br[0, 0])
    check(
        "S1",
        sp.simplify(apb - (d + 4 * h - wx - wy)) == 0
        and sp.simplify(amb - (d - wx + wy)) == 0,
        "singlet A+B = d+4h-wx-wy, A-B = d-wx+wy",
    )
    check(
        "S1",
        sp.simplify(ar[0, 0] - (d - wx)) != 0,
        "negative control: the singlet is not the triplet (the 2h Hartree term)",
    )
    # S2 oscillator normalization.
    vec = sp.Matrix([x, x]) / sp.sqrt(2)
    check(
        "S2",
        sp.simplify((vec.T * vec)[0] - x**2) == 0
        and sp.simplify((vec.T * sp.Matrix([r, r]))[0] - sp.sqrt(2) * x * r) == 0,
        "(x,x)/sqrt2: norm^2 = x^2, dipole = sqrt2 x r",
    )
    # S3 scalar response at omega = i u.
    a, b = sp.symbols("a b", real=True)
    om = sp.I * u
    msys = sp.Matrix([[a - om, b], [b, a + om]])
    z = msys.LUsolve(sp.Matrix([m, m]))
    xpy = sp.simplify(z[0] + z[1])
    ref = 2 * m * (a - b) / ((a - b) * (a + b) + u**2)
    check(
        "S3",
        sp.simplify(xpy - ref) == 0,
        "X+Y = 2m (A-B)/((A-B)(A+B)+u^2) at omega = iu",
    )
    check(
        "S3",
        sp.simplify(xpy - 2 * m * (a + b) / ((a - b) * (a + b) + u**2)) != 0,
        "negative control: (A+B) in the numerator is a different number",
    )


# ============================================================ numerical model
def random_ri(rng, n, naux, scale):
    b = rng.standard_normal((naux, n, n)) * scale
    return 0.5 * (b + b.transpose(0, 2, 1))


def make_model(rng):
    nocc, nvir, naux = 2, 3, 7
    nmo = nocc + nvir
    eps = np.concatenate(
        [np.sort(rng.uniform(-1.2, -0.6, nocc)), np.sort(rng.uniform(0.4, 1.0, nvir))]
    )
    eps_qp = eps + rng.uniform(-0.15, 0.15, nmo)  # a distinct QP spectrum
    b = random_ri(rng, nmo, naux, 0.15)
    r = [0.5 * (m + m.T) for m in rng.standard_normal((3, nmo, nmo))]
    return nocc, nvir, nmo, eps, eps_qp, b, r


def static_dielectric(b, eps, nocc, nvir):
    """eps~(0) = 1 + 4 sum_ia B_ia B_ia / Delta_ia, then its eigendecomposition."""
    naux = b.shape[0]
    bov = b[:, :nocc, nocc:].reshape(naux, -1)
    gaps = (eps[None, nocc:] - eps[:nocc, None]).ravel()
    eps_mat = np.eye(naux) + 4.0 * (bov / gaps) @ bov.T
    lam, u = np.linalg.eigh(eps_mat)
    return eps_mat, lam, u


def code_integrals(b, eps_mat, lam, u, bare_only=False):
    """bse.rs `bare` and `screened` (reduced weights w = 1/lambda - 1)."""
    m_proj = np.einsum("Px,Ppq->xpq", u, b)
    w_red = 1.0 / lam - 1.0

    def bare(p, q, r, s):
        return float(np.dot(b[:, p, q], b[:, r, s]))

    def screened(p, q, r, s):
        acc = bare(p, q, r, s)
        if bare_only:
            return acc
        return acc + float(np.sum(w_red * m_proj[:, p, q] * m_proj[:, r, s]))

    return bare, screened


def code_bse_tda(nocc, nvir, eps_qp, bare, screened):
    n = nocc * nvir
    a_bse = np.zeros((n, n))
    for i, a, j, bb in itertools.product(
        range(nocc), range(nvir), range(nocc), range(nvir)
    ):
        ia, jb = i * nvir + a, j * nvir + bb
        coul = bare(i, nocc + a, j, nocc + bb)
        scr = screened(nocc + a, nocc + bb, i, j)
        a_bse[ia, jb] = 2.0 * coul - scr
        if ia == jb:
            a_bse[ia, jb] += eps_qp[nocc + a] - eps_qp[i]
    return a_bse


def code_rpax(nocc, nvir, eps, bare, screened):
    n = nocc * nvir
    apb, amb = np.zeros((n, n)), np.zeros((n, n))
    for i, a, j, bb in itertools.product(
        range(nocc), range(nvir), range(nocc), range(nvir)
    ):
        ia, jb = i * nvir + a, j * nvir + bb
        a_loc, b_loc = nocc + a, nocc + bb
        coul = bare(i, a_loc, j, b_loc)
        w_abij = screened(a_loc, b_loc, i, j)
        w_ibaj = screened(i, b_loc, a_loc, j)
        apb[ia, jb] = 4.0 * coul - w_abij - w_ibaj
        amb[ia, jb] = w_ibaj - w_abij
        if ia == jb:
            apb[ia, jb] += eps[a_loc] - eps[i]
            amb[ia, jb] += eps[a_loc] - eps[i]
    return apb, amb


def spin_rotation(d):
    eye = np.eye(d)
    return np.block([[eye, eye], [eye, -eye]]) / SQRT2


def so_blocks(nocc, nvir, eps_delta, hartree, exch_a, exch_b):
    """Spin-orbital A and B over pairs ordered (alpha ia..., beta ia...).

    hartree(i,a,j,b) = (ia|jb) between every spin pair; exch_a(i,a,j,b) = W(ab|ij)
    and exch_b(i,a,j,b) = W(ib|ja) only between equal spins.
    """
    n = nocc * nvir
    a_so, b_so = np.zeros((2 * n, 2 * n)), np.zeros((2 * n, 2 * n))
    for s, t in itertools.product((0, 1), repeat=2):
        for i, a, j, bb in itertools.product(
            range(nocc), range(nvir), range(nocc), range(nvir)
        ):
            x, y = s * n + i * nvir + a, t * n + j * nvir + bb
            h = hartree(i, a, j, bb)
            a_so[x, y] = h - (exch_a(i, a, j, bb) if s == t else 0.0)
            b_so[x, y] = h - (exch_b(i, a, j, bb) if s == t else 0.0)
            if x == y:
                a_so[x, y] += eps_delta[i * nvir + a]
    return a_so, b_so


# ---------------------------------------------------------------- Jordan-Wigner CIS
def apply_excitation(det, p, q):
    """a+_p a_q |det>: (sign, new det) or None."""
    if not (det >> q) & 1:
        return None
    sign = -1 if bin(det & ((1 << q) - 1)).count("1") % 2 else 1
    det ^= 1 << q
    if (det >> p) & 1:
        return None
    sign *= -1 if bin(det & ((1 << p) - 1)).count("1") % 2 else 1
    return sign, det | (1 << p)


def apply_h(vec, h1, g2, nso):
    """H = sum h_PQ a+P aQ + 1/2 sum (PQ|RS) a+P a+R aS aQ on a {det: coef} vector."""
    out = {}
    h1 = h1 - 0.5 * np.einsum("pqqs->ps", g2)
    nz1 = [(p, q) for p in range(nso) for q in range(nso) if h1[p, q] != 0.0]
    nz2 = [
        (p, q, r, s)
        for p, q, r, s in itertools.product(range(nso), repeat=4)
        if g2[p, q, r, s] != 0.0
    ]
    for det, cf in vec.items():
        for p, q in nz1:
            res = apply_excitation(det, p, q)
            if res:
                out[res[1]] = out.get(res[1], 0.0) + cf * res[0] * h1[p, q]
        for p, q, r, s in nz2:
            r1 = apply_excitation(det, r, s)
            if not r1:
                continue
            r2 = apply_excitation(r1[1], p, q)
            if r2:
                out[r2[1]] = (
                    out.get(r2[1], 0.0) + cf * r1[0] * r2[0] * 0.5 * g2[p, q, r, s]
                )
    return out


def dot(u, v):
    return sum(cf * v.get(d, 0.0) for d, cf in u.items())


# ============================================================ B: BSE-TDA
def part_b(rng):
    nocc, nvir, nmo, eps, eps_qp, b, rmat = make_model(rng)
    n = nocc * nvir
    eps_mat, lam, u = static_dielectric(b, eps, nocc, nvir)
    bare, screened = code_integrals(b, eps_mat, lam, u)
    a_code = code_bse_tda(nocc, nvir, eps_qp, bare, screened)

    # Independent screened interaction: matrix inverse, no PDEP eigenbasis.
    w_inv = np.linalg.inv(eps_mat)
    w4 = np.einsum("Ppq,PQ,Qrs->pqrs", b, w_inv, b)
    eri = np.einsum("Ppq,Prs->pqrs", b, b)
    err_w = max(
        abs(screened(p, q, r, s) - w4[p, q, r, s])
        for p, q, r, s in itertools.product(range(nmo), repeat=4)
    )
    check("B1", err_w < 1e-12, f"PDEP-mode W = B^T eps^-1 B: max |d| = {err_w:.1e}")

    gaps = np.array(
        [eps_qp[nocc + a] - eps_qp[i] for i in range(nocc) for a in range(nvir)]
    )
    a_so, _ = so_blocks(
        nocc,
        nvir,
        gaps,
        lambda i, a, j, bb: eri[i, nocc + a, j, nocc + bb],
        lambda i, a, j, bb: w4[nocc + a, nocc + bb, i, j],
        lambda i, a, j, bb: 0.0,
    )
    rot = spin_rotation(n)
    ar = rot.T @ a_so @ rot
    sing, trip, off = ar[:n, :n], ar[n:, n:], ar[:n, n:]
    check(
        "B1",
        maxabs(sing - a_code) < 1e-12,
        f"singlet block = code A: max |d| = {maxabs(sing - a_code):.1e}",
    )
    trip_ref = np.diag(gaps) - np.array(
        [
            [w4[nocc + a, nocc + bb, i, j] for j in range(nocc) for bb in range(nvir)]
            for i in range(nocc)
            for a in range(nvir)
        ]
    )
    check(
        "B1",
        maxabs(trip - trip_ref) < 1e-12 and maxabs(off) < 1e-12,
        f"triplet = d - W(ab|ij), off-diagonal blocks {maxabs(off):.1e}",
    )
    spec_so = np.sort(np.linalg.eigvalsh(a_so))
    spec_st = np.sort(
        np.concatenate([np.linalg.eigvalsh(a_code), np.linalg.eigvalsh(trip_ref)])
    )
    check(
        "B1",
        maxabs(spec_so - spec_st) < 1e-12,
        "spin-orbital spectrum = singlet U triplet",
    )
    check(
        "B1",
        maxabs(sing - trip) > 1e-3,
        "negative control: singlet and triplet differ (the 2(ia|jb) term is live)",
    )

    # ---- B2/B3 brute-force CIS (2 occ, 2 vir -> 8 spin orbitals)
    no2, nv2, naux2 = 2, 2, 5
    nmo2 = no2 + nv2
    eps2 = np.array([-1.0, -0.7, 0.5, 0.9])
    b2 = random_ri(rng, nmo2, naux2, 0.2)
    eri2 = np.einsum("Ppq,Prs->pqrs", b2, b2)
    nso = 2 * nmo2
    spin = [k // nmo2 for k in range(nso)]  # spin orbital k = (spin, spatial k % nmo2)
    sp_idx = [k % nmo2 for k in range(nso)]
    g2 = np.zeros((nso,) * 4)
    for p, q, r, s in itertools.product(range(nso), repeat=4):
        if spin[p] == spin[q] and spin[r] == spin[s]:
            g2[p, q, r, s] = eri2[sp_idx[p], sp_idx[q], sp_idx[r], sp_idx[s]]
    occ = [k for k in range(nso) if sp_idx[k] < no2]
    h1 = np.diag([eps2[sp_idx[k]] for k in range(nso)]).astype(float)
    for k in occ:
        h1 -= g2[:, :, k, k] - g2[:, k, k, :]
    phi0 = sum(1 << k for k in occ)
    pairs = [
        (i, a)
        for s_ in (0, 1)
        for i in occ
        if spin[i] == s_
        for a in range(nso)
        if spin[a] == s_ and sp_idx[a] >= no2
    ]
    basis = [{phi0: 1.0}]
    for i, a in pairs:
        sg, dt = apply_excitation(phi0, a, i)
        basis.append({dt: float(sg)})
    hv = [apply_h(bv, h1, g2, nso) for bv in basis]
    hmat = np.array([[dot(bi, hj) for hj in hv] for bi in basis])
    e0 = hmat[0, 0]
    a_brute = hmat[1:, 1:] - e0 * np.eye(len(pairs))
    check(
        "B2",
        maxabs(hmat[0, 1:]) < 1e-12,
        f"Brillouin: |<Phi0|H|Phi_i^a>| = {maxabs(hmat[0, 1:]):.1e}",
    )

    # Code A at W -> v with HF energies on the same model (this is run_cis_tda).
    n2 = no2 * nv2
    bare2 = lambda p, q, r, s: float(np.dot(b2[:, p, q], b2[:, r, s]))  # noqa: E731
    a_cis_code = code_bse_tda(no2, nv2, eps2, bare2, bare2)
    rot2 = spin_rotation(n2)
    # pairs are ordered (alpha ia, beta ia) with i, a ascending: the same ordering as so_blocks.
    ar2 = rot2.T @ a_brute @ rot2
    err_cis = maxabs(ar2[:n2, :n2] - a_cis_code)
    err_off = maxabs(ar2[:n2, n2:])
    check(
        "B2",
        err_cis < 1e-12 and err_off < 1e-12,
        f"brute-force CIS singlet = code A (W -> v): max |d| = {err_cis:.1e}, singlet-triplet {err_off:.1e}",
    )
    check(
        "B2",
        maxabs(ar2[n2:, n2:] - a_cis_code) > 1e-3,
        "negative control: the brute-force triplet block is not the singlet matrix",
    )

    # ---- B3 transition moment
    omega, xmat = np.linalg.eigh(a_cis_code)
    r2 = 0.5 * (lambda m: m + m.T)(rng.standard_normal((nmo2, nmo2)))
    r_ia = np.array([[r2[i, no2 + a] for a in range(nv2)] for i in range(no2)]).ravel()
    worst = 0.0
    for state in range(n2):
        xv = xmat[:, state]
        psi = {}
        for idx, (i, a) in enumerate(pairs):
            ia_loc = (i % nmo2) * nv2 + (a % nmo2 - no2)
            psi_c = xv[ia_loc] / SQRT2
            sgn, dt = apply_excitation(phi0, a, i)
            psi[dt] = psi.get(dt, 0.0) + psi_c * sgn
        # <Phi0| sum_pq r_pq a+_p a_q |Psi>
        amp = 0.0
        for dt, cf in psi.items():
            for p, q in itertools.product(range(nso), repeat=2):
                if spin[p] != spin[q]:
                    continue
                res = apply_excitation(dt, p, q)
                if res and res[1] == phi0:
                    amp += cf * res[0] * r2[sp_idx[p], sp_idx[q]]
        mu_x = SQRT2 * float(xv @ r_ia)  # bse.rs tda_oscillator_strengths
        worst = max(worst, abs(amp - mu_x))
    check(
        "B3",
        worst < 1e-12,
        f"<Phi0|r|Psi_n> = sqrt2 X.r over {n2} CIS roots: max |d| = {worst:.1e}",
    )

    # ---- B4 sum rule on the screened model
    omega, xmat = np.linalg.eigh(a_code)
    dips = [
        np.array([[r_[i, nocc + a] for a in range(nvir)] for i in range(nocc)]).ravel()
        for r_ in rmat
    ]
    f_tot = 0.0
    for state in range(n):
        xv = xmat[:, state]
        mu2 = sum((SQRT2 * float(xv @ dv)) ** 2 for dv in dips)
        f_tot += (2.0 / 3.0) * omega[state] * mu2
    rule = (4.0 / 3.0) * sum(float(dv @ a_code @ dv) for dv in dips)
    check(
        "B4",
        abs(f_tot - rule) < 1e-12 * max(1.0, abs(rule)),
        f"sum f = {f_tot:.10f}, (4/3) sum r^T A r = {rule:.10f}",
    )
    check(
        "B4",
        abs(f_tot / 2.0 - rule) > 1e-6,
        "negative control: dropping the sqrt2 halves the sum",
    )


# ============================================================ R: RPAx
def finite_field_alpha(eri, eps, nocc, r_mo, delta=1e-4):
    nmo = len(eps)

    def fock(h, d):
        j = np.einsum("pqrs,rs->pq", eri, d)
        k = np.einsum("prqs,rs->pq", eri, d)
        return h + j - 0.5 * k

    d0 = np.diag([2.0] * nocc + [0.0] * (nmo - nocc))
    h0 = np.diag(eps) - (fock(np.zeros((nmo, nmo)), d0))

    def dipole(field):
        h = h0 - field * r_mo
        d = d0.copy()
        for _ in range(500):
            _, c = np.linalg.eigh(fock(h, d))
            co = c[:, :nocc]
            d_new = 2.0 * co @ co.T
            if maxabs(d_new - d) < 1e-15:
                d = d_new
                break
            d = d_new
        return float(np.sum(r_mo * d))

    return (dipole(delta) - dipole(-delta)) / (2 * delta)


def part_r(rng):
    nocc, nvir, nmo, eps, _, b, rmat = make_model(rng)
    n = nocc * nvir
    eps_mat, lam, u = static_dielectric(b, eps, nocc, nvir)
    bare, screened = code_integrals(b, eps_mat, lam, u)
    apb, amb = code_rpax(nocc, nvir, eps, bare, screened)
    w_inv = np.linalg.inv(eps_mat)
    w4 = np.einsum("Ppq,PQ,Qrs->pqrs", b, w_inv, b)
    eri = np.einsum("Ppq,Prs->pqrs", b, b)
    gaps = np.array([eps[nocc + a] - eps[i] for i in range(nocc) for a in range(nvir)])

    a_so, b_so = so_blocks(
        nocc,
        nvir,
        gaps,
        lambda i, a, j, bb: eri[i, nocc + a, j, nocc + bb],
        lambda i, a, j, bb: w4[nocc + a, nocc + bb, i, j],
        lambda i, a, j, bb: w4[i, nocc + bb, j, nocc + a],
    )
    rot = spin_rotation(n)
    ar, br = rot.T @ a_so @ rot, rot.T @ b_so @ rot
    apb_so, amb_so = ar + br, ar - br
    e1 = max(maxabs(apb_so[:n, :n] - apb), maxabs(amb_so[:n, :n] - amb))
    e2 = max(maxabs(apb_so[:n, n:]), maxabs(amb_so[:n, n:]))
    check(
        "R1",
        e1 < 1e-12,
        f"code A+B, A-B = singlet blocks of the spin-orbital A, B: max |d| = {e1:.1e}",
    )
    check("R1", e2 < 1e-12, f"no singlet-triplet coupling: {e2:.1e}")
    trip_apb = np.diag(gaps) - np.array(
        [
            [
                w4[nocc + a, nocc + bb, i, j] + w4[i, nocc + bb, j, nocc + a]
                for j in range(nocc)
                for bb in range(nvir)
            ]
            for i in range(nocc)
            for a in range(nvir)
        ]
    )
    check(
        "R1",
        maxabs(apb_so[n:, n:] - trip_apb) < 1e-12,
        "triplet A+B = d - W(ab|ij) - W(ib|ja)",
    )
    check("R1", maxabs(apb - amb) > 1e-3, "negative control: A+B and A-B differ")

    # R2: independent full response solve in the spin-orbital space.
    mu = np.array(
        [[rmat[0][i, nocc + a] for a in range(nvir)] for i in range(nocc)]
    ).ravel()
    mu_so = np.concatenate([mu, mu])
    worst = 0.0
    for uu in (0.0, 0.7):
        om = 1j * uu
        m_full = np.block(
            [[a_so - om * np.eye(2 * n), b_so], [b_so, a_so + om * np.eye(2 * n)]]
        )
        z = np.linalg.solve(m_full, np.concatenate([mu_so, mu_so]).astype(complex))
        alpha_full = mu_so @ (z[: 2 * n] + z[2 * n :])
        sysm = amb @ apb + uu**2 * np.eye(n)
        alpha_code = 4.0 * mu @ np.linalg.solve(sysm, amb @ mu)
        worst = max(worst, abs(alpha_full - alpha_code), abs(alpha_full.imag))
    check(
        "R2",
        worst < 1e-10,
        f"alpha(0), alpha(0.7i): full spin-orbital solve = code form, max |d| = {worst:.1e}",
    )
    rev = 4.0 * mu @ amb @ np.linalg.solve(amb @ apb + 0.49 * np.eye(n), mu)
    check(
        "R2",
        abs(rev - alpha_code) > 1e-6,
        "negative control: the reversed order is a different number",
    )
    check(
        "R2",
        abs(
            4.0 * mu @ np.linalg.solve(apb, mu)
            - 4.0 * mu @ np.linalg.solve(amb @ apb, amb @ mu)
        )
        < 1e-10,
        "u = 0: 4 mu^T (A+B)^-1 mu",
    )

    # R3: W -> v at an HF reference is the CPHF polarizability.
    bare_only = lambda p, q, r, s: float(np.dot(b[:, p, q], b[:, r, s]))  # noqa: E731
    apb_h, _amb_h = code_rpax(nocc, nvir, eps, bare_only, bare_only)
    alpha_tdhf = 4.0 * mu @ np.linalg.solve(apb_h, mu)
    alpha_ff = finite_field_alpha(eri, eps, nocc, rmat[0])
    rel = abs(alpha_tdhf - alpha_ff) / abs(alpha_ff)
    check(
        "R3",
        rel < 1e-6,
        f"TDHF alpha = {alpha_tdhf:.8f}, finite-field RHF = {alpha_ff:.8f}, rel {rel:.1e}",
    )
    alpha_bad = (
        4.0
        * mu
        @ np.linalg.solve(
            np.diag(gaps)
            + 2.0
            * (
                b[:, :nocc, nocc:].reshape(b.shape[0], -1).T
                @ b[:, :nocc, nocc:].reshape(b.shape[0], -1)
            ),
            mu,
        )
    )
    check(
        "R3",
        abs(alpha_bad - alpha_ff) / abs(alpha_ff) > 1e-3,
        "negative control: Hartree 2 instead of 4 and no exchange is not the CPHF polarizability",
    )


def main():
    symbolic()
    part_b(np.random.default_rng(20261115))
    part_r(np.random.default_rng(20261116))
    print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all identities hold")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
