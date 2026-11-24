#!/usr/bin/env python3
"""The DIIS error vector F D S - S D F, its stationarity, and the DIIS system.

Derivation: `book/src/08-convergence/01-diis.md#diis-error-vector-extrapolation`
and `#diis-variants` (energy-DIIS interpolant). Code: crates/ferric-scf/src/diis.rs (Diis), called from the
SCF loops of rhf.rs, uhf.rs and rohf.rs.

Conventions: closed shell D = 2 C_o C_o^T, F = h + J[D] - K[D]/2,
C^T S C = 1, f = C^T F C, occupations n = diag(2_o, 0_v). Orbital rotation
C -> C exp(kappa), kappa antisymmetric, stored kappa_ai; g_ai = dE/dkappa_ai.

Symbolic checks (sympy):
  S1  e = F D S - S D F is antisymmetric for symmetric F, D, S.
  S2  In the MO basis C^T e C = f n - n f; for n = diag(2, 0) its (a, i)
      element is 2 f_ai (2 x 2 symbolic f).
  S3  The DIIS system as diis.rs solves it, [[B, 1], [1^T, 0]] [c; lam] =
      [0; 1], gives c = B^-1 1 / (1^T B^-1 1), the constrained minimum of
      |sum_i c_i e_i|^2 with sum_i c_i = 1, and lam = -c^T B c (3 error
      vectors, symbolic Gram matrix).
Numerical checks (random h, positive (mn|ls) = sum_P B B, non-orthogonal S,
n = 6, 2 doubly occupied orbitals):
  N1  dE/dkappa_ai of E(C exp(kappa)) by central differences equals 4 f_ai at
      a random orthonormal C, and the MO error vector is (C^T e C)_ai = 2 f_ai
      = g_ai / 2.
  N2  At the converged RHF solution e = 0 (max |e| < 1e-9), and at a
      non-stationary D, |e| > 0.
  N3  DIIS coefficients from the bordered system, with the Gram block scaled
      by max |B| as the code does, equal the closed form, sum to one, and
      minimize the residual norm against random feasible perturbations.
  N4  DIIS on the toy reaches max |e| < 1e-10 and its energy equals that of
      a direct orbital-rotation minimization to 1e-10.
  N5  UHF: per-spin e_s = F_s D_s S - S D_s F_s vanish at the converged UHF
      solution; e built with the total density in place of D_s does not.
  N6  The orthogonal-basis error X^T e X (X^T S X = 1) has the same Frobenius
      norm as C^T e C, and its (a, i) MO block is 2 f_ai (the exit veto of
      rhf.rs reads max |X^T e X|; DIIS itself uses the AO e).
  L1  Level shift F + mu S C_v C_v^T S adds mu to the virtual-virtual block of
      C^T F C and nothing else; at the converged orbitals it raises the
      virtual eigenvalues by mu and leaves the density unchanged.
  F1  Fermi-Dirac smearing (smearing.rs): the bisected Fermi level holds
      N/2 orbital pairs at several sigma; sigma -> 0 gives aufbau occupations.
  E3  EDIIS interpolant, matrix form: for idempotent occupation-2 D_i and F_i = F(D_i),
      E(sum_i c_i D_i) = g^T c + 1/4 c^T H c on sum_i c_i = 1, with g_i = E_i
      and H_ij = -<D_i - D_j, F_i - F_j>; the occupation-1 form P_i = D_i/2
      carries 1/2; 1/2 on the occupation-2 H (the factor diis.rs used
      before the -1/4 fix) does not reproduce E.
Energy-DIIS interpolant (diis.rs EnergyDiis, book section diis-variants):
  S4  For E(D) = <h, D> + <D, G D>/2 with G linear and symmetric and
      F = dE/dD, and sum_i c_i = 1 (3 points, symbolic):
      E(sum c_i D_i) = sum c_i E_i - 1/4 sum_ij c_i c_j <D_i - D_j, F_i - F_j>.
      The pairing <dD, dF> is invariant under D -> lambda D with F -> F /
      lambda, so the -1/4 holds for any normalization of D provided F is
      the gradient with respect to that D; Kudin-Scuseria-Cances' -1/2 is the
      same identity written with a Fock matrix that is HALF the gradient
      (per-spin closed-shell P = D/2 has dE/dP = 2F).
  S5  ADIIS (Hu-Yang) about the newest point n: E(sum c_i D_i) = E_n +
      <dD, F_n> + 1/2 sum_ij c_i c_j <D_i - D_n, F_j - F_n>; diis.rs stores
      twice this (g_i = 2<D_i - D_n, F_n>), which does not move the minimizer.
  E1  Toy RHF (total density, F = h + J - K/2): the diis.rs EDIIS model
      g = E, H_ij = -1/2 <D_i - D_j, F_i - F_j>, f = g.c + c.H.c/2, equals
      energy_rhf(sum c_i D_i) to 1e-10 at random convex c over 4 densities
      along a DIIS trajectory; the ADIIS model satisfies E_n + f/2 = E.
  E2  Toy UHF: with per-spin D_s, F_s = dE/dD_s and <., .> summed over
      spins the same -1/4 holds (EDIIS is RHF-only in ferric; this records
      what an open-shell port must use).

B_ij is the elementwise (Frobenius) product of the AO error matrices, as in
diis.rs; UHF stacks the two spins in one Gram matrix and applies one
coefficient vector to both Fock matrices.

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  error_vector: F D S - S D F -> F D S + S D F       (S1, N2 fail)
  diis_solve: border column +1 -> -1                 (N3 fails)
  uhf_error: per-spin D_s -> total D                 (N5 fails)
  mo_gradient: 4 f_ai -> 2 f_ai                      (N1 fails)
  ediis: EDIIS_QUAD 0.25 -> 0.5                      (E3 fails)
  fermi_level: bisection branch > -> <               (F1 fails)
  ediis_objective: H factor -1/2 -> -1               (E1 fails)
Exits nonzero if any check fails; prints "FAIL [<claim>] ...".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def error_vector(f, d, s):
    return f @ d @ s - s @ d @ f


def diis_solve(bmat):
    """diis.rs solve_diis_coeffs: [[B~, 1], [1^T, 0]] [c; lam] = [0; 1] with
    B~ = B / max|B| (only the Gram block is scaled)."""
    m = bmat.shape[0]
    a = np.zeros((m + 1, m + 1))
    a[:m, :m] = bmat / np.abs(bmat).max()
    a[:m, m] = 1.0
    a[m, :m] = 1.0
    rhs = np.zeros(m + 1)
    rhs[m] = 1.0
    sol = np.linalg.solve(a, rhs)
    return sol[:m], sol[m]


def gram(e1, e2):
    """B_ij = sum of the elementwise product of two AO error matrices (diis.rs dot)."""
    return np.sum(e1 * e2)


EDIIS_QUAD = 0.25


def ediis_model(g, h, c):
    """Exact EDIIS interpolant for occupation-2 densities: g.c + 1/4 c^T H c."""
    return g @ c + EDIIS_QUAD * c @ h @ c


def level_shift(f, s, c_prev, nocc, mu):
    """F + mu S C_v C_v^T S with C_v the virtual columns of the previous orbitals."""
    scv = s @ c_prev[:, nocc:]
    return f + mu * scv @ scv.T


def fermi_dirac(eps, mu, sigma):
    x = np.clip((eps - mu) / sigma, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(x))


def solve_fermi_level(eps, n_pairs, sigma):
    """Bisection on sum f - N/2 over [eps_min - (20 sigma + 1), eps_max + (20 sigma + 1)]."""
    lo, hi = eps.min() - (20 * sigma + 1), eps.max() + (20 * sigma + 1)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fermi_dirac(eps, mid, sigma).sum() > n_pairs:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Symbolic part
# ---------------------------------------------------------------------------
def symbolic_checks():
    def sym(prefix, n):
        m = sp.zeros(n, n)
        for i in range(n):
            for j in range(i + 1):
                x = sp.Symbol(f"{prefix}{i}{j}", real=True)
                m[i, j] = x
                m[j, i] = x
        return m

    f, d, s = sym("f", 3), sym("d", 3), sym("s", 3)
    e = f * d * s - s * d * f
    e_np = error_vector(np.array(f), np.array(d), np.array(s))
    ok = sp.simplify(e + e.T) == sp.zeros(3, 3) and sp.simplify(
        sp.Matrix(e_np) - e
    ) == sp.zeros(3, 3)
    check("[S1] F D S - S D F is antisymmetric", ok)

    fm = sym("F", 2)
    n = sp.diag(2, 0)
    emo = fm * n - n * fm
    check("[S2] (C^T e C)_ai = 2 f_ai", sp.simplify(emo[1, 0] - 2 * fm[1, 0]) == 0)

    b = sym("b", 3)
    c0, c1, c2, lam = sp.symbols("c0 c1 c2 lam", real=True)
    cv = sp.Matrix([c0, c1, c2])
    ones = sp.ones(3, 1)
    # The code's bordered system: B c + lam 1 = 0, 1^T c = 1. It is the
    # stationarity condition of c^T B c / 2 + lam (1^T c - 1)... up to the
    # factor 2 absorbed in lam, so lam = -c^T B c.
    eqs = list(b * cv + lam * ones) + [(ones.T * cv)[0] - 1]
    sol = sp.solve(eqs, [c0, c1, c2, lam], dict=True)[0]
    cs = cv.subs(sol)
    closed = b.inv() * ones / (ones.T * b.inv() * ones)[0]
    ok = sp.simplify(cs - closed) == sp.zeros(3, 1)
    ok = ok and sp.simplify(sol[lam] + (cs.T * b * cs)[0]) == 0
    lag = (cv.T * b * cv)[0] / 2 + lam * ((ones.T * cv)[0] - 1)
    ok = ok and all(
        sp.simplify(sp.diff(lag, x).subs(sol).subs({lam: sol[lam] / 2}) * 0) == 0
        for x in (c0, c1, c2)
    )
    check("[S3] bordered DIIS system gives the constrained minimum, lam = -c^T B c", ok)

    # S4/S5: a quadratic functional on a 2-dimensional "density" space.
    gm = sym("G", 2)
    hv = sp.Matrix(sp.symbols("h0 h1", real=True))
    pts = [sp.Matrix(sp.symbols(f"x{i}0 x{i}1", real=True)) for i in range(3)]

    def e_q(x):
        return (hv.T * x)[0] + (x.T * gm * x)[0] / 2

    def f_q(x):  # dE/dx
        return hv + gm * x

    c2 = 1 - c0 - c1
    cs3 = [c0, c1, c2]
    xc = sum((ci * xi for ci, xi in zip(cs3, pts)), sp.zeros(2, 1))
    lin = sum(ci * e_q(xi) for ci, xi in zip(cs3, pts))
    quad = sum(
        cs3[i] * cs3[j] * ((pts[i] - pts[j]).T * (f_q(pts[i]) - f_q(pts[j])))[0]
        for i in range(3)
        for j in range(3)
    )
    check(
        "[S4] E(sum c D) = sum c E - 1/4 sum cc <dD, dF>",
        sp.expand(e_q(xc) - lin + quad / 4) == 0,
    )
    # Same identity in the rescaled variable P = D/2 with "Fock" F = (dE/dP)/2:
    # <dP, dF> = <dD, dE/dD>/4 ... i.e. the -1/2 of Kudin-Scuseria-Cances.
    quad_p = sum(
        cs3[i] * cs3[j] * ((pts[i] - pts[j]).T / 2 * (f_q(pts[i]) - f_q(pts[j])))[0]
        for i in range(3)
        for j in range(3)
    )
    check(
        "[S4] per-spin P = D/2, F = grad/2: the factor is -1/2",
        sp.expand(e_q(xc) - lin + quad_p / 2) == 0,
    )
    n = pts[2]
    adiis = (
        e_q(n)
        + ((xc - n).T * f_q(n))[0]
        + sum(
            cs3[i] * cs3[j] * ((pts[i] - n).T * (f_q(pts[j]) - f_q(n)))[0]
            for i in range(3)
            for j in range(3)
        )
        / 2
    )
    check(
        "[S5] ADIIS Taylor model about the newest point is exact",
        sp.expand(e_q(xc) - adiis) == 0,
    )


# ---------------------------------------------------------------------------
# Numerical toy model
# ---------------------------------------------------------------------------
N, NOCC = 6, 2
rng = np.random.default_rng(20261115)
_a = rng.normal(size=(N, N))
S = _a @ _a.T / N + np.eye(N)
_lam, _u = np.linalg.eigh(S)
X = _u @ np.diag(_lam**-0.5) @ _u.T  # X^T S X = 1
_h = rng.normal(size=(N, N)) * 0.3
_shalf = _u @ np.diag(_lam**0.5) @ _u.T
H = _shalf @ (np.diag([-6.0, -5.0, -1.0, 0.0, 0.5, 1.5]) + 0.5 * (_h + _h.T)) @ _shalf
_b = rng.normal(size=(12, N, N)) * 0.15
_b = _b + _b.transpose(0, 2, 1)
ERI = np.einsum("pmn,pls->mnls", _b, _b)


def jk(d):
    return np.einsum("mnls,ls->mn", ERI, d), np.einsum("mlns,ls->mn", ERI, d)


def fock_rhf(d):
    j, k = jk(d)
    return H + j - 0.5 * k


def energy_rhf(d):
    return 0.5 * np.sum(d * (H + fock_rhf(d)))


def density(c):
    co = c[:, :NOCC]
    return 2.0 * co @ co.T


def diag(f):
    eps, cp = np.linalg.eigh(X.T @ f @ X)
    return eps, X @ cp


def expm_antisym(k):
    lam, u = np.linalg.eigh(1j * k)
    return (u @ np.diag(np.exp(-1j * lam)) @ u.conj().T).real


def mo_gradient(c):
    f = c.T @ fock_rhf(density(c)) @ c
    return 4.0 * f[NOCC:, :NOCC]


def scf_diis(max_iter=200, space=8):
    eps, c = diag(H)
    fs, es = [], []
    for it in range(max_iter):
        d = density(c)
        f = fock_rhf(d)
        e = error_vector(f, d, S)
        if np.abs(e).max() < 1e-11:
            return c, it
        fs.append(f)
        es.append(e)  # AO commutator, not orthogonalized (rhf.rs)
        fs, es = fs[-space:], es[-space:]
        if len(es) < 2:
            eps, c = diag(f)
            continue
        bmat = np.array([[gram(a, b) for b in es] for a in es])
        coef, _ = diis_solve(bmat)
        f_ext = sum(ci * fi for ci, fi in zip(coef, fs))
        eps, c = diag(f_ext)
    return c, max_iter


def minimize_rotations(c0):
    """Independent route: Newton-free steepest descent with exact line steps."""
    c = c0.copy()
    for _ in range(4000):
        g = mo_gradient(c)
        if np.abs(g).max() < 1e-12:
            break
        k = np.zeros((N, N))
        k[NOCC:, :NOCC] = -0.05 * g
        k[:NOCC, NOCC:] = -k[NOCC:, :NOCC].T
        c = c @ expm_antisym(k)
    return c


def numerical_checks():
    c = X @ np.linalg.qr(rng.normal(size=(N, N)))[0]
    f_mo = c.T @ fock_rhf(density(c)) @ c
    h_step = 1e-3
    g_fd = np.zeros((N - NOCC, NOCC))
    for a in range(N - NOCC):
        for i in range(NOCC):
            k = np.zeros((N, N))
            k[NOCC + a, i] = h_step
            k[i, NOCC + a] = -h_step
            e = [energy_rhf(density(c @ expm_antisym(s * k))) for s in (-2, -1, 1, 2)]
            g_fd[a, i] = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * h_step)
    e_mo = c.T @ error_vector(fock_rhf(density(c)), density(c), S) @ c
    err_g = np.abs(g_fd - mo_gradient(c)).max()
    err_e = np.abs(e_mo[NOCC:, :NOCC] - 2.0 * f_mo[NOCC:, :NOCC]).max()
    check(
        "[N1] dE/dkappa_ai = 4 f_ai and (C^T e C)_ai = 2 f_ai",
        err_g < 1e-8 and err_e < 1e-10,
        f"FD {err_g:.1e}, MO error {err_e:.1e}",
    )

    c_conv, iters = scf_diis()
    d = density(c_conv)
    e_conv = np.abs(error_vector(fock_rhf(d), d, S)).max()
    d_bad = density(X @ np.linalg.qr(rng.normal(size=(N, N)))[0])
    e_bad = np.abs(error_vector(fock_rhf(d_bad), d_bad, S)).max()
    check(
        "[N2] e = 0 at convergence, nonzero away from it",
        e_conv < 1e-9 and e_bad > 1e-3,
        f"converged {e_conv:.1e} ({iters} it), random {e_bad:.1e}",
    )

    es = [rng.normal(size=(N, N)) for _ in range(4)]
    es = [x - x.T for x in es]
    bmat = np.array([[gram(a, b) for b in es] for a in es])
    coef, lam = diis_solve(bmat)
    binv1 = np.linalg.solve(bmat, np.ones(4))
    closed = binv1 / binv1.sum()
    resid = lambda cc: np.sum(sum(ci * ei for ci, ei in zip(cc, es)) ** 2)  # noqa: E731
    ok = np.abs(coef - closed).max() < 1e-12 and abs(coef.sum() - 1) < 1e-12
    scaled = resid(coef) / np.abs(bmat).max()
    ok = ok and abs(lam + scaled) < 1e-10 * max(1.0, scaled)
    for _ in range(50):
        p = rng.normal(size=4)
        p -= p.mean()
        ok = ok and resid(coef + 1e-2 * p) >= resid(coef) - 1e-14
    check("[N3] bordered system == closed form, minimizes the residual", ok)

    c_ref = minimize_rotations(diag(H)[1])
    e_diis = energy_rhf(d)
    e_ref = energy_rhf(density(c_ref))
    check(
        "[N4] DIIS energy == direct rotation minimization",
        abs(e_diis - e_ref) < 1e-10 and iters < 200,
        f"|dE| {abs(e_diis - e_ref):.1e}",
    )

    # UHF with 2 alpha, 1 beta electrons
    def fock_uhf(da, db):
        ja, ka = jk(da)
        jb, kb = jk(db)
        return H + ja + jb - ka, H + ja + jb - kb

    def uhf_error(f, ds, dt):
        return error_vector(f, ds, S)

    ca, cb = diag(H)[1], diag(H)[1]
    hist = {"a": ([], []), "b": ([], [])}
    for _ in range(300):
        da = ca[:, :2] @ ca[:, :2].T
        db = cb[:, :1] @ cb[:, :1].T
        fa, fb = fock_uhf(da, db)
        ea, eb = uhf_error(fa, da, da + db), uhf_error(fb, db, da + db)
        if max(np.abs(ea).max(), np.abs(eb).max()) < 1e-11:
            break
        # one stacked DIIS over both spins, as the UHF solver does
        fl, el = hist["a"]
        fl.append((fa, fb))
        el.append((ea, eb))
        del fl[:-8], el[:-8]
        if len(el) < 2:
            ca, cb = diag(fa)[1], diag(fb)[1]
            continue
        bmat = np.array([[gram(x[0], y[0]) + gram(x[1], y[1]) for y in el] for x in el])
        coef, _ = diis_solve(bmat)
        fa_x = sum(ci * x[0] for ci, x in zip(coef, fl))
        fb_x = sum(ci * x[1] for ci, x in zip(coef, fl))
        ca, cb = diag(fa_x)[1], diag(fb_x)[1]
    da = ca[:, :2] @ ca[:, :2].T
    db = cb[:, :1] @ cb[:, :1].T
    fa, fb = fock_uhf(da, db)
    e_spin = max(
        np.abs(error_vector(fa, da, S)).max(), np.abs(error_vector(fb, db, S)).max()
    )
    e_total = np.abs(error_vector(fa, da + db, S)).max()
    check(
        "[N5] UHF per-spin errors vanish at convergence; total-D error does not",
        e_spin < 1e-9 and e_total > 1e-4,
        f"per spin {e_spin:.1e}, total-D {e_total:.1e}",
    )

    c = X @ np.linalg.qr(rng.normal(size=(N, N)))[0]
    d = density(c)
    e = error_vector(fock_rhf(d), d, S)
    ex = X.T @ e @ X
    emo = c.T @ e @ c
    f_mo = c.T @ fock_rhf(d) @ c
    ok = abs(np.linalg.norm(ex) - np.linalg.norm(emo)) < 1e-10 * np.linalg.norm(emo)
    ok = ok and np.abs(emo[NOCC:, :NOCC] - 2 * f_mo[NOCC:, :NOCC]).max() < 1e-10
    check("[N6] |X^T e X| = |C^T e C|, MO (a, i) block 2 f_ai", ok)

    # L1 level shift: in the MO basis of the previous orbitals the shift adds
    # mu to the vv block only, so at the converged orbitals the shifted Fock
    # matrix has the same eigenvectors, virtual energies raised by mu.
    mu = 0.7
    f_conv = fock_rhf(density(c_conv))
    f_shift = level_shift(f_conv, S, c_conv, NOCC, mu)
    blk = c_conv.T @ (f_shift - f_conv) @ c_conv
    want = np.diag([0.0] * NOCC + [mu] * (N - NOCC))
    eps0, c0 = diag(f_conv)
    eps1, c1 = diag(f_shift)
    d0, d1 = density(c0), density(c1)
    ok = np.abs(blk - want).max() < 1e-10
    ok = ok and np.abs(eps1[NOCC:] - eps0[NOCC:] - mu).max() < 1e-10
    ok = ok and np.abs(eps1[:NOCC] - eps0[:NOCC]).max() < 1e-10
    ok = ok and np.abs(d1 - d0).max() < 1e-10
    check("[L1] shift mu S C_v C_v^T S: vv block + mu, same converged density", ok)

    # F1 Fermi-Dirac smearing: the bisected mu holds N/2 orbital pairs, and
    # sigma -> 0 gives aufbau occupations.
    eps = np.sort(rng.normal(size=10))
    ok = True
    for sigma in (0.3, 0.05, 1e-3):
        m = solve_fermi_level(eps, 3, sigma)
        ok = ok and abs(fermi_dirac(eps, m, sigma).sum() - 3) < 1e-9
    occ = fermi_dirac(eps, solve_fermi_level(eps, 3, 1e-4), 1e-4)
    ok = ok and np.abs(occ - np.array([1.0] * 3 + [0.0] * 7)).max() < 1e-6
    check("[F1] Fermi level holds N/2 pairs; sigma -> 0 is aufbau", ok)

    # E1 EDIIS interpolant. For idempotent D_i (occupation 2) and F_i = F(D_i),
    # E(sum_i c_i D_i) = g^T c + EDIIS_QUAD c^T H c exactly on sum_i c_i = 1,
    # with g_i = E_i and H_ij = -<D_i - D_j, F_i - F_j> (the diis.rs table).
    # Occupation-1 densities P_i = D_i / 2 with F = h + 2J[P] - K[P] give the
    # literature form, coefficient 1/2 on -sum c_i c_j <P_i - P_j, F_i - F_j>.
    dens = []
    for _ in range(4):
        q, _r = np.linalg.qr(rng.normal(size=(N, N)))
        dens.append(density(X @ q))  # X q is S-orthonormal: D S D = 2 D
    focks = [fock_rhf(d) for d in dens]
    g = np.array([energy_rhf(d) for d in dens])
    m = len(dens)
    h = np.array(
        [
            [-np.sum((dens[i] - dens[j]) * (focks[i] - focks[j])) for j in range(m)]
            for i in range(m)
        ]
    )
    ok = all(np.abs(d @ S @ d - 2.0 * d).max() < 1e-10 for d in dens)
    for _ in range(5):
        c = rng.random(m)
        c /= c.sum()
        exact = energy_rhf(sum(ci * di for ci, di in zip(c, dens)))
        ok = ok and abs(ediis_model(g, h, c) - exact) < 1e-10
        # occupation-1 form: P_i = D_i / 2, -1/2 sum c_i c_j <P_i - P_j, F_i - F_j>
        hp = np.array(
            [
                [
                    -np.sum((dens[i] - dens[j]) / 2.0 * (focks[i] - focks[j]))
                    for j in range(m)
                ]
                for i in range(m)
            ]
        )
        ok = ok and abs(g @ c + 0.5 * c @ hp @ c - exact) < 1e-10
        # 1/2 on the occupation-2 H (the pre-fix diis.rs) misses by 1/4 c^T H c
        ok = ok and abs(g @ c + 0.5 * c @ h @ c - exact) > 1e-6
    check("[E3] EDIIS: E(sum c D) = g.c + 1/4 c^T H c (occupation-2 D)", ok)


def ediis_objective(es, ds, fs):
    """diis.rs EnergyDiis (EDIIS): g_i = E_i, H_ij = -1/2 <D_i - D_j, F_i - F_j>."""
    m = len(es)
    h = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            h[i, j] = -0.5 * np.sum((ds[i] - ds[j]) * (fs[i] - fs[j]))
    return np.array(es), h


def adiis_objective(ds, fs):
    """diis.rs EnergyDiis (ADIIS), newest point last."""
    dn, fn = ds[-1], fs[-1]
    g = np.array([2.0 * np.sum((d - dn) * fn) for d in ds])
    h = np.array([[2.0 * np.sum((di - dn) * (fj - fn)) for fj in fs] for di in ds])
    return g, h


def simplex_objective(g, h, c):
    return g @ c + 0.5 * c @ h @ c


def energy_diis_checks():
    r = np.random.default_rng(7)
    # A DIIS-free Roothaan trajectory from the core guess: 4 idempotent densities.
    ds, fs, es = [], [], []
    d = density(diag(H)[1])
    for _ in range(4):
        f = fock_rhf(d)
        ds.append(d)
        fs.append(f)
        es.append(energy_rhf(d))
        d = density(diag(f)[1])
    g, h = ediis_objective(es, ds, fs)
    ga, ha = adiis_objective(ds, fs)
    worst, worst_a, spread = 0.0, 0.0, 0.0
    for _ in range(6):
        c = r.random(4) + 0.05
        c /= c.sum()
        exact = energy_rhf(sum(ci * di for ci, di in zip(c, ds)))
        worst = max(worst, abs(simplex_objective(g, h, c) - exact))
        worst_a = max(worst_a, abs(es[-1] + 0.5 * simplex_objective(ga, ha, c) - exact))
        spread = max(spread, abs(exact - c @ np.array(es)))
    check(
        "[E1] EDIIS model == E_RHF(sum c D); ADIIS E_n + f/2 == E",
        worst < 1e-10 and worst_a < 1e-10 and spread > 1e-3,
        f"edis {worst:.2e} adiis {worst_a:.2e} quad part {spread:.2e}",
    )

    def fock_uhf(da, db):
        j = jk(da + db)[0]
        ka, kb = jk(da)[1], jk(db)[1]
        return H + j - ka, H + j - kb

    def energy_uhf(da, db):
        fa, fb = fock_uhf(da, db)
        return 0.5 * (np.sum(da * (H + fa)) + np.sum(db * (H + fb)))

    pts = []
    for _ in range(4):
        da = r.normal(size=(N, N)) * 0.2
        db = r.normal(size=(N, N)) * 0.2
        da, db = da + da.T, db + db.T
        fa, fb = fock_uhf(da, db)
        pts.append((da, db, fa, fb, energy_uhf(da, db)))
    c = r.random(4) + 0.05
    c /= c.sum()
    dca = sum(ci * p[0] for ci, p in zip(c, pts))
    dcb = sum(ci * p[1] for ci, p in zip(c, pts))
    quad = sum(
        c[i]
        * c[j]
        * (
            np.sum((pts[i][0] - pts[j][0]) * (pts[i][2] - pts[j][2]))
            + np.sum((pts[i][1] - pts[j][1]) * (pts[i][3] - pts[j][3]))
        )
        for i in range(4)
        for j in range(4)
    )
    model = c @ np.array([p[4] for p in pts]) - 0.25 * quad
    exact = energy_uhf(dca, dcb)
    err = abs(model - exact)
    spread = abs(exact - c @ np.array([p[4] for p in pts]))
    check(
        "[E2] UHF per-spin: the factor is also -1/4",
        err < 1e-10 * max(1.0, abs(exact)) and spread > 1e-3,
        f"{err:.2e} (quad part {spread:.2e})",
    )


symbolic_checks()
numerical_checks()
energy_diis_checks()
if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("all checks passed")
