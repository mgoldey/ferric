"""ferric-owned ECP integrals: analytic angular projection + numerical radial quadrature.

Design prototype for replacing libecpint (FINDINGS "Iteration 25"). No PySCF code is used here; PySCF appears only
in oracle.py / the tests, as a reference.

Conventions (identical to ferric's `ecp.rs` so the Rust port is a drop-in):
  * A shell is (l, centre[3], exponents[], coefficients[]) with BARE-Cartesian coefficients (gto_norm(l, a) folded in):
    chi_ijk(r) = sum_k c_k (x-Ax)^i (y-Ay)^j (z-Az)^k exp(-a_k |r-A|^2), Cartesian order CCA (lx desc, then ly desc).
  * An ECP centre is (centre[3], ams[], ns[], exponents[], coefficients[]); the channel with the MAXIMUM am is the
    local one (libecpint convention); every other am l is a semi-local projector sum_m |lm> U_l(r) <lm|.
    U(r) = sum d r^(n-2) exp(-zeta r^2)   (BSE r_exponents n; FINDINGS Iteration 22 (2) confirmed the power).
  * Spherical output uses libcint's cart2sph (p order x, y, z; l >= 2 order m = -l..l), exactly ecp.rs's C2S tables,
    reproduced here from our own real-harmonic coefficients (checked against the tables in the tests).

Semi-local (type 2):  <A|U_l P_l|B> = sum_m int_0^inf r^2 U_l(r) F^A_lm(r) F^B_lm(r) dr,
   F_lm(r) = int dOmega Y_lm(Omega) chi(C + r Omega)
           = 4 pi exp(-a (r-|a|)^2) sum_{N,lam} T[c, m, N, lam] r^N ktil_lam(2 a |a| r),
   T[c,m,N,lam] = sum_{p+q+s=N} C(i,p)C(j,q)C(k,s) (-ax)^(i-p)(-ay)^(j-q)(-az)^(k-s)
                  sum_mu Y_lam,mu(a_hat) int Y_lm Y_lam,mu Ox^p Oy^q Oz^s dOmega,
   ktil_n(z) = exp(-z) i_n(z)  (exp-scaled modified spherical Bessel; bounded, cancellation-free), a = A - C.
Local (type 1): the product chi_A chi_B is one Gaussian exp(-(al+be) r^2 + 2 k.r) (k = al a + be b) times a polynomial;
   its angular integral is the lam-expansion of exp(2 k.r) against monomials, same ktil and angular tables.
Radial: every (primitive pair, ECP term) integrand is  exp(-p (r - r0)^2) x (analytic, slowly varying);
   integrated by Gauss-Legendre on a window [max(0, r0 - T/sqrt p), r* + T/sqrt p] (see radial_window()).
"""

import math
from functools import lru_cache

import numpy as np

FOURPI = 4.0 * math.pi

# ----------------------------------------------------------------------------- small combinatorics


@lru_cache(maxsize=None)
def binom(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def dfact(n):  # (n)!!, (-1)!! = 1
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def ncart(l):
    return (l + 1) * (l + 2) // 2


@lru_cache(maxsize=None)
def cart_list(l):
    """CCA Cartesian order (libint2 / libecpint / libcint): lx descending, then ly descending."""
    return tuple(
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    )


def gto_norm(l, a):
    """libcint primitive normalisation (ecp.rs gto_norm)."""
    return math.sqrt(
        2 ** (2 * l + 3)
        * math.factorial(l + 1)
        * (2 * a) ** (l + 1.5)
        / (math.factorial(2 * l + 2) * math.sqrt(math.pi))
    )


@lru_cache(maxsize=None)
def sphere_monomial(a, b, c):
    """int x^a y^b z^c dOmega over the unit sphere (exact up to one rounding)."""
    if a % 2 or b % 2 or c % 2:
        return 0.0
    return FOURPI * dfact(a - 1) * dfact(b - 1) * dfact(c - 1) / dfact(a + b + c + 1)


# ----------------------------------------------------------------------------- real spherical harmonics


@lru_cache(maxsize=None)
def ylm_coeffs(l):
    """Real unit-sphere harmonics Y_lm, m = -l..l, as {(a,b,c): coef} monomial dicts (x^a y^b z^c, a+b+c = l).

    Helgaker/Jorgensen/Olsen Eq. 6.4.47-6.4.50 solid harmonics S_lm (no Condon-Shortley phase, S_11 = x, S_1-1 = y)
    scaled by sqrt((2l+1)/4pi). Orthonormality is asserted in the tests."""
    out = []
    for m in range(-l, l + 1):
        am = abs(m)
        nlm = math.sqrt(
            2.0
            * math.factorial(l + am)
            * math.factorial(l - am)
            / (2.0 if m == 0 else 1.0)
        ) / (2**am * math.factorial(l))
        d = {}
        for t in range((l - am) // 2 + 1):
            for u in range(t + 1):
                for w in range(1 if m < 0 else 0, am + 1, 2):  # w = 2v
                    sgn = -1.0 if ((t + (w - (1 if m < 0 else 0)) // 2) % 2) else 1.0
                    c = (
                        sgn
                        * 0.25**t
                        * binom(l, t)
                        * binom(l - t, am + t)
                        * binom(t, u)
                        * binom(am, w)
                    )
                    key = (2 * t + am - 2 * u - w, 2 * u + w, l - 2 * t - am)
                    d[key] = d.get(key, 0.0) + c * nlm
        out.append(
            {k: v * math.sqrt((2 * l + 1) / FOURPI) for k, v in d.items() if v != 0.0}
        )
    return tuple(out)


def ylm_values(lmax, u):
    """Y_lm(u) for a unit vector u, l = 0..lmax: list of arrays (2l+1)."""
    x, y, z = u
    res = []
    for l in range(lmax + 1):
        res.append(
            np.array(
                [
                    sum(c * x**a * y**b * z**cc for (a, b, cc), c in d.items())
                    for d in ylm_coeffs(l)
                ]
            )
        )
    return res


@lru_cache(maxsize=None)
def c2s(l):
    """libcint cart2sph (ncart x nsph) for libcint-normalised Cartesians = ecp.rs C2S tables.
    Rows: CCA Cartesians; columns: p -> (x, y, z), l >= 2 -> m = -l..l."""
    Y = ylm_coeffs(l)
    order = [2, 0, 1] if l == 1 else list(range(2 * l + 1))  # m = +1 (x), -1 (y), 0 (z)
    M = np.zeros((ncart(l), 2 * l + 1))
    for j, mi in enumerate(order):
        for i, k in enumerate(cart_list(l)):
            M[i, j] = Y[mi].get(k, 0.0)
    return M


@lru_cache(maxsize=None)
def ang_table(l, lam, p, q, s):
    """Q[m, mu] = int Y_lm Y_lam,mu Ox^p Oy^q Oz^s dOmega  ((2l+1) x (2lam+1))."""
    A, B = ylm_coeffs(l), ylm_coeffs(lam)
    Q = np.zeros((2 * l + 1, 2 * lam + 1))
    if (l + lam + p + q + s) % 2 or abs(l - lam) > p + q + s:
        return Q
    for i, da in enumerate(A):
        for j, db in enumerate(B):
            acc = []
            for (a1, b1, c1), ca in da.items():
                for (a2, b2, c2), cb in db.items():
                    v = sphere_monomial(a1 + a2 + p, b1 + b2 + q, c1 + c2 + s)
                    if v:
                        acc.append(ca * cb * v)
            Q[i, j] = math.fsum(acc)
    Q[np.abs(Q) < 1e-15 * max(1.0, np.abs(Q).max())] = 0.0
    return Q


@lru_cache(maxsize=None)
def mono_ylm(lam, p, q, s):
    """int Ox^p Oy^q Oz^s Y_lam,mu dOmega, mu = -lam..lam  (type-1 angular table)."""
    return np.array(
        [
            math.fsum(
                c * sphere_monomial(a + p, b + q, cc + s) for (a, b, cc), c in d.items()
            )
            for d in ylm_coeffs(lam)
        ]
    )


# ----------------------------------------------------------------------------- exp-scaled Bessel functions


def z_switch(nmax):
    """Below: series + downward recurrence; above: upward recurrence (both <= 2.1e-15 rel vs mpmath for nmax <= 12)."""
    return max(16.0, 4.0 * nmax)


def _series(n, z):
    """exp(-z) i_n(z) by the all-positive power series (array z)."""
    h = 0.5 * z * z
    term = np.ones_like(z)
    s = np.ones_like(z)
    for k in range(1, 400):
        term = term * h / (k * (2 * n + 2 * k + 1))
        s = s + term
        if np.all(term <= 1e-17 * s):
            break
    return np.exp(-z) * z**n / dfact(2 * n + 1) * s


def bessel_ktil(nmax, z):
    """ktil_n(z) = exp(-z) i_n(z), n = 0..nmax, z >= 0 (array). Returns (nmax+1, len(z)).

    z == 0:               ktil_n = delta_n0.
    0 < z < 1e-8:         two-term expansion exp(-z) z^n/(2n+1)!! (1 + z^2/(2(2n+3)))  (rel. error O(z^4)); avoids the
                          underflow of z^n that would poison the recurrence.
    1e-8 <= z < z_switch: power series (all terms positive, no cancellation) for n = nmax+1 and nmax only, then the
                          DOWNWARD recurrence ktil_{n-1} = ktil_{n+1} + (2n+1)/z ktil_n (stable: i_n is the minimal
                          solution in n).
    z >= z_switch(nmax):  UPWARD recurrence ktil_{n+1} = ktil_{n-1} - (2n+1)/z ktil_n from the closed forms
                          ktil_0 = (1 - e^{-2z})/(2z), ktil_1 = (1 + e^{-2z})/(2z) - (1 - e^{-2z})/(2z^2); stable for
                          z >= 4n.
    Max relative error vs mpmath (tests): <= 2.1e-15 for n <= 12, z in {0} U [1e-300, 1e5]."""
    z = np.asarray(z, dtype=float)
    out = np.zeros((nmax + 1,) + z.shape)
    zero = z == 0.0
    out[0, zero] = 1.0
    tiny = (z > 0.0) & (z < 1e-8)
    if tiny.any():
        zt = z[tiny]
        for n in range(nmax + 1):
            out[n, tiny] = (
                np.exp(-zt)
                * zt**n
                / dfact(2 * n + 1)
                * (1 + zt * zt / (2 * (2 * n + 3)))
            )
    small = (z >= 1e-8) & (z < z_switch(nmax))
    if small.any():
        zs = z[small]
        kp1 = _series(nmax + 1, zs)
        kn = _series(nmax, zs)
        out[nmax, small] = kn
        for n in range(nmax, 0, -1):
            km1 = kp1 + (2 * n + 1) / zs * kn
            out[n - 1, small] = km1
            kp1, kn = kn, km1
    big = z >= z_switch(nmax)
    if big.any():
        zb = z[big]
        e2 = np.exp(-2 * zb)
        k0 = (1 - e2) / (2 * zb)
        out[0, big] = k0
        if nmax >= 1:
            k1 = (1 + e2) / (2 * zb) - (1 - e2) / (2 * zb * zb)
            out[1, big] = k1
            km, kc = k0, k1
            for n in range(1, nmax):
                kn = km - (2 * n + 1) / zb * kc
                out[n + 1, big] = kn
                km, kc = kc, kn
    return out


# ----------------------------------------------------------------------------- angular projection (type 2)


@lru_cache(maxsize=None)
def _proj_tensor_cached(lA, l, ax, ay, az):
    a = np.array([ax, ay, az])
    an = float(np.linalg.norm(a))
    u = (
        a / an if an > 0 else np.array([0.0, 0.0, 1.0])
    )  # on-centre: only lam = 0 survives (ktil_lam(0) = 0)
    lmax_lam = l + lA
    Yv = ylm_values(lmax_lam, u)
    carts = cart_list(lA)
    T = np.zeros((len(carts), 2 * l + 1, lA + 1, lmax_lam + 1))
    for ic, (i, j, k) in enumerate(carts):
        for p in range(i + 1):
            for q in range(j + 1):
                for s in range(k + 1):
                    pref = (
                        binom(i, p)
                        * binom(j, q)
                        * binom(k, s)
                        * (-ax) ** (i - p)
                        * (-ay) ** (j - q)
                        * (-az) ** (k - s)
                    )
                    if pref == 0.0:
                        continue
                    N = p + q + s
                    for lam in range(
                        abs(l - N) if N < l else (l + N) % 2, l + N + 1, 2
                    ):
                        Q = ang_table(l, lam, p, q, s)
                        T[ic, :, N, lam] += pref * (Q @ Yv[lam])
    return T


def proj_tensor(lA, l, a):
    """T[c, m, N, lam] for a shell of angular momentum lA displaced by a = A - C (r-independent)."""
    return _proj_tensor_cached(lA, l, float(a[0]), float(a[1]), float(a[2]))


def project_prim(lA, alpha, a, l, r):
    """F[c, m, r] = int Y_lm(Omega) chi_c(C + r Omega) dOmega for ONE bare primitive (coefficient 1)."""
    an = float(np.linalg.norm(a))
    T = proj_tensor(lA, l, a)
    kt = bessel_ktil(l + lA, 2 * alpha * an * r)  # (lam, nr)
    rN = r[None, :] ** np.arange(lA + 1)[:, None]  # (N, nr)
    env = FOURPI * np.exp(-alpha * (r - an) ** 2)
    return np.einsum("cmNl,Nr,lr->cmr", T, rN, kt) * env


# ----------------------------------------------------------------------------- radial windows

RAD_T = 6.0  # window half-width in units of 1/sqrt(p) (GL windows)
RAD_N = 40  # max Gauss-Legendre nodes per window
RAD_NMIN = 16  # min Gauss-Legendre nodes per window
RAD_PER_SIGMA = 3.4  # GL nodes per unit of (window length x sqrt p)
RAD_NH = 20  # Gauss-Hermite nodes for interior windows
RAD_TAU = 6.5  # interior iff t0 = r0 sqrt(p) >= RAD_TAU (all GH nodes at r > 0; neglected r < 0 mass <= e^-42)
# (radial_scheme.py: max rel error 5.2e-14 per window over 42.8k windows, mean 27.7 nodes; GL-40 fixed: 1.35e-14, 40)


@lru_cache(maxsize=None)
def _gl(n):
    return np.polynomial.legendre.leggauss(n)


@lru_cache(maxsize=None)
def _gh(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return x, w * np.exp(
        x * x
    )  # weights for the FULL integrand (the envelope stays explicit)


def radial_window(p, r0, deg, T=None):
    """Integration window for exp(-p (r-r0)^2) r^deg on [0, inf): [max(0, r0 - T/sqrt p), rpk + T/sqrt p] where
    rpk = argmax of r^deg exp(-p(r-r0)^2) (shifts the upper edge out for high powers when r0 is small)."""
    T = RAD_T if T is None else T
    sp = math.sqrt(p)
    rpk = 0.5 * (r0 + math.sqrt(r0 * r0 + 2.0 * deg / p))
    return max(0.0, r0 - T / sp), max(r0, rpk) + T / sp


def radial_nodes(p, r0, deg):
    """Production radial rule for one window (envelope exp(-p (r-r0)^2), polynomial degree deg):
    interior (r0 sqrt p >= RAD_TAU): Gauss-Hermite RAD_NH nodes r = r0 + x/sqrt p;
    otherwise Gauss-Legendre on radial_window() with clamp(ceil(RAD_PER_SIGMA (hi-lo) sqrt p), RAD_NMIN, RAD_N) nodes."""
    sp = math.sqrt(p)
    if r0 * sp >= RAD_TAU:
        x, w = _gh(RAD_NH)
        return r0 + x / sp, w / sp
    lo, hi = radial_window(p, r0, deg)
    n = min(RAD_N, max(RAD_NMIN, math.ceil(RAD_PER_SIGMA * (hi - lo) * sp)))
    return gl_nodes(lo, hi, n)


def gl_nodes(lo, hi, n=None):
    x, w = _gl(RAD_N if n is None else n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


# ----------------------------------------------------------------------------- ECP bookkeeping


def split_ecp(ecp):
    """-> (lmax_local, {l: [(n, zeta, d), ...]} semi-local, [(n, zeta, d)] local)."""
    L = max(ecp["ams"])
    semi, loc = {}, []
    for l, n, z, d in zip(ecp["ams"], ecp["ns"], ecp["exponents"], ecp["coefficients"]):
        if l == L:
            loc.append((n, z, d))
        else:
            semi.setdefault(l, []).append((n, z, d))
    return L, semi, loc


class Stats:
    """Operation counters for the cost model (nodes, F evaluations, screened items)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.windows = self.nodes = self.screened = self.t1_windows = self.t1_nodes = (
            self.ang_contractions
        ) = self.ang_flops = 0
        self.t1_prim_pairs = self.t1_wflops = 0


STATS = Stats()
WINDOW_LOG = None  # list -> every evaluated type-2 window is appended (cost.py feeds these to radial_kernel.cc)
SCREEN = 1e-16  # drop a (primitive pair, term) whose rigorous-ish bound is below SCREEN (absolute)


def _prim_screen_bound(la, lb, ca, cb, al, be, A, B, p, r0, K, n, d):
    """Upper-bound estimate of |int r^2 U F^A F^B| for one (primitive pair, term): the Gaussian envelope's integral
    exp(-K) sqrt(pi/p) times (4pi)^2 |c c d| times the polynomial factor at the window edge (|F| <= 4pi sup|chi|,
    |ktil| <= 1, |T| <= (|a| + r)^l-ish). Deliberately loose (x 1e3 safety) — the tests assert screened == unscreened."""
    rmax = r0 + RAD_T / math.sqrt(p)
    poly = (
        (A + rmax) ** la * (B + rmax) ** lb * max(rmax, 1.0) ** n * (2.0 ** (la + lb))
    )
    return (
        1e3
        * abs(ca * cb * d)
        * FOURPI**2
        * math.exp(-K)
        * math.sqrt(math.pi / p)
        * poly
    )


# ----------------------------------------------------------------------------- type 2 (semi-local) Cartesian block


def type2_cart(sa, sb, C, semi, screen=True):
    """sum_l sum_m <sa|l m> U_l <l m|sb> for contracted shells (bare coefficients), Cartesian (ncA x ncB).

    Production form (what the Rust port implements): the angular tensors T^A, T^B depend only on (shell, centre, l),
    so all primitive/term/node work goes into ONE radial tensor per (shell pair, centre, l)
        R[K, lam, lam'] = sum_{al, be} c_al c_be sum_{(n, zeta, d)} d int_0^inf (4pi)^2 r^(n+K)
                          exp(-al (r-A)^2 - be (r-B)^2 - zeta r^2) ktil_lam(2 al A r) ktil_lam'(2 be B r) dr
    and one angular contraction  V[c,c'] = sum_{m,N,lam,N',lam'} T^A[c,m,N,lam] T^B[c',m,N',lam'] R[N+N',lam,lam']."""
    la, lb = sa["l"], sb["l"]
    a = np.asarray(sa["center"], float) - C
    b = np.asarray(sb["center"], float) - C
    A, B = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    V = np.zeros((ncart(la), ncart(lb)))
    rK = np.arange(la + lb + 1)
    for l, terms in semi.items():
        R = radial_tensor_t2(sa, sb, A, B, l, terms, screen)
        if R is None:
            continue
        Ta, Tb = proj_tensor(la, l, a), proj_tensor(lb, l, b)
        # G[c', m, N, lam] = sum_{N', lam'} T^B[c', m, N', lam'] R[N+N', lam, lam']
        Rs = np.stack([R[N : N + lb + 1] for N in range(la + 1)])  # (N, N', lam, lam')
        G = np.einsum("dmPq,NPlq->dmNl", Tb, Rs)
        V += np.einsum("cmNl,dmNl->cd", Ta, G)
        STATS.ang_contractions += 1
        nla, nlb = (
            l + la + 1,
            l + lb + 1,
        )  # (parity halves the live lam; counted in full)
        STATS.ang_flops += (
            2 * ncart(lb) * (2 * l + 1) * (la + 1) * nla * (lb + 1) * nlb
            + 2 * ncart(la) * ncart(lb) * (2 * l + 1) * (la + 1) * nla
        )
    return V


def radial_tensor_t2(sa, sb, A, B, l, terms, screen=True):
    la, lb = sa["l"], sb["l"]
    na, nb = l + la + 1, l + lb + 1
    R = np.zeros((la + lb + 1, na, nb))
    rK = np.arange(la + lb + 1)
    any_ = False
    for al, ca in zip(sa["exponents"], sa["coefficients"]):
        for be, cb in zip(sb["exponents"], sb["coefficients"]):
            for n, zeta, d in terms:
                p = al + be + zeta
                r0 = (al * A + be * B) / p
                K = al * A * A + be * B * B - p * r0 * r0
                if (
                    screen
                    and _prim_screen_bound(la, lb, ca, cb, al, be, A, B, p, r0, K, n, d)
                    < SCREEN
                ):
                    STATS.screened += 1
                    continue
                any_ = True
                if WINDOW_LOG is not None:
                    WINDOW_LOG.append((la, lb, l, al, A, be, B, n, zeta, ca * cb * d))
                r, w = radial_nodes(p, r0, n + la + lb)
                STATS.windows += 1
                STATS.nodes += len(r)
                ka = bessel_ktil(na - 1, 2 * al * A * r)
                kb = bessel_ktil(nb - 1, 2 * be * B * r)
                # exp(-al(r-A)^2 - be(r-B)^2 - zeta r^2) = exp(-p (r-r0)^2 - K): no overflow, no cancellation
                g = (
                    (ca * cb * d * FOURPI * FOURPI)
                    * w
                    * r**n
                    * np.exp(-p * (r - r0) ** 2 - K)
                )
                R += np.einsum("Kr,lr,qr,r->Klq", r[None, :] ** rK[:, None], ka, kb, g)
    return R if any_ else None


def type2_cart_direct(sa, sb, C, semi):
    """Cross-check construction: F^A_lm(r), F^B_lm(r) formed explicitly at every node, then sum_m F^A F^B."""
    la, lb = sa["l"], sb["l"]
    a = np.asarray(sa["center"], float) - C
    b = np.asarray(sb["center"], float) - C
    A, B = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    V = np.zeros((ncart(la), ncart(lb)))
    for l, terms in semi.items():
        Ta, Tb = proj_tensor(la, l, a), proj_tensor(lb, l, b)
        for al, ca in zip(sa["exponents"], sa["coefficients"]):
            for be, cb in zip(sb["exponents"], sb["coefficients"]):
                for n, zeta, d in terms:
                    p = al + be + zeta
                    r, w = radial_nodes(p, (al * A + be * B) / p, n + la + lb)
                    Fa = _F(Ta, la, l, al, A, r)
                    Fb = _F(Tb, lb, l, be, B, r)
                    V += (
                        ca
                        * cb
                        * np.einsum(
                            "amr,bmr,r->ab",
                            Fa,
                            Fb,
                            w * d * r**n * np.exp(-zeta * r * r),
                        )
                    )
    return V


def _F(T, lA, l, alpha, an, r):
    kt = bessel_ktil(l + lA, 2 * alpha * an * r)
    rN = r[None, :] ** np.arange(lA + 1)[:, None]
    env = FOURPI * np.exp(-alpha * (r - an) ** 2)
    return np.einsum("cmNl,Nr,lr->cmr", T, rN, kt) * env


# ----------------------------------------------------------------------------- type 1 (local) Cartesian block


def _poly1d(i, ax):
    """(x - ax)^i as coefficients of x^p, p = 0..i."""
    return np.array([binom(i, p) * (-ax) ** (i - p) for p in range(i + 1)])


def type1_cart(sa, sb, C, loc, screen=True):
    """<sa|U_L|sb> for the local channel: product Gaussian around C, angular integral via the lam expansion."""
    la, lb = sa["l"], sb["l"]
    a = np.asarray(sa["center"], float) - C
    b = np.asarray(sb["center"], float) - C
    A2, B2 = float(a @ a), float(b @ b)
    V = np.zeros((ncart(la), ncart(lb)))
    L = la + lb
    ca_list, cb_list = cart_list(la), cart_list(lb)
    # polynomial of the product in monomials of r (about C): P[ia, ib][(P,Q,S)]
    polys = {}
    for ia, (i1, j1, k1) in enumerate(ca_list):
        for ib, (i2, j2, k2) in enumerate(cb_list):
            px = np.convolve(_poly1d(i1, a[0]), _poly1d(i2, b[0]))
            py = np.convolve(_poly1d(j1, a[1]), _poly1d(j2, b[1]))
            pz = np.convolve(_poly1d(k1, a[2]), _poly1d(k2, b[2]))
            polys[ia, ib] = (px, py, pz)
    for al, cA in zip(sa["exponents"], sa["coefficients"]):
        for be, cB in zip(sb["exponents"], sb["coefficients"]):
            kv = al * a + be * b
            kn = float(np.linalg.norm(kv))
            u = kv / kn if kn > 0 else np.array([0.0, 0.0, 1.0])
            Yv = ylm_values(L, u)
            STATS.t1_prim_pairs += 1
            STATS.t1_wflops += (
                2 * len(ca_list) * len(cb_list) * (L + 1) ** 2
            )  # the final contraction with R1
            # W[ia, ib, N, lam] = sum_{PQS: P+Q+S=N} poly * sum_mu Y_lam,mu(k_hat) int O^PQS Y_lam,mu
            W = np.zeros((len(ca_list), len(cb_list), L + 1, L + 1))
            for (ia, ib), (px, py, pz) in polys.items():
                for P, cx in enumerate(px):
                    if cx == 0.0:
                        continue
                    for Q, cy in enumerate(py):
                        if cy == 0.0:
                            continue
                        for S, cz in enumerate(pz):
                            if cz == 0.0:
                                continue
                            N = P + Q + S
                            STATS.t1_wflops += sum(
                                2 * (2 * lm + 1) + 2 for lm in range(N % 2, N + 1, 2)
                            )
                            for lam in range(N % 2, N + 1, 2):
                                W[ia, ib, N, lam] += (
                                    cx * cy * cz * (mono_ylm(lam, P, Q, S) @ Yv[lam])
                                )
            for n, zeta, d in loc:
                p = al + be + zeta
                r0 = kn / p
                K = al * A2 + be * B2 - kn * kn / p
                if (
                    screen
                    and 1e3
                    * abs(cA * cB * d)
                    * FOURPI
                    * math.exp(-K)
                    * math.sqrt(math.pi / p)
                    * (math.sqrt(max(A2, B2)) + r0 + RAD_T / math.sqrt(p) + 1)
                    ** (L + n)
                    < SCREEN
                ):
                    STATS.screened += 1
                    continue
                r, w = radial_nodes(p, r0, n + L)
                STATS.t1_windows += 1
                STATS.t1_nodes += len(r)
                kt = bessel_ktil(L, 2 * kn * r)
                rN = r[None, :] ** np.arange(L + 1)[:, None]
                env = FOURPI * np.exp(-p * (r - r0) ** 2 - K) * d * r**n * w
                V += cA * cB * np.einsum("abNl,Nr,lr,r->ab", W, rN, kt, env)
    return V


# ----------------------------------------------------------------------------- triples, blocks, derivatives


def triple_cart(sa, sb, ecp, screen=True, channels="all"):
    """<sa|U_ecp|sb> Cartesian block (bare-Cartesian, CCA order) for one ECP centre. channels: all|local|semi."""
    C = np.asarray(ecp["center"], float)
    _, semi, loc = split_ecp(ecp)
    V = np.zeros((ncart(sa["l"]), ncart(sb["l"])))
    if channels in ("all", "semi") and semi:
        V += type2_cart(sa, sb, C, semi, screen)
    if channels in ("all", "local") and loc:
        V += type1_cart(sa, sb, C, loc, screen)
    return V


def to_sph(V, la, lb):
    return c2s(la).T @ V @ c2s(lb)


def ecp_block_spherical(bra, ket, ecps, mask=None, screen=True, channels="all"):
    """Mirror of ferric_integrals::ecp::ecp_block_spherical (rectangular, mask index (a*nket + b)*necp + u)."""
    nr = sum(2 * s["l"] + 1 for s in bra)
    nc = sum(2 * s["l"] + 1 for s in ket)
    out = np.zeros((nr, nc))
    r0 = 0
    for ia, sa in enumerate(bra):
        c0 = 0
        for ib, sb in enumerate(ket):
            acc = np.zeros((ncart(sa["l"]), ncart(sb["l"])))
            for iu, e in enumerate(ecps):
                if mask is not None and not mask[(ia * len(ket) + ib) * len(ecps) + iu]:
                    continue
                acc += triple_cart(sa, sb, e, screen, channels)
            out[r0 : r0 + 2 * sa["l"] + 1, c0 : c0 + 2 * sb["l"] + 1] = to_sph(
                acc, sa["l"], sb["l"]
            )
            c0 += 2 * sb["l"] + 1
        r0 += 2 * sa["l"] + 1
    return out


def _shifted_shell(s, dl, scale_by_alpha):
    """The raised (dl = +1, coefficients 2 a c) or lowered (dl = -1, coefficients c) companion shell."""
    return dict(
        l=s["l"] + dl,
        center=s["center"],
        exponents=list(s["exponents"]),
        coefficients=[
            (2 * a * c if scale_by_alpha else c)
            for a, c in zip(s["exponents"], s["coefficients"])
        ],
    )


def deriv_shell_cart(s, other, ecp, which, screen=True, channels="all"):
    """d/dA_x <s|U|other> (which = 'bra') or d/dB_x <other|U|s> (which = 'ket'): [3][ncart(s) x ncart(other)] with
    rows indexed by s. d chi_ijk/dA_x = 2a (x-Ax)^(i+1).. - i (x-Ax)^(i-1).. (raised/lowered shells)."""
    l = s["l"]
    up = _shifted_shell(s, +1, True)
    Vup = (
        triple_cart(up, other, ecp, screen, channels)
        if which == "bra"
        else triple_cart(other, up, ecp, screen, channels).T
    )
    Vdn = None
    if l > 0:
        dn = _shifted_shell(s, -1, False)
        Vdn = (
            triple_cart(dn, other, ecp, screen, channels)
            if which == "bra"
            else triple_cart(other, dn, ecp, screen, channels).T
        )
    cu, cd = (
        {c: i for i, c in enumerate(cart_list(l + 1))},
        ({c: i for i, c in enumerate(cart_list(l - 1))} if l else {}),
    )
    D = np.zeros((3, ncart(l), Vup.shape[1]))
    for ic, ijk in enumerate(cart_list(l)):
        for x in range(3):
            e = [0, 0, 0]
            e[x] = 1
            upk = tuple(v + d for v, d in zip(ijk, e))
            D[x, ic] = Vup[cu[upk]]  # raised shell already carries 2a c
            if ijk[x] > 0:
                dnk = tuple(v - d for v, d in zip(ijk, e))
                D[x, ic] -= ijk[x] * Vdn[cd[dnk]]
    return D


def ecp_block_deriv_spherical(
    bra, ket, ecps, mask=None, centre_group=None, ngroup=1, screen=True, channels="all"
):
    """Mirror of ferric_integrals::ecp::ecp_block_deriv_spherical: dict(bra=[3], ket=[3], centre=[ngroup][3]),
    each (nsph(bra) x nsph(ket)); centre := -(bra + ket) per triple (exact translation invariance)."""
    if centre_group is None:
        centre_group = [0] * len(ecps)
    nr = sum(2 * s["l"] + 1 for s in bra)
    nc = sum(2 * s["l"] + 1 for s in ket)
    Db = np.zeros((3, nr, nc))
    Dk = np.zeros((3, nr, nc))
    Dc = np.zeros((ngroup, 3, nr, nc))
    r0 = 0
    for ia, sa in enumerate(bra):
        c0 = 0
        ra = slice(r0, r0 + 2 * sa["l"] + 1)
        for ib, sb in enumerate(ket):
            cb = slice(c0, c0 + 2 * sb["l"] + 1)
            for iu, e in enumerate(ecps):
                if mask is not None and not mask[(ia * len(ket) + ib) * len(ecps) + iu]:
                    continue
                dA = deriv_shell_cart(sa, sb, e, "bra", screen, channels)
                dB = deriv_shell_cart(sb, sa, e, "ket", screen, channels)  # rows = sb
                for x in range(3):
                    va = to_sph(dA[x], sa["l"], sb["l"])
                    vb = to_sph(dB[x].T, sa["l"], sb["l"])
                    Db[x, ra, cb] += va
                    Dk[x, ra, cb] += vb
                    Dc[centre_group[iu], x, ra, cb] -= va + vb
            c0 += 2 * sb["l"] + 1
        r0 += 2 * sa["l"] + 1
    return dict(bra=Db, ket=Dk, centre=Dc)


def ecp_matrix_spherical(shells, ecps, screen=True):
    """Mirror of ecp_matrix_spherical: square symmetric matrix, upper triangle evaluated and mirrored."""
    offs = np.cumsum([0] + [2 * s["l"] + 1 for s in shells])
    V = np.zeros((offs[-1], offs[-1]))
    for i, sa in enumerate(shells):
        for j in range(i, len(shells)):
            sb = shells[j]
            acc = np.zeros((ncart(sa["l"]), ncart(sb["l"])))
            for e in ecps:
                acc += triple_cart(sa, sb, e, screen)
            blk = to_sph(acc, sa["l"], sb["l"])
            V[offs[i] : offs[i + 1], offs[j] : offs[j + 1]] = blk
            V[offs[j] : offs[j + 1], offs[i] : offs[i + 1]] = blk.T
    return V
