#!/usr/bin/env python3
"""Proofs: exact exchange of global and range-separated (CAM) hybrids in ferric's KS SCF.

Derivation: crates/ferric-dft/theory.md, "Exact exchange in hybrid and
range-separated hybrid functionals". The code transcribed here:

  * ferric_dft::libxc::XcFunctional::cam_coefficients
      c_sr = cam_alpha + cam_beta,  c_lr = cam_alpha  (libxc xc_hyb_cam_coef)
  * ferric_dft::libxc::k_mix_from_xc_def, ks.rs KsXc::k_mix / KsXcUks::k_mix
      KMix{sr, lr, omega}; a global hybrid is KMix{a, a, 0}
  * ferric_scf::fock_assembly::subtract_rsh_exchange
      K_total = c_sr K[erfc(w r)/r] + c_lr K[erf(w r)/r]
  * rhf.rs solve_rhf:  F = h + J(D) - 1/2 K_total(D),  E = 1/2 Tr[D (h + F)] + E_xc
  * uhf.rs solve_uhf / rohf.rs solve_rohf:
      F_s = h + J(D_a + D_b) - K_total(D_s),  E = 1/2 sum_s Tr[D_s (h + F_s)] + E_xc

Checked:
  A1  erf + erfc = 1; the CAM operator split
      1/r = [a + b erf(w r)]/r + [1 - a - b erf(w r)]/r.
  A2  The kernel libxc reports, cam_alpha/r + cam_beta erfc(w r)/r, equals
      c_sr erfc/r + c_lr erf/r iff c_sr = cam_alpha + cam_beta and
      c_lr = cam_alpha. The Yanai CAM form a/r + b erf/r gives c_sr = a,
      c_lr = a + b.
  A3  Symbolic, 2 basis functions, 8-fold-symmetric generic integrals for both
      kernels and a generic symmetric D: d/dt E_x(D + t dD) = Tr[F_x dD] with
      E_x = -1/4 Tr[D K_total(D)], F_x = -1/2 K_total(D) (RHF), and per spin
      E_x = -1/2 sum_s Tr[D_s K_total(D_s)], F_x,s = -K_total(D_s) (UHF/ROHF).
      Closed-shell limit D_a = D_b = D/2: UHF energy and Fock equal RHF's.
  B   Numerical, 4 s-type Gaussians at random centres:
      B1  erf and erfc ERIs from the closed form and, independently, from
          1/r = (2/sqrt pi) int_0^inf exp(-t^2 r^2) dt split at t = w
          (Gauss-Legendre on [0, w] and on [w, inf)); erf + erfc = Coulomb.
      B2  exact limits: w -> inf gives the Coulomb ERIs; as w -> 0,
          erf(w r)/r -> 2 w / sqrt(pi), so the erf ERIs -> 2 w/sqrt(pi) S_ab S_cd.
      B3  K_total of the code equals K built from the CAM kernel integrated
          directly (weight c_lr for t < w, c_sr for t > w).
      B4  central finite differences of the total electronic energy
          E(D) = Tr[D h] + 1/2 Tr[D J] - 1/4 Tr[D K_total] (+ the UHF
          analogue) against Tr[F dD]; E is quadratic in D, so the central
          difference is exact and the bar is rounding (1e-9 relative).
      B5  closed-shell limit of the UHF energy and Fock, numerically.
  C   libxc parameters -> (c_sr, c_lr, w) for wB97X, wB97X-V, CAM-B3LYP,
      HSE06 and LC-wPBE equal the published short-/long-range HF fractions.
      Optional cross-checks, run only with DERIV_CROSSCHECK=1 and each
      skipped with a message when its package is missing: PySCF's rsh_coeff (alpha = c_LR, beta = c_SR - c_LR)
      reproduces the same (c_sr, c_lr, w); ferric's
      XcFunctional.exchange_mix equals the published values; ferric's
      AoIntegrals.veff (E, V) satisfies dE/dD = V by central differences for
      wB97X-V, CAM-B3LYP and HSE06 on H2/STO-3G.

Mutations (the runner substitutes the code line once; each exits 1):
  libxc_to_ferric returns (alpha, alpha + beta): c_sr and c_lr swapped
      -> C1 (and A2) fail
  rhf_fock's exchange prefactor 1/2 -> 1/4: E and F stay mutually
      consistent, but the RHF/UHF closed-shell limit breaks -> B5 fails
  uhf_fock's per-spin exchange prefactor 1 -> 1/2 -> B5 fails
  make_k_total puts c_sr on the erf and c_lr on the erfc matrix (fitters
      swapped in subtract_rsh_exchange) -> B3 fails
  rhf_energy drops the 1/2 on Tr[D (h + F)] -> B4 fails

Run: python3 scripts/dft_hybrid_rsh_sympy.py
"""

import itertools
import os
import math
import sys

import numpy as np
import sympy as sp

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------ code transcriptions


def libxc_to_ferric(alpha, beta):
    """XcFunctional::cam_coefficients: (cam_alpha, cam_beta) -> (c_sr, c_lr)."""
    return alpha + beta, alpha


def k_of(eri, d):
    """K[D]_mn = sum_ls (ml|ns) D_ls."""
    return np.einsum("mlns,ls->mn", eri, d)


def j_of(eri, d):
    """J[D]_mn = sum_ls (mn|ls) D_ls."""
    return np.einsum("mnls,ls->mn", eri, d)


def make_k_total(eri_erfc, eri_erf):
    """fock_assembly::subtract_rsh_exchange's K_total for KMix (c_sr, c_lr)."""

    def k_total(d, mix):
        c_sr, c_lr = mix
        return c_sr * k_of(eri_erfc, d) + c_lr * k_of(eri_erf, d)

    return k_total


def rhf_fock(h, eri, k_total, d, mix):
    """solve_rhf: F = h + J(D) - 1/2 K_total(D), D the total density."""
    j = j_of(eri, d)
    return h + j - 0.5 * k_total(d, mix)


def uhf_fock(h, eri, k_total, d_tot, ds, mix):
    """solve_uhf / solve_rohf: F_s = h + J(D_a + D_b) - K_total(D_s)."""
    j = j_of(eri, d_tot)
    return h + j - k_total(ds, mix)


def rhf_energy(h, d, f):
    """solve_rhf: e_elec_no_xc = 1/2 Tr[D (h + F)]."""
    return 0.5 * np.sum(d * (h + f))


def uhf_energy(h, da, db, fa, fb):
    """solve_uhf: e_elec_no_xc = 1/2 sum_s Tr[D_s (h + F_s)]."""
    return 0.5 * (np.sum(da * (h + fa)) + np.sum(db * (h + fb)))


# ------------------------------------------------------------ A: symbolic


def part_a():
    x, w, r, a, b = sp.symbols("x omega r a b", positive=True)
    check(
        "A1 erf + erfc = 1",
        sp.simplify((sp.erf(x) + sp.erfc(x) - 1).rewrite(sp.erf)) == 0,
    )
    cam = (a + b * sp.erf(w * r)) / r + (1 - a - b * sp.erf(w * r)) / r
    check("A1 CAM split sums to 1/r (claim A1)", sp.simplify(cam - 1 / r) == 0)

    # A2: kernel identities, matched as polynomials in E = erf(w r).
    e = sp.Symbol("E")
    al, be, csr, clr = sp.symbols("cam_alpha cam_beta c_sr c_lr")
    libxc_kernel = al + be * (1 - e)  # cam_alpha + cam_beta erfc
    ferric_kernel = csr * (1 - e) + clr * e
    sol = sp.solve(
        sp.Poly(sp.expand(libxc_kernel - ferric_kernel), e).coeffs(),
        [csr, clr],
        dict=True,
    )
    code_sr, code_lr = libxc_to_ferric(al, be)
    check(
        "A2 libxc (cam_alpha, cam_beta) -> c_sr = alpha + beta, c_lr = alpha (claim A2)",
        len(sol) == 1
        and sp.simplify(sol[0][csr] - code_sr) == 0
        and sp.simplify(sol[0][clr] - code_lr) == 0,
    )
    ya, yb = sp.symbols("a_Y b_Y")
    sol_y = sp.solve(
        sp.Poly(sp.expand(ya + yb * e - ferric_kernel), e).coeffs(),
        [csr, clr],
        dict=True,
    )
    check(
        "A2 Yanai CAM (a/r + b erf/r) -> c_sr = a, c_lr = a + b",
        sp.simplify(sol_y[0][csr] - ya) == 0
        and sp.simplify(sol_y[0][clr] - (ya + yb)) == 0,
    )

    # A3: generic 8-fold-symmetric integrals over 2 basis functions.
    def generic_eri(tag):
        g = {}
        for p, q, rr, s in itertools.product(range(2), repeat=4):
            pq, rs = tuple(sorted((p, q))), tuple(sorted((rr, s)))
            key = tuple(sorted((pq, rs)))
            if key not in g:
                g[key] = sp.Symbol(f"{tag}{key[0][0]}{key[0][1]}{key[1][0]}{key[1][1]}")
        return lambda p, q, rr, s: g[
            tuple(sorted((tuple(sorted((p, q))), tuple(sorted((rr, s))))))
        ]

    g_sr, g_lr = generic_eri("s"), generic_eri("l")
    c_sr, c_lr, t = sp.symbols("c_sr c_lr t")

    def sym_mat(tag):
        m = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"{tag}{min(i, j)}{max(i, j)}"))
        return m

    def k_sym(dm):
        return sp.Matrix(
            2,
            2,
            lambda m, n: sum(
                (c_sr * g_sr(m, l2, n, s2) + c_lr * g_lr(m, l2, n, s2)) * dm[l2, s2]
                for l2 in range(2)
                for s2 in range(2)
            ),
        )

    def tr(m1, m2):
        return sum(m1[i, j] * m2[i, j] for i in range(2) for j in range(2))

    d, dd = sym_mat("D"), sym_mat("dD")
    e_rhf = lambda dm: -sp.Rational(1, 4) * tr(dm, k_sym(dm))  # noqa: E731
    f_rhf = -sp.Rational(1, 2) * k_sym(d)
    lhs = sp.diff(e_rhf(d + t * dd), t).subs(t, 0)
    check(
        "A3 RHF: d/dt E_x = Tr[F_x dD], F_x = -1/2 K_total (claim A3)",
        sp.expand(lhs - tr(f_rhf, dd)) == 0,
    )

    da, db, dda, ddb = sym_mat("Da"), sym_mat("Db"), sym_mat("dDa"), sym_mat("dDb")
    e_uhf = lambda xa, xb: -sp.Rational(1, 2) * (tr(xa, k_sym(xa)) + tr(xb, k_sym(xb)))  # noqa: E731
    lhs_u = sp.diff(e_uhf(da + t * dda, db + t * ddb), t).subs(t, 0)
    rhs_u = tr(-k_sym(da), dda) + tr(-k_sym(db), ddb)
    check(
        "A3 UHF: d/dt E_x = sum_s Tr[F_x,s dD_s], F_x,s = -K_total(D_s)",
        sp.expand(lhs_u - rhs_u) == 0,
    )
    check(
        "A3 closed-shell limit D_s = D/2: E_x(UHF) = E_x(RHF), F_x,s = F_x",
        sp.expand(e_uhf(d / 2, d / 2) - e_rhf(d)) == 0
        and sp.expand(-k_sym(d / 2) - f_rhf) == sp.zeros(2, 2),
    )


# ------------------------------------------------------------ B: s-Gaussian model


def boys0(tv):
    return (
        1.0 if tv < 1e-14 else 0.5 * math.sqrt(math.pi / tv) * math.erf(math.sqrt(tv))
    )


def pair(a, ca, b, cb):
    p = a + b
    return (
        p,
        (a * ca + b * cb) / p,
        math.exp(-a * b / p * float(np.sum((ca - cb) ** 2))),
    )


def eri_closed(prims, omega=None):
    """(ab|cd) over unnormalized s Gaussians; erf(w r)/r when omega is given."""
    n = len(prims)
    g = np.zeros((n, n, n, n))
    for i, j, k, l2 in itertools.product(range(n), repeat=4):
        p, pc, kab = pair(*prims[i], *prims[j])
        q, qc, kcd = pair(*prims[k], *prims[l2])
        zeta = p * q / (p + q)
        r2 = float(np.sum((pc - qc) ** 2))
        pref = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q)) * kab * kcd
        if omega is None:
            g[i, j, k, l2] = pref * boys0(zeta * r2)
        else:
            s = omega**2 / (omega**2 + zeta)
            g[i, j, k, l2] = pref * math.sqrt(s) * boys0(zeta * s * r2)
    return g


def gauss_kernel(prims, i, j, k, l2, tv):
    """int int rho_ab rho_cd exp(-t^2 r12^2), vectorized over t."""
    p, pc, kab = pair(*prims[i], *prims[j])
    q, qc, kcd = pair(*prims[k], *prims[l2])
    zeta = p * q / (p + q)
    r2 = float(np.sum((pc - qc) ** 2))
    t2 = tv**2
    return (
        kab
        * kcd
        * (math.pi**2 / (p * q)) ** 1.5
        * (zeta / (zeta + t2)) ** 1.5
        * np.exp(-zeta * t2 * r2 / (zeta + t2))
    )


def eri_quadrature(prims, omega, w_low, w_high, npts=400):
    """(2/sqrt pi) [w_low int_0^w + w_high int_w^inf] of the Gaussian kernel."""
    x, wq = np.polynomial.legendre.leggauss(npts)
    t_lo = 0.5 * omega * (x + 1)
    w_lo = 0.5 * omega * wq
    u = 0.5 * (x + 1)  # t = omega / u on [omega, inf)
    t_hi = omega / u
    w_hi = 0.5 * wq * omega / u**2
    n = len(prims)
    g = np.zeros((n, n, n, n))
    for idx in itertools.product(range(n), repeat=4):
        lo = np.dot(w_lo, gauss_kernel(prims, *idx, t_lo))
        hi = np.dot(w_hi, gauss_kernel(prims, *idx, t_hi))
        g[idx] = 2 / math.sqrt(math.pi) * (w_low * lo + w_high * hi)
    return g


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def part_b():
    rng = np.random.default_rng(20261114)
    prims = [
        (float(rng.uniform(0.4, 2.0)), rng.uniform(-1.0, 1.0, 3)) for _ in range(4)
    ]
    omega = 0.4
    eri = eri_closed(prims)
    eri_erf = eri_closed(prims, omega)
    eri_erfc = eri - eri_erf  # the code's erfc operator
    q_erf = eri_quadrature(prims, omega, 1.0, 0.0)
    q_erfc = eri_quadrature(prims, omega, 0.0, 1.0)
    r1, r2 = rel(eri_erf, q_erf), rel(eri_erfc, q_erfc)
    check(
        "B1 erf ERIs: closed form = quadrature on [0, w] (claim B1)",
        r1 < 1e-11,
        f"rel {r1:.1e}",
    )
    check(
        "B1 erfc ERIs: Coulomb - erf = quadrature on [w, inf)",
        r2 < 1e-11,
        f"rel {r2:.1e}",
    )
    r3 = rel(q_erf + q_erfc, eri)
    check(
        "B1 erf + erfc quadratures = Coulomb closed form", r3 < 1e-11, f"rel {r3:.1e}"
    )
    big = rel(eri_closed(prims, 1e5), eri)
    check("B2 w -> inf: erf ERIs -> Coulomb", big < 1e-8, f"rel {big:.1e}")
    # small w: erf(w r)/r = 2 w/sqrt(pi) (1 - w^2 r^2/3 + ...), relative error O(w^2)
    w_small = 1e-5
    ovl = np.array(
        [
            [pair(*pa, *pb)[2] * (math.pi / pair(*pa, *pb)[0]) ** 1.5 for pb in prims]
            for pa in prims
        ]
    )
    lim0 = 2 * w_small / math.sqrt(math.pi) * np.einsum("ab,cd->abcd", ovl, ovl)
    small = rel(eri_closed(prims, w_small), lim0)
    check(
        "B2 w -> 0: erf ERIs -> 2 w/sqrt(pi) S_ab S_cd",
        small < 1e-8,
        f"rel {small:.1e}",
    )

    k_total = make_k_total(eri_erfc, eri_erf)
    mix = (0.167, 1.0)  # wB97X-V's (c_sr, c_lr)
    d = rng.standard_normal((4, 4))
    d = d + d.T
    cam_direct = eri_quadrature(prims, omega, mix[1], mix[0])
    r4 = rel(k_total(d, mix), k_of(cam_direct, d))
    check(
        "B3 K_total = K over the CAM kernel (c_lr below w, c_sr above) (claim B3)",
        r4 < 1e-11,
        f"rel {r4:.1e}",
    )

    h = rng.standard_normal((4, 4))
    h = h + h.T
    dd = rng.standard_normal((4, 4))
    dd = dd + dd.T
    step = 1e-3

    def e_rhf(dm):
        return rhf_energy(h, dm, rhf_fock(h, eri, k_total, dm, mix))

    for label, mx in (
        ("RSH", mix),
        ("global hybrid", (0.25, 0.25)),
        ("HF", (1.0, 1.0)),
    ):

        def e_r(dm, mx=mx):
            return rhf_energy(h, dm, rhf_fock(h, eri, k_total, dm, mx))

        fd = (e_r(d + step * dd) - e_r(d - step * dd)) / (2 * step)
        an = float(np.sum(rhf_fock(h, eri, k_total, d, mx) * dd))
        err = abs(fd - an) / max(abs(an), 1.0)
        check(
            f"B4 RHF {label}: FD dE/dD = Tr[F dD] (claim B4)",
            err < 1e-9,
            f"rel {err:.1e}",
        )

    da = rng.standard_normal((4, 4))
    da = da + da.T
    db = rng.standard_normal((4, 4))
    db = db + db.T
    dda = rng.standard_normal((4, 4))
    dda = dda + dda.T
    ddb = rng.standard_normal((4, 4))
    ddb = ddb + ddb.T

    def uhf_parts(xa, xb):
        fa = uhf_fock(h, eri, k_total, xa + xb, xa, mix)
        fb = uhf_fock(h, eri, k_total, xa + xb, xb, mix)
        return uhf_energy(h, xa, xb, fa, fb), fa, fb

    fd = (
        uhf_parts(da + step * dda, db + step * ddb)[0]
        - uhf_parts(da - step * dda, db - step * ddb)[0]
    ) / (2 * step)
    _, fa, fb = uhf_parts(da, db)
    an = float(np.sum(fa * dda) + np.sum(fb * ddb))
    err = abs(fd - an) / max(abs(an), 1.0)
    check("B4 UHF RSH: FD dE/dD_s = sum_s Tr[F_s dD_s]", err < 1e-9, f"rel {err:.1e}")

    e_u, fa, fb = uhf_parts(d / 2, d / 2)
    f_r = rhf_fock(h, eri, k_total, d, mix)
    lim = max(
        abs(e_u - e_rhf(d)),
        float(np.max(np.abs(fa - f_r))),
        float(np.max(np.abs(fb - f_r))),
    )
    check(
        "B5 closed-shell limit: UHF energy and Fock = RHF (claim B5)",
        lim < 1e-12,
        f"max {lim:.1e}",
    )


# ------------------------------------------------------------ C: parameters

# libxc (cam_alpha, cam_beta, omega) per functional, from the libxc sources
# (hyb_gga_xc_wb97.c, hyb_gga_xc_cam_b3lyp.c, hyb_gga_xc_hse.c,
# hyb_gga_xc_lc_wpbe in hyb_gga_x_wpbeh / lc_wpbe). For a short-range-only
# hybrid such as HSE06, xc_hyb_init_sr sets cam_alpha = 0, cam_beta = fraction.
LIBXC = {
    "wB97X": ("HYB_GGA_XC_WB97X", 1.0, -(1.0 - 0.157706), 0.3),
    "wB97X-V": ("HYB_GGA_XC_WB97X_V", 1.0, -(1.0 - 0.167), 0.3),
    "CAM-B3LYP": ("HYB_GGA_XC_CAM_B3LYP", 0.65, -0.46, 0.33),
    "HSE06": ("HYB_GGA_XC_HSE06", 0.0, 0.25, 0.11),
    "LC-wPBE": ("HYB_GGA_XC_LC_WPBE", 1.0, -1.0, 0.4),
}
# Published (short-range HF, long-range HF, omega in bohr^-1):
# Chai & Head-Gordon JCP 128, 084106 (2008) c_x = 0.157706;
# Mardirossian & Head-Gordon PCCP 16, 9904 (2014) c_x = 0.167;
# Yanai, Tew, Handy CPL 393, 51 (2004) alpha = 0.19, alpha + beta = 0.65;
# Krukau et al. JCP 125, 224106 (2006) 1/4 short range;
# Vydrov & Scuseria JCP 125, 234109 (2006) LC-wPBE, w = 0.4.
PUBLISHED = {
    "wB97X": (0.157706, 1.0, 0.3),
    "wB97X-V": (0.167, 1.0, 0.3),
    "CAM-B3LYP": (0.19, 0.65, 0.33),
    "HSE06": (0.25, 0.0, 0.11),
    "LC-wPBE": (0.0, 1.0, 0.4),
}
PYSCF_NAME = {
    "wB97X": "wb97x",
    "wB97X-V": "wb97x_v",
    "CAM-B3LYP": "camb3lyp",
    "HSE06": "hse06",
    "LC-wPBE": "lc_wpbe",
}


def close3(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def part_c():
    for name, (_, al, be, om) in LIBXC.items():
        c_sr, c_lr = libxc_to_ferric(al, be)
        check(
            f"C1 {name}: ferric (c_sr, c_lr, w) = published",
            close3((c_sr, c_lr, om), PUBLISHED[name]),
        )
    if os.environ.get("DERIV_CROSSCHECK") != "1":
        print("[SKIP] C2-C4 PySCF/ferric cross-checks (set DERIV_CROSSCHECK=1)")
        return
    try:
        from pyscf.dft import libxc as pxc
    except ImportError:
        print("[SKIP] C2 PySCF not importable")
    else:
        for name, pname in PYSCF_NAME.items():
            omega, alpha, beta = pxc.rsh_coeff(pname)
            # PySCF: alpha = c_LR, beta = c_SR - c_LR.
            check(
                f"C2 {name}: PySCF rsh_coeff -> published",
                close3((alpha + beta, alpha, omega), PUBLISHED[name]),
            )
    try:
        import ferric
    except ImportError:
        print("[SKIP] C3/C4 ferric not importable")
        return
    print(f"ferric build: {getattr(ferric, '__build__', 'unknown')}")
    mol = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nH 0 0 0.74\n")
    ints = ferric.AoIntegrals(mol, ferric.BasisSet.bundled("sto-3g"))
    for name in ("wB97X-V", "CAM-B3LYP", "HSE06"):
        xc = ints.xc_functional(LIBXC[name][0])
        check(
            f"C3 {name}: ferric exchange_mix = published",
            close3(xc.exchange_mix, PUBLISHED[name], 1e-6),
        )
        rng = np.random.default_rng(7)
        n = ints.nao
        d = np.full((n, n), 0.6) + 0.1 * np.eye(n)
        dd = rng.standard_normal((n, n))
        dd = dd + dd.T
        step = 1e-4
        ep = ints.veff(d + step * dd, xc)[0]
        em = ints.veff(d - step * dd, xc)[0]
        _, v = ints.veff(d, xc)
        fd = (ep - em) / (2 * step)
        an = float(np.sum(v * dd))
        # E_xc is not quadratic; the h^2 truncation of a smooth functional at
        # h = 1e-4 is far below this bar on a 2-function basis.
        err = abs(fd - an) / max(abs(an), 1.0)
        check(f"C4 {name}: ferric veff dE/dD = V (FD)", err < 1e-6, f"rel {err:.1e}")


def main():
    part_a()
    part_b()
    part_c()
    print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all identities hold")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
