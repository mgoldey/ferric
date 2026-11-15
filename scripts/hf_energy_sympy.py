#!/usr/bin/env python3
"""Hartree-Fock energy functionals and Fock matrices as ferric codes them.

Derivation: crates/ferric-scf/theory.md, "Hartree-Fock energy functionals
(RHF, UHF, ROHF)". Code: crates/ferric-scf/src/rhf.rs (solve_rhf),
uhf.rs (solve_uhf), rohf.rs (solve_rohf, roothaan_fock_inner,
rohf_gradient_mo).

Conventions (as coded): J(D)_mn = sum_ls (mn|ls) D_ls,
K(D)_mn = sum_ls (ml|ns) D_ls.
  RHF   D = 2 C_o C_o^T, F = h + J(D) - K(D)/2, E = tr[D (h + F)]/2 + E_nn.
  UHF   D_s = C_so C_so^T, F_s = h + J(D_a + D_b) - K(D_s),
        E = sum_s tr[D_s (h + F_s)]/2 + E_nn.
  ROHF  one orbital set, D_b = C_c C_c^T, D_a = D_b + C_o C_o^T, the UHF
        functional at (D_a, D_b), and the Roothaan effective Fock
        (projector form of PySCF get_roothaan_fock, Guest-Saunders).

Symbolic checks (sympy, 2 basis functions, the 6 unique integrals, a generic
symmetric D):
  S1  dE_RHF/dD = F_RHF (the off-diagonal symbol carries 2 F_12).
  S2  dE_UHF/dD_s = F_s, both spins.
  S3  UHF(D/2, D/2) = RHF(D), F_a = F_b = F_RHF there.
  S4  E_2e = sum Gamma (mn|ls) with the gradient code's Gamma
      (gradient.rs gamma: D D/2 - D D/4; gamma_uhf: D D/2 - (Da Da + Db Db)/2).
Numerical checks (random h, 8-fold symmetric positive (mn|ls) = sum_P B B,
orthonormal AO basis, n = 7):
  N1  F = dE/dD by central differences along random symmetric directions
      (RHF, UHF alpha, UHF beta; E is quadratic, so the stencil is exact up to
      rounding; bar 1e-9).
  N2  The converged RHF energy equals the MO Slater-Condon form
      2 sum_i h_ii + sum_ij [2 (ii|jj) - (ij|ij)].
  N3  At the converged ROHF orbitals the effective Fock is block-diagonal by
      the block table (MO blocks cc/oo/vv = Fc, oc = Fb, vo = Fa, vc = Fc).
  N4  dE/dkappa of E(C exp(kappa)) by central differences equals 2 g, with g
      the code's rohf_gradient_mo blocks ((Fa+Fb)_vc, (Fa)_vo, (Fb)_oc), at a
      random orthonormal C (fourth-order stencil, step 1e-3,
      bar 1e-9); rotations inside cc, oo, vv leave E unchanged.
  N5  At the converged ROHF C every core-open, core-virtual and open-virtual
      rotation is stationary (|dE/dkappa| < 1e-9).
  N6  The ROHF energy (UHF functional at the ROHF densities) equals the
      high-spin Slater-Condon energy of the same determinant.
  N7  E_UHF <= E_ROHF for the same occupation (variational ordering).

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  rhf_fock: exchange weight 1/2 -> 1                 (N1 RHF fails)
  uhf_focks: exchange weight 1 -> 1/2                (N1 UHF fails)
  roothaan_fock: open-closed block F_b -> F_a        (N3 block table fails)
  rohf_gradient_mo: closed-virtual F_a + F_b -> F_a  (N4 dE/dkappa fails)
Exits nonzero on the first failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def fail_hard(label):
    """A failed SCF is a failed check (exit 1 with a FAIL line), not a crash."""
    print(f"FAIL {label}")
    sys.exit(1)


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------------------
# Symbolic part
# ---------------------------------------------------------------------------
def sym_eri():
    """8-fold symmetric (mn|ls) for n = 2 from the 6 unique values."""
    u = sp.symbols("g1111 g1112 g1122 g1212 g1222 g2222")
    table = {}

    def key(m, n, la, s):
        a, b = sorted((m, n)), sorted((la, s))
        pair = sorted([tuple(a), tuple(b)])
        return (pair[0], pair[1])

    names = {
        ((0, 0), (0, 0)): u[0],
        ((0, 0), (0, 1)): u[1],
        ((0, 0), (1, 1)): u[2],
        ((0, 1), (0, 1)): u[3],
        ((0, 1), (1, 1)): u[4],
        ((1, 1), (1, 1)): u[5],
    }
    for m in range(2):
        for n in range(2):
            for la in range(2):
                for s in range(2):
                    table[(m, n, la, s)] = names[key(m, n, la, s)]
    return table


def sym_matrix(prefix):
    a, b, c = sp.symbols(f"{prefix}11 {prefix}12 {prefix}22")
    return sp.Matrix([[a, b], [b, c]]), (a, b, c)


def sym_j(g, d):
    return sp.Matrix(
        2,
        2,
        lambda m, n: sum(
            g[(m, n, la, s)] * d[la, s] for la in range(2) for s in range(2)
        ),
    )


def sym_k(g, d):
    return sp.Matrix(
        2,
        2,
        lambda m, n: sum(
            g[(m, la, n, s)] * d[la, s] for la in range(2) for s in range(2)
        ),
    )


def symbolic_checks():
    g = sym_eri()
    h, _ = sym_matrix("h")
    d, (d11, d12, d22) = sym_matrix("D")
    f = h + sym_j(g, d) - sym_k(g, d) / 2
    e = sp.expand(sum((d.multiply_elementwise(h + f))) / 2)
    ok = (
        sp.simplify(sp.diff(e, d11) - f[0, 0]) == 0
        and sp.simplify(sp.diff(e, d22) - f[1, 1]) == 0
        and sp.simplify(sp.diff(e, d12) - 2 * f[0, 1]) == 0
    )
    check("S1 dE_RHF/dD = F_RHF (symbolic, n=2)", ok)

    da, (a11, a12, a22) = sym_matrix("Da")
    db, (b11, b12, b22) = sym_matrix("Db")
    jt = sym_j(g, da + db)
    fa = h + jt - sym_k(g, da)
    fb = h + jt - sym_k(g, db)
    eu = sp.expand(
        (sum(da.multiply_elementwise(h + fa)) + sum(db.multiply_elementwise(h + fb)))
        / 2
    )
    ok = all(
        sp.simplify(sp.diff(eu, sym) - w * fs[i, j]) == 0
        for fs, syms in ((fa, (a11, a12, a22)), (fb, (b11, b12, b22)))
        for sym, (i, j), w in zip(syms, ((0, 0), (0, 1), (1, 1)), (1, 2, 1))
    )
    check("S2 dE_UHF/dD_s = F_s (symbolic, n=2)", ok)

    sub = {
        a11: d11 / 2,
        a12: d12 / 2,
        a22: d22 / 2,
        b11: d11 / 2,
        b12: d12 / 2,
        b22: d22 / 2,
    }
    ok = sp.simplify(eu.subs(sub) - e) == 0 and all(
        sp.simplify(x.subs(sub) - y) == 0 for x, y in zip(fa, f)
    )
    check("S3 UHF(D/2, D/2) = RHF(D) and F_a = F_RHF", ok)

    idx = [
        (m, n, la, s)
        for m in range(2)
        for n in range(2)
        for la in range(2)
        for s in range(2)
    ]
    e2_rhf = sum(
        (
            sp.Rational(1, 2) * d[m, n] * d[la, s]
            - sp.Rational(1, 4) * d[m, la] * d[n, s]
        )
        * g[(m, n, la, s)]
        for (m, n, la, s) in idx
    )
    ok = (
        sp.simplify(sp.expand(e2_rhf) - sp.expand(e - sum(d.multiply_elementwise(h))))
        == 0
    )
    dt = da + db
    e2_uhf = sum(
        (
            sp.Rational(1, 2) * dt[m, n] * dt[la, s]
            - sp.Rational(1, 2) * (da[m, la] * da[n, s] + db[m, la] * db[n, s])
        )
        * g[(m, n, la, s)]
        for (m, n, la, s) in idx
    )
    ok_u = (
        sp.simplify(sp.expand(e2_uhf) - sp.expand(eu - sum(dt.multiply_elementwise(h))))
        == 0
    )
    check("S4 E_2e = sum Gamma (mn|ls), RHF and UHF Gamma of gradient.rs", ok and ok_u)


# ---------------------------------------------------------------------------
# Numerical model
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20261114)
N = 7
NAUX = 12


def sym(m):
    return 0.5 * (m + m.T)


def make_model():
    h = np.diag(np.linspace(-6.0, 3.0, N)) + 0.3 * sym(rng.standard_normal((N, N)))
    b = 0.2 * rng.standard_normal((NAUX, N, N))
    b = 0.5 * (b + b.transpose(0, 2, 1))
    g = np.einsum("Pmn,Pls->mnls", b, b)
    return h, g


def jmat(g, d):
    return np.einsum("mnls,ls->mn", g, d)


def kmat(g, d):
    return np.einsum("mlns,ls->mn", g, d)


def rhf_fock(h, g, d):
    jm, km = jmat(g, d), kmat(g, d)
    f = h + jm - 0.5 * km
    return f


def rhf_energy(h, g, d):
    return 0.5 * np.sum(d * (h + rhf_fock(h, g, d)))


def uhf_focks(h, g, da, db):
    jm = jmat(g, da + db)

    def one(d_s):
        km_s = kmat(g, d_s)
        return h + jm - km_s

    return one(da), one(db)


def uhf_energy(h, g, da, db):
    fa, fb = uhf_focks(h, g, da, db)
    return 0.5 * (np.sum(da * (h + fa)) + np.sum(db * (h + fb)))


def roothaan_fock(fa, fb, da, db, s):
    """Transcription of rohf.rs roothaan_fock_inner."""
    n = s.shape[0]
    fc = 0.5 * (fa + fb)
    pc = db @ s
    po = (da - db) @ s
    pv = np.eye(n) - da @ s
    pct, pot, pvt = pc.T, po.T, pv.T
    f = 0.5 * pct @ fc @ pc
    f = f + 0.5 * pot @ fc @ po
    f = f + 0.5 * pvt @ fc @ pv
    f = f + pot @ fb @ pc
    f = f + pot @ fa @ pv
    f = f + pvt @ fc @ pc
    return f + f.T


def rohf_gradient_mo(fa_mo, fb_mo, nc, no):
    """Transcription of rohf.rs rohf_gradient_mo (rows = target, cols = source)."""
    n = fa_mo.shape[0]
    na = nc + no
    g = np.zeros((n, n))
    for p in range(na, n):
        for q in range(nc):
            g[p, q] = fa_mo[p, q] + fb_mo[p, q]
    for p in range(na, n):
        for q in range(nc, na):
            g[p, q] = fa_mo[p, q]
    for p in range(nc, na):
        for q in range(nc):
            g[p, q] = fb_mo[p, q]
    return g


class Diis:
    def __init__(self, size=8):
        self.f, self.e, self.size = [], [], size

    def extrapolate(self, f, e):
        self.f.append(f)
        self.e.append(e)
        if len(self.f) > self.size:
            self.f.pop(0)
            self.e.pop(0)
        m = len(self.f)
        b = -np.ones((m + 1, m + 1))
        b[m, m] = 0.0
        for i in range(m):
            for j in range(m):
                b[i, j] = np.sum(self.e[i] * self.e[j])
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        try:
            c = np.linalg.solve(b, rhs)[:m]
        except np.linalg.LinAlgError:
            return f
        return sum(ci * fi for ci, fi in zip(c, self.f))


def solve_rhf(h, g, nocc):
    _, c = np.linalg.eigh(h)
    diis = Diis()
    for _ in range(300):
        d = 2.0 * c[:, :nocc] @ c[:, :nocc].T
        f = rhf_fock(h, g, d)
        err = f @ d - d @ f
        if np.max(np.abs(err)) < 1e-12:
            break
        eps, c = np.linalg.eigh(diis.extrapolate(f, err))
    else:
        fail_hard("RHF SCF converges on the toy model")
    eps, c = np.linalg.eigh(f)
    return c, eps


def solve_uhf(h, g, na, nb):
    _, c = np.linalg.eigh(h)
    ca, cb = c.copy(), c.copy()
    # Break the alpha/beta symmetry of the guess.
    rot = np.eye(N)
    th = 0.3
    rot[[nb - 1, nb], [nb - 1, nb]] = np.cos(th)
    rot[nb - 1, nb], rot[nb, nb - 1] = np.sin(th), -np.sin(th)
    cb = cb @ rot
    diis = Diis()
    for _ in range(500):
        da = ca[:, :na] @ ca[:, :na].T
        db = cb[:, :nb] @ cb[:, :nb].T
        fa, fb = uhf_focks(h, g, da, db)
        ea, eb = fa @ da - da @ fa, fb @ db - db @ fb
        if max(np.max(np.abs(ea)), np.max(np.abs(eb))) < 1e-12:
            break
        fx = diis.extrapolate(np.concatenate([fa, fb]), np.concatenate([ea, eb]))
        _, ca = np.linalg.eigh(fx[:N])
        _, cb = np.linalg.eigh(fx[N:])
    else:
        fail_hard("UHF SCF converges on the toy model")
    return uhf_energy(h, g, da, db)


def rohf_densities(c, nc, no):
    db = c[:, :nc] @ c[:, :nc].T
    da = db + c[:, nc : nc + no] @ c[:, nc : nc + no].T
    return da, db


def solve_rohf(h, g, nc, no):
    """ROHF by DIIS on R_eff with the rohf_gradient_mo error, a level shift
    (0.5 open, 1.0 virtual) in the current MO frame for the orbital update
    only, then canonicalization inside the c, o, v blocks."""
    s = np.eye(N)
    _, c = np.linalg.eigh(h)
    diis = Diis()
    shift = np.zeros(N)
    shift[nc : nc + no], shift[nc + no :] = 0.5, 1.0
    for _ in range(500):
        da, db = rohf_densities(c, nc, no)
        fa, fb = uhf_focks(h, g, da, db)
        reff = roothaan_fock(fa, fb, da, db, s)
        gm = rohf_gradient_mo(c.T @ fa @ c, c.T @ fb @ c, nc, no)
        err = s @ c @ (gm - gm.T) @ c.T @ s
        if np.max(np.abs(err)) < 1e-12:
            break
        f = diis.extrapolate(reff, err) + s @ c @ np.diag(shift) @ c.T @ s
        _, c = np.linalg.eigh(f)
    else:
        fail_hard("ROHF SCF converges on the toy model")
    reff_mo = c.T @ reff @ c
    for blk in (slice(0, nc), slice(nc, nc + no), slice(nc + no, N)):
        _, u = np.linalg.eigh(reff_mo[blk, blk])
        c[:, blk] = c[:, blk] @ u
    return c


def expm_antisym(k):
    w, v = np.linalg.eigh(1j * k)
    return np.real(v @ np.diag(np.exp(-1j * w)) @ v.conj().T)


def rohf_energy_of_c(h, g, c, nc, no):
    da, db = rohf_densities(c, nc, no)
    return uhf_energy(h, g, da, db)


def numeric_checks():
    h, g = make_model()

    # N1: F = dE/dD.
    step = 1e-3
    d = sym(rng.standard_normal((N, N)))
    da, db = sym(rng.standard_normal((N, N))), sym(rng.standard_normal((N, N)))
    worst = 0.0
    for _ in range(3):
        x = sym(rng.standard_normal((N, N)))
        fd = (rhf_energy(h, g, d + step * x) - rhf_energy(h, g, d - step * x)) / (
            2 * step
        )
        worst = max(worst, abs(fd - np.sum(rhf_fock(h, g, d) * x)))
    check("N1 F = dE/dD (RHF, central FD)", worst < 1e-9, f"max err {worst:.2e}")
    fa, fb = uhf_focks(h, g, da, db)
    wa = wb = 0.0
    for _ in range(3):
        x = sym(rng.standard_normal((N, N)))
        fd_a = (
            uhf_energy(h, g, da + step * x, db) - uhf_energy(h, g, da - step * x, db)
        ) / (2 * step)
        fd_b = (
            uhf_energy(h, g, da, db + step * x) - uhf_energy(h, g, da, db - step * x)
        ) / (2 * step)
        wa = max(wa, abs(fd_a - np.sum(fa * x)))
        wb = max(wb, abs(fd_b - np.sum(fb * x)))
    check("N1 F_a = dE/dD_a (UHF, central FD)", wa < 1e-9, f"max err {wa:.2e}")
    check("N1 F_b = dE/dD_b (UHF, central FD)", wb < 1e-9, f"max err {wb:.2e}")
    e_cs = uhf_energy(h, g, d / 2, d / 2) - rhf_energy(h, g, d)
    check("N1 UHF(D/2, D/2) = RHF(D) (numeric)", abs(e_cs) < 1e-11, f"diff {e_cs:.2e}")

    # N2: converged RHF vs Slater-Condon in the MO basis.
    nocc = 3
    c, _ = solve_rhf(h, g, nocc)
    co = c[:, :nocc]
    e_rhf = rhf_energy(h, g, 2.0 * co @ co.T)
    hmo = co.T @ h @ co
    gmo = np.einsum("mnls,mi,nj,lk,sq->ijkq", g, co, co, co, co)
    e_sc = 2 * np.trace(hmo) + sum(
        2 * gmo[i, i, j, j] - gmo[i, j, i, j] for i in range(nocc) for j in range(nocc)
    )
    check(
        "N2 E_RHF = Slater-Condon closed shell",
        abs(e_rhf - e_sc) < 1e-10,
        f"diff {e_rhf - e_sc:.2e}",
    )

    # ROHF: nc closed, no open (high spin). N3 (table) and N4 hold at ANY
    # orthonormal orbital set, so they run on a random one before the SCF.
    nc, no = 2, 2
    na = nc + no
    blk = {"c": slice(0, nc), "o": slice(nc, na), "v": slice(na, N)}
    k0 = rng.standard_normal((N, N))
    c_r = expm_antisym(k0 - k0.T)
    da_r, db_r = rohf_densities(c_r, nc, no)
    fa_r, fb_r = uhf_focks(h, g, da_r, db_r)
    fa_mo, fb_mo = c_r.T @ fa_r @ c_r, c_r.T @ fb_r @ c_r
    fc_mo = 0.5 * (fa_mo + fb_mo)
    reff_mo = c_r.T @ roothaan_fock(fa_r, fb_r, da_r, db_r, np.eye(N)) @ c_r
    table = {
        ("c", "c"): fc_mo, ("o", "o"): fc_mo, ("v", "v"): fc_mo,
        ("o", "c"): fb_mo, ("v", "o"): fa_mo, ("v", "c"): fc_mo,
    }  # fmt: skip
    worst = max(
        np.max(np.abs(reff_mo[blk[p], blk[q]] - m[blk[p], blk[q]]))
        for (p, q), m in table.items()
    )
    check(
        "N3 Roothaan effective Fock block table (MO)",
        worst < 1e-12,
        f"max err {worst:.2e}",
    )

    # N4: dE/dkappa = 2 g at a non-converged C.
    gm = rohf_gradient_mo(fa_mo, fb_mo, nc, no)
    hk = 1e-3

    def dedk(cc, p, q):
        """Fourth-order central difference of E(C exp(kappa)), kappa_pq = -kappa_qp = t."""

        def e_at(t):
            kk = np.zeros((N, N))
            kk[p, q], kk[q, p] = t, -t
            return rohf_energy_of_c(h, g, cc @ expm_antisym(kk), nc, no)

        return (8 * (e_at(hk) - e_at(-hk)) - (e_at(2 * hk) - e_at(-2 * hk))) / (12 * hk)

    def same_block(p, q):
        return (
            (p < nc and q < nc)
            or (nc <= p < na and nc <= q < na)
            or (p >= na and q >= na)
        )

    pairs = [(p, q) for p in range(N) for q in range(p)]
    worst = 0.0
    inv = 0.0
    for p, q in pairs:
        val = dedk(c_r, p, q)
        if same_block(p, q):
            inv = max(inv, abs(val))
        else:
            worst = max(worst, abs(val - 2.0 * gm[p, q]))
    check(
        "N4 dE/dkappa = 2 g (rohf_gradient_mo) off convergence",
        worst < 1e-9,
        f"max err {worst:.2e}",
    )
    check("N4 cc/oo/vv rotations leave E unchanged", inv < 1e-9, f"max |dE| {inv:.2e}")

    c = solve_rohf(h, g, nc, no)
    da, db = rohf_densities(c, nc, no)
    fa, fb = uhf_focks(h, g, da, db)
    reff_mo = c.T @ roothaan_fock(fa, fb, da, db, np.eye(N)) @ c
    offd = np.max(np.abs(reff_mo - np.diag(np.diag(reff_mo))))
    check(
        "N3 converged R_eff is diagonal in its own orbitals",
        offd < 1e-10,
        f"max off-diag {offd:.2e}",
    )

    # N5: stationarity at the converged ROHF C.
    stat = max(abs(dedk(c, p, q)) for p, q in pairs if not same_block(p, q))
    check(
        "N5 converged ROHF stationary (c-o, c-v, o-v)",
        stat < 1e-9,
        f"max |dE/dk| {stat:.2e}",
    )

    # N6: ROHF energy vs high-spin Slater-Condon in MO integrals.
    e_rohf = uhf_energy(h, g, da, db)
    cm = c[:, :na]
    hmo = cm.T @ h @ cm
    gmo = np.einsum("mnls,mi,nj,lk,sq->ijkq", g, cm, cm, cm, cm)
    cl, op = range(nc), range(nc, na)
    jj = lambda i, j: gmo[i, i, j, j]  # noqa: E731
    kk = lambda i, j: gmo[i, j, i, j]  # noqa: E731
    e_hs = (
        2 * sum(hmo[i, i] for i in cl)
        + sum(hmo[t, t] for t in op)
        + sum(2 * jj(i, j) - kk(i, j) for i in cl for j in cl)
        + sum(2 * jj(i, t) - kk(i, t) for i in cl for t in op)
        + 0.5 * sum(jj(t, u) - kk(t, u) for t in op for u in op)
    )
    check(
        "N6 E_ROHF = high-spin Slater-Condon",
        abs(e_rohf - e_hs) < 1e-10,
        f"diff {e_rohf - e_hs:.2e}",
    )

    # N7: UHF is at or below ROHF.
    e_uhf = solve_uhf(h, g, na, nc)
    check(
        "N7 E_UHF <= E_ROHF",
        e_uhf <= e_rohf + 1e-10,
        f"E_UHF - E_ROHF = {e_uhf - e_rohf:.3e}",
    )


def main():
    symbolic_checks()
    numeric_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
