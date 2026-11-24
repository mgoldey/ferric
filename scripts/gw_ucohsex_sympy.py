#!/usr/bin/env python3
"""Proofs: ferric-gw's open-shell (per-spin) static self-energy and the per-spin Fermi level.

Derivation: `book/src/20-gw-bse/07-open-shell-gw.md#open-shell-gw-spin-summed-screening-per` (open shell). The code
transcribed here (auxiliary index first, `B_s[P,p,q]` per spin, `M_s[xi,p,q] =
sum_P U[P,xi] B_s[P,p,q]` with U the eigenvectors of the SPIN-SUMMED static
dielectric):

  * energy.rs (unrestricted dielectric): eps~_U(0) = 1 + Pi_a + Pi_b,
    Pi_s = 2 sum_{i_s a_s} B_s[:,ia] B_s[:,ia]^T / Delta_{ia}   (prefactor 2 per spin)
  * cohsex.rs sigma_x_diag, cohsex_pieces and u_cohsex.rs run_u_cohsex, per spin:
        Sigma^x_s(p)  = - sum_{i in s} sum_P B_s[P,p,i]^2
        dSigma_SEX_s(p) = - sum_{i in s} sum_xi w_xi M_s[xi,p,i]^2
        Sigma_COH_s(p)  = + 1/2 sum_{q in s} sum_xi w_xi M_s[xi,p,q]^2,  w = 1/lambda - 1
  * u_sigma.rs qp_per_spin_g0w0: each spin continues Sigma_c from its own line
    z = eps_F,s + i u with eps_F,s the mid-gap of THAT spin (fermi_level(&eps_act, n_occ_act)).
  * u_sigma.rs run_u_g0w0 / qp_per_spin_g0w0: the dynamic per-spin Sigma_c
    (claim U3) from the same M_s and the spin-summed eps~(iu).

Checked:
  U1  Lattice model (6 sites, spin-dependent orthogonal orbitals, 2 alpha and
      1 beta electrons, v = L L^T): the code's per-spin Sigma^x, dSigma_SEX
      and Sigma_COH from B, the spin-summed dielectric and its PDEP modes equal
      the spin-orbital real-space construction, with no auxiliary or PDEP basis:
      chi0 = sum_s chi0_s, W = (1 - v chi0)^-1 v, W_d = W - v,
      dSigma_SEX_s(p) = - sum_{i in s} <p i|W_d|i p>, Sigma_COH_s(p) = (1/2) W_d(r,r)
      projected on p (the sum over the complete q set is the delta function).
      Negative controls: summing the SEX over the other spin's occupied
      orbitals, and building W from one spin's Pi only, both fail.
  U1b Closed-shell limit: alpha = beta orbitals and occupations give equal
      alpha and beta self-energies, and the prefactor 4 closed-shell dielectric
      of the same model reproduces them.
  U2  Per-spin Fermi level: the Thiele continuation of the single-pole Sigma_c
      sampled on z = eps_F + i u, with the occupied side set by
      Re(z - eps_q) > 0, reproduces the SAME rational function of z for any
      eps_F inside that spin's gap; a line through the other spin's mid-gap
      (outside this spin's gap) mis-assigns a state and continues to a
      different function. 80-digit arithmetic, tolerance 1e-40.
  U3  Dynamic per-spin Sigma^c (u_sigma.rs run_u_g0w0, module doc):
        Sigma_c,s(m, z) = -(1/pi) sum_{n in s} int_0^inf du W_s,mn(iu)
                           (z - eps_n,s) / ((z - eps_n,s)^2 + u^2),
        W_s,mn(iu) = M_s[:,mn]^T [eps~(iu)^-1 - 1] M_s[:,mn],
        eps~(iu) = 1 + Pi_a(iu) + Pi_b(iu),
        Pi_s(iu) = 2 sum B_s B_s^T Delta / (u^2 + Delta^2),
      on the U1 lattice model at z = eps_F,s + iy (y = 0.3, 1.1), integrated
      with 400 Gauss-Legendre nodes on u = u0 (1+x)/(1-x), equals the
      independent pole sum over the spin-orbital direct-RPA excitations,
      sum_n sum_s V_s,mn^2 / (z - eps_n +- Omega_s) with
      V_s,mn = sum_{jb tau} (mn|jb) (X+Y)_{jb tau, s}  (1e-10). At alpha = beta
      it equals the closed-shell Sigma_c (prefactor-4 Pi). Negative control:
      the other spin's orbital energies in the kernel.

Mutations (the runner substitutes the code line once; each exits 1):
  the per-spin dielectric prefactor 2 -> 4                 -> U1 fails
  the alpha channel's occupied set swapped with beta's     -> U1 fails
  the sign rule Re(z - eps_q) > 0 replaced by True         -> U2 fails
  the frequency-dependent per-spin Pi prefactor 2 -> 4     -> U3 fails
  W_d = eps~^-1 - 1 replaced by eps~^-1 (bare v kept in W)  -> U3 fails

Run: python3 scripts/gw_ucohsex_sympy.py   (needs numpy, mpmath)
"""

import sys

import numpy as np
from mpmath import mp, mpc, mpf

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{name}] {detail}")
    if not ok:
        FAILS.append(name)


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


M = 6  # sites


def orthogonal(rng):
    q, _ = np.linalg.qr(rng.standard_normal((M, M)))
    return q


def spin_channel(rng, nocc):
    phi = orthogonal(rng)
    eps = np.concatenate(
        [
            np.sort(rng.uniform(-1.5, -0.5, nocc)),
            np.sort(rng.uniform(0.3, 1.2, M - nocc)),
        ]
    )
    return phi, eps


def build_b(lmat, phi):
    """B[P,p,q] = sum_r L[r,P] phi[r,p] phi[r,q]."""
    return np.einsum("rP,rp,rq->Ppq", lmat, phi, phi)


def pi_spin(b, eps, nocc, prefactor):
    bov = b[:, :nocc, nocc:].reshape(b.shape[0], -1)
    gaps = (eps[None, nocc:] - eps[:nocc, None]).ravel()
    return prefactor * (bov / gaps) @ bov.T


def code_self_energy(b, eps_unused, nocc_of, lam, u, spin):
    """cohsex.rs / u_cohsex.rs per spin: (Sigma^x, dSigma_SEX, Sigma_COH)."""
    w_red = 1.0 / lam - 1.0
    m_proj = np.einsum("Px,Ppq->xpq", u, b)
    occ = range(nocc_of[spin])
    sx = -sum(np.sum(b[:, :, i] ** 2, axis=0) for i in occ)
    dsex = -sum(np.einsum("x,xp->p", w_red, m_proj[:, :, i] ** 2) for i in occ)
    coh = 0.5 * np.einsum("x,xpq->p", w_red, m_proj**2)
    return sx, dsex, coh


def lattice_self_energy(vmat, phi, eps, nocc, w_d, spin_occ_phi):
    """Spin-orbital real-space construction (no aux or PDEP basis)."""
    nm = phi.shape[1]
    sx = np.zeros(nm)
    dsex = np.zeros(nm)
    coh = np.zeros(nm)
    for p in range(nm):
        for i in range(spin_occ_phi.shape[1]):
            pi_ = phi[:, p] * spin_occ_phi[:, i]
            sx[p] -= pi_ @ vmat @ pi_
            dsex[p] -= pi_ @ w_d @ pi_
        coh[p] = 0.5 * np.sum(phi[:, p] ** 2 * np.diag(w_d))
    return sx, dsex, coh


def part_u1():
    rng = np.random.default_rng(20261117)
    nocc_a, nocc_b = 2, 1
    phi_a, eps_a = spin_channel(rng, nocc_a)
    phi_b, eps_b = spin_channel(rng, nocc_b)
    lmat = 0.35 * rng.standard_normal((M, M))
    vmat = lmat @ lmat.T
    b_a, b_b = build_b(lmat, phi_a), build_b(lmat, phi_b)

    # Code route: spin-summed dielectric (prefactor 2 per spin), PDEP modes.
    pi_a = pi_spin(b_a, eps_a, nocc_a, 2.0)
    pi_b = pi_spin(b_b, eps_b, nocc_b, 2.0)
    eps_mat = np.eye(M) + pi_a + pi_b
    lam, u = np.linalg.eigh(eps_mat)

    # Real-space route: chi0 = sum_s chi0_s on the sites, W = (1 - v chi0)^-1 v.
    def chi0(phi, eps, nocc):
        out = np.zeros((M, M))
        for i in range(nocc):
            for a in range(nocc, M):
                pair = phi[:, i] * phi[:, a]
                out -= 2.0 * np.outer(pair, pair) / (eps[a] - eps[i])
        return out

    chi_tot = chi0(phi_a, eps_a, nocc_a) + chi0(phi_b, eps_b, nocc_b)
    w_full = np.linalg.solve(np.eye(M) - vmat @ chi_tot, vmat)
    w_d = w_full - vmat

    worst = 0.0
    for spin, (b, phi, eps, nocc) in enumerate(
        [(b_a, phi_a, eps_a, nocc_a), (b_b, phi_b, eps_b, nocc_b)]
    ):
        nocc_of = [nocc_a, nocc_b]
        sx, dsex, coh = code_self_energy(b, eps, nocc_of, lam, u, spin)
        rng_occ = nocc_of[spin]
        sx_l, dsex_l, coh_l = lattice_self_energy(
            vmat, phi, eps, nocc, w_d, phi[:, :rng_occ]
        )
        worst = max(
            worst, maxabs(sx - sx_l), maxabs(dsex - dsex_l), maxabs(coh - coh_l)
        )
    check(
        "U1",
        worst < 1e-12,
        f"code per-spin Sigma^x, dSigma_SEX, Sigma_COH = real-space spin-orbital construction: max |d| = {worst:.1e}",
    )

    # Negative controls.
    sx, dsex_a, coh = code_self_energy(
        b_a, eps_a, [nocc_b, nocc_a], lam, u, 0
    )  # swapped occupied set
    _, dsex_l, _ = lattice_self_energy(
        vmat, phi_a, eps_a, nocc_a, w_d, phi_a[:, :nocc_a]
    )
    check(
        "U1",
        maxabs(dsex_a - dsex_l) > 1e-4,
        "negative control: SEX over the other spin's occupied orbitals is wrong",
    )
    lam1, u1 = np.linalg.eigh(np.eye(M) + pi_a)
    _, dsex_one, _ = code_self_energy(b_a, eps_a, [nocc_a, nocc_b], lam1, u1, 0)
    check(
        "U1",
        maxabs(dsex_one - dsex_l) > 1e-4,
        "negative control: W from the alpha Pi alone is wrong",
    )

    # U1b closed-shell limit.
    phi, eps = spin_channel(rng, 2)
    b = build_b(lmat, phi)
    eps_u = np.eye(M) + pi_spin(b, eps, 2, 2.0) + pi_spin(b, eps, 2, 2.0)
    eps_c = np.eye(M) + pi_spin(b, eps, 2, 4.0)
    check(
        "U1b",
        maxabs(eps_u - eps_c) < 1e-14,
        "alpha = beta: 1 + Pi_a + Pi_b = closed-shell 1 + Pi (prefactor 4)",
    )
    lam_u, uu = np.linalg.eigh(eps_u)
    lam_c, uc = np.linalg.eigh(eps_c)
    sa = code_self_energy(b, eps, [2, 2], lam_u, uu, 0)
    sb = code_self_energy(b, eps, [2, 2], lam_u, uu, 1)
    sc = code_self_energy(b, eps, [2, 2], lam_c, uc, 0)
    d = max(maxabs(np.asarray(x) - np.asarray(y)) for x, y in zip(sa, sb))
    d2 = max(maxabs(np.asarray(x) - np.asarray(y)) for x, y in zip(sa, sc))
    check(
        "U1b",
        d < 1e-13 and d2 < 1e-12,
        f"alpha = beta self-energies and = closed-shell COHSEX: {d:.1e}, {d2:.1e}",
    )


# ----------------------------------------------------------------- U2
mp.dps = 80


def thiele_fit(zs, fs):
    n = len(zs)
    g_prev = list(fs)
    coeffs = [g_prev[0]] + [mpc(0)] * (n - 1)
    for pp in range(1, n):
        g_curr = [mpc(0)] * n
        for k in range(pp, n):
            g_curr[k] = (g_prev[pp - 1] - g_prev[k]) / (
                (zs[k] - zs[pp - 1]) * g_prev[k]
            )
        coeffs[pp] = g_curr[pp]
        g_prev = g_curr
    return coeffs


def thiele_eval(zs, coeffs, zz):
    acc = mpc(1)
    for pp in range(len(coeffs) - 1, 0, -1):
        acc = 1 + coeffs[pp] * (zz - zs[pp - 1]) / acc
    return coeffs[0] / acc


def sigma_closed_form(z, states):
    """claim 2 of gw_self_energy_sympy.py: occupied side by Re(z - eps_q) > 0."""
    tot = mpc(0)
    for eps_q, s_q, om_q in states:
        on_occupied_side = (z - eps_q).real > 0
        tot += (
            s_q / (z - eps_q + om_q) if on_occupied_side else s_q / (z - eps_q - om_q)
        )
    return tot


def continued(states, ef, e_real):
    nodes = [mpc(ef, mpf(k) / 3) for k in range(1, 5)]
    vals = [sigma_closed_form(zz, states) for zz in nodes]
    return thiele_eval(nodes, thiele_fit(nodes, vals), mpc(e_real))


def part_u2():
    # (eps_q, s_q, Omega_q): one occupied and one virtual state per spin.
    spin_a = [(mpf(-1), mpf("0.3"), mpf("0.7")), (mpf("0.2"), mpf("0.25"), mpf("0.9"))]
    spin_b = [
        (mpf("-0.3"), mpf("0.2"), mpf("0.8")),
        (mpf("0.5"), mpf("0.15"), mpf("0.6")),
    ]
    e_real = mpf("-0.35")

    def truth(states):
        tot = mpc(0)
        for k, (eps_q, s_q, om_q) in enumerate(states):
            tot += (
                s_q / (e_real - eps_q + om_q)
                if k == 0
                else s_q / (e_real - eps_q - om_q)
            )
        return tot

    ok_all = True
    for name, states, gap in (
        ("alpha", spin_a, (mpf(-1), mpf("0.2"))),
        ("beta", spin_b, (mpf("-0.3"), mpf("0.5"))),
    ):
        ref = truth(states)
        for ef in (gap[0] + mpf("0.01"), (gap[0] + gap[1]) / 2, gap[1] - mpf("0.01")):
            err = abs(continued(states, ef, e_real) - ref)
            ok_all = ok_all and err < mpf("1e-40")
    check(
        "U2",
        ok_all,
        "any eps_F inside the spin's own gap continues to the same Sigma_c(E) (< 1e-40)",
    )
    ef_alpha_mid = (
        spin_a[0][0] + spin_a[1][0]
    ) / 2  # inside alpha's gap, outside beta's
    err_out = abs(continued(spin_b, ef_alpha_mid, e_real) - truth(spin_b))
    check(
        "U2",
        err_out > mpf("1e-3"),
        f"negative control: the alpha mid-gap line mis-assigns a beta state, |d| = {float(err_out):.2e}",
    )


# ----------------------------------------------------------------- U3
def pi_spin_freq(b, eps, nocc, u):
    """energy.rs unrestricted chi0 at iu: Pi_s = 2 sum B B^T Delta/(u^2 + Delta^2)."""
    bov = b[:, :nocc, nocc:].reshape(b.shape[0], -1)
    gaps = (eps[None, nocc:] - eps[:nocc, None]).ravel()
    return 2.0 * (bov * (gaps / (u**2 + gaps**2))) @ bov.T


def code_sigma_c(b_s, eps_s, lam0, u0, chans, m, z, nodes, weights):
    """u_sigma.rs per spin: Sigma_c,s(m, z) = -(1/pi) sum_{n in s} sum_xi,eta
    M_s[xi,mn] [eps~(iu)^-1 - 1]_xi,eta M_s[eta,mn] integrated over u, with
    M_s = U^T B_s (U: static spin-summed PDEP modes, a complete basis here)."""
    m_proj = np.einsum("Px,Ppq->xpq", u0, b_s)
    tot = 0.0
    for uk, wk in zip(nodes, weights):
        eps_mat = np.eye(M) + sum(pi_spin_freq(b, e, no, uk) for b, e, no in chans)
        w_d = u0.T @ np.linalg.inv(eps_mat) @ u0 - np.eye(M)
        for n in range(M):
            w_mn = m_proj[:, m, n] @ w_d @ m_proj[:, m, n]
            kern = z - eps_s[n]
            tot += wk * w_mn * kern / (kern**2 + uk**2)
    return -tot / np.pi


def drpa_pole_sigma(chans, spin, m, z):
    """Independent route: spin-orbital direct RPA on real frequencies,
    Omega^2 from Delta^1/2 (Delta + 2K) Delta^1/2, X+Y = Delta^1/2 Z/sqrt(Omega),
    V_s,mn = sum_{jb tau} (mn|jb)_tau (X+Y)_{jb tau, s}, and claim 2's residues
    Sigma_c(z) = sum_n sum_s V^2 / (z - eps_n +- Omega_s)."""
    bov = np.concatenate(
        [b[:, :no, no:].reshape(b.shape[0], -1) for b, _, no in chans], axis=1
    )
    dlt = np.concatenate([(e[None, no:] - e[:no, None]).ravel() for _, e, no in chans])
    kmat = bov.T @ bov
    sq = np.sqrt(dlt)
    om2, zz = np.linalg.eigh(sq[:, None] * (np.diag(dlt) + 2.0 * kmat) * sq[None, :])
    om = np.sqrt(om2)
    xpy = sq[:, None] * zz / np.sqrt(om)[None, :]
    b_s, eps_s, nocc_s = chans[spin]
    tot = 0.0
    for n in range(M):
        v = b_s[:, m, n] @ bov @ xpy
        sign = 1.0 if n < nocc_s else -1.0
        tot += np.sum(v**2 / (z - eps_s[n] + sign * om))
    return tot


def gl_half_line(n, u0):
    x, w = np.polynomial.legendre.leggauss(n)
    return u0 * (1 + x) / (1 - x), w * 2 * u0 / (1 - x) ** 2


def part_u3():
    rng = np.random.default_rng(20261119)
    nocc_a, nocc_b = 2, 1
    phi_a, eps_a = spin_channel(rng, nocc_a)
    phi_b, eps_b = spin_channel(rng, nocc_b)
    lmat = 0.35 * rng.standard_normal((M, M))
    b_a, b_b = build_b(lmat, phi_a), build_b(lmat, phi_b)
    chans = [(b_a, eps_a, nocc_a), (b_b, eps_b, nocc_b)]
    lam0, u0 = np.linalg.eigh(
        np.eye(M) + sum(pi_spin_freq(b, e, no, 0.0) for b, e, no in chans)
    )
    nodes, weights = gl_half_line(400, 0.5)
    worst = 0.0
    for spin, (b_s, eps_s, nocc_s) in enumerate(chans):
        ef = 0.5 * (eps_s[nocc_s - 1] + eps_s[nocc_s])  # this spin's mid-gap
        for m in range(M):
            for y in (0.3, 1.1):
                z = ef + 1j * y
                c = code_sigma_c(b_s, eps_s, lam0, u0, chans, m, z, nodes, weights)
                worst = max(worst, abs(c - drpa_pole_sigma(chans, spin, m, z)))
    check(
        "U3",
        worst < 1e-10,
        f"per-spin dynamic Sigma_c on z = eps_F,s + iy (code integral) = U-dRPA pole sum: max |d| = {worst:.1e}",
    )
    # Negative control: the other spin's orbital energies in the kernel.
    z = 0.5 * (eps_a[1] + eps_a[2]) + 0.3j
    bad = code_sigma_c(b_a, eps_b, lam0, u0, chans, 0, z, nodes, weights)
    check(
        "U3",
        abs(bad - drpa_pole_sigma(chans, 0, 0, z)) > 1e-4,
        "negative control: the other spin's propagator energies are wrong",
    )

    # Closed-shell limit: alpha = beta equals the closed-shell Sigma_c
    # (prefactor-4 Pi with spatial pairs).
    phi, eps = spin_channel(rng, 2)
    b = build_b(lmat, phi)
    ch2 = [(b, eps, 2), (b, eps, 2)]
    lam_u, uu = np.linalg.eigh(
        np.eye(M) + sum(pi_spin_freq(bb, e, no, 0.0) for bb, e, no in ch2)
    )
    z = 0.5 * (eps[1] + eps[2]) + 0.4j
    d = 0.0
    for m in range(M):
        su = code_sigma_c(b, eps, lam_u, uu, ch2, m, z, nodes, weights)
        sc = 0.0
        m_proj = np.einsum("Px,Ppq->xpq", uu, b)
        for uk, wk in zip(nodes, weights):
            pi_c = 2.0 * pi_spin_freq(b, eps, 2, uk)  # prefactor 4
            w_d = uu.T @ np.linalg.inv(np.eye(M) + pi_c) @ uu - np.eye(M)
            for n in range(M):
                w_mn = m_proj[:, m, n] @ w_d @ m_proj[:, m, n]
                sc += wk * w_mn * (z - eps[n]) / ((z - eps[n]) ** 2 + uk**2)
        d = max(d, abs(su + sc / np.pi))
    check(
        "U3",
        d < 1e-12,
        f"alpha = beta: per-spin dynamic Sigma_c = closed-shell Sigma_c: max |d| = {d:.1e}",
    )


def main():
    part_u1()
    part_u2()
    part_u3()
    print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all identities hold")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
