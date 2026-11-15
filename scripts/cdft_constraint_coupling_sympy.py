#!/usr/bin/env python3
"""Proofs: the cDFT constraint (Fock shift, envelope theorem) and the Wu-Van Voorhis coupling.

Derivation: crates/ferric-dft/theory.md, section "Constrained DFT and the
diabatic coupling". The code transcribed here:

  * ferric-scf/src/cdft_driver.rs, the Fock modifier of the inner UHF solve:
      Total:    F_a += lam W,  F_b += lam W,  N = Tr[W (D_a + D_b)]
      SpinDiff: F_a += lam W,  F_b -= lam W,  N = Tr[W (D_a - D_b)]
    (ferric-dft/src/cdft.rs::population), with V(lam) = E + lam (N - target)
    and E the bare energy at the constrained density.
  * ferric-scf/src/cdft_coupling.rs: biorth_pairing (SVD of
    M = C_a^T S C_b), cofactor_sum (prefix/suffix products, no division),
    cross_one_body (sum over spins of the cofactor sum times the other
    spin's determinant), own_population, coupling_hab:
      F_X   = E_X + lam_X N_X
      H_raw = 1/2 [(F_b S - lam_b <a|W_b|b>) + (F_a S - lam_a <a|W_a|b>)]
      H_ab  = (H_raw - 1/2 (E_a + E_b) S) / (1 - S^2)

Checked (the book cites the labels in parentheses):
  1  (claim cofactor) cross_one_body equals the Fock-space matrix element
     <Psi_a| sum_pq O_pq a+_p a_q |Psi_b> computed by brute force over
     occupation strings (Jordan-Wigner signs) in a Lowdin-orthonormalized
     basis, for two UHF determinants with different MO sets; for the
     generic case, for one exactly zero singular value, and for two. The
     code's value is in the SVD-rotated determinants, so it is compared after
     the phase det(U) det(V) of each spin; the determinant overlap likewise.
  2  (claim lowdin) symbolically, the off-diagonal element of
     S^{-1/2} H S^{-1/2} for H = [[E_a, h], [h, E_b]] and S = [[1, s], [s, 1]]
     is (h - (E_a + E_b) s / 2) / (1 - s^2).
  3  (claim invariance) H_ab of coupling_hab is unchanged when both weight
     operators are shifted, W -> W - c S (target N -> N - c N_e); the form
     with E_X in place of F_X changes.
  4  (claim envelope) for a model UHF energy (random h and a small
     8-fold-symmetric two-electron tensor) solved self-consistently with the
     code's Fock shift: dV/dlam = N - target (Hellmann-Feynman), and the
     bare energy satisfies dE/dN = -lam, by central differences in lam, for
     Total and SpinDiff constraints. This is an independent route: it uses
     only the converged SCF, not the shift formula's derivation.

Mutations (the runner substitutes the code line once; each exits 1):
  the Fock shift sign F_a += lam W -> F_a -= lam W        -> 4 fails
  H_raw with E_b in place of F_b (the non-invariant form) -> 3 fails
  cross_one_body pairs each spin's cofactor sum with its own determinant
      instead of the other spin's                          -> 1 fails

Run: python3 scripts/cdft_constraint_coupling_sympy.py
"""

import itertools
import sys

import numpy as np
import sympy as sp

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# ================================================================ code transcriptions


def biorth_pairing(c_occ_a, c_occ_b, s):
    m = c_occ_a.T @ s @ c_occ_b
    u, sig, vt = np.linalg.svd(m)
    return {
        "det_m": float(np.prod(sig)),
        "s_vals": sig,
        "ct_a": c_occ_a @ u,
        "ct_b": c_occ_b @ vt.T,
        "phase": float(np.linalg.det(u) * np.linalg.det(vt)),
    }


def cofactor_sum(s_vals, d):
    n = len(s_vals)
    suffix = [1.0] * (n + 1)
    for i in reversed(range(n)):
        suffix[i] = suffix[i + 1] * s_vals[i]
    prefix, total = 1.0, 0.0
    for i in range(n):
        total += d[i] * prefix * suffix[i + 1]
        prefix *= s_vals[i]
    return total


def paired_diagonal(op_ao, p):
    return [p["ct_a"][:, i] @ op_ao @ p["ct_b"][:, i] for i in range(len(p["s_vals"]))]


def cross_one_body(op_ao, pa, pb):
    alpha = cofactor_sum(pa["s_vals"], paired_diagonal(op_ao, pa))
    beta = cofactor_sum(pb["s_vals"], paired_diagonal(op_ao, pb))
    return alpha * pb["det_m"] + beta * pa["det_m"]


def own_population(w, c_occ_a, c_occ_b):
    return float(np.trace(c_occ_a.T @ w @ c_occ_a) + np.trace(c_occ_b.T @ w @ c_occ_b))


def coupling_hab(st_a, st_b, s, energy_form=False):
    pa = biorth_pairing(st_a["ca"], st_b["ca"], s)
    pb = biorth_pairing(st_a["cb"], st_b["cb"], s)
    s_ab = pa["det_m"] * pb["det_m"]
    w_a_elem = cross_one_body(st_a["w"], pa, pb)
    w_b_elem = cross_one_body(st_b["w"], pa, pb)
    n_a = own_population(st_a["w"], st_a["ca"], st_a["cb"])
    n_b = own_population(st_b["w"], st_b["ca"], st_b["cb"])
    e_a, e_b = st_a["e"], st_b["e"]
    f_a = e_a + st_a["lam"] * n_a
    f_b = e_b + st_b["lam"] * n_b
    if energy_form:
        f_a, f_b = e_a, e_b
    h_raw = 0.5 * (
        (f_b * s_ab - st_b["lam"] * w_b_elem) + (f_a * s_ab - st_a["lam"] * w_a_elem)
    )
    denom = 1.0 - s_ab * s_ab
    if abs(denom) < 1e-10:
        return e_a, s_ab
    return (h_raw - 0.5 * (e_a + e_b) * s_ab) / denom, s_ab


def fock_shift(f_a, f_b, lam, w, channel):
    """The Fock modifier of cdft_driver.rs (one constraint)."""
    lw = lam * w
    f_a = f_a.copy()
    f_b = f_b.copy()
    if channel == "total":
        f_a, f_b = f_a + lw, f_b + lw
    else:
        f_a += lw
        f_b -= lw
    return f_a, f_b


def population(w, d_a, d_b, channel):
    return float(np.sum(w * (d_a + d_b if channel == "total" else d_a - d_b)))


# ================================================================ helpers


def sym_orth(s):
    e, v = np.linalg.eigh(s)
    return v @ np.diag(e**-0.5) @ v.T, v @ np.diag(e**0.5) @ v.T


def random_spd(rng, n):
    a = rng.standard_normal((n, n)) * 0.3
    return np.eye(n) + 0.5 * (a + a.T) @ (a + a.T).T / n


def s_orthonormal_mos(rng, s_half_inv, n):
    """Columns orthonormal in the S metric: C = S^{-1/2} Q."""
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return s_half_inv @ q


# ================================================================ 1: cofactor form vs Fock space


def apply_excitation(det, p, q):
    """a+_p a_q |det>: (sign, det) or None."""
    if not (det >> q) & 1:
        return None
    sign = -1 if bin(det & ((1 << q) - 1)).count("1") % 2 else 1
    det ^= 1 << q
    if (det >> p) & 1:
        return None
    sign *= -1 if bin(det & ((1 << p) - 1)).count("1") % 2 else 1
    return sign, det | (1 << p)


def fock_space_vector(c_ortho_a, c_ortho_b, nbf):
    """Expand a UHF determinant over occupation strings of 2 nbf spin orbitals.

    Spin orbital k < nbf is alpha k, k >= nbf is beta k - nbf; the determinant
    is a+_{phi_1a} ... a+_{phi_na} a+_{phi_1b} ... |0>, so the coefficient of
    the string with alpha set P and beta set Q is det(C_a[P]) det(C_b[Q]).
    """
    vec = {}
    na, nb = c_ortho_a.shape[1], c_ortho_b.shape[1]
    for pa in itertools.combinations(range(nbf), na):
        da = np.linalg.det(c_ortho_a[list(pa), :]) if na else 1.0
        for pb in itertools.combinations(range(nbf), nb):
            db = np.linalg.det(c_ortho_b[list(pb), :]) if nb else 1.0
            bits = sum(1 << k for k in pa) + sum(1 << (nbf + k) for k in pb)
            vec[bits] = vec.get(bits, 0.0) + da * db
    return vec


def fock_space_element(vec_a, vec_b, o_ortho, nbf):
    out = {}
    for det, cf in vec_b.items():
        for spin in (0, 1):
            off = spin * nbf
            for p in range(nbf):
                for q in range(nbf):
                    if o_ortho[p, q] == 0.0:
                        continue
                    res = apply_excitation(det, off + p, off + q)
                    if res:
                        out[res[1]] = out.get(res[1], 0.0) + cf * res[0] * o_ortho[p, q]
    return sum(cf * out.get(d, 0.0) for d, cf in vec_a.items())


def part_1(rng):
    nbf, na, nb = 4, 2, 1
    s = random_spd(rng, nbf)
    s_mhalf, s_half = sym_orth(s)
    o = rng.standard_normal((nbf, nbf))
    o = 0.5 * (o + o.T)
    worst = 0.0
    worst_s = 0.0
    for case in ("generic", "one zero", "two zeros"):
        ca, cb = (
            s_orthonormal_mos(rng, s_mhalf, nbf),
            s_orthonormal_mos(rng, s_mhalf, nbf),
        )
        ca_o_a, ca_o_b = ca[:, :na], ca[:, na : na + nb]
        cb_o_a, cb_o_b = cb[:, :na], cb[:, na : na + nb]
        if case != "generic":
            # make B's second alpha orbital S-orthogonal to both of A's alpha orbitals
            ra = ca[:, na:]  # A's unoccupied alpha space, S-orthonormal
            cb_o_a = np.column_stack([cb_o_a[:, 0], ra[:, 0]])
        if case == "two zeros":
            # and B's beta orbital S-orthogonal to A's beta orbital
            cb_o_b = (ca[:, na + nb :])[:, :1]

        # S-orthonormalize B's occupied sets (QR in the S metric)
        def s_orth_cols(c):
            q, _ = np.linalg.qr(s_half @ c)
            return s_mhalf @ q

        cb_o_a, cb_o_b = s_orth_cols(cb_o_a), s_orth_cols(cb_o_b)
        pa = biorth_pairing(ca_o_a, cb_o_a, s)
        pb = biorth_pairing(ca_o_b, cb_o_b, s)
        code = cross_one_body(o, pa, pb) * pa["phase"] * pb["phase"]
        code_s = pa["det_m"] * pb["det_m"] * pa["phase"] * pb["phase"]
        va = fock_space_vector(s_half @ ca_o_a, s_half @ ca_o_b, nbf)
        vb = fock_space_vector(s_half @ cb_o_a, s_half @ cb_o_b, nbf)
        brute = fock_space_element(va, vb, s_mhalf @ o @ s_mhalf, nbf)
        brute_s = sum(cf * vb.get(d, 0.0) for d, cf in va.items())
        nzero = int(np.sum(pa["s_vals"] < 1e-12) + np.sum(pb["s_vals"] < 1e-12))
        print(
            f"    {case}: zero singular values {nzero}, <a|O|b> brute {brute:+.6e}, code {code:+.6e}"
        )
        err, err_s = abs(code - brute), abs(code_s - brute_s)
        worst = max(worst, err if np.isfinite(err) else np.inf)
        worst_s = max(worst_s, err_s if np.isfinite(err_s) else np.inf)
        if case == "one zero":
            nonzero_one = abs(brute) > 1e-3 and nzero == 1
    check(
        "1. cross_one_body == Fock-space <a|O|b> (generic, one and two zero singular values) (claim cofactor)",
        worst < 1e-12 and nonzero_one,
        f"max {worst:.1e}",
    )
    check(
        "1. det_m(alpha) det_m(beta) == Fock-space <a|b>",
        worst_s < 1e-12,
        f"max {worst_s:.1e}",
    )


# ================================================================ 2: Lowdin orthogonalization


def part_2():
    ea, eb, h, s = sp.symbols("E_a E_b h s", real=True)
    smat = sp.Matrix([[1, s], [s, 1]])
    hmat = sp.Matrix([[ea, h], [h, eb]])
    a, b = 1 / sp.sqrt(1 + s), 1 / sp.sqrt(1 - s)
    s_mhalf = sp.Rational(1, 2) * sp.Matrix([[a + b, a - b], [a - b, a + b]])
    ok_inv = sp.simplify(s_mhalf * smat * s_mhalf - sp.eye(2)) == sp.zeros(2, 2)
    off = sp.simplify(
        (s_mhalf * hmat * s_mhalf)[0, 1] - (h - (ea + eb) * s / 2) / (1 - s**2)
    )
    check(
        "2. offdiag S^-1/2 H S^-1/2 == (h - (E_a + E_b) s/2) / (1 - s^2) (claim lowdin)",
        ok_inv and off == 0,
    )


# ================================================================ 3: offset invariance


def part_3(rng):
    nbf, na, nb = 5, 3, 2
    s = random_spd(rng, nbf)
    s_mhalf, _ = sym_orth(s)
    states = []
    for _ in range(2):
        c = s_orthonormal_mos(rng, s_mhalf, nbf)
        c2 = s_orthonormal_mos(rng, s_mhalf, nbf)
        w = rng.standard_normal((nbf, nbf))
        states.append(
            {
                "ca": c[:, :na],
                "cb": c2[:, :nb],
                "w": 0.5 * (w + w.T),
                "e": float(rng.uniform(-10, -9)),
                "lam": float(rng.uniform(-1, 1)),
            }
        )
    shifted = [dict(st, w=st["w"] - 0.7 * s) for st in states]
    h0, s0 = coupling_hab(states[0], states[1], s)
    h1, _ = coupling_hab(shifted[0], shifted[1], s)
    he0, _ = coupling_hab(states[0], states[1], s, energy_form=True)
    he1, _ = coupling_hab(shifted[0], shifted[1], s, energy_form=True)
    err = abs(h1 - h0)
    check(
        "3. H_ab invariant under W -> W - c S (F_X = E_X + lam_X N_X form) (claim invariance)",
        err < 1e-12 * max(1.0, abs(h0)) and abs(s0) > 1e-3,
        f"|dH| {err:.1e}, S_ab {s0:.2e}",
    )
    check(
        "3. control: the E_X form is not invariant",
        abs(he1 - he0) > 1e-6,
        f"|dH| {abs(he1 - he0):.1e}",
    )


# ================================================================ 4: Fock shift and envelope theorem


def model_uhf(rng, nbf):
    h = rng.standard_normal((nbf, nbf))
    h = 0.5 * (h + h.T) + np.diag(np.arange(nbf, dtype=float) * 1.5)
    g = rng.standard_normal((nbf, nbf, nbf, nbf)) * 0.05
    g = g + g.transpose(1, 0, 2, 3) + g.transpose(0, 1, 3, 2) + g.transpose(1, 0, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    w = np.zeros((nbf, nbf))
    w[:2, :2] = np.array([[1.0, 0.2], [0.2, 0.6]])
    return h, g, w


def energy(h, g, d_a, d_b):
    dt = d_a + d_b
    j = np.einsum("pqrs,rs->pq", g, dt)
    ka = np.einsum("prqs,rs->pq", g, d_a)
    kb = np.einsum("prqs,rs->pq", g, d_b)
    return float(
        np.sum(h * dt)
        + 0.5 * np.sum(dt * j)
        - 0.5 * (np.sum(d_a * ka) + np.sum(d_b * kb))
    )


def scf(h, g, w, lam, channel, na, nb, d0):
    d_a, d_b = d0
    for _ in range(2000):
        dt = d_a + d_b
        j = np.einsum("pqrs,rs->pq", g, dt)
        f_a = h + j - np.einsum("prqs,rs->pq", g, d_a)
        f_b = h + j - np.einsum("prqs,rs->pq", g, d_b)
        f_a, f_b = fock_shift(f_a, f_b, lam, w, channel)
        ea, va = np.linalg.eigh(f_a)
        eb, vb = np.linalg.eigh(f_b)
        na_new = va[:, :na] @ va[:, :na].T
        nb_new = vb[:, :nb] @ vb[:, :nb].T
        delta = max(np.max(np.abs(na_new - d_a)), np.max(np.abs(nb_new - d_b)))
        d_a, d_b = 0.5 * (d_a + na_new), 0.5 * (d_b + nb_new)
        if delta < 1e-14:
            return na_new, nb_new
    raise RuntimeError("model SCF did not converge")


def part_4(rng):
    nbf, na, nb = 5, 2, 1
    h, g, w = model_uhf(rng, nbf)
    eye = np.eye(nbf)
    d0 = (eye[:, :na] @ eye[:, :na].T, eye[:, :nb] @ eye[:, :nb].T)
    hstep = 1e-4
    for channel in ("total", "spindiff"):
        lam = -0.4
        sol = {}
        for k in (-1, 0, 1):
            d_a, d_b = scf(h, g, w, lam + k * hstep, channel, na, nb, d0)
            sol[k] = (energy(h, g, d_a, d_b), population(w, d_a, d_b, channel))
        e0, n0 = sol[0]
        target = n0 - 0.05
        v = {k: sol[k][0] + (lam + k * hstep) * (sol[k][1] - target) for k in sol}
        dv = (v[1] - v[-1]) / (2 * hstep)
        hf_err = abs(dv - (n0 - target))
        de_dn = (sol[1][0] - sol[-1][0]) / (sol[1][1] - sol[-1][1])
        env_err = abs(de_dn + lam)
        dn = abs(sol[1][1] - sol[-1][1]) / (2 * hstep)
        check(
            f"4. {channel}: dV/dlam == N - target (Hellmann-Feynman)",
            hf_err < 1e-7,
            f"|err| {hf_err:.1e}",
        )
        check(
            f"4. {channel}: dE/dN == -lam with the code's Fock shift (claim envelope)",
            env_err < 1e-5 and dn > 1e-3,
            f"dE/dN {de_dn:+.6f}, lam {lam:+.3f}, |dN/dlam| {dn:.2e}",
        )


def main():
    rng = np.random.default_rng(20261115)
    part_1(rng)
    part_2()
    part_3(rng)
    part_4(rng)
    print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all identities hold")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
