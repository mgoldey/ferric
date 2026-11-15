#!/usr/bin/env python3
"""Proofs: ferric-tddft's spin-adapted response matrices, Casida reduction and oscillator strengths.

Derivation: crates/ferric-tddft/theory.md (sections 1-4, 8-11). The code
transcribed here (index ia = i * nvir + a, chemist integrals, real orbitals):

  * tddft.rs build_a_matrix / build_b_matrix with coulomb_factor (c_J = 2
    singlet, 0 triplet) and ExchangeMix::long_range (dc = c_lr - c_sr):
      A = Delta + c_J (ia|jb) - c_sr (ij|ab) - dc (ij|ab)_w + K
      B = c_J (ia|jb) - c_sr (ib|ja) - dc (ib|ja)_w + K
    (subtract_exchange_a reads (ij|ab) from B_oo^T B_vv at (i*nocc+j, a*nvir+b);
    subtract_exchange_b reads (ib|aj) = (ib|ja) from B_ov^T B_ov at (ib, ja))
  * add_fxc_block / ferric_dft::lr_kernel: K = (ia| f_aa + s f_ab |jb),
    s = +1 singlet, -1 triplet
  * solve_tda (eigh of A); solve_casida: T = (A-B)^{1/2}, M = T (A+B) T,
    M Z = W^2 Z, (X+Y) = W^{-1/2} T Z
  * oscillator_strengths: <0|r|n> = sqrt2 sum_ia (X+Y)_ia <i|r|a>,
    f = 2/3 W |<0|r|n>|^2; triplets report 0 (transition_properties)
  * unrestricted.rs dense_matrices / fill_blocks / transition_properties:
    per-spin build_a/b with c_J = 1, cross-spin Coulomb B_ov^a^T B_ov^b in A
    and B, the spin-resolved kernel, no sqrt2 in the transition dipole

Checked:
  A1  (symbolic) U = [[1,1],[1,-1]]/sqrt2 on the spin index block-diagonalizes
      the closed-shell spin-orbital A and B into A^S = A_aa + A_ab (c_J = 2,
      kernel f_aa + f_ab) and A^T = A_aa - A_ab (c_J = 0, kernel f_aa - f_ab).
  A2  (symbolic) for a spin-symmetric e(rho_a, rho_b), 2 d^2/drho^2 of
      e(rho/2, rho/2) = f_aa + f_ab: the singlet kernel is twice the
      unpolarized second derivative (PySCF get_ab's factor 2 on frr).
  A3  (symbolic, scalar case) the eigenvalues of [[a, b], [-b, -a]] are
      +-sqrt((a-b)(a+b)), and X+Y = W^{-1/2} sqrt(a-b) z with unit z gives
      X^2 - Y^2 = 1.
  A4  (symbolic) the range-separated exchange c_sr erfc + c_lr erf =
      c_sr * 1 + (c_lr - c_sr) erf (theory.md section 8).
  B   Numerical, random closed-shell model (2 occ, 3 vir, RI tensors for
      Coulomb and erf, a random grid kernel f_aa, f_ab):
      B1  the spin-orbital A and B built directly from the textbook
          definition (spin-orbital integrals with spin deltas) equal, after U,
          blockdiag(code singlet, code triplet), for HF, a global hybrid and a
          range-separated hybrid.
      B2  CIS by brute force: for an HF model whose Fock matrix is diagonal,
          <Phi_i^a| H - E0 |Phi_j^b> over determinants a+_a a_i |Phi0>
          (Jordan-Wigner signs, 10 spin orbitals) equals the spin-orbital A
          elementwise; Brillouin <Phi0|H|Phi_i^a> = 0; the TDA singlet
          transition moment <Phi0|r|Psi_n> equals sqrt2 sum X r_ia.
      B3  TDA: the code's A eigenvalues are the X-block roots of the full
          problem with B = 0.
      B4  Casida: the positive eigenvalues of [[A, B], [-B, -A]] equal the
          code's W; the full eigenvector normalized to X.X - Y.Y = 1 gives the
          code's X+Y (up to sign); spin-orbital RPA roots = singlet union
          triplet; sum rule sum_n f_n = 4/3 sum_x r_x.(A-B).r_x (singlets,
          all roots), which fixes both the W^{-1/2} and the sqrt2.
      B5  Unrestricted (2 alpha, 1 beta occupied, different alpha and beta
          orbitals, spin-dependent kernel): the code's dense A and B equal the
          spin-orbital construction; CIS by brute force (spin-dependent one-
          body term making both Fock matrices diagonal) equals A elementwise;
          for C_a = C_b the unrestricted roots are the restricted singlet union
          triplet and the transition moment reduces to the sqrt2 form.
  D   <S^2> of an unrestricted root (theory.md section 11), against an
      independent Fock-space construction <Psi|S^2|Psi> = M^2 + M +
      |S_+ Psi|^2 with S_+ = sum O_pq a+_{p alpha} a_{q beta} and Jordan-Wigner
      signs, 4 spatial orbitals per spin, (n_alpha, n_beta) = (2,1), (3,1),
      (2,2), a general O = C_a^T S C_b (MOs spanning different AO subspaces, so
      O^T O != 1) and an orthogonal O:
      D1  SpinOverlap::reference = the brute force for Phi (claim s2ref).
      D2  SpinOverlap::excited = the brute force for 24 random X (claim s2).
      D3  O = 1: X_a = X_b gives 0, X_a = -X_b gives 2 (claim s2).
      D4  casida_x = the X block of the eigenvector of [[A, B], [-B, -A]]
          normalized to X.X - Y.Y = 1 (claim casidax).
  C   PySCF, run only with DERIV_CROSSCHECK=1 (skipped with a message when
      it is not importable): H2O/STO-3G
      RHF TDA and TDHF singlets and triplets, and H2O+/STO-3G UHF TDA and
      TDHF, built with the code's formulas from exact MO integrals: every
      excitation energy and oscillator strength equals PySCF tdscf's.

Mutations (the runner substitutes the code line once; each exits 1):
  coulomb_factor's singlet value 2 -> 1 -> B1 fails
  fxc_ov_block's triplet sign s = -1 -> +1 -> B1 fails
  subtract_exchange_b reads (ia|jb) instead of (ib|ja) (identity transpose)
      -> B1 fails
  solve_casida scales X+Y by 1/W instead of W^{-1/2} -> B4 fails
  oscillator_strengths drops the singlet sqrt2 -> B2 fails
  u_dense_matrices drops the cross-spin Coulomb block of A -> B5 fails
  SpinOverlap::excited drops the 2 on the cross-spin term -> D2 fails
  SpinOverlap::reference drops the occupied-occupied overlap term -> D1 fails
  casida_x uses (X+Y) - (A+B)(X+Y)/W (the Y block) -> D4 fails

Run: python3 scripts/tddft_response_sympy.py
"""

import itertools
import os
import sys

import numpy as np
import sympy as sp

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


# ================================================================ code transcriptions


def coulomb_factor(spin):
    """tddft.rs coulomb_factor."""
    return 2.0 if spin == "singlet" else 0.0


def long_range(mix):
    """ExchangeMix::long_range: dc = c_lr - c_sr when omega != 0."""
    c_sr, c_lr, omega = mix
    return (c_lr - c_sr) if omega != 0.0 else None


def subtract_exchange_a(a, t, nocc, nvir, coef):
    k = (t["b_oo"].T @ t["b_vv"]).reshape(nocc, nocc, nvir, nvir)  # [i, j, a, b]
    a -= coef * k.transpose(0, 2, 1, 3).reshape(nocc * nvir, nocc * nvir)


def subtract_exchange_b(b, k_ov, nocc, nvir, coef):
    kov4 = k_ov.reshape(nocc, nvir, nocc, nvir)  # [i, b, j, a] = (ib|ja)
    b -= coef * kov4.transpose(0, 3, 2, 1).reshape(nocc * nvir, nocc * nvir)


def build_a_matrix(eps, nocc, nvir, ri, mix, c_j):
    dim = nocc * nvir
    a = np.diag([eps[nocc + x % nvir] - eps[x // nvir] for x in range(dim)])
    if c_j != 0.0:
        a = a + c_j * ri["b_ov"].T @ ri["b_ov"]
    if mix[0] != 0.0:
        subtract_exchange_a(a, ri, nocc, nvir, mix[0])
    dc = long_range(mix)
    if ri.get("lr") is not None and dc is not None:
        subtract_exchange_a(a, ri["lr"], nocc, nvir, dc)
    return a


def build_b_matrix(nocc, nvir, ri, mix, c_j):
    dim = nocc * nvir
    b = np.zeros((dim, dim))
    coulomb = ri["b_ov"].T @ ri["b_ov"]
    if c_j != 0.0:
        b += c_j * coulomb
    if mix[0] != 0.0:
        subtract_exchange_b(b, coulomb, nocc, nvir, mix[0])
    dc = long_range(mix)
    if ri.get("lr") is not None and dc is not None:
        lr = ri["lr"]
        subtract_exchange_b(b, lr["b_ov"].T @ lr["b_ov"], nocc, nvir, dc)
    return b


def fxc_ov_block(grid, spin):
    """lr_kernel singlet/triplet ov block: (ia| f_aa + s f_ab |jb)."""
    if grid is None:
        return 0.0
    s = 1.0 if spin == "singlet" else -1.0
    ov = grid["ov"]  # (ngrid, nocc*nvir) products phi_i phi_a
    return ov.T @ ((grid["w"] * (grid["faa"] + s * grid["fab"]))[:, None] * ov)


def solve_tda(a):
    return np.linalg.eigh(a)


def solve_casida(a, b):
    lam, u = np.linalg.eigh(a - b)
    if np.any(lam < 0):
        raise AssertionError("A-B not positive definite in the model")
    t_half = (u * np.sqrt(lam)) @ u.T
    w2, z = np.linalg.eigh(t_half @ (a + b) @ t_half)
    w = np.sqrt(w2)
    xpy = t_half @ z / np.sqrt(w)
    return w, xpy


def oscillator_strengths(r_ov, w, amps):
    """r_ov: (3, nocc*nvir) <i|r|a>; amps: (dim, nroots)."""
    tdip = np.sqrt(2.0) * np.einsum("xk,kn->nx", r_ov, amps)
    return (2.0 / 3.0) * w * np.sum(tdip**2, axis=1), tdip


def u_dense_matrices(eps, nocc, nvir, ri, mix, kernel, casida):
    """unrestricted.rs dense_matrices + fill_blocks (kernel: full (d, d) block)."""
    da = nocc[0] * nvir[0]
    n = da + nocc[1] * nvir[1]
    cross = ri[0]["b_ov"].T @ ri[1]["b_ov"]
    a = np.zeros((n, n))
    a[:da, :da] = build_a_matrix(eps[0], nocc[0], nvir[0], ri[0], mix, 1.0)
    a[da:, da:] = build_a_matrix(eps[1], nocc[1], nvir[1], ri[1], mix, 1.0)
    a[:da, da:] = cross
    a[da:, :da] = cross.T
    b = None
    if casida:
        b = np.zeros((n, n))
        b[:da, :da] = build_b_matrix(nocc[0], nvir[0], ri[0], mix, 1.0)
        b[da:, da:] = build_b_matrix(nocc[1], nvir[1], ri[1], mix, 1.0)
        b[:da, da:] = cross
        b[da:, :da] = cross.T
    if kernel is not None:
        a = a + kernel
        if b is not None:
            b = b + kernel
    return a, b


def u_transition(r_ov_spins, w, amps, da):
    """unrestricted.rs transition_properties: no sqrt2."""
    tdip = np.einsum("xk,kn->nx", r_ov_spins[0], amps[:da]) + np.einsum(
        "xk,kn->nx", r_ov_spins[1], amps[da:]
    )
    return (2.0 / 3.0) * w * np.sum(tdip**2, axis=1), tdip


# ================================================================ A: symbolic


def part_a():
    dlt, jj, kx, kb, faa, fab, c = sp.symbols("Delta J K_A K_B f_aa f_ab c")
    a_aa, a_ab = dlt + jj - c * kx + faa, jj + fab
    b_aa, b_ab = jj - c * kb + faa, jj + fab
    u = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    for lbl, (m_aa, m_ab), (sing, trip) in (
        (
            "A",
            (a_aa, a_ab),
            (dlt + 2 * jj - c * kx + faa + fab, dlt - c * kx + faa - fab),
        ),
        ("B", (b_aa, b_ab), (2 * jj - c * kb + faa + fab, -c * kb + faa - fab)),
    ):
        rot = sp.simplify(u.T * sp.Matrix([[m_aa, m_ab], [m_ab, m_aa]]) * u)
        ok = sp.simplify(rot - sp.diag(sing, trip)) == sp.zeros(2, 2)
        check(
            f"A1 spin adaptation of {lbl}: U^T {lbl}_so U = diag(singlet, triplet) (claim A1)",
            ok,
        )

    ra, rb, rho = sp.symbols("rho_a rho_b rho", positive=True)
    coeffs = sp.symbols("e0:10")
    # generic spin-symmetric cubic: symmetric in (rho_a, rho_b)
    mons = [1, ra + rb, ra**2 + rb**2, ra * rb, ra**3 + rb**3, ra**2 * rb + ra * rb**2]
    e = sum(cf * m for cf, m in zip(coeffs, mons))
    f_aa = sp.diff(e, ra, 2)
    f_ab = sp.diff(e, ra, rb)
    e_unpol = e.subs({ra: rho / 2, rb: rho / 2}, simultaneous=True)
    lhs = 2 * sp.diff(e_unpol, rho, 2)
    rhs = (f_aa + f_ab).subs({ra: rho / 2, rb: rho / 2}, simultaneous=True)
    check(
        "A2 2 d2e(rho/2, rho/2)/drho2 = f_aa + f_ab (singlet kernel normalization) (claim A2)",
        sp.expand(lhs - rhs) == 0,
    )

    a, b, lam, z = sp.symbols("a b lambda z", positive=True)
    poly = (lam * sp.eye(2) - sp.Matrix([[a, b], [-b, -a]])).det()
    check(
        "A3 eig [[a,b],[-b,-a]] = +-sqrt((a-b)(a+b)) (claim A3)",
        sp.expand(poly - (lam**2 - (a - b) * (a + b))) == 0,
    )
    w = sp.sqrt((a - b) * (a + b))
    xpy = sp.sqrt(a - b) * z / sp.sqrt(w)
    xmy = (a + b) * xpy / w
    check(
        "A3 (X+Y) = W^{-1/2} T z, |z| = 1  =>  X^2 - Y^2 = 1",
        sp.simplify((xpy * xmy).subs(z, 1) - 1) == 0,
    )

    x, csr, clr = sp.symbols("x c_sr c_lr", positive=True)
    check(
        "A4 c_sr erfc + c_lr erf = c_sr + (c_lr - c_sr) erf (claim A4)",
        sp.simplify(
            (
                csr * sp.erfc(x) + clr * sp.erf(x) - (csr + (clr - csr) * sp.erf(x))
            ).rewrite(sp.erf)
        )
        == 0,
    )


# ================================================================ model builders


def random_ri(rng, n, naux, scale):
    """B^P_pq symmetric in pq, shape (naux, n, n)."""
    b = rng.standard_normal((naux, n, n)) * scale
    return 0.5 * (b + b.transpose(0, 2, 1))


def ri_blocks(b_full, nocc):
    """The code's (naux, n*m) ov, oo, vv blocks from a full (naux, nmo, nmo) tensor."""
    naux = b_full.shape[0]
    o, v = slice(0, nocc), slice(nocc, None)
    return {
        "b_ov": b_full[:, o, v].reshape(naux, -1).copy(),
        "b_oo": b_full[:, o, o].reshape(naux, -1).copy(),
        "b_vv": b_full[:, v, v].reshape(naux, -1).copy(),
    }


def eri_from(b_full):
    return np.einsum("Ppq,Prs->pqrs", b_full, b_full)


def so_matrices(spin_orbs, occ, vir, eri, eri_w, kern, eps, mix):
    """Spin-orbital A and B from the textbook definition, over spin-conserving pairs.

    spin_orbs[k] = (spin, spatial index into that spin's integral set).
    eri[(s, t)][p, q, r, s'] = (p_s q_s | r_t s'_t); kern[(s, t)] likewise for f_st.
    """
    c_sr, c_lr, omega = mix
    dc = (c_lr - c_sr) if omega else 0.0
    pairs = [
        (i, a)
        for s in (0, 1)
        for i in occ
        if spin_orbs[i][0] == s
        for a in vir
        if spin_orbs[a][0] == s
    ]

    def g(tbl, p, q, r, s):
        sp_, sq, sr, ss = (spin_orbs[k][0] for k in (p, q, r, s))
        if sp_ != sq or sr != ss or tbl is None:
            return 0.0
        return tbl[(sp_, sr)][
            spin_orbs[p][1], spin_orbs[q][1], spin_orbs[r][1], spin_orbs[s][1]
        ]

    n = len(pairs)
    a_m, b_m = np.zeros((n, n)), np.zeros((n, n))
    for x, (i, a) in enumerate(pairs):
        for y, (j, b) in enumerate(pairs):
            diag = (eps[a] - eps[i]) if (i == j and a == b) else 0.0
            a_m[x, y] = (
                diag
                + g(eri, i, a, j, b)
                - c_sr * g(eri, i, j, a, b)
                - dc * g(eri_w, i, j, a, b)
                + g(kern, i, a, j, b)
            )
            b_m[x, y] = (
                g(eri, i, a, b, j)
                - c_sr * g(eri, i, b, a, j)
                - dc * g(eri_w, i, b, a, j)
                + g(kern, i, a, b, j)
            )
    return a_m, b_m, pairs


def grid_kernel(phi_s, w, f):
    """kern[(s,t)][p,q,r,s'] = sum_g w f_st phi_p phi_q phi_r phi_s' (phi_s: (ngrid, nmo))."""
    out = {}
    for s, t in itertools.product((0, 1), repeat=2):
        out[(s, t)] = np.einsum(
            "g,gp,gq,gr,gs->pqrs", w * f[(s, t)], phi_s[s], phi_s[s], phi_s[t], phi_s[t]
        )
    return out


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
    """H = sum h_PQ a+P aQ + 1/2 sum (PQ|RS) a+P a+R aS aQ on a {det: coef} vector.

    The two-body term is applied as a+P aQ a+R aS, so the normal-ordering
    remainder -1/2 sum_Q (PQ|QS) a+P aS is folded into the one-body term.
    """
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


def cis_brute(spin_orbs, occ, pairs, h1, g2):
    nso = len(spin_orbs)
    phi0 = sum(1 << k for k in occ)
    basis = [{phi0: 1.0}]
    for i, a in pairs:
        s, d = apply_excitation(phi0, a, i)
        basis.append({d: float(s)})
    hv = [apply_h(b, h1, g2, nso) for b in basis]

    def dot(u, v):
        return sum(cf * v.get(d, 0.0) for d, cf in u.items())

    h = np.array([[dot(bi, hj) for hj in hv] for bi in basis])
    e0 = h[0, 0]
    return h[1:, 1:] - e0 * np.eye(len(pairs)), h[0, 1:]


# ================================================================ B: numerical


def closed_model(rng):
    nocc, nvir, naux, ngrid = 2, 3, 9, 40
    nmo = nocc + nvir
    eps = np.concatenate(
        [np.sort(rng.uniform(-1.2, -0.6, nocc)), np.sort(rng.uniform(0.4, 1.0, nvir))]
    )
    b_c = random_ri(rng, nmo, naux, 0.12)
    b_w = random_ri(rng, nmo, naux, 0.08)
    phi = rng.standard_normal((ngrid, nmo))
    w = rng.uniform(0.0, 0.02, ngrid)
    faa = rng.uniform(-0.3, 0.3, ngrid)
    fab = rng.uniform(-0.3, 0.3, ngrid)
    ov = np.einsum("gi,ga->gia", phi[:, :nocc], phi[:, nocc:]).reshape(ngrid, -1)
    return {
        "nocc": nocc,
        "nvir": nvir,
        "nmo": nmo,
        "eps": eps,
        "b_c": b_c,
        "b_w": b_w,
        "phi": phi,
        "w": w,
        "f": {(0, 0): faa, (1, 1): faa, (0, 1): fab, (1, 0): fab},
        "grid": {"ov": ov, "w": w, "faa": faa, "fab": fab},
        "r": [0.5 * (m + m.T) for m in rng.standard_normal((3, nmo, nmo))],
    }


def code_closed(m, mix, spin, with_kernel):
    ri = ri_blocks(m["b_c"], m["nocc"])
    ri["lr"] = ri_blocks(m["b_w"], m["nocc"]) if mix[2] else None
    c_j = coulomb_factor(spin)
    a = build_a_matrix(m["eps"], m["nocc"], m["nvir"], ri, mix, c_j)
    b = build_b_matrix(m["nocc"], m["nvir"], ri, mix, c_j)
    if with_kernel:
        k = fxc_ov_block(m["grid"], spin)
        a, b = a + k, b + k
    return a, b


def spin_rotation(d):
    eye = np.eye(d)
    return np.block([[eye, eye], [eye, -eye]]) / np.sqrt(2.0)


def part_b():
    rng = np.random.default_rng(20261114)
    m = closed_model(rng)
    nocc, nmo, d = m["nocc"], m["nmo"], m["nocc"] * m["nvir"]
    spin_orbs = [(s, p) for s in (0, 1) for p in range(nmo)]
    occ = [s * nmo + p for s in (0, 1) for p in range(nocc)]
    vir = [s * nmo + p for s in (0, 1) for p in range(nocc, nmo)]
    eps_so = np.concatenate([m["eps"], m["eps"]])
    eri_sp, eriw_sp = eri_from(m["b_c"]), eri_from(m["b_w"])
    eri = {k: eri_sp for k in itertools.product((0, 1), repeat=2)}
    eri_w = {k: eriw_sp for k in itertools.product((0, 1), repeat=2)}
    kern = grid_kernel([m["phi"], m["phi"]], m["w"], m["f"])
    u = spin_rotation(d)

    cases = (
        ("HF", (1.0, 1.0, 0.0), False),
        ("global hybrid", (0.25, 0.25, 0.0), True),
        ("RSH", (0.167, 1.0, 0.3), True),
    )
    for label, mix, with_k in cases:
        a_so, b_so, _ = so_matrices(
            spin_orbs, occ, vir, eri, eri_w, kern if with_k else None, eps_so, mix
        )
        a_s, b_s = code_closed(m, mix, "singlet", with_k)
        a_t, b_t = code_closed(m, mix, "triplet", with_k)
        z = np.zeros((d, d))
        err = max(
            maxabs(u.T @ a_so @ u - np.block([[a_s, z], [z, a_t]])),
            maxabs(u.T @ b_so @ u - np.block([[b_s, z], [z, b_t]])),
        )
        check(
            f"B1 {label}: U^T (A,B)_so U = blockdiag(code singlet, code triplet) (claim B1)",
            err < 1e-13,
            f"max {err:.1e}",
        )

    # ---- B2: CIS by brute force (HF model with a diagonal Fock matrix).
    h_sp = (
        np.diag(m["eps"])
        - (2 * np.einsum("pqkk->pq", eri_sp[:, :, :nocc, :nocc]))
        + np.einsum("pkkq->pq", eri_sp[:, :nocc, :nocc, :])
    )
    nso = 2 * nmo
    h1 = np.zeros((nso, nso))
    g2 = np.zeros((nso, nso, nso, nso))
    for s in (0, 1):
        h1[s * nmo : (s + 1) * nmo, s * nmo : (s + 1) * nmo] = h_sp
        for t in (0, 1):
            g2[
                s * nmo : (s + 1) * nmo,
                s * nmo : (s + 1) * nmo,
                t * nmo : (t + 1) * nmo,
                t * nmo : (t + 1) * nmo,
            ] = eri_sp
    a_so, b_so, pairs = so_matrices(
        spin_orbs, occ, vir, eri, eri_w, None, eps_so, (1.0, 1.0, 0.0)
    )
    a_cis, h0s = cis_brute(spin_orbs, occ, pairs, h1, g2)
    err = maxabs(a_cis - a_so)
    check(
        "B2 CIS <Phi_i^a|H - E0|Phi_j^b> (Jordan-Wigner) = spin-orbital A (claim B2)",
        err < 1e-12,
        f"max {err:.1e}",
    )
    check(
        "B2 Brillouin <Phi0|H|Phi_i^a> = 0 for the diagonal-Fock model",
        maxabs(h0s) < 1e-12,
        f"max {maxabs(h0s):.1e}",
    )
    a_s, _ = code_closed(m, (1.0, 1.0, 0.0), "singlet", False)
    w_s, x_s = solve_tda(a_s)
    r_ov = np.array([r[:nocc, nocc:].reshape(-1) for r in m["r"]])
    _, tdip = oscillator_strengths(r_ov, w_s, x_s)
    # brute force: <Phi0| r |Psi_n> with Psi_n = sum_{ia s} X_{ias} a+_a a_i Phi0,
    # X_{ia,alpha} = X_{ia,beta} = X^S/sqrt2, and <Phi0|a+_i a_a a+_a a_i|Phi0> = 1.
    x_so = np.vstack([x_s, x_s]) / np.sqrt(2.0)
    r_so = np.array([np.concatenate([r[:nocc, nocc:].reshape(-1)] * 2) for r in m["r"]])
    tdip_bf = np.einsum("xk,kn->nx", r_so, x_so)
    err = maxabs(tdip - tdip_bf)
    check(
        "B2 TDA singlet <0|r|n> = sqrt2 sum X r_ia (spin-orbital sum)",
        err < 1e-13,
        f"max {err:.1e}",
    )

    # ---- B3, B4 on the RSH TDDFT model.
    mix = (0.167, 1.0, 0.3)
    a_s, b_s = code_closed(m, mix, "singlet", True)
    a_t, b_t = code_closed(m, mix, "triplet", True)
    full0 = np.block([[a_s, np.zeros((d, d))], [np.zeros((d, d)), -a_s]])
    ev0 = np.sort(np.linalg.eigvals(full0).real)
    err = maxabs(ev0[d:] - solve_tda(a_s)[0])
    check(
        "B3 TDA: eig(A) = positive roots of the full problem with B = 0 (claim B3)",
        err < 1e-12,
        f"max {err:.1e}",
    )

    w, xpy = solve_casida(a_s, b_s)
    full = np.block([[a_s, b_s], [-b_s, -a_s]])
    ev, vec = np.linalg.eig(full)
    order = np.argsort(ev.real)
    pos = order[d:]
    err = maxabs(ev.real[pos] - w)
    check(
        "B4 Casida W = positive eigenvalues of [[A,B],[-B,-A]] (claim B4)",
        err < 1e-11,
        f"max {err:.1e}",
    )
    xpy_full = []
    for k in pos:
        v = vec[:, k].real
        x, y = v[:d], v[d:]
        v = v / np.sqrt(x @ x - y @ y)
        xpy_full.append(v[:d] + v[d:])
    xpy_full = np.array(xpy_full).T
    sign = np.sign(np.sum(xpy_full * xpy, axis=0))
    err = maxabs(xpy_full * sign - xpy)
    check(
        "B4 code X+Y = full eigenvector normalized to X.X - Y.Y = 1",
        err < 1e-10,
        f"max {err:.1e}",
    )
    xmy = (a_s + b_s) @ xpy / w
    norm = maxabs(np.einsum("kn,kn->n", xpy, xmy) - 1.0)
    check(
        "B4 |X|^2 - |Y|^2 = (X+Y).(X-Y) = 1 for every root",
        norm < 1e-12,
        f"max {norm:.1e}",
    )

    a_so, b_so, _ = so_matrices(spin_orbs, occ, vir, eri, eri_w, kern, eps_so, mix)
    w_so, _ = solve_casida(a_so, b_so)
    w_union = np.sort(np.concatenate([w, solve_casida(a_t, b_t)[0]]))
    err = maxabs(w_so - w_union)
    check(
        "B4 spin-orbital RPA roots = singlet union triplet",
        err < 1e-12,
        f"max {err:.1e}",
    )

    osc, _ = oscillator_strengths(r_ov, w, xpy)
    trk = 4.0 / 3.0 * sum(rv @ (a_s - b_s) @ rv for rv in r_ov)
    err = abs(np.sum(osc) - trk) / abs(trk)
    check(
        "B4 sum rule: sum_n f_n = 4/3 sum_x r_x.(A-B).r_x (claim sumrule)",
        err < 1e-11,
        f"rel {err:.1e}",
    )

    # ---- B5: unrestricted.
    part_b5(rng, m)


def part_b5(rng, m):
    nmo, ngrid = m["nmo"], len(m["w"])
    nocc, nvir = (2, 1), (nmo - 2, nmo - 1)
    nao = nmo
    b_ao_c = random_ri(rng, nao, 9, 0.12)
    b_ao_w = random_ri(rng, nao, 9, 0.08)
    cs = [np.linalg.qr(rng.standard_normal((nao, nmo)))[0] for _ in (0, 1)]
    eps = [
        np.concatenate(
            [
                np.sort(rng.uniform(-1.2, -0.6, nocc[s])),
                np.sort(rng.uniform(0.4, 1.0, nvir[s])),
            ]
        )
        for s in (0, 1)
    ]
    phi_ao = rng.standard_normal((ngrid, nao))
    f = {(0, 0): rng.uniform(-0.3, 0.3, ngrid), (1, 1): rng.uniform(-0.3, 0.3, ngrid)}
    f[(0, 1)] = f[(1, 0)] = rng.uniform(-0.3, 0.3, ngrid)
    mix = (0.167, 1.0, 0.3)

    def run(cs, eps, nocc, nvir, f, mix, with_kernel=True):
        bmo = [np.einsum("Pmn,mp,nq->Ppq", b_ao_c, c, c) for c in cs]
        bmo_w = [np.einsum("Pmn,mp,nq->Ppq", b_ao_w, c, c) for c in cs]
        ri = []
        for s in (0, 1):
            r = ri_blocks(bmo[s], nocc[s])
            r["lr"] = ri_blocks(bmo_w[s], nocc[s]) if mix[2] else None
            ri.append(r)
        phi = [phi_ao @ c for c in cs]
        kern = grid_kernel(phi, m["w"], f)
        kblock = None
        if with_kernel:
            ov = [
                np.einsum(
                    "gi,ga->gia", phi[s][:, : nocc[s]], phi[s][:, nocc[s] :]
                ).reshape(ngrid, -1)
                for s in (0, 1)
            ]
            kblock = np.block(
                [
                    [ov[s].T @ ((m["w"] * f[(s, t)])[:, None] * ov[t]) for t in (0, 1)]
                    for s in (0, 1)
                ]
            )
        a, b = u_dense_matrices(eps, nocc, nvir, ri, mix, kblock, True)
        spin_orbs = [(s, p) for s in (0, 1) for p in range(nmo)]
        occ = [s * nmo + p for s in (0, 1) for p in range(nocc[s])]
        vir = [s * nmo + p for s in (0, 1) for p in range(nocc[s], nmo)]
        eri = {
            (s, t): np.einsum("Ppq,Prs->pqrs", bmo[s], bmo[t])
            for s in (0, 1)
            for t in (0, 1)
        }
        eri_w = {
            (s, t): np.einsum("Ppq,Prs->pqrs", bmo_w[s], bmo_w[t])
            for s in (0, 1)
            for t in (0, 1)
        }
        a_so, b_so, pairs = so_matrices(
            spin_orbs,
            occ,
            vir,
            eri,
            eri_w,
            kern if with_kernel else None,
            np.concatenate(eps),
            mix,
        )
        return a, b, a_so, b_so, (spin_orbs, occ, pairs, eri)

    a, b, a_so, b_so, _ = run(cs, eps, nocc, nvir, f, mix)
    err = max(maxabs(a - a_so), maxabs(b - b_so))
    check(
        "B5 unrestricted dense A, B (RSH + kernel) = spin-orbital construction (claim B5)",
        err < 1e-13,
        f"max {err:.1e}",
    )

    # CIS brute force with a spin-dependent one-body term making both Fock matrices diagonal.
    a, _, a_so, _, (spin_orbs, occ, pairs, eri) = run(
        cs, eps, nocc, nvir, f, (1.0, 1.0, 0.0), with_kernel=False
    )
    nso = 2 * nmo
    h1 = np.zeros((nso, nso))
    g2 = np.zeros((nso,) * 4)
    for s in (0, 1):
        hs = np.diag(eps[s]).copy()
        for t in (0, 1):
            hs -= np.einsum("pqkk->pq", eri[(s, t)][:, :, : nocc[t], : nocc[t]])
        hs += np.einsum("pkkq->pq", eri[(s, s)][:, : nocc[s], : nocc[s], :])
        h1[s * nmo : (s + 1) * nmo, s * nmo : (s + 1) * nmo] = hs
        for t in (0, 1):
            g2[
                s * nmo : (s + 1) * nmo,
                s * nmo : (s + 1) * nmo,
                t * nmo : (t + 1) * nmo,
                t * nmo : (t + 1) * nmo,
            ] = eri[(s, t)]
    a_cis, h0s = cis_brute(spin_orbs, occ, pairs, h1, g2)
    err = max(maxabs(a_cis - a), maxabs(h0s))
    check(
        "B5 UHF CIS (Jordan-Wigner, 2a + 1b electrons) = code unrestricted A",
        err < 1e-12,
        f"max {err:.1e}",
    )

    # closed-shell limit: C_a = C_b, same eps, same occupations, f_bb = f_aa.
    nc = (2, 2)
    nv = (nmo - 2, nmo - 2)
    fc = {(0, 0): f[(0, 0)], (1, 1): f[(0, 0)], (0, 1): f[(0, 1)], (1, 0): f[(0, 1)]}
    a_u, b_u, _, _, _ = run([cs[0], cs[0]], [eps[0], eps[0]], nc, nv, fc, mix)
    mr = {
        "nocc": 2,
        "nvir": nmo - 2,
        "eps": eps[0],
        "b_c": np.einsum("Pmn,mp,nq->Ppq", b_ao_c, cs[0], cs[0]),
        "b_w": np.einsum("Pmn,mp,nq->Ppq", b_ao_w, cs[0], cs[0]),
    }
    phi = phi_ao @ cs[0]
    ov = np.einsum("gi,ga->gia", phi[:, :2], phi[:, 2:]).reshape(ngrid, -1)
    mr["grid"] = {"ov": ov, "w": m["w"], "faa": fc[(0, 0)], "fab": fc[(0, 1)]}
    a_s, b_s = code_closed(mr, mix, "singlet", True)
    a_t, b_t = code_closed(mr, mix, "triplet", True)
    w_u, xpy_u = solve_casida(a_u, b_u)
    w_s, xpy_s = solve_casida(a_s, b_s)
    w_r = np.sort(np.concatenate([w_s, solve_casida(a_t, b_t)[0]]))
    err = maxabs(w_u - w_r)
    check(
        "B5 closed-shell limit: unrestricted roots = singlet union triplet",
        err < 1e-12,
        f"max {err:.1e}",
    )
    r = [0.5 * (x + x.T) for x in rng.standard_normal((3, nmo, nmo))]
    r_ov = np.array([x[:2, 2:].reshape(-1) for x in r])
    f_s, _ = oscillator_strengths(r_ov, w_s, xpy_s)
    f_u, _ = u_transition([r_ov, r_ov], w_u, xpy_u, r_ov.shape[1])
    # singlet roots carry f; triplet roots (X_a = -X_b) carry exactly zero
    f_u_sorted = np.sort(f_u)[::-1][: len(f_s)]
    err = maxabs(np.sort(f_u_sorted) - np.sort(f_s))
    tiny = np.sort(f_u)[: len(f_u) - len(f_s)]
    check(
        "B5 closed-shell limit: unrestricted f (no sqrt2) = restricted singlet f",
        err < 1e-10,
        f"max {err:.1e}",
    )
    check(
        "B5 closed-shell limit: triplet roots have f = 0",
        maxabs(tiny) < 1e-10,
        f"max {maxabs(tiny):.1e}",
    )


# ================================================================ C: PySCF


def exact_ri(eri_ao):
    n = eri_ao.shape[0]
    lam, v = np.linalg.eigh(eri_ao.reshape(n * n, n * n))
    keep = lam > 1e-12 * lam.max()
    return (v[:, keep] * np.sqrt(lam[keep])).T.reshape(-1, n, n)


def part_c():
    if os.environ.get("DERIV_CROSSCHECK") != "1":
        print("[SKIP] C PySCF cross-check (set DERIV_CROSSCHECK=1)")
        return
    try:
        from pyscf import gto, scf, tdscf
    except ImportError:
        print("[SKIP] C PySCF not importable")
        return
    geom = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
    mol = gto.M(atom=geom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-12).run()
    b_ao = exact_ri(mol.intor("int2e"))
    c = mf.mo_coeff
    nocc = int(np.sum(mf.mo_occ > 0))
    nvir = c.shape[1] - nocc
    ri = ri_blocks(np.einsum("Pmn,mp,nq->Ppq", b_ao, c, c), nocc)
    ri["lr"] = None
    r_ao = mol.intor("int1e_r")
    r_ov = np.array([(c[:, :nocc].T @ x @ c[:, nocc:]).reshape(-1) for x in r_ao])
    hf = (1.0, 1.0, 0.0)
    nst = nocc * nvir
    for spin in ("singlet", "triplet"):
        c_j = coulomb_factor(spin)
        a = build_a_matrix(mf.mo_energy, nocc, nvir, ri, hf, c_j)
        b = build_b_matrix(nocc, nvir, ri, hf, c_j)
        for kind, td in (("TDA", tdscf.TDA(mf)), ("TDHF", tdscf.TDHF(mf))):
            td.singlet = spin == "singlet"
            td.conv_tol = 1e-12
            td.nstates = nst
            td.kernel()
            if kind == "TDA":
                w, amp = solve_tda(a)
            else:
                w, amp = solve_casida(a, b)
            err_e = maxabs(w - td.e)
            check(
                f"C H2O/STO-3G {kind} {spin}: excitation energies = PySCF (claim C)",
                err_e < 1e-8,
                f"max {err_e:.1e}",
            )
            if spin == "singlet":
                osc, _ = oscillator_strengths(r_ov, w, amp)
                err_f = maxabs(osc - td.oscillator_strength(gauge="length"))
                check(
                    f"C H2O/STO-3G {kind} singlet: oscillator strengths = PySCF",
                    err_f < 1e-8,
                    f"max {err_f:.1e}",
                )

    molp = gto.M(atom=geom, basis="sto-3g", charge=1, spin=1, verbose=0)
    mfu = scf.UHF(molp).set(conv_tol=1e-12).run()
    b_ao = exact_ri(molp.intor("int2e"))
    r_ao = molp.intor("int1e_r")
    ri, r_ovs, nos, nvs = [], [], [], []
    for s in (0, 1):
        cs = mfu.mo_coeff[s]
        no = int(np.sum(mfu.mo_occ[s] > 0))
        r = ri_blocks(np.einsum("Pmn,mp,nq->Ppq", b_ao, cs, cs), no)
        r["lr"] = None
        ri.append(r)
        r_ovs.append(
            np.array([(cs[:, :no].T @ x @ cs[:, no:]).reshape(-1) for x in r_ao])
        )
        nos.append(no)
        nvs.append(cs.shape[1] - no)
    for kind, td in (("TDA", tdscf.TDA(mfu)), ("TDHF", tdscf.TDHF(mfu))):
        a, b = u_dense_matrices(mfu.mo_energy, nos, nvs, ri, hf, None, kind == "TDHF")
        td.conv_tol = 1e-12
        td.nstates = 8
        td.kernel()
        w, amp = solve_tda(a) if kind == "TDA" else solve_casida(a, b)
        w, amp = w[:8], amp[:, :8]
        err_e = maxabs(w - td.e)
        check(
            f"C H2O+/STO-3G UHF {kind}: excitation energies = PySCF",
            err_e < 1e-8,
            f"max {err_e:.1e}",
        )
        osc, _ = u_transition(r_ovs, w, amp, nos[0] * nvs[0])
        err_f = maxabs(osc - td.oscillator_strength(gauge="length"))
        check(
            f"C H2O+/STO-3G UHF {kind}: oscillator strengths = PySCF",
            err_f < 1e-8,
            f"max {err_f:.1e}",
        )


# ================================================================ D: <S^2>


def s2_m_terms(na, nb):
    m = 0.5 * (na - nb)
    return m * m + m


def s2_reference(o, na, nb):
    """unrestricted.rs SpinOverlap::reference."""
    beta_occ = np.sum(o[:, :nb] ** 2)
    return s2_m_terms(na, nb) + beta_occ - np.sum(o[:na, :nb] ** 2)


def s2_excited(o, na, nb, x):
    """unrestricted.rs SpinOverlap::excited (x = [X_alpha | X_beta], ia = i*nvir + a)."""
    nva, nvb = o.shape[0] - na, o.shape[1] - nb
    norm2 = x @ x
    xa = x[: na * nva].reshape(na, nva) / np.sqrt(norm2)
    xb = x[na * nva :].reshape(nb, nvb) / np.sqrt(norm2)
    p = o.T @ o
    q = o[:, :nb] @ o[:, :nb].T
    r = o[:na, :].T @ o[:na, :]
    wa, wb = np.sum(xa * xa), np.sum(xb * xb)
    beta_occ = np.sum(o[:, :nb] ** 2)

    def ph(xs, m, nocc):
        return np.trace(xs @ m[nocc:, nocc:] @ xs.T) - np.trace(
            xs.T @ m[:nocc, :nocc] @ xs
        )

    one_body = (wa + wb) * beta_occ + ph(xb, p, nb)
    two_alpha = wa * np.trace(q[:na, :na]) + ph(xa, q, na)
    two_beta = wb * np.trace(r[:nb, :nb]) + ph(xb, r, nb)
    cross = 2.0 * np.trace(xa.T @ o[:na, :nb] @ xb @ o[na:, nb:].T)
    return s2_m_terms(na, nb) + one_body - two_alpha - two_beta - cross


def casida_x(xpy, apb_xpy, omega):
    """unrestricted.rs casida_x: X = ((X+Y) + (A+B)(X+Y)/W) / 2."""
    return 0.5 * (xpy + apb_xpy / omega)


def s2_brute(o, na, nb, x):
    """<Psi|S^2|Psi> = M^2 + M + |S_+ Psi|^2 in the Fock space (Jordan-Wigner).

    Spin orbitals: alpha p -> p, beta q -> n + q. S_+ = sum O_pq a+_{p alpha} a_{q beta}.
    |Psi> = sum X_{ia sigma} a+_{a sigma} a_{i sigma} |Phi>, X normalized.
    """
    n = o.shape[0]
    phi = sum(1 << k for k in range(na)) + sum(1 << (n + k) for k in range(nb))
    nva, nvb = n - na, n - nb
    if x.size == 0:
        psi = {phi: 1.0}
    else:
        x = x / np.sqrt(x @ x)
        psi = {}
        for i in range(na):
            for a in range(nva):
                sg, d = apply_excitation(phi, na + a, i)
                psi[d] = psi.get(d, 0.0) + sg * x[i * nva + a]
        for i in range(nb):
            for a in range(nvb):
                sg, d = apply_excitation(phi, n + nb + a, n + i)
                psi[d] = psi.get(d, 0.0) + sg * x[na * nva + i * nvb + a]
    sp_psi = {}
    for det, cf in psi.items():
        for pp in range(n):
            for qq in range(n):
                res = apply_excitation(det, pp, n + qq)
                if res:
                    sp_psi[res[1]] = sp_psi.get(res[1], 0.0) + cf * res[0] * o[pp, qq]
    return s2_m_terms(na, nb) + sum(v * v for v in sp_psi.values())


def part_d():
    rng = np.random.default_rng(20261115)
    n, nao = 4, 6
    s_ao = rng.standard_normal((nao, nao))
    s_ao = s_ao @ s_ao.T + nao * np.eye(nao)
    s_ao /= np.sqrt(np.outer(np.diag(s_ao), np.diag(s_ao)))

    def s_orthonormal(c):
        w, v = np.linalg.eigh(c.T @ s_ao @ c)
        return c @ v @ np.diag(w**-0.5) @ v.T

    # alpha and beta MOs span different 4-dimensional subspaces of the 6 AOs,
    # so O = C_a^T S C_b is not orthogonal and P = O^T O != 1.
    ca = s_orthonormal(rng.standard_normal((nao, n)))
    cb = s_orthonormal(rng.standard_normal((nao, n)))
    o_gen = ca.T @ s_ao @ cb
    q_a, _ = np.linalg.qr(rng.standard_normal((n, n)))
    q_b, _ = np.linalg.qr(rng.standard_normal((n, n)))
    o_orth = q_a.T @ q_b
    worst_ref = worst_exc = 0.0
    for o in (o_gen, o_orth):
        for na, nb in ((2, 1), (3, 1), (2, 2)):
            dim = na * (n - na) + nb * (n - nb)
            worst_ref = max(
                worst_ref,
                abs(s2_reference(o, na, nb) - s2_brute(o, na, nb, np.zeros(0))),
            )
            for _ in range(4):
                x = rng.standard_normal(dim)
                worst_exc = max(
                    worst_exc, abs(s2_excited(o, na, nb, x) - s2_brute(o, na, nb, x))
                )
    off_orth = maxabs(o_gen.T @ o_gen - np.eye(n))
    check(
        "D1 <S^2> of the reference: SpinOverlap::reference = Fock-space |S+ Phi|^2 "
        "(claim s2ref)",
        worst_ref < 1e-12,
        f"max {worst_ref:.1e}; |O^T O - 1| = {off_orth:.2f}",
    )
    check(
        "D2 <S^2> of a single-excitation state: SpinOverlap::excited = Fock-space "
        "M^2 + M + |S+ Psi|^2, general and orthogonal O (claim s2)",
        worst_exc < 1e-12,
        f"max {worst_exc:.1e} over 24 random X",
    )
    # Closed-shell limit: O = 1, X_a = +-X_b gives 0 / 2.
    eye = np.eye(n)
    xs = rng.standard_normal(2 * (n - 2))
    sing = s2_excited(eye, 2, 2, np.concatenate([xs, xs]))
    trip = s2_excited(eye, 2, 2, np.concatenate([xs, -xs]))
    check(
        "D3 closed shell, O = 1: X_a = X_b gives 0, X_a = -X_b gives 2 (claim s2)",
        abs(sing) < 1e-12 and abs(trip - 2.0) < 1e-12,
        f"{sing:.1e}, {trip:.12f}",
    )
    # casida_x: the X block of the full eigenvector, normalized X.X - Y.Y = 1.
    d = 5
    a_m = rng.standard_normal((d, d))
    a_m = a_m @ a_m.T + 4.0 * d * np.eye(d)
    b_m = 0.3 * rng.standard_normal((d, d))
    b_m = 0.5 * (b_m + b_m.T)
    full = np.block([[a_m, b_m], [-b_m, -a_m]])
    w_all, v_all = np.linalg.eig(full)
    worst_x = 0.0
    for k in np.argsort(w_all.real):
        if w_all[k].real <= 0:
            continue
        v = v_all[:, k].real
        xv, yv = v[:d], v[d:]
        scale = np.sqrt(xv @ xv - yv @ yv)
        xv, yv = xv / scale, yv / scale
        xpy = xv + yv
        got = casida_x(xpy, (a_m + b_m) @ xpy, w_all[k].real)
        worst_x = max(worst_x, maxabs(got - xv))
    check(
        "D4 casida_x: ((X+Y) + (A+B)(X+Y)/W)/2 = the X block of the RPA eigenvector "
        "(claim casidax)",
        worst_x < 1e-10,
        f"max {worst_x:.1e}",
    )
    # Control: dropping the cross term breaks D2 for a mixed-spin X.
    o = o_gen
    x = rng.standard_normal(2 * 2 + 1 * 3)
    nva, nvb = 2, 3
    xa = x[:4].reshape(2, nva) / np.sqrt(x @ x)
    xb = x[4:].reshape(1, nvb) / np.sqrt(x @ x)
    cross = 2.0 * np.trace(xa.T @ o[:2, :1] @ xb @ o[2:, 1:].T)
    ctrl = abs(s2_excited(o, 2, 1, x) + cross - s2_brute(o, 2, 1, x))
    check(
        "D control: omitting the 2 tr(Xa^T O_oo Xb O_vv^T) term is detected",
        ctrl > 1e-3,
        f"error {ctrl:.1e}",
    )


def main():
    part_a()
    part_b()
    part_c()
    part_d()
    print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all identities hold")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
