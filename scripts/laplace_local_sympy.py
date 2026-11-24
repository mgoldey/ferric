#!/usr/bin/env python3
"""Localized Laplace SOS-MP2 (`local` and `ao-sparse`): the rotated energy and its truncations.

Theory: `book/src/12-oo-laplace/04-local-laplace-sos.md#local-laplace-sos-rotated-energy`
and the sections after it. Code: crates/ferric-mp2/src/laplace_local.rs
(rotated_exp, build_local_domains, partners, energy_cut, occ_half_transform,
point_energy, sos_local_energy) and crates/ferric-mp2/src/boys.rs
(build_domains, build_pseudo_density_occ_sparse,
build_pseudo_density_vir_sparse). The functions below keep the Rust names and
loop structure.

Model: NBAS = 10 AO functions with a random overlap S and AO Fock F, solved
for canonical C and eps (NOCC = 4, NVIR = 6); a random 3-index (P|mu nu)
(NAUX = 14) and a random SPD metric V. The localized orbitals are L = C_o U
and C_vl = C_v R with random orthogonal U, R (the identities hold for any
orthogonal rotation, so Boys and VV-HV are not needed). AO functions, LMO and
LVO centroids sit on a line, so a radius cut drops pairs. Each check prints
"FAIL [<name>]" when it fails; the script exits 1 if any did.

  1  rotated-identity (sympy): for 2 occupied, 2 virtual and 2 auxiliary
     functions with symbolic integrals, rotations (rational Cayley
     parametrization) and exponential weights x_i = e^{t eps_i},
     y_a = e^{-t eps_a}: Z_PQ = sum E_ij H^P_ia Q_ab H^Q_jb with
     E = U^T diag(x) U, Q = R^T diag(y) R, H = U^T B R equals
     J_PQ = sum_ia B^P_ia x_i y_a B^Q_ia exactly.
  2  matrix-weight-is-expm (sympy): the columns v_k = U^T e_k are
     eigenvectors of F_loc = U^T diag(eps) U with eigenvalue eps_k, so
     E(t) = sum_k e^{t eps_k} v_k v_k^T = exp(t F_loc), the weight ao-sparse
     builds.
  3  rotated-equals-canonical (numpy, independent route): point_energy with
     vacuous domains, on Cholesky-whitened localized integrals, equals
     ||J||_F^2 from canonical orbitals whitened by the symmetric V^{-1/2},
     at every quadrature point.
  4  scalar-weight-control: replacing E(t) by the scalar diag(e^{t f_ii})
     (a localized orbital paired with one energy) changes the energy.
  5  t2-truncates: a finite `vir` radius drops pairs and changes the energy.
  6  t3-is-an-extra-truncation: with a finite `vir` radius, restricting j to
     partners (V_i and V_j share an LVO) changes the energy when R is a
     rotation, and changes nothing when R = 1 (Q(t) diagonal).
  7  t1-ta-vacuous / t1-truncates: occ_half_transform with every shell kept
     reproduces the full integrals; a finite `occ` radius changes them.
  8  energy-cut-vacuous / energy-cut-truncates: energy_cut at 1e-300 keeps
     every column (anchor); a cut at the median drops columns.
  9  ao-sparse-vacuous-equals-canonical: the masked AO pseudo-densities with
     full domains give the canonical energy.
 10  ao-sparse-retained-elements-exact: at a finite cutoff every element
     build_pseudo_density_occ_sparse keeps equals the canonical P(t)
     element; the per-orbital scalar construction (canonical C and
     e^{t eps_i} masked by the domain of localized orbital i) does not.
 11  ao-sparse-truncates: a finite cutoff changes the energy.

Tolerances. Checks 3, 7, 9, 10 compare two orderings of sums of O(1) terms
in double precision: bar 1e-11 relative. The controls (4, 5, 6, 7, 8, 10,
11) require a relative change above 1e-6; each effect measured on this
model is above 1e-3.

Mutations (each turns the named check red):
  - rotated_exp: m.T @ (w * m) -> m @ (w * m.T)       [rotated-equals-canonical]
  - partners: keep only j = i                          [rotated-equals-canonical]
  - Cholesky whitening L^{-1} -> L^T                   [rotated-equals-canonical]
  - energy_cut denominator f_vir - f_occ -> f_occ - f_vir  [energy-cut-vacuous]
  - occ pseudo-density exp(t F_loc) -> diag(e^{t f_ii})    [ao-sparse-vacuous-equals-canonical]
  - symbolic Q = R^T diag(y) R -> R diag(y) R^T            [rotated-identity]
"""

import sys

import numpy as np
import scipy.linalg as sla
import sympy as sp

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{name}] {detail}")
    if not ok:
        FAILED.append(name)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------- 1, 2 sympy


def cayley2(u):
    c, s = (1 - u**2) / (1 + u**2), 2 * u / (1 + u**2)
    return sp.Matrix([[c, -s], [s, c]])


def check_symbolic():
    u, v = sp.symbols("u v", real=True)
    x = sp.symbols("x0:2", positive=True)
    y = sp.symbols("y0:2", positive=True)
    b = [sp.Matrix(2, 2, sp.symbols(f"b{p}_0:4", real=True)) for p in range(2)]
    U, R = cayley2(u), cayley2(v)
    E = U.T * sp.diag(*x) * U
    Q = R.T * sp.diag(*y) * R
    H = [U.T * bp * R for bp in b]
    worst = 0
    for p in range(2):
        for q in range(2):
            z = sum(
                E[i, j] * H[p][i, a] * Q[a, c] * H[q][j, c]
                for i in range(2)
                for j in range(2)
                for a in range(2)
                for c in range(2)
            )
            jpq = sum(
                b[p][i, a] * x[i] * y[a] * b[q][i, a]
                for i in range(2)
                for a in range(2)
            )
            if sp.cancel(sp.together(z - jpq)) != 0:
                worst += 1
    check("rotated-identity", worst == 0, f"{worst} of 4 Z_PQ - J_PQ nonzero")

    eps = sp.symbols("e0:2", real=True)
    F = U.T * sp.diag(*eps) * U
    bad = 0
    for k in range(2):
        vk = U.T[:, k]
        if sp.simplify(F * vk - eps[k] * vk) != sp.zeros(2, 1):
            bad += 1
    spec = sum((x[k] * U.T[:, k] * U.T[:, k].T for k in range(2)), sp.zeros(2, 2))
    if sp.simplify(spec - E) != sp.zeros(2, 2):
        bad += 1
    check(
        "matrix-weight-is-expm",
        bad == 0,
        "F_loc U^T e_k = eps_k U^T e_k, E = sum x_k v_k v_k^T",
    )


# ---------------------------------------------------------------- the model

NBAS, NOCC, NAUX = 10, 4, 14
NVIR = NBAS - NOCC
RNG = np.random.default_rng(20261119)


def random_spd(n, shift):
    a = RNG.standard_normal((n, n))
    return a @ a.T / n + shift * np.eye(n)


def random_orthogonal(n):
    q, r = np.linalg.qr(RNG.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


S = random_spd(NBAS, 1.0)
F_AO = RNG.standard_normal((NBAS, NBAS))
F_AO = F_AO + F_AO.T + 6.0 * S @ np.diag(np.linspace(-1.0, 1.0, NBAS)) @ S
EPS, C = sla.eigh(F_AO, S)
C_O, C_V = C[:, :NOCC], C[:, NOCC:]
EPS_O, EPS_V = EPS[:NOCC], EPS[NOCC:]
assert EPS_O.max() < EPS_V.min()

PQ = RNG.standard_normal((NAUX, NBAS, NBAS))
PQ = PQ + PQ.transpose(0, 2, 1)  # (P|mu nu)
V = random_spd(NAUX, 0.5)

U = random_orthogonal(NOCC)
R = random_orthogonal(NVIR)
L = C_O @ U  # localized occupied
C_VL = C_V @ R  # localized virtuals
F_LOC = L.T @ F_AO @ L  # = U^T diag(eps_o) U

AO_X = np.linspace(0.0, 18.0, NBAS)  # AO centres on a line (Bohr)
LMO_X = np.array([1.0, 6.0, 11.0, 16.0])
LVO_X = np.array([0.5, 3.5, 7.0, 10.0, 13.5, 17.5])

T_POINTS = [(0.05, 0.4), (0.3, 0.35), (1.1, 0.2)]  # (t_k, w_k); any points do


def cholesky_inverse_sqrt(v):
    return np.linalg.inv(np.linalg.cholesky(v))  # M with M V M^T = 1


def sym_inverse_sqrt(v):
    w, q = np.linalg.eigh(v)
    return q @ np.diag(w**-0.5) @ q.T


# ---------------------------------------------------------------- local


def rotated_exp(m, eps, scale):
    w = np.exp(scale * eps)
    return m.T @ (w[:, None] * m)


def build_vir_domains(r_vir):
    return [
        [a for a in range(NVIR) if abs(LVO_X[a] - LMO_X[i]) <= r_vir]
        for i in range(NOCC)
    ]


def partners(vir):
    lmos_of_lvo = [[] for _ in range(NVIR)]
    for i, v in enumerate(vir):
        for a in v:
            lmos_of_lvo[a].append(i)
    out = []
    for v in vir:
        seen = [False] * len(vir)
        for a in v:
            for j in lmos_of_lvo[a]:
                seen[j] = True
        out.append([j for j in range(len(vir)) if seen[j]])
    return out


def all_partners(vir):
    return [list(range(len(vir))) for _ in vir]


def occ_half_transform(r_occ, pair_tau, q_sh):
    """(P|i lambda) with mu restricted to shells near LMO i (T1) and lambda to
    Schwarz-significant partners of that support (TA); one function per shell."""
    x = np.zeros((NAUX, NOCC, NBAS))
    for i in range(NOCC):
        occ = [m for m in range(NBAS) if abs(AO_X[m] - LMO_X[i]) <= r_occ]
        nsh = [y for y in range(NBAS) if any(q_sh[m, y] >= pair_tau for m in occ)]
        for lam in nsh:
            x[:, i, lam] = PQ[:, occ, lam] @ L[occ, i]
    return x


def local_h(x, vir):
    """Columns (i, a in V_i) of (P|ia), then whitened by M = Cholesky^{-1}."""
    cols = [x[:, i, :] @ C_VL[:, a] for i, v in enumerate(vir) for a in v]
    h = np.array(cols).T if cols else np.zeros((NAUX, 0))
    return cholesky_inverse_sqrt(V) @ h


def offsets(vir):
    off = [0]
    for v in vir:
        off.append(off[-1] + len(v))
    return off


def energy_cut(h_hat, vir, f_occ, f_vir, tau):
    off = offsets(vir)
    keep_cols, new_vir = [], []
    for i, v in enumerate(vir):
        kept = []
        for q, a in enumerate(v):
            col = off[i] + q
            norm2 = float(h_hat[:, col] @ h_hat[:, col])
            if norm2 / (f_vir[a] - f_occ[i]) >= tau:
                kept.append(a)
                keep_cols.append(col)
        new_vir.append(kept)
    return h_hat[:, keep_cols], new_vir


def point_energy(h_hat, vir, nbrs, e, q):
    off = offsets(vir)
    w = np.zeros_like(h_hat)
    for i, vi in enumerate(vir):
        for j in nbrs[i]:
            vj = vir[j]
            g = e[i, j] * q[np.ix_(vj, vi)]
            w[:, off[i] : off[i + 1]] += h_hat[:, off[j] : off[j + 1]] @ g
    z = h_hat @ w.T
    return float((z * z).sum())


def local_energy(h_hat, vir, nbrs, weight=None):
    tot = 0.0
    for t, wk in T_POINTS:
        e = rotated_exp(U, EPS_O, t) if weight is None else weight(t)
        q = rotated_exp(R, EPS_V, -t)
        tot += -wk * point_energy(h_hat, vir, nbrs, e, q)
    return tot


def canonical_energy():
    b = np.einsum("Pmn,mi,na->Pia", PQ, C_O, C_V)
    b = np.einsum("QP,Pia->Qia", sym_inverse_sqrt(V), b)
    tot = 0.0
    for t, wk in T_POINTS:
        bw = b * np.exp(t * EPS_O)[None, :, None] * np.exp(-t * EPS_V)[None, None, :]
        j = np.einsum("Pia,Qia->PQ", bw, b)
        tot += -wk * float((j * j).sum())
    return tot


def check_local():
    ref = canonical_energy()
    q_sh = np.abs(RNG.standard_normal((NBAS, NBAS)))
    q_sh = q_sh + q_sh.T
    x_full = occ_half_transform(1e9, 0.0, q_sh)

    full = build_vir_domains(1e9)
    h_full = local_h(x_full, full)
    e_vac = local_energy(h_full, full, partners(full))
    d = rel(e_vac, ref)
    check("rotated-equals-canonical", d < 1e-11, f"rel {d:.1e} (E_OS {ref:.6f})")

    fdiag = np.diag(F_LOC)
    d = rel(
        local_energy(
            h_full, full, partners(full), lambda t: np.diag(np.exp(t * fdiag))
        ),
        ref,
    )
    check("scalar-weight-control", d > 1e-6, f"rel change {d:.1e}")

    vir = build_vir_domains(4.0)
    h_t = local_h(x_full, vir)
    e_t2 = local_energy(h_t, vir, all_partners(vir))
    d = rel(e_t2, ref)
    kept = sum(map(len, vir))
    check(
        "t2-truncates",
        kept < NOCC * NVIR and d > 1e-6,
        f"|T| {kept}/{NOCC * NVIR}, rel {d:.1e}",
    )

    e_t3 = local_energy(h_t, vir, partners(vir))
    d_rot = rel(e_t3, e_t2)
    dropped = sum(NOCC - len(p) for p in partners(vir))
    # Q(t) diagonal: localized virtuals = canonical virtuals (R = 1).
    h_c = (
        cholesky_inverse_sqrt(V)
        @ np.array(
            [x_full[:, i, :] @ C_V[:, a] for i, v in enumerate(vir) for a in v]
        ).T
    )
    tot_p = tot_a = 0.0
    for t, wk in T_POINTS:
        e, qd = rotated_exp(U, EPS_O, t), np.diag(np.exp(-t * EPS_V))
        tot_p += -wk * point_energy(h_c, vir, partners(vir), e, qd)
        tot_a += -wk * point_energy(h_c, vir, all_partners(vir), e, qd)
    d_diag = rel(tot_p, tot_a)
    check(
        "t3-is-an-extra-truncation",
        dropped > 0 and d_rot > 1e-6 and d_diag < 1e-13,
        f"{dropped} (i,j) dropped; rel change {d_rot:.1e} with R rotated, {d_diag:.1e} with Q diagonal",
    )

    x_t1 = occ_half_transform(3.0, 0.0, q_sh)
    d_x = np.abs(x_t1 - x_full).max() / np.abs(x_full).max()
    x_ta = occ_half_transform(1e9, -1.0, q_sh)
    check(
        "t1-ta-vacuous", np.abs(x_ta - x_full).max() < 1e-12 * np.abs(x_full).max(), ""
    )
    check("t1-truncates", d_x > 1e-6, f"max rel change {d_x:.1e}")

    f_occ = np.diag(F_LOC)
    f_vir = np.diag(C_VL.T @ F_AO @ C_VL)
    h_cut, vir_cut = energy_cut(h_full, full, f_occ, f_vir, 1e-300)
    d = rel(local_energy(h_cut, vir_cut, partners(vir_cut)), ref)
    check(
        "energy-cut-vacuous",
        h_cut.shape[1] == NOCC * NVIR and d < 1e-11,
        f"rel {d:.1e}",
    )
    ratios = [
        float(h_full[:, c] @ h_full[:, c]) / (f_vir[a] - f_occ[i])
        for c, (i, a) in enumerate((i, a) for i, v in enumerate(full) for a in v)
    ]
    h_cut, vir_cut = energy_cut(h_full, full, f_occ, f_vir, float(np.median(ratios)))
    d = rel(local_energy(h_cut, vir_cut, partners(vir_cut)), ref)
    check(
        "energy-cut-truncates",
        h_cut.shape[1] < NOCC * NVIR and d > 1e-6,
        f"rel {d:.1e}",
    )


# ---------------------------------------------------------------- ao-sparse


def build_domains(cutoff):
    doms = [
        [m for m in range(NBAS) if abs(AO_X[m] - LMO_X[i]) <= cutoff]
        for i in range(NOCC)
    ]
    active = sorted({m for d in doms for m in d})
    return doms, active


def expm_sym(a, t):
    w, q = np.linalg.eigh(a)
    return q @ np.diag(np.exp(t * w)) @ q.T


def build_pseudo_density_occ_sparse(c_loc, f_loc, t, doms):
    expf = expm_sym(f_loc, t)
    keep = np.zeros((NBAS, NBAS), bool)
    for d in doms:
        keep[np.ix_(d, d)] = True
    return np.where(keep, (c_loc @ expf) @ c_loc.T, 0.0)


def build_pseudo_density_vir_sparse(c_vir, eps_v, t, active):
    q = np.zeros((NBAS, NBAS))
    qa = (c_vir * np.exp(-t * eps_v)) @ c_vir.T
    q[np.ix_(active, active)] = qa[np.ix_(active, active)]
    return q


def ao_energy(cutoff):
    doms, active = build_domains(cutoff)
    b = np.einsum("QP,Pmn->Qmn", sym_inverse_sqrt(V), PQ)
    tot = 0.0
    for t, wk in T_POINTS:
        p = build_pseudo_density_occ_sparse(L, F_LOC, t, doms)
        q = build_pseudo_density_vir_sparse(C_V, EPS_V, t, active)
        j = np.einsum("Pmn,ml,ns,Qls->PQ", b, p, q, b, optimize=True)
        tot += -wk * float((j * j).sum())
    return tot


def check_ao_sparse():
    ref = canonical_energy()
    d = rel(ao_energy(1e9), ref)
    check("ao-sparse-vacuous-equals-canonical", d < 1e-11, f"rel {d:.1e}")

    doms, _ = build_domains(4.0)
    t = 0.3
    exact = (C_O * np.exp(t * EPS_O)) @ C_O.T
    p = build_pseudo_density_occ_sparse(L, F_LOC, t, doms)
    keep = p != 0.0
    err_new = np.abs(p[keep] - exact[keep]).max() / np.abs(exact).max()
    old = np.zeros((NBAS, NBAS))
    for i, d_i in enumerate(doms):  # the per-orbital scalar construction
        ci = C_O[d_i, i]
        old[np.ix_(d_i, d_i)] += np.exp(t * EPS_O[i]) * np.outer(ci, ci)
    err_old = np.abs(old[keep] - exact[keep]).max() / np.abs(exact).max()
    check(
        "ao-sparse-retained-elements-exact",
        err_new < 1e-11 and err_old > 1e-6,
        f"kept {keep.sum()}/{NBAS * NBAS}; max rel error {err_new:.1e}, scalar construction {err_old:.1e}",
    )

    d = rel(ao_energy(4.0), ref)
    check("ao-sparse-truncates", d > 1e-6, f"rel {d:.1e}")


check_symbolic()
check_local()
check_ao_sparse()
if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall localized Laplace SOS-MP2 identities hold")
