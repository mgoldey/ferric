#!/usr/bin/env python3
"""Atomic charges: Mulliken, Lowdin, Becke, Hirshfeld, CHELPG and RESP.

Derivation: crates/ferric-scf/theory.md, "Atomic charges". Code:
crates/ferric-scf/src/properties.rs (mulliken_charges, lowdin_charges,
becke_charges_chunked, solve_chelpg_normal_equations, solve_resp_restrained)
and crates/ferric-scf/src/properties/hirshfeld_grid.rs (hirshfeld_charges).

As coded, with Z^eff_A = Atom::effective_z and N = Molecule::nelec
(= sum_A Z^eff_A - Q):
  Mulliken   q_A = Z^eff_A - sum_{mu in A} (D S)_mumu
  Lowdin     q_A = Z^eff_A - sum_{mu in A} (S^1/2 D S^1/2)_mumu
  Becke      q_A = Z^eff_A - s n_A,  n_A = sum_{g: home(g)=A} w_g rho_g
  Hirshfeld  q_A = Z^eff_A - s n_A,  n_A = sum_g w_g rho_g rho0_A/(sum_B rho0_B + 1e-12)
             (s = N / sum_A n_A in both grid schemes)
  CHELPG     [A 1; 1^T 0][q; lam] = [b; Q],  A_AB = sum_g 1/(r_gA r_gB),
             b_A = sum_g V_g / r_gA
  RESP       the same system with a/sqrt(q_A^2 + b^2) added to A_AA for every
             non-hydrogen atom, iterated from q = 0 until max |dq| < 1e-8.

Checks (each label carries "(claim X)"):
  A1  Mulliken and Lowdin sum to Q exactly for a density with Tr[D S] = N
      (idempotent closed shell on a random overlap), because
      Tr[S^1/2 D S^1/2] = Tr[D S]; Lowdin populations are nonnegative.
  A2  Becke and Hirshfeld sum to Q exactly after the rescale, whatever the
      quadrature error and the 1e-12 floor in the Hirshfeld weight.
  A3  CHELPG: the bordered normal equations give the same charges as an
      independent null-space solve of min |V - R q|^2 subject to
      sum q = Q, and the charges sum to Q.
  A4  RESP: the coded fixed point is the minimizer of
      1/2 sum_g (V_g - V_fit,g)^2 + a sum_heavy (sqrt(q^2 + b^2) - b)
      subject to sum q = Q, found independently by a constrained quasi-Newton
      minimization; the objective without the 1/2 (weight 2a on the
      restraint relative to the code) has a different minimizer (negative
      control); a -> 0 gives the CHELPG charges (trivial-limit anchor).

Mutations (each turns the script red; exact one-occurrence substitutions):
  Lowdin built with S in place of S^1/2                     (A1 fails)
  CHELPG border right-hand side 0 in place of Q              (A3 fails)
  RESP diagonal a/(2 sqrt(q^2+b^2)) in place of a/sqrt(..)   (A4 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# --- transcriptions of the code ------------------------------------------------


def mulliken(z, ao_atom, d, s):
    m = d @ s
    pop = np.zeros(len(z))
    np.add.at(pop, ao_atom, np.diag(m))
    return z - pop


def lowdin(z, ao_atom, d, s):
    s_half = np.real(sqrtm(s))
    m = s_half @ d @ s_half
    pop = np.zeros(len(z))
    np.add.at(pop, ao_atom, np.diag(m))
    return z - pop


def grid_charges(z, n_e, nelec):
    scale = nelec / n_e.sum() if abs(n_e.sum()) > 1e-12 else 1.0
    return z - scale * n_e


def hirshfeld_counts(rho, w, rho0):
    wa = rho0 / (rho0.sum(axis=0) + 1e-12)
    return (rho[None, :] * wa * w[None, :]).sum(axis=1)


def design(atoms, grid):
    return 1.0 / np.linalg.norm(grid[:, None, :] - atoms[None, :, :], axis=2)


def chelpg(atoms, grid, v, q_total):
    r = design(atoms, grid)
    n = len(atoms)
    mat = np.zeros((n + 1, n + 1))
    mat[:n, :n] = r.T @ r
    mat[:n, n] = mat[n, :n] = 1.0
    rhs = np.append(r.T @ v, q_total)
    return np.linalg.solve(mat, rhs)[:n]


def resp(atoms, grid, v, q_total, heavy, a, b):
    r = design(atoms, grid)
    n = len(atoms)
    base = np.zeros((n + 1, n + 1))
    base[:n, :n] = r.T @ r
    base[:n, n] = base[n, :n] = 1.0
    rhs = np.append(r.T @ v, q_total)
    q = np.zeros(n)
    for _ in range(50):
        mat = base.copy()
        for k in range(n):
            if heavy[k]:
                mat[k, k] += a / np.sqrt(q[k] * q[k] + b * b)
        q_new = np.linalg.solve(mat, rhs)[:n]
        dq = np.abs(q_new - q).max()
        q = q_new
        if dq < 1e-8:
            break
    return q


# --- A1, A2 --------------------------------------------------------------------


def claim_a1_a2():
    rng = np.random.default_rng(13)
    nbf, nocc = 9, 3
    ao_atom = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
    z = np.array([6.0, 1.0, 7.0, 1.0])
    x = rng.standard_normal((nbf, nbf)) * 0.25
    s = np.eye(nbf) + 0.5 * (x + x.T)
    s = s @ s.T / np.max(np.diag(s @ s.T))
    ev, u = np.linalg.eigh(s)
    x_orth = u @ np.diag(ev**-0.5) @ u.T
    c = x_orth @ np.linalg.qr(rng.standard_normal((nbf, nocc)))[0]
    d = 2 * c @ c.T
    nelec = 2 * nocc
    q_mol = z.sum() - nelec
    qm, ql = mulliken(z, ao_atom, d, s), lowdin(z, ao_atom, d, s)
    s_half = np.real(sqrtm(s))
    pop_ok = np.diag(s_half @ d @ s_half).min() > -1e-12
    check(
        "A1 Mulliken and Lowdin sum to Q; Lowdin populations >= 0 (claim A1)",
        abs(qm.sum() - q_mol) < 1e-12 and abs(ql.sum() - q_mol) < 1e-12 and pop_ok,
        f"Mulliken {qm.sum() - q_mol:.1e} Lowdin {ql.sum() - q_mol:.1e}",
    )
    npts = 500
    rho = rng.random(npts)
    w = rng.random(npts) * 0.05
    home = rng.integers(0, 4, npts)
    n_b = np.zeros(4)
    np.add.at(n_b, home, w * rho)
    rho0 = rng.random((4, npts)) ** 6
    rho0[:, :5] = 1e-15
    n_h = hirshfeld_counts(rho, w, rho0)
    qb, qh = grid_charges(z, n_b, nelec), grid_charges(z, n_h, nelec)
    check(
        "A2 Becke and Hirshfeld sum to Q after the rescale (claim A2)",
        abs(qb.sum() - q_mol) < 1e-12 and abs(qh.sum() - q_mol) < 1e-12,
        f"raw N {n_b.sum():.3f}/{n_h.sum():.3f} vs {nelec}; Becke {qb.sum() - q_mol:.1e} "
        f"Hirshfeld {qh.sum() - q_mol:.1e}",
    )


# --- A3, A4 --------------------------------------------------------------------


def esp_model():
    rng = np.random.default_rng(17)
    atoms = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [-0.8, 1.9, 0.0],
            [-0.8, -1.0, 1.7],
            [3.4, 1.4, 0.5],
        ]
    )
    heavy = np.array([True, True, False, False, True])
    pts = rng.standard_normal((900, 3))
    pts = 5.5 * pts / np.linalg.norm(pts, axis=1, keepdims=True) + atoms.mean(axis=0)
    true_q = np.array([-0.9, 0.55, 0.12, 0.08, -0.85])
    v = design(atoms, pts) @ true_q
    dip = np.array([0.2, -0.1, 0.3])
    rel = pts - atoms[0]
    v += rel @ dip / np.linalg.norm(rel, axis=1) ** 3
    return atoms, pts, v, heavy, float(true_q.sum())


def null_space_lsq(r, v, q_total):
    n = r.shape[1]
    q0 = np.full(n, q_total / n)
    basis = np.linalg.svd(np.ones((1, n)))[2][1:].T
    y = np.linalg.lstsq(r @ basis, v - r @ q0, rcond=None)[0]
    return q0 + basis @ y, q0, basis


def claim_a3():
    atoms, pts, v, _, q_total = esp_model()
    qc = chelpg(atoms, pts, v, q_total)
    qn, _, _ = null_space_lsq(design(atoms, pts), v, q_total)
    err = np.abs(qc - qn).max()
    check(
        "A3 CHELPG bordered normal equations = null-space constrained LSQ (claim A3)",
        err < 1e-10 and abs(qc.sum() - q_total) < 1e-12,
        f"max |dq| {err:.1e} sum - Q {qc.sum() - q_total:.1e}",
    )


def claim_a4():
    atoms, pts, v, heavy, q_total = esp_model()
    a, b = 0.0005, 0.1
    r = design(atoms, pts)
    _, q0, basis = null_space_lsq(r, v, q_total)

    def objective(y, chi_weight):
        q = q0 + basis @ y
        res = v - r @ q
        restraint = np.sum(np.where(heavy, np.sqrt(q * q + b * b) - b, 0.0))
        return chi_weight * res @ res + a * restraint

    def solve(chi_weight):
        y0 = np.zeros(basis.shape[1])
        out = minimize(
            objective, y0, args=(chi_weight,), method="BFGS", options={"gtol": 1e-12}
        )
        return q0 + basis @ out.x

    q_code = resp(atoms, pts, v, q_total, heavy, a, b)
    q_half = solve(0.5)
    q_full = solve(1.0)
    err = np.abs(q_code - q_half).max()
    miss = np.abs(q_code - q_full).max()
    q_chelpg = chelpg(atoms, pts, v, q_total)
    anchor = np.abs(resp(atoms, pts, v, q_total, heavy, 1e-14, b) - q_chelpg).max()
    # Stationarity on the constraint surface. The iteration stops at
    # max |dq| < 1e-8, so the projected gradient is bounded by |A| 1e-8.
    grad_r = np.where(heavy, a * q_code / np.sqrt(q_code**2 + b * b), 0.0)
    grad_chi = -r.T @ (v - r @ q_code)
    kkt_half = np.abs(basis.T @ (grad_chi + grad_r)).max()
    kkt_full = np.abs(basis.T @ (2 * grad_chi + grad_r)).max()
    bound = 10 * np.linalg.norm(r.T @ r, 2) * 1e-8
    check(
        "A4 RESP fixed point minimizes 1/2 chi^2 + a R; a -> 0 is CHELPG (claim A4)",
        kkt_half < bound
        and kkt_full > 10 * bound
        and err < 1e-6
        and miss > 10 * err
        and abs(q_code.sum() - q_total) < 1e-12
        and anchor < 1e-10,
        f"projected gradient {kkt_half:.1e} (bound {bound:.1e}), without the 1/2 "
        f"{kkt_full:.1e}; vs minimizers {err:.1e} / {miss:.1e}; a->0 {anchor:.1e}",
    )


def main():
    claim_a1_a2()
    claim_a3()
    claim_a4()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all atomic-charge identities verified")


if __name__ == "__main__":
    main()
