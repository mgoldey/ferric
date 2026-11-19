"""Every ``scripts/*_sympy.py`` derivation proof holds, and turns red under its mutations.

CONTRIBUTING.md, "Every method has a derivation and a sympy proof": each proof
script exits 0 when its identities hold and nonzero when any fails. This runner
executes each script unmodified (it must exit 0), then once per recorded
mutation: a copy of the script with one exact source substitution, which must
exit nonzero on a failed CHECK (any traceback other than an AssertionError is a
crash, not a detection) and print the named failing check. The substitution
must occur exactly once in the source, so a renamed line cannot turn a mutation
into a silent no-op. Each script's header lists the same mutations.

Scripts with a runtime budget of ``SLOW_S`` seconds or more are marked
``validation`` (the weekly tier, ``pytest -m validation``).

The registry (``testdata/derivations/registry.json``,
``test_derivation_registry.py``) counts a script as run when its file name
appears in a test here.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import sympy  # noqa: F401  (a missing sympy must fail, not skip)

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"

# (script, typical-runtime budget in s (the subprocess gets 4x, at least 120 s), [(old source text, mutated text, failing-check marker)])
CASES = [
    (
        "rpa_drpa_pdep_sympy.py",
        30,
        [
            ("SCALE_CS = 4.0", "SCALE_CS = 2.0", "FAIL [freq-vs-definition]"),
            (
                "B2 = 2.0 * K\n    a = np.diag(gaps(eo, ev)) + B2",
                "B2 = 1.0 * K\n    a = np.diag(gaps(eo, ev)) + B2",
                "FAIL [plasmon-vs-definition]",
            ),
            ("SCALE_U = 2.0", "SCALE_U = 4.0", "FAIL [u-freq-vs-definition]"),
            (
                "rhs = B2 + f_t + bt_ + t @ B2 + t @ bt_",
                "rhs = B2 + f_t + bt_ + t @ bt_",
                "FAIL [riccati-vs-definition]",
            ),
        ],
    ),
    (
        "rpa_rs_mp2_rpa_sympy.py",
        15,
        [
            (
                "return e_mp2_c + e_rpa_l - 2.0 * eos_l",
                "return e_mp2_c + e_rpa_l - eos_l",
                "FAIL [b-form-second-order]",
            ),
            (
                "eos += g_ab * g_ab / denom",
                "eos += 2.0 * g_ab * g_ab / denom",
                "FAIL [ring2-equals-2eos]",
            ),
            (
                "ess += g_ab * (g_ab - g_ba) / denom",
                "ess += g_ab * (g_ab + g_ba) / denom",
                "FAIL [mp2-vs-spin-orbital]",
            ),
            (
                "return e_mp2_c + (e_rpa_c - 2.0 * eos_c) - (e_rpa_s - 2.0 * eos_s)",
                "return e_mp2_c + (e_rpa_c - 2.0 * eos_c) - e_rpa_s",
                "FAIL [t-form-second-order]",
            ),
        ],
    ),
    (
        "rpa_terf_signed_fit_sympy.py",
        15,
        [
            ("SIGMA_SR = -1.0", "SIGMA_SR = 1.0", "FAIL [signed-factor]"),
            (
                "keep = np.abs(w) > 1e-10 if signed else w > 1e-10",
                "keep = w > 1e-10 if signed else w > 1e-10",
                "FAIL [signed-is-exact]",
            ),
            ("SCALE_CS = 4.0", "SCALE_CS = 2.0", "FAIL [logdet-vs-plasmon]"),
        ],
    ),
    (
        "terf_kernel_sign_sympy.py",
        30,
        [
            (
                "FIRST_ARG_SIGN = -1",
                "FIRST_ARG_SIGN = 1",
                "FAIL [two-charge]",
            ),
            (
                "return mp.cos(k * r0)  # PHASE",
                "return mp.mpf(1)  # PHASE",
                "FAIL [two-charge-fourier]",
            ),
            ("BRACKET_DIV = 4", "BRACKET_DIV = 2", "FAIL [one-mode-sign]"),
        ],
    ),
    (
        "u_attenuated_sympy.py",
        15,
        [
            (
                "return e_ab + 0.5 * (d_aa + d_bb)",
                "return 2.0 * e_ab",
                "FAIL [u-ring2-equals-dmp2]",
            ),
            ("pi = pi_a + pi_b", "pi = pi_a", "FAIL [u-freq-vs-plasmon]"),
            (
                "return 0.5 * e, direct  # same-spin block",
                "return 1.0 * e, direct  # same-spin block",
                "FAIL [u-mp2-vs-spin-orbital]",
            ),
            ("e_os = e_ab", "e_os = e_ab + e_aa", "FAIL [os-is-ab]"),
            (
                "return e_mp2_c + (e_rpa_c - dmp2_c) - (e_rpa_s - dmp2_s)",
                "return e_mp2_c + (e_rpa_c - dmp2_c) - e_rpa_s",
                "FAIL [u-t-form-second-order]",
            ),
        ],
    ),
    (
        "mbd_coupled_dipole_sympy.py",
        20,
        [
            (
                "return 0.5 * np.sum(np.sqrt(lam)) - 1.5 * om.sum()",
                "return 0.5 * np.sum(np.sqrt(lam)) - 1.0 * om.sum()",
                "FAIL [qho-vs-frequency]",
            ),
            (
                "extra = 2.0 * zeta * zeta * theta / r**5",
                "extra = zeta * zeta * theta / r**5",
                "FAIL [t-gg-is-hessian]",
            ),
            (
                "FRAC = sp.sqrt(2 / sp.pi) / 3",
                "FRAC = sp.Rational(1, 3)",
                "FAIL [sigma-self-energy]",
            ),
            (
                "for b in range(n) for i in range(3))\n",
                "for b in [a] for i in range(3))\n",
                "FAIL [scs-dimer-closed-form]",
            ),
            (
                "return (v * (0.25 / np.sqrt(lam))) @ v.T",
                "return (v * (0.5 / np.sqrt(lam))) @ v.T",
                "FAIL [energy-adjoint]",
            ),
            (
                "dg2_ds = -2.0 * z2t * (3.0 - 2.0 * zeta * zeta) / (sigma * r5)\n    radial",
                "dg2_ds = -2.0 * z2t * (3.0 + 2.0 * zeta * zeta) / (sigma * r5)\n    radial",
                "FAIL [vjp-partials]",
            ),
            (
                'abar = c6t_bar * (6.0 / np.pi) * tp["weights"][k] * ak',
                'abar = c6t_bar * (3.0 / np.pi) * tp["weights"][k] * ak',
                "FAIL [gradient-vs-fd]",
            ),
            (
                "a0t_bar += rt_bar * rt / (3.0 * a0t)",
                "a0t_bar += rt_bar * rt / (2.0 * a0t)",
                "FAIL [gradient-vs-fd]",
            ),
            (
                "            abar = abar + a0t_bar\n",
                "            abar = abar + 0.0 * a0t_bar\n",
                "FAIL [gradient-vs-fd]",
            ),
            (
                "+ r_free / (3.0 * np.cbrt(ratio) ** 2) * gr",
                "+ r_free / (3.0 * np.cbrt(ratio)) * gr",
                "FAIL [ratio-chain-rule]",
            ),
            (
                'errs["closed"] = abs(fd - (-0.5 * np.sum(s1 * (d @ v @ d))))',
                'errs["closed"] = abs(fd - (-1.0 * np.sum(s1 * (d @ v @ d))))',
                "FAIL [orthonormality-term]",
            ),
            (
                "w_q = da @ v @ da + pc @ v @ pc + 0.5 * (cross + cross.T)",
                "w_q = da @ v @ da + pc @ v @ pc",
                "FAIL [orthonormality-term]",
            ),
            (
                "lattice = np.tile(-fixed.sum(axis=0) / n, (n, 1))",
                "lattice = np.tile(0.0 * fixed.sum(axis=0) / n, (n, 1))",
                "FAIL [lattice-term]",
            ),
        ],
    ),
    (
        "laplace_local_sympy.py",
        60,
        [
            (
                "return m.T @ (w[:, None] * m)",
                "return m @ (w[:, None] * m.T)",
                "FAIL [rotated-equals-canonical]",
            ),
            (
                "out.append([j for j in range(len(vir)) if seen[j]])",
                "out.append([j for j in range(len(vir)) if seen[j] and j == len(out)])",
                "FAIL [rotated-equals-canonical]",
            ),
            (
                "return np.linalg.inv(np.linalg.cholesky(v))  # M",
                "return np.linalg.cholesky(v).T  # M",
                "FAIL [rotated-equals-canonical]",
            ),
            (
                "if norm2 / (f_vir[a] - f_occ[i]) >= tau:",
                "if norm2 / (f_occ[i] - f_vir[a]) >= tau:",
                "FAIL [energy-cut-vacuous]",
            ),
            (
                "expf = expm_sym(f_loc, t)",
                "expf = np.diag(np.exp(t * np.diag(f_loc)))",
                "FAIL [ao-sparse-vacuous-equals-canonical]",
            ),
            (
                "Q = R.T * sp.diag(*y) * R",
                "Q = R * sp.diag(*y) * R.T",
                "FAIL [rotated-identity]",
            ),
        ],
    ),
    (
        "u_oo_rimp2_response_sympy.py",
        15,
        [
            (
                "pair_gamma_rows(blk_aa, 1.0, 1.0, 0.5, ba, gamma_a[:oa])",
                "pair_gamma_rows(blk_aa, 1.0, 1.0, 1.0, ba, gamma_a[:oa])",
                "FAIL [kernel-vs-fd]",
            ),
            (
                '"ab_a": 2.0 * g_a[:, oa:]',
                '"ab_a": 1.0 * g_a[:, oa:]',
                "FAIL [kernel-vs-fd]",
            ),
            (
                "pair_ooov_term(blk_aa, 1.0, 1.0, -0.5, ooov_a, i, oa, va, same)",
                "pair_ooov_term(blk_aa, 1.0, 1.0, 0.5, ooov_a, i, oa, va, same)",
                "FAIL [kernel-vs-fd]",
            ),
            (
                "cross += -1.0 * ab(i, jj).T @ o",
                "cross += 1.0 * ab(i, jj).T @ o",
                "FAIL [kernel-vs-fd]",
            ),
            (
                "row = bp[first_occ + i]",
                "row = bp[i]",
                "FAIL [kernel-vs-fd-frozen-core]",
            ),
            (
                "return u_v @ g_act @ u_o.T",
                "return u_v @ g_act @ u_o",
                "FAIL [assembled-vs-fd]",
            ),
            (
                "g[a, i] += 2.0 * f_mo[nocc + a, i]",
                "g[a, i] += 1.0 * f_mo[nocc + a, i]",
                "FAIL [assembled-vs-fd]",
            ),
            (
                "g[:, first_occ:nocc_total] = g_act",
                "g[:, 0 : nocc_total - first_occ] = g_act",
                "FAIL [assembled-vs-fd-frozen-core]",
            ),
            (
                "return p * a - q * a.T + p * b.T - q * b",
                "return p * a - q * a.T + p * b.T + q * b",
                "FAIL [pair-combination]",
            ),
        ],
    ),
    (
        "rpa_gradient_sympy.py",
        20,
        [
            (
                "xs = wk / (4.0 * np.pi)",
                "xs = wk / (2.0 * np.pi)",
                "FAIL [gamma-vs-fd]",
            ),
            (
                "return 4.0 * (w2 - x * y)",
                "return 4.0 * (w2 + x * y)",
                "FAIL [p-vs-fd]",
            ),
            (
                "w_mat[nocc:, :nocc] = z * eps[None, :nocc]",
                "w_mat[nocc:, :nocc] = 0.5 * z * eps[None, :nocc]",
                "FAIL [gradient-vs-fd]",
            ),
            (
                "p_ic = -y_mat[:nfc, nfc:nocc] / (2.0 * (eps[:nfc, None] - eps[None, nfc:nocc]))",
                "p_ic = -y_mat[:nfc, nfc:nocc] / (eps[:nfc, None] - eps[None, nfc:nocc])",
                "FAIL [gradient-vs-fd-frozen-core]",
            ),
            (
                "+ 4.0 * (cv.T @ g_op(lt, p_ao) @ co)",
                "+ 2.0 * (cv.T @ g_op(lt, p_ao) @ co)",
                "FAIL [gradient-vs-fd]",
            ),
        ],
    ),
    (
        "rpa_c6_polarizability_sympy.py",
        15,
        [
            (
                "PREF = 3.0 / np.pi",
                "PREF = 3.0 / (2.0 * np.pi)",
                "FAIL [c6-quadrature-converged]",
            ),
            (
                "tensor[d][j] = 4.0 * bare - 16.0 * coupled",
                "tensor[d][j] = 4.0 * bare - 4.0 * coupled",
                "FAIL [kernel-vs-casida]",
            ),
            (
                "omega_a = (4.0 / 3.0) * c6_eff / (alpha_eff * alpha_eff)",
                "omega_a = c6_eff / (alpha_eff * alpha_eff)",
                "FAIL [ts-pole-reproduces-c6]",
            ),
        ],
    ),
    (
        "hf_energy_sympy.py",
        20,
        [
            ("f = h + jm - 0.5 * km", "f = h + jm - km", "FAIL N2 E_RHF"),
            (
                "return 1 if sp_p == sp_q else 0",
                "return 1",
                "FAIL N8 <0|H|0>",
            ),
            (
                "f = f + pot @ fb @ pc",
                "f = f + pot @ fa @ pc",
                "FAIL N3 Roothaan effective Fock block table",
            ),
            (
                "g[p, q] = fa_mo[p, q] + fb_mo[p, q]",
                "g[p, q] = fa_mo[p, q]",
                "FAIL N4 dE/dkappa",
            ),
            ("return h + jm - km_s", "return h + jm - 0.5 * km_s", "FAIL N6 E_ROHF"),
            (
                "return u[:, kept] / np.sqrt(lam[kept])",
                "return u[:, kept] / lam[kept]",
                "FAIL [O1]",
            ),
            ("return x @ x.T @ s_bs", "return s_bs", "FAIL [O2]"),
        ],
    ),
    (
        "hf_gradient_sympy.py",
        20,
        [
            (
                "return 2.0 * (co * eps[:nocc]) @ co.T",
                "return (co * eps[:nocc]) @ co.T",
                "FAIL G1 RHF gradient",
            ),
            (
                "w = da @ fa @ da + da @ fb @ db",
                "w = da @ fa @ da + da @ fa @ db",
                "FAIL G3 ROHF gradient",
            ),
            ("d, d) - 0.5 * (", "d, d) - 0.25 * (", "FAIL G2 UHF gradient"),
            ("            - pulay", "            + pulay", "FAIL G1 RHF gradient"),
        ],
    ),
    (
        "hessian_factored_k_sympy.py",
        10,
        [
            ("    return g + g.T", "    return g", "FAIL [F1]"),
            (
                "kb = k_factored(BS, xb, cb_o)",
                "kb = k_factored(BS, xb, ca_o[:, :nb])",
                "FAIL [F2]",
            ),
            ("return (c @ u)[:, :k]", "return (c @ u.T)[:, :k]", "FAIL [F3]"),
        ],
    ),
    (
        "hf_hessian_sympy.py",
        20,
        [
            (
                "uf = np.vstack([-0.5 * s[:no], u])",
                "uf = np.vstack([-s[:no], u])",
                "FAIL H1 RHF Hessian",
            ),
            ("hs = 2.0 * occ * (f1_mo", "hs = occ * (f1_mo", "FAIL H1 RHF Hessian"),
            (
                "+ dy[0] @ fb @ db + da @ fy[1] @ db + da @ fb @ dy[1]",
                "+ dy[0] @ fb @ db + da @ fb @ dy[1]",
                "FAIL H4 ROHF Hessian",
            ),
            (
                "c.T @ vs @ c + f @ k - k @ f for",
                "c.T @ vs @ c + (np.diag(f)[:, None] - np.diag(f)[None, :]) * k for",
                "FAIL H4 ROHF Hessian",
            ),
        ],
    ),
    (
        "dft_becke_weight_hessian_sympy.py",
        90,
        [
            (
                "+ 2 * mu0 * r_x * r_y / rv**2",
                "+ 0 * mu0 * r_x * r_y / rv**2",
                "FAIL W1 mu_x, mu_xy and s_xy pair formulas",
            ),
            (
                "+ 2 * p_val[home] * tg * tg.T / t0**3",
                "+ 0 * p_val[home] * tg * tg.T / t0**3",
                "FAIL W2 log-form",
            ),
        ],
    ),
    (
        "df_jk_gradient_sympy.py",
        10,
        [
            (
                "term2 = -sp.Rational(1, 2) * (c.T * v1 * c)[0]",
                "term2 = -(c.T * v1 * c)[0]",
                "FAIL D1 RI-J gradient formula",
            ),
            (
                "return 1.0 / (lk * (lk - ll))",
                "return 0.0",
                "FAIL D3 truncated-metric derivative",
            ),
        ],
    ),
    (
        "fock_hook_gradient_sympy.py",
        10,
        [
            (
                "explicit = kappa * np.sum(d * model.bxp(x)) + lam * np.sum(d * model.sxp(x))",
                "explicit = kappa * np.sum(d * model.bxp(x))",
                "FAIL K1 hooked gradient",
            ),
            (
                "w = 2.0 * eps0[0] * (c0 @ c0.T)",
                "w = 2.0 * (c0.T @ fock_hooked(model, x, d, hook, hooked=False) @ c0)[0, 0] * (c0 @ c0.T)",
                "FAIL K1 hooked gradient",
            ),
        ],
    ),
    (
        "local_mp2_fit_pno_sympy.py",
        10,
        [
            (
                "e_pno, u = np.linalg.eigh(f_pno)",
                "e_pno, u = np.diag(f_pno), np.eye(npno)",
                "FAIL L3 full-rank PNO",
            ),
            (
                "return w_i.T @ w_j",
                "return w_i.T @ w_i",
                "FAIL L1 whitened Gram product",
            ),
        ],
    ),
    (
        "geometry_search_steps_sympy.py",
        10,
        [
            (
                "term1 = (ys + yhy) * rho * rho",
                "term1 = yhy * rho * rho",
                "FAIL B2 inverse of the direct BFGS update",
            ),
            (
                "    if ys <= CURVATURE_EPS * np.linalg.norm(y) * np.linalg.norm(s):",
                "    if ys <= -np.inf:",
                "FAIL B3 curvature guard",
            ),
            (
                "lam_p = 0.5 * (b_p + np.sqrt(b_p * b_p + 4.0 * g_p * g_p))",
                "lam_p = 0.5 * (b_p - np.sqrt(b_p * b_p + 4.0 * g_p * g_p))",
                "FAIL P2 prfo_step equals the augmented-Hessian",
            ),
            (
                "- e_dx * dx[i] * dx[j] / (dx_dx * dx_dx)",
                "- 0.0 * dx[i] * dx[j] / (dx_dx * dx_dx)",
                "FAIL F2 bofill_update",
            ),
            (
                "return -length * g / (masses * gn)",
                "return -length * g / (np.sqrt(masses) * gn)",
                "FAIL I1 mass-weighted gradient",
            ),
            (
                "TRUST_SHRINK_RATIO = 0.25\nTRUST_GROW_RATIO = 0.75",
                "TRUST_SHRINK_RATIO = 0.75\nTRUST_GROW_RATIO = 0.25",
                "FAIL T2 ratio_trust_update decision table",
            ),
            (
                "    if not accept or predicted >= 0.0 or ratio < TRUST_SHRINK_RATIO:",
                "    if ratio < TRUST_SHRINK_RATIO:",
                "FAIL T3 the radius never grows",
            ),
            (
                "    return alpha * p\n",
                "    return clip_per_block(p, angular, dist_trust, ang_trust)\n",
                "FAIL C1 uniform clip always descends",
            ),
        ],
    ),
    (
        "lindh_hessian_sympy.py",
        10,
        [
            (
                "BF_ALPHA = [\n    [1.0000, 0.3949, 0.3949],",
                "BF_ALPHA = [\n    [1.0000, 0.2800, 0.3949],",
                "FAIL L4 water = stored reference",
            ),
            (
                'add((i, j, k), 0.15 * w, "angle")',
                'add((i, j, k), 0.45 * w, "angle")',
                "FAIL L4 water = stored reference",
            ),
            (
                "    dj = -(t1 + 1.0) * di + t2 * dl\n",
                "    dj = (t1 - 1.0) * di + t2 * dl\n",
                "FAIL L2 Wilson tors",
            ),
            (
                "    return np.exp(alpha * (rref * rref - r * r))",
                "    return np.exp(-alpha * (rref * rref - r * r))",
                "FAIL L1 numeric rho below 1 past r_ref",
            ),
        ],
    ),
    (
        "hf_frequencies_sympy.py",
        15,
        [
            (
                "AU_FREQ_TO_CM = 219_474.631_363_2",
                "AU_FREQ_TO_CM = 219_474.631_363_2 / (2.0 * np.pi)",
                "FAIL D1 diatomic wavenumber",
            ),
            (
                "np.array([1.0 / np.sqrt(mass_au[c // 3]) for c in range(n)])",
                "np.array([1.0 / mass_au[c // 3] for c in range(n)])",
                "FAIL D1 diatomic wavenumber",
            ),
            (
                "0: [0.0, -r[2], r[1]],",
                "0: [0.0, r[2], r[1]],",
                "FAIL P1 rotation vectors preserve",
            ),
            (
                "np.outer(ht, t) - np.outer(t, t) * tht",
                "np.outer(ht, t) + np.outer(t, t) * tht",
                "FAIL P4 off-minimum",
            ),
        ],
    ),
    (
        "mp2_family_sympy.py",
        120,
        [
            (
                "return g * (2 * g - gx) / d  # closed-shell kernel",
                "return g * (2 * g + gx) / d  # closed-shell kernel",
                "FAIL A",
            ),
            (
                "quarter = sp.Rational(1, 4)  # u_rimp2 same-spin",
                "quarter = sp.Rational(1, 2)  # u_rimp2 same-spin",
                "FAIL B",
            ),
            (
                "e_ss_win = e_ss2 - e_ss1  # scs.rs",
                "e_ss_win = e_ss1 - e_ss2  # scs.rs",
                "FAIL C",
            ),
            (
                "half_w = sp.Rational(1, 2)  # Laplace factor",
                "half_w = sp.Integer(1)  # Laplace factor",
                "FAIL F",
            ),
            (
                "omega_link = 1 / (r0 * sp.sqrt(2))  # Operator::terfc",
                "omega_link = 1 / (r0 * 2)  # Operator::terfc",
                "FAIL D",
            ),
            (
                "return 1 - tc * tc  # vv10.rs factor_from_r2",
                "return 1 - tc  # vv10.rs factor_from_r2",
                "FAIL E",
            ),
            (
                "d1 = 1 - sp.exp(k * denom)  # 1 - e^{-kappa Delta}",
                "d1 = 1 - sp.exp(-k * denom)  # 1 - e^{-kappa Delta}",
                "FAIL G",
            ),
            (
                "cross = 2.0 * b_ss(g1, gd)  # the g1-d cross term of the window",
                "cross = 1.0 * b_ss(g1, gd)  # the g1-d cross term of the window",
                "FAIL C",
            ),
            (
                "CBS_POWER = 3  # the X^-3 model",
                "CBS_POWER = 2  # the X^-3 model",
                "FAIL H",
            ),
        ],
    ),
    (
        "double_hybrid_sympy.py",
        5,
        [
            ('"c_os": 0.51, "c_ss": 0.36}', '"c_os": 0.51, "c_ss": 0.35}', "FAIL P"),
            ('"cab_3": 11.869,', '"cab_3": -11.869,', "FAIL P"),
            (
                "e_ss = np.sum(g * (g - g.transpose(0, 3, 2, 1)) / den)",
                "e_ss = np.sum(g * g / den)",
                "FAIL E1",
            ),
            (
                "return e_ks + c_os * e_os + c_ss * e_ss",
                "return e_ks + c_ss * e_os + c_os * e_ss",
                "FAIL E2",
            ),
            (
                "f = h + jm - 0.5 * a_x * km + vxc",
                "f = h + jm - a_x * km + vxc",
                "FAIL E1",
            ),
            (
                '"dsdpbep86": {"s6": 0.48,',
                '"dsdpbep86": {"s6": 0.418,',
                "FAIL P",
            ),
            ("e_ss = e_aa + e_bb", "e_ss = e_aa", "FAIL U2"),
            (
                'e_dh = assemble(e_ks, e_os, e_ss, p["c_os"], p["c_ss"])  # U assembly',
                'e_dh = assemble(e_ks, e_os, e_ss, p["c_ss"], p["c_os"])  # U assembly',
                "FAIL U2",
            ),
            (
                "inv_sqrt(lam * v2), lam * t3)",
                "inv_sqrt(v2), lam * t3)",
                "FAIL L1",
            ),
            (
                "e_c = linlccd_hh(so_1, e_so, lam, lam)",
                "e_c = linlccd_hh(so_1, e_so, lam, 1.0)",
                "FAIL L1",
            ),
        ],
    ),
    (
        "nevpt2_sympy.py",
        10,
        [
            (
                'a -= perm(x["f3ca"], (1, 4, 0, 2, 5, 3))',
                'a += perm(x["f3ca"], (1, 4, 0, 2, 5, 3))',
                "FAIL C1",
            ),
            (
                'a -= ein("bi,qpai->pqab", h1e, dm2)',
                'a += ein("bi,qpai->pqab", h1e, dm2)',
                "FAIL C8",
            ),
            ("f = 0.5 if i == j else 1.0", "f = 1.0", "FAIL C4"),
            ("theta = 2.0 * v - g[i, bb, j, a]", "theta = v", "FAIL C3"),
            ("val += dm2[p, u, r, s_]", "val += 0.0", "FAIL R1"),
            (
                "- 2.0 * dl(i, k) * dm1[l_, j]",
                "- 1.0 * dl(i, k) * dm1[l_, j]",
                "FAIL R2",
            ),
            (
                'rp = ein("jfDE,ejE->efD", e, w)',
                'rp = ein("ejDE,fjE->efD", e, w)',
                "FAIL R3",
            ),
            (
                "v_sijr=perm(blk([cc, ca, cc, cv]), (3, 1, 2, 0))",
                "v_sijr=perm(blk([cc, ca, cc, cv]), (3, 1, 0, 2))",
                "FAIL T",
            ),
            (
                '-(x["eps_c"][i] + x["eps_c"][j])))',
                '(x["eps_c"][i] + x["eps_c"][j])))',
                "FAIL C7",
            ),
        ],
    ),
    (
        "att_rimp2_gradient_sympy.py",
        10,
        [
            (
                'grad += -2 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)  # metric derivative',
                'grad += 0.0 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)  # metric derivative',
                "FAIL L",
            ),
            (
                'du_3c = m[f"du_{corr}"]  # 3-centre derivative operator',
                'du_3c = m["du_coul"]  # 3-centre derivative operator',
                "FAIL L",
            ),
            (
                "xvo = veff_mo[nocc:, :nocc] + i_ia - i_ai  # Z-vector RHS",
                "xvo = i_ia - i_ai  # Z-vector RHS",
                "FAIL L",
            ),
            (
                'deri_ref = m["deri_coul"]  # four-centre derivative operator',
                'deri_ref = m["deri_erfc"]  # four-centre derivative operator',
                "FAIL L",
            ),
            (
                "RHO_W_SCALE = 1  # erf-part exponent",
                "RHO_W_SCALE = 0  # erf-part exponent",
                "FAIL E",
            ),
        ],
    ),
    (
        "terfc_rimp2_gradient_sympy.py",
        90,
        [
            (
                'grad += -2 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)  # metric derivative',
                'grad += 0.0 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)  # metric derivative',
                "FAIL L",
            ),
            (
                'dv = m[f"dv_{corr}"]  # 2-centre derivative operator',
                'dv = m["dv_coul"]  # 2-centre derivative operator',
                "FAIL L",
            ),
            (
                'deri_ref = m["deri_coul"]  # four-centre derivative operator',
                'deri_ref = m["deri_terfc"]  # four-centre derivative operator',
                "FAIL L",
            ),
            (
                "jz, kz = jk(eri_ref, dz)  # Z-vector coupling operator",
                'jz, kz = jk(m["eri_terfc"], dz)  # Z-vector coupling operator',
                "FAIL L",
            ),
            (
                "TERFC_SIGN = -1  # terfc = Coulomb + TERFC_SIGN * terf",
                "TERFC_SIGN = 1  # terfc = Coulomb + TERFC_SIGN * terf",
                "FAIL K",
            ),
            (
                "HERMITE_LOWER = 1  # weight of the -i E^{i-1,j} term of the derivative",
                "HERMITE_LOWER = 0  # weight of the -i E^{i-1,j} term of the derivative",
                "FAIL H",
            ),
        ],
    ),
    (
        "scs_2terfc_gradient_sympy.py",
        60,
        [
            (
                "(r2, 0 * c_os, c_ss),  # term 2: the non-bonded cutoff",
                "(r1, 0 * c_os, c_ss),  # term 2: the non-bonded cutoff",
                "FAIL C",
            ),
            (
                "w_ss, w_os = b_, a_  # (same-spin, opposite-spin) weights",
                "w_ss, w_os = a_, b_  # (same-spin, opposite-spin) weights",
                "FAIL U",
            ),
            (
                "z[1] = 1.0 * z[1]  # beta Z-vector",
                "z[1] = 0.0 * z[1]  # beta Z-vector",
                "FAIL U",
            ),
            (
                "eri_z = eri_ref  # Z-vector coupling kernel",
                'eri_z = np.einsum("Pmn,PQ,Qls->mnls", u, np.linalg.inv(v), u)  # Z-vector coupling kernel',
                "FAIL C",
            ),
            (
                "imat_sum = imat_sum + imat  # every term's Imat",
                "imat_sum = imat  # every term's Imat",
                "FAIL C",
            ),
            (
                "w_same, w_opp = w_ss, w_os  # (same-spin, opposite-spin) weights of x",
                "w_same, w_opp = w_os, w_ss  # (same-spin, opposite-spin) weights of x",
                "FAIL S",
            ),
        ],
    ),
    (
        "u_att_rimp2_gradient_sympy.py",
        15,
        [
            (
                "ck = corr  # correlation operator of amplitudes, b_full, Imat, RI response",
                'ck = "coul"  # correlation operator of amplitudes, b_full, Imat, RI response',
                "FAIL U",
            ),
            (
                'deri_ref = m["deri_coul"]  # two-electron derivative operator',
                'deri_ref = m["deri_erfc"]  # two-electron derivative operator',
                "FAIL U",
            ),
            (
                'eri_ref = m["eri_coul"]  # Z-vector and veff operator',
                'eri_ref = m["eri_erfc"]  # Z-vector and veff operator',
                "FAIL U",
            ),
            (
                "dtot = d_hf[0] + d_hf[1] + d_rel[0] + d_rel[1]  # relaxed density",
                "dtot = d_hf[0] + d_hf[1] + d_rel[0]  # relaxed density",
                "FAIL U",
            ),
            (
                "z[1] = 1.0 * z[1]  # beta Z-vector",
                "z[1] = 0.0 * z[1]  # beta Z-vector",
                "FAIL U",
            ),
            (
                "p_scale = scale  # scaling of the unrelaxed densities",
                "p_scale = 1.0  # scaling of the unrelaxed densities",
                "FAIL U",
            ),
        ],
    ),
    (
        "u_rimp2_gradient_sympy.py",
        5,
        [
            (
                "AB_IN_X_BETA = 1.0  # alpha-beta weight in x^beta",
                "AB_IN_X_BETA = 0.0  # alpha-beta weight in x^beta",
                "FAIL S",
            ),
            (
                "z[1] = 1.0 * z[1]  # beta Z-vector",
                "z[1] = 0.0 * z[1]  # beta Z-vector",
                "FAIL G",
            ),
            (
                "coupled = [ja + jb - ka, ja + jb - kb]  # CPHF opposite-spin Coulomb",
                "coupled = [ja - ka, jb - kb]  # CPHF opposite-spin Coulomb",
                "FAIL G",
            ),
            (
                "RI_SPIN_WEIGHT = 1.0  # RI response weight per spin",
                "RI_SPIN_WEIGHT = 2.0  # RI response weight per spin",
                "FAIL G",
            ),
        ],
    ),
    (
        "mp2_gradient_sympy.py",
        60,
        [
            (
                "coupling = 0.5 * az  # Z-vector coupling",
                "coupling = 1.0 * az  # Z-vector coupling",
                "FAIL G",
            ),
            (
                "xvo = veff_mo[nocc:, :nocc] + i_ia - i_ai  # RHS",
                "xvo = veff_mo[nocc:, :nocc] + i_ia + i_ai  # RHS",
                "FAIL G",
            ),
            (
                "p_oo = c_ss * p_mp2_oo + (c_os - c_ss) * p_os_oo  # SCS density weight",
                "p_oo = c_ss * p_mp2_oo + c_os * p_os_oo  # SCS density weight",
                "FAIL G",
            ),
            (
                "zeta[nocc:, :nocc] = eps[None, :nocc] * dm1[nocc:, :nocc]  # zeta occ-vir weight",
                "zeta[nocc:, :nocc] = eps[nocc:, None] * dm1[nocc:, :nocc]  # zeta occ-vir weight",
                "FAIL G",
            ),
            (
                "w_overlap = im1 - zeta_ao - vhf  # oo_rimp2_gradient w_overlap",
                "w_overlap = im1 - zeta_ao  # oo_rimp2_gradient w_overlap",
                "FAIL O",
            ),
            (
                "j_tot = jk(eri, pa + pb)[0]  # J[P^a + P^b]",
                "j_tot = jk(eri, pa)[0]  # J[P^a + P^b]",
                "FAIL U",
            ),
            (
                "rot = 2 * (f_mo @ w - w @ f_mo)  # U-OO rotation part",
                "rot = 2 * (w @ f_mo - f_mo @ w)  # U-OO rotation part",
                "FAIL U",
            ),
        ],
    ),
    (
        "mp2_variant_identities.py",
        15,
        [
            (
                "e_ss = np.sum(giajb * (giajb - gibja) / d)",
                "e_ss = np.sum(giajb * (giajb + gibja) / d)",
                "FAIL E_SS spatial",
            ),
            (
                "def same_spin(bs, eo, ev, scale=0.25):",
                "def same_spin(bs, eo, ev, scale=0.5):",
                "FAIL same-spin kernel",
            ),
            (
                "bt = b * np.exp(-0.5 * t * (ev[None, None, :] - eo[None, :, None]))",
                "bt = b * np.exp(-1.0 * t * (ev[None, None, :] - eo[None, :, None]))",
                "FAIL sum_PQ J_PQ(t)^2",
            ),
            (
                "(4.0 * gs * gl - gs * gl.transpose(0, 1, 3, 2)",
                "(2.0 * gs * gl - gs * gl.transpose(0, 1, 3, 2)",
                "FAIL E[s+l]",
            ),
            (
                "az = cv.T @ (4.0 * jz - kz - kz.T) @ co",
                "az = cv.T @ (4.0 * jz - kz) @ co",
                "FAIL (4J-K-K^T)",
            ),
            (
                "coeffs_u[obs_idx] = -ls12_u @ c2_u",
                "coeffs_u[obs_idx] = ls12_u @ c2_u",
                "FAIL interleaved S_O,R C == 0",
            ),
            (
                "return np.sum(g * (2.0 * h - h.transpose(0, 1, 3, 2)) / d)",
                "return np.sum(g * (2.0 * h + h.transpose(0, 1, 3, 2)) / d)",
                "FAIL X == 2 B(s, l)",
            ),
            (
                "coef = np.linalg.solve(v, three)  # V^-1 (Q|ia)",
                "coef = np.linalg.solve(chi.T @ chi, three)  # V^-1 (Q|ia)",
                "FAIL (ia|jb) - sum_P B B == (delta_ia|delta_jb)",
            ),
            (
                "b += m[p0:p1, q0:q1] @ mo_blk  # beta = 1",
                "b = m[p0:p1, q0:q1] @ mo_blk  # beta = 1",
                "FAIL chunked dressing",
            ),
            (
                "return start, start + (q + 1 if rank < s else q)",
                "return start, start + q",
                "FAIL band partials sum to G^(i)",
            ),
            (
                "U64, U32 = 2.0**-53, 2.0**-24",
                "U64, U32 = 2.0**-70, 2.0**-24",
                "FAIL two f64 orders",
            ),
            (
                "(1 + gamma(b, U32))",
                "(1 + gamma(b, U64))",
                "FAIL mixed bound, sub-ulp panel",
            ),
        ],
    ),
    (
        "mm_cutoff_sympy.py",
        15,
        [
            ("/ w_  # dS/dr", "# dS/dr", "BAD 5"),
            ("2.0 * qq_ * k * rr", "qq_ * k * rr", "BAD 5"),
        ],
    ),
    (
        "mm_dihedral_gradient_sympy.py",
        15,
        [
            (
                "gj = -gi + p * gi - q * gl",
                "gj = -gi + p * gi + q * gl",
                "FAIL: claim 2",
            ),
            ("- ct * B / bn**2)", "- ct * B / an**2)", "FAIL: claim 4"),
        ],
    ),
    (
        "d3_atm_gradient_sympy.py",
        30,
        [
            (
                "- sp.Rational(3, 2) / (q * P**3)",
                "- sp.Rational(3, 4) / (q * P**3)",
                "[FAIL] 3.",
            ),
            ("comp = 2 * dq[q]", "comp = dq[q]", "[FAIL] 5."),
        ],
    ),
    (
        "d3_cn_chain_rule_gradient_sympy.py",
        480,
        [
            (
                "w = (de_dcn[i] + de_dcn[j]) * dg_dr",
                "w = de_dcn[i] * dg_dr",
                "[FAIL] 5a.",
            ),
            ("c8 = 3 * c6 * q  # c8", "c8 = c6 * q  # c8", "[FAIL] 5a."),
        ],
    ),
    (
        "md_integrators_sympy.py",
        10,
        [
            (
                "    pn = ph - dt / 2 * g(xn)  # second kick at the NEW position",
                "    pn = ph - dt / 2 * g(xx)  # second kick at the NEW position",
                "FAIL [verlet-symplectic]",
            ),
            (
                "def vv_step_numeric(xx, pp, h, mm, g):\n    ph = pp - h / 2 * g(xx)",
                "def vv_step_numeric(xx, pp, h, mm, g):\n    ph = pp - h * g(xx)",
                "FAIL [verlet-time-reversal]",
            ),
            (
                "k * xx**2 / 2 * (1 - w2 * dt**2 / 4)\n    d =",
                "k * xx**2 / 2 * (1 - w2 * dt**2 / 2)\n    d =",
                "FAIL [verlet-shadow-energy]",
            ),
            (
                "amp2 = (1 - cc**2) * kT / m",
                "amp2 = (1 - cc) * kT / m",
                "FAIL [baoab-ou-variance]",
            ),
            (
                "O = sp.Matrix([[1, 0], [0, c]])",
                "O = sp.Matrix([[1, 0], [0, c**2]])",
                "FAIL [baoab-harmonic-config]",
            ),
            (
                "lam2 = 1 + dt / tau * (T0 / T - 1)",
                "lam2 = 1 + dt / tau * (T / T0 - 1)",
                "FAIL [berendsen-rate]",
            ),
        ],
    ),
    (
        "d3_hessian_sympy.py",
        10,
        [
            ("            h = madd(h, mixed_tt)", "            h = h", "[FAIL] 6a."),
            ('kw.get("g2_coeff", 72)', 'kw.get("g2_coeff", 36)', "[FAIL] 6a."),
            ("m[i][j] -= g * cij", "m[i][j] -= 0 * cij", "[FAIL] 6a."),
            (
                "code_c2 = -2 / R * c1 + a_ * (1 - 2 * cfun) * c1",
                "code_c2 = -2 / R * c1 + a_ * c1",
                "[FAIL] 1. c''",
            ),
        ],
    ),
    (
        "pvp_ss_sympy.py",
        15,
        [
            ("(2 * dv * pv**2.5)", "(dv * pv**2.5)", "FAIL s/s"),
            (
                "- mpmath.pi * mpmath.exp(-pv * dv**2) / pv**2",
                "+ mpmath.pi * mpmath.exp(-pv * dv**2) / pv**2",
                "FAIL s/s",
            ),
            (
                "resid = abs(-lb - cf - rem)",
                "resid = abs(lb - cf - rem)",
                "FAIL bra-bra",
            ),
            ("zv * av * na * nb", "zv * bv * na * nb", "FAIL bra-bra"),
            (
                "mpmath.exp(-pv * dv**2) / pv\n",
                "mpmath.exp(-pv * dv**2) / pv * (dv > 0)\n",
                "FAIL bra-bra",
            ),
        ],
    ),
    (
        "x2c_sympy.py",
        15,
        [
            ("spin = I * sum(", "spin = sum(", "FAIL [dirac-identity]"),
            ("metric = 2 * c**2", "metric = 4 * c**2", "FAIL [modified-dirac]"),
            (
                "lag = Vm + T * X + X.T * T - X.T * T * X",
                "lag = Vm + T * X + X.T * T + X.T * T * X",
                "FAIL [decoupling]",
            ),
            ("xx.T @ w @ xx * k", "xx.T @ w @ xx * 2 * k", "FAIL [spectrum]"),
            (
                "r = sym_pow(s, -0.5) @ inner @ sym_pow(s, 0.5)",
                "r = sym_pow(s, -0.5) @ sym_pow(s, 0.5)",
                "FAIL [renorm]",
            ),
        ],
    ),
    (
        "terf_identities_sympy.py",
        10,
        [
            (
                "sp.erf(sp.sqrt(S) - sp.sqrt(s))",
                "sp.erf(sp.sqrt(s) - sp.sqrt(S))",
                "FAIL [two-center]",
            ),
            (
                "df.append(df[-1] * (2 * i) / (2 * i + 1))",
                "df.append(df[-1] * (2 * i + 1) / (2 * i + 2))",
                "FAIL [series]",
            ),
            (
                "return pref * mp.sqrt(ph2 / th2) * G0c_num(",
                "return pref * G0c_num(",
                "FAIL [two-center]",
            ),
            (
                "CURVATURE_R0W = 1 / sp.sqrt(2)",
                "CURVATURE_R0W = sp.Rational(1, 2)",
                "FAIL [curvature]",
            ),
        ],
    ),
    (
        "screening_bounds_sympy.py",
        15,
        [
            (
                "1 - mp.cos(kk * r0v) * mp.exp",
                "1 + mp.cos(kk * r0v) * mp.exp",
                "FAIL [fourier]",
            ),
            (
                "m[i, j] = 2 * ph / mp.sqrt(mp.pi) * mp.exp(-(ph**2) * r0v**2)",
                "m[i, j] = 2 * ph / mp.sqrt(mp.pi)",
                "FAIL [indefinite]",
            ),
            (
                'QQR_SAFETY = mp.mpf("1.10")',
                'QQR_SAFETY = mp.mpf("0.50")',
                "FAIL [qqr-s]",
            ),
            (
                "return mp.sqrt(eri_s(name, fa, fb, fa, fb, w, r0v))",
                "return eri_s(name, fa, fb, fa, fb, w, r0v)",
                "FAIL [schwarz]",
            ),
        ],
    ),
    (
        "robust_fit_sympy.py",
        10,
        [
            (
                "return cj + cj.T - coef.T @ VCm @ coef",
                "return cj + cj.T",
                "FAIL [model]",
            ),
            (
                "G = robust_g(coef, VC, J)\n# residual-density",
                "G = robust_g(coef, Vw, J)\n# residual-density",
                "FAIL [model]",
            ),
            ("cj + cj.T - coef", "cj + cj - coef", "FAIL [model]"),
            ("Vinv_used = M.T @ M", "Vinv_used = M @ M", "FAIL [collapse]"),
        ],
    ),
    (
        "lmp2_direct_maps_sympy.py",
        10,
        [
            (
                "schur = A[nk:, nk:] - A[nk:, :nk] * A[:nk, :nk].inv() * A[:nk, nk:]",
                "schur = A[nk:, nk:] + A[nk:, :nk] * A[:nk, :nk].inv() * A[:nk, nk:]",
                "claim 1: Schur",
            ),
            (
                "rhs = 2 * (t3.T * dJ)[0, 0] - (dJ.T * A3.inv() * dJ)[0, 0]",
                "rhs = 2 * (t3.T * dJ)[0, 0]",
                "claim 2: expansion",
            ),
            (
                "+ 2 * (tk.T * dJk)[0, 0]",
                "+ 2 * (t_star[:nk, 0].T * dJk)[0, 0]",
                "claim 3: kept-block",
            ),
        ],
    ),
    (
        "lmp2_hylleraas_vs_wshg23_sympy.py",
        10,
        [
            (
                "- _t[b * 2 + a] * J[a * 2 + b]",
                "+ _t[b * 2 + a] * J[a * 2 + b]",
                "claim 4: exchange sign",
            ),
            (
                "(2 * J_num + A_num * t_m)",
                "(J_num + A_num * t_m)",
                "closed mask must have no first-order term",
            ),
        ],
    ),
    (
        "lmp2_fock_screen_sympy.py",
        10,
        [
            (
                "schur = A_DD - A_DK * A_KK.inv() * A_KD",
                "schur = A_DD + A_DK * A_KK.inv() * A_KD",
                "claim 1: Schur",
            ),
            (
                "t_est = -Delta.inv() * (Jm + lam * Ap * t0)",
                "t_est = -Delta.inv() * Jm",
                "claim 2: first order",
            ),
            (
                "return g + Fo * t + t * Fo",
                "return g + Fo * t + t",
                "claim 3: transpose",
            ),
        ],
    ),
    (
        "lmp2_lambda_boys_sympy.py",
        10,
        [
            (
                "pair_grad = sum(4 * w * (X[p, p] - X[q, q]) * X[p, q]",
                "pair_grad = sum(w * (X[p, p] - X[q, q]) * X[p, q]",
                "S1: gradient",
            ),
            (
                "    return np.arctan2(b, a) / 4",
                "    return np.arctan2(-b, -a) / 4",
                "S2: maximum",
            ),
            (
                "for X, w in list(zip(Xm, (1.0, 1.0, 1.0))) + [(Fm, lam_v)]:",
                "for X, w in list(zip(Xm, (1.0, 1.0, 1.0))) + [(Fm, 0.0)]:",
                "S3: stationarity",
            ),
            (
                "    return sp.atan2(-Bb_, Ab_) / 4",
                "    return sp.atan2(Bb_, Ab_) / 4",
                "S4: Boys",
            ),
        ],
    ),
    (
        "lmp2_combined_sympy.py",
        20,
        [
            (
                "schur_a = A_DD - A[np.ix_(D, K)]",
                "schur_a = A_DD + 0 * A[np.ix_(D, K)]",
                "P1: residual identity",
            ),
            (
                "g -= sum(fo[i, k] * T(k, j, a, b_) for k in range(no_s) if k != i)",
                "g -= 0",
                "P2: Jacobi",
            ),
            (
                "rhs = self.J[K] + self.B[np.ix_(K, D)] @ tau[D]",
                "rhs = self.J[K] - self.B[np.ix_(K, D)] @ tau[D]",
                "P3: exact when f_oo diagonal",
            ),
            (
                "                for X, w in [(X, 1.0) for X in R] + [(F, lam)]:",
                "                for X, w in [(X, 1.0) for X in R] + [(F, 0.0)]:",
                "P4: stationarity",
            ),
            (
                "4 * X3[p, q] * (X3[q, q] - X3[p, p])",
                "X3[p, q] * (X3[q, q] - X3[p, p])",
                "P4: gradient",
            ),
            (
                "md.J4, Uo, Uo, Uv, Uv",
                "md.J4, Uo, np.eye(md.no), Uv, Uv",
                "P6: rotational invariance",
            ),
        ],
    ),
    (
        "casci_slater_condon_phases_sympy.py",
        300,
        [
            (
                "e += gsym(p, p, q, q) - gsym(p, q, q, p)",
                "e += gsym(p, p, q, q) + gsym(p, q, q, p)",
                "FAIL n=",
            ),
            ("- gsym(a, p, p, i)", "", "FAIL n="),
            (
                "def hamiltonians_claim5(n, seed, exchange_sign=-1.0, quarter=0.25)",
                "def hamiltonians_claim5(n, seed, exchange_sign=1.0, quarter=0.25)",
                "FAIL claim 5 forms",
            ),
            (
                "f = hso + np.einsum(",
                "f = hso + 0.0 * np.einsum(",
                "FAIL claim 6 split",
            ),
        ],
    ),
    (
        "cc_ccsd_eom_brute_force_sympy.py",
        60,
        [
            (
                "r += 0.5 * (\n",
                "r += 1.0 * (\n",
                "FAIL [ccd-residual]",
            ),
            (
                "def ccsd_r(t1, t2, wmbej_t2=0.5)",
                "def ccsd_r(t1, t2, wmbej_t2=1.0)",
                "FAIL [ccsd-residual]",
            ),
            (
                "def ccsd_energy(t1, t2, quarter_t1=0.5)",
                "def ccsd_energy(t1, t2, quarter_t1=0.25)",
                "FAIL [ccsd-energy]",
            ),
            (
                "def cs_energy(t1s, t2s, exchange=1.0)",
                "def cs_energy(t1s, t2s, exchange=0.0)",
                "FAIL [spin-adaptation]",
            ),
            (
                "hsd = basis @ hbar @ basis.T - e_cc * np.eye(n)",
                "hsd = basis @ hbar @ basis.T + e_cc * np.eye(n)",
                "FAIL [eom-jacobian]",
            ),
            (
                "def claim_cis(coulomb=2.0)",
                "def claim_cis(coulomb=1.0)",
                "FAIL [cis-limit]",
            ),
            (
                "+ 0.5 * ((r2 - r2.transpose(0, 1, 3, 2)) ** 2).sum()",
                "+ 1.0 * ((r2 - r2.transpose(0, 1, 3, 2)) ** 2).sum()",
                "FAIL [spin-norm]",
            ),
        ],
    ),
    (
        "ci_rdm_casscf_gradient_sympy.py",
        30,
        [
            (
                "def generalized_fock(c, gam, big, core_factor=2.0, k_active=0.5)",
                "def generalized_fock(c, gam, big, core_factor=1.0, k_active=0.5)",
                "FAIL [casscf-gradient]",
            ),
            (
                "def generalized_fock(c, gam, big, core_factor=2.0, k_active=0.5)",
                "def generalized_fock(c, gam, big, core_factor=2.0, k_active=0.0)",
                "FAIL [casscf-gradient]",
            ),
            ("def rdms(c, swap=False)", "def rdms(c, swap=True)", "FAIL [rdm-energy]"),
            (
                "def claim_sigma(h_eff, g_act, hm, half=0.5)",
                "def claim_sigma(h_eff, g_act, hm, half=1.0)",
                "FAIL [sigma-split]",
            ),
        ],
    ),
    (
        "cc_triples_weights_and_linlccd_spin_adaptation_sympy.py",
        15,
        [
            ("4.0 * Wf", "2.0 * Wf", "[FAIL] claim 4b"),
            ('"jbkc,ia->ijkabc"', '"jckb,ia->ijkabc"', "[FAIL] claim 4b"),
            (
                "iajb * (2.0 * T - T.transpose(0, 3, 2, 1))",
                "iajb * (T - T.transpose(0, 3, 2, 1))",
                "[FAIL] claim 6",
            ),
        ],
    ),
    (
        "cc_uccsd_spin_blocks_sympy.py",
        15,
        [
            ("W_AAB = 0.5", "W_AAB = 1.0 / 6.0", "[FAIL] claim 2"),
            (
                "            return -self.t2ab[iq, ir].T",
                "            return self.t2ab[iq, ir].T",
                "[FAIL] claim 2",
            ),
            (
                "- x * 0.25 * np.einsum(",
                "- 0.0 * 0.25 * np.einsum(",
                "[FAIL] claim 3",
            ),
        ],
    ),
    (
        "gw_self_energy_sympy.py",
        30,
        [
            ("Pi_u = 2 * b**2", "Pi_u = 1 * b**2", "[FAIL] unrestricted"),
            (
                "0.5j * (cauchy(x1) - cauchy(1 / x1))",
                "0.5j * (cauchy(x1) + cauchy(1 / x1))",
                "[FAIL] product rule",
            ),
        ],
    ),
    (
        "gw_hybrid_vmf_sympy.py",
        15,
        [
            (
                "- 0.5 * c_x * k_of(eri, D)",
                "- c_x * k_of(eri, D)",
                "FAIL [H1]",
            ),
            ("shift = Sx - v_mf", "shift = Sx - vsl", "FAIL [H2]"),
        ],
    ),
    (
        "gw_bse_rpax_sympy.py",
        30,
        [
            (
                "apb[ia, jb] = 4.0 * coul - w_abij - w_ibaj",
                "apb[ia, jb] = 2.0 * coul - w_abij - w_ibaj",
                "FAIL [R1]",
            ),
            (
                "amb[ia, jb] = w_ibaj - w_abij",
                "amb[ia, jb] = w_abij - w_ibaj",
                "FAIL [R1]",
            ),
            (
                "a_bse[ia, jb] = 2.0 * coul - scr",
                "a_bse[ia, jb] = 1.0 * coul - scr",
                "FAIL [B1]",
            ),
            (
                "mu_x = SQRT2 * float(xv @ r_ia)",
                "mu_x = float(xv @ r_ia)",
                "FAIL [B3]",
            ),
        ],
    ),
    (
        "gw_ucohsex_sympy.py",
        30,
        [
            (
                "pi_a = pi_spin(b_a, eps_a, nocc_a, 2.0)",
                "pi_a = pi_spin(b_a, eps_a, nocc_a, 4.0)",
                "FAIL [U1]",
            ),
            (
                "occ = range(nocc_of[spin])",
                "occ = range(nocc_of[1 - spin])",
                "FAIL [U1]",
            ),
            (
                "on_occupied_side = (z - eps_q).real > 0",
                "on_occupied_side = True",
                "FAIL [U2]",
            ),
        ],
    ),
    (
        "rpa_response_modes_sympy.py",
        15,
        [
            (
                "return bare - 16.0 * w @ np.linalg.solve(eps, w.T), eps, w, bare",
                "return bare - 4.0 * w @ np.linalg.solve(eps, w.T), eps, w, bare",
                "FAIL [M1]",
            ),
            (
                "out -= np.outer(s, s) / lam[k]",
                "out += np.outer(s, s) / lam[k]",
                "FAIL [M2]",
            ),
            (
                "return bare - 4.0 * w_t @ np.linalg.solve(eps, w_t.T), w_t, eps",
                "return bare - 16.0 * w_t @ np.linalg.solve(eps, w_t.T), w_t, eps",
                "FAIL [M4]",
            ),
        ],
    ),
    (
        "cosmo_sympy.py",
        20,
        [
            (
                'return np.einsum("k,kmn->mn", q, m)',
                'return 0.5 * np.einsum("k,kmn->mn", q, m)',
                "FAIL C2 Tr[V dP]",
            ),
            (
                "return (eps - 1) / (eps + sp.Rational(1, 2))",
                "return (eps - 1) / eps",
                "FAIL C3 f = Kirkwood",
            ),
            (
                "return xk * xl / np.sqrt(xk * xk + xl * xl)",
                "return xk * xl / (xk + xl)",
                "FAIL C4 Gaussian S",
            ),
        ],
    ),
    (
        "external_potential_sympy.py",
        20,
        [
            (
                "f2 = -gp * (2.0 / (width * width) + 2.0 / (r * r)) + 2.0 * g / (r * r * r)",
                "f2 = -gp * (1.0 / (width * width) + 2.0 / (r * r)) + 2.0 * g / (r * r * r)",
                "FAIL E2 smeared_radial_derivs",
            ),
            ("    return +1\n", "    return -1\n", "FAIL E4 field origin"),
            (
                "grad[a, c] = (energy_of(cp) - energy_of(cm)) / (2.0 * h)",
                "grad[a, c] = (energy_of(cp) - energy_of(cm)) / h",
                "FAIL E5 FD field-density",
            ),
        ],
    ),
    (
        "qmmm_boundary_sympy.py",
        30,
        [
            (
                "full[q_full] += (1.0 - g) * d",
                "full[q_full] += g * d",
                "FAIL Q3 full_gradient",
            ),
            ("full[m1] -= 0.5 * f", "full[m1] -= 1.0 * f", "FAIL Q3 full_gradient"),
            ("qb = 2 * q0", "qb = q0", "FAIL Q2 RCD conserves"),
        ],
    ),
    (
        "thole_polarizable_sympy.py",
        60,
        [
            (
                "g[k] += -1.0 * d_dot_dr",
                "g[k] += -0.5 * d_dot_dr",
                "FAIL T3 site and charge",
            ),
            (
                'return -np.einsum("ic,icmn->mn", mu, f_ints)',
                'return -0.5 * np.einsum("ic,icmn->mn", mu, f_ints)',
                "FAIL T4 dE/dD",
            ),
            (
                "return lam3, lam3 - v * v * v / 6 * e",
                "return lam3, lam3 - v * v * v / 2 * e",
                "FAIL T1 lam5",
            ),
        ],
    ),
    (
        "atomic_charges_sympy.py",
        60,
        [
            (
                "    m = s_half @ d @ s_half\n",
                "    m = s @ d @ s\n",
                "FAIL A1 Mulliken and Lowdin",
            ),
            (
                "rhs = np.append(r.T @ v, q_total)\n    return np.linalg.solve(mat, rhs)[:n]",
                "rhs = np.append(r.T @ v, 0.0)\n    return np.linalg.solve(mat, rhs)[:n]",
                "FAIL A3 CHELPG",
            ),
            (
                "mat[k, k] += a / np.sqrt(q[k] * q[k] + b * b)",
                "mat[k, k] += a / (2 * np.sqrt(q[k] * q[k] + b * b))",
                "FAIL A4 RESP",
            ),
        ],
    ),
    (
        "pcm_ief_sympy.py",
        150,
        [
            (
                "z8 = y8 - p8 * A8 * dty8",
                "z8 = y8 + p8 * A8 * dty8",
                "[FAIL] 8 response gradient",
            ),
            (
                "np.fill_diagonal(D, -xi * np.sqrt(2 / np.pi) / (2 * rad))",
                "np.fill_diagonal(D, -xi * np.sqrt(2 / np.pi) / rad)",
                "[FAIL] 7 sphere",
            ),
        ],
    ),
    (
        "pbc_ewald_pairft_stress_sympy.py",
        60,
        [
            (
                "kp = -k * (1 / (4 * omega**2) + 1 / g2)",
                "kp = -k * (1 / (4 * omega**2))",
                "[FAIL] S5 Ewald strain",
            ),
            (
                "e_g0 = -math.pi * sum(q) ** 2 / (2 * vol * omega**2)",
                "e_g0 = -math.pi * sum(q) ** 2 / (vol * omega**2)",
                "[FAIL] S1 charged triclinic toy",
            ),
        ],
    ),
    (
        "pbc_rsgdf_sympy.py",
        60,
        [
            (
                "j3_split = j3_split - g0",
                "j3_split = j3_split + 0 * g0",
                "[FAIL] (claim split-j3)",
            ),
            (
                "w = evecs[:, keep] / np.sqrt(evals[keep])",
                "w = evecs[:, keep] / evals[keep]",
                "[FAIL] (claim fit-trivial)",
            ),
        ],
    ),
    (
        "pbc_kpoint_corr_sympy.py",
        60,
        [
            (
                "km[k] += mad * madelung_scale",
                "km[k] += mad * madelung_scale / NK",
                "[FAIL] (claim krhf-supercell)",
            ),
            (
                "def kmesh_jk(dms, madelung_scale=1.0, kernel_at_g=False):",
                "def kmesh_jk(dms, madelung_scale=1.0, kernel_at_g=True):",
                "[FAIL] (claim krhf-supercell)",
            ),
            (
                "kb = (ki + kj - ka) % NK",
                "kb = (ki - kj + ka) % NK",
                "[FAIL] (claim kmp2-supercell)",
            ),
            (
                "pi_q = (4.0 / NK) *",
                "pi_q = (4.0) *",
                "[FAIL] (claim kdrpa-supercell)",
            ),
        ],
    ),
    (
        "dft_becke_weight_grad_sympy.py",
        120,
        [
            (
                "* (1 - 2 * ac * mu_ij) / 2",
                "* (1 - 2 * ac * mu_ij)",
                "[FAIL] 5. becke_weights_and_grad assembly",
            ),
            (
                "/ (rJ * Rij) + mu_ij / Rij**2 * vij[k])",
                "/ (rJ * Rij) - mu_ij / Rij**2 * vij[k])",
                "[FAIL] 5. becke_weights_and_grad assembly",
            ),
            (
                "w_cheb = (math.pi / np1) * math.sin(theta)",
                "w_cheb = (math.pi / np1) * math.sin(theta) ** 2",
                "[FAIL] 6. radial: 75-point",
            ),
        ],
    ),
    (
        "dft_vxc_matrix_elements_sympy.py",
        750,
        [
            ("sp.Rational(1, 2) * vt * DCHI_N", "vt * DCHI_N", "[FAIL] numeric closed"),
            (
                "2 * vss * gs[b] + d[3] * gc[b]",
                "2 * vss * gs[b] + 2 * d[3] * gc[b]",
                "[FAIL] numeric polarized alpha",
            ),
            (
                "def closed_code(m, n, gga_fac=2, tau_fac=sp.Rational(1, 2)):",
                "def closed_code(m, n, gga_fac=1, tau_fac=sp.Rational(1, 2)):",
                "[FAIL] closed: V_mn",
            ),
        ],
    ),
    (
        "dft_fxc_kernel_sympy.py",
        30,
        [
            ("st[2 * c + t_]", "st[3 * t_ + c]", "[FAIL] 5. numeric: linearize_point"),
            (
                "2 * vs[0] * da[c] + vs[1] * db[c]",
                "vs[0] * da[c] + vs[1] * db[c]",
                "[FAIL] 5. numeric: linearize_point",
            ),
            (
                "w_a = [2 * dv_sig_aa * ga[b] + c * dv_sig_ab * gb[b] for b in AX]",
                "w_a = [dv_sig_aa * ga[b] + c * dv_sig_ab * gb[b] for b in AX]",
                "[FAIL] 3. w^a == d G^a / dt",
            ),
        ],
    ),
    (
        "dft_mgga_tau_gradient_sympy.py",
        40,
        [
            (
                "c = [2 * vs * grad_rho[b] for b in AX]",
                "c = [vs * grad_rho[b] for b in AX]",
                "[FAIL] 4. -2 (gga_ao_partials",
            ),
            ("        hess_col = 2 * (", "        hess_col = (", "[FAIL] 5. hess_col"),
        ],
    ),
    (
        "dft_vv10_potential_sympy.py",
        20,
        [
            (
                "    dk_dr = kap(i) / 6",
                "    dk_dr = kap(i) / 3",
                "[FAIL] W_i vrho_i, W_i vsig_i",
            ),
            (
                "    vsig = fac * wv * dw0_dg",
                "    vsig = fac * u * dw0_dg",
                "[FAIL] W_i vrho_i, W_i vsig_i",
            ),
        ],
    ),
    (
        "ks_fixed_grid_hessian_sympy.py",
        700,
        [
            (
                'coef = two_out * w * (k["rs"] * rho1 + 2 * k["ss"] * gg1)',
                'coef = two_out * w * (k["rs"] * rho1 + k["ss"] * gg1)',
                "[FAIL] 6. weights_response",
            ),
            (
                "+ self.d1[m][a] * self.d1[n][c]",
                "- self.d1[m][a] * self.d1[n][c]",
                "[FAIL] 9. first_order_fock",
            ),
        ],
    ),
    (
        "uks_xc_hessian_sympy.py",
        1800,
        [
            (
                "2 * dot(g[0], g1[0]),",
                "dot(g[0], g1[0]),",
                "[FAIL] 5. V1^s[D1^a, D1^b]",
            ),
            (
                "+ k.d1[3] * g1[t][c]",
                "- k.d1[3] * g1[t][c]",
                "[FAIL] 5. V1^s[D1^a, D1^b]",
            ),
        ],
    ),
    (
        "dft_hybrid_rsh_sympy.py",
        10,
        [
            ("return alpha + beta, alpha", "return alpha, alpha + beta", "[FAIL] C1"),
            (
                "return h + j - 0.5 * k_total(d, mix)",
                "return h + j - 0.25 * k_total(d, mix)",
                "[FAIL] B5",
            ),
            (
                "return h + j - k_total(ds, mix)",
                "return h + j - 0.5 * k_total(ds, mix)",
                "[FAIL] B5",
            ),
            (
                "return c_sr * k_of(eri_erfc, d) + c_lr * k_of(eri_erf, d)",
                "return c_sr * k_of(eri_erf, d) + c_lr * k_of(eri_erfc, d)",
                "[FAIL] B3",
            ),
            (
                "return 0.5 * np.sum(d * (h + f))",
                "return np.sum(d * (h + f))",
                "[FAIL] B4",
            ),
        ],
    ),
    (
        "tddft_response_sympy.py",
        10,
        [
            (
                'return 2.0 if spin == "singlet" else 0.0',
                'return 1.0 if spin == "singlet" else 0.0',
                "[FAIL] B1",
            ),
            ('s = 1.0 if spin == "singlet" else -1.0', "s = 1.0", "[FAIL] B1"),
            ("kov4.transpose(0, 3, 2, 1)", "kov4.transpose(0, 1, 2, 3)", "[FAIL] B1"),
            ("xpy = t_half @ z / np.sqrt(w)", "xpy = t_half @ z / w", "[FAIL] B4"),
            ("tdip = np.sqrt(2.0) * np.einsum", "tdip = np.einsum", "[FAIL] B2"),
            ("a[:da, da:] = cross", "a[:da, da:] = 0.0 * cross", "[FAIL] B5"),
            (
                "cross = 2.0 * np.trace(xa.T @ o[:na, :nb]",
                "cross = 1.0 * np.trace(xa.T @ o[:na, :nb]",
                "[FAIL] D2",
            ),
            (
                "return s2_m_terms(na, nb) + beta_occ - np.sum(o[:na, :nb] ** 2)",
                "return s2_m_terms(na, nb) + beta_occ",
                "[FAIL] D1",
            ),
            (
                "return 0.5 * (xpy + apb_xpy / omega)",
                "return 0.5 * (xpy - apb_xpy / omega)",
                "[FAIL] D4",
            ),
        ],
    ),
    (
        "cdft_constraint_coupling_sympy.py",
        30,
        [
            (
                "f_a, f_b = f_a + lw, f_b + lw",
                "f_a, f_b = f_a - lw, f_b - lw",
                "[FAIL] 4. total: dE/dN",
            ),
            ('f_b = e_b + st_b["lam"] * n_b', "f_b = e_b", "[FAIL] 3. H_ab invariant"),
            (
                'return alpha * pb["det_m"] + beta * pa["det_m"]',
                'return alpha * pa["det_m"] + beta * pb["det_m"]',
                "[FAIL] 1. cross_one_body",
            ),
        ],
    ),
    (
        "basis_normalization_sympy.py",
        15,
        [
            (
                "** (l + sp.Rational(3, 2))",
                "** (l + 1)",
                "FAIL S_l(a,b)",
            ),
            (
                "c_unit = coef / np.sqrt(s)  # overlap-matrix renormalization",
                "c_unit = coef / np.sqrt(np.sum(coef**2))  # overlap-matrix renormalization",
                "FAIL max |<phi|phi>",
            ),
        ],
    ),
    (
        "scf_jk_builders_sympy.py",
        10,
        [
            ("return np.linalg.solve(v, gamma)", "return gamma", "FAIL [N1]"),
            (
                "return 2.0 * rik_occupied(b, c_occ)",
                "return 1.0 * rik_occupied(b, c_occ)",
                "FAIL [N4]",
            ),
            (
                "return rik_occupied(b, xp) - rik_occupied(b, xm)",
                "return rik_occupied(b, xp) + rik_occupied(b, xm)",
                "FAIL [N5]",
            ),
            (
                "wt = 1.0 if m == n else 2.0",
                "wt = 1.0 if m == n else 1.0",
                "FAIL [N7]",
            ),
            (
                "jb, kb = jb + j_exact(dd), kb + k_exact(dd)",
                "jb, kb = j_exact(dd), k_exact(dd)",
                "FAIL [N9]",
            ),
            (
                "qk = s_exact @ np.linalg.solve(s_num, k_tilde)",
                "qk = k_tilde",
                "FAIL [C2]",
            ),
            (
                "return -0.25 * np.sum(d * k)",
                "return 0.25 * np.sum(d * k)",
                "FAIL [N10]",
            ),
        ],
    ),
    (
        "scf_diis_sympy.py",
        10,
        [
            (
                "return f @ d @ s - s @ d @ f",
                "return f @ d @ s + s @ d @ f",
                "FAIL [S1]",
            ),
            ("a[:m, m] = 1.0", "a[:m, m] = -1.0", "FAIL [N3]"),
            (
                "return error_vector(f, ds, S)",
                "return error_vector(f, dt, S)",
                "FAIL [N5]",
            ),
            (
                "return 4.0 * f[NOCC:, :NOCC]",
                "return 2.0 * f[NOCC:, :NOCC]",
                "FAIL [N1]",
            ),
            (
                "return f + mu * scv @ scv.T",
                "return f + mu * scv @ scv.T @ s",
                "FAIL [L1]",
            ),
            ("EDIIS_QUAD = 0.25", "EDIIS_QUAD = 0.5", "FAIL [E1]"),
            (
                "if fermi_dirac(eps, mid, sigma).sum() > n_pairs:",
                "if fermi_dirac(eps, mid, sigma).sum() < n_pairs:",
                "FAIL [F1]",
            ),
        ],
    ),
    (
        "scf_orbital_hessian_sympy.py",
        10,
        [
            (
                "dd = 2.0 * c @ ksym @ c.T\n    df = jmat(dd) - 0.5 * kmat(dd)",
                "dd = 1.0 * c @ ksym @ c.T\n    df = jmat(dd) - 0.5 * kmat(dd)",
                "FAIL [H2]",
            ),
            (
                "df = jmat(dd) - 0.5 * kmat(dd)",
                "df = jmat(dd) - 1.0 * kmat(dd)",
                "FAIL [H2]",
            ),
            ("df = jt - kmat(dd)", "df = jmat(dd) - kmat(dd)", "FAIL [H4]"),
            ("df = -0.5 * kmat(dd)", "df = 0.0 * kmat(dd)", "FAIL [S1]"),
            (
                "g += 0.5 * dn * a + 0.5 * b + dn * dfm",
                "g += 0.5 * dn * a + 1.0 * b + dn * dfm",
                "FAIL [H5]",
            ),
            (
                "cheap = 0.5 * (gv @ kv + mu * kv @ kv)",
                "cheap = 0.5 * (gv @ kv - mu * kv @ kv)",
                "FAIL [T1]",
            ),
        ],
    ),
    (
        "ks_zvector_sympy.py",
        10,
        [
            (
                "    rhs = -4.0 * cv.T @ v_ao @ co\n",
                "    rhs = -2.0 * cv.T @ v_ao @ co\n",
                "FAIL Z1 RKS:",
            ),
            (
                "return 0.25 * (g_at(1.0) - g_at(-1.0)) / (2.0 * t), z, hmat",
                "return 0.5 * (g_at(1.0) - g_at(-1.0)) / (2.0 * t), z, hmat",
                "FAIL Z2 RKS:",
            ),
            (
                "w_dot = 0.5 * (dd @ f_ao @ d + d @ f_ao @ dd + d @ df @ d)",
                "w_dot = 0.5 * (dd @ f_ao @ d + d @ f_ao @ dd)",
                "FAIL Z2 RKS:",
            ),
            (
                "rhs = [-2.0 * cv.T @ v_ao @ co for co, cv, _ in blocks]",
                "rhs = [-4.0 * cv.T @ v_ao @ co for co, cv, _ in blocks]",
                "FAIL Z3 UKS",
            ),
            (
                "rhs = [-4.0 * cv.T @ v_ao @ cc, -2.0",
                "rhs = [-2.0 * cv.T @ v_ao @ cc, -2.0",
                "FAIL Z4 ROKS",
            ),
            (
                "return alpha + closed + 0.5 * (cross + cross.T)",
                "return alpha + closed",
                "FAIL Z4 ROKS",
            ),
        ],
    ),
    (
        "mp2_polarizability_sympy.py",
        15,
        [
            (
                'grad -= np.einsum("ijcb,ikjb->ck"',
                'grad += np.einsum("ijcb,ikjb->ck"',
                "FAIL P1",
            ),
            ("xvo = lag + veff_vo(c, dm_p)", "xvo = lag", "FAIL P2"),
            ("return r + jd - 0.5 * kd", "return r + 2 * jd - 0.5 * kd", "FAIL P3"),
            (
                "sysm = amb @ apb + omega**2 * np.eye(n)",
                "sysm = amb @ apb + omega * np.eye(n)",
                "FAIL P4",
            ),
            (
                "return 3.0 / np.pi * np.sum",
                "return 3.0 / (2 * np.pi) * np.sum",
                "FAIL P5",
            ),
        ],
    ),
    (
        "rpa_quadrature_sympy.py",
        15,
        [
            (
                "weights = w * 2.0 * u0 / (1.0 - x) ** 2",
                "weights = w * u0 / (1.0 - x) ** 2",
                "FAIL Q3",
            ),
            (
                "w_k = (math.pi / np1) * math.sin(theta_k) * (u0 * math.pi / 4.0) * sec2",
                "w_k = (math.pi / np1) * math.sin(theta_k) ** 2 * (u0 * math.pi / 4.0) * sec2",
                "FAIL Q2",
            ),
            ("(u0 * math.pi / 4.0) * sec2", "(u0 * math.pi / 2.0) * sec2", "FAIL Q4"),
            (
                "    if n <= 16:\n        return 0.4",
                "    if n <= 16:\n        return 0.5",
                "FAIL Q6",
            ),
        ],
    ),
    (
        "rpa_laplace_chi0_sympy.py",
        15,
        [
            (
                "        coeff = 4.0 * w_l * math.cos(u * t_l)",
                "        coeff = 2.0 * w_l * math.cos(u * t_l)",
                "FAIL L3",
            ),
            (
                "        x = y * np.exp(-0.5 * t_l * e)\n        coeff = 4.0",
                "        x = y * np.exp(-t_l * e)\n        coeff = 4.0",
                "FAIL L3",
            ),
            (
                "    return t / ymin, w / ymin, r_tab",
                "    return t * ymin, w / ymin, r_tab",
                "FAIL L2",
            ),
            (
                "    if u * t_max > math.pi / 2:\n        return dense_dielectric(v, b, eo, ev, u)",
                "    if u * t_max > math.pi:\n        return dense_dielectric(v, b, eo, ev, u)",
                "FAIL L5",
            ),
            (
                "        coeff = 2.0 * w_l * math.cos(u * t_l)",
                "        coeff = 4.0 * w_l * math.cos(u * t_l)",
                "FAIL L6",
            ),
        ],
    ),
    (
        "rrho_thermo_sympy.py",
        15,
        [
            (
                "q = kt_over_h**1.5 * math.sqrt(math.pi) / (sigma * math.sqrt(abc))",
                "q = kt_over_h**1.5 * math.sqrt(math.pi) / math.sqrt(abc)",
                "FAIL N1 water",
            ),
            (
                "return dict(e=1.5 * R_EH * t, h=2.5 * R_EH * t, s=R_EH * (2.5 + math.log(q)))",
                "return dict(e=1.5 * R_EH * t, h=2.5 * R_EH * t, s=R_EH * (1.5 + math.log(q)))",
                "FAIL S1 transcribed S_trans",
            ),
            (
                "s += x * bose - math.log(-math.expm1(-x))",
                "s += x * bose + math.log(-math.expm1(-x))",
                "FAIL S3 transcribed S_vib",
            ),
            (
                "C2 = 100.0 * PLANCK_J_S * LIGHT_SPEED_M_PER_S / BOLTZMANN_J_PER_K",
                "C2 = 100.0 * PLANCK_J_S / BOLTZMANN_J_PER_K",
                "FAIL K1 c2",
            ),
        ],
    ),
    (
        "ir_intensity_sympy.py",
        15,
        [
            (
                "return w[keep], (u[:, keep] * inv[:, None]).T",
                "return w[keep], u[:, keep].T",
                "FAIL N2 diatomic",
            ),
            (
                "AU_TO_KM_PER_MOL = 1_777_097.731_306",
                "AU_TO_KM_PER_MOL = 974.880_111_8",
                "FAIL C1 AU_TO_KM_PER_MOL",
            ),
            (
                "rows.append(-(gp - gm) / (2.0 * h))",
                "rows.append((gp - gm) / (2.0 * h))",
                "FAIL N3 field route",
            ),
            (
                "return AU_TO_KM_PER_MOL * np.sum(dmu**2, axis=0)",
                "return AU_TO_KM_PER_MOL * dmu[2] ** 2",
                "FAIL N1 contraction",
            ),
        ],
    ),
]


def _run(path, timeout):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-I", str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=max(4 * timeout, 120),
    )


def _crashed(stderr):
    if "Traceback" not in stderr:
        return False
    last = [ln for ln in stderr.strip().splitlines() if ln.strip()][-1]
    return not last.startswith("AssertionError")


# Proofs whose own runtime budget is SLOW_S or more (the exact-derivative XC
# Hessian and V_xc models, minutes each) run in the weekly validation tier.
SLOW_S = 600


def _tier(budget, *values, id):
    marks = [pytest.mark.validation] if budget >= SLOW_S else []
    return pytest.param(*values, marks=marks, id=id)


@pytest.mark.parametrize(
    ("script", "timeout"), [_tier(c[1], c[0], c[1], id=c[0]) for c in CASES]
)
def test_script_holds(script, timeout):
    r = _run(SCRIPTS / script, timeout)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]


MUTATIONS = [
    (script, timeout, old, new, marker)
    for script, timeout, muts in CASES
    for old, new, marker in muts
]


@pytest.mark.parametrize(
    ("script", "timeout", "old", "new", "marker"),
    [_tier(m[1], *m, id=f"{m[0]}:{i}") for i, m in enumerate(MUTATIONS)],
)
def test_mutation_turns_the_script_red(tmp_path, script, timeout, old, new, marker):
    src = (SCRIPTS / script).read_text()
    assert src.count(old) == 1, f"mutation anchor not unique in {script}: {old!r}"
    mutated = tmp_path / script
    mutated.write_text(src.replace(old, new))
    r = _run(mutated, timeout)
    out = r.stdout + r.stderr
    assert r.returncode != 0, f"{script} stayed green under {old!r} -> {new!r}"
    assert not _crashed(r.stderr), (
        f"{script} crashed instead of failing:\n{out[-3000:]}"
    )
    assert marker in out, f"failing check {marker!r} not reported:\n{out[-3000:]}"
