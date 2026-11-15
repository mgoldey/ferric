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
        "pvp_ss_sympy.py",
        15,
        [
            ("(2 * dv * pv**2.5)", "(dv * pv**2.5)", "FAIL s/s"),
            (
                "- mpmath.pi * mpmath.exp(-pv * dv**2) / pv**2",
                "+ mpmath.pi * mpmath.exp(-pv * dv**2) / pv**2",
                "FAIL s/s",
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
