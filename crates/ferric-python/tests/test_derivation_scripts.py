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

pytest.importorskip("sympy", reason="the derivation proofs need sympy")

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
