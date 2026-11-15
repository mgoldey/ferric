"""Every ``scripts/*_sympy.py`` derivation proof holds, and turns red under its mutations.

CONTRIBUTING.md, "Every method has a derivation and a sympy proof": each proof
script exits 0 when its identities hold and nonzero when any fails. This runner
executes each script unmodified (it must exit 0), then once per recorded
mutation: a copy of the script with one exact source substitution, which must
exit nonzero on a failed CHECK (any traceback other than an AssertionError is a
crash, not a detection) and print the named failing check. The substitution
must occur exactly once in the source, so a renamed line cannot turn a mutation
into a silent no-op. Each script's header lists the same mutations.

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


@pytest.mark.parametrize(
    ("script", "timeout"), [(c[0], c[1]) for c in CASES], ids=[c[0] for c in CASES]
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
    MUTATIONS,
    ids=[f"{m[0]}:{i}" for i, m in enumerate(MUTATIONS)],
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
