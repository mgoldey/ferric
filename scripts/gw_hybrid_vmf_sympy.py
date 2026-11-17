#!/usr/bin/env python3
"""Sympy proof of the static GW shift on a hybrid / range-separated-hybrid KS
reference (book/src/19-gw-bse/05-continuation-and-qp-equation.md,
"Hybrid references";
ferric_gw::vxc_mo::vxc_diagonal_mo_on_grids and its add_exact_exchange).

The QP equation is eps_qp = eps_p + (Sigma_x - v_mf)_pp + Sigma_c, where v_mf
is the KS Fock minus h + J. Claims:

  H1  Closed shell: the RKS Fock's exchange term -1/2 c K[D] equals -c K[D_a]
      (D = 2 D_a), and in the MO basis (K[D_a])_pp = sum_i (pi|ip) = -Sigma_x,pp.
      Brute force on a random 8-fold-symmetric ERI tensor and a random
      orthonormal C (independent route: the definitions, index by index).
  H2  The shift: Sigma_x - v_mf = (1 - c) Sigma_x - v_sl for a global hybrid.
      Limits: c = 0 (pure functional) leaves Sigma_x - v_sl unchanged; c = 1,
      v_sl = 0 (HF) gives 0, the HF reference's shift. Symbolic.
  H3  Range separation: erf(w r) + erfc(w r) = 1, so with
      v_mf = v_sl - c_sr K_erfc - c_lr K_erf the shift is
      Sigma_x - v_sl - c_sr Sigma_x^SR - c_lr Sigma_x^LR, and c_sr = c_lr = c
      collapses it to H2. Symbolic; brute force on a random ERI split.
  H4  The defect: a v_mf without the K term is off by exactly c Sigma_x,pp
      (<= 0, the QP energy is pushed DOWN by |c Sigma_x|): symbolic.
  H5  Numerical cross-check against PySCF (testdata/reference/validation/
      gw_hybrid/h2o_cc-pvdz.json, scripts/validation/gen_gw_hybrid.py):
      PySCF's v_mf - v_xc^semilocal (from get_veff - get_j and nr_rks) equals
      its separately built -1/2 c K (exact four-centre), and for PBE0
      (c = 1/4) that term is c * Sigma_x(DF) to the RI fitting error of the
      cc-pVDZ-RI Lpq (an independent integral route), < 2e-4 Ha.

Mutations (each turns the script red):
  - drop the 1/2 in the closed-shell Fock exchange (marker "FAIL [H1]");
  - subtract only v_sl in the shift (marker "FAIL [H2]").

Run: python3 scripts/gw_hybrid_vmf_sympy.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

FAILED = []


def check(cond, tag, what):
    ok = bool(cond)
    print(f"  [{'ok' if ok else 'FAIL'}] {'' if ok else 'FAIL '}[{tag}] {what}")
    if not ok:
        FAILED.append(f"[{tag}] {what}")


def random_eri(n, rng):
    """(mu nu|la si) with 8-fold symmetry, positive-semidefinite as a pair
    matrix (a Gram matrix of random pair vectors)."""
    npair = n * (n + 1) // 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1)]
    L = rng.standard_normal((npair, npair + 3))
    g = L @ L.T / npair
    eri = np.zeros((n, n, n, n))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            for (p, q) in ((i, j), (j, i)):
                for (r, s) in ((k, l), (l, k)):
                    eri[p, q, r, s] = g[a, b]
    return eri


def k_of(eri, d):
    """K[D]_mn = sum_ls (m l|n s) D_ls  (ferric_scf::rhf::build_jk)."""
    return np.einsum("mlns,ls->mn", eri, d)


def j_of(eri, d):
    return np.einsum("mnls,ls->mn", eri, d)


# ---------------------------------------------------------------------------
print("\n=== H1  closed-shell exchange term and its MO diagonal")
rng = np.random.default_rng(7)
n, nocc, c_x = 6, 2, 0.25
eri = random_eri(n, rng)
C, _ = np.linalg.qr(rng.standard_normal((n, n)))  # orthonormal MOs (S = 1)
Da = C[:, :nocc] @ C[:, :nocc].T
D = 2.0 * Da
h = rng.standard_normal((n, n))
h = h + h.T
vsl_ao = rng.standard_normal((n, n))
vsl_ao = vsl_ao + vsl_ao.T
# The RKS Fock as rhf.rs assembles it: F = h + J[D] - 1/2 c K[D] + V_xc.
F = h + j_of(eri, D) - 0.5 * c_x * k_of(eri, D) + vsl_ao
v_mf_def = C.T @ (F - h - j_of(eri, D)) @ C  # the definition of v_mf
v_mf_code = C.T @ (vsl_ao - c_x * k_of(eri, Da)) @ C  # vxc_mo.rs
check(np.allclose(v_mf_def, v_mf_code, atol=1e-13, rtol=0), "H1", "v_mf = v_sl - c K[D_a]")
eri_mo = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C)
sx = np.array([-sum(eri_mo[p, i, i, p] for i in range(nocc)) for p in range(n)])
kd = np.diag(C.T @ k_of(eri, Da) @ C)
check(np.allclose(kd, -sx, atol=1e-13, rtol=0), "H1", "(K[D_a])_pp = sum_i (pi|ip) = -Sigma_x,pp")

# ---------------------------------------------------------------------------
print("\n=== H2  the static shift for a global hybrid")
Sx, vsl, c = sp.symbols("Sigma_x v_sl c", real=True)
v_mf = vsl + c * Sx  # H1: -c K_pp = c Sigma_x,pp
shift = Sx - v_mf
check(sp.simplify(shift - ((1 - c) * Sx - vsl)) == 0, "H2", "Sigma_x - v_mf = (1-c) Sigma_x - v_sl")
check(sp.simplify(shift.subs(c, 0) - (Sx - vsl)) == 0, "H2", "c = 0: pure functional unchanged")
check(sp.simplify(shift.subs({c: 1, vsl: 0})) == 0, "H2", "c = 1, v_sl = 0: the HF shift is 0")
shift_num = sx - np.diag(v_mf_code)
check(
    np.allclose(shift_num, (1 - c_x) * sx - np.diag(C.T @ vsl_ao @ C), atol=1e-13, rtol=0),
    "H2",
    "brute force: Sigma_x - v_mf = (1-c) Sigma_x - v_sl",
)

# ---------------------------------------------------------------------------
print("\n=== H3  range separation")
r, w = sp.symbols("r omega", positive=True)
check(sp.simplify((sp.erf(w * r) + sp.erfc(w * r)).rewrite(sp.erf) - 1) == 0, "H3", "erf + erfc = 1")
Ssr, Slr, csr, clr = sp.symbols("Sigma_x_SR Sigma_x_LR c_sr c_lr", real=True)
shift_rsh = (Ssr + Slr) - (vsl + csr * Ssr + clr * Slr)
check(
    sp.simplify(shift_rsh.subs({csr: c, clr: c}) - ((1 - c) * (Ssr + Slr) - vsl)) == 0,
    "H3",
    "c_sr = c_lr = c reduces to H2",
)
# Brute force: a random split of the ERIs into SR + LR parts.
eri_lr = random_eri(n, np.random.default_rng(11)) * 0.3
eri_sr = eri - eri_lr
c_sr, c_lr = 0.19, 0.65
F_rsh = h + j_of(eri, D) - 0.5 * (c_sr * k_of(eri_sr, D) + c_lr * k_of(eri_lr, D)) + vsl_ao
v_rsh_def = np.diag(C.T @ (F_rsh - h - j_of(eri, D)) @ C)
v_rsh_code = np.diag(C.T @ (vsl_ao - c_sr * k_of(eri_sr, Da) - c_lr * k_of(eri_lr, Da)) @ C)
check(np.allclose(v_rsh_def, v_rsh_code, atol=1e-13, rtol=0), "H3", "RSH v_mf brute force")

# ---------------------------------------------------------------------------
print("\n=== H4  size and sign of the double counting")
defect = (Sx - vsl) - shift  # shift without the K term minus the right one
check(sp.simplify(defect - c * Sx) == 0, "H4", "missing K term = c Sigma_x,pp")
check(np.all(sx <= 0.0), "H4", "Sigma_x,pp <= 0, so the defect lowers every QP energy")

# ---------------------------------------------------------------------------
print("\n=== H5  PySCF cross-check (H2O/cc-pVDZ)")


def find_reference():
    rel = Path("testdata/reference/validation/gw_hybrid/h2o_cc-pvdz.json")
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for p in [start, *start.parents]:
            if (p / rel).is_file():
                return p / rel
    return None


ref = find_reference()
check(ref is not None, "H5", "reference JSON found (a missing reference is a failure)")
if ref is not None:
    R = json.loads(ref.read_text())
    for blk in ("g0w0_pbe0", "g0w0_cam_b3lyp"):
        b = R[blk]
        vmf, vsl_r, vk = (np.array(b[k]) for k in ("v_mf", "v_xc_semilocal", "v_k_hybrid"))
        check(
            np.max(np.abs(vmf - vsl_r - vk)) < 1e-10,
            "H5",
            f"{blk}: get_veff - get_j - v_xc^sl = separately built -1/2 K term",
        )
    b = R["g0w0_pbe0"]
    vk, sxdf = np.array(b["v_k_hybrid"]), np.array(b["sigma_x_df"])
    d = float(np.max(np.abs(vk - 0.25 * sxdf)))
    print(f"    PBE0: max |v_K - c Sigma_x(DF)| = {d:.2e} Ha")
    check(d < 2e-4, "H5", "PBE0: v_K = 1/4 Sigma_x to the RI fitting error")
    check(float(np.max(np.abs(vk))) > 0.05, "H5", "PBE0: the K term is not negligible")

print()
if FAILED:
    print(f"FAILED {len(FAILED)} claim(s):")
    for fl in FAILED:
        print(f"  - {fl}")
    sys.exit(1)
print("all claims verified")
