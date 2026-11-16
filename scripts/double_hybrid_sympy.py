#!/usr/bin/env python3
"""Proofs for the double hybrids ferric codes: B2PLYP, DSD-PBEP86 and wB97X-L-V.

Theory: crates/ferric-mp2/theory.md, "16. Double hybrids" (B2PLYP,
DSD-PBEP86) and crates/ferric-cc/theory.md, "ωB97X-L-V correlation".
Code: crates/ferric-mp2/src/double_hybrid.rs (mp2_double_hybrid,
DoubleHybridKind::mp2_config), crates/ferric-dft/src/libxc.rs (b2plyp_def,
dsd_pbep86_def, wb97x_l_v_def, WB97X_L_V_EXT_PARAMS, WB97X_L_V_VV10),
crates/ferric-cc/src/double_hybrid.rs (lambda_scaled_sr_operator, assemble).

Claims (the book cites claim P, claim E and claim L):

  P  Parameter tables. The values in the Rust sources (parsed from the files,
     not copied) equal the cited values held in CITED below:
       B2PLYP, Grimme, J. Chem. Phys. 124, 034108 (2006): 0.53 HF + 0.47 B88
       exchange, 0.73 LYP correlation, c_os = c_ss = 0.27;
       DSD-PBEP86 without dispersion, Kozuch and Martin, J. Comput. Chem. 34,
       2327 (2013): 0.72 HF + 0.28 PBE exchange, 0.44 P86, c_os 0.51, c_ss 0.36;
       wB97X-L-V, Ransford and Carter-Fenk, Phys. Chem. Chem. Phys. 28, 14428
       (2026), Table 2 "Final" column: the 15 B97 coefficients, lambda 0.6,
       omega 0.1 Bohr^-1, VV10 b 10.0 and C 0.01; and the exchange mixing
       alpha 1.0, beta -0.4 read from eqn (27).
     Constraints: a_x + b_x = 1 for B2PLYP and DSD-PBEP86; B2PLYP's
     a_c = 1 - c_os; wB97X-L-V cx0 = 1 - lambda, css0 = cos0 = 1 - lambda^2,
     c_sr = alpha + beta = lambda, c_lr = alpha = 1, one omega everywhere;
     the KS definition wb97x_l_v_def takes no lambda or omega argument.
  E  Energy decomposition, numerically, on a model with orthonormal AOs
     (5 spatial orbitals, 2 doubly occupied), low-rank (RI-like) four-index
     integrals, and a model semilocal exchange (-c rho^(4/3)) and correlation
     (-a rho / (1 + b rho^(1/3))) on a quadrature grid. The KS SCF with
     F = h + J - a_x/2 K + b_x V_x + a_c V_c is converged; E_OS, E_SS are the
     code's closed-shell sums on its orbitals and orbital energies. The
     independent route is a determinant-space computation (210 determinants,
     second quantization applied to the reference by bit strings, no
     Slater-Condon formulas): E0 = <0|H|0>, the singles <S|H|0> and the
     Moller-Plesset second-order doubles sum split by spin.
       E1 plain-MP2 limit (a_x, b_x, a_c, c_os, c_ss) = (1, 0, 0, 1, 1): E_DH
          equals E0 + E2 with orbital energies from the integrals' Fock
          diagonal, and every singles element vanishes (Brillouin): HF + MP2.
       E2 B2PLYP and DSD-PBEP86 fractions: E_OS (alpha-beta doubles) and
          E_SS (alpha-alpha + beta-beta doubles) on the KS orbitals with KS
          orbital energies equal the brute-force doubles sums, and
          E_DH = E_KS + c_os E_OS + c_ss E_SS.
       E3 control: on the KS orbitals the singles elements do not vanish
          (|E_singles| > 1e-6 Ha), so dropping them is a definition, not an
          identity.
  L  wB97X-L-V lambda placement, on the same model's MO integrals. Route A
     (the code): scale the three-centre integrals AND the metric by lambda
     (the composite operator lambda*erfc), B -> sqrt(lambda) B, every
     four-index integral -> lambda, then LinLCCD(hh); its energy is the term
     the code adds. Route B (paper eqn 22): unscaled integrals, lambda on the
     driver and on the hh ladder, Fock terms unscaled, E_c = 1/4 sum g t,
     term lambda * E_c.
       L1 route A == route B (lambda = 0.6);
       L2 ladder off: route A == lambda^2 E_MP2 (sympy: E_MP2[lambda g] is
          lambda^2 E_MP2[g]; B(lambda) = sqrt(lambda) B);
       L3 control: the linear form lambda * E_c(lambda = 1) misses route A by
          more than 10 %;
       L4 plain-MP2 limit lambda = 1, ladder off: route A == the brute-force
          doubles sum of E.

Mutations (the runner applies each as an exact substitution):
  * DSD-PBEP86 cited c_ss 0.36 -> 0.35: FAIL P.
  * wB97X-L-V cited cab_3 11.869 -> -11.869 (the pdftotext sign trap): FAIL P.
  * E_SS without the exchange integral: FAIL E1.
  * c_os and c_ss swapped in the assembly: FAIL E2.
  * the HF exchange in the Fock at a_x instead of a_x/2: FAIL E1.
  * route A's factor scaled by lambda instead of sqrt(lambda): FAIL L1.
  * route B's hh ladder without lambda: FAIL L1.

Tolerances: every comparison is between two evaluations of the same
quantities in double precision on O(1) numbers with a few thousand terms, so
the bar is 1e-10 (about 1e5 * eps * |E|); the SCF is converged to a commutator
of 1e-12, below which the energy is stationary.

Run: python3 scripts/double_hybrid_sympy.py  (needs sympy, numpy)
"""

from __future__ import annotations

import ast
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import sympy as sp

FAILED: list[str] = []
TOL = 1e-10

# ----------------------------------------------------------------- cited values
CITED = {
    "b2plyp": {"a_x": 0.53, "b_x": 0.47, "a_c": 0.73, "c_os": 0.27, "c_ss": 0.27},
    "dsd-pbep86": {"a_x": 0.72, "b_x": 0.28, "a_c": 0.44, "c_os": 0.51, "c_ss": 0.36},
}
CITED_FUNCS = {
    "b2plyp": ("GGA_X_B88", "GGA_C_LYP"),
    "dsd-pbep86": ("GGA_X_PBE", "GGA_C_P86"),
}
# Ransford and Carter-Fenk 2026, Table 2, "Final" column (signs as recovered in
# testdata/reference/wb97x_l_v_params.json); libxc names _cx*, _css*, _cos*.
CITED_WB97XLV = {
    "cx_0": 0.4,
    "cx_1": 0.154,
    "cx_2": 3.884,
    "cx_3": -11.300,
    "cx_4": 8.425,
    "css_0": 0.64,
    "css_1": -1.417,
    "css_2": 4.716,
    "css_3": -2.956,
    "css_4": -0.861,
    "cab_0": 0.64,
    "cab_1": -1.194,
    "cab_2": -1.348,
    "cab_3": 11.869,
    "cab_4": -9.571,
    "lambda": 0.6,
    "omega": 0.1,
    "b": 10.0,
    "C": 0.01,
    "alpha": 1.0,
    "beta": -0.4,
}


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(label)


# ----------------------------------------------------------------- part P
def repo_root():
    cands = [Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents]
    if os.environ.get("FERRIC_ROOT"):
        cands.insert(0, Path(os.environ["FERRIC_ROOT"]))
    for c in cands:
        if (c / "crates" / "ferric-dft" / "src" / "libxc.rs").is_file():
            return c
    return None


def fn_body(src, name):
    start = src.index(f"pub fn {name}(")
    depth, i = 0, src.index("{", start)
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[start : j + 1]
    raise ValueError(name)


def num(s):
    """A Rust float expression of literals, unary minus and +/-, e.g. "1.0 + (-0.4)"."""

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
            a, b = ev(n.left), ev(n.right)
            return a + b if isinstance(n.op, ast.Add) else a - b
        raise ValueError(f"not a float expression: {s!r}")

    return ev(ast.parse(s.strip(), mode="eval").body)


def part_p():
    print("P: parameter tables in the Rust sources = cited values")
    root = repo_root()
    if root is None:
        check("P repository sources found", False, "set FERRIC_ROOT")
        return
    libxc = (root / "crates/ferric-dft/src/libxc.rs").read_text(encoding="utf-8")
    dh_mp2 = (root / "crates/ferric-mp2/src/double_hybrid.rs").read_text(
        encoding="utf-8"
    )
    dh_cc = (root / "crates/ferric-cc/src/double_hybrid.rs").read_text(encoding="utf-8")

    for kind, fname, variant in (
        ("b2plyp", "b2plyp_def", "B2plyp"),
        ("dsd-pbep86", "dsd_pbep86_def", "DsdPbep86"),
    ):
        c = CITED[kind]
        body = fn_body(libxc, fname)
        funcs = tuple(re.findall(r'XcFunctional::new\("(\w+)"', body))
        a_x = float(re.search(r"b3lyp_mix: Some\(([-\d.]+)\)", body).group(1))
        w = [
            float(x)
            for x in re.search(r"weights: Some\(vec!\[([^\]]*)\]", body)
            .group(1)
            .split(",")
        ]
        m = re.search(
            rf"Self::{variant} => Mp2DoubleHybridConfig \{{\s*c_os: ([-\d.]+),\s*c_ss: ([-\d.]+)",
            dh_mp2,
        )
        code = {
            "a_x": a_x,
            "b_x": w[0],
            "a_c": w[1],
            "c_os": float(m.group(1)),
            "c_ss": float(m.group(2)),
        }
        bad = {k: (code[k], c[k]) for k in c if abs(code[k] - c[k]) > 1e-15}
        check(
            f"P {kind} fractions == cited",
            not bad,
            f"code vs cited {bad}" if bad else str(code),
        )
        check(
            f"P {kind} libxc components == cited",
            funcs == CITED_FUNCS[kind],
            str(funcs),
        )
        check(f"P {kind} a_x + b_x == 1", abs(c["a_x"] + c["b_x"] - 1) < 1e-15)
    check(
        "P b2plyp a_c == 1 - c_os (Grimme's link)",
        abs(CITED["b2plyp"]["a_c"] - (1 - CITED["b2plyp"]["c_os"])) < 1e-15,
    )

    cw = CITED_WB97XLV
    blk = libxc[libxc.index("pub const WB97X_L_V_EXT_PARAMS") :]
    blk = blk[: blk.index("];")]
    ext = {k: float(v) for k, v in re.findall(r'\("(_\w+)",\s*(-?[\d.]+)\)', blk)}
    name = {"cx": "_cx", "css": "_css", "cab": "_cos"}
    bad = {}
    for k, v in cw.items():
        if "_" in k:
            fam, n = k.split("_")
            key = f"{name[fam]}{n}"
            if abs(ext.get(key, np.nan) - v) > 1e-15 or key not in ext:
                bad[key] = (ext.get(key), v)
    for key, cv in (
        ("_alpha", cw["alpha"]),
        ("_beta", cw["beta"]),
        ("_omega", cw["omega"]),
    ):
        if abs(ext.get(key, np.nan) - cv) > 1e-15:
            bad[key] = (ext.get(key), cv)
    check(
        "P wB97X-L-V WB97X_L_V_EXT_PARAMS == Table 2 (18 values)",
        not bad and len(ext) == 18,
        f"code vs cited {bad}" if bad else f"{len(ext)} values",
    )
    lam_x = float(
        re.search(r"pub const WB97X_L_V_LAMBDA: f64 = ([\d.]+);", libxc).group(1)
    )
    lam_c = float(
        re.search(r"pub const WB97X_L_V_LAMBDA: f64 = ([\d.]+);", dh_cc).group(1)
    )
    om_c = float(
        re.search(r"pub const WB97X_L_V_OMEGA: f64 = ([\d.]+);", dh_cc).group(1)
    )
    vv = re.search(
        r"WB97X_L_V_VV10: Vv10Params = Vv10Params \{ b: ([\d.]+), c: ([\d.]+) \}", libxc
    )
    body = fn_body(libxc, "wb97x_l_v_def")
    cam = re.search(r"omega: ([\d.]+),\s*c_sr: ([^,]+),\s*c_lr: ([^,]+),", body)
    vals = {
        "lambda (libxc.rs)": (lam_x, cw["lambda"]),
        "lambda (ferric-cc)": (lam_c, cw["lambda"]),
        "omega (ferric-cc)": (om_c, cw["omega"]),
        "omega (CamCoeffs)": (num(cam.group(1)), cw["omega"]),
        "VV10 b": (float(vv.group(1)), cw["b"]),
        "VV10 C": (float(vv.group(2)), cw["C"]),
        "c_sr = alpha + beta": (num(cam.group(2)), cw["alpha"] + cw["beta"]),
        "c_lr = alpha": (num(cam.group(3)), cw["alpha"]),
    }
    bad = {k: v for k, v in vals.items() if abs(v[0] - v[1]) > 1e-15}
    check(
        "P wB97X-L-V lambda, omega, VV10, CAM == cited",
        not bad,
        str(bad) if bad else "",
    )
    lam = cw["lambda"]
    cons = {
        "cx0 = 1 - lambda": (cw["cx_0"], 1 - lam),
        "css0 = 1 - lambda^2": (cw["css_0"], 1 - lam * lam),
        "cos0 = 1 - lambda^2": (cw["cab_0"], 1 - lam * lam),
        "c_sr = lambda": (cw["alpha"] + cw["beta"], lam),
    }
    bad = {k: v for k, v in cons.items() if abs(v[0] - v[1]) > 1e-12}
    check("P wB97X-L-V constrained coefficients", not bad, str(bad) if bad else "")
    sig = re.search(r"pub fn wb97x_l_v_def\(([^)]*)\)", libxc).group(1)
    check(
        "P wb97x_l_v_def takes only nspin (KS half fixed at the published lambda, omega)",
        sig.replace(" ", "") == "nspin:u32",
        sig,
    )


# ----------------------------------------------------------------- the model
RNG = np.random.default_rng(20261115)
N, NOCC, NAUX, NG = 5, 2, 9, 40


def make_model():
    t3 = RNG.normal(size=(NAUX, N, N)) * 0.2
    t3 = 0.5 * (t3 + t3.transpose(0, 2, 1))
    a = RNG.normal(size=(NAUX, NAUX))
    v2 = a @ a.T / NAUX + np.eye(NAUX)
    h = RNG.normal(size=(N, N)) * 0.2
    h = 0.5 * (h + h.T) + np.diag(np.array([-3.0, -2.4, 0.4, 1.0, 1.6]))
    chi = RNG.normal(size=(NG, N)) * 0.6
    w = RNG.uniform(0.05, 0.15, size=NG)
    return h, t3, v2, chi, w


def inv_sqrt(v):
    e, u = np.linalg.eigh(v)
    return (u / np.sqrt(e)) @ u.T


def ri_factor(t3, v2, lam=1.0):
    """B = (lam V)^-1/2 (lam T): the code's composite-operator scaling."""
    return np.einsum("PQ,Qmn->Pmn", inv_sqrt(lam * v2), lam * t3)


CX, CA, CB = 0.7, 0.3, 0.8


def xc_model(rho):
    r3 = np.cbrt(rho)
    ex, vx = -CX * rho * r3, -(4.0 / 3.0) * CX * r3
    ec = -CA * rho / (1 + CB * r3)
    vc = (
        -CA / (1 + CB * r3)
        + CA * rho * (CB / 3.0) / (r3 * r3 + 1e-300) / (1 + CB * r3) ** 2
    )
    return ex, vx, ec, vc


def ks_scf(model, a_x, b_x, a_c):
    h, t3, v2, chi, w = model
    b = ri_factor(t3, v2)
    eri = np.einsum("Pmn,Pls->mnls", b, b)

    def energy_fock(d):
        jm = np.einsum("mnls,ls->mn", eri, d)
        km = np.einsum("mlns,ls->mn", eri, d)
        rho = np.einsum("gm,mn,gn->g", chi, d, chi)
        ex, vx, ec, vc = xc_model(rho)
        vxc = np.einsum("g,gm,gn->mn", w * (b_x * vx + a_c * vc), chi, chi)
        f = h + jm - 0.5 * a_x * km + vxc
        e = (
            np.sum(d * h)
            + 0.5 * np.sum(d * jm)
            - 0.25 * a_x * np.sum(d * km)
            + np.sum(w * (b_x * ex + a_c * ec))
        )
        return e, f

    _, c = np.linalg.eigh(h)
    d = 2 * c[:, :NOCC] @ c[:, :NOCC].T
    fs, rs = [], []
    for _ in range(300):
        e, f = energy_fock(d)
        r = f @ d - d @ f
        if np.abs(r).max() < 1e-12:
            break
        fs.append(f)
        rs.append(r)
        fs, rs = fs[-8:], rs[-8:]
        m = len(fs)
        bm = -np.ones((m + 1, m + 1))
        bm[m, m] = 0
        for i, j in itertools.product(range(m), repeat=2):
            bm[i, j] = np.sum(rs[i] * rs[j])
        rhs = np.zeros(m + 1)
        rhs[m] = -1
        coef = np.linalg.solve(bm, rhs)[:m]
        f = sum(ci * fi for ci, fi in zip(coef, fs))
        eps, c = np.linalg.eigh(f)
        d = 2 * c[:, :NOCC] @ c[:, :NOCC].T
    else:
        raise AssertionError("model SCF did not converge")
    e, f = energy_fock(d)
    eps, c = np.linalg.eigh(f)
    return e, c, eps, b


def spin_components(b_ao, c, eps):
    """E_OS, E_SS as rimp2.rs::pair_energy sums them (closed shell)."""
    co, cv = c[:, :NOCC], c[:, NOCC:]
    bov = np.einsum("Pmn,mi,na->Pia", b_ao, co, cv)
    g = np.einsum("Pia,Pjb->iajb", bov, bov)
    eo, ev = eps[:NOCC], eps[NOCC:]
    den = (
        eo[:, None, None, None]
        + eo[None, None, :, None]
        - ev[None, :, None, None]
        - ev[None, None, None, :]
    )
    e_os = np.sum(g * g / den)
    e_ss = np.sum(g * (g - g.transpose(0, 3, 2, 1)) / den)
    return e_os, e_ss


def assemble(e_ks, e_os, e_ss, c_os, c_ss):
    return e_ks + c_os * e_os + c_ss * e_ss


# ------------------------------------------- determinant space (brute force)
def mo_integrals(model, b_ao, c):
    h = c.T @ model[0] @ c
    bm = np.einsum("Pmn,mp,nq->Ppq", b_ao, c, c)
    return h, np.einsum("Ppq,Prs->pqrs", bm, bm)  # chemist (pq|rs)


def apply_h(h, g, det):
    """H|det> by second quantization on bit strings; spin orbital 2p + s."""
    n = h.shape[0]
    out = {}

    def ann(d, k):
        if not d >> k & 1:
            return None, 0
        return d ^ (1 << k), (-1) ** bin(d & ((1 << k) - 1)).count("1")

    def cre(d, k):
        if d >> k & 1:
            return None, 0
        return d | (1 << k), (-1) ** bin(d & ((1 << k) - 1)).count("1")

    def add(d, v):
        out[d] = out.get(d, 0.0) + v

    so = range(2 * n)
    for p, q in itertools.product(so, repeat=2):
        if p % 2 != q % 2:
            continue
        d1, s1 = ann(det, q)
        if d1 is None:
            continue
        d2, s2 = cre(d1, p)
        if d2 is None:
            continue
        add(d2, h[p // 2, q // 2] * s1 * s2)
    # 1/2 sum <pq|rs> a+p a+q a_s a_r, <pq|rs> = (pr|qs)
    for p, q, r, s in itertools.product(so, repeat=4):
        if p % 2 != r % 2 or q % 2 != s % 2:
            continue
        v = g[p // 2, r // 2, q // 2, s // 2]
        if v == 0.0:
            continue
        d1, s1 = ann(det, r)
        if d1 is None:
            continue
        d2, s2 = ann(d1, s)
        if d2 is None:
            continue
        d3, s3 = cre(d2, q)
        if d3 is None:
            continue
        d4, s4 = cre(d3, p)
        if d4 is None:
            continue
        add(d4, 0.5 * v * s1 * s2 * s3 * s4)
    return out


def brute_force(h_mo, g_mo, eps):
    """E0, singles energy, and the MP doubles sum split (OS, SS)."""
    ref = sum(1 << k for k in range(2 * NOCC))
    hv = apply_h(h_mo, g_mo, ref)
    e0 = hv[ref]
    e_s = e_os = e_ss = 0.0
    max_s = 0.0
    occ0 = [k for k in range(2 * N) if ref >> k & 1]
    for d, v in hv.items():
        if d == ref or abs(v) < 1e-300:
            continue
        holes = [k for k in occ0 if not d >> k & 1]
        parts = [k for k in range(2 * N) if d >> k & 1 and not ref >> k & 1]
        de = sum(eps[k // 2] for k in holes) - sum(eps[k // 2] for k in parts)
        if len(holes) == 1:
            e_s += v * v / de
            max_s = max(max_s, abs(v))
        elif len(holes) == 2:
            if holes[0] % 2 != holes[1] % 2:
                e_os += v * v / de
            else:
                e_ss += v * v / de
    return e0, e_s, max_s, e_os, e_ss


def part_e():
    print("E: E_DH = E_KS + c_os E_OS + c_ss E_SS against a determinant-space route")
    model = make_model()
    # E1: plain-MP2 limit.
    e_ks, c, eps, b = ks_scf(model, 1.0, 0.0, 0.0)
    e_os, e_ss = spin_components(b, c, eps)
    e_dh = assemble(e_ks, e_os, e_ss, 1.0, 1.0)
    h_mo, g_mo = mo_integrals(model, b, c)
    eps_hf = np.array(
        [
            h_mo[p, p]
            + sum(2 * g_mo[p, p, j, j] - g_mo[p, j, j, p] for j in range(NOCC))
            for p in range(N)
        ]
    )
    e0, e_s, max_s, bos, bss = brute_force(h_mo, g_mo, eps_hf)
    err = abs(e_dh - (e0 + e_s + bos + bss))
    check(
        "E1 plain-MP2 limit: E_DH == <0|H|0> + E2 (HF + MP2)",
        err < TOL,
        f"|d| {err:.1e}",
    )
    check(
        "E1 Brillouin: max |<S|H|0>| == 0 at the HF limit", max_s < 1e-9, f"{max_s:.1e}"
    )
    check("E1 E_KS == <0|H|0>", abs(e_ks - e0) < TOL, f"|d| {abs(e_ks - e0):.1e}")
    # E2 / E3: the cited fractions, KS orbitals.
    for kind in ("b2plyp", "dsd-pbep86"):
        p = CITED[kind]
        e_ks, c, eps, b = ks_scf(model, p["a_x"], p["b_x"], p["a_c"])
        e_os, e_ss = spin_components(b, c, eps)
        e_dh = assemble(e_ks, e_os, e_ss, p["c_os"], p["c_ss"])
        h_mo, g_mo = mo_integrals(model, b, c)
        _, e_s, _, bos, bss = brute_force(h_mo, g_mo, eps)
        want = e_ks + p["c_os"] * bos + p["c_ss"] * bss
        d_os, d_ss, d_t = abs(e_os - bos), abs(e_ss - bss), abs(e_dh - want)
        check(f"E2 {kind}: E_OS == alpha-beta doubles", d_os < TOL, f"|d| {d_os:.1e}")
        check(f"E2 {kind}: E_SS == same-spin doubles", d_ss < TOL, f"|d| {d_ss:.1e}")
        check(
            f"E2 {kind}: E_DH == E_KS + c_os E_OS + c_ss E_SS",
            d_t < TOL,
            f"|d| {d_t:.1e}",
        )
        check(
            f"E3 {kind}: KS singles are not zero (dropped by definition)",
            abs(e_s) > 1e-6,
            f"{e_s:.2e}",
        )
    return model


# ----------------------------------------------------------------- part L
def so_integrals(model, lam_scale):
    """Antisymmetrized spin-orbital <pq||rs> on HF orbitals of the model, with
    the RI factor built from lam-scaled 3c and metric (route A), plus eps."""
    h, t3, v2, chi, w = model
    _, c, eps, _ = ks_scf(model, 1.0, 0.0, 0.0)
    b = ri_factor(t3, v2, lam_scale)
    bm = np.einsum("Pmn,mp,nq->Ppq", b, c, c)
    g = np.einsum("Ppq,Prs->pqrs", bm, bm)
    n2 = 2 * N
    so = np.zeros((n2, n2, n2, n2))
    for p, q, r, s in itertools.product(range(n2), repeat=4):
        v = 0.0
        if p % 2 == r % 2 and q % 2 == s % 2:
            v += g[p // 2, r // 2, q // 2, s // 2]
        if p % 2 == s % 2 and q % 2 == r % 2:
            v -= g[p // 2, s // 2, q // 2, r // 2]
        so[p, q, r, s] = v
    e_so = np.repeat(eps, 2)
    return so, e_so


def linlccd_hh(so, e_so, drv, lad):
    """t = (drv g_oovv + lad * 1/2 sum_kl <ij||kl> t_klab) / D, E = 1/4 sum g t."""
    o, v = slice(0, 2 * NOCC), slice(2 * NOCC, 2 * N)
    g = so[o, o, v, v]
    hh = so[o, o, o, o]
    eo, ev = e_so[o], e_so[v]
    den = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )
    t = drv * g / den
    for _ in range(500):
        t_new = (drv * g + lad * 0.5 * np.einsum("ijkl,klab->ijab", hh, t)) / den
        if np.abs(t_new - t).max() < 1e-15:
            t = t_new
            break
        t = t_new
    else:
        raise AssertionError("LinLCCD(hh) model iteration did not converge")
    return 0.25 * np.sum(g * t)


def part_l(model):
    print("L: wB97X-L-V lambda placement (composite operator vs eqn 22)")
    lam = CITED_WB97XLV["lambda"]
    so_l, e_so = so_integrals(model, lam)
    so_1, _ = so_integrals(model, 1.0)
    # Route A: the code; every integral already carries lambda.
    e_a = linlccd_hh(so_l, e_so, 1.0, 1.0)
    # Route B: eqn 22 on unscaled integrals, lambda on driver and hh ladder.
    e_c = linlccd_hh(so_1, e_so, lam, lam)
    d = abs(e_a - lam * e_c)
    check("L1 route A == lambda * E_c(eqn 22)", d < TOL, f"|d| {d:.1e}")
    e_a0 = linlccd_hh(so_l, e_so, 1.0, 0.0)
    e_mp2 = linlccd_hh(so_1, e_so, 1.0, 0.0)
    d = abs(e_a0 - lam * lam * e_mp2)
    check("L2 ladder off: route A == lambda^2 E_MP2", d < TOL, f"|d| {d:.1e}")
    lin = lam * linlccd_hh(so_1, e_so, 1.0, 1.0)
    rel = abs(lin - e_a) / abs(e_a)
    check(
        "L3 control: linear lambda E_c(1) misses route A by > 10 %",
        rel > 0.1,
        f"{rel:.2f}",
    )
    h_mo, g_mo = mo_integrals(
        model, ri_factor(model[1], model[2]), ks_scf(model, 1.0, 0.0, 0.0)[1]
    )
    eps_hf = np.array(
        [
            h_mo[p, p]
            + sum(2 * g_mo[p, p, j, j] - g_mo[p, j, j, p] for j in range(NOCC))
            for p in range(N)
        ]
    )
    _, _, _, bos, bss = brute_force(h_mo, g_mo, eps_hf)
    d = abs(e_mp2 - (bos + bss))
    check(
        "L4 lambda = 1, ladder off == brute-force doubles (MP2)",
        d < TOL,
        f"|d| {d:.1e}",
    )
    # Symbolic: the scalings behind L2.
    lm, v, t, gs, den = sp.symbols("lambda V T g D", positive=True)
    b_l = (lm * v) ** sp.Rational(-1, 2) * (lm * t)
    check(
        "L2 sympy: B(lambda) = sqrt(lambda) B",
        sp.simplify(b_l - sp.sqrt(lm) * t / sp.sqrt(v)) == 0,
    )
    e_l = sp.Rational(1, 4) * (lm * gs) ** 2 / den
    check(
        "L2 sympy: E_MP2[lambda g] = lambda^2 E_MP2[g]",
        sp.simplify(e_l - lm**2 * sp.Rational(1, 4) * gs**2 / den) == 0,
    )


if __name__ == "__main__":
    part_p()
    model = part_e()
    part_l(model)
    if FAILED:
        print("FAILED:", *FAILED, sep="\n  ")
        sys.exit(1)
    print("all double-hybrid identities hold")
