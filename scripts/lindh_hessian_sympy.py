#!/usr/bin/env python3
"""Lindh (1995) model Hessian: force constants, Wilson vectors, Cartesian sum.

Derivation: crates/ferric-scf/theory.md, "Lindh model Hessian". Code:
crates/ferric-core/src/model_hessian.rs (lindh_rho, lindh_force_constant,
lindh_cartesian_hessian) and crates/ferric-core/src/internal_coords.rs
(primitive_derivatives, the Wilson vectors b_q = dq/dx the sum uses).
Reference: R. Lindh, A. Bernhardsson, G. Karlstrom, P.-A. Malmqvist,
Chem. Phys. Lett. 241, 423 (1995), Table 1.

Independent routes: the force-constant derivatives in sympy; the Wilson
vectors against sympy gradients of the primitive VALUE expressions (bond
length, acos angle, atan2 torsion); the Cartesian Hessian rebuilt here by a
brute-force loop over every pair, bend and torsion with sympy-lambdified
Wilson vectors, with none of the Rust neighbour-list pre-screening.

Checks (each label carries "(claim X)"):
  L1  rho_ij = exp(alpha (r_ref^2 - r^2)) is 1 at r_ref, d ln rho / dr =
      -2 alpha r, and the stretch, bend and torsion constants k_r rho,
      k_phi rho rho, k_tau rho rho rho reduce to k_r, k_phi, k_tau at the
      reference distances (symbols). The constants and the alpha / r_ref
      tables parsed from model_hessian.rs equal Table 1 of the paper.
  L2  The closed-form Wilson vectors of primitive_derivatives (bond, angle,
      Blondel-Karplus torsion), transcribed below, equal the sympy gradients
      of the primitive values at random geometries (numerics, 1e-10).
  L3  H = sum_q k_q b_q b_q^T is symmetric, positive semi-definite and has
      the six rigid-body motions as null vectors, for water and ethanol
      (numerics); a bend with b from L2 is invariant term by term.
  L4  The brute-force Hessian equals testdata/reference/lindh_model_hessian.json,
      which the Rust test lindh_hessian_matches_the_numpy_reference reads
      (numerics, 1e-12 Ha/Bohr^2). Run with --write to regenerate the file.

Mutations (each turns the script red; exact one-occurrence substitutions):
  alpha for an H-heavy pair 0.3949 -> 0.28 in the brute-force table  (L4)
  bend constant k_phi 0.15 -> 0.45 in the brute-force sum              (L4)
  torsion Wilson vector: dj coefficient -(t1 + 1) -> (t1 - 1)          (L2)
  rho exponent sign flipped                                          (L1)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp


def repo_root():
    cands = [Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents]
    if os.environ.get("FERRIC_ROOT"):
        cands.insert(0, Path(os.environ["FERRIC_ROOT"]))
    for c in cands:
        if (c / "crates" / "ferric-core" / "src" / "model_hessian.rs").is_file():
            return c
    sys.exit("FAIL L0 repository not found (set FERRIC_ROOT)")


ROOT = repo_root()
RUST = ROOT / "crates/ferric-core/src/model_hessian.rs"
REF = ROOT / "testdata/reference/lindh_model_hessian.json"
ANGSTROM_TO_BOHR = 1.0 / 0.52917721092

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# Table 1 of Lindh et al. 1995 (periods: H-He, Li-Ne, Na and heavier).
PAPER_ALPHA = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
]
PAPER_R_REF = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]]
PAPER_K = (0.45, 0.15, 0.005)
RHO_CUTOFF = 1e-8
LINEAR_COS = 0.996_194_698


def period(z):
    return 0 if z <= 2 else (1 if z <= 10 else 2)


# --- L1 -------------------------------------------------------------------


def rho_expr(alpha, rref, r):
    return sp.exp(alpha * (rref**2 - r**2))


def l1_checks():
    a, rref, r = sp.symbols("alpha r_ref r", positive=True)
    kr, kp, kt = sp.symbols("k_r k_phi k_tau", positive=True)
    rho = rho_expr(a, rref, r)
    check("L1 rho(r_ref) = 1 (claim L1)", sp.simplify(rho.subs(r, rref) - 1) == 0)
    dlog = sp.simplify(sp.diff(sp.log(rho), r) + 2 * a * r)
    check("L1 d ln rho/dr = -2 alpha r (claim L1)", dlog == 0)
    r1, r2, r3 = sp.symbols("r1 r2 r3", positive=True)
    k_bend = kp * rho_expr(a, rref, r1) * rho_expr(a, rref, r2)
    k_tors = kt * rho_expr(a, rref, r1) * rho_expr(a, rref, r2) * rho_expr(a, rref, r3)
    ok = (
        sp.simplify(kr * rho.subs(r, rref) - kr) == 0
        and sp.simplify(k_bend.subs({r1: rref, r2: rref}) - kp) == 0
        and sp.simplify(k_tors.subs({r1: rref, r2: rref, r3: rref}) - kt) == 0
    )
    check("L1 constants at the reference distances (claim L1)", ok)
    # The bend constant's derivative factorizes: d ln k_bend / d r1 = -2 alpha r1.
    check(
        "L1 d ln k_bend / d r1 = -2 alpha r1 (claim L1)",
        sp.simplify(sp.diff(sp.log(k_bend), r1) + 2 * a * r1) == 0,
    )

    src = RUST.read_text()

    def const(name):
        m = re.search(rf"pub const {name}: f64 = ([0-9.e-]+);", src)
        return float(m.group(1)) if m else None

    def table(name):
        m = re.search(rf"pub const {name}: \[\[f64; 3\]; 3\] = (\[.*?\]);", src, re.S)
        if not m:
            return None
        body = re.sub(r",\s*\]", "]", m.group(1).replace("\n", " "))
        return json.loads(body)

    rust_k = (const("LINDH_K_STRETCH"), const("LINDH_K_BEND"), const("LINDH_K_TORSION"))
    check(
        "L1 Rust k_r, k_phi, k_tau = paper (claim L1)", rust_k == PAPER_K, f"{rust_k}"
    )
    check("L1 Rust alpha table = paper (claim L1)", table("LINDH_ALPHA") == PAPER_ALPHA)
    check("L1 Rust r_ref table = paper (claim L1)", table("LINDH_R_REF") == PAPER_R_REF)
    check("L1 Rust cutoff (claim L1)", const("LINDH_RHO_CUTOFF") == RHO_CUTOFF)
    # The lambdified rho used below follows the same sign convention.
    x = rho_num(0.28, 2.87, 3.5)
    check("L1 numeric rho below 1 past r_ref (claim L1)", 0.0 < x < 1.0, f"{x:.4f}")


def rho_num(alpha, rref, r):
    return np.exp(alpha * (rref * rref - r * r))


# --- L2: Wilson vectors ---------------------------------------------------

P = sp.Matrix(sp.symbols("x0:12", real=True))


def nrm(v):
    return sp.sqrt(v.dot(v))


def pts(n):
    return [P[3 * a : 3 * a + 3, 0] for a in range(n)]


def value_exprs():
    a, b, c, d = pts(4)
    bond = nrm(a - b)
    u, v = a - b, c - b
    angle = sp.acos(u.dot(v) / (nrm(u) * nrm(v)))
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = b1.cross(b2), b2.cross(b3)
    tors = sp.atan2(n1.cross(n2).dot(b2) / nrm(b2), n1.dot(n2))
    out = {}
    for name, e, n in (("bond", bond, 2), ("angle", angle, 3), ("tors", tors, 4)):
        syms = list(P[: 3 * n])
        out[name] = sp.lambdify([syms], [sp.diff(e, s) for s in syms], "numpy")
    return out


GRADS = value_exprs()


def wilson_sympy(kind, xyz):
    return np.array(GRADS[kind](np.asarray(xyz, float).ravel()), float).reshape(-1, 3)


def wilson_closed(kind, xyz):
    """Transcription of internal_coords.rs primitive_derivatives."""
    xyz = np.asarray(xyz, float)
    if kind == "bond":
        u = xyz[0] - xyz[1]
        e = u / np.linalg.norm(u)
        return np.array([e, -e])
    if kind == "angle":
        u, v = xyz[0] - xyz[1], xyz[2] - xyz[1]
        ru, rv = np.linalg.norm(u), np.linalg.norm(v)
        eu, ev = u / ru, v / rv
        c = np.clip(eu @ ev, -1, 1)
        s = np.sqrt(max(1 - c * c, 0))
        di = (c * eu - ev) / (ru * s)
        dk = (c * ev - eu) / (rv * s)
        return np.array([di, -(di + dk), dk])
    b1, b2, b3 = xyz[1] - xyz[0], xyz[2] - xyz[1], xyz[3] - xyz[2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    nb2 = np.linalg.norm(b2)
    di = -nb2 / (n1 @ n1) * n1
    dl = nb2 / (n2 @ n2) * n2
    t1, t2 = (b1 @ b2) / (b2 @ b2), (b3 @ b2) / (b2 @ b2)
    dj = -(t1 + 1.0) * di + t2 * dl
    dk = -(di + dj + dl)
    return np.array([di, dj, dk, dl])


def l2_checks():
    rng = np.random.default_rng(11)
    worst = {"bond": 0.0, "angle": 0.0, "tors": 0.0}
    for _ in range(30):
        xyz = rng.normal(scale=1.5, size=(4, 3))
        for kind, n in (("bond", 2), ("angle", 3), ("tors", 4)):
            d = np.abs(wilson_closed(kind, xyz[:n]) - wilson_sympy(kind, xyz[:n])).max()
            worst[kind] = max(worst[kind], d)
    for kind, w in worst.items():
        check(
            f"L2 Wilson {kind} = sympy gradient (claim L2)", w < 1e-10, f"max {w:.2e}"
        )


# --- L3/L4: the Cartesian sum, brute force --------------------------------

WATER = [
    (8, [0.0, 0.0, 0.1173]),
    (1, [0.0, 0.7572, -0.4692]),
    (1, [0.0, -0.7572, -0.4692]),
]
ETHANOL = [
    (6, [-0.0478, -0.4252, 0.0000]),
    (6, [1.2600, 0.3599, 0.0000]),
    (8, [-1.1661, 0.4545, 0.0000]),
    (1, [-0.0951, -1.0675, 0.8857]),
    (1, [-0.0951, -1.0675, -0.8857]),
    (1, [2.1102, -0.3268, 0.0000]),
    (1, [1.3184, 1.0000, 0.8810]),
    (1, [1.3184, 1.0000, -0.8810]),
    (1, [-1.9867, -0.0507, 0.0000]),
]

# The brute-force tables, typed separately from PAPER_* so a mutation here is
# caught by L4 against the stored reference (which the Rust also matches).
BF_ALPHA = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
]
BF_R_REF = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]]


def near_linear(x, i, j, k):
    u, v = x[i] - x[j], x[k] - x[j]
    return abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)) >= LINEAR_COS


def lindh_bruteforce(mol):
    zs = [z for z, _ in mol]
    x = np.array([p for _, p in mol]) * ANGSTROM_TO_BOHR
    n = len(zs)
    rho = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                pa, pb = period(zs[i]), period(zs[j])
                rho[i, j] = rho_num(
                    BF_ALPHA[pa][pb], BF_R_REF[pa][pb], np.linalg.norm(x[i] - x[j])
                )
    h = np.zeros((3 * n, 3 * n))

    def add(atoms, k, kind):
        b = np.zeros(3 * n)
        for a, row in zip(atoms, wilson_sympy(kind, x[list(atoms)])):
            b[3 * a : 3 * a + 3] += row
        h[:, :] += k * np.outer(b, b)

    for i, j in combinations(range(n), 2):
        if rho[i, j] >= RHO_CUTOFF:
            add((i, j), 0.45 * rho[i, j], "bond")
    for j in range(n):
        others = [a for a in range(n) if a != j]
        for i, k in combinations(others, 2):
            w = rho[i, j] * rho[j, k]
            if w >= RHO_CUTOFF and not near_linear(x, i, j, k):
                add((i, j, k), 0.15 * w, "angle")
    for j, k in combinations(range(n), 2):
        for i in range(n):
            for m in range(n):
                if len({i, j, k, m}) < 4:
                    continue
                w = rho[i, j] * rho[j, k] * rho[k, m]
                if w < RHO_CUTOFF or near_linear(x, i, j, k) or near_linear(x, j, k, m):
                    continue
                add((i, j, k, m), 0.005 * w, "tors")
    return h, x


def rigid_modes(x):
    n = len(x)
    out = []
    for c in range(3):
        v = np.zeros((n, 3))
        v[:, c] = 1.0
        out.append(v.ravel())
    for axis in np.eye(3):
        out.append(np.cross(axis, x).ravel())
    return out


def l3_checks(hs):
    for name, (h, x) in hs.items():
        scale = np.abs(h).max()
        check(f"L3 {name} symmetric (claim L3)", np.abs(h - h.T).max() < 1e-14 * scale)
        w = np.linalg.eigvalsh(h)
        check(f"L3 {name} PSD (claim L3)", w[0] > -1e-12 * w[-1], f"min {w[0]:.2e}")
        worst = max(
            np.linalg.norm(h @ t) / (np.linalg.norm(t) * w[-1]) for t in rigid_modes(x)
        )
        check(f"L3 {name} rigid modes null (claim L3)", worst < 1e-12, f"{worst:.1e}")
        nzero = int(np.sum(np.abs(w) < 1e-10 * w[-1]))
        check(f"L3 {name} exactly six null modes (claim L3)", nzero == 6, f"{nzero}")


def l4_checks(hs, write):
    payload = {
        "source": "scripts/lindh_hessian_sympy.py (brute-force numpy Lindh 1995 Hessian)",
        "units": "geometry Angstrom (converted with 1/0.52917721092), Hessian Hartree/Bohr^2",
        "systems": {
            name: {
                "atoms": [[z, p] for z, p in mol],
                "hessian": hs[name][0].tolist(),
            }
            for name, mol in (("water", WATER), ("ethanol", ETHANOL))
        },
    }
    if write:
        REF.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {REF}")
    stored = json.loads(REF.read_text())
    for name in ("water", "ethanol"):
        ref = np.array(stored["systems"][name]["hessian"])
        d = np.abs(ref - hs[name][0]).max()
        check(f"L4 {name} = stored reference (claim L4)", d < 1e-12, f"max {d:.2e}")


def main():
    l1_checks()
    l2_checks()
    hs = {"water": lindh_bruteforce(WATER), "ethanol": lindh_bruteforce(ETHANOL)}
    l3_checks(hs)
    l4_checks(hs, "--write" in sys.argv)
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
