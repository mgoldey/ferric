#!/usr/bin/env python3
"""Laplace-transformed chi0: the kernel identity, the gap scaling, the dielectric.

Derivation: crates/ferric-rpa/theory.md, "Laplace-transformed chi0". Code:
crates/ferric-rpa/src/laplace_chi0.rs (build_laplace_for_gaps,
dielectric_matrix_laplace_into, dielectric_matrix_laplace_unrestricted_into,
accumulate_one_spin), crates/ferric-quadrature/src/minimax.rs
(LaplaceQuadrature::new, select_minimax_points; the R = 5 table entries
are transcribed below and checked against that file) and crates/ferric-rpa/src/sternheimer.rs
(build_scale_factors_with_prefactor, the dense kernel).

As coded, for a trial block V (naux x m), Y = V^T B (m x n_ov):
  dense:    eps_U(iu) = 1 + Y diag(c Delta/(u^2 + Delta^2)) Y^T, c = 4 (closed
            shell), 2 per spin (open shell, summed over spins)
  Laplace:  t_l = t_l^tab/Delta_min, w_l = w_l^tab/Delta_min, the minimax
            1/y table on [1, R] picked as the first entry with R_tab >= 0.99 R,
            R = Delta_max/Delta_min; then
            eps_U(iu) = 1 + sum_l c w_l cos(u t_l) X_l X_l^T,
            X_l = Y diag(exp(-t_l Delta/2));
            if u t_max > pi/2 (t_max over both spins when open shell) the
            dense kernel is used instead, for both spins.

Checks (each label carries "(claim Ln)"):
  L0  The transcribed table entries equal crates/ferric-quadrature/src/minimax.rs
      (run when the repository is reachable; the mutation runner's copies are
      not inside it).
  L1  int_0^oo exp(-Delta t) cos(u t) dt = Delta/(u^2 + Delta^2) and
      sum_l w_l cos(u t_l) exp(-t_l Delta) = Re r(Delta + i u) with
      r(z) = sum_l w_l exp(-t_l z) (symbolic); exp(-t Delta) =
      exp(t eps_i) exp(-t eps_a) and = exp(-t Delta/2)^2 (the X_l X_l^T split).
  L2  Gap scaling: r(x) = (1/Delta_min) r_tab(x/Delta_min) for the scaled
      nodes, so the relative 1/x error on [Delta_min, Delta_max] is the
      table's relative error on [1, R]; numerically against the table.
  L3  The coded Laplace dielectric equals 1 + c Y diag(Re r(Delta + iu)) Y^T to
      rounding (independent route: the complex rational form, one matmul),
      and at u = 0 it differs from the dense dielectric by exactly
      c Y diag(r(Delta) - 1/Delta) Y^T (the minimax error, nothing else; on
      the 3-point table, whose error is far above rounding). Exactness
      anchor: the identity is exact for any (t_l, w_l), so a wrong prefactor
      or exponent split cannot hide behind the quadrature error.
  L4  Accuracy at u = 0: the coded kernel's relative error at every gap is at
      most the table's worst relative error on [1, R] (sampled densely), for
      n = 3, 5, 7, and the dielectric error falls 3 -> 5 -> 7. The relative
      kernel error at the edge u t_max = pi/2 is printed (a measurement).
  L5  Fallback rule: the Laplace branch is taken only when u t_max <= pi/2, so
      every cos(u t_l) >= 0 there; above it the coded result equals the dense
      dielectric exactly. Open shell: the threshold uses the larger t_max of
      the two spins, and both spins fall back together.
  L6  Open shell: with identical alpha and beta channels, 2 + 2 per spin equals
      the closed-shell 4, for the Laplace and the dense branch.

Mutations (each turns the script red; exact one-occurrence substitutions):
  closed-shell coefficient 4 w_l cos -> 2 w_l cos               (L3 fails)
  X_l scaled by exp(-t Delta) instead of exp(-t Delta/2)        (L3 fails)
  nodes scaled t*Delta_min instead of t/Delta_min               (L2 fails)
  fallback threshold pi/2 -> pi                                 (L5 fails)
  open-shell coefficient 2 -> 4                                 (L6 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import math
import re
import sys
from pathlib import Path

import numpy as np
import sympy as sp

failures = []
EPS = np.finfo(float).eps
MINIMAX_REL = Path("crates") / "ferric-quadrature" / "src" / "minimax.rs"


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# --- the minimax tables -----------------------------------------------------------
#
# The R = 5 entries of crates/ferric-quadrature/src/minimax.rs (the only ones the
# model's gap range reaches) and every tabulated R, transcribed. Claim L0 checks
# them against the Rust source when the repository is reachable (the runner's
# mutated copies live outside it).

R_TABLE = {
    3: [5.0, 10.0, 20.0, 50.0, 100.0],
    5: [5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0],
    7: [5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0],
}
R5 = {
    3: (
        [0.16607313750141492, 0.9784372085453776, 3.0530159808682455],
        [0.4365792105033684, 1.2723305210744182, 3.2151540832501007],
    ),
    5: (
        [
            0.10333512117595546,
            0.5658998554798739,
            1.500259486444618,
            3.156388982390542,
            6.193584172208591,
        ],
        [
            0.2675144938257101,
            0.6731648185410648,
            1.2341495375414908,
            2.173074998760648,
            4.240049104595772,
        ],
    ),
    7: (
        [
            0.07517801205393558,
            0.40394057377108084,
            1.0300953839123785,
            2.0257208844822334,
            3.5300416178849785,
            5.8202600945046035,
            9.571879302298791,
        ],
        [
            0.19380100796794852,
            0.4692435586300529,
            0.7946376596686956,
            1.2188241997775222,
            1.8330663504453952,
            2.844141721918306,
            5.0081428000286365,
        ],
    ),
}


def parse_rust_tables(path):
    src = path.read_text(encoding="utf-8")
    tables = {}
    for k, body in re.findall(
        r"static MINIMAX_K(\d+): &\[MinimaxEntry\] = &\[(.*?)\n\];", src, re.S
    ):
        rows = re.findall(
            r"\(\s*([0-9.e+]+),\s*&\[(.*?)\],\s*&\[(.*?)\],\s*\)", body, re.S
        )
        tables[int(k)] = [
            (
                float(r),
                [float(v) for v in t.replace(",", " ").split()],
                [float(v) for v in w.replace(",", " ").split()],
            )
            for r, t, w in rows
        ]
    return tables


# --- transcriptions ---------------------------------------------------------------


def select_minimax_points(k, r):
    for r_tab in R_TABLE[k]:
        if r_tab >= r * 0.99:
            if r_tab != 5.0:
                raise ValueError("only the R = 5 entries are transcribed")
            t, w = R5[k]
            return np.array(t), np.array(w), r_tab
    raise ValueError("R exceeds the table")


def laplace_quadrature(n_quad, ymin, ymax):
    t, w, r_tab = select_minimax_points(n_quad, ymax / ymin)
    return t / ymin, w / ymin, r_tab


def build_laplace_for_gaps(eo, ev, n_quad):
    ymin = min(ev) - max(eo)
    ymax = max(ev) - min(eo)
    return laplace_quadrature(n_quad, ymin, ymax)


def gaps(eo, ev):
    return np.array([a - i for i in eo for a in ev])


def dense_dielectric(v, b, eo, ev, u, pref=4.0):
    e = gaps(eo, ev)
    y = v.T @ b
    return np.eye(v.shape[1]) + (y * (pref * e / (u * u + e * e))) @ y.T


def dense_dielectric_unrestricted(v, chans, u):
    m = v.shape[1]
    out = np.eye(m)
    for b, eo, ev in chans:
        out += dense_dielectric(v, b, eo, ev, u, pref=2.0) - np.eye(m)
    return out


def laplace_dielectric(v, b, eo, ev, u, quad):
    t_pts, w_pts, _ = quad
    t_max = max(t_pts)
    if u * t_max > math.pi / 2:
        return dense_dielectric(v, b, eo, ev, u)
    y = v.T @ b
    e = gaps(eo, ev)
    out = np.zeros((v.shape[1], v.shape[1]))
    for t_l, w_l in zip(t_pts, w_pts):
        x = y * np.exp(-0.5 * t_l * e)
        coeff = 4.0 * w_l * math.cos(u * t_l)
        out += coeff * (x @ x.T)
    return out + np.eye(v.shape[1])


def accumulate_one_spin(v, chan, quad, u, out):
    b, eo, ev = chan
    if len(eo) * len(ev) == 0:
        return
    t_pts, w_pts, _ = quad
    y = v.T @ b
    e = gaps(eo, ev)
    for t_l, w_l in zip(t_pts, w_pts):
        x = y * np.exp(-0.5 * t_l * e)
        coeff = 2.0 * w_l * math.cos(u * t_l)
        out += coeff * (x @ x.T)


def laplace_dielectric_unrestricted(v, chans, quads, u):
    t_max = max(max(q[0]) for q in quads)
    if u * t_max > math.pi / 2:
        return dense_dielectric_unrestricted(v, chans, u)
    out = np.zeros((v.shape[1], v.shape[1]))
    for chan, quad in zip(chans, quads):
        accumulate_one_spin(v, chan, quad, u, out)
    return out + np.eye(v.shape[1])


# --- model ------------------------------------------------------------------------

RNG = np.random.default_rng(20261115)
EO = np.array([-0.62, -0.41])
EV = np.array([0.18, 0.66, 1.43])
NAUX, M = 6, 4
B = RNG.normal(scale=0.2, size=(NAUX, EO.size * EV.size))
V = np.linalg.qr(RNG.normal(size=(NAUX, M)))[0]


def kernel_r(t, w, z):
    return np.sum(w * np.exp(-t * z))


# --- checks -----------------------------------------------------------------------


def claim_l0():
    found = [
        p / MINIMAX_REL
        for p in (Path(__file__).resolve().parents[1], Path.cwd())
        if (p / MINIMAX_REL).is_file()
    ]
    if not found:
        print("skip L0 transcribed tables vs minimax.rs (repository not reachable)")
        return
    rust = parse_rust_tables(found[0])
    same = all(
        [e[0] for e in rust[k]] == R_TABLE[k]
        and (rust[k][0][1], rust[k][0][2]) == R5[k]
        for k in (3, 5, 7)
    )
    check(
        "L0 transcribed minimax tables = minimax.rs (claim L0)",
        same,
        f"source {found[0]}",
    )


def claim_l1():
    d, u, t, ei, ea = sp.symbols("Delta u t epsilon_i epsilon_a", positive=True)
    lhs = sp.integrate(sp.exp(-d * t) * sp.cos(u * t), (t, 0, sp.oo))
    ident = sp.simplify(lhs - d / (u**2 + d**2)) == 0
    tl, wl = sp.symbols("t_l w_l", positive=True)
    real_part = (
        sp.simplify(
            sp.re(sp.expand_complex(wl * sp.exp(-tl * (d + sp.I * u))))
            - wl * sp.cos(u * tl) * sp.exp(-tl * d)
        )
        == 0
    )
    split = (
        sp.simplify(sp.exp(tl * ei) * sp.exp(-tl * ea) - sp.exp(-tl * (ea - ei))) == 0
    )
    half = sp.simplify(sp.exp(-tl * d / 2) ** 2 - sp.exp(-tl * d)) == 0
    check(
        "L1 cosine-Laplace identity and the exponential splits (claim L1)",
        ident and real_part and split and half,
        f"integral {ident} Re-form {real_part} occ/vir split {split} half split {half}",
    )


def claim_l2():
    eo, ev = EO, EV
    t, w, r_tab = build_laplace_for_gaps(eo, ev, 7)
    ymin = ev.min() - eo.max()
    ymax = ev.max() - eo.min()
    t_tab, w_tab, _ = select_minimax_points(7, ymax / ymin)
    worst = 0.0
    for x in np.linspace(ymin, ymax, 201):
        scaled = kernel_r(t, w, x)
        table = kernel_r(t_tab, w_tab, x / ymin) / ymin
        worst = max(worst, abs(scaled - table) / abs(table))
    # Rounding bound: 7 terms, each exp and two divisions.
    check(
        "L2 gap scaling: r(x) = r_tab(x/Delta_min)/Delta_min (claim L2)",
        worst < 50 * EPS,
        f"worst relative {worst:.1e}; R = {ymax / ymin:.2f}, table R {r_tab}",
    )


def claim_l3():
    quad = build_laplace_for_gaps(EO, EV, 7)
    t, w, _ = quad
    e = gaps(EO, EV)
    y = V.T @ B
    worst = 0.0
    for frac in (0.0, 0.3, 0.9):
        u = frac * (math.pi / 2) / max(t)
        coded = laplace_dielectric(V, B, EO, EV, u, quad)
        rre = np.array([np.sum(w * np.exp(-t * (ei + 1j * u))).real for ei in e])
        indep = np.eye(M) + (y * (4.0 * rre)) @ y.T
        scale = np.abs(y).max() ** 2 * 4.0 * np.abs(rre).sum() + 1.0
        worst = max(worst, np.abs(coded - indep).max() / (len(t) * 20 * EPS * scale))
    # u = 0 against the dense dielectric, on the 3-point table whose minimax
    # error (~1e-3) is far above rounding, so "exactly the minimax error" bites.
    q3 = build_laplace_for_gaps(EO, EV, 3)
    coded0 = laplace_dielectric(V, B, EO, EV, 0.0, q3)
    dense0 = dense_dielectric(V, B, EO, EV, 0.0)
    r0 = np.array([kernel_r(q3[0], q3[1], ei) for ei in e])
    pred = (y * (4.0 * (r0 - 1.0 / e))) @ y.T
    diff0 = np.abs((coded0 - dense0) - pred).max()
    bound0 = 3 * 20 * EPS * (np.abs(dense0).max() + 1.0)
    check(
        "L3 Laplace dielectric = 1 + 4 Y Re r(Delta+iu) Y^T (claim L3)",
        worst < 1.0 and diff0 < bound0 and np.abs(pred).max() > 1e3 * bound0,
        f"worst err/bound {worst:.2f}; u=0 residual vs minimax error {diff0:.1e}",
    )


def claim_l4():
    # Accuracy at u = 0: the coded kernel's relative error at every model gap
    # is at most the table's worst relative error on [1, R] (sampled on a fine
    # grid), for every tabulated size; and the dielectric error falls 3 -> 5 -> 7.
    ymin = EV.min() - EO.max()
    e = gaps(EO, EV)
    rows, ok, derr = [], True, []
    for n in (3, 5, 7):
        t, w, r_tab = build_laplace_for_gaps(EO, EV, n)
        sup = max(
            abs(y * kernel_r(t * ymin, w * ymin, y) - 1.0)
            for y in np.linspace(1.0, r_tab, 4001)
        )
        worst = max(abs(ei * kernel_r(t, w, ei) - 1.0) for ei in e)
        ok = ok and worst <= sup * (1 + 1e-9)
        rows.append(f"n={n} {worst:.1e}<={sup:.1e}")
        derr.append(
            np.abs(
                laplace_dielectric(V, B, EO, EV, 0.0, (t, w, r_tab))
                - dense_dielectric(V, B, EO, EV, 0.0)
            ).max()
        )
    falls = derr[0] > derr[1] > derr[2]
    # At the edge of the Laplace region the kernel is still accurate (measured,
    # reported): relative error of Re r(Delta + iu) at u t_max = pi/2.
    t, w, _ = build_laplace_for_gaps(EO, EV, 7)
    u_edge = (math.pi / 2) / max(t)
    edge = max(
        abs(
            np.sum(w * np.cos(u_edge * t) * np.exp(-t * ei))
            / (ei / (u_edge**2 + ei**2))
            - 1.0
        )
        for ei in e
    )
    check(
        "L4 kernel error within the table's on [1, R]; falls with n (claim L4)",
        ok and falls,
        "; ".join(rows)
        + f"; dielectric errors {[f'{x:.1e}' for x in derr]}; edge-u kernel rel err {edge:.1e}",
    )


def claim_l5():
    quad = build_laplace_for_gaps(EO, EV, 7)
    t, w, _ = quad
    t_max = max(t)
    taken = [u for u in np.linspace(0.0, 3.0, 301) if u * t_max <= math.pi / 2]
    cos_ok = all(math.cos(u * tl) >= -1e-15 for u in taken for tl in t)
    u_hi = 1.01 * math.pi / 2 / t_max
    fb = np.array_equal(
        laplace_dielectric(V, B, EO, EV, u_hi, quad),
        dense_dielectric(V, B, EO, EV, u_hi),
    )
    # Open shell: beta has its own (narrower) gap range, so its own t_max.
    eob, evb = np.array([-0.55]), np.array([0.21, 0.97])
    bb = RNG.normal(scale=0.2, size=(NAUX, 2))
    qb = build_laplace_for_gaps(eob, evb, 7)
    t_both = max(max(t), max(qb[0]))
    u_mid = 0.5 * (math.pi / 2 / t_both + math.pi / 2 / min(max(t), max(qb[0])))
    chans = [(B, EO, EV), (bb, eob, evb)]
    both_fb = np.array_equal(
        laplace_dielectric_unrestricted(V, chans, [quad, qb], u_mid),
        dense_dielectric_unrestricted(V, chans, u_mid),
    )
    nontrivial = math.pi / 2 / t_both < u_mid
    check(
        "L5 fallback u t_max > pi/2 -> dense, cos >= 0 on the Laplace side (claim L5)",
        cos_ok and fb and both_fb and nontrivial,
        f"cos>=0 {cos_ok}; closed-shell fallback exact {fb}; open-shell joint fallback {both_fb}",
    )


def claim_l6():
    quad = build_laplace_for_gaps(EO, EV, 7)
    chans = [(B, EO, EV), (B, EO, EV)]
    worst = 0.0
    for u in (0.0, 0.05, 0.2, 5.0):
        cs = laplace_dielectric(V, B, EO, EV, u, quad)
        os_ = laplace_dielectric_unrestricted(V, chans, [quad, quad], u)
        worst = max(worst, np.abs(cs - os_).max())
    check(
        "L6 open shell 2 + 2 = closed shell 4, Laplace and dense branch (claim L6)",
        worst < 1e-13,
        f"max difference {worst:.1e}",
    )


def main():
    claim_l0()
    claim_l1()
    claim_l2()
    claim_l3()
    claim_l4()
    claim_l5()
    claim_l6()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all Laplace chi0 identities verified")


if __name__ == "__main__":
    main()
