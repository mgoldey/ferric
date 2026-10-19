"""Issue #230, the n = 5 E3d experiment: is the E3d residual a genuine 1/N_k (n^-3) term, or an n^-5 tail read
through an extrapolated E_inf?  PRE-REGISTERED below; the design, predictions and decision rule were committed
before any n = 5 E3d number existed.

Fixture: H2/STO-3G, a = 6 Bohr, bond along z, Gamma-centred n x n x n mesh, per-cell MP2, exxdiv ewald, shifted
(Madelung-corrected occupied) denominators.  E3d = the FIXED-k estimator of bloch_head.py:
    E3d = S + L + Qall + (Qlat_direct - Qcub_direct) + F
(cubic head on all channels, Phi-tensor re-weighting of the DIRECT |H|^2 quadratic term only, first-order Fock head).
Anchor of that definition (re-run 2026-10-19 from the saved npz, n = 3 / 4): E3d = -1.3359193576e-2 / -1.3358003359e-2,
E3d(3) - E3d(4) = -1.190e-6, (E3d - E_inf34) n^3 = -5.548e-5 / -5.533e-5 with E_inf34 = (64 S4 - 27 S3)/37
= -1.3357138784e-2 -- the FINDINGS / PR #461 table.

HYPOTHESIS (H-n3, from #230 / FINDINGS): the E3d residual is a genuine n^-3 term, r ~ -5.5e-5 x n^-3 (2.5% of
R = 2.213e-3), as the flat x n^3 at n = 3, 4 suggests.
ALTERNATIVE (H-n5, the 2026-10-17 review): the residual is an n^-5 tail and E_inf is mis-extrapolated; then E3d's own
limit is -1.3357633e-2 (pure n^-5 through n = 3, 4) and nothing is missing at order n^-3.

CORRECTION TO THE PR #461 CRITERION (found while pre-registering, before running).  PR #461 wrote "n^-3 predicts
(E3d - E_inf) n^3 = -5.5e-5 at n = 5, n^-5 predicts about -3.5e-5".  The -3.5e-5 is -5.53e-5 x (4/5)^2, i.e. it
assumes E3d - E_inf34 itself decays as n^-5 -- but then the n = 3 value would have been -5.53e-5 x (4/3)^2 = -9.8e-5,
not the measured -5.55e-5.  That reading is already excluded by n = 3, 4.  The self-consistent pure-n^-5 model fitted
through E3d(3), E3d(4) gives, against the same E_inf34, (E3d(5) - E_inf34) x 125 = -7.69e-5 (and -1.52e-5 against its
own limit).  So the PR's two numbers do not discriminate the two models as stated.  The criterion used here is
E_inf-free.

PREDICTIONS (E_inf-free; Delta_mn = E3d(m) - E3d(n), measured Delta_34 = -1.1902e-6):
    pure n^-3 :  Delta_45 = -4.24e-7   rho = Delta_45 / Delta_34 = 0.3561
    pure n^-5 :  Delta_45 = -2.49e-7   rho = 0.2092
  Equivalent against the fixed E_inf34 (for comparison with PR #461's wording): (E3d(5) - E_inf34) x 125 =
  -5.51e-5 (n^-3) vs -7.69e-5 (pure n^-5); PR #461's -3.5e-5 corresponds to neither self-consistent model.
DECISION RULE (exact three-point fit E3d(n) = E_inf + c3 n^-3 + c5 n^-5 on n = 3, 4, 5):
    SUPPORTED   (genuine n^-3 residual):    c3 in [-7.0e-5, -4.0e-5]   (+-~25% of -5.5e-5)
    REFUTED     (n^-5 / extrapolation):     |c3| <= 1.5e-5             (<= 0.7% of R; the "2.5% residual" is gone)
    UNDETERMINED otherwise, OR if the fit is not self-consistent: its E_inf must agree with the shifted-S series limit
      (S at n = 3..6, forms (3,), (3,5), (3,4) on 3..6 / 4..6; spread 5.9e-7) to within 1.8e-6 (3x that spread);
      a larger gap means terms outside {n^-3, n^-5} matter at n = 3 and a three-point fit cannot decide.
  Bands: numerical noise on E3d is <~1e-12 (SCF 1e-12, FD pair derivatives O(delta^2) ~1e-8 relative on a 1e-5 head),
  so rho is known to ~1e-5; the band widths reflect model inadequacy (n = 3 is pre-asymptotic: in the synthetic Phi
  test the n^-5 piece is as large as the n^-3 piece at n = 3), not noise.  Even SUPPORTED says only that the
  residual is a real 1/N_k term, NOT what explains it (#230's question stays open in that case).
ARTIFACT HYPOTHESES / ANCHORS (must pass before the decision is read):
  N1  the n = 5 pipeline reproduces the INDEPENDENT Iteration-12 n = 5 run (run_kcorr_convergence.py, 2026-09-29):
      S(5) = -1.333924912e-2 and the cubic-head row cub1 = head-restored -1.335689898e-2 (printed to 1e-11).  A broken
      mesh/SCF/pair FT would miss these by >> 1e-11.
  N2  A2 (vectorised MP2 == pbc_kcorr.kmp2, no head and head) < 1e-14; A4 max|P0 - S| < 1e-10; exact quadratic in the
      head scale (cub0.5, cub1, cub2 rows: second difference consistent with cub0.001 to 1e-12 relative of Q).
  N3  bloch_head anchors at n = 5: fixed-k E1 from MO head vectors == row cub1 to < 1e-14; |S_lat - P0| < 1e-10.
  N4  the n = 3, 4 E3d values are re-derived by the SAME analysis code in the same run (must equal the anchor numbers
      above to 1e-13): the analysis cannot silently differ between the old and the new mesh.
  If the implementation (not the physics) were broken at n = 5 I expect N1 to fail; a mis-scaled Nk factor in F or
  the head would move E3d(5) by ~F/125 ~ 5e-6, i.e. rho off by ~10, outside BOTH bands -> UNDETERMINED, not a verdict.

usage:
  e3d_n5.py gen 5        build head_anomaly_a6_n5.npz (lean run_mesh: no Lebedev / Berry rows; ~1-2 h)
  e3d_n5.py analyze      E3d at n = 3, 4, 5, fits, decision
"""

import os
import sys
import time

import numpy as np

PBC = "/home/matt/qc/ferric-pbc/reference/pbc"
sys.path.insert(0, PBC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = f"{PBC}/head_anomaly_data"
A_CELL = 6.0
S_SERIES = {  # shifted MP2 per cell, H2/STO-3G a = 6 z (FINDINGS Iteration 12 + n = 5, 6 addenda)
    3: -1.3275179e-2,
    4: -1.3322562e-2,
    5: -1.333924912e-2,
    6: -1.334680512359e-2,
}
ITER12_N5 = {"sh": -1.333924912e-2, "cub1": -1.335689898e-2}
E3D_ANCHOR = {
    3: -1.3359193576461e-2,
    4: -1.3358003359331e-2,
}  # bloch_head.py re-run 2026-10-19


def gen(m):
    import pbc_kcorr as KC
    import pbc_kpts as PK
    import run_kcorr_head_anomaly as HA
    from pbc_gamma import Cell
    from run_kpts_anchor import H2_ATOMS

    assert os.environ.get("HEAD_ORIENT", "z") == "z"
    out = f"{DATA}/head_anomaly_a{A_CELL:g}_n{m}.npz"
    assert not os.path.exists(out), f"refusing to overwrite {out}"
    cell = Cell(np.eye(3) * A_CELL, H2_ATOMS, "sto-3g")
    gcut = PK.aft_gcut(cell, 1e-10)
    n = (m, m, m)
    t0 = time.time()
    kb = PK.build_k(cell, n, exxdiv="ewald", gcut=gcut, verbose=False)
    e_hf, eps, it, C, occ = PK.krhf(kb, 2, conv=1e-12, kshift=0.0, return_mo=True)
    assert all(int(o.sum()) == 1 and o[0] for o in occ)
    vm = kb["madelung"]
    t1 = time.time()
    print(f"scf {t1 - t0:.0f}s HF {e_hf:.12f} it {it} v_M {vm:.10f}", flush=True)
    st = KC.aft_ov(cell, n, C, 1, gcut=gcut)
    st["Co"], st["Cv"] = KC._mo_stacks(C, 1)
    t2 = time.time()
    print(f"aft_ov {t2 - t1:.0f}s", flush=True)
    eo, ev = KC.k_denominators(eps, 1, vm, "shifted")
    P0, A, B = HA.pair_derivs(cell, n)
    Nk = st["Nk"]
    a4 = max(abs(P0[k] - kb["S"][k]).max() for k in range(Nk))
    e_sh = HA.energies(st, eo, ev)
    a2 = abs(e_sh[0] - KC.kmp2(st, eo, ev)[0])
    cub = A * np.sqrt(4 * np.pi / (3 * cell.vol))
    rows = {"sh": e_sh}
    for lam in (1.0, 0.5, 2.0, 1e-3):
        rows[f"cub{lam:g}"] = HA.energies(HA.with_head(st, cub, lam), eo, ev)
    a2h = abs(rows["cub1"][0] - KC.kmp2(HA.with_head(st, cub), eo, ev)[0])
    deo, dev = HA.fock_head_shifts(cell, st, P0, A, B, C, 1)
    eoF = [x - y for x, y in zip(eo, deo)]
    evF = [x - y for x, y in zip(ev, dev)]
    rows["fock"] = HA.energies(st, eoF, evF)
    rows["fock+cub"] = HA.energies(HA.with_head(st, cub), eoF, evF)
    print(f"anchors: A4 max|P0-S| {a4:.1e}; A2 {a2:.1e} (no head) {a2h:.1e} (head)")
    for key, val in rows.items():
        print(f"   {key:10s} MP2 {val[0]:.12e}  dRPA {val[1]:.12e}", flush=True)
    np.savez(
        out,
        rows=np.array([list(v) for v in rows.values()]),
        keys=np.array(list(rows.keys())),
        eps=np.array(eps),
        vm=vm,
        deo=np.array(deo),
        dev=np.array(dev),
        e_hf=e_hf,
        V=st["V"],
        Kq=st["Kq"],
        C=np.array(C),
        A=A,
        B=B,
        P0=P0,
        anchors=np.array([a4, a2, a2h]),
    )
    print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)


def e3d(m):
    """(E3d_fixed, E3d_true, E1, row, info) at mesh m, the bloch_head.main() construction verbatim."""
    from bloch_head import head_energies, head_vectors, phi_tensor
    from cheap_account import load, st_from
    from pbc_gamma import Cell
    from vao_build import h2_atoms

    cell = Cell(np.eye(3) * A_CELL, h2_atoms("z"), "sto-3g")
    z, row = load(A_CELL, m, "")
    st = st_from(z, cell, m)
    eps, vm = z["eps"], float(z["vm"])
    eo = [e[:1] - vm for e in eps]
    ev = [e[1:] for e in eps]
    S0 = row["sh"][0]
    F = row["fock"][0] - S0
    gfo, gfv, gto_, gtv, info = head_vectors(z, cell, m)
    out = {}
    for tag, (go, gv) in (("fixed", (gfo, gfv)), ("true", (gto_, gtv))):
        L, Qc, Ql, Qall = head_energies(st, go, gv, eo, ev, cell.vol, phi_tensor())
        out[tag] = (S0 + L + Qall + (Ql - Qc) + F, S0 + L + Qall)
    info["E1_anchor"] = abs(out["fixed"][1] - row["cub1"][0])
    return out["fixed"][0], out["true"][0], out["fixed"][1], row, info


def fit(ns, ys, powers):
    A = np.array([[1.0] + [float(x) ** -p for p in powers] for x in ns])
    c, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
    return c


def analyze():
    res = {m: e3d(m) for m in (3, 4, 5)}
    print("== anchors")
    for m in (3, 4):
        d = abs(res[m][0] - E3D_ANCHOR[m])
        print(
            f"  N4 n={m}: E3d {res[m][0]:.13e} vs anchor {E3D_ANCHOR[m]:.13e}  |diff| {d:.1e}"
        )
        assert d < 1e-13
    row5, info5 = res[5][3], res[5][4]
    for k, ref in ITER12_N5.items():
        d = abs(row5[k][0] - ref)
        print(
            f"  N1 n=5 row {k}: {row5[k][0]:.12e} vs Iteration-12 {ref:.11e}  |diff| {d:.1e}"
        )
        assert d < 1e-11, k
    q = {lam: row5[f"cub{lam:g}"][0] - row5["sh"][0] for lam in (1e-3, 0.5, 1.0, 2.0)}
    Q = (q[2.0] - 2 * q[1.0]) / 2
    lin = q[1.0] - Q
    quad_err = max(abs(q[lam] - (lam * lin + lam**2 * Q)) for lam in q) / abs(Q)
    print(f"  N2 n=5 exact quadratic in head scale: max rel err {quad_err:.1e}")
    assert quad_err < 1e-6
    print(
        f"  N3 n=5: |S_lat - P0| {info5['S_err']:.1e}; fixed-k E1 (MO) - cub1 {info5['E1_anchor']:.1e}; "
        f"Fourier-dS aliasing (rel) {info5['dS_alias']:.2e}"
    )
    assert info5["S_err"] < 1e-10 and info5["E1_anchor"] < 1e-14

    print("== raw E3d (fixed-k, decision-bearing) and E3d(true) (H5, reported only)")
    E = {m: res[m][0] for m in (3, 4, 5)}
    for m in (3, 4, 5):
        print(
            f"  n={m}: E3d {res[m][0]:.13e}  E3d(true) {res[m][1]:.13e}  E1 {res[m][2]:.13e}  S {res[m][3]['sh'][0]:.12e}"
        )
    d34, d45 = E[3] - E[4], E[4] - E[5]
    rho = d45 / d34
    print(
        f"  Delta_34 {d34:+.5e}  Delta_45 {d45:+.5e}  rho {rho:.4f}  (pure n^-3 0.3561, pure n^-5 0.2092)"
    )
    einf34 = (64 * S_SERIES[4] - 27 * S_SERIES[3]) / 37
    for m in (3, 4, 5):
        print(
            f"  (E3d - E_inf34) x n^3 at n={m}: {(E[m] - einf34) * m**3:+.4e}   [E_inf34 {einf34:.10e}]"
        )

    print("== fits")
    c = fit((3, 4, 5), [E[3], E[4], E[5]], (3, 5))
    print(
        f"  E3d exact (3,5) fit on 3..5: E_inf {c[0]:.10e} c3 {c[1]:+.4e} c5 {c[2]:+.4e}"
    )
    c34 = fit((3, 4, 5), [E[3], E[4], E[5]], (3, 4))
    print(
        f"  E3d exact (3,4) fit on 3..5 (diagnostic): E_inf {c34[0]:.10e} c3 {c34[1]:+.4e} c4 {c34[2]:+.4e}"
    )
    sinf = []
    for pw, idx in (
        ((3,), (3, 4, 5, 6)),
        ((3, 5), (3, 4, 5, 6)),
        ((3, 4), (3, 4, 5, 6)),
        ((3, 5), (4, 5, 6)),
        ((3,), (4, 5, 6)),
    ):
        cs = fit(idx, [S_SERIES[i] for i in idx], pw)
        sinf.append(cs[0])
        print(
            f"  S fit powers {pw} n={idx}: E_inf {cs[0]:.10e} coefs {np.array2string(cs[1:], precision=4)}"
        )
    lo, hi = min(sinf), max(sinf)
    gap = 0.0 if lo <= c[0] <= hi else min(abs(c[0] - lo), abs(c[0] - hi))
    print(
        f"  S-series E_inf range [{lo:.10e}, {hi:.10e}] (spread {hi - lo:.1e}); E3d-fit E_inf outside it by {gap:.1e}"
    )
    for m in (3, 4, 5):
        print(
            f"  (E3d - E_inf[S range]) x n^3 at n={m}: [{(E[m] - hi) * m**3:+.3e}, {(E[m] - lo) * m**3:+.3e}]"
        )

    print("== decision (pre-registered)")
    c3 = c[1]
    if gap > 1.8e-6:
        verdict = "UNDETERMINED (E3d fit not consistent with the S-series limit)"
    elif -7.0e-5 <= c3 <= -4.0e-5:
        verdict = "SUPPORTED: genuine n^-3 residual"
    elif abs(c3) <= 1.5e-5:
        verdict = "REFUTED: no n^-3 residual (n^-5 tail / extrapolation)"
    else:
        verdict = "UNDETERMINED (c3 between the bands)"
    print(f"  c3 = {c3:+.4e} -> {verdict}")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]))
    elif sys.argv[1] == "analyze":
        analyze()
    else:
        raise SystemExit(__doc__)
