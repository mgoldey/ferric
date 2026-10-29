"""Issue #230, FIQ (Fixed Inner problem, q-refined): is the ~3-5% n^-3 residual of the k-MP2 q = 0 head an
ERI-side over-correction of the coded head (reading (i)), a missing n^-3 term (reading (ii)), or mean-field?
Design: issue #230 comment of 2026-10-29 (pre-registration) and design_230.md (scratchpad copy of that design).

PRE-REGISTRATION (2026-10-29). Step 0 below ran from the saved npz only. Bands, gates and the decision rule were
written and committed BEFORE any FIQ number (any off-mesh integrand value) existed.

Fixture: H2/STO-3G, a = 6 Bohr, bond along z and [111], per-cell MP2, shifted (Madelung) denominators, n0 = 3 inner
mesh (n0 = 4 = replicate). 'direct' = the J-type channel 2|V|^2/D = 2 x OS of cheap_account.kmp2_split;
exchange = SS - OS. Coefficients are "x n^-3" (Ha per cell).

---------------------------------------------------------------- STEP 0 (measured 2026-10-29, npz only, `fiq.py step0`)
Estimator pieces from bloch_head (head vectors) + cheap_account (channel split), E_inf per channel from the shifted
S at n = 3, 4 (pure n^-3: (64 S4 - 27 S3)/37), X = (E3d - E_inf) n^3. Variants: ERI head with fixed-k or Bloch-paired
('true', 'none' interpolant) vectors; first-order Fock head fixed-k (npz deo/dev) or Bloch-paired (dK_*.npz true_none).
Anchors: S, cub1, fock rows reproduced to < 1e-14; Qall_OS x 2 == Qc; fixed-k dK diag/2 == npz deo/dev to 9e-19.

  X (x n^-3)                       z n0=3      z n0=4      [111] n0=3  [111] n0=4
  direct, fixed ERI, fixed F       +2.1295e-4  +2.0400e-4  -7.8889e-5  -9.2158e-5
  direct, true ERI,  fixed F       +2.6128e-4  +2.4221e-4  -6.6006e-5  -8.5770e-5
  direct, true ERI,  true F (*)    +2.4835e-4  +2.2244e-4  -8.6531e-5  (no dK npz)
  total,  fixed ERI, fixed F       -5.5494e-5  -5.5368e-5  -1.8376e-4  -1.8861e-4   (= FINDINGS: anchor)
  total,  true ERI,  true F        -4.0171e-5  -4.7312e-5  -1.8373e-4
  P_code (direct, true vectors)    +3.9990e-3  +4.0122e-3  +5.3606e-3  +5.3734e-3   = -n^3 (L_dir + Qlat_dir)
  P_code (direct, fixed vectors)   +4.0473e-3  +4.0504e-3  +5.3734e-3  +5.3798e-3
(*) PRIMARY X_dir: the ERI head the FIQ compares against (true vectors) plus the best npz-available mean-field
side (Bloch-paired first-order Fock head). Orbital relaxation (FINDINGS (e), total -2.10e-5 z / -1.64e-5 [111], no
channel split) needs the V_AO kernels and is NOT included.

STEP-0 FINDING (not in the design): the residual does NOT live in the direct channel at the size of the total.
At z the direct and exchange residuals are each ~4-5x the total with opposite signs (+2.1e-4 vs -2.7e-4).
The direct-channel X is also not a stable n^-3 coefficient. From the n = 5 npz (z, fixed variant, `fiq.py step0n5`):
  E3d_dir(3, 4, 5) = -2.652973815e-2, -2.653443764e-2, -2.653396195e-2; rho = D45/D34 = -0.101 (pure n^-3 0.356);
  the exact 3-point fit gives c3 = -4.64e-4 (c5 = +4.66e-3), against c3 = +2.19e-4 from (3,4). The total is unchanged
  (rho 0.506, c3 -1.124e-4 = e3d_n5.py). So the SIGN of X_dir(z) depends on the E_inf model.

---------------------------------------------------------------- BANDS AND GATES (fixed before any FIQ number)
Statistic (Stage 1): Delta = P_num - P_code(true, 'none'), direct channel, inner n0 = 3 (n0 = 4 replicate).
  P_num = -Phi[F_lim - r0]. F_lim(u) = Richardson (t^2, t^4 removed) of F(t u), t = 0.04, 0.02, 0.01 Bohr^-1.
  r0 = F(0) in the S convention (K = 0 dropped). Phi is applied to the Lebedev projections onto l = 0 and the cubic
  K4 = sum u_a^4 - 3/5, with Phi[K4] = 3 Phi_aaaa - 3/5. The l = 2 parts and the non-cubic l = 4 parts have Phi = 0.
"~ X":  Delta in [0.45 X_dir, 1.45 X_dir], same sign, X_dir = PRIMARY above:
          z      [+1.1176e-4, +3.6011e-4]        [111]  [-1.2547e-4, -3.8939e-5]
"~ 0":  |Delta| <= 0.15 |X|min. |X|min is the smallest magnitude among ALL the step-0 X definitions in the table
        (direct variants and totals, n0 = 3), because X_dir is E_inf-fragile (above). This is STRICTER than 0.15 |X_dir|.
        It stops an inflated X_dir from manufacturing a "~ 0":
          z      |X|min = 4.0171e-5 -> |Delta| <= 6.03e-6      [111]  |X|min = 6.6006e-5 -> |Delta| <= 9.90e-6
Precision gates (scale |X|min; any failure -> UNDETERMINED (gate)):
  G_t    P_num from t = {0.04, 0.02, 0.01} vs {0.02, 0.01, 0.005}:     <= 0.05 |X|min (z 2.01e-6, [111] 3.30e-6)
  G_leb  P_num from Lebedev 50 vs Lebedev 38 (quadrature order):       <= 0.05 |X|min
  G_int  |Delta('ewald' interpolant) - Delta('none')|:                 <= 0.30 |X|min (z 1.21e-5, [111] 1.98e-5)
  (Stage 2 only, not run here: R_q fit windows 4..12 / 6..12, odd / even n_q, Duffy N = 12 vs 16: <= 0.05 |X|min)
Anchors (must all pass before Delta is read):
  A0  n_q = n0 (3 and 4, both orientations): FIQ per-q direct energies == the stored V/eps/vm values, and the total ==
      stored 2 x S_OS, to <= 1e-12.
  A1  one generic off-mesh quartet (k_i = k_j = Gamma, k_a = (0.137, -0.071, 0.213).b, k_b = -k_a) == PySCF 2.13
      AFTDF.get_eri MO-transformed with the same C, to <= 1e-10.
  A2  Duffy integrator on the realistic_check5 synthetic: N = 12 vs 16 relative <= 1e-9; a smooth periodic function
      integrates exactly (<= 1e-13). (Stage 2 machinery; here it carries the no-Duffy mutant.)
  A4  fixed-k pairing (the 'fixed' head vectors, exact by definition): P_num(fixed) == P_code(fixed) to <= 1e-6 rel.
  A5  body continuity: F_lim at head scale lambda = 0 equals r0 to <= 0.01 |X|min. Time reversal: F(-q) == F(q) to
      <= 1e-12 relative (pilot). The second justifies evaluating one member of each Lebedev inversion pair.
Reachability: inject kappa (sum u^4 - 3/5) e^{-q^2/sig^2} (l = 4), and separately an l = 0 term, into the REAL
  Stage-1 F values. Each is sized so that it moves P_num by exactly X_dir. Recovery must be within 0.05 |X|min,
  i.e. inside the "~ X" band.
Mutants (each must fail an anchor): no Duffy s^2 Jacobian (A2); Bloch phase of leg 1 at k_i instead of k_i + q,
  i.e. the wrong pair FT k' (A0); band interpolant without the Wigner-Seitz half weights on the boundary aliases,
  i.e. the wrong interpolant (A0 at n0 = 4). Plus M1: fixed-k pairing moves Delta by the H5 amount
  (P_code true - fixed = -4.8e-5 z), which shows Stage 1 can see a Bloch-pairing-sized error.
Decision (Stage 1, n0 = 3; z and [111] must land in the same row, else UNDETERMINED):
  both "~ 0"  -> ERI-side reading (i) REFUTED: the coded head equals the numerical q -> 0 limit. (ii)-ERI vs
                 mean-field needs Stage 2 (D_num).
  both "~ X"  -> (i) indicated, but reported as UNDETERMINED pending an E_inf-robust X_dir(z), because X_dir(z)
                 changes sign with the E_inf model. If Delta = X_dir to within 2% at both orientations, audit before
                 reading it ("too clean").
  otherwise   -> UNDETERMINED.
Delta is also reported as a fraction of X_tot (descriptive only, no verdict attached).
Timing: a 10-q pilot calibrates seconds per q. Pre-registered abort (design): if > 20 s per q, lower gcut and re-check
A0 at <= 1e-11. 6 single-thread workers on a shared box; timings are unverified.

usage:
  fiq.py step0        X table above (npz only, ~1 min)
  fiq.py step0n5      z direct-channel n = 3, 4, 5 check (~3 min)
"""

import itertools
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PBC = os.path.dirname(HERE)
sys.path.insert(0, PBC)
sys.path.insert(0, HERE)

DATA = os.environ.get("HEAD_DATA", "/home/matt/qc/ferric/wiki/data/pbc-head-anomaly")
A_CELL = 6.0
ORIENTS = ("z", "111")


def _suf(orient):
    return "" if orient == "z" else f"_{orient}"


def _imports():
    import cheap_account as CA

    CA.DATA = DATA  # the scripts hard-code the old ferric-pbc checkout
    return CA


def cell_of(orient):
    from pbc_gamma import Cell
    from vao_build import h2_atoms

    return Cell(np.eye(3) * A_CELL, h2_atoms(orient), "sto-3g")


# ------------------------------------------------------------------------------ step 0 (npz only)
def split_heads(CA, st, gov, gvo, eo, ev, vol, PHI):
    """Channel-resolved head pieces (per cell): returns dict of (OS, SS) arrays for the linear head L (cubic
    average), the cubic quad Qall (lambda-quadratic, all channels), and the direct-|H|^2 quad Qc (cubic) / Ql (Phi).
    'direct' energy = 2 * OS (the J-type 2|V|^2/D term); exchange = SS - OS."""
    Nk = st["Nk"]
    c = 4 * np.pi / vol
    eo_, ev_ = np.array(eo)[:, 0], np.array(ev)[:, 0]
    D = eo_[:, None] + eo_[None, :] - ev_[:, None] - ev_[None, :]
    G = c * np.einsum("ia,jb->ijab", gov, gvo.conj())
    Hc = np.einsum("ijaa->ij", G) / 3
    Qc = 2 * np.sum(abs(Hc) ** 2 / D) / Nk**3
    Ql = 2 * np.sum(np.einsum("ijab,ijcd,abcd->ij", G, G.conj(), PHI).real / D) / Nk**3

    def e_with(lam):
        new = dict(st)
        V = st["V"].copy()
        for x in range(Nk):
            V[x, :, x, 0, 0, 0, 0] += lam * Hc[x]
        new["V"] = V
        return np.array(CA.kmp2_split(new, eo, ev))

    s0 = np.array(CA.kmp2_split(st, eo, ev))
    e1, e2 = e_with(1.0) - s0, e_with(2.0) - s0
    Qall = (e2 - 2 * e1) / 2
    return dict(S=s0, L=e1 - Qall, Qall=Qall, Qc=Qc, Ql=Ql)


def step0_pieces(orient, m):
    """Every npz-computable piece of the E3d estimator at mesh n = m, channel resolved, x n^3 where noted."""
    CA = _imports()
    import bloch_head as BH

    cell = cell_of(orient)
    PHI = BH.phi_tensor()
    z, row = CA.load(A_CELL, m, _suf(orient))
    st = CA.st_from(z, cell, m)
    eps, vm = z["eps"], float(z["vm"])
    eo = [e[:1] - vm for e in eps]
    ev = [e[1:] for e in eps]
    gfo, gfv, gto_, gtv, info = BH.head_vectors(z, cell, m)
    out = dict(info=info, row=row)
    for tag, (go, gv) in (("fixed", (gfo, gfv)), ("true", (gto_, gtv))):
        out[tag] = split_heads(CA, st, go, gv, eo, ev, cell.vol, PHI)
    S = out["fixed"]["S"]
    # first-order Fock heads (eigenvalue shifts; eps_corr = eps_mesh - shift): fixed-k (stored deo/dev) and
    # Bloch-paired 'none' interpolant (truefock.py dK npz, shift = dK_pp / 2), when present
    eoF = [x - y for x, y in zip(eo, z["deo"])]
    evF = [x - y for x, y in zip(ev, z["dev"])]
    out["F_fixed"] = np.array(CA.kmp2_split(st, eoF, evF)) - S
    dkf = f"{HERE}/dK_a{A_CELL:g}_n{m}{_suf(orient)}.npz"
    if os.path.exists(dkf):
        dk = np.load(dkf)
        out["dK_ana_vs_deo"] = max(
            abs(dk["ana"][:, 0, 0].real / 2 - z["deo"][:, 0]).max(),
            abs(dk["ana"][:, 1, 1].real / 2 - z["dev"][:, 0]).max(),
        )
        t = dk["true_none"]
        eoT = [x - np.array([t[k, 0, 0].real / 2]) for k, x in enumerate(eo)]
        evT = [x - np.array([t[k, 1, 1].real / 2]) for k, x in enumerate(ev)]
        out["F_true"] = np.array(CA.kmp2_split(st, eoT, evT)) - S
    return out


def step0():
    """X_dir (pre-step 0): the direct-channel n^-3 residual coefficient of the estimator, from the npz only."""
    res = {}
    for orient in ORIENTS:
        P = {m: step0_pieces(orient, m) for m in (3, 4)}
        # E_inf from the shifted-denominator S at n = 3, 4 (pure n^-3 through two points), per channel
        Sm = {m: P[m]["fixed"]["S"] for m in (3, 4)}
        einf = (64 * Sm[4] - 27 * Sm[3]) / 37
        for m in (3, 4):
            p, n3 = P[m], m**3
            f, t = p["fixed"], p["true"]
            assert abs(f["S"].sum() - p["row"]["sh"][0]) < 1e-14
            assert abs(np.sum(f["L"] + f["Qall"]) - (p["row"]["cub1"][0] - p["row"]["sh"][0])) < 1e-14
            assert abs(p["F_fixed"].sum() - (p["row"]["fock"][0] - p["row"]["sh"][0])) < 1e-14
            assert abs(f["Qall"][0] * 2 - f["Qc"]) < 1e-16 * n3 + 1e-12 * abs(f["Qc"])  # direct quad == 2 x OS
            # direct channel (2 x OS): E3d_dir = 2(S + L + Qc/2 + F)_OS + (Ql - Qc)  ==  2(S + L + F)_OS + Ql
            def e3d_dir(h, Fv):
                return 2 * (h["S"][0] + h["L"][0] + Fv[0]) + h["Ql"]

            def e3d_tot(h, Fv):
                return np.sum(h["S"] + h["L"] + h["Qall"] + Fv) + (h["Ql"] - h["Qc"])

            ed, et = 2 * einf[0], einf.sum()
            row = dict(
                Xdir_fixed=(e3d_dir(f, p["F_fixed"]) - ed) * n3,
                Xtot_fixed=(e3d_tot(f, p["F_fixed"]) - et) * n3,
                Xdir_trueERI_fixedF=(e3d_dir(t, p["F_fixed"]) - ed) * n3,
                Xtot_trueERI_fixedF=(e3d_tot(t, p["F_fixed"]) - et) * n3,
                Pcode_true=-(2 * t["L"][0] + t["Ql"]) * n3,
                Pcode_fixed=-(2 * f["L"][0] + f["Ql"]) * n3,
                info=p["info"],
            )
            if "F_true" in p:
                row["Xdir_true"] = (e3d_dir(t, p["F_true"]) - ed) * n3
                row["Xtot_true"] = (e3d_tot(t, p["F_true"]) - et) * n3
                row["dK_ana_vs_deo"] = p["dK_ana_vs_deo"]
            row["F_dir_fixed"] = 2 * p["F_fixed"][0] * n3
            res[(orient, m)] = row
            print(f"== {orient} n0={m}: " + " ".join(
                f"{k} {v:+.4e}" for k, v in row.items() if isinstance(v, float)), flush=True)
            print(f"   bloch_head info {p['info']}", flush=True)
    return res


def step0n5():
    """Direct / exchange / total E3d (fixed variant) at n = 3, 4, 5 (z): is X_dir a stable n^-3 coefficient?"""
    P = {m: step0_pieces("z", m) for m in (3, 4, 5)}
    vals = {}
    for m in (3, 4, 5):
        p, f = P[m], P[m]["fixed"]
        ed = 2 * (f["S"][0] + f["L"][0] + p["F_fixed"][0]) + f["Ql"]
        et = np.sum(f["S"] + f["L"] + f["Qall"] + p["F_fixed"]) + f["Ql"] - f["Qc"]
        vals[m] = dict(S_dir=2 * f["S"][0], S_exch=f["S"][1] - f["S"][0], E3d_dir=ed, E3d_exch=et - ed, E3d_tot=et)
        print(f"n={m} " + " ".join(f"{k} {v:.12e}" for k, v in vals[m].items()))
    n = np.array([3, 4, 5.0])
    for key in vals[3]:
        y = np.array([vals[m][key] for m in (3, 4, 5)])
        c = np.linalg.solve(np.c_[np.ones(3), n**-3, n**-5], y)
        print(f"{key:9s} rho {(y[1] - y[2]) / (y[0] - y[1]):+.4f} (n^-3 0.3561) | 3-pt c3 {c[1]:+.4e} c5 {c[2]:+.4e}"
              f" | (3,4) c3 {(y[0] - (64 * y[1] - 27 * y[0]) / 37) * 27:+.4e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step0"
    if mode == "step0":
        step0()
    elif mode == "step0n5":
        step0n5()
