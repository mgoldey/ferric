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

RESULT (2026-10-29, after the pre-registration commit e2f1f1c8b): see the FINDINGS addendum "FIQ Stage 1 result".
Every anchor and gate passes. |Delta| <= 6.2e-11 (z, [111]; n0 = 3, 4) is the "~ 0" row: ERI-side reading (i)
REFUTED (provisional). After the run, P_code('ewald') was corrected to use the ewald interpolant's own eigenvalues
(true_ewald). The original construction is kept as true_ewald_v0 / G_int_v0, and G_int passes under both.

usage (raw per-q npz -> $FIQ_OUT, default <DATA>/fiq; FIQ_WORKERS default 6; single-thread BLAS is forced):
  fiq.py step0                    X table above (npz only, ~1 min)
  fiq.py step0n5                  z direct-channel n = 3, 4, 5 check (~3 min)
  fiq.py a0run z 3 111 3 z 4 111 4   A0 q points (+ mutants mut_phase, ws_full);  fiq.py a0 z 3 ...  analyse
  fiq.py a1                       PySCF off-mesh quartet (+ mut_phase);  fiq.py a2   Duffy synthetic (+ no Jacobian)
  fiq.py pilotrun / pilot         10 q (5 +-q pairs): seconds per q, time reversal
  fiq.py s1run [orient]           Stage 1 q points (Lebedev 50 + 38 directions, t = 0.04 ... 0.005, q = 0)
  fiq.py s1 [orient]              Stage 1 table: Delta, gates, A4, A5, M1, reachability
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
            assert (
                abs(
                    np.sum(f["L"] + f["Qall"])
                    - (p["row"]["cub1"][0] - p["row"]["sh"][0])
                )
                < 1e-14
            )
            assert (
                abs(p["F_fixed"].sum() - (p["row"]["fock"][0] - p["row"]["sh"][0]))
                < 1e-14
            )
            assert abs(f["Qall"][0] * 2 - f["Qc"]) < 1e-16 * n3 + 1e-12 * abs(
                f["Qc"]
            )  # direct quad == 2 x OS

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
            print(
                f"== {orient} n0={m}: "
                + " ".join(
                    f"{k} {v:+.4e}" for k, v in row.items() if isinstance(v, float)
                ),
                flush=True,
            )
            print(f"   bloch_head info {p['info']}", flush=True)
    return res


# ------------------------------------------------------------------------------ the FIQ integrand
GCUT_PREC = 1e-10  # run_kcorr_head_anomaly.run_mesh's K sphere (the stored V used it)
OUT = os.environ.get("FIQ_OUT", f"{DATA}/fiq")


class Inner:
    """The fixed inner problem: n0-mesh occupied orbitals and v_M-shifted occupied energies (saved SCF), and the
    virtual band at any k from the Wigner-Seitz Fourier interpolant of the mesh Fock (truefock.Bands)."""

    def __init__(self, orient, n0, interp="none", mutant=None):
        CA = _imports()
        import truefock as TF

        self.orient, self.n0, self.interp = orient, n0, interp
        self.cell = cell_of(orient)
        z, self.row = CA.load(A_CELL, n0, _suf(orient))
        self.z = z
        self.Nk = n0**3
        b = TF.Bands(self.cell, z, n0, conv=interp)
        if (
            mutant == "ws_full"
        ):  # WRONG interpolant: boundary aliases at full weight (even n0 only)
            Sk = np.array([b.S(k) for k in b.kpts])
            SC = np.einsum("kmn,knp->kmp", Sk, b.C)
            Fk = np.einsum("kmp,kp,knp->kmn", SC, b.eps, SC.conj())
            rng = range(-(n0 // 2), n0 // 2 + 1)
            b.FL = []
            for mv in itertools.product(rng, repeat=3):
                L = np.array(mv, float) @ self.cell.a
                b.FL.append(
                    (
                        L,
                        np.einsum("k,kmn->mn", np.exp(-1j * b.kpts @ L), Fk)
                        / len(b.kpts),
                    )
                )
        self.b = b
        self.kpts = b.kpts
        self.vm = float(z["vm"])
        self.eo = z["eps"][:, 0] - self.vm
        self.Co = z["C"][:, :, 0]
        self.Cv_mesh, self.ev_mesh = z["C"][:, :, 1], z["eps"][:, 1]

    def virt(self, ks):
        out_e, out_c = [], []
        for k in ks:
            e, c = self.b.orbitals(k)
            out_e.append(e[1])
            out_c.append(c[:, 1])
        return np.array(out_e), np.array(out_c)


def eval_q(q, inners, specs, chunk=30000):
    """V_body, V_head (the G = 0 term K = q) and D for every spec at one q.
    spec = (key, pairing): key indexes `inners`; pairing 'true' (a at k_i + q, b at k_j - q, leg-1 pair FT at
    k' = k_i + q), 'fixed' (a at k_i, b at k_j, leg-1 k' = k_i: the head code's fixed-k vectors) or the mutant
    'mut_phase' (true orbitals, leg-1 k' = k_i)."""
    import pbc_kpts as PK
    from pbc_supercell import pair_ft_residues

    any_in = next(iter(inners.values()))
    cell, b0 = any_in.cell, any_in.b
    q = np.asarray(q, float)
    gcut = PK.aft_gcut(cell, GCUT_PREC)
    K = PK.shifted_gvectors(cell, q, gcut)
    is_head = np.linalg.norm(K - q[None, :], axis=1) < 1e-12
    v = 4 * np.pi / np.einsum("gi,gi->g", K, K) / cell.vol
    Ls = b0.pl.Ls
    prep, phase_sets = {}, {}
    for key, pairing in specs:
        inn = inners[key]
        km = inn.kpts
        if pairing == "fixed":
            ea, Ca, eb, Cb = inn.ev_mesh, inn.Cv_mesh, inn.ev_mesh, inn.Cv_mesh
        else:
            ea, Ca = inn.virt(km + q)
            eb, Cb = inn.virt(km - q)
        leg1 = (inn.n0, "shift") if pairing == "true" else (inn.n0, "mesh")
        phase_sets[(inn.n0, "mesh")] = km
        phase_sets[(inn.n0, "shift")] = km + q
        D = inn.eo[:, None] + inn.eo[None, :] - ea[:, None] - eb[None, :]
        prep[(key, pairing)] = dict(
            Ca=Ca,
            Cb=Cb,
            D=D,
            leg1=leg1,
            leg2=(inn.n0, "mesh"),
            Co=inn.Co,
            Vb=np.zeros((inn.Nk, inn.Nk), complex),
            Vh=np.zeros((inn.Nk, inn.Nk), complex),
        )
    ph = {k: np.exp(1j * ks @ Ls.T) for k, ks in phase_sets.items()}
    for c0 in range(0, len(K), chunk):
        sl = slice(c0, c0 + chunk)
        Kc, vc, hc = K[sl], v[sl], is_head[sl]
        Q = pair_ft_residues(b0.pl, Kc, 1e-14) * b0.nn[None, :, :, None]
        nL, g = Q.shape[0], Q.shape[3]
        Qm = Q.reshape(nL, -1)
        P = {k: (p @ Qm).reshape(len(p), 2, 2, g) for k, p in ph.items()}
        del Q, Qm
        for d in prep.values():
            Bov = np.einsum(
                "xm,xmng,xn->xg", d["Co"].conj(), P[d["leg1"]], d["Ca"], optimize=True
            )
            Bvo = np.einsum(
                "ym,ymng,yn->yg", d["Cb"].conj(), P[d["leg2"]], d["Co"], optimize=True
            )
            w = Bov * vc[None, :]
            d["Vb"] += w[:, ~hc] @ Bvo[:, ~hc].conj().T
            if hc.any():
                d["Vh"] += w[:, hc] @ Bvo[:, hc].conj().T
    return {k: (d["Vb"], d["Vh"], d["D"]) for k, d in prep.items()}, len(K)


def F_of(Vb, Vh, D, lam=1.0):
    """Per-q direct energy F(q) = (1/Nk^2) sum 2|V|^2/D (mesh average of F over q = the direct KMP2 per cell)."""
    V = Vb + lam * Vh
    return float(np.sum(2 * abs(V) ** 2 / D).real) / D.shape[0] ** 2


def stored_perq(inn):
    """Per-q direct energies from the stored V, eps, vm (the S convention): E_q = (1/Nk^3) sum 2|V|^2/D."""
    from pbc_kpts import mesh_index, mp_mesh

    n = (inn.n0,) * 3
    ints, _ = mp_mesh(inn.cell, n)
    V = inn.z["V"][..., 0, 0, 0, 0]
    ev = inn.ev_mesh
    out = []
    for iq in range(inn.Nk):
        tot = 0.0
        for ki in range(inn.Nk):
            ka = mesh_index(n, ints[ki] + ints[iq])
            for kj in range(inn.Nk):
                kb = mesh_index(n, ints[kj] - ints[iq])
                D = inn.eo[ki] + inn.eo[kj] - ev[ka] - ev[kb]
                tot += 2 * abs(V[ki, kj, ka]) ** 2 / D
        out.append(tot / inn.Nk**3)
    return np.array(out)


# ------------------------------------------------------------------------------ worker pool
_INNERS = {}


def _inners(orient, keys):
    out = {}
    for key in keys:
        if (orient, key) not in _INNERS:
            n0, interp, mut = key
            _INNERS[(orient, key)] = Inner(orient, n0, interp, mut)
        out[key] = _INNERS[(orient, key)]
    return out


def _job(task):
    import time

    tag, orient, q, specs = task
    path = f"{OUT}/{tag}.npz"
    if os.path.exists(path):
        return tag, None
    t0 = time.time()
    inners = _inners(orient, list(dict.fromkeys(k for k, _ in specs)))
    res, nK = eval_q(q, inners, specs)
    dt = time.time() - t0
    arrs = {}
    for i, ((key, pairing), (Vb, Vh, D)) in enumerate(res.items()):
        arrs[f"Vb{i}"], arrs[f"Vh{i}"], arrs[f"D{i}"] = Vb, Vh, D
    os.makedirs(OUT, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(
        tmp,
        q=q,
        orient=orient,
        specs=np.array([repr(s) for s in res]),
        nK=nK,
        dt=dt,
        **arrs,
    )
    os.replace(tmp, path)
    return tag, dt


def run_pool(tasks, workers=6):
    import multiprocessing as mp
    import time

    t0 = time.time()
    todo = [t for t in tasks if not os.path.exists(f"{OUT}/{t[0]}.npz")]
    print(
        f"pool: {len(todo)} of {len(tasks)} tasks to run, {workers} workers (pid {os.getpid()})",
        flush=True,
    )
    with mp.get_context("fork").Pool(workers) as pool:
        for i, (tag, dt) in enumerate(pool.imap_unordered(_job, todo, chunksize=1)):
            if dt is not None:
                print(
                    f"  [{i + 1}/{len(todo)}] {tag} {dt:.1f}s (wall {time.time() - t0:.0f}s)",
                    flush=True,
                )
    print(f"pool done, wall {time.time() - t0:.0f}s", flush=True)


def load_task(tag):
    import ast

    z = np.load(f"{OUT}/{tag}.npz")
    specs = [ast.literal_eval(s) for s in z["specs"]]
    return {s: (z[f"Vb{i}"], z[f"Vh{i}"], z[f"D{i}"]) for i, s in enumerate(specs)}, z


# ------------------------------------------------------------------------------ anchors A0 + mutants
def a0_tasks(orient, n0):
    inn_specs = [((n0, "none", None), "true"), ((n0, "none", None), "mut_phase")]
    if n0 % 2 == 0:
        inn_specs.append(((n0, "none", "ws_full"), "true"))
    from pbc_kpts import mp_mesh

    _, kpts = mp_mesh(cell_of(orient), (n0,) * 3)
    return [
        (f"a0_{orient}_n{n0}_q{iq}", orient, kpts[iq], inn_specs) for iq in range(n0**3)
    ]


def a0_analyze(orient, n0):
    inn = Inner(orient, n0)
    ref = stored_perq(inn)
    tot_ref = (
        2
        * _imports().kmp2_split(
            _imports().st_from(inn.z, inn.cell, n0),
            [e[:1] for e in inn.eo[:, None]],
            [e[:1] for e in inn.ev_mesh[:, None]],
        )[0]
    )
    rows = {}
    for tag, *_ in a0_tasks(orient, n0):
        if not os.path.exists(f"{OUT}/{tag}.npz"):
            return None
        res, _z = load_task(tag)
        for spec, (Vb, Vh, D) in res.items():
            rows.setdefault(spec, []).append(F_of(Vb, Vh, D) / inn.Nk)
    out = {}
    for spec, vals in rows.items():
        vals = np.array(vals)
        out[spec] = (
            abs(vals - ref).max(),
            abs(vals.sum() - ref.sum()),
            abs(vals.sum() - tot_ref),
        )
        print(
            f"A0 {orient} n0={n0} {spec}: max per-q |FIQ - stored| {out[spec][0]:.2e}; total |FIQ - stored perq sum| "
            f"{out[spec][1]:.2e}; total vs 2 S_OS(kmp2_split) {out[spec][2]:.2e}  (stored total {ref.sum():.12e})"
        )
    print(f"   stored per-q sum vs 2 S_OS: {abs(ref.sum() - tot_ref):.1e}")
    return out


# ------------------------------------------------------------------------------ Stage 1 tasks
S1_SPECS = [
    ((3, "none", None), "true"),
    ((3, "ewald", None), "true"),
    ((3, "none", None), "fixed"),
    ((4, "none", None), "true"),
    ((4, "ewald", None), "true"),
    ((4, "none", None), "fixed"),
]
T_A, T_B = (0.04, 0.02, 0.01), (0.02, 0.01, 0.005)


def lebedev(n):
    """Lebedev grid (lab frame), representatives of the inversion pairs and the map point -> representative."""
    from pyscf.dft import gen_grid

    g = gen_grid.MakeAngularGrid(n)
    u, w = g[:, :3], g[:, 3]
    reps, rmap = [], []
    for x in u:
        y = x if (x[np.nonzero(abs(x) > 1e-10)[0][0]] > 0) else -x
        key = tuple(np.round(y, 10))
        if key not in reps:
            reps.append(key)
        rmap.append(reps.index(key))
    return u, w, [np.array(r) for r in reps], np.array(rmap)


def s1_dirs():
    _, _, r50, _ = lebedev(50)
    _, _, r38, _ = lebedev(38)
    allr = {tuple(np.round(r, 10)): r for r in r50 + r38}
    k50 = {tuple(np.round(r, 10)) for r in r50}
    return allr, k50


def s1_tasks(orient):
    allr, k50 = s1_dirs()
    tasks = [(f"s1_{orient}_q0", orient, np.zeros(3), S1_SPECS)]
    for key, u in allr.items():
        ts = sorted(set(T_A) | (set(T_B) if key in k50 else set()))
        for t in ts:
            tasks.append(
                (
                    f"s1_{orient}_u{'_'.join(f'{x:+.6f}' for x in u)}_t{t:g}",
                    orient,
                    t * u,
                    S1_SPECS,
                )
            )
    return tasks


def pilot_tasks():
    rng = np.random.default_rng(230)
    cell = cell_of("z")
    out = []
    for i in range(5):
        f = rng.uniform(-0.5, 0.5, 3)
        q = f @ cell.b
        out.append((f"pilot_z_{i}_p", "z", q, S1_SPECS))
        out.append((f"pilot_z_{i}_m", "z", -q, S1_SPECS))
    return out


def pilot_analyze():
    dts, tr = [], 0.0
    for i in range(5):
        rp, zp = load_task(f"pilot_z_{i}_p")
        rm, zm = load_task(f"pilot_z_{i}_m")
        dts += [float(zp["dt"]), float(zm["dt"])]
        for s in rp:
            fp, fm = F_of(*rp[s]), F_of(*rm[s])
            tr = max(tr, abs(fp - fm) / abs(fp))
    print(
        f"pilot: seconds per q (6 workers, shared box) mean {np.mean(dts):.1f} min {min(dts):.1f} max {max(dts):.1f}"
    )
    print(
        f"A5 time reversal: max |F(q) - F(-q)| / |F| over 5 generic q, all Stage-1 specs: {tr:.1e}"
    )
    return np.mean(dts), tr


# ------------------------------------------------------------------------------ Stage 1 analysis
PHI4 = -0.17159902
PHIK4 = 3 * PHI4 - 0.6  # Phi[sum u_a^4 - 3/5]


def _quartic_monomials(u):
    ex = [e for e in itertools.product(range(5), repeat=3) if sum(e) == 4]
    M = np.stack([np.prod(u ** np.array(e), axis=1) for e in ex], 1)
    phi = np.array(
        [
            PHI4
            if max(e) == 4
            else ((1 - 3 * PHI4) / 6 if sorted(e) == [0, 2, 2] else 0.0)
            for e in ex
        ]
    )
    return M, phi


def richardson(F3):
    """F at t, t/2, t/4 (even in t) -> limit with t^2 and t^4 removed."""
    return (F3[0] - 20 * F3[1] + 64 * F3[2]) / 45


def p_num(orient, spec, leb=50, tset=T_A, lam=1.0, inject=None, part=None):
    """-Phi[F_lim - r0] and diagnostics. part: None (F at lam), 'L' or 'Q' (lambda-linear / -quadratic parts),
    'B' (lambda = 0 body: its limit minus r0 should vanish)."""
    u, w, reps, rmap = lebedev(leb)

    def Fval(res, qv):
        Vb, Vh, D = res[spec]
        if part is None:
            f = F_of(Vb, Vh, D, lam)
        else:
            f0, f1, f2 = (F_of(Vb, Vh, D, x) for x in (0.0, 1.0, 2.0))
            c2 = (f2 - 2 * f1 + f0) / 2
            f = {"L": f1 - f0 - c2, "Q": c2, "B": f0}[part]
        if inject is not None:
            f += inject(qv)
        return f

    r0_res, _ = load_task(f"s1_{orient}_q0")
    r0 = Fval(r0_res, np.zeros(3)) if part in (None, "B") else 0.0
    hrep = []
    for r in reps:
        Fs = []
        for t in tset:
            res, _ = load_task(
                f"s1_{orient}_u{'_'.join(f'{x:+.6f}' for x in r)}_t{t:g}"
            )
            Fs.append(Fval(res, t * r))
        hrep.append(richardson(Fs) - r0)
    h = np.array(hrep)[rmap]
    K4 = np.sum(u**4, 1) - 0.6
    avg = np.sum(w * h)
    c4 = np.sum(w * h * K4) / np.sum(w * K4 * K4)
    phi = avg + c4 * PHIK4
    M, phim = _quartic_monomials(u)
    coef, *_ = np.linalg.lstsq(M, h, rcond=None)
    resid = h - M @ coef
    return dict(
        P=-phi,
        avg=avg,
        c4=c4,
        phi_ls=float(coef @ phim),
        l6_rms=float(np.sqrt(np.sum(w * resid**2))),
        h_rms=float(np.sqrt(np.sum(w * h**2))),
    )


class Inject:
    """Injected integrand term kappa * a(q^) * e^{-|q|^2/sig^2} (a = 1 for l = 0, sum u^4 - 3/5 for l = 4). It is
    degree 0 at q = 0, so the S convention drops it there. It is added to the F(t u) values, so the Richardson
    extrapolation and the Lebedev projection must both carry it to P_num. Expected shift: -kappa Phi[a]."""

    def __init__(self, kappa, ell, sig=0.3):
        self.kappa, self.ell, self.sig = kappa, ell, sig

    def __call__(self, qv):
        t = float(np.linalg.norm(qv))
        if t == 0.0:
            return 0.0
        u = qv / t
        a = 1.0 if self.ell == 0 else float(np.sum(u**4) - 0.6)
        return self.kappa * a * np.exp(-(t**2) / self.sig**2)


def p_code(orient, n0):
    """Direct-channel coded head coefficient -n^3 (L_dir + Qlat_dir) for true ('none' and 'ewald' interpolants) and
    fixed vectors; also its L / Q split."""
    CA = _imports()
    import bloch_head as BH

    cell = cell_of(orient)
    PHI = BH.phi_tensor()
    z, row = CA.load(A_CELL, n0, _suf(orient))
    st = CA.st_from(z, cell, n0)
    eps, vm = z["eps"], float(z["vm"])
    eo = [e[:1] - vm for e in eps]
    ev = [e[1:] for e in eps]
    n3 = n0**3
    out = {}
    gfo, gfv, gto_, gtv, _ = BH.head_vectors(z, cell, n0)
    from pbc_kpts import mp_mesh

    _, kpts = mp_mesh(cell, (n0,) * 3)
    S, _dS = BH.lattice_S(cell, kpts)
    epsw = eps.copy()
    epsw[:, 0] -= vm
    SC = np.einsum("kmn,knp->kmp", S, z["C"])
    Fk = np.einsum("kmp,kp,knp->kmn", SC, epsw, SC.conj())
    # v0 (as first run): ewald dF but bloch_head's eps_i, eps_a still unshifted (internally inconsistent)
    _, _, gte0, gtev0, _ = BH.head_vectors(
        z, cell, n0, dF_override=BH.fourier_grad(Fk, cell, n0)
    )
    # consistent: the ewald interpolant's own eigenvalues (eps_occ - v_M) everywhere in the Bloch-pairing formula
    zz = dict(C=z["C"], eps=epsw, A=z["A"], P0=z["P0"])
    _, _, gte, gtev, _ = BH.head_vectors(zz, cell, n0)
    for tag, (go, gv) in (
        ("fixed", (gfo, gfv)),
        ("true_none", (gto_, gtv)),
        ("true_ewald", (gte, gtev)),
        ("true_ewald_v0", (gte0, gtev0)),
    ):
        h = split_heads(CA, st, go, gv, eo, ev, cell.vol, PHI)
        out[tag] = dict(
            P=-(2 * h["L"][0] + h["Ql"]) * n3, L=-2 * h["L"][0] * n3, Q=-h["Ql"] * n3
        )
    return out


def stage1_analyze(orients=ORIENTS):
    X_DIR = {"z": 2.4835e-4, "111": -8.6531e-5}
    XMIN = {"z": 4.0171e-5, "111": 6.6006e-5}
    X_TOT = {"z": -4.0171e-5, "111": -1.8373e-4}
    summary = {}
    for orient in orients:
        X, xm = X_DIR[orient], XMIN[orient]
        for n0 in (3, 4):
            pc = p_code(orient, n0)
            key = (n0, "none", None)
            pn = p_num(orient, (key, "true"))
            pnB = p_num(orient, (key, "true"), tset=T_B)
            pn38 = p_num(orient, (key, "true"), leb=38)
            pne = p_num(orient, ((n0, "ewald", None), "true"))
            pnf = p_num(orient, (key, "fixed"))
            pnL = p_num(orient, (key, "true"), part="L")
            pnQ = p_num(orient, (key, "true"), part="Q")
            pnB0 = p_num(orient, (key, "true"), part="B")
            d = pn["P"] - pc["true_none"]["P"]
            de = pne["P"] - pc["true_ewald"]["P"]
            de0 = pne["P"] - pc["true_ewald_v0"]["P"]
            row = dict(
                P_num=pn["P"],
                P_code=pc["true_none"]["P"],
                Delta=d,
                Delta_over_Xdir=d / X,
                Delta_over_Xtot=d / X_TOT[orient],
                G_t=abs(pn["P"] - pnB["P"]),
                G_leb=abs(pn["P"] - pn38["P"]),
                G_int=abs(de - d),
                G_int_v0=abs(de0 - d),
                A4_rel=abs(pnf["P"] - pc["fixed"]["P"]) / abs(pc["fixed"]["P"]),
                A4_abs=pnf["P"] - pc["fixed"]["P"],
                A5_body=abs(pnB0["P"]),
                Delta_L=pnL["P"] - pc["true_none"]["L"],
                Delta_Q=pnQ["P"] - pc["true_none"]["Q"],
                M1_fixed_minus_true=pnf["P"] - pn["P"],
                l6_rms=pn["l6_rms"],
                h_rms=pn["h_rms"],
                phi_ls_minus_proj=pn["phi_ls"] + pn["P"],
                c4=pn["c4"],
                avg=pn["avg"],
            )
            # reachability: inject l = 4 and l = 0 terms sized to move P_num by exactly X_dir
            rec = {}
            for ell, kap in ((4, -X / PHIK4), (0, -X)):
                inj = Inject(kap, ell)
                rec[ell] = p_num(orient, (key, "true"), inject=inj)["P"] - pn["P"]
            row["reach_l4"], row["reach_l0"] = rec[4], rec[0]
            summary[(orient, n0)] = row
            print(f"== Stage 1 {orient} n0={n0}  (X_dir {X:+.4e}, |X|min {xm:.4e})")
            for k, v in row.items():
                print(f"   {k:22s} {v:+.6e}")
    return summary


# ------------------------------------------------------------------------------ A1: external oracle (PySCF)
def a1_pyscf():
    """One generic off-mesh quartet: k_i = k_j = Gamma, k_a = (0.137, -0.071, 0.213).b, k_b = -k_a. FIQ's
    (i a | j b) vs PySCF 2.13 AFTDF.get_eri MO-transformed with the SAME C (complex value, then |V|).
    Two FIQ K spheres: production (prec 1e-10) and a tight one (1e-16, inside PySCF's 61^3 box)."""
    global GCUT_PREC
    from pyscf.pbc import df as pdf
    from pyscf.pbc import gto as pgto

    key = (3, "none", None)
    inn = Inner("z", 3)
    cell = inn.cell
    ka = np.array([0.137, -0.071, 0.213]) @ cell.b
    kb, g0 = -ka, np.zeros(3)
    assert abs(inn.kpts[0]).max() == 0
    ours = {}
    for prec in (1e-10, 1e-16):
        GCUT_PREC = prec
        res, nK = eval_q(ka, {key: inn}, [(key, "true")])
        Vb, Vh, _ = res[(key, "true")]
        ours[prec] = ((Vb + Vh)[0, 0], nK)
    GCUT_PREC = 1e-10
    _, Ca = inn.virt([ka])
    _, Cb = inn.virt([kb])
    Ci = Cj = inn.Co[0]
    pc = pgto.Cell(
        a=cell.a, atom=cell.atoms, basis="sto-3g", unit="B", cart=True, verbose=0
    )
    pc.precision = 1e-12
    pc.max_memory = 1500
    pc.build()
    af = pdf.AFTDF(pc)
    af.mesh = [61] * 3
    eri = af.get_eri(np.array([g0, ka, g0, kb]), compact=False).reshape((pc.nao,) * 4)
    ref = np.einsum("pqrs,p,q,r,s->", eri, Ci.conj(), Ca[0], Cj.conj(), Cb[0])
    for prec, (v, nK) in ours.items():
        print(
            f"A1 FIQ (prec {prec:g}, nK {nK}) {v:.15e}  PySCF {ref:.15e}  |diff| {abs(v - ref):.2e}  "
            f"||V| - |V_pyscf|| {abs(abs(v) - abs(ref)):.2e}"
        )
    # mutant check: the wrong leg-1 Bloch phase must miss PySCF
    res, _ = eval_q(ka, {key: inn}, [(key, "mut_phase")])
    Vb, Vh, _ = res[(key, "mut_phase")]
    print(f"A1 mutant mut_phase |diff| {abs((Vb + Vh)[0, 0] - ref):.2e}")
    return ours, ref


# ------------------------------------------------------------------------------ A2: Duffy integrator (synthetic)
def duffy(F, N, jacobian=True):
    """Integral over the BZ cube [-pi, pi]^3 / (2pi)^3 by six pyramids with apex at q = 0 (q = s v, Jacobian s^2)."""
    x, w = np.polynomial.legendre.leggauss(N)
    s, ws = (x + 1) / 2, w / 2
    u, wu = np.pi * x, np.pi * w
    S, U, V = np.meshgrid(s, u, u, indexing="ij")
    W = np.einsum("i,j,k->ijk", ws, wu, wu)
    jac = S**2 if jacobian else np.ones_like(S)  # mutant: drop the Duffy Jacobian
    tot = 0.0
    for ax in range(3):
        for sg in (1, -1):
            q = np.zeros(S.shape + (3,))
            o = [d for d in range(3) if d != ax]
            q[..., ax] = sg * np.pi * S
            q[..., o[0]] = U * S
            q[..., o[1]] = V * S
            tot += np.sum(W * jac * np.pi * F(q))
    return tot / (2 * np.pi) ** 3


def _synthetic(q, b=0.8, sig=2.2 * np.pi):
    """realistic_check5's periodic, analytic-except-at-0 synthetic (G images |m| <= 3)."""
    tot = 0.0
    for m in itertools.product(range(-3, 4), repeat=3):
        Q = q + 2 * np.pi * np.array(m, float)
        t2 = np.sum(Q**2, -1)
        ok = t2 > 0
        u2 = np.where(ok, Q[..., 2] ** 2 / np.where(ok, t2, 1), 0.0)
        tot = tot + np.where(ok, (1 + b * u2) ** 2, 0.0) * np.exp(-t2 / sig**2)
    return tot * (1 + 0.1 * np.cos(q[..., 0]))


def _smooth(q):
    return 1 + 0.3 * np.cos(q[..., 0]) * np.cos(2 * q[..., 1]) + 0.1 * np.cos(q[..., 2])


def a2_duffy():
    smooth = _smooth
    out = {}
    for jac in (True, False):
        s12, s16 = duffy(_synthetic, 12, jac), duffy(_synthetic, 16, jac)
        sm = duffy(smooth, 24, jac)
        out[jac] = (abs(s16 - s12) / abs(s16), abs(sm - 1.0))
        tag = "Duffy" if jac else "MUTANT no Jacobian"
        print(
            f"A2 {tag}: synthetic N=12 {s12:.15f} N=16 {s16:.15f} rel diff {out[jac][0]:.1e} (gate 1e-9); "
            f"smooth periodic (exact 1) err {out[jac][1]:.1e} (gate 1e-13)"
        )
    return out


def step0n5():
    """Direct / exchange / total E3d (fixed variant) at n = 3, 4, 5 (z): is X_dir a stable n^-3 coefficient?"""
    P = {m: step0_pieces("z", m) for m in (3, 4, 5)}
    vals = {}
    for m in (3, 4, 5):
        p, f = P[m], P[m]["fixed"]
        ed = 2 * (f["S"][0] + f["L"][0] + p["F_fixed"][0]) + f["Ql"]
        et = np.sum(f["S"] + f["L"] + f["Qall"] + p["F_fixed"]) + f["Ql"] - f["Qc"]
        vals[m] = dict(
            S_dir=2 * f["S"][0],
            S_exch=f["S"][1] - f["S"][0],
            E3d_dir=ed,
            E3d_exch=et - ed,
            E3d_tot=et,
        )
        print(f"n={m} " + " ".join(f"{k} {v:.12e}" for k, v in vals[m].items()))
    n = np.array([3, 4, 5.0])
    for key in vals[3]:
        y = np.array([vals[m][key] for m in (3, 4, 5)])
        c = np.linalg.solve(np.c_[np.ones(3), n**-3, n**-5], y)
        print(
            f"{key:9s} rho {(y[1] - y[2]) / (y[0] - y[1]):+.4f} (n^-3 0.3561) | 3-pt c3 {c[1]:+.4e} c5 {c[2]:+.4e}"
            f" | (3,4) c3 {(y[0] - (64 * y[1] - 27 * y[0]) / 37) * 27:+.4e}"
        )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step0"
    args = sys.argv[2:]
    W = int(os.environ.get("FIQ_WORKERS", "6"))
    if mode == "step0":
        step0()
    elif mode == "step0n5":
        step0n5()
    elif mode == "a0run":  # a0run orient n0 [orient n0 ...]
        run_pool(
            [t for o, n in zip(args[::2], args[1::2]) for t in a0_tasks(o, int(n))], W
        )
    elif mode == "a0":
        for o, n in zip(args[::2], args[1::2]):
            a0_analyze(o, int(n))
    elif mode == "pilotrun":
        run_pool(pilot_tasks(), W)
    elif mode == "pilot":
        pilot_analyze()
    elif mode == "s1run":
        run_pool([t for o in (args or ORIENTS) for t in s1_tasks(o)], W)
    elif mode == "s1":
        stage1_analyze(args or ORIENTS)
    elif mode == "a1":
        a1_pyscf()
    elif mode == "a2":
        a2_duffy()
