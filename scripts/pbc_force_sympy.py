#!/usr/bin/env python3
"""Checks of the electronic rows of ferric-pbc's Gamma-point nuclear force and
of the reciprocal and overlap terms of its stress.
Theory: `book/src/24-periodic/07-forces.md#forces-force-term-term` and
`book/src/24-periodic/08-stress.md#stress-assembling-electronic-stress`.

The model is a triclinic cell with two atoms carrying s Gaussians, two
nuclear charges and two fixed, unrelated spin densities D_a != D_b (so the
per-spin exchange and Madelung forms are exercised). The energy at FIXED
density is assembled from the pieces of the Gamma functional:

  E = sum D (V_SR + V_LR + c0 Z_tot S) + sum Gamma I
      - (alpha v_M / 2) sum_s tr(D_s S D_s S),

with I = (1/Omega) sum_{G != 0} (4 pi/G^2) Re[P*_mn P_ls] the dense AFT tensor,
Gamma = 1/2 D D - (alpha/2) sum_s D^s D^s, V_LR = -(1/Omega) sum_G v_eta
Re[P* S(G)], V_SR the erfc attraction of each Gaussian pair charge by each
nucleus (real space), and c0 = pi/(eta^2 Omega).

Claims (each prints "(claim X)" and exits nonzero on a mismatch):

  force-fixed-density  the force rows of the book's table, evaluated
        analytically on the HALF G sphere as grad.rs does, equal a central
        finite difference of E with respect to each atom position (basis
        centre and nucleus moved together):
          two-electron  (8/Omega) sum_half (4 pi/G^2) Re sum_{m in A, n}
                        Q*_mn Z_mn,  Z = 1/2 D rho(G) - (alpha/2) sum_s D_s P D_s,
                        rho(G) = sum_ls D_ls P_ls(G);
          LR basis      -(2/Omega) sum_half v_eta Re[rho'_A* S(G)],
                        rho'_A = 2 sum_{m in A, n} D_mn Q_mn;
          LR nucleus    -(2/Omega) sum_half v_eta Re[rho* dS/dR_A],
                        dS/dR_A = -i G Z_A e^{-i G.R_A};
          SR erfc       the basis derivative plus the nuclear derivative
                        taken as minus (bra + ket) by translation invariance;
          overlap       2 sum_{m in A, n} M_mn dS_mn/dA,
                        M = c0 Z_tot D - alpha v_M sum_s D_s S D_s.
        Q_mn(G) = dP_mn/dA is the bra-centre derivative of the pair
        transform; the ket derivative is Q_nm because P_mn = P_nm at lattice G.

  rsgdf-fit-derivative  the two-electron energy through the fit,
        E = sum Gamma J3 f(J2) J3^T with f(s) = 1/s above lindep and 0 below,
        has dE = sum Y dJ3 + sum Wm dJ2 with Y_P = D (C_P . D) - alpha
        sum_s D_s C_P D_s, C = f(J2) J3^T, and Wm = U (Lo o U^T H U) U^T,
        H = J3^T Gamma J3, Lo the Daleckii-Krein (Loewner) matrix of f:
        -1/(s_i s_j) kept-kept, (1/s_k)/(s_k - s_d) kept-dropped, 0
        dropped-dropped. Checked against a central difference along a random
        perturbation with one metric eigenvalue below the cut.

  stress-fixed-density  dE/deps_ab of the Coulomb, exchange and LR nuclear
        energies, -delta_ab E + (1/Omega) sum_G [v' (-2 G_a G_b) Phi + v dPhi]
        with v' = -v/G^2 (bare) or -v_eta (1/G^2 + 1/4 eta^2) (Ewald LR) and
        dPhi from the pair-transform strain dP/deps_ab = sum_L P_L [-2 (ab/p)
        d_a d_b + G_a G_b/(2p)] (S(G) is strain invariant), plus the overlap
        terms sum M dS/deps and the c0 volume term -delta_ab c0 Z_tot tr(D S),
        equal a central strain difference (h = 1e-5) of the same energy with
        the lattice, the atoms, the integer translations and the Miller set
        strained together (frozen index sets). The SR erfc term is left out of
        this claim (its real-space sum is not part of the model's strain).

Independent route: central differences of the transcribed energy
(h = 1e-5 Bohr). The Pulay term -W of M is not in this check: it comes from
the orthonormality constraint, not from the fixed-density energy (section
@sec:hf-gradient). The Ewald row is claim ewald-gradient of
scripts/pbc_ewald_pairft_stress_sympy.py.

Rust: crates/ferric-pbc/src/grad.rs gamma_rhf_gradient, gamma_uhf_gradient,
assemble (M); crates/ferric-pbc/src/pair_ft.rs pair_ft_deriv_chunked (Q);
crates/ferric-pbc/src/rsgdf/deriv.rs fit_derivatives (Y, Wm).

Mutations (each turns the script red):
  - 4/Omega in place of 8/Omega in the two-electron row, the 1/2 of
    1/2 D rho in Z dropped, or the LR nucleus row dropped (marker
    "[FAIL] (claim force-fixed-density)");
  - the kept-dropped Loewner block dropped, leaving the textbook kept-kept
    metric term (marker "[FAIL] (claim rsgdf-fit-derivative)");
  - the c0 volume term of the stress dropped (marker
    "[FAIL] (claim stress-fixed-density)").

Run: python3 scripts/pbc_force_sympy.py   (a few seconds)
"""

import math
import sys

import numpy as np
from scipy.special import erf

FAILURES = []


def check(claim, ok, detail=""):
    tag = "ok" if ok else "FAIL"
    print(f"[{tag}] (claim {claim}){(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(claim)


LAT = np.array([[3.4, 0.0, 0.0], [0.3, 3.6, 0.0], [-0.2, 0.4, 3.8]])
VOL = abs(np.linalg.det(LAT))
RECIP = 2 * math.pi * np.linalg.inv(LAT).T
ZNUC = np.array([1.0, 2.0])
# Basis: (atom, exponent); unnormalised s Gaussians.
BASIS = [(0, 1.0), (0, 0.45), (1, 0.8)]
NAO = len(BASIS)
ETA = 0.9  # the nuclear split
ALPHA = 0.6  # exact-exchange fraction
VM = 0.31  # a fixed Madelung constant (geometry independent at a fixed lattice)
GCUT = 12.0
NBOX = 4


def _int_box(nbox):
    r = np.arange(-nbox, nbox + 1)
    return np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T


NL = _int_box(NBOX)  # frozen integer translations
_NG_ALL = _int_box(9)
_G_ALL = _NG_ALL @ RECIP
_G2_ALL = np.einsum("ij,ij->i", _G_ALL, _G_ALL)
NG = _NG_ALL[(_G2_ALL > 1e-12) & (_G2_ALL <= GCUT * GCUT)]  # frozen Miller set


def geom(lat):
    """Volume, translations, G vectors and kernels of `lat` on the frozen
    integer sets (a strained cell keeps the reference cell's indices)."""
    recip = 2 * math.pi * np.linalg.inv(lat).T
    gv = NG @ recip
    g2 = np.einsum("ij,ij->i", gv, gv)
    first = np.where(
        gv[:, 0] != 0, gv[:, 0], np.where(gv[:, 1] != 0, gv[:, 1], gv[:, 2])
    )
    vg = 4 * math.pi / g2
    return {
        "vol": abs(np.linalg.det(lat)),
        "tvec": NL @ lat,
        "gv": gv,
        "g2": g2,
        "half": first > 0,
        "vg": vg,
        "veta": vg * np.exp(-g2 / (4 * ETA * ETA)),
    }


GEO = geom(LAT)
TVEC = GEO["tvec"]
SR_RADIUS = 9.0  # erfc(0.9 r) < 1e-35 beyond: the SR kernel's support
TVEC_M = TVEC[np.linalg.norm(TVEC, axis=1) < SR_RADIUS + 12.0]
GV, G2, HALF = GEO["gv"], GEO["g2"], GEO["half"]
VG, VETA = GEO["vg"], GEO["veta"]


def pair_ft(pos, gv, g2, deriv=False, tvec=None, strain=False):
    """P_mn(G) = sum_L (pi/p)^{3/2} e^{-ab|A-B-L|^2/p} e^{-G^2/4p} e^{-iG.Pc};
    with deriv, also Q_mn(G) = dP_mn/dA (3 components); with strain, also
    dP_mn/deps_ab = sum_L P_L [-2 (ab/p) d_a d_b + G_a G_b/(2p)], d = A - B - L
    (the book's pair-transform strain formula for s|s, where [Q_a +
    iG_a (a/p) P] = -2 (ab/p) d_a P and the polynomial factor is 1)."""
    tvec = TVEC if tvec is None else tvec
    p_out = np.zeros((NAO, NAO, len(g2)), complex)
    q_out = np.zeros((3, NAO, NAO, len(g2)), complex)
    e_out = np.zeros((3, 3, NAO, NAO, len(g2)), complex)
    for m, (am, a) in enumerate(BASIS):
        for n, (an, b) in enumerate(BASIS):
            p = a + b
            ra = pos[am]
            rb = pos[an][None, :] + tvec
            dv = ra[None, :] - rb
            d2 = np.einsum("ij,ij->i", dv, dv)
            s = (math.pi / p) ** 1.5 * np.exp(-a * b / p * d2)
            keep = s > 1e-22
            pc = (a * ra[None, :] + b * rb[keep]) / p
            ph = np.exp(-1j * pc @ gv.T) * s[keep][:, None]  # (L, G)
            damp = np.exp(-g2 / (4 * p))
            p_out[m, n] = damp * ph.sum(axis=0)
            if deriv:
                for c in range(3):
                    fac = (
                        -2 * a * b / p * dv[keep, c][:, None]
                        - 1j * gv[None, :, c] * a / p
                    )
                    q_out[c, m, n] = damp * (ph * fac).sum(axis=0)
            if strain:
                dk = dv[keep]
                for i in range(3):
                    for j in range(3):
                        sep = (-2 * a * b / p * dk[:, i] * dk[:, j])[:, None]
                        gg = (gv[:, i] * gv[:, j] / (2 * p))[None, :]
                        e_out[i, j, m, n] = damp * (ph * (sep + gg)).sum(axis=0)
    if strain:
        return p_out, e_out
    return (p_out, q_out) if deriv else p_out


def overlap(pos, deriv=False):
    g0 = np.zeros((1, 3))
    if deriv:
        p, q = pair_ft(pos, g0, np.zeros(1), deriv=True)
        return p[:, :, 0].real, q[:, :, :, 0].real
    return pair_ft(pos, g0, np.zeros(1))[:, :, 0].real


def struct(pos):
    return (ZNUC[:, None] * np.exp(-1j * pos @ GV.T)).sum(axis=0)


def v_sr(pos, nuc_pos=None, only=None):
    """V_SR_mn = -sum_C Z_C sum_{L,M} S_L [erf(sqrt(p) d) - erf(g d)]/d,
    1/g^2 = 1/p + 1/eta^2, d = |Pc_L - R_C - M| (Gaussian pair charge vs the
    erfc kernel). nuc_pos lets the nuclei sit apart from the basis; only
    keeps the terms of one nucleus."""
    nuc_pos = pos if nuc_pos is None else nuc_pos
    out = np.zeros((NAO, NAO))
    for m, (am, a) in enumerate(BASIS):
        for n, (an, b) in enumerate(BASIS):
            p = a + b
            g = 1 / math.sqrt(1 / p + 1 / ETA**2)
            ra = pos[am]
            rb = pos[an][None, :] + TVEC
            d2 = np.einsum("ij,ij->i", ra[None, :] - rb, ra[None, :] - rb)
            s = (math.pi / p) ** 1.5 * np.exp(-a * b / p * d2)
            keep = s > 1e-16
            pc = (a * ra[None, :] + b * rb[keep]) / p
            sk = s[keep]
            tot = 0.0
            for ic, (zc, rc) in enumerate(zip(ZNUC, nuc_pos)):
                if only is not None and ic != only:
                    continue
                dv = pc[:, None, :] - rc[None, None, :] - TVEC_M[None, :, :]
                d = np.sqrt(np.einsum("lmc,lmc->lm", dv, dv))
                near = d < SR_RADIUS
                dn = d[near]
                ds = np.where(dn > 1e-12, dn, 1.0)
                val = np.where(
                    dn > 1e-12,
                    (erf(math.sqrt(p) * ds) - erf(g * ds)) / ds,
                    2 * (math.sqrt(p) - g) / math.sqrt(math.pi),
                )
                w = np.broadcast_to(sk[:, None], d.shape)[near]
                tot -= zc * float(np.sum(w * val))
            out[m, n] = tot
    return out


def energy_parts(pos, das, nuc_pos=None, geo=None, with_sr=True):
    """The fixed-density energy, split as (SR, everything else)."""
    geo = GEO if geo is None else geo
    nuc_pos = pos if nuc_pos is None else nuc_pos
    vol, gv, vg, veta = geo["vol"], geo["gv"], geo["vg"], geo["veta"]
    d = das[0] + das[1]
    pg = pair_ft(pos, gv, geo["g2"], tvec=geo["tvec"])
    s = pair_ft(pos, np.zeros((1, 3)), np.zeros(1), tvec=geo["tvec"])[:, :, 0].real
    sg = (ZNUC[:, None] * np.exp(-1j * nuc_pos @ gv.T)).sum(axis=0)
    rho = np.einsum("lsg,ls->g", pg, d)
    v_lr = -np.einsum("g,mng->mn", veta, np.conj(pg) * sg[None, None, :]).real / vol
    c0 = math.pi / (ETA * ETA * vol)
    e1 = float(np.sum(d * (v_lr + c0 * ZNUC.sum() * s)))
    e_j = 0.5 * float(np.sum(vg * np.abs(rho) ** 2)) / vol
    e_k = 0.0
    for ds in das:
        # sum_{mnls} D_ml D_ns Re[P*_mn P_ls]
        e_k -= (
            0.5
            * ALPHA
            * float(
                np.real(np.einsum("ml,ns,mng,lsg,g->", ds, ds, np.conj(pg), pg, vg))
            )
            / vol
        )
    e_m = -0.5 * ALPHA * VM * sum(float(np.trace(ds @ s @ ds @ s)) for ds in das)
    e_sr = float(np.sum(d * v_sr(pos, nuc_pos))) if with_sr else 0.0
    return e_sr, e1 + e_j + e_k + e_m


def analytic_stress(pos, das, volume_term=True):
    """dE/deps_ab of the reciprocal and overlap-coupled terms at fixed density:
    each reciprocal energy gives -delta_ab E + (1/Omega) sum_G [v' (-2 G_a G_b) Phi
    + v dPhi] (S(G) is strain invariant: its phases are); the overlap terms give
    sum M dS/deps with M = c0 Z_tot D - alpha v_M sum_s D_s S D_s, and c0's volume
    dependence adds -delta_ab c0 Z_tot tr(D S)."""
    d = das[0] + das[1]
    pg, dpg = pair_ft(pos, GV, G2, strain=True)
    s0, ds0 = pair_ft(pos, np.zeros((1, 3)), np.zeros(1), strain=True)
    s, dsm = s0[:, :, 0].real, ds0[:, :, :, :, 0].real
    sg = struct(pos)
    rho = np.einsum("lsg,ls->g", pg, d)
    vg_p = -VG / G2  # d(4 pi/G^2)/dG^2
    veta_p = -VETA * (1 / G2 + 1 / (4 * ETA * ETA))
    c0 = math.pi / (ETA * ETA * VOL)
    e_j = 0.5 * float(np.sum(VG * np.abs(rho) ** 2)) / VOL
    e_lr = -float(np.sum(VETA * np.real(np.conj(rho) * sg))) / VOL
    phi_k = np.zeros(len(G2))
    for dsp in das:
        phi_k -= (
            0.5
            * ALPHA
            * np.real(np.einsum("ml,ns,mng,lsg->g", dsp, dsp, np.conj(pg), pg))
        )
    e_k = float(np.sum(VG * phi_k)) / VOL
    mmat = c0 * ZNUC.sum() * d - ALPHA * VM * sum(dsp @ s @ dsp for dsp in das)
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # the two-electron and LR nuclear energies
            gg = -2 * GV[:, i] * GV[:, j]
            drho = np.einsum("lsg,ls->g", dpg[i, j], d)
            dphi_j = np.real(np.conj(rho) * drho)
            dphi_k = np.zeros(len(G2))
            for dsp in das:
                dphi_k -= ALPHA * np.real(
                    np.einsum("ml,ns,mng,lsg->g", dsp, dsp, np.conj(dpg[i, j]), pg)
                )
            v = np.sum(vg_p * gg * 0.5 * np.abs(rho) ** 2 + VG * dphi_j)
            v += np.sum(vg_p * gg * phi_k + VG * dphi_k)
            v += -np.sum(
                veta_p * gg * np.real(np.conj(rho) * sg)
                + VETA * np.real(np.conj(drho) * sg)
            )
            out[i, j] = float(v) / VOL
            out[i, j] += float(np.sum(mmat * dsm[i, j]))
            if i == j:
                out[i, j] -= e_j + e_k + e_lr
                if volume_term:
                    out[i, j] -= c0 * ZNUC.sum() * float(np.sum(d * s))
    return out


def claim_stress(**mut):
    pos = np.array([[0.2, 0.1, 0.3], [1.5, 1.7, 1.9]])
    das = random_spin_dms(5)
    ana = analytic_stress(pos, das, **mut)
    h, worst, scale = 1e-5, 0.0, 0.0
    for i in range(3):
        for j in range(3):
            vals = []
            for sgn in (1, -1):
                m = np.eye(3)
                m[i, j] += sgn * h
                lat = LAT @ m.T
                vals.append(
                    energy_parts(pos @ m.T, das, geo=geom(lat), with_sr=False)[1]
                )
            fd = (vals[0] - vals[1]) / (2 * h)
            worst = max(worst, abs(fd - ana[i, j]))
            scale = max(scale, abs(fd))
    return worst, scale


def analytic_force(pos, das, two_e_pref=8.0, z_half=0.5, lr_nucleus=True):
    d = das[0] + das[1]
    pg, qg = pair_ft(pos, GV, G2, deriv=True)
    s, q0 = overlap(pos, deriv=True)
    sg = struct(pos)
    rho = np.einsum("lsg,ls->g", pg, d)
    z = z_half * d[:, :, None] * rho[None, None, :]
    for ds in das:
        z -= 0.5 * ALPHA * np.einsum("ml,lsg,sn->mng", ds, pg, ds)
    c0 = math.pi / (ETA * ETA * VOL)
    mmat = c0 * ZNUC.sum() * d - ALPHA * VM * sum(ds @ s @ ds for ds in das)
    h = HALF
    grad = np.zeros((2, 3))
    for atom in range(2):
        on = np.array([BASIS[m][0] == atom for m in range(NAO)])
        for c in range(3):
            qa = qg[c][on]  # (n_on, NAO, G)
            two = np.real(
                np.einsum("mng,mng,g->", np.conj(qa[:, :, h]), z[on][:, :, h], VG[h])
            )
            grad[atom, c] += two_e_pref / VOL * two
            rho_a = 2 * np.einsum("mn,mng->g", d[on], qa)
            grad[atom, c] += (
                -2 / VOL * float(np.sum(VETA[h] * np.real(np.conj(rho_a[h]) * sg[h])))
            )
            if lr_nucleus:
                ds_dr = -1j * GV[:, c] * ZNUC[atom] * np.exp(-1j * GV @ pos[atom])
                grad[atom, c] += (
                    -2
                    / VOL
                    * float(np.sum(VETA[h] * np.real(np.conj(rho[h]) * ds_dr[h])))
                )
            grad[atom, c] += 2 * float(np.sum(mmat[on] * q0[c][on]))
    return grad


def sr_force_by_invariance(pos, das, h=1e-5):
    """SR row: the basis derivative (basis moved, nuclei fixed) plus the
    nuclear derivative as minus the derivative with every basis function moved
    together (translation invariance of each three-centre integral)."""
    d = das[0] + das[1]
    out = np.zeros((2, 3))
    for atom in range(2):
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[atom, c] += h
            pm[atom, c] -= h
            basis = (np.sum(d * v_sr(pp, pos)) - np.sum(d * v_sr(pm, pos))) / (2 * h)

            # Nucleus `atom`: each of its integrals depends on the bra, ket and
            # nuclear centres through their differences only, so its nuclear
            # derivative is minus the derivative with bra and ket moved together.
            def e_nuc(sh, c=c, atom=atom):
                shifted = pos.copy()
                shifted[:, c] += sh
                return np.sum(d * v_sr(shifted, pos, only=atom))

            nucleus = -(e_nuc(h) - e_nuc(-h)) / (2 * h)
            out[atom, c] = basis + nucleus
    return out


def random_spin_dms(seed):
    rng = np.random.default_rng(seed)
    out = []
    for scale in (0.9, 0.4):
        x = rng.standard_normal((NAO, NAO))
        out.append(scale * (x @ x.T) / NAO)
    return out


def claim_force(**mut):
    pos = np.array([[0.2, 0.1, 0.3], [1.5, 1.7, 1.9]])
    das = random_spin_dms(5)
    ana = analytic_force(pos, das, **mut) + sr_force_by_invariance(pos, das)
    h = 1e-5
    worst, scale = 0.0, 0.0
    for atom in range(2):
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[atom, c] += h
            pm[atom, c] -= h
            fd = (sum(energy_parts(pp, das)) - sum(energy_parts(pm, das))) / (2 * h)
            worst = max(worst, abs(fd - ana[atom, c]))
            scale = max(scale, abs(fd))
    return worst, scale


def fit_energy(j2, j3, das, lindep):
    """E = sum Gamma I^fit, I^fit = J3 f(J2) J3^T, f(s) = 1/s above lindep, else 0."""
    s, u = np.linalg.eigh(j2)
    f = u @ np.diag(np.where(s > lindep, 1 / s, 0.0)) @ u.T
    i_fit = j3 @ f @ j3.T  # (n^2, n^2)
    n = das[0].shape[0]
    i4 = i_fit.reshape(n, n, n, n)
    d = das[0] + das[1]
    e = 0.5 * np.einsum("mn,ls,mnls->", d, d, i4)
    for ds in das:
        e -= 0.5 * ALPHA * np.einsum("ml,ns,mnls->", ds, ds, i4)
    return float(e)


def fit_derivative(j2, j3, das, lindep, kept_dropped=True):
    """dE = sum Y dJ3 + sum Wm dJ2 (rsgdf/deriv.rs module doc):
    Y_P = D (C_P . D) - alpha sum_s D_s C_P D_s, C = f(J2) J3^T;
    Wm = U (Lo o U^T H U) U^T, H = J3^T Gamma J3, Lo the Loewner matrix of f."""
    n = das[0].shape[0]
    s, u = np.linalg.eigh(j2)
    keep = s > lindep
    fs = np.where(keep, 1 / s, 0.0)
    c = (u @ np.diag(fs) @ u.T) @ j3.T  # (naux, n^2)
    d = das[0] + das[1]
    y = np.zeros_like(c)
    for p in range(c.shape[0]):
        cp = c[p].reshape(n, n)
        yp = d * np.sum(cp * d)
        for ds in das:
            yp -= ALPHA * ds @ cp @ ds
        y[p] = yp.reshape(-1)
    # H_PQ = sum Gamma J3_{mn,P} J3_{ls,Q}; with Gamma(X, Y) = 1/2 Y^X . Y.
    h = np.zeros((c.shape[0], c.shape[0]))
    for p in range(c.shape[0]):
        jp = j3[:, p].reshape(n, n)
        yp = d * np.sum(jp * d)
        for ds in das:
            yp -= ALPHA * ds @ jp @ ds
        h[p] = 0.5 * yp.reshape(-1) @ j3
    lo = np.zeros((len(s), len(s)))
    for i in range(len(s)):
        for j in range(len(s)):
            if keep[i] and keep[j]:
                lo[i, j] = -1 / (s[i] * s[j])
            elif kept_dropped and keep[i] != keep[j]:
                lo[i, j] = (fs[i] - fs[j]) / (s[i] - s[j])
    wm = u @ (lo * (u.T @ h @ u)) @ u.T
    return y, wm


def claim_fit_derivative(kept_dropped=True):
    """The RS-GDF two-electron force at fixed density through the fit: a
    perturbation J2 + t A, J3 + t B gives dE/dt = sum Y.B^T + sum Wm.A, against
    a central difference, with one metric eigenvalue below the cut so the
    kept-dropped Loewner block carries energy."""
    rng = np.random.default_rng(17)
    n, naux, lindep = 3, 5, 1e-3
    x = rng.standard_normal((naux, naux))
    q, _ = np.linalg.qr(x)
    j2 = q @ np.diag([2.0, 1.3, 0.7, 0.2, 2e-4]) @ q.T
    j3 = rng.standard_normal((n * n, naux))
    j3 = 0.5 * (j3 + j3.reshape(n, n, naux).transpose(1, 0, 2).reshape(n * n, naux))
    das = random_spin_dms(23)[:2]
    das = [ds[:n, :n] for ds in das]
    a = rng.standard_normal((naux, naux))
    a = 0.5 * (a + a.T)
    b = rng.standard_normal((n * n, naux))
    y, wm = fit_derivative(j2, j3, das, lindep, kept_dropped)
    ana = float(np.sum(y * b.T) + np.sum(wm * a))
    h = 1e-6
    fd = (
        fit_energy(j2 + h * a, j3 + h * b, das, lindep)
        - fit_energy(j2 - h * a, j3 - h * b, das, lindep)
    ) / (2 * h)
    return abs(ana - fd), abs(fd)


def main():
    err, scale = claim_fit_derivative()
    check(
        "rsgdf-fit-derivative",
        err < 1e-6 * max(1.0, scale),
        f"|dE analytic - FD| = {err:.1e} (dE/dt {scale:.3f}), one eigenvalue below the cut",
    )
    worst_s, scale_s = claim_stress()
    check(
        "stress-fixed-density",
        worst_s < 1e-7 and scale_s > 1e-2,
        f"max |analytic dE/deps - FD at frozen index sets| = {worst_s:.1e} Ha "
        f"(largest {scale_s:.3f})",
    )
    worst, scale = claim_force()
    check(
        "force-fixed-density",
        worst < 1e-7 and scale > 1e-2,
        f"max |analytic - FD| = {worst:.1e} Ha/Bohr (largest component {scale:.3f})",
    )
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
