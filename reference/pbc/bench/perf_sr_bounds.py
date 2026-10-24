"""ESTIMATES (no ferric run): SR 3-centre triplet counts under alternative, still-rigorous screens.

Reuses reference/pbc/bench/cell_facts.py's shell data and capsule estimator. V0 is ferric's
current rsgdf.rs rule. Alternatives are evaluated on an importance sample of (L, mu-shell, nu-shell)
pair images drawn proportional to their V0 triplet count; the reported count for variant X is
V0_total * mean(X(s)/V0(s)) over the sample (ratio estimator; V0 itself is computed in full).

  V0  current: shell-level q = max_k q_k, (1 + 2 mu_max/sqrt(pi)) e^{-nu_min^2 (d-2)^2}
  V1  V0 with zero-coefficient primitives removed from p_min/p_max
  V2  primitive envelope: per (obs prim pair k, aux prim m) the V0 functional form with that
      combination's own q_k Q_m, mu_km, nu_km; rigorous sum bound -> threshold t/(N_k N_m)
  V3  V2 with the exact s-type SR kernel envelope q Q erfc(nu d')/d' (d' = d - 2, margin kept),
      capped at the contact value q Q 2(eta - nu)/sqrt(pi); erfc(x) < e^{-x^2}/(x sqrt(pi))
  V4  V3 minus every combination whose FULL Coulomb converges inside the existing LR G sphere:
      1/p_k + 1/alpha_m >= 1/w^2 (PySCF RSGDF exclude_d_aux + fft_dd_block, at prim level)
  V4s V3 minus the PySCF-style SHELL-level split: aux prim alpha <= w^2 -> G space; obs prim pair
      with both exponents <= w^2/2 -> G space (partial decontraction into compact/smooth shells)
"""

import math
import os
import sys
import time

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_facts as cf  # noqa: E402

PREC = 1e-13
SQPI = math.sqrt(math.pi)


def radius_v0form(q, Q, mu, nu, t):
    pref = q * Q * (1 + 2 * mu / SQPI)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            pref > t, np.sqrt(np.log(np.maximum(pref, t) / t)) / nu + 2.0, np.nan
        )


def radius_erfc(q, Q, eta, nu, t):
    """Smallest d with q Q erfc(nu d')/d' <= t, d' = d - 2 (via e^{-x^2}/(x sqrt pi) bound);
    NaN if even the contact value q Q 2 (eta - nu)/sqrt(pi) is below t."""
    qq, eta, nu, t = np.broadcast_arrays(q * Q, eta, nu, t)
    contact = qq * 2 * np.maximum(eta - nu, 0) / SQPI
    out = np.full(np.shape(qq), np.nan)
    ok = contact > t
    if not ok.any():
        return out
    a = qq[ok] / (t[ok] * nu[ok] * SQPI)  # e^{-nu^2 d^2} / d^2 = 1/a
    x = np.sqrt(np.maximum(np.log(np.maximum(a * nu[ok] ** 2, 1.0)), 1e-12))  # x = nu d
    for _ in range(6):  # x^2 + 2 ln(x/nu) = ln a
        x = np.sqrt(
            np.maximum(np.log(a) - 2 * np.log(np.maximum(x / nu[ok], 1e-3)), 1e-12)
        )
    d = x / nu[ok]
    # verify with the real erfc (the bound is only for x>0; the fixed point can undershoot)
    for _ in range(40):
        val = qq[ok] * erfc(nu[ok] * d) / np.maximum(d, 1e-12)
        bad = val > t[ok]
        if not bad.any():
            break
        d = np.where(bad, d * 1.05 + 0.05, d)
    out[ok] = d + 2.0
    return out


def prim_arrays(shells):
    out = []
    for s in shells:
        nz = s["k"] != 0.0
        out.append(dict(e=s["e"], k=s["k"], nz=nz, l=s["l"], c=s["c"]))
    return out


def main(cellname, omegas=(1.0,), nsample=800, seed=7):
    sym, xyz, lat = cf.read_cell(cellname)
    obs = cf.ferric_shells("cc-pvdz", sym, xyz)
    aux = cf.ferric_shells("def2-universal-jkfit", sym, xyz)
    vol = abs(np.linalg.det(lat))
    A = np.array([s["c"] for s in obs])
    ns = len(obs)
    # aux primitives (flattened) with shell index
    ap_sh, ap_a, ap_Q, ap_l = [], [], [], []
    for i, p in enumerate(aux):
        for e, k in zip(p["e"], p["k"]):
            ap_sh.append(i)
            ap_a.append(e)
            ap_Q.append(abs(k) * (math.pi / e) ** 1.5 * (1 + e**-0.5) ** p["l"])
            ap_l.append(p["l"])
    ap_sh, ap_a, ap_Q = np.array(ap_sh), np.array(ap_a), np.array(ap_Q)
    naux_sh = len(aux)
    nprim_of_aux = np.bincount(ap_sh, minlength=naux_sh)
    qaux = np.array(
        [
            np.max(
                np.abs(p["k"])
                * (math.pi / p["e"]) ** 1.5
                * (1 + p["e"] ** -0.5) ** p["l"]
            )
            for p in aux
        ]
    )
    aamin = np.array([p["amin"] for p in aux])
    aamax = np.array([p["amax"] for p in aux])
    K, C, pmin, pmax = cf.pair_tables(obs)
    # nonzero-coefficient pmin/pmax (V1)
    pmin_nz = np.zeros((ns, ns))
    pmax_nz = np.zeros((ns, ns))
    for i, a in enumerate(obs):
        for j, b in enumerate(obs):
            ea = a["e"][a["k"] != 0]
            eb = b["e"][b["k"] != 0]
            p = ea[:, None] + eb[None, :]
            pmin_nz[i, j], pmax_nz[i, j] = p.min(), p.max()
    amin_orb = min(s["amin"] for s in obs)
    pmax_orb = 2 * max(s["amax"] for s in obs)
    vfac = max(
        1.0,
        max(
            1 + 2 * math.sqrt(pmax_orb * p["amax"] / (pmax_orb + p["amax"])) / SQPI
            for p in aux
        ),
    )
    pair_thresh = 0.1 * PREC / max(qaux.max() * vfac, 1.0)
    rpair = math.sqrt(2 * math.log(1e3 / pair_thresh) / amin_orb) + 2.0
    images = cf.lattice_points(lat, rpair)
    res = {}
    for w in omegas:
        t0 = time.time()

        def v0_counts(pmn, pmx, prec=PREC):
            rows = []
            for li, L in enumerate(images):
                B = A + L
                d = A[:, None, :] - B[None, :, :]
                r2 = np.einsum("ijk,ijk->ij", d, d)
                q = np.max(C * np.exp(-K * r2[:, :, None]), axis=2)
                seg = np.sqrt(r2)
                mu = np.sqrt(pmx[:, :, None] * aamax / (pmx[:, :, None] + aamax))
                nu = 1 / np.sqrt(1 / pmn[:, :, None] + 1 / aamin + 1 / w**2)
                rad = radius_v0form(q[:, :, None], qaux[None, None, :], mu, nu, prec)
                cap = (
                    math.pi * rad**2 * seg[:, :, None] + 4.0 / 3.0 * math.pi * rad**3
                ) / vol
                cap = np.where(np.isnan(cap), 0.0, cap)
                rows.append(cap.sum(axis=2))
            return np.stack(rows)  # (nL, ns, ns)

        per_pair0 = v0_counts(pmin, pmax)
        tot0 = per_pair0.sum()
        tot1 = v0_counts(pmin_nz, pmax_nz).sum()
        tot_x1000 = v0_counts(pmin, pmax, PREC * 1e3).sum()
        flat = per_pair0.ravel()
        rng = np.random.default_rng(seed)
        prob = flat / flat.sum()
        idx = rng.choice(flat.size, size=nsample, p=prob)
        ratios = {k: [] for k in ("V2", "V3", "V4", "V4s")}
        for s in idx:
            li, i, j = np.unravel_index(s, per_pair0.shape)
            L = images[li]
            a, b = obs[i], obs[j]
            bc = A[j] + L
            r2 = float(((A[i] - bc) ** 2).sum())
            seg = math.sqrt(r2)
            ea, eb = a["e"], b["e"]
            ka, kb = a["k"], b["k"]
            P = ea[:, None] + eb[None, :]
            qk = (
                np.abs(ka[:, None] * kb[None, :])
                * (math.pi / P) ** 1.5
                * np.exp(-ea[:, None] * eb[None, :] / P * r2)
            ).ravel()
            Pk = P.ravel()
            smooth_pair = (
                (ea[:, None] <= w * w / 2) & (eb[None, :] <= w * w / 2)
            ).ravel()
            keep = qk > 0
            qk, Pk, smooth_pair = qk[keep], Pk[keep], smooth_pair[keep]
            Nk = qk.size
            # (k, m) grids
            q2 = qk[:, None]
            p2 = Pk[:, None]
            al = ap_a[None, :]
            Qm = ap_Q[None, :]
            Nm = nprim_of_aux[ap_sh][None, :]
            t = PREC / (Nk * Nm)
            mu = np.sqrt(p2 * al / (p2 + al))
            nu = 1 / np.sqrt(1 / p2 + 1 / al + 1 / w**2)
            eta = mu
            r_v2 = radius_v0form(q2, Qm, mu, nu, t)
            r_v3 = radius_erfc(q2, Qm, eta, nu, t)
            g_ok = (1 / p2 + 1 / al) >= 1 / w**2
            r_v4 = np.where(g_ok, np.nan, r_v3)
            s_ok = smooth_pair[:, None] | (al <= w * w)
            r_v4s = np.where(s_ok, np.nan, r_v3)
            cap0_s = per_pair0[li, i, j]
            for key, rr in (("V2", r_v2), ("V3", r_v3), ("V4", r_v4), ("V4s", r_v4s)):
                rsh = np.full(naux_sh, np.nan)
                with np.errstate(invalid="ignore"):
                    colmax = np.where(
                        np.all(np.isnan(rr), axis=0),
                        np.nan,
                        np.nanmax(np.where(np.isnan(rr), -1, rr), axis=0),
                    )
                np.fmax.at(rsh, ap_sh, colmax)
                capx = (math.pi * rsh**2 * seg + 4.0 / 3.0 * math.pi * rsh**3) / vol
                capx = np.nansum(capx)
                ratios[key].append(capx / cap0_s)
        est = {k: float(tot0 * np.mean(v)) for k, v in ratios.items()}
        sem = {
            k: float(tot0 * np.std(v) / math.sqrt(len(v))) for k, v in ratios.items()
        }
        ng = cf.count_g_half(lat, 2 * w * math.sqrt(math.log(1 / PREC)))
        res[w] = dict(
            V0=float(tot0),
            V1=float(tot1),
            V0_thresh_x1000=float(tot_x1000),
            **est,
            sem=sem,
            n_g_half=ng,
            secs=time.time() - t0,
        )
        print(
            cellname,
            "w",
            w,
            {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in res[w].items()},
            flush=True,
        )
    return res


if __name__ == "__main__":
    cell = sys.argv[1]
    oms = tuple(float(x) for x in sys.argv[2:]) or (1.0,)
    main(cell, oms)
