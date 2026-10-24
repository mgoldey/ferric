"""Pre-run facts and ferric cost PREDICTIONS for the real-size benchmark cells (no ferric run).

Measurement-free except for the lattice overlap spectrum (PySCF pbc_intor, seconds). Everything
else re-implements ferric-pbc's own truncation rules so the counts are what ferric WILL do:

  RS-GDF (crates/ferric-pbc/src/rsgdf.rs, RsGdfConfig defaults w = 1, precision 1e-13, lindep 1e-10):
    pair images      |L| <= r_pair = sqrt(2 ln(1e3/t)/a_min) + 2,  t = 0.1 prec / max(Q_aux vfac, 1)
    LR G half-sphere |G| <= 2 w sqrt(ln 1/prec)
    SR 3c triplets   (mu_0 nu_L | P_T): per (L, mu-shell, nu-shell, P-shell) the radius
                     sr_radius(q_ab, Q_P, ...) and every aux image within it of the segment [A, B+L]
    SR 2c pairs      (P_0 | Q_T) within sr_radius(Q_P, Q_Q, ...)
  periodic hcore (crates/ferric-pbc/src/hcore.rs, precision 1e-14,
                  w_h = min(2.5 sqrt(pi)/V^(1/3), sqrt(ln 1e13 / ln 1/prec)) = default_hcore_omega):
    pair images at t = 1e-15; SR Gaussian-nucleus triplets per (L, mu, nu, nucleus image) within
    nucleus_radius(q, Zmax, pmax, w_p) of the segment; candidate list = nuclei within r_nuc_max + r_pair

Triplet counts are ESTIMATED: the number of lattice images of a centre inside a capsule of radius r
around a segment of length s is taken as its volume (pi r^2 s + 4/3 pi r^3) / Omega. `--exact`
enumerates the lattice points instead (slow; use it only on the smallest cell, to calibrate the
estimator). Primitive data mirror ferric: BSE-JSON contraction columns renormalised to unit
self-overlap, ZERO coefficients KEPT (ferric does not drop them; they enter p_min/p_max).

STALE SINCE WRITTEN (2026-09-25; noted 2026-10-24 when ported to main): these are the ORDERED (s1),
UNROTATED SR walks. ferric now computes each Gamma orbital pair once (c83476b80, orbital-pair
symmetry) and rotates generally contracted columns by default (282be9cac). Compare these counts with
RsGdf's `n_sr3_triplets_ordered` on a `sr_column_rotation = false` run, not with `n_sr3_triplets`.

Usage (OPENBLAS_NUM_THREADS=1; < 1 min per cell except dryice_112 with --counts, ~1-2 min):
  python reference/pbc/bench/cell_facts.py diamond_prim [--aux def2-universal-jkfit] [--counts] [--exact]
Writes reference/pbc/bench/out/facts_<cell>_<aux>.json and prints a summary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BUNDLED = os.path.join(REPO, "crates", "ferric-core", "src", "basis", "bundled")
BOHR = 0.52917721092
Z = {"H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "Si": 14}
DFACT = [1.0, 1.0, 3.0, 15.0, 105.0]


def read_cell(name):
    with open(os.path.join(HERE, name + ".xyz")) as f:
        lines = f.read().split("\n")
    nat = int(lines[0])
    sym, xyz = [], []
    for ln in lines[2 : 2 + nat]:
        p = ln.split()
        sym.append(p[0])
        xyz.append([float(v) / BOHR for v in p[1:4]])
    lat = np.loadtxt(os.path.join(HERE, name + ".lattice"))
    return sym, np.array(xyz), lat


def prim_norm(a, l):
    return (2 * a / math.pi) ** 0.75 * math.sqrt((4 * a) ** l) / math.sqrt(DFACT[l])


def ferric_shells(basis, symbols, coords):
    """Per shell: (l, pure, centre, exps, coefs*prim_norm) exactly as ferric's parse_bse_json
    (renormalise_contraction: unit self-overlap) + md3c1e prim_norm."""
    d = json.load(open(os.path.join(BUNDLED, basis + ".json")))
    out = []
    for s, c in zip(symbols, coords):
        for sh in d["elements"][str(Z[s])]["electron_shells"]:
            exps = np.array([float(x) for x in sh["exponents"]])
            pure = sh.get("function_type") == "gto_spherical"
            ls = sh["angular_momentum"]
            cols = sh["coefficients"]
            pairs = (
                [(ls[0], col) for col in cols] if len(ls) == 1 else list(zip(ls, cols))
            )
            for l, col in pairs:
                cf = np.array([float(x) for x in col])
                # unit self-overlap of the contracted, primitive-normalised function
                n = np.array([prim_norm(a, l) for a in exps])
                p = exps[:, None] + exps[None, :]
                sab = (math.pi / p) ** 1.5 * DFACT[l] / (2 * p) ** l
                norm = np.einsum("i,j,i,j,ij->", cf, cf, n, n, sab)
                cf = cf / math.sqrt(norm)
                out.append(
                    dict(
                        l=l,
                        pure=pure and l >= 2,
                        c=np.array(c),
                        e=exps,
                        k=cf * n,
                        amin=exps.min(),
                        amax=exps.max(),
                    )
                )
    return out


def nfun(sh):
    return 2 * sh["l"] + 1 if sh["pure"] else (sh["l"] + 1) * (sh["l"] + 2) // 2


def lattice_points(lat, r, x0=np.zeros(3)):
    """Lattice translations T = n @ lat (rows of `lat` are the lattice vectors) with |T - x0| <= r.

    The integer box is centred on the lattice point nearest x0 (c0 = rint(x0 @ inv(lat))), not on
    the origin: a box centred on 0 drops the points near a far-from-origin x0. The half-width
    ceil(r |b_j| / 2 pi) + 1 covers the radius plus the <= 1/2 rounding of c0 in each direction.
    """
    b = 2 * np.pi * np.linalg.inv(lat).T
    c0 = np.rint(np.asarray(x0, float) @ np.linalg.inv(lat))
    rng = [int(math.ceil(r * np.linalg.norm(b[j]) / (2 * np.pi))) + 1 for j in range(3)]
    n = c0 + np.array(list(itertools.product(*[range(-k, k + 1) for k in rng])), float)
    t = n @ lat
    return t[np.linalg.norm(t - x0, axis=1) <= r]


def count_g_half(lat, gcut):
    b = 2 * np.pi * np.linalg.inv(lat).T
    rng = [
        int(math.ceil(gcut / (2 * np.pi) * np.linalg.norm(lat[j]))) + 1
        for j in range(3)
    ]
    n = np.array(list(itertools.product(*[range(-k, k + 1) for k in rng])), float)
    g = np.linalg.norm(n @ b, axis=1)
    return int(((g <= gcut) & (g > 0)).sum() // 2)


def g_half_norms2(lat, gcut):
    """Sorted |G|^2 of the half sphere 0 < |G| <= gcut (one of each +-G)."""
    b = 2 * np.pi * np.linalg.inv(lat).T
    rng = [
        int(math.ceil(gcut / (2 * np.pi) * np.linalg.norm(lat[j]))) + 1
        for j in range(3)
    ]
    n = np.array(list(itertools.product(*[range(-k, k + 1) for k in rng])), float)
    g2 = np.einsum("ij,ij->i", n @ b, n @ b)
    g2 = np.sort(g2[(g2 <= gcut * gcut) & (g2 > 0)])
    return g2[::2]


def lr_pairft_predict(obs, lat, nao, naux, gcut, thresh=1e-15, lmax=2):
    """rsgdf.rs lr_accumulate -> pair_ft_chunked: the (primitive pair, image, G) evaluations of
    the analytic pair FT inside each primitive pair's G window, and the per-chunk overhead
    (pair_ft_block re-walks every shell pair x image x primitive pair for EVERY G chunk)."""
    g2 = g_half_norms2(lat, gcut)
    gmax = math.sqrt(g2[-1])
    prims = [(s["c"], e, k, s["l"]) for s in obs for e, k in zip(s["e"], s["k"])]
    C = np.array([p[0] for p in prims])
    E = np.array([p[1] for p in prims])
    Kc = np.array([p[2] for p in prims])
    Lq = np.array([p[3] for p in prims])
    amin = E.min()
    rpair = math.sqrt(2 * math.log(1e3 / thresh) / amin) + 2.0
    images = lattice_points(lat, rpair)
    P = E[:, None] + E[None, :]
    CC = np.abs(Kc[:, None] * Kc[None, :]) * (math.pi / P) ** 1.5
    RED = E[:, None] * E[None, :] / P
    LL = (Lq[:, None] + Lq[None, :]) * math.log(max(gmax, 1.0))
    evals = 0.0
    nsurv = 0
    for L in images:
        d = C[:, None, :] - (C[None, :, :] + L)
        r2 = np.einsum("ijk,ijk->ij", d, d)
        mag = CC * np.exp(-RED * r2)
        ok = mag >= thresh
        g2max = 4 * P[ok] * (np.log(mag[ok] / thresh) + 10.0 + LL[ok])
        evals += np.searchsorted(g2, g2max, side="right").sum()
        nsurv += int(ok.sum())
    ncart = (lmax + 1) * (lmax + 2) // 2
    per_g = (
        16 * nao * nao
        + 40
        + 16 * 3 * (2 * lmax + 1)
        + 16
        + 16 * 3 * (lmax + 1) ** 2
        + 32 * ncart * ncart
    )
    per_g += 16 * nao * nao + 48 * naux + 64
    chunk = max(1, (64 << 20) // per_g)
    nchunks = math.ceil(len(g2) / chunk)
    return dict(
        pairft_rpair=rpair,
        pairft_images=len(images),
        n_primitives=len(prims),
        surviving_prim_pair_images=nsurv,
        prim_pair_G_evals=float(evals),
        g_per_chunk=chunk,
        n_chunks=nchunks,
        per_chunk_prim_pair_image_tests=len(prims) ** 2 * len(images),
        total_prim_pair_image_tests=float(len(prims) ** 2 * len(images) * nchunks),
    )


def pair_tables(shells):
    """Padded primitive-pair arrays over all ordered shell pairs (zero coefs kept)."""
    ns = len(shells)
    mp = max(len(s["e"]) for s in shells) ** 2
    K = np.zeros((ns, ns, mp))  # ab/p
    C = np.zeros((ns, ns, mp))  # |ca cb| (pi/p)^1.5
    pmin = np.zeros((ns, ns))
    pmax = np.zeros((ns, ns))
    for i, a in enumerate(shells):
        for j, b in enumerate(shells):
            p = a["e"][:, None] + b["e"][None, :]
            k = (a["e"][:, None] * b["e"][None, :] / p).ravel()
            c = (
                np.abs(a["k"][:, None] * b["k"][None, :]) * (math.pi / p) ** 1.5
            ).ravel()
            K[i, j, : k.size] = k
            C[i, j, : c.size] = c
            pmin[i, j], pmax[i, j] = p.min(), p.max()
    return K, C, pmin, pmax


def capsule_count(r, s, vol):
    return (math.pi * r * r * s + 4.0 / 3.0 * math.pi * r**3) / vol


def rsgdf_predict(obs, aux, lat, prec=1e-13, w=1.0, exact=False):
    vol = abs(np.linalg.det(lat))
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
    amin_orb = min(s["amin"] for s in obs)
    pmax_orb = 2 * max(s["amax"] for s in obs)
    vfac = max(
        1.0,
        max(
            1
            + 2
            * math.sqrt(pmax_orb * p["amax"] / (pmax_orb + p["amax"]))
            / math.sqrt(math.pi)
            for p in aux
        ),
    )
    pair_thresh = 0.1 * prec / max(qaux.max() * vfac, 1.0)
    rpair = math.sqrt(2 * math.log(1e3 / pair_thresh) / amin_orb) + 2.0
    images = lattice_points(lat, rpair)
    gcut = 2 * w * math.sqrt(math.log(1 / prec))
    ng = count_g_half(lat, gcut)

    def sr_radius(qa, qb, amn, amx, bmn, bmx):
        mu = np.sqrt(amx * bmx / (amx + bmx))
        nu = 1 / np.sqrt(1 / amn + 1 / bmn + 1 / w**2)
        pref = qa * qb * (1 + 2 * mu / math.sqrt(math.pi))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(
                pref > prec,
                np.sqrt(np.log(np.maximum(pref, prec) / prec)) / nu + 2.0,
                np.nan,
            )
        return r

    # SR metric pairs
    n2 = 0.0
    for i, p in enumerate(aux):
        r = sr_radius(qaux[i], qaux, p["amin"], p["amax"], aamin, aamax)
        ok = ~np.isnan(r)
        n2 += np.sum(4.0 / 3.0 * math.pi * r[ok] ** 3 / vol + 1.0)
    # SR 3-centre triplets
    K, C, pmin, pmax = pair_tables(obs)
    A = np.array([s["c"] for s in obs])
    auxc = np.array([p["c"] for p in aux])
    n3 = 0.0
    n3_exact = 0
    ns = len(obs)
    for L in images:
        B = A + L
        d = A[:, None, :] - B[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        q = np.max(C * np.exp(-K * r2[:, :, None]), axis=2)  # (ns, ns)
        seg = np.sqrt(r2)
        rad = sr_radius(
            q[:, :, None],
            qaux[None, None, :],
            pmin[:, :, None],
            pmax[:, :, None],
            aamin,
            aamax,
        )
        ok = ~np.isnan(rad)
        cap = (math.pi * rad**2 * seg[:, :, None] + 4.0 / 3.0 * math.pi * rad**3) / vol
        n3 += np.sum(cap[ok])
        if exact:
            for i, j, k in zip(*np.nonzero(ok)):
                a, bc = A[i], B[j]
                mid = 0.5 * (a + bc)
                pts = (
                    lattice_points(lat, rad[i, j, k] + 0.5 * seg[i, j], mid - auxc[k])
                    + auxc[k]
                )
                ab = bc - a
                l2 = ab @ ab
                t = (
                    np.clip(((pts - a) @ ab) / l2, 0, 1)
                    if l2 > 0
                    else np.zeros(len(pts))
                )
                dist = np.linalg.norm(pts - a - t[:, None] * ab, axis=1)
                n3_exact += int((dist <= rad[i, j, k]).sum())
    return dict(
        rpair=rpair,
        n_pair_images=len(images),
        gcut=gcut,
        n_g_half=ng,
        sr2_pairs_est=float(n2),
        sr3_triplets_est=float(n3),
        sr3_triplets_exact=n3_exact if exact else None,
        qaux_max=float(qaux.max()),
        n_obs_shells=ns,
        n_aux_shells=len(aux),
    )


def default_hcore_omega(vol, prec):
    """ferric's hcore.rs default_hcore_omega: 2.5 x the Ewald split, capped so the hcore LR
    sphere stays inside the default RS-GDF one (omega 1, precision 1e-13)."""
    return min(
        2.5 * math.sqrt(math.pi) / vol ** (1 / 3),
        math.sqrt(math.log(1e13) / math.log(1 / prec)),
    )


def hcore_predict(obs, symbols, coords, lat, prec=1e-14, exact=False):
    vol = abs(np.linalg.det(lat))
    w = default_hcore_omega(vol, prec)
    pair_thresh = 0.1 * prec
    amin = min(s["amin"] for s in obs)
    rpair = math.sqrt(2 * math.log(1e3 / pair_thresh) / amin) + 2.0
    images = lattice_points(lat, rpair)
    zs = np.array([Z[s] for s in symbols], float)
    zmax = zs.max()
    K, C, pmin, pmax = pair_tables(obs)
    q0 = np.max(C, axis=2)
    wp0 = w * np.sqrt(pmin / (pmin + w * w))
    pref0 = q0 * zmax * (1 + 2 * np.sqrt(pmax / math.pi))
    r_nuc_max = np.nanmax(
        np.where(
            pref0 > prec,
            np.sqrt(np.log(np.maximum(pref0, prec) / prec)) / wp0 + 2.0,
            np.nan,
        )
    )
    rc = r_nuc_max + rpair
    ncand = len(lattice_points(lat, rc)) * len(symbols)
    A = np.array([s["c"] for s in obs])
    n3, nseg, nlab = 0.0, 0, 0
    for L in images:
        B = A + L
        d = A[:, None, :] - B[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        q = np.max(C * np.exp(-K * r2[:, :, None]), axis=2)
        pref = q * zmax * (1 + 2 * np.sqrt(pmax / math.pi))
        ok = pref > prec
        rad = np.sqrt(np.log(np.maximum(pref, prec) / prec)) / wp0 + 2.0
        seg = np.sqrt(r2)
        n3 += np.sum(
            ((math.pi * rad**2 * seg + 4.0 / 3.0 * math.pi * rad**3) / vol)[ok]
        ) * len(symbols)
        nlab += int(ok.sum())
    nseg = nlab * ncand
    return dict(
        omega=w,
        rpair=rpair,
        n_pair_images=len(images),
        r_nuc_max=float(r_nuc_max),
        n_candidates=ncand,
        surviving_pair_images=nlab,
        segment_tests=nseg,
        sr_nucleus_triplets_est=float(n3),
    )


def overlap_spectrum(symbols, coords, lat, basis, kmesh=None):
    from pyscf.pbc import gto as pgto

    from pyscf_bench import ferric_basis_pyscf  # same directory

    cell = pgto.Cell()
    cell.a = lat
    cell.unit = "B"
    cell.atom = [(s, c) for s, c in zip(symbols, coords)]
    cell.basis = ferric_basis_pyscf(basis, symbols)
    cell.cart = False
    cell.precision = 1e-12
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(kmesh) if kmesh else np.zeros((1, 3))
    out = []
    for k, s in zip(kpts, cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)):
        e = np.linalg.eigvalsh(s)
        out.append(
            dict(
                k=k.tolist(),
                lam_min=float(e[0]),
                n_below_1e6=int((e < 1e-6).sum()),
                n_below_1e8=int((e < 1e-8).sum()),
            )
        )
    return cell.nao_nr(), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--aux", default="def2-universal-jkfit")
    ap.add_argument(
        "--counts",
        action="store_true",
        help="estimate SR triplet counts (numpy, can take ~1 min)",
    )
    ap.add_argument(
        "--exact",
        action="store_true",
        help="also enumerate SR3 triplets exactly (tiny cells only)",
    )
    ap.add_argument("--kmesh", type=int, nargs=3, default=None)
    ap.add_argument("--no-overlap", action="store_true")
    a = ap.parse_args()
    sym, xyz, lat = read_cell(a.cell)
    obs = ferric_shells(a.basis, sym, xyz)
    aux = ferric_shells(a.aux, sym, xyz)
    nao = sum(nfun(s) for s in obs)
    naux = sum(nfun(s) for s in aux)
    nelec = sum(Z[s] for s in sym)
    vol = abs(np.linalg.det(lat))
    res = dict(
        cell=a.cell,
        basis=a.basis,
        aux=a.aux,
        natoms=len(sym),
        nelec=nelec,
        nocc=nelec // 2,
        nao=nao,
        naux=naux,
        volume_bohr3=vol,
        aux_max_l=max(p["l"] for p in aux),
        memory_bytes=dict(
            B_resident=8 * naux * nao * nao,
            build_peak_j3_bt_or_bt_b=16 * naux * nao * nao,
            ledger_reservation=24 * naux * nao * nao
            + 32 * nao * nao
            + 32 * naux * naux,
        ),
        flops=dict(
            K_per_iter_dense_D=4 * naux * nao**3,
            J_per_iter=4 * naux * nao**2,
            K_per_iter_occ_path=4 * naux * nao**2 * (nelec // 2),
        ),
    )
    t0 = time.time()
    if a.counts:
        res["rsgdf"] = rsgdf_predict(obs, aux, lat, exact=a.exact)
        res["rsgdf"]["lr_gemm_flops"] = 4 * nao * nao * naux * res["rsgdf"]["n_g_half"]
        res["rsgdf"]["lr_pair_ft"] = lr_pairft_predict(
            obs, lat, nao, naux, res["rsgdf"]["gcut"]
        )
        res["hcore"] = hcore_predict(obs, sym, xyz, lat)
    if not a.no_overlap:
        nao_p, spec = overlap_spectrum(sym, xyz, lat, a.basis, a.kmesh)
        assert nao_p == nao, (nao_p, nao)
        res["overlap"] = spec
    res["facts_seconds"] = time.time() - t0
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    path = os.path.join(HERE, "out", f"facts_{a.cell}_{a.aux}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
