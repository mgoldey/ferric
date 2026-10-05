"""Fourier-space terf/terfc integrals and SCS-MP2(2terfc) energies (issue #275).

Consumers:
  crates/ferric-integrals/tests/validation_terfc_integrals.rs  (integrals)
  crates/ferric-mp2/tests/validation_scs_mp2_2terfc.rs         (energies)
Output:
  testdata/reference/validation/scs_mp2_2terfc/<system>_<basis>.json

WHY A FOURIER-SPACE CONSTRUCTION
--------------------------------
terfc exists in no other code. ferric evaluates it with its own
McMurchie-Davidson (MD) driver and the Dutoi/Goldey G_{m,0}(S,s)
interpolation tables (terf-tables/*.bin); every earlier check
(terfc_base_validation.rs, terfc_series_anchors.rs, eri4_terfc_split.rs)
either reuses that MD pass or the same G-function derivation, or is (ss|ss)
only. This script shares NONE of the MD recursion, the Hermite sign
conventions, the tables or the series branches. It uses PySCF's analytic
Gaussian Fourier transforms (`pyscf.gto.ft_ao`) and a k-space quadrature.

THE KERNEL AND ITS FOURIER TRANSFORM (re-derived; verified numerically below)
-----------------------------------------------------------------------------
ferric (shim.cc, terfc_base_validation.rs header):
    terf(r; r0)/r  = [erf(w(r - r0)) + erf(w(r + r0))] / (2r),  w = 1/(r0 sqrt2)
    terfc(r; r0)/r = 1/r - terf(r; r0)/r.
Write h(r) = erf(w(r - r0)) + erf(w(r + r0)). h is ODD in r (erf is odd), so
for a radial f(r) = h(r)/(2r) the 3-D transform
    F(k) = (4 pi / k) int_0^inf r f(r) sin(kr) dr = (2 pi / k) int_0^inf h(r) sin(kr) dr
reduces, because h(r) sin(kr) is EVEN, to a 1-D transform over the whole line:
    F(k) = (pi / k) int_{-inf}^{inf} h(r) sin(kr) dr.
Integrate by parts (h' is even, h(+-inf) = +-2 and the boundary term is the
same distributional limit as for the Coulomb kernel, which gives the 4pi/k^2
prefactor):  int h sin(kr) dr = (1/k) int h'(r) cos(kr) dr, with
    h'(r) = (2w/sqrt(pi)) [exp(-w^2 (r - r0)^2) + exp(-w^2 (r + r0)^2)].
Each Gaussian transforms to sqrt(pi)/w exp(-k^2/4w^2) e^{-+ i k r0}, whose sum
is 2 sqrt(pi)/w exp(-k^2/4w^2) cos(k r0). Hence
    int h'(r) cos(kr) dr = 4 exp(-k^2/4w^2) cos(k r0),
    F_terf(k)  = (4 pi / k^2) exp(-k^2 / 4w^2) cos(k r0),
    F_terfc(k) = (4 pi / k^2) [1 - exp(-k^2 / 4w^2) cos(k r0)].
Limits: r0 -> 0 at fixed w gives (4pi/k^2) exp(-k^2/4w^2), the erf(w r)/r
kernel exactly (asserted below against PySCF's erf integrals). r0 -> inf on
the curvature constraint (w r0 = 1/sqrt2) makes terf(r)/r -> the constant
C(r0) = (2/sqrt(pi)) exp(-1/2) w, so (P|terf|mu nu) -> C N_P S_mu nu (N_P =
int chi_P, S the overlap): asserted below at r0 = 20, 40, 80 Bohr.
`verify_fourier_transform` inverts F_terf radially by adaptive quadrature at
20 radii and compares it with the real-space kernel BEFORE anything else runs.

THE INTEGRALS
-------------
With chi^(k) = int chi(r) e^{-i k.r} d^3r (PySCF `ft_ao` / `ft_aopair`) and
real functions, Parseval gives
    (P|K|mu nu) = (2 pi)^-3 int d^3k F_K(k) Re[ chi_P^(k)* rho_mu nu^(k) ],
    (P|K|Q)     = (2 pi)^-3 int d^3k F_K(k) Re[ chi_P^(k)* chi_Q^(k) ].
The 4pi/k^2 of F_K cancels the k^2 of the spherical Jacobian. The grid is
Gauss-Legendre on k in [0, KMAX_FACTOR * w] (exp(-k^2/4w^2) < 1e-21 beyond it)
times a Lebedev rule (PySCF `MakeAngularGrid`, weights normalized to 1, so
multiplied by 4 pi here). The sign convention of the transform cancels
(Re[A* B] = Re[A B*]); the normalization does not, and is what the anchor tests.

ANCHOR FIRST (the script REFUSES to write unless every check passes):
  1. `verify_fourier_transform`: radial inverse of F_terf vs real space, <= 1e-12.
  2. erf anchor: on every rung of GRID_LADDER the same quadrature with
     F_erf(w) is compared with PySCF int3c2e/int2c2e under
     `with_range_coulomb(w)`; the production rung must reach <= 1e-10.
     The whole ladder is recorded, so the reference's own error is measured.
  3. terf self-convergence: the production-rung terf tensor vs the previous
     rung (recorded; asserted <= 1e-10).
  4. r0 -> 0 (fixed w) reproduces the erf anchor; r0 -> inf approaches
     C(r0) N_P S_mu nu with an error that falls as r0 grows.
terfc = PySCF analytic Coulomb (int3c2e/int2c2e) - terf(k-space).

WHAT ferric's SCS-MP2(2terfc) COMPUTES (read from the code, origin/main c28cfb7f)
--------------------------------------------------------------------------------
crates/ferric-mp2/src/scs.rs `scs_mp2_2terfc`: two calls of
`ri_mp2_spin_components(.., Operator::terfc(r0), .., RiMp2Config{frozen_core,
..Default})` (scs.rs:203-220), then
    E = c_OS E_OS(r0_1) + c_SS [E_SS(r0_2) - E_SS(r0_1)]       (scs.rs:224-226)
with defaults r0 = 0.75/1.05 A * 1.8897259886, c_OS 1.27, c_SS 4.05.
crates/ferric-mp2/src/rimp2.rs `ri_mp2_spin_components` (rimp2.rs:993+):
metric_op None -> the METRIC is the same terfc operator (rimp2.rs:1025,
`coulomb_metric_2c(met_op, ..)`), inverted by `metric_inverse_sqrt`, which for
Terfc is `cholesky_inverse_sqrt` (rimp2.rs:1984-1994): B = L^-1 (Q|ia), V = L L^T.
Occupied range frozen_core..nocc, all virtuals, canonical denominators.
This script does the same in numpy on the k-space terfc tensors with PySCF
exact-J/K RHF orbitals. The numpy assembly is anchored at the Coulomb kernel
against PySCF's own `dfmp2.DFMP2` before any terfc energy is written.

AO ORDER: samples are stored in FERRIC's AO order via
`gen_properties.ferric_ao_permutation` (orbital and aux). The Rust test checks
the permutation first by comparing ferric's libint Coulomb (P|mu nu) / (P|Q)
with PySCF's at the same stored indices.

Run (the k-space quadrature is the expensive step; slot it):
    scripts/validation/run_slot.sh -- \
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_terfc_integrals.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_properties as gp  # noqa: E402

ROW = "scs_mp2_2terfc"
ROW_NAME = "SCS-MP2(2terfc)"
CASES = (
    ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    ("nh3", "cc-pvdz", "cc-pvdz-ri"),
    ("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
)
# ferric's Angstrom->Bohr for r0 (crates/ferric-mp2/src/scs.rs ANGSTROM_TO_BOHR).
FERRIC_R0_ANG_TO_BOHR = 1.8897259886
R0_ANG_INTEGRALS = (0.75, 1.05)
R0_ANG_ENERGIES = (0.75, 1.00, 1.05)
C_OS, C_SS = 1.27, 4.05
R0_INF_BOHR = (20.0, 40.0, 80.0)
R0_ZERO_BOHR = 1e-5
KMAX_FACTOR = 14.0
# (n_radial, n_lebedev); the LAST rung is production.
GRID_LADDER = ((16, 110), (24, 194), (32, 302), (40, 590), (60, 1202), (80, 2030))
BATCH = 4000
N_RANDOM_3C = 300
N_RANDOM_2C = 200
SEED = 275
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9

TOL_FT = 1e-12
TOL_ANCHOR = 1e-10
TOL_SELF = 1e-10
TOL_DFMP2 = 1e-10


def r0_bohr(r0_ang: float) -> float:
    return r0_ang * FERRIC_R0_ANG_TO_BOHR


def w_of(r0: float) -> float:
    return 1.0 / (r0 * 2.0**0.5)


def terf_real(r, r0, w):
    from scipy.special import erf

    return (erf(w * (r - r0)) + erf(w * (r + r0))) / (2.0 * r)


def verify_fourier_transform(r0: float) -> float:
    """Radial inverse of F_terf vs the real-space kernel at 20 radii."""
    import numpy as np
    from scipy import integrate

    w = w_of(r0)
    worst = 0.0
    for r in np.geomspace(0.01, 30.0, 20):
        # f(r) = (1/(2 pi^2)) int k^2 F(k) sin(kr)/(kr) dk, F = 4pi/k^2 g(k)
        def integrand(k, r=r):
            return (
                np.exp(-k * k / (4 * w * w)) * np.cos(k * r0) * np.sin(k * r) / (k * r)
            )

        val, _ = integrate.quad(
            integrand, 0.0, KMAX_FACTOR * w * 3, limit=1000, epsabs=1e-15, epsrel=1e-14
        )
        worst = max(worst, abs(2.0 / np.pi * val - float(terf_real(r, r0, w))))
    if worst > TOL_FT:
        raise RuntimeError(f"F_terf radial inverse misses real space by {worst:.2e}")
    return worst


def kgrid(nrad: int, nang: int, kmax: float):
    import numpy as np
    from pyscf.dft import LebedevGrid

    x, wx = np.polynomial.legendre.leggauss(nrad)
    k = 0.5 * kmax * (x + 1.0)
    wk = 0.5 * kmax * wx
    ang = LebedevGrid.MakeAngularGrid(nang)
    assert abs(ang[:, 3].sum() - 1.0) < 1e-12, "Lebedev weights must sum to 1"
    gv = (k[:, None, None] * ang[None, :, :3]).reshape(-1, 3)
    wt = (wk[:, None] * 4.0 * np.pi * ang[None, :, 3]).reshape(-1)
    kk = np.repeat(k, nang)
    return gv, wt, kk


def kspace_tensors(mol, auxmol, gv, wt, kk, kernels):
    """(P|K|mu nu) as (naux, nao, nao) and (P|K|Q) for each g(k) in `kernels`,
    F_K = 4pi/k^2 g(k)."""
    import numpy as np
    from pyscf.gto import ft_ao

    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    nk = len(kernels)
    i3 = np.zeros((nk, naux, nao * nao))
    i2 = np.zeros((nk, naux, naux))
    pref = 4.0 * np.pi / (2.0 * np.pi) ** 3
    for s in range(0, len(gv), BATCH):
        g = gv[s : s + BATCH]
        ww = wt[s : s + BATCH] * pref
        kb = kk[s : s + BATCH]
        a = ft_ao.ft_ao(auxmol, g)  # (ng, naux)
        b = ft_ao.ft_aopair(mol, g, aosym="s1").reshape(len(g), -1)
        for ik, kern in enumerate(kernels):
            aw = a.conj() * (ww * kern(kb))[:, None]
            i3[ik] += (aw.T @ b).real
            i2[ik] += (aw.T @ a).real
    return i3.reshape(nk, naux, nao, nao), i2


def g_erf(w):
    import numpy as np

    return lambda k: np.exp(-k * k / (4 * w * w))


def g_terf(r0, w):
    import numpy as np

    return lambda k: np.exp(-k * k / (4 * w * w)) * np.cos(k * r0)


def pyscf_3c2c(mol, auxmol, omega=None):
    """PySCF (naux, nao, nao) and (naux, naux); omega>0 erf, <0 erfc, None Coulomb."""
    from pyscf import df

    def build():
        v3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        return v3.transpose(2, 0, 1).copy(), auxmol.intor("int2c2e")

    if omega is None:
        return build()
    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        return build()


def maxabs(a) -> float:
    import numpy as np

    return float(np.max(np.abs(a)))


def ferric_l_list(basis_name: str, symbols) -> list[int]:
    out = []
    for s in symbols:
        for sh in common.ferric_shells(basis_name, common.z_of(s)):
            ell = sh["l"]
            nf = (
                (2 * ell + 1) if (sh["pure"] or ell < 2) else (ell + 1) * (ell + 2) // 2
            )
            out.extend([ell] * nf)
    return out


def ri_mp2(mo_coeff, mo_energy, nocc, frozen, v3, v2):
    """ferric's ri_mp2_spin_components in numpy: v3 (naux,nao,nao), v2 (naux,naux)."""
    import numpy as np
    import scipy.linalg

    co = mo_coeff[:, frozen:nocc]
    cv = mo_coeff[:, nocc:]
    iap = np.einsum("Pmn,mi,na->Pia", v3, co, cv, optimize=True)
    naux = v2.shape[0]
    low = scipy.linalg.cholesky(v2, lower=True)
    b = scipy.linalg.solve_triangular(low, iap.reshape(naux, -1), lower=True)
    no, nv = co.shape[1], cv.shape[1]
    b = b.reshape(naux, no, nv)
    eo, ev = mo_energy[frozen:nocc], mo_energy[nocc:]
    e_os = 0.0
    e_ss = 0.0
    for i in range(no):
        for j in range(no):
            v = np.einsum("Pa,Pb->ab", b[:, i, :], b[:, j, :])
            d = eo[i] + eo[j] - ev[:, None] - ev[None, :]
            e_os += float(np.sum(v * v / d))
            e_ss += float(np.sum(v * (v - v.T) / d))
    return {"e_os": e_os, "e_ss": e_ss, "e_total": e_os + e_ss}


def class_label(ls) -> str:
    return "".join("spdfgh"[x] for x in ls)


def sample_3c(t_ferric, laux, lobs, rng):
    """Per-class max-|value| element (class = l_P, l_mu >= l_nu over mu >= nu)
    plus a seeded random subset, as (P, mu, nu) in ferric order."""
    import numpy as np

    naux, nao, _ = t_ferric.shape
    best: dict[str, tuple[float, tuple[int, int, int]]] = {}
    for p in range(naux):
        for m in range(nao):
            for n in range(m + 1):
                lm, ln = sorted((lobs[m], lobs[n]), reverse=True)
                key = class_label((laux[p], lm, ln))
                v = abs(t_ferric[p, m, n])
                if key not in best or v > best[key][0]:
                    best[key] = (v, (p, m, n))
    idx = [best[k][1] for k in sorted(best)]
    picks = set(idx)
    while len(picks) < len(idx) + N_RANDOM_3C:
        p = int(rng.integers(naux))
        m = int(rng.integers(nao))
        n = int(rng.integers(m + 1))
        picks.add((p, m, n))
    rand = sorted(picks - set(idx))
    return idx + rand, sorted(best), np.asarray([best[k][0] for k in sorted(best)])


def sample_2c(t_ferric, laux, rng):
    naux = t_ferric.shape[0]
    best: dict[str, tuple[float, tuple[int, int]]] = {}
    for p in range(naux):
        for q in range(p + 1):
            lp, lq = sorted((laux[p], laux[q]), reverse=True)
            key = class_label((lp, lq))
            v = abs(t_ferric[p, q])
            if key not in best or v > best[key][0]:
                best[key] = (v, (p, q))
    idx = [best[k][1] for k in sorted(best)]
    picks = set(idx)
    while len(picks) < len(idx) + N_RANDOM_2C:
        p = int(rng.integers(naux))
        q = int(rng.integers(p + 1))
        picks.add((p, q))
    return idx + sorted(picks - set(idx)), sorted(best)


def run_case(system: str, basis_name: str, auxbasis: str) -> Path:
    import numpy as np
    import pyscf
    from pyscf import df, gto, scf
    from pyscf.mp import dfmp2

    t_start = time.time()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    auxb, auxcart = common.pyscf_basis(auxbasis, symbols)
    assert not auxcart, "aux basis must be spherical"
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=auxb,
        cart=False,
        verbose=0,
    )
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    assert naux == common.ferric_nao(auxbasis, symbols), (naux, auxbasis)
    perm = np.asarray(gp.ferric_ao_permutation(mol, basis_name, symbols))
    aperm = np.asarray(gp.ferric_ao_permutation(auxmol, auxbasis, symbols))
    lobs = ferric_l_list(basis_name, symbols)
    laux = ferric_l_list(auxbasis, symbols)
    assert len(lobs) == nao and len(laux) == naux

    def to_f3(t):
        return t[np.ix_(aperm, perm, perm)]

    def to_f2(t):
        return t[np.ix_(aperm, aperm)]

    # ---- Coulomb (analytic, PySCF) ----------------------------------------
    c3, c2 = pyscf_3c2c(mol, auxmol)

    # ---- k-space: ladder + anchors per production r0 ----------------------
    r0_all = sorted(set(R0_ANG_INTEGRALS) | set(R0_ANG_ENERGIES))
    terf3: dict[float, np.ndarray] = {}
    terf2: dict[float, np.ndarray] = {}
    grid_records = []
    ft_checks = {}
    for r0a in r0_all:
        r0 = r0_bohr(r0a)
        w = w_of(r0)
        ft_checks[f"{r0a:.2f}"] = verify_fourier_transform(r0)
        e3, e2 = pyscf_3c2c(mol, auxmol, omega=w)
        kernels = [g_erf(w), g_terf(r0, w), g_terf(R0_ZERO_BOHR, w)]
        ladder = []
        prev = None
        for nrad, nang in GRID_LADDER:
            gv, wt, kk = kgrid(nrad, nang, KMAX_FACTOR * w)
            i3, i2 = kspace_tensors(mol, auxmol, gv, wt, kk, kernels)
            rung = {
                "n_radial": nrad,
                "n_lebedev": nang,
                "n_points": len(gv),
                "erf_anchor_3c_max_abs": maxabs(i3[0] - e3),
                "erf_anchor_2c_max_abs": maxabs(i2[0] - e2),
            }
            if prev is not None:
                rung["terf_change_vs_previous_rung_3c"] = maxabs(i3[1] - prev[0])
                rung["terf_change_vs_previous_rung_2c"] = maxabs(i2[1] - prev[1])
            ladder.append(rung)
            prev = (i3[1], i2[1])
            print(f"{system}/{basis_name} r0={r0a} {rung}", flush=True)
        prod = ladder[-1]
        anchor = max(prod["erf_anchor_3c_max_abs"], prod["erf_anchor_2c_max_abs"])
        if anchor > TOL_ANCHOR:
            raise RuntimeError(
                f"erf anchor {anchor:.2e} > {TOL_ANCHOR:.0e} at r0={r0a}"
            )
        selfc = max(
            prod["terf_change_vs_previous_rung_3c"],
            prod["terf_change_vs_previous_rung_2c"],
        )
        if selfc > TOL_SELF:
            raise RuntimeError(f"terf not converged ({selfc:.2e}) at r0={r0a}")
        r0zero = max(maxabs(i3[2] - e3), maxabs(i2[2] - e2))
        if r0zero > 1e-8:
            raise RuntimeError(f"r0->0 limit misses erf by {r0zero:.2e}")
        terf3[r0a], terf2[r0a] = i3[1], i2[1]
        grid_records.append(
            {
                "r0_angstrom": r0a,
                "r0_bohr": r0,
                "w_bohr_inv": w,
                "kmax": KMAX_FACTOR * w,
                "ladder": ladder,
                "r0_to_zero_at_fixed_w_vs_pyscf_erf_max_abs": r0zero,
            }
        )

    # ---- r0 -> inf (curvature constrained): terf -> C N_P S_mu nu ---------
    s_ao = mol.intor("int1e_ovlp")
    from pyscf.gto import ft_ao

    n_p = ft_ao.ft_ao(auxmol, np.zeros((1, 3)))[0].real
    inf_records = []
    nrad, nang = GRID_LADDER[-1]
    for r0 in R0_INF_BOHR:
        w = w_of(r0)
        gv, wt, kk = kgrid(nrad, nang, KMAX_FACTOR * w)
        i3, i2 = kspace_tensors(mol, auxmol, gv, wt, kk, [g_terf(r0, w)])
        c_r0 = 2.0 / np.sqrt(np.pi) * np.exp(-0.5) * w
        lim3 = c_r0 * n_p[:, None, None] * s_ao[None, :, :]
        lim2 = c_r0 * np.outer(n_p, n_p)
        rel = max(maxabs(i3[0] - lim3), maxabs(i2[0] - lim2)) / c_r0
        inf_records.append({"r0_bohr": r0, "max_abs_dev_over_C": rel})
        print(f"{system}/{basis_name} r0->inf r0={r0}: |terf - C N S|/C = {rel:.3e}")
    devs = [x["max_abs_dev_over_C"] for x in inf_records]
    if not all(b < a for a, b in zip(devs, devs[1:])):
        raise RuntimeError(f"r0->inf limit is not approached monotonically: {devs}")

    # ---- metric conditioning ---------------------------------------------
    metric_eigs = {}
    for r0a in r0_all:
        ev_c = np.linalg.eigvalsh(c2 - terf2[r0a])
        ev_t = np.linalg.eigvalsh(terf2[r0a])
        metric_eigs[f"{r0a:.2f}"] = {
            "terfc_min": float(ev_c[0]),
            "terfc_max": float(ev_c[-1]),
            "terf_min": float(ev_t[0]),
            "terf_max": float(ev_t[-1]),
        }
        if ev_c[0] <= 0.0:
            raise RuntimeError(
                f"terfc metric not positive definite at r0={r0a}: {ev_c[0]:.3e}"
            )

    # ---- integral samples in ferric order ----------------------------------
    rng = np.random.default_rng(SEED)
    r0_ref = R0_ANG_INTEGRALS[0]
    tfc_ref = to_f3(c3 - terf3[r0_ref])
    idx3, classes3, _ = sample_3c(tfc_ref, laux, lobs, rng)
    idx2, classes2 = sample_2c(to_f2(c2 - terf2[r0_ref]), laux, rng)
    a3 = np.asarray(idx3)
    a2 = np.asarray(idx2)

    def pick3(t):
        tf = to_f3(t)
        return [float(x) for x in tf[a3[:, 0], a3[:, 1], a3[:, 2]]]

    def pick2(t):
        tf = to_f2(t)
        return [float(x) for x in tf[a2[:, 0], a2[:, 1]]]

    def summary(t):
        return {"sum_sq": float(np.sum(t * t)), "max_abs": maxabs(t)}

    integrals = {
        "aux_l_ferric_order": laux,
        "obs_l_ferric_order": lobs,
        "eri3_indices_p_mu_nu": [list(map(int, x)) for x in idx3],
        "eri3_n_class_max": len(classes3),
        "eri3_classes": classes3,
        "eri2_indices_p_q": [list(map(int, x)) for x in idx2],
        "eri2_n_class_max": len(classes2),
        "eri2_classes": classes2,
        "coulomb": {"eri3": pick3(c3), "eri2": pick2(c2)},
        "per_r0": [],
    }
    for r0a in R0_ANG_INTEGRALS:
        r0 = r0_bohr(r0a)
        w = w_of(r0)
        ec3, ec2 = pyscf_3c2c(mol, auxmol, omega=-w)
        integrals["per_r0"].append(
            {
                "r0_angstrom": r0a,
                "r0_bohr": r0,
                "terf": {
                    "eri3": pick3(terf3[r0a]),
                    "eri2": pick2(terf2[r0a]),
                    "eri3_full": summary(terf3[r0a]),
                    "eri2_full": summary(terf2[r0a]),
                },
                "terfc": {
                    "eri3": pick3(c3 - terf3[r0a]),
                    "eri2": pick2(c2 - terf2[r0a]),
                    "eri3_full": summary(c3 - terf3[r0a]),
                    "eri2_full": summary(c2 - terf2[r0a]),
                },
                "erfc_same_w_info": {
                    "eri3": pick3(ec3),
                    "eri2": pick2(ec2),
                },
            }
        )

    # ---- RHF + energies ----------------------------------------------------
    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.kernel()
    assert mf.converged
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    n_heavy = sum(1 for s in symbols if common.z_of(s) > 2)

    coul = ri_mp2(mf.mo_coeff, mf.mo_energy, nocc, 0, c3, c2)
    pt = dfmp2.DFMP2(mf)
    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = auxb
    pt.kernel()
    d_assembly = abs(coul["e_total"] - float(pt.e_corr))
    if d_assembly > TOL_DFMP2:
        raise RuntimeError(f"numpy DF-MP2 vs PySCF DFMP2: {d_assembly:.2e}")

    energies = []
    for fc in (0, n_heavy):
        per_r0 = []
        for r0a in R0_ANG_ENERGIES:
            e = ri_mp2(
                mf.mo_coeff, mf.mo_energy, nocc, fc, c3 - terf3[r0a], c2 - terf2[r0a]
            )
            per_r0.append({"r0_angstrom": r0a, "r0_bohr": r0_bohr(r0a), **e})
            print(f"{system}/{basis_name} fc={fc} r0={r0a}: {e}")
        by = {x["r0_angstrom"]: x for x in per_r0}
        e1, e2 = by[0.75], by[1.05]
        scs = C_OS * e1["e_os"] + C_SS * (e2["e_ss"] - e1["e_ss"])
        energies.append(
            {
                "frozen_core": fc,
                "per_r0": per_r0,
                "scs_2terfc": {
                    "r0_bonded_angstrom": 0.75,
                    "r0_nonbonded_angstrom": 1.05,
                    "c_os": C_OS,
                    "c_ss": C_SS,
                    "scs_corr": scs,
                    "total_energy": float(mf.e_tot) + scs,
                },
            }
        )

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "auxbasis": auxbasis,
        "charge": 0,
        "multiplicity": 1,
        "nao": nao,
        "naux": naux,
        "nuclear_repulsion": float(mol.energy_nuc()),
        "rhf_energy": float(mf.e_tot),
        "n_heavy": n_heavy,
        "integrals": integrals,
        "energies": energies,
        "checks": {
            "fourier_transform_radial_inverse_max_abs": ft_checks,
            "kspace_grid": grid_records,
            "r0_to_infinity": inf_records,
            "metric_eigenvalues": metric_eigs,
            "numpy_vs_pyscf_dfmp2_coulomb_abs": d_assembly,
            "coulomb_ri_mp2": coul,
        },
        "provenance": common.provenance(
            code="PySCF (ft_ao/ft_aopair, int3c2e/int2c2e, RHF, DFMP2 anchor) + numpy (k-space quadrature, RI-MP2)",
            version=pyscf.__version__,
            keywords={
                "terf": "k-space: (2pi)^-3 int d3k 4pi/k^2 exp(-k^2/4w^2) cos(k r0) Re[chiP* rho]",
                "terfc": "PySCF analytic Coulomb minus k-space terf",
                "grid": f"Gauss-Legendre k in [0, {KMAX_FACTOR} w] x Lebedev; production {GRID_LADDER[-1]}",
                "fit": "standard, metric = same terfc kernel, Cholesky (ferric metric_op None)",
                "rhf": "scf.RHF exact J/K",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=f"k-space {GRID_LADDER[-1]}",
            aux=auxbasis,
            frozen_core=[0, n_heavy],
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator="scripts/validation/gen_terfc_integrals.py",
            extra={
                "basis_self_check": basis_check,
                "aux_basis_json": str(
                    common.basis_json_path(auxbasis).relative_to(common.ROOT)
                ),
                "aux_basis_sha256": common.sha256_file(
                    common.basis_json_path(auxbasis)
                ),
                "r0_angstrom_to_bohr": FERRIC_R0_ANG_TO_BOHR,
                "seed": SEED,
                "wall_seconds": time.time() - t_start,
            },
        ),
    }
    return common.write_reference(ROW, system, basis_name, payload)


def main() -> int:
    only = set(sys.argv[1:])
    written = []
    for system, basis_name, auxbasis in CASES:
        if only and f"{system}_{basis_name}" not in only:
            continue
        written.append(run_case(system, basis_name, auxbasis))
        print(f"wrote {written[-1]}", flush=True)
    print(f"GEN_TERFC_INTEGRALS_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
