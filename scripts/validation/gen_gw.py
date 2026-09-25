"""PySCF references for the VALIDATION.md GW rows (validation tier W3).

Consumer: crates/ferric-gw/tests/validation_gw.rs.
Output:   testdata/reference/validation/gw/<system>_<basis>.json

Rows covered: "G0W0" (closed shell, @HF and @PBE), "G0W0@HF+ECP",
"U-G0W0", "COHSEX/evGW0/evGW".

WHAT FERRIC COMPUTES (read from crates/ferric-gw/src/{sigma,cohsex,u_sigma}.rs,
not from a doc) AND HOW THIS SCRIPT MATCHES IT
------------------------------------------------------------------------------
ferric builds W from a PDEP eigenbasis of the static dielectric. At
`trunc_thresh = 0` with the default Lanczos eigensolver that basis is the FULL
naux-dimensional RI space (one dense eigh of I + Pi(0)), and Sigma_c contracts
the FULL per-frequency inverse-dielectric matrix in it, so W is the ordinary
RI screened interaction and is basis-rotation invariant. PySCF `gw_ac` builds
the same object in the Cholesky-RI basis. What is left to match:

  * W frequency integral: ferric Gauss-Legendre on [0,inf) via
    w = u0 (1+x)/(1-x), `n_points`, u0 = 0.5; PySCF `_get_scaled_legendre_roots
    (nw, x0=0.5)` -- the same map. Both run at NW = 100 (PySCF's default).
  * Pade support: ferric samples Sigma_c(ef + i w) on [0] + GL(100, u0=0.5)
    (a fixed 101-point grid, independent of n_points), selects 18 nodes with
    PySCF's `_get_ac_idx` (step_ratio 2/3, idx_start 1) and fits Thiele.
    PySCF samples on [0] + its W grid, CUT at `ac_iw_cutoff` (default 5.0 Ha),
    then selects with `_get_ac_idx`. With NW = 100 the grids coincide, and
    `ac_iw_cutoff = None` removes the cut, so the 18 nodes are identical.
    The Thiele COEFFICIENTS are then identical too (same recursion), but the
    EVALUATION is not.
  * PYSCF DEFECT (PySCF 2.13.1, pyscf/gw/utils/ac_grid.py,
    `pade_thiele_ndarray`, called by `PadeAC.ac_eval`): it seeds
    `X = coeff[-1] * (freqs - zn[-2])` and then loops idx = ncoeff-1 ... 1 with
    `X = coeff[idx] * (freqs - zn[idx-1]) / (1 + X)`. The first pass (idx =
    ncoeff-1) applies coeff[-1] a second time, so the innermost level is
    1 + a_{N-1}(z - z_{N-2}) / (1 + a_{N-1}(z - z_{N-2})) instead of
    1 + a_{N-1}(z - z_{N-2}). The resulting fraction does not interpolate the
    data it was fitted to (measured: up to 1.8e-3 Ha off at the 18 nodes,
    H2O/cc-pVDZ). ferric's pade.rs `PadeCF::eval` is the standard Thiele
    fraction a_0 / (1 + a_1(z - z_0) / (1 + ... a_{N-1}(z - z_{N-2}))), which
    reproduces its nodes to 1e-16. Same Sigma_c(ef + i w), same nodes, same
    coefficients: the two evaluations differ in Sigma_c(eps_qp) by <=1.2e-7 Ha
    at @HF (H2O/cc-pVDZ) but by up to 7.14e-3 Ha at @PBE (H2O/cc-pVDZ HOMO-2;
    N2/cc-pVDZ LUMO+2 1.4e-5), where the KS roots sit further from ef.
    THEREFORE every @PBE reference (`g0w0_pbe`, and the ferric-matched @PBE
    sensitivity variants) uses the standard Thiele fraction, re-implemented in
    numpy (`thiele_coeffs` / `thiele_eval` / `textbook_continuation`) on
    PySCF's own Sigma_c(ef + i w); PySCF's numbers are kept as `*_pyscf_pade`.
    The @HF / ECP / open-shell / ev blocks keep PySCF's evaluation (below
    every bar there). Sigma_c(ef + i w) at the 18 nodes is stored under `ac`
    (every @PBE block, and @HF H2O/N2 cc-pVDZ) for an imaginary-axis comparison
    with NO continuation in it -- the like-for-like check of Sigma_c itself.
  * Fermi level: mid-gap of the mean-field spectrum on both sides (closed
    shell). OPEN SHELL DIFFERS: ferric uses a PER-SPIN mid-gap
    (u_sigma.rs: fermi_level(eps_act_sigma, n_occ_act_sigma)) while PySCF
    `ugw_ac` uses ONE ef = (max HOMO + min LUMO)/2. Sigma_c(z) is analytic,
    but the Pade continuation of it depends on the support line, so the two
    conventions give different numbers. The matched reference runs `ugw_ac`
    twice with ef overridden to ef_alpha and ef_beta and keeps the matching
    spin; the PySCF-native (common-ef) numbers are recorded as a diagnostic.
  * QP equation: full (not linearized) root of
        w - eps_mf - (Sigma_x - v_mf) - Re Sigma_c^Pade(w) = 0
    on both sides. PySCF's scipy secant (tol 1e-6) is re-solved here with a
    tight Newton (tol 1e-12) on the Pade (PySCF's own object, or the
    textbook fraction for @PBE); both are recorded.
  * Sigma_x: ferric's Sigma_x is density-fitted with the GW aux
    (cohsex.rs sigma_x_diag: -sum_{P,i} B^P_{mi}^2 over ACTIVE occupieds).
    For an HF reference ferric applies no static shift, and PySCF's
    vk - v_mf is identically 0 for an exact-integral HF (recorded). For a KS
    reference the static shift Sigma_x - v_xc matters, so @PBE runs PySCF
    with `vhf_df = True` (vk from the same Lpq), matching ferric.
  * SCF: exact 4-index integrals on both sides (ferric: HF default; PBE with
    `df_j_aux: Some("")`). PBE grid (75,110) unpruned, Becke partition with
    Becke-1988 radii adjustment -- ferric's default, the KS-DFT row's recipe.
  * Basis and aux: ferric's bundled JSON for BOTH (common.py), AO and aux
    counts checked against ferric's parser.
  * XC density floor: ferric zeroes e_xc/v_xc where rho <= 1e-10; the @PBE
    reference applies the same floor (FERRIC_DENSITY_FLOOR). Without it the
    h2o/aug-cc-pvdz diffuse virtuals differ by up to 5e-7 Ha; that shift is
    recorded as `density_floor_mo_energy_shift_max`.
  * Frozen core: none, except the explicit `g0w0_hf_fc1` block (PySCF
    frozen=1; ferric GwConfig.frozen_core = PdepRpaConfig.frozen_core = 1).

evGW0 / evGW: PySCF 2.13 has neither. They are built here by iterating PySCF's
own `gw_ac.get_sigma` (which takes separate G and W orbital energies,
`mo_energy` / `mo_energy_w`) exactly as ferric iterates: Jacobi updates of the
QP window only (orbitals outside the window keep their mean-field energy), ef
and eps_mf held at the mean-field values, Pade + full QP root each sweep,
W frozen at the mean-field spectrum (evGW0) or rebuilt from the updated one
(evGW), converged to 1e-7 Ha (the change floors at ~1e-8 Ha: Pade refit noise).

COHSEX: static, closed form, numpy on PySCF's Lpq:
    dSigma_SEX(m) = - sum_i  [L^T ((I - Pi(0))^-1 - I) L]_{mi,mi}
    Sigma_COH(m)  = + 1/2 sum_p [ ... ]_{mp,mp}
    eps_qp = eps_mf + dSigma_SEX + Sigma_COH      (HF reference)
which is cohsex.rs's formula with w_alpha = 1/lambda_alpha - 1 in the full-rank
static eigenbasis.

DIAGNOSTICS written alongside (not asserted by the Rust test; they explain the
old loose bars):
  * h2o/cc-pvdz `diagnostics.pbe_recipe_sensitivity`: G0W0@PBE HOMO under
    PySCF's native settings (iw cutoff 5, exact Sigma_x, nw 16/20), and
    `diagnostics.hf_quadrature_sensitivity`: G0W0@HF HOMO with a 16/20/40-point
    W integral (the Pade grid kept at 101 points, as ferric does).
  * open-shell cc-pvdz `diagnostics.old_recipe`: the recipe of
    scripts/gw100/pyscf_u_g0w0.py (DF-UHF with the cc-pvdz-ri aux, common ef).

Run (light; about a minute per system):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_gw.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "gw"
NW = 100
X0 = 0.5
PADE_NPTS = 18
PADE_STEP_RATIO = 2.0 / 3.0
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-8
QP_TOL = 1e-12
EV_TOL = 1e-7  # the sweep-to-sweep change floors at ~1e-8 Ha (Pade refit noise)
EV_MAXIT = 80
MAIN_GRID = (75, 110)

AUX_FOR = {
    "cc-pvdz": "cc-pvdz-ri",
    "aug-cc-pvdz": "aug-cc-pvdz-rifit",
    "aug-cc-pvdz-pp": "def2-tzvp-rifit",
}

# system -> (xyz relative to MOL_DIR, bases); all neutral singlets
CLOSED = {
    "h2o": ("h2o.xyz", ("cc-pvdz", "aug-cc-pvdz")),
    "nh3": ("nh3.xyz", ("cc-pvdz", "aug-cc-pvdz")),
    "n2": ("n2.xyz", ("cc-pvdz", "aug-cc-pvdz")),
}
# Extra closed-shell blocks, by (system, basis).
PBE_CASES = {("h2o", "cc-pvdz"), ("h2o", "aug-cc-pvdz"), ("n2", "cc-pvdz")}
COHSEX_CASES = {("h2o", "cc-pvdz"), ("n2", "cc-pvdz")}
EV_CASES = {("h2o", "cc-pvdz")}
FC_CASES = {("h2o", "cc-pvdz")}
# @HF blocks that also store Sigma_c(ef + i w) at the 18 Pade nodes (the
# imaginary-axis check in validation_gw.rs; every @PBE block stores them).
HF_AC_NODE_CASES = {("h2o", "cc-pvdz"), ("n2", "cc-pvdz")}

ECP = {
    "i2": "ecp/i2.xyz",
    "xe": "ecp/xe.xyz",
    "ag2": "ecp/ag2.xyz",
}
ECP_BASIS = "aug-cc-pvdz-pp"

OPEN = {  # system -> (xyz, multiplicity, bases)
    "oh": ("oh.xyz", 2, ("cc-pvdz", "aug-cc-pvdz")),
    "ch3": ("ch3.xyz", 2, ("cc-pvdz", "aug-cc-pvdz")),
    "nh2": ("nh2.xyz", 2, ("cc-pvdz", "aug-cc-pvdz")),
    "o2": ("o2.xyz", 3, ("aug-cc-pvdz",)),
    "ch2_triplet": ("ch2_triplet.xyz", 3, ("aug-cc-pvdz",)),
}


def _np():
    import numpy as np

    return np


def aux_dict(aux_name, symbols):
    aux, cart = common.pyscf_basis(aux_name, symbols)
    if cart:
        raise ValueError(f"{aux_name}: Cartesian l>=2 aux shells cannot be matched")
    return aux


def check_naux(with_df, aux_name, symbols):
    naux = int(with_df.get_naoaux())
    want = common.ferric_nao(aux_name, symbols)
    assert naux == want, f"{aux_name}: PySCF naux {naux} != ferric {want}"
    return naux


def window(nocc_max, nmo):
    """ferric's default QP range: HOMO-2 .. LUMO+2 (lib.rs default_qp_range)."""
    return list(range(max(0, nocc_max - 3), min(nocc_max + 3, nmo)))


def rhf_exact(mol):
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    stable = common._stability_status(mf)[1] if hasattr(mf, "stability") else None
    return mf, bool(stable)


# ferric zeroes e_xc and v_xc at grid points with rho <= 1e-10
# (ferric-dft vxc.rs DENSITY_FLOOR); libxc's own PBE threshold is lower. The
# floor only reaches the far tail, so it moves occupied eigenvalues ~1e-10 Ha
# but diffuse virtuals up to 5e-7 Ha (h2o/aug-cc-pvdz MO 6). The reference
# applies the same floor; the unfloored shift is recorded per block.
FERRIC_DENSITY_FLOOR = 1e-10


def _apply_density_floor(mf, floor):
    ni = mf._numint
    orig = ni.eval_xc_eff

    def floored(
        xc_code, rho, deriv=1, omega=None, xctype=None, verbose=None, spin=None
    ):
        exc, vxc, fxc, kxc = orig(xc_code, rho, deriv, omega, xctype, verbose, spin)
        mask = (rho[0] if rho.ndim > 1 else rho) <= floor
        exc = exc.copy()
        exc[mask] = 0.0
        vxc = vxc.copy()
        vxc[..., mask] = 0.0
        return exc, vxc, fxc, kxc

    ni.eval_xc_eff = floored


def rks_pbe_exact(mol, density_floor=FERRIC_DENSITY_FLOOR):
    from pyscf import dft

    mf = dft.RKS(mol, xc="PBE,PBE")
    if density_floor is not None:
        _apply_density_floor(mf, density_floor)
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, "RKS/PBE did not converge"
    return mf


def _refine(f, x0):
    from scipy.optimize import newton

    return float(newton(f, x0, tol=QP_TOL, maxiter=200))


def thiele_coeffs(fn, zn):
    """Thiele reciprocal differences, written out here (independent of PySCF's
    `thiele_ndarray`): g_1 = f; g_p(z_k) = (g_{p-1}(z_{p-1}) - g_{p-1}(z_k)) /
    ((z_k - z_{p-1}) g_{p-1}(z_k)); a_p = g_p(z_p). fn: (npts, norb)."""
    np = _np()
    g = np.array(fn, dtype=complex)
    zn = np.asarray(zn, dtype=complex)
    for p in range(1, len(zn)):
        g[p:] = (g[p - 1] - g[p:]) / ((zn[p:, None] - zn[p - 1]) * g[p:])
    return g


def thiele_eval(w, zn, a):
    """Textbook Thiele continued fraction (ferric pade.rs PadeCF::eval):
        C(z) = a_0 / (1 + a_1 (z - z_0) / (1 + ... a_{N-1} (z - z_{N-2}) / 1))
    evaluated bottom-up. It interpolates its data EXACTLY at every node.

    PySCF 2.13 `ac_grid.pade_thiele_ndarray` evaluates a DIFFERENT fraction:
    it seeds X = a_{N-1}(z - z_{N-2}) and then runs its loop from idx = N-1
    again, so the innermost level is 1 + a_{N-1}(z-z_{N-2}) / (1 + a_{N-1}
    (z-z_{N-2})) -- a_{N-1} enters twice. That fraction does NOT pass through
    its own data (measured: up to 1.8e-3 Ha off at the 18 nodes, H2O/cc-pVDZ),
    and far from ef on the real axis it moves Sigma_c by up to 7.1e-3 Ha
    (H2O @PBE, HOMO-2). Same coefficients, same nodes, different evaluation."""
    acc = 1.0 + 0.0j
    for p in range(len(a) - 1, 0, -1):
        acc = 1.0 + a[p] * (w - zn[p - 1]) / acc
    return a[0] / acc


def _qp_roots_in_window(f, e0, half=1.0, n=20001):
    """Sign changes of the QP function on e0 +/- half that are not poles
    (|f| small on both sides): the number of QP solutions in that window."""
    np = _np()
    ws = np.linspace(e0 - half, e0 + half, n)
    fv = np.array([f(w) for w in ws])
    ix = np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]
    return [float(ws[i]) for i in ix if abs(fv[i]) < 0.5 and abs(fv[i + 1]) < 0.5]


def textbook_continuation(gw, mf, orbs_frz, orbs):
    """ferric's continuation, re-implemented in numpy on PySCF's Sigma_c(iw).

    Sigma_c(ef + i w) comes from PySCF's own `gw_ac.get_sigma` on the grid
    GWAC.kernel used ([0] + gw.freqs, same cut); the 18 nodes from
    `_get_ac_idx` (== ferric sigma::pade_node_indices, pinned by a ferric unit
    test); coefficients from `thiele_coeffs` above (cross-checked against
    PySCF's fitted coefficients, so the data and the recursion are the same);
    evaluation with the textbook fraction `thiele_eval`; full QP root.
    Returns (per-orbital eps_qp, sigma_c_at_qp, diagnostics dict)."""
    np = _np()
    from pyscf.gw.gw_ac import _mo_energy_without_core, get_sigma
    from pyscf.gw.utils.ac_grid import _get_ac_idx, _get_scaled_legendre_roots

    qf, qw = _get_scaled_legendre_roots(gw.nw, X0)
    eval_f = np.concatenate(([0.0], gw.freqs))
    e_frz = np.asarray(
        _mo_energy_without_core(gw, np.asarray(mf.mo_energy)), dtype=float
    )
    sig, omega = get_sigma(
        gw,
        orbs_frz,
        gw.Lpq,
        qf,
        qw,
        gw.ef,
        e_frz,
        iw_cutoff=gw.ac_iw_cutoff,
        eval_freqs=eval_f,
    )
    assert np.allclose(omega, gw.acobj.omega, rtol=0, atol=1e-14), (
        "AC grid differs from GWAC's"
    )
    idx = _get_ac_idx(
        len(omega), npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio
    )
    zn = omega[idx]
    fn = sig[:, idx].T  # (npts, norb)
    a = thiele_coeffs(fn, zn)
    coeff_rel = float(
        np.max(np.abs(a - gw.acobj.coeff) / np.maximum(np.abs(a), 1e-300))
    )
    assert coeff_rel < 1e-8, (
        f"Thiele coefficients differ from PySCF's: rel {coeff_rel:.2e}"
    )
    interp_tb = float(
        max(
            abs(thiele_eval(z, zn, a[:, i]) - fn[k, i])
            for i in range(len(orbs))
            for k, z in enumerate(zn)
        )
    )
    interp_py = float(
        max(
            abs(gw.acobj[i].ac_eval(z) - fn[k, i])
            for i in range(len(orbs))
            for k, z in enumerate(zn)
        )
    )
    eqp, sc, resid, nroots = [], [], [], []
    for ip, (pf, pa) in enumerate(zip(orbs_frz, orbs)):
        shift = float(gw.vk[pf, pf] - gw.vxc[pf, pf])
        e0 = float(mf.mo_energy[pa])
        ai = a[:, ip]

        def f(w, ai=ai, e0=e0, shift=shift):
            return w - e0 - (thiele_eval(w + 0.0j, zn, ai).real + shift)

        root = _refine(f, float(gw.mo_energy[pa]))
        eqp.append(root)
        sc.append(float(thiele_eval(root + 0.0j, zn, ai).real))
        resid.append(abs(f(root)))
        nroots.append(len(_qp_roots_in_window(f, e0)))
    diag = {
        "continuation": "textbook Thiele continued fraction (ferric pade.rs), "
        "numpy re-implementation on PySCF's Sigma_c(ef+iw); see gen_gw.thiele_eval",
        "ac_nodes_omega": [float(x) for x in zn.imag],
        "ac_nodes_index": [int(i) for i in idx],
        "sigma_c_iw_nodes_re": [
            [float(x) for x in fn[:, i].real] for i in range(len(orbs))
        ],
        "sigma_c_iw_nodes_im": [
            [float(x) for x in fn[:, i].imag] for i in range(len(orbs))
        ],
        "thiele_coeff_max_rel_vs_pyscf": coeff_rel,
        "node_interp_err_textbook": interp_tb,
        "node_interp_err_pyscf_eval": interp_py,
        "qp_roots_within_1ha_of_eps_mf": nroots,
        "qp_residual_max": float(max(resid)),
    }
    return eqp, sc, diag


def run_gwac(
    mf,
    aux,
    orbs,
    frozen=None,
    vhf_df=False,
    nw=NW,
    iw_cutoff=None,
    nw2=None,
    continuation="pyscf",
):
    """PySCF gw_ac with ferric-matched settings; returns (gw, block dict).

    `nw2` sets the Pade sampling grid separately from the W integral (PySCF's
    `nw2`); ferric always samples on [0] + GL(100), so a W integral with
    nw != 100 is emulated as (nw, nw2=100).

    `continuation`: "pyscf" evaluates the Pade with PySCF's own
    `pade_thiele_ndarray`; "textbook" with ferric's fraction (`thiele_eval`),
    keeping PySCF's numbers under *_pyscf_pade. Used for every @PBE block,
    where the two evaluations differ by up to 7e-3 Ha; at @HF they agree to
    <=1.2e-7 Ha (measured, H2O/cc-pVDZ), so the @HF blocks keep "pyscf"."""
    np = _np()
    from pyscf.gw.gw_ac import GWAC

    gw = GWAC(mf, frozen=frozen, auxbasis=aux)
    gw.nw = nw
    if nw2 is not None:
        gw.nw2 = nw2
    gw.ac_iw_cutoff = iw_cutoff
    gw.ac_pade_npts = PADE_NPTS
    gw.ac_pade_step_ratio = PADE_STEP_RATIO
    gw.vhf_df = vhf_df
    gw.orbs = list(orbs)
    gw.verbose = 0
    gw.kernel()
    nocc_act = gw.nocc
    lpq = gw.Lpq
    eqp, eqp_native, sc, sx_df, sx_ref, vmf, resid = [], [], [], [], [], [], []
    for ip, (pf, pa) in enumerate(zip(gw.orbs_frz, gw.orbs)):
        shift = float(gw.vk[pf, pf] - gw.vxc[pf, pf])
        e0 = float(mf.mo_energy[pa])
        ac = gw.acobj[ip]

        def f(w, ac=ac, e0=e0, shift=shift):
            return w - e0 - (ac.ac_eval(w).real + shift)

        root = _refine(f, float(gw.mo_energy[pa]))
        eqp.append(root)
        eqp_native.append(float(gw.mo_energy[pa]))
        sc.append(float(ac.ac_eval(root).real))
        resid.append(abs(f(root)))
        sx_df.append(float(-np.sum(lpq[:, pf, :nocc_act] ** 2)))
        sx_ref.append(float(gw.vk[pf, pf]))
        vmf.append(float(gw.vxc[pf, pf]))
    extra = {}
    if continuation == "textbook":
        eqp_tb, sc_tb, diag = textbook_continuation(
            gw, mf, list(gw.orbs_frz), list(gw.orbs)
        )
        extra = {
            "eps_qp_pyscf_pade": eqp,
            "sigma_c_at_qp_pyscf_pade": sc,
            "ac": diag,
        }
        eqp, sc = eqp_tb, sc_tb
        resid = [diag["qp_residual_max"]]
    else:
        assert continuation == "pyscf", continuation
    block = {
        "orbs": [int(p) for p in gw.orbs],
        "frozen": 0 if frozen is None else int(frozen),
        "ef": float(gw.ef),
        "eps_mf": [float(mf.mo_energy[p]) for p in gw.orbs],
        "eps_qp": eqp,
        "eps_qp_pyscf_secant": eqp_native,
        "sigma_c_at_qp": sc,
        "sigma_x_df": sx_df,
        "sigma_x_used": sx_ref,
        "v_mf": vmf,
        "qp_residual_max": float(max(resid)),
        "static_shift_max_abs": float(max(abs(a - b) for a, b in zip(sx_ref, vmf))),
        "settings": {
            "nw": nw,
            "nw2": nw2,
            "x0": X0,
            "ac": "pade",
            "ac_pade_npts": PADE_NPTS,
            "ac_pade_step_ratio": PADE_STEP_RATIO,
            "ac_iw_cutoff": iw_cutoff,
            "vhf_df": vhf_df,
            "qpe": f"full QP root, scipy newton, tol {QP_TOL}",
            "pade_evaluation": (
                "textbook Thiele fraction (ferric pade.rs)"
                if continuation == "textbook"
                else "PySCF pade_thiele_ndarray"
            ),
        },
    }
    block.update(extra)
    return gw, block


def ev_loop(gw, mf, update_w):
    """evGW0 (update_w False) / evGW (True), iterated the way ferric does."""
    np = _np()
    from pyscf.gw.gw_ac import get_sigma
    from pyscf.gw.utils.ac_grid import PadeAC, _get_scaled_legendre_roots

    qf, qw = _get_scaled_legendre_roots(NW, X0)
    eval_f = np.concatenate(([0.0], qf))
    orbs_frz = list(gw.orbs_frz)
    nfc = len(mf.mo_energy) - gw.Lpq.shape[1]
    e_mf_act = np.asarray(mf.mo_energy[nfc:], dtype=float)
    eqp = e_mf_act[orbs_frz].copy()
    history = []
    converged = False
    for it in range(EV_MAXIT):
        eprop = e_mf_act.copy()
        eprop[orbs_frz] = eqp
        sig, omega = get_sigma(
            gw,
            orbs_frz,
            gw.Lpq,
            qf,
            qw,
            gw.ef,
            eprop,
            iw_cutoff=None,
            eval_freqs=eval_f,
            mo_energy_w=(eprop if update_w else e_mf_act),
        )
        ac = PadeAC(npts=PADE_NPTS, step_ratio=PADE_STEP_RATIO)
        ac.ac_fit(sig, omega, axis=-1)
        new = []
        for i, p in enumerate(orbs_frz):
            e0 = float(e_mf_act[p])

            def f(w, i=i, e0=e0):
                return w - e0 - ac[i].ac_eval(w).real

            new.append(_refine(f, e0))
        new = np.asarray(new)
        dev = float(np.max(np.abs(new - eqp)))
        eqp = new
        history.append(dev)
        if dev < EV_TOL and (it > 0 or not update_w):
            converged = True
            break
    assert converged, f"ev loop (update_w={update_w}) did not converge: {history}"
    return {
        "orbs": [int(p) + nfc for p in orbs_frz],
        "eps_mf": [float(e_mf_act[p]) for p in orbs_frz],
        "eps_qp": [float(x) for x in eqp],
        "iterations": len(history),
        "max_dev_history": history,
        "ev_tol": EV_TOL,
        "w_updated": update_w,
    }


def cohsex_block(gw, mf, orbs):
    np = _np()
    from pyscf.gw.gw_ac import get_rho_response

    lpq = gw.Lpq
    nocc = gw.nocc
    e = np.asarray(mf.mo_energy, dtype=float)
    pi0 = get_rho_response(0.0, e, np.ascontiguousarray(lpq[:, :nocc, nocc:]))
    naux = pi0.shape[0]
    wp = np.linalg.inv(np.eye(naux) - pi0) - np.eye(naux)
    out_sex, out_coh, eqp = [], [], []
    for m in orbs:
        lm = lpq[:, m, :]  # (naux, nmo)
        wl = wp @ lm
        diag = np.einsum("Pn,Pn->n", lm, wl)
        dsex = -float(np.sum(diag[:nocc]))
        coh = 0.5 * float(np.sum(diag))
        out_sex.append(dsex)
        out_coh.append(coh)
        eqp.append(float(e[m]) + dsex + coh)
    return {
        "orbs": list(orbs),
        "eps_mf": [float(e[m]) for m in orbs],
        "delta_sigma_sex": out_sex,
        "sigma_coh": out_coh,
        "eps_qp": eqp,
    }


def mol_and_prov_base(xyz, basis_name, charge=0, mult=1, ecp_json=None):
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(
        xyz, basis_name, charge=charge, multiplicity=mult, ecp_json=ecp_json
    )
    ll = common.check_basis_like_for_like(mol, basis_name, symbols)
    if ecp_json is not None:
        ll.update(common.check_ecp_like_for_like(mol, ecp_json, symbols))
    return mol, symbols, coords, ll


def provenance(xyz, basis_name, symbols, coords, aux, stability, extra, ecp_json=None):
    import numpy
    import pyscf
    import scipy

    return common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "gw": "pyscf.gw.gw_ac.GWAC / pyscf.gw.ugw_ac.UGWAC",
            "nw": NW,
            "x0": X0,
            "ac_iw_cutoff": None,
            "ac_pade_npts": PADE_NPTS,
            "ac_pade_step_ratio": PADE_STEP_RATIO,
            "qp_root_tol": QP_TOL,
            "ev_tol": EV_TOL,
            "scf": "exact 4-index ERIs, conv_tol 1e-11, conv_tol_grad 1e-8",
            "pbe_grid": f"{MAIN_GRID} unpruned, Becke partition, Becke-1988 radii adjust",
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=list(MAIN_GRID),
        aux=aux,
        frozen_core="none unless the block says frozen > 0",
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=stability,
        ecp_json=ecp_json,
        generator="scripts/validation/gen_gw.py",
        extra=extra,
    )


def header(mol, basis_name, aux_name, naux, ll):
    return {
        "basis": basis_name,
        "aux": aux_name,
        "charge": int(mol.charge),
        "multiplicity": int(mol.spin + 1),
        "nao": int(mol.nao_nr()),
        "naux": naux,
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "like_for_like": ll,
    }


def gen_closed(system):
    xyz_rel, bases = CLOSED[system]
    xyz = common.MOL_DIR / xyz_rel
    for basis_name in bases:
        aux_name = AUX_FOR[basis_name]
        mol, symbols, coords, ll = mol_and_prov_base(xyz, basis_name)
        aux = aux_dict(aux_name, symbols)
        mf, stable = rhf_exact(mol)
        nocc = mol.nelectron // 2
        orbs = window(nocc, mol.nao_nr())
        gw, g_hf = run_gwac(mf, aux, orbs)
        naux = check_naux(gw.with_df, aux_name, symbols)
        payload = header(mol, basis_name, aux_name, naux, ll)
        payload["rhf"] = {
            "energy": float(mf.e_tot),
            "internal_stable": stable,
            "mo_energy": [float(x) for x in mf.mo_energy],
        }
        key = (system, basis_name)
        if key in HF_AC_NODE_CASES:
            # Nodes only: eps_qp / sigma_c_at_qp of this block keep PySCF's
            # evaluation (<=1.2e-7 Ha from the textbook one at @HF).
            _, _, g_hf["ac"] = textbook_continuation(
                gw, mf, list(gw.orbs_frz), list(gw.orbs)
            )
        payload["g0w0_hf"] = g_hf
        if key in FC_CASES:
            _, payload["g0w0_hf_fc1"] = run_gwac(mf, aux, orbs, frozen=1)
        if key in COHSEX_CASES:
            payload["cohsex_hf"] = cohsex_block(gw, mf, orbs)
        if key in EV_CASES:
            payload["evgw0_hf"] = ev_loop(gw, mf, update_w=False)
            payload["evgw_hf"] = ev_loop(gw, mf, update_w=True)
        if key in PBE_CASES:
            mks = rks_pbe_exact(mol)
            _, g_pbe = run_gwac(mks, aux, orbs, vhf_df=True, continuation="textbook")
            g_pbe["rks_energy"] = float(mks.e_tot)
            g_pbe["xc"] = "PBE"
            g_pbe["j"] = "exact (no density fitting)"
            g_pbe["density_floor"] = FERRIC_DENSITY_FLOOR
            raw = rks_pbe_exact(mol, density_floor=None)
            g_pbe["density_floor_mo_energy_shift_max"] = float(
                max(abs(mks.mo_energy[i] - raw.mo_energy[i]) for i in orbs)
            )
            payload["g0w0_pbe"] = g_pbe
            if key == ("h2o", "cc-pvdz"):
                payload["diagnostics"] = {
                    "pbe_recipe_sensitivity": pbe_sensitivity(
                        mol, mks, aux, orbs, nocc
                    ),
                    "hf_quadrature_sensitivity": {
                        f"nw_{n}": _nw_variant(mf, aux, orbs, n, vhf_df=False)[
                            orbs.index(nocc - 1)
                        ]
                        * 27.211386245988
                        for n in (16, 20, 40)
                    },
                }
        payload["provenance"] = provenance(
            xyz,
            basis_name,
            symbols,
            coords,
            aux_name,
            {"rhf_internal_stable": stable},
            {"blocks": [k for k in payload if k not in ("like_for_like",)]},
        )
        path = common.write_reference(ROW, system, basis_name, payload)
        homo = orbs.index(nocc - 1)
        print(
            f"{system}/{basis_name}: HOMO G0W0@HF {g_hf['eps_qp'][homo] * 27.211386245988:.6f} eV"
            f" resid {g_hf['qp_residual_max']:.1e} -> {path.relative_to(common.ROOT)}"
        )


def pbe_sensitivity(mol, mks, aux, orbs, nocc):
    """G0W0@PBE HOMO under PySCF-native knobs, one at a time (diagnostic)."""
    ha = 27.211386245988
    homo = orbs.index(nocc - 1)
    out = {}
    for label, kw in [
        ("matched", {"vhf_df": True}),
        ("exact_sigma_x", {"vhf_df": False}),
        ("iw_cutoff_5", {"vhf_df": True, "iw_cutoff": 5.0}),
        ("nw_16", {"vhf_df": True, "nw": 16}),
        ("nw_20", {"vhf_df": True, "nw": 20}),
        ("pade_npts_16", {"vhf_df": True, "npts": 16}),
        (
            "pyscf_defaults",
            {"vhf_df": False, "iw_cutoff": 5.0, "continuation": "pyscf"},
        ),
    ]:
        # Every variant but pyscf_defaults evaluates the Pade the way ferric
        # does (textbook Thiele); pyscf_defaults is PySCF end to end.
        kw.setdefault("continuation", "textbook")
        if "npts" in kw:
            out[label] = _npts_variant(mks, aux, orbs, kw["npts"])[homo] * ha
            continue
        if "nw" in kw and kw["nw"] != NW:
            # ferric keeps a 101-point Pade grid whatever n_points is; emulate
            # with nw2 so only the W integral changes.
            out[label] = (
                _nw_variant(mks, aux, orbs, kw["nw"], continuation=kw["continuation"])[
                    homo
                ]
                * ha
            )
            continue
        _, b = run_gwac(mks, aux, orbs, **kw)
        out[label] = b["eps_qp"][homo] * ha
    out["units"] = "eV, HOMO QP energy"
    out["pade_evaluation"] = "textbook Thiele (ferric) for all but pyscf_defaults"
    return out


def _npts_variant(mf, aux, orbs, npts):
    """QP energies with a different number of Pade nodes (ferric GwConfig.pade_npts),
    textbook Thiele evaluation (ferric's)."""
    from pyscf.gw.gw_ac import GWAC

    gw = GWAC(mf, auxbasis=aux)
    gw.nw = NW
    gw.ac_iw_cutoff = None
    gw.ac_pade_npts = npts
    gw.ac_pade_step_ratio = PADE_STEP_RATIO
    gw.vhf_df = True
    gw.orbs = list(orbs)
    gw.verbose = 0
    gw.kernel()
    eqp, _, _ = textbook_continuation(gw, mf, list(gw.orbs_frz), list(gw.orbs))
    return eqp


def _nw_variant(mf, aux, orbs, nw, vhf_df=True, continuation="pyscf"):
    """QP energies with an nw-point W integral and ferric's fixed 101-point
    Pade grid (nw2 = 100), refined like the main blocks."""
    _, b = run_gwac(
        mf, aux, orbs, vhf_df=vhf_df, nw=nw, nw2=NW, continuation=continuation
    )
    return b["eps_qp"]


def gen_ecp(system):
    xyz = common.MOL_DIR / ECP[system]
    basis_name = ECP_BASIS
    ecp_json = common.basis_json_path(basis_name)
    aux_name = AUX_FOR[basis_name]
    mol, symbols, coords, ll = mol_and_prov_base(xyz, basis_name, ecp_json=ecp_json)
    aux = aux_dict(aux_name, symbols)
    mf, stable = rhf_exact(mol)
    nocc = mol.nelectron // 2
    assert mf.mo_energy[nocc - 1] < 0, f"{system}: unbound HOMO, unphysical RHF"
    orbs = window(nocc, mol.nao_nr())
    gw, g_hf = run_gwac(mf, aux, orbs)
    naux = check_naux(gw.with_df, aux_name, symbols)
    payload = header(mol, basis_name, aux_name, naux, ll)
    payload["rhf"] = {
        "energy": float(mf.e_tot),
        "internal_stable": stable,
        "mo_energy": [float(x) for x in mf.mo_energy],
    }
    payload["g0w0_hf"] = g_hf
    payload["provenance"] = provenance(
        xyz,
        basis_name,
        symbols,
        coords,
        aux_name,
        {"rhf_internal_stable": stable},
        {"blocks": ["rhf", "g0w0_hf"]},
        ecp_json=ecp_json,
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    homo = orbs.index(nocc - 1)
    print(
        f"{system}/{basis_name}: HOMO G0W0@HF {g_hf['eps_qp'][homo] * 27.211386245988:.6f} eV"
        f" -> {path.relative_to(common.ROOT)}"
    )


def run_ugwac(mf, aux, orbs, ef_override=None, nw=NW):
    from pyscf.gw.ugw_ac import UGWAC

    class _UGWAC(UGWAC):
        def get_ef(self, mo_energy=None):
            if ef_override is None:
                return UGWAC.get_ef(self, mo_energy)
            return ef_override

    gw = _UGWAC(mf, auxbasis=aux)
    gw.nw = nw
    gw.ac_iw_cutoff = None
    gw.ac_pade_npts = PADE_NPTS
    gw.ac_pade_step_ratio = PADE_STEP_RATIO
    gw.orbs = list(orbs)
    gw.verbose = 0
    gw.kernel()
    return gw


def u_spin_block(mf, gw, s, orbs):
    """Refined QP roots of spin `s` from a UGWAC run."""
    np = _np()
    eqp, eqp_native, sc, resid, sx_df = [], [], [], [], []
    nocc_s = gw.nocc[s]
    for ip, p in enumerate(gw.orbs_frz):
        shift = float(gw.vk[s, p, p] - gw.vxc[s, p, p])
        e0 = float(mf.mo_energy[s][p])
        ac = gw.acobj[s, ip]

        def f(w, ac=ac, e0=e0, shift=shift):
            return w - e0 - (ac.ac_eval(w).real + shift)

        root = _refine(f, float(gw.mo_energy[s][p]))
        eqp.append(root)
        eqp_native.append(float(gw.mo_energy[s][p]))
        sc.append(float(ac.ac_eval(root).real))
        resid.append(abs(f(root)))
        sx_df.append(float(-np.sum(gw.Lpq[s][:, p, :nocc_s] ** 2)))
    return {
        "ef": float(gw.ef),
        "eps_mf": [float(mf.mo_energy[s][p]) for p in orbs],
        "eps_qp": eqp,
        "eps_qp_pyscf_secant": eqp_native,
        "sigma_c_at_qp": sc,
        "sigma_x_df": sx_df,
        "qp_residual_max": float(max(resid)),
        "static_shift_max_abs": float(
            max(abs(gw.vk[s, p, p] - gw.vxc[s, p, p]) for p in gw.orbs_frz)
        ),
    }


def gen_open(system):
    xyz_rel, mult, bases = OPEN[system]
    xyz = common.MOL_DIR / xyz_rel
    for basis_name in bases:
        aux_name = AUX_FOR[basis_name]
        mol, symbols, coords, ll = mol_and_prov_base(xyz, basis_name, mult=mult)
        aux = aux_dict(aux_name, symbols)
        uhf, mf = common.run_open_shell(
            mol,
            "uhf",
            conv_tol=CONV_TOL,
            conv_tol_grad=CONV_TOL_GRAD,
            return_mf=True,
        )
        na, nb = mol.nelec
        orbs = window(max(na, nb), mol.nao_nr())
        ea, eb = mf.mo_energy
        ef_a = 0.5 * (ea[na - 1] + ea[na])
        ef_b = 0.5 * (eb[nb - 1] + eb[nb])
        gw_a = run_ugwac(mf, aux, orbs, ef_override=float(ef_a))
        gw_b = run_ugwac(mf, aux, orbs, ef_override=float(ef_b))
        gw_n = run_ugwac(mf, aux, orbs)
        naux = check_naux(gw_a.with_df, aux_name, symbols)
        payload = header(mol, basis_name, aux_name, naux, ll)
        payload["uhf"] = uhf
        payload["u_g0w0_uhf"] = {
            "orbs": orbs,
            "ef_convention": "per-spin mid-gap, as ferric u_sigma.rs",
            "alpha": u_spin_block(mf, gw_a, 0, orbs),
            "beta": u_spin_block(mf, gw_b, 1, orbs),
        }
        payload["diagnostics"] = {
            "pyscf_native_common_ef": {
                "ef": float(gw_n.ef),
                "alpha_eps_qp": u_spin_block(mf, gw_n, 0, orbs)["eps_qp"],
                "beta_eps_qp": u_spin_block(mf, gw_n, 1, orbs)["eps_qp"],
            }
        }
        if basis_name == "cc-pvdz":
            payload["diagnostics"]["old_recipe"] = old_u_recipe(mol, aux, orbs)
        payload["provenance"] = provenance(
            xyz,
            basis_name,
            symbols,
            coords,
            aux_name,
            {"uhf": uhf["stability"]},
            {"blocks": ["uhf", "u_g0w0_uhf", "diagnostics"]},
        )
        path = common.write_reference(ROW, system, basis_name, payload)
        ha = 27.211386245988
        blk = payload["u_g0w0_uhf"]
        print(
            f"{system}/{basis_name}: E_UHF {uhf['energy']:.10f}  "
            f"a-HOMO {blk['alpha']['eps_qp'][orbs.index(na - 1)] * ha:.5f} "
            f"b-HOMO {blk['beta']['eps_qp'][orbs.index(nb - 1)] * ha:.5f} eV"
            f" -> {path.relative_to(common.ROOT)}"
        )


def old_u_recipe(mol, aux, orbs):
    """scripts/gw100/pyscf_u_g0w0.py's recipe: DF-UHF with the GW aux, common ef."""
    from pyscf import scf

    ha = 27.211386245988
    mf = scf.UHF(mol).density_fit(auxbasis=aux)
    mf.conv_tol = CONV_TOL
    mf.verbose = 0
    mf.kernel()
    gw = run_ugwac(mf, aux, orbs)
    na, nb = mol.nelec
    return {
        "e_uhf_df": float(mf.e_tot),
        "alpha_homo_qp_ev": float(gw.mo_energy[0][na - 1] * ha),
        "beta_homo_qp_ev": float(gw.mo_energy[1][nb - 1] * ha),
        "alpha_homo_mf_ev": float(mf.mo_energy[0][na - 1] * ha),
        "beta_homo_mf_ev": float(mf.mo_energy[1][nb - 1] * ha),
        "note": "DF-UHF with an MP2-fitting aux + exact Sigma_x makes vk - v_mf != 0 "
        "(a static shift ferric does not have); ef is PySCF's common one",
    }


def main(argv):
    want = set(argv) or (set(CLOSED) | set(ECP) | set(OPEN))
    for s in CLOSED:
        if s in want:
            gen_closed(s)
    for s in ECP:
        if s in want:
            gen_ecp(s)
    for s in OPEN:
        if s in want:
            gen_open(s)
    unknown = want - set(CLOSED) - set(ECP) - set(OPEN)
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")


if __name__ == "__main__":
    main(sys.argv[1:])
