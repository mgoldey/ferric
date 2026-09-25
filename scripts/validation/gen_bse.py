"""numpy BSE-TDA references for the VALIDATION.md "BSE-TDA" row.

Consumer: crates/ferric-gw/tests/validation_bse.rs
Output:   testdata/reference/validation/bse/<system>_<basis>.json

PySCF 2.13 has no BSE. The reference is an INDEPENDENT numpy BSE-TDA built from
PySCF ingredients (exact-integral RHF, `gw_ac`'s density-fitted Lpq with
ferric's aux, `gw_ac.get_sigma` for the G0W0 self-energy), assembled here with
the conventions ferric uses. Nothing in it calls ferric.

WHAT FERRIC COMPUTES (read from crates/ferric-gw/src/bse.rs `run_bse_tda`,
line numbers as of this generator's commit)
------------------------------------------------------------------------------
  * Spin: SINGLET only; the Tamm-Dancoff A matrix only (bse.rs:300-332). There
    is no triplet and no B block on this path.
  * A_{ia,jb} = (eps^QP_a - eps^QP_i) d_ij d_ab + 2 (ia|jb) - (ab|W|ij)
    (bse.rs:302, :326-331). (ia|jb) is the BARE density-fitted Coulomb
    integral from the full RI tensor b_full (bse.rs:285-291).
  * QP energies: `run_bse_tda` always runs its OWN G0W0@HF, for EVERY MO
    (`qp_mos: Some(0..nmo)`, bse.rs:208-226), with the caller's PdepRpaConfig
    and GwConfig `pade_npts: 0` (-> 18 Pade nodes), `qp_newton_damp: 1.0`.
    It takes no external QP energies and no scissor. The QP solve is
    sigma.rs `solve_qp_for_mo`: textbook Thiele fraction on Sigma_c(ef + i w)
    sampled at [0] + GL(100, u0 = 0.5) (18 nodes by PySCF's `_get_ac_idx`),
    linearized start eps_mf + Z Sigma_c(eps_mf) with Z = 1/(1 - dSigma/dw)
    clamped to [0, 1.5] and a 4-point derivative (h = 0.05), then <= 30 undamped
    Newton steps, stopping when |step| < 1e-7 or |1 - dSigma/dw| < 1e-3.
    `ferric_qp` below re-implements exactly that on PySCF's Sigma_c(ef + i w).
  * W: the STATIC (omega = 0) RPA screened interaction of the SAME PDEP run
    that fed G0W0: (pq|W|rs) = (pq|rs) + sum_a (1/lambda_a(0) - 1) M_a,pq
    M_a,rs (bse.rs:253-298), lambda_a(0) the eigenvalues of the symmetrized
    static dielectric eps~ = I + Pi(0). At trunc_thresh = 0 the mode set is
    the full RI space, so this is L_pq^T (I + Pi(0))^-1 L_rs, basis invariant.
    Pi(0)_PQ = 4 sum_ia L_P,ia L_Q,ia / (eps_a - eps_i) with the MEAN-FIELD
    (HF) orbital energies (G0W0: W is never rebuilt from QP energies), all
    occupied x all virtual pairs (frozen_core = 0 everywhere in this row).
  * (ia) space: all occupied (frozen_core = 0; with frozen_core > 0 the
    all-MO qp_mos range reaches the frozen block and run_gw refuses it) x all
    virtuals, flat index ia = i * nvir + a (bse.rs:315-325).
  * Oscillator strengths: length gauge, f_n = 2/3 Omega_n |sqrt(2) sum_ia
    X_n(ia) <i|r|a>|^2 with unit-normalized X (bse.rs:111-183).

HOW THIS REFERENCE SEPARATES BSE FROM GW
------------------------------------------------------------------------------
The kernel K = 2(ia|jb) - (ab|W|ij) depends only on the mean-field SCF and the
RI tensor; only the diagonal of A depends on the QP energies, and ferric
computes those itself. So K is stored (`bse_kernel`, upper triangle) with the
occ-vir dipoles (`dipole_ia`), and the Rust test diagonalizes
K + diag(eps^QP_a - eps^QP_i) with ferric's OWN QP energies
(BseResult.eps_qp): that comparison contains no GW difference at all.

The stored `bse_singlet` spectrum uses this script's QP energies from
`ferric_qp` (ferric's QP recipe on PySCF's Sigma_c(iw)). Inside HOMO-2..LUMO+2
they agree with PySCF's own roots and the G0W0 row to <= 1.6e-7 Ha. Outside the
window they are NOT reproducible: the Thiele continuation far from ef is
ill-conditioned, and a relative 1e-10 perturbation of Sigma_c(iw) at the nodes
moves core and high-virtual QP energies by up to 0.1-0.3 Ha (`qp.sensitivity`,
measured here; two runs of this script differ by 1.9e-3 Ha on H2O/cc-pVDZ MO
23). ferric's QP energies there are equally arbitrary, so the raw spectrum is
compared at a looser bar, and each MO's QP energy is compared at a bar scaled
by its measured sensitivity.

For a molecule with degenerate orbitals (NH3), that noise also SPLITS the QP
energies of degenerate orbitals, so the diagonal is not invariant under a
rotation inside the degenerate set and Omega depends on the (arbitrary) MO
rotation: `degenerate_rotation_ambiguity` records how far the lowest states
move under random such rotations (zero for the C2v systems).

ANCHORS asserted HERE before anything is written
------------------------------------------------------------------------------
  1. The A-matrix builder, fed EXACT 4-index MO integrals, HF orbital
     energies and W -> v, reproduces PySCF's `tdscf.rhf.get_ab(mf)[0]` (the
     singlet TDA/CIS matrix) to 1e-10: layout, the 2(ia|jb) singlet factor
     and the exchange index order are PySCF's, independently.
  2. Inside HOMO-2..LUMO+2, ferric_qp lands on PySCF's own Pade roots from
     the same run and on the G0W0 row's stored QP energies (where that file
     exists; there is none for CH2O) to < 5e-7 Ha.
  3. Every eigenvalue of the static dielectric I + Pi(0) is >= 1, so
     0 < 1/lambda <= 1 (W screens, never anti-screens).
  4. The stored kernel, read back from its upper triangle and given this
     script's QP diagonal, reproduces `bse_singlet` to 1e-10 (the layout the
     Rust test reads is the layout written).

Stored spectra (lowest N states; N = 5 extended to close a degenerate group):
  bse_singlet       - the reference at this script's QP energies (Omega, f).
  bse_triplet       - A = D - W: must be MISSED by ferric (singlet factor).
  bse_w_off         - A = D + 2(ia|jb): must be MISSED (W reaches the kernel).
  bse_w_bare        - A = D^QP + 2(ia|jb) - (ab|ij): CIS on QP energies,
                      must be MISSED (screening, not bare exchange).
  cis_df            - HF energies, bare DF exchange: ferric `run_cis_tda`.
  cis_exact_tdscf   - PySCF `tdscf.TDA` (exact integrals) + oscillator
                      strengths; ferric's DF-kernel CIS differs by the DF
                      error, recorded as `cis_df_vs_exact_max`.

Run (light; well under a minute per system):
    scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_bse.py [system ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_gw  # noqa: E402  (shared GW recipe: thiele_*, rhf_exact, header)

ROW = "bse"
HA_TO_EV = 27.211386245988
N_STATES = 5
# Ha: states closer than this form one degenerate group. The C3v NH3 geometry
# is given to 6 decimals, which splits its E pairs by up to 3.7e-7 Ha; the
# smallest gap between DISTINCT states in any stored window is asserted to be
# > MIN_GROUP_GAP, so the grouping cannot merge two real states.
DEGEN_TOL = 1e-5
MIN_GROUP_GAP = 1e-3
ANCHOR_TOL = 1e-10
COMPACT_KERNEL = "__BSE_KERNEL_UPPER__"


def encode_kernel(k_upper):
    """base64 of the little-endian float64 bytes of the upper triangle."""
    import base64

    return base64.b64encode(np.asarray(k_upper, dtype="<f8").tobytes()).decode("ascii")


def decode_kernel(text):
    import base64

    return np.frombuffer(base64.b64decode(text), dtype="<f8")


# Pade conditioning probe (qp_sensitivity); the Rust test scales the stored
# per-MO result by its own QP_SENS_FACTOR for the all-MO QP comparison.
QP_SENS_REL = 1e-10
QP_SENS_DRAWS = 20

# ferric sigma.rs solve_qp_for_mo constants
FERRIC_FD_H = 0.05
FERRIC_Z_CLAMP = (0.0, 1.5)
FERRIC_NEWTON_MAXIT = 30
FERRIC_NEWTON_STEP_TOL = 1e-7
FERRIC_NEWTON_FPRIME_MIN = 1e-3

# system -> (xyz under MOL_DIR, bases)
SYSTEMS = {
    "h2o": ("h2o.xyz", ("cc-pvdz", "aug-cc-pvdz")),
    "nh3": ("nh3.xyz", ("cc-pvdz",)),
    "ch2o": ("ch2o.xyz", ("cc-pvdz",)),
}


# ---------------------------------------------------------------------------
# G0W0 for every MO, ferric's QP recipe
# ---------------------------------------------------------------------------


def run_gw_all(mf, aux, nmo):
    """PySCF GWAC over ALL MOs at the G0W0 row's matched settings."""
    from pyscf.gw.gw_ac import GWAC

    gw = GWAC(mf, auxbasis=aux)
    gw.nw = gen_gw.NW
    gw.ac_iw_cutoff = None
    gw.ac_pade_npts = gen_gw.PADE_NPTS
    gw.ac_pade_step_ratio = gen_gw.PADE_STEP_RATIO
    gw.vhf_df = False
    gw.orbs = list(range(nmo))
    gw.verbose = 0
    gw.kernel()
    return gw


def sigma_nodes(gw, mf):
    """Sigma_c(ef + i w) at the 18 Pade nodes for every MO, and the Thiele
    coefficients (checked against PySCF's own fit, as gen_gw does)."""
    from pyscf.gw.gw_ac import _mo_energy_without_core, get_sigma
    from pyscf.gw.utils.ac_grid import _get_ac_idx, _get_scaled_legendre_roots

    qf, qw = _get_scaled_legendre_roots(gw.nw, gen_gw.X0)
    eval_f = np.concatenate(([0.0], gw.freqs))
    e_frz = np.asarray(_mo_energy_without_core(gw, np.asarray(mf.mo_energy)), float)
    orbs = list(gw.orbs_frz)
    sig, omega = get_sigma(
        gw, orbs, gw.Lpq, qf, qw, gw.ef, e_frz, iw_cutoff=None, eval_freqs=eval_f
    )
    assert np.allclose(omega, gw.acobj.omega, rtol=0, atol=1e-14)
    idx = _get_ac_idx(
        len(omega), npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio
    )
    zn = omega[idx]
    fn = sig[:, idx].T
    a = gen_gw.thiele_coeffs(fn, zn)
    rel = float(np.max(np.abs(a - gw.acobj.coeff) / np.maximum(np.abs(a), 1e-300)))
    assert rel < 1e-8, f"Thiele coefficients differ from PySCF's: rel {rel:.2e}"
    return zn, a, fn, [int(i) for i in idx]


def ferric_qp(zn, a, e0):
    """sigma.rs solve_qp_for_mo, line for line (HF reference: no static shift)."""

    def sc(w):
        return gen_gw.thiele_eval(w + 0.0j, zn, a).real

    def dsc(w, h=FERRIC_FD_H):
        return (-sc(w + 2 * h) + 8 * sc(w + h) - 8 * sc(w - h) + sc(w - 2 * h)) / (
            12 * h
        )

    with np.errstate(divide="ignore"):
        z = 1.0 / (1.0 - dsc(e0))
    z = float(np.clip(z, *FERRIC_Z_CLAMP))
    e = e0 + z * sc(e0)
    converged = False
    for _ in range(FERRIC_NEWTON_MAXIT):
        f = e - e0 - sc(e)
        fp = 1.0 - dsc(e)
        if abs(fp) < FERRIC_NEWTON_FPRIME_MIN:
            break
        step = -f / fp
        e += step
        if abs(step) < FERRIC_NEWTON_STEP_TOL:
            converged = True
            break
    return float(e), float(sc(e)), converged


# ---------------------------------------------------------------------------
# BSE-TDA assembly
# ---------------------------------------------------------------------------


def build_a(d_eps_ia, coul_iajb, exch_ijab, c_coul, c_exch):
    """A[ia,jb] = diag(d_eps) + c_coul (ia|jb) - c_exch X[i,j,a,b], flat ia = i*nvir + a.
    `exch_ijab` is (ab|K|ij) stored as [i, j, a, b]."""
    nocc, nvir = d_eps_ia.shape
    n = nocc * nvir
    k = c_coul * coul_iajb.reshape(n, n)
    x = exch_ijab.transpose(0, 2, 1, 3).reshape(n, n)  # [i,a,j,b]
    a = k - c_exch * x
    a[np.diag_indices(n)] += d_eps_ia.ravel()
    return a


def degenerate_groups(omega, n_want):
    groups, cur = [], [0]
    for k in range(1, len(omega)):
        if omega[k] - omega[cur[-1]] < DEGEN_TOL:
            cur.append(k)
        else:
            groups.append(cur)
            if sum(len(g) for g in groups) >= n_want:
                return groups, float(omega[k] - omega[cur[-1]])
            cur = [k]
    groups.append(cur)
    return groups, None


def spectrum(a, dip_ia, n_want, check_gap=False):
    w, x = np.linalg.eigh(a)
    groups, next_gap = degenerate_groups(w, n_want)
    n = sum(len(g) for g in groups)
    mu = np.sqrt(2.0) * np.einsum(
        "dia,ian->nd", dip_ia, x.reshape(*dip_ia.shape[1:], -1)
    )
    f = (2.0 / 3.0) * w * np.sum(mu**2, axis=1)
    # the gap to the next distinct state after the stored ones
    gap_after = float(w[n] - w[n - 1]) if n < len(w) else None
    out = {
        "omega": [float(v) for v in w[:n]],
        "omega_ev": [float(v * HA_TO_EV) for v in w[:n]],
        "osc_strength": [float(v) for v in f[:n]],
        "groups": groups,
        "group_osc_strength": [float(sum(f[k] for k in g)) for g in groups],
        "gap_to_next_state": gap_after,
        "degenerate_tol": DEGEN_TOL,
    }
    if next_gap is not None:
        out["gap_between_last_group_and_next"] = next_gap
    # smallest gap between distinct groups, the next state included
    firsts = [g[0] for g in groups] + ([n] if n < len(w) else [])
    lasts = [g[-1] for g in groups]
    min_gap = min(float(w[firsts[k + 1]] - w[lasts[k]]) for k in range(len(firsts) - 1))
    out["min_gap_between_groups"] = min_gap
    if check_gap:
        assert min_gap > MIN_GROUP_GAP, (
            f"distinct states {min_gap:.2e} Ha apart: grouping at {DEGEN_TOL} is ambiguous"
        )
    return out


def orbital_groups(e_mf, tol=1e-6):
    """Sets of degenerate mean-field orbitals (index lists)."""
    out, p, nmo = [], 0, len(e_mf)
    while p < nmo:
        q = p
        while q + 1 < nmo and abs(e_mf[q + 1] - e_mf[p]) < tol:
            q += 1
        out.append(list(range(p, q + 1)))
        p = q + 1
    return out


def qp_sensitivity(zn, fn, e_mf, rel=QP_SENS_REL, draws=QP_SENS_DRAWS, seed=0):
    """Per-MO conditioning of ferric's QP recipe: the largest change of
    ferric_qp over `draws` random RELATIVE perturbations of size `rel` of
    Sigma_c(ef + i w) at the Pade nodes (the level at which ferric and PySCF
    agree on Sigma_c(iw), validation_gw.rs: 8.6e-11 Ha). Far from ef (core,
    high virtuals) the Thiele continuation is ill-conditioned and this is
    O(1e-2..1e-1) Ha; inside the QP window it is O(1e-9)."""
    rng = np.random.default_rng(seed)
    a0 = gen_gw.thiele_coeffs(fn, zn)
    q0 = np.array(
        [ferric_qp(zn, a0[:, p], float(e_mf[p]))[0] for p in range(len(e_mf))]
    )
    worst = np.zeros(len(e_mf))
    for _ in range(draws):
        ap = gen_gw.thiele_coeffs(fn * (1.0 + rel * rng.normal(size=fn.shape)), zn)
        q = np.array(
            [ferric_qp(zn, ap[:, p], float(e_mf[p]))[0] for p in range(len(e_mf))]
        )
        worst = np.maximum(worst, np.abs(q - q0))
    for g in orbital_groups(e_mf):
        worst[g] = worst[g].max()
    return worst


def kernel_blocks(lpq, e_mf, nocc):
    """(ia|jb), bare (ab|ij) and static (ab|W|ij) from the DF tensor, and the
    eigenvalues of the static dielectric I + Pi(0) (HF energies, all occ x
    all vir). Exchange-type blocks are stored as [i, j, a, b]."""
    naux = lpq.shape[0]
    l_ia = lpq[:, :nocc, nocc:]
    l_ij = lpq[:, :nocc, :nocc]
    l_ab = lpq[:, nocc:, nocc:]
    coul = np.einsum("Pia,Pjb->iajb", l_ia, l_ia)
    bare_x = np.einsum("Pij,Pab->ijab", l_ij, l_ab)
    d_mf = e_mf[None, nocc:] - e_mf[:nocc, None]
    eps_t = np.eye(naux) + 4.0 * np.einsum("Pia,Qia,ia->PQ", l_ia, l_ia, 1.0 / d_mf)
    lam = np.linalg.eigvalsh(eps_t)
    winv = np.linalg.inv(eps_t)
    winv = 0.5 * (winv + winv.T)
    w_x = np.einsum("Pij,PQ,Qab->ijab", l_ij, winv, l_ab)
    return coul, bare_x, w_x, lam


def rotation_ambiguity(lpq, e_mf, eps_qp, nocc, n, draws=4, seed=0):
    """How far the lowest n singlets move when the MOs inside each set of
    degenerate mean-field orbitals are rotated, with THESE QP energies on the
    diagonal. G0W0 splits degenerate orbitals' QP energies by Pade noise
    (see qp_sensitivity), so the diagonal is not invariant under that
    rotation and Omega is defined only to this amount; ferric's MO rotation
    and PySCF's are unrelated. Zero when no MO is degenerate (C2v here)."""
    groups = [g for g in orbital_groups(e_mf) if len(g) > 1]
    if not groups:
        return 0.0
    rng = np.random.default_rng(seed)
    d_qp = eps_qp[None, nocc:] - eps_qp[:nocc, None]

    def lowest(l_):
        coul, _, w_x, _ = kernel_blocks(l_, e_mf, nocc)
        return np.linalg.eigvalsh(build_a(d_qp, coul, w_x, 2.0, 1.0))[:n]

    w0 = lowest(lpq)
    worst = 0.0
    for _ in range(draws):
        u = np.eye(e_mf.size)
        for g in groups:
            q, _ = np.linalg.qr(rng.normal(size=(len(g), len(g))))
            u[np.ix_(g, g)] = q
        lr = np.einsum("Ppq,pr,qs->Prs", lpq, u, u)
        worst = max(worst, float(np.max(np.abs(lowest(lr) - w0))))
    return worst


def max_group_diff(sa, sb):
    """max |Omega| diff over the first min(n) states (for recorded controls)."""
    n = min(len(sa["omega"]), len(sb["omega"]))
    return float(np.max(np.abs(np.array(sa["omega"][:n]) - np.array(sb["omega"][:n]))))


# ---------------------------------------------------------------------------
# One system
# ---------------------------------------------------------------------------


def gen_one(system, basis_name):
    from pyscf import ao2mo, tdscf
    from pyscf.tdscf.rhf import get_ab

    xyz_rel, _ = SYSTEMS[system]
    xyz = common.MOL_DIR / xyz_rel
    aux_name = gen_gw.AUX_FOR[basis_name]
    mol, symbols, coords, ll = gen_gw.mol_and_prov_base(xyz, basis_name)
    aux = gen_gw.aux_dict(aux_name, symbols)
    mf, stable = gen_gw.rhf_exact(mol)
    nmo = mf.mo_energy.size
    nocc = mol.nelectron // 2
    nvir = nmo - nocc
    e_hf = np.asarray(mf.mo_energy, float)
    c = np.asarray(mf.mo_coeff)

    # --- G0W0@HF for every MO, ferric's QP recipe -------------------------
    gw = run_gw_all(mf, aux, nmo)
    naux = gen_gw.check_naux(gw.with_df, aux_name, symbols)
    zn, coef, fn, node_idx = sigma_nodes(gw, mf)
    qp = [ferric_qp(zn, coef[:, p], float(e_hf[p])) for p in range(nmo)]
    eps_qp = np.array([q[0] for q in qp])
    qp_conv = [q[2] for q in qp]
    ef = float(gw.ef)
    assert abs(ef - 0.5 * (e_hf[nocc - 1] + e_hf[nocc])) < 1e-14, "ef is not mid-gap"

    # Anchor 2: in HOMO-2..LUMO+2, ferric_qp against (a) PySCF's own Pade
    # (its `ac_eval`, root refined to 1e-12 as gen_gw does) from THIS run and
    # (b) the G0W0 row's stored numbers where that file exists. Both differ
    # from ferric_qp only by the Pade EVALUATION (<= 1.6e-7 Ha at @HF).
    win = gen_gw.window(nocc, nmo)
    assert all(qp_conv[p] for p in win), "Newton did not converge inside the QP window"
    pyscf_win = []
    for p in win:
        ac, e0 = gw.acobj[p], float(e_hf[p])
        pyscf_win.append(
            gen_gw._refine(
                lambda w, ac=ac, e0=e0: w - e0 - ac.ac_eval(w).real,
                float(gw.mo_energy[p]),
            )
        )
    d_pyscf = max(abs(eps_qp[p] - q) for p, q in zip(win, pyscf_win))
    assert d_pyscf < 5e-7, (
        f"{system}/{basis_name}: ferric_qp vs PySCF Pade {d_pyscf:.2e}"
    )
    window_check = {"pyscf_pade_this_run_max_abs_diff": float(d_pyscf)}
    gw_ref_path = common.reference_path("gw", system, basis_name)
    if gw_ref_path.is_file():
        blk = json.loads(gw_ref_path.read_text())["g0w0_hf"]
        d = max(abs(eps_qp[p] - q) for p, q in zip(blk["orbs"], blk["eps_qp"]))
        window_check["gw_reference"] = str(gw_ref_path.relative_to(common.ROOT))
        window_check["max_abs_diff"] = float(d)
        assert d < 5e-7, (
            f"{system}/{basis_name}: ferric_qp misses the G0W0 row by {d:.2e}"
        )

    # --- integrals --------------------------------------------------------
    lpq = np.asarray(gw.Lpq)  # (naux, nmo, nmo), (pq|rs) = sum_P L_P,pq L_P,rs
    assert lpq.shape == (naux, nmo, nmo)
    coul, bare_x, w_x, lam = kernel_blocks(lpq, e_hf, nocc)
    assert lam.min() >= 1.0 - 1e-12, f"static dielectric eigenvalue {lam.min()} < 1"
    d_hf = e_hf[None, nocc:] - e_hf[:nocc, None]

    # dipole integrals (origin 0, ferric's convention; origin-independent)
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r_ao = mol.intor_symmetric("int1e_r", comp=3)
    dip_ia = np.einsum("up,duv,vq->dpq", c[:, :nocc], r_ao, c[:, nocc:])

    # --- Anchor 1: builder vs PySCF get_ab, exact integrals ---------------
    eri_mo = ao2mo.restore(1, ao2mo.full(mol, c), nmo)
    coul_ex = eri_mo[:nocc, nocc:, :nocc, nocc:]
    x_ex = eri_mo[:nocc, :nocc, nocc:, nocc:]  # (ij|ab) = (ab|ij)
    a_mine = build_a(d_hf, coul_ex, x_ex, 2.0, 1.0)
    a_pyscf = get_ab(mf)[0].reshape(nocc * nvir, nocc * nvir)
    anchor_getab = float(np.max(np.abs(a_mine - a_pyscf)))
    assert anchor_getab < ANCHOR_TOL, f"A builder vs PySCF get_ab: {anchor_getab:.2e}"

    # --- spectra ----------------------------------------------------------
    d_qp = eps_qp[None, nocc:] - eps_qp[:nocc, None]
    zero_x = np.zeros_like(w_x)
    a_singlet = build_a(d_qp, coul, w_x, 2.0, 1.0)
    singlet = spectrum(a_singlet, dip_ia, N_STATES, True)
    n_cmp = len(singlet["omega"])
    qp_sens = qp_sensitivity(zn, fn, e_hf)
    ambiguity = rotation_ambiguity(lpq, e_hf, eps_qp, nocc, n_cmp)

    # The QP-independent kernel K = A - diag(d eps^QP), stored so the Rust
    # test can put ferric's OWN QP energies on the diagonal and re-diagonalize
    # (exact separation of BSE from GW). Self-check of the storage layout:
    # K + diag(this reference's d eps) reproduces `singlet` to ANCHOR_TOL.
    nov = nocc * nvir
    kernel = a_singlet - np.diag(d_qp.ravel())
    iu = np.triu_indices(nov)
    k_upper = kernel[iu]
    k_back = np.zeros((nov, nov))
    k_back[iu] = k_upper
    k_back = k_back + np.triu(k_back, 1).T
    w_back = np.linalg.eigvalsh(k_back + np.diag(d_qp.ravel()))[:n_cmp]
    kernel_roundtrip = float(np.max(np.abs(w_back - np.array(singlet["omega"]))))
    assert kernel_roundtrip < ANCHOR_TOL, f"kernel round trip {kernel_roundtrip:.2e}"
    assert float(np.max(np.abs(kernel - kernel.T))) < 1e-12, "kernel not symmetric"
    triplet = spectrum(build_a(d_qp, coul, w_x, 0.0, 1.0), dip_ia, n_cmp)
    w_off = spectrum(build_a(d_qp, coul, zero_x, 2.0, 1.0), dip_ia, n_cmp)
    w_bare = spectrum(build_a(d_qp, coul, bare_x, 2.0, 1.0), dip_ia, n_cmp)
    cis_df = spectrum(build_a(d_hf, coul, bare_x, 2.0, 1.0), dip_ia, N_STATES, True)

    # Exact-integral CIS: the dense eigenproblem of PySCF's own get_ab A
    # matrix (anchor 1 already tied it to the builder), cross-checked against
    # PySCF's Davidson `tdscf.TDA` and its `oscillator_strength`. PySCF 2.13's
    # per-root `converged` flags are unreliable here (CH2O/cc-pVDZ: root 2
    # flagged unconverged at conv_tol 1e-7 while its energy equals the dense
    # eigenvalue to 1e-14), so the flags are recorded, not asserted, and the
    # VALUES are asserted against the dense solve, which is the stored value.
    n_td = len(cis_df["omega"])
    cis_exact = spectrum(a_pyscf, dip_ia, n_td)
    td = tdscf.TDA(mf)
    td.nstates = n_td + 4
    td.conv_tol = 1e-6
    td.max_cycle = 400
    td.verbose = 0
    td.kernel()
    f_td = np.asarray(td.oscillator_strength(gauge="length"))
    td_e_diff = float(np.max(np.abs(td.e[:n_td] - np.array(cis_exact["omega"][:n_td]))))
    td_f_diff = float(
        np.max(np.abs(f_td[:n_td] - np.array(cis_exact["osc_strength"][:n_td])))
    )
    assert td_e_diff < 1e-8, f"tdscf.TDA vs dense get_ab: {td_e_diff:.2e} Ha"
    assert td_f_diff < 1e-6, (
        f"tdscf.TDA f vs dense get_ab + my formula: {td_f_diff:.2e}"
    )
    cis_exact["tdscf_tda_davidson_max_omega_diff"] = td_e_diff
    cis_exact["tdscf_tda_davidson_max_f_diff"] = td_f_diff
    cis_exact["tdscf_tda_conv_tol"] = td.conv_tol
    cis_exact["tdscf_tda_converged_flags"] = [bool(x) for x in td.converged[:n_td]]
    cis_df_vs_exact = float(
        np.max(np.abs(np.array(cis_df["omega"]) - np.array(cis_exact["omega"])))
    )

    homo, lumo = nocc - 1, nocc
    payload = gen_gw.header(mol, basis_name, aux_name, naux, ll)
    payload["rhf"] = {
        "energy": float(mf.e_tot),
        "internal_stable": stable,
        "mo_energy": [float(x) for x in e_hf],
    }
    payload["nocc"] = nocc
    payload["nvir"] = nvir
    payload["qp"] = {
        "recipe": "ferric sigma.rs solve_qp_for_mo on PySCF gw_ac Sigma_c(ef+iw) "
        "(textbook Thiele, linearized start, Z clamp [0,1.5], 4-pt FD h=0.05, "
        "<=30 Newton steps, |step|<1e-7)",
        "ef": ef,
        "eps_mf": [float(x) for x in e_hf],
        "eps_qp": [float(x) for x in eps_qp],
        "newton_converged": qp_conv,
        "window": win,
        "window_vs_g0w0_row": window_check,
        "ac_nodes_index": node_idx,
        "gap_qp_ev": float((eps_qp[lumo] - eps_qp[homo]) * HA_TO_EV),
        "pyscf_native_eps_qp": [float(x) for x in gw.mo_energy],
        "sensitivity": [float(x) for x in qp_sens],
        "sensitivity_recipe": f"max |d eps_qp| over {QP_SENS_DRAWS} random relative "
        f"perturbations of size {QP_SENS_REL:g} of Sigma_c at the Pade nodes",
    }
    payload["w_static"] = {
        "lambda_min": float(lam.min()),
        "lambda_max": float(lam.max()),
        "n_modes": int(naux),
        "pi0": "4 sum_ia L_ia L_ia^T / (eps_a - eps_i), HF energies, all occ x all vir",
    }
    payload["anchors"] = {
        "a_builder_vs_pyscf_get_ab_exact_max": anchor_getab,
        "a_builder_vs_pyscf_get_ab_tol": ANCHOR_TOL,
    }
    payload["bse_singlet"] = singlet
    payload["bse_kernel"] = {
        "definition": "K = A - diag(eps^QP_a - eps^QP_i) = 2(ia|jb) - (ab|W(0)|ij), "
        "singlet, flat ia = i*nvir + a, row-major upper triangle (i <= j) of the "
        "nov x nov matrix, as base64 of little-endian float64 bytes",
        "nov": nov,
        "upper_f64le_b64": COMPACT_KERNEL,
        "roundtrip_max": kernel_roundtrip,
    }
    payload["dipole_ia"] = {
        "definition": "<i|r|a>, origin 0, [x, y, z][i*nvir + a], PySCF MO phases "
        "(the same MOs as bse_kernel)",
        "values": [[float(v) for v in dip_ia[d].ravel()] for d in range(3)],
    }
    payload["degenerate_rotation_ambiguity"] = ambiguity
    payload["controls"] = {
        "bse_triplet": triplet,
        "bse_w_off": w_off,
        "bse_w_bare": w_bare,
        "max_diff_singlet_vs_triplet": max_group_diff(singlet, triplet),
        "max_diff_singlet_vs_w_off": max_group_diff(singlet, w_off),
        "max_diff_singlet_vs_w_bare": max_group_diff(singlet, w_bare),
    }
    payload["cis_df"] = cis_df
    payload["cis_exact_tdscf"] = cis_exact
    payload["cis_df_vs_exact_max"] = cis_df_vs_exact
    payload["provenance"] = gen_gw.provenance(
        xyz,
        basis_name,
        symbols,
        coords,
        aux_name,
        {"rhf_internal_stable": stable},
        {
            "generator": "scripts/validation/gen_bse.py",
            "bse": "numpy BSE-TDA singlet: A = dQP + 2(ia|jb) - (ab|W(0)|ij), "
            "W(0) = L^T (I + Pi(0))^-1 L, Pi(0) on HF energies, full occ x vir, "
            "frozen core none, DF with the listed aux (PySCF gw_ac Lpq)",
            "blocks": [k for k in payload if k != "like_for_like"],
        },
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    # The kernel is the bulk of the file (CH2O: 28 920 numbers). As a JSON
    # array it is 721 KB, over the repo's 500 KB large-file limit, so it is
    # stored as base64 of the little-endian float64 bytes (bit-exact, 308 KB).
    text = path.read_text()
    assert text.count(f'"{COMPACT_KERNEL}"') == 1
    path.write_text(
        text.replace(f'"{COMPACT_KERNEL}"', json.dumps(encode_kernel(k_upper)))
    )
    back = decode_kernel(json.loads(path.read_text())["bse_kernel"]["upper_f64le_b64"])
    assert np.array_equal(back, k_upper), "kernel did not round-trip"
    print(
        f"{system}/{basis_name}: nocc {nocc} nvir {nvir} naux {naux}; QP gap "
        f"{payload['qp']['gap_qp_ev']:.4f} eV; window vs G0W0 row "
        f"{window_check.get('max_abs_diff', float('nan')):.2e}, vs PySCF Pade "
        f"{window_check['pyscf_pade_this_run_max_abs_diff']:.2e}; "
        f"non-converged QP {[p for p, ok in enumerate(qp_conv) if not ok]}"
    )
    print(
        f"   BSE singlet (eV) {['%.5f' % v for v in singlet['omega_ev']]} groups {singlet['groups']}"
    )
    print(f"   f {['%.6f' % v for v in singlet['osc_strength']]}")
    print(f"   triplet {['%.5f' % (v * HA_TO_EV) for v in triplet['omega']]}")
    print(f"   W off   {['%.5f' % (v * HA_TO_EV) for v in w_off['omega']]}")
    print(f"   W=v     {['%.5f' % (v * HA_TO_EV) for v in w_bare['omega']]}")
    print(
        f"   CIS(DF) {['%.5f' % (v * HA_TO_EV) for v in cis_df['omega']]}; "
        f"vs tdscf.TDA exact {cis_df_vs_exact:.2e} Ha; get_ab anchor {anchor_getab:.1e}"
    )
    print(f"   degenerate-MO rotation ambiguity of Omega {ambiguity:.2e} Ha")
    print(
        "   QP sensitivity (Ha) "
        + " ".join(f"{p}:{v:.0e}" for p, v in enumerate(qp_sens))
    )
    print(f"   -> {path.relative_to(common.ROOT)}")


def main(argv):
    want = set(argv) or set(SYSTEMS)
    unknown = want - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    for s, (_, bases) in SYSTEMS.items():
        if s in want:
            for b in bases:
                gen_one(s, b)


if __name__ == "__main__":
    main(sys.argv[1:])
