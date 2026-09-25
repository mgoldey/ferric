"""PySCF references for the VALIDATION.md "U-RPA" row (open-shell PDEP-RPA).

Consumer: crates/ferric-rpa/tests/validation_urpa.rs.
Output:   testdata/reference/validation/urpa/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-rpa/src/{lib,energy,sternheimer,
quadrature}.rs, not from a doc) AND HOW THIS SCRIPT MATCHES IT
------------------------------------------------------------------------------
`run_u_pdep_rpa` builds, per spin s, B^P_{ia,s} = L^{-1}(Q|ia)_s (Cholesky
inverse of the Coulomb metric, `metric_inverse_sqrt`), and the spin-summed
dielectric
    eps(iw) = I + Pi_a(iw) + Pi_b(iw),
    Pi_s(iw) = B_s diag(2 D / (w^2 + D^2)) B_s^T,   D = e_a - e_i  (spin s)
(`sternheimer::dielectric_matrix_unrestricted`, prefactor 2 per spin). It
diagonalizes eps(0) in full (Lanczos arm = one dense eigh), keeps the modes with
|lambda - 1| > trunc_thresh, and at each quadrature node sums
    E_c = sum_k w_k / (2 pi) sum_alpha [ ln lambda_alpha(iw_k) + 1 - lambda_alpha(iw_k) ]
over the eigenvalues of the projected eps(iw_k) (`energy::rpa_correlation_energy`).

PySCF 2.13.1 `pyscf.gw.urpa.URPA` (kernel = `rpa.kernel`) computes the SAME
object: diel = sum_s L_s^T diag(2 e_ov f_ov / (w^2 + e_ov^2)) L_s with
e_ov = e_i - e_a < 0 and f_ov = 1 per spin, i.e. diel = -Pi, and
    E_c = sum_k w_k / (2 pi) [ ln det(I - diel) + tr(diel) ]
        = sum_k w_k / (2 pi) [ ln det(I + Pi) - tr(Pi) ],
identical at full rank (the eigenbasis is complete, and the log-det/trace are
basis-invariant; L and B differ by an orthogonal aux rotation, since both are
Cholesky-whitened with the same metric). It returns E_c ONLY as `e_corr`
(`e_hf` is a density-fitted HF energy we do not use). Specifically:

  * Frequency grid: PySCF `_get_scaled_legendre_roots(nw, x0)`:
    w = x0 (1 + x)/(1 - x), weight w_GL * 2 x0 / (1 - x)^2 on nw Gauss-Legendre
    nodes. ferric `QuadratureScheme::GaussLegendre` with `u0` is the SAME map
    and weight (`quadrature::gauss_legendre_nodes`). So the grid is MATCHED,
    not converged-and-compared: both sides run NW = 40 (PySCF's default),
    x0 = u0 = 0.5. The NW-convergence of PySCF's value (20/40/80/160) is
    recorded as a diagnostic (and the NW = 20 value feeds a quadrature control).
  * Density fitting: PySCF `df.DF(mol, auxbasis=<ferric's aux JSON>)`; the
    Coulomb metric is Cholesky-factored (dfump2 `_init_mp_df_eris`); naux
    checked against ferric's parser.
  * Reference: EXACT-integral UHF (PySCF `scf.UHF`, 4-index ERIs, not DF),
    converged to 1e-11 and stability-followed (common.run_open_shell); ferric
    runs exact-integral UHF with `check_stability` + stability descent. The
    test asserts E_UHF agreement before any RPA number.
  * Frozen core: none on either side (URPA frozen=None; ferric frozen_core 0).
  * Rank: full (ferric trunc_thresh = 0).

INDEPENDENT CROSS-CHECK (in this script): the energy is ALSO evaluated in plain
numpy from PySCF's per-spin Cholesky ovL tensors and orbital energies
(`numpy_urpa`), without PySCF's kernel; the two must agree to 1e-10 or the
reference is refused. The same numpy code produces the control values:

  * alpha-only / beta-only: Pi from ONE spin channel (what dropping a Pi_s in
    ferric would give) -- the test asserts ferric MISSES both.
  * truncated: ferric's PDEP truncation emulated in numpy (eigh of eps(0), keep
    |lambda - 1| > t, project eps(iw) onto the kept modes). The test asserts
    ferric at trunc_thresh = t MATCHES this and MISSES the full-rank value.

Closed-shell anchor (h2o): PySCF `RPA` on the RHF and `URPA` on the same RHF
converted to UHF; they must agree (PySCF's own consistency), and ferric's
`run_pdep_rpa` and `run_u_pdep_rpa` (on a singlet UHF) must equal both.

Run (light; seconds per system):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_urpa.py [system ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "urpa"
NW = 40
X0 = 0.5
NW_CONVERGENCE = (20, 40, 80, 160)
TRUNC_THRESHES = (1e-3, 1e-2, 1e-1)
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-8
NUMPY_VS_PYSCF_MAX = 1e-10

AUX_FOR = {
    "cc-pvdz": "cc-pvdz-ri",
    "aug-cc-pvdz": "aug-cc-pvdz-rifit",
}

OPEN = {  # system -> (xyz, multiplicity, bases)
    "oh": ("oh.xyz", 2, ("cc-pvdz", "aug-cc-pvdz")),
    "ch3": ("ch3.xyz", 2, ("cc-pvdz",)),
    "nh2": ("nh2.xyz", 2, ("cc-pvdz",)),
    "o2": ("o2.xyz", 3, ("cc-pvdz",)),
}
# (system, basis) that also record a stability-checked ROHF energy.
ROHF_CONTROL = {("oh", "cc-pvdz")}
CLOSED = {  # system -> (xyz, bases)
    "h2o": ("h2o.xyz", ("cc-pvdz",)),
}


def _np():
    import numpy as np

    return np


def aux_dict(aux_name, symbols):
    aux, cart = common.pyscf_basis(aux_name, symbols)
    if cart:
        raise ValueError(f"{aux_name}: Cartesian l>=2 aux shells cannot be matched")
    return aux


def scaled_legendre(nw, x0=X0):
    from pyscf.gw.rpa import _get_scaled_legendre_roots

    return _get_scaled_legendre_roots(nw, x0)


def make_df(mol, aux):
    from pyscf import df

    with_df = df.DF(mol, auxbasis=aux)
    with_df.build()
    return with_df


def check_naux(with_df, aux_name, symbols):
    naux = int(with_df.get_naoaux())
    want = common.ferric_nao(aux_name, symbols)
    assert naux == want, f"{aux_name}: PySCF naux {naux} != ferric {want}"
    return naux


def spin_channels(rpa, eris, unrestricted):
    """[(L_s (nov, naux), D_s = e_a - e_i > 0 (nov,), occupation factor)].

    Closed shell: one channel with the spin factor 2 folded into the
    occupation difference (PySCF f_ov = 2), exactly as rpa.make_f_ov does."""
    np = _np()
    e_ov = rpa.make_e_ov()
    f_ov = rpa.make_f_ov()
    if unrestricted:
        out = []
        for s in (0, 1):
            nov = e_ov[s].size
            L = np.asarray(eris.get_ov_blk(s, 0, nov))
            out.append((L, -np.asarray(e_ov[s]), np.asarray(f_ov[s])))
        return out
    nov = e_ov.size
    L = np.asarray(eris.get_ov_blk(0, nov))
    return [(L, -np.asarray(e_ov), np.asarray(f_ov))]


def pi_matrix(channels, omega, use=None):
    """Pi(iw) = sum_s L_s^T diag(2 f D / (w^2 + D^2)) L_s."""
    np = _np()
    naux = channels[0][0].shape[1]
    pi = np.zeros((naux, naux))
    for s, (L, d, f) in enumerate(channels):
        if use is not None and s not in use:
            continue
        chi = 2.0 * f * d / (omega**2 + d**2)
        pi += (L.T * chi) @ L
    return pi


def numpy_urpa(channels, nw, use=None, trunc=None):
    """E_c = sum_k w_k/(2 pi) [ln det(V^T eps V) + tr(I - V^T eps V)].

    V = I (full rank) or, when `trunc` is given, the static eigenvectors of
    eps(0) with |lambda - 1| > trunc (ferric's PDEP truncation; at least one
    mode kept, as ferric's `n_keep.max(1)`). Returns (E_c, n_keep)."""
    np = _np()
    naux = channels[0][0].shape[1]
    basis = None
    n_keep = naux
    if trunc is not None:
        lam, vec = np.linalg.eigh(np.eye(naux) + pi_matrix(channels, 0.0, use))
        order = np.argsort(-np.abs(lam - 1.0), kind="stable")
        n_keep = max(1, int(np.sum(np.abs(lam - 1.0) > trunc)))
        basis = vec[:, order[:n_keep]]
        # How far the nearest static mode sits from the cut: a mode within
        # ~1e-9 of it could flip n_keep between codes.
        numpy_urpa.last_margin = float(np.min(np.abs(np.abs(lam - 1.0) - trunc)))
    freqs, wts = scaled_legendre(nw)
    e = 0.0
    for w, wt in zip(freqs, wts):
        eps = np.eye(naux) + pi_matrix(channels, w, use)
        if basis is not None:
            eps = basis.T @ eps @ basis
        sign, logdet = np.linalg.slogdet(eps)
        assert sign > 0, "dielectric matrix not positive definite"
        e += wt / (2.0 * np.pi) * (logdet + np.trace(np.eye(eps.shape[0]) - eps))
    return float(e), int(n_keep)


def run_pyscf_rpa(mf, with_df, unrestricted, nw):
    from pyscf.gw.rpa import RPA
    from pyscf.gw.urpa import URPA

    rpa = (URPA if unrestricted else RPA)(mf)
    rpa.with_df = with_df
    rpa.verbose = 0
    eris = rpa.ao2mo()
    t0 = time.perf_counter()
    rpa.kernel(eris=eris, nw=nw, x0=X0)
    return rpa, eris, float(rpa.e_corr), time.perf_counter() - t0


def rpa_block(mf, with_df, unrestricted):
    """PySCF E_c at NW, its NW convergence, the numpy cross-check and the
    spin-channel / truncation control values."""
    rpa, eris, e_corr, dt = run_pyscf_rpa(mf, with_df, unrestricted, NW)
    channels = spin_channels(rpa, eris, unrestricted)
    e_np, _ = numpy_urpa(channels, NW)
    diff = abs(e_np - e_corr)
    if diff > NUMPY_VS_PYSCF_MAX:
        raise RuntimeError(
            f"numpy RPA {e_np:.12f} disagrees with PySCF {e_corr:.12f} ({diff:.2e})"
        )
    conv = {}
    for nw in NW_CONVERGENCE:
        conv[str(nw)] = (
            e_corr if nw == NW else run_pyscf_rpa(mf, with_df, unrestricted, nw)[2]
        )
    trunc = {}
    for t in TRUNC_THRESHES:
        e_t, n_keep = numpy_urpa(channels, NW, trunc=t)
        trunc[f"{t:g}"] = {
            "trunc_thresh": t,
            "e_corr": e_t,
            "n_keep": n_keep,
            "nearest_mode_distance_from_cut": numpy_urpa.last_margin,
        }
    block = {
        "method": "pyscf.gw.urpa.URPA" if unrestricted else "pyscf.gw.rpa.RPA",
        "nw": NW,
        "x0": X0,
        "frozen": None,
        "e_corr": e_corr,
        "e_corr_numpy": e_np,
        "numpy_vs_pyscf_abs_diff": diff,
        "naux_full_rank": int(channels[0][0].shape[1]),
        "nw_convergence": conv,
        "nw_convergence_note": "PySCF e_corr at each NW, x0 = 0.5; the comparison runs at NW = 40 "
        "on BOTH sides (same Gauss-Legendre map), so this only documents the quadrature error",
        "truncated": trunc,
        "truncated_note": "numpy emulation of ferric's PDEP truncation: eigh of eps(0), keep "
        "|lambda-1| > trunc_thresh, project eps(iw) at NW = 40",
        "pyscf_kernel_seconds": dt,
    }
    if unrestricted:
        block["e_corr_alpha_only"] = numpy_urpa(channels, NW, use={0})[0]
        block["e_corr_beta_only"] = numpy_urpa(channels, NW, use={1})[0]
        block["nov"] = [int(c[0].shape[0]) for c in channels]
    return block


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


def provenance(xyz, basis_name, symbols, coords, aux, stability, extra):
    import numpy
    import pyscf
    import scipy

    return common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "rpa": "pyscf.gw.urpa.URPA / pyscf.gw.rpa.RPA (kernel = rpa.kernel)",
            "nw": NW,
            "x0": X0,
            "frequency_grid": "Gauss-Legendre, w = x0 (1+x)/(1-x) (rpa._get_scaled_legendre_roots)",
            "df": "df.DF(mol, auxbasis=ferric aux JSON), Cholesky-whitened metric",
            "scf": "exact 4-index ERIs, conv_tol 1e-11, conv_tol_grad 1e-8",
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=aux,
        frozen_core=None,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=stability,
        generator="scripts/validation/gen_urpa.py",
        extra=extra,
    )


def gen_open(system):
    xyz_rel, mult, bases = OPEN[system]
    xyz = common.MOL_DIR / xyz_rel
    for basis_name in bases:
        t0 = time.perf_counter()
        aux_name = AUX_FOR[basis_name]
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name, multiplicity=mult)
        ll = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = aux_dict(aux_name, symbols)
        uhf, mf = common.run_open_shell(
            mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
        )
        with_df = make_df(mol, aux)
        naux = check_naux(with_df, aux_name, symbols)
        payload = header(mol, basis_name, aux_name, naux, ll)
        payload["uhf"] = uhf
        payload["urpa"] = rpa_block(mf, with_df, unrestricted=True)
        blocks = ["uhf", "urpa"]
        stab = {"uhf": uhf["stability"]}
        if (system, basis_name) in ROHF_CONTROL:
            # The reference-orbital control: ferric's U-RPA on an ROHF state
            # must MISS the UHF reference. Only the ROHF energy is recorded, so
            # the test can prove ferric's control run is on the real ROHF state.
            rohf = common.run_open_shell(
                mol, "rohf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD
            )
            payload["rohf"] = rohf
            blocks.append("rohf")
            stab["rohf"] = rohf["stability"]
        payload["generator_seconds"] = time.perf_counter() - t0
        payload["provenance"] = provenance(
            xyz,
            basis_name,
            symbols,
            coords,
            aux_name,
            stab,
            {"blocks": blocks},
        )
        path = common.write_reference(ROW, system, basis_name, payload)
        b = payload["urpa"]
        print(
            f"{system}/{basis_name}: E_UHF {uhf['energy']:.10f}  E_c(URPA,nw={NW}) "
            f"{b['e_corr']:.10f}  numpy |d| {b['numpy_vs_pyscf_abs_diff']:.1e}  "
            f"nw160-nw40 {b['nw_convergence']['160'] - b['e_corr']:+.2e}  "
            f"a-only {b['e_corr_alpha_only']:.6f} b-only {b['e_corr_beta_only']:.6f}  "
            f"trunc {[(k, v['n_keep'], round(v['e_corr'] - b['e_corr'], 9)) for k, v in b['truncated'].items()]}  "
            f"({payload['generator_seconds']:.1f} s) -> {path.relative_to(common.ROOT)}"
        )


def gen_closed(system):
    from pyscf import scf

    xyz_rel, bases = CLOSED[system]
    xyz = common.MOL_DIR / xyz_rel
    for basis_name in bases:
        t0 = time.perf_counter()
        aux_name = AUX_FOR[basis_name]
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        ll = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux = aux_dict(aux_name, symbols)
        mf = scf.RHF(mol)
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
        mf.max_cycle = 500
        mf.verbose = 0
        mf.kernel()
        assert mf.converged, "RHF did not converge"
        rhf_stable = common._stability_status(mf)[1]
        assert rhf_stable, "RHF not internally stable"
        umf = scf.addons.convert_to_uhf(mf)
        with_df = make_df(mol, aux)
        naux = check_naux(with_df, aux_name, symbols)
        payload = header(mol, basis_name, aux_name, naux, ll)
        payload["rhf"] = {
            "energy": float(mf.e_tot),
            "internal_stable": bool(rhf_stable),
        }
        payload["rpa"] = rpa_block(mf, with_df, unrestricted=False)
        payload["urpa_on_rhf"] = rpa_block(umf, with_df, unrestricted=True)
        d = abs(payload["rpa"]["e_corr"] - payload["urpa_on_rhf"]["e_corr"])
        payload["rpa_vs_urpa_abs_diff"] = d
        if d > NUMPY_VS_PYSCF_MAX:
            raise RuntimeError(f"PySCF RPA vs URPA on the same RHF differ by {d:.2e}")
        payload["generator_seconds"] = time.perf_counter() - t0
        payload["provenance"] = provenance(
            xyz,
            basis_name,
            symbols,
            coords,
            aux_name,
            {"rhf_internal_stable": bool(rhf_stable)},
            {"blocks": ["rhf", "rpa", "urpa_on_rhf"]},
        )
        path = common.write_reference(ROW, system, basis_name, payload)
        print(
            f"{system}/{basis_name}: E_RHF {mf.e_tot:.10f}  E_c(RPA) "
            f"{payload['rpa']['e_corr']:.10f}  URPA-RPA {d:.1e}  "
            f"({payload['generator_seconds']:.1f} s) -> {path.relative_to(common.ROOT)}"
        )


def main(argv):
    known = set(OPEN) | set(CLOSED)
    want = set(argv) or known
    unknown = want - known
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)} (known: {sorted(known)})")
    for s in CLOSED:
        if s in want:
            gen_closed(s)
    for s in OPEN:
        if s in want:
            gen_open(s)


if __name__ == "__main__":
    main(sys.argv[1:])
