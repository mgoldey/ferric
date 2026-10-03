"""PySCF + numpy references for the VALIDATION.md "Amplitude-threshold dRPA" row.

Consumer: crates/ferric-mp2/tests/validation_drpa_amplitude.rs
Output:   testdata/reference/validation/drpa_amplitude/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/drpa_amplitude.rs, not
assumed): closed-shell direct RPA (dRPA@HF) by the spin-adapted drCCD Riccati
equations in the Boys-localized basis, B_iajb = 2 (ia|jb) over the ACTIVE
occupieds (the lowest `frozen_core` canonical MOs dropped), Coulomb-metric
density fitting with the aux basis passed in, E_c = 1/2 sum B o T. At eps = 0
nothing is truncated and the Riccati root equals the plasmon-formula dRPA
(proof notebook wiki/notebooks/12-amplitude-threshold-drpa.ipynb). ferric's
in-crate reference `canonical_plasmon_drpa` states the spin factors this
script reproduces:

    A = D + B,  B = 2 (ia|jb),  D_ia = e_a - e_i,
    Omega^2 = eig[(A - B)(A + B)],   E_c = 1/2 (sum Omega - Tr A),

on the EXACTLY semicanonicalized occupied (active) and virtual blocks of the
converged Fock. dRPA is FIRST order in Fock inconsistency (notebook section 3),
so the semicanonicalization and a tight SCF are load-bearing here.

THE REFERENCE, built independently of ferric's code path:
  * RHF: PySCF `scf.RHF`, exact four-centre J/K (ferric's `RhfConfig::default()`
    has no SCF fitting either), conv_tol 1e-12 / conv_tol_grad 1e-10, ferric's
    bundled orbital basis JSON and ferric's geometry in Bohr (common.py).
  * Integrals: PySCF `df.incore.cholesky_eri` with an auxmol built from
    ferric's bundled aux JSON (as gen_kappa_mp2.py). PySCF Cholesky-factors the
    Coulomb metric where ferric uses V^{-1/2}; (ia|jb) is the same number.
  * Fock: `mf.get_fock()` at the converged density, rotated to the MO basis;
    the active-occupied and virtual blocks are re-diagonalized (the max
    off-diagonal element is recorded: the semicanonicalization is a no-op at
    convergence up to that size).
  * Energy: the SYMMETRIC plasmon form Omega^2 = eigh[D^{1/2}(D + 2B)D^{1/2}]
    (similar to (A - B)(A + B) since A - B = D is diagonal and positive) —
    a different eigensolver route from ferric's non-symmetric `eig`. The
    non-symmetric form is evaluated too and must agree to 1e-10.

Self-anchors asserted HERE before anything is written:
  (a) H2/STO-3G with STO-3G as aux reproduces the proof notebook's
      -0.0126072623 to 1e-9 (the notebook's own symbolic value).
  (b) For EVERY system and frozen-core setting, the numpy plasmon energy
      agrees with PySCF `pyscf.gw.rpa.RPA` (frequency integration of
      ln det(1 - chi0 v) + tr(chi0 v), PySCF's scaled-Legendre grid, x0 = 0.5)
      on the SAME DF integrals and the SAME frozen core (`RPA(mf, frozen=n)`
      drops the lowest n MOs, as ferric does). The 40/80/160-point series and
      the final difference are recorded; the difference at nw = 160 must sit
      below NW160_MAX and must not exceed the nw = 40 one (unless both are at
      the 1e-10 floor set by roundoff and the canonical-vs-semicanonical energies PySCF uses). This ties the plasmon
      formula and its spin factors to an independent dRPA implementation;
      the quadrature error is why it anchors the formula, not ferric.

Controls written for the test's negative controls:
  * `e_corr_ring_dropped`: the second-order ring term -1/2 sum B^2/(D_ia+D_jb)
    (the Riccati root with B T + T B + T B T dropped from the residual).
  * every frozen-core setting is written, so a frozen-core-off-by-one ferric
    lands on a DIFFERENT recorded value.

Run (light; seconds per system, ~1 min for n-octane):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_drpa_amplitude.py [system ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "drpa_amplitude"
ROW_NAME = "Amplitude-threshold dRPA"
# system -> (xyz in testdata/molecules/validation, [(basis, aux)], frozen-core settings)
SYSTEMS = {
    "h2": ("h2.xyz", [("sto-3g", "sto-3g")], (0,)),
    "h2o": ("h2o.xyz", [("6-31g", "cc-pvdz-ri"), ("cc-pvdz", "cc-pvdz-ri")], (0, 1)),
    "alkane_4": (
        "alkane_4.xyz",
        [("6-31g", "cc-pvdz-ri"), ("cc-pvdz", "cc-pvdz-ri")],
        (0, 1, 4),
    ),
    "alkane_8": ("alkane_8.xyz", [("6-31g", "cc-pvdz-ri")], (8,)),
}
H2_NOTEBOOK = -0.0126072623
TOL_NOTEBOOK = 1e-9
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
NONSYM_VS_SYM_MAX = 1e-10
NW_SERIES = (40, 80, 160)
X0 = 0.5
# PySCF frequency integration vs the plasmon formula at nw = 160, every system
# and frozen-core setting. Measured max 3.6e-11 (alkane_8/6-31G); the residual
# is flat in nw beyond 40 points (a roundoff floor, not quadrature error).
NW160_MAX = 4e-10


def df_factors(mol, aux_name, symbols):
    """(L[P, mu, nu], naux, aux_bas): Coulomb-metric DF factors, ferric's aux."""
    from pyscf import df, lib

    aux_bas, aux_cart = common.pyscf_basis(aux_name, symbols)
    assert not aux_cart, f"{aux_name}: Cartesian aux shells not supported here"
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    cderi = df.incore.cholesky_eri(mol, auxmol=auxmol)
    naux = cderi.shape[0]
    assert naux == common.ferric_nao(aux_name, symbols), "naux != ferric's"
    return lib.unpack_tril(cderi), naux, aux_bas


def semicanonical(fock_mo, lo, hi):
    """Re-diagonalize fock_mo[lo:hi, lo:hi]: (U, eigenvalues, max |offdiag|)."""
    blk = fock_mo[lo:hi, lo:hi]
    off = blk - np.diag(np.diag(blk))
    w, u = np.linalg.eigh(blk)
    return u, w, float(np.max(np.abs(off))) if off.size else 0.0


def plasmon(lov, e_occ, e_vir):
    """Numpy plasmon dRPA from L[P, i, a] and semicanonical energies.

    Returns (E_c, E_c via non-symmetric eig, E_ring_dropped, min Omega^2)."""
    naux, no, nv = lov.shape
    n = no * nv
    lmat = lov.reshape(naux, n)
    g = lmat.T @ lmat  # (ia|jb)
    b = 2.0 * g
    d = (e_vir[None, :] - e_occ[:, None]).ravel()
    assert d.min() > 0.0, "non-positive orbital-energy difference"
    sd = np.sqrt(d)
    m_sym = sd[:, None] * (np.diag(d) + 2.0 * b) * sd[None, :]
    w2 = np.linalg.eigvalsh(m_sym)
    assert w2.min() > 0.0, f"RPA instability: min Omega^2 {w2.min()}"
    tr_a = float(np.sum(d) + np.trace(b))
    e_sym = 0.5 * (float(np.sum(np.sqrt(w2))) - tr_a)
    # ferric's literal form: eig of the non-symmetric (A - B)(A + B)
    a_mat = np.diag(d) + b
    lam = np.linalg.eigvals((a_mat - b) @ (a_mat + b))
    assert np.max(np.abs(lam.imag)) < 1e-8 * max(1.0, np.max(np.abs(lam.real)))
    e_nonsym = 0.5 * (float(np.sum(np.sqrt(lam.real))) - tr_a)
    denom = d[:, None] + d[None, :]
    e_ring_dropped = -0.5 * float(np.sum(b * b / denom))
    return e_sym, e_nonsym, e_ring_dropped, float(w2.min())


def pyscf_rpa_series(mf, aux_bas, frozen):
    """PySCF RPA e_corr on the same DF basis at each nw in NW_SERIES."""
    from pyscf import df
    from pyscf.gw.rpa import RPA

    rpa = RPA(mf, frozen=frozen if frozen else None)
    rpa.with_df = df.DF(mf.mol, auxbasis=aux_bas)
    rpa.with_df.build()
    rpa.verbose = 0
    eris = rpa.ao2mo()
    out = {}
    for nw in NW_SERIES:
        rpa.e_hf = 0.0  # skip PySCF's DF-HF energy; only e_corr is used
        rpa.kernel(eris=eris, nw=nw, x0=X0)
        out[str(nw)] = float(rpa.e_corr)
    return out


def run_system(system, xyz_name, basis_name, aux_name, frozen_list):
    import pyscf
    from pyscf import scf

    xyz = common.MOL_DIR / xyz_name
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)

    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{system}/{basis_name}: PySCF RHF did not converge")

    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    c = mf.mo_coeff
    fock_mo = c.T @ mf.get_fock() @ c
    ov_max = float(np.max(np.abs(fock_mo[:nocc, nocc:])))
    lao, naux, aux_bas = df_factors(mol, aux_name, symbols)

    fc_blocks = {}
    for fc in frozen_list:
        t0 = time.perf_counter()
        u_o, e_o, off_o = semicanonical(fock_mo, fc, nocc)
        u_v, e_v, off_v = semicanonical(fock_mo, nocc, nmo)
        c_o = c[:, fc:nocc] @ u_o
        c_v = c[:, nocc:] @ u_v
        lov = np.einsum("Pmn,mi,na->Pia", lao, c_o, c_v, optimize=True)
        e_c, e_nonsym, e_ring, w2min = plasmon(lov, e_o, e_v)
        d_ns = abs(e_c - e_nonsym)
        if d_ns > NONSYM_VS_SYM_MAX:
            raise RuntimeError(
                f"{system}/{basis_name}/fc={fc}: symmetric vs non-symmetric plasmon "
                f"differ by {d_ns:.2e}"
            )
        series = pyscf_rpa_series(mf, aux_bas, fc)
        diffs = {k: v - e_c for k, v in series.items()}
        d160 = abs(diffs[str(NW_SERIES[-1])])
        d40 = abs(diffs[str(NW_SERIES[0])])
        if d160 > NW160_MAX or d160 > max(d40, 1e-10):
            raise RuntimeError(
                f"{system}/{basis_name}/fc={fc}: PySCF RPA does not converge onto the "
                f"plasmon value: diffs {diffs} — refusing to write"
            )
        fc_blocks[str(fc)] = {
            "frozen_core": fc,
            "no_active": nocc - fc,
            "nvir": nmo - nocc,
            "e_corr": e_c,
            "e_corr_nonsymmetric_eig": e_nonsym,
            "nonsymmetric_vs_symmetric_abs_diff": d_ns,
            "e_corr_ring_dropped": e_ring,
            "min_omega_squared": w2min,
            "semicanonical_max_offdiag_occ": off_o,
            "semicanonical_max_offdiag_vir": off_v,
            "pyscf_rpa": {
                "method": "pyscf.gw.rpa.RPA",
                "x0": X0,
                "frozen": fc if fc else None,
                "e_corr_by_nw": series,
                "minus_plasmon_by_nw": diffs,
                "abs_diff_at_max_nw": d160,
            },
        }
        print(
            f"{system:9s} {basis_name:8s} fc={fc} no={nocc - fc:3d} nv={nmo - nocc:4d} "
            f"E_c={e_c:+.12f} ring_dropped={e_ring:+.10f} offdiag(o,v)=({off_o:.1e},"
            f"{off_v:.1e}) PySCF-RPA-minus-plasmon "
            + " ".join(f"nw{k}:{v:+.2e}" for k, v in diffs.items())
            + f" [{time.perf_counter() - t0:.1f}s]"
        )
        if system == "h2":
            d_nb = abs(e_c - H2_NOTEBOOK)
            if d_nb > TOL_NOTEBOOK:
                raise RuntimeError(
                    f"H2/STO-3G plasmon {e_c:.12f} vs notebook {H2_NOTEBOOK} "
                    f"(|d| {d_nb:.2e}) — refusing to write"
                )
            fc_blocks[str(fc)]["proof_notebook_value"] = H2_NOTEBOOK
            fc_blocks[str(fc)]["proof_notebook_abs_diff"] = d_nb
            print(f"  anchor (a): H2 vs notebook |d| {d_nb:.2e}")

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "aux_basis": aux_name,
        "nao": mol.nao_nr(),
        "naux": naux,
        "nocc": nocc,
        "nvir": nmo - nocc,
        "nuclear_repulsion": float(mol.energy_nuc()),
        "e_rhf": float(mf.e_tot),
        "fock_max_occ_vir": ov_max,
        "frozen_core": fc_blocks,
        "provenance": common.provenance(
            code="PySCF + numpy",
            version=pyscf.__version__,
            keywords={
                "scf": f"scf.RHF exact J/K, conv_tol {CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}",
                "integrals": "df.incore.cholesky_eri(mol, auxmol from ferric's aux JSON), "
                "Coulomb metric, Cholesky-factorized",
                "fock": "mf.get_fock() at convergence; active-occ and vir blocks "
                "re-diagonalized (semicanonical)",
                "energy": "Omega^2 = eigvalsh[D^1/2 (D + 2B) D^1/2], B = 2(ia|jb); "
                "E = 1/2 (sum Omega - Tr A), A = D + B",
                "anchor_formula": "pyscf.gw.rpa.RPA on the same DF basis, nw "
                f"{list(NW_SERIES)}, x0 {X0}",
                "numpy": np.__version__,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid=None,
            aux=aux_name,
            frozen_core=list(frozen_list),
            scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            stability=None,
            generator="scripts/validation/gen_drpa_amplitude.py",
            extra={
                "basis_self_check": basis_check,
                "aux_basis_json_sha256": common.sha256_file(
                    common.basis_json_path(aux_name)
                ),
            },
        ),
    }
    path = common.write_reference(ROW, system, basis_name, payload)
    print(f"{system:9s} {basis_name:8s} E_RHF={mf.e_tot:.12f} naux={naux} -> {path}")
    return path


def main() -> int:
    only = set(sys.argv[1:])
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(
            f"unknown systems: {sorted(unknown)}; known: {sorted(SYSTEMS)}"
        )
    # Anchor (a) runs first: no other reference is written if it fails.
    order = ["h2"] + [s for s in SYSTEMS if s != "h2"]
    written = []
    for system in order:
        if only and system not in only and system != "h2":
            continue
        xyz_name, bases, frozen_list = SYSTEMS[system]
        for basis_name, aux_name in bases:
            written.append(
                run_system(system, xyz_name, basis_name, aux_name, frozen_list)
            )
    print(f"GEN_DRPA_AMPLITUDE_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
