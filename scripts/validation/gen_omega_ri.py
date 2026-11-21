"""PySCF/numpy references for omega-RI MP2 and omega-RI RPA.

Consumers: crates/ferric-mp2/tests/omega_ri_anchors.rs,
           crates/ferric-rpa/tests/omega_ri_rpa.rs.
Output:    testdata/reference/validation/omega_ri/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from the code)
-----------------------------------------
`ferric_mp2::omega_ri::fit_dressing` builds, for the erfc-attenuated fitting
metric (Glasbrenner, Graf, Ochsenfeld JCTC 16, 6856 (2020) eqs 15-18;
Luenser, Schurkus, Ochsenfeld JCTC 13, 1647 (2017) eq 29),

    B = D (P|ia)_w,   D = L^T V~^-1,   V = L L^T (Coulomb metric),
    V~ = (P|erfc(w r)/r|Q),  (P|ia)_w = (P|erfc(w r)/r|ia)

so (ia|jb) ~ B_ia . B_jb = (ia|P)_w [V~^-1 V V~^-1]_PQ (Q|jb)_w. Canonical
omega-RI-MP2 (`ri_mp2_fitted`) and omega-RI RPA (`run_omega_ri_rpa`, the
full-rank PDEP-RPA trace-log) take that B unchanged.

THE REFERENCE (independent assembly)
------------------------------------
numpy on PySCF integrals: int3c2e/int2c2e under `with_range_coulomb(-w)`
(PySCF: negative w = erfc), the fit assembled as the explicit 4-index matrix
(ia|jb) = A^T V~^-1 V V~^-1 A from `numpy.linalg.solve` (no Cholesky factor,
no dressing: a different construction from ferric's), then

  * MP2 E_OS/E_SS by the explicit (i,a,j,b) denominator sum;
  * dRPA on the 40-point Gauss-Legendre grid of gen_urpa (x0 = 0.5), from
    a PSD Gram factor of the explicit matrix taken by eigh (an independent
    factorization from ferric's Cholesky-solve dressing).

Checks before anything is written:
  1. omega -> 0: the explicit omega-RI matrix at w = 1e-6 equals the Coulomb
     RI matrix (relative 1e-9);
  2. Coulomb anchors: numpy DF-MP2 == PySCF DFMP2 and numpy dRPA == PySCF
     pyscf.gw.rpa.RPA (both 1e-10);
  3. the charge leak of the fit (recorded, not asserted): with n_P = integral
     of chi_P, the fitted charge q_ia = n . c_ia of every occupied-virtual
     pair density (true charge 0) and of every occupied ii density (true
     charge 1), for the Coulomb and the erfc fits.

Run (light; seconds per system):
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/validation/gen_omega_ri.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_attenuated_rpa as gar  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "omega_ri"
ROW_NAME = "omega-RI MP2 / RPA"
OMEGAS = (0.05, 0.1, 0.2, 0.5)
LIMIT_OMEGA = 1e-6
NUMPY_VS_PYSCF_MAX = 1e-10

CASES = {
    "h2o_cc-pvdz": ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    "nh3_cc-pvdz": ("nh3", "cc-pvdz", "cc-pvdz-ri"),
}


def omega_ri_matrix(mf, mol, auxmol, w):
    """Explicit (nov, nov) omega-RI (ia|jb) and the fit coefficients c (naux, nov)."""
    import numpy as np

    v3w, v2w = gar.ints(mol, auxmol, -w if w > 0 else 0.0)
    _, v2 = gar.ints(mol, auxmol, 0.0)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    a = np.einsum("mnP,mi,na->Pia", v3w, c[:, :nocc], c[:, nocc:], optimize=True)
    a = a.reshape(a.shape[0], -1)
    coef = np.linalg.solve(v2w, a)
    return coef.T @ v2 @ coef, coef


def mp2_from_matrix(g, mf):
    import numpy as np

    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    e = mf.mo_energy
    nvir = e.size - nocc
    t = g.reshape(nocc, nvir, nocc, nvir)
    d = (
        e[:nocc, None, None, None]
        - e[None, nocc:, None, None]
        + e[None, None, :nocc, None]
        - e[None, None, None, nocc:]
    )
    e_os = float(np.einsum("iajb,iajb->", t, t / d))
    e_ss = float(np.einsum("iajb,iajb->", t - t.transpose(0, 3, 2, 1), t / d))
    return e_os, e_ss


def gram_factor(g):
    """B (nov, r) with B B^T = g, from eigh (g is PSD)."""
    import numpy as np

    lam, u = np.linalg.eigh(0.5 * (g + g.T))
    keep = lam > 1e-14 * max(lam[-1], 1.0)
    return u[:, keep] * np.sqrt(lam[keep])


def aux_charges(auxmol):
    """n_P = integral of chi_P: overlap with a normalized s Gaussian of
    exponent 1e-9 at the origin, divided by its normalization."""
    import numpy as np
    from pyscf import gto

    eps = 1e-9
    one = gto.M(
        atom=[["Ghost", (0.0, 0.0, 0.0)]],
        basis={"Ghost": [[0, [eps, 1.0]]]},
        unit="Bohr",
        verbose=0,
    )
    s = gto.intor_cross("int1e_ovlp", auxmol, one)[:, 0]
    norm = (2.0 * eps / np.pi) ** 0.75
    return s / norm


def charge_leak(mf, mol, auxmol, w, n_p):
    import numpy as np

    v3w, v2w = gar.ints(mol, auxmol, -w if w > 0 else 0.0)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    a_ov = np.einsum(
        "mnP,mi,na->Pia", v3w, c[:, :nocc], c[:, nocc:], optimize=True
    ).reshape(v3w.shape[2], -1)
    a_ii = np.einsum("mnP,mi,ni->Pi", v3w, c[:, :nocc], c[:, :nocc], optimize=True)
    q_ov = n_p @ np.linalg.solve(v2w, a_ov)
    q_ii = n_p @ np.linalg.solve(v2w, a_ii)
    return {
        "ov_max_abs": float(np.max(np.abs(q_ov))),
        "ov_rms": float(np.sqrt(np.mean(q_ov**2))),
        "ii_max_abs_minus_1": float(np.max(np.abs(q_ii - 1.0))),
    }


def gen_case(key):
    import numpy as np
    import pyscf
    import scipy
    from pyscf import gto, scf
    from pyscf.mp import dfmp2

    system, basis_name, aux_name = CASES[key]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    ll = common.check_basis_like_for_like(mol, basis_name, symbols)
    aux = gen_urpa.aux_dict(aux_name, symbols)
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=aux,
        cart=False,
        verbose=0,
    )
    naux = int(auxmol.nao_nr())
    assert naux == common.ferric_nao(aux_name, symbols), "aux count differs from ferric"

    mf = scf.RHF(mol)
    mf.conv_tol = gar.CONV_TOL
    mf.conv_tol_grad = gar.CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, "RHF did not converge"

    # Coulomb reference: explicit RI matrix through the same code at w = 0.
    g0, _ = omega_ri_matrix(mf, mol, auxmol, 0.0)
    e_os0, e_ss0 = mp2_from_matrix(g0, mf)
    pm = dfmp2.DFMP2(mf)
    pm.with_df = gen_urpa.make_df(mol, aux)
    pm.verbose = 0
    pm.kernel()
    d_mp2 = abs(e_os0 + e_ss0 - float(pm.e_corr))
    assert d_mp2 < NUMPY_VS_PYSCF_MAX, f"numpy DF-MP2 vs PySCF DFMP2 {d_mp2:.2e}"
    d = gar.gaps(mf)
    e_rpa0 = gar.numpy_rpa(gram_factor(g0), d)
    e_rpa0_pyscf = gar.pyscf_rpa_coulomb(mf, aux)
    d_rpa = abs(e_rpa0 - e_rpa0_pyscf)
    assert d_rpa < NUMPY_VS_PYSCF_MAX, f"numpy dRPA vs PySCF RPA {d_rpa:.2e}"

    glim, _ = omega_ri_matrix(mf, mol, auxmol, LIMIT_OMEGA)
    rel_lim = float(np.max(np.abs(glim - g0)) / np.max(np.abs(g0)))
    assert rel_lim < 1e-9, f"omega -> 0 limit not reached: {rel_lim:.2e}"

    n_p = aux_charges(auxmol)
    blocks = []
    for w in OMEGAS:
        g, _ = omega_ri_matrix(mf, mol, auxmol, w)
        e_os, e_ss = mp2_from_matrix(g, mf)
        e_rpa = gar.numpy_rpa(gram_factor(g), d)
        blk = {
            "omega_bohr_inv": w,
            "mp2": {"e_os": e_os, "e_ss": e_ss, "e_corr": e_os + e_ss},
            "mp2_minus_coulomb": e_os + e_ss - (e_os0 + e_ss0),
            "rpa_e_corr": e_rpa,
            "rpa_minus_coulomb": e_rpa - e_rpa0,
            "charge_leak": charge_leak(mf, mol, auxmol, w, n_p),
        }
        blocks.append(blk)
        print(
            f"{key} w={w:<5} dMP2 {blk['mp2_minus_coulomb']:+.3e}  dRPA {blk['rpa_minus_coulomb']:+.3e}  "
            f"q_ov max {blk['charge_leak']['ov_max_abs']:.2e}"
        )
    leak0 = charge_leak(mf, mol, auxmol, 0.0, n_p)
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "aux": aux_name,
        "nao": int(mol.nao_nr()),
        "naux": naux,
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "like_for_like": ll,
        "rhf": {"energy": float(mf.e_tot)},
        "nw": gar.NW,
        "x0": gar.X0,
        "coulomb": {
            "mp2": {"e_os": e_os0, "e_ss": e_ss0, "e_corr": e_os0 + e_ss0},
            "mp2_pyscf_dfmp2": float(pm.e_corr),
            "rpa_e_corr": e_rpa0,
            "rpa_pyscf": e_rpa0_pyscf,
            "charge_leak": leak0,
        },
        "omega_ri": blocks,
        "checks": {
            "numpy_vs_pyscf_dfmp2_abs": d_mp2,
            "numpy_vs_pyscf_rpa_abs": d_rpa,
            "omega_1e-6_vs_coulomb_matrix_rel": rel_lim,
        },
        "generator_seconds": time.perf_counter() - t0,
    }
    payload["provenance"] = common.provenance(
        code="PySCF (integrals, RHF, DFMP2 and RPA Coulomb anchors) + numpy (omega-RI assembly)",
        version=pyscf.__version__,
        keywords={
            "rhf": "scf.RHF, exact 4-index J/K, conv_tol 1e-11",
            "fit": "(ia|jb) = A^T V~^-1 V V~^-1 A, A = (P|erfc(w r)/r|ia), np.linalg.solve",
            "operator": "Mole.with_range_coulomb(-w) on mol and auxmol (erfc)",
            "rpa": "closed-shell dRPA, eigh Gram factor of the explicit matrix, GL grid",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=aux_name,
        frozen_core=None,
        scf_conv={"conv_tol": gar.CONV_TOL, "conv_tol_grad": gar.CONV_TOL_GRAD},
        stability=None,
        generator="scripts/validation/gen_omega_ri.py",
        extra={
            "aux_basis_sha256": common.sha256_file(common.basis_json_path(aux_name))
        },
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    print(
        f"{key}: Coulomb leak {leak0} -> {path.relative_to(common.ROOT)} ({payload['generator_seconds']:.1f} s)"
    )


def main(argv):
    want = list(argv) or list(CASES)
    for k in want:
        gen_case(k)
    print(f"GEN_OMEGA_RI_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
