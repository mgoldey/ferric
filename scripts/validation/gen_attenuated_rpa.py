"""PySCF/numpy references for the VALIDATION.md "Attenuated PDEP-RPA" row.

Consumer: crates/ferric-rpa/tests/validation_attenuated_rpa.rs.
Output:   testdata/reference/validation/attenuated_rpa/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from the code, not a doc)
----------------------------------------------------
`run_pdep_rpa(mol, obs, dfbs, op, rhf, cfg)` (crates/ferric-rpa/src/lib.rs)
calls `ferric_mp2::rimp2::compute_rpa_intermediates(.., op, ..)`, which threads
the SAME `op` into all three RI pieces:

  * the 2-centre metric      `threeindex::coulomb_metric_2c(op, dfbs)`
  * its inverse square root  `metric_inverse_sqrt(&v2c, op)`: symmetric
    regularized eigh (drop metric eigenvalues < 1e-10) for erf/terf, lower
    Cholesky L^-1 for Coulomb/erfc
  * the 3-centre integrals   `ThreeIndexSource::build(op, obs, dfbs, ..)`

so B^P_ia = [V_op^-1/2 (Q|op|ia)]_P with op = erf(w r)/r or erfc(w r)/r. The
RHF is the ordinary FULL-Coulomb RHF (the operator only enters the RPA). The
energy is the closed-shell dRPA

    E_c = sum_k w_k/(2 pi) [ ln det(I + Pi(iw_k)) - tr Pi(iw_k) ],
    Pi(iw) = B^T diag(4 D / (w^2 + D^2)) B,   D = e_a - e_i,

(closed-shell factor 4 = 2 per spin x 2 spins), full rank (trunc_thresh = 0),
on the SAME 40-point Gauss-Legendre grid w = x0 (1+x)/(1-x), x0 = 0.5 as the
U-RPA row (gen_urpa.scaled_legendre = PySCF's _get_scaled_legendre_roots, and
ferric's QuadratureScheme::GaussLegendre with u0 = 0.5 is the same map).

THE REFERENCE (independent assembly)
------------------------------------
numpy on PySCF integrals: int3c2e (mu nu|P) and int2c2e (P|Q) built with
`Mole.with_range_coulomb(w)` on BOTH the orbital and the auxiliary Mole
(PySCF: positive w = erf, long range; NEGATIVE w = erfc, short range), MO
transform with PySCF's exact-integral RHF orbitals, the metric factorized
exactly as ferric does (eigh + 1e-10 lindep drop for erf, Cholesky for erfc).
Before anything is written the script checks:

  1. kernel identity: erf(w) + erfc(w) == Coulomb for BOTH the int3c2e and the
     int2c2e blocks (proves the sign convention and that both Moles carry the
     attenuation), and that the attenuated metric differs from Coulomb;
  2. Coulomb anchor: the same numpy assembly with the Coulomb operator equals
     PySCF's own `pyscf.gw.rpa.RPA` (DF, same aux) to 1e-10 -- proves the
     numpy RPA assembly and grid;
  3. the omega limits of THIS reference: erfc(w -> 0) and erf(w -> inf)
     reproduce the Coulomb value (recorded as `limits`; the anchor omegas the
     test uses are chosen from that scan).

omega grid (Bohr^-1; ferric's Operator takes Bohr^-1, the CLI/Python take
Angstrom^-1): 0.2, 0.222254 (= 0.420 A^-1, the production default
`RsMp2RpaConfig::omega`), 0.42 (the SAME digits as the default in A^-1: the
unit-slip control) and 1.0; both erf (the LR dRPA of the default DeltaLr
RS-MP2+RPA formulation) and erfc (the SR dRPA of CoupledRings).

CONTROL VALUES (numpy, recorded for the test to assert ferric MISSES them):
  * `mixed_coulomb_metric`: attenuated 3-centre with the COULOMB metric -- what
    ferric gives if `op` stops reaching `coulomb_metric_2c`.
  * `lindep_sensitivity` (erf only): the erf value with the lindep drop at
    1e-12 / 1e-8 instead of 1e-10, and the metric spectrum around the cut --
    documents how much of any erf disagreement the regularization can own.
  * `non_additivity_info`: E(erf) + E(erfc) - E(Coulomb); RPA is nonlinear in
    the kernel, so this is NOT zero and is NOT asserted.

Run (light; seconds per system):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        .venv/bin/python scripts/validation/gen_attenuated_rpa.py [system_basis ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "attenuated_rpa"
ROW_NAME = "Attenuated PDEP-RPA"
NW = gen_urpa.NW
X0 = gen_urpa.X0
CONV_TOL = gen_urpa.CONV_TOL
CONV_TOL_GRAD = gen_urpa.CONV_TOL_GRAD
NUMPY_VS_PYSCF_MAX = 1e-10
IDENTITY_MAX = 1e-10
# ferric: BOHR_INV_PER_ANG_INV = 1/1.8897259886; default omega = 0.420 A^-1.
BOHR_PER_ANG = 1.8897259886
OMEGA_DEFAULT_BOHR = 0.420 / BOHR_PER_ANG
OMEGAS = (0.2, OMEGA_DEFAULT_BOHR, 0.42, 1.0)
# ferric eigh_inverse_sqrt LINDEP_THRESH (crates/ferric-mp2/src/rimp2.rs).
LINDEP = 1e-10
LINDEP_SENSITIVITY = (1e-12, 1e-8)
# Limit scans: erfc(w -> 0) and erf(w -> inf) must tend to Coulomb.
ERFC_LIMIT_OMEGAS = (1e-2, 1e-3, 1e-4, 1e-5)
ERF_LIMIT_OMEGAS = (1e2, 1e3, 1e4, 1e5)
# The anchor omegas the test runs (one of each scan above).
ERFC_ANCHOR = 1e-5
ERF_ANCHOR = 1e5

CASES = {  # key -> (system, basis, aux)
    "h2o_cc-pvdz": ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    "nh3_cc-pvdz": ("nh3", "cc-pvdz", "cc-pvdz-ri"),
    "h2o_aug-cc-pvdz": ("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
}


def _np():
    import numpy as np

    return np


def ints(mol, auxmol, pyscf_omega):
    """(nao,nao,naux) int3c2e and (naux,naux) int2c2e. pyscf_omega: 0 =
    Coulomb, > 0 = erf, < 0 = erfc (PySCF's convention)."""
    from pyscf import df

    with mol.with_range_coulomb(pyscf_omega), auxmol.with_range_coulomb(pyscf_omega):
        v3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        v2 = auxmol.intor("int2c2e")
    return v3, v2


def pyscf_omega(kind, omega):
    return {"coulomb": 0.0, "erf": omega, "erfc": -omega}[kind]


def metric_inv_sqrt(v2, kind, lindep=LINDEP):
    """ferric's metric_inverse_sqrt: eigh + drop < lindep for erf, Cholesky
    L^-1 otherwise. Returns (M, info) with M^T M = V^+ (so B = M (Q|ia))."""
    np = _np()
    import scipy.linalg

    lam = np.linalg.eigvalsh(v2)
    info = {
        "metric_min_eigenvalue": float(lam[0]),
        "metric_max_eigenvalue": float(lam[-1]),
    }
    if kind == "erf":
        lam, u = np.linalg.eigh(v2)
        keep = lam >= lindep
        info["metric_modes_dropped"] = int(np.count_nonzero(~keep))
        info["smallest_kept_eigenvalue"] = float(lam[keep][0])
        # How close the nearest metric eigenvalue sits to the cut, in decades:
        # a mode within ~0.1 decade could be kept by one code and dropped by
        # the other.
        pos = lam[lam > 0]
        info["nearest_mode_log10_distance_from_cut"] = (
            float(np.min(np.abs(np.log10(pos) - np.log10(lindep))))
            if pos.size
            else None
        )
        us = u[:, keep] / np.sqrt(lam[keep])
        return us @ u[:, keep].T, info
    L = scipy.linalg.cholesky(v2, lower=True)
    return scipy.linalg.solve_triangular(L, np.eye(v2.shape[0]), lower=True), info


def b_ov(mf, v3, m):
    """B (nov, naux) = [M (Q|ia)]^T."""
    np = _np()
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    iaP = np.einsum("mnP,mi,na->iaP", v3, c[:, :nocc], c[:, nocc:], optimize=True)
    naux = v3.shape[2]
    return iaP.reshape(-1, naux) @ m.T


def gaps(mf):
    np = _np()
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    e = mf.mo_energy
    return (e[None, nocc:] - e[:nocc, None]).ravel()


def numpy_rpa(B, d, nw=NW):
    """Closed-shell dRPA on the GL grid; B (nov, naux), d = e_a - e_i (nov,)."""
    np = _np()
    naux = B.shape[1]
    freqs, wts = gen_urpa.scaled_legendre(nw, X0)
    e = 0.0
    for w, wt in zip(freqs, wts):
        chi = 4.0 * d / (w**2 + d**2)
        pi = (B.T * chi) @ B
        sign, logdet = np.linalg.slogdet(np.eye(naux) + pi)
        assert sign > 0, "dielectric matrix not positive definite"
        e += wt / (2.0 * np.pi) * (logdet - np.trace(pi))
    return float(e)


def e_rpa(mf, mol, auxmol, kind, omega, lindep=LINDEP, metric_kind=None):
    """numpy dRPA with `kind` 3-centre and `metric_kind` (default: same)
    metric. Returns (E_c, metric info)."""
    metric_kind = metric_kind or kind
    v3, _ = ints(mol, auxmol, pyscf_omega(kind, omega))
    _, v2 = ints(mol, auxmol, pyscf_omega(metric_kind, omega))
    m, info = metric_inv_sqrt(v2, metric_kind, lindep)
    return numpy_rpa(b_ov(mf, v3, m), gaps(mf)), info


def pyscf_rpa_coulomb(mf, aux):
    from pyscf.gw.rpa import RPA

    with_df = gen_urpa.make_df(mf.mol, aux)
    rpa = RPA(mf)
    rpa.with_df = with_df
    rpa.verbose = 0
    rpa.kernel(nw=NW, x0=X0)
    return float(rpa.e_corr)


def kernel_identity(mol, auxmol, omega):
    np = _np()
    v3c, v2c = ints(mol, auxmol, 0.0)
    v3l, v2l = ints(mol, auxmol, omega)
    v3s, v2s = ints(mol, auxmol, -omega)
    ident = max(
        float(np.max(np.abs(v3l + v3s - v3c))), float(np.max(np.abs(v2l + v2s - v2c)))
    )
    assert ident < IDENTITY_MAX, f"erf+erfc != Coulomb ({ident:.2e}) at omega={omega}"
    for name, v2 in (("erf", v2l), ("erfc", v2s)):
        assert np.max(np.abs(v2 - v2c)) > 1e-3, f"{name} metric equals Coulomb"
    return ident


def gen_case(key):
    import numpy as np
    import pyscf
    import scipy
    from pyscf import gto, scf

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
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    rhf_stable = common._stability_status(mf)[1]
    assert rhf_stable, "RHF not internally stable"

    # --- check 2: Coulomb numpy == PySCF RPA -------------------------------
    e_coul, coul_info = e_rpa(mf, mol, auxmol, "coulomb", 0.0)
    e_coul_pyscf = pyscf_rpa_coulomb(mf, aux)
    d_anchor = abs(e_coul - e_coul_pyscf)
    if d_anchor > NUMPY_VS_PYSCF_MAX:
        raise RuntimeError(
            f"numpy Coulomb RPA {e_coul:.12f} vs PySCF {e_coul_pyscf:.12f}"
        )

    worst_ident = 0.0
    blocks = []
    for omega in OMEGAS:
        worst_ident = max(worst_ident, kernel_identity(mol, auxmol, omega))
        entry = {
            "omega_bohr_inv": omega,
            "omega_angstrom_inv": omega * BOHR_PER_ANG,
        }
        for kind in ("erf", "erfc"):
            e, info = e_rpa(mf, mol, auxmol, kind, omega)
            mixed, _ = e_rpa(mf, mol, auxmol, kind, omega, metric_kind="coulomb")
            blk = {"e_corr": e, "metric": info, "mixed_coulomb_metric": mixed}
            if kind == "erf":
                blk["lindep_sensitivity"] = {
                    f"{t:g}": e_rpa(mf, mol, auxmol, kind, omega, lindep=t)[0] - e
                    for t in LINDEP_SENSITIVITY
                }
            entry[kind] = blk
        entry["non_additivity_info"] = (
            entry["erf"]["e_corr"] + entry["erfc"]["e_corr"] - e_coul
        )
        blocks.append(entry)
        print(
            f"{key} w={omega:.6f} erf {entry['erf']['e_corr']:.12f} "
            f"(drop {entry['erf']['metric']['metric_modes_dropped']}, "
            f"lindep sens {entry['erf']['lindep_sensitivity']}) "
            f"erfc {entry['erfc']['e_corr']:.12f}  mixed erf "
            f"{entry['erf']['mixed_coulomb_metric'] - entry['erf']['e_corr']:+.2e} "
            f"mixed erfc {entry['erfc']['mixed_coulomb_metric'] - entry['erfc']['e_corr']:+.2e}  "
            f"nonadd {entry['non_additivity_info']:+.3e}"
        )

    limits = {"erfc": {}, "erf": {}}
    for kind, omegas in (("erfc", ERFC_LIMIT_OMEGAS), ("erf", ERF_LIMIT_OMEGAS)):
        for omega in omegas:
            e, info = e_rpa(mf, mol, auxmol, kind, omega)
            limits[kind][f"{omega:g}"] = {
                "e_corr": e,
                "minus_coulomb": e - e_coul,
                "metric": info,
            }
            print(f"{key} limit {kind}({omega:g}) - coulomb = {e - e_coul:+.3e}")

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis_name,
        "aux": aux_name,
        "charge": 0,
        "multiplicity": 1,
        "nao": int(mol.nao_nr()),
        "naux": naux,
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "like_for_like": ll,
        "rhf": {"energy": float(mf.e_tot), "internal_stable": bool(rhf_stable)},
        "nw": NW,
        "x0": X0,
        "lindep": LINDEP,
        "coulomb": {
            "e_corr": e_coul,
            "e_corr_pyscf_rpa": e_coul_pyscf,
            "numpy_vs_pyscf_abs_diff": d_anchor,
            "metric": coul_info,
        },
        "attenuated": blocks,
        "limits": limits,
        "anchors": {
            "erfc_omega": ERFC_ANCHOR,
            "erfc_e_corr": limits["erfc"][f"{ERFC_ANCHOR:g}"]["e_corr"],
            "erf_omega": ERF_ANCHOR,
            "erf_e_corr": limits["erf"][f"{ERF_ANCHOR:g}"]["e_corr"],
        },
        "checks": {
            "erf_plus_erfc_minus_coulomb_max_abs": worst_ident,
            "numpy_coulomb_vs_pyscf_rpa_abs": d_anchor,
        },
        "generator_seconds": time.perf_counter() - t0,
    }
    payload["provenance"] = common.provenance(
        code="PySCF (integrals, RHF, Coulomb RPA anchor) + numpy (attenuated dRPA assembly)",
        version=pyscf.__version__,
        keywords={
            "rhf": "scf.RHF, exact 4-index Coulomb J/K, conv_tol 1e-11",
            "operator": "Mole.with_range_coulomb(w) on mol AND auxmol: w>0 erf, w<0 erfc",
            "attenuated_integrals": "int3c2e (mu nu|P) and int2c2e (P|Q) with the same kernel",
            "metric_factor": "erf: symmetric eigh V^-1/2, drop eigenvalues < 1e-10; "
            "erfc/Coulomb: Cholesky L^-1 (ferric metric_inverse_sqrt)",
            "rpa": "closed-shell dRPA, full rank, ln det(I+Pi) - tr Pi",
            "nw": NW,
            "x0": X0,
            "frequency_grid": "Gauss-Legendre, w = x0 (1+x)/(1-x) (rpa._get_scaled_legendre_roots)",
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
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability={"rhf_internal_stable": bool(rhf_stable)},
        generator="scripts/validation/gen_attenuated_rpa.py",
        extra={
            "aux_basis_json": str(
                common.basis_json_path(aux_name).relative_to(common.ROOT)
            ),
            "aux_basis_sha256": common.sha256_file(common.basis_json_path(aux_name)),
        },
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    print(
        f"{key}: E_RHF {mf.e_tot:.10f} E_c(Coulomb) {e_coul:.12f} (PySCF RPA |d| "
        f"{d_anchor:.1e}) identity {worst_ident:.1e} ({payload['generator_seconds']:.1f} s) "
        f"-> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = list(argv) or list(CASES)
    unknown = [k for k in want if k not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown} (known: {sorted(CASES)})")
    for k in want:
        gen_case(k)
    print(f"GEN_ATTENUATED_RPA_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
