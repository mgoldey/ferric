"""PySCF references for the VALIDATION row "ROHF-referenced U-RPA / U-G0W0 / U-MP2".

Consumer: crates/ferric-gw/tests/validation_rohf_reference.rs.
Output:   testdata/reference/validation/rohf_semicanonical/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-scf/src/semicanonical.rs
`unrestricted_reference`, called at the top of run_u_pdep_rpa / run_u_gw /
u_ri_mp2) AND HOW THIS SCRIPT MATCHES IT
------------------------------------------------------------------------------
On a ROHF reference ferric semi-canonicalizes before any correlated step: with
the spin Fock matrices F_a = h + J[D] - K[D_a], F_b = h + J[D] - K[D_b] of the
converged ROHF density, the ROHF MOs C are rotated separately inside the
alpha-occupied / alpha-virtual blocks (eigenvectors of C^T F_a C restricted to
each block) and likewise for beta (nocc_b occupied), giving per-spin orbitals
C_a, C_b and orbital energies e_a, e_b (the block eigenvalues). The occupied
span of each spin is unchanged, so density and energy are the ROHF ones. Those
orbitals and energies are then used EXACTLY as a UHF reference's.

This script does the same in numpy from a PySCF ROHF (exact 4-index integrals,
conv_tol 1e-11, stability-followed: common.run_open_shell), with F_a/F_b from
`scf.UHF(mol).get_fock(dm=ROHF (D_a, D_b))` (cross-checked against PySCF's own
ROHF `focka`/`fockb`), then wraps them in a `scf.UHF` object (mo_coeff,
mo_energy, mo_occ set; converged) and runs:

  * U-RPA: `pyscf.gw.urpa.URPA` with the recipe of gen_urpa.py (df.DF with
    ferric's aux JSON, Gauss-Legendre nw = 40, x0 = 0.5, full rank, no frozen
    core); cross-checked by the numpy energy on PySCF's per-spin Cholesky ovL.
  * U-G0W0: `pyscf.gw.ugw_ac.UGWAC` with the recipe of gen_gw.py (nw = 100,
    no iw cutoff, 18 Pade nodes, PER-SPIN mid-gap ef, refined QP roots). For
    this UHF-shaped object PySCF's vk - v_mf is identically 0 (v_mf = -K[D_s]
    of the same density), matching ferric's HF-reference QP equation; the
    residual is recorded (`static_shift_max_abs`).
  * U-MP2 WITHOUT singles: numpy on the same per-spin Cholesky ovL tensors,
        E = sum_s 1/2 sum_ijab (ia|jb)[(ia|jb) - (ib|ja)] / D  +  sum (ia|JB)^2 / D,
    i.e. the doubles-only UMP2 expression on semi-canonical ROHF orbitals --
    what ferric's u_ri_mp2 computes. Cross-checked against PySCF DFUMP2 on the
    same object and DF. This is NOT ROMP2: ROHF is not a UHF stationary point,
    so f_ia^s != 0 and ROMP2 adds sum_s sum_ia |f_ia^s|^2 / (e_i - e_a); that
    singles term is recorded as a diagnostic (`romp2_singles`), not compared.

CONTROL ("legacy"): the pre-fix ferric treatment -- ROHF MOs and PySCF's
ROHF mo_energy (eigenvalues of the Roothaan EFFECTIVE Fock) for BOTH spins --
run through the same three recipes. The Rust test asserts ferric MISSES it.

Run (light; a few seconds per system):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        .venv/bin/python scripts/validation/gen_rohf_semicanonical.py [system ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_gw  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "rohf_semicanonical"
BASIS = "cc-pvdz"
AUX = "cc-pvdz-ri"
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-8
NUMPY_VS_PYSCF_MAX = 1e-10
# Semi-canonical invariants checked before anything is written.
BLOCK_OFFDIAG_MAX = 1e-10
DENSITY_MAX = 1e-10

SYSTEMS = {  # system -> (xyz, multiplicity)
    "oh": ("oh.xyz", 2),
    "ch3": ("ch3.xyz", 2),
}


def _np():
    import numpy as np

    return np


def semicanonicalize(mol, mf):
    """Per-spin semi-canonical (C_s, e_s) from a converged PySCF ROHF."""
    np = _np()
    from pyscf import scf

    dm = np.asarray(mf.make_rdm1())
    assert dm.shape[0] == 2, "ROHF make_rdm1 should return (D_a, D_b)"
    fa, fb = scf.UHF(mol).get_fock(dm=dm)
    # PySCF's own ROHF spin Focks (attached to the Roothaan Fock) must agree.
    froot = mf.get_fock(dm=dm)
    d_pyscf = max(
        float(np.max(np.abs(fa - froot.focka))), float(np.max(np.abs(fb - froot.fockb)))
    )
    assert d_pyscf < 1e-10, f"UHF get_fock vs ROHF focka/fockb differ by {d_pyscf:.2e}"
    c = mf.mo_coeff
    na, nb = mol.nelec
    out = []
    diag = {"uhf_fock_vs_rohf_focka_fockb": d_pyscf}
    for s, (f, nocc, d_ref) in enumerate(((fa, na, dm[0]), (fb, nb, dm[1]))):
        fmo = c.T @ f @ c
        nmo = c.shape[1]
        cs = np.zeros_like(c)
        es = np.zeros(nmo)
        for lo, hi in ((0, nocc), (nocc, nmo)):
            if hi == lo:
                continue
            blk = 0.5 * (fmo[lo:hi, lo:hi] + fmo[lo:hi, lo:hi].T)
            w, u = np.linalg.eigh(blk)
            cs[:, lo:hi] = c[:, lo:hi] @ u
            es[lo:hi] = w
        fnew = cs.T @ f @ cs
        occ = np.arange(nmo) < nocc
        same = occ[:, None] == occ[None, :]
        off = np.where(same & ~np.eye(nmo, dtype=bool), np.abs(fnew), 0.0).max()
        assert off < BLOCK_OFFDIAG_MAX, f"spin {s}: in-block off-diagonal {off:.2e}"
        assert np.max(np.abs(np.diag(fnew) - es)) < BLOCK_OFFDIAG_MAX
        dnew = cs[:, :nocc] @ cs[:, :nocc].T
        dd = float(np.max(np.abs(dnew - d_ref)))
        assert dd < DENSITY_MAX, f"spin {s}: occupied span changed ({dd:.2e})"
        fov = fnew[:nocc, nocc:]
        out.append((cs, es, fov))
        diag[f"spin{s}_max_abs_f_ov"] = float(np.max(np.abs(fov))) if fov.size else 0.0
    return out, diag


def uhf_object(mol, mf_rohf, c_a, c_b, e_a, e_b):
    """A converged-looking scf.UHF carrying the given per-spin orbitals."""
    np = _np()
    from pyscf import scf

    na, nb = mol.nelec
    nmo = c_a.shape[1]
    umf = scf.UHF(mol)
    umf.verbose = 0
    umf.mo_coeff = np.array([c_a, c_b])
    umf.mo_energy = np.array([e_a, e_b])
    umf.mo_occ = np.array(
        [(np.arange(nmo) < na).astype(float), (np.arange(nmo) < nb).astype(float)]
    )
    umf.converged = True
    umf.e_tot = umf.energy_tot()
    d = abs(umf.e_tot - mf_rohf.e_tot)
    assert d < 1e-9, f"UHF-shaped object energy differs from ROHF by {d:.2e}"
    return umf


def ump2_numpy(channels, nocc, nvir):
    """Doubles-only UMP2 from per-spin (L (nov, naux), D = e_a - e_i) channels."""
    np = _np()
    ovs = []
    for s in (0, 1):
        L, d, _f = channels[s]
        ovs.append((L.reshape(nocc[s], nvir[s], -1), d.reshape(nocc[s], nvir[s])))
    e_ss = []
    for s in (0, 1):
        B, d = ovs[s]
        g = np.einsum("iaP,jbP->iajb", B, B)
        den = d[:, :, None, None] + d[None, None, :, :]
        e_ss.append(float(0.5 * np.sum(g * (g - g.transpose(0, 3, 2, 1)) / -den)))
    (Ba, da), (Bb, db) = ovs
    g = np.einsum("iaP,jbP->iajb", Ba, Bb)
    den = da[:, :, None, None] + db[None, None, :, :]
    e_os = float(np.sum(g * g / -den))
    return {
        "e_aa": e_ss[0],
        "e_bb": e_ss[1],
        "e_ab": e_os,
        "e_corr": e_ss[0] + e_ss[1] + e_os,
    }


def check_channel_order(channels, umf, nocc):
    """ovL rows are i-major (i * nvir + a): D must equal e_a - e_i in that order."""
    np = _np()
    for s in (0, 1):
        e = umf.mo_energy[s]
        want = (e[None, nocc[s] :] - e[: nocc[s], None]).ravel()
        assert np.max(np.abs(channels[s][1] - want)) < 1e-12, (
            "ovL ordering is not i-major"
        )


def dfump2_pyscf(umf, with_df):
    from pyscf.mp import dfump2

    pt = dfump2.DFUMP2(umf, mo_energy=umf.mo_energy)
    pt.with_df = with_df
    pt.verbose = 0
    pt.kernel(with_t2=False)
    return float(pt.e_corr)


def correlated_block(mol, umf, aux_dict, with_df, orbs, symbols):
    """U-RPA, doubles-only U-MP2 and U-G0W0 on one UHF-shaped object."""
    na, nb = mol.nelec
    nmo = umf.mo_coeff[0].shape[1]
    nocc, nvir = (na, nb), (nmo - na, nmo - nb)

    rpa, eris, e_rpa, _ = gen_urpa.run_pyscf_rpa(umf, with_df, True, gen_urpa.NW)
    channels = gen_urpa.spin_channels(rpa, eris, True)
    check_channel_order(channels, umf, nocc)
    e_rpa_np, _ = gen_urpa.numpy_urpa(channels, gen_urpa.NW)
    d = abs(e_rpa_np - e_rpa)
    if d > NUMPY_VS_PYSCF_MAX:
        raise RuntimeError(
            f"numpy URPA {e_rpa_np:.12f} vs PySCF {e_rpa:.12f} ({d:.2e})"
        )
    e_rpa_20 = gen_urpa.run_pyscf_rpa(umf, with_df, True, 20)[2]

    mp2 = ump2_numpy(channels, nocc, nvir)
    e_df = dfump2_pyscf(umf, with_df)
    dmp2 = abs(e_df - mp2["e_corr"])
    if dmp2 > NUMPY_VS_PYSCF_MAX:
        raise RuntimeError(
            f"numpy UMP2 {mp2['e_corr']:.12f} vs DFUMP2 {e_df:.12f} ({dmp2:.2e})"
        )
    mp2["e_corr_pyscf_dfump2"] = e_df
    mp2["numpy_vs_pyscf_abs_diff"] = dmp2

    ea, eb = umf.mo_energy
    ef_a = 0.5 * (ea[na - 1] + ea[na])
    ef_b = 0.5 * (eb[nb - 1] + eb[nb])
    gw_a = gen_gw.run_ugwac(umf, aux_dict, orbs, ef_override=float(ef_a))
    gw_b = gen_gw.run_ugwac(umf, aux_dict, orbs, ef_override=float(ef_b))
    naux = gen_gw.check_naux(gw_a.with_df, AUX, symbols)
    gw = {
        "orbs": orbs,
        "nw": gen_gw.NW,
        "ef_convention": "per-spin mid-gap, as ferric u_sigma.rs",
        "alpha": gen_gw.u_spin_block(umf, gw_a, 0, orbs),
        "beta": gen_gw.u_spin_block(umf, gw_b, 1, orbs),
    }
    return {
        "eps_alpha": [float(x) for x in ea],
        "eps_beta": [float(x) for x in eb],
        "urpa": {
            "method": "pyscf.gw.urpa.URPA",
            "nw": gen_urpa.NW,
            "x0": gen_urpa.X0,
            "e_corr": e_rpa,
            "e_corr_numpy": e_rpa_np,
            "numpy_vs_pyscf_abs_diff": d,
            "e_corr_nw20": e_rpa_20,
        },
        "ump2_doubles": mp2,
        "u_g0w0": gw,
        "naux": int(naux),
    }


def gen(system):
    np = _np()
    xyz_rel, mult = SYSTEMS[system]
    xyz = common.MOL_DIR / xyz_rel
    t0 = time.perf_counter()
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, multiplicity=mult)
    ll = common.check_basis_like_for_like(mol, BASIS, symbols)
    aux = gen_urpa.aux_dict(AUX, symbols)
    rohf, mf = common.run_open_shell(
        mol, "rohf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
    )
    with_df = gen_urpa.make_df(mol, aux)
    naux = gen_urpa.check_naux(with_df, AUX, symbols)
    na, nb = mol.nelec
    nmo = mf.mo_coeff.shape[1]
    orbs = gen_gw.window(max(na, nb), nmo)

    sc, sc_diag = semicanonicalize(mol, mf)
    (ca, ea, fov_a), (cb, eb, fov_b) = sc
    semi = uhf_object(mol, mf, ca, cb, ea, eb)
    # ROMP2 singles on the semi-canonical orbitals (diagnostic, NOT compared).
    singles = 0.0
    for fov, e, nocc in ((fov_a, ea, na), (fov_b, eb, nb)):
        den = e[:nocc, None] - e[None, nocc:]
        singles += float(np.sum(fov**2 / den))

    e_eff = np.asarray(mf.mo_energy)
    legacy = uhf_object(mol, mf, mf.mo_coeff, mf.mo_coeff, e_eff, e_eff)

    payload = gen_urpa.header(mol, BASIS, AUX, naux, ll)
    payload["rohf"] = rohf
    payload["rohf_orbital_energies"] = {
        "effective_fock": [float(x) for x in e_eff],
        "pyscf_mo_ea": [float(x) for x in mf.mo_energy.mo_ea],
        "pyscf_mo_eb": [float(x) for x in mf.mo_energy.mo_eb],
        "note": "effective_fock = PySCF ROHF mo_energy (Roothaan effective Fock); "
        "mo_ea/mo_eb = diag of F_a/F_b in the ROHF MOs (NOT semi-canonical)",
    }
    payload["semicanonical"] = correlated_block(mol, semi, aux, with_df, orbs, symbols)
    payload["semicanonical"]["checks"] = sc_diag
    payload["semicanonical"]["romp2_singles"] = singles
    payload["semicanonical"]["romp2_total"] = (
        payload["semicanonical"]["ump2_doubles"]["e_corr"] + singles
    )
    payload["legacy_effective_fock"] = correlated_block(
        mol, legacy, aux, with_df, orbs, symbols
    )
    payload["legacy_effective_fock"]["note"] = (
        "CONTROL: ROHF MOs + effective-Fock eigenvalues for BOTH spins (the pre-fix ferric "
        "treatment); ferric must MISS these"
    )
    payload["generator_seconds"] = time.perf_counter() - t0
    import numpy
    import pyscf
    import scipy

    payload["provenance"] = common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "reference": "scf.ROHF (exact 4-index ERIs), semi-canonicalized in numpy "
            "(F_s = UHF get_fock of the ROHF (D_a, D_b); occ/vir block eigh per spin)",
            "rpa": "pyscf.gw.urpa.URPA, nw 40, x0 0.5 (gen_urpa.py recipe)",
            "gw": "pyscf.gw.ugw_ac.UGWAC, nw 100, ac_iw_cutoff None, 18 Pade nodes, "
            "per-spin ef (gen_gw.py recipe)",
            "mp2": "doubles-only UMP2 in numpy on URPA's Cholesky ovL; cross-checked vs DFUMP2",
            "df": "df.DF(mol, auxbasis=ferric aux JSON), Cholesky-whitened metric",
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        basis_name=BASIS,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=AUX,
        frozen_core=None,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability={"rohf": rohf["stability"]},
        generator="scripts/validation/gen_rohf_semicanonical.py",
        extra={"blocks": ["rohf", "semicanonical", "legacy_effective_fock"]},
    )
    path = common.write_reference(ROW, system, BASIS, payload)
    s, lg = payload["semicanonical"], payload["legacy_effective_fock"]
    ha = 27.211386245988
    i_a, i_b = orbs.index(na - 1), orbs.index(nb - 1)
    print(
        f"{system}/{BASIS}: E_ROHF {rohf['energy']:.10f}\n"
        f"  U-RPA E_c   semi {s['urpa']['e_corr']:.10f}  legacy {lg['urpa']['e_corr']:.10f}"
        f"  (numpy |d| {s['urpa']['numpy_vs_pyscf_abs_diff']:.1e})\n"
        f"  U-MP2 E_c   semi {s['ump2_doubles']['e_corr']:.10f}  legacy "
        f"{lg['ump2_doubles']['e_corr']:.10f}  (DFUMP2 |d| "
        f"{s['ump2_doubles']['numpy_vs_pyscf_abs_diff']:.1e}; ROMP2 singles "
        f"{singles:.10f})\n"
        f"  U-G0W0 a-HOMO semi {s['u_g0w0']['alpha']['eps_qp'][i_a] * ha:.5f} legacy "
        f"{lg['u_g0w0']['alpha']['eps_qp'][i_a] * ha:.5f} eV; b-HOMO semi "
        f"{s['u_g0w0']['beta']['eps_qp'][i_b] * ha:.5f} legacy "
        f"{lg['u_g0w0']['beta']['eps_qp'][i_b] * ha:.5f} eV\n"
        f"  max|f_ov| a {sc_diag['spin0_max_abs_f_ov']:.3e} b {sc_diag['spin1_max_abs_f_ov']:.3e}"
        f"  ({payload['generator_seconds']:.1f} s) -> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = set(argv) or set(SYSTEMS)
    unknown = want - set(SYSTEMS)
    if unknown:
        raise SystemExit(
            f"unknown systems: {sorted(unknown)} (known: {sorted(SYSTEMS)})"
        )
    for s in SYSTEMS:
        if s in want:
            gen(s)


if __name__ == "__main__":
    main(sys.argv[1:])
