"""PySCF/numpy references for the VALIDATION.md "Attenuated RI-MP2" row.

Consumer: crates/ferric-mp2/tests/validation_attenuated_mp2.rs.
Output:   testdata/reference/validation/attenuated_mp2/<system>_aug-cc-pvdz.json

What ferric computes (crates/ferric-mp2/src/attenuated.rs, unscreened path =
`ri_mp2_spin_components(.., Operator::erfc(omega), ..)` in rimp2.rs):

* Reference: canonical RHF with the FULL Coulomb operator (exact J/K).
* 3-center (P|mu nu) AND 2-center metric (P|Q) both with erfc(omega r)/r
  (`RiMp2Config::metric_op = None` -> the metric uses the physical kernel;
  standard, non-robust fit), inverse square root by Cholesky.
* (ia|jb) ~= sum_PQ (ia|P) (P|Q)^-1 (Q|jb), all erfc.
* E_os = sum_ijab (ia|jb)^2 / D,  E_ss = sum_ijab (ia|jb)[(ia|jb)-(ib|ja)] / D,
  D = e_i + e_j - e_a - e_b; no frozen core. omega is in BOHR^-1 in the
  library (the CLI/Python take Angstrom^-1 and convert).

This generator reproduces that in numpy on top of PySCF integrals, with the
erfc kernel set by `with_range_coulomb(-omega)` (PySCF: negative omega =
short-range erfc) on BOTH the orbital and the auxiliary Mole, so the int3c2e
and int2c2e builds are both attenuated. Before any energy is written it checks:

  1. kernel identity: erfc + erf == Coulomb for the int3c2e and int2c2e blocks
     (proves the sign convention picked erfc, and that both Moles carry it);
  2. omega = 0 numpy DF-MP2 == PySCF's own `dfmp2.DFMP2` with the same aux
     basis (proves the numpy assembly);
  3. records the exact 4-index erfc MP2 (no RI) as information, so the RI
     error of the attenuated fit is visible next to the RI reference.

omega grid (Bohr^-1): 0.2, 0.222234 (= 0.42 A^-1, ferric's default), 0.42, 1.0;
plus omega = 0 (plain Coulomb RI-MP2) for the exactness anchor.

Run (light step; well under a minute):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_attenuated_mp2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "attenuated_mp2"
ROW_NAME = "Attenuated RI-MP2"
BASIS = "aug-cc-pvdz"
AUXBASIS = "aug-cc-pvdz-rifit"
SYSTEMS = ("h2o", "nh3")
# ferric: BOHR_INV_PER_ANG_INV = 1/1.8897259886; default omega = 0.420 A^-1.
OMEGA_DEFAULT_BOHR = 0.420 * (1.0 / 1.8897259886)
OMEGAS = (0.2, OMEGA_DEFAULT_BOHR, 0.42, 1.0)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9


def ints(mol, auxmol, omega):
    """(nao,nao,naux) int3c2e and (naux,naux) int2c2e under erfc(omega r)/r
    (omega > 0), erf (omega < 0 in THIS helper's convention is not used), or
    Coulomb (omega == 0)."""
    from pyscf import df

    if omega == 0.0:
        return (
            df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1"),
            auxmol.intor("int2c2e"),
        )
    with mol.with_range_coulomb(-omega), auxmol.with_range_coulomb(-omega):
        v3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        v2 = auxmol.intor("int2c2e")
    return v3, v2


def ints_erf(mol, auxmol, omega):
    from pyscf import df

    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        v3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
        v2 = auxmol.intor("int2c2e")
    return v3, v2


def ri_mp2(mf, v3, v2):
    import numpy as np
    import scipy.linalg

    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    e = mf.mo_energy
    co, cv = c[:, :nocc], c[:, nocc:]
    iaP = np.einsum("mnP,mi,na->iaP", v3, co, cv, optimize=True)
    naux = v2.shape[0]
    L = scipy.linalg.cholesky(v2, lower=True)
    B = scipy.linalg.solve_triangular(L, iaP.reshape(-1, naux).T, lower=True)
    nvir = cv.shape[1]
    B = B.reshape(naux, nocc, nvir)
    eo, ev = e[:nocc], e[nocc:]
    e_os = 0.0
    e_ss = 0.0
    for i in range(nocc):
        for j in range(nocc):
            v = np.einsum("Pa,Pb->ab", B[:, i, :], B[:, j, :])
            d = eo[i] + eo[j] - ev[:, None] - ev[None, :]
            e_os += float(np.sum(v * v / d))
            e_ss += float(np.sum(v * (v - v.T) / d))
    return {"e_os": e_os, "e_ss": e_ss, "e_corr": e_os + e_ss}


def exact_mp2(mf, mol, omega):
    import numpy as np
    from pyscf import ao2mo

    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    c = mf.mo_coeff
    co, cv = c[:, :nocc], c[:, nocc:]
    with mol.with_range_coulomb(-omega):
        eri = ao2mo.general(mol, (co, cv, co, cv), compact=False)
    nvir = cv.shape[1]
    ovov = eri.reshape(nocc, nvir, nocc, nvir)
    e = mf.mo_energy
    d = (
        e[:nocc, None, None, None]
        - e[None, nocc:, None, None]
        + e[None, None, :nocc, None]
        - e[None, None, None, nocc:]
    )
    t = ovov / d
    return float(np.einsum("iajb,iajb->", t, 2 * ovov - ovov.transpose(0, 3, 2, 1)))


def main() -> int:
    import numpy as np
    import pyscf
    from pyscf import df, gto, scf
    from pyscf.mp import dfmp2

    only = set(sys.argv[1:])
    written = []
    for system in SYSTEMS:
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, BASIS)
        basis_check = common.check_basis_like_for_like(mol, BASIS, symbols)
        auxbasis, auxcart = common.pyscf_basis(AUXBASIS, symbols)
        assert not auxcart, "aux basis must be spherical for a like-for-like build"
        auxmol = gto.M(
            atom=common.pyscf_atom_bohr(symbols, coords),
            unit="Bohr",
            basis=auxbasis,
            cart=False,
            verbose=0,
        )
        naux_ferric = common.ferric_nao(AUXBASIS, symbols)
        assert auxmol.nao_nr() == naux_ferric, (auxmol.nao_nr(), naux_ferric)

        mf = scf.RHF(mol)
        mf.conv_tol = CONV_TOL
        mf.conv_tol_grad = CONV_TOL_GRAD
        mf.kernel()
        assert mf.converged

        # --- check 2: numpy assembly == PySCF DFMP2 at omega = 0 ------------
        v3c, v2c = ints(mol, auxmol, 0.0)
        coul = ri_mp2(mf, v3c, v2c)
        pt = dfmp2.DFMP2(mf)
        pt.with_df = df.DF(mol)
        pt.with_df.auxbasis = auxbasis
        pt.kernel()
        d_assembly = abs(coul["e_corr"] - float(pt.e_corr))
        assert d_assembly < 1e-10, f"numpy DF-MP2 vs PySCF DFMP2: {d_assembly:.2e}"

        blocks = []
        worst_identity = 0.0
        for omega in OMEGAS:
            v3, v2 = ints(mol, auxmol, omega)
            # --- check 1: erfc + erf == Coulomb, and erfc != Coulomb --------
            v3e, v2e = ints_erf(mol, auxmol, omega)
            ident = max(
                float(np.max(np.abs(v3 + v3e - v3c))),
                float(np.max(np.abs(v2 + v2e - v2c))),
            )
            worst_identity = max(worst_identity, ident)
            assert ident < 1e-10, f"erfc+erf != Coulomb ({ident:.2e}) at omega={omega}"
            assert np.max(np.abs(v2 - v2c)) > 1e-3, (
                "erfc metric equals Coulomb: attenuation not applied"
            )
            r = ri_mp2(mf, v3, v2)
            r_exact = exact_mp2(mf, mol, omega)
            blocks.append(
                {
                    "omega_bohr_inv": omega,
                    "omega_angstrom_inv": omega * 1.8897259886,
                    "e_corr": r["e_corr"],
                    "e_os": r["e_os"],
                    "e_ss": r["e_ss"],
                    "exact_4index_e_corr_info": r_exact,
                    "ri_error_info": r["e_corr"] - r_exact,
                }
            )
            print(
                f"{system:4s} omega={omega:.6f} E_corr(RI erfc)={r['e_corr']:.12f} "
                f"os={r['e_os']:.10f} ss={r['e_ss']:.10f} exact4={r_exact:.10f} "
                f"RIerr={r['e_corr'] - r_exact:+.2e}"
            )
        print(
            f"{system:4s} coulomb E_corr={coul['e_corr']:.12f} (PySCF DFMP2 |d|={d_assembly:.1e}) "
            f"E_RHF={mf.e_tot:.12f} identity_max={worst_identity:.1e}"
        )
        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": BASIS,
            "auxbasis": AUXBASIS,
            "charge": 0,
            "multiplicity": 1,
            "nao": mol.nao_nr(),
            "naux": auxmol.nao_nr(),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "rhf_energy": float(mf.e_tot),
            "coulomb": coul,
            "attenuated": blocks,
            "checks": {
                "numpy_vs_pyscf_dfmp2_omega0_abs": d_assembly,
                "erfc_plus_erf_minus_coulomb_max_abs": worst_identity,
            },
            "provenance": common.provenance(
                code="PySCF (integrals, RHF, DFMP2 anchor) + numpy (attenuated RI-MP2 assembly)",
                version=pyscf.__version__,
                keywords={
                    "rhf": "scf.RHF, exact Coulomb J/K",
                    "operator": "erfc(omega r)/r via Mole.with_range_coulomb(-omega) on mol AND auxmol",
                    "attenuated_integrals": "int3c2e (P|mu nu) and int2c2e (P|Q) both erfc",
                    "fit": "standard (non-robust), metric = same erfc kernel, Cholesky",
                    "energy": "E_os + E_ss, canonical denominators",
                    "numpy": np.__version__,
                },
                basis_name=BASIS,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid=None,
                aux=AUXBASIS,
                frozen_core=0,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=None,
                generator="scripts/validation/gen_attenuated_mp2.py",
                extra={
                    "basis_self_check": basis_check,
                    "aux_basis_json": str(
                        common.basis_json_path(AUXBASIS).relative_to(common.ROOT)
                    ),
                    "aux_basis_sha256": common.sha256_file(
                        common.basis_json_path(AUXBASIS)
                    ),
                },
            ),
        }
        written.append(common.write_reference(ROW, system, BASIS, payload))
    print(f"GEN_ATTENUATED_MP2_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
