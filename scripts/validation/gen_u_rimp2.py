"""PySCF references for the VALIDATION.md "Unrestricted RI-MP2" row.

Consumer: crates/ferric-mp2/tests/validation_u_rimp2.rs.
Output:   testdata/reference/validation/u_rimp2/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/u_rimp2.rs and the CLI,
not assumed)
-------------------------------------------------------------------------
`method.kind = "rimp2"` on a molecule with multiplicity > 1 solves a UHF
reference (crates/ferric-cli/src/lib.rs `solve_open_shell_reference`: plain
`solve_uhf` with the `[scf]` keys, EXACT four-centre J/K -- the CLI's
`scf_xc_and_aux_defaults` returns no SCF aux for rimp2) and dispatches to
`run_u_rimp2` -> `ferric_mp2::u_rimp2::u_ri_mp2` with
`RiMp2Config { frozen_core: [mp2] frozen_core (default 0), .. }`. The
correlation energy is

    E_aa = 1/4 sum_{ij,ab} |(ia|jb) - (ib|ja)|^2 / D     (alpha occ/vir)
    E_bb = same with beta
    E_ab =     sum_{iJ,aB}  (ia|JB)^2 / D                  (no exchange)

with (ia|jb) = sum_P B^P_ia B^P_jb, B = (P|Q)^{-1/2} (P|ia) in the Coulomb
metric of the correlation aux (`[mp2] auxbasis`, default cc-pvdz-ri for
cc-pVDZ), canonical UHF orbital energies, and `frozen_core` = the number of
lowest SPATIAL orbitals dropped from EACH spin's occupied space
(`rimp2::active_occ(nocc_sigma, frozen_core)`).

REFERENCE
---------
* UHF: PySCF `scf.UHF`, exact integrals, ferric's basis JSON and Bohr
  geometry, converged from three guesses and followed to an internally
  STABLE state (`common.run_open_shell`); an unstable state is refused.
* E_corr: PySCF `mp.dfump2.DFUMP2` on that UHF with `with_df.auxmol` built
  from ferric's aux JSON (AO count checked against ferric's parser), and
  `frozen = n` for the frozen-core block (PySCF freezes the n lowest orbitals
  of each spin -- ferric's convention).
* Spin components: DFUMP2 returns only e_corr_ss (aa + bb) and e_corr_os.
  The aa / bb split is computed here INDEPENDENTLY in numpy from PySCF's
  int3c2e / int2c2e with a Cholesky-metric B (no DFUMP2 code), and anchored:
  numpy aa + bb == DFUMP2 e_corr_ss and numpy ab == DFUMP2 e_corr_os
  (asserted to 1e-11 Ha, recorded as `numpy_anchor_*`).

SYSTEMS (cc-pVDZ / cc-pvdz-ri): OH 2Pi, CH3 2A2'', NH2 2B1, O2 3Sg-, HO2 2A''
all electrons correlated; HO2 also with frozen core 2 (O, O 1s) in block
`ump2_fc2`.

Run (light; seconds per system):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_u_rimp2.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "u_rimp2"
ROW_NAME = "Unrestricted RI-MP2"
BASIS = "cc-pvdz"
AUX = "cc-pvdz-ri"
SYSTEMS = {  # system -> (charge, multiplicity, frozen-core blocks beyond 0)
    "oh": (0, 2, ()),
    "ch3": (0, 2, ()),
    "nh2": (0, 2, ()),
    "o2": (0, 3, ()),
    "ho2": (0, 2, (2,)),
}
TOL_ANCHOR = 1e-11


def _auxmol(mol, symbols):
    from pyscf import df

    aux_bas, aux_cart = common.pyscf_basis(AUX, symbols)
    assert not aux_cart, f"{AUX}: Cartesian aux shells cannot be matched"
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    want = common.ferric_nao(AUX, symbols)
    assert auxmol.nao_nr() == want, (
        f"{AUX}: PySCF naux {auxmol.nao_nr()} != ferric {want}"
    )
    return auxmol


def dfump2(mf, auxmol, frozen):
    from pyscf import df
    from pyscf.mp import dfump2 as m

    mp = m.DFUMP2(mf, frozen=frozen or None)
    mp.with_df = df.DF(mf.mol)
    mp.with_df.auxmol = auxmol
    mp.verbose = 0
    mp.kernel()
    return float(mp.e_corr), float(mp.e_corr_ss), float(mp.e_corr_os)


def numpy_components(mf, auxmol, frozen):
    """Independent aa / bb / ab from PySCF int3c2e + Cholesky-metric B."""
    import numpy as np
    import scipy.linalg as sla
    from pyscf import df

    mol = mf.mol
    n = mol.nao_nr()
    int3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    naux = auxmol.nao_nr()
    low = np.linalg.cholesky(auxmol.intor("int2c2e"))
    b_ao = sla.solve_triangular(low, int3.reshape(n * n, naux).T, lower=True)
    b_ao = b_ao.reshape(naux, n, n)
    na, nb = mol.nelec
    bov, eo, ev = [], [], []
    for s, nocc in ((0, na), (1, nb)):
        c = mf.mo_coeff[s]
        e = mf.mo_energy[s]
        co, cv = c[:, frozen:nocc], c[:, nocc:]
        bov.append(np.einsum("Pmn,mi,na->Pia", b_ao, co, cv, optimize=True))
        eo.append(e[frozen:nocc])
        ev.append(e[nocc:])

    def denom(eo1, ev1, eo2, ev2):
        return (
            eo1[:, None, None, None]
            - ev1[None, :, None, None]
            + eo2[None, None, :, None]
            - ev2[None, None, None, :]
        )

    def same(s):
        g = np.einsum("Pia,Pjb->iajb", bov[s], bov[s], optimize=True)
        k = g - g.transpose(0, 3, 2, 1)
        return 0.25 * float(np.sum(k * k / denom(eo[s], ev[s], eo[s], ev[s])))

    g = np.einsum("Pia,PJB->iaJB", bov[0], bov[1], optimize=True)
    e_ab = float(np.sum(g * g / denom(eo[0], ev[0], eo[1], ev[1])))
    return same(0), same(1), e_ab, int(naux)


def block(mf, auxmol, frozen) -> dict:
    e_corr, e_ss, e_os = dfump2(mf, auxmol, frozen)
    e_aa, e_bb, e_ab, naux = numpy_components(mf, auxmol, frozen)
    d_ss = (e_aa + e_bb) - e_ss
    d_os = e_ab - e_os
    assert abs(d_ss) < TOL_ANCHOR, f"numpy aa+bb vs DFUMP2 e_corr_ss: {d_ss:.2e}"
    assert abs(d_os) < TOL_ANCHOR, f"numpy ab vs DFUMP2 e_corr_os: {d_os:.2e}"
    return {
        "frozen_core": int(frozen),
        "e_corr": e_corr,
        "e_corr_ss": e_ss,
        "e_corr_os": e_os,
        "e_aa": e_aa,
        "e_bb": e_bb,
        "e_ab": e_ab,
        "e_total": float(mf.e_tot) + e_corr,
        "numpy_anchor_ss_diff": float(d_ss),
        "numpy_anchor_os_diff": float(d_os),
        "naux": naux,
    }


def gen(system) -> Path:
    import numpy
    import pyscf

    charge, mult, fc_blocks = SYSTEMS[system]
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, charge=charge, multiplicity=mult)
    ll = common.check_basis_like_for_like(mol, BASIS, symbols)
    auxmol = _auxmol(mol, symbols)
    uhf, mf = common.run_open_shell(
        mol, "uhf", conv_tol=1e-11, conv_tol_grad=1e-8, return_mf=True
    )
    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": BASIS,
        "aux_basis": AUX,
        "charge": charge,
        "multiplicity": mult,
        "nao": int(mol.nao_nr()),
        "naux": int(auxmol.nao_nr()),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "uhf": uhf,
        "ump2": block(mf, auxmol, 0),
    }
    for fc in fc_blocks:
        payload[f"ump2_fc{fc}"] = block(mf, auxmol, fc)
    payload["provenance"] = common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "scf": "scf.UHF exact integrals, 3 guesses + stability loop (common.run_open_shell)",
            "mp2": "pyscf.mp.dfump2.DFUMP2, with_df.auxmol from ferric's aux JSON",
            "spin_components": "numpy int3c2e/int2c2e Cholesky-metric B, anchored to DFUMP2",
            "numpy": numpy.__version__,
        },
        basis_name=BASIS,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux={
            "name": AUX,
            "json": str(common.basis_json_path(AUX).relative_to(common.ROOT)),
            "sha256": common.sha256_file(common.basis_json_path(AUX)),
            "metric": "Coulomb",
        },
        frozen_core={"ump2": 0, **{f"ump2_fc{fc}": fc for fc in fc_blocks}},
        scf_conv={"conv_tol": 1e-11, "conv_tol_grad": 1e-8},
        stability={"uhf": uhf["stability"]},
        generator="scripts/validation/gen_u_rimp2.py",
        extra={"basis_self_check": ll},
    )
    path = common.write_reference(ROW, system, BASIS, payload)
    b = payload["ump2"]
    print(
        f"{system:4s} E_UHF {uhf['energy']:.10f} <S2> {uhf['s_squared']:.5f} "
        f"E_corr {b['e_corr']:.10f} aa {b['e_aa']:.8f} bb {b['e_bb']:.8f} ab {b['e_ab']:.8f} "
        f"multi={uhf['multiple_stable_minima']}",
        flush=True,
    )
    for fc in fc_blocks:
        bf = payload[f"ump2_fc{fc}"]
        print(
            f"{system:4s} fc{fc} E_corr {bf['e_corr']:.10f} (fc effect {bf['e_corr'] - b['e_corr']:+.3e})"
        )
    return path


def main(argv) -> int:
    want = argv or list(SYSTEMS)
    unknown = set(want) - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    n = 0
    for s in SYSTEMS:
        if s in want:
            gen(s)
            n += 1
    print(f"GEN_U_RIMP2_DONE written={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
