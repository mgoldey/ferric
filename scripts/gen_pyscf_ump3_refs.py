"""PySCF references for the spin-blocked unrestricted MP3 driver (ferric_mp2::u_mp3).

Consumer: crates/ferric-mp2/tests/u_mp3.rs.
Output:   testdata/reference/ump3/<system>_<ref>_cc-pvdz.json

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/u_mp3.rs)
------------------------------------------------------------
Doubles-only MP2 and MP3 on a UHF reference, or on an ROHF reference
semicanonicalized on the UHF Fock matrices of the ROHF density (each spin's
occ-occ and vir-vir blocks diagonalized separately; the f_ov singles terms
are NOT included). Every electron is correlated. Integrals are exact
(dense four-centre) or RI (Coulomb metric, cc-pvdz-ri).

REFERENCE (independent code)
----------------------------
* SCF: PySCF `scf.RHF` / `common.run_open_shell` (UHF or ROHF, three guesses
  + stability loop), exact integrals, ferric's basis JSON and Bohr geometry.
* ROHF semicanonicalization: done here in numpy (UHF Fock of the ROHF density,
  oo and vv blocks diagonalized per spin). The semicanonical orbitals are put
  on a plain `scf.UHF` object, so the downstream PySCF codes see a UHF
  determinant with non-zero f_ov that they ignore (doubles only), which is the
  quantity ferric computes.
* E2: `mp.UMP2` (exact) or `mp.dfump2.DFUMP2` (RI). The aa / bb / ab split is
  computed here in numpy from `ao2mo` (exact) or int3c2e/int2c2e (RI) and
  anchored to PySCF's e_corr_ss / e_corr_os at TOL_ANCHOR.
* E3: `adc.UADC` (or `adc.RADC` for RHF) ground-state MP3 correlation,
  method "adc(3)", minus its own "adc(2)" (MP2) ground-state energy; the ADC
  MP2 is anchored to UMP2 at TOL_ANCHOR so a mismatch in orbitals or
  denominators would refuse to write. The RI blocks use the same ADC run with
  `density_fit(auxbasis)`; whether PySCF's DF-ADC fits every block is checked by
  its own MP2 agreeing with DFUMP2 at TOL_ANCHOR.

SYSTEMS (cc-pVDZ): H2O (RHF), OH 2Pi (UHF, ROHF), CH2 3B1 (UHF, ROHF).

Run (light; seconds per system):
    box-lease exec -- uv run --no-sync python scripts/gen_pyscf_ump3_refs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "validation"))

import common  # noqa: E402

BASIS = "cc-pvdz"
AUX = "cc-pvdz-ri"
OUT_DIR = common.ROOT / "testdata" / "reference" / "ump3"
SYSTEMS = {  # system -> (charge, multiplicity, references)
    "h2o": (0, 1, ("rhf",)),
    "oh": (0, 2, ("uhf", "rohf")),
    "ch2_triplet": (0, 3, ("uhf", "rohf")),
}
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-9
TOL_ANCHOR = 1e-10


def _auxmol(mol, symbols):
    from pyscf import df

    aux_bas, aux_cart = common.pyscf_basis(AUX, symbols)
    assert not aux_cart, f"{AUX}: Cartesian aux shells cannot be matched"
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    want = common.ferric_nao(AUX, symbols)
    assert auxmol.nao_nr() == want, f"{AUX}: naux {auxmol.nao_nr()} != ferric {want}"
    return auxmol


def semicanonical_uhf(mol, rohf):
    """Plain UHF object carrying the ROHF determinant in semicanonical orbitals."""
    import numpy as np
    from pyscf import scf

    na, nb = mol.nelec
    c = rohf.mo_coeff
    dm = rohf.make_rdm1()
    veff = scf.uhf.get_veff(mol, dm)
    h1 = rohf.get_hcore()
    coeffs, energies = [], []
    for s, nocc in ((0, na), (1, nb)):
        f = c.T @ (h1 + veff[s]) @ c
        cs = np.zeros_like(c)
        es = np.zeros(c.shape[1])
        for lo, hi in ((0, nocc), (nocc, c.shape[1])):
            w, u = np.linalg.eigh(f[lo:hi, lo:hi])
            cs[:, lo:hi] = c[:, lo:hi] @ u
            es[lo:hi] = w
        coeffs.append(cs)
        energies.append(es)
    mfu = scf.UHF(mol)
    mfu.verbose = 0
    mfu.mo_coeff = np.array(coeffs)
    mfu.mo_energy = np.array(energies)
    occ = np.zeros((2, c.shape[1]))
    occ[0, :na] = 1.0
    occ[1, :nb] = 1.0
    mfu.mo_occ = occ
    mfu.e_tot = float(rohf.e_tot)
    mfu.converged = True
    return mfu


def numpy_split(mo_coeff, mo_energy, nelec, eri_mo):
    """aa / bb / ab MP2 from MO integrals: eri_mo(cs1, cs2) -> (ia|jb) [i,a,j,b]."""
    import numpy as np

    co = [mo_coeff[s][:, : nelec[s]] for s in (0, 1)]
    cv = [mo_coeff[s][:, nelec[s] :] for s in (0, 1)]
    eo = [mo_energy[s][: nelec[s]] for s in (0, 1)]
    ev = [mo_energy[s][nelec[s] :] for s in (0, 1)]

    def denom(s, t):
        return (
            eo[s][:, None, None, None]
            - ev[s][None, :, None, None]
            + eo[t][None, None, :, None]
            - ev[t][None, None, None, :]
        )

    out = []
    for s in (0, 1):
        g = eri_mo((co[s], cv[s]), (co[s], cv[s]))
        k = g - g.transpose(0, 3, 2, 1)
        out.append(0.25 * float(np.sum(k * k / denom(s, s))))
    g = eri_mo((co[0], cv[0]), (co[1], cv[1]))
    out.append(float(np.sum(g * g / denom(0, 1))))
    return out


def exact_eri_mo(mol):
    from pyscf import ao2mo

    def f(p, q):
        g = ao2mo.general(mol, (p[0], p[1], q[0], q[1]), compact=False)
        return g.reshape(p[0].shape[1], p[1].shape[1], q[0].shape[1], q[1].shape[1])

    return f


def ri_eri_mo(mol, auxmol):
    import numpy as np
    import scipy.linalg as sla
    from pyscf import df

    n = mol.nao_nr()
    int3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    naux = auxmol.nao_nr()
    low = np.linalg.cholesky(auxmol.intor("int2c2e"))
    b_ao = sla.solve_triangular(low, int3.reshape(n * n, naux).T, lower=True)
    b_ao = b_ao.reshape(naux, n, n)

    def f(p, q):
        bp = np.einsum("Pmn,mi,na->Pia", b_ao, p[0], p[1], optimize=True)
        bq = np.einsum("Pmn,mi,na->Pia", b_ao, q[0], q[1], optimize=True)
        return np.einsum("Pia,Pjb->iajb", bp, bq, optimize=True)

    return f


def as_unrestricted(mf):
    import numpy as np

    if mf.mo_coeff.ndim == 3:
        return mf.mo_coeff, mf.mo_energy
    return np.array([mf.mo_coeff] * 2), np.array([mf.mo_energy] * 2)


def adc_mp(mf, restricted, auxmol=None):
    """(E2, E3) from PySCF ADC ground-state energies at adc(2) and adc(3)."""
    from pyscf import adc

    out = []
    for method in ("adc(2)", "adc(3)"):
        a = adc.RADC(mf) if restricted else adc.UADC(mf)
        if auxmol is not None:
            a = a.density_fit(auxmol.basis)
            a.with_df.auxmol = auxmol
        a.method = method
        a.verbose = 0
        out.append(float(a.kernel_gs()[0]))
    return out[0], out[1] - out[0]


def mp2_pyscf(mf, restricted, auxmol=None):
    from pyscf import df, mp
    from pyscf.mp import dfump2

    if auxmol is None:
        m = mp.MP2(mf) if restricted else mp.UMP2(mf)
        m.verbose = 0
        m.kernel()
        return float(m.e_corr), float(m.e_corr_ss), float(m.e_corr_os)
    if restricted:
        from pyscf.mp import dfmp2

        m = dfmp2.DFMP2(mf)
    else:
        m = dfump2.DFUMP2(mf)
    m.with_df = df.DF(mf.mol)
    m.with_df.auxmol = auxmol
    m.verbose = 0
    m.kernel()
    return float(m.e_corr), float(m.e_corr_ss), float(m.e_corr_os)


def block(mf, mol, restricted, integrals, auxmol):
    mo_c, mo_e = as_unrestricted(mf)
    eri = exact_eri_mo(mol) if integrals == "exact" else ri_eri_mo(mol, auxmol)
    use_aux = auxmol if integrals == "ri" else None
    e_aa, e_bb, e_ab = numpy_split(mo_c, mo_e, mol.nelec, eri)
    e2, e2_ss, e2_os = mp2_pyscf(mf, restricted, use_aux)
    d_ss = e_aa + e_bb - e2_ss
    d_os = e_ab - e2_os
    assert abs(d_ss) < TOL_ANCHOR, (
        f"{integrals}: numpy aa+bb vs PySCF e_corr_ss {d_ss:.2e}"
    )
    assert abs(d_os) < TOL_ANCHOR, (
        f"{integrals}: numpy ab vs PySCF e_corr_os {d_os:.2e}"
    )
    adc_e2, adc_e3 = adc_mp(mf, restricted, use_aux)
    d_adc = adc_e2 - e2
    assert abs(d_adc) < TOL_ANCHOR, f"{integrals}: ADC(2) E2 vs MP2 {d_adc:.2e}"
    return {
        "e2": e2,
        "e2_aa": e_aa,
        "e2_bb": e_bb,
        "e2_ab": e_ab,
        "e3": adc_e3,
        "anchor_numpy_ss_diff": d_ss,
        "anchor_numpy_os_diff": d_os,
        "anchor_adc_mp2_diff": d_adc,
    }


def scf_reference(mol, ref):
    from pyscf import scf

    if ref == "rhf":
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.verbose = CONV_TOL, CONV_TOL_GRAD, 0
        mf.kernel()
        assert mf.converged, "RHF did not converge"
        return {"energy": float(mf.e_tot)}, mf
    info, mf = common.run_open_shell(
        mol, ref, conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
    )
    keep = ("energy", "s_squared", "stability", "multiple_stable_minima")
    info = {k: info[k] for k in keep}
    if ref == "rohf":
        mf = semicanonical_uhf(mol, mf)
    return info, mf


def gen(system: str, ref: str) -> Path:
    import numpy
    import pyscf

    charge, mult, _ = SYSTEMS[system]
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, charge=charge, multiplicity=mult)
    ll = common.check_basis_like_for_like(mol, BASIS, symbols)
    auxmol = _auxmol(mol, symbols)
    info, mf = scf_reference(mol, ref)
    restricted = ref == "rhf"
    payload = {
        "system": system,
        "reference": ref,
        "basis": BASIS,
        "aux_basis": AUX,
        "charge": charge,
        "multiplicity": mult,
        "nao": int(mol.nao_nr()),
        "naux": int(auxmol.nao_nr()),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "frozen_core": 0,
        "scf": info,
        "exact": block(mf, mol, restricted, "exact", auxmol),
        "ri": block(mf, mol, restricted, "ri", auxmol),
        "provenance": {
            "generator": "scripts/gen_pyscf_ump3_refs.py",
            "git_head": common.git_head(),
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "xyz": str(xyz.relative_to(common.ROOT)),
            "coords_bohr": coords,
            "symbols": symbols,
            "basis_json_sha256": common.sha256_file(common.basis_json_path(BASIS)),
            "aux_json_sha256": common.sha256_file(common.basis_json_path(AUX)),
            "scf_conv": {"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
            "e2": "mp.MP2/UMP2 (exact), DFMP2/DFUMP2 (ri); split from numpy",
            "e3": "adc.RADC/UADC kernel_gs: adc(3) minus adc(2)",
            "rohf": "numpy semicanonicalization on the UHF Fock of the ROHF density",
            "basis_self_check": ll,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{system}_{ref}_{BASIS}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    for kind in ("exact", "ri"):
        b = payload[kind]
        print(
            f"{system:11s} {ref:4s} {kind:5s} E_scf {info['energy']:.10f} E2 {b['e2']:.10f} "
            f"(aa {b['e2_aa']:.8f} bb {b['e2_bb']:.8f} ab {b['e2_ab']:.8f}) E3 {b['e3']:.10f}",
            flush=True,
        )
    return path


def main(argv) -> int:
    want = argv or list(SYSTEMS)
    unknown = set(want) - set(SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    n = 0
    for system, (_, _, refs) in SYSTEMS.items():
        if system in want:
            for ref in refs:
                gen(system, ref)
                n += 1
    print(f"GEN_UMP3_DONE written={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
