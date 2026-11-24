"""PySCF references for the VALIDATION.md "Unrestricted CCSD / CCSD(T)" row.

Consumer: crates/ferric-cc/tests/validation_uccsd.rs.
Output:   testdata/reference/validation/uccsd/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-cc/src/u_ccsd.rs, u_ccsd_t.rs
and the CLI, not assumed)
-------------------------------------------------------------------------
`method.kind = "ccsd"` / `"ccsd(t)"` on a molecule with multiplicity > 1
solves a UHF reference with EXACT four-centre J/K (`solve_uhf`, no SCF aux),
then `ferric_cc::u_ccsd::u_ccsd` (and `u_ccsd_t::u_ccsd_t`). Every
correlation integral is (pq|rs) = sum_P B^P_pq B^P_rs with
B = (P|Q)^{-1/2} (P|pq) in the Coulomb metric of `[mp2] auxbasis`
(default cc-pvdz-ri), per spin. The Fock operator is taken as diagonal
(canonical UHF orbital energies only). `frozen_core = n` drops the n lowest
orbitals of EACH spin.

REFERENCE
---------
* UHF: PySCF `scf.UHF`, exact integrals, ferric's basis JSON and Bohr
  geometry, followed to an internally STABLE state (`common.run_open_shell`).
* UCCSD: `pyscf.cc.dfuccsd.UCCSD` on that UHF with `with_df.auxmol` built
  from ferric's aux JSON (passed as `auxbasis`; the built auxmol is checked
  against ferric's AO count and metric) and
  `frozen = n`. DF-UCCSD builds every block, vvvv included, from the same
  DF factor, so the approximation is ferric's. Its Fock is the exact-SCF MO
  Fock, whose off-diagonal part is recorded (`fock_offdiag_max`) as the
  size of the one difference left.
* (T): `mycc.ccsd_t()` on the DF ERIs (PySCF `uccsd_t`).
* Spin components, computed here and anchored:
  - CCSD: e_aa / e_ab / e_bb from PySCF's converged amplitudes with the
    `uccsd.energy` expression split by spin on numpy chemist blocks from a
    Cholesky-metric B (no dfuccsd code); aa + ab + bb == mycc.e_corr is
    asserted to 1e-10 Ha.
  - (T): the four occupied spin classes (aaa, aab = PySCF's "baa",
    abb = "bba", bbb) from the dense formulas of
    `pyscf/cc/uccsd_t_slow.py::kernel`, transcribed per class; their sum
    == mycc.ccsd_t() is asserted to 1e-10 Ha.

SYSTEMS (cc-pVDZ / cc-pvdz-ri): OH 2Pi, NH2 2B1, O2 3Sg-, frozen core 0 and 1.

Run (light; well under a minute per system):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_uccsd.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "uccsd"
ROW_NAME = "Unrestricted CCSD / CCSD(T)"
BASIS = "cc-pvdz"
AUX = "cc-pvdz-ri"
SYSTEMS = {  # system -> (charge, multiplicity)
    "oh": (0, 2),
    "nh2": (0, 2),
    "o2": (0, 3),
}
FROZEN = (0, 1)
TOL_ANCHOR = 1e-10
CC_CONV_TOL = 1e-12
CC_CONV_TOL_NORMT = 1e-10


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
    return auxmol, aux_bas


def b_factor(mol, auxmol):
    """B[P, m, n] = sum_Q (L^{-1})_PQ (Q|mn), L = chol((P|Q))."""
    import numpy as np
    import scipy.linalg as sla
    from pyscf import df

    n = mol.nao_nr()
    naux = auxmol.nao_nr()
    int3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    low = np.linalg.cholesky(auxmol.intor("int2c2e"))
    b = sla.solve_triangular(low, int3.reshape(n * n, naux).T, lower=True)
    return b.reshape(naux, n, n)


def energy_components(mycc, b_ao):
    """aa / ab / bb split of `uccsd.energy` on numpy DF blocks."""
    import numpy as np

    t1a, t1b = mycc.t1
    t2aa, t2ab, t2bb = mycc.t2
    mask = mycc.get_frozen_mask()
    nocc = mycc.nocc
    ovov = []
    bov = []
    for s in (0, 1):
        c = mycc.mo_coeff[s][:, mask[s]]
        no = nocc[s]
        bov.append(
            np.einsum("Pmn,mi,na->Pia", b_ao, c[:, :no], c[:, no:], optimize=True)
        )
    for s in (0, 1):
        ovov.append(np.einsum("Pia,Pjb->iajb", bov[s], bov[s], optimize=True))
    ovOV = np.einsum("Pia,Pjb->iajb", bov[0], bov[1], optimize=True)

    def same(t1, t2, g):
        e = 0.25 * np.einsum("ijab,iajb", t2, g) - 0.25 * np.einsum("ijab,ibja", t2, g)
        e += 0.5 * np.einsum("ia,jb,iajb", t1, t1, g) - 0.5 * np.einsum(
            "ia,jb,ibja", t1, t1, g
        )
        return float(e)

    e_aa = same(t1a, t2aa, ovov[0])
    e_bb = same(t1b, t2bb, ovov[1])
    e_ab = float(
        np.einsum("iJaB,iaJB", t2ab, ovOV) + np.einsum("ia,jb,iajb", t1a, t1b, ovOV)
    )
    return e_aa, e_ab, e_bb


def triples_classes(mcc, eris):
    """uccsd_t_slow.kernel split by occupied spin class (dense, PySCF formulas)."""
    import numpy
    from pyscf import lib

    def p6(t):
        return (
            t
            + t.transpose(1, 2, 0, 4, 5, 3)
            + t.transpose(2, 0, 1, 5, 3, 4)
            + t.transpose(0, 2, 1, 3, 5, 4)
            + t.transpose(2, 1, 0, 5, 4, 3)
            + t.transpose(1, 0, 2, 4, 3, 5)
        )

    def r6(w):
        return (
            w
            + w.transpose(2, 0, 1, 3, 4, 5)
            + w.transpose(1, 2, 0, 3, 4, 5)
            - w.transpose(2, 1, 0, 3, 4, 5)
            - w.transpose(0, 2, 1, 3, 4, 5)
            - w.transpose(1, 0, 2, 3, 4, 5)
        )

    t1a, t1b = mcc.t1
    t2aa, t2ab, t2bb = mcc.t2
    nocca, noccb = t2ab.shape[:2]
    mo_ea, mo_eb = eris.mo_energy
    eia = mo_ea[:nocca, None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb, None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:, :nocca]
    fVO = eris.fockb[noccb:, :noccb]
    A = numpy.asarray
    out = {}

    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eia, eia)
    w = numpy.einsum("ijae,kceb->ijkabc", t2aa, A(eris.get_ovvv()))
    w -= numpy.einsum("mkbc,iajm->ijkabc", t2aa, A(eris.ovoo))
    r = r6(w)
    v = numpy.einsum("jbkc,ia->ijkabc", A(eris.ovov), t1a)
    v += numpy.einsum("jkbc,ai->ijkabc", t2aa, fvo) * 0.5
    wvd = p6(w + v) / d3
    out["aaa"] = 0.25 * float(numpy.einsum("ijkabc,ijkabc", wvd, r))

    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eIA, eIA)
    w = numpy.einsum("ijae,kceb->ijkabc", t2bb, A(eris.get_OVVV()))
    w -= numpy.einsum("imab,kcjm->ijkabc", t2bb, A(eris.OVOO))
    r = r6(w)
    v = numpy.einsum("jbkc,ia->ijkabc", A(eris.OVOV), t1b)
    v += numpy.einsum("jkbc,ai->ijkabc", t2bb, fVO) * 0.5
    wvd = p6(w + v) / d3
    out["bbb"] = 0.25 * float(numpy.einsum("ijkabc,ijkabc", wvd, r))

    # baa: one beta, two alpha occupied (ferric's aab class)
    w = numpy.einsum("jIeA,kceb->IjkAbc", t2ab, A(eris.get_ovvv())) * 2
    w += numpy.einsum("jIbE,kcEA->IjkAbc", t2ab, A(eris.get_ovVV())) * 2
    w += numpy.einsum("jkbe,IAec->IjkAbc", t2aa, A(eris.get_OVvv()))
    w -= numpy.einsum("mIbA,kcjm->IjkAbc", t2ab, A(eris.ovoo)) * 2
    w -= numpy.einsum("jMbA,kcIM->IjkAbc", t2ab, A(eris.ovOO)) * 2
    w -= numpy.einsum("jmbc,IAkm->IjkAbc", t2aa, A(eris.OVoo))
    r = w - w.transpose(0, 2, 1, 3, 4, 5)
    r = r + r.transpose(0, 2, 1, 3, 5, 4)
    v = numpy.einsum("jbkc,IA->IjkAbc", A(eris.ovov), t1b)
    v += numpy.einsum("kcIA,jb->IjkAbc", A(eris.ovOV), t1a)
    v += numpy.einsum("kcIA,jb->IjkAbc", A(eris.ovOV), t1a)
    v += numpy.einsum("jkbc,AI->IjkAbc", t2aa, fVO) * 0.5
    v += numpy.einsum("kIcA,bj->IjkAbc", t2ab, fvo) * 2
    w += v
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eia, eia)
    r /= d3
    out["aab"] = 0.25 * float(numpy.einsum("ijkabc,ijkabc", w, r))

    # bba: two beta, one alpha occupied (ferric's abb class)
    w = numpy.einsum("ijae,kceb->ijkabc", t2ab, A(eris.get_OVVV())) * 2
    w += numpy.einsum("ijeb,kcea->ijkabc", t2ab, A(eris.get_OVvv())) * 2
    w += numpy.einsum("jkbe,iaec->ijkabc", t2bb, A(eris.get_ovVV()))
    w -= numpy.einsum("imab,kcjm->ijkabc", t2ab, A(eris.OVOO)) * 2
    w -= numpy.einsum("mjab,kcim->ijkabc", t2ab, A(eris.OVoo)) * 2
    w -= numpy.einsum("jmbc,iakm->ijkabc", t2bb, A(eris.ovOO))
    r = w - w.transpose(0, 2, 1, 3, 4, 5)
    r = r + r.transpose(0, 2, 1, 3, 5, 4)
    v = numpy.einsum("jbkc,ia->ijkabc", A(eris.OVOV), t1a)
    v += numpy.einsum("iakc,jb->ijkabc", A(eris.ovOV), t1b)
    v += numpy.einsum("iakc,jb->ijkabc", A(eris.ovOV), t1b)
    v += numpy.einsum("JKBC,ai->iJKaBC", t2bb, fvo) * 0.5
    v += numpy.einsum("iKaC,BJ->iJKaBC", t2ab, fVO) * 2
    w += v
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eIA, eIA)
    r /= d3
    out["abb"] = 0.25 * float(numpy.einsum("ijkabc,ijkabc", w, r))
    return out


def block(mf, auxmol, aux_bas, b_ao, frozen) -> dict:
    import numpy as np
    from pyscf import df
    from pyscf.cc import dfuccsd

    mycc = dfuccsd.UCCSD(mf, frozen=frozen or None)
    # DF.build() rebuilds auxmol from `auxbasis`, so a preset auxmol would be
    # discarded (PySCF's default fitting set would be used instead): pass
    # ferric's aux through `auxbasis` and check the built auxmol below.
    mycc.with_df = df.DF(mf.mol, auxbasis=aux_bas)
    mycc.conv_tol = CC_CONV_TOL
    mycc.conv_tol_normt = CC_CONV_TOL_NORMT
    mycc.max_cycle = 300
    mycc.verbose = 0
    eris = mycc.ao2mo()
    built = mycc.with_df.auxmol
    assert built.nao_nr() == auxmol.nao_nr() and np.allclose(
        built.intor("int2c2e"), auxmol.intor("int2c2e"), atol=1e-12
    ), "DF-UCCSD did not build ferric's aux basis"
    mycc.kernel(eris=eris)
    if not mycc.converged:
        raise RuntimeError("DF-UCCSD did not converge")
    e_t = float(mycc.ccsd_t(eris=eris))
    e_aa, e_ab, e_bb = energy_components(mycc, b_ao)
    d_cc = (e_aa + e_ab + e_bb) - float(mycc.e_corr)
    assert abs(d_cc) < TOL_ANCHOR, f"numpy aa+ab+bb vs DF-UCCSD e_corr: {d_cc:.2e}"
    cls = triples_classes(mycc, eris)
    d_t = sum(cls.values()) - e_t
    assert abs(d_t) < TOL_ANCHOR, f"(T) classes vs ccsd_t(): {d_t:.2e}"
    offdiag = 0.0
    for f in (eris.focka, eris.fockb):
        offdiag = max(offdiag, float(np.max(np.abs(f - np.diag(np.diag(f))))))
    t1a, t1b = mycc.t1
    return {
        "frozen_core": int(frozen),
        "e_corr": float(mycc.e_corr),
        "e_aa": e_aa,
        "e_ab": e_ab,
        "e_bb": e_bb,
        "e_t": e_t,
        "e_t_aaa": cls["aaa"],
        "e_t_aab": cls["aab"],
        "e_t_abb": cls["abb"],
        "e_t_bbb": cls["bbb"],
        "e_total_ccsd": float(mf.e_tot) + float(mycc.e_corr),
        "e_total_ccsd_t": float(mf.e_tot) + float(mycc.e_corr) + e_t,
        "t1_norm": float(np.sqrt(np.sum(t1a**2) + np.sum(t1b**2))),
        "numpy_anchor_ccsd_diff": float(d_cc),
        "numpy_anchor_t_diff": float(d_t),
        "fock_offdiag_max": offdiag,
    }


def gen(system) -> Path:
    import numpy
    import pyscf

    charge, mult = SYSTEMS[system]
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, charge=charge, multiplicity=mult)
    ll = common.check_basis_like_for_like(mol, BASIS, symbols)
    auxmol, aux_bas = _auxmol(mol, symbols)
    uhf, mf = common.run_open_shell(
        mol, "uhf", conv_tol=1e-12, conv_tol_grad=1e-9, return_mf=True
    )
    b_ao = b_factor(mol, auxmol)
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
    }
    for fc in FROZEN:
        payload[f"uccsd_fc{fc}"] = block(mf, auxmol, aux_bas, b_ao, fc)
    payload["provenance"] = common.provenance(
        code="PySCF",
        version=pyscf.__version__,
        keywords={
            "scf": "scf.UHF exact integrals, 3 guesses + stability loop (common.run_open_shell)",
            "ccsd": "pyscf.cc.dfuccsd.UCCSD, with_df auxbasis = ferric's aux JSON",
            "triples": "mycc.ccsd_t() on the DF ERIs; classes from uccsd_t_slow formulas",
            "spin_components": "uccsd.energy split by spin on numpy Cholesky-metric B",
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
        frozen_core={f"uccsd_fc{fc}": fc for fc in FROZEN},
        scf_conv={"conv_tol": 1e-12, "conv_tol_grad": 1e-9},
        stability={"uhf": uhf["stability"]},
        generator="scripts/validation/gen_uccsd.py",
        extra={
            "basis_self_check": ll,
            "cc_conv": {"conv_tol": CC_CONV_TOL, "conv_tol_normt": CC_CONV_TOL_NORMT},
        },
    )
    path = common.write_reference(ROW, system, BASIS, payload)
    for fc in FROZEN:
        b = payload[f"uccsd_fc{fc}"]
        print(
            f"{system:4s} fc{fc} E_UHF {uhf['energy']:.10f} E_CCSD {b['e_corr']:.10f} "
            f"E_T {b['e_t']:.10f} fock_offdiag {b['fock_offdiag_max']:.1e} "
            f"multi={uhf['multiple_stable_minima']}",
            flush=True,
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
    print(f"GEN_UCCSD_DONE written={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
