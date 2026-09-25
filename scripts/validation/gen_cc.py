"""PySCF/numpy references for the VALIDATION.md rows "MP3", "RI-CCD",
"RI-CCSD spin-orbital / spin-adapted" and "CCSD(T) / spin-adapted (T)"
(validation tier W2).

Consumer: crates/ferric-cc/tests/validation_cc.rs.
Output:   testdata/reference/validation/cc/<system>_<basis>.json

LIKE-FOR-LIKE DENSITY FITTING (the point of this generator)
-----------------------------------------------------------
ferric's MP3/CCD/CCSD/(T) (crates/ferric-mp2/src/mp3.rs,
crates/ferric-cc/src/{ccd,ccsd,ccsd_closed_shell,ccsd_t,ccsd_t_closed_shell}.rs)
all build EVERY two-electron MO integral from one DF factorization,

    B^P_pq = sum_Q [V^{-1/2}]_PQ (Q|pq),   (pq|rs) ~= sum_P B^P_pq B^P_rs,

with V the Coulomb metric (P|Q) of the auxiliary basis, V^{-1/2} from a
Cholesky factor (`cholesky_inverse_sqrt`), and the reference orbitals from an
EXACT-integral RHF (`solve_rhf` default: no RI-J, no RI-K). The Fock operator
is taken as diagonal: only the RHF orbital energies enter the denominators and
no off-diagonal Fock term appears. The occupied/virtual split is canonical
(frozen core = the lowest `frozen_core` RHF orbitals, dropped).

This generator reproduces exactly that approximation, and nothing else:

1. Exact-integral PySCF RHF with ferric's own bundled orbital basis
   (common.build_pyscf_mol) and ferric's geometry in Bohr.
2. The SAME auxiliary basis, read from ferric's bundled JSON
   (common.pyscf_basis on the aux name), with the AO count checked against
   ferric's parser. B = L^{-1} (P|mn) with L the Cholesky factor of (P|Q),
   built here in numpy from PySCF's `int3c2e`/`int2c2e` (any factorization
   V = L L^T gives the same (pq|rs), so this is the same approximation as
   ferric's V^{-1/2}).
3. MO integrals (pq|rs) = sum_P B^P_pq B^P_rs, fed to PySCF's own CC codes
   through a hand-built `ccsd._ChemistsERIs` (same block layout as PySCF's
   `_make_eris_incore`) whose Fock matrix is set to diag(RHF mo_energy), the
   canonical-diagonal Fock ferric uses. The largest off-diagonal element of
   PySCF's recomputed MO Fock is recorded (`fock_offdiag_max`) as the evidence
   that dropping it is below the bar.
4. CCD  = pyscf.cc.ccd.CCD (CCSD with T1 frozen at zero) on those eris.
   CCSD = pyscf.cc.ccsd.CCSD on those eris. (T) = pyscf.cc.ccsd_t.kernel on
   those eris and the CCSD amplitudes.
5. MP3 = two INDEPENDENT numpy constructions on the same DF MO integrals:
   (a) spin-orbital textbook sum-over-states, Shavitt & Bartlett, "Many-Body
       Methods in Chemistry and Physics" (CUP 2009), Eq. (10.12) diagrams:
         E3 = 1/8 sum <ij||ab><ab||cd><cd||ij> / (D_ijab D_ijcd)   (pp ladder)
            + 1/8 sum <ij||ab><kl||ij><ab||kl> / (D_ijab D_klab)   (hh ladder)
            +     sum <ij||ab><kb||cj><ac||ik> / (D_ijab D_ikac)   (ring)
       with D_ijab = e_i + e_j - e_a - e_b;
   (b) closed-shell spatial: E3 = sum_ijab [2(ia|jb) - (ib|ja)] t2^(2)_ijab,
       where t2^(2) = (L t2^(1))/D is the LINEAR part of PySCF's own RCCSD
       doubles residual applied to the first-order amplitudes, extracted
       exactly by a symmetric difference (the CCD residual is a quadratic
       polynomial in t2, so [U(t) - U(-t)]/2 is its linear part with no
       truncation error).
   (a) and (b) share no code; their agreement (`mp3_spinorbital_vs_spatial`)
   is asserted here at 1e-10.

Independent check of the DF construction itself: the DF-MP2 energy from B
(numpy) is compared with PySCF's `mp.MP2(mf).density_fit(auxbasis=...)` (a
different code path through `pyscf.df`), asserted at 1e-10.

Also recorded, for the Rust test's negative controls (values ferric must MISS):
  * frozen-core (one core orbital per first-row heavy atom) MP3/CCD/CCSD/(T);
  * the conventional (exact 4-index integral) CCSD correlation energy — the
    DF error the like-for-like construction removes;
  * CCD vs CCSD differ on every system (asserted by the Rust side).

Systems (geometries in testdata/molecules/validation/):
    h2o   / cc-pvdz, def2-svp, aug-cc-pvdz
    nh3   / cc-pvdz, def2-svp
    hcn   / cc-pvdz
    c2h6  / cc-pvdz
Aux basis per orbital basis: cc-pvdz -> cc-pvdz-ri, def2-svp -> def2-svp-rifit,
aug-cc-pvdz -> aug-cc-pvdz-rifit (ferric's bundled names).

Open-shell CC is NOT generated: ferric's CC/MP3 drivers accept only a
restricted reference (`ScfResult::eps_r`/`mos_r` assert Spin::Restricted), so
a UCCSD reference would have no consumer.

Run (light step; ~1 min total):
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_cc.py [system ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "cc"
AUX = {
    "cc-pvdz": "cc-pvdz-ri",
    "def2-svp": "def2-svp-rifit",
    "aug-cc-pvdz": "aug-cc-pvdz-rifit",
}
# (system, basis) -> methods. MP3's numpy spin-orbital construction is
# (2 nmo)^4 dense, so it is restricted to the small systems it validates.
CASES = {
    ("h2o", "cc-pvdz"): ("mp3", "ccd", "ccsd", "ccsd_t"),
    ("h2o", "def2-svp"): ("mp3", "ccd", "ccsd"),
    ("nh3", "cc-pvdz"): ("mp3", "ccd", "ccsd"),
    ("nh3", "def2-svp"): ("mp3", "ccd", "ccsd"),
    ("hcn", "cc-pvdz"): ("ccsd", "ccsd_t"),
    ("h2o", "aug-cc-pvdz"): ("ccsd", "ccsd_t"),
    ("c2h6", "cc-pvdz"): ("ccsd", "ccsd_t"),
}
SCF_CONV_TOL = 1e-12
SCF_CONV_TOL_GRAD = 1e-9
CC_CONV_TOL = 1e-12
CC_CONV_TOL_NORMT = 1e-10


def n_core_first_row(symbols: list[str]) -> int:
    """One 1s core orbital per Li-Ne atom (all systems here are H + first row)."""
    n = 0
    for s in symbols:
        z = common.z_of(s)
        if z > 10:
            raise ValueError(f"{s}: frozen-core count here is for first-row atoms only")
        if z > 2:
            n += 1
    return n


def df_factor(mol, auxmol):
    """B[P, m, n] = sum_Q (L^{-1})_PQ (Q|mn), L = chol((P|Q)) (lower)."""
    import numpy as np
    import scipy.linalg
    from pyscf import df

    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    j3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    j3c = j3c.reshape(nao * nao, naux).T  # (P, mn)
    j2c = auxmol.intor("int2c2e")
    evals = np.linalg.eigvalsh(j2c)
    low = scipy.linalg.cholesky(j2c, lower=True)
    b = scipy.linalg.solve_triangular(low, j3c, lower=True)
    return b.reshape(naux, nao, nao), float(evals[0]), float(evals[-1])


def mo_eri(b_ao, c):
    """Chemist (pq|rs) over the columns of c from the DF factor."""
    import numpy as np

    b_mo = np.einsum("Pmn,mp,nq->Ppq", b_ao, c, c, optimize=True)
    return np.einsum("Ppq,Prs->pqrs", b_mo, b_mo, optimize=True), b_mo


def build_eris(mycc, eri, mo_energy_active):
    """PySCF `_ChemistsERIs` from a dense chemist MO eri over the ACTIVE MOs,
    with the canonical-diagonal Fock ferric uses. Block layout copied from
    pyscf.cc.ccsd._make_eris_incore (its commented reference lines)."""
    import numpy as np
    from pyscf import ao2mo, lib
    from pyscf.cc import ccsd

    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc)  # exact-RHF Fock in the active MO basis
    fock_full = eris.fock.copy()
    nocc = eris.nocc
    nmo = fock_full.shape[0]
    assert eri.shape == (nmo,) * 4
    nvir = nmo - nocc
    offdiag = float(np.max(np.abs(fock_full - np.diag(np.diag(fock_full)))))
    diag_dev = float(np.max(np.abs(np.diag(fock_full) - mo_energy_active)))
    eris.fock = np.diag(mo_energy_active)
    eris.mo_energy = np.asarray(mo_energy_active).copy()
    o, v = slice(0, nocc), slice(nocc, nmo)
    eris.oooo = eri[o, o, o, o].copy()
    eris.ovoo = eri[o, v, o, o].copy()
    eris.ovvo = eri[o, v, v, o].copy()
    eris.ovov = eri[o, v, o, v].copy()
    eris.oovv = eri[o, o, v, v].copy()
    ovvv = eri[o, v, v, v].copy()
    eris.ovvv = lib.pack_tril(ovvv.reshape(-1, nvir, nvir)).reshape(nocc, nvir, -1)
    eris.vvvv = ao2mo.restore(4, eri[v, v, v, v].copy(), nvir)
    return eris, offdiag, diag_dev


def mp2_mp3_spinorbital(eri, eps, nocc):
    """Textbook spin-orbital MP2 and MP3 (Shavitt & Bartlett Eq. 10.12).
    `eri` chemist spatial over active MOs, `nocc` active occupied spatial."""
    import numpy as np

    nmo = eps.shape[0]
    n2 = 2 * nmo
    sp = np.arange(n2) // 2  # spatial index
    sg = np.arange(n2) % 2  # spin (interleaved)
    # <pq|rs> = (pr|qs) d(sp,sr) d(sq,ss)
    phys = eri[np.ix_(sp, sp, sp, sp)].transpose(0, 2, 1, 3)
    phys = phys * (sg[:, None, None, None] == sg[None, None, :, None])
    phys = phys * (sg[None, :, None, None] == sg[None, None, None, :])
    g = phys - phys.transpose(0, 1, 3, 2)  # <pq||rs>
    e = np.repeat(eps, 2)
    no = 2 * nocc
    o, v = slice(0, no), slice(no, n2)
    eo, ev = e[o], e[v]
    d = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )
    t = g[o, o, v, v] / d
    e2 = 0.25 * np.einsum("ijab,ijab->", g[o, o, v, v], t)
    e_pp = 0.125 * np.einsum("ijab,abcd,ijcd->", t, g[v, v, v, v], t, optimize=True)
    e_hh = 0.125 * np.einsum("ijab,klij,klab->", t, g[o, o, o, o], t, optimize=True)
    e_ring = np.einsum("ijab,kbcj,ikac->", t, g[o, v, v, o], t, optimize=True)
    return (
        float(e2),
        float(e_pp + e_hh + e_ring),
        {
            "pp_ladder": float(e_pp),
            "hh_ladder": float(e_hh),
            "ring": float(e_ring),
        },
    )


def mp3_spatial_via_pyscf_residual(mycc, eris):
    """Closed-shell MP3 from the linear part of PySCF's RCCSD doubles residual."""
    import numpy as np
    from pyscf.cc import ccsd

    nocc = eris.nocc
    e = eris.mo_energy
    eo, ev = e[:nocc], e[nocc:]
    d = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )
    ovov = np.asarray(eris.ovov)  # (ia|jb) at [i,a,j,b]
    iajb = ovov.transpose(0, 2, 1, 3)  # [i,j,a,b] = (ia|jb)
    t1amp = iajb / d
    nvir = ev.shape[0]
    z1 = np.zeros((nocc, nvir))
    _, up = ccsd.update_amps(mycc, z1, t1amp, eris)
    _, um = ccsd.update_amps(mycc, z1, -t1amp, eris)
    t2_2 = 0.5 * (up - um)  # = (L t1amp) / D exactly (quadratic cancels)
    w = 2.0 * iajb - iajb.transpose(0, 1, 3, 2)
    e2 = float(np.einsum("ijab,ijab->", w, t1amp))
    e3 = float(np.einsum("ijab,ijab->", w, t2_2))
    return e2, e3


def run_df_block(mf, mol, b_ao, frozen, methods):
    """All DF correlation energies for one frozen-core setting."""
    import numpy as np
    from pyscf.cc import ccd, ccsd

    nfz = frozen
    c_act = mf.mo_coeff[:, nfz:]
    eps_act = mf.mo_energy[nfz:]
    nocc_act = mol.nelectron // 2 - nfz
    eri, b_mo = mo_eri(b_ao, c_act)
    out = {
        "frozen_core": nfz,
        "nocc_active": nocc_act,
        "nvir": int(eps_act.size - nocc_act),
    }

    # DF-MP2 in numpy from B (for the DF construction cross-check).
    o, v = slice(0, nocc_act), slice(nocc_act, eps_act.size)
    iajb = eri[o, v, o, v].transpose(0, 2, 1, 3)
    eo, ev = eps_act[o], eps_act[v]
    d = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )
    out["mp2_numpy"] = float(
        np.einsum("ijab,ijab->", 2.0 * iajb - iajb.transpose(0, 1, 3, 2), iajb / d)
    )

    def new_cc(cls):
        cc_obj = cls(mf, frozen=nfz if nfz else None)
        cc_obj.conv_tol = CC_CONV_TOL
        cc_obj.conv_tol_normt = CC_CONV_TOL_NORMT
        cc_obj.max_cycle = 300
        cc_obj.verbose = 0
        cc_obj.direct = False
        return cc_obj

    probe = new_cc(ccsd.CCSD)
    eris, offdiag, diag_dev = build_eris(probe, eri, eps_act)
    out["fock_offdiag_max"] = offdiag
    out["fock_diag_vs_mo_energy_max"] = diag_dev

    if "mp3" in methods:
        e2_so, e3_so, parts = mp2_mp3_spinorbital(eri, eps_act, nocc_act)
        e2_sp, e3_sp = mp3_spatial_via_pyscf_residual(probe, eris)
        dmp3 = abs(e3_so - e3_sp)
        dmp2 = abs(e2_so - e2_sp)
        assert dmp3 < 1e-10 and dmp2 < 1e-10, (
            f"MP3 constructions disagree: spin-orbital {e3_so!r} vs spatial {e3_sp!r} ({dmp3:.2e})"
        )
        out["mp3"] = {
            "e_mp2": e2_so,
            "e_mp3": e3_so,
            "e_corr": e2_so + e3_so,
            "e_mp3_components_spinorbital": parts,
            "e_mp3_spatial_pyscf_residual": e3_sp,
            "mp3_spinorbital_vs_spatial": dmp3,
            "mp2_spinorbital_vs_spatial": dmp2,
        }
    if "ccd" in methods:
        mycc = new_cc(ccd.CCD)
        mycc.kernel(eris=eris)
        if not mycc.converged:
            raise RuntimeError("DF-CCD did not converge")
        out["ccd"] = {"e_corr": float(mycc.e_corr)}
    if "ccsd" in methods:
        mycc = new_cc(ccsd.CCSD)
        mycc.kernel(eris=eris)
        if not mycc.converged:
            raise RuntimeError("DF-CCSD did not converge")
        out["ccsd"] = {
            "e_corr": float(mycc.e_corr),
            "t1_norm": float(np.linalg.norm(mycc.t1)),
            "t1_diagnostic": float(np.linalg.norm(mycc.t1) / np.sqrt(2 * nocc_act)),
        }
        if "ccsd_t" in methods:
            from pyscf.cc import ccsd_t

            e_t = ccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, verbose=0)
            out["ccsd_t"] = {
                "e_t": float(e_t),
                "e_corr_ccsd_t": float(mycc.e_corr + e_t),
            }
    return out


def main() -> int:
    import numpy
    import pyscf
    import scipy
    from pyscf import gto, mp, scf
    from pyscf.cc import ccsd

    only = set(sys.argv[1:])
    written = []
    for (system, basis_name), methods in CASES.items():
        if only and system not in only:
            continue
        aux_name = AUX[basis_name]
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        aux_basis, aux_cart = common.pyscf_basis(aux_name, symbols)
        if aux_cart:
            raise ValueError(f"{aux_name}: Cartesian aux shells; not handled")
        auxmol = gto.M(
            atom=common.pyscf_atom_bohr(symbols, coords),
            unit="Bohr",
            basis=aux_basis,
            cart=False,
            verbose=0,
        )
        naux = auxmol.nao_nr()
        naux_ferric = common.ferric_nao(aux_name, symbols)
        assert naux == naux_ferric, (
            f"{aux_name}: PySCF naux {naux} != ferric {naux_ferric}"
        )

        mf = scf.RHF(mol)
        mf.conv_tol = SCF_CONV_TOL
        mf.conv_tol_grad = SCF_CONV_TOL_GRAD
        mf.max_cycle = 200
        mf.verbose = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"{system}/{basis_name}: RHF did not converge")

        b_ao, j2c_min, j2c_max = df_factor(mol, auxmol)
        nocc = mol.nelectron // 2
        ncore = n_core_first_row(symbols)

        all_e = run_df_block(mf, mol, b_ao, 0, methods)
        fc = run_df_block(mf, mol, b_ao, ncore, methods)

        # Independent DF path: PySCF's own density-fitted MP2 (pyscf.df), same aux.
        mymp = mp.MP2(mf).density_fit(auxbasis=aux_basis)
        mymp.verbose = 0
        e_dfmp2_pyscf = float(mymp.kernel()[0])
        d_dfmp2 = abs(e_dfmp2_pyscf - all_e["mp2_numpy"])
        assert d_dfmp2 < 1e-10, (
            f"DF-MP2 numpy {all_e['mp2_numpy']} vs pyscf.df {e_dfmp2_pyscf}"
        )

        # Conventional (exact 4-index) CCSD: the DF error the like-for-like
        # construction removes, recorded as a must-miss control.
        conv = ccsd.CCSD(mf)
        conv.conv_tol = CC_CONV_TOL
        conv.conv_tol_normt = CC_CONV_TOL_NORMT
        conv.max_cycle = 300
        conv.verbose = 0
        conv.kernel()
        if not conv.converged:
            raise RuntimeError("conventional CCSD did not converge")

        payload = {
            "system": system,
            "basis": basis_name,
            "aux_basis": aux_name,
            "charge": 0,
            "multiplicity": 1,
            "methods": list(methods),
            "nao": int(mol.nao_nr()),
            "naux": int(naux),
            "nocc": int(nocc),
            "nvir": int(mol.nao_nr() - nocc),
            "n_frozen_core_control": int(ncore),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "rhf": {
                "energy": float(mf.e_tot),
                "homo": float(mf.mo_energy[nocc - 1]),
                "lumo": float(mf.mo_energy[nocc]),
            },
            "df": {
                "metric_min_eigenvalue": j2c_min,
                "metric_max_eigenvalue": j2c_max,
                "mp2_numpy_from_B": all_e["mp2_numpy"],
                "mp2_pyscf_density_fit": e_dfmp2_pyscf,
                "mp2_numpy_vs_pyscf_df": d_dfmp2,
            },
            "all_electron": all_e,
            "frozen_core": fc,
            "conventional_exact_integrals": {"ccsd_e_corr": float(conv.e_corr)},
            "provenance": common.provenance(
                code="PySCF + numpy",
                version=pyscf.__version__,
                keywords={
                    "reference": "scf.RHF, exact 4-index J/K (no density fitting)",
                    "scf_conv_tol": SCF_CONV_TOL,
                    "scf_conv_tol_grad": SCF_CONV_TOL_GRAD,
                    "correlation_integrals": "DF: (pq|rs) = sum_P B^P_pq B^P_rs, "
                    "B = L^-1 (P|mn), L = chol((P|Q)) (numpy on PySCF int3c2e/int2c2e)",
                    "fock": "diag(RHF mo_energy) (canonical, off-diagonal dropped)",
                    "ccd": "pyscf.cc.ccd.CCD on hand-built DF _ChemistsERIs",
                    "ccsd": "pyscf.cc.ccsd.CCSD on hand-built DF _ChemistsERIs",
                    "ccsd_t": "pyscf.cc.ccsd_t.kernel on the same eris",
                    "mp3": "numpy spin-orbital (Shavitt-Bartlett 10.12) == "
                    "spatial via linear PySCF RCCSD residual",
                    "cc_conv_tol": CC_CONV_TOL,
                    "cc_conv_tol_normt": CC_CONV_TOL_NORMT,
                    "numpy": numpy.__version__,
                    "scipy": scipy.__version__,
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                aux=aux_name,
                frozen_core="0 (all_electron block); n_frozen_core_control (frozen_core block)",
                scf_conv=SCF_CONV_TOL,
                generator="scripts/validation/gen_cc.py",
                extra={
                    "aux_basis_json": str(
                        common.basis_json_path(aux_name).relative_to(common.ROOT)
                    ),
                    "aux_basis_sha256": common.sha256_file(
                        common.basis_json_path(aux_name)
                    ),
                    "basis_check": basis_check,
                },
            ),
        }
        path = common.write_reference(ROW, system, basis_name, payload)
        written.append(path)
        summary = {
            k: all_e[k].get("e_corr", all_e[k].get("e_t"))
            for k in ("mp3", "ccd", "ccsd", "ccsd_t")
            if k in all_e
        }
        if "ccsd_t" in all_e:
            summary["ccsd_t"] = all_e["ccsd_t"]["e_t"]
        print(
            f"{system}/{basis_name} aux={aux_name} nao={mol.nao_nr()} naux={naux} "
            f"E_RHF={mf.e_tot:.10f} {summary} "
            f"fock_offdiag={all_e['fock_offdiag_max']:.1e} dfmp2|d|={d_dfmp2:.1e} "
            f"conv-DF CCSD={conv.e_corr - all_e.get('ccsd', {}).get('e_corr', float('nan')):.2e}",
            flush=True,
        )
    for p in written:
        print("wrote", p.relative_to(common.ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
