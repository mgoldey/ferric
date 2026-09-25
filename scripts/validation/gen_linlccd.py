"""PySCF + numpy references for the VALIDATION.md "LinLCCD(hh)" row.

Consumer: crates/ferric-cc/tests/validation_linlccd.rs
Output:   testdata/reference/validation/linlccd/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-cc/src/linlccd.rs and
linlccd_u.rs, not assumed): LinLCCD(hh) (Carter-Fenk, JPCA 129, 7251 (2025),
eq. 14) — linearized CCD keeping ONLY the driver, the (canonical) Fock
diagonal and the hole–hole ladder:

    D_ijab t_ij^ab = <ij||ab> + ½ Σ_kl <kl||ij> t_kl^ab,
    D_ijab = ε_i + ε_j − ε_a − ε_b,      E = ¼ Σ_ijab <ij||ab> t_ij^ab

in spin orbitals, with Coulomb-metric RI integrals from the same aux basis,
all electrons correlated (`CcConfig::default().frozen_core == 0`). ferric
solves it by Jacobi iteration + DIIS on a spin-orbital amplitude tensor.
`LadderVariant::DriversOnly` drops the ladder, which is exactly MP2.

THE REFERENCE, built independently of ferric's construction:
  * SCF: PySCF, exact four-centre J/K, ferric's bundled basis and Bohr
    geometry (common.py). Closed shell: scf.RHF. Open shell (OH ²Π): scf.UHF
    via common.run_open_shell — three guesses, each followed to an internally
    stable state, lowest kept, unstable refused.
  * Integrals: PySCF `df.incore.cholesky_eri` with ferric's aux JSON
    (Cholesky metric factor; same fitted (pq|rs) as ferric's V^{-1/2}).
  * Solver: NOT iterative. The equation is linear in t and, for fixed (a,b),
    couples only the occupied pairs, so with
        H[(ij),(kl)] = (ε_i + ε_j) δ_ik δ_jl − W[(ij),(kl)]
    it reads (H − (ε_a + ε_b)) t_ab = v_ab; H is symmetric, diagonalized ONCE,
    and every (a,b) column is solved exactly in its eigenbasis (the paper's
    eq. 15 "dressed pair energy" form, which ferric does not use).
  * Two constructions of the closed-shell answer, which must agree:
      - SPIN-ADAPTED (spatial orbitals, derived by spin integration):
            D T_ijab = (ia|jb) + Σ_kl (ki|lj) T_klab,
            E = Σ_ijab T_ijab [2 (ia|jb) − (ib|ja)];
      - SPIN-ORBITAL, from explicitly spin-blocked MO coefficients
        (Lso^P_pq = C_pα^T L^P C_qα + C_pβ^T L^P C_qβ), which is also the
        construction used for UHF OH. Agreement to 1e-12 is asserted here.

Anchors asserted HERE before anything is written:
  * ladder off (MP2) numpy == PySCF `mp.dfmp2.DFMP2` (RHF) or
    `mp.dfump2.DFUMP2` (UHF), frozen=None, to 1e-12;
  * closed shell: spin-adapted == spin-orbital to 1e-12 (MP2 and hh);
  * the solved amplitudes satisfy the amplitude equation (dense residual
    evaluated in spin orbitals) to 1e-11, and the hh correction
    E_hh − E_MP2 is resolvable (> 1e-5 Eh).

Run (light; seconds per system):
    scripts/validation/run_slot.sh --light -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_linlccd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "linlccd"
ROW_NAME = "LinLCCD(hh)"
AUX = "cc-pvdz-ri"
BASES = ("6-31g", "cc-pvdz")
# system -> (charge, multiplicity)
SYSTEMS = {
    "h2o": (0, 1),
    "nh3": (0, 1),
    "oh": (0, 2),
}
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
TOL_ANCHOR = 1e-12
TOL_RESIDUAL = 1e-11


def df_factors(mol, symbols):
    from pyscf import df, lib

    aux_bas, aux_cart = common.pyscf_basis(AUX, symbols)
    assert not aux_cart
    auxmol = df.addons.make_auxmol(mol, aux_bas)
    auxmol.cart = False
    auxmol.build()
    cderi = df.incore.cholesky_eri(mol, auxmol=auxmol)
    return lib.unpack_tril(cderi), cderi.shape[0], aux_bas


def pyscf_df_mp2(mf, aux_bas, unrestricted):
    from pyscf import df
    from pyscf.mp import dfmp2, dfump2

    cls = dfump2.DFUMP2 if unrestricted else dfmp2.DFMP2
    mp = cls(mf, frozen=None)
    mp.with_df = df.DF(mf.mol)
    mp.with_df.auxmol = df.addons.make_auxmol(mf.mol, aux_bas)
    mp.with_df.auxmol.cart = False
    mp.with_df.auxmol.build()
    mp.with_df.auxbasis = aux_bas
    mp.kernel()
    return float(mp.e_corr)


def solve_pair_linear(h_pair, drive, s_ab):
    """Solve (H − s_ab) t_ab = drive_ab for every (a,b) column exactly.

    h_pair: (npair, npair) symmetric; drive: (npair, nab); s_ab: (nab,).
    Returns t (npair, nab). Asserts the pair spectrum lies below every s_ab.
    """
    lam, u = np.linalg.eigh(h_pair)
    gap = (s_ab[None, :] - lam[:, None]).min()
    assert gap > 0.0, f"no gap: min(s_ab - lambda) = {gap}"
    return u @ ((u.T @ drive) / (lam[:, None] - s_ab[None, :]))


# ---------------------------------------------------------------------------
# Closed shell, spin-adapted (spatial orbitals)
# ---------------------------------------------------------------------------


def spin_adapted(lov, loo, eo, ev, ladder):
    no, nv = len(eo), len(ev)
    g = np.einsum("Pia,Pjb->ijab", lov, lov, optimize=True)  # (ia|jb) as [i,j,a,b]
    g_x = g.transpose(0, 1, 3, 2)  # (ib|ja)
    s_ab = (ev[:, None] + ev[None, :]).reshape(-1)
    pair_e = (eo[:, None] + eo[None, :]).reshape(-1)
    drive = g.reshape(no * no, nv * nv)  # (H - s) T = (ia|jb)
    if ladder:
        # W[(ij),(kl)] = (ki|lj)
        w = np.einsum("Pki,Plj->ijkl", loo, loo, optimize=True).reshape(
            no * no, no * no
        )
        h_pair = np.diag(pair_e) - w
    else:
        h_pair = np.diag(pair_e)
    t = solve_pair_linear(h_pair, drive, s_ab).reshape(no, no, nv, nv)
    e = float(np.sum(t * (2.0 * g - g_x)))
    sym = float(np.max(np.abs(t - t.transpose(1, 0, 3, 2))))
    return e, sym


# ---------------------------------------------------------------------------
# Spin-orbital (RHF or UHF)
# ---------------------------------------------------------------------------


def spin_orbital(lao, c_a, c_b, e_a, e_b, na, nb, ladder):
    """LinLCCD(hh) / MP2 in spin orbitals from per-spin MOs.

    Spin-orbital order: occ = [α occ..., β occ...], vir = [α vir..., β vir...].
    """
    nao = c_a.shape[0]
    nmo = c_a.shape[1]

    def spin_blocked(cols_a, cols_b):
        """(nao, n) α-row and β-row coefficient blocks for a spin-orbital set."""
        n = cols_a.shape[1] + cols_b.shape[1]
        ra = np.zeros((nao, n))
        rb = np.zeros((nao, n))
        ra[:, : cols_a.shape[1]] = cols_a
        rb[:, cols_a.shape[1] :] = cols_b
        return ra, rb

    oa, ob = spin_blocked(c_a[:, :na], c_b[:, :nb])
    va, vb = spin_blocked(c_a[:, na:], c_b[:, nb:])
    eo = np.concatenate([e_a[:na], e_b[:nb]])
    ev = np.concatenate([e_a[na:], e_b[nb:]])
    no, nv = len(eo), len(ev)
    assert nv == 2 * nmo - na - nb

    def l3(pa, pb, qa, qb):
        return np.einsum("Pmn,mp,nq->Ppq", lao, pa, qa, optimize=True) + np.einsum(
            "Pmn,mp,nq->Ppq", lao, pb, qb, optimize=True
        )

    lov = l3(oa, ob, va, vb)
    g = np.einsum("Pia,Pjb->ijab", lov, lov, optimize=True)  # (ia|jb)
    v_oovv = g - g.transpose(0, 1, 3, 2)  # <ij||ab> = (ia|jb) - (ib|ja)
    pair_e = (eo[:, None] + eo[None, :]).reshape(-1)
    s_ab = (ev[:, None] + ev[None, :]).reshape(-1)
    d = (eo[:, None, None, None] + eo[None, :, None, None]
         - ev[None, None, :, None] - ev[None, None, None, :])  # fmt: skip
    if ladder:
        loo = l3(oa, ob, oa, ob)
        c_oooo = np.einsum(
            "Pki,Plj->klij", loo, loo, optimize=True
        )  # (ki|lj) as [k,l,i,j]
        v_oooo = c_oooo - c_oooo.transpose(0, 1, 3, 2)  # <kl||ij>
        m = 0.5 * v_oooo.transpose(2, 3, 0, 1).reshape(no * no, no * no)  # [(ij),(kl)]
        assert np.max(np.abs(m - m.T)) < 1e-12
        h_pair = np.diag(pair_e) - m
    else:
        v_oooo = None
        h_pair = np.diag(pair_e)
    t = solve_pair_linear(h_pair, v_oovv.reshape(no * no, nv * nv), s_ab)
    t = t.reshape(no, no, nv, nv)
    e = float(0.25 * np.sum(v_oovv * t))
    # Dense residual of the amplitude equation (independent of the solve).
    r = v_oovv - d * t
    if v_oooo is not None:
        r = r + 0.5 * np.einsum("klij,klab->ijab", v_oooo, t, optimize=True)
    return e, float(np.max(np.abs(r)))


def main() -> int:
    import pyscf
    from pyscf import scf

    only = set(sys.argv[1:])
    written = []
    for system, (charge, mult) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        for basis_name in BASES:
            mol = common.build_pyscf_mol(
                xyz, basis_name, charge=charge, multiplicity=mult
            )
            basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
            lao, naux, aux_bas = df_factors(mol, symbols)
            unrestricted = mult != 1
            stability = None
            scf_block = {}
            if unrestricted:
                res, mf = common.run_open_shell(
                    mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD,
                    return_mf=True,
                )  # fmt: skip
                stability = res["stability"]
                scf_block = {"method": "UHF", **res}
                c_a, c_b = mf.mo_coeff
                e_a, e_b = mf.mo_energy
                na, nb = mol.nelec
            else:
                mf = scf.RHF(mol)
                mf.conv_tol = CONV_TOL
                mf.conv_tol_grad = CONV_TOL_GRAD
                mf.max_cycle = 200
                mf.kernel()
                if not mf.converged:
                    raise RuntimeError(f"{system}/{basis_name}: RHF not converged")
                scf_block = {"method": "RHF", "energy": float(mf.e_tot)}
                c_a = c_b = mf.mo_coeff
                e_a = e_b = mf.mo_energy
                na = nb = mol.nelectron // 2

            e_pyscf_mp2 = pyscf_df_mp2(mf, aux_bas, unrestricted)
            e_mp2_so, res_mp2 = spin_orbital(
                lao, c_a, c_b, e_a, e_b, na, nb, ladder=False
            )
            e_hh_so, res_hh = spin_orbital(lao, c_a, c_b, e_a, e_b, na, nb, ladder=True)
            ctx = f"{system}/{basis_name}"
            checks = {
                "mp2_numpy_so_vs_pyscf": abs(e_mp2_so - e_pyscf_mp2),
                "residual_mp2_max": res_mp2,
                "residual_hh_max": res_hh,
            }
            if checks["mp2_numpy_so_vs_pyscf"] > TOL_ANCHOR:
                raise RuntimeError(f"{ctx}: MP2 anchor vs PySCF failed: {checks}")
            if max(res_mp2, res_hh) > TOL_RESIDUAL:
                raise RuntimeError(f"{ctx}: amplitude residual too large: {checks}")
            if abs(e_hh_so - e_mp2_so) < 1e-5:
                raise RuntimeError(f"{ctx}: hh ladder correction unresolvable")
            if not unrestricted:
                cocc = c_a[:, :na]
                cvir = c_a[:, na:]
                lov = np.einsum("Pmn,mi,na->Pia", lao, cocc, cvir, optimize=True)
                loo = np.einsum("Pmn,mi,nj->Pij", lao, cocc, cocc, optimize=True)
                e_mp2_sa, sym_mp2 = spin_adapted(
                    lov, loo, e_a[:na], e_a[na:], ladder=False
                )
                e_hh_sa, sym_hh = spin_adapted(
                    lov, loo, e_a[:na], e_a[na:], ladder=True
                )
                checks.update(
                    {
                        "mp2_spin_adapted_vs_spin_orbital": abs(e_mp2_sa - e_mp2_so),
                        "hh_spin_adapted_vs_spin_orbital": abs(e_hh_sa - e_hh_so),
                        "mp2_spin_adapted_vs_pyscf": abs(e_mp2_sa - e_pyscf_mp2),
                        "pair_symmetry_max_hh": sym_hh,
                    }
                )
                for k in (
                    "mp2_spin_adapted_vs_spin_orbital",
                    "hh_spin_adapted_vs_spin_orbital",
                    "mp2_spin_adapted_vs_pyscf",
                ):
                    if checks[k] > TOL_ANCHOR:
                        raise RuntimeError(f"{ctx}: {k} = {checks[k]:.2e}")

            payload = {
                "row": ROW_NAME,
                "system": system,
                "basis": basis_name,
                "aux_basis": AUX,
                "charge": charge,
                "multiplicity": mult,
                "nao": mol.nao_nr(),
                "naux": naux,
                "nelec_alpha": int(na),
                "nelec_beta": int(nb),
                "nuclear_repulsion": float(mol.energy_nuc()),
                "scf": scf_block,
                "e_scf": float(mf.e_tot),
                "mp2": {"e_corr": e_mp2_so, "pyscf_df_mp2_e_corr": e_pyscf_mp2},
                "linlccd_hh": {"e_corr": e_hh_so},
                "hh_minus_mp2": e_hh_so - e_mp2_so,
                "checks": checks,
                "provenance": common.provenance(
                    code="PySCF + numpy",
                    version=pyscf.__version__,
                    keywords={
                        "scf": (
                            "scf.UHF via common.run_open_shell"
                            if unrestricted
                            else "scf.RHF"
                        )
                        + f", exact J/K, conv_tol {CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}",
                        "integrals": "df.incore.cholesky_eri with auxmol from ferric's aux "
                        "JSON (Coulomb metric)",
                        "method": "LinLCCD(hh): D t = <ij||ab> + 1/2 sum_kl <kl||ij> t_kl^ab, "
                        "E = 1/4 sum <ij||ab> t; solved EXACTLY per (a,b) in the eigenbasis "
                        "of the occupied-pair matrix (no iteration)",
                        "mp2_anchor": (
                            "mp.dfump2.DFUMP2" if unrestricted else "mp.dfmp2.DFMP2"
                        )
                        + "(frozen=None)",
                        "numpy": np.__version__,
                    },
                    basis_name=basis_name,
                    xyz_path=xyz,
                    coords_bohr=coords,
                    symbols=symbols,
                    grid=None,
                    aux=AUX,
                    frozen_core=None,
                    scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                    stability=stability,
                    generator="scripts/validation/gen_linlccd.py",
                    extra={
                        "basis_self_check": basis_check,
                        "aux_basis_json_sha256": common.sha256_file(
                            common.basis_json_path(AUX)
                        ),
                    },
                ),
            }
            path = common.write_reference(ROW, system, basis_name, payload)
            written.append(path)
            print(
                f"{ctx:14s} E_scf={mf.e_tot:.12f} MP2={e_mp2_so:+.12f} "
                f"hh={e_hh_so:+.12f} (hh-MP2 {e_hh_so - e_mp2_so:+.3e}) "
                + " ".join(f"{k}={v:.1e}" for k, v in checks.items())
            )
    print(f"GEN_LINLCCD_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
