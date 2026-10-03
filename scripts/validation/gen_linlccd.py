"""PySCF + numpy references for the VALIDATION.md "LinLCCD(hh)" and
"Amplitude-threshold LinLCCD" rows.

Consumers: crates/ferric-cc/tests/validation_linlccd.rs (canonical hh),
           crates/ferric-cc/tests/validation_linlccd_amplitude.rs (local,
           all three tiers incl. the `linlccd_full` block)
Output:    testdata/reference/validation/linlccd/<system>_<basis>.json

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

FULL LinLCCD (closed shell only, block `linlccd_full`): ferric's
`LadderVariant::Full` (linlccd.rs, eq. 7) adds the particle–particle ladder
½ Σ_cd <ab||cd> t_ij^cd. The hh ladder dresses only the occupied-pair index
and the pp ladder only the virtual-pair index, so the equation is the
Sylvester equation H_occ T − T H_vir = v with H_occ = diag(ε_i + ε_j) − W_oo
and H_vir = diag(ε_a + ε_b) + W_vv, solved exactly in the two eigenbases
(solve_sylvester). Built twice — spin-orbital (½<ab||cd>) and spin-adapted
(Σ_cd (ac|bd) T_ijcd) — and refused unless they agree to 1e-12; the
Sylvester solver with W_vv = 0 must reproduce the pair-eigenbasis hh solve
to 1e-12 (`checks/full_solver_pp_off_vs_hh`).

NOT an anchor: full LinLCCD is NOT exact for two-electron systems (the
paper, §3.1: "LinLCCD is no longer exact for all two electron systems").
Measured 2026-10-02 on the same DF integrals: H2/STO-3G full LinLCCD
−0.008498 vs PySCF CCD (t1 frozen at 0) −0.020585 Eh; H2/cc-pVDZ −0.017450
vs −0.034609 Eh. The ring terms LinLCCD drops are what CCD needs there, so
no H2-vs-CCD anchor is used.

Existing references are MERGED, not rewritten: PySCF's RHF is not
bit-reproducible run to run, so every committed field is kept verbatim and
only new fields are added (merge_into_committed; `--fresh` overrides).

Anchors asserted HERE before anything is written:
  * ladder off (MP2) numpy == PySCF `mp.dfmp2.DFMP2` (RHF) or
    `mp.dfump2.DFUMP2` (UHF), frozen=None, to 1e-12;
  * closed shell: spin-adapted == spin-orbital to 1e-12 (MP2 and hh);
  * the solved amplitudes satisfy the amplitude equation (dense residual
    evaluated in spin orbitals) to 1e-11, and the hh correction
    E_hh − E_MP2 is resolvable (> 1e-5 Eh).

Run (light; seconds per system):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_linlccd.py   # the reference env (PySCF)
"""

from __future__ import annotations

import json
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


def solve_sylvester(h_occ, h_vir, drive):
    """Solve H_occ T − T H_vir = drive exactly (both symmetric).

    The full-LinLCCD operator is H_occ ⊗ 1 − 1 ⊗ H_vir on the (occ pair) ×
    (vir pair) amplitude matrix: the hh ladder dresses only the occupied-pair
    index and the pp ladder only the virtual-pair index, so the two
    eigenbases diagonalize it together. Asserts max λ_occ < min μ_vir (the
    operator is then negative definite and the solve is unique).
    Returns (T, gap).
    """
    lam, u = np.linalg.eigh(h_occ)
    mu, v = np.linalg.eigh(h_vir)
    gap = float(mu.min() - lam.max())
    assert gap > 0.0, f"no gap: min(mu) - max(lambda) = {gap}"
    tt = (u.T @ drive @ v) / (lam[:, None] - mu[None, :])
    return u @ tt @ v.T, gap


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


def spin_adapted_full(lov, loo, lvv, eo, ev, pp=True):
    """Full LinLCCD (hh + pp ladders) in spatial orbitals, closed shell:

        D T_ijab = (ia|jb) + Σ_kl (ki|lj) T_klab + Σ_cd (ac|bd) T_ijcd,
        E = Σ_ijab T_ijab [2 (ia|jb) − (ib|ja)].

    The pp term is the spin integration of ½ <ab||cd> t_ij^cd exactly as the
    hh term is that of ½ <kl||ij> t_kl^ab; agreement with spin_orbital(...,
    pp=True), which never spin-integrates, is the proof. `pp=False` runs the
    SAME Sylvester solver with the pp block zeroed (solver check against the
    pair-eigenbasis hh solve).
    """
    no, nv = len(eo), len(ev)
    g = np.einsum("Pia,Pjb->ijab", lov, lov, optimize=True)  # (ia|jb) as [i,j,a,b]
    g_x = g.transpose(0, 1, 3, 2)  # (ib|ja)
    w_oo = np.einsum("Pki,Plj->ijkl", loo, loo, optimize=True).reshape(no * no, no * no)
    h_occ = np.diag((eo[:, None] + eo[None, :]).reshape(-1)) - w_oo
    h_vir = np.diag((ev[:, None] + ev[None, :]).reshape(-1))
    if pp:
        # W_vv[(ab),(cd)] = (ac|bd)
        h_vir = h_vir + np.einsum("Pac,Pbd->abcd", lvv, lvv, optimize=True).reshape(
            nv * nv, nv * nv
        )
    t, gap = solve_sylvester(h_occ, h_vir, g.reshape(no * no, nv * nv))
    t = t.reshape(no, no, nv, nv)
    e = float(np.sum(t * (2.0 * g - g_x)))
    sym = float(np.max(np.abs(t - t.transpose(1, 0, 3, 2))))
    return e, sym, gap


# ---------------------------------------------------------------------------
# Spin-orbital (RHF or UHF)
# ---------------------------------------------------------------------------


def spin_orbital(lao, c_a, c_b, e_a, e_b, na, nb, ladder, pp=False):
    """LinLCCD(hh) / MP2 in spin orbitals from per-spin MOs; `pp=True`
    (requires `ladder`) adds the particle–particle ladder ½ <ab||cd> t_ij^cd —
    full LinLCCD, ferric's `LadderVariant::Full` (linlccd.rs, eq. 7) — and
    solves the coupled hh+pp system exactly with solve_sylvester.

    Spin-orbital order: occ = [α occ..., β occ...], vir = [α vir..., β vir...].
    """
    assert ladder or not pp
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
    if pp:
        lvv = l3(va, vb, va, vb)
        c_vvvv = np.einsum(
            "Pac,Pbd->abcd", lvv, lvv, optimize=True
        )  # (ac|bd) = <ab|cd> as [a,b,c,d]
        v_vvvv = c_vvvv - c_vvvv.transpose(0, 1, 3, 2)  # <ab||cd>
        mv = 0.5 * v_vvvv.reshape(nv * nv, nv * nv)  # [(ab),(cd)]
        assert np.max(np.abs(mv - mv.T)) < 1e-12
        t, _gap = solve_sylvester(
            h_pair, np.diag(s_ab) + mv, v_oovv.reshape(no * no, nv * nv)
        )
    else:
        v_vvvv = None
        t = solve_pair_linear(h_pair, v_oovv.reshape(no * no, nv * nv), s_ab)
    t = t.reshape(no, no, nv, nv)
    e = float(0.25 * np.sum(v_oovv * t))
    # Dense residual of the amplitude equation (independent of the solve).
    r = v_oovv - d * t
    if v_oooo is not None:
        r = r + 0.5 * np.einsum("klij,klab->ijab", v_oooo, t, optimize=True)
    if v_vvvv is not None:
        r = r + 0.5 * np.einsum("abcd,ijcd->ijab", v_vvvv, t, optimize=True)
    return e, float(np.max(np.abs(r)))


def _numeric_leaves(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _numeric_leaves(v, f"{pre}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _numeric_leaves(v, f"{pre}/{i}")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield pre, float(d)


def _keep_committed(old, new):
    """`new`'s key order; every value already in `old` kept verbatim."""
    if not (isinstance(old, dict) and isinstance(new, dict)):
        return old
    return {k: (_keep_committed(old[k], v) if k in old else v) for k, v in new.items()}


def merge_into_committed(system, basis_name, payload):
    """Add only the NEW fields to an existing reference, byte-stable otherwise.

    PySCF's RHF is not bit-reproducible run to run (measured: existing
    energies move by up to ~4e-15 Eh), so a fresh write would churn every
    committed number. Instead every field already in the committed JSON is
    kept verbatim, the new fields are added, and the largest drift of the
    shared numeric fields (outside provenance) is recorded and capped at
    1e-12 so a real change cannot hide behind the merge. `--fresh` skips it.
    """
    path = common.reference_path(ROW, system, basis_name)
    if not path.exists():
        return payload
    old = json.loads(path.read_text())
    old_num = dict(_numeric_leaves({k: v for k, v in old.items() if k != "provenance"}))
    new_num = dict(
        _numeric_leaves({k: v for k, v in payload.items() if k != "provenance"})
    )
    added = sorted(set(new_num) - set(old_num))
    if not added:
        return old
    drift = max(abs(new_num[k] - old_num[k]) for k in old_num if k in new_num)
    if drift > 1e-12:
        raise RuntimeError(f"{system}/{basis_name}: committed fields drift {drift:.2e}")
    merged = _keep_committed(old, payload)
    merged["provenance"]["added_fields"] = {
        "fields": added,
        "git_head": payload["provenance"]["git_head"],
        "generated_utc": payload["provenance"]["generated_utc"],
        "max_abs_drift_of_committed_numeric_fields": drift,
    }
    return merged


def main() -> int:
    import pyscf
    from pyscf import scf

    fresh = "--fresh" in sys.argv[1:]
    only = set(sys.argv[1:]) - {"--fresh"}
    unknown = only - set(SYSTEMS)
    if unknown:
        raise SystemExit(
            f"unknown systems: {sorted(unknown)}; known: {sorted(SYSTEMS)}"
        )
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
            e_full_so = None
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

                # ---- full LinLCCD (hh + pp ladders), closed shell only ----
                lvv = np.einsum("Pmn,ma,nb->Pab", lao, cvir, cvir, optimize=True)
                e_full_so, res_full = spin_orbital(
                    lao, c_a, c_b, e_a, e_b, na, nb, ladder=True, pp=True
                )
                e_full_sa, sym_full, gap_full = spin_adapted_full(
                    lov, loo, lvv, e_a[:na], e_a[na:], pp=True
                )
                e_hh_syl, _, _ = spin_adapted_full(
                    lov, loo, lvv, e_a[:na], e_a[na:], pp=False
                )
                checks.update(
                    {
                        "residual_full_max": res_full,
                        "full_spin_adapted_vs_spin_orbital": abs(e_full_sa - e_full_so),
                        "full_solver_pp_off_vs_hh": abs(e_hh_syl - e_hh_so),
                        "pair_symmetry_max_full": sym_full,
                        "full_operator_gap": gap_full,
                    }
                )
                for k in (
                    "full_spin_adapted_vs_spin_orbital",
                    "full_solver_pp_off_vs_hh",
                ):
                    if checks[k] > TOL_ANCHOR:
                        raise RuntimeError(f"{ctx}: {k} = {checks[k]:.2e}")
                if res_full > TOL_RESIDUAL:
                    raise RuntimeError(f"{ctx}: full residual {res_full:.2e}")
                if abs(e_full_so - e_hh_so) < 1e-5:
                    raise RuntimeError(f"{ctx}: pp ladder correction unresolvable")

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
                **(
                    {}
                    if e_full_so is None
                    else {
                        "linlccd_full": {"e_corr": e_full_so},
                        "full_minus_hh": e_full_so - e_hh_so,
                    }
                ),
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
                        **(
                            {}
                            if unrestricted
                            else {
                                "method_full": "full LinLCCD: D t = <ij||ab> + 1/2 "
                                "sum_kl <kl||ij> t_kl^ab + 1/2 sum_cd <ab||cd> t_ij^cd; "
                                "solved EXACTLY as the Sylvester equation H_occ T - T H_vir "
                                "= <ij||ab> in the joint eigenbasis (no iteration)"
                            }
                        ),
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
            if not fresh:
                payload = merge_into_committed(system, basis_name, payload)
            path = common.write_reference(ROW, system, basis_name, payload)
            written.append(path)
            full_txt = (
                ""
                if e_full_so is None
                else f"full={e_full_so:+.12f} (full-hh {e_full_so - e_hh_so:+.3e}) "
            )
            print(
                f"{ctx:14s} E_scf={mf.e_tot:.12f} MP2={e_mp2_so:+.12f} "
                f"hh={e_hh_so:+.12f} (hh-MP2 {e_hh_so - e_mp2_so:+.3e}) "
                + full_txt
                + " ".join(f"{k}={v:.1e}" for k, v in checks.items())
            )
    print(f"GEN_LINLCCD_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
