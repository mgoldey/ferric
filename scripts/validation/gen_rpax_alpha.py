"""numpy RPAx static-polarizability references for the VALIDATION.md row
"TDHF / RPAx polarizability" (CLI kind `tdhf-static-polarizability`).

Consumer: crates/ferric-gw/tests/validation_rpax_alpha.rs
Output:   testdata/reference/validation/rpax_alpha/<system>_<basis>_<ref>.json

PySCF has no screened RPAx. The reference is an INDEPENDENT numpy assembly of
the kernel ferric uses, built from PySCF ingredients (libcint integrals,
PySCF SCF, PySCF's density-fitting tensor with ferric's aux basis). Nothing in
it calls ferric.

WHAT FERRIC COMPUTES (read from crates/ferric-gw/src/bse.rs
`run_rpax_static_polarizability`, closed shell, frozen_core = 0 here)
------------------------------------------------------------------------------
  * eps = the reference's orbital energies, with `scissor` added to every
    VIRTUAL energy. The shifted copy is used ONLY on the A+-B diagonal;
    `run_pdep_rpa` reads the UNSHIFTED reference, so W is built at the
    unshifted gap.
  * W(0): (pq|W|rs) = (pq|rs) + sum_a (1/lambda_a(0) - 1) M_a,pq M_a,rs over
    the PDEP modes of the static dielectric eps~ = I + Pi(0). At
    trunc_thresh = 0 that is L_pq^T (I + Pi(0))^-1 L_rs, with
    Pi(0)_PQ = 4 sum_ia L_P,ia L_Q,ia / (eps_a - eps_i), all occ x all vir,
    UNSHIFTED mean-field energies.
  * (A+B)_{ia,jb} = d eps d + 4 (ia|jb) - (ab|W|ij) - (ib|W|aj)
    (A-B)_{ia,jb} = d eps d +            (ib|W|aj) - (ab|W|ij)
    with (ia|jb) the bare DF integral; flat ia = i * nvir + a.
  * alpha_xy = 4 mu_x^T t_y, (A-B)(A+B) t_y = (A-B) mu_y, mu = <i|r|a>
    (origin 0). Algebraically t = (A+B)^-1 mu: the static alpha does NOT
    depend on A-B at all. Both routes are computed and their difference is
    recorded (`alpha_route_identity_max`).
  * The library refuses any alpha_dd <= 1e-8 (`check_alpha_diagonal_positive`).

BLOCKS PER FILE
------------------------------------------------------------------------------
  screened  - the reference: full 3x3 alpha, iso, min eig of A+B and A-B.
  bare      - the same assembly with W -> v (the TDHF kernel): a control the
              screened ferric result must MISS, and the target of ferric's
              bare-kernel hook.
  anchors   - (RHF references) the exactness anchor, asserted HERE before the
              file is written:
                (a) numpy bare alpha on a DENSITY-FITTED RHF (same aux) equals
                    PySCF DF-CPHF alpha (the CPHF matrix built from PySCF's
                    `mf.gen_response(hermi=1)`, dense solve) to <= 1e-9
                    relative; PySCF's iterative `scf.cphf.solve` on the same
                    operator is recorded beside it (it stops ~5e-8 short);
                (b) numpy bare alpha with EXACT MO ERIs on the exact RHF
                    equals PySCF exact CPHF alpha to <= 1e-9 relative.
              Recorded: the DF-vs-exact gap of the bare alpha on the exact
              RHF (the size of the fitting error, not a defect).
  scissor0  - (PBE) the screened alpha at scissor 0, the documented
              excitonic-instability case: recorded, either sign.

Run (whole-CPU slot, as every validation step):
    scripts/validation/run_slot.sh -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_rpax_alpha.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_gw  # noqa: E402  (aux_dict, rhf_exact, rks_pbe_exact, header)

ROW = "rpax_alpha"
ANCHOR_REL_TOL = 1e-9
CPHF_TOL = 1e-13
CPHF_MAX_CYCLE = 400
# ferric's documented physical setting for water at PBE/cc-pVDZ (the shift
# that matches its GW gap); bse.rs `run_rpax_static_polarizability` docs.
PBE_SCISSOR = 0.36

# (system, basis, reference, scissor)
CASES = [
    ("h2o", "aug-cc-pvdz", "rhf", 0.0),
    ("nh3", "aug-cc-pvdz", "rhf", 0.0),
    ("h2o", "cc-pvdz", "pbe", PBE_SCISSOR),
]


# ---------------------------------------------------------------------------
# Ingredients
# ---------------------------------------------------------------------------


def lpq_mo(mol, aux, c):
    """L_P,pq in the MO basis: (pq|rs) = sum_P L_P,pq L_P,rs (Coulomb metric)."""
    from pyscf import df, lib

    dfobj = df.DF(mol, auxbasis=aux)
    dfobj.verbose = 0
    dfobj.build()
    naux = int(dfobj.get_naoaux())
    nmo = c.shape[1]
    out = np.empty((naux, nmo, nmo))
    p0 = 0
    for cderi in dfobj.loop():
        l_ao = lib.unpack_tril(np.asarray(cderi))
        p1 = p0 + l_ao.shape[0]
        out[p0:p1] = np.einsum("Puv,up,vq->Ppq", l_ao, c, c, optimize=True)
        p0 = p1
    assert p0 == naux
    return out, naux


def dipole_ia(mol, c, nocc):
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r_ao = mol.intor_symmetric("int1e_r", comp=3)
    return np.einsum("up,duv,vq->dpq", c[:, :nocc], r_ao, c[:, nocc:]).reshape(3, -1)


def screening_metric(lpq, e_mf, nocc):
    """(I + Pi(0))^-1 on UNSHIFTED mean-field energies, and its eigenvalues."""
    naux = lpq.shape[0]
    l_ia = lpq[:, :nocc, nocc:]
    d = e_mf[None, nocc:] - e_mf[:nocc, None]
    eps_t = np.eye(naux) + 4.0 * np.einsum("Pia,Qia,ia->PQ", l_ia, l_ia, 1.0 / d)
    lam = np.linalg.eigvalsh(eps_t)
    winv = np.linalg.inv(eps_t)
    return 0.5 * (winv + winv.T), lam


def apb_amb_df(lpq, eps_diag, nocc, metric):
    """A+B and A-B as the bse.rs loop builds them. `metric` = None is W -> v."""
    nmo = lpq.shape[1]
    nvir = nmo - nocc
    n = nocc * nvir
    l_ia = lpq[:, :nocc, nocc:]
    l_ij = lpq[:, :nocc, :nocc]
    l_ab = lpq[:, nocc:, nocc:]
    l_ai = lpq[:, nocc:, :nocc]
    wl_ij = l_ij if metric is None else np.einsum("PQ,Qij->Pij", metric, l_ij)
    wl_ai = l_ai if metric is None else np.einsum("PQ,Qaj->Paj", metric, l_ai)
    coul = np.einsum("Pia,Pjb->iajb", l_ia, l_ia).reshape(n, n)
    # (ab|W|ij) at [i,a,j,b]
    w_abij = np.einsum("Pab,Pij->iajb", l_ab, wl_ij).reshape(n, n)
    # (ib|W|aj) at [i,a,j,b]
    w_ibaj = np.einsum("Pib,Paj->iajb", l_ia, wl_ai).reshape(n, n)
    return _assemble(coul, w_abij, w_ibaj, eps_diag, nocc)


def apb_amb_exact(eri_mo, eps_diag, nocc):
    """Bare kernel from exact 4-index MO integrals (chemist notation)."""
    nmo = eri_mo.shape[0]
    nvir = nmo - nocc
    n = nocc * nvir
    o, v = slice(0, nocc), slice(nocc, nmo)
    coul = eri_mo[o, v, o, v].reshape(n, n)
    w_abij = eri_mo[v, v, o, o].transpose(2, 0, 3, 1).reshape(n, n)
    w_ibaj = eri_mo[o, v, v, o].transpose(0, 2, 3, 1).reshape(n, n)
    return _assemble(coul, w_abij, w_ibaj, eps_diag, nocc)


def _assemble(coul, w_abij, w_ibaj, eps_diag, nocc):
    d = (eps_diag[None, nocc:] - eps_diag[:nocc, None]).ravel()
    apb = 4.0 * coul - w_abij - w_ibaj
    amb = w_ibaj - w_abij
    apb[np.diag_indices_from(apb)] += d
    amb[np.diag_indices_from(amb)] += d
    for m, name in ((apb, "A+B"), (amb, "A-B")):
        asym = float(np.max(np.abs(m - m.T)))
        assert asym < 1e-11, f"{name} not symmetric: {asym:.2e}"
    return apb, amb


def alpha_block(apb, amb, mu):
    """ferric's route ((A-B)(A+B) t = (A-B) mu) and the direct (A+B)^-1 route."""
    sysm = amb @ apb
    t = np.linalg.solve(sysm, amb @ mu.T)
    a_ferric = 4.0 * mu @ t
    a_direct = 4.0 * mu @ np.linalg.solve(apb, mu.T)
    route = float(np.max(np.abs(a_ferric - a_direct)) / np.max(np.abs(a_direct)))
    return {
        "tensor": [[float(x) for x in row] for row in a_ferric],
        "iso": float(np.trace(a_ferric) / 3.0),
        "tensor_direct_apb_inverse": [[float(x) for x in row] for row in a_direct],
        "alpha_route_identity_max_rel": route,
        "min_eig_apb": float(np.linalg.eigvalsh(apb)[0]),
        "min_eig_amb": float(np.linalg.eigvalsh(amb)[0]),
    }


def max_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---------------------------------------------------------------------------
# PySCF CPHF (the independent response code for the bare-limit anchor)
# ---------------------------------------------------------------------------


def cphf_alpha(mf):
    """PySCF CPHF alpha two ways, from PySCF's own response operator
    (`mf.gen_response(hermi=1)`, i.e. PySCF's J/K code, DF or exact as `mf`):

      dense    - the CPHF matrix (e_a - e_i) + v[.] built column by column from
                 that operator on unit vectors, then a dense solve. This is
                 the anchor value: it is the exact CPHF solution.
      solver   - `scf.cphf.solve` (PySCF's iterative Krylov solver) on the same
                 operator. Its stopping rule leaves ~5e-8 relative in alpha
                 even at tol 1e-13 (measured on H2O/aug-cc-pVDZ, where the
                 dense matrix equals the numpy A+B to 1.3e-14), so it is
                 recorded, not used as the anchor.

    alpha = -4 sum_ai h_ai x_ai with (e_a - e_i) x + v[x] = -h, h = <a|r|i>.
    """
    from pyscf.scf import cphf

    mol = mf.mol
    c = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ) > 0
    orbo, orbv = c[:, occ], c[:, ~occ]
    nocc, nvir = orbo.shape[1], orbv.shape[1]
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r_ao = mol.intor_symmetric("int1e_r", comp=3)
    h1 = np.einsum("xpq,pa,qi->xai", r_ao, orbv, orbo)
    vresp = mf.gen_response(hermi=1)

    def fvind(x):
        x = np.asarray(x).reshape(-1, nvir, nocc)
        dm1 = np.einsum("xai,pa,qi->xpq", x, orbv, orbo * 2.0)
        dm1 = dm1 + dm1.transpose(0, 2, 1)
        v1 = vresp(dm1)
        return np.einsum("xpq,pa,qi->xai", v1, orbv, orbo).reshape(x.shape)

    e = np.asarray(mf.mo_energy)
    d_ai = (e[~occ][:, None] - e[occ][None, :]).ravel()
    n = nvir * nocc
    m = fvind(np.eye(n).reshape(n, nvir, nocc)).reshape(n, n).T + np.diag(d_ai)
    sym = float(np.max(np.abs(m - m.T)))
    assert sym < 1e-11, f"PySCF CPHF matrix not symmetric: {sym:.2e}"
    x = np.linalg.solve(m, -h1.reshape(3, n).T)
    a_dense = -4.0 * h1.reshape(3, n) @ x
    mo1 = cphf.solve(
        fvind,
        mf.mo_energy,
        mf.mo_occ,
        h1,
        None,
        max_cycle=CPHF_MAX_CYCLE,
        tol=CPHF_TOL,
    )[0]
    a_iter = -4.0 * np.einsum("xai,yai->xy", h1, mo1)
    return 0.5 * (a_dense + a_dense.T), 0.5 * (a_iter + a_iter.T)


def df_rhf(mol, aux):
    from pyscf import scf

    mf = scf.RHF(mol).density_fit(auxbasis=aux)
    mf.conv_tol = gen_gw.CONV_TOL
    mf.conv_tol_grad = gen_gw.CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, "DF-RHF did not converge"
    return mf


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def gen_one(system, basis_name, ref, scissor):
    from pyscf import ao2mo

    xyz = common.MOL_DIR / f"{system}.xyz"
    aux_name = gen_gw.AUX_FOR[basis_name]
    mol, symbols, coords, ll = gen_gw.mol_and_prov_base(xyz, basis_name)
    aux = gen_gw.aux_dict(aux_name, symbols)
    nocc = mol.nelectron // 2

    if ref == "rhf":
        mf, stable = gen_gw.rhf_exact(mol)
        scf_block = {"kind": "RHF", "j_k": "exact 4-index", "energy": float(mf.e_tot)}
        stability = {"rhf_internal_stable": stable}
    else:
        mf = gen_gw.rks_pbe_exact(mol)
        scf_block = {
            "kind": "RKS",
            "xc": "PBE",
            "j": "exact (no density fitting)",
            "grid": f"{gen_gw.MAIN_GRID} unpruned, Becke partition",
            "density_floor": gen_gw.FERRIC_DENSITY_FLOOR,
            "energy": float(mf.e_tot),
        }
        stability = None
    c = np.asarray(mf.mo_coeff)
    e_mf = np.asarray(mf.mo_energy, float)
    nmo = e_mf.size
    nvir = nmo - nocc
    scf_block["mo_energy"] = [float(x) for x in e_mf]

    lpq, naux = lpq_mo(mol, aux, c)
    assert naux == common.ferric_nao(aux_name, symbols), "aux count differs"
    mu = dipole_ia(mol, c, nocc)
    metric, lam = screening_metric(lpq, e_mf, nocc)
    assert lam.min() >= 1.0 - 1e-12, f"static dielectric eigenvalue {lam.min()} < 1"

    eps_diag = e_mf.copy()
    eps_diag[nocc:] += scissor

    screened = alpha_block(*apb_amb_df(lpq, eps_diag, nocc, metric), mu)
    bare = alpha_block(*apb_amb_df(lpq, eps_diag, nocc, None), mu)
    payload = gen_gw.header(mol, basis_name, aux_name, naux, ll)
    payload.update(
        {
            "reference": ref,
            "scissor": scissor,
            "nocc": nocc,
            "nvir": nvir,
            "scf": scf_block,
            "w_static": {
                "lambda_min": float(lam.min()),
                "lambda_max": float(lam.max()),
                "n_modes": int(naux),
                "pi0": "4 sum_ia L_ia L_ia^T / (eps_a - eps_i), UNSHIFTED "
                "mean-field energies, all occ x all vir",
            },
            "screened": screened,
            "bare": bare,
            "screened_vs_bare_max_rel": max_rel(screened["tensor"], bare["tensor"]),
        }
    )

    if ref == "rhf":
        # (a) bare numpy vs PySCF DF-CPHF on a DF-RHF with the SAME aux.
        mf_df = df_rhf(mol, aux)
        c_df = np.asarray(mf_df.mo_coeff)
        lpq_df, _ = lpq_mo(mol, aux, c_df)
        bare_df = alpha_block(
            *apb_amb_df(lpq_df, np.asarray(mf_df.mo_energy), nocc, None),
            dipole_ia(mol, c_df, nocc),
        )
        cphf_df, cphf_df_iter = cphf_alpha(mf_df)
        a_rel = max_rel(bare_df["tensor"], cphf_df)
        assert a_rel <= ANCHOR_REL_TOL, f"anchor (a) DF: {a_rel:.2e}"
        # (b) bare numpy with EXACT MO ERIs vs PySCF exact CPHF.
        eri_mo = ao2mo.restore(1, ao2mo.full(mol, c), nmo)
        bare_ex = alpha_block(*apb_amb_exact(eri_mo, e_mf, nocc), mu)
        cphf_ex, cphf_ex_iter = cphf_alpha(mf)
        b_rel = max_rel(bare_ex["tensor"], cphf_ex)
        assert b_rel <= ANCHOR_REL_TOL, f"anchor (b) exact: {b_rel:.2e}"
        payload["anchors"] = {
            "a_df_bare_numpy_vs_pyscf_df_cphf_max_rel": a_rel,
            "a_df_rhf_energy": float(mf_df.e_tot),
            "a_df_cphf_tensor": [[float(x) for x in r] for r in cphf_df],
            "b_exact_bare_numpy_vs_pyscf_exact_cphf_max_rel": b_rel,
            "b_exact_cphf_tensor": [[float(x) for x in r] for r in cphf_ex],
            "cphf_matrix": "dense, built from PySCF gen_response(hermi=1)",
            "a_df_cphf_iterative_solver_vs_dense_max_rel": max_rel(
                cphf_df_iter, cphf_df
            ),
            "b_exact_cphf_iterative_solver_vs_dense_max_rel": max_rel(
                cphf_ex_iter, cphf_ex
            ),
            "tolerance_rel": ANCHOR_REL_TOL,
            "cphf_tol": CPHF_TOL,
            "df_fitting_gap_bare_df_vs_exact_cphf_max_rel": max_rel(
                bare["tensor"], cphf_ex
            ),
        }
    else:
        e0 = alpha_block(*apb_amb_df(lpq, e_mf, nocc, metric), mu)
        payload["scissor0"] = e0
        payload["scissor0_negative_diagonal"] = [
            d for d in range(3) if not e0["tensor"][d][d] > 1e-8
        ]

    payload["provenance"] = common.provenance(
        code="PySCF + numpy",
        version=__import__("pyscf").__version__,
        keywords={
            "scf": "conv_tol 1e-11, conv_tol_grad 1e-8",
            "df": "pyscf.df.DF with ferric's aux JSON (Coulomb metric)",
            "cphf": f"dense matrix from gen_response(hermi=1); also scf.cphf.solve tol {CPHF_TOL}",
            "numpy": np.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=list(gen_gw.MAIN_GRID) if ref == "pbe" else None,
        aux=aux_name,
        frozen_core=None,
        scf_conv={"conv_tol": gen_gw.CONV_TOL, "conv_tol_grad": gen_gw.CONV_TOL_GRAD},
        stability=stability,
        generator="scripts/validation/gen_rpax_alpha.py",
    )
    path = common.write_reference(ROW, system, f"{basis_name}_{ref}", payload)
    print(
        f"{system}/{basis_name}/{ref} scissor {scissor}: E {mf.e_tot:.10f} "
        f"nocc {nocc} nvir {nvir} naux {naux} lambda_max {lam.max():.4f}"
    )
    print(
        f"   screened iso {screened['iso']:.8f} diag "
        f"{[round(screened['tensor'][d][d], 6) for d in range(3)]} "
        f"min eig A+B {screened['min_eig_apb']:.6f} A-B {screened['min_eig_amb']:.6f} "
        f"route {screened['alpha_route_identity_max_rel']:.1e}"
    )
    print(
        f"   bare     iso {bare['iso']:.8f}; screened vs bare "
        f"{payload['screened_vs_bare_max_rel']:.3e} rel"
    )
    if "anchors" in payload:
        a = payload["anchors"]
        print(
            f"   anchor (a) DF {a['a_df_bare_numpy_vs_pyscf_df_cphf_max_rel']:.2e} "
            f"(b) exact {a['b_exact_bare_numpy_vs_pyscf_exact_cphf_max_rel']:.2e} "
            f"DF gap {a['df_fitting_gap_bare_df_vs_exact_cphf_max_rel']:.2e}; "
            f"cphf.solve vs dense {a['a_df_cphf_iterative_solver_vs_dense_max_rel']:.1e}"
        )
    if "scissor0" in payload:
        print(
            f"   scissor 0 diag {[round(payload['scissor0']['tensor'][d][d], 6) for d in range(3)]}"
            f" iso {payload['scissor0']['iso']:.6f}"
        )
    print(f"   -> {path.relative_to(common.ROOT)}")


def main(argv):
    want = set(argv)
    for case in CASES:
        if not want or case[0] in want:
            gen_one(*case)


if __name__ == "__main__":
    main(sys.argv[1:])
