"""numpy references for the VALIDATION.md "U-COHSEX" row (validation tier W3).

Consumer: crates/ferric-gw/tests/validation_u_cohsex.rs.
Output:   testdata/reference/validation/u_cohsex/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-gw/src/u_cohsex.rs, cohsex.rs,
w_pdep.rs, mo_b.rs and ferric-rpa lib.rs / sternheimer.rs) AND HOW THIS SCRIPT
MATCHES IT
------------------------------------------------------------------------------
Entry point: `run_u_gw` (ferric-gw lib.rs) with `GwMethod::Cohsex` dispatches to
`u_cohsex::run_u_cohsex` after `run_u_pdep_rpa` and `build_full_b_both_spins`.

  * W is ONE spin-summed static screened interaction shared by both spins
    (u_cohsex.rs:26 `static_weights(&pdep.eigenvalues_static)`, used for both
    spins at :46-47). Its dielectric is eps(0) = I + Pi_a(0) + Pi_b(0) with a
    per-spin prefactor 2 (sternheimer.rs `dielectric_apply_unrestricted`,
    `build_scale_factors_with_prefactor(.., 2.0)`), assembled from each spin's
    own occupied-virtual B tensor and orbital energies (ferric-rpa lib.rs
    `run_u_pdep_rpa`: eps_a for alpha, eps_b for beta). PySCF
    `ugw_ac.get_rho_response` is the same object with the opposite sign
    convention (e_ia = e_i - e_a, eps = I - Pi); this script builds Pi itself
    and asserts it equals PySCF's function.
  * Static weights: w_alpha = 1/lambda_alpha(0) - 1 (w_pdep.rs:86-87). At
    trunc_thresh = 0 with the Lanczos solver the PDEP basis is the full aux
    space, so sum_alpha w_alpha M_alpha,mn^2 = L_mn^T (eps^-1 - I) L_mn, a
    basis-invariant quadratic form (any square root of the Coulomb metric).
    The script evaluates it BOTH ways (matrix inverse, and eigh + weights as
    ferric does) and asserts they agree.
  * Per spin s (cohsex.rs `cohsex_pieces`, called once per spin with that
    spin's MoB and projection, u_cohsex.rs:46-47):
        dSEX_s(m) = - sum_{i occ in s}   [L^s_mi]^T (eps^-1 - I) L^s_mi
        COH_s(m)  = + 1/2 sum_{p all in s} [L^s_mp]^T (eps^-1 - I) L^s_mp
    The SEX sum runs over the n_occ_act occupied orbitals OF THAT SPIN
    (mo_b.n_occ_act, nocc_a / nocc_b from mo_b.rs build_full_b_both_spins);
    the COH sum over ALL active orbitals of that spin.
  * QP energy (u_cohsex.rs:75-78): eps_qp_s = eps_mf_s + dSEX_s + COH_s. The
    bare exchange is NOT replaced (HF reference: Sigma_x - v_x = 0), it is
    reported separately as sigma_x_s = -sum_{P, i occ in s} L^s_{P,mi}^2
    (cohsex.rs `sigma_x_diag`). `sigma_c` in UGwResult is dSEX + COH.
  * Z = 1, no frequency integral, no Pade: closed form. W's frequency grid
    (PdepRpaConfig.quadrature) is not read by COHSEX.
  * Frozen core: none (GwConfig.frozen_core = PdepRpaConfig.frozen_core = 0).
  * QP window: HOMO-2 .. LUMO+2 of the spin with more electrons
    (lib.rs default_u_qp_range; gen_gw.window), same indices for both spins.
  * SCF: exact 4-index UHF, stability-checked (common.run_open_shell), the
    U-G0W0 row's state. Basis and aux: ferric's bundled JSON (common.py), AO
    and aux counts checked against ferric's parser.

INDEPENDENT CROSS-CHECKS (asserted here, recorded in `checks`)
  1. Pi(0): this script's spin-summed Pi equals PySCF ugw_ac.get_rho_response.
  2. W route: inverse-matrix route == eigh/weights route (ferric's PDEP form).
  3. Closed-shell limit (h2o): numpy U-COHSEX on a singlet UHF equals
     gen_gw.cohsex_block (the closed-shell COHSEX formula, factor-4 Pi) on the
     RHF, for both spins, AND equals the committed closed-shell reference
     testdata/reference/validation/gw/h2o_cc-pvdz.json `cohsex_hf`.

NEGATIVE-CONTROL VARIANTS (stored under `controls`; ferric must MISS them)
  * alpha_only_w: W from eps = I + Pi_a only (the beta polarizability dropped).
  * closed_formula_per_spin: each spin's Sigma through the closed-shell COHSEX
    formula applied to that spin's channel alone (Pi = 4 * the spin's
    occupied-virtual block, i.e. 2 Pi_s), as if the open shell were an RHF
    built from that spin's orbitals.

Run (light; seconds per system):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_u_cohsex.py [system ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_gw  # noqa: E402

ROW = "u_cohsex"

# system -> (xyz, multiplicity, bases). The U-G0W0 row's open-shell set.
OPEN = {
    "oh": ("oh.xyz", 2, ("cc-pvdz", "aug-cc-pvdz")),
    "ch3": ("ch3.xyz", 2, ("cc-pvdz", "aug-cc-pvdz")),
    "nh2": ("nh2.xyz", 2, ("cc-pvdz", "aug-cc-pvdz")),
    "o2": ("o2.xyz", 3, ("aug-cc-pvdz",)),
    "ch2_triplet": ("ch2_triplet.xyz", 3, ("aug-cc-pvdz",)),
}
# Closed-shell anchor: U-COHSEX on a singlet UHF == closed-shell COHSEX.
ANCHOR = {"h2o": ("h2o.xyz", ("cc-pvdz",))}

TOL_PI = 1e-12  # own Pi vs PySCF get_rho_response (same arithmetic)
TOL_ROUTE = 1e-10  # inverse route vs eigh route
TOL_CLOSED_LIMIT = 1e-8  # singlet UHF vs RHF (SCF convergence floor)
TOL_COMMITTED = 1e-8  # vs the committed gw/h2o_cc-pvdz.json cohsex_hf


def _np():
    import numpy as np

    return np


def df_lpq(mol, aux, mo_coeffs):
    """L^s_{P,pq} for each MO coefficient matrix, from PySCF's in-core DF
    Cholesky vectors with ferric's aux (independent of the GW module)."""
    np = _np()
    from pyscf import df, lib

    with_df = df.DF(mol, auxbasis=aux)
    with_df.build()
    cderi = np.asarray(with_df._cderi)
    naux = cderi.shape[0]
    ao = lib.unpack_tril(cderi).reshape(naux, mol.nao_nr(), mol.nao_nr())
    out = [np.einsum("Pmn,mp,nq->Ppq", ao, c, c, optimize=True) for c in mo_coeffs]
    return with_df, out


def pi_static(lpq, e, nocc, prefactor):
    """Pi(0) = prefactor * sum_ia L_ia L_ia / (e_a - e_i)  (positive, ferric sign)."""
    np = _np()
    lia = lpq[:, :nocc, nocc:]
    eia = e[nocc:][None, :] - e[:nocc][:, None]
    return prefactor * np.einsum("Pia,Qia->PQ", lia / eia[None], lia, optimize=True)


def w_reduced(pi):
    np = _np()
    n = pi.shape[0]
    return np.linalg.inv(np.eye(n) + pi) - np.eye(n)


def w_reduced_eigh(pi):
    """ferric's form: eigh of eps, w_alpha = 1/lambda_alpha - 1 (w_pdep.rs)."""
    np = _np()
    lam, u = np.linalg.eigh(np.eye(pi.shape[0]) + pi)
    return (u * (1.0 / lam - 1.0)) @ u.T, lam


def spin_cohsex(lpq, e, nocc, wr, orbs):
    np = _np()
    dsex, coh, sx, eqp = [], [], [], []
    for m in orbs:
        lm = lpq[:, m, :]
        diag = np.einsum("Pn,Pn->n", lm, wr @ lm)
        d = -float(np.sum(diag[:nocc]))
        c = 0.5 * float(np.sum(diag))
        dsex.append(d)
        coh.append(c)
        sx.append(-float(np.sum(lm[:, :nocc] ** 2)))
        eqp.append(float(e[m]) + d + c)
    return {
        "eps_mf": [float(e[m]) for m in orbs],
        "delta_sigma_sex": dsex,
        "sigma_coh": coh,
        "sigma_c": [a + b for a, b in zip(dsex, coh)],
        "sigma_x_df": sx,
        "eps_qp": eqp,
    }


def u_cohsex(mf, lpqs, orbs, nelec):
    """Returns (block, controls, checks) for a UHF `mf`."""
    np = _np()
    from pyscf.gw.ugw_ac import get_rho_response

    ea, eb = (np.asarray(x, dtype=float) for x in mf.mo_energy)
    na, nb = nelec
    pa = pi_static(lpqs[0], ea, na, 2.0)
    pb = pi_static(lpqs[1], eb, nb, 2.0)
    pi = pa + pb
    py = get_rho_response(
        0.0,
        np.asarray([ea, eb]),
        np.ascontiguousarray(lpqs[0][:, :na, na:]),
        np.ascontiguousarray(lpqs[1][:, :nb, nb:]),
    )
    pi_dev = float(np.max(np.abs(pi + py)))  # PySCF sign: eps = I - Pi
    assert pi_dev < TOL_PI, f"Pi(0) differs from PySCF get_rho_response: {pi_dev:.2e}"
    wr = w_reduced(pi)
    wr_e, lam = w_reduced_eigh(pi)
    route_dev = float(np.max(np.abs(wr - wr_e)))
    assert route_dev < TOL_ROUTE, f"inverse vs eigh W route: {route_dev:.2e}"
    block = {
        "orbs": list(orbs),
        "nocc": [int(na), int(nb)],
        "alpha": spin_cohsex(lpqs[0], ea, na, wr, orbs),
        "beta": spin_cohsex(lpqs[1], eb, nb, wr, orbs),
        "dielectric_eig_min": float(lam.min()),
        "dielectric_eig_max": float(lam.max()),
    }
    wr_a = w_reduced(pa)
    controls = {
        "alpha_only_w": {
            "what": "W from eps = I + Pi_alpha (beta polarizability dropped)",
            "alpha_eps_qp": spin_cohsex(lpqs[0], ea, na, wr_a, orbs)["eps_qp"],
            "beta_eps_qp": spin_cohsex(lpqs[1], eb, nb, wr_a, orbs)["eps_qp"],
        },
        "closed_formula_per_spin": {
            "what": "closed-shell COHSEX (Pi prefactor 4) on each spin's channel alone",
            "alpha_eps_qp": spin_cohsex(
                lpqs[0], ea, na, w_reduced(pi_static(lpqs[0], ea, na, 4.0)), orbs
            )["eps_qp"],
            "beta_eps_qp": spin_cohsex(
                lpqs[1], eb, nb, w_reduced(pi_static(lpqs[1], eb, nb, 4.0)), orbs
            )["eps_qp"],
        },
    }
    checks = {
        "pi_vs_pyscf_get_rho_response_max_abs": pi_dev,
        "w_inverse_vs_eigh_route_max_abs": route_dev,
    }
    return block, controls, checks


def _max_dev(a, b):
    return float(max(abs(x - y) for x, y in zip(a, b)))


def _provenance(xyz, basis_name, symbols, coords, aux_name, stability, extra):
    import numpy
    import pyscf

    return common.provenance(
        code="numpy on PySCF DF integrals",
        version=pyscf.__version__,
        keywords={
            "method": "static COHSEX, spin-unrestricted, HF reference",
            "w": "spin-summed static RPA W, eps = I + Pi_a + Pi_b (prefactor 2 per spin), full rank",
            "qp": "eps_mf + dSEX + COH (Z = 1)",
            "scf": "exact 4-index ERIs, conv_tol 1e-11, conv_tol_grad 1e-8",
            "numpy": numpy.__version__,
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        aux=aux_name,
        frozen_core="none",
        scf_conv={"conv_tol": gen_gw.CONV_TOL, "conv_tol_grad": gen_gw.CONV_TOL_GRAD},
        stability=stability,
        generator="scripts/validation/gen_u_cohsex.py",
        extra=extra,
    )


def _uhf(mol):
    return common.run_open_shell(
        mol,
        "uhf",
        conv_tol=gen_gw.CONV_TOL,
        conv_tol_grad=gen_gw.CONV_TOL_GRAD,
        return_mf=True,
    )


def gen_open(system):
    xyz_rel, mult, bases = OPEN[system]
    xyz = common.MOL_DIR / xyz_rel
    for basis_name in bases:
        aux_name = gen_gw.AUX_FOR[basis_name]
        mol, symbols, coords, ll = gen_gw.mol_and_prov_base(xyz, basis_name, mult=mult)
        aux = gen_gw.aux_dict(aux_name, symbols)
        uhf, mf = _uhf(mol)
        na, nb = mol.nelec
        orbs = gen_gw.window(max(na, nb), mol.nao_nr())
        with_df, lpqs = df_lpq(mol, aux, mf.mo_coeff)
        naux = gen_gw.check_naux(with_df, aux_name, symbols)
        block, controls, checks = u_cohsex(mf, lpqs, orbs, (na, nb))
        payload = gen_gw.header(mol, basis_name, aux_name, naux, ll)
        payload["uhf"] = uhf
        payload["u_cohsex_uhf"] = block
        payload["controls"] = controls
        payload["checks"] = checks
        payload["provenance"] = _provenance(
            xyz,
            basis_name,
            symbols,
            coords,
            aux_name,
            {"uhf": uhf["stability"]},
            {"blocks": ["uhf", "u_cohsex_uhf", "controls", "checks"]},
        )
        path = common.write_reference(ROW, system, basis_name, payload)
        ha = 27.211386245988
        print(
            f"{system}/{basis_name}: E_UHF {uhf['energy']:.10f}  "
            f"a-HOMO {block['alpha']['eps_qp'][orbs.index(na - 1)] * ha:.5f} "
            f"b-HOMO {block['beta']['eps_qp'][orbs.index(nb - 1)] * ha:.5f} eV  "
            f"Pi {checks['pi_vs_pyscf_get_rho_response_max_abs']:.1e} "
            f"route {checks['w_inverse_vs_eigh_route_max_abs']:.1e}"
            f" -> {path.relative_to(common.ROOT)}"
        )


def gen_anchor(system):
    xyz_rel, bases = ANCHOR[system]
    xyz = common.MOL_DIR / xyz_rel
    for basis_name in bases:
        aux_name = gen_gw.AUX_FOR[basis_name]
        mol, symbols, coords, ll = gen_gw.mol_and_prov_base(xyz, basis_name)
        aux = gen_gw.aux_dict(aux_name, symbols)
        rhf, rhf_stable = gen_gw.rhf_exact(mol)
        uhf, mf = _uhf(mol)
        na, nb = mol.nelec
        assert na == nb, "anchor must be a singlet"
        s2 = uhf["s_squared"]
        assert abs(s2) < 1e-8, f"singlet UHF is spin-contaminated: <S^2> {s2}"
        de = abs(uhf["energy"] - float(rhf.e_tot))
        assert de < 1e-9, f"singlet UHF did not collapse to RHF: dE {de:.2e}"
        orbs = gen_gw.window(na, mol.nao_nr())
        with_df, lpqs = df_lpq(mol, aux, mf.mo_coeff)
        _, (lpq_r,) = df_lpq(mol, aux, [rhf.mo_coeff])
        naux = gen_gw.check_naux(with_df, aux_name, symbols)
        block, controls, checks = u_cohsex(mf, lpqs, orbs, (na, nb))
        closed = gen_gw.cohsex_block(SimpleNamespace(Lpq=lpq_r, nocc=na), rhf, orbs)
        closed["sigma_c"] = [
            a + b for a, b in zip(closed["delta_sigma_sex"], closed["sigma_coh"])
        ]
        for spin in ("alpha", "beta"):
            d = _max_dev(block[spin]["eps_qp"], closed["eps_qp"])
            checks[f"closed_limit_{spin}_vs_cohsex_block_max_abs"] = d
            assert d < TOL_CLOSED_LIMIT, f"closed-shell limit ({spin}): {d:.2e}"
        committed_path = common.reference_path("gw", system, basis_name)
        committed = json.loads(committed_path.read_text())["cohsex_hf"]
        assert committed["orbs"] == list(orbs), "committed cohsex_hf window differs"
        for spin in ("alpha", "beta"):
            d = _max_dev(block[spin]["eps_qp"], committed["eps_qp"])
            checks[f"closed_limit_{spin}_vs_committed_gw_cohsex_hf_max_abs"] = d
            assert d < TOL_COMMITTED, f"vs committed gw cohsex_hf ({spin}): {d:.2e}"
        payload = gen_gw.header(mol, basis_name, aux_name, naux, ll)
        payload["rhf"] = {
            "energy": float(rhf.e_tot),
            "internal_stable": rhf_stable,
            "mo_energy": [float(x) for x in rhf.mo_energy],
        }
        payload["uhf"] = uhf
        payload["u_cohsex_uhf"] = block
        payload["cohsex_rhf"] = closed
        payload["controls"] = controls
        payload["checks"] = checks
        payload["provenance"] = _provenance(
            xyz,
            basis_name,
            symbols,
            coords,
            aux_name,
            {"uhf": uhf["stability"], "rhf_internal_stable": rhf_stable},
            {
                "blocks": [
                    "rhf",
                    "uhf",
                    "u_cohsex_uhf",
                    "cohsex_rhf",
                    "controls",
                    "checks",
                ],
                "closed_shell_anchor": "u_cohsex_uhf (singlet UHF) == cohsex_rhf "
                "(gen_gw.cohsex_block on the RHF) == gw/h2o_cc-pvdz.json cohsex_hf",
            },
        )
        path = common.write_reference(ROW, system, basis_name, payload)
        worst = max(v for k, v in checks.items() if k.startswith("closed_limit"))
        print(
            f"{system}/{basis_name}: singlet UHF == RHF COHSEX to {worst:.1e} Ha"
            f" (dE_scf {de:.1e}) -> {path.relative_to(common.ROOT)}"
        )


def main(argv):
    known = set(OPEN) | set(ANCHOR)
    want = set(argv) or known
    unknown = want - known
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)} (known: {sorted(known)})")
    for s in ANCHOR:
        if s in want:
            gen_anchor(s)
    for s in OPEN:
        if s in want:
            gen_open(s)


if __name__ == "__main__":
    main(sys.argv[1:])
