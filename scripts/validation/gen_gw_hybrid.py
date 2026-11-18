#!/usr/bin/env python3
"""G0W0 on HYBRID and range-separated-hybrid KS references (validation row
"G0W0@hybrid"): PySCF `gw_ac` vs ferric `run_gw` + `vxc_mo::vxc_diagonal_mo`.

The QP equation at a KS reference is

    eps_qp = eps_KS + (Sigma_x - v_mf)_pp + Sigma_c(eps_qp),

where v_mf is EVERYTHING in the KS Fock besides h + J: the semilocal v_xc AND
the exact-exchange part the hybrid put there,

    v_mf = v_xc^semilocal - (1/2) [ c_sr K_erfc(w)[D] + c_lr K_erf(w)[D] ]

(closed shell, D the total density; c_sr = c_lr = c_x and K_erfc + K_erf = K
for a global hybrid). PySCF `gw_ac` takes v_mf = mf.get_veff() - mf.get_j(),
which contains that term (book/src/19-gw-bse/05-continuation-and-qp-equation.md,
"Hybrid references").

Per functional the block stores, besides gen_gw.run_gwac's usual fields:
  * `v_mf`            the full diagonal (semilocal + exact-exchange part),
  * `v_xc_semilocal`  the semilocal (libxc) diagonal alone (`nr_rks`),
  * `v_k_hybrid`      the exact-exchange diagonal v_mf - v_xc_semilocal,
  * `exact`           the grid-free pole-sum sub-block (gen_gw.closed_exact_block),
  * `exact/eps_qp_hybrid_double_counted`: the same QP root with the shift
    Sigma_x - v_xc_semilocal, i.e. the exact-exchange fraction counted twice
    (ferric before the hybrid fix) -- the size of that defect, from PySCF alone.

Settings are gen_gw.py's @PBE recipe: exact 4-index ERIs, (75,110) unpruned
Becke grid with Becke-1988 radii adjustment, ferric's 1e-14 XC density floor,
`vhf_df = True` (Sigma_x from the same Lpq as ferric), the textbook Thiele
continuation. Functionals by libxc identifier on both sides.

Run (one heavy op at a time):
    box-lease run --label gw-hybrid -- uv run --no-sync python \
        scripts/validation/gen_gw_hybrid.py
-> testdata/reference/validation/gw_hybrid/h2o_cc-pvdz.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_gw as G  # noqa: E402

ROW = "gw_hybrid"
SYSTEM, BASIS = "h2o", "cc-pvdz"
# block name -> (libxc identifier, both codes)
FUNCTIONALS = {
    "g0w0_pbe0": "HYB_GGA_XC_PBEH",
    "g0w0_cam_b3lyp": "HYB_GGA_XC_CAM_B3LYP",
}


def rks_exact(mol, xc):
    from pyscf import dft

    mf = dft.RKS(mol, xc=xc)
    G._apply_density_floor(mf, G.FERRIC_DENSITY_FLOOR)
    mf.grids.atom_grid = G.MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = G.CONV_TOL
    mf.conv_tol_grad = G.CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    assert mf.converged, f"RKS/{xc} did not converge"
    return mf


def hybrid_block(mol, aux, orbs, xc):
    import numpy as np

    mks = rks_exact(mol, xc)
    gw, blk = G.run_gwac(mks, aux, orbs, vhf_df=True, continuation="textbook")
    c = mks.mo_coeff
    dm = mks.make_rdm1()
    _n, _exc, vxc_sl = mks._numint.nr_rks(mol, mks.grids, mks.xc, dm)
    vsl = np.einsum("mp,mn,np->p", c, vxc_sl, c)
    vmf_full = np.einsum("mp,mn,np->p", c, mks.get_veff() - mks.get_j(), c)
    # Independent build of the exact-exchange term from the coefficients.
    omega, c_lr, c_sr = mks._numint.rsh_and_hybrid_coeff(mks.xc, mol.spin)
    if omega == 0:
        vk = -0.5 * c_sr * mks.get_k(dm=dm)
    else:
        vk = -0.5 * (
            c_sr * (mks.get_k(dm=dm, omega=-omega))
            + c_lr * mks.get_k(dm=dm, omega=omega)
        )
    vk_mo = np.einsum("mp,mn,np->p", c, vk, c)
    split = float(np.max(np.abs(vmf_full - (vsl + vk_mo))))
    assert split < 1e-10, f"v_mf != v_xc^sl + v_K: {split:.2e}"
    assert max(abs(vmf_full[p] - v) for p, v in zip(orbs, blk["v_mf"])) < 1e-12
    blk["xc"] = xc
    blk["rsh"] = {"omega": float(omega), "c_sr": float(c_sr), "c_lr": float(c_lr)}
    blk["rks_energy"] = float(mks.e_tot)
    blk["mo_energy"] = [float(x) for x in mks.mo_energy]
    blk["j"] = "exact (no density fitting)"
    blk["density_floor"] = G.FERRIC_DENSITY_FLOOR
    blk["v_xc_semilocal"] = [float(vsl[p]) for p in orbs]
    blk["v_k_hybrid"] = [float(vk_mo[p]) for p in orbs]
    blk["v_mf_split_residual"] = split
    blk["exact"] = G.closed_exact_block(gw, mks, blk, True)
    # The defect, from PySCF alone: Sigma_x - v_xc^sl (exact exchange twice).
    e = np.asarray(mks.mo_energy, dtype=float)
    from pyscf.gw.utils.ac_grid import _get_ac_idx, _get_scaled_legendre_roots

    lpq = np.asarray(gw.Lpq)
    nocc = int(gw.nocc)
    om, rho = G.exact_r_drpa(lpq, e, nocc)
    leg, _ = _get_scaled_legendre_roots(G.NW, G.X0)
    eval_f = np.concatenate(([0.0], leg))
    idx = _get_ac_idx(len(eval_f), npts=G.PADE_NPTS, step_ratio=G.PADE_STEP_RATIO)
    zn = blk["ef"] + 1j * eval_f[idx]
    dc = []
    for ip, p in enumerate(orbs):
        f, _poles, _wts = G.exact_sigma_c(lpq, e, nocc, om, rho, p)
        a = G.thiele_coeffs(f(zn)[:, None], zn)[:, 0]

        def sc_re(w, a=a):
            return G.thiele_eval(w + 0.0j, zn, a).real

        shift = blk["sigma_x_df"][ip] - blk["v_xc_semilocal"][ip]
        root, _rule = G.ferric_qp_rule(sc_re, float(e[p]), shift)
        dc.append(root)
    blk["exact"]["eps_qp_hybrid_double_counted"] = dc
    return blk


def main():
    xyz = common.MOL_DIR / f"{SYSTEM}.xyz"
    aux_name = G.AUX_FOR[BASIS]
    mol, symbols, coords, ll = G.mol_and_prov_base(xyz, BASIS)
    aux = G.aux_dict(aux_name, symbols)
    nocc = mol.nelectron // 2
    orbs = G.window(nocc, mol.nao_nr())
    payload = None
    ha = 27.211386245988
    for name, xc in FUNCTIONALS.items():
        blk = hybrid_block(mol, aux, orbs, xc)
        if payload is None:
            from pyscf.df import DF

            naux = G.check_naux(DF(mol, auxbasis=aux), aux_name, symbols)
            payload = G.header(mol, BASIS, aux_name, naux, ll)
        payload[name] = blk
        h, lu = orbs.index(nocc - 1), orbs.index(nocc)
        x = blk["exact"]
        print(
            f"{name}: HOMO {x['eps_qp'][h] * ha:.4f} eV (double-counted "
            f"{x['eps_qp_hybrid_double_counted'][h] * ha:.4f}), LUMO "
            f"{x['eps_qp'][lu] * ha:.4f} eV (double-counted "
            f"{x['eps_qp_hybrid_double_counted'][lu] * ha:.4f})",
            flush=True,
        )
    prov = G.provenance(
        xyz,
        BASIS,
        symbols,
        coords,
        aux_name,
        None,
        {"blocks": list(FUNCTIONALS), "functionals": FUNCTIONALS},
    )
    prov["generator"] = "scripts/validation/gen_gw_hybrid.py"
    payload["provenance"] = prov
    path = common.write_reference(ROW, SYSTEM, BASIS, payload)
    print(f"-> {path.relative_to(common.ROOT)}")


if __name__ == "__main__":
    main()
