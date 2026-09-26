"""Reference for the OO-RI-MP2 + ECP validation row (energy and, via the Rust
side's self-FD, the nuclear gradient's ECP term).

Consumer: crates/ferric-mp2/tests/validation_oo_rimp2_ecp.rs
Output:   testdata/reference/validation/oo_rimp2_ecp/<system>_<basis>.json

WHAT IS REFERENCED: the same functional as gen_oo_rimp2.py's `numpy_oo`
block (E_RHF(C) + E_MP2^RI(C), MP2 in the semicanonical frame of C, exact
four-centre J/K, Coulomb-metric RI with ferric's aux, all electrons outside
the ECP core correlated), written directly in numpy on PySCF integrals, but on
a molecule whose heavy atom carries an ECP. The core Hamiltonian is PySCF's
`mf.get_hcore()` = T + V_nuc(Z - N_core) + V_ECP (asserted equal to
int1e_kin + int1e_nuc + ECPscalar); the ECP reaches PySCF from ferric's own
basis JSON through `common.pyscf_ecp`, checked like-for-like by
`common.check_ecp_like_for_like` (per-atom N_core, electron count, term count),
exactly as the RHF+ECP row (gen_ecp.py) does. The rotation, pair,
denominator, FD-gradient and minimiser helpers are gen_oo_rimp2.py's
(imported, unchanged); J/K here come from PySCF's 8-fold-symmetric `int2e` via
`scf.hf.dot_eri_dm`.

System: HI (testdata/molecules/validation/ecp/hi.xyz, r = 1.75 A against
r_e ~ 1.609 A, so the nuclear gradient is far from zero), def2-SVP with the
def2 28-electron ECP on I, def2-svp-rifit aux, closed-shell RHF reference.

MINIMISER: gen_oo_rimp2.py has no analytic OO orbital gradient (its gradient is
a 5-point FD in kappa by design, so nothing is shared with ferric's orbital
gradient). To make the FD-gradient BFGS converge in few iterations the
variables are preconditioned by the diagonal orbital Hessian: the minimiser
works in y = s * kappa with s_ai = sqrt(max(4 (e_a - e_i), PRECOND_FLOOR)),
e from the PySCF RHF. The minimum is the same point; convergence is gated on
max |dE/dkappa| <= 1e-8 re-measured by FD in the ORIGINAL kappa.

Anchor (asserted): at kappa = 0 the numpy energy equals PySCF RHF + DF-MP2
(same auxmol) to TOL_ANCHOR = 1e-11.

EFFECT-SIZE block (`ecp_effect`): tr(D_RHF V_ECP) and the kappa = 0 total of
the functional with V_ECP LEFT OUT of the core Hamiltonian (same C_RHF, same
V_nuc(Z - N_core), same E_nn) -- the functional an OO-RI-MP2 whose hcore
lacks V_ECP evaluates at its first point.

MEASURED 2026-09-25 (HI/def2-SVP, nao 31, naux 128, 234 rotations): anchor
-5.7e-14 Ha; E_RHF -297.224288781605, E_corr(DF-MP2) -0.143859083541; numpy
OO minimum -297.369197888246 (reference -297.223197432701, doubles
-0.146000455545), 1.05e-3 Ha below RI-MP2; max |dE/dkappa| 7.0e-10 after 8
BFGS + 17 gradient-only iterations, 179 s. tr(D V_ECP) = 50.464 Ha; kappa = 0
total -347.7718 Ha without V_ECP vs -297.3681 Ha with it.

Run (light step):
    OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_oo_rimp2_ecp.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from gen_oo_rimp2 import (  # noqa: E402
    NUMPY_FD_STEP,
    _denom,
    _fd_gradient,
    _fitted_integrals,
    _minimise,
    _pair,
    _rotate,
    _semicanonical_b,
)

ROW = "oo_rimp2_ecp"
ROW_NAME = "OO-RI-MP2 with an ECP (energy; nuclear gradient via ferric self-FD)"
ECP_MOL_DIR = common.MOL_DIR / "ecp"
# system -> (orbital basis (carries the ECP inline), aux basis, charge, mult)
SYSTEMS = {
    "hi": ("def2-svp", "def2-svp-rifit", 0, 1),
}
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
TOL_ANCHOR = 1e-11
# The written minimum's max |dE/dkappa|, re-measured in kappa (not in the
# preconditioned variable the minimiser works in).
MAX_ORBITAL_GRAD = 1e-8
# Floor on the diagonal orbital-Hessian estimate 4 (e_a - e_i) (Ha).
PRECOND_FLOOR = 0.1
TOL_HCORE = 1e-12


def run(system: str) -> Path:
    import pyscf
    import scipy
    from pyscf import df, scf
    from pyscf.mp import dfmp2

    basis, aux, charge, mult = SYSTEMS[system]
    if mult != 1:
        raise ValueError("closed shell only (ferric has no U-OO nuclear gradient)")
    xyz = ECP_MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    ecp_json = common.basis_json_path(basis)
    mol = common.build_pyscf_mol(
        xyz, basis, charge=charge, multiplicity=mult, ecp_json=ecp_json
    )
    basis_check = common.check_basis_like_for_like(mol, basis, symbols)
    ecp_check = common.check_ecp_like_for_like(mol, ecp_json, symbols)
    if not any(ecp_check["ecp_core_electrons"]):
        raise RuntimeError(f"{system}/{basis}: no atom carries an ECP; wrong input")

    # ---- RHF (exact J/K), stability-checked ----
    mf = scf.RHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{system}: RHF did not converge")
    _mo_i, stable = common._stability_status(mf)
    if not stable:
        raise RuntimeError(f"{system}: RHF internal instability; refusing to write")

    h = mf.get_hcore()
    v_ecp = mol.intor("ECPscalar")
    h_no_ecp = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    d_h = float(np.abs(h - (h_no_ecp + v_ecp)).max())
    if d_h > TOL_HCORE:
        raise RuntimeError(f"hcore != T + V_nuc + V_ECP ({d_h:.2e})")

    # ---- DF-MP2 with ferric's aux (the kappa = 0 anchor) ----
    _eri_j, _eri_k, b, auxmol = _fitted_integrals(mol, aux, symbols)
    del _eri_j, _eri_k
    mp = dfmp2.DFMP2(mf)
    mp.with_df = df.DF(mol)
    mp.with_df.auxmol = auxmol
    mp.kernel()
    e_corr_df = float(mp.e_corr)

    eri = mol.intor("int2e", aosym="s8")
    enuc = mol.energy_nuc()
    c0 = mf.mo_coeff
    nocc = mol.nelectron // 2
    nvir = c0.shape[1] - nocc

    def parts_with(hh, kappa):
        c = _rotate(c0, kappa, nocc)
        co, cv = c[:, :nocc], c[:, nocc:]
        d = 2.0 * co @ co.T
        j, k = scf.hf.dot_eri_dm(eri, d, hermi=1)
        f = hh + j - 0.5 * k
        e_ref = 0.5 * np.sum(d * (hh + f)) + enuc
        eo, ev, bia = _semicanonical_b(b, f, co, cv)
        g = _pair(bia, bia)
        e_d = np.sum(g * (2.0 * g - g.transpose(0, 3, 2, 1)) / _denom(eo, ev, eo, ev))
        return float(e_ref), float(e_d)

    def energy(kappa):
        e_ref, e_d = parts_with(h, kappa)
        return e_ref + e_d

    x0 = np.zeros(nvir * nocc)
    e0 = energy(x0)
    anchor = float(e0 - (mf.e_tot + e_corr_df))
    if abs(anchor) > TOL_ANCHOR:
        raise RuntimeError(f"numpy OO kappa=0 vs PySCF RHF+DFMP2: {anchor:.2e}")
    e0_ref_noecp, e0_d_noecp = parts_with(h_no_ecp, x0)
    tr_d_vecp = float(np.sum(mf.make_rdm1() * v_ecp))
    print(
        f"{system}: nao {mol.nao_nr()} naux {auxmol.nao_nr()} nocc {nocc} nvir {nvir} "
        f"E_RHF {mf.e_tot:.12f} E_corr(DF) {e_corr_df:.12f} anchor {anchor:.1e} "
        f"tr(D V_ECP) {tr_d_vecp:.6f} kappa0 without V_ECP "
        f"{e0_ref_noecp + e0_d_noecp:.10f}",
        flush=True,
    )

    # Diagonal orbital-Hessian preconditioning: minimise over y = s * kappa,
    # s_ai = sqrt(4 (e_a - e_i)) from the RHF orbital energies (the leading
    # diagonal of the RHF orbital Hessian in this kappa convention). The
    # minimiser is gen_oo_rimp2's unchanged BFGS + gradient-only polish on a
    # 5-point FD gradient; only the variables are rescaled, so the minimum is
    # the same point. Nothing here comes from ferric.
    eps = mf.mo_energy
    hdiag = 4.0 * (eps[nocc:, None] - eps[None, :nocc])
    s = np.sqrt(np.maximum(hdiag, PRECOND_FLOOR)).ravel()

    def energy_y(y):
        return energy(y / s)

    t0 = time.time()
    y, gmax_y, nit, msg = _minimise(energy_y, nvir * nocc)
    x = y / s
    # Convergence gate in the ORIGINAL variable kappa (independent FD pass).
    gmax = float(np.abs(_fd_gradient(energy, x)).max())
    wall = time.time() - t0
    if gmax > MAX_ORBITAL_GRAD:
        raise RuntimeError(
            f"{system}: numpy OO not converged: max |dE/dkappa| {gmax:.2e} "
            f"(max |dE/dy| {gmax_y:.2e}; {msg})"
        )
    e_ref, e_d = parts_with(h, x)
    print(
        f"{system}: numpy E_OO {e_ref + e_d:.12f} (ref {e_ref:.12f} dbl {e_d:.12f}) "
        f"max|g| {gmax:.1e} {nit} it {wall:.0f} s",
        flush=True,
    )

    payload = {
        "row": ROW_NAME,
        "system": system,
        "basis": basis,
        "aux_basis": aux,
        "charge": charge,
        "multiplicity": mult,
        "nao": int(mol.nao_nr()),
        "naux": int(auxmol.nao_nr()),
        "nelectron": int(mol.nelectron),
        "ecp_core_electrons": ecp_check["ecp_core_electrons"],
        "nuclear_repulsion": float(enuc),
        "pyscf": {
            "e_scf": float(mf.e_tot),
            "e_corr_dfmp2": e_corr_df,
            "stability": {"internal_stable": True, "kind": "PySCF RHF internal (real)"},
        },
        "numpy_oo": {
            "e_total": float(e_ref + e_d),
            "e_reference": e_ref,
            "e_doubles": e_d,
            "max_orbital_gradient": gmax,
            "max_preconditioned_gradient": gmax_y,
            "iterations": nit,
            "optimizer_message": msg,
            "anchor_kappa0_diff": anchor,
            "wall_s": wall,
            "method": (
                "numpy OO-RI-MP2 (RHF) on an ECP molecule: E(kappa) = E_RHF(C) + "
                "E_MP2^RI(C), C = C_RHF expm(K), K antisymmetric vir-occ; hcore = "
                "PySCF get_hcore() = T + V_nuc(Z - N_core) + V_ECP (ECP from ferric's "
                "basis JSON via common.pyscf_ecp); exact J/K from PySCF int2e (s8, "
                "scf.hf.dot_eri_dm); doubles in the semicanonical frame with "
                "Coulomb-metric fitted B from PySCF int3c2e/int2c2e and ferric's aux; "
                "scipy BFGS + gradient-only BFGS polish (gen_oo_rimp2._minimise) "
                "over y = s kappa, s = sqrt(max(4 (e_a - e_i), "
                f"{PRECOND_FLOOR})) from the RHF orbital energies, on a 5-point "
                f"central-difference gradient (h = {NUMPY_FD_STEP} in y); "
                "max_orbital_gradient re-measured by FD in kappa; anchor at "
                "kappa = 0: PySCF scf.RHF + mp.dfmp2.DFMP2 with the same auxmol"
            ),
        },
        "ecp_effect": {
            "tr_d_rhf_vecp": tr_d_vecp,
            "kappa0_total_with_vecp": float(e0),
            "kappa0_total_without_vecp": float(e0_ref_noecp + e0_d_noecp),
            "note": (
                "kappa = 0 (C_RHF) value of the same functional with V_ECP omitted "
                "from hcore (V_nuc(Z - N_core) and E_nn kept): what an OO-RI-MP2 "
                "whose hcore lacks V_ECP evaluates at its first point"
            ),
        },
    }
    payload["provenance"] = common.provenance(
        code="PySCF + numpy",
        version=f"PySCF {pyscf.__version__}",
        keywords={
            "pyscf": f"scf.RHF exact J/K conv_tol {CONV_TOL} conv_tol_grad "
            f"{CONV_TOL_GRAD} + mp.dfmp2.DFMP2 with auxmol from ferric's aux JSON",
            "ecp": "mol.ecp built by common.pyscf_ecp from the basis JSON's inline "
            "ecp_potentials",
            "numpy_oo": payload["numpy_oo"]["method"],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        basis_name=basis,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux={
            "correlation": aux,
            "correlation_json": str(
                common.basis_json_path(aux).relative_to(common.ROOT)
            ),
            "correlation_sha256": common.sha256_file(common.basis_json_path(aux)),
            "scf_and_oo_fock": "none (exact four-centre J/K)",
        },
        frozen_core="none (all non-ECP electrons correlated)",
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability=payload["pyscf"]["stability"],
        ecp_json=ecp_json,
        generator="scripts/validation/gen_oo_rimp2_ecp.py",
        extra={"basis_self_check": basis_check, "ecp_self_check": ecp_check},
    )
    return common.write_reference(ROW, system, basis, payload)


def main() -> int:
    only = set(sys.argv[1:])
    written = [run(s) for s in SYSTEMS if not only or s in only]
    print(f"GEN_OO_RIMP2_ECP_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
