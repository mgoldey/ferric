"""References for unrestricted OO-RI-MP2 with a frozen core.

Consumer: crates/ferric-mp2/tests/u_oo_rimp2_frozen_core.rs
Output:   testdata/reference/validation/u_oo_rimp2_frozen_core/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/u_oo_rimp2.rs):
E(C_a, C_b) = E_UHF(C_a, C_b) + E_UMP2^RI(C_a, C_b), minimised over
active-occupied/virtual rotations of each spin. The lowest `frozen_core`
orbitals of each spin are excluded from the MP2 sum and are NOT rotated: they
stay the reference UHF's canonical core orbitals. The MP2 part is evaluated in
each spin's semicanonical frame (active-occupied and virtual blocks of
C^T F C diagonalised separately; the core block is left alone). E_UHF and F
use exact four-centre J/K; only the MP2 part is density-fitted (Coulomb
metric, ferric's aux).

NO EXTERNAL CODE computes this functional. PySCF has no orbital-optimised MP2
module; its OMP2 example (examples/mp/42-omp2.py) is CASSCF with an MP2
"FCI solver", closed shell, conventional integrals, and with a frozen core it
would be a CASSCF core that IS rotated against the active and virtual spaces.
So the reference is an independent numpy OO-RI-MP2 on PySCF integrals, the
`numpy_oo_uhf` construction of gen_oo_rimp2.py with the frozen core added:
the free parameters are the active-occupied/virtual blocks of K_a and K_b,
C_s = C_UHF,s expm(K_s), the energy gradient is a 5-point central difference
(nothing is shared with ferric's analytic orbital gradient), and scipy BFGS
plus a gradient-only BFGS polish minimise it to max |dE/dkappa| <= 1e-10.

Anchor at kappa = 0: the numpy energy equals PySCF UHF + DFUMP2(frozen=fc)
with the same auxmol (asserted to TOL_ANCHOR_UHF). The UHF is
`common.run_open_shell("uhf")`, stability-checked.

Each system is written at frozen core 0 and 1. The frozen-core-0 block is the
control that ties this generator to the existing CH3 reference construction;
the 1 - 0 difference is the effect the test must resolve.

Run (light: under a minute per system on one core):
    scripts/validation/run_slot.sh --light -- \\
        uv run --no-sync python scripts/validation/gen_u_oo_rimp2_frozen_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from gen_oo_rimp2 import (  # noqa: E402
    NUMPY_FD_STEP,
    TOL_ANCHOR_UHF,
    _denom,
    _fitted_integrals,
    _minimise,
    _pair,
)

ROW = "u_oo_rimp2_frozen_core"
ROW_NAME = "U-OO-RI-MP2 energy with a frozen core"

# system -> (orbital basis, aux basis, charge, multiplicity)
SYSTEMS = {
    "oh": ("6-31g", "cc-pvdz-ri", 0, 2),
    "nh2": ("6-31g", "cc-pvdz-ri", 0, 2),
}
FROZEN = (0, 1)


def _rotate_active(c, kappa, fc, nocc):
    """C expm(K) with only the active-occupied/virtual block of K free."""
    import scipy.linalg as sla

    nmo = c.shape[1]
    k = np.zeros((nmo, nmo))
    k[nocc:, fc:nocc] = kappa.reshape(nmo - nocc, nocc - fc)
    k -= k.T
    return c @ sla.expm(k)


def _semicanonical_active(b, f, co_act, cv):
    eo, uo = np.linalg.eigh(co_act.T @ f @ co_act)
    ev, uv = np.linalg.eigh(cv.T @ f @ cv)
    return eo, ev, np.matmul(np.matmul((co_act @ uo).T, b), cv @ uv)


def numpy_u_oo_frozen(mol, mf, auxmol, eri_j, eri_k, b, fc: int) -> dict:
    from pyscf import df
    from pyscf.mp import dfump2

    mp = dfump2.DFUMP2(mf, frozen=fc if fc else None)
    mp.with_df = df.DF(mol)
    mp.with_df.auxmol = auxmol
    mp.kernel()
    e_corr_df = float(mp.e_corr)

    h = mf.get_hcore()
    enuc = mol.energy_nuc()
    ca0, cb0 = mf.mo_coeff
    noa, nob = mol.nelec
    n = ca0.shape[0]
    nva, nvb = ca0.shape[1] - noa, cb0.shape[1] - nob
    npa = nva * (noa - fc)
    npb = nvb * (nob - fc)

    def parts(kappa):
        ca = _rotate_active(ca0, kappa[:npa], fc, noa)
        cb = _rotate_active(cb0, kappa[npa:], fc, nob)
        da = ca[:, :noa] @ ca[:, :noa].T
        db = cb[:, :nob] @ cb[:, :nob].T
        j = (eri_j @ (da + db).ravel()).reshape(n, n)
        fa = h + j - (eri_k @ da.ravel()).reshape(n, n)
        fb = h + j - (eri_k @ db.ravel()).reshape(n, n)
        e_ref = 0.5 * (np.sum(da * (h + fa)) + np.sum(db * (h + fb))) + enuc
        eoa, eva, ba = _semicanonical_active(b, fa, ca[:, fc:noa], ca[:, noa:])
        eob, evb, bb = _semicanonical_active(b, fb, cb[:, fc:nob], cb[:, nob:])
        e_d = 0.0
        for bx, eo, ev in ((ba, eoa, eva), (bb, eob, evb)):
            g = _pair(bx, bx)
            e_d += 0.5 * np.sum(
                g * (g - g.transpose(0, 3, 2, 1)) / _denom(eo, ev, eo, ev)
            )
        g = _pair(ba, bb)
        e_d += np.sum(g * g / _denom(eoa, eva, eob, evb))
        return e_ref, e_d

    def energy(kappa):
        e_ref, e_d = parts(kappa)
        return e_ref + e_d

    anchor = float(energy(np.zeros(npa + npb)) - (mf.e_tot + e_corr_df))
    if abs(anchor) > TOL_ANCHOR_UHF:
        raise RuntimeError(f"numpy U-OO fc={fc} kappa=0 vs PySCF: {anchor:.2e}")
    x, gmax, nit, msg = _minimise(energy, npa + npb)
    e_ref, e_d = parts(x)
    return {
        "frozen_core": fc,
        "e_total": float(e_ref + e_d),
        "e_reference": float(e_ref),
        "e_doubles": float(e_d),
        "max_orbital_gradient": gmax,
        "iterations": nit,
        "optimizer_message": msg,
        "anchor_kappa0_diff": anchor,
        "e_dfump2_corr_kappa0": e_corr_df,
        "n_parameters": npa + npb,
    }


def main() -> int:
    import pyscf
    from pyscf import gto

    only = {a for a in sys.argv[1:] if not a.startswith("--")}
    written = 0
    for system, (basis, aux, charge, mult) in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        bas, cart = common.pyscf_basis(basis, symbols)
        assert not cart
        mol = gto.M(
            atom=common.pyscf_atom_bohr(symbols, coords),
            unit="Bohr",
            basis=bas,
            cart=False,
            charge=charge,
            spin=mult - 1,
            verbose=0,
        )
        res, mf = common.run_open_shell(
            mol, "uhf", conv_tol=1e-12, conv_tol_grad=1e-9, return_mf=True
        )
        eri_j, eri_k, b, auxmol = _fitted_integrals(mol, aux, symbols)
        blocks = {
            f"fc{fc}": numpy_u_oo_frozen(mol, mf, auxmol, eri_j, eri_k, b, fc)
            for fc in FROZEN
        }
        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis,
            "aux_basis": aux,
            "charge": charge,
            "multiplicity": mult,
            "nao": int(mol.nao_nr()),
            "naux": int(auxmol.nao_nr()),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "pyscf": {"e_scf": float(mf.e_tot), "uhf": res},
            "numpy_oo": blocks,
        }
        payload["provenance"] = common.provenance(
            code=f"PySCF {pyscf.__version__} + numpy",
            version=f"PySCF {pyscf.__version__}",
            keywords={
                "pyscf": "common.run_open_shell('uhf', conv_tol 1e-12, "
                "conv_tol_grad 1e-9), exact J/K, stability loop; "
                "mp.dfump2.DFUMP2(frozen=fc) with auxmol from ferric's aux JSON",
                "numpy_oo": (
                    "numpy U-OO-RI-MP2 with frozen core fc: C_s = C_UHF,s expm(K_s), "
                    "K_s antisymmetric with only the active-occupied/virtual block "
                    "free (core orbitals fixed); exact four-centre J/K from PySCF "
                    "int2e; doubles over the active occupied orbitals in each spin's "
                    "semicanonical frame (active-occupied and virtual blocks), "
                    "Coulomb-metric fitted B; scipy BFGS on a 5-point central-"
                    f"difference gradient (h = {NUMPY_FD_STEP}) + gradient-only BFGS"
                ),
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
            frozen_core="0 and 1 per spin (blocks fc0, fc1); core orbitals not rotated",
            scf_conv="PySCF conv_tol 1e-12, conv_tol_grad 1e-9",
            stability=res["stability"],
            generator="scripts/validation/gen_u_oo_rimp2_frozen_core.py",
        )
        common.write_reference(ROW, system, basis, payload)
        written += 1
        for key, blk in blocks.items():
            print(
                f"{system:4s} {key} E_OO {blk['e_total']:.12f} "
                f"(ref {blk['e_reference']:.12f} dbl {blk['e_doubles']:.12f}, "
                f"max|g| {blk['max_orbital_gradient']:.1e}, {blk['iterations']} it, "
                f"anchor {blk['anchor_kappa0_diff']:.1e})"
            )
    print(f"GEN_U_OO_RIMP2_FROZEN_CORE_DONE written={written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
