"""PySCF references for EE-EOM-CCSD singlets (ferric-cc/src/eom_ccsd.rs).

Consumer: crates/ferric-cc/tests/validation_eom_ccsd.rs.
Output:   testdata/reference/validation/eom_ccsd/<system>_<basis>.json

Like-for-like density fitting, exactly as gen_cc.py: exact-integral PySCF
RHF with ferric's bundled orbital basis and geometry, then every correlation
integral from one DF factorization with ferric's bundled auxiliary basis,
(pq|rs) = sum_P B^P_pq B^P_rs, fed to PySCF's CCSD through a hand-built
`_ChemistsERIs` whose Fock matrix is diag(RHF mo_energy). PySCF's
`EOMEESinglet` then runs on those eris and the CCSD amplitudes. All electrons
are correlated.

Root selection: PySCF's EOM Davidson starts from unit vectors on the lowest
diagonal elements and its product never mixes point-group irreps, so a low
root of an irrep none of those vectors touches is skipped. The guesses here
are those unit vectors plus a small deterministic dense part (the same
construction as ferric's solver), and NGUESS of them are run so the lowest
NROOTS are covered.

Each root's residual ||(H-bar - E) v - w v|| / ||v|| is recomputed here with
PySCF's own matvec and recorded, so the reference carries its own accuracy
(the eigenvalue error of this non-symmetric problem is first order in it).

For a packed dimension up to DENSE_MAX_DIM (water), the whole operator is
also built column by column and diagonalized; its lowest roots must equal the
Davidson ones (asserted at 1e-8), so the root selection is checked too.

Run (light; about two minutes for both systems):
    box-lease exec -- uv run --no-sync python scripts/validation/gen_eom_ccsd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_cc  # noqa: E402

ROW = "eom_ccsd"
CASES = {
    ("h2o", "cc-pvdz"): common.MOL_DIR / "h2o.xyz",
    ("c2h4", "cc-pvdz"): common.ROOT / "testdata" / "molecules" / "c2h4.xyz",
}
AUX = {"cc-pvdz": "cc-pvdz-ri"}
NROOTS = 5
NGUESS = 10
MAX_MEMORY_MB = 4000
EOM_CONV_TOL = 1e-11
EOM_RESIDUAL_TOL = 1e-8
GUESS_SPREAD = 0.1
DENSE_MAX_DIM = 6000
# PySCF drops a correction whose squared norm is below lindep (default 1e-14,
# a residual floor of 1e-7); lowered so the residual can reach EOM_RESIDUAL_TOL.
EOM_LINDEP = 1e-22


def guesses(diag, n):
    import numpy as np

    dim = diag.size
    order = np.argsort(diag, kind="stable")[:n]
    rng = np.random.default_rng(11)
    v = GUESS_SPREAD / np.sqrt(dim) * rng.uniform(-1.0, 1.0, (dim, n))
    for k, i in enumerate(order):
        v[i, k] += 1.0
    q, _ = np.linalg.qr(v)
    return [q[:, k].copy() for k in range(n)]


def run_case(system, basis_name, xyz):
    import numpy as np
    from pyscf import gto, lib, scf
    from pyscf.cc import ccsd, eom_rccsd

    aux_name = AUX[basis_name]
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name)
    mol.max_memory = MAX_MEMORY_MB
    basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
    aux_basis, aux_cart = common.pyscf_basis(aux_name, symbols)
    assert not aux_cart
    auxmol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=aux_basis,
        verbose=0,
    )
    assert auxmol.nao_nr() == common.ferric_nao(aux_name, symbols)

    mf = scf.RHF(mol)
    mf.conv_tol = gen_cc.SCF_CONV_TOL
    mf.conv_tol_grad = gen_cc.SCF_CONV_TOL_GRAD
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    assert mf.converged

    b_ao, _, _ = gen_cc.df_factor(mol, auxmol)
    eri, _ = gen_cc.mo_eri(b_ao, mf.mo_coeff)
    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = gen_cc.CC_CONV_TOL
    mycc.conv_tol_normt = gen_cc.CC_CONV_TOL_NORMT
    mycc.max_cycle = 300
    mycc.verbose = 0
    mycc.direct = False
    mycc.max_memory = MAX_MEMORY_MB
    eris, offdiag, _ = gen_cc.build_eris(mycc, eri, mf.mo_energy)
    mycc.kernel(eris=eris)
    assert mycc.converged

    eom = eom_rccsd.EOMEESinglet(mycc)
    eom.max_memory = MAX_MEMORY_MB
    eom.verbose = 0
    imds = eom.make_imds(eris)
    matvec, diag = eom.gen_matvec(imds)

    def precond(r, e0, x0):
        return r / (e0 - diag + 1e-12)

    def pick_lowest_real(w, v, nroots, envs):
        real_idx = np.where(abs(w.imag) < 1e-3)[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, True)

    # PySCF's own Davidson and product; lowest real roots (its user-guess path
    # follows overlap with the guesses instead, so it is not used).
    conv, e, v = lib.davidson_nosym1(
        matvec,
        guesses(diag, NGUESS),
        precond,
        pick=pick_lowest_real,
        tol=EOM_CONV_TOL,
        tol_residual=EOM_RESIDUAL_TOL,
        lindep=EOM_LINDEP,
        max_cycle=300,
        max_space=80,
        max_memory=MAX_MEMORY_MB,
        nroots=NGUESS,
    )
    e = np.asarray(e).real
    order = np.argsort(e)
    if not all(np.asarray(conv)[order[:NROOTS]]):
        raise RuntimeError(f"{system}: EOM Davidson did not converge: {conv} {e}")
    e = e[order][:NROOTS]
    vecs = [np.asarray(v[k]) for k in order[:NROOTS]]
    res = []
    for w, x in zip(e, vecs):
        hx = eom.matvec(x, imds)
        res.append(float(np.linalg.norm(hx - w * x) / np.linalg.norm(x)))

    dense_check = None
    dim = diag.size
    if dim <= DENSE_MAX_DIM:
        # The lowest roots of the whole packed operator, built column by
        # column from the same product: the root-selection check.
        cols = [eom.matvec(u, imds) for u in np.eye(dim)]
        w_all = np.linalg.eigvals(np.array(cols).T)
        w_all = np.sort(w_all[w_all.real > 1e-10].real)[:NROOTS]
        dense_check = {
            "dense_lowest": [float(x) for x in w_all],
            "max_abs_davidson_minus_dense": float(np.max(np.abs(w_all - e))),
        }
        assert dense_check["max_abs_davidson_minus_dense"] < 1e-8, dense_check

    nocc = mol.nelectron // 2
    return {
        "system": system,
        "basis": basis_name,
        "aux_basis": aux_name,
        "nao": int(mol.nao_nr()),
        "naux": int(auxmol.nao_nr()),
        "nocc": int(nocc),
        "nvir": int(mol.nao_nr() - nocc),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "rhf": {"energy": float(mf.e_tot)},
        "ccsd": {"e_corr": float(mycc.e_corr)},
        "fock_offdiag_max": offdiag,
        "eom_ccsd_singlets": {
            "excitation_energies": [float(x) for x in e],
            "residual_norms": res,
            "n_guess": NGUESS,
            "dense_check": dense_check,
        },
        "provenance": common.provenance(
            code="PySCF + numpy",
            version=__import__("pyscf").__version__,
            keywords={
                "reference": "scf.RHF, exact 4-index J/K",
                "correlation_integrals": "DF, B = L^-1 (P|mn) as gen_cc.py",
                "fock": "diag(RHF mo_energy)",
                "ccsd": "pyscf.cc.ccsd.CCSD on hand-built DF _ChemistsERIs",
                "eom": "pyscf.cc.eom_rccsd.EOMEESinglet product + lib.davidson_nosym1, "
                "lowest real roots",
                "eom_conv_tol": EOM_CONV_TOL,
                "eom_lindep": EOM_LINDEP,
                "eom_residual_tol": EOM_RESIDUAL_TOL,
                "guess": f"{NGUESS} unit vectors on the lowest diagonal + {GUESS_SPREAD} dense part",
                "cc_conv_tol": gen_cc.CC_CONV_TOL,
                "cc_conv_tol_normt": gen_cc.CC_CONV_TOL_NORMT,
            },
            basis_name=basis_name,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            aux=aux_name,
            frozen_core=0,
            scf_conv=gen_cc.SCF_CONV_TOL,
            generator="scripts/validation/gen_eom_ccsd.py",
            extra={"basis_check": basis_check},
        ),
    }


def main() -> int:
    only = set(sys.argv[1:])
    for (system, basis_name), xyz in CASES.items():
        if only and system not in only:
            continue
        payload = run_case(system, basis_name, xyz)
        path = common.write_reference(ROW, system, basis_name, payload)
        print(
            f"{system}/{basis_name}: E_corr={payload['ccsd']['e_corr']:.10f} "
            f"omega={payload['eom_ccsd_singlets']['excitation_energies']} "
            f"res={['%.1e' % r for r in payload['eom_ccsd_singlets']['residual_norms']]}",
            flush=True,
        )
        print("wrote", path.relative_to(common.ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
