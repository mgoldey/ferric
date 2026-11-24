"""References for the "SCS-MP2 analytic gradient" validation row.

Consumer: crates/ferric-mp2/tests/validation_scs_mp2_gradient.rs
Output:   testdata/reference/validation/scs_mp2_gradient/<system>_<basis>.json

WHAT FERRIC COMPUTES (read from crates/ferric-mp2/src/gradient.rs):
`scs_mp2_gradient_analytical` on an exact four-centre-J/K RHF
(`RhfConfig` fitting fields None); only the MP2 correlation is density
fitted (Coulomb metric, ferric's aux JSON). The energy is
    E = E_RHF + c_os E_OS + c_ss E_SS,
so the gradient is linear in the scaling:
    g = g_RHF + c_os g_OS + c_ss g_SS,
and the reference stores the three gradients separately, so the Rust test
composes any scaling (Grimme's, SOS, the unit scaling) from one reference.

THE INDEPENDENT CODE: PySCF 2.13.1, exact-integral `scf.RHF` +
`pyscf.mp.dfmp2.DFMP2` with `with_df.auxmol` from ferric's aux JSON.
`DFMP2.kernel()` tags its result with `e_corr_os` and `e_corr_ss`. PySCF has
NO analytic DF-MP2 gradient (`DFMP2.nuc_grad_method` raises
NotImplementedError; `pyscf.grad` has only conventional mp2/ump2, which has
no SCS scaling and no density fitting), so each gradient is a 5-point central
finite difference of PySCF's own energy component, step H_FD Bohr: an
independent code AND an independent method (analytic Z-vector vs FD).
Conventional-integral MP2 is deliberately not used: its DF-vs-conventional
difference is the RI error, which is not bounded here.

Both frozen_core = 0 and frozen_core = 1 (the O or N 1s) are generated; the
SCF is shared between them at each displaced geometry.

FD floor: truncation O(h^4 E^(5)) ~ 1e-11; PySCF SCF conv_tol 1e-12 /
conv_tol_grad 1e-10 gives ~1e-12 Ha noise in each energy, / h, times the
stencil weight sum 2.5/... ~ 1e-9 in g. The generator re-runs one coordinate
at H_FD/2 and refuses to write if the step moves it by more than 5e-9.

Run (one PySCF process, serial BLAS):
    box-lease run --label gradgrade -- \\
        /home/matt/qc/ferric/.venv/bin/python scripts/validation/gen_scs_mp2_gradient.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "scs_mp2_gradient"
ROW_NAME = "SCS-MP2 analytic gradient"

# system -> (orbital basis, aux basis, frozen core orbitals for the fc variant)
SYSTEMS = {
    "h2o_distorted": ("cc-pvdz", "cc-pvdz-ri", 1),
    "nh3_distorted": ("def2-svp", "def2-svp-rifit", 1),
}

H_FD = 2e-3  # Bohr, 5-point stencil
STEP_CHECK_TOL = 5e-9
STENCIL = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]


def energies(symbols, coords_bohr, basis, aux, ncore, dm0=None):
    """(E_RHF, [E_OS, E_SS] all-electron, [E_OS, E_SS] frozen core, dm, naux)."""
    from pyscf import df, gto, scf
    from pyscf.mp import dfmp2

    bas, cart = common.pyscf_basis(basis, symbols)
    assert not cart
    mol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords_bohr),
        unit="Bohr",
        basis=bas,
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge")
    aux_bas, aux_cart = common.pyscf_basis(aux, symbols)
    assert not aux_cart
    out = []
    naux = None
    for frozen in (None, list(range(ncore))):
        mp = dfmp2.DFMP2(mf, frozen=frozen)
        mp.with_df = df.DF(mol)
        mp.with_df.auxmol = df.addons.make_auxmol(mol, aux_bas)
        mp.with_df.auxmol.cart = False
        mp.with_df.auxmol.build()
        mp.with_df.auxbasis = aux_bas
        mp.kernel()
        out.append([float(mp.e_corr_os), float(mp.e_corr_ss)])
        naux = mp.with_df.auxmol.nao_nr()
    return float(mf.e_tot), out[0], out[1], mf.make_rdm1(), naux, mol


def fd_gradients(symbols, coords, basis, aux, ncore, dm0, step=H_FD, only=None):
    """5-point FD of E_RHF, E_OS, E_SS (all-electron and fc) -> dict of (natm, 3)."""
    c0 = np.array(coords)
    natm = len(symbols)
    keys = ["hf", "os", "ss", "os_fc", "ss_fc"]
    g = {k: np.zeros((natm, 3)) for k in keys}
    atoms = range(natm) if only is None else [only[0]]
    for a in atoms:
        for k in range(3) if only is None else [only[1]]:
            for s, w in STENCIL:
                c = c0.copy()
                c[a, k] += s * step
                e_hf, ae, fc, *_ = energies(symbols, c.tolist(), basis, aux, ncore, dm0)
                vals = {
                    "hf": e_hf,
                    "os": ae[0],
                    "ss": ae[1],
                    "os_fc": fc[0],
                    "ss_fc": fc[1],
                }
                for key in keys:
                    g[key][a, k] += w * vals[key] / step
    return g


def main() -> int:
    import pyscf

    only_sys = {a for a in sys.argv[1:] if not a.startswith("--")}
    written = []
    for system, (basis, aux, ncore) in SYSTEMS.items():
        if only_sys and system not in only_sys:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        e_hf, ae, fc, dm, naux, mol = energies(symbols, coords, basis, aux, ncore)
        g = fd_gradients(symbols, coords, basis, aux, ncore, dm)

        # step check on one coordinate: the largest-|g_os| component
        a, k = np.unravel_index(np.argmax(np.abs(g["os"])), g["os"].shape)
        g_half = fd_gradients(
            symbols, coords, basis, aux, ncore, dm, step=H_FD / 2, only=(int(a), int(k))
        )
        step_move = max(abs(g_half[key][a, k] - g[key][a, k]) for key in g)
        if step_move > STEP_CHECK_TOL:
            raise RuntimeError(f"{system}: FD moves {step_move:.2e} with the step")

        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis,
            "aux_basis": aux,
            "charge": 0,
            "multiplicity": 1,
            "nao": int(mol.nao_nr()),
            "naux": int(naux),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "n_frozen": ncore,
            "pyscf": {
                "e_rhf": e_hf,
                "e_os": ae[0],
                "e_ss": ae[1],
                "e_os_fc": fc[0],
                "e_ss_fc": fc[1],
                "gradient_fd": {key: g[key].tolist() for key in g},
                "fd": {"stencil": "5-point central", "step_bohr": H_FD},
                "step_check": {
                    "coordinate": [int(a), int(k)],
                    "step_bohr": H_FD / 2,
                    "max_change": step_move,
                },
            },
            "provenance": common.provenance(
                code=f"PySCF {pyscf.__version__}",
                version=f"PySCF {pyscf.__version__}",
                keywords={
                    "pyscf": "scf.RHF (exact J/K, conv_tol 1e-12, conv_tol_grad "
                    "1e-10) + mp.dfmp2.DFMP2 (frozen None and range(n_frozen)) "
                    "with auxmol from ferric's aux JSON; e_corr_os / e_corr_ss "
                    "tags; gradient = 5-point central FD of E_RHF, E_OS, E_SS "
                    f"separately, h = {H_FD} Bohr"
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
                    "correlation_sha256": common.sha256_file(
                        common.basis_json_path(aux)
                    ),
                    "scf": "none (exact four-centre J/K)",
                },
                frozen_core=f"none, and the first {ncore} orbital(s)",
                scf_conv="PySCF conv_tol 1e-12 conv_tol_grad 1e-10",
                stability=None,
                generator="scripts/validation/gen_scs_mp2_gradient.py",
            ),
        }
        path = common.write_reference(ROW, system, basis, payload)
        written.append(path)
        print(f"{system:14s} {basis:8s} step check {step_move:.1e} -> {path}")
    print(f"GEN_SCS_MP2_GRADIENT_DONE written={len(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
