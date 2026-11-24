"""PySCF finite-difference gradients of open-shell attenuated (erfc) DF-UMP2.

Consumer: crates/ferric-mp2/tests/validation_u_att_rimp2_gradient_pyscf.rs.
Output:   testdata/reference/validation/u_att_rimp2_gradient_pyscf/<system>_cc-pvdz.json

WHAT IS BUILT
-------------
E(R) = E_UHF[1/r] + E_c[erfc(w r)/r], w = 0.420 A^-1 (0.222254 Bohr^-1):
PySCF `scf.UHF` (exact Coulomb integrals) and `mp.dfump2.DFUMP2` on a
`df.DF` built inside `mol.with_range_coulomb(-w)`, so the 3-centre integrals
AND the metric of the fit are erfc, as in ferric's attenuated RI-MP2.
No frozen core. Basis cc-pVDZ, fit cc-pVDZ-RI, both from ferric's bundled
JSON (`common.pyscf_basis`).

At the reference geometry the UHF is the lowest stable solution
(`common.run_open_shell`); every displaced UHF starts from its density, so
the displaced points stay on the same state (their <S^2> is recorded).

For each listed Cartesian coordinate, central differences
D(h) = (E(x+h) - E(x-h)) / 2h at h = 2e-3 Bohr and 2h, and the Richardson
value R = (4 D(h) - D(2h)) / 3. The JSON keeps all four energies per
coordinate, so a consumer can rebuild D(h), D(2h) and R.

This is an independent route: PySCF's SCF, integrals (libcint), fit and
correlation code; ferric's analytic gradient is compared with R.

Coordinates: OH (bond on z) the two z components; NH2 (in the yz plane) the
six in-plane components. The out-of-plane ones vanish by symmetry.

Run (light; about a minute):
    OPENBLAS_NUM_THREADS=1 box-lease exec -- \\
        uv run --no-sync python scripts/validation/gen_u_att_rimp2_gradient_pyscf.py [system ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_attenuated_rpa as gar  # noqa: E402
import gen_urpa  # noqa: E402

ROW = "u_att_rimp2_gradient_pyscf"
OMEGA = gar.OMEGA_DEFAULT_BOHR
H = 2e-3
CONV_TOL = 1e-13
CONV_TOL_GRAD = 1e-11
SYSTEMS = {  # system -> (xyz, multiplicity, basis, aux, [(atom, coord)])
    "oh": ("oh.xyz", 2, "cc-pvdz", "cc-pvdz-ri", [(0, 2), (1, 2)]),
    "nh2": (
        "nh2.xyz",
        2,
        "cc-pvdz",
        "cc-pvdz-ri",
        [(a, c) for a in range(3) for c in (1, 2)],
    ),
}


def total_energy(symbols, coords, mult, basis, aux, dm0):
    """(E_UHF + E_c[erfc], E_UHF, <S^2>) at `coords` (Bohr), UHF from `dm0`."""
    from pyscf import df, gto, scf
    from pyscf.mp import dfump2

    mol = gto.M(
        atom=common.pyscf_atom_bohr(symbols, coords),
        unit="Bohr",
        basis=basis,
        cart=False,
        spin=mult - 1,
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 300
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("displaced UHF did not converge")
    with mol.with_range_coulomb(-OMEGA):
        dfo = df.DF(mol, auxbasis=aux)
        dfo.verbose = 0
        dfo.build()
    assert mol.omega == 0.0 and dfo.auxmol.omega == -OMEGA
    pt = dfump2.DFUMP2(mf)
    pt.with_df = dfo
    pt.verbose = 0
    pt.kernel()
    return float(mf.e_tot + pt.e_corr), float(mf.e_tot), float(mf.spin_square()[0])


def gen(system):
    import pyscf

    xyz_rel, mult, basis_name, aux_name, coords_fd = SYSTEMS[system]
    t0 = time.perf_counter()
    xyz = common.MOL_DIR / xyz_rel
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, basis_name, multiplicity=mult)
    ll = common.check_basis_like_for_like(mol, basis_name, symbols)
    basis, cart = common.pyscf_basis(basis_name, symbols)
    assert not cart
    aux = gen_urpa.aux_dict(aux_name, symbols)
    uhf, mf = common.run_open_shell(
        mol, "uhf", conv_tol=CONV_TOL, conv_tol_grad=CONV_TOL_GRAD, return_mf=True
    )
    dm_ref = mf.make_rdm1()
    e0, e_uhf0, s2_0 = total_energy(symbols, coords, mult, basis, aux, dm_ref)
    if abs(e_uhf0 - float(mf.e_tot)) > 1e-10:
        raise RuntimeError("the reference UHF moved when re-solved from its density")
    rows = []
    for atom, c in coords_fd:
        e = {}
        s2 = {}
        for k in (-2, -1, 1, 2):
            x = [list(r) for r in coords]
            x[atom][c] += k * H
            e[k], _, s2[k] = total_energy(symbols, x, mult, basis, aux, dm_ref)
        d1 = (e[1] - e[-1]) / (2 * H)
        d2 = (e[2] - e[-2]) / (4 * H)
        rows.append(
            {
                "atom": atom,
                "coord": c,
                "energies": {str(k): e[k] for k in (-2, -1, 1, 2)},
                "s2": {str(k): s2[k] for k in (-2, -1, 1, 2)},
                "d_h": d1,
                "d_2h": d2,
                "richardson": (4 * d1 - d2) / 3,
            }
        )
        print(f"  {system} atom {atom} coord {c}: R = {(4 * d1 - d2) / 3:+.10e}")
    payload = {
        "system": system,
        "basis": basis_name,
        "aux": aux_name,
        "charge": 0,
        "multiplicity": mult,
        "like_for_like": ll,
        "omega_bohr_inv": OMEGA,
        "frozen_core": 0,
        "scaling": 1.0,
        "h_bohr": H,
        "uhf": uhf,
        "energy_total": e0,
        "s2": s2_0,
        "fd": rows,
        "generator_seconds": time.perf_counter() - t0,
    }
    payload["provenance"] = common.provenance(
        code="PySCF (scf.UHF exact integrals; df.DF under Mole.with_range_coulomb(-w); "
        "mp.dfump2.DFUMP2), central differences of the total energy",
        version=pyscf.__version__,
        keywords={
            "uhf": f"scf.UHF, conv_tol {CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}; "
            "reference stability-followed, displaced points from its density",
            "operator": "erfc(w r)/r on the fit's 3-centre integrals and metric",
            "fd": f"h = {H} Bohr and 2h, Richardson (4 D(h) - D(2h)) / 3",
        },
        basis_name=basis_name,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid=None,
        aux=aux_name,
        frozen_core=0,
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        stability={"uhf": uhf["stability"]},
        generator="scripts/validation/gen_u_att_rimp2_gradient_pyscf.py",
        extra={},
    )
    path = common.write_reference(ROW, system, basis_name, payload)
    print(
        f"{system}: E {e0:.12f} <S^2> {s2_0:.6f} ({payload['generator_seconds']:.1f} s) "
        f"-> {path.relative_to(common.ROOT)}"
    )


def main(argv):
    want = list(argv) or list(SYSTEMS)
    unknown = [k for k in want if k not in SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown systems: {unknown} (known: {sorted(SYSTEMS)})")
    for k in want:
        gen(k)
    print(f"GEN_U_ATT_RIMP2_GRADIENT_PYSCF_DONE written={len(want)}")


if __name__ == "__main__":
    main(sys.argv[1:])
