"""PySCF references for the point-group-blocked RHF row (issue #294, stage 2).

Consumer: crates/ferric-scf/tests/scf_symmetry.rs (the `#[ignore]`d PySCF
comparison) and testdata/reference/validation/scf_symmetry/*.json.

For each system the reference records PySCF's symmetry-blocked RHF
(`mol.symmetry = True`) energy, point group and the number of doubly occupied
orbitals per irrep, next to the unblocked energy. Extra blocks:

* `h2o`: `excited` moves the out-of-plane 1b1 pair into the first unoccupied
  a1 orbital (`irrep_nelec`), the state plain aufbau cannot reach.
* `h2_r50`, `h2_r10000` (H2 at 50 and 10000 Angstrom, testdata/molecules/validation/h2_r*.xyz): `sigma_g2` and
  `sigma_u2` pin each occupation through `irrep_nelec`; `unblocked` is PySCF
  with `symmetry = False`.

RKS (`rks_pbe` block, h2o and both H2 files): PySCF `dft.RKS` PBE with
`mol.symmetry = True` on ferric's KS recipe, the same one
gen_ks_energies.py uses: (75, 110) unpruned atomic grids, Becke partition
with Becke (1988) radii adjustment, RI-J with def2-universal-jkfit fed from
ferric's JSON (`density_fit`; ferric's KS default). The grid is placed in
each code's own symmetry frame; both frames put the symmetry elements on the
Cartesian axes, so they differ by an axis permutation or sign flip, an Oh
operation that maps every Lebedev rule onto itself: the grids coincide.
For H2, `sigma_g2` / `sigma_u2` pin the occupation through `irrep_nelec`
and `unblocked` is PySCF RKS with `symmetry = False`.

Geometry and basis are ferric's own (common.py): bundled BSE JSON, ferric's
Angstrom->Bohr constant, coordinates handed to PySCF in Bohr.

Labels. D2h irrep labels (B1g/B2g/B3g, B1u/B2u/B3u) depend on which axes the
detector picked, and PySCF and ferric pick differently; compare per-irrep
occupations as a MULTISET for D2h. C2v and Cs labels are fixed by convention.

Run:
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_scf_symmetry.py

`--rks-only` recomputes only the `rks_pbe` blocks and merges them into the
existing files, leaving every other block byte-identical (each `rks_pbe`
block carries its own `provenance`). Unblocked stretched-H2 runs do not
converge, so their recorded energy depends on the SCF trajectory and the BLAS
kernel; regenerating everything moves those numbers without any change in
the physics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "scf_symmetry"
MOL = common.ROOT / "testdata" / "molecules"
SYSTEMS = {
    "h2o": MOL / "water.xyz",
    "nh3": MOL / "nh3.xyz",
    "c2h4": MOL / "c2h4.xyz",
    "benzene": MOL / "benzene.xyz",
}
BASES = ("sto-3g", "6-31g")
CONV = {"conv_tol": 1e-12, "conv_tol_grad": 1e-8, "max_cycle": 200}
AUX = "def2-universal-jkfit"
KS_GRID = (75, 110)
KS_RECIPE = {
    "method": "dft.RKS",
    "xc": "PBE,PBE",
    "grid": {
        "atom_grid": list(KS_GRID),
        "prune": None,
        "partition": "Becke (original_becke)",
        "radii_adjust": "becke_atomic_radii_adjust",
    },
    "jk": f"RI-J (density_fit, {AUX} fed from ferric JSON); K unused",
}


def build(xyz: Path, basis: str, symmetry: bool):
    mol = common.build_pyscf_mol(xyz, basis)
    mol.symmetry = symmetry
    mol.build()
    return mol


def run_rhf(
    xyz: Path, basis: str, symmetry: bool, irrep_nelec=None, require_converged=True
):
    from pyscf import scf

    mol = build(xyz, basis, symmetry)
    mf = scf.RHF(mol)
    for k, v in CONV.items():
        setattr(mf, k, v)
    if irrep_nelec:
        mf.irrep_nelec = irrep_nelec
    mf.kernel()
    if require_converged and not mf.converged:
        raise RuntimeError(f"{xyz.name}/{basis} symmetry={symmetry} did not converge")
    return mol, mf


def run_rks(
    xyz: Path, basis: str, symmetry: bool, irrep_nelec=None, require_converged=True
):
    from pyscf import dft

    mol = build(xyz, basis, symmetry)
    symbols, _ = common.read_xyz(xyz)
    aux, cart = common.pyscf_basis(AUX, symbols)
    if cart:
        raise ValueError(f"{AUX}: Cartesian aux shells; not matched like-for-like")
    mf = dft.RKS(mol, xc=KS_RECIPE["xc"]).density_fit(auxbasis=aux)
    mf.grids.atom_grid = KS_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    for k, v in CONV.items():
        setattr(mf, k, v)
    mf.verbose = 0
    if irrep_nelec:
        mf.irrep_nelec = irrep_nelec
    mf.kernel()
    if require_converged and not mf.converged:
        raise RuntimeError(
            f"RKS {xyz.name}/{basis} symmetry={symmetry} did not converge"
        )
    return mol, mf


def occupations(mol, mf) -> dict:
    """Doubly occupied orbital count per irrep (PySCF labels)."""
    from pyscf import symm

    if not mol.symmetry:
        return {}
    ids = mf.mo_coeff.orbsym
    out: dict = {name: 0 for name in mol.irrep_name}
    for i, occ in zip(ids, mf.mo_occ):
        if occ > 0:
            out[symm.irrep_id2name(mol.groupname, i)] += 1
    return out


def homo_irrep(mol, mf) -> str:
    from pyscf import symm

    nocc = int(mf.mo_occ.sum() // 2)
    return symm.irrep_id2name(mol.groupname, mf.mo_coeff.orbsym[nocc - 1])


def block(mol, mf, e_unblocked=None) -> dict:
    d = {
        "energy": float(mf.e_tot),
        "group": mol.groupname,
        "irreps": list(mol.irrep_name),
        "docc_per_irrep": occupations(mol, mf),
        "nuclear_repulsion": float(mol.energy_nuc()),
    }
    if e_unblocked is not None:
        d["energy_unblocked"] = float(e_unblocked)
    return d


def rks_prov(system, xyz, basis) -> dict:
    p = prov(system, xyz, basis, {})
    p["keywords"] = {**KS_RECIPE, "symmetry": True, **CONV}
    p["grid"] = KS_RECIPE["grid"]
    p["aux_basis"] = AUX
    return p


def rks_ground(xyz: Path, basis: str) -> dict:
    _, off = run_rks(xyz, basis, False)
    mol, mf = run_rks(xyz, basis, True)
    return {
        "provenance": rks_prov("h2o", xyz, basis),
        "recipe": KS_RECIPE,
        "ground": block(mol, mf, off.e_tot),
    }


def rks_stretched(xyz: Path, basis: str) -> dict:
    _, off = run_rks(xyz, basis, False, require_converged=False)
    mol_g, g = run_rks(xyz, basis, True, irrep_nelec={"A1g": 2, "A1u": 0})
    mol_u, u = run_rks(xyz, basis, True, irrep_nelec={"A1g": 0, "A1u": 2})
    return {
        "provenance": rks_prov(xyz.stem, xyz, basis),
        "recipe": KS_RECIPE,
        "unblocked": {"energy": float(off.e_tot), "converged": bool(off.converged)},
        "sigma_g2": block(mol_g, g),
        "sigma_u2": block(mol_u, u),
    }


def prov(system, xyz, basis, extra):
    import pyscf

    symbols, coords = common.read_xyz(xyz)
    return common.provenance(
        code="pyscf",
        version=pyscf.__version__,
        keywords={"method": "RHF", "symmetry": True, **CONV, **extra},
        basis_name=basis,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        scf_conv=CONV,
        generator="scripts/validation/gen_scf_symmetry.py",
    )


def standard(system: str, xyz: Path, basis: str) -> dict:
    mol_off, mf_off = run_rhf(xyz, basis, False)
    mol, mf = run_rhf(xyz, basis, True)
    payload = {
        "provenance": prov(system, xyz, basis, {}),
        "unblocked": {"energy": float(mf_off.e_tot)},
        "ground": block(mol, mf, mf_off.e_tot),
    }
    if system == "h2o":
        out_of_plane = homo_irrep(mol, mf)
        occ = occupations(mol, mf)
        # Move the out-of-plane pair into the next a1 orbital.
        nelec = {name: 2 * n for name, n in occ.items()}
        nelec[out_of_plane] -= 2
        nelec["A1"] += 2
        mol_x, mf_x = run_rhf(xyz, basis, True, irrep_nelec=nelec)
        payload["excited"] = block(mol_x, mf_x)
        payload["excited"]["irrep_nelec"] = nelec
        payload["excited"]["moved_from"] = out_of_plane
        payload["rks_pbe"] = rks_ground(xyz, basis)
    return payload


def h2_stretched(tag: str) -> None:
    xyz = common.MOL_DIR / f"{tag}.xyz"
    basis = "cc-pvdz"
    _, off = run_rhf(xyz, basis, False, require_converged=False)
    mol_g, g = run_rhf(xyz, basis, True, irrep_nelec={"A1g": 2, "A1u": 0})
    mol_u, u = run_rhf(xyz, basis, True, irrep_nelec={"A1g": 0, "A1u": 2})
    payload = {
        "provenance": prov(tag, xyz, basis, {}),
        "unblocked": {
            "energy": float(off.e_tot),
            "converged": bool(off.converged),
        },
        "sigma_g2": block(mol_g, g),
        "sigma_u2": block(mol_u, u),
        "rks_pbe": rks_stretched(xyz, basis),
    }
    print(f"wrote {common.write_reference(ROW, tag, basis, payload)}")


def merge_rks(system: str, basis: str, block_: dict) -> None:
    """Replace only the `rks_pbe` block of an existing reference file."""
    path = common.reference_path(ROW, system, basis)
    payload = json.loads(path.read_text())
    payload["rks_pbe"] = block_
    print(f"merged rks_pbe into {common.write_reference(ROW, system, basis, payload)}")


def main_rks_only() -> int:
    for basis in BASES:
        merge_rks("h2o", basis, rks_ground(SYSTEMS["h2o"], basis))
    for tag in ("h2_r50", "h2_r10000"):
        merge_rks(
            tag, "cc-pvdz", rks_stretched(common.MOL_DIR / f"{tag}.xyz", "cc-pvdz")
        )
    return 0


def main() -> int:
    if "--rks-only" in sys.argv[1:]:
        return main_rks_only()
    for system, xyz in SYSTEMS.items():
        for basis in BASES:
            path = common.write_reference(
                ROW, system, basis, standard(system, xyz, basis)
            )
            print(f"wrote {path}")
    for tag in ("h2_r50", "h2_r10000"):
        h2_stretched(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
