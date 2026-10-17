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

Geometry and basis are ferric's own (common.py): bundled BSE JSON, ferric's
Angstrom->Bohr constant, coordinates handed to PySCF in Bohr.

Labels. D2h irrep labels (B1g/B2g/B3g, B1u/B2u/B3u) depend on which axes the
detector picked, and PySCF and ferric pick differently; compare per-irrep
occupations as a MULTISET for D2h. C2v and Cs labels are fixed by convention.

Run:
    scripts/validation/run_slot.sh --light -- \
        uv run --no-sync python scripts/validation/gen_scf_symmetry.py
"""

from __future__ import annotations

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
    }
    print(f"wrote {common.write_reference(ROW, tag, basis, payload)}")


def main() -> int:
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
