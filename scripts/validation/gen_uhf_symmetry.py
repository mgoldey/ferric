"""PySCF references for point-group-blocked UHF and ROHF (issue #294, stage 3a).

Consumer: crates/ferric-scf/tests/validation_uhf_symmetry.rs (ignored PySCF
comparison); references land in testdata/reference/validation/uhf_symmetry/.

Per system and basis the reference records, for UHF and for ROHF, PySCF's
symmetry-blocked energy next to the unblocked energy, and the (n_alpha, n_beta)
orbital count per irrep. Extras:

* `h2o_cation`: `excited` moves the hole from the out-of-plane 1b1 orbital
  to the 3a1 orbital (2A1 instead of 2B1), reached through `irrep_nelec`
  ({irrep: (n_alpha, n_beta)}), a state plain aufbau cannot reach.
* `h2_r2p0` (singlet H2 at 2.0 Angstrom, 6-31G): the negative control.
  `blocked` is PySCF's symmetric UHF (sigma_g^2, equal to RHF); `unblocked` is
  PySCF's UHF after following its internal instability to the lower
  spatially symmetry-broken solution.

Geometry and basis are ferric's own (common.py). Irrep labels: C2v and Cs
labels follow the standard convention, D2h labels depend on which axes the
detector picked (PySCF and ferric differ), so D2h occupations are compared as
a multiset.

Run (after committing this script so provenance.git_head names it):
    OMP_NUM_THREADS=2 ~/qc/ferric/.venv/bin/python scripts/validation/gen_uhf_symmetry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "uhf_symmetry"
MOL = common.ROOT / "testdata" / "molecules"
VAL = common.MOL_DIR
CONV = {"conv_tol": 1e-12, "conv_tol_grad": 1e-8, "max_cycle": 300}

# system -> (xyz, charge, multiplicity, bases)
SYSTEMS = {
    "nh2": (VAL / "nh2.xyz", 0, 2, ("sto-3g", "6-31g")),
    "ch3": (VAL / "ch3.xyz", 0, 2, ("sto-3g", "6-31g")),
    "o2": (MOL / "o2.xyz", 0, 3, ("sto-3g", "6-31g")),
    "ch2_triplet": (VAL / "ch2_triplet.xyz", 0, 3, ("sto-3g", "6-31g")),
    # NO2/STO-3G has two distinct converged UHF states (minao vs hcore guess);
    # 6-31G has one, so only that basis is a reference.
    "no2": (VAL / "no2.xyz", 0, 2, ("6-31g",)),
    "h2o_cation": (MOL / "water.xyz", 1, 2, ("sto-3g", "6-31g")),
}
NEGATIVE = {"h2_r2p0": (VAL / "h2_r2p0.xyz", "6-31g")}


def build(xyz, basis, charge, mult, symmetry):
    mol = common.build_pyscf_mol(xyz, basis, charge, mult)
    mol.symmetry = symmetry
    mol.build()
    return mol


def run(xyz, basis, charge, mult, kind, symmetry, irrep_nelec=None):
    from pyscf import scf

    mol = build(xyz, basis, charge, mult, symmetry)
    mf = (scf.UHF if kind == "uhf" else scf.ROHF)(mol)
    for k, v in CONV.items():
        setattr(mf, k, v)
    if irrep_nelec:
        mf.irrep_nelec = irrep_nelec
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{xyz.name}/{basis} {kind} sym={symmetry} not converged")
    return mol, mf


def occupations(mol, mf, kind) -> dict:
    """(n_alpha, n_beta) orbital count per irrep (PySCF labels)."""
    from pyscf import symm

    out = {name: [0, 0] for name in mol.irrep_name}
    if kind == "uhf":
        for s in (0, 1):
            ids = mf.mo_coeff[s].orbsym
            for i, occ in zip(ids, mf.mo_occ[s]):
                if occ > 0:
                    out[symm.irrep_id2name(mol.groupname, i)][s] += 1
    else:
        ids = mf.mo_coeff.orbsym
        for i, occ in zip(ids, mf.mo_occ):
            name = symm.irrep_id2name(mol.groupname, i)
            if occ > 0:
                out[name][0] += 1
            if occ > 1:
                out[name][1] += 1
    return {k: tuple(v) for k, v in out.items()}


def s_squared(mf, kind) -> float:
    if kind == "uhf":
        return float(mf.spin_square()[0])
    ss = mf.spin_square()[0]
    return float(ss)


def block(mol, mf, kind) -> dict:
    return {
        "energy": float(mf.e_tot),
        "group": mol.groupname,
        "irreps": list(mol.irrep_name),
        "occ_per_irrep": {k: list(v) for k, v in occupations(mol, mf, kind).items()},
        "s_squared": s_squared(mf, kind),
        "nuclear_repulsion": float(mol.energy_nuc()),
    }


def prov(xyz, basis, charge, mult, extra):
    import pyscf

    symbols, coords = common.read_xyz(xyz)
    return common.provenance(
        code="pyscf",
        version=pyscf.__version__,
        keywords={
            "method": "UHF/ROHF",
            "symmetry": True,
            "charge": charge,
            "multiplicity": mult,
            **CONV,
            **extra,
        },
        basis_name=basis,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        scf_conv=CONV,
        generator="scripts/validation/gen_uhf_symmetry.py",
    )


def standard(xyz, basis, charge, mult) -> dict:
    payload = {"provenance": prov(xyz, basis, charge, mult, {})}
    for kind in ("uhf", "rohf"):
        _, off = run(xyz, basis, charge, mult, kind, False)
        mol, on = run(xyz, basis, charge, mult, kind, True)
        payload[kind] = {
            "unblocked": {
                "energy": float(off.e_tot),
                "s_squared": s_squared(off, kind),
            },
            "blocked": block(mol, on, kind),
        }
    return payload


def cation_excited(payload, xyz, basis, charge, mult) -> None:
    """Move the beta-spin hole: ground is 2B1 (1b1 hole), excited 2A1 (3a1 hole)."""
    for kind in ("uhf", "rohf"):
        occ = {
            k: tuple(v) for k, v in payload[kind]["blocked"]["occ_per_irrep"].items()
        }
        # The beta electron count of the singly occupied irrep (alpha > beta).
        hole = next(k for k, (a, b) in occ.items() if a > b)
        req = dict(occ)
        a, b = req["A1"]
        req["A1"] = (a, b - 1)
        ha, hb = req[hole]
        req[hole] = (ha, hb + 1)
        mol, mf = run(xyz, basis, charge, mult, kind, True, irrep_nelec=req)
        payload[kind]["excited"] = block(mol, mf, kind)
        payload[kind]["excited"]["irrep_nelec"] = {k: list(v) for k, v in req.items()}
        payload[kind]["excited"]["moved_from"] = hole


def followed_unrestricted(xyz, basis):
    """Singlet UHF without symmetry, instability followed to a stable minimum."""
    from pyscf import scf

    mol = build(xyz, basis, 0, 1, False)
    mf = scf.UHF(mol)
    for k, v in CONV.items():
        setattr(mf, k, v)
    mf.kernel()
    for _ in range(10):
        mo, _, stable, _ = mf.stability(internal=True, return_status=True)
        if stable:
            break
        dm = mf.make_rdm1(mo, mf.mo_occ)
        mf.kernel(dm)
    if not mf.converged:
        raise RuntimeError("unrestricted H2 reference did not converge")
    return mf


def negative(tag, xyz, basis) -> None:
    mol, on = run(xyz, basis, 0, 1, "uhf", True)
    off = followed_unrestricted(xyz, basis)
    payload = {
        "provenance": prov(xyz, basis, 0, 1, {}),
        "unblocked": {"energy": float(off.e_tot), "s_squared": s_squared(off, "uhf")},
        "blocked": block(mol, on, "uhf"),
    }
    print(f"wrote {common.write_reference(ROW, tag, basis, payload)}")


def main() -> int:
    for system, (xyz, charge, mult, bases) in SYSTEMS.items():
        for basis in bases:
            payload = standard(xyz, basis, charge, mult)
            if system == "h2o_cation":
                cation_excited(payload, xyz, basis, charge, mult)
            print(f"wrote {common.write_reference(ROW, system, basis, payload)}")
    for tag, (xyz, basis) in NEGATIVE.items():
        negative(tag, xyz, basis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
