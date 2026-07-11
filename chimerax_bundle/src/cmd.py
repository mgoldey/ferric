"""ChimeraX commands: thin wrappers over tools.active_site.

Each command mirrors one composable Python-side stage 1:1 — this file must
never contain domain logic (charge derivation, SCF, properties) itself, only
argument marshaling and a session-scoped name registry so results from one
command can be referenced by name in the next (`activesite embed` takes a
pocket name produced by `activesite charges`, etc.). Anything you can do from
these commands, you can do identically from a plain Python script or
notebook by importing tools.active_site directly — that's the point.
"""
from __future__ import annotations

import sys
from pathlib import Path

from chimerax.core.commands import CmdDesc, StringArg, BoolArg
from chimerax.atomic import AtomicStructuresArg

# tools/active_site lives in the main ferric repo, not inside this bundle —
# it is the single source of truth for the domain logic; this bundle only
# imports it. Path is inserted once, at import time.
_FERRIC_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_FERRIC_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_FERRIC_REPO_ROOT))


# Session-scoped registries: name -> object, so later commands can refer to
# earlier results by a user-chosen name instead of recomputing them.
_POCKETS: dict[str, object] = {}
_EMBEDDED: dict[str, object] = {}
_ENERGIES: dict[str, object] = {}


def _write_structure_pdb(session, structure, out_path: Path) -> None:
    from chimerax.pdb import save_pdb
    save_pdb(session, str(out_path), models=[structure])


def _write_structure_xyz(session, structure, out_path: Path) -> None:
    # ChimeraX's atomic API gives element symbols + coordinates directly —
    # no need to round-trip through a PDB writer + external xyz converter.
    atoms = structure.atoms
    lines = [str(len(atoms)), structure.name or "ligand"]
    for elem, (x, y, z) in zip(atoms.element_names, atoms.coords):
        lines.append(f"{elem} {x:.10f} {y:.10f} {z:.10f}")
    out_path.write_text("\n".join(lines) + "\n")


def activesite_charges(session, structures, name: str, ff: str = "AMBER"):
    """Derive a pocket's point charges from an open PDB structure and store
    it under `name` for later `activesite embed` calls.
    """
    from tools.active_site.pocket_charges import derive_pocket_charges
    import tempfile

    if len(structures) != 1:
        raise ValueError("activesite_charges expects exactly one structure")
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "pocket.pdb"
        _write_structure_pdb(session, structures[0], pdb_path)
        pocket = derive_pocket_charges(pdb_path, ff=ff)
    _POCKETS[name] = pocket
    session.logger.info(
        f"activesite charges: stored pocket '{name}' ({pocket.n_charges} point charges, ff={ff})"
    )


activesite_charges_desc = CmdDesc(
    required=[("structures", AtomicStructuresArg), ("name", StringArg)],
    keyword=[("ff", StringArg)],
    synopsis="Derive pocket point charges from an open PDB structure",
)


def activesite_embed(session, structures, pocket_name: str, name: str, basis: str = "def2-svp"):
    """Embed an open ligand structure against a stored pocket, under `name`."""
    from tools.active_site.ligand_embedding import embed_ligand
    import tempfile

    if pocket_name not in _POCKETS:
        raise ValueError(f"no pocket named '{pocket_name}' — run activesite_charges first")
    if len(structures) != 1:
        raise ValueError("activesite_embed expects exactly one structure")
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = Path(tmpdir) / "ligand.xyz"
        _write_structure_xyz(session, structures[0], xyz_path)
        embedded = embed_ligand(xyz_path, pocket=_POCKETS[pocket_name], basis=basis)
    _EMBEDDED[name] = embedded
    n_pc = len(embedded.point_charges) if embedded.point_charges else 0
    session.logger.info(
        f"activesite embed: stored embedded ligand '{name}' "
        f"({embedded.mol.natoms()} atoms, basis={basis}, {n_pc} pocket charges after overlap filter)"
    )


activesite_embed_desc = CmdDesc(
    required=[("structures", AtomicStructuresArg), ("pocket_name", StringArg), ("name", StringArg)],
    keyword=[("basis", StringArg)],
    synopsis="Embed a ligand structure against a stored pocket",
)


def activesite_energy(session, embedded_name: str, name: str,
                       method: str = "rhf", xc: str = None, field: bool = True):
    """Compute an energy for a stored embedded ligand, under `name`."""
    from tools.active_site.energy import compute_energy

    if embedded_name not in _EMBEDDED:
        raise ValueError(f"no embedded ligand named '{embedded_name}' — run activesite_embed first")
    result = compute_energy(_EMBEDDED[embedded_name], method=method, xc=xc, use_field=field)
    _ENERGIES[name] = result
    session.logger.info(
        f"activesite energy: '{name}' = {result.energy:.8f} Ha "
        f"(method={method}, field={result.field}, converged={result.converged})"
    )


activesite_energy_desc = CmdDesc(
    required=[("embedded_name", StringArg), ("name", StringArg)],
    keyword=[("method", StringArg), ("xc", StringArg), ("field", BoolArg)],
    synopsis="Compute an energy for a stored embedded ligand",
)


def activesite_properties(session, embedded_name: str, energy_name: str):
    """Report Hirshfeld/Löwdin charges for a stored energy result."""
    from tools.active_site.properties import compute_charges

    if embedded_name not in _EMBEDDED:
        raise ValueError(f"no embedded ligand named '{embedded_name}'")
    if energy_name not in _ENERGIES:
        raise ValueError(f"no energy result named '{energy_name}'")
    charges = compute_charges(_EMBEDDED[embedded_name], _ENERGIES[energy_name])
    session.logger.info(f"activesite properties ('{energy_name}'):")
    session.logger.info(f"  Hirshfeld: {charges['hirshfeld']}")
    session.logger.info(f"  Lowdin:    {charges['lowdin']}")


activesite_properties_desc = CmdDesc(
    required=[("embedded_name", StringArg), ("energy_name", StringArg)],
    synopsis="Report Hirshfeld/Löwdin charges for a stored energy result",
)


def activesite_bindingenergy(session, ligand_structures, pocket_structures,
                              basis: str = "def2-svp", method: str = "rhf", xc: str = None,
                              ff: str = "AMBER"):
    """Straight passthrough to compute_binding_energy — the common-case
    one-shot command, built from the same stages as the others above.
    """
    from tools.active_site.binding_energy import compute_binding_energy
    import tempfile

    if len(ligand_structures) != 1 or len(pocket_structures) != 1:
        raise ValueError("activesite_bindingenergy expects exactly one ligand and one pocket structure")
    with tempfile.TemporaryDirectory() as tmpdir:
        ligand_xyz = Path(tmpdir) / "ligand.xyz"
        pocket_pdb = Path(tmpdir) / "pocket.pdb"
        _write_structure_xyz(session, ligand_structures[0], ligand_xyz)
        _write_structure_pdb(session, pocket_structures[0], pocket_pdb)
        result = compute_binding_energy(
            ligand_xyz, pocket_pdb, basis=basis, method=method, xc=xc, ff=ff,
        )
    session.logger.info(
        f"activesite bindingenergy: dE = {result.delta_e_kcal_mol:.2f} kcal/mol "
        f"(E_vacuum={result.e_vacuum:.8f} Ha, E_field={result.e_field:.8f} Ha, "
        f"n_pocket_charges={result.n_pocket_charges})"
    )


activesite_bindingenergy_desc = CmdDesc(
    required=[("ligand_structures", AtomicStructuresArg), ("pocket_structures", AtomicStructuresArg)],
    keyword=[("basis", StringArg), ("method", StringArg), ("xc", StringArg), ("ff", StringArg)],
    synopsis="Compute field-vs-vacuum binding energy for a ligand in a pocket",
)
