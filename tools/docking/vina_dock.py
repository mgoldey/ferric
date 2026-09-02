"""AutoDock Vina pose search, as tier 1 of a cost hierarchy.

## Where this sits

    tier 1  Vina (empirical)     ~10 us/pose   10^5-10^6 poses   SEARCH
    tier 2  MMFF / GFN-FF        ~ms           10^2-10^3         relax, declash
    tier 3  GFN2-xTB             ~0.5 s        10-10^2           rank
    tier 4  DFT + dispersion     minutes-hours 1-10              final energetics

Each tier exists to DISCARD, cheaply, what the next cannot afford to examine.
Vina's scoring is empirical and crude; that is fine, because only its POSES are
wanted here. The physics comes from the tiers above it.

## What Vina actually searches

6 rigid-body degrees of freedom plus the ligand's rotatable torsions (9 for
danuglipron), by Monte Carlo with BFGS local refinement. That is the same
~15-dimensional space a 4 ps GFN2 anneal failed to explore in 62 minutes, and
Vina covers it in seconds -- because its per-pose cost is ~5 orders of magnitude
lower. Cheap scoring is what MAKES a search possible.

## Honest limits

- The receptor is RIGID. Induced fit is not modelled.
- Vina's score is an empirical sum fitted to binding data. It is a pose-ranking
  heuristic, not a binding free energy, and it is NOT used as one here.
- A docking pose is a hypothesis. `redock_rmsd` against a known bound pose is
  the only way to know whether the search works on a given target, and it is
  the first thing this module should be used for.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Vina and Meeko are the `docking` extra, not core dependencies (see
# pyproject.toml for why). Every import of them below is deferred to call time
# and routed through this, so that a missing extra says what to install instead
# of surfacing a bare ModuleNotFoundError from three frames down.
_INSTALL_HINT = (
    "install the docking extra: `pip install 'ferric[docking]'` "
    "(or `uv sync --extra docking`). Note that PyPI's vina ships wheels for "
    "CPython 3.8-3.12 only; on 3.13+ it builds from source and needs Boost."
)


def _require(module: str):
    """Import an optional docking dependency, or raise an actionable error."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"{module} is required for docking -- {_INSTALL_HINT}") from e


@dataclass
class DockedPose:
    """One pose from the search, in the receptor's coordinate frame."""
    symbols: list[str]
    coords_angstrom: list[tuple[float, float, float]]
    vina_score: float          # kcal/mol, empirical -- a ranking heuristic only
    rank: int


@dataclass
class DockResult:
    """Outcome of one docking run.

    `poses` is empty and `error` set when the search did not run. As everywhere
    in this codebase, a failure never comes back as a neutral-looking number.
    """
    poses: list[DockedPose] = field(default_factory=list)
    error: str | None = None
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.poses)

    @property
    def best(self) -> DockedPose | None:
        return self.poses[0] if self.poses else None


def prepare_receptor(pdb_path: str | Path, out_pdbqt: str | Path) -> Path:
    """Convert a receptor PDB to the PDBQT Vina requires.

    Uses Meeko's receptor path. Raises with an actionable message rather than
    returning a half-prepared file, because a silently malformed receptor gives
    poses that look plausible and are meaningless.
    """
    _require("meeko")  # fail here, not inside the CLI subprocess below

    pdb_path, out_pdbqt = Path(pdb_path), Path(out_pdbqt)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"receptor PDB not found: {pdb_path}")

    # Meeko's polymer/receptor prep is version-sensitive; shell out to its CLI,
    # which is the supported entry point and gives a readable error.
    import subprocess

    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    # --allow_bad_res: a 2.82 A cryo-EM model has residues with missing heavy
    # atoms, and Meeko refuses the whole structure without this. The flag DROPS
    # such residues, so it is only safe once you have checked that none of them
    # line the binding site -- verified for 7LCJ on 2026-08-29: 0 of the 47
    # residues within 6 A of the bound ligand were lost (atom count rose
    # 3223 -> 3882, which is Meeko adding hydrogens). Re-run that check for any
    # new receptor rather than assuming it carries over.
    proc = subprocess.run(
        ["mk_prepare_receptor.py", "--read_pdb", str(pdb_path),
         "-o", str(out_pdbqt.with_suffix("")), "-p", "--allow_bad_res"],
        capture_output=True, text=True,
    )
    produced = out_pdbqt.with_suffix(".pdbqt")
    if not produced.is_file():
        raise RuntimeError(
            "mk_prepare_receptor.py did not produce a PDBQT.\n"
            f"stdout: {proc.stdout[-800:]}\nstderr: {proc.stderr[-800:]}"
        )
    return produced


def _ligand_pdbqt_from_rdkit(mol) -> str:
    """Meeko-prepared PDBQT string for an RDKit mol WITH 3D coordinates."""
    meeko = _require("meeko")

    prep = meeko.MoleculePreparation()
    setups = prep.prepare(mol)
    if not setups:
        raise RuntimeError("Meeko produced no setup for this ligand")
    pdbqt, ok, err = meeko.PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"Meeko PDBQT write failed: {err}")
    return pdbqt


# PDBQT's last column is an AUTODOCK TYPE, not an element symbol. The types
# encode chemistry the docking scoring function needs -- "OA" is an H-bond
# ACCEPTING oxygen, "NA" an accepting nitrogen, "A" aromatic carbon -- and they
# collide with real element symbols: naive `.capitalize()` turns OA into the
# non-existent element "Oa" and NA into sodium. That is not cosmetic: the first
# run of the redocking validation produced 20 perfectly good poses and reported
# "NO ALIGNABLE POSE", because RDKit rejected every geometry over the bogus
# element. Mapped explicitly, with a documented fallback.
_AUTODOCK_TO_ELEMENT = {
    "A": "C",    # aromatic carbon
    "C": "C",
    "OA": "O",   # H-bond acceptor oxygen
    "O": "O",
    "NA": "N",   # H-bond acceptor nitrogen -- NOT sodium
    "NS": "N",
    "N": "N",
    "SA": "S",   # H-bond acceptor sulfur
    "S": "S",
    "HD": "H",   # polar hydrogen (donor)
    "H": "H",
    "F": "F", "Cl": "CL", "Br": "BR", "I": "I", "P": "P",
    "Mg": "MG", "Mn": "MN", "Zn": "ZN", "Ca": "CA", "Fe": "FE",
}


def _element_from_autodock_type(raw: str) -> str:
    """Element symbol for a PDBQT AutoDock type.

    Unknown types fall back to the leading alphabetic run, title-cased -- which
    is right for real two-letter elements (CL, BR) and is the best guess for
    anything this table has not seen. It never invents a two-letter symbol out
    of a one-letter element plus a type suffix, which is the bug this replaces.
    """
    t = raw.strip()
    for key, el in _AUTODOCK_TO_ELEMENT.items():
        if t.upper() == key.upper():
            return el.capitalize()
    lead = "".join(ch for ch in t if ch.isalpha())
    return (lead[:2] if len(lead) >= 2 and lead[:2].upper() in
            ("CL", "BR", "SI", "SE", "ZN", "FE", "MG", "MN", "CA")
            else lead[:1]).capitalize()


def _parse_pdbqt_models(text: str):
    """Split a Vina output PDBQT into (symbols, coords, score) per MODEL."""
    models, cur, score, syms, crds = [], False, None, [], []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            cur, score, syms, crds = True, None, [], []
        elif line.startswith("REMARK VINA RESULT"):
            parts = line.split()
            if len(parts) >= 4:
                score = float(parts[3])
        elif line.startswith(("ATOM", "HETATM")) and cur:
            raw = (line[77:79].strip() or line[12:16].strip())
            syms.append(_element_from_autodock_type(raw))
            crds.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        elif line.startswith("ENDMDL") and cur:
            models.append((syms, crds, score))
            cur = False
    return models


def dock_ligand(
    mol,
    receptor_pdbqt: str | Path,
    box_center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (24.0, 24.0, 24.0),
    exhaustiveness: int = 32,
    n_poses: int = 20,
    seed: int = 0xF00D,
) -> DockResult:
    """Search poses for an RDKit mol (with 3D coords) in a prepared receptor.

    `seed` is fixed by default: Vina's search is stochastic, and an unseeded run
    would make a pose ranking irreproducible for reasons unrelated to chemistry
    -- the same discipline `tools.morph.embed` applies to ETKDG.

    `exhaustiveness` trades wall time for search thoroughness. Vina's default is
    8; 32 is used here because the failure being fixed is a SEARCH failure, and
    under-searching would reproduce it in a new form.
    """
    Vina = _require("vina").Vina

    receptor_pdbqt = Path(receptor_pdbqt)
    if not receptor_pdbqt.is_file():
        return DockResult(error=f"receptor PDBQT not found: {receptor_pdbqt}")

    try:
        lig_pdbqt = _ligand_pdbqt_from_rdkit(mol)
    except Exception as e:  # noqa: BLE001
        return DockResult(error=f"ligand preparation failed: {type(e).__name__}: {e}")

    try:
        v = Vina(sf_name="vina", seed=seed, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(center=list(box_center), box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        out = v.poses(n_poses=n_poses)
    except Exception as e:  # noqa: BLE001
        return DockResult(error=f"Vina docking failed: {type(e).__name__}: {e}",
                          box_center=box_center, box_size=box_size)

    models = _parse_pdbqt_models(out)
    poses = [
        DockedPose(symbols=s, coords_angstrom=c,
                   vina_score=sc if sc is not None else float("nan"), rank=i)
        for i, (s, c, sc) in enumerate(models)
    ]
    if not poses:
        return DockResult(error="Vina returned no parseable pose",
                          box_center=box_center, box_size=box_size)
    return DockResult(poses=poses, box_center=box_center, box_size=box_size)
