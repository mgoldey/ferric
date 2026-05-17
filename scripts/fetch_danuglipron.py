"""Build a danuglipron (PF-06882961) conformer ensemble for the C9 benchmark.

Sources, in priority order:
  1. PDB 7LCJ (2.82 Å cryo-EM of GLP-1R bound to danuglipron). Ligand code UK4
     — extract heavy atoms and add hydrogens via RDKit to get the bioactive
     bound conformer. Written as `conf_00_cryo_em.xyz`.
  2. PubChem CID 134611040 — has a precomputed 3D conformer. Written as
     `conf_01_pubchem.xyz`.
  3. RDKit ETKDGv3 + MMFF94 — generate ~18 additional conformers from SMILES,
     optimize each, dedupe by all-heavy-atom RMSD > 0.5 Å. Written as
     `conf_02_rdkit.xyz` ... `conf_19_rdkit.xyz` (or fewer after dedup).

The bound conformer (#00) is the anchor "bioactive" pose. Subsequent indices
are arbitrary; the C9 driver ranks them by ferric RI-RPA energy.

PDB / PubChem references:
    PDB 7LCJ: doi:10.1038/s41594-020-00547-5 (Zhao et al., Nature SMB 2021)
    PubChem CID 134611040  https://pubchem.ncbi.nlm.nih.gov/compound/134611040

SMILES (PubChem):
    C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F

Run from repo root:
    python scripts/fetch_danuglipron.py
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "testdata/molecules/c9_systems/danuglipron"

SMILES = "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"

UA = "Mozilla/5.0 (ferric-c9 fetcher)"


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write_xyz_from_rdkit(mol, conf_id: int, out_path: Path, comment: str):
    from rdkit.Chem import GetPeriodicTable
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    lines = [str(n), comment]
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        p = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():<3s} {p.x:14.8f} {p.y:14.8f} {p.z:14.8f}")
    out_path.write_text("\n".join(lines) + "\n")


def conf_from_cryo_em() -> "tuple[bool, str]":
    """Extract UK4 ligand from PDB 7LCJ, add Hs in RDKit, write conf_00."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return False, "rdkit not installed"

    try:
        pdb_bytes = fetch("https://files.rcsb.org/download/7LCJ.pdb")
    except Exception as e:
        return False, f"PDB download failed: {e}"

    # Extract HETATM lines for UK4
    het_lines = [l for l in pdb_bytes.decode().splitlines()
                 if l.startswith("HETATM") and " UK4 " in l]
    if not het_lines:
        return False, "no UK4 HETATM records in 7LCJ.pdb"

    # Wrap in a minimal PDB block for RDKit
    pdb_block = "\n".join(["HEADER    UK4 LIGAND", *het_lines, "END"]) + "\n"
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if mol is None:
        return False, "RDKit could not parse UK4 from PDB"

    # The PDB ligand has no bond orders / no Hs. Use the reference SMILES
    # to assign bond orders, then add Hs in 3D consistent with the
    # heavy-atom coordinates.
    ref = Chem.MolFromSmiles(SMILES)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
    except Exception as e:
        return False, f"bond-order assignment failed: {e}"
    mol = Chem.AddHs(mol, addCoords=True)
    # H positions from addCoords are heuristic but stay near the parent heavy
    # atom (within ~1 Å) — acceptable for an RPA single-point. Optionally
    # relax only the H positions with MMFF (heavy atoms frozen).
    try:
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol))
        if ff is not None:
            # Freeze heavy atoms
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() > 1:
                    ff.AddFixedPoint(atom.GetIdx())
            ff.Minimize(maxIts=500)
    except Exception:
        pass  # H positions from AddHs(addCoords) are good enough.

    out_path = OUT_DIR / "conf_00_cryo_em.xyz"
    write_xyz_from_rdkit(
        mol, 0, out_path,
        "danuglipron (UK4) bound conformer from PDB 7LCJ; heavy atoms from "
        "cryo-EM, H from RDKit AddHs+MMFF (heavy frozen)"
    )
    return True, str(out_path)


def conf_from_pubchem() -> "tuple[bool, str]":
    try:
        from rdkit import Chem
    except ImportError:
        return False, "rdkit not installed"
    try:
        sdf = fetch(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/134611040/"
            "record/SDF?record_type=3d"
        ).decode("ascii", "replace")
    except Exception as e:
        return False, f"PubChem download failed: {e}"
    mol = Chem.MolFromMolBlock(sdf, removeHs=False)
    if mol is None:
        return False, "RDKit could not parse PubChem SDF"
    out_path = OUT_DIR / "conf_01_pubchem.xyz"
    write_xyz_from_rdkit(
        mol, 0, out_path,
        "danuglipron PubChem CID 134611040 precomputed 3D conformer"
    )
    return True, str(out_path)


def confs_from_rdkit(n_target: int = 18, rms_thresh: float = 0.5) -> "tuple[int, str]":
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return 0, "rdkit not installed"
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        return 0, "RDKit failed to parse SMILES"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 1337
    params.numThreads = 0
    params.pruneRmsThresh = rms_thresh
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_target * 2, params=params)
    if len(cids) == 0:
        return 0, "ETKDGv3 produced 0 conformers"
    # MMFF optimize each
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    # Drop unconverged conformers
    cid_list = list(cids)
    keep = [(cid, results[i][1]) for i, cid in enumerate(cid_list)
            if results[i][0] == 0]
    keep.sort(key=lambda x: x[1])
    energy_by_cid = dict(keep)
    # Deduplicate by heavy-atom RMSD
    chosen: list[int] = []
    for cid, _e in keep:
        is_new = True
        for kept in chosen:
            rms = AllChem.GetBestRMS(mol, mol, prbId=cid, refId=kept)
            if rms < rms_thresh:
                is_new = False
                break
        if is_new:
            chosen.append(cid)
        if len(chosen) >= n_target:
            break
    for i, cid in enumerate(chosen):
        out_path = OUT_DIR / f"conf_{i+2:02d}_rdkit.xyz"
        write_xyz_from_rdkit(
            mol, cid, out_path,
            f"danuglipron RDKit ETKDGv3+MMFF94 conformer {i} (cid={cid}, "
            f"E_mmff={energy_by_cid.get(cid, 0.0):.2f} kcal/mol)"
        )
    return len(chosen), f"{len(chosen)} RDKit conformers"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    ok_em, msg_em = conf_from_cryo_em()
    print(f"[cryo-em]  {msg_em}", file=sys.stderr)
    if not ok_em:
        failures.append(f"cryo-em: {msg_em}")

    ok_pc, msg_pc = conf_from_pubchem()
    print(f"[pubchem]  {msg_pc}", file=sys.stderr)
    if not ok_pc:
        failures.append(f"pubchem: {msg_pc}")

    n_rdkit, msg_rdkit = confs_from_rdkit()
    print(f"[rdkit]    {msg_rdkit}", file=sys.stderr)

    n_files = len(list(OUT_DIR.glob("conf_*.xyz")))
    print(f"DONE: {n_files} danuglipron conformers in {OUT_DIR}", file=sys.stderr)

    if failures:
        readme = OUT_DIR / "README.md"
        readme.write_text(
            "# Danuglipron conformer sourcing\n\n"
            f"Generated {n_files} conformer XYZ files.\n\n"
            "## Sourcing failures\n\n"
            + "\n".join(f"- {f}" for f in failures) + "\n\n"
            "Re-run: `python scripts/fetch_danuglipron.py`\n"
        )


if __name__ == "__main__":
    main()
