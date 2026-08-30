"""Align a generated conformer into the pocket frame of a reference bound pose.

## Why this module has to exist

Measured 2026-08-29 on the committed danuglipron ensemble: only
`conf_00_cryo_em` lives in the 7LCJ pocket's coordinate frame (centroid
~(125.4, 157.0, 113.3) Å, closest pocket charge 1.99 Å, 353 charges within
10 Bohr). Every other conformer -- PubChem's and all 18 RDKit ones -- is
origin-centred, i.e. **176-179 Å away from the pocket**. A freshly embedded
analogue is likewise at the origin.

So any fit score computed on an unaligned conformer is meaningless. Without
this step the honest outcome is "no pocket charges within the cutoff"
(`tools.campaign.fit` reports exactly that, rather than an interaction of 0),
and the whole analogue arm would produce no data.

## What the alignment does, and its one real limitation

`align_to_reference` performs a least-squares rigid-body superposition
(Kabsch) of a common substructure onto the bound reference pose. For an
analogue, the common substructure is found by **maximum common substructure
(MCS)** against the parent, so the shared scaffold -- benzimidazole core,
piperidine linker, distal aryl -- is what gets overlaid, and the modified
region is placed by whatever the scaffold dictates.

**This is a rigid overlay, not docking.** It answers "if this analogue keeps the
parent's scaffold placement, what does its electrostatic environment look like?"
It does NOT search for the analogue's own best pose. A modification that would
genuinely rebind in a different orientation is therefore scored pessimistically,
and that limitation is a property of the measurement, not a bug -- it is stated
in `AlignedPose.method` so it travels with every number derived from it.

The alternative (full redocking per analogue) needs a docking engine this repo
does not have and would introduce a second, larger source of error. A rigid
scaffold overlay is the honest cheap answer, and it is the right one for a set
of analogues designed specifically to preserve that scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedPose:
    """A conformer moved into the reference's frame.

    `rmsd_angstrom` is the post-fit RMSD over the matched atom pairs. A large
    value means the analogue's scaffold genuinely cannot adopt the parent's
    placement, which is itself a fit finding -- so it is reported, not hidden.
    """
    symbols: list[str]
    coords_angstrom: list[tuple[float, float, float]]
    n_matched_atoms: int
    rmsd_angstrom: float
    method: str = (
        "rigid Kabsch superposition of an MCS scaffold onto a bound reference "
        "pose; NOT docking -- the analogue is not allowed to find its own "
        "orientation, so fit scores are conditional on keeping the parent's "
        "scaffold placement"
    )
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.coords_angstrom)


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimal rigid-body rotation+translation taking `mobile` onto `target`.

    Both are (n, 3). Returns (R, t, rmsd) with `mobile @ R.T + t ~= target`.
    Includes the reflection guard: a naive SVD can return an improper rotation
    (det = -1), which would MIRROR the molecule -- silently inverting every
    stereocentre while producing an excellent-looking RMSD.
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"expected matching (n,3) arrays, got {mobile.shape} and {target.shape}")
    if len(mobile) < 3:
        raise ValueError(
            f"need at least 3 atom pairs to define a rigid orientation, got {len(mobile)}"
        )

    cm, ct = mobile.mean(axis=0), target.mean(axis=0)
    m, t = mobile - cm, target - ct
    u, _, vt = np.linalg.svd(m.T @ t)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    # Flip the least-significant axis if the naive solution is a reflection.
    correction = np.diag([1.0, 1.0, d])
    R = vt.T @ correction @ u.T
    aligned = m @ R.T
    rmsd = float(np.sqrt(((aligned - t) ** 2).sum(axis=1).mean()))
    return R, ct - cm @ R.T, rmsd


# Conventional bar for a successful docking pose (Kramer/Gedeck and the wider
# docking literature). A pose beyond this is not "slightly off" -- it is a
# different binding mode, and no scoring function can rank it meaningfully.
DOCKING_SUCCESS_RMSD_ANGSTROM = 2.0


def pose_quality_gate(
    aligned: "list[AlignedPose]",
    threshold_angstrom: float = DOCKING_SUCCESS_RMSD_ANGSTROM,
) -> tuple[bool, str]:
    """Are ANY generated poses close enough to a known bound pose to score?

    **Run this before scoring anything.** It is the check whose absence cost
    this campaign four measurement rounds (2026-08-29): the fit metric was
    characterised in detail -- precision, charge confound, size correlation,
    relaxation response -- while being fed poses 2-4 A from the binding mode.
    Every one of those measurements was valid and none of them mattered,
    because the geometries could not express the differences being ranked.

    The measurement that should have come first: align the generated conformers
    onto the experimentally determined pose and look at the RMSD. On the
    committed danuglipron ensemble the best of 20 was **2.23 A** and the rest
    2.70-3.70 A, against a 2.0 A success bar -- i.e. zero usable poses.

    Why unbiased conformer generation fails here, and will fail again on any
    similar target: danuglipron has 9 rotatable bonds over 41 heavy atoms.
    ETKDG samples FREE-SOLUTION torsional space; the bound conformer is one
    receptor-selected point in it. More conformers does not fix this in any
    practical number (100 fresh poses did no better than the committed 20).
    The fix is to CONSTRAIN generation to the bound scaffold, or to dock.

    Returns `(passed, detail)`. `passed` is True only if at least one pose is
    within `threshold_angstrom`.
    """
    usable = [a for a in aligned if a.ok]
    if not usable:
        return False, "no pose could be aligned at all; nothing to assess"

    rmsds = sorted(a.rmsd_angstrom for a in usable)
    best = rmsds[0]
    n_ok = sum(1 for r in rmsds if r <= threshold_angstrom)
    if n_ok == 0:
        return False, (
            f"NO USABLE POSE: best scaffold RMSD vs the reference pose is "
            f"{best:.2f} A over {len(usable)} poses, against a "
            f"{threshold_angstrom:.1f} A docking-success bar. These geometries "
            "are a different binding mode, so no scoring function can rank them "
            "meaningfully -- any fit number computed from them measures the "
            "pose error, not the chemistry. Constrain conformer generation to "
            "the bound scaffold, or dock."
        )
    return True, (
        f"{n_ok}/{len(usable)} poses within {threshold_angstrom:.1f} A "
        f"(best {best:.2f} A)"
    )


def align_by_index_map(
    symbols: list[str],
    coords_angstrom: list[tuple[float, float, float]],
    reference_coords_angstrom: list[tuple[float, float, float]],
    index_pairs: list[tuple[int, int]],
) -> AlignedPose:
    """Align using an explicit (mobile_index, reference_index) pair list."""
    if len(index_pairs) < 3:
        return AlignedPose(
            symbols, [], 0, float("nan"),
            error=f"only {len(index_pairs)} matched atom pairs; need >= 3",
        )
    mob = np.asarray([coords_angstrom[i] for i, _ in index_pairs], dtype=float)
    ref = np.asarray([reference_coords_angstrom[j] for _, j in index_pairs], dtype=float)
    R, t, rmsd = kabsch(mob, ref)
    moved = np.asarray(coords_angstrom, dtype=float) @ R.T + t
    return AlignedPose(
        symbols=list(symbols),
        coords_angstrom=[tuple(row) for row in moved],
        n_matched_atoms=len(index_pairs),
        rmsd_angstrom=rmsd,
    )


def align_to_reference(
    mobile_smiles: str,
    mobile_symbols: list[str],
    mobile_coords_angstrom: list[tuple[float, float, float]],
    reference_smiles: str,
    reference_symbols: list[str],
    reference_coords_angstrom: list[tuple[float, float, float]],
    timeout_seconds: int = 30,
) -> AlignedPose:
    """Align a conformer onto a bound reference via their maximum common substructure.

    Both molecules are rebuilt from their SMILES and matched to their coordinate
    arrays by element+connectivity, because the reference (from a PDB) and the
    analogue (from RDKit) do not share an atom ordering -- the committed
    danuglipron ensemble mixes three orderings, so an index-wise alignment would
    superimpose a carbon onto a fluorine.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    def _mol_with_coords(smiles, symbols, coords):
        """Build a molecule carrying `coords`, with connectivity perceived from
        the geometry itself rather than imposed from the SMILES order.

        The atom ORDER cannot be taken from the SMILES: the three provenances in
        the committed ensemble use three different orderings (grouped-by-element
        from the PDB, a different grouping from PubChem, RDKit canonical for the
        generated ones). So the molecule is built directly from the xyz block,
        which preserves the geometry's own order, and `smiles` is used only to
        CHECK that the perceived molecule is the intended one.

        Bonds are perceived by RDKit's `DetermineConnectivity` (distance-based),
        which is reliable for a well-formed 3D organic structure and is what
        makes the MCS meaningful without trusting any external atom mapping.
        """
        from rdkit.Chem import rdDetermineBonds

        if len(symbols) != len(coords):
            return None, (
                f"{len(symbols)} symbols but {len(coords)} coordinate rows"
            )
        block = [str(len(symbols)), "from_coords"]
        for sym, (x, y, z) in zip(symbols, coords):
            block.append(f"{sym} {float(x):.8f} {float(y):.8f} {float(z):.8f}")
        mol = Chem.MolFromXYZBlock("\n".join(block) + "\n")
        if mol is None:
            return None, "RDKit could not read the geometry as an xyz block"
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
        except Exception as e:  # noqa: BLE001
            return None, f"bond perception failed ({type(e).__name__}: {e})"

        # Sanity-check the perceived molecule against the declared SMILES by
        # heavy-atom formula. A full canonical-SMILES match is too strict here:
        # DetermineConnectivity assigns no bond orders, so aromaticity and
        # charges are not comparable. The formula check still catches the error
        # that matters -- a geometry that is not the molecule it claims to be.
        declared = Chem.MolFromSmiles(smiles)
        if declared is None:
            return None, f"unparseable SMILES {smiles!r}"
        from collections import Counter

        # HEAVY ATOMS ONLY. A geometry may legitimately carry all hydrogens
        # (RDKit/xyz), only polar ones, or none at all -- AutoDock PDBQT is
        # UNITED-ATOM, merging nonpolar H into their carbons, so a docked
        # danuglipron pose has 41 atoms where the RDKit mol has 70. Demanding a
        # full-formula match rejected every docked pose as "not this molecule",
        # which surfaced as a bogus NO-ALIGNABLE-POSE verdict on a run whose
        # poses were in fact all within 2.8 A of the site.
        #
        # Alignment is heavy-atom anyway (H positions from two different tools
        # are not comparable), so the heavy-atom formula is the right identity
        # check: it still catches a geometry that is the wrong MOLECULE, which
        # is what this guard is for.
        want = Counter(a.GetSymbol() for a in declared.GetAtoms()
                       if a.GetSymbol() != "H")
        got = Counter(a.GetSymbol() for a in mol.GetAtoms()
                      if a.GetSymbol() != "H")
        if want != got:
            return None, (
                f"the geometry's heavy-atom formula {dict(sorted(got.items()))} "
                f"does not match the declared SMILES "
                f"{dict(sorted(want.items()))}"
            )
        return mol, None

    mob_mol, err = _mol_with_coords(mobile_smiles, mobile_symbols, mobile_coords_angstrom)
    if err:
        return AlignedPose(mobile_symbols, [], 0, float("nan"),
                           error=f"mobile molecule: {err}")
    ref_mol, err = _mol_with_coords(
        reference_smiles, reference_symbols, reference_coords_angstrom
    )
    if err:
        return AlignedPose(mobile_symbols, [], 0, float("nan"),
                           error=f"reference molecule: {err}")

    # Heavy atoms only: hydrogens on the reference PDB pose were added by a
    # different tool than the analogue's, so their positions are not comparable
    # and their count can differ. Bond ORDERS are absent (DetermineConnectivity
    # only perceives connectivity), so the comparison must ignore them --
    # `bondCompare=CompareAny` and no ring constraint, since ring perception
    # also depends on bond orders that were never assigned.
    mob_heavy = Chem.RemoveHs(mob_mol, sanitize=False)
    ref_heavy = Chem.RemoveHs(ref_mol, sanitize=False)
    mcs = rdFMCS.FindMCS(
        [mob_heavy, ref_heavy],
        timeout=timeout_seconds,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        matchValences=False,
    )
    if mcs.canceled or mcs.numAtoms < 3:
        return AlignedPose(
            mobile_symbols, [], 0, float("nan"),
            error=(
                f"MCS found only {mcs.numAtoms} common atoms"
                + (" (search timed out)" if mcs.canceled else "")
                + "; too little shared scaffold to define a rigid overlay"
            ),
        )

    patt = Chem.MolFromSmarts(mcs.smartsString)
    if patt is None:
        return AlignedPose(mobile_symbols, [], 0, float("nan"),
                           error=f"MCS produced unusable SMARTS {mcs.smartsString!r}")
    mob_match = mob_heavy.GetSubstructMatch(patt)
    ref_match = ref_heavy.GetSubstructMatch(patt)
    if not mob_match or not ref_match or len(mob_match) != len(ref_match):
        return AlignedPose(
            mobile_symbols, [], 0, float("nan"),
            error="the MCS pattern did not map onto both molecules consistently",
        )

    # MCS indices are into the heavy-atom molecules; the coordinate arrays are
    # indexed by the FULL atom list. Map back via each element's position among
    # the non-hydrogens, which RemoveHs preserves in order.
    def _heavy_to_full(symbols):
        return [i for i, s in enumerate(symbols) if s != "H"]

    mob_map = _heavy_to_full(mobile_symbols)
    ref_map = _heavy_to_full(reference_symbols)
    try:
        pairs = [(mob_map[i], ref_map[j]) for i, j in zip(mob_match, ref_match)]
    except IndexError:
        return AlignedPose(
            mobile_symbols, [], 0, float("nan"),
            error="heavy-atom index map is inconsistent with the MCS match",
        )

    return align_by_index_map(
        mobile_symbols, mobile_coords_angstrom, reference_coords_angstrom, pairs,
    )
