"""Arm A: conformer strain — what the bound pose costs relative to free solution.

## The quantity and why it is a toxicity lever

E_strain(pose) = E(pose geometry, relaxed in the pocket field)
               - E(global free minimum, relaxed in vacuum)

A ligand that must adopt a high-energy conformation to bind pays that strain out
of its binding free energy. The potency you observe is the intrinsic
complementarity MINUS the strain penalty, so a strained binder needs more
*concentration* for the same receptor occupancy — i.e. a higher dose, i.e. more
systemic exposure. For danuglipron the dose-limiting toxicity was
dose-dependent GI intolerability, so **strain is a toxicity lever that does not
require changing the molecule at all**: a modification that preserves the
contacts but relieves strain lowers the efficacious dose.

## Reference-state discipline

Strain is a DIFFERENCE, so its value is entirely determined by what you subtract.
Two traps, both of which produce a plausible-looking number:

1.  **Comparing different relaxation levels.** A bound pose relaxed in-field
    against a free minimum that was only MMFF-optimized mixes two energy
    functions; the "strain" is then mostly the GFN2-vs-MMFF offset. Everything
    here is GFN2 on both sides, and `StrainResult.method` records it.
2.  **Using a local free minimum as "the" free minimum.** The reference must be
    the LOWEST free-solution conformer found, not whichever one happened to be
    conformer 0. `free_reference` scans the whole ensemble and reports how many
    conformers it considered, so a single-conformer "reference" is visible as
    such.

Strain is reported in kcal/mol because that is the scale of the effect (a few
kcal/mol matters) and Hartree hides it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .xtb_engine import HARTREE_TO_KCAL_MOL, XtbRun, relax, singlepoint


@dataclass
class ConformerEnergy:
    """One conformer's vacuum energetics."""
    label: str
    e_singlepoint: float | None
    e_relaxed: float | None
    converged: bool
    error: str | None = None
    relaxed_coords: list[tuple[float, float, float]] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.e_relaxed is not None


@dataclass
class FreeReference:
    """The free-solution reference state for a strain calculation.

    `n_considered` is reported so a reader can tell a genuine global scan from a
    one-conformer stand-in. `e_min` is `None` if nothing converged, which makes
    every strain built on it `None` rather than silently zero.
    """
    e_min: float | None
    label: str | None
    n_considered: int
    n_converged: int
    per_conformer: list[ConformerEnergy] = field(default_factory=list)
    method: str = "GFN2-xTB"

    @property
    def spread_kcal(self) -> float | None:
        """Energy spread across converged conformers, kcal/mol.

        A near-zero spread across a 9-rotatable-bond molecule is a red flag
        (per CLAUDE.md: too clean is a stop condition) -- it usually means every
        "conformer" collapsed to the same minimum, not that the molecule is
        rigid.
        """
        es = [c.e_relaxed for c in self.per_conformer if c.ok]
        if len(es) < 2:
            return None
        return (max(es) - min(es)) * HARTREE_TO_KCAL_MOL


@dataclass
class StrainResult:
    """Strain penalty for one pose against a free reference."""
    label: str
    e_pose_in_field: float | None
    e_pose_relaxed_in_field: float | None
    strain_kcal: float | None
    reference_label: str | None
    method: str = "GFN2-xTB"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.strain_kcal is not None


def free_reference(
    symbols: "list[str] | list[list[str]]",
    conformers: list[list[tuple[float, float, float]]],
    labels: list[str] | None = None,
    charge: int = 0,
    max_conformers: int | None = None,
) -> FreeReference:
    """Relax every conformer in vacuum and return the lowest as the reference.

    This is the expensive-but-necessary half: the reference must be a real
    minimum of the same Hamiltonian used for the bound pose.

    `symbols` accepts either one shared list or a per-conformer list of lists.
    The per-conformer form exists because a real ensemble can mix atom orderings
    (the committed danuglipron set mixes three); a total energy is invariant to
    that, so such an ensemble is perfectly usable HERE even though it is unsafe
    for per-atom work. See `XyzEnsemble.shared_order`.
    """
    labels = labels or [f"conf_{i:02d}" for i in range(len(conformers))]
    per_conf_symbols: list[list[str]]
    if symbols and isinstance(symbols[0], list):
        per_conf_symbols = symbols  # type: ignore[assignment]
        if len(per_conf_symbols) != len(conformers):
            raise ValueError(
                f"got {len(per_conf_symbols)} symbol lists for "
                f"{len(conformers)} conformers -- must match 1:1"
            )
    else:
        per_conf_symbols = [symbols] * len(conformers)  # type: ignore[list-item]

    if max_conformers is not None:
        conformers = conformers[:max_conformers]
        labels = labels[:max_conformers]
        per_conf_symbols = per_conf_symbols[:max_conformers]

    per: list[ConformerEnergy] = []
    for label, coords, syms in zip(labels, conformers, per_conf_symbols):
        sp = singlepoint(syms, coords, charge=charge)
        rx = relax(syms, coords, charge=charge)
        per.append(
            ConformerEnergy(
                label=label,
                e_singlepoint=sp.energy,
                e_relaxed=rx.energy,
                converged=rx.converged,
                error=rx.error or sp.error,
                relaxed_coords=rx.coords_angstrom,
            )
        )

    good = [c for c in per if c.ok]
    if not good:
        return FreeReference(None, None, len(per), 0, per)
    best = min(good, key=lambda c: c.e_relaxed)
    return FreeReference(best.e_relaxed, best.label, len(per), len(good), per)


def pose_strain(
    symbols: list[str],
    pose_coords: list[tuple[float, float, float]],
    reference: FreeReference,
    point_charges: list[tuple[float, float, float, float]] | None = None,
    label: str = "pose",
    charge: int = 0,
) -> StrainResult:
    """Strain of one pose, optionally relaxed inside a pocket point-charge field.

    NOTE on what is and is not comparable: `e_pose_*` are in-field energies when
    `point_charges` is given, while `reference.e_min` is a vacuum energy. Their
    difference therefore contains the ligand-field interaction as well as the
    conformational strain, and is NOT a pure strain number. For a pure
    conformational strain, pass `point_charges=None` -- then both sides are
    vacuum GFN2 and the difference is the deformation energy alone. Both are
    useful; conflating them is the error. `strain_kcal` is the vacuum-consistent
    quantity: it is computed from a VACUUM single point at the (possibly
    in-field-relaxed) pose geometry, so the field enters only through the
    geometry, never through the energy.
    """
    if reference.e_min is None:
        return StrainResult(
            label, None, None, None, None,
            error="free reference has no converged conformer; strain undefined",
        )

    in_field = relax(symbols, pose_coords, charge=charge, point_charges=point_charges)
    if not in_field.ok:
        return StrainResult(
            label, None, None, None, reference.label,
            error=f"pose relaxation failed: {in_field.error}",
        )

    sp_in_field = singlepoint(
        symbols, pose_coords, charge=charge, point_charges=point_charges
    )

    # The strain number itself: a VACUUM energy at the relaxed pose geometry,
    # minus the vacuum global minimum. Both sides are vacuum GFN2, so the
    # difference is purely the cost of holding this geometry.
    geom = in_field.coords_angstrom or pose_coords
    vac_at_pose = singlepoint(symbols, geom, charge=charge)
    if not vac_at_pose.ok:
        return StrainResult(
            label, sp_in_field.energy, in_field.energy, None, reference.label,
            error=f"vacuum single point at the relaxed pose failed: {vac_at_pose.error}",
        )

    strain = (vac_at_pose.energy - reference.e_min) * HARTREE_TO_KCAL_MOL
    return StrainResult(
        label=label,
        e_pose_in_field=sp_in_field.energy,
        e_pose_relaxed_in_field=in_field.energy,
        strain_kcal=strain,
        reference_label=reference.label,
    )


@dataclass
class XyzEnsemble:
    """A directory of xyz conformers, with PER-CONFORMER symbol lists.

    Symbols are per-conformer, not shared, because the committed danuglipron
    ensemble genuinely mixes three atom orderings (measured 2026-08-29):
    `conf_00_cryo_em` is grouped-by-element from the PDB, `conf_01_pubchem` is a
    different element grouping, and the `conf_*_rdkit` members follow RDKit's
    canonical order. All 71 atoms and the C31H30FN5O4 formula agree.

    That is SAFE for a total energy -- GFN2, like any Hamiltonian, is invariant
    to atom labelling -- and UNSAFE for anything per-atom: comparing atom i of
    the cryo-EM pose to atom i of a PubChem pose pairs a carbon with a fluorine.
    So `shared_order` records whether a single ordering holds across the whole
    ensemble, and consumers that need per-atom correspondence (per-atom charges,
    RMSD, prescreen charge tables) must check it and re-map if it is False.
    """
    symbols_per_conformer: list[list[str]]
    conformers: list[list[tuple[float, float, float]]]
    labels: list[str]
    shared_order: bool
    formula: str

    @property
    def symbols(self) -> list[str]:
        """The first conformer's symbols.

        Only meaningful for whole-molecule quantities. Raises if the ensemble
        does not share one order, so a caller cannot accidentally use it as
        though it applied to every member.
        """
        if not self.shared_order:
            raise ValueError(
                "this ensemble mixes atom orderings "
                f"({self.formula}); `.symbols` is ambiguous. Use "
                "`symbols_per_conformer[i]` alongside `conformers[i]`."
            )
        return self.symbols_per_conformer[0]

    def __len__(self) -> int:
        return len(self.conformers)


def _formula(symbols: list[str]) -> str:
    from collections import Counter

    c = Counter(symbols)
    return "".join(f"{el}{c[el]}" for el in sorted(c))


def load_xyz_ensemble(directory: str | Path, pattern: str = "conf_*.xyz") -> XyzEnsemble:
    """Read a directory of xyz conformers.

    Raises if the conformers are not the same MOLECULE (differing formula or
    atom count) -- energies across different molecules are not comparable, and
    that failure would otherwise surface as an inexplicable strain spread. A
    differing atom ORDER is recorded (`shared_order=False`), not rejected: it
    does not affect a total energy, and the committed danuglipron ensemble
    really does mix orderings across its three provenances.
    """
    from tools.active_site.ligand_embedding import _read_xyz_atoms

    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")

    symbols_per: list[list[str]] = []
    conformers, labels = [], []
    reference_formula: str | None = None
    for p in paths:
        atoms = _read_xyz_atoms(p)
        syms = [a.symbol for a in atoms]
        f = _formula(syms)
        if reference_formula is None:
            reference_formula = f
        elif f != reference_formula:
            raise ValueError(
                f"{p.name} is a different molecule than {paths[0].name}: "
                f"formula {f} vs {reference_formula}. Energies across these "
                "geometries are not comparable."
            )
        symbols_per.append(syms)
        conformers.append([(a.x, a.y, a.z) for a in atoms])
        labels.append(p.stem)

    shared = all(s == symbols_per[0] for s in symbols_per)
    return XyzEnsemble(
        symbols_per_conformer=symbols_per,
        conformers=conformers,
        labels=labels,
        shared_order=shared,
        formula=reference_formula or "",
    )
