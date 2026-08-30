"""Designed structural analogues of a lead compound, and the pharmacophore
constraint they must respect.

GENERIC LIBRARY. The types here describe *how* to express a set of designed
analogues; they contain no molecule. A specific campaign's hypothesis set lives
with that campaign, in `experiments/<name>/`, and is passed in.

## The shape of an analogue set

Each `Analogue` pairs a structure with the liability hypothesis it tests and the
`PharmacophoreSpec` it must not break. A set should also contain **negative
controls** — variants that break exactly one pharmacophore feature on purpose —
because without them a fit or liability metric cannot be shown to discriminate
anything. See `Analogue.is_negative_control`.

## Ionization state is part of the design, not a detail

`Analogue.smiles_ionized`/`net_charge` carry the species that exists at
physiological pH. This is load-bearing: scoring a neutral acid when the bound
species is an anion omitted ~143 kcal/mol of the interaction under study in one
campaign (see `experiments/danuglipron/RESULTS.md`), and made a
cannot-ionize control look equivalent to its parent.

## Honesty boundary

An analogue is a *hypothesis expressed as a structure*. Nothing here is a claim
about what happens in an organism.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PharmacophoreSpec:
    """SMARTS features an analogue must retain to be a fit candidate.

    Each entry is `(feature_name, smarts, min_count)`. The check is deliberately
    coarse — presence of a substructure class, not a geometric overlay — because
    a SMARTS test is the strongest claim a 2D structure can support. Geometric
    fit is measured later, in the pocket, by the QM pipeline; conflating the two
    would let a 2D pattern match stand in for an actual pose.
    """
    features: tuple[tuple[str, str, int], ...]

    def check(self, mol) -> dict[str, bool]:
        from rdkit import Chem

        out: dict[str, bool] = {}
        for name, smarts, min_count in self.features:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(f"pharmacophore feature {name!r} has invalid SMARTS {smarts!r}")
            out[name] = len(mol.GetSubstructMatches(patt)) >= min_count
        return out




@dataclass
class Analogue:
    """One designed structural variant.

    `is_negative_control=True` marks a variant designed to BREAK a
    pharmacophore feature on purpose. These are kept in the set so the fit
    metric can be shown to discriminate: a control that deletes the potency
    anchor and still scores as well as the parent means the metric is not
    measuring complementarity. Per CLAUDE.md's protocol, this is the stated
    artifact hypothesis made executable — and it earned its place, catching
    three independent errors in one campaign (see
    experiments/danuglipron/RESULTS.md).

    On what a failure means: the original guess was "measuring size or noise".
    Measured 2026-08-29, SIZE is ruled out (r(MW, fit) = +0.132). NOISE was real
    but fixable (SEM 6.5 kcal/mol at n=40). What remains is pose determination.

    `smiles_ionized` / `net_charge` carry the state the molecule is actually IN
    at physiological pH, and they are the ones that must be scored.

    WHY THIS FIELD EXISTS (measured 2026-08-29, and it invalidated a whole run):
    a carboxylic acid with pKa ~4 is **anionic** at pH 7.4, and that anion is
    the salt bridge that anchors it. Scoring the NEUTRAL acid in a protein
    point-charge field gave −22.86 kcal/mol; the ANION at the SAME geometry gave
    **−165.81** kcal/mol. Modelling everything as neutral therefore omitted ~143
    kcal/mol of exactly the interaction under study, and made a cannot-ionize
    control look equivalent to its parent. An ionization state is not a detail;
    it is the measurement.
    """
    label: str
    smiles: str
    hypothesis: str
    rationale: str
    # Required, with no default: the library ships no pharmacophore of its own,
    # and silently defaulting to a permissive one would let an analogue set
    # claim a constraint it never checked.
    pharmacophore: PharmacophoreSpec
    is_negative_control: bool = False
    # SMILES of the dominant species at pH 7.4, and its net charge. Defaults to
    # the neutral form (charge 0) for a molecule with no ionizable group -- e.g.
    # the methyl-ester control, whose entire point is that it CANNOT ionize.
    smiles_ionized: str | None = None
    net_charge: int = 0

    @property
    def scoring_smiles(self) -> str:
        """The species to score: the ionized form when one is declared."""
        return self.smiles_ionized or self.smiles

    def check_pharmacophore(self) -> dict[str, bool]:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"analogue {self.label!r} has unparseable SMILES: {self.smiles}")
        return self.pharmacophore.check(mol)

    @property
    def retains_pharmacophore(self) -> bool:
        return all(self.check_pharmacophore().values())
