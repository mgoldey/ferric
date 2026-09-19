"""Propose viable substitutions at a drug active site, scored RELATIVELY.

## What the danuglipron prototype established (2026-09-19)

Run against the real parent (PubChem CID 134611040) and the real receptor
(7LCJ, GLP-1R), 9 aromatic CH sites x 6 substituents = 54 analogues:

| stage | result |
|---|---|
| enumerate | 54 analogues |
| pharmacophore gate | 54 kept, **0 rejected** |
| absolute Lipinski/Veber | **0 clean**, 54 flagged |

Both cheap gates were useless, for OPPOSITE reasons, and neither is a bug:

* The pharmacophore gate rejected nothing because substituting an aromatic CH
  genuinely cannot break an acid, a fused diazole, a basic amine or a nitrile.
  Confirmed reachable by feeding it molecules that MUST fail (a methyl ester
  broke `acid_or_bioisostere`; benzene and ethanol broke all four), so 0/54 is
  a true negative. That gate belongs on SCAFFOLD moves -- `bioisostere_swaps`,
  `ring_contractions` -- where those features are actually at risk.
* The absolute liability gate rejected everything because **the parent already
  violates it**: danuglipron is MW 555.6, cLogP 4.89. It is a Phase-2 clinical
  compound. The rule of 5 is a hit-finding filter; applied to an optimized
  molecule it is a constant, not a discriminator.

What discriminated was the CHANGE each substitution makes:

    subst    dMW   dcLogP   verdict
    CN     +25.0    -0.13   the only one that LOWERS lipophilicity
    OMe    +30.0    +0.01   marginal
    F      +18.0    +0.14
    Me     +14.0    +0.31
    Cl     +34.4    +0.65
    CF3    +68.0    +1.02   worst, against a parent already at 4.89

So this module enumerates and scores RELATIVE to the parent. That is the same
argument as reporting ddE rather than an absolute binding energy at the QM
tier: for lead OPTIMIZATION, every gate must be relative, because the absolute
is dominated by the scaffold you are not changing.

## What this module deliberately does NOT do

No docking, no xtb, no QM. Those are tiers 4-7 and live in `tools/docking`,
`tools/campaign` and `tools/active_site`; this produces the POPULATION they
narrow. See `wiki/substitution-pipeline-danuglipron-2026-09-19.md` for the full
stage list and the four blockers (unwritten pose geometry -- since fixed in
PR #93 -- missing dispersion, pose noise, and the missing connector).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SubstitutionProposal", "propose_substitutions", "relative_descriptors"]


@dataclass(frozen=True)
class SubstitutionProposal:
    """One proposed analogue, scored against the parent it came from."""

    smiles: str
    #: The substituent label ("F", "CF3", ...), or "parent" for the reference row.
    label: str
    #: True for the single reference row. Its deltas are exactly zero.
    is_parent: bool
    d_mw: float
    d_clogp: float
    d_tpsa: float

    def __str__(self) -> str:  # pragma: no cover - display only
        tag = "PARENT" if self.is_parent else self.label
        return f"{tag:<8} dMW {self.d_mw:+7.1f}  dcLogP {self.d_clogp:+6.2f}  dTPSA {self.d_tpsa:+6.1f}"


def _mol(smiles: str):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    return m


def relative_descriptors(smiles: str, parent_smiles: str) -> tuple[float, float, float]:
    """`(dMW, dcLogP, dTPSA)` of `smiles` against `parent_smiles`.

    Relative by construction: a molecule against itself is exactly
    `(0.0, 0.0, 0.0)`, not approximately, because the same descriptor call is
    subtracted from itself.
    """
    from tools.tox.alerts import _descriptors

    a = _descriptors(_mol(smiles))
    b = _descriptors(_mol(parent_smiles))
    return (a.mw - b.mw, a.clogp - b.clogp, a.tpsa - b.tpsa)


def propose_substitutions(
    parent_smiles: str,
    substituents: dict[str, str],
    site_smarts: str = "[cH:1]",
    require_smarts: tuple[str, str] | None = None,
) -> list[SubstitutionProposal]:
    """Enumerate single substitutions of `site_smarts` and score them relatively.

    `substituents` maps a label to a replacement fragment, e.g.
    `{"F": "F", "CF3": "C(F)(F)F"}`. An EMPTY dict returns exactly the parent --
    the trivial limit, pinned by
    `test_an_empty_substituent_set_returns_exactly_the_parent`.

    `site_smarts` chooses WHERE. The default `[cH:1]` hits every aromatic CH,
    which on a drug-sized parent is dozens of sites and more than any QM tier
    can afford. **In real use, derive this from the pocket contact map**: site
    choice is the budget control, and it is human judgement, not a default.

    `require_smarts` is an OPTIONAL `(name, pattern)` pharmacophore gate. It is
    off by default because, as the module docstring records, it cannot reject a
    substituent scan -- supply it for scaffold moves, where it can. The parent
    row is never gated out; it is the reference the deltas are measured
    against, and dropping it would leave the caller unable to see what they
    were compared to.

    Returns the parent row FIRST, then substitutions sorted by
    (label, canonical SMILES) -- deterministic, because `RunReactants` does not
    guarantee a stable product order and an unstable population makes every
    downstream tier irreproducible.
    """
    from rdkit import Chem

    parent = _mol(parent_smiles)  # raises ValueError on bad input
    parent_canonical = Chem.MolToSmiles(parent)

    rows = [
        SubstitutionProposal(
            smiles=parent_canonical,
            label="parent",
            is_parent=True,
            d_mw=0.0,
            d_clogp=0.0,
            d_tpsa=0.0,
        )
    ]
    if not substituents:
        return rows

    from tools.isomers.substitutional import substituent_scan

    gate = None
    if require_smarts is not None:
        _, pattern = require_smarts
        gate = Chem.MolFromSmarts(pattern)
        if gate is None:
            raise ValueError(f"could not parse require_smarts pattern: {pattern!r}")

    scored = []
    for iso in substituent_scan(
        parent_canonical, substituents, site_smarts=site_smarts
    ):
        m = Chem.MolFromSmiles(iso.canonical)
        if m is None:
            continue  # substituent_scan already skips these; belt and braces
        if gate is not None and not m.HasSubstructMatch(gate):
            continue
        dmw, dlp, dtp = relative_descriptors(iso.canonical, parent_canonical)
        # iso.transform is the full "[cH:1] -> F" string; the label is its tail.
        label = iso.transform.rsplit("->", 1)[-1].strip() or iso.transform
        scored.append(
            SubstitutionProposal(
                smiles=iso.canonical,
                label=label,
                is_parent=False,
                d_mw=dmw,
                d_clogp=dlp,
                d_tpsa=dtp,
            )
        )
    scored.sort(key=lambda p: (p.label, p.smiles))
    return rows + scored
