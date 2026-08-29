"""Offline toxicity-liability baseline: published structural alerts + rules.

This is the provider that ALWAYS works. The web services in `web.py` are
better sources when reachable, but they are third-party endpoints that move
(ADMETlab 3.0's documented `/api/admet` was returning 404 on 2026-08-29), and
an experiment whose ranking silently disappears when a website changes is not
an experiment. So the ranking is anchored here.

## What this is, and what it is not

These are *published alert sets*, evaluated by RDKit's `FilterCatalog`:

- **Brenk** (Brenk et al., ChemMedChem 2008) — 105 substructures flagged as
  unsuitable for lead-like libraries: reactive electrophiles, known toxicophores,
  metabolically labile groups.
- **PAINS** (Baell & Holloway, J Med Chem 2010) — frequent-hitter substructures.
- **NIH** — NIH/MLSMR undesirable-functionality filters.
- **ChEMBL curated sets** (Glaxo, Dundee, BMS, Inpharmatica, LINT, SureChEMBL)
  — pharma in-house rejection filters as curated into ChEMBL.

An alert is a *liability flag from the medicinal-chemistry literature*, not a
toxicity prediction. A compound with zero alerts is not predicted safe; it is
merely unflagged by these particular rule sets. That distinction is preserved in
the endpoint names (`alert_*`) and in the `note` fields, so a downstream table
cannot present these as predicted probabilities of harm.

## Why physicochemical rules are here too

Dose-driven GI toxicity — the danuglipron failure mode — is largely an *exposure*
problem, and exposure tracks the classic developability descriptors. Lipinski
(Ro5) and Veber criteria are included on that basis, normalized to a 0-1
"fraction of rules violated" so they aggregate with the alert densities.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import ToxEndpoint

# Catalogs evaluated, in a fixed order so endpoint lists are reproducible.
# Each entry: (endpoint suffix, RDKit FilterCatalogs attribute name, citation).
_CATALOGS: list[tuple[str, str, str]] = [
    ("brenk", "BRENK", "Brenk et al., ChemMedChem 2008 (unsuitable for lead-like libs)"),
    ("pains", "PAINS", "Baell & Holloway, J Med Chem 2010 (frequent hitters)"),
    ("nih", "NIH", "NIH/MLSMR undesirable functionality filters"),
    ("chembl_glaxo", "CHEMBL_Glaxo", "Glaxo hard filters, via ChEMBL"),
    ("chembl_dundee", "CHEMBL_Dundee", "Dundee NTD screening filters, via ChEMBL"),
    ("chembl_bms", "CHEMBL_BMS", "BMS HTS deck filters, via ChEMBL"),
]

# A single alert-set hit count is an integer, but the aggregate score needs a
# 0-1 probability-like scale. Saturating at this many hits maps "clean" to 0.0
# and "several independent flags" to 1.0. The value is a presentation choice,
# not a fitted parameter, and is stated in every endpoint's note so it can
# never be mistaken for a calibrated probability.
_ALERT_SATURATION = 3.0


@dataclass
class _Descriptors:
    mw: float
    clogp: float
    hbd: int
    hba: int
    tpsa: float
    rotb: int


def _descriptors(mol) -> _Descriptors:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    return _Descriptors(
        mw=Descriptors.MolWt(mol),
        clogp=Crippen.MolLogP(mol),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
        tpsa=rdMolDescriptors.CalcTPSA(mol),
        rotb=Lipinski.NumRotatableBonds(mol),
    )


def lipinski_violations(d: _Descriptors) -> list[str]:
    """Rule-of-5 violations (Lipinski et al., Adv Drug Deliv Rev 1997)."""
    v = []
    if d.mw > 500:
        v.append(f"MW {d.mw:.0f} > 500")
    if d.clogp > 5:
        v.append(f"cLogP {d.clogp:.2f} > 5")
    if d.hbd > 5:
        v.append(f"HBD {d.hbd} > 5")
    if d.hba > 10:
        v.append(f"HBA {d.hba} > 10")
    return v


def veber_violations(d: _Descriptors) -> list[str]:
    """Veber criteria for oral bioavailability (Veber et al., J Med Chem 2002)."""
    v = []
    if d.rotb > 10:
        v.append(f"rotatable bonds {d.rotb} > 10")
    if d.tpsa > 140:
        v.append(f"TPSA {d.tpsa:.0f} > 140")
    return v


class RdkitAlertsProvider:
    """Offline provider: RDKit `FilterCatalog` alert sets + Lipinski/Veber.

    Constructed once and reused across a whole ensemble — building the
    FilterCatalog objects parses several hundred SMARTS patterns, which is slow
    enough that per-molecule construction dominates the runtime of a batch.
    """

    name = "rdkit-alerts"

    def __init__(self) -> None:
        from rdkit.Chem import FilterCatalog

        self._catalogs: list[tuple[str, object, str]] = []
        for suffix, attr, citation in _CATALOGS:
            enum_val = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, attr, None)
            if enum_val is None:
                # A future RDKit dropping a catalog must not break the run; the
                # endpoint simply won't be emitted (and its absence is visible).
                continue
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(enum_val)
            self._catalogs.append((suffix, FilterCatalog.FilterCatalog(params), citation))

    def fetch(self, smiles: str) -> list[ToxEndpoint]:
        from rdkit import Chem

        if not isinstance(smiles, str):
            raise TypeError(f"smiles must be str, got {type(smiles).__name__}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # An unparseable SMILES is a caller error, but per the provider
            # contract we do not raise -- an empty list lets a batch continue
            # and the driver records the failure.
            return []

        out: list[ToxEndpoint] = []
        total_hits = 0
        for suffix, catalog, citation in self._catalogs:
            matches = catalog.GetMatches(mol)
            n = len(matches)
            total_hits += n
            names = sorted({m.GetDescription() for m in matches})
            out.append(
                ToxEndpoint(
                    name=f"alert_{suffix}",
                    value=min(n / _ALERT_SATURATION, 1.0),
                    higher_is_worse=True,
                    source=self.name,
                    units="probability",
                    note=(
                        f"{n} substructure hit(s) [{citation}]; scaled n/"
                        f"{_ALERT_SATURATION:.0f} capped at 1.0 -- a rank-only "
                        f"liability density, NOT a probability of toxicity."
                        + (f" Hits: {'; '.join(names[:6])}" if names else "")
                    ),
                )
            )

        out.append(
            ToxEndpoint(
                name="alert_total_count",
                value=float(total_hits),
                higher_is_worse=True,
                source=self.name,
                units="count",
                note="Total structural-alert hits across all evaluated catalogs. "
                "units='count' keeps it out of the probability aggregate.",
            )
        )

        d = _descriptors(mol)
        lip = lipinski_violations(d)
        veb = veber_violations(d)
        out.append(
            ToxEndpoint(
                name="lipinski_violation_fraction",
                value=len(lip) / 4.0,
                higher_is_worse=True,
                source=self.name,
                units="probability",
                note=(
                    f"MW={d.mw:.1f} cLogP={d.clogp:.2f} HBD={d.hbd} HBA={d.hba}; "
                    + (f"violations: {', '.join(lip)}" if lip else "no Ro5 violations")
                ),
            )
        )
        out.append(
            ToxEndpoint(
                name="veber_violation_fraction",
                value=len(veb) / 2.0,
                higher_is_worse=True,
                source=self.name,
                units="probability",
                note=(
                    f"TPSA={d.tpsa:.1f} rotB={d.rotb}; "
                    + (f"violations: {', '.join(veb)}" if veb else "no Veber violations")
                ),
            )
        )
        # Descriptors themselves, on 'raw' units so they are reported but never
        # averaged into the liability score.
        for nm, val, unit in [
            ("mw", d.mw, "Da"), ("clogp", d.clogp, "log10"), ("tpsa", d.tpsa, "A^2"),
            ("rotatable_bonds", float(d.rotb), "count"),
        ]:
            out.append(
                ToxEndpoint(
                    name=f"desc_{nm}", value=val, higher_is_worse=True,
                    source=self.name, units=unit,
                    note="Physicochemical descriptor, reported for context only.",
                )
            )
        return out
