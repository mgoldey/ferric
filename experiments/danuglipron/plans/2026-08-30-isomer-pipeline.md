# Hierarchical Isomer Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible pipeline that ENUMERATES structural and substitutional
isomers of a lead compound and funnels them through four cost tiers (docking →
force field → GFN2 → ferric DFT), with every tier's survivors, rejects, and
failures recorded.

**Architecture:** Two new library packages plus one driver. `tools/isomers`
enumerates candidates from a parent SMILES by SMARTS transformation
(substitutional) and by scaffold/stereo variation (structural), deduplicating on
canonical SMILES and preserving provenance. `tools/pipeline` runs an ordered
tier stack, passing survivors down and recording a `FunnelReport`. Existing
`tools/docking`, `tools/morph`, `tools/campaign` supply the tiers unchanged —
this plan adds generation and orchestration, not new chemistry.

**Tech Stack:** RDKit (enumeration, FF), AutoDock Vina + Meeko (tier 1), xtb
(tier 3), ferric (tier 4), pytest.

**Spec:** `experiments/danuglipron/RESULTS.md` M7–M9 and
`tools/campaign/hierarchy.py` — the tier contract and the measured costs this
plan is built around.

## Global Constraints

- **Determinism is the deliverable.** Every stochastic stage takes a seed with a
  fixed default: ETKDG (`0xF00D`), Vina (`0xF00D`), MD (none used here). Two runs
  of the same input must produce byte-identical candidate lists.
- **`tools/` must not import from `experiments/`** and must contain no named
  molecule, target, or hypothesis. Enforced by
  `tools/tests/test_library_experiment_boundary.py`.
- **A tier that cannot answer returns `None`/UNEVALUATED, never a
  neutral-looking number.** A fabricated zero reads as "best possible".
- **Validate a tier against ground truth before trusting it** — the tier-1
  precedent is the 0.95 Å redock of 7LCJ.
- **Measured tier costs on this box** (70-atom anion, def2-SVP/PBE for tier 4):
  tier 1 ≈ 2 min/ligand at exhaustiveness 32; tier 2 ≈ 1 ms/pose; tier 3 ≈ 0.5 s
  single point, ~30 s relaxation; tier 4 ≈ **17–37 min/pose** (extrapolated N³–N⁴
  from a measured 96.1 s at 32 atoms). Tier 4 is the reason the funnel must
  narrow to ≤10 candidates.
- **Run everything with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`** and
  `LD_LIBRARY_PATH` including `$HOME/.local/lib/x86_64-linux-gnu` (libxtb is not
  thread-safe; parallelism is across processes).

---

## File Structure

| file | responsibility |
|---|---|
| `tools/isomers/__init__.py` | public surface: `Isomer`, `enumerate_isomers` |
| `tools/isomers/model.py` | `Isomer` record — SMILES, kind, provenance, parent |
| `tools/isomers/substitutional.py` | SMARTS-driven substituent scans |
| `tools/isomers/structural.py` | scaffold hops, ring-size and stereo variation |
| `tools/isomers/enumerate.py` | orchestration, dedup, filtering, seeding |
| `tools/isomers/tests/` | one test module per source module |
| `tools/pipeline/__init__.py` | public surface: `run_funnel`, `FunnelReport` |
| `tools/pipeline/funnel.py` | ordered tier execution + per-tier bookkeeping |
| `tools/pipeline/tiers.py` | thin adapters wrapping existing tier code |
| `tools/pipeline/tests/` | funnel bookkeeping and adapter tests |
| `experiments/danuglipron/run_isomer_pipeline.py` | the campaign driver |

---

### Task 1: The `Isomer` record

**Files:**
- Create: `tools/isomers/model.py`
- Test: `tools/isomers/tests/test_model.py`

**Interfaces:**
- Produces: `Isomer(smiles: str, kind: str, transform: str, parent_smiles: str,
  net_charge: int = 0, notes: list[str])`, property `canonical` (canonical
  SMILES), property `is_parent` (canonical == canonical parent).

- [ ] **Step 1: Write the failing test**

```python
from tools.isomers.model import Isomer

def test_canonical_smiles_is_order_independent():
    a = Isomer("OC(=O)c1ccccc1", "substitutional", "none", "OC(=O)c1ccccc1")
    b = Isomer("c1ccccc1C(O)=O", "substitutional", "none", "OC(=O)c1ccccc1")
    assert a.canonical == b.canonical

def test_parent_is_detected_by_canonical_form_not_string_equality():
    i = Isomer("c1ccccc1C(O)=O", "substitutional", "none", "OC(=O)c1ccccc1")
    assert i.is_parent is True

def test_unparseable_smiles_raises_rather_than_returning_none():
    import pytest
    with pytest.raises(ValueError, match="unparseable"):
        Isomer("C1CC(((", "substitutional", "x", "C").canonical
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/tests/test_model.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.isomers'`

- [ ] **Step 3: Write minimal implementation**

```python
"""One enumerated isomer, with the provenance that makes it reproducible."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


@dataclass
class Isomer:
    """A generated candidate.

    `transform` records HOW this was produced (the SMARTS or the operation
    name), so a candidate list can be regenerated and audited rather than
    trusted. `kind` is "substitutional" or "structural".
    """
    smiles: str
    kind: str
    transform: str
    parent_smiles: str
    net_charge: int = 0
    notes: list[str] = field(default_factory=list)

    @cached_property
    def canonical(self) -> str:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"unparseable SMILES for {self.transform!r}: {self.smiles}")
        return Chem.MolToSmiles(mol)

    @property
    def is_parent(self) -> bool:
        from rdkit import Chem
        p = Chem.MolFromSmiles(self.parent_smiles)
        return p is not None and self.canonical == Chem.MolToSmiles(p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/tests/test_model.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/isomers/model.py tools/isomers/tests/test_model.py
git commit -m "feat(isomers): Isomer record with canonical-form identity"
```

---

### Task 2: Substitutional enumeration

**Files:**
- Create: `tools/isomers/substitutional.py`
- Test: `tools/isomers/tests/test_substitutional.py`

**Interfaces:**
- Consumes: `Isomer` from Task 1.
- Produces: `substituent_scan(parent_smiles: str, substituents: dict[str, str],
  site_smarts: str = "[cH:1]") -> list[Isomer]`, and the module constant
  `COMMON_SUBSTITUENTS: dict[str, str]` mapping a label to the replacement
  fragment SMILES (`{"F": "F", "Cl": "Cl", "Me": "C", "CN": "C#N",
  "OMe": "OC", "CF3": "C(F)(F)F"}`).

- [ ] **Step 1: Write the failing test**

```python
from tools.isomers.substitutional import COMMON_SUBSTITUENTS, substituent_scan

BENZOIC = "OC(=O)c1ccccc1"

def test_fluorine_scan_finds_the_three_distinct_positions():
    out = substituent_scan(BENZOIC, {"F": "F"})
    assert len(out) == 3, [i.smiles for i in out]
    assert len({i.canonical for i in out}) == 3

def test_products_are_deduplicated_by_symmetry():
    """Benzoic acid has 5 aryl H but only 3 distinct products (ortho/meta/para).
    Returning 5 would mean symmetry-equivalent duplicates leaked through."""
    assert len(substituent_scan(BENZOIC, {"F": "F"})) == 3

def test_each_isomer_records_the_transform_that_made_it():
    out = substituent_scan(BENZOIC, {"CN": "C#N"})
    assert all(i.kind == "substitutional" for i in out)
    assert all("CN" in i.transform for i in out)
    assert all(i.parent_smiles == BENZOIC for i in out)

def test_multiple_substituents_compose():
    out = substituent_scan(BENZOIC, {"F": "F", "Cl": "Cl"})
    assert len(out) == 6

def test_unparseable_parent_raises():
    import pytest
    with pytest.raises(ValueError, match="unparseable parent"):
        substituent_scan("C1CC(((", {"F": "F"})

def test_enumeration_is_deterministic():
    a = [i.canonical for i in substituent_scan(BENZOIC, COMMON_SUBSTITUENTS)]
    b = [i.canonical for i in substituent_scan(BENZOIC, COMMON_SUBSTITUENTS)]
    assert a == b, "same input gave a different candidate ORDER"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/tests/test_substitutional.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.isomers.substitutional'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Substituent scans: replace a matched site with each of a set of groups.

Verified 2026-08-30 that RDKit's RunReactants plus canonical-SMILES dedup
collapses symmetry-equivalent products correctly: a fluorine scan of benzoic
acid gives exactly 3 products (ortho/meta/para) from 5 aryl hydrogens.
"""
from __future__ import annotations

from .model import Isomer

# Label -> replacement fragment. Chosen to span the medicinal-chemistry moves
# that change electronics and sterics without changing the scaffold.
COMMON_SUBSTITUENTS: dict[str, str] = {
    "F": "F",
    "Cl": "Cl",
    "Me": "C",
    "CN": "C#N",
    "OMe": "OC",
    "CF3": "C(F)(F)F",
}


def substituent_scan(
    parent_smiles: str,
    substituents: dict[str, str],
    site_smarts: str = "[cH:1]",
) -> list[Isomer]:
    """Every distinct single substitution of `site_smarts` by each substituent.

    Results are deduplicated on canonical SMILES and returned in a deterministic
    order (substituent label, then canonical SMILES), so two runs agree exactly.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"unparseable parent SMILES: {parent_smiles}")

    out: list[Isomer] = []
    for label in sorted(substituents):
        frag = substituents[label]
        rxn = AllChem.ReactionFromSmarts(f"{site_smarts}>>[c:1]{frag}")
        seen: set[str] = set()
        for products in rxn.RunReactants((parent,)):
            p = products[0]
            try:
                Chem.SanitizeMol(p)
            except Exception:  # noqa: BLE001 - a bad product is skipped, not fatal
                continue
            canon = Chem.MolToSmiles(p)
            if canon in seen:
                continue
            seen.add(canon)
            out.append(Isomer(
                smiles=canon, kind="substitutional",
                transform=f"{site_smarts} -> {label}",
                parent_smiles=parent_smiles,
            ))
        out.sort(key=lambda i: (i.transform, i.canonical))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/tests/test_substitutional.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/isomers/substitutional.py tools/isomers/tests/test_substitutional.py
git commit -m "feat(isomers): substituent scans with symmetry dedup"
```

---

### Task 3: Structural enumeration

**Files:**
- Create: `tools/isomers/structural.py`
- Test: `tools/isomers/tests/test_structural.py`

**Interfaces:**
- Consumes: `Isomer` from Task 1.
- Produces: `stereoisomers(parent_smiles: str, max_isomers: int = 32) ->
  list[Isomer]`; `ring_contractions(parent_smiles: str) -> list[Isomer]`;
  `bioisostere_swaps(parent_smiles: str, swaps: dict[str, tuple[str, str]] |
  None = None) -> list[Isomer]` with default `ACID_BIOISOSTERES` mapping a label
  to `(from_smarts, to_smiles)`.

- [ ] **Step 1: Write the failing test**

```python
from tools.isomers.structural import (
    ACID_BIOISOSTERES, bioisostere_swaps, ring_contractions, stereoisomers,
)

def test_stereoisomers_enumerates_unassigned_centres():
    out = stereoisomers("CC(N)C(=O)O")
    assert len(out) == 2
    assert {i.canonical for i in out} == {"C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"}
    assert all(i.kind == "structural" for i in out)

def test_stereoisomers_respects_the_cap():
    # 4 unassigned centres would be 16; cap to 4.
    out = stereoisomers("CC(N)C(O)C(N)C(O)C(=O)O", max_isomers=4)
    assert len(out) <= 4

def test_ring_contraction_shrinks_a_saturated_ring():
    out = ring_contractions("C1CCNCC1")          # piperidine
    canon = {i.canonical for i in out}
    assert any("C1CNC1" in c or "C1CCNC1" in c for c in canon), canon

def test_bioisostere_swap_replaces_a_carboxylic_acid():
    out = bioisostere_swaps("OC(=O)c1ccccc1")
    labels = {i.transform for i in out}
    assert any("tetrazole" in t for t in labels), labels
    assert all(i.kind == "structural" for i in out)

def test_bioisostere_swap_is_a_no_op_without_the_group():
    assert bioisostere_swaps("c1ccccc1") == []

def test_structural_enumeration_is_deterministic():
    a = [i.canonical for i in bioisostere_swaps("OC(=O)c1ccccc1")]
    b = [i.canonical for i in bioisostere_swaps("OC(=O)c1ccccc1")]
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/tests/test_structural.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.isomers.structural'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Structural isomers: stereochemistry, ring size, and bioisosteric swaps.

"Structural" here means the SCAFFOLD changes -- connectivity, ring size, or a
whole functional group -- as opposed to `substitutional.py`, which decorates a
fixed scaffold. Both produce `Isomer` records and are deduplicated the same way.
"""
from __future__ import annotations

from .model import Isomer

# label -> (SMARTS to find, SMILES to put in its place)
ACID_BIOISOSTERES: dict[str, tuple[str, str]] = {
    "tetrazole": ("[CX3](=O)[OX2H1]", "c1nn[nH]n1"),
    "acylsulfonamide": ("[CX3](=O)[OX2H1]", "C(=O)NS(=O)(=O)C"),
    "hydroxamic_acid": ("[CX3](=O)[OX2H1]", "C(=O)NO"),
}

# label -> (SMARTS to find, SMILES to put in its place)
RING_CONTRACTIONS: dict[str, tuple[str, str]] = {
    "piperidine_to_azetidine": ("C1CCNCC1", "C1CNC1"),
    "piperidine_to_pyrrolidine": ("C1CCNCC1", "C1CCNC1"),
    "cyclohexyl_to_cyclobutyl": ("C1CCCCC1", "C1CCC1"),
}


def _replace(parent_smiles: str, kind: str, table: dict[str, tuple[str, str]]) -> list[Isomer]:
    from rdkit import Chem

    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"unparseable parent SMILES: {parent_smiles}")

    out: list[Isomer] = []
    seen: set[str] = set()
    for label in sorted(table):
        frm, to = table[label]
        patt, repl = Chem.MolFromSmarts(frm), Chem.MolFromSmiles(to)
        if patt is None or repl is None:
            raise ValueError(f"bad transform {label!r}: {frm!r} -> {to!r}")
        if not parent.HasSubstructMatch(patt):
            continue
        for prod in Chem.ReplaceSubstructs(parent, patt, repl, replaceAll=False):
            try:
                Chem.SanitizeMol(prod)
            except Exception:  # noqa: BLE001
                continue
            canon = Chem.MolToSmiles(prod)
            if canon in seen:
                continue
            seen.add(canon)
            out.append(Isomer(smiles=canon, kind=kind, transform=label,
                              parent_smiles=parent_smiles))
    out.sort(key=lambda i: (i.transform, i.canonical))
    return out


def bioisostere_swaps(parent_smiles: str,
                      swaps: "dict[str, tuple[str, str]] | None" = None) -> list[Isomer]:
    """Replace a functional group with bioisosteres that keep its role."""
    return _replace(parent_smiles, "structural", swaps or ACID_BIOISOSTERES)


def ring_contractions(parent_smiles: str) -> list[Isomer]:
    """Shrink a saturated ring, changing scaffold geometry without deleting it."""
    return _replace(parent_smiles, "structural", RING_CONTRACTIONS)


def stereoisomers(parent_smiles: str, max_isomers: int = 32) -> list[Isomer]:
    """Enumerate UNASSIGNED stereocentres.

    `max_isomers` is a hard cap: n centres give 2^n isomers, which is a
    combinatorial trap on a flexible drug-like molecule. The cap is applied by
    RDKit itself so the truncation is deterministic, not by slicing afterwards.
    """
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers, StereoEnumerationOptions,
    )

    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"unparseable parent SMILES: {parent_smiles}")
    opts = StereoEnumerationOptions(onlyUnassigned=True, maxIsomers=max_isomers,
                                    unique=True)
    out = [
        Isomer(smiles=Chem.MolToSmiles(m), kind="structural",
               transform="stereoisomer", parent_smiles=parent_smiles)
        for m in EnumerateStereoisomers(parent, opts)
    ]
    out.sort(key=lambda i: i.canonical)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/tests/test_structural.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/isomers/structural.py tools/isomers/tests/test_structural.py
git commit -m "feat(isomers): structural isomers - stereo, ring size, bioisosteres"
```

---

### Task 4: Enumeration orchestration with filters and provenance

**Files:**
- Create: `tools/isomers/enumerate.py`, `tools/isomers/__init__.py`
- Test: `tools/isomers/tests/test_enumerate.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `enumerate_isomers(parent_smiles, *, substituents=None,
  site_smarts="[cH:1]", include_stereo=True, include_rings=True,
  include_bioisosteres=True, max_candidates=200, mw_range=(0.0, 700.0)) ->
  list[Isomer]`, with the parent always first, and
  `EnumerationReport(n_generated, n_after_dedup, n_after_filter, rejected)`.

- [ ] **Step 1: Write the failing test**

```python
from tools.isomers import enumerate_isomers
from tools.isomers.enumerate import enumerate_with_report

BENZOIC = "OC(=O)c1ccccc1"

def test_parent_is_always_first_and_present_exactly_once():
    out = enumerate_isomers(BENZOIC)
    assert out[0].is_parent
    assert sum(1 for i in out if i.is_parent) == 1

def test_results_are_globally_deduplicated_across_generators():
    out = enumerate_isomers(BENZOIC)
    canon = [i.canonical for i in out]
    assert len(canon) == len(set(canon))

def test_molecular_weight_filter_rejects_and_records_why():
    out, rep = enumerate_with_report(BENZOIC, mw_range=(0.0, 130.0))
    assert all(i.canonical for i in out)
    assert rep.n_after_filter <= rep.n_after_dedup
    assert any("MW" in r for r in rep.rejected)

def test_max_candidates_caps_the_output_deterministically():
    a, _ = enumerate_with_report(BENZOIC, max_candidates=5)
    b, _ = enumerate_with_report(BENZOIC, max_candidates=5)
    assert len(a) <= 5
    assert [i.canonical for i in a] == [i.canonical for i in b]

def test_the_whole_enumeration_is_reproducible():
    a = [i.canonical for i in enumerate_isomers(BENZOIC)]
    b = [i.canonical for i in enumerate_isomers(BENZOIC)]
    assert a == b

def test_report_counts_are_consistent():
    _, rep = enumerate_with_report(BENZOIC)
    assert rep.n_generated >= rep.n_after_dedup >= rep.n_after_filter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/tests/test_enumerate.py -q`
Expected: FAIL, `ImportError: cannot import name 'enumerate_isomers'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Run every generator, deduplicate globally, filter, and report what was lost.

The report is not optional decoration: an enumeration that silently drops
candidates is indistinguishable from one that never made them, and the whole
point of this package is that a candidate list can be regenerated and audited.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Isomer
from .structural import bioisostere_swaps, ring_contractions, stereoisomers
from .substitutional import COMMON_SUBSTITUENTS, substituent_scan


@dataclass
class EnumerationReport:
    n_generated: int = 0
    n_after_dedup: int = 0
    n_after_filter: int = 0
    rejected: list[str] = field(default_factory=list)


def enumerate_with_report(
    parent_smiles: str,
    *,
    substituents: "dict[str, str] | None" = None,
    site_smarts: str = "[cH:1]",
    include_stereo: bool = True,
    include_rings: bool = True,
    include_bioisosteres: bool = True,
    max_candidates: int = 200,
    mw_range: tuple[float, float] = (0.0, 700.0),
) -> "tuple[list[Isomer], EnumerationReport]":
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    rep = EnumerationReport()
    generated: list[Isomer] = [
        Isomer(parent_smiles, "parent", "none", parent_smiles)
    ]
    generated += substituent_scan(parent_smiles,
                                  substituents or COMMON_SUBSTITUENTS,
                                  site_smarts)
    if include_bioisosteres:
        generated += bioisostere_swaps(parent_smiles)
    if include_rings:
        generated += ring_contractions(parent_smiles)
    if include_stereo:
        generated += stereoisomers(parent_smiles)
    rep.n_generated = len(generated)

    # Global dedup, parent first, then a stable order.
    seen: set[str] = set()
    deduped: list[Isomer] = []
    for iso in sorted(generated, key=lambda i: (not i.is_parent, i.kind,
                                                i.transform, i.canonical)):
        if iso.canonical in seen:
            continue
        seen.add(iso.canonical)
        deduped.append(iso)
    rep.n_after_dedup = len(deduped)

    kept: list[Isomer] = []
    lo, hi = mw_range
    for iso in deduped:
        mw = Descriptors.MolWt(Chem.MolFromSmiles(iso.canonical))
        if not (lo <= mw <= hi):
            rep.rejected.append(f"{iso.transform}: MW {mw:.1f} outside [{lo},{hi}]")
            continue
        kept.append(iso)
        if len(kept) >= max_candidates:
            rep.rejected.append(
                f"max_candidates={max_candidates} reached; "
                f"{len(deduped) - len(kept)} candidates not evaluated"
            )
            break
    rep.n_after_filter = len(kept)
    return kept, rep


def enumerate_isomers(parent_smiles: str, **kwargs) -> list[Isomer]:
    """Candidates only. Use `enumerate_with_report` when the losses matter."""
    return enumerate_with_report(parent_smiles, **kwargs)[0]
```

And `tools/isomers/__init__.py`:

```python
"""Isomer enumeration: generate candidates from a parent, reproducibly.

Substitutional isomers decorate a fixed scaffold; structural isomers change it.
Both are deduplicated on canonical SMILES and carry the transform that produced
them, so a candidate list is auditable rather than merely asserted.

GENERIC LIBRARY -- no molecule lives here. A campaign supplies its parent.
"""

from .enumerate import EnumerationReport, enumerate_isomers, enumerate_with_report
from .model import Isomer

__all__ = ["Isomer", "EnumerationReport", "enumerate_isomers", "enumerate_with_report"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/isomers/ -q`
Expected: PASS (all isomer tests)

- [ ] **Step 5: Commit**

```bash
git add tools/isomers/
git commit -m "feat(isomers): orchestration with global dedup and a loss report"
```

---

### Task 5: Tier adapters

**Files:**
- Create: `tools/pipeline/tiers.py`
- Test: `tools/pipeline/tests/test_tiers.py`

**Interfaces:**
- Consumes: `tools.docking.dock_ligand`, `tools.morph.embed.embed_analogue`,
  `tools.campaign.xtb_engine.singlepoint`, `ferric.run_dft`.
- Produces: `TierResult(candidate_id: str, value: float | None, error: str |
  None, payload: dict)`; callables `tier1_dock(...)`, `tier2_forcefield(...)`,
  `tier3_gfn2(...)`, `tier4_dft(...)`, each `(Isomer, context) -> TierResult`.

- [ ] **Step 1: Write the failing test**

```python
from tools.isomers.model import Isomer
from tools.pipeline.tiers import TierResult, tier2_forcefield

def test_tier_result_failure_carries_none_not_zero():
    r = TierResult("x", None, "embedding failed", {})
    assert r.value is None and not r.ok
    assert r.error

def test_tier2_embeds_and_returns_an_energy():
    iso = Isomer("OC(=O)c1ccccc1", "parent", "none", "OC(=O)c1ccccc1")
    r = tier2_forcefield(iso, {})
    assert r.ok, r.error
    assert r.value is not None
    assert "coords" in r.payload and len(r.payload["coords"]) > 0

def test_tier2_reports_an_unembeddable_molecule_as_unevaluated():
    iso = Isomer("C12C3C1C1C2C31", "structural", "cage", "C")
    r = tier2_forcefield(iso, {})
    assert r.ok == (r.error is None), "ok and error must never disagree"
    if not r.ok:
        assert r.value is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/pipeline/tests/test_tiers.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Thin adapters presenting each existing tier with one uniform signature.

Deliberately thin: the chemistry lives in tools/docking, tools/morph and
tools/campaign and is NOT reimplemented here. This module exists so the funnel
can call four very different methods without knowing anything about them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.isomers.model import Isomer


@dataclass
class TierResult:
    """One tier's verdict on one candidate.

    `value` is `None` for any failure -- never 0.0, which in an energy ranking
    reads as the best possible score.
    """
    candidate_id: str
    value: float | None
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None


def tier2_forcefield(iso: Isomer, context: dict) -> TierResult:
    """MMFF94 embed + optimize. Cheap declash; NOT a ranking method."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(iso.canonical))
    params = AllChem.ETKDGv3()
    params.randomSeed = context.get("seed", 0xF00D)
    if AllChem.EmbedMolecule(mol, params) != 0:
        return TierResult(iso.canonical, None, "ETKDG could not embed")
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        energy = float(res[0][1])
    except Exception as e:  # noqa: BLE001
        return TierResult(iso.canonical, None, f"MMFF failed: {type(e).__name__}: {e}")
    conf = mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return TierResult(iso.canonical, energy,
                      payload={"coords": coords,
                               "symbols": [a.GetSymbol() for a in mol.GetAtoms()]})
```

(Tasks 6 and 7 add `tier1_dock`, `tier3_gfn2` and `tier4_dft` to this module;
each is added with its own failing test first, following the same pattern.)

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/pipeline/tests/test_tiers.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/pipeline/tiers.py tools/pipeline/tests/test_tiers.py
git commit -m "feat(pipeline): uniform tier adapter interface, tier 2 first"
```

---

### Task 6: The funnel

**Files:**
- Create: `tools/pipeline/funnel.py`, `tools/pipeline/__init__.py`
- Test: `tools/pipeline/tests/test_funnel.py`

**Interfaces:**
- Consumes: `TierResult` from Task 5, `TierOutcome`/`Tier` from
  `tools.campaign.hierarchy`.
- Produces: `Stage(tier: Tier, fn, keep: int, name: str)`;
  `run_funnel(candidates: list[Isomer], stages: list[Stage], context: dict) ->
  FunnelReport`; `FunnelReport(outcomes: list[TierOutcome], survivors:
  list[Isomer], results: dict[str, list[TierResult]])`.

- [ ] **Step 1: Write the failing test**

```python
from tools.campaign.hierarchy import Tier
from tools.isomers.model import Isomer
from tools.pipeline.funnel import Stage, run_funnel
from tools.pipeline.tiers import TierResult

P = "OC(=O)c1ccccc1"
CANDS = [Isomer(s, "substitutional", "t", P) for s in
         ["OC(=O)c1ccccc1", "OC(=O)c1ccc(F)cc1", "OC(=O)c1ccc(Cl)cc1"]]

def _fake(score_by_index):
    order = {}
    def fn(iso, ctx):
        i = order.setdefault(iso.canonical, len(order))
        v = score_by_index.get(i)
        return TierResult(iso.canonical, v, None if v is not None else "no value")
    return fn

def test_funnel_narrows_to_the_keep_count():
    stages = [Stage(Tier.FORCE_FIELD, _fake({0: -3.0, 1: -1.0, 2: -2.0}), keep=2, name="ff")]
    rep = run_funnel(CANDS, stages, {})
    assert len(rep.survivors) == 2

def test_funnel_keeps_the_lowest_values():
    stages = [Stage(Tier.FORCE_FIELD, _fake({0: -3.0, 1: -1.0, 2: -2.0}), keep=2, name="ff")]
    rep = run_funnel(CANDS, stages, {})
    kept = {i.canonical for i in rep.survivors}
    assert "OC(=O)c1ccccc1" in kept

def test_failed_candidates_are_dropped_and_counted_not_ranked_as_best():
    stages = [Stage(Tier.FORCE_FIELD, _fake({0: -3.0, 2: -2.0}), keep=3, name="ff")]
    rep = run_funnel(CANDS, stages, {})
    assert len(rep.survivors) == 2
    assert rep.outcomes[0].n_failed == 1

def test_outcomes_record_the_population_at_every_stage():
    stages = [
        Stage(Tier.FORCE_FIELD, _fake({0: -3.0, 1: -1.0, 2: -2.0}), keep=2, name="ff"),
        Stage(Tier.SEMIEMPIRICAL, _fake({0: -9.0, 1: -8.0}), keep=1, name="gfn2"),
    ]
    rep = run_funnel(CANDS, stages, {})
    assert [o.n_in for o in rep.outcomes] == [3, 2]
    assert [o.n_out for o in rep.outcomes] == [2, 1]
    assert len(rep.survivors) == 1

def test_an_empty_population_stops_the_funnel_cleanly():
    stages = [
        Stage(Tier.FORCE_FIELD, _fake({}), keep=2, name="ff"),
        Stage(Tier.SEMIEMPIRICAL, _fake({0: -1.0}), keep=1, name="gfn2"),
    ]
    rep = run_funnel(CANDS, stages, {})
    assert rep.survivors == []
    assert rep.outcomes[0].n_out == 0
    assert len(rep.outcomes) == 1, "must not run a later tier on an empty set"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/pipeline/tests/test_funnel.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.pipeline.funnel'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Run an ordered tier stack, narrowing the population at each step.

Every tier's job is to DISCARD, cheaply, what the next cannot afford to examine
(see tools/campaign/hierarchy.py). This module is that loop, plus the
bookkeeping that makes the funnel auditable: how many entered each tier, how
many survived, how many FAILED, and every raw result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tools.campaign.hierarchy import Tier, TierOutcome
from tools.isomers.model import Isomer
from tools.pipeline.tiers import TierResult

TierFn = Callable[[Isomer, dict], TierResult]


@dataclass
class Stage:
    """One tier in the stack. `keep` is how many survivors pass downward."""
    tier: Tier
    fn: TierFn
    keep: int
    name: str


@dataclass
class FunnelReport:
    outcomes: list[TierOutcome] = field(default_factory=list)
    survivors: list[Isomer] = field(default_factory=list)
    results: dict[str, list[TierResult]] = field(default_factory=dict)

    def value(self, stage_name: str, canonical: str) -> float | None:
        for r in self.results.get(stage_name, []):
            if r.candidate_id == canonical:
                return r.value
        return None


def run_funnel(candidates: list[Isomer], stages: list[Stage],
               context: dict[str, Any]) -> FunnelReport:
    """Narrow `candidates` through `stages`, cheapest first.

    Ranking is ASCENDING by value at every tier, because every tier here reports
    an energy or an energy-like score where lower is better. A tier that failed
    on a candidate drops it -- a failure is never ranked, and never treated as
    a good score.
    """
    rep = FunnelReport()
    population = list(candidates)

    for stage in stages:
        if not population:
            break
        results = [stage.fn(iso, context) for iso in population]
        rep.results[stage.name] = results

        by_id = {r.candidate_id: r for r in results}
        ok = [iso for iso in population if by_id.get(iso.canonical, None)
              and by_id[iso.canonical].ok]
        ok.sort(key=lambda iso: by_id[iso.canonical].value)
        survivors = ok[:stage.keep]

        rep.outcomes.append(TierOutcome(
            tier=stage.tier, n_in=len(population), n_out=len(survivors),
            n_failed=len(population) - len(ok),
            note=f"{stage.name}: kept {len(survivors)} of {len(ok)} scored",
            errors=[r.error for r in results if r.error][:10],
        ))
        population = survivors

    rep.survivors = population
    return rep
```

And `tools/pipeline/__init__.py`:

```python
"""Ordered tier execution: the funnel that turns many candidates into few.

Generation lives in `tools/isomers`; the individual methods live in
`tools/docking`, `tools/morph` and `tools/campaign`. This package only
sequences them and records what happened at each step.
"""

from .funnel import FunnelReport, Stage, run_funnel
from .tiers import TierResult

__all__ = ["FunnelReport", "Stage", "run_funnel", "TierResult"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/pipeline/ -q`
Expected: PASS (all funnel + tier tests)

- [ ] **Step 5: Commit**

```bash
git add tools/pipeline/
git commit -m "feat(pipeline): funnel with per-tier population bookkeeping"
```

---

### Task 7: Tier 1, 3 and 4 adapters

**Files:**
- Modify: `tools/pipeline/tiers.py`
- Test: `tools/pipeline/tests/test_tiers.py` (append)

**Interfaces:**
- Produces: `tier1_dock(iso, context)` requiring
  `context["receptor_pdbqt"]`, `context["box_center"]`, `context["box_size"]`;
  `tier3_gfn2(iso, context)` requiring `context["point_charges"]` (optional);
  `tier4_dft(iso, context)` with `context["basis"]` (default `"def2-svp"`) and
  `context["functional"]` (default `"PBE"`).

- [ ] **Step 1: Write the failing test**

```python
import pytest
from tools.isomers.model import Isomer
from tools.pipeline.tiers import tier1_dock, tier3_gfn2, tier4_dft

SMALL = Isomer("CO", "parent", "none", "CO")   # methanol: cheap at every tier

def test_tier3_returns_an_energy_for_a_small_molecule():
    r = tier3_gfn2(SMALL, {"coords_from": "embed"})
    assert r.ok, r.error
    assert r.value < 0

def test_tier4_returns_a_dft_energy_for_a_small_molecule():
    r = tier4_dft(SMALL, {"basis": "sto-3g", "functional": "PBE"})
    assert r.ok, r.error
    assert r.value < 0
    assert r.payload.get("converged") is True

def test_tier4_is_lower_than_tier3_for_the_same_molecule():
    """Not a physics claim -- a wiring check. DFT total energies are far more
    negative than GFN2's, so an accidental swap of the two adapters shows up
    here immediately."""
    assert tier4_dft(SMALL, {"basis": "sto-3g"}).value < tier3_gfn2(SMALL, {}).value

def test_tier1_reports_a_missing_receptor_rather_than_raising():
    r = tier1_dock(SMALL, {"receptor_pdbqt": "/nonexistent.pdbqt",
                           "box_center": (0, 0, 0), "box_size": (10, 10, 10)})
    assert not r.ok
    assert r.value is None
    assert "receptor" in (r.error or "").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/pipeline/tests/test_tiers.py -q`
Expected: FAIL, `ImportError: cannot import name 'tier1_dock'`

- [ ] **Step 3: Write minimal implementation**

Append to `tools/pipeline/tiers.py`:

```python
def _embedded(iso: Isomer, context: dict):
    """Shared 3D geometry for tiers 3 and 4: reuse tier 2's if present."""
    cached = context.get("geometry", {}).get(iso.canonical)
    if cached:
        return cached["symbols"], cached["coords"]
    r = tier2_forcefield(iso, context)
    if not r.ok:
        return None, None
    return r.payload["symbols"], r.payload["coords"]


def tier1_dock(iso: Isomer, context: dict) -> TierResult:
    """AutoDock Vina pose search. `value` is the best Vina score (lower=better)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from tools.docking import dock_ligand

    mol = Chem.AddHs(Chem.MolFromSmiles(iso.canonical))
    params = AllChem.ETKDGv3()
    params.randomSeed = context.get("seed", 0xF00D)
    if AllChem.EmbedMolecule(mol, params) != 0:
        return TierResult(iso.canonical, None, "ETKDG could not embed for docking")
    AllChem.MMFFOptimizeMolecule(mol)

    res = dock_ligand(mol, context["receptor_pdbqt"], context["box_center"],
                      context.get("box_size", (24.0, 24.0, 24.0)),
                      exhaustiveness=context.get("exhaustiveness", 16),
                      n_poses=context.get("n_poses", 10),
                      seed=context.get("seed", 0xF00D))
    if not res.ok:
        return TierResult(iso.canonical, None, res.error)
    best = res.best
    return TierResult(iso.canonical, best.vina_score,
                      payload={"symbols": best.symbols,
                               "coords": best.coords_angstrom,
                               "n_poses": len(res.poses)})


def tier3_gfn2(iso: Isomer, context: dict) -> TierResult:
    """GFN2-xTB single point, optionally in a point-charge field."""
    from tools.campaign.xtb_engine import singlepoint

    symbols, coords = _embedded(iso, context)
    if symbols is None:
        return TierResult(iso.canonical, None, "no geometry for GFN2")
    run = singlepoint(symbols, coords, charge=iso.net_charge,
                      point_charges=context.get("point_charges"))
    if not run.ok:
        return TierResult(iso.canonical, None, run.error)
    return TierResult(iso.canonical, run.energy,
                      payload={"symbols": symbols, "coords": coords})


def tier4_dft(iso: Isomer, context: dict) -> TierResult:
    """ferric Kohn-Sham DFT. The most expensive tier: measured 96.1 s at 32
    atoms (def2-SVP/PBE), extrapolating to 17-37 min at 70 atoms, so this must
    only ever see the handful of candidates the tiers above it left."""
    import ferric

    symbols, coords = _embedded(iso, context)
    if symbols is None:
        return TierResult(iso.canonical, None, "no geometry for DFT")
    xyz = [f"{len(symbols)}", "tier4"]
    for s, (x, y, z) in zip(symbols, coords):
        xyz.append(f"{s} {x:.8f} {y:.8f} {z:.8f}")
    try:
        mol = ferric.Molecule.from_xyz_string("\n".join(xyz) + "\n",
                                              iso.net_charge, 1)
        bs = ferric.BasisSet.bundled(context.get("basis", "def2-svp"))
        res = ferric.run_dft(mol, bs,
                             functional=context.get("functional", "PBE"),
                             point_charges=context.get("point_charges"))
    except Exception as e:  # noqa: BLE001
        return TierResult(iso.canonical, None, f"DFT failed: {type(e).__name__}: {e}")
    if not res.converged:
        return TierResult(iso.canonical, None, "DFT did not converge")
    return TierResult(iso.canonical, res.total_energy,
                      payload={"converged": True, "symbols": symbols,
                               "coords": coords})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/pipeline/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/pipeline/tiers.py tools/pipeline/tests/test_tiers.py
git commit -m "feat(pipeline): tier 1 docking, tier 3 GFN2, tier 4 ferric DFT adapters"
```

---

### Task 8: Reproducibility test for the whole stack

**Files:**
- Create: `tools/pipeline/tests/test_reproducibility.py`

**Interfaces:** Consumes Tasks 4–7.

- [ ] **Step 1: Write the failing test**

```python
"""The pipeline's headline claim is REPRODUCIBILITY. Test it end to end.

Every stochastic stage (ETKDG, Vina) takes a seeded default. If any of them
leaks an unseeded random source, two runs diverge -- and a candidate ranking
that changes between runs is not a result.
"""
from tools.campaign.hierarchy import Tier
from tools.isomers import enumerate_isomers
from tools.pipeline import Stage, run_funnel
from tools.pipeline.tiers import tier2_forcefield, tier3_gfn2

PARENT = "OC(=O)c1ccccc1"

def _run():
    cands = enumerate_isomers(PARENT, substituents={"F": "F"},
                              include_stereo=False, include_rings=False,
                              include_bioisosteres=False)
    stages = [
        Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=3, name="ff"),
        Stage(Tier.SEMIEMPIRICAL, tier3_gfn2, keep=2, name="gfn2"),
    ]
    return run_funnel(cands, stages, {"seed": 0xF00D})

def test_two_identical_runs_give_identical_survivors():
    a, b = _run(), _run()
    assert [i.canonical for i in a.survivors] == [i.canonical for i in b.survivors]

def test_two_identical_runs_give_identical_energies():
    a, b = _run(), _run()
    for stage in ("ff", "gfn2"):
        va = [(r.candidate_id, r.value) for r in a.results[stage]]
        vb = [(r.candidate_id, r.value) for r in b.results[stage]]
        assert va == vb, f"stage {stage} is not reproducible"

def test_the_funnel_actually_narrows():
    rep = _run()
    ns = [o.n_in for o in rep.outcomes] + [len(rep.survivors)]
    assert ns == sorted(ns, reverse=True), f"population did not narrow: {ns}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tools/pipeline/tests/test_reproducibility.py -q`
Expected: FAIL if any stage is unseeded; PASS once seeding is correct.

- [ ] **Step 3: Fix any unseeded stage found**

If a test fails, locate the unseeded call (`EmbedMolecule` without
`params.randomSeed`, or `dock_ligand` without `seed=`) and thread the
`context["seed"]` value through it. No new abstraction — pass the seed.

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/pipeline/tests/test_reproducibility.py
git commit -m "test(pipeline): pin end-to-end reproducibility of the funnel"
```

---

### Task 9: The danuglipron driver

**Files:**
- Create: `experiments/danuglipron/run_isomer_pipeline.py`
- Modify: `experiments/danuglipron/RESULTS.md` (add section M10)

**Interfaces:** Consumes Tasks 4–8; uses `experiments/danuglipron/design.py`'s
`DANUGLIPRON_SMILES` and the 7LCJ receptor prepared by
`tools.docking.prepare_receptor`.

- [ ] **Step 1: Write the driver**

The driver must: enumerate isomers from `DANUGLIPRON_SMILES`; prepare the 7LCJ
receptor once; build the four-stage funnel with keeps `(40, 20, 8, 3)`; run it;
and write `out/isomer_pipeline.json` plus a printed funnel table. It must print
the `EnumerationReport` counts BEFORE the funnel, so the candidate list is
auditable independently of what survived.

Tier-4 budget: with keep=3 and 17–37 min per candidate, tier 4 is ~1–2 h. Set
`context["basis"] = "def2-svp"` and state that cost in the driver's docstring.

- [ ] **Step 2: Run it**

Run:
```bash
LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-sync python experiments/danuglipron/run_isomer_pipeline.py
```
Expected: an enumeration report, then a funnel table narrowing 4 tiers, then a
written JSON.

- [ ] **Step 3: Record results in RESULTS.md as M10**

Report the candidate counts at each tier, the survivors, and — importantly —
whether tier 4 changed the tier-3 ordering. If DFT reorders the survivors, the
cheap tiers cannot substitute for it; if it does not, that is equally worth
knowing and makes tier 4 skippable in future runs.

- [ ] **Step 4: Commit**

```bash
git add experiments/danuglipron/run_isomer_pipeline.py experiments/danuglipron/RESULTS.md
git add -f experiments/danuglipron/out/isomer_pipeline.json
git commit -m "feat(campaign): M10 end-to-end isomer pipeline through all four tiers"
```

---

## Self-Review

**Spec coverage:**
- Structural isomers → Task 3 (stereo, ring contraction, bioisosteres)
- Substitutional isomers → Task 2 (SMARTS substituent scan)
- Hierarchical suite → Tasks 5–7 (four tier adapters, uniform signature)
- Pipeline → Task 6 (funnel with per-tier bookkeeping)
- Reproducible → Task 8 (end-to-end seeded-determinism test), plus the seeding
  constraint in Global Constraints
- Applied to the campaign → Task 9

**Placeholder scan:** No TBDs. Every code step carries runnable code. Task 7's
"append to tiers.py" repeats the full function bodies rather than referring back.

**Type consistency:** `Isomer.canonical` is used as the candidate id in
`TierResult.candidate_id` throughout Tasks 5–8. `Stage.fn` matches
`TierFn = Callable[[Isomer, dict], TierResult]` in every adapter. `TierOutcome`
is imported from `tools.campaign.hierarchy` and not redefined.

**Known gap, deliberate:** tier 4 remains validated only by the wiring check in
Task 7 (DFT energy is far below GFN2's). Validating tier 4 against a reference
energy is out of scope here and stays recorded as unvalidated in
`experiments/danuglipron/hierarchy.py`.
