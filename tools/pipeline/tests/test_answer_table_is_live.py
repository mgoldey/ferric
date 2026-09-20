"""Every call named in the golden path's answer table must still exist.

## Why this guard

Section 0a is the first thing a reader sees and the only place the six pharma
use cases are answered in one view. Its whole value is that each row was
EXECUTED rather than asserted -- which makes it exactly the kind of table that
rots silently when a function is renamed, because prose does not fail to
compile.

## What it does and does not check

It checks the named callables are IMPORTABLE and CALLABLE with the documented
signature shape. It does NOT re-run the timings: a DFT call is ~1 s and docking
is ~26 s/ligand, which does not belong in the fast tier. The costs carry their
own date and measurement note in the table.

A renamed function fails here. A function whose COST changed does not, and that
is deliberate -- a cost is a measurement with a date, not an invariant.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GOLDEN = REPO / "site/src/reference/pipeline-golden-path.md"

#: Every `module.callable` the answer table names, mapped to the module that
#: must export it. Kept explicit rather than parsed out of the markdown: a
#: regex over prose would silently match nothing and pass, which is the
#: failure mode this guard exists to prevent.
TABLE_CALLS = {
    "tools.pipeline.substitution": ["propose_substitutions"],
    "tools.tox.alerts": ["RdkitAlertsProvider"],
    "tools.tox.assess": ["assess_smiles"],
    "tools.pipeline.tiers": ["tier2_forcefield", "tier3_gfn2", "tier4_dft"],
    "tools.viz.energy_plots": [
        "site_substituent_heatmap",
        "liability_profile",
        "pose_ensemble",
        "funnel_survival",
        "tier_comparison",
        "energy_profile",
        "imaginary_mode",
        "reaction_path",
        "optimization_trace",
        "qmmm_partition",
    ],
    "tools.viz.molecules": ["depict"],
    # The two the table names by MODULE shorthand. Without them, renaming
    # either callable leaves the answer table stale while this guard passes --
    # exactly the drift it exists to catch.
    "tools.docking.vina_dock": ["dock_ligand"],
    "tools.active_site.binding_energy": ["compute_binding_energy"],
}

#: `ferric.*` entry points the table names. The extension may be absent in a
#: fast-tier environment, so these are skipped rather than failed there.
FERRIC_CALLS = [
    "run_saddle",
    "run_frequencies",
    "run_irc",
    "run_optimize",
    "QmmmSystem",
]


@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_the_answer_table_exists_and_names_its_costs():
    text = GOLDEN.read_text()
    assert "## 0a. THE ANSWER TABLE" in text, (
        "the answer table is gone; it is the only place the six pharma use "
        "cases are answered in one view"
    )
    block = text[text.index("## 0a.") : text.index("\n## ", text.index("## 0a.") + 5)]
    # It must say the rows were executed, not estimated -- that claim is the
    # reason the table is trustworthy and this guard is worth having.
    assert "EXECUTED" in block
    # And it must carry the two caveats, which are the rows most likely to be
    # misread as licensing something they do not.
    assert "not** licensed" in block or "not licensed" in block, (
        "the binding-energy row lost its 'ranking is not licensed' caveat"
    )
    assert "gate, not a ranker" in block, "the xtb row lost its gate-not-ranker caveat"


@pytest.mark.parametrize("module,names", sorted(TABLE_CALLS.items()))
def test_every_named_call_is_importable_and_callable(module, names):
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # Skip ONLY for a genuinely absent optional dependency. A blanket
        # ImportError catch turns a removed symbol or a broken internal import
        # into a SKIP, which reads as "not applicable here" rather than as the
        # breakage it is -- a guard that skips itself green.
        if exc.name not in {"rdkit", "matplotlib", "vina", "meeko"}:
            raise
        pytest.skip(f"{module} needs the optional dependency {exc.name!r}")
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, (
        f"the answer table names {missing} in {module}, and they are gone. "
        "Either restore them or update section 0a -- a table naming a function "
        "that does not exist is worse than no table."
    )
    for n in names:
        assert callable(getattr(mod, n)), f"{module}.{n} is no longer callable"


def test_the_tier_calls_still_take_an_isomer_and_a_context():
    """The table shows `tiers.tierN(...)`; the shape is part of the claim."""
    tiers = pytest.importorskip("tools.pipeline.tiers")
    for name in ("tier2_forcefield", "tier3_gfn2", "tier4_dft"):
        params = list(inspect.signature(getattr(tiers, name)).parameters)
        assert params[:2] == ["iso", "context"], (
            f"{name} now takes {params[:2]}; the answer table's example call "
            "no longer runs as written"
        )


def test_the_ferric_entry_points_the_table_names_exist():
    ferric = pytest.importorskip("ferric", reason="compiled extension not built")
    missing = [n for n in FERRIC_CALLS if not hasattr(ferric, n)]
    assert not missing, f"the answer table names ferric.{missing}, now absent"


def test_the_measured_costs_carry_a_date():
    """A cost with no date is unfalsifiable; the table must say when it was run."""
    text = GOLDEN.read_text()
    block = text[text.index("## 0a.") : text.index("\n## ", text.index("## 0a.") + 5)]
    assert re.search(r"20\d\d-\d\d-\d\d", block), (
        "section 0a quotes costs with no date, so nobody can tell whether they "
        "describe the current code"
    )


def test_embedded_proposals_are_origin_centred_not_pocket_placed():
    """The silent failure between `embed_proposals` and an embedded SCF.

    `embed_proposals` returns ETKDG conformers centred on the ORIGIN. A pocket
    derived from a PDB sits at its crystal coordinates. MEASURED on 7LCJ: 226 A
    apart. Feeding one to `run_rhf(point_charges=...)` is not an error -- it is
    a confident ~0.00 kcal/mol, a gas-phase answer wearing a QM/MM label.

    This pins the PRECONDITION, not a defect: origin-centring is correct for an
    embedder that knows nothing about a receptor. The guard exists so that if
    `embed_proposals` ever starts placing structures, whoever changes it sees
    that the golden path's placement warning needs updating too.
    """
    pytest.importorskip("rdkit")
    import statistics

    from tools.pipeline.substitution import embed_proposals, propose_substitutions

    props = propose_substitutions("CC(=O)Oc1ccccc1C(=O)O", {"F": "F"})
    good = [e for e in embed_proposals(props[:2]) if e.error is None]
    assert good, "no proposal embedded, so this guard checked nothing"

    for e in good:
        centroid = [statistics.fmean(c[k] for c in e.coords) for k in range(3)]
        assert all(abs(v) < 1.0 for v in centroid), (
            f"embedded proposal centroid is {centroid}, not the origin. If "
            "embed_proposals now places structures, the golden path's "
            "'the analogue must be in the pocket' warning needs revisiting."
        )


@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_the_placement_warning_is_present():
    """A silent wrong answer needs the warning to survive edits."""
    text = GOLDEN.read_text()
    assert "226 A apart" in text or "226 A" in text, (
        "the measured origin-vs-pocket separation is gone from the note"
    )
    assert "gas-phase answer wearing a QM/MM label" in text


@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_every_plot_the_table_cites_is_one_the_guard_checks():
    """The two lists are kept in sync BY HAND, so make that enforced.

    Section 0a's last column names the figure that answers each use case. If a
    row cites a plot the guard does not know about, that plot can be renamed
    and the table goes stale while every test still passes -- the exact drift
    the guard exists to catch, reintroduced one column over.
    """
    text = GOLDEN.read_text()
    block = text[text.index("## 0a.") : text.index("\n## ", text.index("## 0a.") + 5)]
    # Only the plot column: names in backticks that look like viz functions.
    cited = {
        m
        for m in re.findall(r"`([a-z_][a-z_0-9]*)`", block)
        if m.endswith(
            (
                "_profile",
                "_survival",
                "_comparison",
                "_heatmap",
                "_ensemble",
                "_path",
                "_mode",
                "_trace",
                "_partition",
            )
        )
    }
    assert cited, "no plot names found in section 0a; has the column changed shape?"
    guarded = set(TABLE_CALLS.get("tools.viz.energy_plots", []))
    missing = sorted(cited - guarded)
    assert not missing, (
        f"section 0a cites {missing}, which TABLE_CALLS does not check. Add "
        "them, or a rename leaves the table stale with every test green."
    )


@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_the_quickstart_binds_every_name_it_uses():
    """Section 0b promises "code you can paste". It must at least COMPILE.

    It did not: section E referenced an undefined `candidates` (the funnel
    takes `Isomer`s and section B produces `SubstitutionProposal`s, with no
    conversion between them) and then an undefined `parent`. Both are runtime
    NameErrors, so the block ran four sections and died on the fifth.

    This compiles the block and checks every name it reads is either bound
    earlier in the block or imported there -- which catches the omission
    without needing rdkit, ferric or a 30-second DFT run in the fast tier.
    """
    import ast

    text = GOLDEN.read_text()
    block = text[text.index("## 0b.") : text.index("\n## ", text.index("## 0b.") + 5)]
    code = "\n".join(
        line
        for chunk in re.findall(r"```python\n(.*?)```", block, re.S)
        for line in chunk.splitlines()
    )
    assert code.strip(), "no python block found in section 0b"

    tree = ast.parse(code)  # a syntax error fails here, which is the point

    bound: set[str] = set(dir(__builtins__)) | {"__name__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)

    used = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    missing = sorted(used - bound - set(dir(__import__("builtins"))))
    assert not missing, (
        f"the quickstart reads {missing} without binding them. It advertises "
        "code you can paste, so a NameError in it is a broken promise, not a "
        "typo."
    )


def test_substitution_labels_are_NOT_unique_so_they_cannot_key_a_heatmap():
    """The natural key for a per-site figure silently collapses candidates.

    `SubstitutionProposal.label` is the substituent NAME. Aspirin with {F, Cl}
    gives 9 proposals and 9 distinct SMILES but only 3 distinct labels, because
    each halogen has four ring positions. A `{(label, site): ddE}` dict built
    with a constant site keeps 3 of 9 -- in a figure whose premise is that
    WHERE a group goes matters as much as which group.

    Pinned as a PROPERTY of the enumerator, not a defect: labels are for
    display. If they ever become unique, the heatmap's docstring warning should
    be revisited rather than left claiming a hazard that no longer exists.
    """
    pytest.importorskip("rdkit")
    from tools.pipeline.substitution import propose_substitutions

    props = propose_substitutions("CC(=O)Oc1ccccc1C(=O)O", {"F": "F", "Cl": "Cl"})
    labels = [p.label for p in props]
    smiles = [p.smiles for p in props]

    assert len(set(smiles)) == len(props), "the enumerator produced duplicates"
    assert len(set(labels)) < len(props), (
        f"labels are now unique ({len(set(labels))} of {len(props)}). The "
        "heatmap docstring warns that they are not -- update it."
    )
    # And the collapse is large, not a one-off tie.
    assert len(set(labels)) * 2 <= len(props), (
        f"{len(set(labels))} labels for {len(props)} proposals is a milder "
        "collision than the docstring describes; re-measure before trusting it"
    )


def test_the_two_geometry_paths_differ_and_the_note_says_which_is_placed():
    """Only one of the two ways to get coordinates is in the receptor frame.

    `DockedPose.coords_angstrom` is placed (its own docstring says "in the
    receptor's coordinate frame"). `embed_proposals` returns origin-centred
    ETKDG conformers -- MEASURED 226 A from the 7LCJ pocket. The two look
    interchangeable from a call site, and feeding the wrong one to an embedded
    SCF is a silent ~0.00 kcal/mol rather than an error.

    Asserted against the SOURCE rather than by docking (which needs vina and a
    receptor), so it stays in the fast tier.
    """
    from pathlib import Path

    vina_src = (
        Path(__file__).resolve().parents[2] / "docking" / "vina_dock.py"
    ).read_text()
    assert "receptor's coordinate frame" in vina_src, (
        "DockedPose no longer documents its frame; the golden path's "
        "two-paths table depends on that claim"
    )
    assert "coords_angstrom" in vina_src

    text = GOLDEN.read_text()
    assert "TWO PATHS TO A GEOMETRY" in text, (
        "the note lost the distinction between the docked and embedded paths"
    )
    assert "_harvest_geometry" in text, (
        "the note no longer names the function that carries a docked pose "
        "into the later tiers"
    )
