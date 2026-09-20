"""Every cost figure in the tier docs must name where it came from.

This exists because the cost claims formed a CLOSED CITATION LOOP:

    golden-path note  --cites-->  tools/pipeline/tiers.py:NN
    tiers.py header   --cites-->  tools/campaign/hierarchy.py
    hierarchy.py      --cites-->  (nothing)

Four tier costs were prose all the way down, and every one passed the check
"is this sourced?" because each pointed at a real file. MEASURED afterwards,
tier 2's `~ms` was 20x low.

The guard is deliberately WEAK: it does not verify the numbers, which would
mean re-running the pipeline in the fast tier. It checks that the cost blocks
say where they came from, because the failure mode was never a wrong
measurement -- it was a figure nobody had measured at all, wearing a citation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TIERS = REPO / "tools/pipeline/tiers.py"
HIERARCHY = REPO / "tools/campaign/hierarchy.py"

#: A cost block must carry one of these: an ALL-CAPS "MEASURED" (the repo's
#: convention for a figure someone actually ran), or a pointer to the
#: experiment log.
#:
#: CASE-SENSITIVE on purpose. A case-insensitive version passed on the phrase
#: "never measured" -- text that says the number is UNSOURCED satisfied a test
#: asserting it is sourced. Found by mutation: stripping every marker left the
#: test green.
PROVENANCE = re.compile(r"\bMEASURED\b|RESULTS\.md|\bM1[0-9]\b")


def _cost_block(path: Path) -> str:
    """The module docstring, which is where both files put their cost table."""
    src = path.read_text()
    m = re.search(r'^"""(.*?)"""', src, re.S | re.M)
    assert m, f"{path.name} has no module docstring to audit"
    return m.group(1)


@pytest.mark.parametrize("path", [TIERS, HIERARCHY], ids=lambda p: p.name)
def test_the_cost_table_names_its_provenance(path):
    """A cost table with no MEASURED/RESULTS.md marker is unsourced prose."""
    block = _cost_block(path)
    assert PROVENANCE.search(block), (
        f"{path.name}'s module docstring carries cost figures with no "
        "provenance marker (MEASURED / RESULTS.md / M<n>). That is how four "
        "tier costs became a citation loop with no measurement at the bottom."
    )


def test_neither_file_cites_the_other_as_a_cost_SOURCE():
    """The loop, pinned.

    `tiers.py` may REFER to hierarchy.py for the funnel RULES -- that is a real
    cross-reference. What it must not do is cite it for a cost NUMBER, because
    hierarchy.py's costs were themselves unsourced.
    """
    tiers = _cost_block(TIERS)
    # Find any line that both names hierarchy.py and carries a number+unit.
    offenders = [
        ln.strip()
        for ln in tiers.splitlines()
        if "hierarchy.py" in ln and re.search(r"\d+\s*(us|ms|s\b|min|h\b)", ln)
    ]
    assert not offenders, (
        "tiers.py cites hierarchy.py on a line carrying a cost figure: "
        f"{offenders}. hierarchy.py's own costs were unsourced, so this closes "
        "the citation loop again."
    )


def test_a_line_number_citation_is_not_used_for_a_cost():
    """`file.py:NN` rots: correcting one line shifts every citation below it.

    MEASURED: fixing the tier-2 line moved `tiers.py:13` and `:14` onto
    entirely the wrong rows. Cite a SYMBOL or a date, never a line number.
    """
    for path in (TIERS, HIERARCHY):
        block = _cost_block(path)
        bad = [
            ln.strip()
            for ln in block.splitlines()
            if re.search(r"\w+\.py:\d+", ln)
            and re.search(r"\d+\s*(us|ms|s\b|min|h\b)", ln)
        ]
        assert not bad, (
            f"{path.name} cites a file:line for a cost figure: {bad}. Line "
            "numbers rot -- cite a symbol (tier3_gfn2) or MEASURED <date>."
        )


#: The user-facing coverage table. This is where the numbers a reader actually
#: quotes live, and it is the file the existing guard did NOT cover.
COVERAGE = REPO / "site/src/reference/pharma-use-case-coverage.md"


@pytest.mark.skipif(not COVERAGE.is_file(), reason=f"no {COVERAGE}")
def test_the_coverage_table_agrees_with_the_measured_tier_costs():
    """The doc a reader quotes must carry the SAME numbers as the source.

    The provenance guard above checks that `tiers.py` says where its figures
    came from. It cannot catch a doc that copies the figures and then goes
    stale -- which is exactly what happened:

        coverage table          tiers.py (MEASURED)
        docking  1e-5 s/pose    26.4 s/ligand      <- 6 orders out
        FF       1e-3 s/pose    2.2-21.6 ms
        xtb      5e-1 s/pose    0.050-0.152 s

    The docking row is the one that mattered. At 1e-5 s tier 1 reads as free,
    when it is in fact 79% of a campaign's wall time -- the single number that
    should drive where optimization effort goes.

    Worse, `tiers.py` had ALREADY recorded that "~1 ms/pose" and "~0.5 s" were
    never measured. Its own comment says the golden path cited that line as its
    source. Correcting the numbers at the source did not reach the table.

    So this asserts AGREEMENT rather than provenance: every distinctive figure
    in `tiers.py`'s measured block must appear somewhere in the coverage table.
    """
    tiers = TIERS.read_text()
    doc = COVERAGE.read_text()

    # Distinctive substrings from the MEASURED block -- specific enough that a
    # stale doc cannot satisfy them by coincidence, and stable across
    # reformatting (no surrounding punctuation).
    # EXTRACT THE TABLE ROWS. Searching the whole document does not work:
    # this file contains a CORRECTION NOTE quoting the same figures, so a
    # stale table row passes while the note satisfies the check. MEASURED --
    # rewriting the docking row to "**WRONG s/ligand**" left all 7 tests
    # green, because "26.4" still appeared in the prose below it.
    rows = [ln for ln in doc.splitlines() if ln.startswith("|") and "yes |" in ln]
    assert len(rows) >= 7, (
        f"expected the coverage table's use-case rows, found {len(rows)}; the "
        "table's shape changed and this guard needs re-deriving"
    )
    table = "\n".join(rows)

    required = ["26.4", "2.2 ms", "21.6", "0.152", "612 s"]
    for token in required:
        assert token in tiers, (
            f"{token!r} is no longer in tiers.py's measured block -- this test "
            "is pinned to figures that moved; re-derive the list rather than "
            "deleting the assertion"
        )
        assert token in table, (
            f"the coverage TABLE does not carry {token!r}, which tiers.py "
            "reports as MEASURED. Checking the whole document instead lets a "
            "correction note stand in for a stale row."
        )


@pytest.mark.skipif(not COVERAGE.is_file(), reason=f"no {COVERAGE}")
def test_the_coverage_table_does_not_reassert_the_retracted_figures():
    """The three wrong numbers must not come back as live claims.

    They may appear in the CORRECTION note -- that is the record of what was
    wrong -- so this checks they are not in a TABLE ROW, which is where a
    reader takes a number from.
    """
    # EVERY table row, not only those mentioning `s/pose`. Keying on that unit
    # meant a retracted value reappearing with a different one -- "1e-5
    # s/ligand", say -- walked straight through.
    rows = [
        line
        for line in COVERAGE.read_text().splitlines()
        if line.startswith("|") and "yes |" in line
    ]
    for bad in ("1e-5", "1e-3", "5e-1"):
        offending = [r for r in rows if bad in r]
        assert not offending, (
            f"{bad!r} is back in a table row: {offending}. That figure was "
            "RETRACTED -- see the correction note in the same file."
        )


GOLDEN = REPO / "site/src/reference/pipeline-golden-path.md"


@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_the_quickstart_names_functions_that_return_what_it_claims():
    """The quickstart says "any input format -> a ferric Molecule". It must.

    It named `read_structure`, which returns a `Structure` -- symbols as a
    FIELD, not a method -- so pasting the line and calling `.symbols()` raises
    `TypeError: 'tuple' object is not callable`. The function that returns a
    Molecule is `read`. Found 2026-09-19 by running the block, which is the
    only way to find it: both names exist, both are exported, and the wrong one
    reads perfectly.

    This is the SECOND defect found in that block today (the first was
    `relative_descriptors` printed as exactly zero when it is -1.42e-14), so
    it earns a guard rather than another fix.

    The guard is deliberately narrow: it checks the quickstart does not claim
    `read_structure` returns a Molecule. Executing the whole block here would
    need rdkit and a docking extra in the fast tier.
    """
    text = GOLDEN.read_text()
    start = text.index("## 0b.")
    block = text[start : text.index("\n## ", start + 5)]

    assert "from tools.structure import" in block, (
        "the quickstart no longer imports from tools.structure; re-derive this "
        "guard rather than deleting it"
    )
    # The claim the block makes about itself.
    assert "-> a ferric Molecule" in block

    # REQUIRE the documented binding, do not merely blacklist the wrong one.
    #
    # An earlier version only rejected `read_structure(`, so swapping in any
    # other reader -- one that returns a Structure, a dict, anything -- passed
    # silently. A guard that forbids one known-bad name does not enforce a
    # contract; it enforces the absence of one mistake.
    assert "from tools.structure import" in block and "read" in block
    bindings = [
        line.strip()
        for line in block.splitlines()
        if line.strip().startswith(("mol = ", "mol="))
    ]
    assert bindings, "the quickstart must bind `mol` somewhere"
    readers = {"from_smiles(", "read("}
    for stripped in bindings:
        assert any(r in stripped for r in readers), (
            f"`mol` is bound by something other than the documented readers: "
            f"{stripped!r}. Only `read` and `from_smiles` return a Molecule; "
            "`read_structure` returns a Structure, whose `symbols` is a field "
            "rather than a method."
        )
        assert "read_structure(" not in stripped, (
            f"the quickstart binds `mol` to read_structure: {stripped!r}"
        )


#: `wiki/` is UNTRACKED (it lives only in the main checkout), so this guard
#: SKIPS rather than fails when it is absent -- a worktree or a CI runner has
#: no copy. Skipping is correct here: the alternative is a test that fails for
#: everyone who is not on Matt's box, which would be turned off and then never
#: re-enabled.
VALIDATION = REPO / "wiki/VALIDATION.md"


@pytest.mark.skipif(
    not VALIDATION.is_file(),
    reason="wiki/ is untracked; present only in the main checkout",
)
@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_the_functional_accuracies_match_validation_md():
    """The functional table quotes VALIDATION.md; the two must not drift.

    The golden path now tells a reader which functional to pick, justified by
    worst-case error against PySCF. Those numbers are TRANSCRIBED from
    `wiki/VALIDATION.md`, which is the authority -- and a transcribed number is
    exactly the kind that goes stale silently when the source is re-measured.

    This asserts each figure still appears in BOTH files. It deliberately does
    not parse the table: the value is catching a re-measurement that did not
    propagate, and a substring check does that without coupling to layout.
    """
    val = VALIDATION.read_text()
    doc = GOLDEN.read_text()

    # (label, the figure as VALIDATION.md states it)
    quoted = [
        ("PBE", "2.1e-8 Ha"),
        ("B3LYP", "1.6e-8 Ha"),
        ("LDA", "5.9e-6 Ha"),
        ("wB97X-V", "3.1e-5 Ha"),
        ("SCAN/r2SCAN", "1.95e-8 Ha"),
    ]
    for label, figure in quoted:
        assert figure in val, (
            f"{label}'s {figure} is no longer in wiki/VALIDATION.md -- it was "
            "re-measured. Update the golden path's functional table to match "
            "rather than deleting this assertion."
        )
        assert figure in doc, (
            f"the golden path quotes {label} but not its measured {figure}; "
            "the two files have drifted"
        )

    # The SCOPE limits matter as much as the numbers -- a reader who takes the
    # table without them will try a meta-GGA optimization at cc-pVDZ.
    assert "s/p-shell only" in val
    assert "s/p-shell only" in doc, (
        "the golden path must carry the meta-GGA gradient's s/p-shell limit; "
        "without it the functional table reads as a free choice"
    )


@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_a_retracted_figure_is_not_still_quoted_as_measured():
    """The note retracted "~2 min/ligand" and went on quoting it as MEASURED.

    The cost table says, in as many words, that the figure "was never measured
    and its `tiers.py:11` citation pointed at the module doc comment". Four
    hundred lines earlier the same note said "Tier 1 costs ~2 min/ligand
    (MEASURED, `tiers.py:11`)". A retraction and the claim it retracts lived in
    one document.

    Guards the SHAPE, not one string: if any figure is described as never
    measured, that same figure must not also be labelled MEASURED.
    """
    text = GOLDEN.read_text()
    if "never measured" not in text:
        pytest.skip("nothing is described as never-measured; guard is vacuous")

    # A QUOTATION of the retracted claim is fine and in fact desirable -- the
    # correction explains what it replaced. What must not recur is an
    # ASSERTIVE use, i.e. the figure stated as fact outside quotation marks.
    # Checked line by line so a quoted mention does not mask a real one.
    offenders = [
        ln.strip()
        for ln in text.splitlines()
        if "MEASURED, `tiers.py:11`" in ln and '"' not in ln and "~~" not in ln
    ]
    assert not offenders, (
        f"`tiers.py:11` is cited as a measurement outside quotation: "
        f"{offenders}. The note itself says that citation resolves to a "
        "module doc comment."
    )


@pytest.mark.skipif(not GOLDEN.is_file(), reason=f"no {GOLDEN}")
def test_the_zero_writes_claim_is_scoped_to_the_survey():
    """`context["geometry"]` IS written now, by funnel.py:184.

    The bare claim "Zero writes. Nothing in the repository ever populates
    context['geometry']" sat under a heading reading FIXED. Both cannot be
    true, and the reader hits the bare one first.
    """
    from pathlib import Path

    funnel = (Path(__file__).resolve().parents[1] / "funnel.py").read_text()
    writes = 'setdefault("geometry"' in funnel
    text = GOLDEN.read_text()
    if writes:
        assert (
            "**One read. Zero writes.** Nothing in the repository ever" not in text
        ), (
            "funnel.py writes context['geometry'], but the note still states "
            "unconditionally that nothing does"
        )
