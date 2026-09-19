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
    required = ["26.4", "2.2 ms", "21.6", "0.152", "612 s"]
    for token in required:
        assert token in tiers, (
            f"{token!r} is no longer in tiers.py's measured block -- this test "
            "is pinned to figures that moved; re-derive the list rather than "
            "deleting the assertion"
        )
        assert token in doc, (
            f"the coverage table does not mention {token!r}, which tiers.py "
            "reports as MEASURED. A doc that carries different numbers from "
            "its source is how 1e-5 s/pose survived for docking."
        )


@pytest.mark.skipif(not COVERAGE.is_file(), reason=f"no {COVERAGE}")
def test_the_coverage_table_does_not_reassert_the_retracted_figures():
    """The three wrong numbers must not come back as live claims.

    They may appear in the CORRECTION note -- that is the record of what was
    wrong -- so this checks they are not in a TABLE ROW, which is where a
    reader takes a number from.
    """
    rows = [
        line
        for line in COVERAGE.read_text().splitlines()
        if line.startswith("|") and "s/pose" in line
    ]
    for bad in ("1e-5 s/pose", "1e-3 s/pose", "5e-1 s/pose"):
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

    # `read_structure` may be MENTIONED (the note explaining the difference is
    # useful), but it must not be the call bound to `mol`.
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("mol = ") or stripped.startswith("mol="):
            assert "read_structure(" not in stripped, (
                f"the quickstart binds `mol` to read_structure, which returns a "
                f"Structure and not a Molecule: {stripped!r}. Use `read`."
            )
