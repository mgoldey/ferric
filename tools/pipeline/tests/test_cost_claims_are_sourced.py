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
