"""The between-substituent spread must come from MATCHED SITES.

`_report_site_dependence` answers one question: does WHERE a group goes matter
as much as WHICH group it is? That determines the pipeline's unit of work --
per-substituent mean, or the (substituent, site) pair.

The first version computed the "between substituents" number as the range over
EVERY (substituent, site) score. That pool already contains the
within-substituent site variation, so it compared a subset against its own
superset. Two consequences, and the second is why this file exists:

1. the ratio is bounded by 1 BY CONSTRUCTION, so it can never report that site
   matters MORE than identity, however strongly the data say so;
2. it cannot distinguish "site matters as much as identity" from "site is most
   of what the pooled range is measuring" -- which are opposite conclusions.

These tests are built around inputs where the two formulas DISAGREE. An input
where they happen to agree would pass either way and prove nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_substitution_scan import (  # noqa: E402
    HARTREE_TO_KCAL,
    _report_site_dependence,
)


def _rows(triples):
    """(score_kcal, label, site) -> the (hartree, label, site) rows the fn takes."""
    return [(k / HARTREE_TO_KCAL, lab, site) for k, lab, site in triples]


def test_between_is_measured_at_a_matched_site(capsys):
    """Site variation must NOT inflate the between-substituent number.

    Constructed so the two formulas give different answers:

      site 0:  F = 0.0,  Cl = 1.0     -> matched between = 1.0
      site 1:  F = 10.0, Cl = 11.0    -> matched between = 1.0
      within F  = 10.0, within Cl = 10.0

    Pooled range is 11.0, so the OLD ratio is 10/11 = 0.91 -- under 1, and it
    reads as "site matters slightly less than identity". The matched ratio is
    10/1 = 10.0: site dominates identity by an order of magnitude. Opposite
    stories from the same data.
    """
    _report_site_dependence(
        _rows(
            [
                (0.0, "F", 0),
                (10.0, "F", 1),
                (1.0, "Cl", 0),
                (11.0, "Cl", 1),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "matched site" in out, "the between number must say it is matched"
    assert "  1.00 kcal/mol (matched site)" in out, (
        f"between should be 1.00 from the matched-site comparison, got:\n{out}"
    )
    assert " 10.00" in out, f"within should be 10.00, got:\n{out}"
    assert " 10.00\n" in out or "ratio                 :  10.00" in out, (
        f"the ratio must be able to EXCEED 1; the pooled formula caps it at "
        f"0.91 here. Got:\n{out}"
    )


def test_the_ratio_can_exceed_one(capsys):
    """THE discriminating property: the old formula could not produce this.

    `within / (max(pool) - min(pool))` has the within-substituent values inside
    its own denominator, so it is <= 1 always. A ratio above 1 is therefore
    proof the matched comparison is being used, not an incidental value.
    """
    _report_site_dependence(
        _rows([(0.0, "F", 0), (100.0, "F", 1), (0.5, "Cl", 0), (100.5, "Cl", 1)])
    )
    out = capsys.readouterr().out
    ratio = float(out.split("ratio")[1].split(":")[1].split()[0])
    assert ratio > 1.0, (
        f"ratio {ratio} is <= 1, which is exactly the ceiling the pooled-range "
        "formula imposes -- the matched comparison is not being used"
    )
    assert ratio == pytest.approx(200.0, rel=1e-6), (
        f"within 100.0 / matched-between 0.5 = 200; got {ratio}"
    )


def test_confounded_input_refuses_rather_than_falling_back(capsys):
    """When no site holds two substituents, say so -- do not use the pool.

    Identity and placement are perfectly confounded in that case. Reporting the
    pooled range instead would silently reinstate the original bug in exactly
    the situation where it is most misleading.
    """
    _report_site_dependence(_rows([(0.0, "F", 0), (10.0, "F", 1), (3.0, "Cl", 2)]))
    out = capsys.readouterr().out
    assert "NOT MEASURABLE" in out, f"a confounded run must refuse, got:\n{out}"
    assert "confounded" in out
    assert "ratio" not in out, (
        "no ratio may be printed when the comparison is not identified"
    )


def test_a_single_site_per_substituent_is_still_reported_as_unmeasurable(capsys):
    """The pre-existing early return: nothing to say about site dependence."""
    _report_site_dependence(_rows([(0.0, "F", 0), (1.0, "Cl", 1)]))
    out = capsys.readouterr().out
    assert "not measurable" in out.lower()
