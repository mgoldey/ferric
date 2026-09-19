import pytest

from tools.active_site.binding_energy import check_available_memory


def test_check_available_memory_passes_with_low_threshold():
    check_available_memory(0.001)


def test_check_available_memory_raises_with_impossible_threshold():
    with pytest.raises(MemoryError):
        check_available_memory(1_000_000.0)


def test_compute_binding_energy_states_what_it_cannot_RANK():
    """The ranking limit belongs at the CALL SITE, not only in a doc.

    `compute_binding_energy` is what a user calls to ask "which analogue binds
    better". MEASURED (RESULTS.md M4-M14), five protocols for getting a ddE out
    of a pose ensemble were tried and all five closed -- the best available
    noise is 4.07 kcal/mol against substituent effects of 1-2.

    A reader who finds this function from an IDE never sees the golden-path
    note. The limit has to travel with the function, and this asserts it still
    does: a docstring is exactly the kind of thing a later edit trims.

    Checks for the MEASURED FIGURE and the explicit prohibition, not for prose
    style -- a rewrite that keeps both is fine, one that drops the number is
    not.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "binding_energy.py").read_text()
    doc_start = src.index("def compute_binding_energy")
    doc = src[doc_start : doc_start + 4000]

    assert "4.07" in doc, (
        "compute_binding_energy no longer quotes the 4.07 kcal/mol ddE noise "
        "floor. Without it the function reads as a ranking tool, which five "
        "measured protocols say it is not."
    )
    assert "Do not order two analogues by it" in doc or "CANNOT DO: RANK" in doc, (
        "the explicit prohibition is gone; a noise figure alone invites the "
        "reader to decide for themselves whether 1-2 kcal/mol clears it"
    )
