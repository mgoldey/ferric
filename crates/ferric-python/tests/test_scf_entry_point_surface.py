"""The six molecular SCF entry points against their refusal table.

`ScfEntryPoint::refused` (ferric-scf `options::entry`, exported as the private
`ferric._scf_refused()`) is the one list of the knobs each of `run_rhf`,
`run_uhf`, `run_rohf`, `run_dft`, `run_uks`, `run_roks` refuses, with the
reason its solver cannot honour them. Every keyword any of the six takes must
be, for each entry point, either in its signature or in its refusal table,
never both: a new keyword is either offered or refused with a reason, and a
refused knob cannot creep into a signature.
"""

from __future__ import annotations

import inspect

import ferric
import pytest

SCF_ENTRY_POINTS = ["run_rhf", "run_uhf", "run_rohf", "run_dft", "run_uks", "run_roks"]


def _offered(fn):
    params = inspect.signature(getattr(ferric, fn)).parameters
    return set(params) - {"mol", "basis_set"}


def _refused(fn):
    return {name for names, _ in ferric._scf_refused()[fn] for name in names}


EVERY = set().union(*(_offered(fn) for fn in SCF_ENTRY_POINTS))


def test_the_table_lists_exactly_the_six_entry_points():
    assert sorted(ferric._scf_refused()) == sorted(SCF_ENTRY_POINTS)


@pytest.mark.parametrize("fn", SCF_ENTRY_POINTS)
def test_a_refused_knob_is_not_in_the_signature(fn):
    assert not _offered(fn) & _refused(fn)


@pytest.mark.parametrize("fn", SCF_ENTRY_POINTS)
def test_every_scf_knob_is_offered_or_refused(fn):
    missing = EVERY - _offered(fn) - _refused(fn)
    assert not missing, f"{fn}: neither offered nor refused: {sorted(missing)}"
    phantom = _refused(fn) - EVERY
    assert not phantom, (
        f"{fn}: refuses a keyword no entry point takes: {sorted(phantom)}"
    )


@pytest.mark.parametrize("fn", SCF_ENTRY_POINTS)
def test_every_refusal_has_a_reason(fn):
    for names, reason in ferric._scf_refused()[fn]:
        assert names and len(reason) > 20, (fn, names, reason)
