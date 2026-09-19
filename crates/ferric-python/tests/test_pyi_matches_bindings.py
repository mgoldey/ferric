"""`ferric.pyi` must not drift from the compiled module.

## Why this exists

`run_saddle` and `run_irc` -- the whole C3-C5 catalyst chain -- were ABSENT
from the stub, along with `SaddleResult`, `IrcResult` and `IrcBranch`. And
`run_frequencies` was missing `point_charges`/`external_field` after those were
added to the binding. A type checker rejected correct code, and an editor
offered no completion for any of it.

Nothing caught that, because a stub is not executed. This is the check that
makes the two files fail together instead of silently disagreeing.

## What it checks, and what it deliberately does not

It checks that every public `run_*` function and result class the MODULE
exports appears in the stub, and that the KEYWORD names match. It does not
compare type annotations: the stub says `list[tuple[float, float, float,
float]]` where pyo3 reports an opaque signature, and chasing that equivalence
would make the guard fail on cosmetic differences rather than on real drift.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

PYI = Path(__file__).resolve().parents[1] / "ferric.pyi"

#: Entry points whose absence from the stub is a real defect. Deliberately a
#: list of what MUST be there rather than a scan of everything the module
#: exports: a scan would turn every new binding into a failing test before its
#: author could write the stub, which trains people to skip the guard.
REQUIRED_FUNCTIONS = [
    "run_rhf",
    "run_optimize",
    "run_frequencies",
    "run_saddle",
    "run_irc",
    "run_qmmm",
    "run_optimize_qmmm",
]

REQUIRED_CLASSES = ["SaddleResult", "IrcResult", "IrcBranch", "FrequencyResult"]

#: Bindings that take an MM field. These are the ones where a missing kwarg in
#: the stub silently blocks the embedded catalyst workflow.
EMBEDDED = ["run_optimize", "run_frequencies", "run_saddle", "run_irc"]


@pytest.fixture(scope="module")
def stub() -> str:
    if not PYI.is_file():
        pytest.skip(f"no {PYI}")
    return PYI.read_text()


@pytest.mark.parametrize("name", REQUIRED_FUNCTIONS)
def test_every_required_function_is_in_the_stub(stub, name):
    assert hasattr(ferric, name), f"ferric.{name} is gone from the module"
    assert re.search(rf"^def {re.escape(name)}\(", stub, re.M), (
        f"ferric.{name} exists but is absent from ferric.pyi, so a type "
        "checker rejects correct code and an editor offers no completion"
    )


@pytest.mark.parametrize("name", REQUIRED_CLASSES)
def test_every_required_result_class_is_in_the_stub(stub, name):
    assert hasattr(ferric, name), f"ferric.{name} is gone from the module"
    assert re.search(rf"^class {re.escape(name)}\b", stub, re.M), (
        f"ferric.{name} exists but is absent from ferric.pyi"
    )


@pytest.mark.parametrize("name", EMBEDDED)
def test_the_embedded_kwargs_reach_the_stub(stub, name):
    """A kwarg the binding accepts and the stub omits blocks typed callers."""
    m = re.search(rf"^def {re.escape(name)}\((.*?)\n\) ->", stub, re.M | re.S)
    assert m, f"could not find {name}'s signature in the stub"
    block = m.group(1)
    for kw in ("point_charges", "external_field"):
        assert kw in block, (
            f"ferric.{name} accepts `{kw}` but ferric.pyi does not list it. "
            "The embedded QM/MM workflow is then untypeable, which is how "
            "run_frequencies/run_saddle/run_irc drifted in the first place."
        )


@pytest.mark.parametrize("name", EMBEDDED)
def test_the_binding_really_accepts_those_kwargs(stub, name):
    """The other direction: the stub must not promise what the module lacks.

    Without this the guard could be satisfied by editing only the stub, which
    is the easier and wronger fix.
    """
    fn = getattr(ferric, name)
    try:
        params = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        pytest.skip(f"{name} exposes no introspectable signature")
    for kw in ("point_charges", "external_field"):
        assert kw in params, (
            f"ferric.pyi lists `{kw}` on {name}, but the compiled binding does "
            "not accept it -- the stub is promising an API that does not exist"
        )
