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
    "run_xtb_energy_gradient",
    "run_xtb_optimize",
    "run_xtb_frequencies",
    "run_qmmm",
    "run_optimize_qmmm",
    "run_aimd",
    "run_ffmd",
    "run_rhf_gamma",
    "run_uhf_gamma",
    "run_rohf_gamma",
    "run_uks_gamma",
    "run_roks_gamma",
    "run_rks_gamma",
    "run_mp2_gamma",
    "run_drpa_gamma",
    "run_rhf_kpts",
    "run_uhf_kpts",
    "run_rks_kpts",
    "run_mp2_kpts",
    "run_drpa_kpts",
]

REQUIRED_CLASSES = [
    "SaddleResult",
    "IrcResult",
    "IrcBranch",
    "FrequencyResult",
    "MdResult",
    "GammaRhfResult",
    "GammaOpenShellResult",
    "GammaRksResult",
    "GammaCorrelationResult",
    "KpointScfResult",
    "KpointCorrelationResult",
]

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


def _stub_class_members(stub: str, cls: str) -> set[str]:
    m = re.search(rf"^class {re.escape(cls)}\b.*?(?=^\S)", stub, re.M | re.S)
    assert m, f"no class {cls} in the stub"
    return set(re.findall(r"^    def (\w+)\(", m.group(0), re.M))


def test_molecule_members_match_the_stub_both_ways(stub):
    """`Molecule` is the type every caller touches first, so its stub lists
    exactly the compiled class's public members (methods and getters such as
    `charge`/`multiplicity`), and nothing the class lacks."""
    compiled = {n for n in dir(ferric.Molecule) if not n.startswith("_")}
    stubbed = {
        n for n in _stub_class_members(stub, "Molecule") if not n.startswith("_")
    }
    assert {"charge", "multiplicity"} <= compiled, "Molecule lost its getters"
    assert compiled - stubbed == set(), f"missing from ferric.pyi: {compiled - stubbed}"
    assert stubbed - compiled == set(), (
        f"stub-only Molecule members: {stubbed - compiled}"
    )


def test_molecule_getters_and_repr():
    mol = ferric.Molecule.from_arrays(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
        charge=1,
        multiplicity=2,
    )
    assert (mol.charge, mol.multiplicity) == (1, 2)
    assert repr(mol) == "Molecule(H2O, 3 atoms, charge=1, multiplicity=2)"
    with pytest.raises(AttributeError):
        mol.charge = 0  # read-only
    ch4 = ferric.Molecule.from_arrays(
        ["H", "C", "H", "H", "H"],
        [[0, 0, 0], [0, 0, 1.1], [1, 0, 1.5], [0, 1, 1.5], [-1, -1, 1.5]],
    )
    assert repr(ch4) == "Molecule(CH4, 5 atoms, charge=0, multiplicity=1)"
    he = ferric.Molecule.from_arrays(["He"], [[0, 0, 0]])
    assert repr(he) == "Molecule(He, 1 atom, charge=0, multiplicity=1)"
    assert str(mol) == repr(mol)
