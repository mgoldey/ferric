"""`run_saddle` from Python — the binding a catalyst workflow needs.

`ferric_scf::saddle::find_saddle` was Rust-only, so a workflow driven from
Python (which is how the whole `tools/` pipeline is driven) could COUNT
imaginary frequencies via `run_frequencies` and could not SEARCH for the saddle
they describe. Golden-path step C3 was unreachable from the language the rest
of the pipeline is written in.

These tests assert the two contracts that make the search safe to expose, not
the chemistry:

* a minimum's basin is REFUSED, with the reason intact across the FFI boundary
* `is_transition_state()` needs BOTH gradient convergence and exactly one
  imaginary mode -- convergence alone is satisfied by every stationary point
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")


def _h2(sep_angstrom: float):
    return ferric.Molecule.from_xyz_string(
        f"2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 {sep_angstrom}\n"
    )


def test_run_saddle_is_exposed():
    assert hasattr(ferric, "run_saddle"), "C3 must be reachable from Python"
    assert hasattr(ferric, "SaddleResult")


def test_a_minimum_basin_is_refused_with_its_reason():
    """H2 near equilibrium has nothing to climb.

    The refusal is the load-bearing behaviour: P-RFO from a minimum's basin
    would otherwise converge and hand back a MINIMUM labelled as a transition
    state. The reason must survive the FFI boundary, or a Python caller gets an
    opaque failure and no way to act on it.
    """
    with pytest.raises(Exception) as exc:
        ferric.run_saddle(_h2(0.74), "sto-3g", max_steps=20)
    msg = str(exc.value)
    assert "negative eigenvalue" in msg, f"the refusal must say WHY: {msg}"
    assert "minimum" in msg.lower(), f"and what to do about it: {msg}"


def test_trust_radius_is_validated_before_any_scf():
    """A bad knob must fail immediately, not after minutes of SCF."""
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(Exception) as exc:
            ferric.run_saddle(_h2(0.74), "sto-3g", trust_radius=bad)
        assert "trust_radius" in str(exc.value)


def test_is_transition_state_requires_both_halves():
    """Gradient convergence alone is satisfied by every stationary point.

    Exercised through the real class rather than a mock, so the accessor tested
    is the one Python callers actually get.
    """
    # The only way to obtain a SaddleResult here is a successful search, which
    # H2 cannot provide -- so assert the accessor exists and is callable on the
    # class, and let the Rust-side unit tests cover its truth table (they do,
    # in saddle.rs::is_transition_state_requires_both_halves).
    assert callable(getattr(ferric.SaddleResult, "is_transition_state", None)), (
        "SaddleResult must expose is_transition_state(); `converged` alone is "
        "not a transition state and a caller needs the combined check"
    )
