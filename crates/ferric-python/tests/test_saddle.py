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


def test_the_whole_catalyst_procedure_is_reachable_from_python():
    """C0-C5 of the golden path, every step callable from Python.

    The catalyst procedure is documented in
    `wiki/golden-path-pipeline-2026-09-18.md` section 3b(b). Its steps were
    landing one at a time across several PRs, and the failure mode is a
    procedure that reads as complete while one step lives only in Rust -- which
    is exactly what happened to C3 until `run_saddle` was bound, and to C4's
    mode vectors until #97.

    This asserts REACHABILITY, not correctness: each step has its own
    correctness tests. What it catches is a step quietly becoming unreachable
    from the language the `tools/` pipeline is written in.
    """
    for step, name in [
        ("C0/C1 QM region + link atoms", "QmmmSystem"),
        ("C2 optimize reactant/product", "run_optimize_qmmm"),
        ("C3 FIND the transition state", "run_saddle"),
        ("C4 verify the transition state", "run_frequencies"),
        # C5 is arithmetic on C2/C3 energies -- no entry point to check.
    ]:
        assert hasattr(ferric, name), (
            f"{step} is not reachable from Python (ferric.{name} missing); the "
            "catalyst procedure documents it as available"
        )


def test_C4_has_BOTH_halves_from_python():
    """Counting imaginary modes is necessary, not sufficient.

    One imaginary frequency means first-order saddle, not "the saddle you
    meant" -- a methyl rotor gives one too. Completing C4 needs the MODE
    VECTOR, to check it displaces atoms along the reaction coordinate. That was
    Rust-only until #97, and the golden path carried a stale "MODE VECTORS are
    Rust-only" caveat for a day afterwards.
    """
    fr = ferric.run_frequencies(_h2(0.74), "sto-3g")

    # C4a: the count.
    assert hasattr(fr, "frequencies")
    n_imag = sum(1 for f in fr.frequencies if f < 0)
    assert n_imag == 0, f"H2 at equilibrium is a MINIMUM, got {n_imag} imaginary"

    # C4b: the vectors. 3N entries per mode, in Cartesians.
    assert hasattr(fr, "normal_modes"), "C4 cannot be completed without the modes"
    assert len(fr.normal_modes) == len(fr.frequencies)
    n_atoms = 2
    assert all(len(m) == 3 * n_atoms for m in fr.normal_modes), (
        f"each mode must have 3N = {3 * n_atoms} Cartesian components"
    )
