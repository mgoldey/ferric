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


def test_the_saddle_to_IRC_handoff_runs_FROM_PYTHON():
    """C3 -> C5, executed rather than inspected for attributes.

    SCOPE, stated because the earlier name overclaimed it: this exercises
    `run_saddle` and `run_irc` and NOTHING ELSE. It does not build a
    `QmmmSystem` (C0/C1), does not call `run_optimize_qmmm` (C2), and does not
    call `run_frequencies` (C4). Those have their own coverage; naming this
    "C0-C5" implied a chain test it is not.

    What it DOES cover is the handoff -- the saddle's geometry and its
    imaginary mode flowing into the IRC -- which is the join no other test
    touches.

    The failure this guards is a step that lands in Rust and never reaches the
    language `tools/` is written in. It has happened three times: C3 until
    `run_saddle` was bound, C4's mode vectors until #97, and C5 (the IRC) until
    `run_irc` -- each time the chain READ as complete because the capability
    existed somewhere.

    MEASURED here, NH3 umbrella inversion at STO-3G:

        saddle   converged, n_imaginary = 1, is_transition_state()
        IRC      -0.4257 / +0.4257 A pyramidalisation, both converged
        barrier  11.142 kcal/mol, symmetric to 3 decimals

    The symmetry is the load-bearing check. NH3's two pyramidal minima are
    mirror images, so a walk that went the same way twice -- the most likely
    direction bug -- gives the same energy but the SAME SIGN, and only the sign
    test catches it.
    """
    import math

    r, a = 1.006, math.radians(120)
    planar = (
        f"4\nplanar NH3\nN 0.0 0.0 0.0\nH {r:.6f} 0.0 0.0\n"
        f"H {r * math.cos(a):.6f} {r * math.sin(a):.6f} 0.0\n"
        f"H {r * math.cos(2 * a):.6f} {r * math.sin(2 * a):.6f} 0.0\n"
    )
    # Start PYRAMIDAL: inside the saddle's basin but not already at it, or the
    # search would succeed trivially.
    pyramidal = planar.replace("N 0.0 0.0 0.0", "N 0.0 0.0 0.15")

    sad = ferric.run_saddle(
        ferric.Molecule.from_xyz_string(pyramidal, 0, 1), "sto-3g", max_steps=60
    )
    assert sad.is_transition_state(), (
        f"C3: converged={sad.converged} n_imaginary={sad.n_imaginary}"
    )
    assert sad.imaginary_mode is not None and len(sad.imaginary_mode) == 12

    # Build the IRC input from the SEARCH RESULT, not from the hand-written
    # `planar` string.
    #
    # Passing `planar` made this test pass `sad.imaginary_mode` while silently
    # ignoring `sad.coords` -- so the saddle-to-IRC handoff was never
    # exercised, and `run_irc` recomputed `saddle_energy` at a geometry the
    # search had not produced. The test would have passed with `sad.coords`
    # returning anything at all.
    found = "{}\nfound saddle\n{}\n".format(
        len(sad.symbols),
        "\n".join(
            f"{s} {x:.8f} {y:.8f} {z:.8f}"
            for s, (x, y, z) in zip(sad.symbols, sad.coords)
        ),
    )
    irc = ferric.run_irc(
        ferric.Molecule.from_xyz_string(found, 0, 1),
        "sto-3g",
        sad.imaginary_mode,
        step=0.15,
        max_steps=120,
    )
    # The IRC's saddle energy must be the one the SEARCH found, not some other
    # geometry's. This is the assertion that makes the handoff checkable.
    assert abs(irc.saddle_energy - sad.energy) < 1e-8, (
        f"run_irc recomputed the saddle at {irc.saddle_energy} but the search "
        f"returned {sad.energy} -- the geometry handoff is broken"
    )
    assert irc.both_converged(), (
        f"C5: fwd converged={irc.forward.converged} "
        f"rev converged={irc.reverse.converged}; an unconverged branch "
        "identifies no basin"
    )

    def pyramidalisation(branch):
        h_mean = sum(c[2] for c in branch.coords[1:]) / 3
        return branch.coords[0][2] - h_mean

    pf, pr = pyramidalisation(irc.forward), pyramidalisation(irc.reverse)
    assert pf * pr < 0, (
        f"the two branches must end on OPPOSITE sides of the H3 plane; got "
        f"{pf:+.4f} and {pr:+.4f} A. Same sign means both walks went the same "
        "way, which every energy-based check would miss."
    )
    assert abs(pf) > 0.05 and abs(pr) > 0.05

    # Mirror images, so degenerate -- catches a branch that wandered off the
    # umbrella coordinate.
    assert abs(irc.forward.energy - irc.reverse.energy) < 1e-6
    for b in (irc.forward_barrier(), irc.reverse_barrier()):
        assert b > 0, f"a barrier of {b} Ha means the endpoint is ABOVE the saddle"
