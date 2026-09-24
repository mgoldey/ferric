"""Constrained DFT (`run_cdft`, `CdftConstraint`, `cdft_coupling`) bindings.

Every numerical test here re-runs a configuration the Rust cDFT suite already
validates (`crates/ferric-scf/tests/cdft_*.rs`) through the Python surface, so
a pass is evidence that the binding reaches the validated library path with the
same inputs -- not a new physics claim. The pinned HeNe+ anchor is READ from the
Rust test source rather than copied, so the two cannot drift apart.

Each test says what it catches and how it fails if the binding is broken.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

REPO = Path(__file__).resolve().parents[3]
OUTER_LOOP_RS = REPO / "crates" / "ferric-scf" / "tests" / "cdft_outer_loop.rs"


def _rust_const(name: str) -> float:
    """A `const NAME: f64 = ...;` literal from the Rust anchor test."""
    src = OUTER_LOOP_RS.read_text()
    m = re.search(rf"const {name}: f64 = ([^;]+);", src)
    assert m, f"{name} not found in {OUTER_LOOP_RS}"
    return float(m.group(1).replace("_", ""))


def _lih(charge: int = 0, mult: int = 1):
    return ferric.Molecule.from_xyz_string(
        "2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", charge, mult
    )


def _svp():
    return ferric.BasisSet.bundled("def2-svp")


# ---------------------------------------------------------------------------
# Cached solves (each is a full constrained SCF; several tests read each one)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _lih_charge_22():
    # == cdft_uhf.rs::charge_constraint_is_satisfied: LiH/def2-SVP, Li (atom 0)
    # charge population 2.2, lambda_tol 1e-5, max_outer 40, defaults otherwise.
    return ferric.run_cdft(
        _lih(),
        _svp(),
        [ferric.CdftConstraint([0], 2.2, kind="charge")],
        lambda_tol=1e-5,
        max_outer=40,
    )


@functools.lru_cache(maxsize=None)
def _lih_uhf():
    return ferric.run_uhf(_lih(), _svp())


@functools.lru_cache(maxsize=None)
def _lih_plus_spin_045():
    # == cdft_uhf.rs::spin_constraint_is_satisfied_lih_plus: LiH+ doublet, Li
    # spin population N_a - N_b = 0.45 (baseline ~0.274).
    return ferric.run_cdft(
        _lih(1, 2),
        _svp(),
        [ferric.CdftConstraint([0], 0.45, kind="spin")],
        lambda_tol=1e-5,
        max_outer=40,
    )


@functools.lru_cache(maxsize=None)
def _he2_plus_states(r_ang: float):
    # == cdft_coupling.rs::he2_plus_hab: He2+ doublet, def2-SVP, hole on atom 0
    # (N(atom0) = 1.0) vs on atom 1, lambda_tol 1e-2 (flat-response plateau),
    # level_shift 0.2, 99x302 grid, max_outer 40, defaults otherwise.
    mol = ferric.Molecule.from_xyz_string(f"2\nHe2+\nHe 0 0 0\nHe 0 0 {r_ang}\n", 1, 2)
    kw = dict(
        lambda_tol=1e-2,
        max_outer=40,
        level_shift=0.2,
        grid_radial=99,
        grid_angular=302,
    )
    a = ferric.run_cdft(mol, _svp(), [ferric.CdftConstraint([0], 1.0)], **kw)
    b = ferric.run_cdft(mol, _svp(), [ferric.CdftConstraint([1], 1.0)], **kw)
    return a, b


# ---------------------------------------------------------------------------
# Anchor to the validated library path
# ---------------------------------------------------------------------------


def test_hene_plus_reproduces_the_rust_pinned_constrained_solution():
    """HeNe+/def2-SVP, R = 2.0 A, N(He) = 2.0, hcore start: the exact config of
    `cdft_outer_loop.rs::hcore_started_path_reaches_the_same_constrained_solution`,
    asserted at that test's own bars (|dE| < 1e-5 and the tighter 1e-6,
    |dlambda| < 1e-3, |N - 2| < 1e-5).

    Catches: any knob the binding drops or maps to the wrong RhfConfig field
    (guess, level_shift, max_iter, lambda_tol, max_outer, stability_descent,
    the fragment index, the charge/spin channel). E.g. ignoring guess="hcore"
    runs from MINAO and lands elsewhere; ignoring stability_descent=False lets
    the descent move E ~0.0245 Ha lower (the documented lower state), which
    fails the 1e-5 energy bar by three orders of magnitude.
    """
    anchor_e = _rust_const("ANCHOR_E")
    anchor_lam = _rust_const("ANCHOR_LAMBDA")
    mol = ferric.Molecule.from_xyz_string(
        "2\nHeNe+\nHe 0.0 0.0 0.0\nNe 0.0 0.0 2.0\n", 1, 2
    )
    r = ferric.run_cdft(
        mol,
        _svp(),
        [ferric.CdftConstraint([0], 2.0, kind="charge")],
        guess="hcore",
        level_shift=0.5,
        max_iter=400,
        lambda_tol=1e-5,
        max_outer=40,
        stability_descent=False,
        grid_radial=99,
        grid_angular=302,
    )
    print(
        f"HeNe+ anchor: E={r.energy:.12f} (rust {anchor_e:.12f}) "
        f"lambda={r.lambdas[0]:.10f} (rust {anchor_lam:.10f}) N={r.populations[0]:.10f}"
    )
    assert r.converged, r
    assert abs(r.populations[0] - 2.0) < 1e-5, r.populations
    assert abs(r.energy - anchor_e) < 1e-5, (r.energy, anchor_e)
    assert abs(r.energy - anchor_e) < 1e-6, (r.energy, anchor_e)
    assert abs(r.lambdas[0] - anchor_lam) < 1e-3, (r.lambdas, anchor_lam)


# ---------------------------------------------------------------------------
# Trivial-limit anchor
# ---------------------------------------------------------------------------


def test_target_equal_to_the_unconstrained_population_is_a_no_op():
    """Constrain Li to the population the UNCONSTRAINED UHF density already has.

    The outer loop starts at lambda = 0; there the inner solve IS plain UHF (the
    Fock modifier adds 0 * W, cf. cdft_uhf.rs::lambda_zero_equals_plain_uhf),
    so the residual is already below lambda_tol and the driver must return on
    outer iteration 1 with lambda exactly 0 and the run_uhf energy.

    Catches: a binding that passes the wrong fragment, the wrong channel
    (the spin population of this closed shell is ~0, not ~N0, so lambda would
    move off zero), a target transformed on the way in (e.g. treated as a net
    charge, Z - N), or a constraint term left in the reported energy. Each
    shows up as lambda != 0, outer_iterations > 1, or an energy shift.
    """
    ref = _lih_charge_22()
    uhf = _lih_uhf()
    assert uhf.converged
    w = ref.weight_matrix(0)
    n0 = float(np.sum(w * (uhf.density_alpha() + uhf.density_beta())))
    # Reachability: N0 must be far from the other test's target, or this would
    # be the same solve and prove nothing about the trivial limit.
    assert abs(n0 - 2.2) > 0.05, n0

    r = ferric.run_cdft(
        _lih(),
        _svp(),
        [ferric.CdftConstraint([0], n0)],
        lambda_tol=1e-5,
        max_outer=40,
        stability_descent=False,
    )
    print(
        f"trivial limit: N0={n0:.10f} lambda={r.lambdas} E={r.energy} vs {uhf.energy}"
    )
    assert r.converged
    assert r.outer_iterations == 1, r.outer_iterations
    assert r.lambdas == [0.0], r.lambdas
    assert abs(r.populations[0] - n0) < 1e-5
    assert abs(r.energy - uhf.energy) < 1e-8, (r.energy, uhf.energy)


# ---------------------------------------------------------------------------
# The constraint is applied
# ---------------------------------------------------------------------------


def test_charge_constraint_is_met_and_raises_the_energy():
    """LiH, N(Li) = 2.2 (cdft_uhf.rs::charge_constraint_is_satisfied).

    Catches: a constraint that never reaches the Fock matrix (population stays
    at the unconstrained N0 far from 2.2, lambda = 0), a population reported
    on a different channel than solved (the independent trace below disagrees),
    and an energy that includes the lambda * (N - target) term or is not the
    constrained minimum (a constrained minimum cannot lie below the
    unconstrained one).
    """
    r = _lih_charge_22()
    uhf = _lih_uhf()
    assert r.converged and r.scf_converged
    assert r.kinds == ["charge"] and r.targets == [2.2]
    assert abs(r.populations[0] - 2.2) < 1e-5, r.populations
    assert r.lambda_tol == 1e-5
    assert r.max_constraint_error < r.lambda_tol
    assert abs(r.lambdas[0]) > 1e-3, r.lambdas
    # Independent population from the returned density and weight operator.
    w = r.weight_matrix(0)
    n = float(np.sum(w * (r.density_alpha() + r.density_beta())))
    assert abs(n - r.populations[0]) < 1e-8, (n, r.populations)
    assert r.energy > uhf.energy + 1e-6, (r.energy, uhf.energy)


def test_spin_constraint_is_met_on_the_spin_channel():
    """LiH+, N_a - N_b on Li = 0.45 (cdft_uhf.rs::spin_constraint_is_satisfied_lih_plus).

    Catches: kind="spin" mapped to the charge channel. Then the solve would
    drive N_a + N_b (about 2) to 0.45 -- which either fails or lands on a
    state whose trace(W (Da - Db)) is not 0.45; the independent spin trace
    below fails either way. Also catches lambda stuck at 0.
    """
    r = _lih_plus_spin_045()
    assert r.converged and r.kinds == ["spin"]
    assert abs(r.populations[0] - 0.45) < 1e-5, r.populations
    assert abs(r.lambdas[0]) > 1e-3, r.lambdas
    w = r.weight_matrix(0)
    spin = float(np.sum(w * (r.density_alpha() - r.density_beta())))
    assert abs(spin - 0.45) < 1e-5, spin
    na, nb = r.nocc
    assert (na, nb) == (2, 1)


def test_uks_constraint_is_met_and_the_functional_is_applied():
    """SMOKE (no Rust UKS-cDFT test exists to anchor to): LiH/PBE, N(Li) = 2.2.

    Catches: `functional` silently dropped (the energy would equal the UHF-cDFT
    energy of `_lih_charge_22` instead of differing by the several-10 mHa
    PBE-vs-HF gap) and a UKS path that does not honour the constraint.
    """
    r = ferric.run_cdft(
        _lih(),
        _svp(),
        [ferric.CdftConstraint([0], 2.2)],
        functional="PBE",
        lambda_tol=1e-5,
        max_outer=40,
    )
    assert r.converged
    assert abs(r.populations[0] - 2.2) < 1e-5, r.populations
    assert abs(r.energy - _lih_charge_22().energy) > 1e-2, r.energy


# ---------------------------------------------------------------------------
# Strict validation (all raise before any SCF runs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["charges", "total", "spin_diff", "spindiff", ""])
def test_unknown_kind_raises(bad):
    """Catches a non-strict kind parser: any silent default picks a Lagrangian."""
    with pytest.raises(ValueError) as exc:
        ferric.CdftConstraint([0], 1.0, kind=bad)
    assert "'charge'" in str(exc.value) and "'spin'" in str(exc.value)


def test_kind_is_case_insensitive_but_canonicalised():
    c = ferric.CdftConstraint([0], 1.0, kind="Spin")
    assert c.kind == "spin" and c.atoms == [0] and c.target == 1.0


def test_empty_atom_list_raises():
    """An empty fragment has population 0 identically: a solve could never succeed."""
    with pytest.raises(ValueError, match="non-empty"):
        ferric.CdftConstraint([], 1.0)


@pytest.mark.parametrize("atoms", [[-1], [0, -2]])
def test_negative_atom_index_raises_valueerror(atoms):
    """Negative indices would otherwise be an OverflowError from usize extraction."""
    with pytest.raises(ValueError, match="negative"):
        ferric.CdftConstraint(atoms, 1.0)


def test_duplicate_atom_index_raises():
    """A repeated atom would be weighted twice in W^C."""
    with pytest.raises(ValueError, match="duplicate"):
        ferric.CdftConstraint([0, 0], 1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_target_raises(bad):
    with pytest.raises(ValueError, match="finite"):
        ferric.CdftConstraint([0], bad)


def test_out_of_range_atom_index_raises():
    """LiH has atoms 0 and 1; index 2 must be refused naming the atom count,
    not panic inside the Becke weight build."""
    with pytest.raises(ValueError) as exc:
        ferric.run_cdft(_lih(), _svp(), [ferric.CdftConstraint([2], 1.0)])
    assert "atom 2" in str(exc.value) and "2 atoms" in str(exc.value)


@pytest.mark.parametrize("target", [-0.1, 4.5])
def test_charge_target_outside_the_electron_count_raises(target):
    """LiH has 4 electrons. Catches a caller passing a NET CHARGE by mistake
    for negative values, and an impossible population."""
    with pytest.raises(ValueError, match="POPULATION"):
        ferric.run_cdft(_lih(), _svp(), [ferric.CdftConstraint([0], target)])


def test_spin_target_beyond_the_electron_count_raises():
    with pytest.raises(ValueError, match="spin"):
        ferric.run_cdft(
            _lih(1, 2), _svp(), [ferric.CdftConstraint([0], -3.5, kind="spin")]
        )


def test_empty_constraint_list_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ferric.run_cdft(_lih(), _svp(), [])


def test_repeated_constraint_raises():
    """Two constraints on the same fragment+kind make the Jacobian singular."""
    cons = [ferric.CdftConstraint([0, 1], 3.0), ferric.CdftConstraint([1, 0], 3.1)]
    with pytest.raises(ValueError, match="singular"):
        ferric.run_cdft(_lih(), _svp(), cons)


@pytest.mark.parametrize(
    "kw",
    [
        {"lambda_tol": 0.0},
        {"lambda_tol": -1e-5},
        {"lambda_tol": math.nan},
        {"max_outer": 0},
        {"grid_angular": 111},
        {"grid_radial": 0},
        {"functional": "PBEE"},
        {"guess": "hcroe"},
    ],
)
def test_bad_knobs_raise(kw):
    with pytest.raises(ValueError):
        ferric.run_cdft(_lih(), _svp(), [ferric.CdftConstraint([0], 2.2)], **kw)


# ---------------------------------------------------------------------------
# Wu-Van Voorhis coupling
# ---------------------------------------------------------------------------


def _hab_reference(a, b, mispair: bool = False) -> tuple[float, float]:
    """H_ab and S_ab by an INDEPENDENT construction.

    `mispair=True` returns the value a binding that swapped the two states'
    lambdas (lambda_a with W_b) would produce, for the reachability check.

    Generalized Slater-Condon transition-density form with explicit inverses
    (the same reference `cdft_coupling.rs::coupling_pairs_each_lambda_with_its_own_weight`
    uses against the SVD kernel): <a|O|b> = S_ab * sum_spin tr(O C_b M^-1 C_a^T),
    M = C_a^T S C_b. S itself is recovered from the MOs (C^T S C = 1 for a full,
    square MO set, so S = (C C^T)^-1), so nothing here comes from the kernel.
    """
    ca = a.mo_coeff_alpha()
    assert ca.shape[0] == ca.shape[1], "S reconstruction needs a square MO set"
    s = np.linalg.inv(ca @ ca.T)
    na, nb = a.nocc
    blocks = [
        (a.mo_coeff_alpha()[:, :na], b.mo_coeff_alpha()[:, :na]),
        (a.mo_coeff_beta()[:, :nb], b.mo_coeff_beta()[:, :nb]),
    ]
    s_ab = 1.0
    for ab, bb in blocks:
        s_ab *= np.linalg.det(ab.T @ s @ bb)

    def elem(op):
        tr = 0.0
        for ab, bb in blocks:
            m_inv = np.linalg.inv(ab.T @ s @ bb)
            tr += np.trace(op @ bb @ m_inv @ ab.T)
        return s_ab * tr

    wa = elem(a.weight_matrix(0))
    wb = elem(b.weight_matrix(0))
    e_a, e_b = a.energy, b.energy
    l_a, l_b = a.lambdas[0], b.lambdas[0]
    if mispair:
        l_a, l_b = l_b, l_a
    h_raw = 0.5 * ((e_b * s_ab - l_b * wb) + (e_a * s_ab - l_a * wa))
    h_ab = (h_raw - 0.5 * (e_a + e_b) * s_ab) / (1.0 - s_ab * s_ab)
    return h_ab, s_ab


def test_he2_plus_coupling_matches_an_independent_construction():
    """He2+ at 2.5 A (cdft_coupling.rs::he2_plus_coupling_is_finite_and_symmetric):
    E_a == E_b to 1e-4, 0 < |H_ab| < 1, PLUS agreement with `_hab_reference`.

    The reference check catches binding wiring errors the Rust bars cannot:
    passing the alpha MOs as beta, the wrong energy or lambda value, a wrong
    nocc, a missing W. It does NOT catch lambda/W MISPAIRING: on this symmetric
    pair lambda_a == lambda_b and <a|W_0|b> == <a|W_1|b> by symmetry, so a swap
    is invisible (the Rust file documents the same trap);
    `test_asymmetric_coupling_pairs_each_lambda_with_its_own_weight` covers it.
    Compared as |.| because the sign of H_ab and S_ab is a determinant phase.
    """
    a, b = _he2_plus_states(2.5)
    c = ferric.cdft_coupling(a, b)
    h_ref, s_ref = _hab_reference(a, b)
    print(
        f"He2+ 2.5A: {c!r}; reference |H_ab|={abs(h_ref):.10f} |S_ab|={abs(s_ref):.6e}"
    )
    assert abs(c.e_a - c.e_b) < 1e-4, (c.e_a, c.e_b)
    assert c.e_a == a.energy and c.e_b == b.energy
    assert math.isfinite(c.h_ab) and 0.0 < abs(c.h_ab) < 1.0
    assert abs(abs(c.s_ab) - abs(s_ref)) < 1e-8, (c.s_ab, s_ref)
    assert abs(abs(c.h_ab) - abs(h_ref)) < 1e-6 * max(1.0, abs(h_ref)), (c.h_ab, h_ref)
    # Swap symmetry (necessary, not sufficient -- see the reference check).
    cba = ferric.cdft_coupling(b, a)
    assert abs(abs(cba.h_ab) - abs(c.h_ab)) < 1e-10


def test_he2_plus_coupling_decays_with_distance():
    """cdft_coupling.rs::he2_plus_coupling_decays_with_distance, via Python.

    Catches a coupling that ignores the states' geometry-dependent inputs
    (e.g. a constant, or E_a returned as H_ab from the degenerate branch).
    """
    h = [abs(ferric.cdft_coupling(*_he2_plus_states(r)).h_ab) for r in (2.5, 3.0, 3.5)]
    print(f"He2+ |H_ab| at 2.5/3.0/3.5 A: {h}")
    assert h[0] > h[1] > h[2], h


def test_coupling_refuses_a_state_with_itself():
    """|S_ab| = 1: the Rust kernel returns E_a as a sentinel. The binding must
    raise instead of handing that back as a coupling."""
    a, _ = _he2_plus_states(2.5)
    with pytest.raises(ValueError, match="undefined"):
        ferric.cdft_coupling(a, a)


def test_coupling_refuses_states_from_different_geometries():
    a, _ = _he2_plus_states(2.5)
    _, b = _he2_plus_states(3.0)
    with pytest.raises(ValueError, match="overlap"):
        ferric.cdft_coupling(a, b)


def _he2_plus_state_b_with(**extra):
    """State b of the R = 3.0 A He2+ pair, solved with extra run_cdft kwargs."""
    mol = ferric.Molecule.from_xyz_string("2\nHe2+\nHe 0 0 0\nHe 0 0 3.0\n", 1, 2)
    kw = dict(
        lambda_tol=1e-2,
        max_outer=40,
        level_shift=0.2,
        grid_radial=99,
        grid_angular=302,
    )
    kw.update(extra)
    return ferric.run_cdft(mol, _svp(), [ferric.CdftConstraint([1], 1.0)], **kw)


@pytest.mark.parametrize(
    "extra",
    [
        # A 1e-4 a.u. field changes H but not the overlap or the occupations,
        # so only the Hamiltonian check can refuse it.
        dict(external_field=(0.0, 0.0, 1e-4)),
        # RI-J vs exact J: same geometry and basis, different Coulomb operator.
        dict(df_j_aux="def2-universal-jkfit"),
    ],
    ids=["external_field", "df_j_aux"],
)
def test_coupling_refuses_states_solved_with_different_hamiltonians(extra):
    """Wu-Van Voorhis needs ONE shared Hamiltonian. The overlap and occupation
    checks cannot see the functional, the fitting, or the external potential.
    Fails if the hamiltonian_key comparison in cdft_coupling is removed: the
    call then returns a number instead of raising."""
    a, b = _he2_plus_states(3.0)
    ferric.cdft_coupling(a, b)  # same Hamiltonian: accepted
    b_other = _he2_plus_state_b_with(**extra)
    with pytest.raises(ValueError, match="different Hamiltonians"):
        ferric.cdft_coupling(a, b_other)


def test_coupling_refuses_a_spin_constrained_state():
    """The kernel applies one operator to both spins; a spin constraint acts
    with opposite signs, so the Wu-VV element would be wrong, not approximate."""
    s = _lih_plus_spin_045()
    with pytest.raises(ValueError, match="spin"):
        ferric.cdft_coupling(s, s)


def test_asymmetric_coupling_pairs_each_lambda_with_its_own_weight():
    """He2+ at 2.5 A with DIFFERENT targets (hole on atom 0 at N = 1.0, atom 1
    held at N = 1.2), so lambda_a != lambda_b and the two weight elements differ.

    Catches a binding that hands state A's lambda to state B's DiabaticState (or
    either state's W to the other): the kernel then matches the MISPAIRED
    reference instead of the correct one. The reachability assert first proves
    the two references are distinguishable on this pair; if it fails, this
    test cannot detect the mispairing and must be redesigned, not loosened.
    Not a Rust-pinned configuration: the N = 1.2 state is chosen only to break
    the symmetry, and no coupling magnitude is asserted.
    """
    mol = ferric.Molecule.from_xyz_string("2\nHe2+\nHe 0 0 0\nHe 0 0 2.5\n", 1, 2)
    kw = dict(
        lambda_tol=1e-2,
        max_outer=40,
        level_shift=0.2,
        grid_radial=99,
        grid_angular=302,
    )
    a, _ = _he2_plus_states(2.5)
    b = ferric.run_cdft(mol, _svp(), [ferric.CdftConstraint([1], 1.2)], **kw)
    h_ref, _ = _hab_reference(a, b)
    h_bad, _ = _hab_reference(a, b, mispair=True)
    print(
        f"asymmetric: lambdas {a.lambdas[0]:.6f} / {b.lambdas[0]:.6f}; "
        f"|H_ab| correct {abs(h_ref):.8f} vs mispaired {abs(h_bad):.8f}"
    )
    assert abs(abs(h_ref) - abs(h_bad)) > 1e-4, (
        "the mispaired and correct references coincide; this pair cannot "
        "detect a lambda/W swap"
    )
    c = ferric.cdft_coupling(a, b)
    assert abs(abs(c.h_ab) - abs(h_ref)) < 1e-6, (c.h_ab, h_ref, h_bad)
