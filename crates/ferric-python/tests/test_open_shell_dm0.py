"""`dm0=(D_alpha, D_beta)` on `run_uhf` / `run_rohf`, and the fock_hook noise floor.

Anchors:

* `dm0` equal to the converged `(D_alpha, D_beta)` of an identical run
  restarts at the solution: at most two iterations, and the same energy
  within the sum of both runs' `energy_conv`;
* `(D / 2, D / 2)` through `run_uhf` and `run_rohf` on a closed-shell
  molecule gives `run_rhf(dm0=D)`'s energy within the same bound, for a
  converged `D` and for one taken from an unconverged run;
* a zero hook leaves the noise-floor advisories unchanged (the floor value
  itself is checked bitwise in `ferric_scf::noise_floor`'s unit tests);
* `mom_reference="initial"` holds the occupation of a core-hole `dm0`
  (beta 1s population below 0.1, against above 0.9 when aufbau or MOM armed
  after iteration 1 refill it); its energy against an independent numpy MOM
  is checked in test_block_ports_set2.py.

The hook advisory fires for the mu = 1e7 projector on water/cc-pVDZ at
`energy_conv = 1e-9`: the hook term `eps * sum|D*V|` is ~9e-9 Ha there and
the floor without it ~2e-11 Ha.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}
EPS = np.finfo(float).eps


def _water(basis: str, charge: int = 0, multiplicity: int = 1):
    mol = ferric.Molecule.from_xyz_string(
        WATER, charge=charge, multiplicity=multiplicity
    )
    return mol, ferric.BasisSet.bundled(basis)


def _pair(r):
    return r.density_alpha(), r.density_beta()


RUNS = {"uhf": ferric.run_uhf, "rohf": ferric.run_rohf}


# ── dm0 anchors ──


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("charge, mult", [(1, 2), (2, 3)], ids=["doublet", "triplet"])
def test_converged_pair_restarts_at_the_solution(kind, charge, mult):
    run = RUNS[kind]
    mol, bs = _water("cc-pvdz", charge, mult)
    first = run(mol, bs, **TIGHT)
    again = run(mol, bs, dm0=_pair(first), **TIGHT)
    assert first.converged and again.converged
    assert first.iterations > 2
    assert again.iterations <= 2
    assert abs(again.energy - first.energy) < 2 * TIGHT["energy_conv"]


def _unconverged_rhf_density(mol, bs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = ferric.run_rhf(mol, bs, max_iter=2, guess="hcore", **TIGHT)
    assert not r.converged
    return r.density()


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("start", ["converged", "unconverged"])
def test_half_density_pair_equals_rhf_dm0(kind, start):
    mol, bs = _water("cc-pvdz")
    if start == "converged":
        d = ferric.run_rhf(mol, bs, **TIGHT).density()
    else:
        d = _unconverged_rhf_density(mol, bs)
    rhf = ferric.run_rhf(mol, bs, dm0=d, **TIGHT)
    open_shell = RUNS[kind](mol, bs, dm0=(d / 2, d / 2), **TIGHT)
    assert rhf.converged and open_shell.converged
    assert abs(open_shell.energy - rhf.energy) < 2 * TIGHT["energy_conv"]


def test_hooked_uhf_restarts_from_its_converged_pair():
    # dm0 with a hook: the natural-orbital start reproduces the hooked
    # density itself, so the hook's potential is in the first Fock matrix.
    mol, bs = _water("cc-pvdz", 1, 2)
    v = 0.01 * ferric.AoIntegrals(mol, bs).overlap()

    def hook(d_a, d_b):
        return float(np.sum((d_a + d_b) * v)), v, v

    first = ferric.run_uhf(mol, bs, fock_hook=hook, **TIGHT)
    again = ferric.run_uhf(mol, bs, fock_hook=hook, dm0=_pair(first), **TIGHT)
    assert first.iterations > 2
    assert again.iterations <= 2
    assert abs(again.energy - first.energy) < 2 * TIGHT["energy_conv"]


# ── dm0 refusals ──


@pytest.mark.parametrize("kind", sorted(RUNS))
def test_swapped_spins_are_rejected(kind):
    mol, bs = _water("sto-3g", 1, 2)
    d_a, d_b = _pair(RUNS[kind](mol, bs))
    with pytest.raises(ValueError, match=r"dm0\[0\] \(D_alpha\).*system has 5"):
        RUNS[kind](mol, bs, dm0=(d_b, d_a))


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda a, b: a + b, "must be a pair"),
        (lambda a, b: (a, b, b), "must be a pair"),
        (lambda a, b: (a[:-1, :-1], b), "shape"),
        (lambda a, b: (a, b + np.triu(np.full_like(b, 1e-6), 1)), "not symmetric"),
        (lambda a, b: (np.where(np.eye(len(a), dtype=bool), np.nan, a), b), "NaN"),
        (lambda a, b: (a, 1.0000001 * b), r"dm0\[1\] \(D_beta\).*electrons"),
    ],
    ids=["single-array", "triple", "shape", "asymmetric", "nan", "electron-count"],
)
@pytest.mark.parametrize("kind", sorted(RUNS))
def test_bad_dm0_pair_is_rejected(kind, make, match):
    mol, bs = _water("sto-3g", 1, 2)
    d_a, d_b = _pair(RUNS[kind](mol, bs))
    with pytest.raises(ValueError, match=match):
        RUNS[kind](mol, bs, dm0=make(d_a, d_b))


@pytest.mark.parametrize(
    "kind, kwargs",
    [
        ("uhf", {"guess": "hcore"}),
        ("uhf", {"guess_basis": "sto-3g"}),
        ("rohf", {"guess_basis": "sto-3g"}),
    ],
)
def test_dm0_with_another_guess_is_rejected(kind, kwargs):
    mol, bs = _water("6-31g", 1, 2)
    pair = _pair(RUNS[kind](mol, bs))
    with pytest.raises(ValueError, match="dm0 and"):
        RUNS[kind](mol, bs, dm0=pair, **kwargs)


# ── mom_reference="initial": the dm0 start as the MOM reference ──


def _core_hole_start(basis: str = "cc-pvdz"):
    """The water cation and its O 1s core-hole start: the neutral RHF
    orbitals with the beta 1s emptied. Returns (mol, bs, dm0, c_1s)."""
    neutral, bs = _water(basis)
    c = ferric.run_rhf(neutral, bs, **TIGHT).mo_coefficients()
    mol, _ = _water(basis, 1, 2)
    d_a = c[:, :5] @ c[:, :5].T
    d_b = c[:, 1:5] @ c[:, 1:5].T
    return mol, bs, (d_a, d_b), c[:, 0]


def _beta_1s_population(r, mol, bs, c_1s):
    s = ferric.AoIntegrals(mol, bs).overlap()
    return float(c_1s @ s @ r.density_beta() @ s @ c_1s)


def test_initial_mom_holds_the_core_hole_of_the_start():
    # MEASURED (2026-11-13): the core-hole cation at -56.1268 Ha in 15
    # iterations, beta population of the neutral's 1s 1.5e-5; MOM armed after
    # iteration 1 refills the hole: -75.6317 Ha, population 1 - 1.5e-6.
    mol, bs, dm0, c_1s = _core_hole_start()
    held = ferric.run_uhf(mol, bs, dm0=dm0, mom_reference="initial", **TIGHT)
    late = ferric.run_uhf(mol, bs, dm0=dm0, mom_after_iter=1, **TIGHT)
    assert held.converged and late.converged
    assert held.energy - late.energy > 19.0
    assert _beta_1s_population(held, mol, bs, c_1s) < 0.1
    assert _beta_1s_population(late, mol, bs, c_1s) > 0.9


def test_previous_reference_is_the_default():
    mol, bs, dm0, _ = _core_hole_start("sto-3g")
    default = ferric.run_uhf(mol, bs, dm0=dm0, **TIGHT)
    named = ferric.run_uhf(mol, bs, dm0=dm0, mom_reference="previous", **TIGHT)
    assert named.energy == default.energy
    assert named.iterations == default.iterations


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mom_reference": "start"}, "previous.*initial"),
        ({"mom_reference": "initial", "mom_after_iter": 3}, "mom_after_iter"),
        ({"mom_reference": "initial", "stability_descent": True}, "stability_descent"),
    ],
    ids=["unknown", "with-mom-after-iter", "with-descent"],
)
def test_mom_reference_refusals(kwargs, match):
    mol, bs = _water("sto-3g", 1, 2)
    with pytest.raises(ValueError, match=match):
        ferric.run_uhf(mol, bs, **kwargs)


# ── hook-aware noise floor ──


def _projector(mol, bs, mu):
    """V = mu S C C^T S over the two lowest core-Hamiltonian orbitals."""
    ints = ferric.AoIntegrals(mol, bs)
    s = ints.overlap()
    w, u = np.linalg.eigh(s)
    x = u / np.sqrt(w)
    _, c = np.linalg.eigh(x.T @ ints.hcore() @ x)
    c = x @ c[:, :2]
    v = mu * (s @ c @ c.T @ s)
    return 0.5 * (v + v.T)


def _messages(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = fn()
    return r, [str(w.message) for w in caught if w.category is RuntimeWarning]


def test_zero_hook_leaves_the_floor_advisories_unchanged():
    # energy_conv = 0 is below any floor (dE can reach 0 exactly, so a tiny
    # positive value can still be met): the pre-SCF advisory fires and the
    # exit report blames the floor, with and without a hook.
    mol, bs = _water("sto-3g")
    kw = {"energy_conv": 0.0, "density_conv": 1e-6, "max_iter": 15}

    def zero(d):
        return 0.0, np.zeros_like(d)

    plain, m_plain = _messages(lambda: ferric.run_rhf(mol, bs, **kw))
    hooked, m_hooked = _messages(lambda: ferric.run_rhf(mol, bs, fock_hook=zero, **kw))
    assert any("noise floor" in m for m in m_plain), m_plain
    assert "noise floor" in plain.convergence_summary
    assert m_hooked == m_plain
    assert hooked.convergence_summary == plain.convergence_summary


@pytest.mark.parametrize("who", ["run_rhf", "run_uhf", "run_rohf"])
def test_projector_hook_term_raises_the_floor_advisory(who):
    mu, ec = 1e7, 1e-9
    if who == "run_rhf":
        mol, bs = _water("cc-pvdz", 2)
        v = _projector(mol, bs, mu)

        def hook(d):
            return float(np.sum(d * v)), v

        def run(h):
            return ferric.run_rhf(mol, bs, fock_hook=h, energy_conv=ec, max_iter=40)
    else:
        mol, bs = _water("cc-pvdz", 3, 2)
        v = _projector(mol, bs, mu)

        def hook(d_a, d_b):
            return float(np.sum((d_a + d_b) * v)), v, v

        def run(h):
            return getattr(ferric, who)(mol, bs, fock_hook=h, energy_conv=ec)

    _, msgs = _messages(lambda: run(hook))
    hits = [m for m in msgs if "fock_hook contributes" in m]
    assert len(hits) == 1, msgs
    assert hits[0].startswith(
        f"{who}: energy_conv 1.0e-9 is below the estimated noise floor"
    )
    # The pre-SCF advisory, which has no hook term, stays silent.
    assert not [m for m in msgs if "noise floor" in m and m not in hits]
    zero = (
        (lambda d: (0.0, 0 * v))
        if who == "run_rhf"
        else (lambda a, b: (0.0, 0 * v, 0 * v))
    )
    _, msgs = _messages(lambda: run(zero))
    assert not [m for m in msgs if "noise floor" in m], msgs
