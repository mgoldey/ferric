"""The ``df_guess`` keyword of ``run_rhf`` and ``run_dft``.

It runs ``ferric_scf::ladder``'s DF pre-stage (Psi4's ``DF_SCF_GUESS``): the
SCF converged loosely with RI-J/RI-K on ``def2-universal-jkfit``, then the
exact four-centre SCF from that density. ``None`` follows
``ladder::DF_GUESS_DEFAULT_ON`` wherever the SCF has a four-centre build to
save and no starting point of its own (``DfGuessPlan::decide``, shared with the
CLI ``rhf`` ladder); ``False`` never runs; ``True`` where it cannot apply
raises ``ValueError`` naming why. ``result.df_guess`` reports the decision.
The Rust side of the rule is ``crates/ferric-scf/tests/df_guess_plan.rs``.
"""

from __future__ import annotations

import ferric
import numpy as np
import pytest

TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-8}
EXACT = {"df_j_aux": "", "df_k_aux": "", "eri": "direct"}
WATER = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz_string(WATER)


@pytest.fixture(scope="module")
def dz():
    return ferric.BasisSet.bundled("cc-pvdz")


@pytest.fixture(scope="module")
def plain(water, dz):
    return ferric.run_rhf(water, dz, df_guess=False, **EXACT, **TIGHT)


def test_true_runs_the_pre_stage_at_the_same_energy_in_fewer_exact_builds(water, dz, plain):
    on = ferric.run_rhf(water, dz, df_guess=True, **EXACT, **TIGHT)
    assert plain.converged and on.converged
    assert on.df_guess["ran"] and on.df_guess["df_iterations"] > 0
    assert on.df_guess["aux"] == "def2-universal-jkfit"
    assert not plain.df_guess["ran"]
    assert plain.df_guess["df_iterations"] == 0
    # Same fixed point to the convergence level; only the start differs.
    assert abs(on.energy - plain.energy) < 1e-9
    assert on.iterations < plain.iterations, (on.iterations, plain.iterations)


def test_unset_follows_the_default(water, dz, plain):
    unset = ferric.run_rhf(water, dz, **EXACT, **TIGHT)
    assert unset.df_guess["ran"] is True
    assert unset.df_guess["reason"] is None
    assert abs(unset.energy - plain.energy) < 1e-9


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"guess": "hcore"}, "core-Hamiltonian"),
        ({"guess_basis": "sto-3g"}, "secondary-basis"),
        ({"mom_after_iter": 3}, "MOM"),
        ({"df_j_aux": "def2-universal-jkfit", "df_k_aux": "def2-universal-jkfit"}, "no four-centre"),
        ({"fock_hook": lambda d: (0.0, np.zeros_like(d))}, "fock_hook"),
        ({"with_gradient": True}, "with_gradient"),
    ],
)
def test_each_gate_keeps_unset_off_and_refuses_true(water, dz, kwargs, needle):
    kw = {**EXACT, **TIGHT, **kwargs}
    r = ferric.run_rhf(water, dz, **kw)
    if isinstance(r, tuple):  # with_gradient=True returns (result, gradient)
        r = r[0]
    assert r.df_guess["ran"] is False
    assert needle in r.df_guess["reason"], r.df_guess
    with pytest.raises(ValueError, match=needle):
        ferric.run_rhf(water, dz, df_guess=True, **kw)


def test_dm0_keeps_unset_off(water, dz, plain):
    r = ferric.run_rhf(water, dz, dm0=np.asarray(plain.density()), **EXACT, **TIGHT)
    assert r.df_guess["ran"] is False and "starting density" in r.df_guess["reason"]
    with pytest.raises(ValueError, match="starting density"):
        ferric.run_rhf(water, dz, dm0=np.asarray(plain.density()), df_guess=True, **EXACT, **TIGHT)


def test_symmetry_refuses_true(water, dz):
    with pytest.raises(ValueError, match="symmetry"):
        ferric.run_rhf(water, dz, symmetry=True, df_guess=True, **EXACT, **TIGHT)


def test_run_dft_hybrid_exact_k(water, dz):
    kw = {"functional": "PBE0", **EXACT, **TIGHT}
    off = ferric.run_dft(water, dz, df_guess=False, **kw)
    on = ferric.run_dft(water, dz, df_guess=True, **kw)
    assert on.df_guess["ran"] and not off.df_guess["ran"]
    assert abs(on.energy - off.energy) < 1e-9
    assert on.iterations < off.iterations, (on.iterations, off.iterations)


def test_run_dft_default_fits_jk_so_unset_stays_off(water, dz):
    r = ferric.run_dft(water, dz, functional="PBE0", **TIGHT)
    assert r.df_guess["ran"] is False and "no four-centre" in r.df_guess["reason"]
    with pytest.raises(ValueError, match="no four-centre"):
        ferric.run_dft(water, dz, functional="PBE0", df_guess=True, **TIGHT)
