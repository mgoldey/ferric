"""AIMD energy/force consistency smoke test."""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric")


def _water():
    return ferric.Molecule.from_xyz_string(
        "3\n\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
    )


def test_run_aimd_conserves_energy_approximately():
    mol = _water()
    res = ferric.run_aimd(mol, "sto-3g", n_steps=5, dt_fs=0.5, thermostat="nve", seed=1)
    assert len(res.total_energies) == 6
    # Total energy should not drift wildly over 5 steps.
    drift = max(res.total_energies) - min(res.total_energies)
    assert drift < 1.0
