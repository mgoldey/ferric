"""`scf_warm_start` on the geometry drivers (`[optimize] scf_warm_start`).

The default (True) starts each geometry's SCF from the previous geometry's
converged orbitals; False re-guesses every geometry. The two must reach the
same optimization within the optimizer's own tolerances: the same number of
steps and final energies within `e_conv` (1e-6 Ha, the run_optimize default).
The first evaluation is cold in both (nothing to carry yet), so it is
bit-identical. Later evaluations are NOT compared one by one: after the first
step the two runs sit at geometries that differ by the SCF-level drift of the
previous gradients (~1e-6 Bohr), and on a steep early surface that alone moves
the energy by ~1e-8 Ha (measured 2.1e-8 on NH2). The per-geometry SCF bound is
derived and asserted in the Rust unit tests of `ferric_scf::warm_start`.
"""

from __future__ import annotations

import ferric

WATER = ferric.Molecule.from_xyz_string(
    "3\ndistorted water\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n", 0, 1
)
NH2 = ferric.Molecule.from_xyz_string(
    "3\nNH2 radical\nN 0 0 0\nH 0 0.95 0.2\nH 0 -0.6 0.85\n", 0, 2
)
E_CONV = 1e-6  # run_optimize's default optimizer energy criterion


def _agree(warm, cold):
    assert warm.converged and cold.converged
    assert warm.steps == cold.steps
    assert warm.energy_trace[0] == cold.energy_trace[0]
    assert abs(warm.energy - cold.energy) < E_CONV


def test_rhf_warm_start_matches_cold(capfd):
    warm = ferric.run_optimize(WATER, "sto-3g")
    out_warm = capfd.readouterr().out
    cold = ferric.run_optimize(WATER, "sto-3g", scf_warm_start=False)
    out_cold = capfd.readouterr().out
    _agree(warm, cold)
    # The keyword reaches the driver: only the warm run reports warm starts
    # (the summary line is printed by the Rust driver when the knob is on).
    assert f"{warm.steps} of {warm.steps + 1} evaluations warm-started" in out_warm
    assert "SCF warm start" not in out_cold
    assert (
        warm.energy == ferric.run_optimize(WATER, "sto-3g", scf_warm_start=True).energy
    )


def test_uhf_warm_start_matches_cold():
    warm = ferric.run_optimize_uhf(NH2, "sto-3g")
    cold = ferric.run_optimize_uhf(NH2, "sto-3g", scf_warm_start=False)
    _agree(warm, cold)


def test_qmmm_optimizer_takes_the_keyword():
    sys = ferric.QmmmSystem(
        ["O", "H", "H"],
        [(0.0, 0.0, 0.0), (0.0, 0.80, 0.60), (0.0, -0.80, 0.60)],
        [0.0, 0.0, 0.0],
        qm_indices=[0, 1, 2],
    )
    warm = ferric.run_optimize_qmmm(sys, "sto-3g", max_steps=3)
    cold = ferric.run_optimize_qmmm(sys, "sto-3g", max_steps=3, scf_warm_start=False)
    assert warm.steps == cold.steps
    assert abs(warm.energy - cold.energy) < E_CONV
