"""`scf_warm_start` on run_saddle, run_irc and run_frequencies.

The default (True) starts each SCF from converged orbitals nearby: the
previous step's (saddle search, IRC) or, for a finite-difference Hessian, the
undisplaced reference geometry's. False re-guesses every SCF. Warm and cold
runs must agree within each driver's OWN tolerance, not bitwise:

* saddle: gradient-convergence-limited, the same bars the analytic-vs-FD
  saddle test uses (energy 1e-6 Ha, coordinates 5e-3 Bohr, lowest eigenvalue
  1% relative);
* IRC: both branches converge to the same two minima (endpoint energies within
  1e-6 Ha, the optimizer's e_conv, mirror-image pyramidalisation);
* frequencies: within the SCF-noise bound derived and asserted per Hessian
  element in the Rust unit test
  `frequencies::tests::warm_fd_hessian_matches_cold_within_the_scf_noise_bound`;
  here only that the kwarg reaches the FD path, plus a smoke bar of 0.5 cm^-1
  on the wavenumbers (the same system measured 0.05 cm^-1 in that Rust test;
  this bar is a smoke check, not the derived bound).
"""

from __future__ import annotations

import math

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

BOHR = 0.529177210903


def _near_planar_ammonia():
    return ferric.Molecule.from_xyz_string(
        "4\nnh3 near planar\n"
        "N  0.0000  0.0000  0.0000\n"
        "H  0.0000  1.0100  0.1000\n"
        "H  0.8747 -0.5050  0.1000\n"
        "H -0.8747 -0.5050  0.1000\n"
    )


def _xyz(res, title):
    return "{}\n{}\n{}\n".format(
        len(res.symbols),
        title,
        "\n".join(
            f"{s} {x:.10f} {y:.10f} {z:.10f}"
            for s, (x, y, z) in zip(res.symbols, res.coords)
        ),
    )


def _max_dx_bohr(a, b):
    return max(abs(p - q) for ca, cb in zip(a, b) for p, q in zip(ca, cb)) / BOHR


def test_saddle_warm_start_matches_cold(capfd):
    warm = ferric.run_saddle(_near_planar_ammonia(), "sto-3g", max_steps=40)
    out_warm = capfd.readouterr().out
    cold = ferric.run_saddle(
        _near_planar_ammonia(), "sto-3g", max_steps=40, scf_warm_start=False
    )
    out_cold = capfd.readouterr().out
    assert warm.is_transition_state() and cold.is_transition_state()
    assert warm.energy == pytest.approx(cold.energy, abs=1e-6)
    assert _max_dx_bohr(warm.coords, cold.coords) < 5e-3
    assert warm.lowest_eigenvalue == pytest.approx(cold.lowest_eigenvalue, rel=1e-2)
    # The kwarg reaches the Hessian callback: only the warm run's FD Hessians
    # report warm starts (printed by the Rust driver when the knob is on).
    assert "SCF warm start" in out_warm
    assert "SCF warm start" not in out_cold


def test_irc_warm_start_matches_cold():
    sad = ferric.run_saddle(_near_planar_ammonia(), "sto-3g", max_steps=40)
    assert sad.is_transition_state()
    ts = ferric.Molecule.from_xyz_string(_xyz(sad, "saddle"), 0, 1)
    run = lambda warm: ferric.run_irc(  # noqa: E731
        ts, "sto-3g", sad.imaginary_mode, step=0.15, max_steps=120, scf_warm_start=warm
    )
    warm, cold = run(True), run(False)
    assert warm.both_converged() and cold.both_converged()
    # The saddle point itself is the first SCF of both walks (cold in both).
    assert warm.saddle_energy == cold.saddle_energy
    for bw, bc in ((warm.forward, cold.forward), (warm.reverse, cold.reverse)):
        assert abs(bw.energy - bc.energy) < 1e-6

    def pyramidalisation(branch):
        h_mean = sum(c[2] for c in branch.coords[1:]) / 3
        return branch.coords[0][2] - h_mean

    for r in (warm, cold):
        pf, pr = pyramidalisation(r.forward), pyramidalisation(r.reverse)
        assert pf * pr < 0 and abs(pf) > 0.05 and abs(pr) > 0.05
    assert math.copysign(1, pyramidalisation(warm.forward)) == math.copysign(
        1, pyramidalisation(cold.forward)
    )
    # Not a vacuum test: the warm walk ran different SCFs (bitwise-different
    # endpoint energies), i.e. the kwarg reached the callback.
    assert (warm.forward.energy, warm.reverse.energy) != (
        cold.forward.energy,
        cold.reverse.energy,
    )


def test_fd_frequencies_warm_start_matches_cold(capfd):
    water = ferric.Molecule.from_xyz_string(
        "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n", 0, 1
    )
    warm = ferric.run_frequencies(water, "6-31g", hessian="fd")
    out_warm = capfd.readouterr().out
    cold = ferric.run_frequencies(water, "6-31g", hessian="fd", scf_warm_start=False)
    out_cold = capfd.readouterr().out
    assert "18 of 19 evaluations warm-started" in out_warm
    assert "SCF warm start" not in out_cold
    assert warm.energy == cold.energy  # the reference SCF is cold in both
    assert max(abs(a - b) for a, b in zip(warm.frequencies, cold.frequencies)) < 0.5
