"""run_rhf reports WHERE the SCF stopped: exit reason, final residuals, unmet criteria.

Issue #410: a 71-atom RHF in a 6458-charge field returned converged=False with no
way to tell an iteration-cap failure (density still draining) from a stall or a
noise-floor energy gate. These tests pin the diagnostics and the warning.
"""

from __future__ import annotations

import warnings

import ferric
import pytest


def _water():
    return ferric.Molecule.from_xyz("testdata/molecules/water.xyz", 0, 1)


def _bs():
    return ferric.BasisSet.bundled("sto-3g")


def test_converged_run_reports_met_criteria_and_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ferric.run_rhf(_water(), _bs())
    assert r.converged
    assert r.exit_reason == "Converged"
    assert r.unmet_criteria == []
    assert r.conv_thresholds == (1e-3, 1e-6)  # run_rhf defaults: energy_conv, density_conv
    assert r.final_dp_rms < 1e-6
    assert r.final_dp_max < 1e-5
    assert r.final_de < 1e-3
    assert r.final_diis_err_max >= 0.0


def test_iteration_capped_run_names_the_unmet_criterion_in_a_warning():
    with pytest.warns(RuntimeWarning, match="did not converge") as rec:
        r = ferric.run_rhf(_water(), _bs(), max_iter=3, density_conv=1e-12)
    assert not r.converged
    assert r.iterations == 3
    assert r.exit_reason == "MaxIter"
    assert "dp_rms" in r.unmet_criteria
    assert r.final_dp_rms > 1e-12
    assert r.conv_thresholds[1] == 1e-12
    text = str(rec[0].message)
    # iteration count, the unmet criterion and the residual-vs-threshold all appear
    assert "MaxIter" in text and "3 iterations" in text
    assert "unmet: dp_rms" in text
    assert "need < 1.0e-12" in text
    assert r.convergence_summary in text


def test_energy_gate_below_the_noise_floor_is_reported_as_the_unmet_one():
    """A converged-density run that fails only on dE says so: unmet == ['dE'] is
    reachable with an energy_conv no iteration can satisfy (0 < dE always)."""
    with pytest.warns(RuntimeWarning):
        r = ferric.run_rhf(_water(), _bs(), max_iter=40, energy_conv=0.0)
    assert not r.converged
    assert r.unmet_criteria == ["dE"]
    assert r.final_dp_rms < 1e-6
