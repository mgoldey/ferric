"""A user-set energy_conv below the estimated noise floor of dE draws one warning.

Issue #410: with E = -1936 Ha the energy change of a settled SCF jitters at ~1e-9 Ha,
so energy_conv = 1e-10 is met only by chance (12-27 extra iterations). The maintainer's
decision is to WARN, not to alter when the SCF stops. These tests pin (a) the warning
fires for that class of setting and not for ordinary ones, (b) it is a RuntimeWarning
that never changes control flow or the numbers, and (c) the failure message repeats it.

The floor model and its measured anchors are unit-tested in
crates/ferric-scf/src/noise_floor.rs.
"""

from __future__ import annotations

import warnings

import ferric
import pytest

FLOOR_TEXT = "noise floor"


def _mol(name: str, mult: int = 1):
    return ferric.Molecule.from_xyz(f"testdata/molecules/{name}.xyz", 0, mult)


def _bs(name: str = "sto-3g"):
    return ferric.BasisSet.bundled(name)


def _floor_warnings(fn, *args, **kwargs):
    """Run fn; return (result_or_exception, [floor-warning messages])."""
    out = None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            out = fn(*args, **kwargs)
        except Exception as e:  # open-shell solvers raise on non-convergence
            out = e
    # The pre-SCF advisory only: the non-convergence warning of a capped run may
    # also quote the floor (see the summary tests below).
    return out, [
        str(w.message)
        for w in rec
        if FLOOR_TEXT in str(w.message) and "did not converge" not in str(w.message)
    ]


def test_default_energy_conv_never_warns_even_under_error_filter():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ferric.run_rhf(_mol("alkane_10"), _bs())
    assert r.converged


@pytest.mark.parametrize("ec", [1e-6, 1e-8, 1e-10])
def test_common_values_on_a_small_system_do_not_warn(ec):
    _, w = _floor_warnings(ferric.run_rhf, _mol("water"), _bs(), energy_conv=ec)
    assert w == []


def test_reporter_class_setting_warns_once_before_the_scf():
    # alkane C20 / STO-3G: 142 functions, conventional closed-shell J/K. Its dE
    # floor is ~4e-10 Ha, so 1e-10 is met only by chance. max_iter=1 keeps the run
    # short: the warning is issued BEFORE the SCF, not after.
    r, w = _floor_warnings(
        ferric.run_rhf,
        _mol("alkane_20"),
        _bs(),
        energy_conv=1e-10,
        density_conv=1e-8,
        max_iter=1,
    )
    assert len(w) == 1, w
    assert "run_rhf" in w[0] and "energy_conv 1.0e-10" in w[0]
    assert "142 basis functions" in w[0]
    assert "density_conv" in w[0]  # tells the user what actually gates the SCF
    assert not r.converged


def test_ordinary_values_on_the_same_system_do_not_warn():
    for ec in (1e-3, 1e-6, 1e-8):
        _, w = _floor_warnings(
            ferric.run_rhf, _mol("alkane_20"), _bs(), energy_conv=ec, max_iter=1
        )
        assert w == [], (ec, w)


def test_density_fitted_coulomb_has_only_the_rounding_floor_so_no_warning():
    # Same system and setting that warns above: with RI-J the dE noise is rounding
    # level (~1e-12 Ha), so 1e-10 is reachable.
    _, w = _floor_warnings(
        ferric.run_rhf,
        _mol("alkane_20"),
        _bs(),
        energy_conv=1e-10,
        df_j_aux="def2-universal-jkfit",
        max_iter=1,
    )
    assert w == []


def test_integral_thresh_enters_the_estimate():
    # Tightening integral_thresh lowers the screening floor (it is linear in it):
    # the setting that warns at the default thresh is reachable at 1e-14.
    _, w_default = _floor_warnings(
        ferric.run_rhf, _mol("alkane_20"), _bs(), energy_conv=1e-10, max_iter=1
    )
    _, w_tight = _floor_warnings(
        ferric.run_rhf,
        _mol("alkane_20"),
        _bs(),
        energy_conv=1e-10,
        integral_thresh=1e-14,
        max_iter=1,
    )
    assert len(w_default) == 1 and w_tight == []


@pytest.mark.parametrize("fn_name,mult", [("run_uhf", 2), ("run_rohf", 2)])
def test_open_shell_solvers_warn_when_below_their_rounding_floor(fn_name, mult):
    fn = getattr(ferric, fn_name)
    _, w = _floor_warnings(fn, _mol("oh", mult), _bs(), energy_conv=1e-16, max_iter=3)
    assert len(w) == 1 and fn_name in w[0], w
    # ... and an ordinary tight value is fine for them.
    _, w = _floor_warnings(fn, _mol("oh", mult), _bs(), energy_conv=1e-10, max_iter=3)
    assert w == []


def test_warning_changes_neither_the_energy_nor_the_iteration_count():
    """The advisory is read-only: the same call with the warning suppressed, raised
    to an error (the warn failure is ignored on purpose) and recorded gives the same
    energy to the last bit and the same iteration count."""
    kw = dict(energy_conv=1e-12, density_conv=1e-8, max_iter=12)
    runs = []
    for action in ("ignore", "error", "always"):
        with warnings.catch_warnings():
            warnings.simplefilter(action)
            r = ferric.run_rhf(_mol("benzene"), _bs(), **kw)
        runs.append((repr(r.energy), r.iterations, r.converged))
    assert runs[0] == runs[1] == runs[2], runs


def test_unmet_energy_gate_below_the_floor_is_explained_in_the_summary():
    with pytest.warns(RuntimeWarning):
        r = ferric.run_rhf(_mol("benzene"), _bs(), max_iter=40, energy_conv=1e-13)
    assert r.unmet_criteria == ["dE"]
    assert FLOOR_TEXT in r.convergence_summary
    assert "energy_conv 1.0e-13" in r.convergence_summary


def test_summary_does_not_blame_the_floor_when_the_density_is_the_problem():
    with pytest.warns(RuntimeWarning):
        r = ferric.run_rhf(
            _mol("water"), _bs(), max_iter=2, density_conv=1e-12, energy_conv=1e-3
        )
    assert "dp_rms" in r.unmet_criteria
    assert FLOOR_TEXT not in r.convergence_summary
