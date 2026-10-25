"""The smooth neighbour switch (issue #495) through the Python KS bindings.

`neighbour_switch` W (Angstrom, default 0.0 = the hard cutoff) maps to
`ferric_pbc::dft::PeriodicGridConfig::neighbour_switch` (Bohr) on every
binding that takes a periodic XC grid: run_rks_gamma, run_uks_gamma,
run_roks_gamma, run_rks_kpts, run_uks_kpts, run_roks_kpts. The library's
force / stress / scan validation is crates/ferric-pbc/tests/
validation_pbc_grid_switch*.rs; this file checks the BINDING:

* W reaches the library, with the Angstrom -> Bohr conversion: with the
  automatic cutoff the library grows D by W (D = max(10 Bohr, bound) + W), so
  an automatic-cutoff run equals the run at the explicit cutoff 10 Bohr + W,
  energy to 1e-12 and grid-point count exactly. A W left in Angstrom (a 1.89x
  unit slip) moves the automatic D to 10 Bohr + 1.06 while the explicit run
  stays at 12 Bohr, so the energies split; a dropped W leaves the automatic
  run at the hard 10 Bohr (asserted different from the switched one).
* The switch changes the energy by the library's measured amount: at
  D = 10 Bohr, W = 2 Bohr on the #504 H2 cell (20, 50) grid, Gamma,
  E(W) - E(hard) is the library's +7.236e-6 Ha
  (`energy_shift_of_the_switch_against_cutoff_convergence`: +1.112e-5 and
  +3.884e-6 against hard D = 20), and lies between the hard energies at D - W
  and D (the #504 bracket).
* Gamma == 1x1x1 k mesh with the switch on (RKS, UKS, ROKS): two library
  drivers fed the same converted W.
* Omitting the kwarg is bitwise the explicit 0.0.
* Refusals: negative, NaN and infinite W raise ValueError on all six bindings
  before any SCF.

Not covered here: a bit-for-bit comparison with a direct library call. The
extension is a cdylib (no Rust harness in this crate) and an energy pinned
across machines is not bitwise, so the binding is compared with the library
through the shift above (to 2e-8) and the cutoff identity (to 1e-12).
"""

from __future__ import annotations

import math

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import (  # noqa: E402,F401  (fixtures re-exported)
    BOHR_IN_ANGSTROM,
    H2_A,
    H2_ATOMS,
    OMEGA,
    _lat,
    _mol,
    basis,
)

H3_ATOMS = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (0.3, 1.9, 0.4)]
D_BOHR = 10.0
W_BOHR = 2.0
D = D_BOHR * BOHR_IN_ANGSTROM
W = W_BOHR * BOHR_IN_ANGSTROM
GRID = dict(n_radial=20, n_angular=50)
# E(D = 10, W = 2) - E(hard, D = 10) at Gamma, H2 / STO-3G cubic 4 Bohr,
# (20, 50), exxdiv none, omega 0.8 Bohr^-1: the library's
# `energy_shift_of_the_switch_against_cutoff_convergence` measured
# +1.112e-5 (switched) and +3.884e-6 (hard) against hard D = 20, i.e.
# +7.236e-6 (each printed to 4 digits, so +-1e-9). SCF settings differ between that test and the binding by
# convergence noise only (< 1e-10 Ha).
LIB_SHIFT = 7.236e-6
SHIFT_TOL = 2e-8

BINDINGS = (
    "run_rks_gamma",
    "run_uks_gamma",
    "run_roks_gamma",
    "run_rks_kpts",
    "run_uks_kpts",
    "run_roks_kpts",
)


def _run(name, basis, cutoff, switch=None, **kw):
    """One binding on H2 (closed shell) or the H3 doublet (open shell)."""
    open_shell = name not in ("run_rks_gamma", "run_rks_kpts")
    mol = _mol(H3_ATOMS, multiplicity=2) if open_shell else _mol(H2_ATOMS)
    args = [mol, _lat(H2_A), basis, "PBE"]
    if name.endswith("_kpts"):
        args.append(kw.pop("mesh", (1, 1, 1)))
        kw.setdefault("energy_conv", 1e-13)
        kw.setdefault("grad_conv", 1e-10)
    else:
        kw.setdefault("density_conv", 1e-11)
    if switch is not None:
        kw["neighbour_switch"] = switch
    return getattr(ferric, name)(
        *args, exxdiv="none", omega=OMEGA, neighbour_cutoff=cutoff, **GRID, **kw
    )


@pytest.mark.parametrize("name", BINDINGS)
def test_switch_reaches_the_library_cutoff_in_angstrom(basis, name):
    """Automatic cutoff with W == explicit 10 Bohr + W (the library adds W)."""
    auto = _run(name, basis, None, W)
    explicit = _run(name, basis, D + W, W)
    hard_auto = _run(name, basis, None)
    assert auto.converged and explicit.converged and hard_auto.converged
    assert auto.n_grid_points == explicit.n_grid_points, name
    assert abs(auto.energy - explicit.energy) < 1e-12, (
        name,
        auto.energy,
        explicit.energy,
    )
    # The switch did something. (The grid-point count does not tell: the
    # (20, 50) radial grid has no shell between 10 and 12 Bohr here, measured.)
    assert abs(auto.energy - hard_auto.energy) > 1e-7, name
    if name == "run_rks_gamma":
        assert math.isclose(auto.neighbour_cutoff, D + W, rel_tol=0, abs_tol=1e-12)


def test_gamma_energy_shift_matches_the_library(basis):
    """E(W) - E(hard) at fixed D equals the library's measured shift and lies
    between the hard energies at D - W and D."""
    hard = _run("run_rks_gamma", basis, D).energy
    switched = _run("run_rks_gamma", basis, D, W).energy
    hard_lo = _run("run_rks_gamma", basis, D - W).energy
    shift = switched - hard
    print(
        f"Gamma RKS PBE: E(W=2) - E(hard), D = 10 Bohr: {shift:+.4e} (library {LIB_SHIFT:+.4e})"
    )
    assert abs(shift - LIB_SHIFT) < SHIFT_TOL, shift
    assert min(hard, hard_lo) <= switched <= max(hard, hard_lo), (
        hard_lo,
        switched,
        hard,
    )


@pytest.mark.parametrize(
    "gamma, kpts",
    [
        ("run_rks_gamma", "run_rks_kpts"),
        ("run_uks_gamma", "run_uks_kpts"),
        ("run_roks_gamma", "run_roks_kpts"),
    ],
)
def test_gamma_equals_one_point_mesh_with_the_switch(basis, gamma, kpts):
    g = _run(gamma, basis, D, W)
    k = _run(kpts, basis, D, W, mesh=(1, 1, 1))
    hard = _run(gamma, basis, D)
    assert g.converged and k.converged
    assert abs(g.energy - k.energy) < 1e-10, (gamma, g.energy, k.energy)
    assert abs(g.energy - hard.energy) > 1e-7, "the switch did not reach the Gamma grid"


@pytest.mark.parametrize("name", ["run_rks_gamma", "run_uks_kpts"])
def test_omitted_kwarg_is_bitwise_zero(basis, name):
    assert _run(name, basis, D).energy == _run(name, basis, D, 0.0).energy


@pytest.mark.parametrize("name", BINDINGS)
@pytest.mark.parametrize("bad", [-1e-3, float("nan"), float("inf"), -float("inf")])
def test_bad_switch_is_a_value_error(basis, name, bad):
    with pytest.raises(ValueError, match=r"neighbour_switch must be finite and >= 0"):
        _run(name, basis, D, bad)
