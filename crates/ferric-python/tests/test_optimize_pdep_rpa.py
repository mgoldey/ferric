"""`run_optimize` / `run_frequencies` with `method="pdep-rpa"`.

The route calls `ferric_rpa::optimize::optimize_geometry_rpa`, the driver of
the CLI's pdep-rpa `task = "optimize"`. `rpa_gradient` is "analytic" (the
RI-dRPA Z-vector gradient: full rank, dense chi0, Coulomb) or "fd"; unset it
resolves to analytic where `check_analytic_scope` accepts the configuration
and to finite differences otherwise, the reason recorded in the certificate.

Pinned here:
- the default is analytic (certificate `gradient` entry), an explicit "fd"
  is finite differences, an unset gradient with a truncating `trunc_thresh`
  falls back to finite differences and says why, an explicit "analytic" with
  a truncating `trunc_thresh` raises;
- analytic and finite-difference optimizations end at the same geometry and
  energy;
- the strict parsers and the kwargs a method does not read raise ValueError;
- `run_frequencies(method="pdep-rpa")` agrees between the two gradients.

Mutations measured to fail (each on an assertion, then restored):
- `run_optimize_pdep_rpa` forcing `RpaGradientMethod::FiniteDifference` where
  the default or "analytic" was asked fails `test_default_is_analytic`,
  `test_default_falls_back_to_fd_when_analytic_is_out_of_scope` and
  `test_explicit_analytic_with_truncation_raises` (the certificate names the
  finite-difference gradient / nothing raises);
- `RpaGradientMethod::resolve_default` always returning the analytic method
  fails the same fallback test (and the unit test
  `unset_gradient_defaults_to_analytic_and_falls_back_with_a_reason`);
- `rpa_gradient_config` ignoring `rpa_gradient` fails
  `test_explicit_fd_is_finite_difference`,
  `test_explicit_analytic_with_truncation_raises`,
  `test_analytic_and_fd_reach_the_same_minimum` and
  `test_frequencies_analytic_matches_fd`;
- `run_frequencies_pdep_rpa` forcing finite differences fails
  `test_frequencies_analytic_matches_fd` (certificate gradient entry).
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")

WATER = "3\ndistorted water\nO 0 0 0\nH 0 0.9 0.1\nH 0 -0.3 0.85\n"
AUX = "cc-pvdz-ri"
BASIS = "sto-3g"
KW = {"auxbasis": AUX, "n_quad": 8}


def _gradient_detail(result) -> str:
    details = {a["name"]: a["detail"] for a in result.certificate["approximations"]}
    return details["gradient"]


@pytest.fixture(scope="module")
def water():
    return ferric.Molecule.from_xyz_string(WATER)


@pytest.fixture(scope="module")
def default_run(water):
    return ferric.run_optimize(water, BASIS, method="pdep-rpa", max_steps=60, **KW)


def test_default_is_analytic(default_run):
    assert default_run.converged
    assert "analytic RHF + RI-dRPA" in _gradient_detail(default_run)
    assert "fell back" not in _gradient_detail(default_run)
    assert default_run.certificate["capability"] == "optimize:pdep-rpa"


def test_explicit_fd_is_finite_difference(water):
    r = ferric.run_optimize(
        water, BASIS, method="pdep-rpa", rpa_gradient="fd", max_steps=1, **KW
    )
    assert "finite difference" in _gradient_detail(r)


def test_default_falls_back_to_fd_when_analytic_is_out_of_scope(water):
    r = ferric.run_optimize(
        water, BASIS, method="pdep-rpa", trunc_thresh=1e-4, max_steps=1, **KW
    )
    detail = _gradient_detail(r)
    assert "finite difference" in detail
    assert "default fell back from the analytic gradient" in detail
    assert "trunc_thresh" in detail


def test_explicit_analytic_with_truncation_raises(water):
    with pytest.raises(Exception, match="trunc_thresh"):
        ferric.run_optimize(
            water,
            BASIS,
            method="pdep-rpa",
            rpa_gradient="analytic",
            trunc_thresh=1e-4,
            max_steps=1,
            **KW,
        )


def test_analytic_and_fd_reach_the_same_minimum(water, default_run):
    fd = ferric.run_optimize(
        water, BASIS, method="pdep-rpa", rpa_gradient="fd", max_steps=60, **KW
    )
    assert fd.converged
    assert "finite difference" in _gradient_detail(fd)
    assert "analytic" in _gradient_detail(default_run)
    assert abs(fd.energy - default_run.energy) < 2e-6
    a = default_run.mol().coords_bohr()
    b = fd.mol().coords_bohr()
    assert max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)) < 5e-3


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rpa_gradient": "numeric"}, "rpa_gradient"),
        ({"quadrature": "simpson"}, "quadrature"),
        ({"scf_jk": "rijk"}, "scf_jk"),
        ({"point_charges": [(0.1, 3.0, 0.0, 0.0)]}, "point_charges"),
        ({"xc": "pbe"}, "xc"),
        ({"c_os": 1.0}, "c_os"),
        ({"reference": "uhf"}, "closed-shell"),
    ],
)
def test_pdep_rpa_refuses(water, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ferric.run_optimize(
            water, BASIS, method="pdep-rpa", max_steps=1, **KW, **kwargs
        )


def test_pdep_rpa_needs_auxbasis(water):
    with pytest.raises(ValueError, match="auxbasis"):
        ferric.run_optimize(water, BASIS, method="pdep-rpa", max_steps=1)


@pytest.mark.parametrize(
    "kwargs", [{"rpa_gradient": "fd"}, {"n_quad": 8}, {"trunc_thresh": 0.0}]
)
def test_rpa_kwargs_are_refused_by_other_methods(water, kwargs):
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_optimize(water, BASIS, max_steps=1, **kwargs)


H2 = "2\nh2\nH 0 0 0\nH 0 0 0.78\n"


def test_frequencies_analytic_matches_fd():
    h2 = ferric.Molecule.from_xyz_string(H2)
    common = {"method": "pdep-rpa", "auxbasis": AUX, "n_quad": 8, "delta": 5e-3}
    a = ferric.run_frequencies(h2, "cc-pvdz", **common)
    f = ferric.run_frequencies(h2, "cc-pvdz", rpa_gradient="fd", **common)
    assert len(a.frequencies) == 1
    assert "analytic RHF + RI-dRPA" in _gradient_detail_freq(a)
    assert "finite difference" in _gradient_detail_freq(f)
    assert a.frequencies[0] == pytest.approx(f.frequencies[0], rel=2e-3)
    assert a.n_gradient_evaluations == 12


def _gradient_detail_freq(result) -> str:
    details = {a["name"]: a["detail"] for a in result.certificate["approximations"]}
    return details["gradient"]


def test_frequencies_pdep_rpa_refuses_scf_kwargs_and_scf_refuses_rpa_kwargs():
    h2 = ferric.Molecule.from_xyz_string(H2)
    with pytest.raises(ValueError, match="xc"):
        ferric.run_frequencies(h2, "cc-pvdz", method="pdep-rpa", auxbasis=AUX, xc="pbe")
    with pytest.raises(ValueError, match="auxbasis"):
        ferric.run_frequencies(h2, "cc-pvdz", method="pdep-rpa")
    with pytest.raises(ValueError, match="not read by"):
        ferric.run_frequencies(h2, "cc-pvdz", rpa_gradient="fd")
    with pytest.raises(ValueError, match="unknown method"):
        ferric.run_frequencies(h2, "cc-pvdz", method="mp2")
