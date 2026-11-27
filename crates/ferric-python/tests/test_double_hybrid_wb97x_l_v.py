"""run_double_hybrid(kind="wb97x-l-v"): the binding of the CLI's
method.kind = "wb97x-l-v" (``ferric_cc::double_hybrid::run_wb97x_l_v_on``).

The functional itself is validated in ``crates/ferric-cc/tests``; these tests
pin what the binding adds: lambda_ / omega reach the driver, omitted ones are
the published values, the result's fields satisfy the decomposition the
docstring states, the certificate describes the wB97X-L-V reference, and the
keywords the route cannot honour raise instead of being dropped.
"""

from __future__ import annotations

import json

import pytest

ferric = pytest.importorskip("ferric")

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"


def _args(multiplicity=1, charge=0):
    return (
        ferric.Molecule.from_xyz_string(
            WATER, charge=charge, multiplicity=multiplicity
        ),
        ferric.BasisSet.bundled("sto-3g"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
    )


def _run(**kw):
    return ferric.run_double_hybrid(*_args(), kind="wb97x-l-v", **kw)


@pytest.fixture(scope="module")
def published():
    return _run()


def _approximation(result, name):
    found = [a for a in result.certificate["approximations"] if a["name"] == name]
    assert len(found) == 1, result.certificate["approximations"]
    return found[0]["detail"]


def test_omitted_lambda_and_omega_are_the_published_values(published):
    explicit = _run(lambda_=0.6, omega=0.1)
    assert published.c_os == 0.6
    assert explicit.total_energy == pytest.approx(published.total_energy, abs=1e-10)
    assert _approximation(published, "double-hybrid").startswith(
        "lambda = 0.6, omega = 0.1 Bohr^-1"
    )


def test_result_fields_decompose_the_total(published):
    r = published
    assert r.e_os < 0.0
    assert (r.e_ss, r.c_ss, r.e_dispersion) == (0.0, 0.0, None)
    assert r.reference == "restricted"
    assert r.e_corr_scaled == pytest.approx(r.c_os * r.e_os, abs=1e-14)
    assert r.total_energy == pytest.approx(r.e_ks + r.c_os * r.e_os, abs=1e-12)


def test_lambda_and_omega_reach_the_driver(published):
    off = _run(lambda_=0.0)
    # lambda scales the operator and the energy: at 0 nothing is added, and the
    # Kohn-Sham reference does not depend on it.
    assert off.total_energy == off.e_ks
    assert off.e_ks == pytest.approx(published.e_ks, abs=1e-9)
    # A larger omega shortens erfc(omega r)/r, so less correlation is recovered.
    short = _run(omega=0.4)
    assert short.e_ks == pytest.approx(published.e_ks, abs=1e-9)
    assert published.e_os < short.e_os < 0.0
    assert abs(short.e_os - published.e_os) > 1e-5
    assert _approximation(short, "double-hybrid").startswith(
        "lambda = 0.6, omega = 0.4 Bohr^-1"
    )


def test_certificate_describes_the_wb97x_l_v_reference(published):
    cert = published.certificate
    assert cert["capability"] == "wb97x-l-v"
    text = json.dumps(cert)
    assert "wB97X-L-V" in text
    # The MP2 double hybrids' functionals must not leak into this route.
    assert "B2PLYP" not in text.upper().replace("-", "")


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"dispersion": "d3bj"}, "dispersion"),
        ({"unrestricted": True}, "closed-shell only"),
        ({"lambda_": 1.5}, "lambda_ must lie in"),
        ({"omega": 0.0}, "omega must be a positive"),
    ],
)
def test_keywords_the_route_cannot_honour_raise(kw, needle):
    with pytest.raises(ValueError, match=needle):
        _run(**kw)


def test_an_open_shell_molecule_is_refused():
    with pytest.raises(ValueError, match="closed-shell only"):
        ferric.run_double_hybrid(*_args(multiplicity=2, charge=1), kind="wb97x-l-v")


@pytest.mark.parametrize("kw", [{"lambda_": 0.5}, {"omega": 0.2}])
def test_lambda_and_omega_are_refused_for_the_mp2_kinds(kw):
    with pytest.raises(ValueError, match="wb97x-l-v' only"):
        ferric.run_double_hybrid(*_args(), kind="b2plyp", **kw)
