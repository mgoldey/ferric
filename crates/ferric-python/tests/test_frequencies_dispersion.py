"""`run_frequencies(dispersion=...)`: the dispersion-corrected Hessian.

The Hessian itself (D3(BJ) against second differences of the D3 energy,
MBD@rsSCS against second differences of the full SCF + MBD energy) is
validated in `crates/ferric-cli/tests/dispersion_frequencies.rs`. These tests
pin the Python surface: the strict parsing shared with `run_dft`, the
refusals, and that the reported energy is the corrected total.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="needs the compiled extension")


def _h2():
    return ferric.Molecule.from_xyz("testdata/molecules/h2.xyz", 0, 1)


def test_d3_frequencies_report_the_corrected_energy():
    mol = _h2()
    plain = ferric.run_frequencies(mol, "sto-3g", xc="PBE")
    d3 = ferric.run_frequencies(mol, "sto-3g", xc="PBE", dispersion="d3bj")
    e_d3 = ferric.d3bj_energy(mol, "pbe")
    assert plain.e_dispersion is None
    assert d3.e_dispersion == pytest.approx(e_d3, abs=1e-14)
    assert d3.energy == pytest.approx(plain.energy + e_d3, abs=1e-12)
    assert d3.hessian_source == "finite-difference"
    # The dispersion Hessian reaches the frequency: the stretch moves.
    assert d3.frequencies[0] != plain.frequencies[0]


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"dispersion": "d3bj"}, "requires xc"),
        ({"xc": "PBE", "dispersion": "d4"}, "unknown dispersion"),
        ({"xc": "PBE", "dispersion": "d3bj", "reference": "uhf"}, "closed-shell"),
        ({"xc": "PBE", "dispersion": "d3bj", "hessian": "analytic"}, "analytic"),
    ],
)
def test_unsupported_dispersion_requests_raise(kwargs, fragment):
    with pytest.raises(Exception, match=fragment):
        ferric.run_frequencies(_h2(), "sto-3g", **kwargs)
