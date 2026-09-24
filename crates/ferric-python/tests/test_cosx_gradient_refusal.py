"""`run_dft(k_builder="cosx", with_gradient=True)` must refuse, not mismatch.

`ks_gradient_closed` builds its exchange term from exact four-centre
derivative integrals. After a COSX SCF the returned gradient is therefore not
the derivative of the returned energy. MEASURED on water (central FD of the
RHF energy along one H z, 1e-3 Angstrom): COSX minus exact-K = -8.9e-6 Ha/Bohr
at STO-3G and -1.4e-5 at cc-pVDZ. The CLI refuses `k_builder = "cosx"` on
gradient tasks for the same reason, and the published SCF docs list COSX as
"no gradients".

HOW THIS FAILS IF THE FIX IS REVERTED: `pytest.raises(ValueError)` does not
raise -- the call runs a COSX B3LYP SCF and returns an exact-exchange gradient.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

WATER = str(
    Path(__file__).resolve().parents[3] / "testdata" / "molecules" / "water.xyz"
)


@pytest.fixture(scope="module")
def setup():
    return ferric.Molecule.from_xyz(WATER), ferric.BasisSet.bundled("sto-3g")


def test_cosx_with_a_gradient_is_refused(setup):
    mol, bs = setup
    with pytest.raises(ValueError, match="cosx"):
        ferric.run_dft(mol, bs, "B3LYP", k_builder="cosx", with_gradient=True)


def test_cosx_energy_and_exact_exchange_gradient_still_run(setup):
    """Reachability anchor: only the COMBINATION is refused."""
    mol, bs = setup
    e = ferric.run_dft(mol, bs, "B3LYP", k_builder="cosx").total_energy
    assert e < 0.0
    r = ferric.run_dft(mol, bs, "B3LYP", with_gradient=True)
    assert r.gradient() is not None
