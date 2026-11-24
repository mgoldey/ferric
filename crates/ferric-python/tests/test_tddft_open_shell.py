"""run_tddft on an open-shell molecule: the unrestricted (UHF/UKS) route.

Pins the binding's reference selection and result fields; the numbers
themselves are validated against PySCF in
crates/ferric-tddft/tests/validation_tddft_open_shell.rs.

Bars: a closed-shell molecule forced through reference="unrestricted" must
return the restricted singlet and triplet roots merged (the exactness anchor
of the Rust unit tests, here at the binding level). The bar 1e-6 Ha is the
SCF density convergence of the two reference SCFs (1e-8 default energy, roots
move linearly with the density error), far below the smallest gap between
distinct roots of water/STO-3G (> 1e-2 Ha).

Mutation check: making the binding report reference="restricted" on the
unrestricted route fails the first and third tests on their assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")

MOLS = Path(__file__).resolve().parents[3] / "testdata" / "molecules"


@pytest.fixture(scope="module")
def bases():
    return ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")


@pytest.fixture(scope="module")
def oh():
    return ferric.Molecule.from_xyz(str(MOLS / "oh.xyz"), multiplicity=2)


def test_open_shell_molecule_runs_unrestricted_and_reports_s2(oh, bases):
    r = ferric.run_tddft(oh, *bases, functional="PBE", n_roots=4)
    assert r.reference == "unrestricted"
    assert r.spin is None
    assert r.s2.shape == (4,)
    # A doublet UKS reference: <S^2> slightly above 3/4; roots at or above it.
    assert 0.75 <= r.s2_reference < 0.8
    assert np.all(r.s2 > 0.74)
    assert "<S^2>" in str(r)
    approx = {a["name"]: a["detail"] for a in r.certificate["approximations"]}
    assert "unrestricted reference" in approx["spin"]


def test_restricted_on_open_shell_and_triplet_on_unrestricted_are_refused(oh, bases):
    with pytest.raises(ValueError, match="closed-shell molecule"):
        ferric.run_tddft(oh, *bases, reference="restricted")
    with pytest.raises(ValueError, match="spin-conserving"):
        ferric.run_tddft(oh, *bases, spin="triplet")
    with pytest.raises(ValueError, match="not one of"):
        ferric.run_tddft(oh, *bases, reference="rohf")


def test_closed_shell_through_unrestricted_is_singlets_plus_triplets(bases):
    water = ferric.Molecule.from_xyz(str(MOLS / "water.xyz"))
    n = 4
    u = ferric.run_tddft(water, *bases, n_roots=2 * n, reference="unrestricted")
    s = ferric.run_tddft(water, *bases, n_roots=2 * n)
    t = ferric.run_tddft(water, *bases, n_roots=2 * n, spin="triplet")
    assert u.reference == "unrestricted" and s.reference == "restricted"
    merged = np.sort(np.concatenate([s.excitation_energies, t.excitation_energies]))
    np.testing.assert_allclose(u.excitation_energies, merged[: 2 * n], atol=1e-6)
    # Each unrestricted root is a pure singlet (0) or the M_S = 0 triplet (2).
    assert np.all(np.minimum(np.abs(u.s2), np.abs(u.s2 - 2.0)) < 1e-6)
