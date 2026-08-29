"""Cross-module unit-convention consistency.

`tools/campaign/fit.py` converts ligand Ångström coordinates to Bohr in order to
compare distances against pocket charges, which `tools/active_site` already
generated in Bohr. If those two conversions ever disagree, ligand and pocket end
up in different unit systems and every pose falls outside the field cutoff.

That failure is loud rather than silently wrong (the pose is reported
UNEVALUATED, not scored as zero — see `test_strain_and_fit.py`), but it is
confusing to debug from the symptom, so the constant is sourced from one place
and pinned here.
"""
from __future__ import annotations

import pytest


def test_fit_sources_the_conversion_from_pocket_charges():
    """One definition, not two."""
    from tools.active_site.pocket_charges import ANGSTROM_TO_BOHR as pocket_a2b
    from tools.campaign.fit import ANGSTROM_TO_BOHR as fit_a2b

    assert fit_a2b is pocket_a2b, (
        "fit.py re-declares the Angstrom->Bohr factor instead of importing it; "
        "the two can now drift"
    )


def test_the_conversion_matches_the_literal_ferric_uses():
    """Compared against the literal, not read back from the module under test:
    a test that sources its expected value from the code it tests cannot detect
    a change to that value."""
    from tools.campaign.fit import ANGSTROM_TO_BOHR

    assert ANGSTROM_TO_BOHR == pytest.approx(1.0 / 0.529_177_210_92, rel=1e-15)


def test_ferric_and_the_tools_agree_on_the_conversion():
    """An INDEPENDENT construction: ferric's own binding must produce the same
    factor. This is the check with teeth -- the tools could be internally
    consistent and still disagree with the engine they feed."""
    ferric = pytest.importorskip("ferric")
    from tools.campaign.fit import ANGSTROM_TO_BOHR

    mol = ferric.Molecule.from_xyz_string("1\n\nHe 1.0 2.0 3.0\n")
    ang = mol.coords()[0]
    bohr = mol.coords_bohr()[0]
    for a, b in zip(ang, bohr):
        assert b == pytest.approx(a * ANGSTROM_TO_BOHR, rel=1e-12)


def test_pocket_charge_coordinates_really_are_bohr():
    """A pocket derived from a real PDB must have coordinates on the Bohr scale.

    7LCJ's pocket sits ~125-157 Angstrom from the origin in the PDB frame, so in
    Bohr the magnitudes must be ~1.89x larger. Checking the SCALE catches a
    dropped conversion that a shape check would miss.
    """
    pytest.importorskip("ferric")
    import shutil

    if shutil.which("pdb2pqr30") is None:
        pytest.skip("pdb2pqr30 not on PATH")

    from tools.active_site.pocket_charges import ANGSTROM_TO_BOHR, derive_pocket_charges

    pocket = derive_pocket_charges(
        "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
    )
    assert pocket.n_charges > 1000
    # Centroid of the charge cloud, in the stored units.
    n = pocket.n_charges
    cx = sum(c[1] for c in pocket.charges) / n
    # The PDB frame puts this pocket at ~125 A in x; in Bohr that is ~236.
    assert cx > 150.0, (
        f"pocket x-centroid is {cx:.1f}; for the 7LCJ frame this should be "
        f"~{125 * ANGSTROM_TO_BOHR:.0f} in Bohr, not ~125 (Angstrom)"
    )
