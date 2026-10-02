"""MBD@rsSCS through the Python bindings (`ferric.mbd_rsscs_energy`).

The full validation (8 systems, two beta, pymbd + libMBD) is the Rust
`validation_mbd_rsscs.rs`; this file checks the binding plumbing: units (the
Molecule is Angstrom, the model is Bohr), argument strictness, and one
same-inputs comparison against pymbd when pymbd is installed.
"""

import pytest

ferric = pytest.importorskip(
    "ferric",
    reason="the compiled ferric extension is not importable; build it with "
    "`cargo build --release -p ferric-python` and make sure the .so is on the "
    "site-packages path (see CLAUDE.md).",
)

WATER_XYZ = """3
water
O  0.0000000  0.0000000  0.1177900
H  0.0000000  0.7554530 -0.4711610
H  0.0000000 -0.7554530 -0.4711610
"""
RATIOS = [0.85, 0.62, 0.62]


@pytest.fixture
def water():
    return ferric.Molecule.from_xyz_string(WATER_XYZ, 0, 1)


def test_matches_pymbd_with_identical_inputs(water):
    pymbd = pytest.importorskip("pymbd")
    import numpy as np

    res = ferric.mbd_rsscs_energy(water, RATIOS, functional="PBE")
    assert res.beta == 0.83
    xyz = np.asarray(water.coords_bohr(), dtype=float)
    a0, c6, rv = pymbd.from_volumes(["O", "H", "H"], RATIOS)
    want = pymbd.mbd_energy(xyz, a0, c6, rv, 0.83)
    # Validation-tier bar is 1e-10 rel on E (see validation_mbd_rsscs.rs);
    # here the only extra step is the Angstrom->Bohr conversion, which both
    # sides take from the same Molecule.
    assert res.energy == pytest.approx(want, rel=1e-10, abs=0)
    assert res.energy < 0
    a_rs, c6_rs, r_rs = pymbd.screening(xyz, a0, c6, rv, 0.83)
    assert res.alpha_0_rsscs == pytest.approx(list(a_rs), rel=1e-12)
    assert res.c6_rsscs == pytest.approx(list(c6_rs), rel=1e-12)
    assert res.r_vdw_rsscs == pytest.approx(list(r_rs), rel=1e-12)


def test_beta_is_required_and_strict(water):
    with pytest.raises(ValueError, match="beta"):
        ferric.mbd_rsscs_energy(water, RATIOS)
    with pytest.raises(ValueError, match="not both"):
        ferric.mbd_rsscs_energy(water, RATIOS, beta=0.83, functional="PBE")
    with pytest.raises(ValueError, match="b3lyp"):
        ferric.mbd_rsscs_energy(water, RATIOS, functional="b3lyp")
    explicit = ferric.mbd_rsscs_energy(water, RATIOS, beta=0.85)
    named = ferric.mbd_rsscs_energy(water, RATIOS, functional="pbe0")
    assert explicit.energy == named.energy


def test_ratio_count_must_match(water):
    with pytest.raises(ValueError, match="volume ratios"):
        ferric.mbd_rsscs_energy(water, RATIOS[:2], beta=0.83)
